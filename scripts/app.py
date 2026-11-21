import json
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from src.models.drift import calculate_feature_drift
from src.models.inference import prepare_inference_features
from src.schema import FEATURES, ComputeNodeTelemetry

EXPECTED_FEATURES = len(FEATURES)
MODEL_PATH = os.getenv("MODEL_PATH", "model/model.joblib")
LOGGER = logging.getLogger(__name__)
DRIFT_THRESHOLD = 3.0
MIN_DRIFT_SAMPLES = 30
DRIFT_MAX_SAMPLES = 1000


def _load_model() -> object:
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH!r}. Train it first or set MODEL_PATH."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from {MODEL_PATH!r}: {exc}") from exc

    # Anti-skew train/serve : le modèle doit consommer exactement les features du contrat.
    actual_features = getattr(model, "n_features_in_", None)
    if actual_features is not None and int(actual_features) != EXPECTED_FEATURES:
        raise RuntimeError(
            f"Model at {MODEL_PATH!r} expects {actual_features} features, "
            f"but the contract defines {EXPECTED_FEATURES}"
        )
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = _load_model()
    app.state.metrics = {
        "prediction_count": 0,
        "prediction_error_count": 0,
        "latency_total_seconds": 0.0,
        "last_prediction_score": None,
    }
    app.state.drift_samples = {feature: [] for feature in FEATURES}
    app.state.reference_statistics = {}
    manifest_path = Path(MODEL_PATH).with_name("model_metadata.json")
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            app.state.reference_statistics = manifest.get("reference_statistics", {})
        except json.JSONDecodeError:
            LOGGER.warning("Model metadata is invalid JSON; drift monitoring disabled")
    yield


app = FastAPI(title="Cloud Compute Node Saturation Risk API", lifespan=lifespan)


def _feature_bounds(feature: str) -> tuple[float, float]:
    """Bornes [ge, le] déclarées dans le contrat Pydantic (source unique)."""
    constraints = ComputeNodeTelemetry.model_fields[feature].metadata
    minimum = next((c.ge for c in constraints if hasattr(c, "ge")), 0.0)
    maximum = next((c.le for c in constraints if hasattr(c, "le")), math.inf)
    return float(minimum), float(maximum)


class TelemetryVector(BaseModel):
    features: list[float]

    @field_validator("features")
    @classmethod
    def validate_length(cls, v: list[float]) -> list[float]:
        if len(v) != EXPECTED_FEATURES:
            raise ValueError(f"Expected {EXPECTED_FEATURES} features, got {len(v)}")
        for feature, value in zip(FEATURES, v, strict=True):
            if not math.isfinite(value):
                raise ValueError(f"{feature} must be finite")
            minimum, maximum = _feature_bounds(feature)
            if value < minimum or value > maximum:
                raise ValueError(f"{feature} must be between {minimum} and {maximum}")
        return v


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    model_loaded = getattr(app.state, "model", None) is not None
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.get("/metrics")
def metrics() -> dict[str, object]:
    values = getattr(app.state, "metrics", {})
    count = int(values.get("prediction_count", 0))
    latency_total = float(values.get("latency_total_seconds", 0.0))
    return {
        **values,
        "average_latency_seconds": latency_total / count if count else 0.0,
    }


@app.get("/drift")
def drift() -> dict[str, object]:
    results: list[dict[str, float | str | bool]] = []
    sample_count = len(app.state.drift_samples.get(FEATURES[0], []))
    status = "insufficient_samples" if sample_count < MIN_DRIFT_SAMPLES else "ready"

    for feature in FEATURES:
        reference = app.state.reference_statistics.get(feature)
        if reference is None:
            results.append({
                "feature": feature,
                "live_mean": 0.0,
                "reference_mean": 0.0,
                "z_score": 0.0,
                "drift_detected": False,
            })
            continue
        results.append(calculate_feature_drift(
            feature,
            app.state.drift_samples[feature],
            float(reference["mean"]),
            float(reference["std"]),
            DRIFT_THRESHOLD,
        ))

    any_drift = any(bool(result["drift_detected"]) for result in results)
    return {
        "status": status,
        "sample_count": sample_count,
        "minimum_samples": MIN_DRIFT_SAMPLES,
        "threshold_z_score": DRIFT_THRESHOLD,
        "drift_detected": any_drift,
        "features": results,
    }


@app.post("/predict")
def predict(data: ComputeNodeTelemetry | TelemetryVector) -> dict[str, float]:
    started_at = time.perf_counter()
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if isinstance(data, ComputeNodeTelemetry):
        arr = prepare_inference_features(data)
        live_values = [float(getattr(data, feature)) for feature in FEATURES]
    else:
        arr = np.array(data.features, dtype=float).reshape(1, -1)
        live_values = data.features

    try:
        raw = float(model.predict(arr)[0])
    except Exception:
        app.state.metrics["prediction_error_count"] += 1
        LOGGER.exception("Model inference failed")
        raise HTTPException(status_code=500, detail="Inference failed") from None

    # Le score de saturation est borné à [0, 1] par contrat métier.
    score = float(np.clip(raw, 0.0, 1.0))

    # N'enregistrer les échantillons de drift qu'après une prédiction réussie.
    for feature, value in zip(FEATURES, live_values, strict=True):
        samples = app.state.drift_samples[feature]
        samples.append(float(value))
        if len(samples) > DRIFT_MAX_SAMPLES:
            samples.pop(0)
    app.state.metrics["prediction_count"] += 1
    app.state.metrics["latency_total_seconds"] += time.perf_counter() - started_at
    app.state.metrics["last_prediction_score"] = score
    return {"saturation_risk_score": score}
