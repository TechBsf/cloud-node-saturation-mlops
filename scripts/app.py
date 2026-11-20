import os
from typing import Union

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from src.schema import ComputeNodeTelemetry

EXPECTED_FEATURES = 6
MODEL_PATH = os.getenv("MODEL_PATH", "model/model.joblib")

app = FastAPI(title="Cloud Compute Node Saturation Risk API")
model = joblib.load(MODEL_PATH)


class TelemetryVector(BaseModel):
    features: list[float]

    @field_validator("features")
    @classmethod
    def validate_length(cls, v: list[float]) -> list[float]:
        if len(v) != EXPECTED_FEATURES:
            raise ValueError(f"Expected {EXPECTED_FEATURES} features, got {len(v)}")
        return v


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(data: Union[ComputeNodeTelemetry, TelemetryVector]) -> dict[str, float]:
    try:
        if isinstance(data, ComputeNodeTelemetry):
            row = [
                data.cpu_utilization_pct,
                data.memory_utilization_pct,
                data.request_rate_rps,
                data.p99_latency_ms,
                float(data.queue_depth),
                float(data.uptime_hours),
            ]
            arr = np.array(row, dtype=float).reshape(1, -1)
        else:
            arr = np.array(data.features, dtype=float).reshape(1, -1)

        pred = float(model.predict(arr)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return {"saturation_risk_score": pred, "prediction": pred}
