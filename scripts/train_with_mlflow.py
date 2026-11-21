import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression

from src.data.processing import get_csv_path, load_data, validate_training_data
from src.data.splitting import (
    RANDOM_STATE,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
    split_features_target,
)
from src.models.evaluation import cross_val_metrics, eval_metrics
from src.models.model_selection import candidate_passes_model_gate
from src.schema import FEATURES, TARGET


def try_mlflow_setup(uri: str) -> Any | None:
    if not uri:
        return None
    try:
        import mlflow
        import mlflow.models
        import mlflow.xgboost

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("cloud-node-saturation-risk")
        return mlflow
    except Exception as e:
        print(f"[WARN] MLflow setup failed, continuing without MLflow. Error: {e}")
        return None


def main() -> None:
    env_train = os.environ.get("SM_CHANNEL_TRAIN", "")
    train_dir = env_train if (env_train and Path(env_train).exists()) else "data"
    model_dir = os.environ.get("SM_MODEL_DIR", "") or "model"
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "").rstrip("/")

    csv_path = get_csv_path(train_dir)
    df = load_data(csv_path)
    validate_training_data(df, FEATURES, TARGET)
    dataset_hash = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()

    x_train, x_validation, x_test, y_train, y_validation, y_test = split_features_target(
        df,
        FEATURES,
        TARGET,
    )
    x_train_array = x_train.to_numpy()
    x_validation_array = x_validation.to_numpy()
    x_test_array = x_test.to_numpy()
    x_cv_array = np.vstack((x_train_array, x_validation_array))
    y_cv = np.concatenate((y_train.to_numpy(), y_validation.to_numpy()))

    baseline = LinearRegression()
    baseline.fit(x_train_array, y_train)
    base_metrics = eval_metrics(y_validation, baseline.predict(x_validation_array))
    baseline_cv_metrics = cross_val_metrics(baseline, x_cv_array, y_cv)

    xgb_params = {
        "n_estimators": 120,
        "max_depth": 4,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": RANDOM_STATE,
    }
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(x_train_array, y_train)
    xgb_metrics = eval_metrics(y_validation, model.predict(x_validation_array))
    candidate_test_metrics = eval_metrics(y_test, model.predict(x_test_array))
    candidate_cv_metrics = cross_val_metrics(model, x_cv_array, y_cv)
    candidate_approved = candidate_passes_model_gate(
        baseline_cv_metrics,
        candidate_cv_metrics,
    )
    selected_model = model if candidate_approved else baseline
    selected_model_name = "XGBRegressor" if candidate_approved else "LinearRegression"
    selected_test_metrics = eval_metrics(y_test, selected_model.predict(x_test_array))
    baseline_test_metrics = eval_metrics(y_test, baseline.predict(x_test_array))

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    art = Path(model_dir) / "model.joblib"
    joblib.dump(selected_model, art)

    root_art = Path("model/model.joblib")
    root_art.parent.mkdir(parents=True, exist_ok=True)
    if art.resolve() != root_art.resolve():
        shutil.copy2(art, root_art)

    manifest = {
        "selected_model": selected_model_name,
        "model_type": type(selected_model).__name__,
        "dataset_sha256": dataset_hash,
        "row_count": int(len(df)),
        "features": FEATURES,
        "target": TARGET,
        "reference_statistics": {
            feature: {
                "mean": float(df[feature].mean()),
                "std": float(df[feature].std()),
            }
            for feature in FEATURES
        },
        "random_state": RANDOM_STATE,
        "split": {
            "train": TRAIN_FRACTION,
            "validation": VALIDATION_FRACTION,
            "test": TEST_FRACTION,
        },
        "model_gate": {
            "approved": candidate_approved,
            "minimum_rmse_reduction": 0.01,
            "candidate_cv_rmse": candidate_cv_metrics["rmse"],
            "baseline_cv_rmse": baseline_cv_metrics["rmse"],
        },
        "test_metrics": selected_test_metrics,
    }
    manifest_path = root_art.with_name("model_metadata.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    mlflow = try_mlflow_setup(mlflow_uri)
    if mlflow is not None:
        with mlflow.start_run(run_name="baseline_linear_regression"):
            mlflow.log_param("model_type", "LinearRegression")
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("train_fraction", TRAIN_FRACTION)
            mlflow.log_param("validation_fraction", VALIDATION_FRACTION)
            mlflow.log_param("test_fraction", TEST_FRACTION)
            mlflow.log_metrics({f"val_{k}": v for k, v in base_metrics.items()})
            mlflow.log_metrics({f"cv_{k}": v for k, v in baseline_cv_metrics.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in baseline_test_metrics.items()})

        with mlflow.start_run(run_name="candidate_xgb_regressor"):
            mlflow.log_param("model_type", "XGBRegressor")
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("train_fraction", TRAIN_FRACTION)
            mlflow.log_param("validation_fraction", VALIDATION_FRACTION)
            mlflow.log_param("test_fraction", TEST_FRACTION)
            mlflow.log_param("candidate_approved", candidate_approved)
            mlflow.log_params(xgb_params)
            mlflow.log_metrics({f"val_{k}": v for k, v in xgb_metrics.items()})
            mlflow.log_metrics({f"cv_{k}": v for k, v in candidate_cv_metrics.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in candidate_test_metrics.items()})
            mlflow.xgboost.log_model(
                model,
                name="model",
                signature=mlflow.models.infer_signature(
                    x_validation_array,
                    model.predict(x_validation_array),
                ),
                input_example=x_validation.iloc[:1],
            )

    print(
        f"OK train_with_mlflow -> "
        f"lin_rmse={base_metrics['rmse']:.4f} (r2={base_metrics['r2']:.4f}) | "
        f"xgb_rmse={xgb_metrics['rmse']:.4f} (r2={xgb_metrics['r2']:.4f}) | "
        f"xgb_cv_rmse={candidate_cv_metrics['rmse']:.4f} "
        f"(std={candidate_cv_metrics['rmse_std']:.4f}) | "
        f"test_rmse={selected_test_metrics['rmse']:.4f} | "
        f"gate={'approved' if candidate_approved else 'rejected'} | "
        f"selected={selected_model_name} | saved={root_art}"
    )


if __name__ == "__main__":
    main()
