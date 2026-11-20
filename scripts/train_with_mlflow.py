import os
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def get_csv_path(train_dir: str) -> str:
    p = Path(train_dir)
    if p.is_file() and p.suffix == ".csv":
        return str(p)
    pref = p / "compute_capacity.csv"
    if pref.exists():
        return str(pref)
    files = sorted(p.glob("*.csv"))
    if not files:
        raise RuntimeError(f"No CSV found in {train_dir}")
    return str(files[0])


def try_mlflow_setup(uri: str):
    if not uri:
        return None
    try:
        import mlflow
        import mlflow.xgboost

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("cloud-node-saturation-risk")
        return mlflow
    except Exception as e:
        print(f"[WARN] MLflow setup failed, continuing without MLflow. Error: {e}")
        return None


def eval_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    env_train = os.environ.get("SM_CHANNEL_TRAIN", "")
    train_dir = env_train if (env_train and Path(env_train).exists()) else "data"
    model_dir = os.environ.get("SM_MODEL_DIR", "") or "model"
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "").rstrip("/")

    csv_path = get_csv_path(train_dir)
    df = pd.read_csv(csv_path, sep=",")
    if df.shape[1] == 1:
        df = pd.read_csv(csv_path, sep=";")

    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)

    baseline = LinearRegression()
    baseline.fit(X_tr, y_tr)
    base_metrics = eval_metrics(y_va, baseline.predict(X_va))

    xgb_params = {
        "n_estimators": 120,
        "max_depth": 4,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
    }
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_tr, y_tr)
    xgb_metrics = eval_metrics(y_va, model.predict(X_va))

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    art = Path(model_dir) / "model.joblib"
    joblib.dump(model, art)

    root_art = Path("model/model.joblib")
    root_art.parent.mkdir(parents=True, exist_ok=True)
    if art.resolve() != root_art.resolve():
        shutil.copy2(art, root_art)

    mlflow = try_mlflow_setup(mlflow_uri)
    if mlflow is not None:
        with mlflow.start_run(run_name="baseline_linear_regression"):
            mlflow.log_param("model_type", "LinearRegression")
            mlflow.log_metrics({f"val_{k}": v for k, v in base_metrics.items()})

        with mlflow.start_run(run_name="candidate_xgb_regressor"):
            mlflow.log_param("model_type", "XGBRegressor")
            mlflow.log_params(xgb_params)
            mlflow.log_metrics({f"val_{k}": v for k, v in xgb_metrics.items()})
            mlflow.xgboost.log_model(model, artifact_path="model")

    print(
        f"OK train_with_mlflow -> "
        f"lin_rmse={base_metrics['rmse']:.4f} (r2={base_metrics['r2']:.4f}) | "
        f"xgb_rmse={xgb_metrics['rmse']:.4f} (r2={xgb_metrics['r2']:.4f}) | "
        f"saved={root_art}"
    )


if __name__ == "__main__":
    main()
