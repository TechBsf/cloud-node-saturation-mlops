import os, shutil
from pathlib import Path
import joblib, numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
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
        import mlflow, mlflow.xgboost
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("cloud-node-saturation-risk")
        return mlflow
    except Exception as e:
        print(f"[WARN] MLflow setup failed, continuing without MLflow. Error: {e}")
        return None

def main():
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
    model = xgb.XGBRegressor(n_estimators=120, max_depth=4, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, random_state=42)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_va)
    rmse = float(np.sqrt(mean_squared_error(y_va, preds)))
    r2 = float(r2_score(y_va, preds))
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    art = Path(model_dir) / "model.joblib"
    joblib.dump(model, art)
    root_art = Path("model/model.joblib")
    root_art.parent.mkdir(parents=True, exist_ok=True)
    if art.resolve() != root_art.resolve():
        shutil.copy2(art, root_art)
    mlflow = try_mlflow_setup(mlflow_uri)
    if mlflow is not None:
        with mlflow.start_run():
            mlflow.log_params({"n_estimators": 120, "max_depth": 4, "learning_rate": 0.08})
            mlflow.log_metrics({"rmse": rmse, "r2": r2})
            mlflow.xgboost.log_model(model, artifact_path="model")
    print(f"OK train_with_mlflow -> rmse={rmse:.4f}, r2={r2:.4f}, saved={root_art}")

if __name__ == "__main__":
    main()
