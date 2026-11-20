import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = ["cpu_utilization_pct", "memory_utilization_pct", "request_rate_rps",
            "p99_latency_ms", "queue_depth", "uptime_hours"]
TARGET = "saturation_risk_score"

df = pd.read_csv("data/compute_capacity.csv")
X, y = df[FEATURES], df[TARGET]

# IMPORTANT : mets exactement le même test_size et random_state que train_with_mlflow.py
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Baseline (mean predictor)": DummyRegressor(strategy="mean").fit(X_train, y_train),
    "Linear Regression": LinearRegression().fit(X_train, y_train),
    "XGBRegressor (final)": joblib.load("model/model.joblib"),
}

print("| Model | RMSE | MAE | R2 |")
print("|---|---|---|---|")
rmses = {}
for name, m in models.items():
    p = m.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, p)))
    rmses[name] = rmse
    print(f"| {name} | {rmse:.4f} | {mean_absolute_error(y_test, p):.4f} | {r2_score(y_test, p):.4f} |")

gain = 100 * (1 - rmses["XGBRegressor (final)"] / rmses["Linear Regression"])
print(f"\nRMSE reduction vs linear: {gain:.1f} %")