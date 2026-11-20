from typing import Union
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from src.schema import ComputeNodeTelemetry

app = FastAPI(title="Cloud Compute Node Saturation Risk API")
model = joblib.load("model/model.joblib")

class TelemetryVector(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: Union[ComputeNodeTelemetry, TelemetryVector]):
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
    return {"saturation_risk_score": pred, "prediction": pred}
