from pydantic import BaseModel, Field

# Ordre canonique des features — source de vérité pour l'entraînement ET l'inférence.
FEATURES = [
    "cpu_utilization_pct",
    "memory_utilization_pct",
    "request_rate_rps",
    "p99_latency_ms",
    "queue_depth",
    "uptime_hours",
]
TARGET = "saturation_risk_score"


class ComputeNodeTelemetry(BaseModel):
    cpu_utilization_pct: float = Field(..., ge=0.0, le=100.0)
    memory_utilization_pct: float = Field(..., ge=0.0, le=100.0)
    request_rate_rps: float = Field(..., ge=0.0, le=10000.0)
    p99_latency_ms: float = Field(..., ge=0.0, le=5000.0)
    queue_depth: int = Field(..., ge=0, le=1000)
    uptime_hours: int = Field(..., ge=0, le=200000)
