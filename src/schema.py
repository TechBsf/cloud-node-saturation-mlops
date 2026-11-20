from pydantic import BaseModel, Field

class ComputeNodeTelemetry(BaseModel):
    cpu_utilization_pct: float = Field(..., ge=0.0, le=100.0)
    memory_utilization_pct: float = Field(..., ge=0.0, le=100.0)
    request_rate_rps: float = Field(..., ge=0.0, le=10000.0)
    p99_latency_ms: float = Field(..., ge=0.0, le=5000.0)
    queue_depth: int = Field(..., ge=0, le=1000)
    uptime_hours: int = Field(..., ge=0, le=200000)
