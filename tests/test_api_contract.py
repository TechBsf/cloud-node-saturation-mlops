import pytest
from pydantic import ValidationError
from src.schema import ComputeNodeTelemetry
from scripts import app as app_module

def test_telemetry_contract_and_app_module():
    assert hasattr(app_module, 'app')
    sample = ComputeNodeTelemetry(
        cpu_utilization_pct=78.0,
        memory_utilization_pct=71.5,
        request_rate_rps=1200.0,
        p99_latency_ms=140.0,
        queue_depth=18,
        uptime_hours=168.0,
    )
    assert sample.cpu_utilization_pct == 78.0

def test_invalid_telemetry_rejected():
    with pytest.raises(ValidationError):
        ComputeNodeTelemetry(
            cpu_utilization_pct=150.0,
            memory_utilization_pct=71.5,
            request_rate_rps=1200.0,
            p99_latency_ms=140.0,
            queue_depth=18,
            uptime_hours=168.0,
        )