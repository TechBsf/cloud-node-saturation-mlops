import pytest
from pydantic import ValidationError

from src.schema import ComputeNodeTelemetry


def test_valid_compute_node_telemetry():
    sample = ComputeNodeTelemetry(
        cpu_utilization_pct=55.0,
        memory_utilization_pct=60.0,
        request_rate_rps=180.0,
        p99_latency_ms=65.0,
        queue_depth=4,
        uptime_hours=320,
    )
    assert sample.cpu_utilization_pct == 55.0


def test_reject_invalid_cpu_utilization():
    with pytest.raises(ValidationError):
        ComputeNodeTelemetry(
            cpu_utilization_pct=145.0,
            memory_utilization_pct=60.0,
            request_rate_rps=180.0,
            p99_latency_ms=65.0,
            queue_depth=4,
            uptime_hours=320,
        )
