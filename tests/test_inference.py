import numpy as np

from src.models.inference import prepare_inference_features
from src.schema import ComputeNodeTelemetry


def test_feature_order_matches_schema():
    data = ComputeNodeTelemetry(
        cpu_utilization_pct=1.0,
        memory_utilization_pct=2.0,
        request_rate_rps=3.0,
        p99_latency_ms=4.0,
        queue_depth=5,
        uptime_hours=6,
    )
    arr = prepare_inference_features(data)
    assert arr.shape == (1, 6)
    assert np.array_equal(arr, np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]))
