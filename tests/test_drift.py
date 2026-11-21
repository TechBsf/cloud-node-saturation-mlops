from src.models.drift import calculate_feature_drift


def test_drift_detected_when_live_mean_is_far_from_reference():
    result = calculate_feature_drift(
        "cpu_utilization_pct",
        [90.0, 92.0, 91.0],
        reference_mean=50.0,
        reference_std=10.0,
        threshold=3.0,
    )
    assert result["drift_detected"] is True
    assert float(result["z_score"]) >= 4.0


def test_drift_not_detected_for_reference_like_values():
    result = calculate_feature_drift(
        "cpu_utilization_pct",
        [48.0, 50.0, 52.0],
        reference_mean=50.0,
        reference_std=10.0,
        threshold=3.0,
    )
    assert result["drift_detected"] is False
