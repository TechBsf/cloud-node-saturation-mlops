from collections.abc import Iterable
from math import isfinite


def calculate_feature_drift(
    feature: str,
    live_values: Iterable[float],
    reference_mean: float,
    reference_std: float,
    threshold: float = 3.0,
) -> dict[str, float | str | bool]:
    values = [float(value) for value in live_values]
    if not values:
        return {
            "feature": feature,
            "live_mean": 0.0,
            "reference_mean": reference_mean,
            "z_score": 0.0,
            "drift_detected": False,
        }

    live_mean = sum(values) / len(values)
    scale = reference_std if reference_std > 0.0 else 1.0
    z_score = abs(live_mean - reference_mean) / scale
    return {
        "feature": feature,
        "live_mean": float(live_mean),
        "reference_mean": float(reference_mean),
        "z_score": float(z_score) if isfinite(z_score) else 0.0,
        "drift_detected": bool(isfinite(z_score) and z_score >= threshold),
    }
