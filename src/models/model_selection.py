def candidate_passes_model_gate(
    baseline_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    minimum_rmse_reduction: float = 0.01,
) -> bool:
    """Return whether the candidate improves RMSE enough to replace the baseline."""
    baseline_rmse = baseline_metrics["rmse"]
    candidate_rmse = candidate_metrics["rmse"]
    if baseline_rmse <= 0.0:
        return False
    return candidate_rmse <= baseline_rmse * (1.0 - minimum_rmse_reduction)
