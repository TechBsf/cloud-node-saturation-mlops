import numpy as np
from sklearn.linear_model import LinearRegression

from src.models.evaluation import cross_val_metrics, eval_metrics


def test_eval_metrics_perfect():
    y = np.array([1.0, 2.0, 3.0])
    m = eval_metrics(y, y)
    assert m["rmse"] == 0.0
    assert m["mae"] == 0.0
    assert m["r2"] == 1.0


def test_eval_metrics_keys_and_types():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.5, 2.5])
    m = eval_metrics(y_true, y_pred)
    assert set(m) == {"rmse", "mae", "r2"}
    assert all(isinstance(v, float) for v in m.values())


def test_cross_val_metrics_has_stable_shape():
    features = np.array([[i] for i in range(20)], dtype=float)
    target = np.array([float(i) for i in range(20)])
    metrics = cross_val_metrics(LinearRegression(), features, target)
    assert set(metrics) == {"rmse", "rmse_std", "r2", "r2_std"}
    assert metrics["rmse"] >= 0.0
