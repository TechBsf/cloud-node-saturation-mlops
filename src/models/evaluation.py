import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate


def eval_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Calculate and return regression metrics."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def cross_val_metrics(
    model: object,
    features: ArrayLike,
    target: ArrayLike,
    folds: int = 5,
) -> dict[str, float]:
    """Calculate mean RMSE and R2 over deterministic folds."""
    scores = cross_validate(
        model,
        features,
        target,
        cv=folds,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        return_train_score=False,
    )
    rmse = -scores["test_rmse"]
    r2 = scores["test_r2"]
    return {
        "rmse": float(np.mean(rmse)),
        "rmse_std": float(np.std(rmse)),
        "r2": float(np.mean(r2)),
        "r2_std": float(np.std(r2)),
    }
