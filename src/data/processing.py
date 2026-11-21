from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


def get_csv_path(train_dir: str) -> str:
    """Find the first CSV file in the given directory or use a default name."""
    p = Path(train_dir)
    if p.is_file() and p.suffix == ".csv":
        return str(p)
    pref = p / "compute_capacity.csv"
    if pref.exists():
        return str(pref)
    files = sorted(p.glob("*.csv"))
    if not files:
        raise RuntimeError(f"No CSV found in {train_dir}")
    return str(files[0])


def load_data(csv_path: str) -> pd.DataFrame:
    """Load data from CSV, handling common separators."""
    df = pd.read_csv(csv_path, sep=",")
    if df.shape[1] == 1:
        df = pd.read_csv(csv_path, sep=";")
    return df


def validate_training_data(
    frame: pd.DataFrame,
    features: Iterable[str],
    target: str,
) -> None:
    """Fail fast on malformed or unusable training data."""
    feature_names = list(features)
    missing = [name for name in [*feature_names, target] if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    if frame.empty:
        raise ValueError("Training dataset is empty")

    required = frame[[*feature_names, target]]
    if required.isna().any().any():
        raise ValueError("Training data contains missing values")
    numeric = required.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Training features and target must be numeric")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Training features and target must contain finite values")

    target_values = numeric[target]
    if ((target_values < 0.0) | (target_values > 1.0)).any():
        raise ValueError("Training target must be in [0, 1]")
