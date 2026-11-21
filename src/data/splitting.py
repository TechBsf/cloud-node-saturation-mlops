import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 1.0 - TRAIN_FRACTION - VALIDATION_FRACTION


def split_features_target(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Create deterministic train, validation and test feature/target splits."""
    features_frame = frame[features]
    target_series = frame[target]
    row_count = len(frame)
    train_count = int(row_count * TRAIN_FRACTION)
    validation_count = int(row_count * VALIDATION_FRACTION)
    test_count = row_count - train_count - validation_count

    x_train, x_remainder, y_train, y_remainder = train_test_split(
        features_frame,
        target_series,
        train_size=train_count,
        random_state=RANDOM_STATE,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_remainder,
        y_remainder,
        train_size=validation_count,
        test_size=test_count,
        random_state=RANDOM_STATE,
    )
    return (
        x_train,
        x_validation,
        x_test,
        y_train,
        y_validation,
        y_test,
    )
