import numpy as np
from numpy.typing import NDArray

from src.schema import FEATURES, ComputeNodeTelemetry


def prepare_inference_features(data: ComputeNodeTelemetry) -> NDArray[np.float64]:
    """Convert telemetry Pydantic model to a numpy array for prediction.

    The feature order is driven by ``src.schema.FEATURES`` so it can never
    drift from the training order.
    """
    row = [float(getattr(data, feature)) for feature in FEATURES]
    return np.asarray(row, dtype=np.float64).reshape(1, -1)
