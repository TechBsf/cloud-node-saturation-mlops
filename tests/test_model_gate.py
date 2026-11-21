from pathlib import Path

import joblib
import numpy as np
import pytest

MODEL_PATH = Path("model/model.joblib")


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Train the model first (make train) — CI trains before running the suite",
)
def test_model_artifact_loads_and_predicts():
    model = joblib.load(MODEL_PATH)
    samples = np.array(
        [
            [55.0, 60.0, 180.0, 65.0, 4.0, 320.0],
            [33.0, 48.0, 156.0, 34.0, 2.0, 1879.0],
        ]
    )
    preds = model.predict(samples)
    assert preds.shape == (2,)
    assert np.isfinite(preds).all()
    # Sanity check tolérant : le clipping à [0,1] est testé au niveau de l'API.
    assert (preds >= -0.1).all()
    assert (preds <= 1.1).all()
