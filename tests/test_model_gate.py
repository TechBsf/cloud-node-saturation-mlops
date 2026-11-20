from pathlib import Path
import joblib
import numpy as np


def test_model_artifact_loads_and_predicts():
    model_path = Path("model/model.joblib")
    assert model_path.exists(), "model/model.joblib must exist"
    model = joblib.load(model_path)
    sample = np.array([[55.0, 60.0, 180.0, 65.0, 4.0, 320.0]])
    preds = model.predict(sample)
    assert len(preds) == 1
    assert np.isfinite(preds[0])
    assert 0.0 <= float(preds[0]) <= 1.0
