from pathlib import Path
import joblib
import numpy as np


def test_model_artifact_loads_and_predicts():
    model_path = Path("model/model.joblib")
    assert model_path.exists(), "model/model.joblib must exist"

    model = joblib.load(model_path)
    sample = np.array([[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]])
    preds = model.predict(sample)

    assert len(preds) == 1
    assert np.isfinite(preds[0])
    assert 0.0 <= float(preds[0]) <= 10.0
