import pandas as pd
import pytest

from src.data.processing import get_csv_path, load_data, validate_training_data
from src.data.splitting import split_features_target
from src.models.model_selection import candidate_passes_model_gate


def test_splits_are_deterministic_and_cover_dataset():
    frame = pd.DataFrame({
        "feature_a": range(100),
        "feature_b": range(100, 200),
        "target": range(200, 300),
    })

    first = split_features_target(frame, ["feature_a", "feature_b"], "target")
    second = split_features_target(frame, ["feature_a", "feature_b"], "target")

    assert len(first[0]) == 70
    assert len(first[1]) == 15
    assert len(first[2]) == 15
    for first_item, second_item in zip(first, second, strict=True):
        assert first_item.equals(second_item)


def test_model_gate_rejects_candidate_without_enough_improvement():
    baseline = {"rmse": 0.0300, "mae": 0.0200, "r2": 0.90}
    worse_candidate = {"rmse": 0.0299, "mae": 0.0198, "r2": 0.905}
    assert not candidate_passes_model_gate(baseline, worse_candidate)


def test_model_gate_accepts_candidate_with_sufficient_improvement():
    baseline = {"rmse": 0.0300, "mae": 0.0200, "r2": 0.90}
    improved_candidate = {"rmse": 0.0280, "mae": 0.0180, "r2": 0.91}
    assert candidate_passes_model_gate(baseline, improved_candidate)


def test_load_data_comma_separator(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    df = load_data(str(csv))
    assert df.shape == (2, 2)
    assert list(df.columns) == ["a", "b"]


def test_load_data_semicolon_separator(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("a;b\n1;2\n3;4\n", encoding="utf-8")
    df = load_data(str(csv))
    assert df.shape == (2, 2)


def test_get_csv_path_prefers_default(tmp_path):
    (tmp_path / "other.csv").write_text("a\n1\n", encoding="utf-8")
    (tmp_path / "compute_capacity.csv").write_text("a\n1\n", encoding="utf-8")
    assert get_csv_path(str(tmp_path)).endswith("compute_capacity.csv")


def test_get_csv_path_missing(tmp_path):
    with pytest.raises(RuntimeError):
        get_csv_path(str(tmp_path))


def test_validate_training_data_rejects_missing_values():
    frame = pd.DataFrame({
        "feature_a": [1.0, None],
        "feature_b": [2.0, 3.0],
        "target": [0.2, 0.8],
    })
    with pytest.raises(ValueError, match="missing values"):
        validate_training_data(frame, ["feature_a", "feature_b"], "target")


def test_validate_training_data_rejects_target_out_of_range():
    frame = pd.DataFrame({
        "feature_a": [1.0, 2.0],
        "feature_b": [2.0, 3.0],
        "target": [0.2, 1.2],
    })
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_training_data(frame, ["feature_a", "feature_b"], "target")
