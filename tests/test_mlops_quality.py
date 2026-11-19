import pytest
from pydantic import ValidationError
from src.schema import WineFeatures


def test_valid_wine_features():
    sample = WineFeatures(
        fixed_acidity=7.4,
        volatile_acidity=0.7,
        citric_acid=0.0,
        residual_sugar=1.9,
        chlorides=0.076,
        free_sulfur_dioxide=11.0,
        total_sulfur_dioxide=34.0,
        density=0.9978,
        pH=3.51,
        sulphates=0.56,
        alcohol=9.4,
    )
    assert sample.alcohol == 9.4


def test_reject_invalid_ph():
    with pytest.raises(ValidationError):
        WineFeatures(
            fixed_acidity=7.4,
            volatile_acidity=0.7,
            citric_acid=0.0,
            residual_sugar=1.9,
            chlorides=0.076,
            free_sulfur_dioxide=11.0,
            total_sulfur_dioxide=34.0,
            density=0.9978,
            pH=19.5,
            sulphates=0.56,
            alcohol=9.4,
        )
