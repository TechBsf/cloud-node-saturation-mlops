from pydantic import BaseModel, Field


class WineFeatures(BaseModel):
    fixed_acidity: float = Field(..., ge=0.0, le=25.0)
    volatile_acidity: float = Field(..., ge=0.0, le=5.0)
    citric_acid: float = Field(..., ge=0.0, le=5.0)
    residual_sugar: float = Field(..., ge=0.0, le=100.0)
    chlorides: float = Field(..., ge=0.0, le=2.0)
    free_sulfur_dioxide: float = Field(..., ge=0.0, le=300.0)
    total_sulfur_dioxide: float = Field(..., ge=0.0, le=500.0)
    density: float = Field(..., ge=0.5, le=2.0)
    pH: float = Field(..., ge=0.0, le=14.0)
    sulphates: float = Field(..., ge=0.0, le=5.0)
    alcohol: float = Field(..., ge=0.0, le=30.0)
