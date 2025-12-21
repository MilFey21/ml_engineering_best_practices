"""Pydantic schemas for configuration validation."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, validator


class DataConfig(BaseModel):
    """Data configuration schema."""

    raw_data_path: str
    features_path: str
    labels_path: str
    processing: Dict[str, Any]
    features: Dict[str, Any]


class ModelConfig(BaseModel):
    """Model configuration schema."""

    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type/class name")
    params: Dict[str, Any] = Field(..., description="Model hyperparameters")

    @validator("type")
    def validate_model_type(cls, v):
        """Validate model type."""
        allowed_types = [
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "LogisticRegression",
        ]
        if v not in allowed_types:
            raise ValueError(f"Model type must be one of {allowed_types}")
        return v


class MLflowConfig(BaseModel):
    """MLflow configuration schema."""

    enabled: bool = True
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "churn_prediction"
    run_name: Optional[str] = None


class TrainingConfig(BaseModel):
    """Training configuration schema."""

    model_path: str
    metrics_path: str
    mlflow: MLflowConfig
    evaluation: Dict[str, Any]


class ProjectConfig(BaseModel):
    """Main project configuration schema."""

    project: Dict[str, Any]
    paths: Dict[str, str]
    seed: int = Field(42, ge=0, description="Random seed")
    logging: Dict[str, str]
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

    @validator("seed")
    def validate_seed(cls, v):
        """Validate seed is non-negative."""
        if v < 0:
            raise ValueError("Seed must be non-negative")
        return v
