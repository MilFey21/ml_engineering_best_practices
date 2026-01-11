"""Configuration for churn prediction project."""

from dataclasses import dataclass
from pathlib import Path

from src.config import DATA_DIR, MODELS_DIR


@dataclass
class ChurnPredictionConfig:
    """Configuration for churn prediction experiments."""

    # Data paths
    raw_data_dir: Path = DATA_DIR / "raw"
    processed_data_dir: Path = DATA_DIR / "processed"
    models_dir: Path = MODELS_DIR

    # Dataset settings
    dataset_name: str = "telco_customer_churn"
    dataset_kaggle: str = "blastchar/telco-customer-churn"
    features_file: str = "features.csv"
    labels_file: str = "labels.csv"

    # Model settings
    test_size: float = 0.2
    random_state: int = 42

    # ClearML settings
    clearml_project: str = "Churn Prediction"
    clearml_task_type: str = "training"

    def __post_init__(self):
        """Create directories if they don't exist."""
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
