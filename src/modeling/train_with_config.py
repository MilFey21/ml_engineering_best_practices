"""Training script with Hydra configuration management."""

import json
from pathlib import Path
import sys

import hydra
import joblib
from loguru import logger
import mlflow
import mlflow.sklearn
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Add project root to path for imports
PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

from src.config import MODELS_DIR  # noqa: E402
from src.modeling.config_schema import ProjectConfig  # noqa: E402

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def create_model(model_config: DictConfig, random_state: int = 42):
    """Create a model instance based on configuration."""
    model_type = model_config.model.type
    params = dict(model_config.model.params)

    # Ensure random_state is set
    if "random_state" not in params:
        params["random_state"] = random_state

    model_classes = {
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "LogisticRegression": LogisticRegression,
    }

    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")

    ModelClass = model_classes[model_type]
    return ModelClass(**params)


# Determine config path relative to project root
PROJ_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = str(PROJ_ROOT / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """Train a model using Hydra configuration."""
    logger.info("Starting training with Hydra configuration")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Validate configuration using Pydantic
    try:
        config_dict: dict = OmegaConf.to_container(cfg, resolve=True)
        ProjectConfig(**config_dict)
        logger.info("Configuration validation passed")
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise

    # Allow access to nested config
    OmegaConf.set_struct(cfg, False)

    # Set MLflow tracking if enabled
    if cfg.training.mlflow.enabled:
        mlflow.set_tracking_uri(cfg.training.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.training.mlflow.experiment_name)

    # Load data
    logger.info(f"Loading features from {cfg.data.features_path}")
    logger.info(f"Loading labels from {cfg.data.labels_path}")

    X = pd.read_csv(cfg.data.features_path)
    y = pd.read_csv(cfg.data.labels_path).squeeze()

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Labels shape: {y.shape}")

    # Split data
    test_size = cfg.data.processing.test_size
    random_state = cfg.seed

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if cfg.data.processing.stratify else None,
    )

    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")

    # Create and train model
    logger.info(f"Training {cfg.model.name} model...")
    model = create_model(cfg, random_state=random_state)
    model.fit(X_train, y_train)

    # Make predictions
    logger.info("Making predictions...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate probabilities for ROC-AUC
    try:
        y_test_proba = model.predict_proba(X_test)[:, 1]
        test_roc_auc = roc_auc_score(y_test, y_test_proba)
    except (AttributeError, IndexError):
        y_test_proba = None
        test_roc_auc = None

    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    metrics = {
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "test_f1_score": float(test_f1),
    }

    if test_roc_auc is not None:
        metrics["test_roc_auc"] = float(test_roc_auc)

    logger.info("Metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")

    # Log to MLflow if enabled
    if cfg.training.mlflow.enabled:
        with mlflow.start_run(run_name=cfg.training.mlflow.run_name):
            # Log parameters
            mlflow.log_params(
                {
                    "model_name": cfg.model.name,
                    "model_type": cfg.model.type,
                    **{f"model.{k}": v for k, v in cfg.model.params.items()},
                    "test_size": test_size,
                    "random_state": random_state,
                }
            )

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=f"{cfg.model.name}_churn",
            )

    # Save model
    model_path = Path(cfg.training.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save metrics
    metrics_path = Path(cfg.training.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
