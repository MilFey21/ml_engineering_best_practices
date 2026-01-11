"""Training script using Hydra for configuration management."""
import json
from pathlib import Path
from typing import Any, Dict

import hydra
import joblib
from loguru import logger
import mlflow
import mlflow.sklearn
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import MODELS_DIR

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def validate_config(cfg: DictConfig) -> None:
    """Validate configuration using Hydra's validation capabilities."""
    # Hydra автоматически валидирует структуру конфигурации
    # Дополнительная валидация значений
    assert cfg.data.test_size > 0 and cfg.data.test_size < 1, "test_size must be between 0 and 1"
    assert cfg.project.random_state >= 0, "random_state must be non-negative"
    assert Path(cfg.paths.features).exists(), f"Features file not found: {cfg.paths.features}"
    assert Path(cfg.paths.labels).exists(), f"Labels file not found: {cfg.paths.labels}"
    
    logger.info("Configuration validation passed")


def instantiate_model(cfg: DictConfig):
    """Instantiate model using Hydra's instantiate functionality."""
    # Используем Hydra для создания объекта модели из конфигурации
    model = hydra.utils.instantiate(cfg.model)
    logger.info(f"Instantiated model: {type(model).__name__}")
    logger.info(f"Model parameters: {OmegaConf.to_container(cfg.model, resolve=True)}")
    return model


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function using Hydra for configuration management."""
    # Hydra автоматически инициализирует конфигурацию из файлов
    logger.info("=" * 50)
    logger.info("Training with Hydra Configuration")
    logger.info("=" * 50)
    
    # Выводим полную конфигурацию
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Валидация конфигурации
    validate_config(cfg)
    
    # Загрузка данных
    logger.info(f"Loading features from {cfg.paths.features}")
    logger.info(f"Loading labels from {cfg.paths.labels}")
    
    X = pd.read_csv(cfg.paths.features)
    y = pd.read_csv(cfg.paths.labels).squeeze()
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Labels shape: {y.shape}")
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state,
        stratify=y if cfg.data.stratify else None,
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Создание модели с помощью Hydra
    model = instantiate_model(cfg)
    
    # Настройка MLflow
    mlflow.set_tracking_uri(cfg.training.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.training.experiment_name)
    
    # Обучение модели
    logger.info("Training model...")
    model.fit(X_train, y_train)
    
    # Предсказания
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Вычисление метрик
    metrics = {
        "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
        "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
        "test_precision": float(precision_score(y_test, y_test_pred)),
        "test_recall": float(recall_score(y_test, y_test_pred)),
        "test_f1_score": float(f1_score(y_test, y_test_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, y_test_proba)),
    }
    
    # Логирование в MLflow
    if cfg.training.log_metrics:
        with mlflow.start_run(run_name=cfg.model_name):
            # Логируем параметры модели
            model_params = OmegaConf.to_container(cfg.model, resolve=True)
            for key, value in model_params.items():
                if key != "_target_":
                    mlflow.log_param(key, value)
            
            # Логируем метрики
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Логируем модель
            if cfg.training.save_model:
                mlflow.sklearn.log_model(model, "model")
    
    # Сохранение модели
    if cfg.training.save_model:
        model_path = Path(cfg.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
    
    # Сохранение метрик
    metrics_path = Path(cfg.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Вывод результатов
    logger.info("=" * 50)
    logger.info("Training Results:")
    logger.info("=" * 50)
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
