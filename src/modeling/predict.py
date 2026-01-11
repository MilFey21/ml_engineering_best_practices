"""Prediction module for model inference.

This module provides functionality to perform inference using trained models
for customer churn prediction.
"""

from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
):
    """Perform inference using a trained model.

    Args:
        features_path: Path to the input features CSV file.
        model_path: Path to the trained model file.
        predictions_path: Path where predictions will be saved.

    Returns:
        None. Predictions are saved to predictions_path.
    """
    logger.info("Performing inference for model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")


if __name__ == "__main__":
    app()
