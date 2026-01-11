"""Dataset download module.

This module provides functionality to download the Telco Customer Churn
dataset from Kaggle using the Kaggle API.
"""

import os
from pathlib import Path
from typing import Optional

from loguru import logger
import pandas as pd
import typer

from src.config import RAW_DATA_DIR

app = typer.Typer(rich_markup_mode=None)

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


@app.command()
def main(
    output_path: Path,
    kaggle_username: Optional[str] = None,
    kaggle_key: Optional[str] = None,
):
    """Download Telco Customer Churn dataset from Kaggle.

    Downloads the Telco Customer Churn dataset from Kaggle and saves it
    to the specified output path. Credentials can be provided via environment
    variables (KAGGLE_USERNAME and KAGGLE_KEY) or as command-line arguments.

    Args:
        output_path: Path where the downloaded dataset will be saved.
        kaggle_username: Optional Kaggle username (defaults to KAGGLE_USERNAME env var).
        kaggle_key: Optional Kaggle API key (defaults to KAGGLE_KEY env var).

    Returns:
        None. Dataset is saved to output_path.

    Raises:
        FileNotFoundError: If no CSV file is found in the downloaded dataset.
        Exception: If Kaggle API authentication fails.
    """
    logger.info("Downloading Telco Customer Churn dataset from Kaggle...")

    # Lazy import to allow CLI help without Kaggle installed
    from kaggle.api.kaggle_api_extended import KaggleApi

    try:
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Set Kaggle credentials from environment variables or command-line arguments
        username = kaggle_username or os.environ.get("KAGGLE_USERNAME")
        key = kaggle_key or os.environ.get("KAGGLE_KEY")

        if username and key:
            # Set credentials as environment variables for kaggle library
            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"] = key
            logger.info("Using Kaggle credentials from environment variables or arguments")
        else:
            logger.info("Using Kaggle credentials from kaggle.json file")

        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()

        # Download dataset
        dataset = "blastchar/telco-customer-churn"
        logger.info(f"Downloading dataset: {dataset}")

        # Download to a temporary directory first
        temp_dir = output_path.parent / "temp_kaggle"
        temp_dir.mkdir(exist_ok=True)

        api.dataset_download_files(
            dataset,
            path=str(temp_dir),
            unzip=True,
        )

        # Find the CSV file in the downloaded directory
        csv_files = list(temp_dir.glob("*.csv"))
        if not csv_files:
            # Try to find CSV in subdirectories
            csv_files = list(temp_dir.rglob("*.csv"))

        if not csv_files:
            raise FileNotFoundError("No CSV file found in downloaded dataset")

        # Use the first CSV file found (usually there's only one)
        source_csv = csv_files[0]
        logger.info(f"Found CSV file: {source_csv}")

        # Load and save to the target location
        df = pd.read_csv(source_csv)
        df.to_csv(output_path, index=False)

        logger.info(f"Dataset downloaded successfully. Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"First 5 records:\n{df.head()}")
        logger.success(f"Dataset saved to {output_path}")

        # Clean up temporary directory
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        logger.error(
            f"""Error downloading dataset: {e}
        Make sure Kaggle API credentials are configured.
        Options:
        1. Set environment variables: KAGGLE_USERNAME and KAGGLE_KEY
        2. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)"""
        )
        raise


if __name__ == "__main__":
    app()
