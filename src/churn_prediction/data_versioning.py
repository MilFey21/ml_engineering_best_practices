"""Data versioning with ClearML Dataset for churn prediction."""

import os
from pathlib import Path

from clearml import Dataset
from loguru import logger

# Import local config setup to ensure environment variables are set
from src.churn_prediction import clearml_local_config  # noqa: F401
from src.churn_prediction.config import ChurnPredictionConfig


def check_clearml_config():
    """Check and warn if ClearML is not configured for local server."""
    api_host = os.getenv("CLEARML_API_HOST", "")
    files_host = os.getenv("CLEARML_FILES_HOST", "")

    if not api_host.startswith("http://localhost"):
        logger.warning(
            "ClearML API_HOST not set for local server.\n"
            "Current: {}\n"
            "Set CLEARML_API_HOST=http://localhost:8008".format(api_host or "not set")
        )

    if not files_host.startswith("http://localhost"):
        logger.warning(
            "ClearML FILES_HOST not set for local server.\n"
            "Current: {}\n"
            "Set CLEARML_FILES_HOST=http://localhost:8081".format(files_host or "not set")
        )


def create_dataset_version(config: ChurnPredictionConfig, version: str = "v1.0") -> str:
    """Create a versioned dataset in ClearML.

    Args:
        config: Configuration object
        version: Version string for the dataset

    Returns:
        Dataset ID
    """
    logger.info(f"Creating ClearML dataset version: {version}")

    # Check configuration
    check_clearml_config()

    # Create dataset
    dataset = Dataset.create(
        dataset_name="Telco Customer Churn",
        dataset_project=config.clearml_project,
    )

    # Add files from processed data directory
    features_path = config.processed_data_dir / config.features_file
    labels_path = config.processed_data_dir / config.labels_file
    raw_data_path = config.raw_data_dir / "customer_churn.csv"

    if features_path.exists():
        dataset.add_files(path=str(features_path), dataset_path="features.csv")
        logger.info(f"Added {features_path} to dataset")

    if labels_path.exists():
        dataset.add_files(path=str(labels_path), dataset_path="labels.csv")
        logger.info(f"Added {labels_path} to dataset")

    if raw_data_path.exists():
        dataset.add_files(path=str(raw_data_path), dataset_path="raw/customer_churn.csv")
        logger.info(f"Added {raw_data_path} to dataset")

    # Add metadata
    dataset.set_metadata(
        {
            "version": version,
            "dataset_name": config.dataset_name,
            "source": "Kaggle",
            "kaggle_dataset": config.dataset_kaggle,
        }
    )

    # Upload files first, then finalize
    logger.info("Uploading dataset files...")
    dataset.upload()
    logger.info("Files uploaded successfully")

    # Finalize dataset
    logger.info("Finalizing dataset...")
    dataset.finalize()
    dataset_id = dataset.id

    logger.success(f"Dataset created with ID: {dataset_id}")
    logger.info(f"Dataset version: {version}")

    return dataset_id


def get_dataset_version(dataset_id: str, output_path: Path):
    """Download a specific dataset version.

    Args:
        dataset_id: ClearML dataset ID
        output_path: Path to save downloaded dataset
    """
    logger.info(f"Downloading dataset ID: {dataset_id}")

    dataset = Dataset.get(dataset_id=dataset_id)
    dataset.get_mutable_copy(str(output_path))

    logger.success(f"Dataset downloaded to {output_path}")


def main():
    """Main function."""
    config = ChurnPredictionConfig()
    create_dataset_version(config)


if __name__ == "__main__":
    main()
