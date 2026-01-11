import os
from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from src.config import RAW_DATA_DIR

app = typer.Typer()

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


@app.command()
def main(
    output_path: str = typer.Argument(default=str(RAW_DATA_DIR / "customer_churn.csv")),
    kaggle_username: str = typer.Option(None, "--kaggle-username"),
    kaggle_key: str = typer.Option(None, "--kaggle-key"),
):
    """Download Telco Customer Churn dataset from Kaggle.

    Can use environment variables KAGGLE_USERNAME and KAGGLE_KEY,
    or pass them as command-line arguments.
    """
    output_path_obj = Path(output_path).resolve()
    # Check if output already exists
    if output_path_obj.exists():
        logger.info(f"Data file already exists at {output_path_obj}, skipping download")
        # Create metrics file for DVC
        metrics_file = Path("metrics/data_download.json")
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        import json

        metrics_file.write_text(
            json.dumps({"status": "skipped", "reason": "file_exists"}, indent=2)
        )
        return

    # Try to pull from DVC cache if .dvc file exists
    dvc_file = output_path_obj.with_suffix(output_path_obj.suffix + ".dvc")
    if dvc_file.exists():
        logger.info(f"DVC file found at {dvc_file}, attempting to pull from cache...")
        import subprocess

        try:
            subprocess.run(
                ["dvc", "pull", str(dvc_file)],
                capture_output=True,
                text=True,
                check=True,
            )
            if output_path_obj.exists():
                logger.info(f"Successfully pulled data from DVC cache to {output_path_obj}")
                # Create metrics file for DVC
                metrics_file = Path("metrics/data_download.json")
                metrics_file.parent.mkdir(parents=True, exist_ok=True)
                import json

                metrics_file.write_text(json.dumps({"status": "pulled_from_cache"}, indent=2))
                return
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to pull from DVC cache: {e.stderr}")
            logger.info("Will attempt to download data instead")

    logger.info("Downloading Telco Customer Churn dataset from Kaggle...")

    try:
        # Import Kaggle API only when needed (after checking if file exists)
        from kaggle.api.kaggle_api_extended import KaggleApi

        # Ensure output directory exists
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

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
        temp_dir = output_path_obj.parent / "temp_kaggle"
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
        df.to_csv(output_path_obj, index=False)

        logger.info(f"Dataset downloaded successfully. Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"First 5 records:\n{df.head()}")
        logger.success(f"Dataset saved to {output_path_obj}")

        # Create metrics file for DVC
        metrics_file = Path("metrics/data_download.json")
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        import json

        metrics_file.write_text(
            json.dumps(
                {
                    "status": "downloaded",
                    "shape": df.shape,
                    "columns": list(df.columns),
                },
                indent=2,
            )
        )

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
