"""Script to run DVC pipeline with monitoring."""

from pathlib import Path
import subprocess
import sys
import time

from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.monitor import PipelineMonitor, send_notification  # noqa: E402


def run_dvc_stage(stage_name: str, monitor: PipelineMonitor, skip_if_exists: bool = False):
    """Run a DVC pipeline stage with monitoring.

    Args:
        stage_name: Name of the DVC stage
        monitor: Pipeline monitor instance
        skip_if_exists: If True, skip stage if output already exists
    """
    start_time = time.time()
    monitor.log_stage_start(stage_name)

    try:
        # Check if we should skip this stage
        if skip_if_exists and stage_name == "download_data":
            data_file = Path("data/raw/customer_churn.csv")
            if data_file.exists():
                logger.info(f"Skipping {stage_name} - output already exists")
                duration = time.time() - start_time
                monitor.log_stage_complete(stage_name, duration)
                return True

        # Run DVC stage
        subprocess.run(
            ["dvc", "repro", stage_name],
            capture_output=True,
            text=True,
            check=True,
        )

        duration = time.time() - start_time
        monitor.log_stage_complete(stage_name, duration)

        logger.info(f"Stage {stage_name} completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        error_msg = e.stderr or e.stdout or str(e)

        # If download_data fails and data exists, continue
        if stage_name == "download_data" and Path("data/raw/customer_churn.csv").exists():
            logger.warning(f"Stage {stage_name} failed but output exists, continuing...")
            monitor.log_stage_complete(stage_name, duration)
            return True

        monitor.log_stage_failed(stage_name, error_msg)
        send_notification(f"Pipeline stage {stage_name} failed: {error_msg}", "error")
        logger.error(f"Stage {stage_name} failed: {error_msg}")
        return False


def main():
    """Run the complete DVC pipeline with monitoring."""
    logger.info("Starting DVC pipeline execution with monitoring")

    monitor = PipelineMonitor()

    # Check if data exists - determine which stages to run
    data_file = Path("data/raw/customer_churn.csv")
    data_dvc_file = Path("data/raw/customer_churn.csv.dvc")

    # Try to get data from DVC cache if .dvc file exists but data doesn't
    if not data_file.exists() and data_dvc_file.exists():
        logger.info(
            "Data file not found but .dvc file exists. Attempting to pull from DVC cache..."
        )
        try:
            subprocess.run(
                ["dvc", "pull", "data/raw/customer_churn.csv.dvc"],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("Successfully pulled data from DVC cache")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to pull from DVC cache: {e.stderr}")
            logger.warning("Will attempt to download data instead")

    # Determine which stages to run
    stages = []

    # Add download_data only if raw data doesn't exist
    if not data_file.exists():
        logger.info("Raw data not found, will attempt to download")
        stages.append("download_data")
    else:
        logger.info("Raw data exists, skipping download_data stage")

    # Always run these stages (DVC will handle caching)
    stages.extend(
        [
            "process_features",
            "train_model",
            "evaluate_model",
        ]
    )

    # Run all stages
    for stage in stages:
        # Skip download_data if data already exists
        skip_if_exists = stage == "download_data"
        success = run_dvc_stage(stage, monitor, skip_if_exists=skip_if_exists)
        if not success:
            # For download_data, if it fails but data exists, continue
            if stage == "download_data":
                if data_file.exists():
                    logger.warning("Download failed but data exists, continuing pipeline...")
                    continue
                else:
                    logger.error("Download failed and data doesn't exist. Cannot continue.")
                    logger.error("Please either:")
                    logger.error(
                        "  1. Set up Kaggle API credentials (KAGGLE_USERNAME and KAGGLE_KEY)"
                    )
                    logger.error("  2. Or manually place data/raw/customer_churn.csv")
                    logger.error("  3. Or run: dvc pull to get data from DVC cache")
                    sys.exit(1)
            logger.error(f"Pipeline failed at stage: {stage}")
            sys.exit(1)

    # Get and save summary
    summary = monitor.get_summary()
    summary_path = Path("reports/pipeline_summary.json")
    monitor.save_summary(summary_path)

    logger.info("Pipeline execution completed successfully")
    logger.info(f"Summary saved to {summary_path}")

    # Send success notification
    send_notification(
        f"Pipeline completed successfully. "
        f"Stages: {summary['total_stages_completed']}/{summary['total_stages_started']}, "
        f"Duration: {summary['total_duration_seconds']:.2f}s",
        "info",
    )


if __name__ == "__main__":
    main()
