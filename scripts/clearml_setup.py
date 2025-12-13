"""Script for setting up ClearML configuration."""

import os
from pathlib import Path

from loguru import logger


def setup_clearml():
    """Set up ClearML configuration."""
    logger.info("Setting up ClearML...")

    # Check if ClearML credentials file exists
    clearml_config_path = Path.home() / ".clearml" / "clearml.conf"

    if clearml_config_path.exists():
        logger.info(f"ClearML config file already exists at {clearml_config_path}")
        logger.info("To reconfigure, run: clearml-init")
        return

    logger.info("ClearML config file not found.")
    logger.info("To set up ClearML, run the following command:")
    logger.info("  clearml-init")
    logger.info("")
    logger.info("This will prompt you for:")
    logger.info("  1. ClearML server URL (or use default: https://app.clear.ml)")
    logger.info("  2. API credentials (Access Key and Secret Key)")
    logger.info("")
    logger.info("You can get your credentials from:")
    logger.info(
        "  - ClearML Web UI: https://app.clear.ml -> Settings -> Workspace -> Create new credentials"
    )
    logger.info("")
    logger.info("For local development, you can use the default server or set up your own:")
    logger.info("  - Default: https://app.clear.ml")
    logger.info("  - Local: Run 'clearml-server' docker container")

    # Check environment variables
    api_access_key = os.getenv("CLEARML_API_ACCESS_KEY")
    api_secret_key = os.getenv("CLEARML_API_SECRET_KEY")
    api_server = os.getenv("CLEARML_API_HOST", "https://app.clear.ml")

    if api_access_key and api_secret_key:
        logger.info("\nEnvironment variables detected:")
        logger.info(f"  CLEARML_API_HOST: {api_server}")
        logger.info(f"  CLEARML_API_ACCESS_KEY: {'*' * len(api_access_key)}")
        logger.info(f"  CLEARML_API_SECRET_KEY: {'*' * len(api_secret_key)}")
        logger.info("\nClearML will use these environment variables for authentication.")
    else:
        logger.info("\nNo environment variables found. Using config file method.")


if __name__ == "__main__":
    setup_clearml()
