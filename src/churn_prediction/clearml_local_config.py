"""Utility to configure ClearML for local server."""

import os
from pathlib import Path

from loguru import logger


def setup_local_clearml_env():
    """Set up ClearML environment variables for local server.

    This function sets environment variables for local ClearML server.
    Environment variables take precedence over config file, so setting them
    will force ClearML to use local server even if config file points to cloud.
    """
    config_path = Path.home() / ".clearml" / "clearml.conf"

    # Always set environment variables for local server
    # This ensures that even if config file points to cloud server,
    # environment variables will override it
    os.environ["CLEARML_API_HOST"] = "http://localhost:8008"
    os.environ["CLEARML_FILES_HOST"] = "http://localhost:8081"
    os.environ["CLEARML_WEB_HOST"] = "http://localhost:8080"

    if config_path.exists():
        logger.info("ClearML config file exists, but environment variables will override it")
        logger.info("Using local server configuration:")
    else:
        logger.info("Configured ClearML for local server:")

    logger.info("  CLEARML_API_HOST=http://localhost:8008")
    logger.info("  CLEARML_FILES_HOST=http://localhost:8081")
    logger.info("  CLEARML_WEB_HOST=http://localhost:8080")
    logger.info("")
    logger.info("Note: Make sure ClearML Server is running:")
    logger.info("  pixi run clearml-server-start")
    logger.info("")
    logger.info("And configure credentials:")
    logger.info("  clearml-init")
    logger.info("  # Use http://localhost:8008 as API server")
    logger.info("  # Use http://localhost:8080 as Web UI")


# Auto-setup when module is imported
setup_local_clearml_env()
