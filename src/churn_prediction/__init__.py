"""Churn prediction module with ClearML integration."""

# Auto-configure ClearML for local server when module is imported
from src.churn_prediction import clearml_local_config  # noqa: F401
