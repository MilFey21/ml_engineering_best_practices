"""Simplified utilities for ClearML integration - using ClearML API directly."""

from typing import Any, Dict, List, Optional

from clearml import Task
from loguru import logger


def compare_experiments(
    project_name: str,
    metric_name: str = "test_f1_score",
    top_n: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> list:
    """Compare experiments in a ClearML project - simplified version using ClearML API directly.

    Args:
        project_name: Name of the ClearML project
        metric_name: Name of the metric to compare
        top_n: Number of top experiments to return
        filters: Optional filters for experiments

    Returns:
        List of top experiments sorted by metric
    """
    # Get all tasks from the project using ClearML API directly
    tasks: List[Any] = Task.get_tasks(project_name=project_name, task_filter=filters or {})
    logger.info(f"Found {len(tasks)} tasks in project '{project_name}'")

    results = []
    for task in tasks:
        try:
            metric_value = None

            # Try to get from scalar metrics (from logger.report_scalar)
            scalar_metrics = task.get_last_scalar_metrics()
            if scalar_metrics:
                for title, series_dict in scalar_metrics.items():
                    if isinstance(series_dict, dict) and metric_name in series_dict:
                        value_dict = series_dict[metric_name]
                        if isinstance(value_dict, dict) and "last" in value_dict:
                            metric_value = value_dict["last"]
                            break

            # Try to get from configuration/parameters (from task.connect)
            if metric_value is None:
                try:
                    config = task.get_parameters()
                    if config:
                        if metric_name in config:
                            metric_value = config[metric_name]
                        elif "General" in config and metric_name in config["General"]:
                            metric_value = config["General"][metric_name]
                except Exception:
                    pass

            if metric_value is not None:
                try:
                    metric_value = float(metric_value)
                    results.append(
                        {
                            "task_id": task.id,
                            "task_name": task.name,
                            metric_name: metric_value,
                            "status": task.status,
                        }
                    )
                except (ValueError, TypeError):
                    pass
        except Exception:
            pass

    # Sort by metric value
    results.sort(key=lambda x: x.get(metric_name, 0), reverse=True)
    return results[:top_n]
