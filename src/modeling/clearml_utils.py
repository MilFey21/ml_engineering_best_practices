"""Utilities for ClearML integration: decorators, context managers, and experiment helpers."""

from contextlib import contextmanager
import functools
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from clearml import Logger, Task
from loguru import logger


class ClearMLExperiment:
    """Context manager for ClearML experiments."""

    def __init__(
        self,
        project_name: str,
        task_name: str,
        task_type: Any = Task.TaskTypes.training,
        tags: Optional[list] = None,
    ):
        """Initialize ClearML experiment context manager.

        Args:
            project_name: Name of the ClearML project
            task_name: Name of the experiment task
            task_type: Type of task (default: training)
            tags: Optional list of tags for the experiment
        """
        self.project_name = project_name
        self.task_name = task_name
        self.task_type = task_type
        self.tags = tags or []
        self.task: Optional[Task] = None
        self.logger: Optional[Logger] = None

    def __enter__(self):
        """Enter the context and initialize ClearML task."""
        self.task = Task.init(
            project_name=self.project_name,
            task_name=self.task_name,
            task_type=self.task_type,
            tags=self.tags,
        )
        assert self.task is not None  # Task.init always returns a Task
        self.logger = self.task.get_logger()
        logger.info(f"Started ClearML experiment: {self.task_name} (ID: {self.task.id})")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and close the task."""
        if self.task:
            if exc_type is not None:
                logger.error(f"Experiment {self.task_name} failed: {exc_val}")
                self.task.mark_failed()
            else:
                logger.info(f"Completed ClearML experiment: {self.task_name}")
            self.task.close()

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to ClearML.

        Args:
            params: Dictionary of parameters to log
        """
        if self.task:
            self.task.connect(params)

    def log_metrics(self, metrics: Dict[str, float], iteration: int = 0):
        """Log metrics to ClearML.

        Args:
            metrics: Dictionary of metrics to log
            iteration: Iteration number (default: 0)
        """
        if self.task:
            # Log metrics as configuration (appears in UI)
            self.task.connect(metrics)

            # Also log as scalars for plots
            if self.logger:
                for metric_name, metric_value in metrics.items():
                    self.logger.report_scalar(
                        "Metrics", metric_name, metric_value, iteration=iteration
                    )

    def log_artifact(self, name: str, artifact_path: Path):
        """Upload artifact to ClearML.

        Args:
            name: Name of the artifact
            artifact_path: Path to the artifact file
        """
        if self.task:
            self.task.upload_artifact(name=name, artifact_object=str(artifact_path))

    def log_confusion_matrix(self, cm, class_names=None, title="Confusion Matrix"):
        """Log confusion matrix as image and table to ClearML.

        Args:
            cm: Confusion matrix array (from sklearn.metrics.confusion_matrix)
            class_names: List of class names for labels
            title: Title for the confusion matrix
        """
        import tempfile

        import matplotlib.pyplot as plt
        import seaborn as sns

        if self.logger:
            # Create confusion matrix as table
            if class_names is None:
                class_names = [f"Class {i}" for i in range(len(cm))]

            # Log as table
            table_data = []
            for i, row in enumerate(cm):
                table_data.append([class_names[i]] + row.tolist())

            self.logger.report_table(
                title=title,
                series="Confusion Matrix",
                table_plot=table_data,
                iteration=0,
            )

            # Create and log as image (heatmap)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={"label": "Count"},
                ax=ax,
            )
            ax.set_title(title)
            ax.set_ylabel("True Label")
            ax.set_xlabel("Predicted Label")
            plt.tight_layout()

            # Save to temporary file and log
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
                plt.savefig(tmp_path, dpi=150, bbox_inches="tight")
                plt.close()

                # Log image
                self.logger.report_image(
                    title=title,
                    series="Confusion Matrix",
                    iteration=0,
                    local_path=str(tmp_path),
                )

                # Clean up
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

    def log_model(self, name: str, model_path: Path):
        """Upload model to ClearML.

        Args:
            name: Name of the model
            model_path: Path to the model file
        """
        if self.task:
            # Upload model as artifact
            self.task.upload_artifact(name=name, artifact_object=str(model_path))

            # Also register as output model for model registry
            try:
                self.task.register_artifact(name=name, artifact_object=str(model_path))
            except Exception:
                # If register_artifact doesn't work, just use upload_artifact
                pass

    def add_tags(self, tags: list):
        """Add tags to the experiment.

        Args:
            tags: List of tags to add
        """
        if self.task:
            self.task.add_tags(tags)


def clearml_experiment(
    project_name: str,
    task_name: Optional[str] = None,
    task_type: Any = Task.TaskTypes.training,
    tags: Optional[list] = None,
    log_params: bool = True,
    log_return_value: bool = False,
):
    """Decorator for automatic ClearML experiment logging.

    Args:
        project_name: Name of the ClearML project
        task_name: Name of the experiment task (if None, uses function name)
        task_type: Type of task (default: training)
        tags: Optional list of tags for the experiment
        log_params: Whether to log function parameters
        log_return_value: Whether to log return value as metric

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate task name if not provided
            exp_task_name = task_name or f"{func.__name__}_{id(kwargs)}"

            # Extract parameters for logging
            params = {}
            if log_params:
                # Log keyword arguments
                params.update(kwargs)
                # Try to extract meaningful args if they're named
                import inspect

                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                params.update(dict(bound_args.arguments))

            with ClearMLExperiment(
                project_name=project_name,
                task_name=exp_task_name,
                task_type=task_type,
                tags=tags,
            ) as exp:
                # Log parameters
                if params:
                    exp.log_params(params)

                # Execute function
                result = func(*args, **kwargs)

                # Log return value if it's a dict with metrics
                if log_return_value and isinstance(result, dict):
                    metrics = {k: v for k, v in result.items() if isinstance(v, (int, float))}
                    if metrics:
                        exp.log_metrics(metrics)

                return result

        return wrapper

    return decorator


def compare_experiments(
    project_name: str,
    metric_name: str = "test_f1_score",
    top_n: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> list:
    """Compare experiments in a ClearML project.

    Args:
        project_name: Name of the ClearML project
        metric_name: Name of the metric to compare
        top_n: Number of top experiments to return
        filters: Optional filters for experiments

    Returns:
        List of top experiments sorted by metric
    """
    from clearml import Task

    # Get all tasks from the project
    tasks: List[Any] = Task.get_tasks(project_name=project_name, task_filter=filters or {})

    results = []
    for task in tasks:
        try:
            metric_value = None

            # Method 1: Try to get from scalar metrics (from logger.report_scalar)
            # Structure: {"Metrics": {"metric_name": {"last": value}}}
            scalar_metrics = task.get_last_scalar_metrics()
            if scalar_metrics:
                for title, series_dict in scalar_metrics.items():
                    if isinstance(series_dict, dict) and metric_name in series_dict:
                        value_dict = series_dict[metric_name]
                        if isinstance(value_dict, dict) and "last" in value_dict:
                            metric_value = value_dict["last"]
                            break

            # Method 2: Try to get from configuration/parameters (from task.connect)
            if metric_value is None:
                try:
                    config = task.get_parameters()
                    # Parameters can be nested, try both direct access and nested access
                    if config:
                        # Try direct access
                        if metric_name in config:
                            metric_value = config[metric_name]
                        # Try accessing through "General" section
                        elif "General" in config and metric_name in config["General"]:
                            metric_value = config["General"][metric_name]
                        # Try accessing through nested structure
                        else:
                            # Flatten nested structure
                            def find_in_dict(d, key):
                                if isinstance(d, dict):
                                    if key in d:
                                        return d[key]
                                    for v in d.values():
                                        result = find_in_dict(v, key)
                                        if result is not None:
                                            return result
                                return None

                            metric_value = find_in_dict(config, metric_name)
                except Exception as e:
                    logger.debug(f"Could not get parameters for task {task.id}: {e}")

            if metric_value is not None:
                # Ensure metric_value is numeric
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
                    logger.warning(
                        f"Metric {metric_name} for task {task.id} is not numeric: {metric_value}"
                    )
        except Exception as e:
            logger.warning(f"Failed to get metrics for task {task.id}: {e}")

    # Sort by metric value
    results.sort(key=lambda x: x.get(metric_name, 0), reverse=True)

    return results[:top_n]


def search_experiments(
    project_name: str,
    search_query: Optional[str] = None,
    tags: Optional[list] = None,
    status: Optional[str] = None,
) -> list:
    """Search experiments in a ClearML project.

    Args:
        project_name: Name of the ClearML project
        search_query: Optional search query for task names
        tags: Optional list of tags to filter by
        status: Optional status filter (e.g., 'completed', 'failed')

    Returns:
        List of matching experiments
    """
    from clearml import Task

    # Build task filter
    task_filter: Dict[str, Any] = {}
    if tags:
        task_filter["tags"] = tags
    if status:
        task_filter["status"] = str(status)

    # Get tasks
    tasks: List[Any] = Task.get_tasks(project_name=project_name, task_filter=task_filter)

    results = []
    for task in tasks:
        # Filter by search query if provided
        if search_query and search_query.lower() not in task.name.lower():
            continue

        results.append(
            {
                "task_id": task.id,
                "task_name": task.name,
                "status": task.status,
                "tags": task.tags,
                "created": task.created,
            }
        )

    return results


@contextmanager
def clearml_context(
    project_name: str,
    task_name: str,
    task_type: Any = Task.TaskTypes.training,
    tags: Optional[list] = None,
):
    """Context manager for ClearML experiments (alias for ClearMLExperiment).

    Args:
        project_name: Name of the ClearML project
        task_name: Name of the experiment task
        task_type: Type of task (default: training)
        tags: Optional list of tags for the experiment

    Yields:
        ClearMLExperiment instance
    """
    with ClearMLExperiment(project_name, task_name, task_type, tags) as exp:
        yield exp
