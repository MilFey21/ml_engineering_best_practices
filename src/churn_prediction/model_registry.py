"""Model registry with ClearML for churn prediction."""

from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from clearml import InputModel, OutputModel, Task
    from clearml.backend_api.session import Session

    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    # Dummy classes for type hints when clearml is not available
    InputModel = None  # type: ignore
    OutputModel = None  # type: ignore
    Task = None  # type: ignore
    Session = None  # type: ignore

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

# Import local config setup to ensure environment variables are set
try:
    from src.churn_prediction import clearml_local_config  # noqa: F401
except ImportError:
    pass

try:
    from src.churn_prediction.config import ChurnPredictionConfig
except ImportError:
    # Dummy class for type hints when config is not available
    class ChurnPredictionConfig:  # type: ignore
        def __init__(self):
            self.clearml_project = "Churn Prediction"


def register_model(
    task_id: str,
    model_path: Path,
    model_name: str = "churn_prediction_model",
    tags: list = None,
) -> str:
    """Register a model in ClearML Model Registry.

    Args:
        task_id: ClearML task ID that produced the model
        model_path: Path to the model file
        model_name: Name for the model in registry
        tags: Optional tags for the model

    Returns:
        Model ID
    """
    logger.info(f"Registering model from task {task_id}")

    # Get the task
    task = Task.get_task(task_id=task_id)

    # Create output model
    output_model = OutputModel(
        task=task,
        name=model_name,
        tags=tags or ["churn-prediction"],
    )

    # Add model files
    if model_path.is_dir():
        # Add all files in directory
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(model_path)
                output_model.update_weights(
                    weights_filename=str(file_path),
                    target_filename=str(relative_path),
                )
    else:
        # Single file
        output_model.update_weights(weights_filename=str(model_path))

    # Add metadata
    output_model.set_metadata(
        {
            "model_type": "sklearn",
            "framework": "scikit-learn",
            "task": "binary_classification",
        }
    )

    # Upload
    output_model.upload()
    model_id = output_model.id

    logger.success(f"Model registered with ID: {model_id}")
    logger.info(f"Model name: {model_name}")

    return model_id


def load_model_from_registry(
    model_name: str,
    version: str = "latest",
    output_path: Path = None,
) -> Path:
    """Load a model from ClearML Model Registry.

    Args:
        model_name: Name of the model in registry
        version: Version to load (default: "latest")
        output_path: Path to save the model (optional)

    Returns:
        Path to downloaded model
    """
    logger.info(f"Loading model {model_name} version {version}")

    # Get model from registry
    if version == "latest":
        model = InputModel(name=model_name)
    else:
        model = InputModel(name=model_name, version=version)

    # Download
    if output_path is None:
        output_path = Path("models") / "downloaded" / model_name

    output_path.mkdir(parents=True, exist_ok=True)
    model_path = model.get_local_copy(str(output_path))

    logger.success(f"Model downloaded to {model_path}")

    return Path(model_path)


def list_models(project_name: str = None):
    """List all models in registry.

    Args:
        project_name: Optional project name to filter
    """
    logger.info("Listing models in registry...")

    session = Session()
    project_name = project_name or "Churn Prediction"

    # Get project ID
    projects_response = session.send_request(
        service="projects",
        action="get_all_ex",
        json={"name": project_name},
    )

    projects_data = projects_response.json().get("data", {}).get("projects", [])

    if not projects_data:
        logger.warning(f"Project '{project_name}' not found")
        return

    project_id = projects_data[0]["id"]

    # Get models for the project
    models_response = session.send_request(
        service="models",
        action="get_all_ex",
        json={"project": [project_id]},
    )

    models_data = models_response.json().get("data", {}).get("models", [])

    if not models_data:
        logger.info(f"No models found in project '{project_name}'")
        logger.info("To register a model, run a training task first")
        return

    logger.info(f"Found {len(models_data)} models in project '{project_name}':")
    logger.info("")

    for model_data in models_data:
        model_name = model_data.get("name", "Unknown")
        model_id = model_data.get("id", "Unknown")

        # Handle created field - can be dict or string
        created_raw = model_data.get("created", {})
        if isinstance(created_raw, dict):
            created = created_raw.get("$date", "Unknown")
        else:
            created = created_raw if created_raw else "Unknown"

        tags = model_data.get("tags", [])

        # Handle framework field - can be dict or None
        framework_raw = model_data.get("framework")
        if isinstance(framework_raw, dict):
            framework = framework_raw.get("framework", "Unknown")
        else:
            framework = framework_raw if framework_raw else "Unknown"

        logger.info(f"  Model: {model_name}")
        logger.info(f"    ID: {model_id}")
        logger.info(f"    Framework: {framework}")
        logger.info(f"    Created: {created}")
        logger.info(f"    Tags: {tags}")
        logger.info("")


def compare_models(
    project_name: str = None,
    metric_name: str = "test_f1_score",
    top_n: int = 5,
    model_name_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Compare models in registry by metrics from their associated tasks.

    Args:
        project_name: Optional project name to filter models
        metric_name: Name of the metric to compare (e.g., "test_f1_score", "test_accuracy")
        top_n: Number of top models to return
        model_name_filter: Optional model name filter

    Returns:
        List of top models sorted by metric value, each containing:
        - model_id: Model ID
        - model_name: Model name
        - metric_value: Value of the metric
        - task_id: Associated task ID
        - task_name: Associated task name
        - created: Creation date
        - tags: Model tags
    """
    logger.info(f"Comparing models by metric: {metric_name}")

    session = Session()
    project_name = project_name or "Churn Prediction"

    # Get project ID
    projects_response = session.send_request(
        service="projects",
        action="get_all_ex",
        json={"name": project_name},
    )

    projects_data = projects_response.json().get("data", {}).get("projects", [])

    if not projects_data:
        logger.warning(f"Project '{project_name}' not found")
        return []

    project_id = projects_data[0]["id"]

    # Get models for the project
    models_response = session.send_request(
        service="models",
        action="get_all_ex",
        json={"project": [project_id]},
    )

    models_data = models_response.json().get("data", {}).get("models", [])

    if not models_data:
        logger.info(f"No models found in project '{project_name}'")
        return []

    # Filter by model name if specified
    if model_name_filter:
        models_data = [
            m for m in models_data if model_name_filter.lower() in m.get("name", "").lower()
        ]

    results = []

    for model_data in models_data:
        try:
            model_id = model_data.get("id")
            model_name = model_data.get("name", "Unknown")

            # Handle created field - can be dict or string
            created_raw = model_data.get("created", {})
            if isinstance(created_raw, dict):
                created = created_raw.get("$date", "Unknown")
            else:
                created = created_raw if created_raw else "Unknown"

            tags = model_data.get("tags", [])

            # Get associated task ID from model - can be dict or string or None
            task_raw = model_data.get("task")
            if isinstance(task_raw, dict):
                task_id = task_raw.get("id")
            elif isinstance(task_raw, str):
                task_id = task_raw
            else:
                task_id = None

            if not task_id:
                logger.debug(f"Model {model_name} has no associated task, skipping")
                continue

            # Get task to extract metrics
            try:
                task = Task.get_task(task_id=task_id)
            except Exception as e:
                logger.warning(f"Could not get task {task_id} for model {model_name}: {e}")
                continue

            metric_value = None

            # Method 1: Try to get from scalar metrics (from logger.report_scalar)
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
                # Ensure metric_value is numeric
                try:
                    metric_value = float(metric_value)
                    results.append(
                        {
                            "model_id": model_id,
                            "model_name": model_name,
                            metric_name: metric_value,
                            "task_id": task_id,
                            "task_name": task.name,
                            "created": created,
                            "tags": tags,
                        }
                    )
                except (ValueError, TypeError):
                    logger.warning(
                        f"Metric {metric_name} for model {model_name} is not numeric: {metric_value}"
                    )
            else:
                logger.debug(f"Metric {metric_name} not found for model {model_name}")

        except Exception as e:
            logger.warning(f"Failed to process model {model_data.get('name', 'Unknown')}: {e}")

    # Sort by metric value (descending)
    results.sort(key=lambda x: x.get(metric_name, 0), reverse=True)

    logger.info(f"Found {len(results)} models with metric {metric_name}")

    return results[:top_n]


def register_existing_model(
    model_path: Path,
    model_name: str = "churn_prediction_model",
    task_id: str = None,
):
    """Register an existing model file in Model Registry.

    Args:
        model_path: Path to the model file
        model_name: Name for the model in registry
        task_id: Optional task ID to associate with the model
    """
    logger.info(f"Registering model from file: {model_path}")

    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return None

    # If task_id is provided, use it; otherwise create a dummy task
    if task_id:
        task = Task.get_task(task_id=task_id)
    else:
        # Create a temporary task for model registration
        task = Task.create(
            project_name="Churn Prediction",
            task_name=f"Model Registration: {model_name}",
            task_type=Task.TaskTypes.inference,
        )
        task.close()

    model_id = register_model(
        task_id=task.id,
        model_path=model_path,
        model_name=model_name,
    )

    return model_id


def main():
    """Main function."""
    import sys
    import typer

    app = typer.Typer(help="ClearML Model Registry CLI")
    config = ChurnPredictionConfig()

    @app.command()
    def list():
        """List all models in registry."""
        list_models(config.clearml_project)

    @app.command()
    def register(
        model_path: Path = typer.Argument(..., help="Path to model file"),
        model_name: str = typer.Option("churn_prediction_model", help="Name for the model"),
        task_id: str = typer.Option(None, help="Optional task ID to associate with model"),
    ):
        """Register an existing model file in Model Registry."""
        register_existing_model(model_path, model_name, task_id)

    @app.command()
    def compare(
        metric_name: str = typer.Option("test_f1_score", help="Metric name to compare"),
        top_n: int = typer.Option(5, help="Number of top models to return"),
        project_name: str = typer.Option(None, help="Project name to filter models"),
    ):
        """Compare models by metrics."""
        import json
        from rich.console import Console
        from rich.table import Table

        proj_name = project_name or config.clearml_project
        results = compare_models(
            project_name=proj_name,
            metric_name=metric_name,
            top_n=top_n,
        )

        if not results:
            logger.info("No models found with the specified metric")
            return

        console = Console()
        table = Table(title=f"Top {len(results)} Models by {metric_name}")
        table.add_column("Model Name", style="cyan")
        table.add_column(metric_name, style="magenta", justify="right")
        table.add_column("Task Name", style="green")
        table.add_column("Tags", style="yellow")

        for result in results:
            tags_str = ", ".join(result.get("tags", []))[:30] or "None"
            table.add_row(
                result["model_name"],
                f"{result[metric_name]:.4f}",
                result["task_name"],
                tags_str,
            )

        console.print(table)
        logger.info("\nDetailed results (JSON):")
        logger.info(json.dumps(results, indent=2, default=str))

    # Default to list if no command provided
    if len(sys.argv) == 1:
        list_models(config.clearml_project)
    else:
        app()


if __name__ == "__main__":
    main()
