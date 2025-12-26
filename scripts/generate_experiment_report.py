"""Script to generate experiment reports with visualizations."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger
from clearml import Task

from src.churn_prediction.model_registry import compare_models
from src.churn_prediction.config import ChurnPredictionConfig
from src.modeling.clearml_utils import compare_experiments


def download_task_plots(task_id: str, output_dir: Path, max_plots: int = 5) -> List[str]:
    """Download plots from a ClearML task.
    
    Args:
        task_id: ClearML task ID
        output_dir: Directory to save plots
        max_plots: Maximum number of plots to download
        
    Returns:
        List of relative paths to downloaded plot images
    """
    try:
        logger.info(f"Fetching task {task_id}...")
        task = Task.get_task(task_id=task_id)
        plot_paths = []
        
        # Create plots directory for this task
        task_plots_dir = output_dir / f"task_{task_id}"
        task_plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Get scalar metrics and create combined plot
        try:
            # Get scalar metrics - returns dict like {"Metrics": {"test_f1_score": {"last": value, "min": value, "max": value}}}
            scalar_metrics = task.get_last_scalar_metrics()
            logger.info(f"Found {len(scalar_metrics) if scalar_metrics else 0} scalar metric groups for task {task_id}")
            
            if scalar_metrics:
                import matplotlib
                matplotlib.use('Agg')  # Use non-interactive backend
                import matplotlib.pyplot as plt
                import numpy as np
                
                # Collect all metrics from "Metrics" group
                metrics_data = {}
                for title, series_dict in scalar_metrics.items():
                    if title == "Metrics" and isinstance(series_dict, dict):
                        for series_name, metric_info in series_dict.items():
                            if isinstance(metric_info, dict):
                                last_value = metric_info.get('last')
                                if last_value is not None:
                                    metrics_data[series_name] = last_value
                
                # Create combined metrics plot if we have metrics
                if metrics_data:
                    logger.info(f"Creating combined metrics plot with {len(metrics_data)} metrics")
                    
                    # Create figure with subplots
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                    
                    # Plot 1: Bar chart with all metrics
                    metric_names = list(metrics_data.keys())
                    metric_values = list(metrics_data.values())
                    colors = plt.cm.viridis(np.linspace(0, 1, len(metric_names)))
                    
                    bars = ax1.barh(metric_names, metric_values, color=colors, alpha=0.7)
                    ax1.set_xlabel('Metric Value', fontsize=12)
                    ax1.set_title('All Metrics', fontsize=14, fontweight='bold')
                    ax1.grid(True, alpha=0.3, axis='x')
                    
                    # Add value labels on bars
                    for i, (name, value) in enumerate(zip(metric_names, metric_values)):
                        ax1.text(value, i, f' {value:.4f}', va='center', fontsize=10)
                    
                    # Plot 2: Radar/spider chart or grouped bar chart
                    # Use grouped bar chart for better readability
                    x_pos = np.arange(len(metric_names))
                    ax2.bar(x_pos, metric_values, color=colors, alpha=0.7)
                    ax2.set_xticks(x_pos)
                    ax2.set_xticklabels([name.replace('_', ' ').title() for name in metric_names], rotation=45, ha='right')
                    ax2.set_ylabel('Metric Value', fontsize=12)
                    ax2.set_title('Metrics Comparison', fontsize=14, fontweight='bold')
                    ax2.grid(True, alpha=0.3, axis='y')
                    
                    # Add value labels on top of bars
                    for i, value in enumerate(metric_values):
                        ax2.text(i, value, f'{value:.4f}', ha='center', va='bottom', fontsize=9)
                    
                    plt.tight_layout()
                    
                    # Save combined plot
                    plot_filename = f"combined_metrics.png"
                    plot_path = task_plots_dir / plot_filename
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    if plot_path.exists():
                        plot_paths.append(f"task_{task_id}/{plot_filename}")
                        logger.success(f"Saved combined metrics plot: {plot_path}")
                    else:
                        logger.warning(f"Combined plot file was not created: {plot_path}")
                        
        except Exception as e:
            logger.warning(f"Could not create combined metrics plot for task {task_id}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        # Get confusion matrix
        confusion_matrix_found = False
        try:
            # Method 1: Try to get confusion matrix from artifacts
            try:
                artifacts_dict = task.artifacts
                if artifacts_dict:
                    logger.info(f"Found {len(artifacts_dict)} artifacts for task {task_id}")
                    for artifact_name, artifact_obj in artifacts_dict.items():
                        if 'confusion' in artifact_name.lower():
                            try:
                                artifact_path_str = artifact_obj.get_local_copy()
                                if artifact_path_str:
                                    artifact_path = Path(artifact_path_str)
                                    if artifact_path.exists():
                                        # Check if it's a CSV file (confusion matrix saved as CSV)
                                        if artifact_path.suffix.lower() == '.csv':
                                            # Read CSV and create confusion matrix visualization
                                            import pandas as pd
                                            import seaborn as sns
                                            import matplotlib
                                            matplotlib.use('Agg')  # Use non-interactive backend
                                            import matplotlib.pyplot as plt
                                            
                                            cm_df = pd.read_csv(artifact_path, index_col=0)
                                            cm = cm_df.values
                                            
                                            # Create confusion matrix heatmap
                                            fig, ax = plt.subplots(figsize=(10, 8))
                                            sns.heatmap(
                                                cm,
                                                annot=True,
                                                fmt="d",
                                                cmap="Blues",
                                                xticklabels=cm_df.columns,
                                                yticklabels=cm_df.index,
                                                cbar_kws={"label": "Count"},
                                                ax=ax,
                                                square=True,
                                                linewidths=0.5,
                                            )
                                            ax.set_title(f"Confusion Matrix - {task.name}", fontsize=14, fontweight='bold')
                                            ax.set_ylabel("True Label", fontsize=12)
                                            ax.set_xlabel("Predicted Label", fontsize=12)
                                            plt.tight_layout()
                                            
                                            cm_filename = "confusion_matrix.png"
                                            cm_path = task_plots_dir / cm_filename
                                            plt.savefig(cm_path, dpi=150, bbox_inches='tight')
                                            plt.close()
                                            
                                            if cm_path.exists():
                                                plot_paths.append(f"task_{task_id}/{cm_filename}")
                                                confusion_matrix_found = True
                                                logger.success(f"Created confusion matrix from CSV: {cm_path}")
                                        elif str(artifact_path).lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                                            # Copy image file directly
                                            cm_filename = "confusion_matrix.png"
                                            cm_path = task_plots_dir / cm_filename
                                            import shutil
                                            shutil.copy(artifact_path, cm_path)
                                            if cm_path.exists():
                                                plot_paths.append(f"task_{task_id}/{cm_filename}")
                                                confusion_matrix_found = True
                                                logger.success(f"Copied confusion matrix image: {cm_path}")
                            except Exception as e_art:
                                logger.debug(f"Could not process confusion matrix artifact {artifact_name}: {e_art}")
            except Exception as e_artifacts:
                logger.debug(f"Could not get artifacts: {e_artifacts}")
            
            # Method 2: Try to get confusion matrix from tables (if logged as table)
            if not confusion_matrix_found:
                try:
                    # Try to get confusion matrix from reported tables
                    # Note: ClearML may store confusion matrix as a table
                    logger.debug("Trying to get confusion matrix from tables...")
                    # This would require accessing ClearML's table API, which may not be directly available
                    # For now, we'll skip this and rely on artifacts
                except Exception as e_table:
                    logger.debug(f"Could not get confusion matrix from tables: {e_table}")
            
            # Method 3: Create confusion matrix from metrics if available
            if not confusion_matrix_found:
                try:
                    # Try to reconstruct confusion matrix from metrics if possible
                    # This is a fallback - ideally confusion matrix should be in artifacts
                    logger.debug("Could not find confusion matrix in artifacts")
                except Exception as e_fallback:
                    logger.debug(f"Could not create confusion matrix from metrics: {e_fallback}")
                    
        except Exception as e:
            logger.warning(f"Could not get confusion matrix for task {task_id}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        logger.info(f"Downloaded {len(plot_paths)} plots/images for task {task_id}")
        return plot_paths
        
    except Exception as e:
        logger.warning(f"Could not download plots from task {task_id}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return []


def generate_comparison_table(results: List[Dict[str, Any]], metric_name: str) -> str:
    """Generate a markdown table from comparison results."""
    if not results:
        return "No results found."
    
    table = "| Model Name | Task Name | " + metric_name + " | Created | Tags |\n"
    table += "|------------|-----------|" + "-" * (len(metric_name) + 2) + "|---------|------|\n"
    
    for result in results:
        model_name = result.get("model_name", "N/A")
        task_name = result.get("task_name", "N/A")
        metric_value = result.get(metric_name, 0)
        created = result.get("created", "N/A")
        tags = ", ".join(result.get("tags", []))[:30] or "None"
        
        # Format date
        if isinstance(created, str) and "T" in created:
            created = created.split("T")[0]
        
        table += f"| {model_name} | {task_name} | {metric_value:.4f} | {created} | {tags} |\n"
    
    return table


def generate_experiment_report(
    output_path: Path = Path("reports/experiments/experiment_report.md"),
    metrics: List[str] = None,
    top_n: int = 10,
    include_plots: bool = True,
    plots_dir: Optional[Path] = None,
):
    """Generate a comprehensive experiment report with plots.
    
    Args:
        output_path: Path to save the report
        metrics: List of metrics to include (default: ["test_f1_score", "test_accuracy"])
        top_n: Number of top models to include per metric
        include_plots: Whether to download and include plots
        plots_dir: Directory to save plots (default: output_path.parent / "plots")
    """
    if metrics is None:
        metrics = ["test_f1_score", "test_accuracy", "test_precision", "test_recall"]
    
    logger.info("Generating experiment report...")
    
    config = ChurnPredictionConfig()
    # Use project name from config (should be "Churn Prediction Experiments")
    project_name = config.clearml_project
    
    # Verify project exists and has tasks
    try:
        tasks = Task.get_tasks(project_name=project_name, task_filter={})
        if tasks:
            logger.info(f"Found {len(tasks)} tasks in project '{project_name}'")
        else:
            logger.warning(f"No tasks found in project '{project_name}'. Trying alternative project names...")
            # Fallback to alternative project names
            for alt_name in ["Churn Prediction Experiments", "Churn Prediction"]:
                if alt_name == project_name:
                    continue  # Skip if already checked
                try:
                    alt_tasks = Task.get_tasks(project_name=alt_name, task_filter={})
                    if alt_tasks:
                        project_name = alt_name
                        logger.info(f"Found {len(alt_tasks)} tasks in project '{alt_name}'")
                        break
                except Exception:
                    continue
    except Exception as e:
        logger.warning(f"Could not get tasks from project '{project_name}': {e}")
        logger.info("Trying alternative project names...")
        # Fallback to alternative project names
        for alt_name in ["Churn Prediction Experiments", "Churn Prediction"]:
            if alt_name == project_name:
                continue  # Skip if already checked
            try:
                alt_tasks = Task.get_tasks(project_name=alt_name, task_filter={})
                if alt_tasks:
                    project_name = alt_name
                    logger.info(f"Found {len(alt_tasks)} tasks in project '{alt_name}'")
                    break
            except Exception:
                continue
    
    if not project_name:
        logger.error("Could not find any tasks in any project!")
        raise ValueError("No ClearML project with experiments found. Please ensure experiments are run first.")
    
    # Setup plots directory
    if include_plots:
        if plots_dir is None:
            plots_dir = output_path.parent / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
    
    report = f"""# Experiment Report

Generated automatically from ClearML experiments.

## Project: {project_name}

## Top Models by Metrics

"""
    
    all_task_ids = set()
    # Track experiments per metric to ensure diversity
    experiments_by_metric = {}  # metric_name -> list of (task_id, task_name, metric_value)
    # Track experiments already shown in tables to avoid duplicates across metrics
    shown_in_tables = set()  # task_name
    
    for metric_name in metrics:
        logger.info(f"Comparing experiments by {metric_name}...")
        try:
            # First try to get from all experiments (not just registered models)
            results = compare_experiments(
                project_name=project_name,
                metric_name=metric_name,
                top_n=top_n * 5,  # Get many more results to find diverse ones
            )
            
            # Remove duplicates by task_id and task_name (keep first occurrence)
            seen_task_ids = set()
            seen_task_names = set()
            unique_results = []
            for result in results:
                task_id = result.get("task_id")
                task_name = result.get("task_name", "")
                
                # Skip if we've already seen this task_id or task_name
                if task_id and task_id not in seen_task_ids:
                    if task_name and task_name in seen_task_names:
                        logger.debug(f"Skipping duplicate task name: {task_name} (task_id: {task_id})")
                        continue
                    
                    seen_task_ids.add(task_id)
                    if task_name:
                        seen_task_names.add(task_name)
                    unique_results.append(result)
            
            # Store all unique results for this metric (for plots)
            all_unique_for_metric = unique_results[:top_n * 3]
            experiments_by_metric[metric_name] = all_unique_for_metric
            
            # For table display: select first experiment that hasn't been shown yet
            table_results = []
            for result in unique_results:
                task_name = result.get("task_name", "")
                if task_name and task_name not in shown_in_tables:
                    table_results.append(result)
                    shown_in_tables.add(task_name)
                    if len(table_results) >= top_n:
                        break
            
            # If we couldn't find enough unique ones, take best ones anyway
            if not table_results:
                table_results = unique_results[:top_n]
            
            logger.info(f"Found {len(table_results)} unique experiments for table (from {len(unique_results)} total)")
            
            # If no results from experiments, try model registry as fallback
            if not table_results:
                logger.info(f"No experiments found, trying model registry...")
                results = compare_models(
                    project_name=project_name,
                    metric_name=metric_name,
                    top_n=top_n * 5,  # Get more for diversity
                )
                # Convert model registry format to experiment format
                if results:
                    # Filter by unique names
                    seen_names = set()
                    filtered_results = []
                    for r in results:
                        task_name = r.get("task_name", r.get("model_name", "Unknown"))
                        if task_name not in seen_names:
                            seen_names.add(task_name)
                            filtered_results.append({
                                "task_id": r.get("task_id"),
                                "task_name": task_name,
                                metric_name: r.get(metric_name),
                                "status": "completed",
                            })
                    
                    # Store all for plots
                    experiments_by_metric[metric_name] = filtered_results[:top_n * 3]
                    
                    # For table: select first not shown
                    table_results = []
                    for r in filtered_results:
                        task_name = r.get("task_name", "")
                        if task_name and task_name not in shown_in_tables:
                            table_results.append(r)
                            shown_in_tables.add(task_name)
                            if len(table_results) >= top_n:
                                break
                    
                    if not table_results:
                        table_results = filtered_results[:top_n]
            
            # Display table results
            if table_results:
                report += f"### Top {len(table_results)} Experiments by {metric_name}\n\n"
                # Create table from results
                table = "| Task Name | " + metric_name + " | Task ID |\n"
                table += "|-----------|" + "-" * (len(metric_name) + 2) + "|----------|\n"
                for result in table_results:
                    task_name = result.get("task_name", "N/A")
                    metric_value = result.get(metric_name, 0)
                    task_id = result.get("task_id", "N/A")
                    # Truncate task_id for display
                    task_id_short = task_id[:8] + "..." if len(task_id) > 8 else task_id
                    table += f"| {task_name} | {metric_value:.4f} | {task_id_short} |\n"
                report += table
                report += "\n\n"
            else:
                report += f"### {metric_name}\n\n"
                report += "No experiments found with this metric.\n\n"
        except Exception as e:
            logger.warning(f"Could not generate comparison for {metric_name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            report += f"### {metric_name}\n\n"
            report += f"Error: {e}\n\n"
    
    # Collect diverse experiments from different metrics for plots
    if include_plots and experiments_by_metric:
        # Collect diverse experiments ensuring we get different ones
        selected_experiments = []  # list of (task_id, task_name, metric_name)
        seen_task_names = set()
        seen_task_ids = set()
        
        # Strategy: Try to get one unique experiment from each metric
        # If same experiment is best for multiple metrics, take next best from those metrics
        max_experiments_per_metric = 3  # Look deeper into each metric's top list
        
        # First pass: collect all unique experiments from all metrics
        all_candidates = []  # list of (task_id, task_name, metric_name, rank_in_metric)
        for metric_name, experiments in experiments_by_metric.items():
            for rank, exp in enumerate(experiments[:max_experiments_per_metric]):
                task_id = exp.get("task_id")
                task_name = exp.get("task_name", "")
                if task_id and task_name:
                    all_candidates.append((task_id, task_name, metric_name, rank))
        
        # Second pass: select diverse experiments prioritizing:
        # 1. Different task names
        # 2. Lower rank (better performance) in their metric
        # 3. Different metrics
        
        # Group by task_name to see which metrics each experiment appears in
        experiments_by_name = {}  # task_name -> list of (task_id, metric_name, rank)
        for task_id, task_name, metric_name, rank in all_candidates:
            if task_name not in experiments_by_name:
                experiments_by_name[task_name] = []
            experiments_by_name[task_name].append((task_id, metric_name, rank))
        
        # Select up to 4 different experiments
        # Priority: experiments that appear in fewer metrics (more unique)
        # Then: experiments with better ranks
        candidate_list = []
        for task_name, appearances in experiments_by_name.items():
            # Find best rank for this experiment across all metrics
            best_rank = min(rank for _, _, rank in appearances)
            best_metric = next(metric for _, metric, rank in appearances if rank == best_rank)
            task_id = next(task_id for task_id, _, _ in appearances)
            candidate_list.append((task_id, task_name, best_metric, best_rank, len(appearances)))
        
        # Sort by: 1) number of appearances (fewer = more unique), 2) rank (lower = better)
        candidate_list.sort(key=lambda x: (x[4], x[3]))
        
        # Select top 4 unique experiments
        for task_id, task_name, metric_name, _, _ in candidate_list[:4]:
            if task_name not in seen_task_names and task_id not in seen_task_ids:
                selected_experiments.append((task_id, task_name, metric_name))
                seen_task_names.add(task_name)
                seen_task_ids.add(task_id)
                if len(selected_experiments) >= 4:
                    break
        
        logger.info(f"Selected {len(selected_experiments)} diverse experiments for visualization:")
        for task_id, task_name, metric_name in selected_experiments:
            logger.info(f"  - {task_name} (best by {metric_name})")
        
        report += "## Visualizations\n\n"
        report += "### Training Metrics Plots\n\n"
        
        plot_count = 0
        max_total_plots = 15  # Limit total plots in report
        
        for task_id, task_name, metric_name in selected_experiments[:4]:  # Limit to 4 different experiments
            if plot_count >= max_total_plots:
                break
                
            logger.info(f"Downloading plots for task {task_id}...")
            plot_paths = download_task_plots(task_id, plots_dir, max_plots=3)
            
            if plot_paths:
                # Get task name for display
                report += f"#### {task_name} (ID: {task_id}) - Best by {metric_name}\n\n"
                
                # Separate confusion matrix and metrics plots
                confusion_matrix_path = None
                metrics_plot_path = None
                
                for plot_path in plot_paths:
                    if "confusion_matrix" in plot_path.lower():
                        confusion_matrix_path = plot_path
                    elif "combined_metrics" in plot_path.lower():
                        metrics_plot_path = plot_path
                
                # Show combined metrics plot first
                if metrics_plot_path:
                    relative_plot_path = f"plots/{metrics_plot_path}"
                    report += f"**Combined Metrics**\n\n"
                    report += f"![Combined Metrics Plot]({relative_plot_path})\n\n"
                    plot_count += 1
                
                # Show confusion matrix separately
                if confusion_matrix_path:
                    relative_cm_path = f"plots/{confusion_matrix_path}"
                    report += f"**Confusion Matrix**\n\n"
                    report += f"![Confusion Matrix]({relative_cm_path})\n\n"
                    plot_count += 1
                
                # Show other plots if any
                for plot_path in plot_paths:
                    if plot_count >= max_total_plots:
                        break
                    if plot_path not in [confusion_matrix_path, metrics_plot_path]:
                        relative_plot_path = f"plots/{plot_path}"
                        plot_name = Path(plot_path).stem.replace("_", " ").title()
                        report += f"**{plot_name}**\n\n"
                        report += f"![Plot]({relative_plot_path})\n\n"
                        plot_count += 1
        
        if plot_count == 0:
            report += "No plots available for top experiments.\n\n"
            report += "> Note: Plots are downloaded from ClearML tasks. Make sure ClearML Server is running and experiments have logged plots.\n\n"
    
    report += """## Notes

- Reports are generated automatically from ClearML experiments
- Metrics are extracted from the last scalar metrics or task parameters
- Models are sorted by the specified metric in descending order
- Plots are downloaded from ClearML tasks and embedded in the report

## Reproducibility

To reproduce these results:

1. Ensure ClearML Server is running: `pixi run clearml-server-start`
2. Run experiments: `pixi run churn-experiments`
3. View results in ClearML UI: http://localhost:8080

"""
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write report
    output_path.write_text(report, encoding="utf-8")
    logger.success(f"Report saved to {output_path}")
    if include_plots:
        logger.success(f"Plots saved to {plots_dir}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate experiment report")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/experiments/experiment_report.md"),
        help="Output path for the report",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["test_f1_score", "test_accuracy", "test_precision", "test_recall"],
        help="Metrics to include in the report",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top models to include per metric",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip downloading and including plots",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Directory to save plots (default: reports/experiments/plots)",
    )
    
    args = parser.parse_args()
    
    generate_experiment_report(
        output_path=args.output,
        metrics=args.metrics,
        top_n=args.top_n,
        include_plots=not args.no_plots,
        plots_dir=args.plots_dir,
    )


if __name__ == "__main__":
    main()

