"""Script to generate all experiment reports with visualizations."""

import json
from pathlib import Path
from datetime import datetime

from loguru import logger

from scripts.generate_experiment_report import generate_experiment_report


def main():
    """Generate all reports."""
    reports_dir = Path("reports")
    experiments_dir = reports_dir / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate main comprehensive experiment report
    logger.info("Generating main comprehensive experiment report...")
    generate_experiment_report(
        output_path=experiments_dir / "experiment_report.md",
        metrics=["test_f1_score", "test_accuracy", "test_precision", "test_recall", "test_roc_auc"],
        top_n=10,
        include_plots=True,
    )
    
    # Generate focused F1-Score report
    logger.info("Generating F1-Score focused report...")
    generate_experiment_report(
        output_path=experiments_dir / f"experiment_report_f1_{timestamp}.md",
        metrics=["test_f1_score"],
        top_n=15,  # More models for F1-Score
        include_plots=True,
    )
    
    # Generate accuracy-focused report
    logger.info("Generating Accuracy focused report...")
    generate_experiment_report(
        output_path=experiments_dir / f"experiment_report_accuracy_{timestamp}.md",
        metrics=["test_accuracy"],
        top_n=15,  # More models for Accuracy
        include_plots=True,
    )
    
    # Generate summary report without plots (faster)
    logger.info("Generating summary report (without plots)...")
    generate_experiment_report(
        output_path=experiments_dir / f"experiment_report_summary_{timestamp}.md",
        metrics=["test_f1_score", "test_accuracy"],
        top_n=5,
        include_plots=False,  # No plots for faster generation
    )
    
    logger.success(f"Generated 4 reports successfully!")
    logger.info(f"Reports saved in: {experiments_dir}")


if __name__ == "__main__":
    main()

