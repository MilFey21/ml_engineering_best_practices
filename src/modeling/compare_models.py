"""Compare multiple trained models.

This module provides functionality to compare metrics from different
trained models and generate comparison reports in JSON and Markdown formats.
"""
import json
from pathlib import Path
from typing import Dict, List

from loguru import logger
import pandas as pd

from src.config import REPORTS_DIR

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def compare_models() -> None:
    """Compare metrics from different trained models.

    Loads metrics from multiple model files and creates comparison reports
    in both JSON and Markdown formats. The comparison includes metrics such as
    accuracy, F1-score, and ROC-AUC.

    Returns:
        None. Comparison results are saved to the reports directory.
    """
    logger.info("Comparing models...")
    
    models_dir = Path("models")
    metrics_files = {
        "Random Forest": models_dir / "rf_metrics.json",
        "Logistic Regression": models_dir / "lr_metrics.json",
        "Gradient Boosting": models_dir / "gb_metrics.json",
    }
    
    results = []
    for model_name, metrics_path in metrics_files.items():
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            metrics["model_name"] = model_name
            results.append(metrics)
            logger.info(f"Loaded metrics for {model_name}")
        else:
            logger.warning(f"Metrics file not found: {metrics_path}")
    
    if not results:
        logger.error("No model metrics found!")
        return
    
    # Создание DataFrame для сравнения
    df = pd.DataFrame(results)
    df = df.set_index("model_name")
    
    logger.info("\nModel Comparison:")
    logger.info("=" * 50)
    logger.info(f"\n{df.to_string()}")
    logger.info("=" * 50)
    
    # Сохранение результатов
    reports_dir = REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON формат
    json_path = reports_dir / "model_comparison.json"
    df.to_json(json_path, indent=2)
    logger.info(f"\nComparison saved to {json_path}")
    
    # Markdown формат
    md_path = reports_dir / "model_comparison.md"
    with open(md_path, "w") as f:
        f.write("# Model Comparison\n\n")
        f.write("## Metrics Comparison\n\n")
        f.write(df.to_markdown())
        f.write("\n\n")
        
        # Лучшая модель по каждой метрике
        f.write("## Best Models by Metric\n\n")
        for metric in ["test_accuracy", "test_f1_score", "test_roc_auc"]:
            if metric in df.columns:
                best_idx = df[metric].idxmax()
                best_value = df.loc[best_idx, metric]
                f.write(f"- **{metric}**: {best_idx} ({best_value:.4f})\n")
    
    logger.info(f"Markdown report saved to {md_path}")


if __name__ == "__main__":
    compare_models()
