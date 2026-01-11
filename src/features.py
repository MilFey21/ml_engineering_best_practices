from pathlib import Path

from loguru import logger
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import typer

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


@app.command()
def main(
    input_path: str = typer.Argument(default=str(RAW_DATA_DIR / "customer_churn.csv")),
    features_path: str = typer.Argument(default=str(PROCESSED_DATA_DIR / "features.csv")),
    labels_path: str = typer.Argument(default=str(PROCESSED_DATA_DIR / "labels.csv")),
):
    """Process raw data and create features for modeling."""
    input_path_obj = Path(input_path)
    features_path_obj = Path(features_path)
    labels_path_obj = Path(labels_path)
    logger.info(f"Loading data from {input_path_obj}...")
    df = pd.read_csv(input_path_obj)

    logger.info(f"Original data shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    # Make a copy for processing
    df_processed = df.copy()

    # Handle missing values - TotalCharges might be empty strings
    df_processed["TotalCharges"] = pd.to_numeric(df_processed["TotalCharges"], errors="coerce")

    # Fill missing TotalCharges with 0 (likely new customers)
    df_processed["TotalCharges"].fillna(0, inplace=True)

    # Remove customerID as it's not a feature
    if "customerID" in df_processed.columns:
        df_processed = df_processed.drop("customerID", axis=1)

    # Separate target variable
    if "Churn" in df_processed.columns:
        y = df_processed["Churn"].copy()
        df_processed = df_processed.drop("Churn", axis=1)

        # Encode target: Yes -> 1, No -> 0
        y = (y == "Yes").astype(int)
        y.to_csv(labels_path_obj, index=False)
        logger.info(f"Labels saved to {labels_path_obj}")
        logger.info(f"Churn distribution:\n{y.value_counts()}")
    else:
        logger.warning("Churn column not found. Skipping label creation.")
        y = None

    # Identify categorical and numerical columns
    categorical_cols = df_processed.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = df_processed.select_dtypes(include=["int64", "float64"]).columns.tolist()

    logger.info(f"Categorical columns: {categorical_cols}")
    logger.info(f"Numerical columns: {numerical_cols}")

    # Encode categorical variables using LabelEncoder
    label_encoders = {}
    df_encoded = df_processed.copy()

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le

    # Ensure all columns are numeric
    for col in df_encoded.columns:
        df_encoded[col] = pd.to_numeric(df_encoded[col], errors="coerce")

    # Fill any remaining NaN values
    df_encoded = df_encoded.fillna(0)

    logger.info(f"Processed features shape: {df_encoded.shape}")
    logger.info(f"Feature columns: {list(df_encoded.columns)}")

    # Save features
    df_encoded.to_csv(features_path_obj, index=False)
    logger.success(f"Features saved to {features_path_obj}")

    if y is not None:
        logger.info(
            f"Features and labels ready for modeling. Features: {df_encoded.shape}, Labels: {y.shape}"
        )

    # Create metrics file for DVC
    metrics_file = Path("metrics/features.json")
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    import json

    metrics_data = {
        "features_shape": list(df_encoded.shape),
        "feature_columns": list(df_encoded.columns),
        "categorical_columns": categorical_cols,
        "numerical_columns": numerical_cols,
    }
    if y is not None:
        metrics_data["labels_shape"] = list(y.shape)
        metrics_data["churn_distribution"] = y.value_counts().to_dict()
    metrics_file.write_text(json.dumps(metrics_data, indent=2))


if __name__ == "__main__":
    app()
