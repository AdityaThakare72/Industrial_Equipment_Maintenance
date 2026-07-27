import json
from pathlib import Path

import joblib
from loguru import logger
import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
import typer
import yaml

from industrial_maintenance.config import (
    MLFLOW_EXPERIMENT_NAME,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)

app = typer.Typer()


def evaluate_model(
    features_path: Path,
    model_path: Path,
    params_path: Path,
    metrics_path: Path,
    predictions_path: Path,
):
    with open(params_path, "r") as f:
        config = yaml.safe_load(f)
    shared_params = config["train"]

    df = pd.read_csv(features_path)
    X = df.drop(columns=["faulty"])
    y = df["faulty"]

    # Recreate the same train/test split used during training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=shared_params["random_state"],
        stratify=y
    )

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df = pd.DataFrame({"actual": y_test.values, "predicted": y_pred})
    predictions_df.to_csv(predictions_path, index=False)

    logger.success(f"Metrics saved to {metrics_path}")
    logger.success(f"Predictions saved to {predictions_path}")


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "featured_data.csv",
    model_path: Path = MODELS_DIR / "model.joblib",
    params_path: Path = Path("params.yaml"),
    metrics_path: Path = REPORTS_DIR / "metrics.json",
    predictions_path: Path = REPORTS_DIR / "predictions.csv",
):
    evaluate_model(features_path, model_path, params_path, metrics_path, predictions_path)


if __name__ == "__main__":
    app()
