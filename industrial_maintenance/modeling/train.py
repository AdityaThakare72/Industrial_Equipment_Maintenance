import pandas as pd
import joblib
import yaml
import mlflow
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

def train_selected_model(input_path: Path, model_output_path: Path, params_path: Path):
    """
    Reads model type and parameters from YAML, performs GridSearchCV,
    and logs the champion to MLflow.
    """
    # Load the sacred parameters
    with open(params_path, "r") as f:
        config = yaml.safe_load(f)
    
    model_type = config["model_type"]
    shared_params = config["train"]
    
    # Load processed data
    df = pd.read_csv(input_path)
    X = df.drop(columns=['faulty'])
    y = df['faulty']

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, 
        random_state=shared_params["random_state"], 
        stratify=y
    )

    # Factory Logic: Choose the weapon
    if model_type == "random_forest":
        estimator = RandomForestClassifier(random_state=shared_params["random_state"])
        param_grid = config["random_forest"]["param_grid"]
    elif model_type == "xgboost":
        # XGBoost handles categories better if they were encoded
        estimator = XGBClassifier(
            random_state=shared_params["random_state"],
            use_label_encoder=False,
            eval_metric='logloss'
        )
        param_grid = config["xgboost"]["param_grid"]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Setup MLflow
    mlflow.set_experiment("Industrial_Maintenance_Experiments")

    with mlflow.start_run(run_name=f"GridSearch_{model_type}"):
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=shared_params["cv_folds"],
            scoring='f1',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)

        # Champion extraction
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        # Log everything to the MLflow librarian
        mlflow.log_param("model_family", model_type)
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_f1_score", best_score)
        mlflow.sklearn.log_model(best_model, "champion_model")

        # Save the physical artifact for DVC tracking
        model_output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, model_output_path)

        print(f"{model_type} training complete. Best F1: {best_score:.4f}")

if __name__ == "__main__":
    train_selected_model(
        input_path=Path("data/processed/featured_data.csv"),
        model_output_path=Path("models/model.joblib"),
        params_path=Path("params.yaml")
    )