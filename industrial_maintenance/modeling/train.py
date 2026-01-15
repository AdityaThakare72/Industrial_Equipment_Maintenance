import pandas as pd
import joblib
import yaml
import mlflow
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

def train_selected_model(input_path: Path, model_output_path: Path, params_path: Path):
    with open(params_path, "r") as f:
        config = yaml.safe_load(f)
    
    model_type = config["model_type"]
    shared_params = config["train"]
    
    df = pd.read_csv(input_path)
    X = df.drop(columns=['faulty'])
    y = df['faulty']

    # Calculate class imbalance for XGBoost
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    imbalance_ratio = neg_count / pos_count

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, 
        random_state=shared_params["random_state"], 
        stratify=y
    )

    if model_type == "random_forest":
        # 'balanced' automatically adjusts weights inversely proportional to class frequencies
        estimator = RandomForestClassifier(
            random_state=shared_params["random_state"],
            class_weight='balanced'
        )
        param_grid = config["random_forest"]["param_grid"]
    elif model_type == "xgboost":
        estimator = XGBClassifier(
            random_state=shared_params["random_state"],
            scale_pos_weight=imbalance_ratio,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        param_grid = config["xgboost"]["param_grid"]

    mlflow.set_experiment("Industrial_Maintenance_Experiments")

    with mlflow.start_run(run_name=f"Final_{model_type}"):
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=shared_params["cv_folds"],
            scoring='f1',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        
        # Log to MLflow
        mlflow.log_param("model_family", model_type)
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_f1_score", grid_search.best_score_)
        mlflow.sklearn.log_model(best_model, "champion_model")

        # Save for DVC
        model_output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, model_output_path)

        print(f"Final {model_type} trained with class balancing. F1: {grid_search.best_score_:.4f}")

if __name__ == "__main__":
    train_selected_model(
        input_path=Path("data/processed/featured_data.csv"),
        model_output_path=Path("models/model.joblib"),
        params_path=Path("params.yaml")
    )