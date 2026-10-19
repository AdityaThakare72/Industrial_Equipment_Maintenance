from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yaml


def generate_features(input_path: Path, output_path: Path, model_path: Path, params_path: Path):
    """
    Transform interim data into processed features:
    1. Scaling numerical values.
    2. Encoding categorical values.

    The preprocessor is fitted on the training split only, so test-set
    statistics never leak into training. train.py and predict.py recreate
    the same split from the same params.
    """
    with open(params_path, "r") as f:
        split = yaml.safe_load(f)["train"]

    df = pd.read_csv(input_path)
    
    # Define feature groups
    target = 'faulty'
    categorical_features = ['equipment', 'location']
    numerical_features = ['temperature', 'pressure', 'vibration', 'humidity']
    
    # Build preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine the transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Separate Features and Target
    X = df.drop(columns=[target])
    y = df[target]

    # Fit on the training rows only, then transform every row
    X_train, _, _, _ = train_test_split(
        X, y,
        test_size=split["test_size"],
        random_state=split["random_state"],
        stratify=y,
    )
    preprocessor.fit(X_train)
    X_processed = preprocessor.transform(X)
    
    # Get feature names for the final dataframe
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = numerical_features + list(cat_feature_names)
    
    X_final = pd.DataFrame(X_processed, columns=feature_names)
    X_final[target] = y.values

    # Save the processed data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    X_final.to_csv(output_path, index=False)
    
    # Save fitted preprocessor for inference
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, model_path)
    
    print(f"Features generated and saved to {output_path}")
    print(f"Preprocessor saved to {model_path}")

if __name__ == "__main__":
    generate_features(
        input_path=Path("data/interim/cleaned_data.csv"),
        output_path=Path("data/processed/featured_data.csv"),
        model_path=Path("models/preprocessor.joblib"),
        params_path=Path("params.yaml"),
    )