import pandas as pd
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def generate_features(input_path: Path, output_path: Path, model_path: Path):
    """
    Transform interim data into processed features:
    1. Scaling numerical values.
    2. Encoding categorical values.
    """
    df = pd.read_csv(input_path)
    
    # Define our pillars
    target = 'faulty'
    categorical_features = ['equipment', 'location']
    numerical_features = ['temperature', 'pressure', 'vibration', 'humidity']
    
    # The Alchemy: Building the Pipeline
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

    # Fit and Transform the data
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names for the final dataframe
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = numerical_features + list(cat_feature_names)
    
    X_final = pd.DataFrame(X_processed, columns=feature_names)
    X_final[target] = y.values

    # Save the processed data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    X_final.to_csv(output_path, index=False)
    
    # Save the preprocessor itself (The 'Golden Seal')
    # We will need this for the production API!
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, model_path)
    
    print(f"Features generated and saved to {output_path}")
    print(f"Preprocessor saved to {model_path}")

if __name__ == "__main__":
    generate_features(
        input_path=Path("data/interim/cleaned_data.csv"),
        output_path=Path("data/processed/featured_data.csv"),
        model_path=Path("models/preprocessor.joblib")
    )