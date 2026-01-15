from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="Industrial Anomaly Agent")

# Load the "Sacred Relics" (Preprocessor and the best XGBoost Model)
MODEL_PATH = Path("models/model.joblib")
PREPROCESSOR_PATH = Path("models/preprocessor.joblib")

# Ensure the hermit has his tools before starting
if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
    raise RuntimeError("Model or Preprocessor missing. Run 'dvc repro' first.")

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# The "Bouncer": Defining exactly what sensor data looks like
class SensorData(BaseModel):
    temperature: float
    pressure: float
    vibration: float
    humidity: float
    equipment: str
    location: str

@app.get("/")
def home():
    return {"message": "Industrial Anomaly Detection API is Online."}

@app.post("/predict")
def predict(data: SensorData):
    try:
        # 1. Convert incoming JSON to a DataFrame row
        input_df = pd.DataFrame([data.model_dump()])
        
        # 2. Transform the data (Scaling and One-Hot Encoding)
        # CRITICAL: We use .transform(), never .fit_transform() here!
        processed_data = preprocessor.transform(input_df)
        
        # 3. Get the prediction (0 or 1) and the probability
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)[0][1]
        
        result = "Faulty" if prediction[0] == 1 else "Healthy"
        
        return {
            "prediction": result,
            "failure_probability": f"{probability:.2%}",
            "status": "Success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))