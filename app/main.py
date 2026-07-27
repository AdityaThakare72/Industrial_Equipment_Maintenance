from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel

from industrial_maintenance.config import MODEL_PATH, PREPROCESSOR_PATH

app = FastAPI(title="Industrial Anomaly Agent")

# Load trained model and preprocessor at startup

# Fail fast if artifacts are missing
if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
    raise RuntimeError("Model or Preprocessor missing. Run 'dvc repro' first.")

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Request schema for sensor readings
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
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))