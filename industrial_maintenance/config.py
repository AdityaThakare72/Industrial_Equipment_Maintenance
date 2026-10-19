import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- API Configuration ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# --- Model Paths ---
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
PREPROCESSOR_PATH = Path(os.getenv("PREPROCESSOR_PATH", "models/preprocessor.joblib"))

# --- Training data ranges ---
# Min/max of each sensor in the raw training data (7,672 rows, see the EDA
# notebook). Predictions outside these ranges are extrapolation, so the API
# flags them and the dashboard keeps its sliders close to them.
TRAINING_RANGES = {
    "temperature": (10.3, 149.7),
    "pressure": (3.6, 79.9),
    "vibration": (-0.4, 5.0),
    "humidity": (10.2, 90.0),
}

# --- MLflow ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Industrial_Maintenance_Experiments")

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
