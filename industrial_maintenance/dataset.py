import pandas as pd
from pathlib import Path
import logging

# Set up logging 
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_raw_data(file_path: Path) -> pd.DataFrame:
    """Loads the raw industrial anomaly CSV from the data folder."""
    if not file_path.exists():
        logging.error(f" The file at {file_path} is missing!")
        raise FileNotFoundError
    return pd.read_csv(file_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Refines the industrial data:
    1. Ensures target 'faulty' is integer.
    2. Strips whitespace from category names.
    """
    logging.info("Performing initial cleaning")
    
    # Ensure our target is a discrete 0 or 1
    if 'faulty' in df.columns:
        df['faulty'] = df['faulty'].astype(int)
    
    # Clean categorical strings to prevent ' Turbine' vs 'Turbine' issues
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
        
    return df

def main():
    """Main pipeline for the industrial anomaly dataset."""
    # Defining paths relative to project root (standard CCDS)
    input_path = Path("data/raw/industrial_equipment_anomaly_data.csv")
    output_path = Path("data/interim/cleaned_data.csv")
    
    logging.info("Starting Data Ingestion...")
    
    # 1. Load
    raw_df = load_raw_data(input_path)
    
    # 2. Clean
    processed_df = clean_data(raw_df)
    
    # 3. Save to the Interim vault
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    
    logging.info(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    main()