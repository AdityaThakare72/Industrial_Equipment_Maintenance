import streamlit as st
import requests
import pandas as pd

# Page Config
st.set_page_config(page_title="Industrial Anomaly Agent", page_icon="⚙️")

st.title("🛡️ Industrial Maintenance Predictor")
st.markdown("---")

# 1. Sidebar for Sensor Inputs
st.sidebar.header("📡 Live Sensor Data")

temperature = st.sidebar.slider("Temperature (°C)", 200.0, 600.0, 300.0)
pressure = st.sidebar.slider("Pressure (PSI)", 50.0, 200.0, 100.0)
vibration = st.sidebar.slider("Vibration (mm/s)", 0.0, 200.0, 20.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 45.0)

equipment = st.sidebar.selectbox("Equipment Type", ["Turbine", "Compressor", "Generator"])
location = st.sidebar.selectbox("Plant Location", ["Atlanta", "Chicago", "Pune", "Guwahati"])

# 2. Preparation for API Call

API_URL = "http://127.0.0.1:8000/predict"

payload = {
    "temperature": temperature,
    "pressure": pressure,
    "vibration": vibration,
    "humidity": humidity,
    "equipment": equipment,
    "location": location
}

# 3. Prediction Button
if st.button("🔍 Analyze Equipment Health"):
    with st.spinner("Agent is analyzing sensor patterns..."):
        try:
            response = requests.post(API_URL, json=payload)
            result = response.json()
            
            # 4. Display Results
            st.markdown("### Analysis Result")
            
            if result["prediction"] == "Faulty":
                st.error(f"⚠️ **ALERT**: {result['prediction']}")
                st.metric("Failure Probability", result["failure_probability"])
                st.warning("Recommendation: Immediate maintenance check required for the " + equipment)
            else:
                st.success(f"✅ **STATUS**: {result['prediction']}")
                st.metric("Failure Probability", result["failure_probability"])
                st.info("System is operating within safe parameters.")
                
            st.caption(f"Run ID: {result.get('model_version', 'N/A')}")
            
        except Exception as e:
            st.error(f"Error connecting to Anomaly Agent: {e}")