import base64
import os
import threading
from urllib.parse import urlparse

from dotenv import load_dotenv
import requests
import streamlit as st

load_dotenv()

API_BASE = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
API_URL = API_BASE + "/predict"
IS_LOCAL_API = urlparse(API_BASE).hostname in ("127.0.0.1", "localhost")

# The free Render instance sleeps when idle and needs ~30-60s to wake, so
# allow a long read timeout instead of failing the first scan
REQUEST_TIMEOUT = (10, 90)

# One-click scenarios. Values sit inside the training data ranges
# (industrial_maintenance/config.py TRAINING_RANGES) so the model isn't
# extrapolating; the dashboard can't import that module on Streamlit Cloud.
PRESETS = {
    "✅ Normal": {"temperature": 70.0, "pressure": 35.0, "vibration": 1.5, "humidity": 50.0},
    "🔥 Overheating": {"temperature": 130.0, "pressure": 35.0, "vibration": 1.5, "humidity": 50.0},
    "📳 High vibration": {"temperature": 70.0, "pressure": 35.0, "vibration": 4.5, "humidity": 50.0},
    "💥 Over-pressure": {"temperature": 70.0, "pressure": 75.0, "vibration": 1.5, "humidity": 50.0},
}

# ======================================================
# SESSION STATE
# ======================================================
if "scan_ran" not in st.session_state:
    st.session_state.scan_ran = False

for field, value in PRESETS["✅ Normal"].items():
    st.session_state.setdefault(field, value)


def apply_preset(name: str):
    for field, value in PRESETS[name].items():
        st.session_state[field] = value
    st.session_state.scan_ran = True


def wake_api():
    try:
        requests.get(API_BASE + "/", timeout=REQUEST_TIMEOUT)
    except requests.RequestException:
        pass


# Start waking the API as soon as someone opens the page, so it's usually
# up by the time they press "Run Diagnostic Scan"
if "api_wake_started" not in st.session_state:
    st.session_state.api_wake_started = True
    threading.Thread(target=wake_api, daemon=True).start()

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Industrial Anomaly Agent",
    page_icon="⚙️",
    layout="wide"
)

# ======================================================
# LOAD BACKGROUND IMAGE
# ======================================================
def get_base64_of_bin_file(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_base64 = get_base64_of_bin_file("app/background.png")

# ======================================================
# CUSTOM CSS
# ======================================================
st.markdown(
    f"""
    <style>
        .stApp {{
            background-image:
                linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.8)),
                url("data:image/png;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# ======================================================
# HEADER
# ======================================================
col1, col2 = st.columns([1, 4])

with col1:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/1541/1541416.png",
        width=100
    )

with col2:
    st.title("Industrial Maintenance AI")
    st.caption(
        "v2.0 | Inference API: "
        + ("local FastAPI server" if IS_LOCAL_API else f"FastAPI on {urlparse(API_BASE).hostname}")
    )

# Divider
st.divider()

# ======================================================
# SIDEBAR CONTROLS
# ======================================================
with st.sidebar:
    st.header("🎛️ Control Panel")
    st.markdown("Adjust live sensor feeds below:")

    with st.expander("🌡️ Environmental Sensors", expanded=True):
        temperature = st.slider(
            "Temperature (°C)", 0.0, 160.0, key="temperature"
        )
        humidity = st.slider(
            "Humidity (%)", 0.0, 100.0, key="humidity"
        )

    with st.expander("⚙️ Mechanical Sensors", expanded=True):
        pressure = st.slider(
            "Pressure (PSI)", 0.0, 90.0, key="pressure"
        )
        vibration = st.slider(
            "Vibration (mm/s)", 0.0, 6.0, step=0.1, key="vibration"
        )

    st.markdown("### 🏭 Configuration")
    equipment = st.selectbox(
        "Equipment Type",
        ["Turbine", "Compressor", "Pump"]
    )
    location = st.selectbox(
        "Plant Location",
        ["Atlanta", "Chicago", "San Francisco", "New York", "Houston"]
    )

# ======================================================
# SCENARIO PRESETS
# ======================================================
st.markdown("**Try a scenario** (sets the sensors and runs a scan):")
preset_cols = st.columns(len(PRESETS))
for col, name in zip(preset_cols, PRESETS):
    col.button(name, on_click=apply_preset, args=(name,), use_container_width=True)

# ======================================================
# LIVE TELEMETRY
# ======================================================
st.subheader("📡 Live Sensor Telemetry")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Temperature", f"{temperature:.1f} °C")
c2.metric("Pressure", f"{pressure:.1f} PSI")
c3.metric("Vibration", f"{vibration:.1f} mm/s")
c4.metric("Humidity", f"{humidity:.1f} %")

# Spacing
st.write("")

payload = {
    "temperature": temperature,
    "pressure": pressure,
    "vibration": vibration,
    "humidity": humidity,
    "equipment": equipment,
    "location": location
}

# ======================================================
# RUN DIAGNOSTIC BUTTON
# ======================================================
if st.button(
    "🔍 Run Diagnostic Scan",
    type="primary",
    use_container_width=True
):
    st.session_state.scan_ran = True

# ======================================================
# RESULTS (ONLY AFTER BUTTON CLICK)
# ======================================================
if st.session_state.scan_ran:

    with st.spinner(
        "🔄 Running XGBoost diagnostic analysis... "
        "(the first scan can take up to a minute while the free inference server wakes up)"
    ):
        try:
            response = requests.post(
                API_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                result = response.json()
                prob_percent = (
                    float(result["failure_probability"].strip("%")) / 100
                )

                st.divider()
                st.subheader("📊 Diagnostic Results")

                r1, r2 = st.columns([2, 1])

                with r1:
                    st.write("Failure Probability Analysis:")

                    st.progress(
                        prob_percent,
                        text=f"RISK LEVEL: {result['failure_probability']}"
                    )

                    if result["prediction"] == "Faulty":
                        st.error(
                            f"⚠️ CRITICAL ALERT: {equipment} failure detected!"
                        )
                        st.markdown(
                            f"**Action Required:** Dispatch maintenance team to **{location}**."
                        )
                    else:
                        st.success(
                            f"✅ SYSTEM HEALTHY: {equipment} operating normally."
                        )

                for warning in result.get("warnings", []):
                    st.warning(f"⚠️ {warning}")

                with r2:
                    st.metric(
                        "Model Prediction",
                        result["prediction"]
                    )
                    st.caption(
                        f"Model Version: {result.get('model_version', 'Production_v1')}"
                    )

            else:
                st.error(
                    f"API Error {response.status_code}: {response.text}"
                )

        except requests.exceptions.Timeout:
            st.error(
                "⏳ The inference server is still waking up. Please press "
                "**Run Diagnostic Scan** again in a few seconds."
            )

        except requests.exceptions.ConnectionError:
            st.error(f"🔌 Connection Failed: couldn't reach the inference API at {API_BASE}")
            if IS_LOCAL_API:
                st.info(
                    "Run: uvicorn app.main:app --port 8000"
                )

        except Exception as e:  # noqa: BLE001
            st.error(f"Unexpected error: {e}")