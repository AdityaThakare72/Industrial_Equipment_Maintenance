import streamlit as st
import requests
import base64

# ======================================================
# SESSION STATE
# ======================================================
if "scan_ran" not in st.session_state:
    st.session_state.scan_ran = False

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
# CUSTOM CSS (STEAMPUNK / GHIBLI)
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
    st.caption("v2.0 | Status: Connected to Localhost Inference Engine")

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
            "Temperature (°C)", 200.0, 600.0, 300.0
        )
        humidity = st.slider(
            "Humidity (%)", 0.0, 100.0, 45.0
        )

    with st.expander("⚙️ Mechanical Sensors", expanded=True):
        pressure = st.slider(
            "Pressure (PSI)", 50.0, 200.0, 100.0
        )
        vibration = st.slider(
            "Vibration (mm/s)", 0.0, 200.0, 20.0
        )

    st.markdown("### 🏭 Configuration")
    equipment = st.selectbox(
        "Equipment Type",
        ["Turbine", "Compressor", "Generator"]
    )
    location = st.selectbox(
        "Plant Location",
        ["Atlanta", "Chicago", "Pune", "Guwahati"]
    )

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

# ======================================================
# API CONFIG
# ======================================================
API_URL = "http://127.0.0.1:8000/predict"

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

    with st.spinner("🔄 Neural Network analyzing sensor patterns..."):
        try:
            response = requests.post(
                API_URL,
                json=payload,
                timeout=5
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

        except requests.exceptions.ConnectionError:
            st.error(
                "🔌 Connection Failed: FastAPI server not running on port 8000"
            )
            st.info(
                "Run: uvicorn app.main:app --port 8000"
            )

        except Exception as e:
            st.error(f"Unexpected error: {e}")