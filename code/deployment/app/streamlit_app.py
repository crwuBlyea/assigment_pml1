"""Web application (Stage 3) – Streamlit UI calling the model API."""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Risk Predictor")
st.caption("Enter the patient's measurements and press **Predict**. The prediction is "
           "computed by a model served by a separate FastAPI container.")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=250, value=117)
    blood_pressure = st.number_input("Blood pressure (mm Hg)", min_value=0, max_value=150, value=72)
    skin_thickness = st.number_input("Skin thickness (mm)", min_value=0, max_value=110, value=23)

with col2:
    insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=30)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=32.0, step=0.1)
    pedigree = st.number_input("Diabetes pedigree function", min_value=0.0, max_value=3.0, value=0.37, step=0.01)
    age = st.number_input("Age", min_value=1, max_value=120, value=29)

if st.button("Predict", type="primary", use_container_width=True):
    payload = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age,
    }
    with st.spinner("Calling the model API..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.ConnectionError:
            st.error("Could not reach the model API. Is the API container running?")
        except requests.exceptions.RequestException as exc:
            st.error(f"The model API returned an error: {exc}")
        else:
            if result["prediction"] == 1:
                st.error(f"⚠️ Prediction: the patient is likely **{result['label']}**.")
            else:
                st.success(f"✅ Prediction: the patient is likely **{result['label']}**.")
            prob = result["probability_diabetes"]
            st.progress(min(max(prob, 0.0), 1.0), text=f"Probability of diabetes: **{prob:.1%}**")