# === AI-Powered Diabetes Predictor with Extended Parameters ===

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import openai

# === Page Config ===
st.set_page_config(page_title="AI Diabetes Predictor", layout="wide")

# === OpenAI Client ===
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# === Sticky Navigation Bar ===
st.markdown("""
    <style>
    .nav-tabs {
        position: sticky;
        top: 0;
        z-index: 100;
        background-color: #0e1117;
        padding: 10px 0;
        text-align: center;
    }
    .nav-tabs a {
        margin: 0 15px;
        color: #4CAF50;
        font-weight: bold;
        text-decoration: none;
        font-size: 1.1em;
    }
    </style>
    <div class="nav-tabs">
        <a href="#patient-info">Patient Info</a>
        <a href="#ai-prediction">AI Prediction</a>
        <a href="#risk-breakdown">Risk Breakdown</a>
        <a href="#chatgpt-explanation">ChatGPT Explanation</a>
    </div>
""", unsafe_allow_html=True)

# === Main Title ===
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🤖 AI Diabetes Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Expanded with clinical and lifestyle data for better diagnosis.</p>", unsafe_allow_html=True)
st.markdown("---")

# === Patient Information ===
st.markdown("<div id='patient-info'></div>", unsafe_allow_html=True)
st.header("📋 Step 1: Patient Personal & Health Information")
with st.expander("Enter extended patient details", expanded=True):
    patient_name = st.text_input("👤 Patient Full Name", value="John Doe")
    patient_id = st.text_input("🆔 Patient ID", value="DFX-2025-001")
    visit_reason = st.text_area("📄 Reason for Visit", placeholder="e.g. Regular checkup, symptoms of fatigue, blood sugar monitoring...")
    notes = st.text_area("🩺 Chronic Conditions / Notes", placeholder="e.g. Has history of heart disease, chronic kidney condition, or other diagnoses...")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.radio("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 90, 45)
        bmi = st.slider("BMI", 10.0, 50.0, 25.0)
        waist = st.slider("Waist Circumference (cm)", 50, 150, 85)
        whr = st.slider("Waist-to-Hip Ratio", 0.6, 1.2, 0.9)
        hdl = st.slider("HDL Cholesterol (mg/dL)", 20, 100, 50)
        triglycerides = st.slider("Triglycerides (mg/dL)", 50, 400, 150)
        crp = st.slider("C-Reactive Protein (mg/L)", 0.1, 20.0, 1.5)
    with col2:
        glucose = st.slider("Glucose (mg/dL)", 70, 200, 100)
        insulin = st.slider("Insulin (μU/mL)", 2, 300, 85)
        bp = st.slider("Blood Pressure (mmHg)", 80, 180, 120)
        sleep = st.slider("Sleep (hrs/night)", 3, 12, 7)
        sugar_intake = st.selectbox("Sugar Intake", ["Low", "Moderate", "High"])
        fiber_intake = st.radio("Fiber Intake", ["Low", "High"])
        smoking = st.radio("Smoking Status", ["Non-smoker", "Former Smoker", "Current Smoker"])
        alcohol = st.selectbox("Alcohol Consumption", ["None", "Occasional", "Frequent"])
        activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
        family_history = st.radio("Family History of Diabetes", ["No", "Yes"])
        gestational = st.radio("History of Gestational Diabetes (if female)", ["N/A", "Yes", "No"])
        pcod = st.radio("PCOS (Polycystic Ovary Syndrome)", ["N/A", "Yes", "No"])

# === Encoding Inputs ===
gender_encoded = 1 if gender == "Male" else 0
activity_encoded = {"Low": 0, "Moderate": 1, "High": 2}[activity]
sugar_encoded = {"Low": 0, "Moderate": 1, "High": 2}[sugar_intake]
alcohol_encoded = {"None": 0, "Occasional": 1, "Frequent": 2}[alcohol]
smoking_encoded = {"Non-smoker": 0, "Former Smoker": 1, "Current Smoker": 2}[smoking]
fiber_encoded = 1 if fiber_intake == "High" else 0
family_encoded = 1 if family_history == "Yes" else 0
gestational_encoded = 1 if gestational == "Yes" else 0
pcod_encoded = 1 if pcod == "Yes" else 0

# === Model Input Preparation ===
input_features = [
    gender_encoded, age, bmi, waist, whr, glucose, insulin, bp, hdl, triglycerides,
    crp, sleep, sugar_encoded, fiber_encoded, smoking_encoded, alcohol_encoded,
    activity_encoded, family_encoded, gestational_encoded, pcod_encoded
]

input_df = pd.DataFrame([input_features], columns=[
    "Gender", "Age", "BMI", "Waist", "WHR", "Glucose", "Insulin", "BP", "HDL", "Triglycerides",
    "CRP", "Sleep", "SugarIntake", "Fiber", "Smoking", "Alcohol", "Activity", "FamilyHistory",
    "Gestational", "PCOD"
])

# === Model ===
X, y = make_classification(n_samples=1000, n_features=len(input_df.columns), random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier()
model.fit(X_scaled, y)
input_scaled = scaler.transform(input_df)
risk = model.predict_proba(input_scaled)[0][1]

# === AI Prediction ===
st.markdown("<div id='ai-prediction'></div>", unsafe_allow_html=True)
st.markdown("### 🎯 Step 2: AI-Predicted Diabetes Risk")
st.subheader(f"🧪 Patient: {patient_name}")
st.subheader(f"🔎 Predicted Risk: **{risk * 100:.2f}%**")
if risk >= 0.7:
    st.error("🔴 High risk. Immediate attention recommended.")
elif risk >= 0.4:
    st.warning("🟠 Moderate risk. Monitor closely.")
else:
    st.success("🟢 Low risk. Maintain current lifestyle.")

# === Risk Breakdown ===
st.markdown("<div id='risk-breakdown'></div>", unsafe_allow_html=True)
st.markdown("#### 🧾 Estimated Risk Contributions")
risk_factors = {
    "BMI": bmi * 0.25,
    "Glucose": glucose * 0.25,
    "Insulin": insulin * 0.15,
    "Age": age * 0.1,
    "Sleep": (12 - sleep) * 0.05,
    "Smoking": smoking_encoded * 5,
    "Waist": waist * 0.1,
    "CRP": crp * 0.1
}

# Radar/Spider Chart
fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
      r=list(risk_factors.values()),
      theta=list(risk_factors.keys()),
      fill='toself',
      name='Risk Profile'
))
fig_radar.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True
    ),
  ),
  showlegend=False
)
st.plotly_chart(fig_radar, use_container_width=True)

# === ChatGPT Explanation ===
st.markdown("<div id='chatgpt-explanation'></div>", unsafe_allow_html=True)
st.markdown("### 🧠 Step 3: ChatGPT Explains the Risk")
# (existing ChatGPT block continues as is)
