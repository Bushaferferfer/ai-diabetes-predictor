# === AI-Powered Diabetes Predictor with Extended Parameters ===

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import openai

# === Page Config ===
st.set_page_config(page_title="AI Diabetes Predictor", layout="wide")

# === Apply Dark Theme Styling ===
st.markdown("""
    <style>
    body { background-color: #0e1117; color: white; }
    .stApp { background-color: #0e1117; color: white; }
    h1, h2, h3, h4, h5, h6 { color: #4CAF50; }
    .stMarkdown, .stRadio, .stSlider, .stSelectbox, .stTextInput, .stTextArea, .stButton {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# === OpenAI Client ===
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# === Sidebar ===
with st.sidebar:
    st.image("https://uskudar.edu.tr/assets/img/logo-en.png", width=180)
    st.title("\U0001F4D8 Project Info")
    st.markdown("""
    **\U0001F393 Graduation Thesis**  
    *AI in Diagnosis & Early Detection of Type 2 Diabetes*  
    \U0001F468‍\U0001F393 Achraf Farki – Üsküdar University  
    \U0001F4C5 2025  
    —  
    \U0001F4A1 Powered by GPT-3.5 Turbo & Machine Learning
    """)

# === Main Title ===
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>\U0001F916 AI Diabetes Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Expanded with clinical and lifestyle data for better diagnosis.</p>", unsafe_allow_html=True)
st.markdown("---")

# === Patient Information ===
st.header("\U0001F4CB Step 1: Patient Personal & Health Information")
with st.expander("Enter extended patient details", expanded=True):
    patient_name = st.text_input("\U0001F464 Patient Full Name", value="John Doe")
    patient_id = st.text_input("\U0001F194 Patient ID", value="DFX-2025-001")
    visit_reason = st.text_area("\U0001F4C4 Reason for Visit", placeholder="e.g. Regular checkup, symptoms of fatigue, blood sugar monitoring...")
    notes = st.text_area("\U0001FA7A Chronic Conditions / Notes", placeholder="e.g. Has history of heart disease, chronic kidney condition, or other diagnoses...")

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

# === Output ===
st.markdown("### \U0001F3AF Step 2: AI-Predicted Diabetes Risk")
st.subheader(f"\U0001F9EA Patient: {patient_name}")
st.subheader(f"\U0001F50E Predicted Risk: **{risk * 100:.2f}%**")
if risk >= 0.7:
    st.error("\U0001F534 High risk. Immediate attention recommended.")
elif risk >= 0.4:
    st.warning("\U0001F7E0 Moderate risk. Monitor closely.")
else:
    st.success("\U0001F7E2 Low risk. Maintain current lifestyle.")

# === Visual Risk Breakdown ===
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
st.markdown("#### \U0001FA7E Estimated Risk Contributions")
fig, ax = plt.subplots(figsize=(3.8, 2.6))
ax.bar(risk_factors.keys(), risk_factors.values(), color='salmon')
ax.set_title("Estimated Risk Contributions", fontsize=10)
ax.set_ylabel("Weighted Value", fontsize=9)
plt.xticks(rotation=30, fontsize=8)
plt.yticks(fontsize=8)
st.pyplot(fig, use_container_width=False)

# === ChatGPT Explanation ===
st.markdown("### \U0001F9E0 Step 3: ChatGPT Explains the Risk")
input_summary = "\n".join([
    f"Patient Name: {patient_name}",
    f"Patient ID: {patient_id}",
    f"Visit Reason: {visit_reason if visit_reason else 'N/A'}",
    f"Age: {age}",
    f"Gender: {gender}",
    f"BMI: {bmi}",
    f"Glucose: {glucose}",
    f"Blood Pressure: {bp}",
    f"Insulin: {insulin}",
    f"HDL: {hdl}",
    f"Triglycerides: {triglycerides}",
    f"CRP: {crp}",
    f"Smoking: {smoking}",
    f"Activity: {activity}",
    f"Sleep Hours: {sleep}",
    f"Sugar Intake: {sugar_intake}",
    f"Family History: {family_history}",
    f"Other Notes: {notes if notes else 'N/A'}",
    f"Predicted Risk: {risk * 100:.2f}%"
])

chat_messages = [
    {"role": "system", "content": "You are a medical AI assistant. Provide a gentle, simple, professional explanation for diabetes risk."},
    {"role": "user", "content": f"Please explain the following patient risk report:\n\n{input_summary}"}
]

if st.button("\U0001F916 Generate AI Explanation"):
    with st.spinner("ChatGPT is analyzing the patient's data..."):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=chat_messages
            )
            explanation = response.choices[0].message.content
            st.success("\U0001F4A1 Explanation:")
            st.markdown(explanation)
        except Exception as e:
            st.error(f"❌ ChatGPT API Error: {str(e)}")

# === Footer ===
st.markdown("""
---
<div style='text-align: center; font-size: 0.9em;'>
\U0001F4D8 <em>Thesis-based AI Project</em> | <strong>Üsküdar University</strong><br>
\U0001F9EC Developed by <strong>Achraf Farki</strong> | 2025
</div>
""", unsafe_allow_html=True)
