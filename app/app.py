import streamlit as st
import pandas as pd
import joblib
import json
import os
 
st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="centered")
 
MODEL_PATH = os.path.join("..", "output", "heart_models", "heart_disease_best_pipeline.pkl")
INFO_PATH = os.path.join("..", "output", "heart_models", "best_model_info.json")
 
@st.cache_resource
def load_model():
    pipeline = joblib.load(MODEL_PATH)
    with open(INFO_PATH) as f:
        info = json.load(f)
    return pipeline, info
 
pipeline, info = load_model()
 
st.title("❤️ Heart Disease Risk Predictor")
st.caption(
    f"Powered by a tuned **{info['best_model']}** model "
    f"(Test Accuracy: {info['test_accuracy']*100:.1f}% · ROC-AUC: {info['test_roc_auc']:.3f})"
)
st.markdown(
    "Enter the patient's clinical details below to estimate the likelihood of heart disease. "
    "This tool is for **educational purposes only** and is not a substitute for professional medical diagnosis."
)
 
st.divider()
 
with st.form("patient_form"):
    st.subheader("Patient Details")
 
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=50)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
        cp = st.selectbox(
            "Chest Pain Type", options=[
                ("Typical Angina", 0), ("Atypical Angina", 1),
                ("Non-Anginal Pain", 2), ("Asymptomatic", 3)
            ], format_func=lambda x: x[0]
        )
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        restecg = st.selectbox(
            "Resting ECG Result", options=[
                ("Normal", 0), ("ST-T Wave Abnormality", 1), ("Left Ventricular Hypertrophy", 2)
            ], format_func=lambda x: x[0]
        )
 
    with col2:
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment", options=[
                ("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)
            ], format_func=lambda x: x[0]
        )
        ca = st.selectbox("Number of Major Vessels Colored (0-4)", options=[0, 1, 2, 3, 4])
        thal = st.selectbox(
            "Thalassemia", options=[
                ("Normal", 1), ("Fixed Defect", 2), ("Reversible Defect", 3), ("Unknown", 0)
            ], format_func=lambda x: x[0]
        )
 
    submitted = st.form_submit_button("🔍 Predict Heart Disease Risk", use_container_width=True)
 
if submitted:
    input_df = pd.DataFrame([{
        "age": age, "sex": sex[1], "cp": cp[1], "trestbps": trestbps, "chol": chol,
        "fbs": fbs[1], "restecg": restecg[1], "thalach": thalach, "exang": exang[1],
        "oldpeak": oldpeak, "slope": slope[1], "ca": ca, "thal": thal[1],
    }])[info["feature_columns"]]
 
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]
 
    # Risk category thresholds
    if probability < 0.33:
        risk_level, risk_color = "Low", "green"
    elif probability < 0.66:
        risk_level, risk_color = "Medium", "orange"
    else:
        risk_level, risk_color = "High", "red"
 
    st.divider()
    st.subheader("Prediction Result")
 
    c1, c2 = st.columns(2)
    with c1:
        if prediction == 1:
            st.error("⚠️ Heart Disease Risk Detected")
        else:
            st.success("✅ No Significant Heart Disease Risk Detected")
    with c2:
        st.metric("Predicted Probability", f"{probability*100:.1f}%")
 
    st.markdown(f"### Risk Category: :{risk_color}[**{risk_level}**]")
    st.progress(float(probability))
 
    st.divider()
    st.subheader("💡 Health Recommendations")
 
    if risk_level == "Low":
        st.markdown(
            "- Continue maintaining a heart-healthy lifestyle (balanced diet, regular exercise).\n"
            "- Routine annual check-ups are recommended.\n"
            "- Monitor blood pressure and cholesterol periodically."
        )
    elif risk_level == "Medium":
        st.markdown(
            "- Schedule a consultation with a cardiologist for further evaluation.\n"
            "- Consider lifestyle modifications: reduce sodium/saturated fat intake, increase physical activity.\n"
            "- Monitor blood pressure, cholesterol, and blood sugar more frequently.\n"
            "- Manage stress and avoid tobacco/excessive alcohol use."
        )
    else:
        st.markdown(
            "- **Seek prompt medical evaluation from a cardiologist.**\n"
            "- Discuss further diagnostic tests (ECG, stress test, angiography) with a physician.\n"
            "- Closely monitor blood pressure, cholesterol, and blood sugar.\n"
            "- Adopt an immediate heart-healthy lifestyle plan under medical supervision.\n"
            "- Do not delay care if experiencing chest pain, shortness of breath, or related symptoms."
        )
 
    st.caption(
        "⚠️ Disclaimer: This prediction is generated by a statistical model trained on a limited dataset "
        "and is intended for educational/demonstration purposes only. It is **not** a medical diagnosis. "
        "Always consult a qualified healthcare professional for medical advice."
    )
 
st.divider()
with st.expander("ℹ️ About this model"):
    st.markdown(
        f"""
        This app uses a **{info['best_model']}** model trained on the UCI Heart Disease dataset
        (302 unique patient records after de-duplication), optimized via GridSearchCV with 5-fold
        stratified cross-validation.
 
        - **Test Accuracy:** {info['test_accuracy']*100:.1f}%
        - **Test ROC-AUC:** {info['test_roc_auc']:.3f}
 
        Logistic Regression was selected as the final deployed model because it offers the best
        balance of accuracy, precision, and F1-score among tuned candidates, while remaining fully
        interpretable — a requirement for generating clinically meaningful explanations.
        """
    )
 