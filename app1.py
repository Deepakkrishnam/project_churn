import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load model, scaler, and feature names
model = joblib.load(open('churn_model.pkl', 'rb'))
scaler = joblib.load(open('scaler.pkl', 'rb'))
feature_names = joblib.load('feature_names.pkl')

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

h1 {
    color: #1f4e79;
    text-align: center;
    font-size: 42px;
}

.block-container {
    padding-top: 2rem;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.stButton>button {
    background-color: #1f77b4;
    color: white;
    border-radius: 12px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background-color: #145a86;
    color: white;
}

.prediction-box {
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    font-size: 25px;
    font-weight: bold;
    margin-top: 25px;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #555;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# App Title
st.title("📊 Customer Churn Prediction System")

st.markdown("""
<div class="subtitle">
Predict whether a customer is likely to leave the service based on customer details.
</div>
""", unsafe_allow_html=True)

# Input Section
st.markdown("## 👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Personal Details")
    gender = st.selectbox('Gender', ['Female', 'Male'])
    senior = st.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.selectbox('Has Partner?', ['No', 'Yes'])
    dependents = st.selectbox('Has Dependents?', ['No', 'Yes'])
    tenure = st.number_input('Tenure Months', min_value=0, max_value=72, value=1)

with col2:
    st.markdown("### Service Details")
    phone = st.selectbox('Phone Service', ['No', 'Yes'])
    multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
    internet = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])

with col3:
    st.markdown("### Billing Details")
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless = st.selectbox('Paperless Billing', ['No', 'Yes'])
    payment = st.selectbox(
        'Payment Method',
        [
            'Electronic check',
            'Mailed check',
            'Bank transfer (automatic)',
            'Credit card (automatic)'
        ]
    )
    monthly_charges = st.number_input('Monthly Charges', min_value=0.0, max_value=150.0, value=50.0)
    total_charges = st.number_input('Total Charges', min_value=0.0, max_value=10000.0, value=50.0)

st.markdown("---")

# Prediction Button
if st.button('Predict Churn'):

    cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
        'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes',
        'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
        'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
        'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
        'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
        'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
        'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
        'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes',
        'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
        'PaymentMethod_Bank transfer (automatic)',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed check'
    ]

    input_df = pd.DataFrame(0, index=[0], columns=cols)

    input_df['gender'] = 1 if gender == 'Male' else 0
    input_df['SeniorCitizen'] = 1 if senior == 'Yes' else 0
    input_df['Partner'] = 1 if partner == 'Yes' else 0
    input_df['Dependents'] = 1 if dependents == 'Yes' else 0
    input_df['PhoneService'] = 1 if phone == 'Yes' else 0
    input_df['PaperlessBilling'] = 1 if paperless == 'Yes' else 0

    input_df['tenure'] = tenure
    input_df['MonthlyCharges'] = monthly_charges
    input_df['TotalCharges'] = total_charges

    input_df[f'MultipleLines_{multiple_lines}'] = 1
    input_df[f'InternetService_{internet}'] = 1
    input_df[f'OnlineSecurity_{security}'] = 1
    input_df[f'OnlineBackup_{backup}'] = 1
    input_df[f'Contract_{contract}'] = 1
    input_df[f'PaymentMethod_{payment}'] = 1

    # Match training feature order
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Scaling
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0][1]

    # Output
    if prediction[0] == 1:
        st.markdown(f"""
        <div class="prediction-box" style="background-color:#ffdddd; color:#b30000;">
            ❌ CUSTOMER WILL LEAVE <br><br>
            Churn Risk Score: {prob:.2f}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box" style="background-color:#ddffdd; color:#006600;">
            ✅ CUSTOMER WILL STAY <br><br>
            Churn Risk Score: {prob:.2f}
        </div>
        """, unsafe_allow_html=True)

    st.progress(float(prob))

    st.info("Risk Score closer to 1 means higher chance of customer churn.")