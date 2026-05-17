import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# 1. Load the model and the scaler
model = joblib.load(open('churn_model.pkl', 'rb'))
scaler = joblib.load(open('scaler.pkl', 'rb'))
feature_names = joblib.load('feature_names.pkl')


st.title('Customer Churn Prediction App')
st.write("Fill in the customer details to predict churn risk.")

# 2. Define User Inputs
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox('Gender', ['Female', 'Male'])
    senior = st.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.selectbox('Has Partner?', ['No', 'Yes'])
    dependents = st.selectbox('Has Dependents?', ['No', 'Yes'])
    tenure = st.number_input('Tenure (Months)', 0, 72, 1)

with col2:
    phone = st.selectbox('Phone Service', ['No', 'Yes'])
    multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
    internet = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])

with col3:
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless = st.selectbox('Paperless Billing', ['No', 'Yes'])
    payment = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges = st.number_input('Monthly Charges', 0.0, 150.0, 50.0)
    total_charges = st.number_input('Total Charges', 0.0, 10000.0, 50.0)

# 3. Preprocessing Logic 
if st.button('Predict Churn'):
    
    cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 
        'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'MultipleLines_No', 
        'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_DSL', 
        'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No', 
        'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No', 
        'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No', 
        'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No', 
        'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No', 
        'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No', 
        'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_Month-to-month', 
        'Contract_One year', 'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)', 
        'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
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
    
    # 4. Scaling 
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    input_scaled = scaler.transform(input_df)

    # 5. Final Prediction
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction[0] == 1:
        st.error(f"Prediction: CUSTOMER WILL LEAVE (Risk Score: {prob:.2f})")
    else:
        st.success(f"Prediction: CUSTOMER WILL STAY (Risk Score: {prob:.2f})")