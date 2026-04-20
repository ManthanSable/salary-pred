import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and encoders
model = joblib.load('linear_regression_model.pkl')

# Load encoders (you must save these during training)
company_encoder = joblib.load('company_encoder.pkl')
job_title_encoder = joblib.load('job_title_encoder.pkl')
location_encoder = joblib.load('location_encoder.pkl')
employment_encoder = joblib.load('employment_encoder.pkl')
job_roles_encoder = joblib.load('job_roles_encoder.pkl')

st.title('💼 Salary Prediction App')
st.write('Select the details below to predict the salary.')

# User Inputs (Categorical)
rating = st.slider('Rating', 0.0, 5.0, 3.5)

company = st.selectbox('Company Name', company_encoder.classes_)
job_title = st.selectbox('Job Title', job_title_encoder.classes_)
location = st.selectbox('Location', location_encoder.classes_)
employment_status = st.selectbox('Employment Status', employment_encoder.classes_)
job_roles = st.selectbox('Job Roles', job_roles_encoder.classes_)

salaries_reported = st.number_input('Salaries Reported', min_value=1, value=1)

# Prediction
if st.button('Predict Salary'):
    
    # Encode categorical values
    company_encoded = company_encoder.transform([company])[0]
    job_title_encoded = job_title_encoder.transform([job_title])[0]
    location_encoded = location_encoder.transform([location])[0]
    employment_encoded = employment_encoder.transform([employment_status])[0]
    job_roles_encoded = job_roles_encoder.transform([job_roles])[0]

    # Create DataFrame
    input_data = pd.DataFrame([{
        'Rating': rating,
        'Company Name': company_encoded,
        'Job Title': job_title_encoded,
        'Salaries Reported': salaries_reported,
        'Location': location_encoded,
        'Employment Status': employment_encoded,
        'Job Roles': job_roles_encoded
    }])

    # Prediction
    prediction = model.predict(input_data)[0]

    st.success(f'💰 Predicted Salary: {prediction:,.2f} INR')
