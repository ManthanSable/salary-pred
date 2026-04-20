
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('linear_regression_model.pkl')

st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary.')

# Input features from the user
# Note: For 'Company Name', 'Job Title', 'Location', 'Employment Status', 'Job Roles',
# the model expects the LabelEncoded integer values. In a real application, you would
# either provide dropdowns with original categories and map them to encoded values,
# or load the LabelEncoders used during training.

rating = st.slider('Rating', min_value=0.0, max_value=5.0, value=3.5, step=0.1)
company_name = st.number_input('Company Name (Encoded Integer)', min_value=0, value=5000)
job_title = st.number_input('Job Title (Encoded Integer)', min_value=0, value=50)
salaries_reported = st.number_input('Salaries Reported', min_value=0, value=1)
location = st.number_input('Location (Encoded Integer)', min_value=0, value=5)
employment_status = st.number_input('Employment Status (Encoded Integer)', min_value=0, value=1)
job_roles = st.number_input('Job Roles (Encoded Integer)', min_value=0, value=10)


if st.button('Predict Salary'):
    # Create a DataFrame from user inputs
    input_data = pd.DataFrame([{
        'Rating': rating,
        'Company Name': company_name,
        'Job Title': job_title,
        'Salaries Reported': salaries_reported,
        'Location': location,
        'Employment Status': employment_status,
        'Job Roles': job_roles
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.success(f'Predicted Salary: {prediction:,.2f} INR')
