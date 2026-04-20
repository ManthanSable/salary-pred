import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

# --- 1. Load the trained model ---
model = joblib.load('linear_regression_model.pkl')

# --- 2. Load the original dataset to reconstruct encoders ---
try:
    original_df = pd.read_csv('Salary_Dataset_DataScienceLovers.csv')
except FileNotFoundError:
    st.error("Error: 'Salary_Dataset_DataScienceLovers.csv' not found. Please ensure it's in the same directory as app.py.")
    st.stop()

# --- 3. Apply the same preprocessing steps as during training and create individual LabelEncoders ---
# Handle null values for object type columns (specifically 'Company Name')
for column in original_df.columns:
    if original_df[column].dtype == 'object':
        original_df[column] = original_df[column].fillna(original_df[column].mode()[0])

# Create separate LabelEncoder instances for each categorical column
le_company_name = LabelEncoder()
le_job_title = LabelEncoder()
le_location = LabelEncoder()
le_employment_status = LabelEncoder()
le_job_roles = LabelEncoder()

# Ensure columns are string type and fit encoders
original_df['Company Name'] = original_df['Company Name'].astype(str)
le_company_name.fit(original_df['Company Name'])

original_df['Job Title'] = original_df['Job Title'].astype(str)
le_job_title.fit(original_df['Job Title'])

original_df['Location'] = original_df['Location'].astype(str)
le_location.fit(original_df['Location'])

original_df['Employment Status'] = original_df['Employment Status'].astype(str)
le_employment_status.fit(original_df['Employment Status'])

original_df['Job Roles'] = original_df['Job Roles'].astype(str)
le_job_roles.fit(original_df['Job Roles'])

st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary.')

# --- 4. Input features from the user using dropdowns for categorical data ---
rating = st.slider('Rating', min_value=0.0, max_value=5.0, value=3.5, step=0.1)

# For categorical features, use selectbox and map back to encoded integers
selected_company_name = st.selectbox('Company Name', sorted(le_company_name.classes_))
selected_job_title = st.selectbox('Job Title', sorted(le_job_title.classes_))
selected_location = st.selectbox('Location', sorted(le_location.classes_))
selected_employment_status = st.selectbox('Employment Status', sorted(le_employment_status.classes_))
selected_job_roles = st.selectbox('Job Roles', sorted(le_job_roles.classes_))

salaries_reported = st.number_input('Salaries Reported', min_value=0, value=1)

if st.button('Predict Salary'):
    # Encode selected categorical values
    encoded_company_name = le_company_name.transform([selected_company_name])[0]
    encoded_job_title = le_job_title.transform([selected_job_title])[0]
    encoded_location = le_location.transform([selected_location])[0]
    encoded_employment_status = le_employment_status.transform([selected_employment_status])[0]
    encoded_job_roles = le_job_roles.transform([selected_job_roles])[0]

    # Create a DataFrame from user inputs (using encoded values for categorical features)
    input_data = pd.DataFrame([{
        'Rating': rating,
        'Company Name': encoded_company_name,
        'Job Title': encoded_job_title,
        'Salaries Reported': salaries_reported,
        'Location': encoded_location,
        'Employment Status': encoded_employment_status,
        'Job Roles': encoded_job_roles
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.success(f'Predicted Salary: {prediction:,.2f} INR')
