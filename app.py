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

# --- 3. Apply the same preprocessing steps as during training ---
# Handle null values for object type columns (specifically 'Company Name')
for column in original_df.columns:
    if original_df[column].dtype == 'object':
        original_df[column] = original_df[column].fillna(original_df[column].mode()[0])

# Re-initialize and fit LabelEncoders for categorical columns
label_encoders = {}
categorical_cols = ['Company Name', 'Job Title', 'Location', 'Employment Status', 'Job Roles']

for col in categorical_cols:
    le = LabelEncoder()
    # Ensure column is string type before fitting encoder
    original_df[col] = original_df[col].astype(str)
    le.fit(original_df[col])
    label_encoders[col] = le

st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary.')

# --- 4. Input features from the user using dropdowns for categorical data ---
rating = st.slider('Rating', min_value=0.0, max_value=5.0, value=3.5, step=0.1)

# For categorical features, use selectbox and map back to encoded integers
selected_company_name = st.selectbox('Company Name', sorted(label_encoders['Company Name'].classes_))
selected_job_title = st.selectbox('Job Title', sorted(label_encoders['Job Title'].classes_))
selected_location = st.selectbox('Location', sorted(label_encoders['Location'].classes_))
selected_employment_status = st.selectbox('Employment Status', sorted(label_encoders['Employment Status'].classes_))
selected_job_roles = st.selectbox('Job Roles', sorted(label_encoders['Job Roles'].classes_))

salaries_reported = st.number_input('Salaries Reported', min_value=0, value=1)

if st.button('Predict Salary'):
    # Encode selected categorical values
    encoded_company_name = label_encoders['Company Name'].transform([selected_company_name])[0]
    encoded_job_title = label_encoders['Job Title'].transform([selected_job_title])[0]
    encoded_location = label_encoders['Location'].transform([selected_location])[0]
    encoded_employment_status = label_encoders['Employment Status'].transform([selected_employment_status])[0]
    encoded_job_roles = label_encoders['Job Roles'].transform([selected_job_roles])[0]

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

    # Ensure column order matches training data (important for consistent predictions)
    # Assuming X_train columns order were: Rating, Company Name, Job Title, Salaries Reported, Location, Employment Status, Job Roles
    # This is critical if the model was trained with a specific column order.
    # For this model, the order during training was: Rating, Company Name, Job Title, Salaries Reported, Location, Employment Status, Job Roles
    # It's good practice to ensure this for prediction.
    # The order of input_data creation already matches the X_train used in the notebook based on the original problem's X definition.

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.success(f'Predicted Salary: {prediction:,.2f} INR')
