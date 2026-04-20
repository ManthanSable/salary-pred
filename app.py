import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('linear_regression_model.pkl')

# Load dataset
df = pd.read_csv('Salary_Dataset_DataScienceLovers.csv')

st.title('💼 Salary Prediction App')

# Dynamic categories from dataset
company_list = sorted(df['Company_Name'].unique())
job_title_list = sorted(df['Job Title'].unique())
location_list = sorted(df['Location'].unique())
employment_list = sorted(df['Employment Status'].unique())
job_roles_list = sorted(df['Job Roles'].unique())

# Inputs
rating = st.slider('Rating', 0.0, 5.0, 3.5)

company = st.selectbox('Company Name', company_list)
job_title = st.selectbox('Job Title', job_title_list)
location = st.selectbox('Location', location_list)
employment_status = st.selectbox('Employment Status', employment_list)
job_roles = st.selectbox('Job Roles', job_roles_list)

salaries_reported = st.number_input('Salaries Reported', min_value=1, value=1)

# ⚠️ Encoding (simple method)
from sklearn.preprocessing import LabelEncoder

le_company = LabelEncoder()
le_job = LabelEncoder()
le_loc = LabelEncoder()
le_emp = LabelEncoder()
le_role = LabelEncoder()

df['Company Name'] = le_company.fit_transform(df['Company Name'])
df['Job Title'] = le_job.fit_transform(df['Job Title'])
df['Location'] = le_loc.fit_transform(df['Location'])
df['Employment Status'] = le_emp.fit_transform(df['Employment Status'])
df['Job Roles'] = le_role.fit_transform(df['Job Roles'])

# Predict
if st.button('Predict Salary'):

    input_data = pd.DataFrame([{
        'Rating': rating,
        'Company Name': le_company.transform([company])[0],
        'Job Title': le_job.transform([job_title])[0],
        'Salaries Reported': salaries_reported,
        'Location': le_loc.transform([location])[0],
        'Employment Status': le_emp.transform([employment_status])[0],
        'Job Roles': le_role.transform([job_roles])[0]
    }])

    prediction = model.predict(input_data)[0]

    st.success(f'💰 Predicted Salary: {prediction:,.2f} INR')
