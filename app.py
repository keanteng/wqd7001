import streamlit as st
import pandas as pd
from backend.model import load_model
import time

# Page title
st.set_page_config(page_title='Employee Turnover Predictor', layout='wide')

# add sidebar
#st.sidebar.title("Employee Turnover Predictor")

# ~~~~ Title ~~~~
st.title("🧿 Employee Turnover Prediction")
st.markdown("""An example app powered by Streamlit to predict the likelihood of employee turnover using machine learning model""")

# ~~~~ Add Guidelines to the App ~~~~
with st.sidebar:
    toggle = st.toggle("Show Guidelines", True)
    st.write("**Input Details**")

if toggle:
    with st.expander("💡 Guidelines", expanded=True): 
        st.write("Follow the steps below to predict the likelihood of employee turnover:")
        st.write("1. Fill in the employee details, job-related information, and salary information in the form.")
        st.write("2. Click on the 'Predict' button to get the prediction results.")
        st.write("3. The prediction results will show whether the employee is likely to leave the company or not.")
        st.write("4. The 'Project Information' section provides details about employee turnover, statistics, case study, and data source.")
        st.write("5. The predictioon results will be displayed below the 'Predict' button.")

# ~~~~ Layout: 2 Columns ~~~~


# ~~~~ Column 1 ~~~~
# ~~~~ Employee Details Input ~~~~
with st.sidebar:
    with st.expander("👤 Employee Details", expanded=False):
        age = st.number_input("Age", min_value=18, max_value=65, value=25)
        total_working_years = st.number_input("Total Working Years", min_value=0, max_value=50, value=5)

        # ~~~~ Employee Job Related Information ~~~~
    with st.expander("🏢 Job Related Information", expanded=False):
        job_involvement = st.slider("Job Involvement", min_value=1, max_value=4, value=3)
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=50, value=3) # notes: what if the employee is new?  
        years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=50, value=2) # notes: what if the employee is new?

        # ~~~~ Employee Salary Information ~~~~
    with st.expander("💰 Salary Information", expanded=False):
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000) # notes: what is the currency? Might need to specify
        daily_rate = st.number_input("Daily Rate", min_value=100, max_value=1500, value=500)
        hourly_rate = st.number_input("Hourly Rate", min_value=5, max_value=100, value=20)
        percent_salary_hike = st.number_input("Percent Salary Hike", min_value=0, max_value=50, value=12) # notes: what is percent salary hike? Might need to explain
    
    submit = st.button("Compute", type="primary")

# Notes: What if the user do not input the data? Might need to add validation

# ~~~~ Column 2 ~~~~
# ~~~~Display the project information ~~~~
with st.sidebar:
    st.divider()
    st.caption("MIT License © 2025 Khor Kean Teng, Ng Jing Wen, Lim Sze Chie, Tan Yee Thong, Yee See Marn")

model = load_model('model/model.pkl')

# Do data transformation here
monthly_income = (monthly_income - 1000)/(20000 - 1000)
daily_rate = (daily_rate - 100)/(1500 - 100)

# Create a DataFrame for the input data
# input_data = pd.DataFrame({
#     'Age': [age],
#     'TotalWorkingYears': [total_working_years],
#     'JobInvolvement': [job_involvement],
#     'YearsAtCompany': [years_at_company],
#     'YearsInCurrentRole': [years_in_current_role],
#     'MonthlyIncome': [monthly_income],
#     'DailyRate': [daily_rate],
#     'HourlyRate': [hourly_rate],
#     'PercentSalaryHike': [percent_salary_hike]
# })

# default data (random)
business_travel = 1
department = 1
distance_from_home = 2
education = 3
education_field = 1
environment_satisfaction = 3
gender = 1
job_role = 1
job_satisfaction = 4
marital_status = 1
monthly_rate = 1
num_companies_worked = 2
overtime = 1
performance_rating = 3
relationship_satisfaction = 3
stock_option_level = 0
training_times_last_year = 2
work_life_balance = 3
years_since_last_promotion = 0
years_with_curr_manager = 0

input_data = pd.DataFrame({
    "Age": [age],
    "BusinessTravel": [business_travel],
    "DailyRate": [daily_rate],
    "Department": [department],
    "DistanceFromHome": [distance_from_home],
    "Education": [education],
    "EducationField": [education_field],
    "EnvironmentSatisfaction": [environment_satisfaction],
    "Gender": [gender],
    "HourlyRate": [hourly_rate],
    "JobInvolvement": [job_involvement],
    "JobRole": [job_role],
    "JobSatisfaction": [job_satisfaction],
    "MaritalStatus": [marital_status],
    "MonthlyIncome": [monthly_income],
    "MonthlyRate": [monthly_rate],
    "NumCompaniesWorked": [num_companies_worked],
    "OverTime": [overtime],
    "PercentSalaryHike": [percent_salary_hike],
    "PerformanceRating": [performance_rating],
    "RelationshipSatisfaction": [relationship_satisfaction],
    "StockOptionLevel": [stock_option_level],
    "TotalWorkingYears": [total_working_years],
    "TrainingTimesLastYear": [training_times_last_year],
    "WorkLifeBalance": [work_life_balance],
    "YearsAtCompany": [years_at_company],
    "YearsInCurrentRole": [years_in_current_role],
    "YearsSinceLastPromotion": [years_since_last_promotion],
    "YearsWithCurrManager": [years_with_curr_manager]
})

# ~~~~ Predict Button ~~~~
if submit:
    with st.status("Predicting...", expanded = True) as status:
         # set a 1.5 seconds delay
        # Get the prediction
        prediction = model.predict(input_data)
        time.sleep(1) 
        status.update(
            label = "Prediction Results", state="complete", expanded = True
        )
        
        # Display the prediction
        if prediction[0] == 0:
            st.error("The employee is not likely to leave the company.")
        else:
            st.success("The employee is likely to leave the company.")
