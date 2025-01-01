import streamlit as st
import pandas as pd
from backend.model import load_model
from backend.transform import *
import time

# Page title
st.set_page_config(page_title='Employee Turnover Predictor', layout='wide')

# add sidebar
#st.sidebar.title("Employee Turnover Predictor")

# ~~~~ Title ~~~~
with st.sidebar:
    st.title("🕸️ Employee Turnover Prediction Web App")
    # st.write("Predict the likelihood of employee turnover using machine learning model.")

# ~~~~ Add Guidelines to the App ~~~~
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
st.subheader("👤 Employee Details")
col1, col2= st.columns([2,1])
with col1:
    age = st.number_input("Age", min_value=18, max_value=65, value=25)
with col2:
    total_working_years = st.number_input("Total Working Years", min_value=0, max_value=50, value=5)

# ~~~~ Employee Job Related Information ~~~~
st.subheader("🏢 Job Related Information")
job_involvement = st.slider("Job Involvement", min_value=1, max_value=4, value=3) # notes: what is job involvement? Might need to explain
col1, col2= st.columns([2,1])
with col1:
    years_at_company = st.number_input("Years at Company", min_value=0, max_value=50, value=3) # notes: what if the employee is new?
with col2:    
    years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=50, value=2) # notes: what if the employee is new?

# ~~~~ Employee Salary Information ~~~~
st.subheader("💰 Salary Information")
col1, col2= st.columns([2,1])
with col1:
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000) # notes: what is the currency? Might need to specify
    daily_rate = st.number_input("Daily Rate", min_value=100, max_value=1500, value=500)
with col2:
    hourly_rate = st.number_input("Hourly Rate", min_value=5, max_value=100, value=20)
    percent_salary_hike = st.number_input("Percent Salary Hike", min_value=0, max_value=50, value=12) # notes: what is percent salary hike? Might need to explain

# Notes: What if the user do not input the data? Might need to add validation

# ~~~~ Column 2 ~~~~
# ~~~~Display the project information ~~~~
with st.sidebar:
    # st.header("Project Information")
    tab1, tab2, tab3, tab4 = st.tabs(["About", "Statistics", "Case Study", "Data Source"])

    with tab1:
        st.subheader("About Employee Turnover")
        st.write(
            "Employee turnover, also known as employee attrition, refers to the number of workers leaving an \
            organization over a specified time. It disrupts operations, increases recruitment costs, and impacts \
            productivity, competitiveness, and profitability.")
    with tab2:
        st.subheader("Malaysian Employee Turnover Statistics")
        st.write("- Nearly 49% of Malaysian organizations face employee turnover issues (Al-Suraihi et al., 2021).")
        st.write("- Voluntary turnover rates rose from 6.5% (early 2019) to 8.7% (2020) (Bibi Nabi & Zahir, 2024).")
        st.write("- Manufacturing sector turnover: 24% in 2019 (Kin et al., 2022).")
        st.write("- FMCG sector voluntary turnover: 8.4% in 2020.")

    with tab3:
        st.subheader("Case Study Highlight")
        st.write("Xerox’s Call Centre reduced turnover by 20% using predictive analytics to identify patterns \
#             linked to attrition and improve employee engagement. (Solutyics, 2023)")

    with tab4:
        st.subheader("Data Source")
        st.write(
            "The dataset used in this project is from the IBM HR Analytics Employee Attrition & Performance dataset \
            available on Kaggle. It is a synthetic dataset with 1,470 observations and 35 features, covering \
            employee background, employment details, and satisfaction metrics.")

    # Independent Section
    # ~~~~Disclaimer~~~~
    st.markdown("---")  # Horizontal separator
    st.caption("""
        **Disclaimer:** 
        This web app is intended for prediction purposes only. The results are based on the input data provided and \
        the performance of the machine learning model. The accuracy of the predictions may vary depending on data quality \
        and model reliability.  

        It is recommended to consult HR professionals for accurate and comprehensive employee turnover analysis. \
        Please avoid making critical decisions or imposing penalties solely based on the predictions generated by this web app.
    """)

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
st.markdown("---")  # Horizontal separator
if st.button("Predict", type="primary"):
    with st.spinner("Predicting..."):
         # set a 1.5 seconds delay
        time.sleep(1.5)
        
        # Get the prediction
        prediction = model.predict(input_data)

        # Display the prediction
        if prediction[0] == 0:
            st.error("The employee is not likely to leave the company.")
        else:
            st.success("The employee is likely to leave the company.")
