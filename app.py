import streamlit as st
import pandas as pd
from backend.model import load_model
import time
import google.generativeai as genai

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
        st.write("1. Fill in the employee details, job-related information, salary information and satisfactory information in the form.")
        st.write("2. Click on the 'Predict' button to get the prediction results.")
        st.write("3. The prediction results will show whether the employee is likely to leave the company or not.")
        st.write("4. The 'Project Information' section provides details about employee turnover, statistics, case study, and data source.")
        st.write("5. The prediction results will be displayed below the 'Predict' button.")

# ~~~~ Layout: 2 Columns ~~~~


# ~~~~ Column 1 ~~~~
# ~~~~ Employee Details Input ~~~~
with st.sidebar:
    with st.expander("👤 Employee Details", expanded=False):
        age = st.number_input("Age", min_value=18, max_value=65, value=25)
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        total_working_years = st.number_input("Total Working Years", min_value=0, max_value=50, value=5)

        # ~~~~ Employee Job Related Information ~~~~
    with st.expander("🏢 Job Related Information", expanded=False):
        job_involvement = st.slider("Job Involvement", min_value=1, max_value=4, value=3)
        overtime = st.radio("Overtime", ["Yes", "No"])
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=50, value=3) 
        years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=50, value=2)

        # ~~~~ Employee Salary Information ~~~~
    with st.expander("💰 Salary Information (RM)", expanded=False):
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000) 
        daily_rate = st.number_input("Daily Rate", min_value=100, max_value=1500, value=500)
        hourly_rate = st.number_input("Hourly Rate", min_value=5, max_value=100, value=20)
        percent_salary_hike = st.number_input("Percent Salary Hike", min_value=0, max_value=50, value=12) 
    
        # ~~~~ Employee Satisfaction Information ~~~~
    with st.expander("😊 Satisfaction Information", expanded=False):
        job_satisfaction = st.slider("Job Satisfaction", min_value=1, max_value=4, value=3)
        environment_satisfaction = st.slider("Environment Satisfaction", min_value=1, max_value=4, value=3)

    submit = st.button("Compute", type="primary")
    st.divider()
    
    with st.expander("🧪 Experimental Features", expanded=False):
        st.caption("API token can be obtained at https://aistudio.google.com/.")
        gemini_api = st.text_input("Gemini Token", "", type='password')
        try:
            genai.configure(api_key=gemini_api)
            ai_model = genai.GenerativeModel("gemini-1.5-flash")
            test = ai_model.generate_content("Explain how AI works")
            st.success("API key is valid. Experimental feature access granted.")
        except Exception as e:
            st.error("API key is invalid. You don't have access to experimental features.")
            

# ~~~~ Column 2 ~~~~
# ~~~~Display the project information ~~~~
with st.sidebar:
    st.caption("MIT License © 2025 Khor Kean Teng, Ng Jing Wen, Lim Sze Chie, Tan Yee Thong, Yee See Marn")

model = load_model('model/model.pkl')

# Do data transformation here
monthly_income = (monthly_income - 1000)/(20000 - 1000)
daily_rate = (daily_rate - 100)/(1500 - 100)
hourly_rate = (hourly_rate - 5)/(100 - 5)
marital_status = 1 if marital_status == "Divorced" else 2 if marital_status == "Married" else 3
overtime = 1 if overtime == "Yes" else 0

business_travel = 1
department = 1
distance_from_home = 2
education = 3
education_field = 1
gender = 1
job_role = 1
monthly_rate = 1
num_companies_worked = 2
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
    prediction = model.predict(input_data)
    if prediction[0] == 0:
        message = "The employee is not likely to leave the company."
    else:
        message = "The employee is likely to leave the company."
        
    with st.status("Predicting...", expanded = True) as status:
        # Get the prediction
        time.sleep(1) 
        status.update(
            label = "Prediction Results", state="complete", expanded = True
        )
        # Display the prediction
        if prediction[0] == 0:
            st.error(message)
        else:
            st.success(message)

    with st.status("AI Opinion", expanded=True):
        try:
            response = ai_model.generate_content(f"Give some opinions in about 100 word based on the prediction results where the employee is {message}")
            st.write(response.text)
        except Exception as e:
            st.write("You don't have access to this feature. Please authenticate to use this feature.")