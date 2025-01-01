import streamlit as st
import pandas as pd

st.set_page_config(page_title='Documentation', layout='wide')

with st.sidebar:
    with st.expander("⚠️ Disclaimer", expanded=True):
        st.write("This web app is intended for prediction purposes only. The results are based on the input data provided and \
        the performance of the machine learning model. The accuracy of the predictions may vary depending on data quality \
        and model reliability.")
        
    st.caption("MIT License © 2025 Khor Kean Teng, Ng Jing Wen, Lim Sze Chie, Tan Yee Thong, Yee See Marn")

st.title("📄 Documentation")
st.markdown("""
            To learn more about the project, please refer to the sections below.
            """)
st.subheader("About Employee Turnover")
st.write(
    "Employee turnover, also known as employee attrition, refers to the number of workers leaving an \
    organization over a specified time. It disrupts operations, increases recruitment costs, and impacts \
    productivity, competitiveness, and profitability.")

st.subheader("Malaysian Employee Turnover Statistics")
st.write("- Nearly 49% of Malaysian organizations face employee turnover issues (Al-Suraihi et al., 2021).")
st.write("- Voluntary turnover rates rose from 6.5% (early 2019) to 8.7% (2020) (Bibi Nabi & Zahir, 2024).")
st.write("- Manufacturing sector turnover: 24% in 2019 (Kin et al., 2022).")
st.write("- FMCG sector voluntary turnover: 8.4% in 2020.")

st.subheader("Case Study Highlight")
st.write("Xerox’s Call Centre reduced turnover by 20% using predictive analytics to identify patterns \
#             linked to attrition and improve employee engagement. (Solutyics, 2023)")

st.subheader("Data Source")
st.write(
    "The dataset used in this project is from the IBM HR Analytics Employee Attrition & Performance dataset \
    available on Kaggle. It is a synthetic dataset with 1,470 observations and 35 features, covering \
    employee background, employment details, and satisfaction metrics. Below attached the sample data for reference.")

# set up the download data
data = pd.read_csv("data/sample_data.csv")
data = data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Sample Data",
    data=data,
    file_name="sample_data.csv",
    mime="text/csv"
)