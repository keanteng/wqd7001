import streamlit as st
from backend.bot import *
from backend.model import *
import pandas as pd

# Page title
st.set_page_config(page_title='Batch Processing', layout='wide')

st.title("Attrition Prediction Engine")
st.write("Welcome to the Attrition Prediction Engine! This tool is designed to help you batch process employee data and predict attrition.")

with st.sidebar:
    with st.expander("⚠️ Disclaimer", expanded=True):
        st.write("This web app is intended for prediction purposes only. The results are based on the input data provided and \
        the performance of the machine learning model. The accuracy of the predictions may vary depending on data quality \
        and model reliability.")
        
    st.caption("MIT License © 2025 Khor Kean Teng, Ng Jing Wen, Lim Sze Chie, Tan Yee Thong, Yee See Marn")

    
# Display assistant response in chat message container
with st.chat_message("assistant", avatar="https://cdn4.iconfinder.com/data/icons/heroes-villains-vol-2-colored/100/Terminator-512.png"):
    # response = st.write_stream(response_generator())
    response = st.write("Hello admin! I am Az-147. How can I assist you today?")
    st.caption("If you use predefined data, the file upload step will be hidden.")
    toggle = st.toggle('Use Predefined Data', True)
    data= get_data("data/sample_data.csv")
    
    if toggle == False:
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
    
    submit = st.button("Execute", type='primary')
    
model = load_model("model/model.pkl")

if submit:
    prediction = model.predict(data)