import streamlit as st
from backend.bot import *
from backend.model import *
import pandas as pd
import time
import google.generativeai as genai

# Page title
st.set_page_config(page_title='Batch Processing', layout='wide')

st.title("Attrition Prediction Engine")
st.write("Welcome to the Attrition Prediction Engine! This tool is designed to help you batch process employee data and predict attrition.")

with st.sidebar:
    with st.expander("🧪 Experimental Features", expanded=True):
        st.caption("API token can be obtained at https://aistudio.google.com/.")
        gemini_api = st.text_input("Gemini Token", "", type='password')
        try:
            genai.configure(api_key=gemini_api)
            ai_model = genai.GenerativeModel("gemini-1.5-flash")
            test = ai_model.generate_content("Explain how AI works")
            st.success("API key is valid. Experimental feature access granted.")
        except Exception as e:
            st.error("API key is invalid. You don't have access to experimental features.")

    
    with st.expander("⚠️ Disclaimer", expanded=False):
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

def count_attrition(predictions):
    return sum(predictions)

if submit:
    with st.status("Data Preview", expanded=True):
        time.sleep(.5)
        st.write(f"You've uploaded a data file of {data.shape[0]} rows and {data.shape[1]} columns. Here's a preview of the data:")
        st.write(data.head())
    
    with st.status("Predicting Attrition...", expanded=True):
        time.sleep(2)
        prediction = model.predict(data)
        data['Attrition'] = prediction
        attrition_count = count_attrition(prediction)
        output = f"Prediction completed! There are {attrition_count} cases of attrition. Here's a preview of the data with the predicted attrition status:"
        st.write(output)
        st.write(data.head())
    
    with st.status("AI Opinion", expanded=True):
        try:
            response = ai_model.generate_content(f"Give some opinions in about 100 word based on the prediction results where there are {attrition_count} cases of attrition.")
            st.write(response.text)
        except Exception as e:
            st.write("You don't have access to this feature. Please authenticate to use this feature.")

