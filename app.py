import streamlit as st
import pandas as pd
from backend.model import load_model

# Load the model
model = load_model('model/model.pkl')


# make a prediction

def predict(data):
    return model.predict(data)

