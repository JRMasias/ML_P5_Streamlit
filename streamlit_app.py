import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App For Project 5')

st.info('This is a machine learning app for project 5')

df = pd.read_csv('https://github.com/JRMasias/ML_P5_Streamlit/blob/master/dataset')
df
