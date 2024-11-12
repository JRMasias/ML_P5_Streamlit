import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App For Project 5')

st.info('This is a machine learning app for project 5')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/JRMasias/ML_P5_Streamlit/refs/heads/master/dataset.csv')
  df
