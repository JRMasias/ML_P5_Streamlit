import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App For Project 5')

st.info('This is a machine learning app for project 5')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/JRMasias/ML_P5_Streamlit/refs/heads/master/dataset.csv')
  df

  st.write('**X**')
  X = df.drop('Turnout', axis=1)
  X

  st.write('**Y**')
  y = df.Turnout
  y

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='Voting Eligible Population (VEP)', y='Registered Voters', color='Turnout')
