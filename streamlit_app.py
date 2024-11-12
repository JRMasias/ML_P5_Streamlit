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

with st.sidebar:
  st.header('Input features')
  
  # NUMERICAL FEATURES
  # Year, Voting Age Population (VAP), Voting Eligible Population (VEP), Registered Voters, Turnout as VAP, Turnout as VEP, Men Voters, Women Voters,
  # Age 18-24, Age 25-44, Age 45-64, Age 65+, High School/GED, Some College/ASC, BAS or more, White Registered, White Voted, Black Registered, Black Voted,
  # Asian Registered, Asian Voted, Hispanic Registered, Hispanic % Voted, Average Income
  year = st.number_input('Year', value=)

  # CATEGORICAL FEATURES
  # House Majority_Republican, Senate Majority_Republican, Current Presidential Party_Republican
