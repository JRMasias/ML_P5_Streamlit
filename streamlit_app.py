import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

st.title('🤖 Machine Learning App For Project 5')

st.info('This is a machine learning app for project 5')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/JRMasias/ML_P5_Streamlit/refs/heads/master/dataset.csv')
  df

  st.write('**X**')
  X_raw = df.drop('Turnout', axis=1)
  X_raw

  st.write('**Y**')
  y_raw = df.Turnout
  y_raw

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='Voting Eligible Population (VEP)', y='Registered Voters', color='Turnout')

# Input Features
with st.sidebar:
  st.header('Input features')
  
  # NUMERICAL FEATURES
  year = st.number_input('Year', value=None, placeholder=datetime.now().year)
  vap = st.number_input('Voting Age Population (VAP)', value=None, placeholder="Enter a value")
  vep = st.number_input('Voting Eligible Population (VEP)', value=None, placeholder="Enter a value")
  registered = st.number_input('Registered Voters', value=None, placeholder="Enter a value")
  vap_turnout = st.number_input('Turnout as VAP', value=None, placeholder="Enter a value")
  vep_turnout = st.number_input('Turnout as VEP', value=None, placeholder="Enter a value")
  men = st.number_input('Men Voters', value=None, placeholder="Enter a value")
  women = st.number_input('Women Voters', value=None, placeholder="Enter a value")
  age18_24 = st.number_input('Age 18-24', value=None, placeholder="Enter a value")
  age25_44 = st.number_input('Age 25-44', value=None, placeholder="Enter a value")
  age45_64 = st.number_input('Age 45-64', value=None, placeholder="Enter a value")
  age65 = st.number_input('Age 65+', value=None, placeholder="Enter a value")
  hs_ged = st.number_input('High School/GED', value=None, placeholder="Enter a value")
  some_college = st.number_input('Some College/ASC', value=None, placeholder="Enter a value")
  bas = st.number_input('BAS or more', value=None, placeholder="Enter a value")
  white_reg = st.number_input('White Registered', value=None, placeholder="Enter a value")
  white_vote = st.number_input('White Voted', value=None, placeholder="Enter a value")
  black_reg = st.number_input('Black Registered', value=None, placeholder="Enter a value")
  black_vote = st.number_input('Black Voted', value=None, placeholder="Enter a value")
  asian_reg = st.number_input('Asian Registered', value=None, placeholder="Enter a value")
  asian_vote = st.number_input('Asian Voted', value=None, placeholder="Enter a value")
  hisp_reg = st.number_input('Hispanic Registered', value=None, placeholder="Enter a value")
  hisp_vote = st.number_input('Hispanic % Voted', value=None, placeholder="Enter a value")
  avg_income = st.number_input('Average Income', value=None, placeholder="Enter a value")

  # CATEGORICAL FEATURES
  house_majority = st.selectbox('House Majority', ("Democrat", "Republican"))
  senate_majority = st.selectbox('Senate Majority', ("Democrat", "Republican"))
  current_party = st.selectbox('Current Presidential Party', ("Democrat", "Republican"))

  # Create a dataframe for input features
  data = {
    'Year': year,
    'Voting Age Population (VAP)': vap,
    'Voting Eligible Population (VEP)': vep,
    'Registered Voters': registered,
    'Turnout as VAP': vap_turnout,
    'Turnout as VEP': vep_turnout,
    'Men Voters': men,
    'Women Voters': women,
    'Age 18-24': age18_24,
    'Age 25-44': age25_44,
    'Age 44-64': age45_64,
    'Age 65+': age65,
    'High School/GED': hs_ged,
    'Some College/ASC': some_college,
    'BAS or more': bas,
    'White Registered': white_reg,
    'White Voted': white_vote,
    'Black Registered': black_reg,
    'Black Voted': black_vote,
    'Asian Registered': asian_reg,
    'Asian Voted': asian_vote,
    'Hispanic Registered': hisp_reg,
    'Hispanic % Voted': hisp_vote,
    'Average Income': avg_income,
    'House Majority_Republican': True if house_majority == "Republican" else False,
    'Senate Majority_Republican': True if senate_majority == "Republican" else False,
    'Current Presidential Party_Republican': True if current_party == "Republican" else False,
  }

input_df = pd.DataFrame(data, index=[0])
input_features = pd.concat([input_df, X_raw], axis=0)

#Model training and inference

# Train the model
X = input_features.iloc[1:]
input_data = input_features.iloc[0:1]

model = make_pipeline(StandardScaler(), Lasso(alpha=0.1, random_state=42))
model.fit(X, y_raw)

# Make predictions
prediction = model.predict(input_data)
st.write(f"Predicted Turnout: {prediction[0]:.0f}")
