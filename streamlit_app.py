import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

st.title('🤖 Machine Learning App For Project 5')

st.info('This is a machine learning app for project 5')

# Load and display data
with st.expander('Data'):
    st.write('**Raw Data**')
    df = pd.read_csv('https://raw.githubusercontent.com/JRMasias/ML_P5_Streamlit/refs/heads/master/dataset.csv')
    st.write(df)

    st.write('**X**')
    X_raw = df.drop('Turnout', axis=1)
    st.write(X_raw)

    st.write('**Y**')
    y_raw = df.Turnout
    st.write(y_raw)

# Data Visualization
with st.expander('Data Visualization'):
    st.write('All Data')
    st.scatter_chart(df)
    
    st.write('VEP & Turnout')
    st.scatter_chart(data=df, x='Voting Eligible Population (VEP)', y='Registered Voters', color='Turnout')

# Sidebar for Input Features
with st.sidebar:
    st.header('Input features')
    
    # NUMERICAL FEATURES
    year = st.number_input('Year', value=datetime.now().year)
    vap = st.number_input('Voting Age Population (VAP)')
    vep = st.number_input('Voting Eligible Population (VEP)')
    registered = st.number_input('Registered Voters')
    vap_turnout = st.number_input('Turnout as VAP')
    vep_turnout = st.number_input('Turnout as VEP')
    men = st.number_input('Men Voters')
    women = st.number_input('Women Voters')
    age18_24 = st.number_input('Age 18-24')
    age25_44 = st.number_input('Age 25-44')
    age45_64 = st.number_input('Age 45-64')
    age65 = st.number_input('Age 65+')
    hs_ged = st.number_input('High School/GED')
    some_college = st.number_input('Some College/ASC')
    bas = st.number_input('BAS or more')
    white_reg = st.number_input('White Registered')
    white_vote = st.number_input('White Voted')
    black_reg = st.number_input('Black Registered')
    black_vote = st.number_input('Black Voted')
    asian_reg = st.number_input('Asian Registered')
    asian_vote = st.number_input('Asian Voted')
    hisp_reg = st.number_input('Hispanic Registered')
    hisp_vote = st.number_input('Hispanic % Voted')
    avg_income = st.number_input('Average Income')

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
        'House Majority_Republican': 1 if house_majority == "Republican" else 0,
        'Senate Majority_Republican': 1 if senate_majority == "Republican" else 0,
        'Current Presidential Party_Republican': 1 if current_party == "Republican" else 0,
    }

input_df = pd.DataFrame(data, index=[0])

# Concatenate the input row to the dataset (for feature compatibility)
input_features = pd.concat([input_df, X_raw], axis=0)

# Handle any missing values
input_features.fillna(0, inplace=True)  # Replace NaNs in input_features with zeros

# Model training and inference
# Separate the user input row after concatenation
X = input_features.iloc[1:]  # Use only original dataset for training
input_data = input_features.iloc[0:1]  # Input data for prediction

# Handle missing values in y_raw
y_raw.fillna(y_raw.mean(), inplace=True)  # Replace NaNs in y_raw with the column mean

# Define and train model
model = make_pipeline(StandardScaler(), Lasso(alpha=0.1, random_state=42))
model.fit(X, y_raw)

# Make predictions with the user input
prediction = model.predict(input_data)

# Display Prediction
st.title('Predicted Turnout')
st.success(f"{prediction[0]:.0f}")
