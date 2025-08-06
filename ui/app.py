import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sys

# Load trained model
model = joblib.load('../models/final_model.pkl')
preprocessing_inputs = joblib.load('../models/scaler.pkl')

# Create app
st.title('Heart Disease Prediction App')
st.write("This app predicts the likelihood of heart disease based on patient characteristics.")

# Input fields
st.sidebar.header('Patient Input Features')

def user_input_features():
    age = st.sidebar.slider('Age', 20, 100, 50)
    cp = st.sidebar.selectbox('Chest Pain Type', 
                            ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 60, 220, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', ['No', 'Yes'])
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise', 0.0, 6.2, 1.0)
    ca = st.sidebar.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 3, 1)
    thal = st.sidebar.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])
    
    # Convert inputs to model format
    age_scaled = preprocessing_inputs['age'].transform(np.array([[age]]))[0][0]
    thalach_scaled = preprocessing_inputs['thalach'].transform(np.array([[thalach]]))[0][0]
    oldpeak_scaled = preprocessing_inputs['oldpeak'].transform(np.array([[oldpeak]]))[0][0]
    data = {
        'age': age_scaled,
        'thalach': thalach_scaled,
        'exang': 1 if exang == 'Yes' else 0,
        'oldpeak': oldpeak_scaled,
        'ca': ca,
        'cp': {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}[cp],
        'thal': {'Normal': 0, 'Fixed Defect': 1, 'Reversible Defect': 2}[thal]
    }
    return pd.DataFrame(data, index=[0])

# Data visualization section
if st.sidebar.checkbox('Show Heart Disease Trends'):
    # Load sample dataset for visualization
    df = pd.read_csv('../data/heart_disease.csv')
    st.subheader('Heart Disease Trends')
    st.write('Distribution of Age by Heart Disease Status:')
    st.bar_chart(df.groupby('target')['age'].mean())
    st.write('Chest Pain Type Distribution:')
    st.bar_chart(df['cp'].value_counts())
    st.write('Correlation Heatmap:')
    st.write(df.corr())

input_df = user_input_features()

# Display input
st.subheader('Patient Input Features')
st.write(input_df)

# Get model expected features
model_features = model.feature_names_in_
input_df = input_df[model_features]

# Make prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
heart_disease = np.array(['No', 'Yes'])
st.write(heart_disease[prediction])

st.subheader('Prediction Probability')
st.write(f"No Heart Disease: {prediction_proba[0][0]:.2f}")
st.write(f"Heart Disease: {prediction_proba[0][1]:.2f}")