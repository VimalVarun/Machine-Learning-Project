import streamlit as st
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline

# Instantiate the prediction pipeline
predict_pipeline = PredictPipeline()

# Streamlit App
st.title('Student Performance Prediction App')

# Sidebar for Inputs
st.sidebar.header("User Input Features")

# Input features for the app

# Gender selection
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])

# Race/Ethnicity selection
race_ethnicity = st.sidebar.selectbox('Race/Ethnicity', 
                                      ['group A', 'group B', 'group C', 'group D', 'group E'])

# Parental level of education
parental_education = st.sidebar.selectbox("Parental Level of Education", 
                                          ["bachelor's degree", 'some college', "master's degree", 
                                           "associate's degree", 'high school', 'some high school'])

# Lunch type
lunch = st.sidebar.selectbox('Lunch Type', ['standard', 'free/reduced'])

# Test preparation course
test_prep_course = st.sidebar.selectbox('Test Preparation Course', ['none', 'completed'])

# Numerical input for scores
reading_score = st.sidebar.slider('Reading Score', 0, 100, 50)
writing_score = st.sidebar.slider('Writing Score', 0, 100, 50)

# When the button is clicked, predict the student's math score
if st.button('Predict Students Math Score'):
    try:
        # Prepare input data as a DataFrame (without math_score, as it's the target)
        input_data = pd.DataFrame({
            'gender': [gender],
            'race_ethnicity': [race_ethnicity],
            'parental_level_of_education': [parental_education],
            'lunch': [lunch],
            'test_preparation_course': [test_prep_course],
            'reading_score': [reading_score],
            'writing_score': [writing_score]
        })

        # Use the PredictPipeline to make predictions (for math_score)
        prediction = predict_pipeline.predict(input_data)[0]

        predictions = np.clip(prediction, 0, 100)

        # Convert prediction to string to handle formatting
        prediction_str = f"{predictions:.2f}"  # Format to two decimal places

        # Extract first two digits from the integer part and keep two decimal places
        integer_part = int(prediction)  # Get the integer part
        formatted_prediction = f"{integer_part:02d}.{prediction_str.split('.')[-1]}"  # Combine two digits and decimal part

        # Display the formatted prediction
        st.success(f'The predicted math score is: {formatted_prediction}')

    except Exception as e:
        st.error(f"Error during prediction: {e}")