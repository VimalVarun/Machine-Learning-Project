from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Try to fetch and convert the user input
            reading_score = request.form.get('reading_score')
            writing_score = request.form.get('writing_score')

            # Check if scores are valid numbers
            if not reading_score or not writing_score:
                raise ValueError("Reading and Writing scores cannot be empty")
            
            reading_score = float(reading_score)
            writing_score = float(writing_score)

            # Create an instance of CustomData with the valid inputs
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=reading_score,
                writing_score=writing_score
            )

            # Convert data into a DataFrame
            pred_df = data.get_data_as_data_frame()
            print("Data for prediction:", pred_df)

            # Run the prediction pipeline
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            print("Prediction result:", results)

            return render_template('home.html', results=results[0])

        except ValueError as ve:
            # Handle invalid input like non-numeric scores
            return render_template('home.html', error=f"Invalid input: {ve}")

        except FileNotFoundError as fnf_error:
            # Handle missing files like model.pkl or preprocessor.pkl
            return render_template('home.html', error="Model files are missing. Please ensure the model and preprocessor files are available.")

        except Exception as e:
            # Catch all other exceptions and provide a generic error message
            return render_template('home.html', error=f"An error occurred: {str(e)}")
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)   