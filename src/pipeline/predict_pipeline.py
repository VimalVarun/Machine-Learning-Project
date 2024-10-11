import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
import logging

class PredictPipeline:
    def __init__(self):
        # Paths to model and preprocessor
        self.model_path = "artifacts/model.pkl"  # Update with the correct path to your model
        self.preprocessor_path = "artifacts/preprocessor.pkl"  # Update with the correct path to your preprocessor
    
    def load_model(self):
        """Loads the saved machine learning model."""
        try:
            model = joblib.load(self.model_path)
            return model
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise CustomException(f"Error loading model: {str(e)}", sys)
    
    def load_preprocessor(self):
        """Loads the saved preprocessor (StandardScaler or any other preprocessing pipeline)."""
        try:
            preprocessor = joblib.load(self.preprocessor_path)
            return preprocessor
        except Exception as e:
            logging.error(f"Error loading preprocessor: {str(e)}")
            raise CustomException(f"Error loading preprocessor: {str(e)}", sys)
    
    def predict(self, features: pd.DataFrame):
        """
        Predict the target variable (math score) based on input features.

        Args:
            features (pd.DataFrame): Input features for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        try:
            model = self.load_model()  # Load the model
            preprocessor = self.load_preprocessor()  # Load the preprocessor

            # Transform features using the preprocessor
            data_scaled = preprocessor.transform(features)

            # Predict using the model
            predictions = model.predict(data_scaled)

            # Clip predictions to fall within 0 to 100 range
            predictions = np.clip(predictions, 0, 100)

            return predictions
        
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise CustomException(f"Prediction failed: {str(e)}", sys)