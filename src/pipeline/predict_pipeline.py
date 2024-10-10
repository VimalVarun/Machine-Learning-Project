import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import logging

class PredictPipeline:
    def __init__(self):
        self.model_path = "artifacts/model.pkl"
        self.preprocessor_path = "artifacts/preprocessor.pkl"

    def predict(self, features):
        """
        Predict the target variable based on input features.

        Args:
            features (pd.DataFrame): Input features for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        try:
            # Validate input features
            if not isinstance(features, pd.DataFrame):
                raise ValueError("Features should be a pandas DataFrame.")
            logging.info("Input features validated.")

            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            # Transform features
            data_scaled = preprocessor.transform(features)
            logging.info("Features transformed using preprocessor.")

            # Make predictions
            preds = model.predict(data_scaled)
            logging.info("Predictions made successfully.")

            return preds

        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise CustomException(f"Prediction failed: {str(e)}", sys)
