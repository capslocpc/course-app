"""
This module procices functionality for managing a ML model

It contains the ModelService class, which handles loading and using
a pre-trained model. The class offers methods to load a model from a file,
building it if it doesnt exist and make predictions with the model.
"""

import pickle as pk
from pathlib import Path

import pandas as pd
from loguru import logger

from config import model_settings
from model.pipeline.model import build_model


class ModelServices:
    """
    A service class for managing machine learning model loading and prediction.

    This class is responsible for:
    - Loading a pre-trained model from disk.
    - Checking if the model exists; if not, triggering model training.
    - Performing predictions using the loaded model.

    Attributes:
        model (object): The machine learning model loaded from disk.

    Methods:
        __init__ : Initializes the ModelServices class.
        load_model(): Loads the pre-trained model from the specified path.
            If the model does not exist, it triggers training before loading.
        predict(input_parameters: list) -> float:
            Predicts an output based on the provided input parameters.
    """

    def __init__(self) -> None:
        """
        Initializes the ModelServices class with an uninitialized model.
        """
        self.model = None

    def load_model(self) -> None:
        """Loads the machine learning model from the configured path."""
        logger.info(f'Check model exists{model_settings.model_name}')

        model_path = Path(
            f'{model_settings.model_path}/{model_settings.model_name}',
            )

        if not model_path.exists():
            logger.warning(
                f'Model {model_settings.model_name} does not exist.'
                f'Training a new model --> {model_settings.model_name}',
                )
            build_model()

        logger.info(
            f'Model exists loading model {model_settings.model_name}',
            )

        with open(model_path, 'rb') as model_file:
            self.model = pk.load(model_file)

    def predict(self, input_parameters: list) -> list:
        """
        Predicts the output based on input features.

        Args:
            input_parameters (list): list of numerical values representing
                                     the input features for the model.

        Returns:
            float: The predicted value from the model.

        Raises:
            ValueError: If model has not been loaded before calling predict.
        """
        logger.info(f'Predicting with model {model_settings.model_name}')

        feature_names = [
            'area',
            'constraction_year',
            'bedrooms',
            'garden',
            'balcony_yes',
            'parking_yes',
            'furnished_yes',
            'garage_yes',
            'storage_yes',
            ]  # match training features

        # convert to DataFrame
        input_df = pd.DataFrame([input_parameters], columns=feature_names)

        return self.model.predict(input_df)
