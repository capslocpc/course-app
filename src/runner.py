"""
This module serves as the entry point for running the machine learning model
service.

It initializes the `ModelServices` class, loads a trained model, and makes a
sample prediction using predefined input data.

Modules Used:
- `ModelServices`: Handles model loading and prediction.
- `settings`: Provides configuration details for the model.
- `loguru.logger`: Facilitates logging for debugging and tracking execution.

Functions:
- `main()`:
    - Starts the application.
    - Loads the trained model.
    - Makes a sample prediction with predefined input.
    - Logs relevant information about the process.

Usage:
Run this script as the main entry point to execute the model service:
"""

from loguru import logger

from model.model_service import ModelServices


@logger.catch
def main() -> None:
    """
    Executes the main workflow of the application.

    This function:
    - Initializes the `ModelServices` class.
    - Loads the trained machine learning model.
    - Makes a sample prediction using predefined input data.
    - Logs relevant information throughout the process.

    Logs:
        - Application start.
        - Model loading.
        - Prediction result.
        - Application completion.

    Returns:
        None

    Example:
        To run the application, execute:
        ```
        if __name__ == '__main__':
            main()
        ```
    """

    logger.info('Starting the application')
    ml_svc = ModelServices()
    ml_svc.load_model()
    feature_values = {
        'area': 85,
        'constraction_year': 2015,
        'bedrooms': 2,
        'garden': 20,
        'balcony_yes': 1,
        'parking_yes': 1,
        'furnished_yes': 0,
        'garage_yes': 0,
        'storage_yes': 1
    }
    pred = ml_svc.predict(list(feature_values.values()))  # passed as a list
    logger.info(f'Prediction: {pred}')
    logger.info('Application finished')


if __name__ == '__main__':
    main()
