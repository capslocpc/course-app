"""
This module handles the training pipeline for a rental price prediction model.

It includes functions for:
- Data preparation and preprocessing
- Feature selection and dataset splitting
- Training a RandomForestRegressor model with hyperparameter tuning
- Evaluating the trained model
- Saving the final model for deployment
"""

import pickle as pk

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from config import model_settings
from model.pipeline.preparation import prepare_data


def build_model() -> None:
    """
    Build the model by training a RandomForestRegressor model with
    hyperparameter tuning and save the final model in a configuration file.

    This function manages the model build pipeline by calling the necessary
    functions in the correct order.
    """

    logger.info('Building the model')

    # to train and save the model
    # 1. load the preprossessed dataset
    df = prepare_data()

    # 2. identify X and y
    feature_names = [
        'area',
        'constraction_year',
        'bedrooms',
        'garden',
        'balcony_yes',
        'parking_yes',
        'furnished_yes',
        'garage_yes',
        'storage_yes',]
    X, y = _get_X_y(
        df,
        col_X=feature_names,
        )

    # 3. split the dataset into train and test
    X_train, X_test, y_train, y_test = _split_train_test(X, y)
    # 4. train the model
    rf = _train_model(X_train, y_train)
    # 5. evaluate the model
    score = _evaluate_model(rf, X_test, y_test)  # noqa: F841
    # 6. save the model in a configuration file
    save_model(rf)


def _get_X_y(
        dataframe: pd.DataFrame,
        col_X: list[str],
        col_y: str = 'rent',
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extracts feature variables (X) and variable (y) from the dataset.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing the dataset.
        col_X (list[str]): A list of column names to be used as features (X).
        col_y (str, optional): The column name of the target variable (y).

    Returns:
        tuple[pd.DataFrame, pd.Series]:
            - A DataFrame containing the selected feature columns (X).
            - A Series containing the target variable (y).
    """

    logger.info(f'defining X and y variables. X vars: {col_X}: y var: {col_y}')

    return dataframe[col_X], dataframe[col_y]


def _split_train_test(
        X: pd.DataFrame,
        y: pd.Series
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into training and testing sets.

    Args:
        X (pd.DataFrame): The feature variables.
        y (pd.Series): The target variable.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            - X_train (pd.DataFrame): Training set features.
            - X_test (pd.DataFrame): Testing set features.
            - y_train (pd.Series): Training set target variable.
            - y_test (pd.Series): Testing set target variable.
    """
    logger.info('splitting the dataset into train and test')

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2  # noqa: WPS432
    )
    return X_train, X_test, y_train, y_test


def _train_model(X_train, y_train) -> RandomForestRegressor:
    """
    Trains a RandomForestRegressor model with hyperparameter tuning using
    GridSearchCV.

    Args:
        X_train (pd.DataFrame): The training dataset containing feature
        variables.
        y_train (pd.Series): The target variable for training.

    Returns:
        RandomForestRegressor: The best model selected based on GridSearchCV.
    """
    logger.info('training the model')

    grid_space = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9, 12, 15],
    }

    logger.debug(f'grid space: {grid_space}')

    grid = GridSearchCV(
        RandomForestRegressor(),
        param_grid=grid_space,
        cv=2,
        scoring='r2',
     )

    model_grid = grid.fit(
        X_train,
        y_train
    )

    return model_grid.best_estimator_


def _evaluate_model(model, X_test, y_test) -> float:
    """
    Evaluates the trained model on the test dataset using the R² score.

    Args:
        model (RandomForestRegressor): The trained model to be evaluated.
        X_test (pd.DataFrame): The test dataset containing feature variables.
        y_test (pd.Series): The actual target values for evaluation.

    Returns:
        float: The R² score of the model on the test dataset.
    """
    logger.info(f'evaluating the model: score ={model.score(X_test, y_test)}')

    return model.score(X_test, y_test)


def save_model(model):
    """
    Saves the trained model to a specified file location using pickle.

    Args:
        model (RandomForestRegressor): The trained model to be saved.

    Returns:
        None

    Logs:
        - Logs the file path where the model is saved.

    Notes:
        - The model is saved as a binary file using `pickle.dump()`.
        - The file path is determined by `settings.model_path` and
          `settings.model_name`.

    Example:
        save_model(trained_model)
    """
    model_path = f'{model_settings.model_path}/{model_settings.model_name}'
    logger.info(f'saving model in {model_path}')
    with open(model_path, 'wb') as model_file:
        pk.dump(model, model_file)
