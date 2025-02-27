
"""
This module contains functions to prepare the data for the model.

This module has functions to load the data from the database,
encode categorical columns, and parse the garden column.
"""

import re
from loguru import logger
import pandas as pd

from model.pipeline.collection import load_data_from_db


def prepare_data() -> pd.DataFrame:
    """
    Prepare the data for the model

    This module loads the data, encodes the categorical columns,
    and parses the garden column.

    Returns:
        pd.DataFrame: The prepared data
    """
    logger.info('Starting up preprocessing pipeline')
    dataframe = load_data_from_db()
    data_encoded = _encode_cat_cols(dataframe)
    df = _parse_garden_col(data_encoded)
    return df


def _encode_cat_cols(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical columns in the given DataFrame using one-hot encoding.

    This function applies `pd.get_dummies` to specified categorical columns
    and converts the resulting boolean columns to integers.

    Args:
        dataframe (pd.DataFrame): The input DF containing categorical columns.

    Returns:
        pd.DataFrame: The transformed DataFrame with encoded
        categorical columns.
    """
    cols = ['balcony', 'storage', 'parking', 'furnished', 'garage']
    logger.info(f'encoding catagory columns: {cols}')

    data_encoded = pd.get_dummies(
        dataframe,
        columns=cols,
        drop_first=True,
        )

    # convert only boolean columns to integers
    for col in (
        'balcony_yes',
        'storage_yes',
        'parking_yes',
        'furnished_yes',
        'garage_yes',
    ):
        if col in data_encoded.columns:  # Check if the column exists
            data_encoded[col] = data_encoded[col].astype(int)

    return data_encoded


def _parse_garden_col(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Parses the garden column in the given DataFrame.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing
        the 'garden' column.

    Returns:
        pd.DataFrame: The DataFrame with the 'garden' column parsed to an
        integer
    """
    logger.info('parsing garden column')
    dataframe['garden'] = dataframe['garden'].apply(
        lambda x: (
            0 if x == 'Not present'
            else int(re.findall(r'\d+', str(x))[0])
        )
    )

    return dataframe  # ✅ Moved outside the loop
