"""
This module is responsible for loading data from the database

Includes a function to extract data from RentAppartments
table in the database and load it into a pandas dataframe.
"""

import pandas as pd
from loguru import logger
from sqlalchemy import select

from config import engine
from db.db_model import RentApartments


def load_data_from_db():
    """
    Load data from the RentApartments table in the database

    Returns:
        pd.DataFrame: dataframe containing the RentApartments table
    """
    logger.info('loading data from database')
    query = select(RentApartments)
    return pd.read_sql(
        query,
        engine,
    )
