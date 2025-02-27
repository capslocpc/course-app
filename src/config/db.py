
"""
This module is used for the database configuration

It utilises Pydanamic's BaseSettings for configuration management.
allowing settings to be read from environmental varaiables and a .env file.
"""


from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine


class DbSettings(BaseSettings):
    """
    Database settings for the class

    Attributes:
    model_config: SettingsConfigDict: model congig settings loaded .env file
    db_conn_str: str: database connection string
    rent_apartment_table_name: str: name of the table in the database
    """

    model_config = SettingsConfigDict(
        env_file='config/.env',
        env_file_encoding='utf-8',
        extra='ignore'
        )

    db_conn_str: str
    rent_apartment_table_name: str


# load settings
db_settings = DbSettings()

# create sqlalchemy database engine
engine = create_engine(db_settings.db_conn_str)
