
"""
This module is used for the model configuration

It utilises Pydanamic's BaseSettings for configuration management.
allowing settings to be read from environmental varaiables and a .env file.
"""

from pydantic import DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseSettings):
    """
    Configuration settings for the class

    Attributes:
    model_config: SettingsConfigDict: model congig settings loaded  .env file
    model_path: DirectoryPath: path to the model
    model_name: str: name of the model
    """

    model_config = SettingsConfigDict(
        env_file='config/.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    model_path: DirectoryPath
    model_name: str


# load settings
model_settings = ModelSettings()
