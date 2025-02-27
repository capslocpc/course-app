
"""
This module is used for the logger configuration

It utilises Pydanamic's BaseSettings for configuration management.
allowing settings to be read from environmental varaiables and a .env file.
"""

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggerSettings(BaseSettings):
    """
    Logger configuration settings for the class

    Attributes:
    model_config: SettingsConfigDict: model congig settings loaded .env file
    log_level: str: log level
    """

    model_config = SettingsConfigDict(
        env_file='config/.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    log_level: str


def configure_logging(log_level: str) -> None:
    """
    Configures the loguru logger

    Args:
    log_level: str: log level
    """
    logger.remove()
    logger.add(
        'logs/app.log',
        rotation='1 day',
        retention='2 days',
        level=log_level,
        compression='zip',
    )


# configure logging no need to load settings in a class
configure_logging(log_level=LoggerSettings().log_level)
