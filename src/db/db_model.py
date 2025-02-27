"""
This module contains the SQLAlchemy model for the rent_apartments table.

The module uses SQLAlchemy's ORM to map python classes to the database tables.
"""

from sqlalchemy import INTEGER, REAL, VARCHAR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from config import db_settings


# Define the base class for declarative models
class Base(DeclarativeBase):
    """Base class for declarative models"""

    pass  # noqa: WPS420, WPS604


# Create a class for the table that inherits from Base
class RentApartments(Base):
    """
    Class for the rent_apartments tableß

    Attributes:
        address (str): The primary key representing the apartment address.
        area (float): The total area of the apartment in square meters.
        construction_year (int): The year the apartment was built.
        rooms (int): The total number of rooms in the apartment.
        bedrooms (int): The number of bedrooms in the apartment.
        bathrooms (int): The number of bathrooms in the apartment.
        balcony (str): Indicates whether the apartment has a balcony.
        storage (str): Indicates whether the apartment has a storage room.
        parking (str): Specifies the type of parking available.
        furnished (str): Indicates whether the apartment is furnished.
        garage (str): Indicates whether the apartment includes a garage.
        garden (str): Indicates whether the apartment has a garden.
        energy (str): The apartment's energy rating.
        facilities (str): Additional facilities available in the apartment.
        zip (str): The postal code of the apartment's location.
        neighborhood (str): The neighborhood where the apartment is located.
        rent (int): The monthly rent price of the apartment.
    """
    __tablename__ = db_settings.rent_apartment_table_name

    address: Mapped[str] = mapped_column(VARCHAR(), primary_key=True)
    area: Mapped[float] = mapped_column(REAL())
    constraction_year: Mapped[int] = mapped_column(INTEGER())
    rooms: Mapped[int] = mapped_column(INTEGER())
    bedrooms: Mapped[int] = mapped_column(INTEGER())
    bathrooms: Mapped[int] = mapped_column(INTEGER())
    balcony: Mapped[str] = mapped_column(VARCHAR())
    storage: Mapped[str] = mapped_column(VARCHAR())
    parking: Mapped[str] = mapped_column(VARCHAR())
    furnished: Mapped[str] = mapped_column(VARCHAR())
    garage: Mapped[str] = mapped_column(VARCHAR())
    garden: Mapped[str] = mapped_column(VARCHAR())
    energy: Mapped[str] = mapped_column(VARCHAR())
    facilities: Mapped[str] = mapped_column(VARCHAR())
    zip: Mapped[str] = mapped_column(VARCHAR())
    neighborhood: Mapped[str] = mapped_column(VARCHAR())
    rent: Mapped[int] = mapped_column(INTEGER())
