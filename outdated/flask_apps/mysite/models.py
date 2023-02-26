from __future__ import annotations
import collections
import dataclasses
import datetime
import enum
import time
from typing import Union
import PIL
import pytz
import math
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import Column, Index, ForeignKey
from sqlalchemy.sql.sqltypes import Boolean
from sqlalchemy.types import String, Integer, BigInteger, Float, DateTime, Enum
from sqlalchemy.orm import backref, relationship
from mysite.database import Base, db_session
from PIL import Image


@dataclasses.dataclass(init=False)
class User(Base):
    __tablename__ = "user"
    id: int = Column(Integer, primary_key=True)
    name: str = Column(String(50), unique=False, nullable=False)
    age: int = Column(Integer, unique=False)
    image: str = Column(String(256), unique=False)

    def __init__(self, name: str, age: int = None, image: str = None):
        self.name = name
        self.age = age
        self.image = image
