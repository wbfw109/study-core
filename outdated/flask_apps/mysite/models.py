from __future__ import annotations
import dataclasses
from sqlalchemy.schema import Column
from sqlalchemy.types import String, Integer
from mysite.database import Base


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
