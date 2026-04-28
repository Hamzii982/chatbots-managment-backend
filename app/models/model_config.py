from sqlalchemy import Column, Integer, String, Boolean, Float
from app.db.base import Base

class ModelConfig(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)

    provider = Column(String, nullable=False)
    model_name = Column(String, nullable=False) 

    api_key = Column(String, nullable=False)  # encrypted ideally

    temperature = Column(Float, default=0.7)
    max_tokens = Column(Integer, default=1000)

    is_active = Column(Boolean, default=True)