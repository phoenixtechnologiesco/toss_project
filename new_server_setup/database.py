# database.py
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Update with your actual PostgreSQL credentials AND PORT
DATABASE_URL = "postgresql://tos_user:phoenix123@localhost:5433/tos_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100))
    service_no = Column(String(50))
    service_year = Column(String(4))
    timestamp = Column(DateTime, default=datetime.utcnow)
    hits = Column(Integer)
    group_size = Column(Float)
    holes_data = Column(JSON)
    image_path = Column(String(255))

# Create tables
Base.metadata.create_all(bind=engine)