from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base
import os

# Database file path
DB_DIR = os.path.join(os.path.dirname(__file__))
DB_FILE = os.path.join(DB_DIR, "requirements.db")
DATABASE_URL = f"sqlite:///{DB_FILE}"

# Create engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize database - create all tables"""
    Base.metadata.create_all(bind=engine)
    print(f"✅ Database initialized at: {DB_FILE}")

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()