import os
from pathlib import Path

class Settings:
    # Directories
    BASE_DIR = Path(__file__).resolve().parent
    UPLOAD_DIR = BASE_DIR / "uploads"
    DATABASE_DIR = BASE_DIR / "database"
    
    # Database
    DATABASE_URL = f"sqlite:///{DATABASE_DIR}/requirements.db"
    
    # File Upload
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {".xlsx", ".xls", ".csv", ".txt"}
    
    # AI/ML Settings
    SIMILARITY_THRESHOLD = 0.6
    MAX_IMPACT_DEPTH = 3

settings = Settings()
