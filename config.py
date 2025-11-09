"""
Configuration module for VSA application.
Loads settings from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# Model Configuration
MODEL_PATH = os.getenv('MODEL_PATH', str(BASE_DIR / 'models' / 'vsa.sav'))

# Data Configuration
DATA_CACHE_DIR = os.getenv('DATA_CACHE_DIR', str(BASE_DIR / 'data' / 'daily_data'))
STOCKS_CSV_PATH = os.getenv('STOCKS_CSV_PATH', str(BASE_DIR / 'data' / 'stocks.csv'))

# Gmail API Configuration
GMAIL_CLIENT_SECRET_PATH = os.getenv('GMAIL_CLIENT_SECRET_PATH', str(BASE_DIR / 'credentials' / 'client_secret.json'))
GMAIL_TOKEN_PATH = os.getenv('GMAIL_TOKEN_PATH', str(BASE_DIR / 'credentials' / 'token.pickle'))
EMAIL_RECIPIENT = os.getenv('EMAIL_RECIPIENT', 'repque@yahoo.com')

# Database Configuration
POSITIONS_DB_PATH = os.getenv('POSITIONS_DB_PATH', str(BASE_DIR / 'db.json'))

# Application Settings
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Ensure required directories exist
Path(DATA_CACHE_DIR).mkdir(parents=True, exist_ok=True)
Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(GMAIL_CLIENT_SECRET_PATH).parent.mkdir(parents=True, exist_ok=True)
