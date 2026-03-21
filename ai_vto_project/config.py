import os
from urllib.parse import quote_plus

# -------------------------------------------------------------------
# DATABASE (MySQL)
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# DATABASE (SQLite for deployment)
# -------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

SQLALCHEMY_DATABASE_URI = os.environ.get(
    "DATABASE_URL",
    "sqlite:///" + os.path.join(BASE_DIR, "site.db")
)

SQLALCHEMY_TRACK_MODIFICATIONS = False


# -------------------------------------------------------------------
# FLASK
# -------------------------------------------------------------------
SECRET_KEY = os.environ.get('SECRET_KEY', 'parshva-jewellers-secret-key-change-in-production')

# -------------------------------------------------------------------
# TWILIO (SMS OTP) — optional, OTP prints to console without it
# -------------------------------------------------------------------
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID', 'YOUR_TWILIO_SID_HERE')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN', 'YOUR_TWILIO_TOKEN_HERE')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER', '+1XXXXXXXXXX')

# -------------------------------------------------------------------
# OTP SETTINGS
# -------------------------------------------------------------------
OTP_EXPIRY_SECONDS = 300  # 5 minutes
OTP_LENGTH = 6