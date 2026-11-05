"""
Minimal Django settings. This app has no database, users or admin panel,
so only the pieces it uses are enabled.
"""
import os
from pathlib import Path

from django.core.management.utils import get_random_secret_key
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")  # reads LLM_API_KEY etc. into environment variables

# Local development: DEBUG on, no config needed.
# Production (Dockerfile): DJANGO_DEBUG=false and DJANGO_ALLOWED_HOSTS are set there.
DEBUG = os.getenv("DJANGO_DEBUG", "true").lower() == "true"

# This app has no logins or sessions, so a random key per start is fine when none is set.
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY") or get_random_secret_key()

# Comma-separated domains this app may be served on. ".hf.space" = any Hugging Face Space.
ALLOWED_HOSTS = os.getenv("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

INSTALLED_APPS = ["rag"]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,  # finds templates in rag/templates/
        "OPTIONS": {},
    }
]

DATABASES = {}  # no database needed: data lives in the FAISS index
