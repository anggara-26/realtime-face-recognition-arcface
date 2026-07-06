"""Django settings for the face-recognition prototype.

Minimal single-app config for a local demo (BAB III.3.8 Development /
BAB IV.4.5 Demo Sistem): SQLite storage, media-served gallery photos, no
auth — this is a research prototype, not a production deployment.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# DATA_DIR holds db.sqlite3 + media/ so they can live on a mounted persistent
# volume in production (Railway containers otherwise wipe the filesystem on
# every redeploy). Defaults to the project dir for local dev.
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR))

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-secret-key-not-for-production")

DEBUG = os.environ.get("DJANGO_DEBUG", "true").lower() == "true"

ALLOWED_HOSTS = [h for h in os.environ.get("DJANGO_ALLOWED_HOSTS", "*").split(",") if h]
_railway_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN")
_render_domain = os.environ.get("RENDER_EXTERNAL_HOSTNAME")
for _host in (_railway_domain, _render_domain):
    if _host:
        ALLOWED_HOSTS.append(_host)

# Trust ngrok tunnel origins for CSRF (subdomain changes each session, so
# wildcard the whole ngrok-free.app domain rather than pinning one URL), plus
# Railway's *.up.railway.app / Render's *.onrender.com domains and anything
# set via env for a custom domain.
CSRF_TRUSTED_ORIGINS = [
    "https://*.ngrok-free.app",
    "https://*.up.railway.app",
    "https://*.onrender.com",
]
for _host in (_railway_domain, _render_domain):
    if _host:
        CSRF_TRUSTED_ORIGINS.append(f"https://{_host}")
_extra_origins = os.environ.get("DJANGO_CSRF_TRUSTED_ORIGINS", "")
CSRF_TRUSTED_ORIGINS += [o for o in _extra_origins.split(",") if o]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "faceid",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": DATA_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Jakarta"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage"},
}

MEDIA_URL = "media/"
MEDIA_ROOT = DATA_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- Face recognition prototype settings -----------------------------------
# Default backend: "arcface" (optimized, 512-D) or "dlib" (baseline, 128-D).
FACE_DEFAULT_ENGINE = "arcface"
# Cosine-similarity threshold for the "dikenal / tak dikenal" decision, per
# backend (the two embedding spaces have very different genuine/impostor
# score ranges -- see BAB IV.4.1's dlib EER threshold of 0.84 vs ArcFace's
# typically much lower operating point).
FACE_MATCH_THRESHOLD = {
    "arcface": 0.38,
    "dlib": 0.84,
}
