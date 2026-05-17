#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Odin Configuration - Tüm Ayarlar
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ========== API KEYS ==========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# ========== MODELS ==========
CHAT_MODEL = "gpt-4"
IMAGE_MODEL = "dall-e-3"
EMBEDDING_MODEL = "text-embedding-3-small"

# ========== AYARLAR ==========
MAX_TOKENS = 2048
TEMPERATURE = 0.7
TOP_P = 0.9

# ========== DİL ==========
DEFAULT_LANGUAGE = "tr"  # Türkçe
LANGUAGES = ["tr", "en"]

# ========== DOSYA YOLLARI ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODULES_DIR = os.path.join(BASE_DIR, "modules")

# Klasörleri oluştur
for directory in [DATA_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ========== VERİTABANI ==========
DB_PATH = os.path.join(DATA_DIR, "odin.db")
TASKS_FILE = os.path.join(DATA_DIR, "tasks.json")
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")

# ========== SES AYARLARI ==========
MIC_INDEX = None  # Otomatik
AUDIO_RATE = 16000
AUDIO_CHUNK = 1024

# ========== WEB ARAŞTIRMA ==========
SEARCH_RESULTS_LIMIT = 10
TIMEOUT = 10

# ========== RAPOR ==========
REPORT_FORMAT = "json"  # json, txt, html
REPORT_PATH = os.path.join(DATA_DIR, "reports")

os.makedirs(REPORT_PATH, exist_ok=True)
