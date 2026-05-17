#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Odin AI - Advanced AI Assistant
Sesli, Yazılı, Web Araması, Resim Oluşturma, Görev Yönetimi
"""

import os
import sys
from dotenv import load_dotenv
from assistant import OdinAssistant

# .env dosyasını yükle
load_dotenv()

def main():
    print("\n" + "="*60)
    print("🤖 ODIN AI ASSISTANT - BAŞLATILIYOR...")
    print("="*60)
    
    # API Key kontrolü
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ HATA: OPENAI_API_KEY bulunamadı!")
        print("📝 Lütfen .env dosyasına API key ekle.")
        sys.exit(1)
    
    # Asistanı başlat
    try:
        odin = OdinAssistant()
        print("✅ Odin başarıyla yüklendi!\n")
        odin.run()
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
