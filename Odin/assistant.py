#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Odin Assistant - Ana Asistan Sınıfı
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv

# Modülleri import et
from modules.chatbot import ChatBot
from modules.speech import SpeechModule
from modules.web_search import WebSearcher
from modules.file_manager import FileManager
from modules.tasks import TaskManager
from modules.image_gen import ImageGenerator
from modules.analytics import Analytics

load_dotenv()

class OdinAssistant:
    def __init__(self):
        """Odin Asistanını başlat"""
        print("📚 Modüller yükleniyor...")
        
        self.chatbot = ChatBot()
        self.speech = SpeechModule()
        self.web_search = WebSearcher()
        self.file_manager = FileManager()
        self.task_manager = TaskManager()
        self.image_gen = ImageGenerator()
        self.analytics = Analytics()
        
        self.running = True
        self.history = []
        
        print("✅ Tüm modüller hazır!")
    
    def process_command(self, user_input):
        """Kullanıcı komutunu işle"""
        
        # Geçmişe ekle
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "input": user_input,
            "type": "command"
        })
        
        # Komut analiz et
        command = user_input.lower().strip()
        
        # Sesli komut
        if command == "dinle":
            print("🎤 Konuşun...")
            text = self.speech.listen()
            if text:
                return self.process_command(text)
        
        # Resim oluştur
        elif command.startswith("resim yap:") or command.startswith("resim:"):
            prompt = command.replace("resim yap:", "").replace("resim:", "").strip()
            print(f"🎨 '{prompt}' için resim oluşturuluyor...")
            result = self.image_gen.generate(prompt)
            print(result)
        
        # Web araması
        elif command.startswith("ara:") or command.startswith("araştır:"):
            query = command.replace("ara:", "").replace("araştır:", "").strip()
            print(f"🔍 '{query}' aranıyor...")
            results = self.web_search.search(query)
            print(results)
        
        # Dosya oluştur
        elif command.startswith("dosya oluştur:"):
            filename = command.replace("dosya oluştur:", "").strip()
            print(f"📄 '{filename}' oluşturuluyor...")
            self.file_manager.create_file(filename)
        
        # Görev ekle
        elif command.startswith("görev ekle:"):
            task = command.replace("görev ekle:", "").strip()
            print(f"📋 Görev ekleniyor: {task}")
            self.task_manager.add_task(task)
        
        # Görevleri listele
        elif command == "görevler" or command == "task list":
            print("📋 GÖREVLER:")
            self.task_manager.list_tasks()
        
        # Rapor
        elif command == "rapor" or command == "report":
            print("📊 RAPOR OLUŞTURULUYOR...")
            report = self.analytics.generate_report(self.history)
            print(report)
        
        # Çıkış
        elif command == "çık" or command == "quit" or command == "exit":
            print("\n👋 Odin kapatılıyor... Hoşça kalın!")
            self.running = False
        
        # Normal sohbet
        else:
            print("🤖 Odin düşünüyor...")
            response = self.chatbot.chat(user_input)
            print(f"\n💬 Odin: {response}\n")
    
    def run(self):
        """Asistanı çalıştır"""
        print("\n" + "="*60)
        print("💡 TİPLER:")
        print("  • Sohbet: Normal mesaj yaz")
        print("  • Sesli: 'dinle' yaz")
        print("  • Resim: 'resim yap: [açıklama]'")
        print("  • Ara: 'ara: [sorgu]'")
        print("  • Dosya: 'dosya oluştur: [ad]'")
        print("  • Görev: 'görev ekle: [görev]'")
        print("  • Rapor: 'rapor'")
        print("  • Çık: 'çık'")
        print("="*60 + "\n")
        
        while self.running:
            try:
                user_input = input("📝 Sen: ").strip()
                if user_input:
                    self.process_command(user_input)
            except KeyboardInterrupt:
                print("\n\n👋 Kapatılıyor...")
                self.running = False
            except Exception as e:
                print(f"❌ Hata: {e}")
