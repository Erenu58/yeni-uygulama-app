#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speech Modülü - Ses Tanıma ve Konuşma
"""

import speech_recognition as sr
import pyttsx3
from config import DEFAULT_LANGUAGE

class SpeechModule:
    def __init__(self):
        """Ses modülünü başlat"""
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.9)
    
    def listen(self):
        """Mikrofondan ses al"""
        try:
            with sr.Microphone() as source:
                audio = self.recognizer.listen(source, timeout=10)
                text = self.recognizer.recognize_google(audio, language='tr-TR')
                return text
        except sr.UnknownValueError:
            return "❌ Üzgünüm, anlamadım. Lütfen tekrar et."
        except Exception as e:
            return f"❌ Hata: {str(e)}"
    
    def speak(self, text):
        """Metni sesle oku"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"❌ Konuşma hatası: {e}")
