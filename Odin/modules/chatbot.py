#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatBot Modülü - GPT-4 ile Sohbet
"""

import os
from openai import OpenAI
from config import OPENAI_API_KEY, CHAT_MODEL, MAX_TOKENS, TEMPERATURE

class ChatBot:
    def __init__(self):
        """ChatBot'u başlat"""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = CHAT_MODEL
        self.conversation_history = []
    
    def chat(self, user_message):
        """Kullanıcı mesajına cevap ver"""
        try:
            # Konversasyon geçmişine ekle
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            
            # API'ye istek gönder
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            
            # Cevabı al
            assistant_message = response.choices[0].message.content
            
            # Geçmişe ekle
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
        
        except Exception as e:
            return f"❌ Hata: {str(e)}"
    
    def clear_history(self):
        """Konversasyon geçmişini temizle"""
        self.conversation_history = []
