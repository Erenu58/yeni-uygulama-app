#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Generation Modülü - Resim Oluşturma (DALL-E 3)
"""

import os
from openai import OpenAI
from config import OPENAI_API_KEY, IMAGE_MODEL

class ImageGenerator:
    def __init__(self):
        """Resim oluşturucuyu başlat"""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = IMAGE_MODEL
    
    def generate(self, prompt):
        """Resim oluştur"""
        try:
            response = self.client.images.generate(
                model=self.model,
                prompt=prompt,
                size="1024x1024",
                quality="hd",
                n=1
            )
            
            image_url = response.data[0].url
            return f"🎨 Resim oluşturuldu!\n📸 {image_url}"
        
        except Exception as e:
            return f"❌ Resim oluşturma hatası: {str(e)}"
