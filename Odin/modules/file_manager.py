#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Manager Modülü - Dosya Yönetimi
"""

import os
from datetime import datetime
from config import DATA_DIR

class FileManager:
    def __init__(self):
        """Dosya yöneticisini başlat"""
        self.base_dir = DATA_DIR
        os.makedirs(self.base_dir, exist_ok=True)
    
    def create_file(self, filename, content=""):
        """Dosya oluştur"""
        try:
            filepath = os.path.join(self.base_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"✅ '{filename}' başarıyla oluşturuldu."
        except Exception as e:
            return f"❌ Dosya oluşturma hatası: {str(e)}"
    
    def read_file(self, filename):
        """Dosya oku"""
        try:
            filepath = os.path.join(self.base_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"❌ Dosya okuma hatası: {str(e)}"
    
    def delete_file(self, filename):
        """Dosya sil"""
        try:
            filepath = os.path.join(self.base_dir, filename)
            os.remove(filepath)
            return f"✅ '{filename}' başarıyla silindi."
        except Exception as e:
            return f"❌ Dosya silme hatası: {str(e)}"
    
    def list_files(self):
        """Dosyaları listele"""
        try:
            files = os.listdir(self.base_dir)
            return "\n".join([f"📄 {f}" for f in files]) if files else "❌ Dosya yok."
        except Exception as e:
            return f"❌ Liste hatası: {str(e)}"
