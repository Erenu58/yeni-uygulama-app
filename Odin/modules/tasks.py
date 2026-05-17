#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task Manager Modülü - Görev Yönetimi
"""

import json
import os
from datetime import datetime
from config import TASKS_FILE

class TaskManager:
    def __init__(self):
        """Görev yöneticisini başlat"""
        self.tasks_file = TASKS_FILE
        self.tasks = self.load_tasks()
    
    def load_tasks(self):
        """Görevleri yükle"""
        if os.path.exists(self.tasks_file):
            try:
                with open(self.tasks_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_tasks(self):
        """Görevleri kaydet"""
        with open(self.tasks_file, 'w', encoding='utf-8') as f:
            json.dump(self.tasks, f, ensure_ascii=False, indent=2)
    
    def add_task(self, task_text):
        """Görev ekle"""
        task = {
            "id": len(self.tasks) + 1,
            "text": task_text,
            "created_at": datetime.now().isoformat(),
            "completed": False
        }
        self.tasks.append(task)
        self.save_tasks()
        return f"✅ Görev eklendi: {task_text}"
    
    def list_tasks(self):
        """Görevleri listele"""
        if not self.tasks:
            return "❌ Görev yok."
        
        for task in self.tasks:
            status = "✅" if task['completed'] else "⏳"
            print(f"{status} {task['id']}. {task['text']}")
    
    def complete_task(self, task_id):
        """Görev tamamla"""
        for task in self.tasks:
            if task['id'] == task_id:
                task['completed'] = True
                self.save_tasks()
                return f"✅ Görev tamamlandı: {task['text']}"
        return "❌ Görev bulunamadı."
