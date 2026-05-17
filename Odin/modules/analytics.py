#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analytics Modülü - Veri Analizi ve Raporlar
"""

import json
import os
from datetime import datetime
from config import REPORT_PATH

class Analytics:
    def __init__(self):
        """Analytics'i başlat"""
        self.report_path = REPORT_PATH
        os.makedirs(self.report_path, exist_ok=True)
    
    def generate_report(self, history):
        """Rapor oluştur"""
        try:
            report = {
                "generated_at": datetime.now().isoformat(),
                "total_interactions": len(history),
                "interactions": history
            }
            
            # Rapor dosyasını kaydet
            report_file = os.path.join(
                self.report_path,
                f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            return f"📊 Rapor oluşturuldu!\n💾 {report_file}"
        
        except Exception as e:
            return f"❌ Rapor hatası: {str(e)}"
