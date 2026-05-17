#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web Search Modülü - İnternet Araştırması
"""

import requests
from bs4 import BeautifulSoup
from config import BRAVE_API_KEY, SEARCH_RESULTS_LIMIT, TIMEOUT

class WebSearcher:
    def __init__(self):
        """Web araştırıcısını başlat"""
        self.api_key = BRAVE_API_KEY
        self.results_limit = SEARCH_RESULTS_LIMIT
    
    def search(self, query):
        """Web araştırması yap"""
        try:
            # Google'da arama (Brave API yerine)
            url = f"https://www.google.com/search"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
            }
            params = {'q': query}
            
            response = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
            response.raise_for_status()
            
            # Sonuçları parse et
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            for g in soup.find_all('div', class_='g'):
                anchor = g.find('a', href=True)
                if anchor:
                    link = anchor['href']
                    title = anchor.get_text()
                    results.append(f"📌 {title}\n   {link}")
            
            if results:
                return "\n\n".join(results[:self.results_limit])
            else:
                return "❌ Sonuç bulunamadı."
        
        except Exception as e:
            return f"❌ Arama hatası: {str(e)}"
