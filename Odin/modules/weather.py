#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weather Module - Hava Durumu Verileri
OpenWeatherMap API kullanan gerçek zamanlı hava durumu modülü
"""

import requests
import json
from datetime import datetime
from typing import Dict, Optional
from config import TIMEOUT

class WeatherDashboard:
    def __init__(self, api_key: str = None):
        """Weather Dashboard'u başlat"""
        # OpenWeatherMap API (ücretsiz)
        self.api_key = api_key or "a6d3634cd0c58ca0e92c72caebb1e209"
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.timeout = TIMEOUT
        self.weather_cache = {}
    
    def get_current_weather(self, city: str, units: str = "metric") -> Dict:
        """Şehrin güncel hava durumunu al"""
        try:
            endpoint = f"{self.base_url}/weather"
            params = {
                "q": city,
                "appid": self.api_key,
                "units": units,  # metric = Celsius, imperial = Fahrenheit
                "lang": "tr"  # Türkçe açıklama
            }
            
            response = requests.get(endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            if response.status_code == 200:
                # Verileri format et
                weather_info = {
                    "şehir": data.get("name"),
                    "ülke": data.get("sys", {}).get("country"),
                    "sıcaklık": data["main"]["temp"],
                    "hissedilen": data["main"]["feels_like"],
                    "min_sıcaklık": data["main"]["temp_min"],
                    "max_sıcaklık": data["main"]["temp_max"],
                    "nem": data["main"]["humidity"],
                    "basınç": data["main"]["pressure"],
                    "hava_durumu": data["weather"][0]["main"],
                    "açıklama": data["weather"][0]["description"],
                    "rüzgar_hızı": data["wind"]["speed"],
                    "bulutluluk": data["clouds"]["all"],
                    "görüş_mesafesi": data.get("visibility"),
                    "yağmur": data.get("rain", {}).get("1h", 0),
                    "kar": data.get("snow", {}).get("1h", 0),
                    "zaman": datetime.fromtimestamp(data["dt"]).strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Cache'e kaydı
                self.weather_cache[city.lower()] = weather_info
                return weather_info
            else:
                return {"hata": "Şehir bulunamadı"}
        
        except requests.exceptions.RequestException as e:
            return {"hata": f"API hatası: {str(e)}"}
        except Exception as e:
            return {"hata": f"Hata: {str(e)}"}
    
    def get_forecast(self, city: str, days: int = 5) -> Dict:
        """5 günlük hava durumu tahmini al"""
        try:
            endpoint = f"{self.base_url}/forecast"
            params = {
                "q": city,
                "appid": self.api_key,
                "units": "metric",
                "lang": "tr",
                "cnt": days * 8  # Her 3 saatte bir (5 gün = 40 veri)
            }
            
            response = requests.get(endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            if response.status_code == 200:
                forecasts = []
                for forecast in data["list"][:days*8]:
                    forecasts.append({
                        "zaman": datetime.fromtimestamp(forecast["dt"]).strftime("%Y-%m-%d %H:%M"),
                        "sıcaklık": forecast["main"]["temp"],
                        "hava_durumu": forecast["weather"][0]["main"],
                        "açıklama": forecast["weather"][0]["description"],
                        "nem": forecast["main"]["humidity"],
                        "rüzgar_hızı": forecast["wind"]["speed"],
                        "yağış_olasılığı": forecast.get("pop", 0) * 100
                    })
                
                return {
                    "şehir": data["city"]["name"],
                    "ülke": data["city"]["country"],
                    "tahminler": forecasts
                }
            else:
                return {"hata": "Tahmin alınamadı"}
        
        except Exception as e:
            return {"hata": f"Tahmin hatası: {str(e)}"}
    
    def get_air_quality(self, lat: float, lon: float) -> Dict:
        """Hava kalitesi verileri al (AQI)"""
        try:
            endpoint = f"{self.base_url}/air_quality"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key
            }
            
            response = requests.get(endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            aqi_levels = {
                1: "Çok İyi",
                2: "İyi",
                3: "Orta",
                4: "Kötü",
                5: "Çok Kötü"
            }
            
            if data["list"]:
                aqi = data["list"][0]
                return {
                    "AQI_seviyesi": aqi_levels.get(aqi["main"]["aqi"], "Bilinmiyor"),
                    "CO": aqi["components"].get("co"),
                    "NO2": aqi["components"].get("no2"),
                    "O3": aqi["components"].get("o3"),
                    "PM2.5": aqi["components"].get("pm2_5"),
                    "PM10": aqi["components"].get("pm10")
                }
            else:
                return {"hata": "Hava kalitesi verisi bulunamadı"}
        
        except Exception as e:
            return {"hata": f"Hava kalitesi hatası: {str(e)}"}
    
    def display_weather(self, city: str) -> str:
        """Hava durumunu güzel format'ta göster"""
        weather = self.get_current_weather(city)
        
        if "hata" in weather:
            return f"❌ {weather['hata']}"
        
        emoji_map = {
            "Clouds": "☁️",
            "Clear": "☀️",
            "Rain": "🌧️",
            "Drizzle": "🌦️",
            "Thunderstorm": "⚡",
            "Snow": "❄️",
            "Mist": "🌫️"
        }
        
        emoji = emoji_map.get(weather["hava_durumu"], "🌡️")
        
        output = f"""
╔════════════════════════════════════════════════════════════╗
║           🌍 HAVA DURUMU DASHBOARD - {weather['şehir'].upper()}, {weather['ülke']}           ║
╚════════════════════════════════════════════════════════════╝

{emoji} Hava Durumu: {weather['hava_durumu']} ({weather['açıklama']})

🌡️  SICAKLIK BİLGİSİ:
   • Güncel: {weather['sıcaklık']}°C
   • Hissedilen: {weather['hissedilen']}°C
   • Minimum: {weather['min_sıcaklık']}°C
   • Maksimum: {weather['max_sıcaklık']}°C

💧 REM VE BASINÇ:
   • Nem: {weather['nem']}%
   • Basınç: {weather['basınç']} hPa

💨 RÜZGAR BİLGİSİ:
   • Rüzgar Hızı: {weather['rüzgar_hızı']} m/s
   • Bulutluluk: {weather['bulutluluk']}%

📊 GÖRÜNÜRLÜK VE YAĞIŞ:
   • Görüş Mesafesi: {weather['görüş_mesafesi']} m
   • Yağmur (son 1 saat): {weather['yağmur']} mm
   • Kar (son 1 saat): {weather['kar']} mm

🕐 Güncellenme: {weather['zaman']}
"""
        return output
    
    def display_forecast(self, city: str) -> str:
        """5 günlük tahmini göster"""
        forecast = self.get_forecast(city, days=5)
        
        if "hata" in forecast:
            return f"❌ {forecast['hata']}"
        
        output = f"""
╔════════════════════════════════════════════════════════════╗
║        📅 5 GÜNLÜK HAVA DURUMU TAHMİNİ - {forecast['şehir']}          ║
╚════════════════════════════════════════════════════════════╝
\n"""
        
        for i, forecast_data in enumerate(forecast["tahminler"][:8], 1):
            output += f"""
📍 {forecast_data['zaman']}
   • Sıcaklık: {forecast_data['sıcaklık']}°C
   • Durum: {forecast_data['açıklama']}
   • Nem: {forecast_data['nem']}%
   • Rüzgar: {forecast_data['rüzgar_hızı']} m/s
   • Yağış Olasılığı: {forecast_data['yağış_olasılığı']:.1f}%
"""
        
        return output
    
    def get_multiple_cities(self, cities: list) -> Dict:
        """Birden fazla şehrin hava durumunu al"""
        results = {}
        for city in cities:
            results[city] = self.get_current_weather(city)
        return results
    
    def compare_cities(self, cities: list) -> str:
        """Şehirleri karşılaştır"""
        weather_data = self.get_multiple_cities(cities)
        
        output = "\n╔════════════════════════════════════════════════════════════╗\n"
        output += "║            🌍 ŞEHİRLER KARŞILAŞTIRMASI                      ║\n"
        output += "╚════════════════════════════════════════════════════════════╝\n\n"
        
        # Tablo başlığı
        output += f"{'Şehir':<15} {'Sıcaklık':<12} {'Durum':<20} {'Nem':<10}\n"
        output += "-" * 57 + "\n"
        
        for city, data in weather_data.items():
            if "hata" not in data:
                output += f"{data['şehir']:<15} {data['sıcaklık']}°C{'':<6} {data['açıklama']:<20} {data['nem']}%\n"
        
        return output
    
    def export_to_json(self, city: str, filename: str = None) -> str:
        """Hava durumu verilerini JSON'a kaydet"""
        weather = self.get_current_weather(city)
        forecast = self.get_forecast(city)
        
        data = {
            "güncel": weather,
            "tahmin": forecast,
            "kaydedilme_zamanı": datetime.now().isoformat()
        }
        
        if filename is None:
            filename = f"weather_{city}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return f"✅ Veriler {filename} dosyasına kaydedildi"
        except Exception as e:
            return f"❌ Kayıt hatası: {str(e)}"
