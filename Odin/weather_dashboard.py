#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hava Durumu Dashboard - Web Arayüzü
Flask ile gerçek zamanlı hava durumu dashboard'u
"""

from flask import Flask, render_template, request, jsonify
from modules.weather import WeatherDashboard
from datetime import datetime
import json
import os

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# Weather Dashboard örneğini oluştur
weather = WeatherDashboard()

# Favori şehirler
FAVORITE_CITIES = ["Istanbul", "Ankara", "Izmir", "Bursa", "Antalya"]

@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html', favorite_cities=FAVORITE_CITIES)

@app.route('/api/weather/<city>')
def get_weather(city):
    """Belirli şehir için hava durumu"""
    data = weather.get_current_weather(city)
    return jsonify(data)

@app.route('/api/forecast/<city>')
def get_forecast(city):
    """5 günlük tahmin"""
    data = weather.get_forecast(city, days=5)
    return jsonify(data)

@app.route('/api/compare', methods=['POST'])
def compare_cities():
    """Şehirleri karşılaştır"""
    cities = request.json.get('cities', [])
    data = weather.get_multiple_cities(cities)
    return jsonify(data)

@app.route('/api/export/<city>')
def export_weather(city):
    """Hava durumunu dışa aktar"""
    result = weather.export_to_json(city)
    return jsonify({"mesaj": result})

if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════════╗
    ║    🌤️  HAVA DURUMU DASHBOARD                ║
    ║    🔗 http://localhost:5000                ║
    ╚════════════════════════════════════════════╝
    """)
    app.run(debug=True, host='0.0.0.0', port=5000)
