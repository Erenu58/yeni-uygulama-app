#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PWA Routes - Progressive Web App Rotaları
"""

from flask import render_template, jsonify
from weather_dashboard import app
from modules.weather import WeatherDashboard

weather = WeatherDashboard()

@app.route('/pwa')
def pwa():
    """PWA Ana Sayfa"""
    return render_template('pwa.html')

@app.route('/offline')
def offline():
    """Offline Sayfası"""
    return render_template('offline.html')

@app.route('/api/pwa/weather/<city>')
def pwa_weather(city):
    """PWA İçin Hava Durumu"""
    data = weather.get_current_weather(city)
    return jsonify(data)

@app.route('/api/pwa/forecast/<city>')
def pwa_forecast(city):
    """PWA İçin 5 Günlük Tahmin"""
    data = weather.get_forecast(city, days=5)
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
