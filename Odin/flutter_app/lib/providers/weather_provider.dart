import 'package:flutter/material.dart';

class WeatherProvider extends ChangeNotifier {
  dynamic _currentWeather;
  List<dynamic> _forecast = [];
  bool _isLoading = false;
  String? _error;

  dynamic get currentWeather => _currentWeather;
  List<dynamic> get forecast => _forecast;
  bool get isLoading => _isLoading;
  String? get error => _error;

  Future<void> fetchWeather(String city) async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      // API çağrısı simülasyonu
      await Future.delayed(const Duration(seconds: 2));
      
      _currentWeather = {
        'city': city,
        'country': 'TR',
        'temp': 22.5,
        'feelsLike': 20.0,
        'tempMin': 18.0,
        'tempMax': 25.0,
        'humidity': 65,
        'pressure': 1013,
        'windSpeed': 12.5,
        'description': 'Parçalı Bulutlu',
      };
    } catch (e) {
      _error = 'Hava durumu alınamadı: $e';
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }
}
