import 'package:flutter/material.dart';

class LocationProvider extends ChangeNotifier {
  String _currentLocation = 'İstanbul';
  bool _isLoading = false;

  String get currentLocation => _currentLocation;
  bool get isLoading => _isLoading;

  Future<void> getCurrentLocation() async {
    _isLoading = true;
    notifyListeners();

    try {
      // Lokasyon simülasyonu
      await Future.delayed(const Duration(seconds: 1));
      _currentLocation = 'İstanbul';
    } catch (e) {
      debugPrint('Lokasyon hatası: $e');
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }
}
