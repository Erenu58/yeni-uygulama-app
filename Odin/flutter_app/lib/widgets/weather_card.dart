import 'package:flutter/material.dart';

class WeatherCard extends StatelessWidget {
  final dynamic weather;

  const WeatherCard({Key? key, required this.weather}) : super(key: key);

  String _getWeatherEmoji(String description) {
    description = description.toLowerCase();
    if (description.contains('bulut')) return '☁️';
    if (description.contains('açık') || description.contains('sunny')) return '☀️';
    if (description.contains('yağmur')) return '🌧️';
    if (description.contains('gök gürültü')) return '⛈️';
    if (description.contains('kar')) return '❄️';
    if (description.contains('sis')) return '🌫️';
    return '🌤️';
  }

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.all(16),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            Text(
              '${weather.city}, ${weather.country}',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 16),
            Text(
              _getWeatherEmoji(weather.description),
              style: const TextStyle(fontSize: 64),
            ),
            const SizedBox(height: 16),
            Text(
              '${weather.temp.toStringAsFixed(0)}°C',
              style: Theme.of(context).textTheme.displayMedium,
            ),
            const SizedBox(height: 8),
            Text(
              weather.description,
              style: Theme.of(context).textTheme.titleMedium,
            ),
          ],
        ),
      ),
    );
  }
}
