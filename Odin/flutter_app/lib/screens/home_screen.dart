import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/weather_provider.dart';
import '../providers/location_provider.dart';
import '../widgets/weather_card.dart';
import '../widgets/search_bar.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({Key? key}) : super(key: key);

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  @override
  void initState() {
    super.initState();
    Future.delayed(Duration.zero, () {
      context.read<LocationProvider>().getCurrentLocation();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('🌤️ Odin Weather'),
        elevation: 0,
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: SearchBar(
              onSearch: (city) {
                context.read<WeatherProvider>().fetchWeather(city);
              },
              onLocationTap: () {
                context.read<LocationProvider>().getCurrentLocation();
              },
            ),
          ),
          Expanded(
            child: Consumer<WeatherProvider>(
              builder: (context, weatherProvider, _) {
                if (weatherProvider.isLoading) {
                  return const Center(child: CircularProgressIndicator());
                }

                if (weatherProvider.error != null) {
                  return Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Icon(Icons.error, size: 64, color: Colors.red),
                        const SizedBox(height: 16),
                        Text(weatherProvider.error ?? 'Hata oluştu'),
                      ],
                    ),
                  );
                }

                if (weatherProvider.currentWeather == null) {
                  return const Center(child: Text('İstanbul için hava durumu yükleniyor...'));
                }

                return SingleChildScrollView(
                  child: Column(
                    children: [
                      WeatherCard(weather: weatherProvider.currentWeather!),
                      Padding(
                        padding: const EdgeInsets.all(16.0),
                        child: Text(
                          'Detaylı Bilgiler',
                          style: Theme.of(context).textTheme.headlineSmall,
                        ),
                      ),
                      _buildDetailsGrid(context, weatherProvider),
                    ],
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDetailsGrid(BuildContext context, WeatherProvider provider) {
    final weather = provider.currentWeather!;
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: GridView.count(
        crossAxisCount: 2,
        shrinkWrap: true,
        physics: const NeverScrollableScrollPhysics(),
        childAspectRatio: 1.5,
        children: [
          _DetailCard(icon: '💧', label: 'Nem', value: '${weather.humidity}%'),
          _DetailCard(icon: '💨', label: 'Rüzgar', value: '${weather.windSpeed} m/s'),
          _DetailCard(icon: '🌡️', label: 'Hissedilen', value: '${weather.feelsLike.toStringAsFixed(1)}°C'),
          _DetailCard(icon: '📊', label: 'Basınç', value: '${weather.pressure} hPa'),
          _DetailCard(icon: '🔥', label: 'Min Sıcaklık', value: '${weather.tempMin}°C'),
          _DetailCard(icon: '❄️', label: 'Max Sıcaklık', value: '${weather.tempMax}°C'),
        ],
      ),
    );
  }
}

class _DetailCard extends StatelessWidget {
  final String icon;
  final String label;
  final String value;

  const _DetailCard({
    required this.icon,
    required this.label,
    required this.value,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(icon, style: const TextStyle(fontSize: 32)),
            const SizedBox(height: 8),
            Text(label, style: Theme.of(context).textTheme.labelSmall),
            const SizedBox(height: 4),
            Text(value, style: Theme.of(context).textTheme.headlineSmall),
          ],
        ),
      ),
    );
  }
}
