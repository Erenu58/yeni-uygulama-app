# 🤖 Odin AI - Advanced AI Assistant

Odin, sesli komut, yapay zeka sohbeti, web araştırması, resim oluşturma, dosya yönetimi ve görev takibi yapabilen **gelişmiş yapay zeka asistanı**.

## ✨ Özellikler

- 🗣️ **Sesli Komut** - Konuşarak komut verme
- 💬 **AI Sohbeti** - GPT-4 ile akıllı sohbet
- 🌐 **Web Araştırması** - İnternet'te arama yapma
- 🎨 **Resim Oluşturma** - DALL-E 3 ile resim oluşturma
- 📁 **Dosya Yönetimi** - Dosya oluşturma ve silme
- 📋 **Görev Yönetimi** - Görev ekle, listele, tamamla
- 📊 **Raporlar** - Otomatik rapor oluşturma
- 🔍 **Veri Analizi** - İstatistik ve analiz

## 🚀 Başlangıç

### 1. Kütüphaneleri Kur

```bash
cd Odin
pip install -r requirements.txt
```

### 2. API Keys Ayarla

`.env.example` dosyasını `.env` olarak kopyala ve API key'lerini ekle:

```bash
cp .env.example .env
```

`.env` dosyasını düzenle:

```
OPENAI_API_KEY=sk-proj-YOUR_KEY_HERE
BRAVE_API_KEY=YOUR_KEY
GOOGLE_API_KEY=YOUR_KEY
```

### 3. Programı Çalıştır

```bash
python main.py
```

## 📝 Komutlar

| Komut | Açıklama |
|-------|----------|
| `dinle` | Sesli komut al |
| `resim yap: [açıklama]` | Resim oluştur |
| `ara: [sorgu]` | Web araştırması yap |
| `dosya oluştur: [ad]` | Dosya oluştur |
| `görev ekle: [görev]` | Görev ekle |
| `görevler` | Görevleri listele |
| `rapor` | Rapor oluştur |
| `çık` | Programı kapat |

## 📁 Dosya Yapısı

```
Odin/
├── main.py                 # Ana giriş noktası
├── assistant.py            # Asistan çekirdeği
├── config.py               # Ayarlar
├── requirements.txt        # Kütüphaneler
├── .env.example            # API key şablonu
├── .env                    # API key'ler (gizli)
├── README.md               # Dokümantasyon
├── modules/
��   ├── chatbot.py          # GPT-4 sohbet
│   ├── speech.py           # Ses tanıma
│   ├── web_search.py       # Web araştırması
│   ├── file_manager.py     # Dosya yönetimi
│   ├── tasks.py            # Görev yönetimi
│   ├── image_gen.py        # Resim oluşturma
│   └── analytics.py        # Raporlar
└── data/
    ├── tasks.json          # Görevler
    ├── history.json        # Geçmiş
    └── reports/            # Rapor dosyaları
```

## 🔑 API Keys

### OpenAI (Gerekli)
1. https://platform.openai.com/account/api-keys
2. "Create new secret key" tıkla
3. Key'i kopyala ve `.env` dosyasına ekle

### Brave Search (Opsiyonel)
1. https://api.search.brave.com
2. Ücretsiz key al

### Google Cloud (Opsiyonel)
1. https://console.cloud.google.com
2. Proje oluştur
3. Speech-to-Text API aç

## ⚙️ Konfigürasyon

`config.py` dosyasında ayarlamaları değiştirebilirsin:

```python
CHAT_MODEL = "gpt-4"           # AI modeli
MAX_TOKENS = 2048             # Maksimum token
TEMPERATURE = 0.7             # Yaratıcılık seviyesi
```

## 📊 Veri Depolama

- **SQLite** - Yapılandırılmış veri
- **JSON** - Görevler ve geçmiş
- **CSV** - Raporlar

## 🔒 Güvenlik

⚠️ **API Key'leri asla herkese gösterme!**

- `.env` dosyası `.gitignore`'a eklenmiş
- Tokens şifrelenmiş olarak depolanır
- Tüm API istekleri HTTPS üzerinde

## 🐛 Hata Çözümü

### Python bulunamıyor
```bash
# Windows
pip install python
```

### OpenAI hatası
```bash
# API key'i kontrol et
echo %OPENAI_API_KEY%
```

### Ses tanıma çalışmıyor
```bash
# PyAudio'yu yeniden kur
pip install --upgrade pyaudio
```

## 📞 Destek

Probleminiz varsa:
1. `config.py`'de DEBUG=True yap
2. Hata mesajını kopyala
3. GitHub issue'sine yapıştır

## 📝 Lisans

MIT License

## 🚀 İleri Seviye Özellikler (Planlanmış)

- [ ] Veritabanı entegrasyonu
- [ ] Machine Learning modelleri
- [ ] Çoklu dil desteği
- [ ] Plugin sistemi
- [ ] Web arayüzü
- [ ] Mobil uygulama

---

**Odin AI - Yapay Zeka, Senin Kontolünde** 🎯
