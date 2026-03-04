# 🏦 TCMB USD/TRY Kur Tahmini — Makine Öğrenmesi Projesi

Bu proje, Türkiye Cumhuriyet Merkez Bankası (TCMB) verilerini kullanarak USD/TRY
döviz kuru tahmini ve kur sıçraması tespiti yapmak amacıyla geliştirilmiştir.

## 📁 Proje Yapısı

```
tcmb-ml-proje/
├── README.md                    # Bu dosya
├── requirements.txt             # Python bağımlılıkları
├── data/
│   ├── raw/                     # Ham veri dosyaları (CSV)
│   └── processed/               # İşlenmiş veri
├── src/
│   ├── __init__.py
│   ├── utils.py                 # Sabitler ve yardımcı fonksiyonlar
│   ├── download_data.py         # Veri indirme (Kaggle / yerel)
│   ├── preprocess.py            # Veri ön işleme ve birleştirme
│   ├── features.py              # Özellik mühendisliği
│   ├── train_regression.py      # Regresyon modelleri (Ridge + RF)
│   ├── train_classification.py  # Sınıflandırma modelleri (LR + RF)
│   └── evaluate.py              # Değerlendirme, grafikler, rapor
├── models/                      # Eğitilmiş modeller (.joblib)
├── reports/
│   ├── report.md                # Türkçe detaylı rapor
│   ├── figures/                 # Grafikler (PNG)
│   └── metrics.json             # Birleşik metrikler
└── metrics/
    └── metrics.json             # Metrik dosyası
```

## 🚀 Kurulum

### 1. Python Bağımlılıkları

```bash
pip install -r requirements.txt
```

### 2. Kaggle API Kurulumu (Opsiyonel)

Veri zaten `dataset/` klasöründe mevcutsa bu adım **gerekli değildir**.
Kaggle'dan indirmek isterseniz:

1. [Kaggle hesabınıza](https://www.kaggle.com/settings) giriş yapın
2. **"Create New Token"** butonuna tıklayın
3. İndirilen `kaggle.json` dosyasını aşağıdaki konuma kopyalayın:
   - **Windows:** `C:\Users\<kullanıcı_adı>\.kaggle\kaggle.json`
   - **Linux/Mac:** `~/.kaggle/kaggle.json`
4. Kaggle CLI kurun: `pip install kaggle`

## 📊 Çalıştırma

Projeyi aşağıdaki sırada çalıştırın:

```bash
# 1. Veriyi indir / kopyala
python -m src.download_data

# 2. Veriyi ön işle ve birleştir
python -m src.preprocess

# 3. Özellik mühendisliği
python -m src.features

# 4. Regresyon modellerini eğit
python -m src.train_regression

# 5. Sınıflandırma modellerini eğit
python -m src.train_classification

# 6. Değerlendirme, grafikler ve rapor oluştur
python -m src.evaluate
```

### Opsiyonel Parametreler

```bash
# Sıçrama eşiğini değiştir (default: %2.0)
python -m src.features --esik 1.5

# Veriyi yeniden indir
python -m src.download_data --force
```

## 🎯 Modeller

### Regresyon (USD/TRY 1 Gün Sonra Tahmini)
| Model | Açıklama |
|-------|----------|
| **Naive Baseline** | Yarın = bugün (karşılaştırma için) |
| **Ridge Regression** | L2 regularizasyonlu doğrusal regresyon |
| **RandomForestRegressor** | Topluluk öğrenmesi tabanlı regresyon |

### Sınıflandırma (Kur Sıçraması Tespiti)
| Model | Açıklama |
|-------|----------|
| **Baseline (0)** | Her zaman "sıçrama yok" tahmini |
| **LogisticRegression** | Dengeli sınıf ağırlıklı lojistik regresyon |
| **RandomForestClassifier** | Dengeli sınıf ağırlıklı rastgele orman |

## 📈 Çıktılar

- `reports/report.md` — Detaylı Türkçe rapor
- `reports/figures/` — 5 adet görselleştirme
- `reports/metrics.json` — Birleşik model metrikleri
- `models/best_regression_model.joblib` — En iyi regresyon modeli
- `models/best_classification_model.joblib` — En iyi sınıflandırma modeli

## 📋 Değerlendirme Metrikleri

- **Regresyon:** MAE, RMSE, MAPE
- **Sınıflandırma:** F1, ROC-AUC, Precision, Recall, Confusion Matrix
- **Cross-validation:** TimeSeriesSplit (5 katlama)

## ⚠️ Önemli Notlar

- Train/test ayrımı **kronolojik**tir (shuffle yapılmaz)
- Lag ve rolling özelliklerde **veri sızıntısı (leakage)** olmamasına dikkat edilmiştir
- Sınıflandırmada sınıf dengesizliği `class_weight="balanced"` ile ele alınmıştır
- Rastgele sayı üreteci sabit seed (42) ile tekrarlanabilirlik sağlanmıştır
