# 🏦 TCMB ML Dashboard

TCMB USD/TRY Makine Öğrenmesi projesinin sonuçlarını görselleştiren
Streamlit tabanlı interaktif dashboard.

## 🚀 Kurulum ve Çalıştırma

```bash
# Bağımlılıkları kur
pip install -r requirements.txt

# Dashboard'u başlat
streamlit run app.py
```

Tarayıcınızda `http://localhost:8501` adresinde açılacaktır.

## 📁 Yapı

```
tcmb-dashboard/
├── app.py               # Ana dashboard uygulaması
├── utils.py             # Veri yükleme yardımcıları
├── requirements.txt     # Python bağımlılıkları
├── assets/              # Logo ve statik dosyalar
└── README.md            # Bu dosya
```

## 📑 Sayfalar

| Sayfa | Açıklama |
|-------|----------|
| 🏠 Genel Bakış | Proje özeti, ana metrikler, kur trendi |
| 📊 Veri İnceleme | Tablo, kolon tipleri, eksik değer analizi |
| 📈 Regresyon | Tahmin vs gerçek, hata dağılımı, MAE/RMSE/MAPE |
| 🎯 Sınıflandırma | Confusion matrix, ROC-AUC/F1, model karşılaştırma |
| 📝 Rapor | report.md içeriğini görüntüleme |
| ⬇️ İndir | Metrik, rapor ve tahmin dosyalarını indirme |

## 🔗 ML Projesi Bağlantısı

Dashboard, `../tcmb-ml-proje/` klasöründeki çıktıları otomatik algılar.
Proje bulunamazsa demo verilerle çalışır.

Gerçek veriler için önce ML pipeline'ını çalıştırın:

```bash
cd ../tcmb-ml-proje
python -m src.download_data
python -m src.preprocess
python -m src.features
python -m src.train_regression
python -m src.train_classification
python -m src.evaluate
```
