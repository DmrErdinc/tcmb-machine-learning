# 🎓 TCMB USD/TRY Kur Tahmini — Sunum Konuşma Metni

> **Ders:** Makine Öğrenmesi  
> **Konu:** TCMB Verileri ile USD/TRY Döviz Kuru Tahmini ve Kur Sıçraması Tespiti  

---

## 🟢 SLAYT 1 — Açılış ve Tanıtım

**Konuşma metni:**

> Hocam, arkadaşlar merhabalar. Ben bugün sizlere TCMB — yani Türkiye Cumhuriyet Merkez Bankası — verilerini kullanarak USD/TRY döviz kurunu makine öğrenmesi yöntemleriyle nasıl tahmin ettiğimizi anlatacağım.
>
> Projemizde iki temel görev var:
> 1. **Regresyon** — Bir sonraki iş gününün USD/TRY kurunu tahmin etmek
> 2. **Sınıflandırma** — Ertesi gün büyük bir kur sıçraması olup olmayacağını tespit etmek
>
> Yani hem "yarın kur ne olur?" sorusunu hem de "yarın büyük bir hareket olacak mı?" sorusunu cevaplamaya çalışıyoruz.

---

## 🟢 SLAYT 2 — Veri Seti

**Konuşma metni:**

> Projemizde Kaggle üzerinden aldığımız **9 farklı CSV dosyası** kullandık. Bu veriler TCMB'den derlenen gerçek piyasa verilerdir.
>
> Verilerin çeşitliliğine bakacak olursak:
>
> - **Günlük veriler:** USD/TRY döviz kuru (ana hedef değişkenimiz), FX swap mevduat miktarı, döviz işlem hacmi ve TCMB net fonlama
> - **Haftalık veriler:** TL faiz oranı (6 aylık mevduat) ve USD faiz oranı
> - **Aylık veriler:** TÜFE genel endeksi (enflasyon), 12 aylık enflasyon beklentisi ve repo faiz oranı
>
> Burada önemli bir teknik detay var: Farklı frekanstaki verileri nasıl birleştirdik? Haftalık ve aylık verileri günlük seriye **forward-fill** yöntemiyle hizaladık. Yani bir haftalık veri, o haftanın sonraki tüm iş günlerine kopyalanmış oldu. Bu sayede tüm verileri tek bir günlük tablo halinde birleştirebildik.

---

## 🟢 SLAYT 3 — Veri Ön İşleme Pipeline'ı

**Konuşma metni:**

> Veri ön işleme sürecimiz şöyle çalışıyor:
>
> **1. Veri İndirme** (`download_data.py`): Veriler ya Kaggle API ile otomatik indirilir, ya da yerel `dataset/` klasöründen kopyalanır.
>
> **2. Ön İşleme** (`preprocess.py`): 
>    - Her CSV dosyasını okuyoruz
>    - Tarih formatlarını otomatik algılıyoruz (dd-mm-yyyy veya yyyy-mm-dd)
>    - Günlük, haftalık ve aylık verileri USD/TRY ana serisine **left join** ile birleştiriyoruz
>    - Haftalık ve aylık veriler **forward-fill** ile günlüğe dönüştürülüyor
>    - Eksik değerler temizleniyor
>    - Sonuçta `merged_data.csv` adında tek bir birleştirilmiş dosya elde ediyoruz
>
> Bu pipeline modüler tasarlanmıştır — her adım bağımsız çalıştırılabilir.

---

## 🟢 SLAYT 4 — Özellik Mühendisliği

**Konuşma metni:**

> Feature engineering aşamasında birleştirilmiş veriden **20'den fazla yeni özellik** türettik. Bunları dört ana kategoride ele aldık:
>
> **1. Gecikme (Lag) Özellikleri:**
>    - `lag_1`, `lag_2`, `lag_5`, `lag_10` — Yani 1, 2, 5 ve 10 gün önceki kur değerleri
>    - Bu, modelin geçmiş kur hareketlerini görmesini sağlıyor
>
> **2. Hareketli İstatistikler (Rolling):**
>    - 5 ve 20 günlük hareketli ortalama ve standart sapma hesapladık
>    - Bu, kurun kısa ve orta vadeli trendini yakalamamızı sağlıyor
>
> **3. Takvim Özellikleri:**
>    - Haftanın günü, ay ve çeyrek bilgisi
>    - Mevsimsel desenleri yakalamak için
>
> **4. Yüzde Değişim:**
>    - Günlük kur değişim oranı
>
> **Çok önemli bir nokta:** Tüm lag ve rolling hesaplamalarında `shift(1)` kullandık. Peki bu neden önemli? Çünkü aksi halde **veri sızıntısı — yani data leakage** oluşur. Model, tahmin etmesi gereken günün bilgisini eğitim sırasında görmüş olur ve bu yapay olarak yüksek başarı oranları verir. Biz bunu önledik.

---

## 🟢 SLAYT 5 — Hedef Değişkenler

**Konuşma metni:**

> İki ayrı hedef değişken oluşturduk:
>
> **Regresyon hedefi (`target_reg`):** Ertesi günün USD/TRY kuru. Yani bugünün verisinden yarının kuru tahmin edilecek.
>
> **Sınıflandırma hedefi (`target_cls`):** Ertesi gün kur sıçraması var mı? Bunun için yüzde 2'lik bir eşik belirledik. Eğer ertesi günkü yüzde değişimin mutlak değeri %2'den büyükse, bunu "sıçrama" (1) olarak etiketledik, değilse "normal" (0).
>
> Bu %2 eşiği sübjektif bir değerdir ve ayarlanabilir. Kodumuzdaki `--esik` parametresiyle değiştirilebilir.

---

## 🟢 SLAYT 6 — Modelleme Yaklaşımı

**Konuşma metni:**

> Modelleme aşamasında çok dikkat etmemiz gereken bir konu var: **zaman serisi verisiyle çalışıyoruz**. Bu yüzden normal random split yani rastgele bölümleme yapamayız. 
>
> **Veri bölünmesi:**
> - İlk %80'i eğitim seti, son %20'si test seti olarak ayırdık
> - **Kronolojik sıralama** korundu, shuffle yapılmadı
> - Bu sayede model geçmiş verilerle eğitilip gelecek verilerde test ediliyor; gerçek hayat senaryosunun simülasyonu bu
>
> **Cross-validation:** 
> - Normal k-fold yerine **TimeSeriesSplit** kullandık (5 katlama)
> - TimeSeriesSplit'te her katlamada eğitim seti genişler ve test seti hep "gelecek" olarak kalır
> - Bu, zaman serisi verisi için en uygun validasyon stratejisidir

---

## 🟢 SLAYT 7 — Regresyon Modelleri ve Sonuçları

**Konuşma metni:**

> Regresyon için üç model karşılaştırdık:
>
> **1. Naive Baseline:** Basitçe "yarın = bugün" diyen bir model. Döviz kurları güçlü otokorelasyon gösterdiği için bu bile MAE 0.028 ile gayet düşük bir hata veriyor.
>
> **2. Ridge Regression:** L2 regularizasyonlu doğrusal regresyon. **En iyi sonucu bu model verdi.** MAE 0.022, RMSE 0.029, MAPE yüzde 0.05. Naive baseline'a göre MAE'de yaklaşık **%21 iyileşme** sağladık.
>
> **3. RandomForest Regressor:** Topluluk öğrenmesi tabanlı. Ancak bu modelde MAE 1.65 ile çok daha yüksek hata çıktı. Bunun nedeni, zaman serisinin doğrusal yapısını Random Forest'ın iyi yakalayamaması ve overfitting riskinin daha yüksek olmasıdır.
>
> **Sonuç:** Ridge Regression, döviz kuru gibi güçlü otokorelasyonlu zaman serilerinde basit ama etkili bir çözüm sunuyor.

---

## 🟢 SLAYT 8 — Sınıflandırma Modelleri ve Sonuçları

**Konuşma metni:**

> Sınıflandırmada da üç model karşılaştırdık:
>
> **1. Baseline (her zaman 0 tahmin et):** Kur sıçramaları nadir olaylar olduğu için, her zaman "sıçrama yok" desen bile accuracy yüksek çıkar. Ama F1 sıfır olur çünkü hiç pozitif tahminde bulunmuyorsun.
>
> **2. Logistic Regression:** Dengeli sınıf ağırlıkları ile eğittik (`class_weight="balanced"`). F1 0.29, ROC-AUC 0.92 çıktı. ROC-AUC yüksek ama F1 düşük — bu, modelin karar eşiğinin optimal olmadığını ve recall'un sadece %17 olduğunu gösteriyor.
>
> **3. RandomForest Classifier:** `class_weight="balanced"` ile eğittik. **Bu model en iyi sonucu verdi:** F1 0.92, ROC-AUC 0.98, Precision 0.85, Recall %100.
>
> **Recall %100 ne demek?** Modelin gerçekleşen tüm kur sıçramalarını yakalayabildiğini gösteriyor. Hiçbir sıçramayı kaçırmıyor. Precision %85 olduğundan, bazı yanlış alarmlar var ama bunlar kabul edilebilir düzeyde.
>
> Sınıf dengesizliği problemiyle `class_weight="balanced"` parametresi ile başa çıktık — bu, azınlık sınıfına daha fazla ağırlık veriyor.

---

## 🟢 SLAYT 9 — Görselleştirmeler

**Konuşma metni:**

> Projemiz otomatik olarak 5 farklı grafik üretmektedir. Bunlar:
>
> **1. Zaman Serisi Grafiği:** USD/TRY kurunun eğitim ve test dönemi ayrımını gösteren grafik. Mavi eğitim, kırmızı test dönemi.
>
> **2. Tahmin vs Gerçek:** Test setindeki gerçek kur değerleri ile Ridge ve RandomForest tahminlerini karşılaştıran bir grafik. Ridge'in gerçek değerlere ne kadar yakın tahmin ettiğini burada net görebilirsiniz.
>
> **3. Hata Dağılımı (Residual):** Tahmin hatalarının histogramı. İdeal durumda sıfır etrafında ve dar bir dağılım görmek isteriz. Ridge'in hata dağılımı çok dar ve merkezlenmiş.
>
> **4. Karmaşıklık Matrisi (Confusion Matrix):** Sınıflandırma modellerinin doğru ve yanlış tahminlerini gösteren matris.
>
> **5. Özellik Önemi (Feature Importance):** En etkili özelliklerin sıralaması. Hangi özelliklerin modelin kararını en çok etkilediğini burada görüyoruz.

---

## 🟢 SLAYT 10 — İnteraktif Dashboard

**Konuşma metni:**

> Sadece grafikler ve raporla sınırlı kalmadık. Proje sonuçlarını interaktif olarak keşfetmek için **Streamlit** tabanlı profesyonel bir dashboard da geliştirdik.
>
> Dashboard'da **6 sayfa** bulunuyor:
>
> - **Genel Bakış:** Ana metrik kartları, model karşılaştırma tabloları ve USD/TRY trend grafiği
> - **Veri İnceleme:** Veri tablosu, kolon tipleri, eksik değer analizi ve istediğiniz sütunun zaman serisi grafiği
> - **Regresyon Sonuçları:** Plotly ile interaktif tahmin vs gerçek grafikleri, hata dağılımı ve detaylı model karşılaştırması
> - **Sınıflandırma:** İnteraktif confusion matrix, model metrikleri karşılaştırma grafiği
> - **Rapor:** Otomatik üretilen Türkçe rapor ve tüm grafikler
> - **İndir:** Tüm sonuçları (JSON, CSV, Markdown) indirme butonları
>
> Dashboard, ML projesi bulunamadığında bile demo veriyle çalışabilecek şekilde tasarlanmıştır. Plotly ile grafikler interaktif, yani zoom yapabilir, hover ile detay görebilirsiniz.

---

## 🟢 SLAYT 11 — Proje Mimarisi ve Teknik Kararlar

**Konuşma metni:**

> Projenin teknik mimarisinden bahsetmek istiyorum. Birkaç önemli tasarım kararı aldık:
>
> **1. Modüler Pipeline:** Her adım bağımsız bir Python modülü — `download_data`, `preprocess`, `features`, `train_regression`, `train_classification`, `evaluate`. Her biri ayrı ayrı çalıştırılabilir.
>
> **2. Tekrarlanabilirlik:** Rastgele sayı üreteci sabit seed (42) ile belirlendi. Bu sayede modeli her çalıştırdığınızda aynı sonuçları alırsınız.
>
> **3. Leakage Önlemi:** Bu belki de en kritik teknik kararımız. `shift(1)` kullanarak gelecek bilgisinin modele sızmasını engelledik.
>
> **4. Kronolojik Bölünme:** Shuffle yapmadık. Zaman serisi verisinde kronolojik sıralamanın korunması şart yoksa model gerçekçi olmayan sonuçlar verir.
>
> **5. Sınıf Dengesizliği:** `class_weight="balanced"` ile sınıf dengesizliğini ele aldık. Kur sıçramaları nadir olaylar, bu yüzden bu parametre olmazsa model azınlık sınıfını tamamen görmezden gelir.
>
> **Teknoloji Yığını:**
> - Python, pandas, NumPy — veri işleme
> - scikit-learn — model eğitimi
> - matplotlib, seaborn — statik grafikler
> - Streamlit, Plotly — interaktif dashboard
> - joblib — model serileştirme

---

## 🟢 SLAYT 12 — Sınırlılıklar

**Konuşma metni:**

> Projenin bazı sınırlılıkları var, bunları açıkça belirtmek isterim:
>
> **1. Veri boyutu:** Yaklaşık 500-700 iş günlük veri ile çalıştık. Bu bir makine öğrenmesi modeli için sınırlı bir boyut. Daha fazla tarihsel veri ile modelin performansı artabilir.
>
> **2. Dış faktörler:** Siyasi olaylar, küresel piyasa şokları, seçimler, savaş gibi dışsal faktörler modele dahil değil. Döviz kuru sadece ekonomik göstergelerle açıklanamaz.
>
> **3. Frekans uyumsuzluğu:** Aylık ve haftalık veriler forward-fill ile günlüğe çevrildi. Bu yöntem basit ama bilgi kaybına yol açabilir. Daha sofistike interpolasyon yöntemleri kullanılabilir.
>
> **4. Eşik değeri:** %2'lik sıçrama eşiği sübjektif olarak belirlenmiştir. Farklı dönemlerde volatilite seviyesi değişeceği için sabit eşik her zaman optimal olmayabilir.

---

## 🟢 SLAYT 13 — Geliştirme Önerileri ve Sonuç

**Konuşma metni:**

> Son olarak, bu projenin nasıl daha da geliştirilebileceğinden bahsetmek istiyorum:
>
> - **Daha fazla veri** ile daha geniş zaman diliminde eğitim
> - **Dışsal değişkenler:** S&P 500, petrol fiyatı, VIX volatilite endeksi gibi küresel göstergeler eklenebilir
> - **Gelişmiş modeller:** XGBoost, LightGBM veya LSTM gibi derin öğrenme modelleri denenebilir
> - **Hiperparametre optimizasyonu:** GridSearchCV veya Optuna ile sistematik hiperparametre araması yapılabilir
> - **Ensemble yöntemler:** Birden fazla modelin tahminlerini birleştiren topluluk yöntemleri kullanılabilir
> - **Olay bazlı özellikler:** TCMB faiz kararı tarihleri gibi kategorik değişkenler eklenebilir
>
> **Genel sonuç olarak:** Bu projede TCMB verilerini kullanarak döviz kuru tahmini için uçtan uca bir makine öğrenmesi pipeline'ı oluşturduk. Ridge Regression ile MAE 0.022 gibi düşük bir hata oranına ulaştık. RandomForest Classifier ile kur sıçramalarını F1 0.92 ve Recall %100 ile tespit edebildik. Ayrıca sonuçları görselleştirmek için profesyonel bir Streamlit dashboard geliştirdik.
>
> Teşekkür ederim. Sorularınız varsa alabilirim.

---

## 📌 Yaptıklarımızın Özet Listesi

Projede adım adım şunları yaptık:

### 1. Veri Toplama ve İndirme
- Kaggle'dan 9 CSV dosyası alındı
- `download_data.py` ile otomatik indirme / kopyalama mekanizması kuruldu
- Veriler: USD/TRY kuru, TL faiz, USD faiz, TÜFE, enflasyon beklentisi, repo faiz, FX swap, FX işlem hacmi, TCMB net fonlama

### 2. Veri Ön İşleme (`preprocess.py`)
- Her dosya tip ve frekansına göre (günlük/haftalık/aylık) okundu
- Tarih formatları otomatik algılandı (dd-mm-yyyy veya yyyy-mm-dd)
- Haftalık ve aylık veriler **forward-fill** ile günlük seriye dönüştürüldü
- Tüm veri setleri USD/TRY ana serisine **left join** ile birleştirildi
- Tamamen boş sütunlar temizlendi, eksik değerler forward-fill + backward-fill ile dolduruldu
- Çıktı: `data/processed/merged_data.csv`

### 3. Özellik Mühendisliği (`features.py`)
- **Lag özellikleri:** 1, 2, 5, 10 günlük gecikmeler
- **Rolling istatistikler:** 5 ve 20 günlük hareketli ortalama ve standart sapma
- **Takvim özellikleri:** Haftanın günü, ay, çeyrek
- **Yüzde değişim:** Günlük kur değişim oranı
- **Hedef değişkenler:** `target_reg` (ertesi gün kuru), `target_cls` (sıçrama var mı — eşik %2)
- **Leakage önlemi:** Tüm lag/rolling hesaplamalarında `shift(1)` kullanıldı
- NaN satırlar temizlendi
- Çıktı: `data/processed/features.csv`

### 4. Regresyon Model Eğitimi (`train_regression.py`)
- Veri %80 train / %20 test olarak **kronolojik** bölündü (shuffle yok)
- `StandardScaler` ile normalizasyon yapıldı
- **Naive Baseline** (yarın = bugün) referans model olarak hesaplandı
- **Ridge Regression** (L2 regularizasyon) eğitildi
- **RandomForest Regressor** eğitildi
- **TimeSeriesSplit** (5 katlama) cross-validation uygulandı
- Metrikler: MAE, RMSE, MAPE
- En iyi model **Ridge** olarak kaydedildi (`best_regression_model.joblib`)
- Sonuçlar: `reports/regression_results.json`, `reports/regression_predictions.csv`

### 5. Sınıflandırma Model Eğitimi (`train_classification.py`)
- Aynı kronolojik train/test bölünmesi kullanıldı
- **Baseline** (her zaman 0 tahmini) referans
- **Logistic Regression** (`class_weight="balanced"`) eğitildi
- **RandomForest Classifier** (`class_weight="balanced"`) eğitildi
- **TimeSeriesSplit** (5 katlama) cross-validation uygulandı
- Metrikler: F1, ROC-AUC, Precision, Recall, Confusion Matrix
- En iyi model **RandomForest Classifier** olarak kaydedildi (`best_classification_model.joblib`)
- Sonuçlar: `reports/classification_results.json`, `reports/classification_predictions.csv`

### 6. Değerlendirme ve Raporlama (`evaluate.py`)
- Tüm model sonuçları yüklendi
- **5 adet grafik** oluşturuldu:
  1. USD/TRY zaman serisi (train/test ayrımı)
  2. Tahmin vs Gerçek (regresyon test seti)
  3. Residual histogramı (hata dağılımı)
  4. Confusion matrix (sınıflandırma)
  5. Feature importance (özellik önemi)
- **Birleşik metrik** dosyası: `reports/metrics.json`
- **Türkçe detaylı rapor** otomatik üretildi: `reports/report.md`

### 7. İnteraktif Dashboard (`tcmb-dashboard/`)
- **Streamlit** tabanlı 6 sayfalık profesyonel dashboard geliştirildi
- **Plotly** ile interaktif grafikler (zoom, hover, pan)
- Demo modda ML projesi olmadan da çalışabiliyor
- Özel CSS ile stillenmiş, gradient metrik kartları, karanlık temali grafikler
- Tüm sonuçları indirme butonları
- Sayfalar: Genel Bakış, Veri İnceleme, Regresyon Sonuçları, Sınıflandırma, Rapor, İndir

### 8. Proje Dokümantasyonu
- `README.md` — GitHub için kapsamlı proje açıklaması
- `requirements.txt` — Bağımlılık listesi (pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, kaggle)
- `reports/report.md` — Otomatik üretilen Türkçe rapor
- Modüler ve tekrarlanabilir kod yapısı (seed=42)
