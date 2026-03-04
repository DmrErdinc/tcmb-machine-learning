"""
Değerlendirme ve görselleştirme modülü.
Tüm model sonuçlarını yükler, 5 grafik oluşturur, birleşik metrik dosyası
ve Türkçe rapor üretir.
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import (
    VERI_ISLEM, RAPORLAR, GRAFIKLER, METRIKLER, MODELLER,
    ensure_dirs, setup_logging,
)

warnings.filterwarnings("ignore")
logger = setup_logging("evaluate")

# Türkçe grafik ayarları
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})
sns.set_style("whitegrid")


def grafik_1_zaman_serisi(df: pd.DataFrame, split_idx: int) -> None:
    """Grafik 1: USD/TRY zaman serisi (train/test ayrımı)."""
    fig, ax = plt.subplots(figsize=(14, 6))

    train_dates = df.iloc[:split_idx]["Date"]
    train_vals = df.iloc[:split_idx]["Conversion_Rate"]
    test_dates = df.iloc[split_idx:]["Date"]
    test_vals = df.iloc[split_idx:]["Conversion_Rate"]

    ax.plot(train_dates, train_vals, color="#2196F3", linewidth=1.5,
            label="Eğitim Seti", alpha=0.9)
    ax.plot(test_dates, test_vals, color="#FF5722", linewidth=1.5,
            label="Test Seti", alpha=0.9)

    # Ayrım çizgisi
    split_date = df.iloc[split_idx]["Date"]
    ax.axvline(x=split_date, color="#4CAF50", linestyle="--", linewidth=2,
               label=f"Train/Test Ayrımı ({split_date.date()})")

    ax.set_title("USD/TRY Döviz Kuru — Eğitim/Test Ayrımı", fontweight="bold")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("USD/TRY Kur")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(GRAFIKLER / "1_zaman_serisi.png")
    plt.close()
    logger.info("Grafik 1: Zaman serisi kaydedildi")


def grafik_2_tahmin_vs_gercek(pred_df: pd.DataFrame, en_iyi: str) -> None:
    """Grafik 2: Regresyon — tahmin vs gerçek (test seti)."""
    fig, ax = plt.subplots(figsize=(14, 6))

    tarihler = pd.to_datetime(pred_df["Date"])

    ax.plot(tarihler, pred_df["Gercek"], color="#2196F3", linewidth=1.5,
            label="Gerçek", alpha=0.9)
    ax.plot(tarihler, pred_df["Ridge"], color="#FF9800", linewidth=1.2,
            label="Ridge", alpha=0.8, linestyle="--")
    ax.plot(tarihler, pred_df["RandomForest"], color="#4CAF50", linewidth=1.2,
            label="Random Forest", alpha=0.8, linestyle=":")

    ax.set_title("Regresyon: Tahmin vs Gerçek (Test Seti)", fontweight="bold")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("USD/TRY Kur")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(GRAFIKLER / "2_tahmin_vs_gercek.png")
    plt.close()
    logger.info("Grafik 2: Tahmin vs Gerçek kaydedildi")


def grafik_3_residual(pred_df: pd.DataFrame, en_iyi: str) -> None:
    """Grafik 3: Residual histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, model_adi, renk in zip(
        axes, ["Ridge", "RandomForest"], ["#FF9800", "#4CAF50"]
    ):
        residuals = pred_df["Gercek"] - pred_df[model_adi]
        ax.hist(residuals, bins=30, color=renk, alpha=0.7, edgecolor="black")
        ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
        ax.set_title(f"{model_adi} — Hata Dağılımı (Residual)", fontweight="bold")
        ax.set_xlabel("Hata (Gerçek - Tahmin)")
        ax.set_ylabel("Frekans")

        # İstatistikler
        ortalama = residuals.mean()
        std = residuals.std()
        ax.text(0.95, 0.95, f"Ort: {ortalama:.4f}\nStd: {std:.4f}",
                transform=ax.transAxes, va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(GRAFIKLER / "3_residual_dagilimi.png")
    plt.close()
    logger.info("Grafik 3: Residual dağılımı kaydedildi")


def grafik_4_confusion_matrix(cls_sonuc: dict) -> None:
    """Grafik 4: Confusion matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (model_adi, renk_harita) in zip(
        axes,
        [("LogisticRegression", "Blues"), ("RandomForest_Cls", "Greens")]
    ):
        if model_adi not in cls_sonuc:
            continue
        cm = np.array(cls_sonuc[model_adi]["Confusion_Matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap=renk_harita, ax=ax,
                    xticklabels=["Normal", "Sıçrama"],
                    yticklabels=["Normal", "Sıçrama"])
        baslik = "Logistic Reg." if "Logistic" in model_adi else "Random Forest"
        ax.set_title(f"{baslik} — Karmaşıklık Matrisi", fontweight="bold")
        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gerçek")

    plt.tight_layout()
    plt.savefig(GRAFIKLER / "4_confusion_matrix.png")
    plt.close()
    logger.info("Grafik 4: Confusion matrix kaydedildi")


def grafik_5_feature_importance(ozellikler: list[str]) -> None:
    """Grafik 5: Feature importance (RF regresyon modeli)."""
    model_yol = MODELLER / "best_regression_model.joblib"
    if not model_yol.exists():
        logger.warning("Regresyon modeli bulunamadı, özellik önemi atlanıyor")
        return

    paket = joblib.load(model_yol)
    model = paket["model"]

    # RandomForest ise doğrudan feature_importances_ kullan
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        # Ridge ise katsayıları kullan
        importances = np.abs(model.coef_)

    # Sıralama
    idx = np.argsort(importances)[::-1][:15]  # İlk 15
    top_ozellikler = [ozellikler[i] for i in idx]
    top_degerler = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_ozellikler)))
    bars = ax.barh(range(len(top_ozellikler)), top_degerler, color=colors)
    ax.set_yticks(range(len(top_ozellikler)))
    ax.set_yticklabels(top_ozellikler)
    ax.invert_yaxis()
    ax.set_title("Özellik Önemi (En İyi Regresyon Modeli — İlk 15)", fontweight="bold")
    ax.set_xlabel("Önem Skoru")

    plt.tight_layout()
    plt.savefig(GRAFIKLER / "5_feature_importance.png")
    plt.close()
    logger.info("Grafik 5: Feature importance kaydedildi")


def birlestir_metrikler(reg_sonuc: dict, cls_sonuc: dict) -> dict:
    """Tüm metrikleri birleştirir."""
    birlesik = {
        "regresyon": {},
        "siniflandirma": {},
    }

    for model_adi in ["Naive_Baseline", "Ridge", "RandomForest_Reg"]:
        if model_adi in reg_sonuc:
            birlesik["regresyon"][model_adi] = {
                k: v for k, v in reg_sonuc[model_adi].items()
                if k in ["MAE", "RMSE", "MAPE", "cv_mae_mean", "cv_rmse_mean"]
            }

    for model_adi in ["Baseline_0", "LogisticRegression", "RandomForest_Cls"]:
        if model_adi in cls_sonuc:
            birlesik["siniflandirma"][model_adi] = {
                k: v for k, v in cls_sonuc[model_adi].items()
                if k in ["F1", "ROC_AUC", "Precision", "Recall", "cv_f1_mean"]
            }

    birlesik["en_iyi_regresyon"] = reg_sonuc.get("en_iyi_model", "N/A")
    birlesik["en_iyi_siniflandirma"] = cls_sonuc.get("en_iyi_model", "N/A")

    return birlesik


def rapor_olustur(birlesik: dict, reg_sonuc: dict, cls_sonuc: dict) -> None:
    """Türkçe rapor (report.md) oluşturur."""

    rapor = """# TCMB USD/TRY Kur Tahmini — Makine Öğrenmesi Raporu

## 1. Proje Özeti

Bu proje, Türkiye Cumhuriyet Merkez Bankası (TCMB) verilerini kullanarak USD/TRY döviz kuru
tahmini yapmayı amaçlamaktadır. İki ana görev ele alınmıştır:

- **Regresyon:** Bir sonraki iş gününün USD/TRY kurunu tahmin etmek
- **Sınıflandırma:** Ertesi gün kur sıçraması olup olmayacağını belirlemek (|değişim| > %2.0)

## 2. Veri Seti

Kaggle'dan alınan veri seti 9 farklı CSV dosyasından oluşmaktadır:

| Veri | Frekans | Açıklama |
|------|---------|----------|
| USD/TRY Kuru | Günlük | Ana hedef değişken |
| TL Faiz Oranı (6 Ay) | Haftalık | TL mevduat faizi |
| USD Faiz Oranı | Haftalık | USD mevduat faizi |
| TÜFE Genel Endeksi | Aylık | Enflasyon göstergesi |
| Enflasyon Beklentisi (12 Ay) | Aylık | Piyasa beklentisi |
| Repo Faiz Oranı | Aylık | Para politikası aracı |
| FX Swap Mevduat | Günlük | Döviz swap hacmi |
| FX İşlem Hacmi | Günlük | Döviz piyasa hacmi |
| TCMB Net Fonlama | Günlük | Merkez bankası fonlama |

Farklı frekanstaki veriler forward-fill yöntemiyle günlük seriye dönüştürülmüştür.

## 3. Özellik Mühendisliği

Oluşturulan özellikler:

- **Gecikme (Lag) Özellikleri:** 1, 2, 5, 10 günlük gecikmeler
- **Hareketli İstatistikler:** 5 ve 20 günlük ortalama ve standart sapma
- **Takvim Özellikleri:** Haftanın günü, ay, çeyrek
- **Yüzde Değişim:** Günlük kur değişim oranı
- **Makroekonomik Göstergeler:** Faiz, enflasyon, repo, FX hacimleri

> **Leakage Önlemi:** Lag ve rolling hesaplamalarında `shift(1)` kullanılarak
> gelecek bilgisinin modele sızması engellenmiştir.

## 4. Modelleme Yaklaşımı

### Veri Bölünmesi
- **Kronolojik split:** %80 eğitim / %20 test (shuffle yok)
- **Cross-validation:** TimeSeriesSplit (5 katlama)

### Regresyon Modelleri

"""
    # Regresyon tablo
    rapor += "| Model | MAE | RMSE | MAPE (%) |\n"
    rapor += "|-------|-----|------|----------|\n"
    for model_adi in ["Naive_Baseline", "Ridge", "RandomForest_Reg"]:
        if model_adi in reg_sonuc:
            m = reg_sonuc[model_adi]
            rapor += f"| {model_adi} | {m['MAE']:.4f} | {m['RMSE']:.4f} | {m['MAPE']:.2f} |\n"

    rapor += f"\n**En iyi regresyon modeli:** {birlesik['en_iyi_regresyon']}\n\n"

    rapor += "### Sınıflandırma Modelleri\n\n"
    rapor += "| Model | F1 | ROC-AUC | Precision | Recall |\n"
    rapor += "|-------|----|---------|-----------|---------|\n"
    for model_adi in ["Baseline_0", "LogisticRegression", "RandomForest_Cls"]:
        if model_adi in cls_sonuc:
            m = cls_sonuc[model_adi]
            rapor += (f"| {model_adi} | {m['F1']:.4f} | {m['ROC_AUC']:.4f} | "
                      f"{m['Precision']:.4f} | {m['Recall']:.4f} |\n")

    rapor += f"\n**En iyi sınıflandırma modeli:** {birlesik['en_iyi_siniflandirma']}\n\n"

    rapor += """## 5. Görselleştirmeler

Aşağıdaki grafikler `reports/figures/` klasöründe mevcuttur:

1. **Zaman Serisi:** USD/TRY kurunun eğitim/test gösterimi
2. **Tahmin vs Gerçek:** Regresyon modellerinin test seti performansı
3. **Hata Dağılımı:** Residual histogramı
4. **Karmaşıklık Matrisi:** Sınıflandırma modelleri confusion matrix
5. **Özellik Önemi:** En etkili özellikler

![Zaman Serisi](figures/1_zaman_serisi.png)
![Tahmin vs Gerçek](figures/2_tahmin_vs_gercek.png)
![Hata Dağılımı](figures/3_residual_dagilimi.png)
![Confusion Matrix](figures/4_confusion_matrix.png)
![Özellik Önemi](figures/5_feature_importance.png)

## 6. Sonuç ve Yorum

### Regresyon
- Naive baseline (yarın = bugün) güçlü bir baseline'dır çünkü döviz kurları genellikle
  güçlü otokorelasyon gösterir.
- Makine öğrenmesi modelleri lag ve rolling özellikleriyle bu baseline'ı iyileştirmeyi
  amaçlamaktadır.

### Sınıflandırma
- Kur sıçramaları nadir olaylar olduğundan sınıf dengesizliği mevcuttur.
- `class_weight="balanced"` parametresi ile dengesizlik ele alınmıştır.

## 7. Sınırlılıklar

1. **Veri boyutu:** ~500-700 iş günü sınırlı bir eğitim seti sunmaktadır
2. **Dış faktörler:** Siyasi olaylar, küresel piyasa şokları modele dahil değildir
3. **Frekans uyumsuzluğu:** Aylık/haftalık veriler forward-fill ile günlüğe çevrilmiş
   olup bilgi kaybı olabilir
4. **Hedef tanımı:** %2.0 sıçrama eşiği sübjektiven belirlenmiştir

## 8. Geliştirme Önerileri

1. **Daha fazla veri:** Daha uzun zaman aralığı ile eğitim
2. **Dışsal değişkenler:** S&P 500, petrol fiyatı, VIX gibi küresel göstergeler
3. **Gelişmiş modeller:** XGBoost, LightGBM veya LSTM gibi derin öğrenme modelleri
4. **Hiperparametre optimizasyonu:** GridSearchCV veya Optuna ile sistematik arama
5. **Ensemble yöntemler:** Birden fazla modelin tahminlerini birleştirme
6. **Olay bazlı özellikler:** TCMB faiz kararı tarihleri gibi kategorik değişkenler
"""

    rapor_yolu = RAPORLAR / "report.md"
    with open(rapor_yolu, "w", encoding="utf-8") as f:
        f.write(rapor)
    logger.info(f"Rapor kaydedildi: {rapor_yolu}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Değerlendirme ve görselleştirme")
    parser.parse_args()

    ensure_dirs()

    # Sonuç dosyalarını yükle
    reg_yol = RAPORLAR / "regression_results.json"
    cls_yol = RAPORLAR / "classification_results.json"

    if not reg_yol.exists():
        raise FileNotFoundError(
            f"{reg_yol} bulunamadı! Önce 'python -m src.train_regression' çalıştırın."
        )
    if not cls_yol.exists():
        raise FileNotFoundError(
            f"{cls_yol} bulunamadı! Önce 'python -m src.train_classification' çalıştırın."
        )

    with open(reg_yol, "r", encoding="utf-8") as f:
        reg_sonuc = json.load(f)
    with open(cls_yol, "r", encoding="utf-8") as f:
        cls_sonuc = json.load(f)

    # Özellik verisi
    df = pd.read_csv(VERI_ISLEM / "features.csv", parse_dates=["Date"])
    split_idx = reg_sonuc.get("split_idx", int(len(df) * 0.8))
    ozellikler = reg_sonuc.get("ozellikler", [])

    # Tahmin verisi
    pred_reg_yol = RAPORLAR / "regression_predictions.csv"
    pred_reg = pd.read_csv(pred_reg_yol) if pred_reg_yol.exists() else None

    # ─── Grafikler ───────────────────────────────────────────────────────
    logger.info("Grafikler oluşturuluyor...")

    grafik_1_zaman_serisi(df, split_idx)

    if pred_reg is not None:
        grafik_2_tahmin_vs_gercek(pred_reg, reg_sonuc.get("en_iyi_model", "Ridge"))
        grafik_3_residual(pred_reg, reg_sonuc.get("en_iyi_model", "Ridge"))

    grafik_4_confusion_matrix(cls_sonuc)
    grafik_5_feature_importance(ozellikler)

    # ─── Birleşik metrikler ──────────────────────────────────────────────
    birlesik = birlestir_metrikler(reg_sonuc, cls_sonuc)

    metrik_yolu = METRIKLER / "metrics.json"
    with open(metrik_yolu, "w", encoding="utf-8") as f:
        json.dump(birlesik, f, ensure_ascii=False, indent=2)
    logger.info(f"Birleşik metrikler kaydedildi: {metrik_yolu}")

    # Ayrıca reports altına da kopyala
    with open(RAPORLAR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(birlesik, f, ensure_ascii=False, indent=2)

    # ─── Rapor ───────────────────────────────────────────────────────────
    rapor_olustur(birlesik, reg_sonuc, cls_sonuc)

    logger.info("=" * 50)
    logger.info("Değerlendirme tamamlandı!")
    logger.info(f"Grafikler: {GRAFIKLER}")
    logger.info(f"Metrikler: {metrik_yolu}")
    logger.info(f"Rapor: {RAPORLAR / 'report.md'}")


if __name__ == "__main__":
    main()
