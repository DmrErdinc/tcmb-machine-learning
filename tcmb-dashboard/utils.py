"""
Dashboard yardımcı fonksiyonları.
ML proje çıktılarını bulan, yükleyen ve demo veri üreten araçlar.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ─── Proje yolları ───────────────────────────────────────────────────────────
DASHBOARD_KOK = Path(__file__).resolve().parent
ML_PROJE = DASHBOARD_KOK.parent / "tcmb-ml-proje"

REPORTS_DIR = ML_PROJE / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = ML_PROJE / "metrics"
DATA_DIR = ML_PROJE / "data" / "processed"
MODELS_DIR = ML_PROJE / "models"


def proje_mevcut() -> bool:
    """ML proje klasörünün varlığını kontrol eder."""
    return ML_PROJE.exists() and (REPORTS_DIR / "report.md").exists()


def dosya_oku(yol: Path, varsayilan: Any = None) -> Any:
    """Dosyayı güvenli şekilde okur."""
    if yol.exists():
        return yol.read_text(encoding="utf-8")
    return varsayilan


def json_oku(yol: Path) -> dict:
    """JSON dosyasını güvenli şekilde okur."""
    if yol.exists():
        with open(yol, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def csv_oku(yol: Path) -> pd.DataFrame | None:
    """CSV dosyasını güvenli şekilde okur."""
    if yol.exists():
        return pd.read_csv(yol, parse_dates=["Date"] if "Date" in pd.read_csv(yol, nrows=0).columns else None)
    return None


def grafik_yolu(isim: str) -> Path | None:
    """Grafik dosyasının yolunu döndürür."""
    yol = FIGURES_DIR / isim
    return yol if yol.exists() else None


def tum_grafikleri_bul() -> dict[str, Path]:
    """Mevcut tüm grafikleri bulur."""
    if not FIGURES_DIR.exists():
        return {}
    return {f.stem: f for f in sorted(FIGURES_DIR.glob("*.png"))}


def metrikleri_yukle() -> dict:
    """Metrik verilerini yükler (JSON veya CSV)."""
    # JSON dene
    json_yol = METRICS_DIR / "metrics.json"
    if json_yol.exists():
        return json_oku(json_yol)

    # reports altındaki JSON dene
    json_yol2 = REPORTS_DIR / "metrics.json"
    if json_yol2.exists():
        return json_oku(json_yol2)

    return {}


def regresyon_sonuclari() -> dict:
    """Regresyon sonuçlarını yükler."""
    return json_oku(REPORTS_DIR / "regression_results.json")


def siniflandirma_sonuclari() -> dict:
    """Sınıflandırma sonuçlarını yükler."""
    return json_oku(REPORTS_DIR / "classification_results.json")


def regresyon_tahminleri() -> pd.DataFrame | None:
    """Regresyon tahminlerini yükler."""
    return csv_oku(REPORTS_DIR / "regression_predictions.csv")


def siniflandirma_tahminleri() -> pd.DataFrame | None:
    """Sınıflandırma tahminlerini yükler."""
    return csv_oku(REPORTS_DIR / "classification_predictions.csv")


def features_veri() -> pd.DataFrame | None:
    """Özellik verisini yükler."""
    return csv_oku(DATA_DIR / "features.csv")


def merged_veri() -> pd.DataFrame | None:
    """Birleştirilmiş ham veriyi yükler."""
    return csv_oku(DATA_DIR / "merged_data.csv")


def rapor_oku() -> str:
    """report.md içeriğini okur."""
    icerik = dosya_oku(REPORTS_DIR / "report.md")
    return icerik if icerik else "# Rapor bulunamadı\n\nRapor dosyası mevcut değil."


# ─── Demo veri üreticiler ────────────────────────────────────────────────────

def demo_metrikler() -> dict:
    """Demo metrik verisi üretir."""
    return {
        "regresyon": {
            "Naive_Baseline": {"MAE": 0.0279, "RMSE": 0.0403, "MAPE": 0.0661},
            "Ridge": {"MAE": 0.0221, "RMSE": 0.0287, "MAPE": 0.0524},
            "RandomForest_Reg": {"MAE": 1.647, "RMSE": 1.85, "MAPE": 3.862},
        },
        "siniflandirma": {
            "Baseline_0": {"F1": 0.0, "ROC_AUC": 0.0, "Precision": 0.0, "Recall": 0.0},
            "LogisticRegression": {"F1": 0.29, "ROC_AUC": 0.92, "Precision": 0.80, "Recall": 0.17},
            "RandomForest_Cls": {"F1": 0.92, "ROC_AUC": 0.98, "Precision": 0.85, "Recall": 1.0},
        },
        "en_iyi_regresyon": "Ridge",
        "en_iyi_siniflandirma": "RandomForest_Cls",
    }


def demo_zaman_serisi() -> pd.DataFrame:
    """Demo zaman serisi verisi üretir."""
    np.random.seed(42)
    tarihler = pd.bdate_range("2023-06-01", periods=500)
    kur = 26.0 + np.cumsum(np.random.randn(500) * 0.05)
    return pd.DataFrame({"Date": tarihler, "Conversion_Rate": kur})
