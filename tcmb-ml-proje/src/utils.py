"""
Yardımcı fonksiyonlar ve proje sabitleri.
Tüm modüller tarafından ortaklaşa kullanılan sabitler, yol tanımları ve
yardımcı araçlar bu dosyada tutulur.
"""

from pathlib import Path
import logging
import sys

# ─── Sabitler ────────────────────────────────────────────────────────────────
SEED: int = 42
SICRAMA_ESIK: float = 0.15       # Sınıflandırma eşiği (yüzde değişim)
TEST_ORANI: float = 0.20         # Test seti oranı
CV_SPLITS: int = 5               # TimeSeriesSplit katlama sayısı
KAGGLE_SLUG: str = "emrekaany/usd-try-conv-rates-and-related-data"

# ─── Proje Yolları ───────────────────────────────────────────────────────────
PROJE_KOK: Path = Path(__file__).resolve().parent.parent
VERI_HAM: Path = PROJE_KOK / "data" / "raw"
VERI_ISLEM: Path = PROJE_KOK / "data" / "processed"
MODELLER: Path = PROJE_KOK / "models"
RAPORLAR: Path = PROJE_KOK / "reports"
GRAFIKLER: Path = RAPORLAR / "figures"
METRIKLER: Path = PROJE_KOK / "metrics"

# Ham veri kaynak klasörü (proje dışı, dataset/ klasörü)
VERI_KAYNAK: Path = PROJE_KOK.parent / "dataset"


def ensure_dirs() -> None:
    """Gerekli proje klasörlerini oluşturur."""
    for d in [VERI_HAM, VERI_ISLEM, MODELLER, RAPORLAR, GRAFIKLER, METRIKLER]:
        d.mkdir(parents=True, exist_ok=True)


def setup_logging(name: str = "tcmb_ml") -> logging.Logger:
    """Standart logging yapılandırması döndürür."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
