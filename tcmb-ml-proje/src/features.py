"""
Özellik mühendisliği modülü.
Birleştirilmiş veriden lag, rolling, takvim ve hedef değişken özelliklerini
oluşturur. Leakage olmamasına dikkat edilir.
"""

import argparse
import pandas as pd
import numpy as np

from src.utils import VERI_ISLEM, SICRAMA_ESIK, ensure_dirs, setup_logging

logger = setup_logging("features")

# ─── Lag ve Rolling parametreleri ────────────────────────────────────────────
LAG_GUNLER: list[int] = [1, 2, 5, 10]
ROLLING_PENCERELER: list[int] = [5, 20]


def lag_olustur(df: pd.DataFrame, sutun: str = "Conversion_Rate") -> pd.DataFrame:
    """Lag özellikleri oluşturur."""
    for lag in LAG_GUNLER:
        df[f"lag_{lag}"] = df[sutun].shift(lag)
    logger.info(f"Lag özellikleri oluşturuldu: {LAG_GUNLER}")
    return df


def rolling_olustur(df: pd.DataFrame, sutun: str = "Conversion_Rate") -> pd.DataFrame:
    """Rolling ortalama ve standart sapma oluşturur."""
    for pencere in ROLLING_PENCERELER:
        df[f"rolling_mean_{pencere}"] = df[sutun].shift(1).rolling(window=pencere).mean()
        df[f"rolling_std_{pencere}"] = df[sutun].shift(1).rolling(window=pencere).std()
    logger.info(f"Rolling özellikleri oluşturuldu: pencereler={ROLLING_PENCERELER}")
    return df


def takvim_olustur(df: pd.DataFrame) -> pd.DataFrame:
    """Takvim özellikleri oluşturur."""
    df["haftanin_gunu"] = df["Date"].dt.dayofweek      # 0=Pazartesi, 4=Cuma
    df["ay"] = df["Date"].dt.month
    df["ceyrek"] = df["Date"].dt.quarter
    logger.info("Takvim özellikleri oluşturuldu: haftanin_gunu, ay, ceyrek")
    return df


def yuzde_degisim_olustur(df: pd.DataFrame, sutun: str = "Conversion_Rate") -> pd.DataFrame:
    """Günlük yüzde değişim hesaplar."""
    df["yuzde_degisim"] = df[sutun].pct_change() * 100
    logger.info("Yüzde değişim özelliği oluşturuldu")
    return df


def hedef_olustur(df: pd.DataFrame, sutun: str = "Conversion_Rate",
                  esik: float = SICRAMA_ESIK) -> pd.DataFrame:
    """
    Hedef değişkenleri oluşturur:
    - target_reg: ertesi günün kuru (regresyon)
    - target_cls: kur sıçraması var mı (sınıflandırma)
    """
    # Regresyon hedefi: ertesi gün kur
    df["target_reg"] = df[sutun].shift(-1)

    # Sınıflandırma hedefi: ertesi gün yüzde değişim > esik mi
    ertesi_gun_degisim = ((df[sutun].shift(-1) - df[sutun]) / df[sutun]) * 100
    df["target_cls"] = (ertesi_gun_degisim.abs() > esik).astype(int)

    logger.info(f"Hedef değişkenler oluşturuldu (sıçrama eşiği: %{esik})")

    # Sıçrama dağılımını logla
    if "target_cls" in df.columns:
        dagılım = df["target_cls"].value_counts()
        logger.info(f"Sınıf dağılımı:\n{dagılım}")

    return df


def ozellik_muhendisligi(esik: float = SICRAMA_ESIK) -> pd.DataFrame:
    """Tüm özellik mühendisliği adımlarını uygular."""

    # Birleştirilmiş veriyi oku
    girdi_yolu = VERI_ISLEM / "merged_data.csv"
    if not girdi_yolu.exists():
        raise FileNotFoundError(
            f"{girdi_yolu} bulunamadı! Önce 'python -m src.preprocess' çalıştırın."
        )

    df = pd.read_csv(girdi_yolu, parse_dates=["Date"])
    logger.info(f"Girdi verisi: {df.shape[0]} satır, {df.shape[1]} sütun")

    # Sıralama
    df = df.sort_values("Date").reset_index(drop=True)

    # Özellikler
    df = yuzde_degisim_olustur(df)
    df = lag_olustur(df)
    df = rolling_olustur(df)
    df = takvim_olustur(df)
    df = hedef_olustur(df, esik=esik)

    # Hedefi olmayan son satırı kaldır
    df = df.dropna(subset=["target_reg"])

    # Lag/rolling nedeniyle ilk satırları kaldır
    df = df.dropna().reset_index(drop=True)

    logger.info(f"Son veri: {df.shape[0]} satır, {df.shape[1]} sütun")
    logger.info(f"Sütunlar: {list(df.columns)}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Özellik mühendisliği")
    parser.add_argument("--esik", type=float, default=SICRAMA_ESIK,
                        help=f"Sıçrama eşiği (yüzde, default={SICRAMA_ESIK})")
    args = parser.parse_args()

    ensure_dirs()

    df = ozellik_muhendisligi(esik=args.esik)

    # Kaydet
    cikti_yolu = VERI_ISLEM / "features.csv"
    df.to_csv(cikti_yolu, index=False)
    logger.info(f"Kaydedildi: {cikti_yolu}")


if __name__ == "__main__":
    main()
