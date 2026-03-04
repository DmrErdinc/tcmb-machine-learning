"""
Veri ön işleme modülü.
Ham CSV dosyalarını okur, tarih formatlarını düzenler, farklı frekanstaki
verileri USD/TRY günlük serisine birleştirir ve temizlenmiş veriyi kaydeder.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils import VERI_HAM, VERI_ISLEM, ensure_dirs, setup_logging

logger = setup_logging("preprocess")

# ─── Dosya tanımları ─────────────────────────────────────────────────────────
GUNLUK_DOSYALAR: dict[str, str] = {
    "USD_TRY_CONVERSION_RATE.csv": "Conversion_Rate",
    "FX_Swap_Deposit_Amount.csv": None,
    "FX_TRANSACTION_VOLUME.csv": None,
    "TCMB_Net_Funding.csv": None,
}

# FX_TRANSACTION_VOLUME ve TCMB_Net_Funding da günlük ama
# yyyy-mm-dd formatında — otomatik algılama ile parse edilecek

HAFTALIK_DOSYALAR: dict[str, str] = {
    "TL_INTEREST_RATE.csv": "TRY_Interest_Rate_6Month",
    "USD_INTEREST_RATE.csv": None,
}

AYLIK_DOSYALAR: dict[str, str] = {
    "CPI_General_Index.csv": "CPI_Index",
    "Inflation_Expectation_12M.csv": "Inflation_Expectation_12M",
    "Repo_1Day_Weighted_Average_Rate.csv": "Repo_1Day_Weighted_Average_Rate",
}


def parse_gunluk_tarih_auto(seri: pd.Series) -> pd.Series:
    """Otomatik tarih formatı algılama: dd-mm-yyyy veya yyyy-mm-dd."""
    # Önce dd-mm-yyyy dene
    result = pd.to_datetime(seri, format="%d-%m-%Y", errors="coerce")
    # Çoğu NaT ise yyyy-mm-dd dene
    if result.isna().sum() > len(result) * 0.5:
        result = pd.to_datetime(seri, format="%Y-%m-%d", errors="coerce")
    # Hala NaT ise genel parse dene
    if result.isna().sum() > len(result) * 0.5:
        result = pd.to_datetime(seri, errors="coerce")
    return result


def parse_aylik_tarih(tarih_str: str) -> pd.Timestamp:
    """YYYY-M formatını parse eder (ayın ilk günü)."""
    try:
        parcalar = str(tarih_str).split("-")
        yil = int(parcalar[0])
        ay = int(parcalar[1])
        return pd.Timestamp(year=yil, month=ay, day=1)
    except (ValueError, IndexError):
        return pd.NaT


def oku_gunluk(dosya_adi: str) -> pd.DataFrame:
    """Günlük frekanslı CSV okur (dd-mm-yyyy veya yyyy-mm-dd)."""
    yol = VERI_HAM / dosya_adi
    if not yol.exists():
        logger.warning(f"Dosya bulunamadı: {yol}")
        return pd.DataFrame()

    df = pd.read_csv(yol)
    df["Date"] = parse_gunluk_tarih_auto(df["Date"])
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def oku_haftalik(dosya_adi: str) -> pd.DataFrame:
    """Haftalık frekanslı CSV okur."""
    yol = VERI_HAM / dosya_adi
    if not yol.exists():
        logger.warning(f"Dosya bulunamadı: {yol}")
        return pd.DataFrame()

    df = pd.read_csv(yol)
    df["Date"] = parse_gunluk_tarih_auto(df["Date"])
    df = df.dropna(subset=["Date"])
    # Year_Week sütununu kaldır
    if "Year_Week" in df.columns:
        df = df.drop(columns=["Year_Week"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def oku_aylik(dosya_adi: str) -> pd.DataFrame:
    """Aylık frekanslı CSV okur (YYYY-M formatı)."""
    yol = VERI_HAM / dosya_adi
    if not yol.exists():
        logger.warning(f"Dosya bulunamadı: {yol}")
        return pd.DataFrame()

    df = pd.read_csv(yol)
    df["Date"] = df["Date"].apply(parse_aylik_tarih)
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def birlestir_veri() -> pd.DataFrame:
    """Tüm veri setlerini USD/TRY günlük serisine birleştirir."""

    # Ana seri: USD/TRY dönüşüm kuru
    ana_df = oku_gunluk("USD_TRY_CONVERSION_RATE.csv")
    if ana_df.empty:
        raise FileNotFoundError("USD_TRY_CONVERSION_RATE.csv bulunamadı veya boş!")

    # Hafta sonlarını (NaN kur) filtrele
    ana_df = ana_df.dropna(subset=["Conversion_Rate"])
    ana_df = ana_df.set_index("Date")
    logger.info(f"Ana seri: {len(ana_df)} iş günü, "
                f"{ana_df.index.min().date()} - {ana_df.index.max().date()}")

    # Diğer günlük serileri birleştir
    for dosya_adi, _ in GUNLUK_DOSYALAR.items():
        if dosya_adi == "USD_TRY_CONVERSION_RATE.csv":
            continue
        df = oku_gunluk(dosya_adi)
        if df.empty:
            continue
        df = df.set_index("Date")
        # Date dışındaki tüm sütunları al
        yeni_sutunlar = [c for c in df.columns if c not in ana_df.columns]
        if yeni_sutunlar:
            ana_df = ana_df.join(df[yeni_sutunlar], how="left")
            logger.info(f"  + {dosya_adi}: {yeni_sutunlar}")

    # Haftalık serileri birleştir (forward-fill)
    for dosya_adi, _ in HAFTALIK_DOSYALAR.items():
        df = oku_haftalik(dosya_adi)
        if df.empty:
            continue
        df = df.set_index("Date")
        yeni_sutunlar = [c for c in df.columns if c not in ana_df.columns]
        if yeni_sutunlar:
            # Reindex to daily ve forward fill
            df_reindexed = df[yeni_sutunlar].reindex(ana_df.index, method="ffill")
            ana_df = ana_df.join(df_reindexed, how="left")
            logger.info(f"  + {dosya_adi} (haftalık→günlük ffill): {yeni_sutunlar}")

    # Aylık serileri birleştir (forward-fill)
    for dosya_adi, _ in AYLIK_DOSYALAR.items():
        df = oku_aylik(dosya_adi)
        if df.empty:
            continue
        df = df.set_index("Date")
        yeni_sutunlar = [c for c in df.columns if c not in ana_df.columns]
        if yeni_sutunlar:
            df_reindexed = df[yeni_sutunlar].reindex(ana_df.index, method="ffill")
            ana_df = ana_df.join(df_reindexed, how="left")
            logger.info(f"  + {dosya_adi} (aylık→günlük ffill): {yeni_sutunlar}")

    # Tamamen boş sütunları kaldır
    bos_sutunlar = ana_df.columns[ana_df.isnull().all()].tolist()
    if bos_sutunlar:
        logger.info(f"Tamamen boş sütunlar kaldırılıyor: {bos_sutunlar}")
        ana_df = ana_df.drop(columns=bos_sutunlar)

    # Kalan eksik değerleri forward-fill
    ana_df = ana_df.ffill()

    # Hala kalan NaN'leri backward-fill
    ana_df = ana_df.bfill()

    ana_df = ana_df.reset_index()
    logger.info(f"Birleştirilmiş veri: {ana_df.shape[0]} satır, {ana_df.shape[1]} sütun")

    return ana_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Veri ön işleme")
    parser.parse_args()

    ensure_dirs()

    logger.info("Veri ön işleme başlatılıyor...")
    df = birlestir_veri()

    # Kaydet
    cikti_yolu = VERI_ISLEM / "merged_data.csv"
    df.to_csv(cikti_yolu, index=False)
    logger.info(f"Kaydedildi: {cikti_yolu}")
    logger.info(f"Sütunlar: {list(df.columns)}")
    logger.info(f"Eksik değerler:\n{df.isnull().sum()}")


if __name__ == "__main__":
    main()
