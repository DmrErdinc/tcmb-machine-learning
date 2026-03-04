"""
Regresyon modelleri eğitim modülü.
Ridge Regression ve RandomForestRegressor modellerini eğitir,
naive baseline ile karşılaştırır. Sonuçları JSON olarak kaydeder.
"""

import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.utils import (
    SEED, TEST_ORANI, CV_SPLITS, VERI_ISLEM,
    MODELLER, RAPORLAR, ensure_dirs, setup_logging,
)

warnings.filterwarnings("ignore")
logger = setup_logging("train_regression")


def mape_hesapla(y_gercek: np.ndarray, y_tahmin: np.ndarray) -> float:
    """Mean Absolute Percentage Error hesaplar."""
    mask = y_gercek != 0
    return float(np.mean(np.abs((y_gercek[mask] - y_tahmin[mask]) / y_gercek[mask])) * 100)


def veri_yukle() -> tuple[pd.DataFrame, list[str]]:
    """Özellik verisini yükler ve özellik sütunlarını belirler."""
    yol = VERI_ISLEM / "features.csv"
    if not yol.exists():
        raise FileNotFoundError(
            f"{yol} bulunamadı! Önce 'python -m src.features' çalıştırın."
        )

    df = pd.read_csv(yol, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Hedef ve tarih dışındaki tüm sayısal sütunlar özellik
    haric_sutunlar = {"Date", "target_reg", "target_cls"}
    ozellik_sutunlar = [c for c in df.columns
                        if c not in haric_sutunlar and df[c].dtype in ["float64", "int64", "int32"]]

    logger.info(f"Özellik sayısı: {len(ozellik_sutunlar)}")
    return df, ozellik_sutunlar


def kronolojik_split(df: pd.DataFrame, ozellikler: list[str],
                     hedef: str = "target_reg") -> tuple:
    """Kronolojik train/test split yapar (shuffle yok)."""
    n = len(df)
    split_idx = int(n * (1 - TEST_ORANI))

    X_train = df.iloc[:split_idx][ozellikler].values
    X_test = df.iloc[split_idx:][ozellikler].values
    y_train = df.iloc[:split_idx][hedef].values
    y_test = df.iloc[split_idx:][hedef].values
    tarihler_test = df.iloc[split_idx:]["Date"].values

    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)} | "
                f"Split tarihi: {df.iloc[split_idx]['Date'].date()}")

    return X_train, X_test, y_train, y_test, tarihler_test, split_idx


def cv_skorlar(model, X_train: np.ndarray, y_train: np.ndarray,
               scaler: StandardScaler = None) -> dict:
    """TimeSeriesSplit ile cross-validation skorları hesaplar."""
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    mae_list, rmse_list = [], []

    for train_idx, val_idx in tscv.split(X_train):
        X_t, X_v = X_train[train_idx], X_train[val_idx]
        y_t, y_v = y_train[train_idx], y_train[val_idx]

        if scaler:
            X_t = scaler.fit_transform(X_t)
            X_v = scaler.transform(X_v)

        model.fit(X_t, y_t)
        y_pred = model.predict(X_v)

        mae_list.append(mean_absolute_error(y_v, y_pred))
        rmse_list.append(root_mean_squared_error(y_v, y_pred))

    return {
        "cv_mae_mean": float(np.mean(mae_list)),
        "cv_mae_std": float(np.std(mae_list)),
        "cv_rmse_mean": float(np.mean(rmse_list)),
        "cv_rmse_std": float(np.std(rmse_list)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Regresyon modeli eğitimi")
    parser.parse_args()

    ensure_dirs()
    np.random.seed(SEED)

    df, ozellikler = veri_yukle()
    X_train, X_test, y_train, y_test, tarihler_test, split_idx = kronolojik_split(df, ozellikler)

    sonuclar: dict = {}

    # ─── Naive Baseline ──────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Naive Baseline (yarın = bugün)")
    # Naive: tahmin = conversion_rate (bugünkü kur)
    y_naive = df.iloc[split_idx:]["Conversion_Rate"].values
    sonuclar["Naive_Baseline"] = {
        "MAE": float(mean_absolute_error(y_test, y_naive)),
        "RMSE": float(root_mean_squared_error(y_test, y_naive)),
        "MAPE": mape_hesapla(y_test, y_naive),
    }
    logger.info(f"  MAE:  {sonuclar['Naive_Baseline']['MAE']:.4f}")
    logger.info(f"  RMSE: {sonuclar['Naive_Baseline']['RMSE']:.4f}")
    logger.info(f"  MAPE: {sonuclar['Naive_Baseline']['MAPE']:.4f}%")

    # ─── Model 1: Ridge Regression ───────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Model 1: Ridge Regression")
    scaler = StandardScaler()
    ridge = Ridge(alpha=1.0, random_state=SEED)

    # CV
    cv_ridge = cv_skorlar(Ridge(alpha=1.0, random_state=SEED), X_train, y_train, StandardScaler())
    logger.info(f"  CV MAE: {cv_ridge['cv_mae_mean']:.4f} ± {cv_ridge['cv_mae_std']:.4f}")

    # Final eğitim
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)

    sonuclar["Ridge"] = {
        "MAE": float(mean_absolute_error(y_test, y_pred_ridge)),
        "RMSE": float(root_mean_squared_error(y_test, y_pred_ridge)),
        "MAPE": mape_hesapla(y_test, y_pred_ridge),
        **cv_ridge,
    }
    logger.info(f"  Test MAE:  {sonuclar['Ridge']['MAE']:.4f}")
    logger.info(f"  Test RMSE: {sonuclar['Ridge']['RMSE']:.4f}")
    logger.info(f"  Test MAPE: {sonuclar['Ridge']['MAPE']:.4f}%")

    # ─── Model 2: Random Forest Regressor ────────────────────────────────
    logger.info("=" * 50)
    logger.info("Model 2: RandomForestRegressor")
    rf_reg = RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        random_state=SEED, n_jobs=-1,
    )

    # CV
    cv_rf = cv_skorlar(
        RandomForestRegressor(n_estimators=100, max_depth=10,
                              min_samples_leaf=5, random_state=SEED, n_jobs=-1),
        X_train, y_train
    )
    logger.info(f"  CV MAE: {cv_rf['cv_mae_mean']:.4f} ± {cv_rf['cv_mae_std']:.4f}")

    # Final eğitim
    rf_reg.fit(X_train, y_train)
    y_pred_rf = rf_reg.predict(X_test)

    sonuclar["RandomForest_Reg"] = {
        "MAE": float(mean_absolute_error(y_test, y_pred_rf)),
        "RMSE": float(root_mean_squared_error(y_test, y_pred_rf)),
        "MAPE": mape_hesapla(y_test, y_pred_rf),
        **cv_rf,
    }
    logger.info(f"  Test MAE:  {sonuclar['RandomForest_Reg']['MAE']:.4f}")
    logger.info(f"  Test RMSE: {sonuclar['RandomForest_Reg']['RMSE']:.4f}")
    logger.info(f"  Test MAPE: {sonuclar['RandomForest_Reg']['MAPE']:.4f}%")

    # ─── En iyi modeli kaydet ────────────────────────────────────────────
    en_iyi_isim = min(
        ["Ridge", "RandomForest_Reg"],
        key=lambda m: sonuclar[m]["MAE"]
    )
    logger.info(f"\nEn iyi regresyon modeli: {en_iyi_isim}")

    if en_iyi_isim == "Ridge":
        joblib.dump({"model": ridge, "scaler": scaler}, MODELLER / "best_regression_model.joblib")
    else:
        joblib.dump({"model": rf_reg, "scaler": None}, MODELLER / "best_regression_model.joblib")

    # Tahminleri kaydet (grafik için)
    tahmin_df = pd.DataFrame({
        "Date": tarihler_test,
        "Gercek": y_test,
        "Naive": y_naive,
        "Ridge": y_pred_ridge,
        "RandomForest": y_pred_rf,
    })
    tahmin_df.to_csv(RAPORLAR / "regression_predictions.csv", index=False)

    # Sonuçları kaydet
    sonuclar["en_iyi_model"] = en_iyi_isim
    sonuclar["split_idx"] = split_idx
    sonuclar["ozellikler"] = ozellikler

    with open(RAPORLAR / "regression_results.json", "w", encoding="utf-8") as f:
        json.dump(sonuclar, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"Sonuçlar kaydedildi: {RAPORLAR / 'regression_results.json'}")
    logger.info(f"Model kaydedildi: {MODELLER / 'best_regression_model.joblib'}")


if __name__ == "__main__":
    main()
