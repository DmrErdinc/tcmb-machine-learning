"""
Sınıflandırma modelleri eğitim modülü.
LogisticRegression ve RandomForestClassifier modellerini eğitir,
baseline ile karşılaştırır. Sonuçları JSON olarak kaydeder.
"""

import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
from sklearn.preprocessing import StandardScaler

from src.utils import (
    SEED, TEST_ORANI, CV_SPLITS, SICRAMA_ESIK,
    VERI_ISLEM, MODELLER, RAPORLAR, ensure_dirs, setup_logging,
)

warnings.filterwarnings("ignore")
logger = setup_logging("train_classification")


def veri_yukle() -> tuple[pd.DataFrame, list[str]]:
    """Özellik verisini yükler."""
    yol = VERI_ISLEM / "features.csv"
    if not yol.exists():
        raise FileNotFoundError(
            f"{yol} bulunamadı! Önce 'python -m src.features' çalıştırın."
        )

    df = pd.read_csv(yol, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    haric_sutunlar = {"Date", "target_reg", "target_cls"}
    ozellik_sutunlar = [c for c in df.columns
                        if c not in haric_sutunlar and df[c].dtype in ["float64", "int64", "int32"]]

    logger.info(f"Özellik sayısı: {len(ozellik_sutunlar)}")
    logger.info(f"Sınıf dağılımı:\n{df['target_cls'].value_counts()}")
    return df, ozellik_sutunlar


def kronolojik_split(df: pd.DataFrame, ozellikler: list[str]) -> tuple:
    """Kronolojik train/test split."""
    n = len(df)
    split_idx = int(n * (1 - TEST_ORANI))

    X_train = df.iloc[:split_idx][ozellikler].values
    X_test = df.iloc[split_idx:][ozellikler].values
    y_train = df.iloc[:split_idx]["target_cls"].values
    y_test = df.iloc[split_idx:]["target_cls"].values
    tarihler_test = df.iloc[split_idx:]["Date"].values

    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    logger.info(f"Train sınıf dağılımı: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    logger.info(f"Test sınıf dağılımı: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    return X_train, X_test, y_train, y_test, tarihler_test, split_idx


def sinif_metrikleri(y_gercek: np.ndarray, y_tahmin: np.ndarray,
                     y_proba: np.ndarray = None) -> dict:
    """Sınıflandırma metriklerini hesaplar."""
    metrikler = {
        "F1": float(f1_score(y_gercek, y_tahmin, zero_division=0)),
        "Precision": float(precision_score(y_gercek, y_tahmin, zero_division=0)),
        "Recall": float(recall_score(y_gercek, y_tahmin, zero_division=0)),
        "Confusion_Matrix": confusion_matrix(y_gercek, y_tahmin).tolist(),
    }

    # ROC-AUC: sadece her iki sınıf mevcutsa hesapla
    if y_proba is not None and len(np.unique(y_gercek)) > 1:
        metrikler["ROC_AUC"] = float(roc_auc_score(y_gercek, y_proba))
    else:
        metrikler["ROC_AUC"] = 0.0

    return metrikler


def cv_skorlar(model, X_train: np.ndarray, y_train: np.ndarray,
               scaler: StandardScaler = None) -> dict:
    """TimeSeriesSplit ile CV skorları."""
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    f1_list = []

    for train_idx, val_idx in tscv.split(X_train):
        X_t, X_v = X_train[train_idx], X_train[val_idx]
        y_t, y_v = y_train[train_idx], y_train[val_idx]

        if scaler:
            X_t = scaler.fit_transform(X_t)
            X_v = scaler.transform(X_v)

        model.fit(X_t, y_t)
        y_pred = model.predict(X_v)
        f1_list.append(f1_score(y_v, y_pred, zero_division=0))

    return {
        "cv_f1_mean": float(np.mean(f1_list)),
        "cv_f1_std": float(np.std(f1_list)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sınıflandırma modeli eğitimi")
    parser.add_argument("--esik", type=float, default=SICRAMA_ESIK,
                        help=f"Sıçrama eşiği (yüzde, default={SICRAMA_ESIK})")
    parser.parse_args()

    ensure_dirs()
    np.random.seed(SEED)

    df, ozellikler = veri_yukle()
    X_train, X_test, y_train, y_test, tarihler_test, split_idx = kronolojik_split(df, ozellikler)

    sonuclar: dict = {}

    # ─── Baseline: Her zaman 0 (sıçrama yok) ────────────────────────────
    logger.info("=" * 50)
    logger.info("Baseline: Her zaman 0 tahmin")
    y_baseline = np.zeros_like(y_test)
    sonuclar["Baseline_0"] = sinif_metrikleri(y_test, y_baseline)
    logger.info(f"  F1: {sonuclar['Baseline_0']['F1']:.4f}")

    # ─── Model 1: Logistic Regression ────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Model 1: LogisticRegression")
    scaler = StandardScaler()

    # CV
    cv_lr = cv_skorlar(
        LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced"),
        X_train, y_train, StandardScaler()
    )
    logger.info(f"  CV F1: {cv_lr['cv_f1_mean']:.4f} ± {cv_lr['cv_f1_std']:.4f}")

    # Final eğitim
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1] if len(np.unique(y_train)) > 1 else None

    sonuclar["LogisticRegression"] = {
        **sinif_metrikleri(y_test, y_pred_lr, y_proba_lr),
        **cv_lr,
    }
    logger.info(f"  Test F1:      {sonuclar['LogisticRegression']['F1']:.4f}")
    logger.info(f"  Test ROC-AUC: {sonuclar['LogisticRegression']['ROC_AUC']:.4f}")
    logger.info(f"  Test Prec:    {sonuclar['LogisticRegression']['Precision']:.4f}")
    logger.info(f"  Test Recall:  {sonuclar['LogisticRegression']['Recall']:.4f}")

    # ─── Model 2: Random Forest Classifier ───────────────────────────────
    logger.info("=" * 50)
    logger.info("Model 2: RandomForestClassifier")

    rf_cls = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    )

    # CV
    cv_rf = cv_skorlar(
        RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5,
                               class_weight="balanced", random_state=SEED, n_jobs=-1),
        X_train, y_train,
    )
    logger.info(f"  CV F1: {cv_rf['cv_f1_mean']:.4f} ± {cv_rf['cv_f1_std']:.4f}")

    # Final eğitim
    rf_cls.fit(X_train, y_train)
    y_pred_rf = rf_cls.predict(X_test)
    y_proba_rf = rf_cls.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) > 1 else None

    sonuclar["RandomForest_Cls"] = {
        **sinif_metrikleri(y_test, y_pred_rf, y_proba_rf),
        **cv_rf,
    }
    logger.info(f"  Test F1:      {sonuclar['RandomForest_Cls']['F1']:.4f}")
    logger.info(f"  Test ROC-AUC: {sonuclar['RandomForest_Cls']['ROC_AUC']:.4f}")
    logger.info(f"  Test Prec:    {sonuclar['RandomForest_Cls']['Precision']:.4f}")
    logger.info(f"  Test Recall:  {sonuclar['RandomForest_Cls']['Recall']:.4f}")

    # ─── En iyi modeli kaydet ────────────────────────────────────────────
    en_iyi_isim = max(
        ["LogisticRegression", "RandomForest_Cls"],
        key=lambda m: sonuclar[m]["F1"]
    )
    logger.info(f"\nEn iyi sınıflandırma modeli: {en_iyi_isim}")

    if en_iyi_isim == "LogisticRegression":
        joblib.dump({"model": lr, "scaler": scaler}, MODELLER / "best_classification_model.joblib")
    else:
        joblib.dump({"model": rf_cls, "scaler": None}, MODELLER / "best_classification_model.joblib")

    # Tahminleri kaydet
    tahmin_df = pd.DataFrame({
        "Date": tarihler_test,
        "Gercek": y_test,
        "Baseline": y_baseline,
        "LogisticRegression": y_pred_lr,
        "RandomForest": y_pred_rf,
    })
    tahmin_df.to_csv(RAPORLAR / "classification_predictions.csv", index=False)

    # Sonuçları kaydet
    sonuclar["en_iyi_model"] = en_iyi_isim
    sonuclar["split_idx"] = split_idx
    sonuclar["ozellikler"] = ozellikler
    sonuclar["sicrama_esik"] = SICRAMA_ESIK

    with open(RAPORLAR / "classification_results.json", "w", encoding="utf-8") as f:
        json.dump(sonuclar, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"Sonuçlar kaydedildi: {RAPORLAR / 'classification_results.json'}")
    logger.info(f"Model kaydedildi: {MODELLER / 'best_classification_model.joblib'}")


if __name__ == "__main__":
    main()
