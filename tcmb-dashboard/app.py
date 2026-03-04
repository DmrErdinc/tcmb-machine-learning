"""
TCMB USD/TRY Makine Öğrenmesi Dashboard'u.
Streamlit tabanlı profesyonel arayüz.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path

from utils import (
    proje_mevcut, metrikleri_yukle, regresyon_sonuclari,
    siniflandirma_sonuclari, regresyon_tahminleri,
    siniflandirma_tahminleri, features_veri, merged_veri,
    rapor_oku, tum_grafikleri_bul, demo_metrikler, demo_zaman_serisi,
    REPORTS_DIR, METRICS_DIR,
)

# ─── Sayfa ayarları ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TCMB USD/TRY ML Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Özel CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Metrik kartları */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #667eea33;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.85rem !important;
        color: #8b95a5 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* Sidebar başlık */
    .sidebar-title {
        font-size: 1.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 8px 0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 8px 20px !important;
    }

    /* Başlık banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border-radius: 16px;
        padding: 32px;
        color: white;
        margin-bottom: 24px;
    }
    .hero-banner h1 {
        color: white !important;
        font-size: 2rem !important;
        margin-bottom: 4px !important;
    }
    .hero-banner p {
        color: #b8c0cc !important;
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-title">🏦 TCMB ML</p>', unsafe_allow_html=True)
    st.caption("USD/TRY Kur Tahmini Dashboard'u")
    st.divider()

    sayfa = st.radio(
        "📑 Sayfa Seçin",
        [
            "🏠 Genel Bakış",
            "📊 Veri İnceleme",
            "📈 Regresyon Sonuçları",
            "🎯 Sınıflandırma",
            "📝 Rapor",
            "⬇️ İndir",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # Proje durumu
    if proje_mevcut():
        st.success("✅ ML Projesi bağlı", icon="🔗")
    else:
        st.warning("⚠️ ML projesi bulunamadı — demo veri kullanılıyor", icon="⚠️")

    st.divider()
    st.caption("© 2026 TCMB ML Projesi")


# ─── Veri yükleme ────────────────────────────────────────────────────────────
@st.cache_data
def veri_yukle():
    """Tüm verileri bir kere yükler."""
    demo = not proje_mevcut()

    if demo:
        metrikler = demo_metrikler()
        reg_son = {
            "Naive_Baseline": {"MAE": 0.0279, "RMSE": 0.0403, "MAPE": 0.0661},
            "Ridge": {"MAE": 0.0221, "RMSE": 0.0287, "MAPE": 0.0524},
            "RandomForest_Reg": {"MAE": 1.647, "RMSE": 1.85, "MAPE": 3.862},
            "en_iyi_model": "Ridge",
        }
        cls_son = {
            "Baseline_0": {"F1": 0.0, "ROC_AUC": 0.0, "Precision": 0.0, "Recall": 0.0,
                           "Confusion_Matrix": [[100, 0], [20, 0]]},
            "LogisticRegression": {"F1": 0.29, "ROC_AUC": 0.92, "Precision": 0.80, "Recall": 0.17,
                                   "Confusion_Matrix": [[95, 5], [14, 6]]},
            "RandomForest_Cls": {"F1": 0.92, "ROC_AUC": 0.98, "Precision": 0.85, "Recall": 1.0,
                                 "Confusion_Matrix": [[108, 3], [0, 23]]},
            "en_iyi_model": "RandomForest_Cls",
        }
        merged = demo_zaman_serisi()
        feat = None
        reg_pred = None
        cls_pred = None
    else:
        metrikler = metrikleri_yukle()
        reg_son = regresyon_sonuclari()
        cls_son = siniflandirma_sonuclari()
        merged = merged_veri()
        feat = features_veri()
        reg_pred = regresyon_tahminleri()
        cls_pred = siniflandirma_tahminleri()

    return {
        "demo": demo,
        "metrikler": metrikler,
        "reg_son": reg_son,
        "cls_son": cls_son,
        "merged": merged,
        "feat": feat,
        "reg_pred": reg_pred,
        "cls_pred": cls_pred,
    }


veri = veri_yukle()

if veri["demo"]:
    st.warning(
        "⚠️ **ML proje dosyaları bulunamadı.** Demo verilerle çalışılıyor. "
        "Gerçek veriler için `tcmb-ml-proje/` pipeline'ını çalıştırın.",
        icon="⚠️"
    )


# ═════════════════════════════════════════════════════════════════════════════
# SAYFA 1: GENEL BAKIŞ
# ═════════════════════════════════════════════════════════════════════════════
if sayfa == "🏠 Genel Bakış":
    st.markdown("""
    <div class="hero-banner">
        <h1>🏦 TCMB USD/TRY Kur Tahmini</h1>
        <p>Makine Öğrenmesi ile döviz kuru regresyon ve sınıflandırma analizi</p>
    </div>
    """, unsafe_allow_html=True)

    # Ana metrik kartları
    col1, col2, col3, col4 = st.columns(4)
    reg = veri["reg_son"]
    cls = veri["cls_son"]

    en_iyi_reg = reg.get("en_iyi_model", "Ridge")
    en_iyi_cls = cls.get("en_iyi_model", "RandomForest_Cls")

    reg_mae = reg.get(en_iyi_reg, {}).get("MAE", 0)
    reg_mape = reg.get(en_iyi_reg, {}).get("MAPE", 0)
    cls_f1 = cls.get(en_iyi_cls, {}).get("F1", 0)
    cls_auc = cls.get(en_iyi_cls, {}).get("ROC_AUC", 0)

    col1.metric("🎯 En İyi Reg. MAE", f"{reg_mae:.4f}", help="Mean Absolute Error — düşük daha iyi")
    col2.metric("📊 En İyi Reg. MAPE", f"%{reg_mape:.3f}", help="Mean Absolute Percentage Error")
    col3.metric("🏆 En İyi Cls. F1", f"{cls_f1:.4f}", help="F1 Score — yüksek daha iyi")
    col4.metric("📈 En İyi Cls. ROC-AUC", f"{cls_auc:.4f}", help="ROC-AUC — yüksek daha iyi")

    st.divider()

    # Model özetleri
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📈 Regresyon Modelleri")
        reg_tablo = []
        for model in ["Naive_Baseline", "Ridge", "RandomForest_Reg"]:
            if model in reg:
                m = reg[model]
                reg_tablo.append({
                    "Model": model,
                    "MAE": f"{m.get('MAE', 0):.4f}",
                    "RMSE": f"{m.get('RMSE', 0):.4f}",
                    "MAPE (%)": f"{m.get('MAPE', 0):.3f}",
                })
        if reg_tablo:
            df_reg = pd.DataFrame(reg_tablo)
            st.dataframe(df_reg, use_container_width=True, hide_index=True)
        st.info(f"✅ En iyi model: **{en_iyi_reg}**")

    with c2:
        st.subheader("🎯 Sınıflandırma Modelleri")
        cls_tablo = []
        for model in ["Baseline_0", "LogisticRegression", "RandomForest_Cls"]:
            if model in cls:
                m = cls[model]
                cls_tablo.append({
                    "Model": model,
                    "F1": f"{m.get('F1', 0):.4f}",
                    "ROC-AUC": f"{m.get('ROC_AUC', 0):.4f}",
                    "Precision": f"{m.get('Precision', 0):.4f}",
                    "Recall": f"{m.get('Recall', 0):.4f}",
                })
        if cls_tablo:
            df_cls = pd.DataFrame(cls_tablo)
            st.dataframe(df_cls, use_container_width=True, hide_index=True)
        st.info(f"✅ En iyi model: **{en_iyi_cls}**")

    # Zaman serisi önizleme
    st.divider()
    st.subheader("📉 USD/TRY Kur Trendi")
    if veri["merged"] is not None:
        df_m = veri["merged"]
        fig = px.line(df_m, x="Date", y="Conversion_Rate",
                      labels={"Date": "Tarih", "Conversion_Rate": "USD/TRY"},
                      template="plotly_dark")
        fig.update_traces(line_color="#667eea", line_width=2)
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# SAYFA 2: VERİ İNCELEME
# ═════════════════════════════════════════════════════════════════════════════
elif sayfa == "📊 Veri İnceleme":
    st.header("📊 Veri İnceleme")

    tab1, tab2, tab3 = st.tabs(["📋 Tablo", "📊 Kolon Bilgileri", "📉 Zaman Serisi"])

    df_inceleme = veri["feat"] if veri["feat"] is not None else veri["merged"]

    if df_inceleme is None:
        st.info("Veri bulunamadı — demo modda çalışılıyor.")
        df_inceleme = demo_zaman_serisi()

    with tab1:
        st.subheader("Veri Tablosu")
        st.caption(f"Toplam: {len(df_inceleme)} satır × {len(df_inceleme.columns)} sütun")

        # Filtre
        goster_n = st.slider("Gösterilecek satır sayısı", 5, min(100, len(df_inceleme)), 20)
        st.dataframe(df_inceleme.head(goster_n), use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Kolon Tipleri ve Eksik Değerler")

        c1, c2 = st.columns(2)
        with c1:
            tip_df = pd.DataFrame({
                "Sütun": df_inceleme.columns,
                "Tip": df_inceleme.dtypes.astype(str).values,
                "Eksik": df_inceleme.isnull().sum().values,
                "Eksik %": (df_inceleme.isnull().sum() / len(df_inceleme) * 100).round(2).values,
            })
            st.dataframe(tip_df, use_container_width=True, hide_index=True)

        with c2:
            eksik_df = df_inceleme.isnull().sum()
            eksik_df = eksik_df[eksik_df > 0]
            if len(eksik_df) > 0:
                fig = px.bar(x=eksik_df.index, y=eksik_df.values,
                             labels={"x": "Sütun", "y": "Eksik Değer Sayısı"},
                             title="Eksik Değer Dağılımı",
                             template="plotly_dark")
                fig.update_traces(marker_color="#e74c3c")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ Hiç eksik değer yok!")

    with tab3:
        st.subheader("Zaman Serisi Grafiği")
        sayisal_kolonlar = df_inceleme.select_dtypes(include=[np.number]).columns.tolist()
        if "Date" in df_inceleme.columns and sayisal_kolonlar:
            secilen_kolon = st.selectbox("Çizilecek sütun", sayisal_kolonlar,
                                         index=sayisal_kolonlar.index("Conversion_Rate")
                                         if "Conversion_Rate" in sayisal_kolonlar else 0)
            fig = px.line(df_inceleme, x="Date", y=secilen_kolon,
                          template="plotly_dark", title=f"{secilen_kolon} Zaman Serisi")
            fig.update_traces(line_color="#2ecc71", line_width=1.5)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# SAYFA 3: REGRESYON SONUÇLARI
# ═════════════════════════════════════════════════════════════════════════════
elif sayfa == "📈 Regresyon Sonuçları":
    st.header("📈 Regresyon Sonuçları")

    reg = veri["reg_son"]
    pred = veri["reg_pred"]

    # Metrik kartları
    col1, col2, col3 = st.columns(3)
    en_iyi = reg.get("en_iyi_model", "Ridge")
    best = reg.get(en_iyi, {})
    naive = reg.get("Naive_Baseline", {})

    col1.metric("MAE (En İyi)", f"{best.get('MAE', 0):.4f}",
                delta=f"{best.get('MAE', 0) - naive.get('MAE', 0):.4f} vs Naive",
                delta_color="inverse")
    col2.metric("RMSE (En İyi)", f"{best.get('RMSE', 0):.4f}")
    col3.metric("MAPE (En İyi)", f"%{best.get('MAPE', 0):.4f}")

    st.divider()

    # Grafikler
    tab1, tab2, tab3 = st.tabs(["📉 Tahmin vs Gerçek", "📊 Hata Dağılımı", "🖼️ Proje Grafikleri"])

    with tab1:
        if pred is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(pred["Date"]), y=pred["Gercek"],
                mode="lines", name="Gerçek",
                line=dict(color="#3498db", width=2)
            ))
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(pred["Date"]), y=pred["Ridge"],
                mode="lines", name="Ridge",
                line=dict(color="#e67e22", width=1.5, dash="dash")
            ))
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(pred["Date"]), y=pred["RandomForest"],
                mode="lines", name="Random Forest",
                line=dict(color="#2ecc71", width=1.5, dash="dot")
            ))
            fig.update_layout(
                title="Gerçek vs Tahmin (Test Seti)",
                xaxis_title="Tarih", yaxis_title="USD/TRY",
                template="plotly_dark", height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            grafik = tum_grafikleri_bul().get("2_tahmin_vs_gercek")
            if grafik:
                st.image(str(grafik), caption="Tahmin vs Gerçek")
            else:
                st.info("Tahmin verisi bulunamadı.")

    with tab2:
        if pred is not None:
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=["Ridge Hata Dağılımı", "RF Hata Dağılımı"])
            ridge_err = pred["Gercek"] - pred["Ridge"]
            rf_err = pred["Gercek"] - pred["RandomForest"]

            fig.add_trace(go.Histogram(x=ridge_err, nbinsx=30, marker_color="#e67e22",
                                        name="Ridge"), row=1, col=1)
            fig.add_trace(go.Histogram(x=rf_err, nbinsx=30, marker_color="#2ecc71",
                                        name="RF"), row=1, col=2)
            fig.update_layout(template="plotly_dark", height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            grafik = tum_grafikleri_bul().get("3_residual_dagilimi")
            if grafik:
                st.image(str(grafik), caption="Residual Dağılımı")

    with tab3:
        grafikler = tum_grafikleri_bul()
        for isim, yol in grafikler.items():
            if "tahmin" in isim or "residual" in isim or "zaman" in isim:
                st.image(str(yol), caption=isim.replace("_", " ").title())

    # Model karşılaştırma tablosu
    st.divider()
    st.subheader("📊 Model Karşılaştırma")
    karsilastirma = []
    for model in ["Naive_Baseline", "Ridge", "RandomForest_Reg"]:
        if model in reg:
            m = reg[model]
            karsilastirma.append({
                "Model": "⭐ " + model if model == en_iyi else model,
                "MAE": m.get("MAE", 0),
                "RMSE": m.get("RMSE", 0),
                "MAPE (%)": m.get("MAPE", 0),
            })
    if karsilastirma:
        st.dataframe(pd.DataFrame(karsilastirma), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# SAYFA 4: SINIFLANDIRMA
# ═════════════════════════════════════════════════════════════════════════════
elif sayfa == "🎯 Sınıflandırma":
    st.header("🎯 Kur Sıçraması Sınıflandırma")

    cls = veri["cls_son"]
    en_iyi = cls.get("en_iyi_model", "RandomForest_Cls")

    # Metrik kartları
    col1, col2, col3, col4 = st.columns(4)
    best = cls.get(en_iyi, {})
    col1.metric("F1 Score", f"{best.get('F1', 0):.4f}")
    col2.metric("ROC-AUC", f"{best.get('ROC_AUC', 0):.4f}")
    col3.metric("Precision", f"{best.get('Precision', 0):.4f}")
    col4.metric("Recall", f"{best.get('Recall', 0):.4f}")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📊 Confusion Matrix", "📈 Model Karşılaştırma", "🖼️ Proje Grafikleri"])

    with tab1:
        # Model seçimi
        model_sec = st.selectbox("Model seçin", ["LogisticRegression", "RandomForest_Cls"],
                                  index=1 if en_iyi == "RandomForest_Cls" else 0)
        if model_sec in cls and "Confusion_Matrix" in cls[model_sec]:
            cm = np.array(cls[model_sec]["Confusion_Matrix"])
            fig = px.imshow(cm,
                           labels=dict(x="Tahmin", y="Gerçek", color="Sayı"),
                           x=["Normal", "Sıçrama"],
                           y=["Normal", "Sıçrama"],
                           text_auto=True,
                           color_continuous_scale="Blues" if "Logistic" in model_sec else "Greens",
                           template="plotly_dark")
            fig.update_layout(title=f"{model_sec} — Karmaşıklık Matrisi", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            grafik = tum_grafikleri_bul().get("4_confusion_matrix")
            if grafik:
                st.image(str(grafik), caption="Confusion Matrix")

    with tab2:
        st.subheader("Model Karşılaştırma")
        tablo = []
        for model in ["Baseline_0", "LogisticRegression", "RandomForest_Cls"]:
            if model in cls:
                m = cls[model]
                tablo.append({
                    "Model": "⭐ " + model if model == en_iyi else model,
                    "F1": m.get("F1", 0),
                    "ROC-AUC": m.get("ROC_AUC", 0),
                    "Precision": m.get("Precision", 0),
                    "Recall": m.get("Recall", 0),
                })
        if tablo:
            st.dataframe(pd.DataFrame(tablo), use_container_width=True, hide_index=True)

        # Karşılaştırma grafik
        if len(tablo) > 1:
            df_karsi = pd.DataFrame(tablo)
            metrik_cols = ["F1", "ROC-AUC", "Precision", "Recall"]
            df_melt = df_karsi.melt(id_vars="Model", value_vars=metrik_cols,
                                     var_name="Metrik", value_name="Değer")
            fig = px.bar(df_melt, x="Model", y="Değer", color="Metrik",
                         barmode="group", template="plotly_dark",
                         title="Model Metrikleri Karşılaştırması")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        grafikler = tum_grafikleri_bul()
        for isim, yol in grafikler.items():
            if "confusion" in isim or "feature" in isim:
                st.image(str(yol), caption=isim.replace("_", " ").title())

    # Sıçrama eşiği bilgi
    st.divider()
    esik = cls.get("sicrama_esik", 0.15)
    st.info(f"ℹ️ Sıçrama eşiği: **%{esik}** — `src/utils.py` içinde `SICRAMA_ESIK` değeriyle ayarlanabilir.")


# ═════════════════════════════════════════════════════════════════════════════
# SAYFA 5: RAPOR
# ═════════════════════════════════════════════════════════════════════════════
elif sayfa == "📝 Rapor":
    st.header("📝 Proje Raporu")

    rapor_icerik = rapor_oku()
    st.markdown(rapor_icerik)

    # Grafikler göster
    st.divider()
    st.subheader("📊 Tüm Grafikler")
    grafikler = tum_grafikleri_bul()
    if grafikler:
        cols = st.columns(2)
        for i, (isim, yol) in enumerate(grafikler.items()):
            with cols[i % 2]:
                st.image(str(yol), caption=isim.replace("_", " ").title(), use_column_width=True)
    else:
        st.info("Grafik dosyaları bulunamadı.")


# ═════════════════════════════════════════════════════════════════════════════
# SAYFA 6: İNDİR
# ═════════════════════════════════════════════════════════════════════════════
elif sayfa == "⬇️ İndir":
    st.header("⬇️ Dosya İndirme")

    st.subheader("📊 Metrikler")
    metrikler = veri["metrikler"]
    if metrikler:
        metrik_json = json.dumps(metrikler, ensure_ascii=False, indent=2)
        st.download_button(
            label="📥 metrics.json indir",
            data=metrik_json,
            file_name="metrics.json",
            mime="application/json",
        )
        with st.expander("Metrik içeriği"):
            st.json(metrikler)

    st.divider()

    st.subheader("📝 Rapor")
    rapor_icerik = rapor_oku()
    st.download_button(
        label="📥 report.md indir",
        data=rapor_icerik,
        file_name="report.md",
        mime="text/markdown",
    )

    st.divider()

    st.subheader("📈 Regresyon Sonuçları")
    reg = veri["reg_son"]
    if reg:
        st.download_button(
            label="📥 regression_results.json indir",
            data=json.dumps(reg, ensure_ascii=False, indent=2, default=str),
            file_name="regression_results.json",
            mime="application/json",
        )

    st.subheader("🎯 Sınıflandırma Sonuçları")
    cls = veri["cls_son"]
    if cls:
        st.download_button(
            label="📥 classification_results.json indir",
            data=json.dumps(cls, ensure_ascii=False, indent=2, default=str),
            file_name="classification_results.json",
            mime="application/json",
        )

    st.divider()
    st.subheader("📋 Tahmin Verileri")
    pred = veri["reg_pred"]
    if pred is not None:
        st.download_button(
            label="📥 Regresyon tahminleri (CSV)",
            data=pred.to_csv(index=False),
            file_name="regression_predictions.csv",
            mime="text/csv",
        )
    cls_pred = veri["cls_pred"]
    if cls_pred is not None:
        st.download_button(
            label="📥 Sınıflandırma tahminleri (CSV)",
            data=cls_pred.to_csv(index=False),
            file_name="classification_predictions.csv",
            mime="text/csv",
        )
