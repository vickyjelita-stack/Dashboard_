import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️", layout="wide")
st.title("⚙️ Dashboard Predictive Maintenance")
st.caption("Prototipe alat bantu prioritas pemeriksaan aset berdasarkan prediksi kegagalan 24 jam.")

@st.cache_resource
def load_assets():
    model = joblib.load(BASE / "pdm_model.joblib")
    config = json.loads((BASE / "model_config.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(BASE / "data_predictive_maintenance.csv")
    importance = pd.read_csv(BASE / "permutation_importance.csv")
    return model, config, raw, importance

model, config, raw, importance = load_assets()
target, group = config["target"], config["group"]
threshold = float(config["threshold"])

data = raw.copy().sort_values([group, "operating_hours"]).reset_index(drop=True)
sensor_cols = [
    "operating_hours", "load_pct", "vibration_mm_s", "temperature_c",
    "rpm", "lubrication_hours_since_service", "noise_db", "current_amp",
    "ambient_temp_c"
]
sensor_cols = [c for c in sensor_cols if c in data.columns]
for col in sensor_cols:
    data[f"{col}_rollmean3"] = data.groupby(group, sort=False)[col].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )

features = config["features"]
latest = data.sort_values([group, "operating_hours"]).groupby(group, as_index=False).tail(1).copy()
latest["risk_probability"] = model.predict_proba(latest[features])[:, 1]
frac = float(config.get("alarm_medium_fraction", .60))
latest["alarm_level"] = np.where(
    latest.risk_probability >= threshold, "Tinggi",
    np.where(latest.risk_probability >= frac * threshold, "Sedang", "Rendah")
)
latest = latest.sort_values("risk_probability", ascending=False).reset_index(drop=True)
latest.insert(0, "Peringkat", np.arange(1, len(latest)+1))

st.warning(
    f"Threshold: {threshold:.3f} | Asumsi biaya relatif FN={config['cost_fn']} dan FP={config['cost_fp']}. "
    "Skor merupakan alat bantu prioritas pemeriksaan, bukan keputusan otomatis. "
    "Pada evaluasi eksperimen ini, performa model pada data test masih terbatas."
)
a,b,c,d = st.columns(4)
a.metric("Jumlah aset", int(latest[group].nunique()))
b.metric("Alarm tinggi", int((latest.alarm_level=="Tinggi").sum()))
c.metric("Alarm sedang", int((latest.alarm_level=="Sedang").sum()))
d.metric("Alarm rendah", int((latest.alarm_level=="Rendah").sum()))

st.subheader("Peringkat risiko aset")
levels = st.multiselect("Filter level alarm", ["Tinggi","Sedang","Rendah"], default=["Tinggi","Sedang","Rendah"])
view = latest[latest.alarm_level.isin(levels)][["Peringkat",group,"operating_hours","risk_probability","alarm_level"]].copy()
view["risk_probability"] = (view["risk_probability"]*100).round(1).astype(str)+"%"
st.dataframe(view, use_container_width=True, hide_index=True)

st.subheader("Rincian aset dan rekomendasi")
selected = st.selectbox("Pilih aset", latest[group].tolist())
row = latest.loc[latest[group]==selected].iloc[0]
left,right = st.columns([1,2])
with left:
    st.metric("Probabilitas model", f"{row.risk_probability*100:.1f}%")
    st.metric("Level alarm", row.alarm_level)
    st.caption(f"Jam operasi pengamatan terakhir: {row.operating_hours:,.0f}")
with right:
    st.markdown("**Indikator untuk diperiksa**")
    top = importance.feature.head(5).tolist()
    shown = 0
    for feat in top:
        if feat in row.index and pd.notna(row[feat]):
            st.write(f"- {feat}: {row[feat]:.2f}")
            shown += 1
    if shown == 0:
        st.write("Tidak ada indikator tambahan yang tersedia.")
    st.caption("Fitur penting bersifat prediktif global, bukan penjelasan sebab-akibat untuk satu aset.")

if row.alarm_level == "Tinggi":
    rec = "Prioritaskan inspeksi sesuai SOP. Verifikasi pembacaan sensor, kondisi fisik, dan riwayat servis. Jangan menghentikan mesin hanya berdasarkan skor model."
elif row.alarm_level == "Sedang":
    rec = "Jadwalkan pemeriksaan lebih dekat, pantau tren sensor, dan tinjau riwayat servis."
else:
    rec = "Lanjutkan pemantauan rutin dan preventive maintenance. Level rendah bukan jaminan aset bebas risiko."
st.info(rec)

st.subheader("Permutation importance")
st.caption("Mengukur kontribusi prediktif global pada data uji; bukan hubungan sebab-akibat.")
st.dataframe(importance.head(10), use_container_width=True, hide_index=True)

with st.expander("Metodologi dan batasan"):
    st.markdown(f"""
    - Target: `{target}`.
    - Validasi: split berdasarkan aset; aset train dan test terpisah.
    - Rolling features: tiga pengamatan sebelumnya, diurutkan berdasarkan `operating_hours`.
    - Dataset tidak memiliki timestamp kalender; jam operasi adalah proxy urutan.
    - Asumsi biaya FN={config['cost_fn']} dan FP={config['cost_fp']} merupakan asumsi relatif.
    - Model adalah prototipe dan bukan sistem diagnosis atau keselamatan otomatis.
    """)
st.caption("Dataset: data_predictive_maintenance.csv | Model dan threshold: artefak notebook.")
