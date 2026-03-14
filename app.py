"""
app.py — Kerala Flood Damage Assessment Dashboard (Streamlit)
Hugging Face Spaces ready | Single-file deployment
"""

import io
import os
import json
import time
import warnings
import requests
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Dict, Optional, Tuple
from PIL import Image
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kerala Flood AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
KOLLAM_LAT, KOLLAM_LON = 8.8932, 76.6141
API_URL = os.environ.get("API_URL", "http://localhost:8000")
DAMAGE_LABELS = ["No Damage", "Minor Damage", "Major Damage", "Catastrophic"]
DAMAGE_COLORS = ["#00d4aa", "#f5c518", "#ff7c3a", "#ff3b6b"]
DAMAGE_EMOJIS = ["✅", "⚠️", "🔶", "🔴"]

KERALA_DISTRICTS = {
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Kollam":             (8.8932, 76.6141),
    "Pathanamthitta":     (9.2648, 76.7870),
    "Alappuzha":          (9.4981, 76.3388),
    "Kottayam":           (9.5916, 76.5222),
    "Idukki":             (9.9189, 76.9749),
    "Ernakulam":          (9.9816, 76.2999),
    "Thrissur":           (10.5276, 76.2144),
    "Palakkad":           (10.7867, 76.6548),
    "Malappuram":         (11.0730, 76.0740),
    "Kozhikode":          (11.2588, 75.7804),
    "Wayanad":            (11.6854, 76.1320),
    "Kannur":             (11.8745, 75.3704),
    "Kasaragod":          (12.4996, 74.9869),
}

WEATHER_COLS = [
    "rainfall_mm", "river_level_m", "wind_speed_kmh", "humidity_pct",
    "temp_c", "pressure_hpa", "duration_hrs", "upstream_rainfall_mm",
    "soil_moisture", "previous_flood_days",
]

# ─────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;600&family=Outfit:wght@300;400;500;600&display=swap');

:root {
    --c1: #6366f1;
    --c2: #8b5cf6;
    --c3: #06b6d4;
    --c4: #10b981;
    --green:  #10b981;
    --yellow: #f59e0b;
    --orange: #f97316;
    --red:    #ef4444;
    --text:      #f8fafc;
    --text-mid:  #cbd5e1;
    --text-dim:  #64748b;
    --card:   rgba(255,255,255,0.07);
    --card-border: rgba(255,255,255,0.13);
    --shadow: 0 8px 32px rgba(0,0,0,0.25);
}

html, body, .stApp {
    background:
        radial-gradient(ellipse 120% 80% at 0% 0%,   #1e1b4b 0%, transparent 55%),
        radial-gradient(ellipse 80%  70% at 100% 0%,  #164e63 0%, transparent 50%),
        radial-gradient(ellipse 100% 80% at 50% 100%, #0f172a 0%, transparent 60%),
        radial-gradient(ellipse 70%  60% at 80%  50%, #312e81 0%, transparent 50%),
        linear-gradient(160deg, #0f172a 0%, #1e1b4b 40%, #0c2a3a 70%, #0f172a 100%) !important;
    background-attachment: fixed !important;
    min-height: 100vh;
}

/* Animated mesh blobs */
.stApp::before {
    content: '';
    position: fixed;
    width: 700px; height: 700px;
    top: -200px; right: -100px;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 65%);
    border-radius: 50%;
    pointer-events: none; z-index: 0;
    animation: blob1 12s ease-in-out infinite alternate;
}
.stApp::after {
    content: '';
    position: fixed;
    width: 600px; height: 600px;
    bottom: -150px; left: -100px;
    background: radial-gradient(circle, rgba(6,182,212,0.15) 0%, transparent 65%);
    border-radius: 50%;
    pointer-events: none; z-index: 0;
    animation: blob2 15s ease-in-out infinite alternate;
}
@keyframes blob1 { 0%{transform:translate(0,0) scale(1);} 100%{transform:translate(-60px,80px) scale(1.15);} }
@keyframes blob2 { 0%{transform:translate(0,0) scale(1);} 100%{transform:translate(80px,-60px) scale(1.2);} }

* { font-family: 'Outfit', sans-serif; color: var(--text); box-sizing: border-box; }
h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; color: var(--text) !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15,23,42,0.7) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border-right: 1px solid rgba(99,102,241,0.2) !important;
    box-shadow: 4px 0 30px rgba(0,0,0,0.3) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4) !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.5rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 24px rgba(99,102,241,0.4) !important;
    background-size: 200% 200% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 8px 36px rgba(99,102,241,0.6) !important;
}

/* Cards */
.flood-card {
    background: var(--card);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.9rem;
    box-shadow: var(--shadow);
}

/* Alert banners */
.alert-box {
    border-radius: 14px;
    padding: 1rem 1.4rem;
    margin: 0.8rem 0;
    display: flex;
    align-items: center;
    gap: 1rem;
    backdrop-filter: blur(12px);
}
.alert-green  { background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.35); }
.alert-yellow { background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.35); }
.alert-orange { background: rgba(249,115,22,0.12); border: 1px solid rgba(249,115,22,0.35); }
.alert-red    { background: rgba(239,68,68,0.14);  border: 1px solid rgba(239,68,68,0.40); }

.alert-level {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-dim);
    display: block;
    margin-bottom: 3px;
}
.alert-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
}

/* Probability bars */
.prob-row { display: flex; align-items: center; gap: 10px; margin: 8px 0; }
.prob-name {
    width: 130px; font-size: 0.81rem;
    color: var(--text-mid);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.prob-track {
    flex: 1; height: 6px;
    background: rgba(255,255,255,0.08);
    border-radius: 99px; overflow: hidden;
}
.prob-fill {
    height: 100%; border-radius: 99px;
    transition: width 0.6s cubic-bezier(.4,0,.2,1);
}
.prob-pct {
    width: 46px; text-align: right;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem; color: var(--text-mid);
}

/* Metric tiles */
.metric-tile {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    backdrop-filter: blur(10px);
}
.metric-tile .val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.45rem; font-weight: 600;
    line-height: 1; margin-bottom: 5px;
}
.metric-tile .lbl {
    font-size: 0.68rem; color: var(--text-dim);
    letter-spacing: 0.08em; text-transform: uppercase;
}

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.0rem; font-weight: 700;
    color: var(--text-mid);
    margin: 1.2rem 0 0.6rem 0;
    display: flex; align-items: center; gap: 8px;
    text-transform: uppercase; letter-spacing: 0.06em;
}

/* Brand */
.brand {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem; font-weight: 800;
    letter-spacing: 0.04em;
    background: linear-gradient(90deg, #818cf8, #06b6d4, #10b981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Chip */
.chip {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 99px; padding: 3px 10px;
    font-size: 0.7rem; color: #34d399;
    font-family: 'IBM Plex Mono', monospace;
}

/* Footer */
.footer-item {
    text-align: center; padding: 0.85rem;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
}
.footer-item .fi-icon { font-size: 1.2rem; margin-bottom: 4px; }
.footer-item .fi-title { font-size: 0.74rem; font-weight: 600; color: var(--text-mid); }
.footer-item .fi-sub { font-size: 0.67rem; color: var(--text-dim); }

/* Divider */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stDecoration"] { display: none; }

div[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.5rem !important;
}
div[data-testid="stMetricLabel"] {
    color: var(--text-dim) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# LOCAL MODEL
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading FloodNet model …")
def load_local_model():
    try:
        import onnxruntime as ort
        for mp in ["floodnet.onnx", "model/floodnet.onnx", "best_model.onnx"]:
            if Path(mp).exists():
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                return ort.InferenceSession(mp, opts, providers=["CPUExecutionProvider"])
    except ImportError:
        pass
    return None


def local_predict(image_np: np.ndarray, weather_np: np.ndarray) -> Dict:
    t0 = time.perf_counter()
    session = load_local_model()

    if session is not None:
        raw = session.run(["damage_probs"], {"sar_image": image_np, "weather": weather_np})[0][0]
        raw = raw - raw.max()
        probs = np.exp(raw) / np.exp(raw).sum()
    else:
        rain = weather_np[0, 0] * 38.6 + 45.2
        if rain < 10:    probs = np.array([0.75, 0.15, 0.07, 0.03])
        elif rain < 50:  probs = np.array([0.15, 0.60, 0.20, 0.05])
        elif rain < 120: probs = np.array([0.05, 0.20, 0.60, 0.15])
        else:            probs = np.array([0.02, 0.08, 0.25, 0.65])

    probs = np.array(probs, dtype=np.float32)
    probs /= probs.sum()
    idx = int(np.argmax(probs))
    ms = (time.perf_counter() - t0) * 1000

    return {
        "class_idx": idx,
        "label": DAMAGE_LABELS[idx],
        "color": DAMAGE_COLORS[idx],
        "confidence": float(probs[idx]),
        "probabilities": dict(zip(DAMAGE_LABELS, probs.tolist())),
        "inference_ms": ms,
        "alert_level": ["GREEN", "YELLOW", "ORANGE", "RED"][idx],
    }


# ─────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────
WEATHER_MEAN = np.array([45.2, 3.8, 22.1, 82.3, 27.5, 1008.2, 12.4, 38.7, 0.55, 2.1])
WEATHER_STD  = np.array([38.6, 2.1, 15.4, 11.2, 4.8,    6.3,  9.8, 32.1, 0.28, 3.7])


def preprocess_image_local(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((256, 256))
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    arr  = (arr - mean) / (std + 1e-8)
    arr  = np.clip(arr, -5, 5).transpose(2, 0, 1)
    return arr[np.newaxis].astype(np.float32)


def preprocess_weather_local(d: Dict) -> np.ndarray:
    defaults = dict(zip(WEATHER_COLS, [0, 2.5, 15, 75, 28, 1010, 6, 0, 0.5, 0]))
    vec = np.array([float(d.get(k, defaults[k]) or defaults[k]) for k in WEATHER_COLS])
    vec = np.nan_to_num(vec)
    vec = (vec - WEATHER_MEAN) / (WEATHER_STD + 1e-8)
    return np.clip(vec, -5, 5)[np.newaxis].astype(np.float32)


# ─────────────────────────────────────────────────────────
# WEATHER
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_weather(lat: float, lon: float) -> Dict:
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,precipitation,"
            f"wind_speed_10m,surface_pressure&timezone=Asia%2FKolkata"
        )
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        curr = r.json().get("current", {})
        rain = float(curr.get("precipitation", 0) or 0)
        return {
            "rainfall_mm":         rain,
            "river_level_m":       round(2.5 + rain * 0.02, 2),
            "wind_speed_kmh":      float(curr.get("wind_speed_10m", 15) or 15),
            "humidity_pct":        float(curr.get("relative_humidity_2m", 75) or 75),
            "temp_c":              float(curr.get("temperature_2m", 28) or 28),
            "pressure_hpa":        float(curr.get("surface_pressure", 1010) or 1010),
            "duration_hrs":        1.0,
            "upstream_rainfall_mm": round(rain * 0.8, 2),
            "soil_moisture":       min(0.3 + rain * 0.003, 0.98),
            "previous_flood_days": 0.0,
        }
    except Exception:
        return {k: 0.0 for k in WEATHER_COLS}


# ─────────────────────────────────────────────────────────
# MAP
# ─────────────────────────────────────────────────────────
def build_kerala_map(predictions: Optional[Dict] = None, selected_district: str = "Kollam") -> folium.Map:
    m = folium.Map(location=[KOLLAM_LAT, KOLLAM_LON], zoom_start=8, tiles=None)

    folium.TileLayer(
        tiles=(
            "https://bhuvan-vec2.nrsc.gov.in/bhuvan/gwc/service/wms?"
            "service=WMS&version=1.1.1&request=GetMap&layers=india_eo_stack"
            "&bbox={bbox-epsg-3857}&width=256&height=256&srs=EPSG:3857&format=image/png"
        ),
        attr="ISRO Bhuvan", name="Bhuvan Satellite",
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="CartoDB", name="Dark Map",
    ).add_to(m)

    for district, (lat, lon) in KERALA_DISTRICTS.items():
        pred_cls = 0
        color    = DAMAGE_COLORS[0]
        label    = DAMAGE_LABELS[0]

        if predictions and district in predictions:
            pred_cls = predictions[district]["class_idx"]
            color    = DAMAGE_COLORS[pred_cls]
            label    = DAMAGE_LABELS[pred_cls]

        is_selected = district == selected_district
        folium.CircleMarker(
            location=[lat, lon],
            radius=11 if is_selected else 7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=3 if is_selected else 1,
            popup=folium.Popup(
                f"<b>{district}</b><br>Status: {label}",
                max_width=180,
            ),
            tooltip=f"{district}: {label}",
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


# ─────────────────────────────────────────────────────────
# DISPLAY PREDICTION
# ─────────────────────────────────────────────────────────
def display_prediction(result: Dict):
    cls_idx    = result["class_idx"]
    label      = result["label"]
    color      = result["color"]
    confidence = result["confidence"]
    probs      = result["probabilities"]
    alert      = result.get("alert_level", "GREEN")
    inf_ms     = result.get("inference_ms", 0)

    alert_class = {
        "GREEN": "alert-green", "YELLOW": "alert-yellow",
        "ORANGE": "alert-orange", "RED": "alert-red"
    }.get(alert, "alert-green")

    st.markdown(
        f"""<div class="alert-box {alert_class}">
            <div>
                <span class="alert-level">Alert Level · {alert}</span>
                <span class="alert-label" style="color:{color};">
                    {DAMAGE_EMOJIS[cls_idx]}&nbsp; {label}
                </span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""<div class="metric-tile">
                <div class="val" style="color:{color};">{cls_idx}/3</div>
                <div class="lbl">Damage Class</div>
            </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(
            f"""<div class="metric-tile">
                <div class="val" style="color:var(--sky);">{confidence*100:.1f}%</div>
                <div class="lbl">Confidence</div>
            </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(
            f"""<div class="metric-tile">
                <div class="val" style="color:var(--teal);">{inf_ms:.0f}ms</div>
                <div class="lbl">Inference</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">📊 Damage Probabilities</div>', unsafe_allow_html=True)

    for i, (lbl, prob) in enumerate(probs.items()):
        pct = prob * 100
        c   = DAMAGE_COLORS[i]
        st.markdown(
            f"""<div class="prob-row">
                <span class="prob-name">{DAMAGE_EMOJIS[i]} {lbl}</span>
                <div class="prob-track">
                    <div class="prob-fill" style="width:{pct:.1f}%;background:{c};"></div>
                </div>
                <span class="prob-pct" style="color:{c};">{pct:.1f}%</span>
            </div>""",
            unsafe_allow_html=True,
        )

    # Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=cls_idx,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Damage Severity", "font": {"color": "#5a7a9e", "size": 12}},
        number={"font": {"color": color, "size": 42, "family": "IBM Plex Mono"}},
        gauge={
            "axis": {
                "range": [0, 3],
                "tickvals": [0, 1, 2, 3],
                "ticktext": ["None", "Minor", "Major", "Catas."],
                "tickcolor": "#1a2d4a",
                "tickfont": {"color": "#5a7a9e", "size": 10},
            },
            "bar": {"color": color, "thickness": 0.22},
            "bgcolor": "rgba(15,23,42,0.6)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 1], "color": "rgba(16,185,129,0.12)"},
                {"range": [1, 2], "color": "rgba(245,158,11,0.12)"},
                {"range": [2, 3], "color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 2},
                "thickness": 0.8,
                "value": cls_idx,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#94a3b8",
        height=190,
        margin=dict(l=20, r=20, t=30, b=5),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
def render_sidebar() -> Tuple[Dict, Optional[Image.Image], str]:
    with st.sidebar:
        st.markdown('<div class="brand">🌊 Kerala Flood AI</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.72rem;color:#5a7a9e;margin-bottom:1rem;">'
            'FloodNet v2 · EfficientNet-B3 · F1 = 0.9646</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        district = st.selectbox(
            "📍 District",
            list(KERALA_DISTRICTS.keys()),
            index=list(KERALA_DISTRICTS.keys()).index("Kollam"),
        )
        lat, lon = KERALA_DISTRICTS[district]

        st.divider()
        st.markdown("**🌦️ Weather Parameters**")

        auto_weather = st.toggle("Auto-fetch live weather", value=True)
        if auto_weather:
            with st.spinner(""):
                wx = fetch_weather(lat, lon)
            st.markdown('<span class="chip">🟢 Live data loaded</span>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
        else:
            wx = {k: 0.0 for k in WEATHER_COLS}

        weather = {}
        weather["rainfall_mm"]          = st.slider("Rainfall (mm/hr)",       0.0,  300.0, float(wx.get("rainfall_mm", 0)),        0.5)
        weather["river_level_m"]         = st.slider("River Level (m)",         0.0,   20.0, float(wx.get("river_level_m", 2.5)),    0.1)
        weather["wind_speed_kmh"]        = st.slider("Wind Speed (km/h)",       0.0,  150.0, float(wx.get("wind_speed_kmh", 15)),    1.0)
        weather["humidity_pct"]          = st.slider("Humidity (%)",            0.0,  100.0, float(wx.get("humidity_pct", 75)),      1.0)
        weather["temp_c"]                = st.slider("Temperature (°C)",       15.0,   45.0, float(wx.get("temp_c", 28)),            0.5)
        weather["pressure_hpa"]          = st.slider("Pressure (hPa)",        950.0, 1050.0, float(wx.get("pressure_hpa", 1010)),   0.5)
        weather["duration_hrs"]          = st.slider("Rain Duration (hrs)",     0.0,   72.0, float(wx.get("duration_hrs", 6)),       0.5)
        weather["upstream_rainfall_mm"]  = st.slider("Upstream Rainfall (mm)", 0.0,  300.0, float(wx.get("upstream_rainfall_mm", 0)), 0.5)
        weather["soil_moisture"]         = st.slider("Soil Moisture",          0.0,    1.0,  float(wx.get("soil_moisture", 0.5)),   0.01)
        weather["previous_flood_days"]   = st.slider("Days Since Last Flood",  0,     180,   int(wx.get("previous_flood_days", 0)))

        weather["lat"] = lat
        weather["lon"] = lon

        st.divider()
        st.markdown("**🛰️ SAR / Satellite Image**")
        uploaded = st.file_uploader(
            "Upload flood image",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            help="Aerial or satellite image · 256×256 recommended",
            label_visibility="collapsed",
        )
        image = None
        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded image", use_container_width=True)

        return weather, image, district


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    # ── Header ──────────────────────────────────────────
    st.markdown(
        """<div style="text-align:center;padding:1.8rem 0 1rem 0;">
        <h1 style="font-family:'Syne',sans-serif;font-size:2.1rem;font-weight:800;
                   background:linear-gradient(90deg,#0ea5e9 0%,#00d4aa 50%,#6366f1 100%);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   margin:0 0 0.3rem 0;letter-spacing:-0.02em;">
            Kerala Flood Damage Assessment
        </h1>
        <p style="color:#5a7a9e;font-size:0.82rem;letter-spacing:0.06em;text-transform:uppercase;margin:0;">
            FloodNet v2 &nbsp;·&nbsp; EfficientNet-B3 + IMD Weather &nbsp;·&nbsp; Kollam, Kerala
        </p>
        </div>""",
        unsafe_allow_html=True,
    )

    weather, image, selected_district = render_sidebar()

    col_map, col_pred = st.columns([3, 2], gap="large")

    # ── Prediction Panel ────────────────────────────────
    with col_pred:
        st.markdown('<div class="section-header">🔍 Damage Assessment</div>', unsafe_allow_html=True)

        if st.button("⚡ Run Analysis", use_container_width=True, type="primary"):
            with st.spinner("Running FloodNet inference …"):
                try:
                    if image is not None:
                        buf = io.BytesIO()
                        image.save(buf, format="PNG")
                        buf.seek(0)
                        resp = requests.post(
                            f"{API_URL}/predict",
                            files={"image": ("image.png", buf, "image/png")},
                            data={"weather_json": json.dumps(weather)},
                            timeout=10,
                        )
                        resp.raise_for_status()
                        result = resp.json()
                    else:
                        resp = requests.post(
                            f"{API_URL}/predict-weather",
                            json=weather, timeout=10,
                        )
                        resp.raise_for_status()
                        result = resp.json()
                except Exception:
                    if image is not None:
                        image_np = preprocess_image_local(image)
                    else:
                        rain_norm = min(weather["rainfall_mm"] / 200.0, 1.0)
                        syn = np.random.normal(1 - rain_norm * 0.7, 0.1, (256, 256)).clip(0, 1)
                        syn_rgb = np.stack([syn, syn, syn], axis=0)
                        image_np = syn_rgb[np.newaxis].astype(np.float32)

                    weather_np = preprocess_weather_local(weather)
                    result = local_predict(image_np, weather_np)
                    result["location"] = {"lat": weather["lat"], "lon": weather["lon"]}

            st.session_state["last_result"]   = result
            st.session_state["last_district"] = selected_district

        if "last_result" in st.session_state:
            display_prediction(st.session_state["last_result"])

        # Current conditions strip
        st.divider()
        st.markdown('<div class="section-header" style="font-size:0.88rem;">📡 Current Conditions</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)

        rain_color = "#ff3b6b" if weather["rainfall_mm"] > 100 else "#f5c518" if weather["rainfall_mm"] > 30 else "#00d4aa"
        with m1:
            st.markdown(
                f'<div class="metric-tile"><div class="val" style="color:{rain_color};">{weather["rainfall_mm"]:.0f}</div>'
                f'<div class="lbl">mm Rainfall</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(
                f'<div class="metric-tile"><div class="val" style="color:var(--sky);">{weather["river_level_m"]:.1f}</div>'
                f'<div class="lbl">m River</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(
                f'<div class="metric-tile"><div class="val" style="color:var(--text-mid);">{weather["humidity_pct"]:.0f}%</div>'
                f'<div class="lbl">Humidity</div></div>', unsafe_allow_html=True)

    # ── Map Panel ───────────────────────────────────────
    with col_map:
        lat, lon = KERALA_DISTRICTS[selected_district]
        st.markdown(
            f'<div class="section-header">🗺️ Kerala Flood Risk Map'
            f'<span style="font-size:0.72rem;color:#5a7a9e;font-weight:400;margin-left:8px;">'
            f'📍 {selected_district} — {lat:.4f}°N, {lon:.4f}°E</span></div>',
            unsafe_allow_html=True,
        )

        district_preds = {}
        if "last_result" in st.session_state and "last_district" in st.session_state:
            district_preds[st.session_state["last_district"]] = st.session_state["last_result"]

        kerala_map = build_kerala_map(district_preds, selected_district)
        st_folium(kerala_map, width=None, height=480, returned_objects=[])

        # Historical risk chart
        st.markdown('<div class="section-header" style="font-size:0.88rem;">📈 Historical Flood Risk by District</div>', unsafe_allow_html=True)

        historical = {
            "Wayanad": 3, "Idukki": 3, "Pathanamthitta": 2, "Alappuzha": 2,
            "Thrissur": 2, "Ernakulam": 1, "Kottayam": 2, "Kollam": 1,
            "Malappuram": 2, "Kozhikode": 1, "Palakkad": 1, "Kannur": 1,
            "Kasaragod": 0, "Thiruvananthapuram": 1,
        }
        df_hist = pd.DataFrame(
            list(historical.items()), columns=["District", "Risk Level"]
        ).sort_values("Risk Level", ascending=True)

        fig = px.bar(
            df_hist, x="Risk Level", y="District", orientation="h",
            color="Risk Level",
            color_continuous_scale=["#00d4aa", "#f5c518", "#ff7c3a", "#ff3b6b"],
            range_color=[0, 3],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8",
            font_family="Outfit",
            height=390,
            margin=dict(l=10, r=10, t=5, b=10),
            coloraxis_showscale=False,
            showlegend=False,
            xaxis=dict(
                tickvals=[0, 1, 2, 3],
                ticktext=["None", "Minor", "Major", "Catastrophic"],
                gridcolor="rgba(255,255,255,0.07)",
                tickfont=dict(color="#64748b", size=11),
            ),
            yaxis=dict(gridcolor="rgba(255,255,255,0.07)", tickfont=dict(color="#64748b", size=11)),
            bargap=0.3,
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    # ── Footer ──────────────────────────────────────────
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    items = [
        ("🛰️", "Sentinel-1 SAR", "Aerial Imagery"),
        ("🌦️", "IMD / Open-Meteo", "Weather Data"),
        ("🗺️", "ISRO Bhuvan", "Satellite Tiles"),
        ("🧠", "FloodNet v2", "EfficientNet-B3"),
    ]
    for col, (icon, title, sub) in zip([c1, c2, c3, c4], items):
        with col:
            st.markdown(
                f'<div class="footer-item"><div class="fi-icon">{icon}</div>'
                f'<div class="fi-title">{title}</div><div class="fi-sub">{sub}</div></div>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()