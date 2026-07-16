import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import timedelta
from prophet_analysis import (
    run_prophet,
    weather_risk_score,
    port_risk_score,
    get_future_date_risk,
    fleet_wide_analysis,
    get_route_map_data,
    PORT_COORDINATES
)
from google import genai
from dotenv import load_dotenv
load_dotenv()

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title = "🚢 Container Risk Monitor",
    page_icon  = "🚢",
    layout     = "wide"
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

# ═══════════════════════════════════════════════════════════════
# THEME — Maritime Command Center
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Share+Tech+Mono&family=Inter:wght@300;400;500;600&display=swap');

/* ── Root Variables ─────────────────────────────── */
:root {
    --ocean-deep   : #040d1a;
    --ocean-mid    : #071428;
    --ocean-surface: #0a1f3d;
    --panel        : #0d2444;
    --panel-border : #1a3a5c;
    --accent-blue  : #00aaff;
    --accent-teal  : #00e5cc;
    --accent-gold  : #ffd700;
    --danger-red   : #ff3b3b;
    --warn-orange  : #ff8c00;
    --warn-yellow  : #ffd700;
    --safe-green   : #00e676;
    --text-primary : #e8f4ff;
    --text-secondary: #7ab3d4;
    --text-dim     : #3a6080;
    --glow-blue    : 0 0 20px rgba(0,170,255,0.4);
    --glow-teal    : 0 0 20px rgba(0,229,204,0.3);
    --glow-red     : 0 0 20px rgba(255,59,59,0.5);
}

/* ── Global Background ──────────────────────────── */
.stApp {
    background: var(--ocean-deep);
    background-image:
        radial-gradient(ellipse at 20% 50%, rgba(0,50,100,0.15) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 20%, rgba(0,100,150,0.1) 0%, transparent 50%),
        linear-gradient(180deg, #040d1a 0%, #050f20 100%);
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

/* Animated grid background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(0,170,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,170,255,0.03) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050f20 0%, #071428 100%) !important;
    border-right: 1px solid var(--panel-border) !important;
    box-shadow: 4px 0 30px rgba(0,0,0,0.5);
}

[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

[data-testid="stSidebar"] .stRadio label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em;
    color: var(--text-secondary) !important;
    padding: 8px 12px;
    border-radius: 6px;
    transition: all 0.2s;
}

[data-testid="stSidebar"] .stRadio label:hover {
    color: var(--accent-blue) !important;
    background: rgba(0,170,255,0.08);
}

[data-testid="stSidebar"] h1 {
    font-family: 'Orbitron', monospace !important;
    font-size: 1rem !important;
    color: var(--accent-teal) !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--panel-border);
    padding-bottom: 12px;
    margin-bottom: 16px;
}

/* ── Main Title ─────────────────────────────────── */
h1 {
    font-family: 'Orbitron', monospace !important;
    font-weight: 800 !important;
    font-size: 2rem !important;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-teal));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 4px !important;
    filter: drop-shadow(0 0 20px rgba(0,170,255,0.3));
}

/* ── Subheaders ─────────────────────────────────── */
h2, h3 {
    font-family: 'Orbitron', monospace !important;
    font-weight: 600 !important;
    color: var(--accent-teal) !important;
    font-size: 1rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-left: 3px solid var(--accent-blue);
    padding-left: 12px;
    margin-top: 8px !important;
}

/* ── Metric Cards ───────────────────────────────── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg,
        rgba(13,36,68,0.9) 0%,
        rgba(7,20,40,0.9) 100%) !important;
    border: 1px solid var(--panel-border) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}

[data-testid="stMetric"]:hover {
    border-color: var(--accent-blue) !important;
    box-shadow: var(--glow-blue);
    transform: translateY(-2px);
}

[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 2px;
    background: linear-gradient(90deg,
        var(--accent-blue), var(--accent-teal));
}

[data-testid="stMetricLabel"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.12em !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase;
}

[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: var(--accent-blue) !important;
    line-height: 1.2;
}

/* ── Buttons ────────────────────────────────────── */
.stButton > button {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    background: linear-gradient(135deg,
        rgba(0,170,255,0.15) 0%,
        rgba(0,229,204,0.1) 100%) !important;
    color: var(--accent-teal) !important;
    border: 1px solid var(--accent-blue) !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    position: relative;
    overflow: hidden;
}

.stButton > button:hover {
    background: linear-gradient(135deg,
        rgba(0,170,255,0.3) 0%,
        rgba(0,229,204,0.2) 100%) !important;
    box-shadow: var(--glow-blue) !important;
    transform: translateY(-2px) !important;
    border-color: var(--accent-teal) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,
        rgba(0,170,255,0.3) 0%,
        rgba(0,229,204,0.2) 100%) !important;
    border-color: var(--accent-teal) !important;
    box-shadow: 0 0 15px rgba(0,229,204,0.2);
}

/* ── Alerts / Status Boxes ──────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-width: 1px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.03em;
}

/* Error = Critical */
[data-testid="stAlert"][data-baseweb="notification"][kind="error"],
div[data-testid="stAlert"] div[class*="st-emotion-cache"][style*="red"] {
    background: rgba(255,59,59,0.08) !important;
    border-color: var(--danger-red) !important;
    box-shadow: var(--glow-red) !important;
}

/* Warning = High/Medium */
[data-testid="stAlert"][kind="warning"] {
    background: rgba(255,140,0,0.08) !important;
    border-color: var(--warn-orange) !important;
}

/* Success = Normal */
[data-testid="stAlert"][kind="success"] {
    background: rgba(0,230,118,0.06) !important;
    border-color: var(--safe-green) !important;
}

/* ── Expander ───────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(13,36,68,0.6) !important;
    border: 1px solid var(--panel-border) !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
    overflow: hidden;
    transition: all 0.2s;
}

[data-testid="stExpander"]:hover {
    border-color: rgba(0,170,255,0.4) !important;
}

[data-testid="stExpander"] summary {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
    color: var(--text-primary) !important;
    letter-spacing: 0.05em;
    padding: 14px 18px !important;
    background: rgba(0,170,255,0.04) !important;
}

/* ── Dataframe / Table ──────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--panel-border) !important;
    border-radius: 10px !important;
    overflow: hidden;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ── Selectbox / Slider ─────────────────────────── */
[data-testid="stSelectbox"] > div,
[data-testid="stSlider"] {
    background: transparent !important;
}

[data-testid="stSelectbox"] > div > div {
    background: var(--panel) !important;
    border: 1px solid var(--panel-border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* Slider track */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent-blue) !important;
    box-shadow: var(--glow-blue);
}

/* ── Divider ────────────────────────────────────── */
hr {
    border-color: var(--panel-border) !important;
    margin: 24px 0 !important;
}

/* ── Spinner ────────────────────────────────────── */
[data-testid="stSpinner"] {
    color: var(--accent-blue) !important;
}

/* ── Markdown text ──────────────────────────────── */
.stMarkdown p {
    color: var(--text-secondary) !important;
    font-size: 0.9rem;
    line-height: 1.6;
}

.stMarkdown strong {
    color: var(--text-primary) !important;
}

/* ── Scrollbar ──────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--ocean-deep); }
::-webkit-scrollbar-thumb {
    background: var(--panel-border);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: var(--accent-blue); }

/* ── Status Badge ───────────────────────────────── */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.badge-critical {
    background: rgba(255,59,59,0.15);
    border: 1px solid var(--danger-red);
    color: var(--danger-red);
    box-shadow: var(--glow-red);
}

.badge-high {
    background: rgba(255,140,0,0.15);
    border: 1px solid var(--warn-orange);
    color: var(--warn-orange);
}

.badge-medium {
    background: rgba(255,215,0,0.1);
    border: 1px solid var(--warn-yellow);
    color: var(--warn-yellow);
}

.badge-normal {
    background: rgba(0,230,118,0.1);
    border: 1px solid var(--safe-green);
    color: var(--safe-green);
}

/* ── Info Panel ─────────────────────────────────── */
.info-panel {
    background: linear-gradient(135deg,
        rgba(13,36,68,0.9), rgba(7,20,40,0.9));
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    padding: 20px 24px;
    margin: 12px 0;
    position: relative;
}

.info-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg,
        var(--accent-blue), var(--accent-teal));
    border-radius: 3px 0 0 3px;
}

/* ── Matplotlib charts ──────────────────────────── */
[data-testid="stImage"] img,
[data-testid="stPyplotUserWarning"] {
    border-radius: 10px;
}

/* ── Top header bar ─────────────────────────────── */
.header-bar {
    background: linear-gradient(90deg,
        rgba(0,170,255,0.08), rgba(0,229,204,0.05));
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    padding: 12px 20px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-secondary);
}

/* ── Pulse animation for critical status ────────── */
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 10px rgba(255,59,59,0.3); }
    50%       { box-shadow: 0 0 25px rgba(255,59,59,0.7); }
}

@keyframes pulse-blue {
    0%, 100% { opacity: 0.7; }
    50%       { opacity: 1; }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

.stApp > div {
    animation: fadeInUp 0.4s ease forwards;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# MATPLOTLIB DARK THEME
# ═══════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor"  : "#040d1a",
    "axes.facecolor"    : "#071428",
    "axes.edgecolor"    : "#1a3a5c",
    "axes.labelcolor"   : "#7ab3d4",
    "axes.titlecolor"   : "#00e5cc",
    "axes.titlesize"    : 11,
    "axes.titleweight"  : "bold",
    "grid.color"        : "#1a3a5c",
    "grid.alpha"        : 0.4,
    "grid.linestyle"    : "--",
    "xtick.color"       : "#7ab3d4",
    "ytick.color"       : "#7ab3d4",
    "xtick.labelsize"   : 8,
    "ytick.labelsize"   : 8,
    "text.color"        : "#e8f4ff",
    "legend.facecolor"  : "#0d2444",
    "legend.edgecolor"  : "#1a3a5c",
    "legend.labelcolor" : "#e8f4ff",
    "legend.fontsize"   : 8,
    "figure.dpi"        : 120,
    "font.family"       : "monospace",
})

# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv(
        'sensor_weather_reduced_timeseries.csv',
        index_col  = 'timestamp',
        parse_dates= True
    )
    return df.reset_index()

df = load_data()

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
# Sidebar branding
st.sidebar.markdown("""
<div style="text-align:center; padding: 16px 0 24px 0;">
    <div style="font-family: Orbitron, monospace; font-size: 1.1rem;
                font-weight: 800; letter-spacing: 0.15em;
                background: linear-gradient(135deg, #00aaff, #00e5cc);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;">
        CARGO WATCH
    </div>
    <div style="font-family: Share Tech Mono, monospace;
                font-size: 0.65rem; color: #3a6080;
                letter-spacing: 0.2em; margin-top: 4px;">
        RISK INTELLIGENCE SYSTEM
    </div>
    <div style="width: 60%; height: 1px;
                background: linear-gradient(90deg, transparent, #1a3a5c, transparent);
                margin: 12px auto 0;"></div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### ⚙️ CONTROLS")

page = st.sidebar.radio(
    "Navigation",
    [
        "🗺️ Route Map",
        "📊 Single Container",
        "🚨 Fleet Dashboard",
        "🤖 AI Agent Analysis",
        "🎭 Scenario Simulator"
    ]
)

container_id = st.sidebar.selectbox(
    "Select Container",
    df['containerID'].unique()
)

forecast_days = st.sidebar.slider(
    "Forecast Days",
    min_value=30, max_value=90,
    value=60, step=10
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Anomaly Sensitivity**")
interval_width = st.sidebar.slider(
    "Interval Width",
    min_value=0.80, max_value=0.99,
    value=0.95, step=0.01,
    help="Higher = fewer anomalies detected"
)


# ═══════════════════════════════════════════════════════════════
# HELPER — Draw India Map with Routes
# ═══════════════════════════════════════════════════════════════
def draw_india_map(route_data, port_scores,
                   highlight_container=None,
                   selected_port=None):
    PORT_RISK_DICT = {}
    if not port_scores.empty:
        PORT_RISK_DICT = dict(zip(
            port_scores["final_port"],
            port_scores["risk_score"]
        ))

    fig, ax = plt.subplots(figsize=(22, 17))
    fig.patch.set_facecolor("#020c1b")
    ax.set_facecolor("#020c1b")
    ax.set_xlim(50, 130)
    ax.set_ylim(-5, 36)

    # Ocean bg
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((50,-5),80,41,color="#040e20",alpha=1.0,zorder=0))
    ax.add_patch(Rectangle((50,-5),80,41,color="#061530",alpha=0.5,zorder=0))

    # Grid
    for lat in range(-5,37,5):
        ax.axhline(lat, color="#0a2040", lw=0.4, alpha=0.6, linestyle=":", zorder=1)
        if 0 <= lat <= 35:
            ax.text(51, lat, f"{lat}°N", fontsize=8, color="#1a4060",
                    va="center", fontfamily="monospace", alpha=0.8)
    for lon in range(55,130,10):
        ax.axvline(lon, color="#0a2040", lw=0.4, alpha=0.6, linestyle=":", zorder=1)
        ax.text(lon, -4.2, f"{lon}°E", fontsize=8, color="#1a4060",
                ha="center", fontfamily="monospace", alpha=0.8)

    # India
    india_lon = [67.5,68.5,70,71,72,72.8,73.2,74,75,76,
                 77,78,78.5,80,80.3,80.1,79.5,78.8,78,77.5,
                 77,76.5,76,77,79,80,81,82,83,84,
                 85,86,87,87.5,88,88.5,89,89.5,67.5]
    india_lat = [23.5,23,22.5,22,21,19,18.5,17.5,16,15,
                 14,13.5,13,10.5,13,14,15,14,13,11,
                 9.5,8.5,8,10,11,12,13,14,15,17,
                 18,19,20,21,22,22.5,23,23.5,23.5]
    ax.fill(india_lon, india_lat, color="#0f2a0f", alpha=1.0, zorder=2)
    ax.fill(india_lon, india_lat, color="#1e4a1e", alpha=0.5, zorder=2)
    ax.plot(india_lon, india_lat, color="#2d6e2d", lw=1.5, alpha=0.9, zorder=3)

    # Sri Lanka
    sl_lon = [79.8,80.0,80.5,81.0,81.5,81.8,81.5,81.0,80.5,80.0,79.8]
    sl_lat = [9.8,9.5,9.0,8.5,8.0,8.2,7.5,7.0,7.2,7.8,9.8]
    ax.fill(sl_lon, sl_lat, color="#0f2a0f", alpha=0.9, zorder=2)
    ax.plot(sl_lon, sl_lat, color="#2d6e2d", lw=0.8, alpha=0.7, zorder=3)

    # Routes
    if not route_data.empty:
        max_d = max(route_data["damage_count"].max(), 1)
        drawn  = set()
        for _, seg in route_data.iterrows():
            key   = tuple(sorted([seg["from"], seg["to"]]))
            ratio = seg["damage_count"] / max_d
            is_hl = (highlight_container and
                     highlight_container in seg["containers"])
            if   is_hl:      color="#ffd700"; lw=3.2; alpha=1.0
            elif ratio>0.7:  color="#ff3333"; lw=2.8; alpha=0.85
            elif ratio>0.4:  color="#ff8800"; lw=2.2; alpha=0.75
            else:            color="#0099ee"; lw=1.6; alpha=0.55
            if key not in drawn or is_hl:
                ax.plot([seg["lon1"],seg["lon2"]],[seg["lat1"],seg["lat2"]],
                        color=color, lw=lw*4, alpha=alpha*0.06, zorder=4)
                ax.plot([seg["lon1"],seg["lon2"]],[seg["lat1"],seg["lat2"]],
                        color=color, lw=lw*2, alpha=alpha*0.14, zorder=4)
                ax.plot([seg["lon1"],seg["lon2"]],[seg["lat1"],seg["lat2"]],
                        color=color, lw=lw,   alpha=alpha,       zorder=5)
                drawn.add(key)

    # Label offsets
    OFFSETS = {
        "Mumbai":"(-7.5,0.5)",  "Nhava Sheva":"(0.5,-1.3)",
        "Chennai":"(0.5,-1.3)", "Vizag":"(0.5,0.5)",
        "Kolkata":"(0.5,0.5)",  "Kochi":"(-6.5,0.4)",
        "Mundra":"(-6.5,0.5)",  "Kandla":"(0.5,0.6)",
        "Pipavav":"(0.5,-1.3)", "Paradip":"(0.5,0.5)",
        "Ennore":"(0.5,-1.3)",  "Tuticorin":"(-8.0,0.3)",
        "Haldia":"(0.5,0.5)",   "Goa":"(-4.5,0.5)",
        "New Mangalore":"(-10.5,0.4)", "Colombo":"(-7.0,-1.1)",
        "Singapore":"(0.5,-0.9)",  "Shanghai":"(0.5,0.5)",
        "Dubai":"(-6.0,0.6)",   "Jebel Ali":"(0.5,-1.3)",
        "Abu Dhabi":"(0.5,0.6)","Port Klang":"(0.5,0.5)",
    }

    # Ports
    for port, (lat, lon) in PORT_COORDINATES.items():
        risk = PORT_RISK_DICT.get(port, 0)
        if   risk>=75: dc="#ff3b3b"; rc="#ff7070"
        elif risk>=50: dc="#ff8c00"; rc="#ffaa44"
        elif risk>=25: dc="#ffd700"; rc="#ffe055"
        else:          dc="#00e676"; rc="#55ffaa"

        is_sel = (port == selected_port)
        for s,a in [(600,0.04),(350,0.09),(200,0.16),(110,0.30),(60,1.0)]:
            ax.scatter(lon,lat,s=s,color=dc,alpha=a,zorder=6,linewidths=0)
        ax.scatter(lon,lat,s=28, color="white",alpha=0.95,zorder=12,linewidths=0)
        ax.scatter(lon,lat,s=130,color="none",edgecolors=rc,lw=2.0,alpha=0.65,zorder=11)
        if is_sel:
            ax.scatter(lon,lat,s=400,color="white",alpha=0.12,zorder=5,linewidths=0)
            ax.scatter(lon,lat,s=35, color="white",alpha=0.95,zorder=13,linewidths=0)

        ox,oy = eval(OFFSETS.get(port,"(0.5,0.5)"))
        lc = "#ffd700" if is_sel else "#e8f4ff"
        ax.text(lon+ox, lat+oy, f" {port} ",
                fontsize=11, color=lc, fontweight="bold",
                fontfamily="monospace", zorder=15,
                bbox=dict(boxstyle="round,pad=0.35",
                          facecolor="#020c1b", edgecolor=dc,
                          linewidth=1.4, alpha=0.93))
        if risk > 0:
            ax.text(lon+ox, lat+oy-1.1, f"  RISK {int(risk)}  ",
                    fontsize=8, color=rc,
                    fontfamily="monospace", zorder=15,
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor=dc, alpha=0.2,
                              edgecolor=dc, linewidth=0.7))

    # Ocean labels
    ocean_lbl = [
        (62,16,"ARABIAN SEA"),
        (88, 5,"BAY OF BENGAL"),
        (76, 1,"INDIAN OCEAN"),
        (114,13,"SOUTH CHINA SEA"),
    ]
    for olon,olat,otxt in ocean_lbl:
        ax.text(olon,olat,otxt,fontsize=9.5,color="#0d3060",
                ha="center",va="center",fontfamily="monospace",
                fontstyle="italic",alpha=0.7,fontweight="bold",
                linespacing=1.7)

    # Legend
    legend_ax = fig.add_axes([0.01,0.01,0.155,0.36])
    legend_ax.set_facecolor("#020c1b")
    legend_ax.set_xlim(0,1); legend_ax.set_ylim(0,1)
    legend_ax.set_xticks([]); legend_ax.set_yticks([])
    for sp in legend_ax.spines.values():
        sp.set_edgecolor("#1a3a5c"); sp.set_linewidth(1.2)
    items = [
        ("ROUTE DAMAGE",None,None),
        ("High damage","#ff3333","line"),
        ("Medium damage","#ff8800","line"),
        ("Low damage","#0099ee","line"),
        ("PORT RISK SCORE",None,None),
        ("Critical  ≥75","#ff3b3b","dot"),
        ("High      ≥50","#ff8c00","dot"),
        ("Medium    ≥25","#ffd700","dot"),
        ("Low        <25","#00e676","dot"),
    ]
    y = 0.96
    for label,color,kind in items:
        if kind is None:
            legend_ax.text(0.08,y,label,fontsize=8.5,
                           color="#00e5cc",fontweight="bold",
                           fontfamily="monospace",va="top")
            y -= 0.06
            legend_ax.plot([0.04,0.96],[y+0.03,y+0.03],
                           color="#1a3a5c",lw=0.8)
        elif kind=="line":
            legend_ax.plot([0.05,0.25],[y-0.025,y-0.025],color=color,lw=3)
            legend_ax.text(0.30,y-0.01,label,fontsize=8,
                           color="#aaccee",fontfamily="monospace",va="top")
            y -= 0.09
        elif kind=="dot":
            legend_ax.scatter([0.14],[y-0.03],s=100,color=color,zorder=5)
            legend_ax.text(0.30,y-0.01,label,fontsize=8,
                           color="#aaccee",fontfamily="monospace",va="top")
            y -= 0.09

    # Title bar
    title_ax = fig.add_axes([0.0,0.962,1.0,0.038])
    title_ax.set_facecolor("#030f22")
    title_ax.set_xlim(0,1); title_ax.set_ylim(0,1)
    title_ax.set_xticks([]); title_ax.set_yticks([])
    for sp in title_ax.spines.values():
        sp.set_edgecolor("#1a3a5c"); sp.set_linewidth(1)
    hl_txt = f" | Container: {highlight_container}" if highlight_container else ""
    title_ax.text(0.5,0.5,
        "◈   C A R G O   W A T C H   —   I N D I A N   O C E A N   R I S K   M A P   ◈",
        ha="center",va="center",fontsize=13,color="#00e5cc",
        fontweight="bold",fontfamily="monospace")
    title_ax.text(0.015,0.5,"● LIVE",
        ha="left",va="center",fontsize=9.5,
        color="#00ff88",fontfamily="monospace")
    title_ax.text(0.985,0.5,
        f"Port glow = risk score  |  Route color = damage history{hl_txt}",
        ha="right",va="center",fontsize=8.5,
        color="#3a6080",fontfamily="monospace")

    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a3a5c"); sp.set_linewidth(1.2)
    plt.subplots_adjust(left=0,right=1,top=0.962,bottom=0)
    return fig


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — ROUTE MAP
# ═══════════════════════════════════════════════════════════════
if page == "🗺️ Route Map":

    st.markdown("""
    <div class="header-bar">
        <span style="color:#00aaff">◈</span>
        <span style="color:#00e5cc; font-weight:600;">CARGO WATCH</span>
        <span style="color:#3a6080">|</span>
        <span>ROUTE INTELLIGENCE MAP</span>
        <span style="margin-left:auto; color:#00e5cc;">● LIVE</span>
    </div>
    """, unsafe_allow_html=True)
    st.title("🗺️ Indian Ocean Shipping Route Map")
    st.markdown(
        "Click a container to highlight its routes. "
        "Port size = risk level. Route color = damage history."
    )
    st.divider()

    # Controls
    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### 🔍 Filter")
        show_container = st.selectbox(
            "Highlight Container",
            ["All Containers"] + list(df['containerID'].unique())
        )

        selected_port = st.selectbox(
            "Inspect Port",
            ["None"] + sorted(list(PORT_COORDINATES.keys()))
        )

        st.markdown("---")
        st.markdown("### 📊 Port Risk Legend")
        st.markdown("🔴 Critical (≥75)")
        st.markdown("🟠 High    (≥50)")
        st.markdown("🟡 Medium  (≥25)")
        st.markdown("🟢 Low     (<25)")

    with col2:
        # Get route data
        highlight = (
            None if show_container == "All Containers"
            else show_container
        )
        route_data  = get_route_map_data(df, highlight)
        port_scores = port_risk_score(df)

        # Draw map
        fig = draw_india_map(
            route_data, port_scores,
            highlight_container = highlight,
            selected_port = (
                None if selected_port == "None"
                else selected_port
            )
        )
        st.pyplot(fig)

    st.divider()

    # Port details when selected
    if selected_port != "None":
        st.subheader(f"⚓ Port Details — {selected_port}")

        port_info = port_scores[
            port_scores['final_port'] == selected_port
        ]

        if not port_info.empty:
            row = port_info.iloc[0]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Visits",   int(row['total']))
            col2.metric("Damage Events",  int(row['damaged']))
            col3.metric("Damage Rate",    f"{row['damage_rate']}%")
            col4.metric("Risk Score",     f"{row['risk_score']}/100")
            st.markdown(f"**Risk Level:** {row['risk_label']}")

        # Which containers visit this port
        containers_at_port = df[
            df['final_port'] == selected_port
        ]['containerID'].unique()

        st.markdown(
            f"**Containers visiting {selected_port}:** "
            f"{', '.join(containers_at_port)}"
        )

        # Damage events at this port
        damages_at_port = df[
            (df['final_port'] == selected_port) &
            (df['damaged(T)'] == 1)
        ][['timestamp', 'containerID',
           'shock_g', 'wave_m', 'storm']]

        if len(damages_at_port) > 0:
            st.markdown(
                f"**Damage events at {selected_port}:**"
            )
            st.dataframe(
                damages_at_port.reset_index(drop=True),
                use_container_width=True
            )
        else:
            st.success(f"No damage events recorded at {selected_port}")

    st.divider()

    # Route summary table
    st.subheader("📋 All Routes Summary")
    if not route_data.empty:
        route_summary = route_data.groupby('route').agg(
            segments     = ('from', 'count'),
            damage_count = ('damage_count', 'first'),
            containers   = ('containers', 'first')
        ).reset_index()
        route_summary['container_count'] = route_summary[
            'containers'
        ].apply(len)
        st.dataframe(
            route_summary[[
                'route', 'segments',
                'damage_count', 'container_count'
            ]].sort_values('damage_count', ascending=False),
            use_container_width=True
        )


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — SINGLE CONTAINER
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Single Container":

    st.markdown("""
    <div class="header-bar">
        <span style="color:#00aaff">◈</span>
        <span style="color:#00e5cc; font-weight:600;">CARGO WATCH</span>
        <span style="color:#3a6080">|</span>
        <span>CONTAINER SENSOR ANALYSIS</span>
        <span style="margin-left:auto; color:#00e5cc;">● LIVE</span>
    </div>
    """, unsafe_allow_html=True)
    st.title("📊 Single Container Analysis")
    st.markdown(
        f"Analyzing **{container_id}** | "
        f"Forecast: **{forecast_days} days**"
    )
    st.divider()

    # Run Prophet
    with st.spinner(f"Running Prophet for {container_id}..."):
        (shock_hist, shock_future,
         wave_hist,  wave_future,
         one_container) = run_prophet(
            df, container_id, forecast_days
        )
        port_scores = port_risk_score(df)

    anomalies = shock_hist[shock_hist['anomaly'] == True]

    # Key Metrics
    st.subheader("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📅 Days Monitored",
                len(shock_hist))
    col2.metric("🚨 Shock Anomalies",
                len(anomalies))
    col3.metric("💥 Real Damage Events",
                int(shock_hist['damaged'].sum()))
    col4.metric("📦 Cargo Type",
                one_container['type'].mode()[0])
    st.divider()

    # Risk Analysis Dashboard
    # Shock Forecast
    st.subheader("⚡ Shock_g Forecast")
    fig1, ax1 = plt.subplots(figsize=(16, 5))

    ax1.plot(shock_hist['ds'], shock_hist['actual'],
             color='steelblue', linewidth=0.8,
             label='Actual shock_g')
    ax1.plot(shock_hist['ds'], shock_hist['yhat'],
             color='orange', linewidth=1.5,
             label='Prophet Forecast')
    ax1.fill_between(
        shock_hist['ds'],
        shock_hist['yhat_lower'],
        shock_hist['yhat_upper'],
        alpha=0.2, color='orange',
        label='Normal Zone (95%)'
    )
    ax1.plot(shock_future['ds'], shock_future['yhat'],
             color='green', linewidth=2, linestyle='--',
             label=f'Next {forecast_days} days')
    ax1.fill_between(
        shock_future['ds'],
        shock_future['yhat_lower'],
        shock_future['yhat_upper'],
        alpha=0.15, color='green',
        label='Future Normal Zone'
    )
    ax1.axvline(
        shock_hist['ds'].max(),
        color='gray', linestyle=':', linewidth=1.5,
        label='Forecast Start'
    )
    ax1.scatter(
        anomalies['ds'], anomalies['actual'],
        color='red', s=60, zorder=5,
        label=f'Anomalies ({len(anomalies)})'
    )
    ax1.set_xlabel('Date')
    ax1.set_ylabel('shock_g')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)

    # Wave Forecast
    st.subheader("🌊 Wave Height Forecast")
    fig2, ax2 = plt.subplots(figsize=(16, 4))

    ax2.plot(wave_hist['ds'], wave_hist['actual'],
             color='steelblue', linewidth=0.8,
             label='Actual wave_m')
    ax2.plot(wave_hist['ds'], wave_hist['yhat'],
             color='orange', linewidth=1.5,
             label='Prophet Forecast')
    ax2.fill_between(
        wave_hist['ds'],
        wave_hist['yhat_lower'],
        wave_hist['yhat_upper'],
        alpha=0.2, color='orange',
        label='Normal Zone'
    )
    ax2.plot(wave_future['ds'], wave_future['yhat'],
             color='green', linewidth=2, linestyle='--',
             label=f'Next {forecast_days} days')
    ax2.fill_between(
        wave_future['ds'],
        wave_future['yhat_lower'],
        wave_future['yhat_upper'],
        alpha=0.15, color='green'
    )
    ax2.axvline(
        wave_hist['ds'].max(),
        color='gray', linestyle=':', linewidth=1.5,
        label='Forecast Start'
    )
    ax2.axhline(
        3.0, color='red', linestyle='--', linewidth=1,
        label='Danger threshold (3.0m)'
    )
    ax2.set_xlabel('Date')
    ax2.set_ylabel('wave_m')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)
    st.divider()

    # Future Date Risk
    st.subheader("🔮 Future Date Risk Predictor")
    st.markdown(
        "Pick any future date to see predicted "
        "shock + wave + port risk"
    )

    min_future    = (
        shock_hist['ds'].max().date() + timedelta(days=1)
    )
    max_future    = shock_future['ds'].max().date()
    selected_date = st.date_input(
        "Select a future date",
        value=min_future,
        min_value=min_future,
        max_value=max_future
    )

    container_ports = one_container['final_port'].unique()
    risk_info = get_future_date_risk(
        shock_future, wave_future,
        port_scores, selected_date,
        container_ports
    )

    if risk_info:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "⚡ Predicted shock_g",
            f"{risk_info['predicted_shock']}g",
            help=(
                f"Normal range: "
                f"{risk_info['shock_lower']}g → "
                f"{risk_info['shock_upper']}g | "
                f"Band position: {risk_info['shock_position']}%"
            )
        )
        col2.metric(
            "🌊 Predicted wave_m",
            f"{risk_info['predicted_wave']}m",
            help=f"Upper bound: {risk_info['wave_upper']}m"
        )
        col3.metric(
            "🌪️ Weather Risk",
            f"{risk_info['weather_score']}/100",
            help=risk_info['weather_label']
        )
        col4.metric(
            "⚓ Port Risk",
            f"{risk_info['max_port_risk']}/100",
            help=f"Riskiest: {risk_info['riskiest_port']}"
        )

        st.markdown("---")
        cargo = one_container['type'].mode()[0]
        flags = risk_info['flags']

        causes = []
        if risk_info['shock_flag']:
            causes.append("shock anomaly exceeding safe threshold")
        if risk_info['weather_flag']:
            causes.append(
                f"high wave forecast "
                f"({risk_info['predicted_wave']}m)"
            )
        if risk_info['port_flag']:
            causes.append(
                f"high risk port "
                f"({risk_info['riskiest_port']} "
                f"score={risk_info['max_port_risk']})"
            )

        flag_msg = (
            "Container flagged due to "
            + " + ".join(causes)
            + f" | Cargo: {cargo}"
            if causes else "All conditions normal"
        )

        if   flags == 3:
            st.error(f"🔴 CRITICAL on {selected_date}\n\n{flag_msg}")
        elif flags == 2:
            st.warning(f"🟠 HIGH on {selected_date}\n\n{flag_msg}")
        elif flags == 1:
            st.warning(f"🟡 MEDIUM on {selected_date}\n\n{flag_msg}")
        else:
            st.success(
                f"🟢 LOW RISK on {selected_date} — "
                f"All conditions within normal range."
            )

    st.divider()

    # Anomaly Table
    st.subheader("🚨 Historical Anomaly Details")
    if len(anomalies) > 0:
        st.dataframe(
            anomalies[[
                'ds', 'actual', 'yhat',
                'yhat_upper', 'yhat_lower',
                'wave_m', 'wind_kph',
                'storm', 'route', 'damaged'
            ]].reset_index(drop=True),
            use_container_width=True
        )
    else:
        st.success("No anomalies detected!")

    st.divider()

    # Port Risk
    st.subheader("⚓ Port Risk Scores")
    st.dataframe(
            port_scores[[
                'final_port', 'total', 'damaged',
                'damage_rate', 'risk_score', 'risk_label'
            ]],
            use_container_width=True
        )

    st.divider()

    # Overall Risks
    st.subheader("📊 Overall Risks")

    import seaborn as sns

    fig_risk, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig_risk.patch.set_facecolor("#040d1a")

    # Plot 1 — Total Risk Score distribution
    axes[0,0].set_facecolor("#071428")
    sns.histplot(df["risk_score"], bins=30, kde=True,
                 color="#ff6b6b", ax=axes[0,0],
                 edgecolor="none", alpha=0.8)
    axes[0,0].set_title("Total Risk Score Distribution",
                         color="#00e5cc", fontsize=10)
    axes[0,0].set_xlabel("Risk Score", color="#7ab3d4")
    axes[0,0].set_ylabel("Count",      color="#7ab3d4")
    axes[0,0].tick_params(colors="#7ab3d4")
    axes[0,0].spines["bottom"].set_color("#1a3a5c")
    axes[0,0].spines["left"].set_color("#1a3a5c")
    axes[0,0].spines["top"].set_visible(False)
    axes[0,0].spines["right"].set_visible(False)

    # Plot 2 — Port Volume
    axes[0,1].set_facecolor("#071428")
    port_counts = df["final_port"].value_counts().head(10)
    colors_red  = sns.color_palette("Reds_r", len(port_counts))
    axes[0,1].barh(port_counts.index, port_counts.values,
                   color=colors_red, edgecolor="none")
    axes[0,1].set_title("Top 10 Ports by Volume",
                         color="#00e5cc", fontsize=10)
    axes[0,1].set_xlabel("Visits", color="#7ab3d4")
    axes[0,1].tick_params(colors="#7ab3d4")
    axes[0,1].spines["bottom"].set_color("#1a3a5c")
    axes[0,1].spines["left"].set_color("#1a3a5c")
    axes[0,1].spines["top"].set_visible(False)
    axes[0,1].spines["right"].set_visible(False)

    # Plot 3 — Average Risk by Port
    axes[0,2].set_facecolor("#071428")
    port_avg = (df.groupby("final_port")["risk_score"]
                  .mean()
                  .sort_values(ascending=False)
                  .head(8))
    colors_v = sns.color_palette("viridis", len(port_avg))
    axes[0,2].barh(port_avg.index, port_avg.values,
                   color=colors_v, edgecolor="none")
    axes[0,2].set_title("Avg Risk Score by Port",
                         color="#00e5cc", fontsize=10)
    axes[0,2].set_xlabel("Avg Risk", color="#7ab3d4")
    axes[0,2].tick_params(colors="#7ab3d4")
    axes[0,2].spines["bottom"].set_color("#1a3a5c")
    axes[0,2].spines["left"].set_color("#1a3a5c")
    axes[0,2].spines["top"].set_visible(False)
    axes[0,2].spines["right"].set_visible(False)

    # Plot 4 — Risk by Damage History
    axes[1,0].set_facecolor("#071428")
    dmg_0 = df[df["damaged(T)"] == 0]["risk_score"]
    dmg_1 = df[df["damaged(T)"] == 1]["risk_score"]
    axes[1,0].boxplot(
        [dmg_0, dmg_1],
        labels=["Not Damaged", "Damaged"],
        patch_artist=True,
        boxprops    =dict(facecolor="#0d2444", color="#00aaff"),
        medianprops =dict(color="#00e5cc", lw=2),
        whiskerprops=dict(color="#1a3a5c"),
        capprops    =dict(color="#1a3a5c"),
        flierprops  =dict(marker="o", markerfacecolor="#ff3b3b",
                          markersize=4, alpha=0.5)
    )
    axes[1,0].set_title("Risk Score vs Damage History",
                         color="#00e5cc", fontsize=10)
    axes[1,0].set_ylabel("Risk Score", color="#7ab3d4")
    axes[1,0].tick_params(colors="#7ab3d4")
    axes[1,0].spines["bottom"].set_color("#1a3a5c")
    axes[1,0].spines["left"].set_color("#1a3a5c")
    axes[1,0].spines["top"].set_visible(False)
    axes[1,0].spines["right"].set_visible(False)

    # Plot 5 — Top 10 Riskiest Containers
    axes[1,1].set_facecolor("#071428")
    top10 = df.nlargest(10, "risk_score")
    colors_cw = sns.color_palette("coolwarm", len(top10))
    axes[1,1].barh(top10["containerID"],
                   top10["risk_score"],
                   color=colors_cw, edgecolor="none")
    for i, cid in enumerate(top10["containerID"]):
        if cid == container_id:
            axes[1,1].get_children()[i].set_edgecolor("#ffd700")
            axes[1,1].get_children()[i].set_linewidth(2)
    axes[1,1].set_title("Top 10 Riskiest Containers",
                         color="#00e5cc", fontsize=10)
    axes[1,1].set_xlabel("Risk Score", color="#7ab3d4")
    axes[1,1].tick_params(colors="#7ab3d4")
    axes[1,1].spines["bottom"].set_color("#1a3a5c")
    axes[1,1].spines["left"].set_color("#1a3a5c")
    axes[1,1].spines["top"].set_visible(False)
    axes[1,1].spines["right"].set_visible(False)

    # Plot 6 — This container risk over time
    axes[1,2].set_facecolor("#071428")
    one_c = df[df["containerID"] == container_id].copy()
    one_c["timestamp"] = pd.to_datetime(one_c["timestamp"])
    axes[1,2].plot(one_c["timestamp"], one_c["risk_score"],
                   color="#00aaff", lw=1.2, alpha=0.85)
    axes[1,2].fill_between(one_c["timestamp"],
                            one_c["risk_score"],
                            alpha=0.15, color="#00aaff")
    dmg_events = one_c[one_c["damaged(T)"] == 1]
    axes[1,2].scatter(dmg_events["timestamp"],
                      dmg_events["risk_score"],
                      color="#ff3b3b", s=50, zorder=5,
                      label="Damage events")
    axes[1,2].set_title(
        f"Risk Over Time — {container_id}",
        color="#00e5cc", fontsize=10)
    axes[1,2].set_xlabel("Date",       color="#7ab3d4")
    axes[1,2].set_ylabel("Risk Score", color="#7ab3d4")
    axes[1,2].tick_params(colors="#7ab3d4", axis="x", rotation=30)
    axes[1,2].legend(fontsize=8)
    axes[1,2].spines["bottom"].set_color("#1a3a5c")
    axes[1,2].spines["left"].set_color("#1a3a5c")
    axes[1,2].spines["top"].set_visible(False)
    axes[1,2].spines["right"].set_visible(False)

    plt.tight_layout(pad=2.0)
    st.pyplot(fig_risk)

    st.divider()

    # AI Analysis
    st.subheader("🤖 AI Agent Analysis")
    if st.button("🔍 Analyze with Gemini AI", type="primary"):

        anomaly_summary = (
            anomalies[[
                'ds', 'actual', 'yhat',
                'wave_m', 'wind_kph',
                'storm', 'route', 'damaged'
            ]].to_string()
            if len(anomalies) > 0
            else "No anomalies detected"
        )

        container_port_risks = port_scores[
            port_scores['final_port'].isin(container_ports)
        ][['final_port', 'risk_score', 'risk_label']].to_string()

        future_summary = f"""
        Next {forecast_days} days forecast:
        Avg predicted shock_g : {shock_future['yhat'].mean():.3f}g
        Max predicted shock_g : {shock_future['yhat'].max():.3f}g
        Avg predicted wave_m  : {wave_future['yhat'].mean():.3f}m
        Max predicted wave_m  : {wave_future['yhat'].max():.3f}m
        """

        cargo = one_container['type'].mode()[0]

        prompt = f"""
        You are a shipping container safety expert.

        Container ID  : {container_id}
        Cargo Type    : {cargo}
        Days monitored: {len(shock_hist)}
        Anomalies     : {len(anomalies)}
        Real damages  : {int(shock_hist['damaged'].sum())}

        HISTORICAL ANOMALIES:
        {anomaly_summary}

        PORT RISKS FOR THIS CONTAINER:
        {container_port_risks}

        FUTURE FORECAST ({forecast_days} days):
        {future_summary}

        Generate a structured report with:

        1. CONTAINER STATUS
           One line like:
           "Container flagged due to [reason1] + [reason2]"

        2. ROOT CAUSE ANALYSIS
           Why did anomalies happen?
           Link shock spikes to wave/storm events.

        3. PORT RISK WARNING
           Which ports are dangerous and why?

        4. FUTURE OUTLOOK
           What does the forecast suggest?
           Is risk increasing or decreasing?

        5. SPECIFIC RECOMMENDATIONS
           Give 3-4 specific actions like:
           - "High wave forecast + {cargo} cargo →
              Recommend reinforced securing"
           - "Port X shows high damage rate →
              Flag for priority inspection"

        6. OVERALL RISK: Low/Medium/High/Critical

        Be specific, practical and concise.
        """

        with st.spinner("🤖 Gemini is analyzing..."):
            response = client.models.generate_content(
                model   = "gemini-2.5-flash",
                contents= prompt
            )
        st.markdown(response.text)


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — FLEET DASHBOARD
# ═══════════════════════════════════════════════════════════════
elif page == "🚨 Fleet Dashboard":

    st.markdown("""
    <div class="header-bar">
        <span style="color:#ff3b3b">◈</span>
        <span style="color:#00e5cc; font-weight:600;">CARGO WATCH</span>
        <span style="color:#3a6080">|</span>
        <span>FLEET ALERT COMMAND CENTER</span>
        <span style="margin-left:auto; color:#ff3b3b;">⚠ MONITORING</span>
    </div>
    """, unsafe_allow_html=True)
    st.title("🚨 Fleet-Wide Alert Dashboard")
    st.markdown(
        "Scans **all 10 containers** and checks "
        "shock + weather + port risk automatically"
    )
    st.divider()

    if st.button("🔍 Scan Entire Fleet", type="primary"):

        with st.spinner(
            "Scanning all containers... (~2-3 mins)"
        ):
            fleet_df, port_scores = fleet_wide_analysis(
                df, forecast_days
            )

        # Summary
        st.subheader("📊 Fleet Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🔴 Critical",
            len(fleet_df[fleet_df['flags_triggered'] == 3]))
        col2.metric("🟠 High",
            len(fleet_df[fleet_df['flags_triggered'] == 2]))
        col3.metric("🟡 Medium",
            len(fleet_df[fleet_df['flags_triggered'] == 1]))
        col4.metric("🟢 Normal",
            len(fleet_df[fleet_df['flags_triggered'] == 0]))

        st.divider()

        # Fleet map
        st.subheader("🗺️ Fleet Route Map")
        all_routes  = get_route_map_data(df)
        fig_fleet   = draw_india_map(all_routes, port_scores)
        st.pyplot(fig_fleet)

        st.divider()

        # Container cards
        st.subheader("🚢 Container Alert Cards")
        for _, row in fleet_df.sort_values(
            'flags_triggered', ascending=False
        ).iterrows():

            with st.expander(
                f"{row['combined_alert']} — "
                f"{row['containerID']} | "
                f"Cargo: {row['cargo_type']}"
            ):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**⚡ Shock Anomaly**")
                    st.markdown(f"Level: {row['shock_level']}")
                    st.markdown(
                        f"Anomalies: `{row['shock_anomalies']}`"
                    )
                    st.markdown(
                        f"Max shock: `{row['max_shock_g']}g`"
                    )
                    if row['shock_flag']:
                        st.error("⚠️ Threshold exceeded!")
                    else:
                        st.success("✅ Normal")

                with col2:
                    st.markdown("**🌊 Weather Risk**")
                    st.markdown(f"Level: {row['weather_level']}")
                    st.markdown(
                        f"Avg risk: `{row['avg_weather_risk']}/100`"
                    )
                    st.markdown(
                        f"High risk days: "
                        f"`{row['high_weather_days']}`"
                    )
                    if row['weather_flag']:
                        st.error("⚠️ Dangerous conditions!")
                    else:
                        st.success("✅ Acceptable")

                with col3:
                    st.markdown("**⚓ Port Risk**")
                    st.markdown(f"Level: {row['port_level']}")
                    st.markdown(
                        f"Riskiest: `{row['riskiest_port']}`"
                    )
                    st.markdown(
                        f"Score: `{row['max_port_risk']}/100`"
                    )
                    if row['port_flag']:
                        st.error("⚠️ High risk port!")
                    else:
                        st.success("✅ Acceptable")

                # Verdict
                st.markdown("---")
                flags  = row['flags_triggered']
                causes = []
                if row['shock_flag']:
                    causes.append(
                        "shock anomaly exceeding safe threshold"
                    )
                if row['weather_flag']:
                    causes.append("dangerous weather conditions")
                if row['port_flag']:
                    causes.append(
                        f"high risk port "
                        f"({row['riskiest_port']})"
                    )

                msg = (
                    "Container flagged due to "
                    + " + ".join(causes)
                    + f" | Cargo: {row['cargo_type']}"
                    + f" | Real damages: {row['real_damages']}"
                    if causes
                    else f"{row['containerID']} — all normal"
                )

                if   flags == 3: st.error(f"🔴 CRITICAL — {msg}")
                elif flags == 2: st.warning(f"🟠 HIGH — {msg}")
                elif flags == 1: st.warning(f"🟡 MEDIUM — {msg}")
                else:            st.success(f"🟢 NORMAL — {msg}")

        # Full table
        st.divider()
        st.subheader("📋 Full Fleet Table")
        st.dataframe(
            fleet_df[[
                'containerID', 'combined_alert',
                'cargo_type',
                'shock_level', 'shock_anomalies',
                'weather_level', 'avg_weather_risk',
                'port_level', 'riskiest_port',
                'flags_triggered', 'real_damages'
            ]],
            use_container_width=True
        )

        # Gemini fleet report
        st.divider()
        st.subheader("🤖 Gemini Fleet Report")
        if st.button("Get AI Fleet Analysis"):
            fleet_summary = fleet_df[[
                'containerID', 'combined_alert',
                'cargo_type', 'shock_anomalies',
                'avg_weather_risk', 'riskiest_port',
                'max_port_risk', 'flags_triggered',
                'real_damages'
            ]].to_string()

            prompt = f"""
            You are a shipping fleet safety expert.

            Full fleet analysis — 10 containers:
            {fleet_summary}

            Provide:
            1. FLEET HEALTH SUMMARY
               Overall status in 2 sentences

            2. CRITICAL CONTAINERS
               Each critical container with reason like:
               "Container X flagged due to shock anomaly
                + dangerous port + fragile cargo"

            3. COMMON RISK PATTERNS
               Shared risks across containers?

            4. DAMAGE PREDICTIONS
               Most likely to be damaged in 60 days and why?

            5. TOP 3 FLEET-WIDE ACTIONS
               Most important steps for operations team

            Be specific about container IDs.
            """

            with st.spinner("🤖 Gemini analyzing fleet..."):
                response = client.models.generate_content(
                    model   = "gemini-2.5-flash",
                    contents= prompt
                )
            st.markdown(response.text)


# ═══════════════════════════════════════════════════════════════
# PAGE 4 — AI AGENT ANALYSIS
# ═══════════════════════════════════════════════════════════════
elif page == "🤖 AI Agent Analysis":

    from agents import run_agent_analysis

    st.markdown("""
    <div class="header-bar">
        <span style="color:#00aaff">◈</span>
        <span style="color:#00e5cc; font-weight:600;">CARGO WATCH</span>
        <span style="color:#3a6080">|</span>
        <span>MULTI-AGENT RISK INTELLIGENCE</span>
        <span style="margin-left:auto; color:#00e5cc;">● LIVE</span>
    </div>
    """, unsafe_allow_html=True)

    st.title("🤖 Multi-Agent Risk Analysis")
    st.markdown(
        "**4 specialized AI agents** analyze your container "
        "from different angles and combine findings into "
        "actionable recommendations."
    )
    st.divider()

    # Agent architecture diagram
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0d2444, #071428);
        border: 1px solid #1a3a5c;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.82rem;
        color: #7ab3d4;
        line-height: 2;
    ">
        <span style="color:#00e5cc; font-weight:bold;">AGENT PIPELINE</span><br>
        <span style="color:#00aaff">⚡ Agent 1</span> Sensor Monitoring
        → analyzes shock_g anomalies via Prophet<br>
        <span style="color:#0099dd">🌊 Agent 2</span> Weather Risk
        → evaluates wave height + storm severity<br>
        <span style="color:#00bbcc">⚓ Agent 3</span> Port Risk
        → identifies historical port damage patterns<br>
        <span style="color:#00e5cc">🧠 Agent 4</span> Decision Agent
        → consults Agents 1-3 → generates recommendations
    </div>
    """, unsafe_allow_html=True)

    # Container selector
    st.subheader(f"🚢 Select Container to Analyze")
    selected = st.selectbox(
        "Container ID",
        df['containerID'].unique(),
        key="agent_container"
    )

    # Show container quick stats
    one_c = df[df['containerID'] == selected]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📦 Cargo Type",
                one_c['type'].mode()[0])
    col2.metric("📅 Days of Data",
                len(one_c))
    col3.metric("💥 Damage Events",
                int(one_c['damaged(T)'].sum()))
    col4.metric("🌊 Avg Wave Height",
                f"{one_c['wave_m'].mean():.2f}m")

    st.divider()

    # Run agents button
    if st.button(
        "🚀 Run All 4 Agents",
        type="primary",
        key="run_agents"
    ):
        st.markdown("""
        <div style="
            background: rgba(0,170,255,0.05);
            border: 1px solid #1a3a5c;
            border-radius: 10px;
            padding: 16px 20px;
            margin: 12px 0;
            font-family: monospace;
            font-size: 0.8rem;
            color: #3a6080;
        ">
        ⚠️ Each agent makes separate Gemini API calls.
        This takes ~60-90 seconds. Please wait...
        </div>
        """, unsafe_allow_html=True)

        # ── Agent 1 — Sensor ──────────────────────────────────
        with st.container():
            st.markdown("""
            <div style="
                background: rgba(0,170,255,0.06);
                border-left: 3px solid #00aaff;
                border-radius: 0 8px 8px 0;
                padding: 12px 16px;
                margin: 8px 0;
                font-family: monospace;
                font-size: 0.82rem;
                color: #00aaff;
            ">⚡ AGENT 1 — SENSOR MONITORING AGENT running...</div>
            """, unsafe_allow_html=True)

            with st.spinner("Analyzing shock anomalies..."):
                try:
                    import agents as _ag
                    from agents import build_sensor_agent
                    _ag.setup(df, GOOGLE_API_KEY)
                    sensor_agent = build_sensor_agent()
                    sensor_out   = sensor_agent.invoke({
                        "input": (
                            f"Analyze shock anomalies and sensor "
                            f"statistics for container {selected}. "
                            f"Provide complete sensor risk assessment."
                        )
                    })
                    sensor_result = sensor_out.get(
                        'output', 'No output'
                    )
                except Exception as e:
                    sensor_result = f"Error: {str(e)}"

            st.markdown(f"""
            <div style="
                background: rgba(13,36,68,0.8);
                border: 1px solid #1a3a5c;
                border-top: 2px solid #00aaff;
                border-radius: 8px;
                padding: 16px 20px;
                margin: 4px 0 16px 0;
                font-family: monospace;
                font-size: 0.85rem;
                color: #e8f4ff;
                line-height: 1.7;
                white-space: pre-wrap;
            ">{sensor_result}</div>
            """, unsafe_allow_html=True)

        # ── Agent 2 — Weather ─────────────────────────────────
        with st.container():
            st.markdown("""
            <div style="
                background: rgba(0,153,221,0.06);
                border-left: 3px solid #0099dd;
                border-radius: 0 8px 8px 0;
                padding: 12px 16px;
                margin: 8px 0;
                font-family: monospace;
                font-size: 0.82rem;
                color: #0099dd;
            ">🌊 AGENT 2 — WEATHER RISK AGENT running...</div>
            """, unsafe_allow_html=True)

            with st.spinner("Evaluating sea conditions..."):
                try:
                    import agents as _ag
                    from agents import build_weather_agent
                    _ag.setup(df, GOOGLE_API_KEY)
                    weather_agent = build_weather_agent()
                    weather_out   = weather_agent.invoke({
                        "input": (
                            f"Evaluate all weather risks and storm "
                            f"events for container {selected}. "
                            f"Assess impact on cargo safety."
                        )
                    })
                    weather_result = weather_out.get(
                        'output', 'No output'
                    )
                except Exception as e:
                    weather_result = f"Error: {str(e)}"

            st.markdown(f"""
            <div style="
                background: rgba(13,36,68,0.8);
                border: 1px solid #1a3a5c;
                border-top: 2px solid #0099dd;
                border-radius: 8px;
                padding: 16px 20px;
                margin: 4px 0 16px 0;
                font-family: monospace;
                font-size: 0.85rem;
                color: #e8f4ff;
                line-height: 1.7;
                white-space: pre-wrap;
            ">{weather_result}</div>
            """, unsafe_allow_html=True)

        # ── Agent 3 — Port ────────────────────────────────────
        with st.container():
            st.markdown("""
            <div style="
                background: rgba(0,187,204,0.06);
                border-left: 3px solid #00bbcc;
                border-radius: 0 8px 8px 0;
                padding: 12px 16px;
                margin: 8px 0;
                font-family: monospace;
                font-size: 0.82rem;
                color: #00bbcc;
            ">⚓ AGENT 3 — PORT RISK AGENT running...</div>
            """, unsafe_allow_html=True)

            with st.spinner("Scanning port damage patterns..."):
                try:
                    import agents as _ag
                    from agents import build_port_agent
                    _ag.setup(df, GOOGLE_API_KEY)
                    port_agent = build_port_agent()
                    port_out   = port_agent.invoke({
                        "input": (
                            f"Identify all port risks and damage "
                            f"patterns for container {selected}. "
                            f"Which ports need priority inspection?"
                        )
                    })
                    port_result = port_out.get(
                        'output', 'No output'
                    )
                except Exception as e:
                    port_result = f"Error: {str(e)}"

            st.markdown(f"""
            <div style="
                background: rgba(13,36,68,0.8);
                border: 1px solid #1a3a5c;
                border-top: 2px solid #00bbcc;
                border-radius: 8px;
                padding: 16px 20px;
                margin: 4px 0 16px 0;
                font-family: monospace;
                font-size: 0.85rem;
                color: #e8f4ff;
                line-height: 1.7;
                white-space: pre-wrap;
            ">{port_result}</div>
            """, unsafe_allow_html=True)

        # ── Agent 4 — Decision ────────────────────────────────
        with st.container():
            st.markdown("""
            <div style="
                background: rgba(0,229,204,0.06);
                border-left: 3px solid #00e5cc;
                border-radius: 0 8px 8px 0;
                padding: 12px 16px;
                margin: 8px 0;
                font-family: monospace;
                font-size: 0.82rem;
                color: #00e5cc;
                font-weight: bold;
            ">🧠 AGENT 4 — DECISION AGENT combining all findings...</div>
            """, unsafe_allow_html=True)

            with st.spinner(
                "Decision Agent consulting all agents..."
            ):
                try:
                    import agents as _ag
                    from agents import build_decision_agent
                    _ag.setup(df, GOOGLE_API_KEY)
                    decision_agent = build_decision_agent()
                    decision_out   = decision_agent.invoke({
                        "input": (
                            f"Provide complete risk analysis and "
                            f"preventive recommendations for "
                            f"container {selected}. "
                            f"Consult all 3 specialist agents and "
                            f"combine findings into specific "
                            f"actionable recommendations."
                        )
                    })
                    decision_result = decision_out.get(
                        'output', 'No output'
                    )
                except Exception as e:
                    decision_result = f"Error: {str(e)}"

            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg,
                    rgba(0,229,204,0.06),
                    rgba(0,170,255,0.04));
                border: 1px solid #00e5cc;
                border-radius: 10px;
                padding: 20px 24px;
                margin: 4px 0 16px 0;
                font-size: 0.9rem;
                color: #e8f4ff;
                line-height: 1.8;
                white-space: pre-wrap;
            ">{decision_result}</div>
            """, unsafe_allow_html=True)

        # Final summary banner
        st.markdown("""
        <div style="
            background: linear-gradient(90deg,
                rgba(0,170,255,0.08),
                rgba(0,229,204,0.05));
            border: 1px solid #1a3a5c;
            border-radius: 10px;
            padding: 14px 20px;
            margin-top: 20px;
            font-family: monospace;
            font-size: 0.8rem;
            color: #3a6080;
            text-align: center;
        ">
        ✅ Analysis complete — 4 agents ran successfully
        | Sensor Agent + Weather Agent + Port Agent
        → Decision Agent
        </div>
        """, unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════
# PAGE 5 — SCENARIO SIMULATOR
# ═══════════════════════════════════════════════════════════════
elif page == "🎭 Scenario Simulator":

    import seaborn as sns

    st.markdown("""
    <div class="header-bar">
        <span style="color:#ffd700">◈</span>
        <span style="color:#00e5cc; font-weight:600;">CARGO WATCH</span>
        <span style="color:#3a6080">|</span>
        <span>SCENARIO SIMULATOR</span>
        <span style="margin-left:auto; color:#ffd700;">⚡ SIMULATION MODE</span>
    </div>
    """, unsafe_allow_html=True)

    st.title("🎭 Scenario Simulator")
    st.markdown(
        "Simulate **what-if scenarios** on any container. "
        "See how different conditions affect risk levels "
        "and get AI recommendations."
    )
    st.divider()

    # ── Scenario selector ─────────────────────────────────────
    st.subheader("⚙️ Configure Simulation")

    col1, col2 = st.columns(2)
    with col1:
        sim_container = st.selectbox(
            "Select Container",
            df['containerID'].unique(),
            key="sim_container"
        )
    with col2:
        scenario = st.selectbox(
            "Select Scenario",
            [
                "🌪️ Sudden Storm Formation",
                "⚓ Port Congestion — Rough Handling",
                "🌡️ Refrigeration System Failure",
                "📦 Fragile Cargo Shipment"
            ],
            key="scenario_select"
        )

    st.markdown("---")

    # ── Scenario parameters ───────────────────────────────────
    sim_params = {}

    if scenario == "🌪️ Sudden Storm Formation":
        st.markdown("### 🌪️ Storm Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            sim_params['storm_intensity'] = st.select_slider(
                "Storm Intensity",
                options=["Mild", "Moderate", "Severe", "Extreme"],
                value="Severe"
            )
        with col2:
            sim_params['wave_spike'] = st.slider(
                "Wave Height (m)",
                min_value=2.0, max_value=6.0,
                value=4.5, step=0.1
            )
        with col3:
            sim_params['wind_spike'] = st.slider(
                "Wind Speed (kph)",
                min_value=20, max_value=80,
                value=45, step=5
            )
        sim_params['affected_days'] = st.slider(
            "Duration (days)",
            min_value=1, max_value=14,
            value=5, step=1
        )
        st.markdown(f"""
        <div class="info-panel">
        <strong style="color:#ffd700">Scenario:</strong>
        A sudden storm forms mid-route with
        <strong>{sim_params['wave_spike']}m waves</strong> and
        <strong>{sim_params['wind_spike']} kph winds</strong>
        lasting <strong>{sim_params['affected_days']} days</strong>.
        </div>
        """, unsafe_allow_html=True)

    elif scenario == "⚓ Port Congestion — Rough Handling":
        st.markdown("### ⚓ Port Congestion Parameters")
        col1, col2 = st.columns(2)
        with col1:
            all_ports = sorted(df['final_port'].unique().tolist())
            sim_params['congested_port'] = st.selectbox(
                "Congested Port",
                all_ports,
                index=all_ports.index('Colombo')
                      if 'Colombo' in all_ports else 0
            )
        with col2:
            sim_params['handling_severity'] = st.select_slider(
                "Handling Severity",
                options=["Light", "Moderate", "Heavy", "Extreme"],
                value="Heavy"
            )
        sim_params['shock_multiplier'] = st.slider(
            "Shock Increase at Port (×)",
            min_value=1.5, max_value=5.0,
            value=3.0, step=0.5
        )
        st.markdown(f"""
        <div class="info-panel">
        <strong style="color:#ffd700">Scenario:</strong>
        Port <strong>{sim_params['congested_port']}</strong>
        is congested causing
        <strong>{sim_params['handling_severity'].lower()}</strong>
        rough handling —
        shock_g increases by
        <strong>{sim_params['shock_multiplier']}×</strong>
        during port visits.
        </div>
        """, unsafe_allow_html=True)

    elif scenario == "🌡️ Refrigeration System Failure":
        st.markdown("### 🌡️ Refrigeration Parameters")
        col1, col2 = st.columns(2)
        with col1:
            sim_params['failure_day'] = st.slider(
                "Failure starts on day",
                min_value=1, max_value=600,
                value=200, step=10
            )
        with col2:
            sim_params['temp_spike'] = st.slider(
                "Temperature spike (°C)",
                min_value=5, max_value=40,
                value=18, step=1
            )
        sim_params['recovery_days'] = st.slider(
            "Days until repaired",
            min_value=1, max_value=30,
            value=7, step=1
        )
        st.markdown(f"""
        <div class="info-panel">
        <strong style="color:#ffd700">Scenario:</strong>
        Refrigeration fails on day
        <strong>{sim_params['failure_day']}</strong>,
        temperature spikes by
        <strong>+{sim_params['temp_spike']}°C</strong>
        for <strong>{sim_params['recovery_days']} days</strong>
        before repair.
        </div>
        """, unsafe_allow_html=True)

    elif scenario == "📦 Fragile Cargo Shipment":
        st.markdown("### 📦 Fragile Cargo Parameters")
        col1, col2 = st.columns(2)
        with col1:
            sim_params['cargo_type'] = st.selectbox(
                "Fragile Cargo Type",
                ["Electronics", "Pharma", "Glassware",
                 "Ceramics", "Medical Equipment"]
            )
        with col2:
            sim_params['shock_threshold'] = st.slider(
                "Safe shock threshold (g)",
                min_value=0.3, max_value=2.0,
                value=0.8, step=0.1
            )
        sim_params['sensitivity'] = st.select_slider(
            "Cargo Sensitivity",
            options=["Low", "Medium", "High", "Extreme"],
            value="High"
        )
        st.markdown(f"""
        <div class="info-panel">
        <strong style="color:#ffd700">Scenario:</strong>
        Container carrying
        <strong>{sim_params['cargo_type']}</strong>
        with safe shock limit of only
        <strong>{sim_params['shock_threshold']}g</strong>
        (vs normal 2.5g threshold).
        Sensitivity: <strong>{sim_params['sensitivity']}</strong>.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Run simulation button ─────────────────────────────────
    if st.button("🚀 Run Simulation", type="primary",
                 key="run_sim"):

        # Get container data
        one_c = df[df['containerID'] == sim_container].copy()
        one_c = one_c.reset_index()
        if 'timestamp' not in one_c.columns:
            one_c = one_c.rename(columns={'index': 'timestamp'})
        one_c['timestamp'] = pd.to_datetime(one_c['timestamp'])

        # ── Apply scenario modifications ──────────────────────
        sim_df = one_c.copy()

        if scenario == "🌪️ Sudden Storm Formation":
            # Pick middle of dataset as storm period
            mid = len(sim_df) // 2
            days = sim_params['affected_days']
            storm_level = {
                "Mild": "Yellow", "Moderate": "Orange",
                "Severe": "Red",   "Extreme": "Red"
            }[sim_params['storm_intensity']]
            shock_boost = {
                "Mild": 1.5, "Moderate": 2.5,
                "Severe": 4.0, "Extreme": 6.0
            }[sim_params['storm_intensity']]

            sim_df.loc[mid:mid+days, 'wave_m']   = sim_params['wave_spike']
            sim_df.loc[mid:mid+days, 'wind_kph'] = sim_params['wind_spike']
            sim_df.loc[mid:mid+days, 'storm']    = storm_level
            sim_df.loc[mid:mid+days, 'shock_g']  = (
                sim_df.loc[mid:mid+days, 'shock_g'] * shock_boost
            )
            affected_mask = sim_df.index.isin(range(mid, mid+days))

        elif scenario == "⚓ Port Congestion — Rough Handling":
            port = sim_params['congested_port']
            mult = sim_params['shock_multiplier']
            port_mask = sim_df['final_port'] == port
            sim_df.loc[port_mask, 'shock_g'] = (
                sim_df.loc[port_mask, 'shock_g'] * mult
            )
            affected_mask = port_mask

        elif scenario == "🌡️ Refrigeration System Failure":
            start = sim_params['failure_day']
            end   = start + sim_params['recovery_days']
            sim_df.loc[start:end, 'temperature_c'] = (
                sim_df.loc[start:end, 'temperature_c'] +
                sim_params['temp_spike']
            )
            affected_mask = sim_df.index.isin(range(start, end))

        elif scenario == "📦 Fragile Cargo Shipment":
            threshold = sim_params['shock_threshold']
            sim_df['type'] = sim_params['cargo_type']
            affected_mask = sim_df['shock_g'] > threshold

        # ── Run Prophet on ORIGINAL data ──────────────────────
        with st.spinner("Running Prophet on original data..."):
            orig_hist, orig_future, _, _, _ = run_prophet(
                df, sim_container, forecast_days=30
            )

        # ── Run Prophet on SIMULATED data ─────────────────────
        with st.spinner("Running Prophet on simulated data..."):
            # Create temp df with simulated values
            temp_df = df.copy()
            temp_df = temp_df.reset_index()
            if 'timestamp' not in temp_df.columns:
                temp_df = temp_df.rename(
                    columns={'index': 'timestamp'}
                )
            temp_df['timestamp'] = pd.to_datetime(
                temp_df['timestamp']
            )
            # Replace container rows with simulated data
            mask = temp_df['containerID'] == sim_container
            for col in ['shock_g', 'wave_m', 'wind_kph',
                        'storm', 'temperature_c']:
                if col in sim_df.columns:
                    temp_df.loc[
                        mask, col
                    ] = sim_df[col].values

            temp_df = temp_df.set_index('timestamp')
            sim_hist, sim_future, _, _, _ = run_prophet(
                temp_df.reset_index(), sim_container,
                forecast_days=30
            )

        st.divider()

        # ── Results header ────────────────────────────────────
        st.subheader("📊 Simulation Results")

        # Anomaly counts
        orig_anomalies = orig_hist[
            orig_hist['anomaly'] == True
        ]
        sim_anomalies  = sim_hist[
            sim_hist['anomaly'] == True
        ]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Original Anomalies",
            len(orig_anomalies)
        )
        col2.metric(
            "Simulated Anomalies",
            len(sim_anomalies),
            delta=len(sim_anomalies) - len(orig_anomalies),
            delta_color="inverse"
        )
        col3.metric(
            "Original Max Shock",
            f"{orig_hist['actual'].max():.3f}g"
        )
        col4.metric(
            "Simulated Max Shock",
            f"{sim_hist['actual'].max():.3f}g",
            delta=f"{sim_hist['actual'].max() - orig_hist['actual'].max():.3f}g",
            delta_color="inverse"
        )

        # ── Before vs After Chart ─────────────────────────────
        st.subheader("📈 Before vs After — shock_g")

        fig_sim, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig_sim.patch.set_facecolor('#040d1a')

        # Original
        axes[0].set_facecolor('#071428')
        axes[0].plot(
            orig_hist['ds'], orig_hist['actual'],
            color='#00aaff', lw=0.8, label='Actual shock_g'
        )
        axes[0].plot(
            orig_hist['ds'], orig_hist['yhat'],
            color='orange', lw=1.5, label='Prophet forecast'
        )
        axes[0].fill_between(
            orig_hist['ds'],
            orig_hist['yhat_lower'],
            orig_hist['yhat_upper'],
            alpha=0.2, color='orange', label='Normal zone'
        )
        axes[0].scatter(
            orig_anomalies['ds'],
            orig_anomalies['actual'],
            color='red', s=50, zorder=5,
            label=f'Anomalies ({len(orig_anomalies)})'
        )
        axes[0].set_title(
            f'ORIGINAL DATA — {sim_container}',
            color='#00e5cc', fontsize=11
        )
        axes[0].set_ylabel('shock_g', color='#7ab3d4')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Simulated
        axes[1].set_facecolor('#071428')
        axes[1].plot(
            sim_hist['ds'], sim_hist['actual'],
            color='#ff8800', lw=0.8,
            label='Simulated shock_g'
        )
        axes[1].plot(
            sim_hist['ds'], sim_hist['yhat'],
            color='orange', lw=1.5, label='Prophet forecast'
        )
        axes[1].fill_between(
            sim_hist['ds'],
            sim_hist['yhat_lower'],
            sim_hist['yhat_upper'],
            alpha=0.2, color='orange', label='Normal zone'
        )
        axes[1].scatter(
            sim_anomalies['ds'],
            sim_anomalies['actual'],
            color='#ff3b3b', s=50, zorder=5,
            label=f'Anomalies ({len(sim_anomalies)})'
        )
        # Highlight scenario-affected period
        if scenario == "🌪️ Sudden Storm Formation":
            mid = len(sim_hist) // 2
            days = sim_params['affected_days']
            if mid < len(sim_hist) and mid+days < len(sim_hist):
                axes[1].axvspan(
                    sim_hist['ds'].iloc[mid],
                    sim_hist['ds'].iloc[min(
                        mid+days, len(sim_hist)-1
                    )],
                    alpha=0.15, color='red',
                    label='Storm period'
                )

        axes[1].set_title(
            f'SIMULATED — {scenario}',
            color='#ffd700', fontsize=11
        )
        axes[1].set_ylabel('shock_g', color='#7ab3d4')
        axes[1].set_xlabel('Date',    color='#7ab3d4')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout(pad=2.0)
        st.pyplot(fig_sim)

        # ── Refrigeration — extra temp chart ──────────────────
        if scenario == "🌡️ Refrigeration System Failure":
            st.subheader("🌡️ Temperature — Before vs After")
            fig_temp, ax_temp = plt.subplots(figsize=(16, 4))
            fig_temp.patch.set_facecolor('#040d1a')
            ax_temp.set_facecolor('#071428')

            ax_temp.plot(
                one_c['timestamp'],
                one_c['temperature_c'],
                color='#00aaff', lw=1, label='Original temp'
            )
            ax_temp.plot(
                sim_df['timestamp'],
                sim_df['temperature_c'],
                color='#ff3b3b', lw=1.2,
                label='Simulated temp'
            )
            safe_temp = one_c['temperature_c'].mean() + 5
            ax_temp.axhline(
                safe_temp, color='orange',
                linestyle='--', lw=1,
                label=f'Safe threshold ({safe_temp:.1f}°C)'
            )
            start = sim_params['failure_day']
            end   = start + sim_params['recovery_days']
            if start < len(sim_df) and end < len(sim_df):
                ax_temp.axvspan(
                    sim_df['timestamp'].iloc[start],
                    sim_df['timestamp'].iloc[
                        min(end, len(sim_df)-1)
                    ],
                    alpha=0.15, color='red',
                    label='Failure period'
                )
            ax_temp.set_title(
                'Temperature During Refrigeration Failure',
                color='#00e5cc', fontsize=11
            )
            ax_temp.set_ylabel(
                'Temperature (°C)', color='#7ab3d4'
            )
            ax_temp.legend(fontsize=8)
            ax_temp.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_temp)

        st.divider()

        # ── Risk comparison ───────────────────────────────────
        st.subheader("⚠️ Risk Level Change")

        orig_risk = len(orig_anomalies)
        sim_risk  = len(sim_anomalies)

        if sim_risk > orig_risk * 2:
            st.error(f"""
            🔴 **CRITICAL RISK INCREASE**

            Scenario **{scenario}** dramatically increases risk for
            container **{sim_container}**:

            Original anomalies : {orig_risk}
            Simulated anomalies: {sim_risk}
            Increase           : +{sim_risk - orig_risk} anomalies
            """)
        elif sim_risk > orig_risk:
            st.warning(f"""
            🟠 **ELEVATED RISK**

            Scenario increases anomalies from
            **{orig_risk}** to **{sim_risk}**
            """)
        else:
            st.success(f"""
            🟢 **RISK UNCHANGED**
            Scenario does not significantly increase anomalies.
            """)

        st.divider()

        # ── AI Recommendations for scenario ───────────────────
        st.subheader("🤖 AI Recommendations for This Scenario")

        if st.button("Get Gemini Recommendations",
                     key="sim_gemini"):

            cargo = one_c['type'].mode()[0]

            scenario_context = {
                "🌪️ Sudden Storm Formation": (
                    f"A sudden storm formed with "
                    f"{sim_params['wave_spike']}m waves and "
                    f"{sim_params['wind_spike']} kph winds "
                    f"for {sim_params['affected_days']} days."
                ),
                "⚓ Port Congestion — Rough Handling": (
                    f"Port {sim_params['congested_port']} "
                    f"is severely congested causing "
                    f"{sim_params['shock_multiplier']}x "
                    f"increase in shock during handling."
                ),
                "🌡️ Refrigeration System Failure": (
                    f"Refrigeration failed on day "
                    f"{sim_params['failure_day']}, "
                    f"temperature spiked by "
                    f"+{sim_params['temp_spike']}°C "
                    f"for {sim_params['recovery_days']} days."
                ),
                "📦 Fragile Cargo Shipment": (
                    f"Container now carrying "
                    f"{sim_params['cargo_type']} with "
                    f"safe shock limit of only "
                    f"{sim_params['shock_threshold']}g "
                    f"(sensitivity: "
                    f"{sim_params['sensitivity']})."
                )
            }[scenario]

            prompt = f"""
You are a shipping container safety expert analyzing
a simulated risk scenario.

Container     : {sim_container}
Original cargo: {cargo}
Scenario      : {scenario}
Details       : {scenario_context}

IMPACT:
Original anomalies : {orig_risk}
Simulated anomalies: {sim_risk}
Max shock original : {orig_hist['actual'].max():.3f}g
Max shock simulated: {sim_hist['actual'].max():.3f}g

Based on this scenario provide:

1. SCENARIO IMPACT
   How serious is this scenario for this container?

2. IMMEDIATE ACTIONS (within 24 hours)
   What must be done RIGHT NOW?

3. PREVENTIVE MEASURES
   How to prevent or minimize damage?

4. SPECIFIC RECOMMENDATIONS
   Give 3-4 specific actions like:
   - "{scenario} + {cargo} cargo → [specific action]"
   - "Shock increased to [X]g → [specific action]"

5. RECOVERY PLAN
   What to do after the scenario ends?

Be specific, practical and urgent in tone.
"""

            with st.spinner(
                "🤖 Gemini generating recommendations..."
            ):
                response = client.models.generate_content(
                    model   = "gemini-2.5-flash",
                    contents= prompt
                )
            st.markdown(response.text)

        st.divider()

        # ── Scenario summary comparison table ─────────────────
        st.subheader("📋 Scenario Comparison Summary")

        port_scores_local = port_risk_score(df)
        avg_port_risk = port_scores_local['risk_score'].mean()

        summary_data = {
            "Metric"           : [
                "Total Anomalies",
                "Max Shock (g)",
                "Avg Shock (g)",
                "Anomaly Rate (%)",
                "Avg Port Risk"
            ],
            "Original"         : [
                len(orig_anomalies),
                f"{orig_hist['actual'].max():.3f}",
                f"{orig_hist['actual'].mean():.3f}",
                f"{len(orig_anomalies)/len(orig_hist)*100:.1f}%",
                f"{avg_port_risk:.1f}/100"
            ],
            "Simulated"        : [
                len(sim_anomalies),
                f"{sim_hist['actual'].max():.3f}",
                f"{sim_hist['actual'].mean():.3f}",
                f"{len(sim_anomalies)/len(sim_hist)*100:.1f}%",
                f"{avg_port_risk:.1f}/100"
            ],
            "Change"           : [
                f"+{len(sim_anomalies)-len(orig_anomalies)}",
                f"+{sim_hist['actual'].max()-orig_hist['actual'].max():.3f}",
                f"+{sim_hist['actual'].mean()-orig_hist['actual'].mean():.3f}",
                f"+{(len(sim_anomalies)-len(orig_anomalies))/len(orig_hist)*100:.1f}%",
                "—"
            ]
        }

        st.dataframe(
            pd.DataFrame(summary_data),
            use_container_width=True
        )
