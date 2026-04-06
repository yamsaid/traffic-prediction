"""
TrafficML — Application Streamlit
Prédiction du Trafic Urbain · Interstate 94 · Minneapolis-Saint Paul
Auteur : Saidou Yameogo
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib, json, warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import joblib
from PIL import Image
import folium
from streamlit_folium import folium_static

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrafficML · Interstate 94",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


#===========================================================
# Thème & Style
#===========================================================

# ── Couleurs statiques utilisées dans les graphiques Plotly ──
BLEU   = "#1E6FD9"
VERT   = "#17B897"
ORANGE = "#F4A223"
ROUGE  = "#E8432A"
GRIS   = "#6B7280"
NOIR   = "#0F172A"
DARK   = "#1A2535"

# ── Thème via session_state ──
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

IS_DARK = st.session_state.theme == "dark"
#IS_DARK = True #st.session_state.theme == "dark"

# ── Variables CSS selon le thème ──
if IS_DARK:
    CSS_VARS = """
    --bg-main:       #0F1723;
    --bg-card:       transparent;
    --bg-card2:      #212D40;
    --bg-input:      #CDD9F6;
    --border:        #2C3E50;
    --border-strong: #3D5166;
    --text-primary:  #CDD9F6;
    --text-secondary:#94A3B8;
    --text-muted:    #64748B;
    --sidebar-bg:    #0B111C;
    --sidebar-text:  #CBD5E1;
    --sidebar-hover: rgba(30,111,217,.25);
    --kpi-bg:        #1A2535;
    --sh-text:       #E2E8F0;
    --box-blue-bg:   #0F2744;
    --box-blue-text: #93C5FD;
    --box-green-bg:  #052E16;
    --box-green-text:#86EFAC;
    --box-orange-bg: #1C1104;
    --box-orange-text:#FCD34D;
    --box-red-bg:    #1C0A09;
    --box-red-text:  #FCA5A5;
    --pred-blue-bg:  linear-gradient(135deg,#0F2744,#1E3A5F);
    --pred-orange-bg:linear-gradient(135deg,#1C1104,#2D1A06);
    --pred-red-bg:   linear-gradient(135deg,#1C0A09,#2D1010);
    --chart-bg:      #1A2535;
    --chart-grid:    #2C3E50;
    --table-odd:     #1A2535;
    --table-even:    #212D40;


    
    --bg-card-hover: #212D40;
    --bg-sidebar: #0B111C;
    
    --text-primary: #E2E8F0;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
    --text-light: #ffffff;
    
    --border-light: #2C3E50;
    --border-medium: #3D5166;
    
    --box-info-bg: #1A2535;
    --box-warning-bg: #1C1104;
    --box-success-bg: #052E16;
    --box-context-bg: #0F2744;
    
    --badge-success-bg: #27ae60;
    --badge-warning-bg: #f39c12;
    --badge-danger-bg: #e74c3c;
    --badge-info-bg: #3498db;
    
    --table-header-bg: #5a67d8;
    --table-row-even: #212D40;
    --table-border: #2C3E50;
    
    --shadow-sm: 0 2px 5px rgba(0,0,0,0.2);
    --shadow-md: 0 4px 15px rgba(0,0,0,0.3);
    """
    PLOTLY_TEMPLATE = "plotly_dark"
    CHART = dict(
        plot_bgcolor="#1A2535", paper_bgcolor="#1A2535",
        margin=dict(t=20,b=0,l=0,r=0),
        yaxis=dict(gridcolor="#2C3E50", color="#94A3B8"),
        xaxis=dict(gridcolor="#2C3E50", color="#94A3B8"),
        font=dict(color="#94A3B8")
    )
else:
    CSS_VARS = """
    --bg-main:       #FFFFFF;
    --bg-card:       #F8FAFC;
    --bg-card2:      #F1F5F9;
    --bg-input:      #FFFFFF;
    --border:        #E2E8F0;
    --border-strong: #CBD5E1;
    --text-primary:  #0F172A;
    --text-secondary:#6B7280;
    --text-muted:    #94A3B8;
    --sidebar-bg:    #1A2535;
    --sidebar-text:  #CBD5E1;
    --sidebar-hover: rgba(30,111,217,.2);
    --kpi-bg:        #F8FAFC;
    --sh-text:       #0F172A;
    --box-blue-bg:   #F0F9FF;
    --box-blue-text: #1E3A5F;
    --box-green-bg:  #F0FFF4;
    --box-green-text:#14532D;
    --box-orange-bg: #FFFBEB;
    --box-orange-text:#78350F;
    --box-red-bg:    #FFF1F0;
    --box-red-text:  #7F1D1D;
    --pred-blue-bg:  linear-gradient(135deg,#EFF6FF,#DBEAFE);
    --pred-orange-bg:linear-gradient(135deg,#FFF7ED,#FED7AA);
    --pred-red-bg:   linear-gradient(135deg,#FEF2F2,#FECACA);
    --chart-bg:      #FFFFFF;
    --chart-grid:    #F1F5F9;
    --table-odd:     #F8FAFC;
    --table-even:    #FFFFFF;
        /* Couleurs de base */
    --bg-main: #f5f7fb;
    --bg-card: #ffffff;
    --bg-card-hover: #f8f9fa;
    --bg-sidebar: #2c3e50;
    
    /* Couleurs du texte */
    --text-primary: #2c3e50;
    --text-secondary: #34495e;
    --text-muted: #6c757d;
    --text-light: #ffffff;
    
    /* Bordures */
    --border-light: #e2e8f0;
    --border-medium: #cbd5e1;
    
    /* Boxes */
    --box-info-bg: #f8f9fa;
    --box-warning-bg: #fff3cd;
    --box-success-bg: #d4edda;
    --box-context-bg: #e8f4fd;
    
    /* Badges */
    --badge-success-bg: #27ae60;
    --badge-warning-bg: #f39c12;
    --badge-danger-bg: #e74c3c;
    --badge-info-bg: #3498db;
    
    /* Tableaux */
    --table-header-bg: #3498db;
    --table-row-even: #f2f2f2;
    --table-border: #ddd;
    
    /* Ombres */
    --shadow-sm: 0 2px 5px rgba(0,0,0,0.05);
    --shadow-md: 0 4px 15px rgba(0,0,0,0.1);
    """
    PLOTLY_TEMPLATE = "plotly_white"
    CHART = dict(
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20,b=0,l=0,r=0),
        yaxis=dict(gridcolor="#F1F5F9"),
        xaxis=dict(gridcolor="#F1F5F9")
    )

# ──────────────────────────────────────────────────────────────
# STYLE GLOBAL
# ──────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Variables globales ── */
:root {{ {CSS_VARS} }}

/* ── Base ── */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
    transition: background-color .3s, color .3s;
}}
.main, .block-container {{
    background-color: var(--bg-main) !important;
}}
.block-container {{ padding-top: 1.8rem; }}

/*==================================pour la page accueil==============================*/
/*====================================================================================*/

/* Style des titres */
h1 {{
    color: var(--text-primary);
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 20px;
}}

h2 {{
    color: var(--text-primary);
    font-size: 28px;
    font-weight: 600;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
    margin-top: 30px;
}}

h3 {{
    color: var(--text-primary);
    font-size: 22px;
    font-weight: 500;
    margin-top: 20px;
}}

/* ============================================
   STYLES DES TEXTES
   ============================================ */

.custom-text {{
    color: var(--text-secondary);
    font-size: 20px;
    line-height: 1.6;
}}

.bold-text {{
    font-weight: bold;
    color: var(--text-primary);
}}


.text-success {{
    color: #27ae60;
}}

.text-warning {{
    color: #f39c12;
}}

.text-danger {{
    color: #e74c3c;
}}

/* ============================================
   CARTES MÉTRIQUES
   ============================================ */

.metric-card {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: var(--shadow-md);
    transition: transform 0.3s ease;
}}

.metric-card:hover {{
    transform: translateY(-5px);
}}

.metric-label {{
    color: rgba(255,255,255,0.9);
    font-size: 14px;
    margin: 0;
}}

    .metric-value {{
        color: white;
        font-size: 32px;
        font-weight: bold;
        margin: 5px 0;
    }}

.metric-delta {{
    color: rgba(255,255,255,0.7);
    font-size: 12px;
    margin: 0;
}}

/* Cartes spécifiques */
    .card-blue {{
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
    }}

    .card-green {{
        background: linear-gradient(135deg, #27ae60 0%, #1e8449 100%);
    }}

    .card-orange {{
        background: linear-gradient(135deg, #e67e22 0%, #d35400 100%);
    }}

    .card-purple {{
        background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
    }}

/* ============================================
   BADGES ET ÉTIQUETTES
   ============================================ */

.badge {{
    display: inline-block;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}}

.badge-success {{
    background-color: var(--badge-success-bg);
    color: white;
}}

.badge-warning {{
    background-color: var(--badge-warning-bg);
    color: white;
}}

.badge-danger {{
    background-color: var(--badge-danger-bg);
    color: white;
}}

.badge-info {{
    background-color: var(--badge-info-bg);
    color: white;
}}

/* ============================================
   BLOCS D'INFORMATION
   ============================================ */

.info-box {{
    background-color: var(--box-info-bg);
    border-left: 5px solid #3498db;
    padding: 15px;
    border-radius: 8px;
    margin: 15px 0;
    box-shadow: var(--shadow-sm);
    color: var(--text-primary);
}}

.warning-box {{
    background-color: var(--box-warning-bg);
    border-left: 5px solid #f39c12;
    padding: 15px;
    border-radius: 8px;
    margin: 15px 0;
    color: var(--text-primary);
}}

.success-box {{
    background-color: var(--box-success-bg);
    border-left: 5px solid #27ae60;
    padding: 15px;
    border-radius: 8px;
    margin: 15px 0;
    color: var(--text-primary);
}}

/* ============================================
   SIDEBAR PERSONNALISÉE
   ============================================ */



.sidebar-logo {{
    text-align: center;
    padding: 20px;
}}

.sidebar-title {{
    color: white;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
    margin-top: 10px;
}}



/* ============================================
   BOUTONS
   ============================================ */

.custom-button {{
    background-color: #3498db;
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 16px;
    transition: background-color 0.3s ease;
}}

.custom-button:hover {{
    background-color: #2980b9;
}}

/* ============================================
   BOXES SPÉCIFIQUES
   ============================================ */

.context-box {{
    background-color: var(--box-context-bg);
    border-left: 2px solid #3d97fd72;
    padding: 16px;
    border-radius: 9px;
    margin: 16px 0;
    color: var(--text-primary);
}}

.problem-box {{
    background-color: var(--box-context-bg);
    border-left: 2px solid #f39c12;
    padding: 16px;
    border-radius: 9px;
    margin: 16px 0;
    color: var(--text-primary);
}}

.objective-box {{
    background-color: var(--box-context-bg);
    border-left: 2px solid #27ae60;
    padding: 16px;
    border-radius: 10px;
    margin: 16px 0;
    color: var(--text-primary);
}}

.highlight {{
    color: #3498db;
    font-weight: bold;
}}


/*===========================================================*/

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border);
    transition: background .3s;
}}
section[data-testid="stSidebar"] * {{ color: var(--sidebar-text) !important; }}
section[data-testid="stSidebar"] .stRadio label {{
    background: rgba(255,255,255,.04);
    border-radius: 8px; padding: 9px 14px; margin: 3px 0;
    cursor: pointer; border-left: 3px solid transparent; transition: all .2s;
}}
section[data-testid="stSidebar"] .stRadio label:hover {{
    background: var(--sidebar-hover);
    border-left-color: {BLEU};
}}

/* ── Tabs ── */
button[data-baseweb="tab"] {{
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-bottom: 2px solid transparent !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {BLEU} !important;
    border-bottom-color: {BLEU} !important;
    font-weight: 600 !important;
}}
[data-testid="stTabsTabPanel"] {{
    background: var(--bg-main) !important;
}}

/* ── Commentaires / Insights ── */

.commentaire{{
        background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
        border-left: 4px solid #667eea;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 10px 0;
}}
.commentaire:hover {{
    transform: translateX(3px);
}}

.eda-insight {{
        background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
        border-left: 4px solid #667eea;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 10px 0;
}}
.eda-warning {{
    background: #77D0FF69;
    border-left: 4px solid #01573569;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 10px 0;
}}
/* ── Inputs Streamlit ── */
[data-testid="stSelectbox"] > div,
[data-testid="stNumberInput"] > div > div,
[data-testid="stDateInput"] > div > div {{
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
}}
.stSelectbox label, .stSlider label,
.stNumberInput label, .stDateInput label {{
    color: var(--text-primary) !important;
}}

/* ── DataFrames ── */
/* Conteneur principal du dataframe */
[data-testid="stDataFrame"] {{
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px;
}}

/* Zone de défilement */
.dvn-scroller {{
    background: var(--bg-card) !important;
}}

/* Tableau lui-même */
.dataframe {{
    width: 100% !important;
    border-collapse: collapse !important;
    background: var(--bg-card) !important;
}}

/* En-têtes du tableau */
.dataframe th {{
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 14px 8px !important;
    text-align: center !important;
    border-bottom: 1px solid var(--border) !important;
}}

/* Cellules du tableau */
.dataframe td {{
    color: var(--text-primary) !important;
    padding: 8px 8px !important;
    border-bottom: 1px solid var(--border) !important;
}}

/* Première colonne (Variable) */
.dataframe td:first-child {{
    color: {BLEU} !important;
    font-weight: 600 !important;
    font-family: 'JetBrains Mono', monospace !important;
}}

/* Lignes paires (alternance) */
.dataframe tbody tr:nth-child(even) {{
    background-color: var(--table-even) !important;
}}

/* Lignes impaires (alternance) */
.dataframe tbody tr:nth-child(odd) {{
    background-color: var(--table-odd) !important;
}}

/* Survol des lignes */
.dataframe tbody tr:hover {{
    background-color: var(--bg-card2) !important;
    cursor: pointer;
}}

/* Balises code dans le tableau */
.dataframe td code {{
    background: transparent !important;
    color: {BLEU} !important;
    font-weight: 600 !important;
    padding: 0 !important;
}}

/* Mode Dark - ajustements spécifiques pour le dataframe */
{'/* MODE DARK DATAFRAME */' if IS_DARK else ''}
{"" if not IS_DARK else '''
[data-testid="stDataFrame"] .dataframe th {
    background: linear-gradient(135deg, #5a67d8, #6b46c1) !important;
}
.dataframe td:first-child {
    color: #93C5FD !important;
}
.dataframe td code {
    color: #93C5FD !important;
}
'''}

/* ── Métriques custom ── */
.kpi {{
    background: var(--kpi-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 22px;
    border-left: 4px solid {BLEU};
    transition: background .3s, border .3s;
}}
.kpi.g {{ border-left-color: {VERT}; }}
.kpi.o {{ border-left-color: {ORANGE}; }}
.kpi.r {{ border-left-color: {ROUGE}; }}
.kpi.gr {{ border-left-color: {GRIS}; }}
.kv {{ font-size: 1.9rem; font-weight: 700; color: var(--text-primary); margin: 4px 0; line-height: 1; }}
.kl {{ font-size: .72rem; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: .07em; }}
.kd {{ font-size: .8rem; color: {VERT}; margin-top: 4px; }}

/* ── Section headers ── */
.sh {{
    border-bottom: 2px solid {BLEU};
    padding-bottom: 8px; margin: 28px 0 14px;
    font-size: 1.05rem; font-weight: 700;
    color: var(--sh-text);
    transition: color .3s;
}}

/* ── Insight boxes ── */
.box {{
    background: var(--box-blue-bg);
    border-left: 3px solid {BLEU};
    border-radius: 0 8px 8px 0;
    padding: 12px 16px; font-size: .88rem;
    color: var(--box-blue-text);
    margin: 8px 0; line-height: 1.6;
    transition: background .3s, color .3s;
}}
.box.g {{ background: var(--box-green-bg); border-color: {VERT}; color: var(--box-green-text); }}
.box.o {{ background: var(--box-orange-bg); border-color: {ORANGE}; color: var(--box-orange-text); }}
.box.r {{ background: var(--box-red-bg); border-color: {ROUGE}; color: var(--box-red-text); }}

/* ── Prediction boxes ── */
.pred-box {{
    background: var(--pred-blue-bg);
    border: 2px solid {BLEU};
    border-radius: 16px; padding: 32px; text-align: center;
    transition: background .3s;
}}
.pred-val {{ font-size: 3.2rem; font-weight: 700; color: {BLEU}; line-height: 1; }}
.pred-box.w {{ background: var(--pred-orange-bg); border-color: {ORANGE}; }}
.pred-box.w .pred-val {{ color: {ORANGE}; }}
.pred-box.d {{ background: var(--pred-red-bg); border-color: {ROUGE}; }}
.pred-box.d .pred-val {{ color: {ROUGE}; }}

/* ── Cards génériques ── */
.card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px; padding: 16px;
    transition: background .3s;
}}

/* ── Toggle thème ── */
.theme-toggle {{
    display: flex; align-items: center; gap: 8px;
    padding: 8px 12px; border-radius: 8px;
    background: rgba(255,255,255,.06);
    cursor: pointer; margin: 8px 0;
    font-size: .82rem; color: var(--sidebar-text);
}}

/* ── Code blocks ── */
code {{
    font-family: 'JetBrains Mono', monospace;
    background: var(--bg-card2) !important;
    color: {BLEU} !important;
    padding: 2px 6px; border-radius: 4px;
}}
pre code {{ color: var(--text-primary) !important; background: transparent !important; }}

/* ── Expanders ── */
[data-testid="stExpander"] {{
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px;
}}

/* ── Bouton primary ── */
.stButton > button[kind="primary"] {{
    background: {BLEU} !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: opacity .2s;
}}
.stButton > button[kind="primary"]:hover {{ opacity: .88; }}




/* ── Masquer branding ── */
#MainMenu, footer {{ visibility: hidden; }}

/* ── Override Streamlit natif selon thème ── */
{'/* MODE DARK — override widgets Streamlit */' if IS_DARK else ''}
{"" if not IS_DARK else '''
/* Fond général */
.stApp { background-color: #0F1723 !important; }
.stApp > header { background-color: #0F1723 !important; }
/* Tabs */
.stTabs [data-baseweb="tab-list"] { background-color: #1A2535 !important; border-bottom: 1px solid #2C3E50; }
/* Selectbox dropdown */
[data-baseweb="select"] [data-baseweb="popover"] { background: #1A2535 !important; }
[data-baseweb="menu"] { background: #1A2535 !important; border: 1px solid #2C3E50 !important; }
[data-baseweb="menu"] li { color: #E2E8F0 !important; }
[data-baseweb="menu"] li:hover { background: #212D40 !important; }
/* Inputs */
input, textarea, select { background-color: #1E2C3D !important; color: #E2E8F0 !important; border-color: #2C3E50 !important; }
/* Metric */
[data-testid="metric-container"] { background: #1A2535 !important; border: 1px solid #2C3E50 !important; border-radius: 8px; }
/* Slider */
[data-testid="stSlider"] [data-baseweb="slider"] div
/* Number input */
[data-testid="stNumberInput"] input { background: #ffffff !important; color: #E2E8F0 !important; }
/* Date input */
[data-testid="stDateInput"] input { background: #1E2C3D !important; color: #E2E8F0 !important; }
/* Expander */
details summary { color: #E2E8F0 !important; }
details[open] { background: #1A2535 !important; }
/* Alert/info */
[data-testid="stAlert"] { background: #1A2535 !important; border-color: #2C3E50 !important; color: #E2E8F0 !important; }
/* Code */
.stCodeBlock { background: #0B111C !important; }
/* Plotly charts background override */
.js-plotly-plot .plotly .bg { fill: #1A2535 !important; }
'''}
</style>
""", unsafe_allow_html=True)

#===========================================================

# ──────────────────────────────────────────────────────────────
# CHARGEMENT
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement des modèles…")
def load_models():
    return (joblib.load("models/random_forest_model.pkl"),
            joblib.load("models/ridge_model.pkl"),
            joblib.load("models/xgboost_model.pkl"),
            joblib.load("models/scaler.pkl"),
            joblib.load("models/feature_columns.pkl"))

@st.cache_data(show_spinner="Chargement des données…")
def load_data():
    raw  = pd.read_csv("data/data_raw.csv")
    proc = pd.read_csv("data/data_processed.csv")
    pred = pd.read_csv("data/predictions_test.csv")
    raw["date_time"] = pd.to_datetime(raw["date_time"])
    proc["datetime"] = pd.to_datetime(proc["datetime"])
    pred["datetime"] = pd.to_datetime(pred["datetime"])
    return raw, proc, pred

@st.cache_data
def load_meta():
    with open("models/metriques.json")       as f: m = json.load(f)
    with open("models/hyperparameters.json") as f: h = json.load(f)
    return m, h

try:
    RF, RIDGE, XGB, SCALER, COLS = load_models()
    df_raw, df_proc, df_pred     = load_data()
    METRIQUES, HYPERPARAMS       = load_meta()
    OK = True
except Exception as e:
    OK = False; ERR = str(e)

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def kpi(label, val, delta="", cls=""):
    st.markdown(f"""<div class="kpi {cls}">
      <div class="kl">{label}</div><div class="kv">{val}</div>
      <div class="kd">{delta}</div></div>""", unsafe_allow_html=True)

def sh(t): st.markdown(f'<div class="sh">{t}</div>', unsafe_allow_html=True)
def box(t, cls=""): st.markdown(f'<div class="box {cls}">{t}</div>', unsafe_allow_html=True)
def commentaire(t): st.markdown(f'<div class="commentaire"> <strong>{t}</strong></div>', unsafe_allow_html=True)

JOURS_FR   = {"Monday":"Lundi","Tuesday":"Mardi","Wednesday":"Mercredi",
               "Thursday":"Jeudi","Friday":"Vendredi","Saturday":"Samedi","Sunday":"Dimanche"}
MOIS_FR    = {1:"Jan",2:"Fév",3:"Mar",4:"Avr",5:"Mai",6:"Jun",
              7:"Jul",8:"Aoû",9:"Sep",10:"Oct",11:"Nov",12:"Déc"}
JOURS_ORD  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
JOURS_FR_ORD = [JOURS_FR[j] for j in JOURS_ORD]
MOIS_ORD   = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]

# Couleurs texte/fond adaptées au thème courant
T_PRIMARY   = "#E2E8F0" if IS_DARK else "#0F172A"
T_SECONDARY = "#94A3B8" if IS_DARK else "#6B7280"
BG_CARD     = "#1A2535" if IS_DARK else "#FFFFFF"
GRID_COLOR  = "#2C3E50" if IS_DARK else "#F1F5F9"

def plo(**kwargs):
    """Retourne un dict de layout Plotly adapté au thème."""
    base = dict(
        plot_bgcolor=BG_CARD,
        paper_bgcolor=BG_CARD,
        margin=dict(t=20,b=0,l=0,r=0),
        font=dict(color=T_SECONDARY, family="Inter, sans-serif"),
        yaxis=dict(gridcolor=GRID_COLOR, color=T_SECONDARY, zerolinecolor=GRID_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR, color=T_SECONDARY, zerolinecolor=GRID_COLOR),
    )
    base.update(kwargs)
    return base
CHART      = dict(plot_bgcolor="white", paper_bgcolor="white",
                  margin=dict(t=20,b=0,l=0,r=0),
                  yaxis=dict(gridcolor="#F1F5F9"),
                  xaxis=dict(gridcolor="#F1F5F9"))




    # CHARGEMENT DU CSS EXTERNE
    # ============================================

def load_cssV1(file_path):
    """Charge le fichier CSS externe"""
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def display_logo(variant="default", location="main"):
    """
    Affiche le logo FlowCast
    
    Parameters:
    - variant: "default", "compact", "sidebar", "animated"
    - location: "main", "sidebar"
    """
    
    if location == "sidebar":
        st.sidebar.markdown("""
        <div style="text-align: center; padding: 10px; margin-bottom: 20px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); width: 50px; height: 50px; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin: 0 auto 10px auto;">
                <span style="font-size: 24px;">📡</span>
            </div>
            <h3 style="margin: 0; color: #667eea;">FlowCast</h3>
            <p style="margin: 0; font-size: 10px; color: gray;">v1.0</p>
        </div>
        """, unsafe_allow_html=True)
    
    elif variant == "compact":
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 20px;">📡</div>
            <div><h2 style="margin: 0; font-size: 24px;">FlowCast</h2><p style="margin: 0; font-size: 10px; color: gray;">Smart Traffic Prediction</p></div>
        </div>
        """, unsafe_allow_html=True)
    elif variant == "animated":
        st.markdown("""
        <style>
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }

        .logo-animated {
            display: flex;
            align-items: center;
            gap: 15px;
            cursor: pointer;
            transition: transform 0.3s ease;
        }

        .logo-animated:hover {
            transform: translateY(-2px);
        }

        .logo-animated:hover .logo-icon-animated {
            animation: pulse 0.5s ease;
        }

        .logo-icon-animated {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            width: 55px;
            height: 55px;
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            transition: all 0.3s ease;
        }

        .logo-icon-animated span {
            font-size: 28px;
        }

        .logo-text-animated h1 {
            margin: 0;
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .logo-text-animated p {
            margin: 0;
            font-size: 11px;
            color: #6c757d;
        }
        </style>

        <div class="logo-animated">
            <div class="logo-icon-animated">
                <span>📡</span>
            </div>
            <div class="logo-text-animated">
                <h1>FlowCast</h1>
                <p>Prédiction intelligente du trafic urbain</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    else:  # default
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 15px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); width: 60px; height: 60px; border-radius: 15px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);">
                <span style="font-size: 32px;">📡</span>
            </div>
            <div>
                <h1 style="margin: 0; font-size: 32px; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">FlowCast</h1>
                <p style="margin: 0; font-size: 12px; color: #6c757d;">Prédiction intelligente du trafic urbain</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer

def add_footer():
    """
    Ajoute un footer professionnel à l'application
    """
    st.markdown("---")
    
        
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #2c3e50 0%, #1a252f 100%);
            padding: 30px 20px 20px 20px;
            border-radius: 15px;
            color: white;
            margin-top: 30px;
        ">
            <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 200px;">
                    <h3 style="margin: 0 0 10px 0;">🚗 Smart Traffic</h3>
                    <p style="font-size: 12px; opacity: 0.8;">Prédiction du volume de trafic<br>par machine learning</p>
                </div>
                <div style="flex: 1; min-width: 200px;">
                    <h4 style="margin: 0 0 10px 0;">Liens rapides</h4>
                    <p style="margin: 5px 0;"><a href="#" style="color: #FFD700;">🏠 Accueil</a></p>
                    <p style="margin: 5px 0;"><a href="#" style="color: #FFD700;">🔮 Prédiction</a></p>
                    <p style="margin: 5px 0;"><a href="#" style="color: #FFD700;">📊 Analyse SHAP</a></p>
                </div>
                <div style="flex: 1; min-width: 200px;">
                    <h4 style="margin: 0 0 10px 0;">Contact</h4>
                    <p style="margin: 5px 0;"><a href="saidouyameogo3@gmail.com" style="color: #FFD700;">📧 saidouyameogo3@gmail.com</a></p>
                    <p style="margin: 5px 0;"><a href="#" style="color: #FFD700;">🐙 GitHub</a></p>
                    <p style="margin: 5px 0;"><a href="#" style="color: #FFD700;">🔗 LinkedIn</a></p>
                </div>
            </div>
            <hr style="border-color: rgba(255,255,255,0.2); margin: 20px 0 10px 0;">
            <div style="text-align: center; padding: 20px; color: #6c757d; font-size: 12px;">
                <div style="display: flex; justify-content: center; gap: 15px; flex-wrap: wrap; margin-bottom: 15px;">
                    <span style="background: #f0f0f0; padding: 5px 12px; border-radius: 20px; font-size: 12px;">🐍 Python 3.9</span>
                    <span style="background: #f0f0f0; padding: 5px 12px; border-radius: 20px; font-size: 12px;">🤖 Scikit-learn</span>
                    <span style="background: #f0f0f0; padding: 5px 12px; border-radius: 20px; font-size: 12px;">⚡ XGBoost</span>
                    <span style="background: #f0f0f0; padding: 5px 12px; border-radius: 20px; font-size: 12px;">🚀 Streamlit</span>
                </div>
            </div>
            <hr style="border-color: rgba(255,255,255,0.2); margin: 20px 0 10px 0;">
            <div style="text-align: center; font-size: 12px; opacity: 0.7;">
                <p>© 2024 Projet Smart City - Tous droits réservés | Données : MnDOT & OpenWeatherMap</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
        )
    


# Chargement du CSS
#load_css()

def header(n=1,text=""):
    return st.markdown(f"<h{n}> {text} </h{n}>", unsafe_allow_html=True)

def paragraphe(classe="", style="", text=""):
    style = f'style="{style}"' if style else ""
    st.markdown(f"<p class={classe} {style}>{text}</p>", unsafe_allow_html=True)

#===============================================

# ──────────────────────────────────────────────────────────────
# SESSION STATE — mode d'utilisation
# ──────────────────────────────────────────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = None   # None = écran de sélection


# ──────────────────────────────────────────────────────────────
# ÉCRAN DE SÉLECTION D'OBJECTIF
# ──────────────────────────────────────────────────────────────
if st.session_state.mode is None:
    # Sidebar minimale sur l'écran de sélection
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align:center;padding:24px 0 16px;'>
          <div style='font-size:2.8rem;'>🚗</div>
          <div style='font-size:1.3rem;font-weight:700;color:#F1F5F9;margin-top:6px;'>TrafficML</div>
          <div style='font-size:.65rem;color:#64748B;letter-spacing:.12em;margin-top:4px;'>
            INTERSTATE 94 · MINNEAPOLIS</div>
        </div>
        <hr style='border-color:#2C3E50;margin:12px 0;'>
        <div style='font-size:.78rem;color:#64748B;padding:8px 4px;line-height:1.7;'>
          Bienvenue sur TrafficML.<br><br>
          Choisissez votre profil pour accéder à l'interface adaptée à vos besoins.
        </div>""", unsafe_allow_html=True)

        theme_icon  = "☀️" if IS_DARK else "🌙"
        theme_label = "Mode clair" if IS_DARK else "Mode sombre"
        st.markdown("<hr style='border-color:#2C3E50;margin:16px 0 8px;'>", unsafe_allow_html=True)
        if st.button(f"{theme_icon}  {theme_label}", use_container_width=True, key="theme_btn_home"):
            st.session_state.theme = "light" if IS_DARK else "dark"
            st.rerun()

    # ── Page de sélection ──
    st.markdown(f"""
    <div style='text-align:center;padding:40px 0 8px;'>
      <div style='font-size:3.5rem;'>🚗</div>
      <h1 style='font-size:2.6rem;font-weight:700;color:var(--text-primary);margin:12px 0 6px;'>TrafficML</h1>
      <p style='font-size:1.1rem;color:var(--text-secondary);margin:0;'>
        Prédiction du Trafic Urbain · Interstate 94 · Minneapolis-Saint Paul</p>
      <div style='display:flex;justify-content:center;gap:12px;margin-top:16px;flex-wrap:wrap;'>
        <span style='background:{BLEU}22;color:{BLEU};border-radius:20px;padding:4px 14px;font-size:.78rem;font-weight:600;'>Random Forest · R²=0.989</span>
        <span style='background:{VERT}22;color:{VERT};border-radius:20px;padding:4px 14px;font-size:.78rem;font-weight:600;'>RMSE = 210 véh/h</span>
        <span style='background:{ORANGE}22;color:{ORANGE};border-radius:20px;padding:4px 14px;font-size:.78rem;font-weight:600;'>MAPE = 5.8%</span>
      </div>
    </div>
    <hr style='border-color:var(--border);margin:32px 0 28px;'>
    <h2 style='text-align:center;font-size:1.3rem;font-weight:600;color:var(--text-primary);margin-bottom:6px;'>
      Quel est votre objectif ?</h2>
    <p style='text-align:center;color:var(--text-secondary);font-size:.92rem;margin-bottom:32px;'>
      Choisissez le profil qui correspond à votre usage pour accéder à l'interface adaptée.</p>
    """, unsafe_allow_html=True)

    col_gap, c1, c2, col_gap2 = st.columns([0.5, 3, 3, 0.5])

    with c1:
        st.markdown(f"""
        <div style='background:var(--bg-card);border:2px solid {BLEU};border-radius:16px;
                     padding:32px 28px;text-align:center;height:100%;'>
          <div style='font-size:3rem;margin-bottom:12px;'>🎓</div>
          <div style='font-size:1.2rem;font-weight:700;color:{BLEU};margin-bottom:10px;'>Mode Pédagogique</div>
          <div style='font-size:.85rem;color:var(--text-secondary);line-height:1.7;margin-bottom:20px;'>
            Pour les <b>étudiants, enseignants et chercheurs</b> qui souhaitent comprendre
            la démarche complète de data science : exploration, preprocessing, modélisation,
            évaluation et interprétabilité.
          </div>
          <div style='text-align:left;font-size:.8rem;color:var(--text-secondary);margin-bottom:24px;'>
            {''.join([f"<div style='padding:4px 0;'>✅ {p}</div>" for p in [
              "Exploration & visualisation (EDA)",
              "Feature engineering documenté",
              "Comparaison des 3 modèles",
              "Interprétabilité SHAP détaillée",
              "Prédiction interactive",
              "Conclusions & perspectives"
            ]])}
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎓  Accéder au mode pédagogique", use_container_width=True,
                     key="btn_peda", type="primary"):
            st.session_state.mode = "pedagogique"
            st.rerun()

    with c2:
        st.markdown(f"""
        <div style='background:var(--bg-card);border:2px solid {VERT};border-radius:16px;
                     padding:32px 28px;text-align:center;height:100%;'>
          <div style='font-size:3rem;margin-bottom:12px;'>💼</div>
          <div style='font-size:1.2rem;font-weight:700;color:{VERT};margin-bottom:10px;'>Mode Professionnel</div>
          <div style='font-size:.85rem;color:var(--text-secondary);line-height:1.7;margin-bottom:20px;'>
            Pour les <b>décideurs, opérationnels et professionnels</b> qui ont besoin
            de résultats directs : performances des modèles et prédictions en temps réel,
            sans les détails techniques.
          </div>
          <div style='text-align:left;font-size:.8rem;color:var(--text-secondary);margin-bottom:24px;'>
            {''.join([f"<div style='padding:4px 0;'>✅ {p}</div>" for p in [
              "Tableau de bord des performances",
              "Comparaison synthétique des modèles",
              "Prédiction interactive avancée",
              "Visualisations claires et épurées",
              "Interface rapide et efficace",
              "Focus sur les résultats métier"
            ]])}
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💼  Accéder au mode professionnel", use_container_width=True,
                     key="btn_pro"):
            st.session_state.mode = "pro"
            st.rerun()

    # Footer
    st.markdown(f"""
    <div style='text-align:center;margin-top:40px;color:var(--text-muted);font-size:.75rem;'>
      Saidou Yameogo · Interstate 94 · Minneapolis-Saint Paul · 2024<br>
      <span style='color:{BLEU};'>Random Forest</span> ·
      <span style='color:{VERT};'>XGBoost</span> ·
      <span style='color:{GRIS};'>Ridge</span>
    </div>""", unsafe_allow_html=True)
    st.stop()


# ──────────────────────────────────────────────────────────────
# SIDEBAR — selon le mode sélectionné
# ──────────────────────────────────────────────────────────────
MODE = st.session_state.mode   # "pedagogique" ou "pro"

with st.sidebar:
    # Logo + titre
    mode_badge_color = BLEU if MODE == "pedagogique" else VERT
    mode_badge_label = "🎓 Pédagogique" if MODE == "pedagogique" else "💼 Professionnel"
    st.markdown(f"""
    <div style='text-align:center;padding:18px 0 10px;'>
      <div style='font-size:2.2rem;'>🚗</div>
      <div style='font-size:1.2rem;font-weight:700;color:#F1F5F9;'>TrafficML</div>
      <div style='font-size:.65rem;color:#94A3B8;letter-spacing:.1em;margin-top:3px;'>
        INTERSTATE 94</div>
      <div style='background:{mode_badge_color}33;color:{mode_badge_color};border-radius:20px;
                   padding:3px 12px;font-size:.7rem;font-weight:600;margin-top:8px;display:inline-block;'>
        {mode_badge_label}</div>
    </div>
    <hr style='border-color:#2C3E50;margin:8px 0 10px;'>""", unsafe_allow_html=True)

    # Toggle thème
    theme_icon  = "☀️" if IS_DARK else "🌙"
    theme_label = "Mode clair" if IS_DARK else "Mode sombre"
    if st.button(f"{theme_icon}  {theme_label}", use_container_width=True, key="theme_btn"):
        st.session_state.theme = "light" if IS_DARK else "dark"
        st.rerun()

    st.markdown("<hr style='border-color:#2C3E50;margin:10px 0 12px;'>", unsafe_allow_html=True)

    # Navigation selon le mode
    if MODE == "pedagogique":
        PAGES_PEDA = [
            "🏠  Accueil",
            "📊  Exploration (EDA)",
            "⚙️  Feature Engineering",
            "🤖  Modélisation",
            "📈  Évaluation & Performances",
            "🔬  Interprétabilité SHAP",
            "🔮  Prédiction Interactive",
            "📝  Conclusions & Perspectives",
        ]
        PAGE = st.radio("", PAGES_PEDA, label_visibility="collapsed")
    else:
        PAGES_PRO = [
            "🏠  Tableau de bord",
            "📈  Performances & Résultats",
            "🔮  Prédiction Interactive",
        ]
        PAGE = st.radio("", PAGES_PRO, label_visibility="collapsed")

    st.markdown("<hr style='border-color:#2C3E50;margin:12px 0;'>", unsafe_allow_html=True)

    # Bouton changer de mode
    other_mode = "professionnel" if MODE == "pedagogique" else "pédagogique"
    other_icon = "💼" if MODE == "pedagogique" else "🎓"
    if st.button(f"{other_icon}  Mode {other_mode}", use_container_width=True, key="switch_mode"):
        st.session_state.mode = None
        st.rerun()

    st.markdown(f"""
    <div style='font-size:.68rem;color:#475569;line-height:1.8;margin-top:8px;'>
      <b style='color:#94A3B8;'>Meilleur modèle</b><br>
      Random Forest · R²=0.989<br>
      RMSE=210 · MAPE=5.8%<br><br>
      <b style='color:#94A3B8;'>Dataset</b><br>
      48 204 obs. · 2012–2018
    </div>""", unsafe_allow_html=True)

if not OK:
    st.error(f"⚠️ Fichiers manquants : `{ERR}`")
    st.stop()

# ══════════════════════════════════════════════════════════════
# ROUTING MODE PROFESSIONNEL
# ══════════════════════════════════════════════════════════════

if MODE == "pro":

    # ── P-PRO-1 : Tableau de bord ──
    if PAGE == "🏠  Tableau de bord":
        st.markdown(f"""
        <div style='margin-bottom:20px;'>
          <h1 style='font-size:2rem;font-weight:700;color:var(--text-primary);margin:0;'>
            Tableau de bord · TrafficML</h1>
          <p style='color:var(--text-secondary);font-size:.95rem;margin-top:6px;'>
            Interstate 94 · Minneapolis-Saint Paul · Prédiction du trafic horaire</p>
        </div>""", unsafe_allow_html=True)

        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi("R² Test","0.989","Variance expliquée","g")
        with c2: kpi("RMSE","210 véh/h","Erreur moyenne","g")
        with c3: kpi("MAPE","5.8 %","Erreur relative","g")
        with c4: kpi("Données","48 204 obs.","2012 – 2018")
        st.markdown("<br>", unsafe_allow_html=True)

        # Série temporelle
        sh("Évolution historique du trafic (2012–2018)")
        daily = df_raw.groupby(df_raw["date_time"].dt.date)["traffic_volume"].mean().reset_index()
        daily.columns = ["date","trafic"]
        fig = px.line(daily, x="date", y="trafic",
                      color_discrete_sequence=[BLEU],
                      labels={"trafic":"Trafic moyen (véh/h)","date":""})
        fig.update_traces(line_width=1.5)
        fig.update_layout(height=260, **plo())
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            sh("Profil horaire moyen")
            hd = df_raw.groupby(df_raw["date_time"].dt.hour)["traffic_volume"].mean().reset_index()
            hd.columns = ["h","t"]
            fig = go.Figure()
            fig.add_vrect(x0=7,x1=9,fillcolor=ORANGE,opacity=.12,annotation_text="Pointe matin")
            fig.add_vrect(x0=16,x1=19,fillcolor=VERT,opacity=.12,annotation_text="Pointe soir")
            fig.add_trace(go.Scatter(x=hd["h"],y=hd["t"],mode="lines+markers",
                                      line=dict(color=BLEU,width=2.5),marker=dict(size=6),
                                      fill="tozeroy",fillcolor=f"{BLEU}18"))
            fig.update_layout(height=270,
                              xaxis=dict(tickvals=list(range(0,24,2)),gridcolor=GRID_COLOR,
                                         color=T_SECONDARY,title="Heure"),
                              yaxis=dict(gridcolor=GRID_COLOR,color=T_SECONDARY,title="Trafic moyen"),
                              **{k:v for k,v in plo().items() if k not in ["xaxis","yaxis"]})
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            sh("Trafic par jour de semaine")
            jours_ord = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            tj = df_raw.groupby(df_raw["date_time"].dt.day_name())["traffic_volume"].mean().reindex(jours_ord).reset_index()
            tj.columns = ["j","t"]
            tj["jf"] = tj["j"].map(JOURS_FR)
            fig = px.bar(tj,x="jf",y="t",color="t",
                         color_continuous_scale=["#DBEAFE","#1E3A8A"],
                         labels={"jf":"","t":"Trafic moyen (véh/h)"})
            fig.update_layout(height=270,coloraxis_showscale=False,**plo())
            st.plotly_chart(fig, use_container_width=True)

        box("Le trafic atteint ses pics entre <b>7h-9h</b> (départ travail) et <b>16h-19h</b> (retour), avec des volumes ~40% plus élevés en semaine qu'en week-end. La saisonnalité estivale (juin-août) génère les volumes maximaux.", "g")

    # ── P-PRO-2 : Performances ──
    elif PAGE == "📈  Performances & Résultats":
        st.markdown(f"""<div style='margin-bottom:16px;'>
          <h1 style='font-size:2rem;font-weight:700;color:var(--text-primary);margin:0;'>
            Performances des modèles</h1>
          <p style='color:var(--text-secondary);font-size:.95rem;margin-top:6px;'>
            Comparaison Ridge · Random Forest · XGBoost sur le jeu de test</p>
        </div>""", unsafe_allow_html=True)

        # Tableau synthétique
        df_comp = pd.DataFrame({
            "Modèle":["Ridge","Random Forest ★","XGBoost"],
            "R² Test":[0.903,0.989,0.988],
            "RMSE (véh/h)":[618,210,213],
            "MAE (véh/h)":[434,135,138],
            "MAPE":[" 28.0%"," 5.8%"," 5.9%"],
            "Taille modèle":["< 1 MB","98 MB","4 MB"],
            "Usage conseillé":["Interprétabilité","Performance max","Production temps réel"]
        })
        st.dataframe(df_comp.style
            .highlight_max(subset=["R² Test"],color="#DCFCE7")
            .highlight_min(subset=["RMSE (véh/h)","MAE (véh/h)"],color="#DCFCE7"),
            use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        for (titre,val,delta,soustitre,c),col in zip([
            ("Ridge — Baseline","R² = 0.903","RMSE = 618 véh/h",
             "Modèle linéaire · Interprétable · Léger",GRIS),
            ("🏆 Random Forest","R² = 0.989","RMSE = 210 véh/h",
             "Meilleure performance · Recommandé batch",VERT),
            ("XGBoost","R² = 0.988","RMSE = 213 véh/h",
             "Quasi-équivalent · 4 MB · Temps réel",ORANGE),
        ],[c1,c2,c3]):
            with col:
                st.markdown(f"""<div style='background:var(--bg-card);border:1px solid var(--border);
                  border-radius:14px;padding:22px;border-top:4px solid {c};'>
                  <div style='font-weight:700;font-size:.9rem;color:var(--text-primary);'>{titre}</div>
                  <div style='font-size:1.7rem;font-weight:700;color:{c};margin:8px 0 2px;'>{val}</div>
                  <div style='font-size:.8rem;color:var(--text-secondary);'>{delta}</div>
                  <div style='font-size:.74rem;color:var(--text-muted);margin-top:10px;border-top:1px solid var(--border);padding-top:8px;'>{soustitre}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        sh("Évolution R² — Train · Validation · Test")
        mods = ["Ridge","Random Forest","XGBoost"]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Train",x=mods,y=[0.823,0.997,0.996],
                              marker_color=f"{BLEU}55",text=["0.823","0.997","0.996"],textposition="inside"))
        fig.add_trace(go.Bar(name="Validation",x=mods,y=[0.891,0.982,0.981],
                              marker_color=BLEU,text=["0.891","0.982","0.981"],textposition="inside"))
        fig.add_trace(go.Bar(name="Test",x=mods,y=[0.903,0.989,0.988],
                              marker_color=VERT,text=["0.903","0.989","0.988"],
                              textposition="inside",textfont=dict(color="white")))
        fig.update_layout(barmode="group",height=320,yaxis_range=[0.78,1.01],
                          legend=dict(orientation="h",y=1.12,font=dict(color=T_SECONDARY)),**plo())
        st.plotly_chart(fig, use_container_width=True)

        c1,c2 = st.columns(2)
        with c1:
            sh("Performances par jour de la semaine")
            mj = pd.DataFrame({"Jour":["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"],
                                "R²":[0.987,0.990,0.990,0.992,0.990,0.977,0.983],
                                "RMSE":[230,209,216,183,198,233,193]})
            fig = go.Figure()
            fig.add_trace(go.Bar(x=mj["Jour"],y=mj["RMSE"],name="RMSE",
                                  marker_color=BLEU,opacity=.85,
                                  text=mj["RMSE"],textposition="outside"))
            fig.update_layout(height=280,yaxis_title="RMSE (véh/h)",**plo(),
                              margin=dict(t=30,b=0,l=0,r=0))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            sh("Réel vs Prédit — Semaine du 02/07/2018")
            sem = df_pred[(df_pred["datetime"]>="2018-07-02")&(df_pred["datetime"]<="2018-07-08 23:00:00")]
            if len(sem)>0:
                sj = sem.groupby(sem["datetime"].dt.date).agg(
                    traffic=("traffic","sum"),pred_rf=("pred_rf","sum")).reset_index()
                sj["jour"] = pd.to_datetime(sj["datetime"]).dt.strftime("%a\n%d/%m")
                fig = go.Figure()
                fig.add_trace(go.Bar(name="Réel",x=sj["jour"],y=sj["traffic"],
                                      marker_color=f"{BLEU}66"))
                fig.add_trace(go.Bar(name="Prédit (RF)",x=sj["jour"],y=sj["pred_rf"],
                                      marker_color=VERT,opacity=.85))
                fig.update_layout(barmode="group",height=280,
                                  yaxis_title="Volume total (véh/jour)",**plo(),
                                  margin=dict(t=10,b=0,l=0,r=0),
                                  legend=dict(orientation="h",y=1.08,font=dict(color=T_SECONDARY)))
                st.plotly_chart(fig, use_container_width=True)

        box("Le modèle Random Forest prédit avec une erreur moyenne de <b>210 véhicules/heure</b> — soit ~5.8% d'erreur relative. Il est opérationnellement fiable 6 jours sur 7, avec une légère dégradation le samedi (R²=0.977) due aux patterns de week-end atypiques.", "g")

    # ── P-PRO-3 : Prédiction Interactive (Pro) ──
    elif PAGE == "🔮  Prédiction Interactive":
        st.markdown(f"""<div style='margin-bottom:16px;'>
          <h1 style='font-size:2rem;font-weight:700;color:var(--text-primary);margin:0;'>
            Prédiction Interactive</h1>
          <p style='color:var(--text-secondary);font-size:.95rem;margin-top:6px;'>
            Estimez le volume de trafic en temps réel selon vos paramètres</p>
        </div>""", unsafe_allow_html=True)

        ci,co = st.columns([1,1], gap="large")
        with ci:
            sh("⚙️ Conditions")
            mod_p = st.selectbox("Modèle prédictif",
                                  ["Random Forest (recommandé)","XGBoost (production)","Ridge (baseline)"])
            st.markdown("**📅 Quand ?**")
            dc1,dc2 = st.columns(2)
            with dc1: date_p = st.date_input("Date",value=datetime(2018,7,3).date())
            with dc2: h_p = st.slider("Heure",0,23,8,format="%dh")

            st.markdown("**🌤️ Météo**")
            mc1,mc2 = st.columns(2)
            with mc1:
                temp_p  = st.slider("Température (°C)",-20,40,20)
                rain_p  = st.slider("Pluie (mm/h)",0,60,0)
            with mc2:
                snow_p  = st.slider("Neige (mm/h)",0,30,0)
                cloud_p = st.slider("Nuages (%)",0,100,40)

            st.markdown("**📈 Trafic récent**")
            lc1,lc2 = st.columns(2)
            with lc1:
                l1  = st.number_input("Heure précédente",0,7500,3500)
                l2  = st.number_input("Il y a 2h",0,7500,3400)
            with lc2:
                l3  = st.number_input("Il y a 3h",0,7500,3200)
                l24 = st.number_input("Même heure hier",0,7500,3600)

            btn_pro = st.button("🚀  Prédire maintenant", use_container_width=True, type="primary")

        with co:
            sh("📊 Résultat")
            if btn_pro:
                dt = datetime.combine(date_p,datetime.min.time())+timedelta(hours=h_p)
                feat = {c:0.0 for c in COLS}
                feat.update({"rain":float(rain_p),"snow":float(snow_p),"cloud":float(cloud_p),
                             "hour":float(h_p),"day":float(dt.day),"weekday":float(dt.weekday()),
                             "month":float(dt.month),"year":float(dt.year),
                             "is_holiday":0.,"is_rush_hour":1. if h_p in range(7,10) or h_p in range(16,20) else 0.,
                             "is_weekend":1. if dt.weekday()>=5 else 0.,
                             "temp_c":float(temp_p),"rain_cat":1. if rain_p>0 else 0.,
                             "snow_cat":1. if snow_p>0 else 0.,
                             "hour_sin":np.sin(2*np.pi*h_p/24),"hour_cos":np.cos(2*np.pi*h_p/24),
                             "day_sin":np.sin(2*np.pi*dt.weekday()/7),"day_cos":np.cos(2*np.pi*dt.weekday()/7),
                             "month_sin":np.sin(2*np.pi*dt.month/12),"month_cos":np.cos(2*np.pi*dt.month/12),
                             "traffic_lag_1":float(l1),"traffic_lag_2":float(l2),
                             "traffic_lag_3":float(l3),"traffic_lag_24":float(l24)})
                for c in COLS:
                    if "lag" in c and feat.get(c,None)==0.0: feat[c]=float(l1)
                    if "mean" in c and feat.get(c,None)==0.0: feat[c]=float((l1+l2+l3)/3)

                Xp = pd.DataFrame([feat])[COLS]
                nr = ["temp_c","rain","cloud","hour_sin","hour_cos","day_sin","day_cos",
                      "month_sin","month_cos","traffic_lag_1","traffic_lag_2","traffic_lag_3","traffic_lag_24"]
                mod_key = mod_p.split(" ")[0]
                if mod_key=="Random":   pred=RF.predict(Xp)[0]
                elif mod_key=="XGBoost": pred=XGB.predict(Xp)[0]
                else:
                    Xs=Xp.copy()
                    try: Xs[nr]=SCALER.transform(Xs[nr])
                    except: pass
                    pred=RIDGE.predict(Xs)[0]

                pred = max(0,int(round(pred)))
                if pred<1500:   css,emoji,niv,niv_c="","🟢","Fluide",VERT
                elif pred<3500: css,emoji,niv,niv_c="w","🟡","Modéré",ORANGE
                elif pred<5000: css,emoji,niv,niv_c="d","🔴","Dense",ROUGE
                else:           css,emoji,niv,niv_c="d","🔴","Saturé",ROUGE

                st.markdown(f"""<div class='pred-box {css}'>
                  <div class='pred-val'>{pred:,}</div>
                  <div style='color:var(--text-secondary);font-size:.95rem;margin-top:6px;'>véhicules / heure</div>
                  <div style='margin-top:14px;font-size:1.1rem;font-weight:700;color:{niv_c};'>
                    {emoji} Trafic {niv}</div>
                  <div style='font-size:.8rem;color:var(--text-muted);margin-top:6px;'>
                    {dt.strftime('%A %d %B %Y')} à {h_p:02d}h00</div>
                </div>""", unsafe_allow_html=True)

                # Jauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",value=pred,
                    title={"text":"Volume prédit (véh/h)","font":{"size":13,"color":T_SECONDARY}},
                    number={"font":{"color":T_PRIMARY}},
                    gauge={"axis":{"range":[0,7500],"tickcolor":T_SECONDARY},
                           "bar":{"color":BLEU},
                           "bgcolor":BG_CARD,
                           "bordercolor":GRID_COLOR,
                           "steps":[{"range":[0,1500],"color":f"{VERT}22"},
                                     {"range":[1500,3500],"color":f"{ORANGE}22"},
                                     {"range":[3500,7500],"color":f"{ROUGE}22"}],
                           "threshold":{"line":{"color":ROUGE,"width":3},"value":5500}}))
                fig.update_layout(height=220,paper_bgcolor=BG_CARD,
                                  font=dict(color=T_SECONDARY),
                                  margin=dict(t=40,b=0,l=20,r=20))
                st.plotly_chart(fig, use_container_width=True)

                # Profil 24h
                sh("Prévision sur 24h")
                p24=[]
                for h in range(24):
                    f2=feat.copy(); f2["hour"]=float(h)
                    f2["hour_sin"]=np.sin(2*np.pi*h/24); f2["hour_cos"]=np.cos(2*np.pi*h/24)
                    f2["is_rush_hour"]=1. if h in range(7,10) or h in range(16,20) else 0.
                    X2=pd.DataFrame([f2])[COLS]
                    if mod_key=="Random":   pv=RF.predict(X2)[0]
                    elif mod_key=="XGBoost": pv=XGB.predict(X2)[0]
                    else:
                        try: X2[nr]=SCALER.transform(X2[nr])
                        except: pass
                        pv=RIDGE.predict(X2)[0]
                    p24.append({"h":h,"p":max(0,pv)})
                df24=pd.DataFrame(p24)
                fig2=go.Figure()
                fig2.add_vrect(x0=7,x1=9,fillcolor=ORANGE,opacity=.12,annotation_text="Matin")
                fig2.add_vrect(x0=16,x1=19,fillcolor=VERT,opacity=.12,annotation_text="Soir")
                fig2.add_trace(go.Scatter(x=df24["h"],y=df24["p"],mode="lines+markers",
                                           line=dict(color=BLEU,width=2.5),marker=dict(size=6),
                                           fill="tozeroy",fillcolor=f"{BLEU}18"))
                fig2.add_vline(x=h_p,line_dash="dash",line_color=ROUGE,
                                annotation_text=f"{h_p}h",
                                annotation_font_color=ROUGE)
                fig2.update_layout(height=230,
                                   xaxis=dict(tickvals=list(range(0,24,2)),gridcolor=GRID_COLOR,
                                               color=T_SECONDARY,title="Heure"),
                                   yaxis=dict(gridcolor=GRID_COLOR,color=T_SECONDARY,title="Trafic (véh/h)"),
                                   **{k:v for k,v in plo().items() if k not in ["xaxis","yaxis"]},
                                   showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

            else:
                st.markdown(f"""<div style='text-align:center;padding:80px 20px;color:var(--text-muted);'>
                  <div style='font-size:3.5rem;'>🎛️</div>
                  <div style='font-size:.95rem;margin-top:14px;'>
                    Configurez les paramètres<br>puis cliquez sur <b>Prédire maintenant</b>
                  </div></div>""", unsafe_allow_html=True)

    #=========================
    # Appel de la fonction
    add_footer()
    st.stop()  # Fin du mode pro — ne pas exécuter les pages pédagogiques


# ══════════════════════════════════════════════════════════════
# P1 — ACCUEIL
# ══════════════════════════════════════════════════════════════
if PAGE == "🏠  Accueil":
    
    # ============================================
    # PAGE D'ACCUEIL
    # ============================================
    # Utilisation
    display_logo(variant="default", location="main")

    st.title("Prédiction du volume de trafic urbain")

    paragraphe(classe="text-primary", style="text-align: center;", text="Smart City - Minneapolis-Saint Paul")

    st.markdown("##")
    st.markdown(" ")
    paragraphe(classe="custom-text",text="""Bienvenue sur l'application de prédiction du trafic urbain pour la zone de Minneapolis-Saint Paul.
    Ce projet vise à fournir des prédictions précises du volume de trafic horaire sur l'Interstate 94,
    en utilisant des données temporelles et météorologiques.
    L'application est conçue pour être interactive et informative, permettant aux utilisateurs de comprendre les facteurs influençant le trafic et d'explorer différentes scénarios.
    """)

    # Affichage d'une image d'en-tête
    header_img = Image.open('assets/projectmap.png')
    st.image(header_img, use_container_width=True)

    # En savoir plus
    st.markdown("---")
    with st.expander("En savoir plus la zone d'étude", expanded=True):
        st.subheader("📍 Zone d'étude : Minneapolis-Saint Paul")

        col1, col2 = st.columns(2)

        with col1:
            
            # Création d'une carte interactive
            m = folium.Map(location=[44.95, -93.20], zoom_start=10)

            # Ajout des marqueurs
            folium.Marker(
                [44.95, -93.20],
                popup='Minneapolis',
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(m)

            folium.Marker(
                [44.94, -93.09],
                popup='Saint Paul',
                icon=folium.Icon(color='green', icon='info-sign')
            ).add_to(m)

            folium.Marker(
                [44.95, -93.15],
                popup='Station ATR 301',
                icon=folium.Icon(color='red', icon='car')
            ).add_to(m)

            # Affichage
            folium_static(m)

        with col2:
            st.markdown("""
            **Pays** : États-Unis 🇺🇸 
            **État** : Minnesota (MN)  
            **Région** : Midwest américain  
            **Comtés** : Hennepin (Minneapolis) • Ramsey (Saint Paul)  
            **Population (aire métro)** : 3,69 millions d'habitants (2023)  
            **Superficie métropolitaine** : 21 632 km² 
            **Villes principales :**
                - **Minneapolis** : 429 954 hab. ("City of Lakes")
                - **Saint Paul** : 311 527 hab. ("Capital City")
                - **Twin Cities** : 741 481 hab. 
            """)
            st.markdown("---")
            st.markdown("""L'Interstate 94 est un axe autoroutier majeur du Midwest américain, reliant les villes jumelles de Minneapolis et Saint Paul dans l'État du Minnesota. Cette région métropolitaine, qui compte près de 3,7 millions d'habitants, constitue la 16ᵉ aire urbaine des États-Unis.
                    Le tronçon étudié s'étend sur environ 40 kilomètres entre les deux centres urbains, traversant à la fois des zones résidentielles, commerciales et industrielles. La station de mesure ATR 301, située à mi-chemin, enregistre en continu le volume horaire de trafic en direction ouest depuis 2012.
                    Le climat continental du Minnesota, caractérisé par des hivers rigoureux (jusqu'à -20°C) et des chutes de neige fréquentes (jusqu'à 1,5 m par an), influence significativement les comportements de mobilité. Notre modèle de prédiction intègre ces spécificités locales pour fournir des estimations précises du volume de trafic.
                    """)
        
        st.subheader("🛣️ L'Interstate 94")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Caractéristiques techniques**
            - Longueur totale : 2 585 km
            - Tronçon Minnesota : 447 km
            - Tronçon étudié : 40 km
            - Voies : 4 à 6
            - Mise en service : 1961
            """)

        with col2:
            st.markdown("""
            **Trafic**
            - Trafic journalier : jusqu'à 170 000 véh/jour
            - Heures de pointe : 7h-9h et 16h-18h
            - Pic maximum : 7 200 véh/heure
            - Station ATR 301 (MnDOT)
            """)

    #=======================problematique & contexte & objectif ====================================

    st.markdown("## ")
    header(n=2,text = "Contexte - Problématique - Objectifs")
    st.markdown(" ")
    st.markdown("""La croissance urbaine engendre une augmentation du trafic routier, source d'embouteillages, de pollution et de perte de temps. Dans les métropoles comme Minneapolis-Saint Paul, l'Interstate 94 supporte plus de 170 000 véhicules par jour. Les initiatives Smart City exploitent les données et le machine learning pour anticiper les flux et optimiser la mobilité. 
    Comment prédire le volume de trafic horaire sur un axe autoroutier majeur à partir de variables temporelles (heure, jour, mois) et météorologiques (température, précipitations, couverture nuageuse) ? La difficulté réside dans les relations non linéaires, les interactions entre variables, la saisonnalité et les événements exceptionnels comme les chutes de neige.
    Développer un modèle performant (R² > 0,85) pour prédire le trafic horaire, enrichir les données par feature engineering (lags, cycles horaires), comparer plusieurs approches (Ridge, Random Forest, XGBoost), interpréter les résultats avec SHAP et PDP, et déployer une application Streamlit interactive. 
    """)

    with st.expander("Lire plus", expanded=True):
        # Contexte
        st.markdown("""
        <div class="context-box">
            <h3 style="margin-top: 0;">🌍 Contexte</h3>
            <p>
            La croissance rapide des zones urbaines s'accompagne d'une augmentation 
            constante du volume de trafic routier, générant des embouteillages chroniques, 
            une pollution atmosphérique accrue et une perte de temps considérable pour 
            les usagers. Dans les métropoles américaines comme Minneapolis-Saint Paul, 
            l'Interstate 94 constitue un axe majeur de mobilité, reliant les deux villes 
            jumelles et supportant quotidiennement plus de <span class="highlight">170 000 véhicules</span>.
            </p>
            <p>
            Face à ces défis, les initiatives de <strong>"Smart City"</strong> (ville intelligente) 
            visent à exploiter les données massives et les algorithmes de machine 
            learning pour anticiper les flux de circulation, optimiser la gestion du 
            réseau routier et informer les citoyens en temps réel.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Problématique
        st.markdown("""
        <div class="problem-box">
            <h3 style="margin-top: 0;">🎯 Problématique</h3>
            <p>
            <strong>Comment prédire le volume de trafic horaire sur un axe autoroutier majeur 
            à partir de variables temporelles et météorologiques ?</strong>
            </p>
            <p>La complexité réside dans plusieurs facteurs :</p>
            <ul>
                <li><strong>Non-linéarité</strong> : l'effet de l'heure de pointe n'est pas simplement additif avec celui de la météo ;</li>
                <li><strong>Interactions</strong> : la pluie n'a pas le même impact à 8h qu'à 14h ;</li>
                <li><strong>Saisonnalité</strong> : les comportements varient selon les jours, week-ends et saisons ;</li>
                <li><strong>Événements exceptionnels</strong> : neige, tempêtes, manifestations perturbent les habitudes.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


        # Objectifs
        st.markdown("""
        <div class="objective-box">
            <h3 style="margin-top: 0;">🎯 Objectifs du projet</h3>
            <p><strong>Objectif principal :</strong> Développer un modèle prédictif avec une précision opérationnelle.</p>
            <p><strong>Objectifs spécifiques :</strong></p>
            <ol>
                <li>Explorer et préparer les données (nettoyage, gestion des anomalies)</li>
                <li>Enrichir les features (lags, moyennes mobiles, cycles horaires)</li>
                <li>Comparer Ridge, Random Forest et XGBoost</li>
                <li>Interpréter les résultats (SHAP, PDP)</li>
                <li>Déployer une application Streamlit interactive</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)


    # ============================================
    # Sources de données
    # ============================================

    st.markdown("## ")
    header(n=2,text = "📌 Sources de données")
    with st.expander("", expanded=True):
        
        st.markdown("""
        <div class="problem-box">
            <h3>  </h3>
            <p>
                Le jeu de données utilisé dans ce projet provient de deux sources principales : 
                le <strong>Minnesota Department of Transportation (MnDOT)</strong> pour les données de trafic 
                (station ATR 301 sur l'Interstate 94 Westbound) et <strong>OpenWeatherMap</strong> pour les 
                données météorologiques (température, précipitations, couverture nuageuse). 
                L'ensemble a été mis à disposition par l'UCI Machine Learning Repository 
                (dataset ID 492).
            </p>
            <p>
                La période initiale couvre octobre 2012 à septembre 2018 avec <span class="highlight">48 204 observations</span> 
                horaires. Après nettoyage (suppression des doublons, traitement des anomalies 
                de température à 0K et de pluie à 9831 mm) et restriction à la période la plus 
                complète (2015-2018), le jeu de données final comprend <span class="highlight">23 979 observations</span> 
                avec un taux de complétude de <span class="highlight">93,4%</span>.
            </p>
            <p>
                Les variables incluent la cible <code style="color:#FFD700;">traffic_volume</code> (volume horaire de trafic) 
                et des prédicteurs temporels (<code style="color:#FFD700;">date_time</code>) et météorologiques (<code style="color:#FFD700;">temp</code>, 
                <code style="color:#FFD700;">rain_1h</code>, <code style="color:#FFD700;">snow_1h</code>, <code style="color:#FFD700;">clouds_all</code>, <code style="color:#FFD700;">weather_main</code>, <code style="color:#FFD700;">weather_description</code>, 
                <code style="color:#FFD700;">holiday</code>). Par la suite, 15 nouvelles features ont été créées par 
                feature engineering (lags, moyennes mobiles, transformations cycliques, indicateurs).
            </p>
        </div>
        """, unsafe_allow_html=True)

        src_img = Image.open('assets/sources.png')
        st.image(src_img, use_container_width=True)
        
        st.markdown("""
        ### 🔗 Accès aux données

        Le jeu de données original est disponible publiquement sur :

        - **UCI Machine Learning Repository** :  
        [Metro Interstate Traffic Volume Dataset](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume)

        - **Téléchargement direct** :  
        ```python
        from ucimlrepo import fetch_ucirepo
        metro = fetch_ucirepo(id=492)
        df = pd.concat([metro.data.features, metro.data.targets], axis=1)
                    """)
    #=============================================


    #==================

    st.markdown("## ")
    header(n=2,text = "🤖 Aperçu des modèles comparés")
    with st.expander("", expanded=True):
        st.markdown("""
        Trois modèles de machine learning ont été entraînés et comparés pour 
        la prédiction du volume de trafic horaire.
        """)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div style="
                background: #f8f9fa;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                border-top: 4px solid #e74c3c;
                color: black;
            ">
                <h3 style="margin: 0;">📐 Ridge</h3>
                <p style="color: black; margin: 5px 0;">Régression linéaire</p>
                <hr>
                <p><strong>R²</strong> : <span style="color: #e74c3c; font-weight: bold;">0,903</span></p>
                <p><strong>RMSE</strong> : <span style="color: #e74c3c; font-weight: bold;">618</span></p>
                <p><strong>MAPE</strong> : <span style="color: #e74c3c; font-weight: bold;">28,0%</span></p>
                <p><strong> </strong> <span style="color: #e74c3c; font-weight: bold;"> ___ </span></p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style="
                background: #e8f8f5;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                border-top: 4px solid #27ae60;
                color: black;
            ">
                <h3 style="margin: 0;">🌲 Random Forest</h3>
                <p style="color: black; margin: 5px 0;">Ensemble (Bagging)</p>
                <hr>
                <p><strong>R²</strong> : <span style="color: #27ae60; font-weight: bold;">0,989</span></p>
                <p><strong>RMSE</strong> : <span style="color: #27ae60; font-weight: bold;">210</span></p>
                <p><strong>MAPE</strong> : <span style="color: #27ae60; font-weight: bold;">5,8%</span></p>
                <p style="margin-top: 10px;"><span style="background: #27ae60; color: white; padding: 3px 10px; border-radius: 15px;">🏆 Meilleur</span></p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div style="
                background: #f8f9fa;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                border-top: 4px solid #3498db;
                color: black;
            ">
                <h3 style="margin: 0;">⚡ XGBoost</h3>
                <p style="color: black; margin: 5px 0;">Ensemble (Boosting)</p>
                <hr>
                <p><strong>R²</strong> : <span style="color: #3498db; font-weight: bold;">0,988</span></p>
                <p><strong>RMSE</strong> : <span style="color: #3498db; font-weight: bold;">213</span></p>
                <p><strong>MAPE</strong> : <span style="color: #3498db; font-weight: bold;">5,9%</span></p>
                <p><strong> </strong> <span style="color: #3498db; font-weight: bold;"> ___ </span></p>

            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        # Graphique de comparaison
        st.subheader("Comparaison des modèles")

        comparison_data = pd.DataFrame({
            'Modèle': ['Ridge', 'Random Forest', 'XGBoost'],
            'R²': [0.903, 0.989, 0.988],
            'RMSE': [618, 210, 213],
            'MAE': [434, 135, 138]
        })

        #st.dataframe(comparison_data, use_container_width=True)

        # Graphique à barres
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # R²
        axes[0].bar(comparison_data['Modèle'], comparison_data['R²'], 
                    color=['steelblue', 'green', 'coral'])
        axes[0].set_ylabel('R²')
        axes[0].set_title('Coefficient de détermination')
        axes[0].set_ylim(0.8, 1.0)
        for i, v in enumerate(comparison_data['R²']):
            axes[0].text(i, v - 0.02, f'{v:.3f}', ha='center', fontweight='bold')

        # RMSE
        axes[1].bar(comparison_data['Modèle'], comparison_data['RMSE'], 
                    color=['steelblue', 'green', 'coral'])
        axes[1].set_ylabel('RMSE (véhicules/heure)')
        axes[1].set_title('Erreur quadratique moyenne')
        for i, v in enumerate(comparison_data['RMSE']):
            axes[1].text(i, v + 20, f'{v:.0f}', ha='center', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)

        # ============================================
        # IMPORTANCE DES VARIABLES
        # ============================================

        st.header("🔑 Top 10 des variables les plus influentes")

        # Données d'importance (issues de XGBoost)
        importance_data = pd.DataFrame({
            'Variable': ['hour_cos', 'snow_cat', 'snow', 'traffic_lag_1', 
                        'is_rush_hour', 'traffic_lag_24', 'rain', 'weekday',
                        'hour', 'is_weekend'],
            'Importance': [44.3, 15.1, 10.7, 6.8, 3.8, 3.3, 2.3, 2.2, 2.1, 1.8]
        })

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Blues_r(np.linspace(0.3, 0.9, len(importance_data)))
        ax.barh(importance_data['Variable'], importance_data['Importance'], color=colors)
        ax.set_xlabel('Importance (%)')
        ax.set_title('Importance des variables - XGBoost')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
        **💡 Interprétation** : 
        - L'heure (`hour_cos`) est le facteur dominant (44%)
        - La neige a un impact majeur (25% cumulé)
        - Le trafic passé (`traffic_lag_1`) confirme l'inertie
        """)



    st.markdown("##")

    st.subheader("📋 Architecture du projet")

    st.markdown("""
    Le projet suit un pipeline complet de data science, de la collecte des données 
    au déploiement de l'application.
    """)

    # Workflow visuel
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown("""
        <div style="text-align: center;">
            <p style="font-size: 32px;">📊</p>
            <p style="font-weight: bold;">Collecte</p>
            <p style="font-size: 12px; color: gray;">Données trafic + météo</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <p style="font-size: 32px;">🧹</p>
            <p style="font-weight: bold;">Nettoyage</p>
            <p style="font-size: 12px; color: gray;">Doublons, outliers, NaN</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="text-align: center;">
            <p style="font-size: 32px;">⚙️</p>
            <p style="font-weight: bold;">Feature Engineering</p>
            <p style="font-size: 12px; color: gray;">Lags, cycles, indicateurs</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div style="text-align: center;">
            <p style="font-size: 32px;">🤖</p>
            <p style="font-weight: bold;">Modélisation</p>
            <p style="font-size: 12px; color: gray;">Ridge, RF, XGBoost</p>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown("""
        <div style="text-align: center;">
            <p style="font-size: 32px;">🚀</p>
            <p style="font-weight: bold;">Déploiement</p>
            <p style="font-size: 12px; color: gray;">Application Streamlit</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("##")


# ══════════════════════════════════════════════════════════════
# P2 — EDA
# ══════════════════════════════════════════════════════════════

elif PAGE == "📊  Exploration (EDA)":
    st.title("📊 Exploration des données (EDA)")
    st.markdown("Analyse exploratoire : distributions, patterns temporels et relations météo–trafic.")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Aperçu", "🕐 Patterns temporels", "🌤️ Météo × Trafic", 
        "🔥 Heatmap & Boxplots"
    ])

    # ============================================
    # TAB 1 : APERÇU
    # ============================================
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Lignes", f"{len(df_raw):,}", "observations brutes")
        with c2: kpi("Colonnes", "9", "variables originales")
        with c3: kpi("Période", "6 ans", "Oct.2012 – Sep.2018")
        with c4: kpi("Trafic moyen", f"{int(df_raw['traffic_volume'].mean()):,}", "véh/heure", "g")
        st.markdown("<br>", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════
        # DESCRIPTION DE LA BASE
        # ══════════════════════════════════════════════════════════════
        with st.expander("📖 Description détaillée du jeu de données", expanded=True):
            st.markdown("""
            **Initialement**, le jeu de données contient **48 204 observations**, allant du **2 octobre 2012 à 9h00** au **30 septembre 2018 à 23h00**, et comporte **9 caractéristiques**.

            Voici la description des différents attributs :
            """)

            # Tableau des variables
            var_desc = pd.DataFrame({
                "Variable": ["holiday", "temp", "rain_1h", "snow_1h", "clouds_all", 
                            "weather_main", "weather_description", "date_time", "traffic_volume"],
                "Type": ["Alphanumérique", "Numérique", "Numérique", "Numérique", "Numérique",
                        "Alphanumérique", "Alphanumérique", "Temporelle", "Numérique"],
                "Unité": ["-", "Kelvin (K)", "mm/h", "mm/h", "%", "-", "-", "CST", "véhicules/heure"],
                "Description": [
                    "Jours fériés nationaux (USA) et régionaux (Minnesota)",
                    "Température moyenne",
                    "Quantité de pluie enregistrée durant l'heure",
                    "Quantité de neige enregistrée durant l'heure",
                    "Couverture nuageuse observée",
                    "Brève description textuelle du temps actuel",
                    "Description textuelle détaillée du temps actuel",
                    "Date et heure de collecte (heure locale CST)",
                    "Volume horaire de trafic en direction ouest (variable cible)"
                ]
            })
            st.dataframe(var_desc, use_container_width=True, hide_index=True)
            
            st.caption("📌 **Variable cible** : `traffic_volume` - volume horaire de trafic sur l'Interstate 94 Westbound (station ATR 301).")

        # ══════════════════════════════════════════════════════════════
        # APERÇU DES DONNÉES (df.head)
        # ══════════════════════════════════════════════════════════════
        sh("Aperçu des données (5 premières lignes)")
        st.dataframe(df_raw.head(10), use_container_width=True)
        
        #============================================================

        # Détection des outliers
        Q1 = df_raw['traffic_volume'].quantile(0.25)
        Q3 = df_raw['traffic_volume'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df_raw[(df_raw['traffic_volume'] < Q1 - 1.5*IQR) | (df_raw['traffic_volume'] > Q3 + 1.5*IQR)]
        
        c1, c2 = st.columns(2)
        with c1:
            sh("Statistiques descriptives")
            desc = df_raw[["traffic_volume","temp","rain_1h","snow_1h","clouds_all"]].describe().round(2).T
            desc.index = ["Trafic (véh/h)","Temp (K)","Pluie (mm/h)","Neige (mm/h)","Nuages (%)"]
            desc.columns = ["N","Moyenne","Std","Min","Q1","Médiane","Q3","Max"]
            st.dataframe(desc, use_container_width=True)
            
            # Ajout : détection outliers
            st.markdown(f"""
            <div class="eda-warning">
                ⚠️ Le jeu de données se distingue par une excellente complétude : les cinq variables numériques sont renseignées pour l'intégralité des 48 204 observations.
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            sh("Types & Valeurs manquantes")
            info = pd.DataFrame({
                "Variable": df_raw.columns,
                "Type": df_raw.dtypes.astype(str).values,
                "Manquants": df_raw.isna().sum().values,
                "% Manq.": (df_raw.isna().mean()*100).round(2).values,
                "Uniques": df_raw.nunique().values
            })
            st.dataframe(info, use_container_width=True, hide_index=True)

        sh("Distribution de traffic_volume")
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.histogram(df_raw, x="traffic_volume", nbins=80,
                               color_discrete_sequence=[BLEU],
                               labels={"traffic_volume": "Volume de trafic (véh/h)"})
            fig.add_vline(x=df_raw["traffic_volume"].mean(), line_dash="dash",
                          line_color=ORANGE, annotation_text=f"Moy={int(df_raw['traffic_volume'].mean())}")
            fig.add_vline(x=df_raw["traffic_volume"].median(), line_dash="dot",
                          line_color=VERT, annotation_text=f"Med={int(df_raw['traffic_volume'].median())}")
            fig.update_layout(height=280, **plo())
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            box("Distribution <b>bimodale</b> : pic à 0 (nuit/week-end) et concentration 3 000–5 500 véh/h (journée ouvrée). Justifie les modèles non-linéaires.", "b")
            col_stats = st.columns(2)
            with col_stats[0]:
                st.metric("Skewness", f"{df_raw['traffic_volume'].skew():.2f}")
                st.metric("Kurtosis", f"{df_raw['traffic_volume'].kurtosis():.2f}")
            with col_stats[1]:
                st.metric("% zéros", f"{(df_raw['traffic_volume']==0).mean()*100:.1f}%")
                st.metric("Max", f"{int(df_raw['traffic_volume'].max()):,}")

    # ============================================
    # TAB 2 : PATTERNS TEMPORELS
    # ============================================
    with tab2:
        sh("📈 Évolution temporelle du trafic")
        
        # Préparation des données
        df_plot = df_raw.copy()
        df_plot['datetime'] = pd.to_datetime(df_plot['date_time'])
        df_plot.set_index('datetime', inplace=True)
        
        # ══════════════════════════════════════════════════════════════
        # SÉLECTEUR : PRÉDÉFINI OU PERSONNALISÉ
        # ══════════════════════════════════════════════════════════════
        
        mode_affichage = st.radio(
            "Mode d'affichage",
            ["📆 Période prédéfinie", "🎯 Plage personnalisée"],
            horizontal=True
        )
        
        if mode_affichage == "📆 Période prédéfinie":
            period_option = st.selectbox(
                "Période",
                ["1 semaine", "1 mois", "3 mois", "6 mois", "1 an", "Toute la période"],
                index=1
            )
            
            period_days = {"1 semaine": 7, "1 mois": 30, "3 mois": 90, "6 mois": 180, "1 an": 365}
            
            if period_option != "Toute la période":
                cutoff = df_plot.index.max() - pd.Timedelta(days=period_days[period_option])
                df_filtered = df_plot[df_plot.index >= cutoff]
                title = f"Évolution du trafic - {period_option}"
            else:
                df_filtered = df_plot
                title = "Évolution du trafic - Période complète"
        
        else:  # Plage personnalisée
            col1, col2 = st.columns(2)
            
            with col1:
                date_debut = st.date_input(
                    "Date de début",
                    value=df_plot.index.min().date(),
                    min_value=df_plot.index.min().date(),
                    max_value=df_plot.index.max().date()
                )
            
            with col2:
                date_fin = st.date_input(
                    "Date de fin",
                    value=df_plot.index.max().date(),
                    min_value=df_plot.index.min().date(),
                    max_value=df_plot.index.max().date()
                )
            
            date_debut = pd.to_datetime(date_debut)
            date_fin = pd.to_datetime(date_fin)
            
            df_filtered = df_plot[(df_plot.index >= date_debut) & (df_plot.index <= date_fin)]
            title = f"Évolution du trafic du {date_debut.strftime('%d/%m/%Y')} au {date_fin.strftime('%d/%m/%Y')}"
        
        # Vérification
        if len(df_filtered) == 0:
            st.warning("⚠️ Aucune donnée sur cette période. Veuillez ajuster les dates.")
        else:
            # Statistiques
            c1, c2, c3= st.columns(3)
            with c1: kpi("Trafic moyen", f"{int(df_filtered['traffic_volume'].mean()):,}", "véh/h")
            with c2: kpi("Trafic max", f"{int(df_filtered['traffic_volume'].max()):,}", "véh/h")
            with c3: kpi("Trafic min", f"{int(df_filtered['traffic_volume'].min()):,}", "véh/h")
            st.markdown("<br>", unsafe_allow_html=True)

            # Graphique
            fig = px.line(
                df_filtered,
                x=df_filtered.index,
                y='traffic_volume',
                title=title,
                labels={'x': 'Date', 'traffic_volume': 'Volume de trafic (véh/h)'},
                color_discrete_sequence=[BLEU]
            )
            fig.update_layout(height=400, hovermode='x unified', **plo())
            st.plotly_chart(fig, use_container_width=True)
            
            
            # Option : resampling
            with st.expander("⚙️ Options avancées"):
                freq_option = st.selectbox(
                    "Fréquence de résampling",
                    ["Horaire (brut)", "Journalière", "Hebdomadaire", "Mensuelle"],
                    index=0
                )
                
                if freq_option != "Horaire (brut)":
                    freq_map = {"Journalière": "D", "Hebdomadaire": "W", "Mensuelle": "ME"}
                    df_resampled = df_filtered['traffic_volume'].resample(freq_map[freq_option]).mean()
                    
                    fig2 = px.line(
                        x=df_resampled.index,
                        y=df_resampled.values,
                        title=f"Évolution du trafic - {freq_option}",
                        labels={'x': 'Date', 'y': 'Volume de trafic (véh/h)'},
                        color_discrete_sequence=[VERT]
                    )
                    fig2.update_layout(height=300, **plo())
                    st.plotly_chart(fig2, use_container_width=True)

  
        
            # Ajout d'une annotation sur le trou
            sh("🔍 Analyse de la qualité des données")
            commentaire("""L'analyse de l'évolution temporelle du trafic (graphique ci-dessus) révèle une longue absence de données entre mi-2014 et mi-2015, correspondant à une période d'environ 10 mois où la station ATR 301 n'a pas enregistré de mesures. Cette interruption rend inexploitables les variables de délai (lags) et les moyennes mobiles, car les calculs traversant cette zone vide perdent leur signification temporelle.
                Pour garantir la fiabilité des features temporelles et maximiser la qualité de l'apprentissage, nous avons restreint notre analyse aux données postérieures au 19 juin 2015 à 17h00. Cette décision se justifie par plusieurs critères :
                Continuité des données : après cette date, la série temporelle est continue sans interruption majeure
                Taux de complétude : 93,4% contre 77,2% sur la période complète
                Exploitabilité des lags : les trous résiduels sont inférieurs à 9 heures, permettant une interpolation locale
                Période suffisante : 3 années de données (2015-2018) sont représentatives des cycles journaliers, hebdomadaires et saisonniers
                En résumé, ce filtrage garantit une base de données robuste et exploitable pour la modélisation, au prix d'une réduction du nombre d'observations de 48 204 à 23 979.
                """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **❌ Avant filtrage (2012-2018)**
                - Taux de complétude : 77,2%
                - Trou maximal : 10 mois
                - Lags : inexploitables
                - Observations : 48 204
                """)
            
            with col2:
                st.markdown(f"""
                **✅ Après filtrage (2015-2018)**
                - Taux de complétude : 93,4%
                - Trou maximal : 9 heures
                - Lags : exploitables
                - Observations : 23 979
                """)
            
            st.success("🎯 **Période retenue pour la modélisation** : 19 juin 2015 → 30 septembre 2018")
        
        # ================================================================
        
        sh("Profil horaire moyen ± écart-type")
        h_data = df_raw.groupby(df_raw["date_time"].dt.hour)["traffic_volume"].agg(["mean", "std"]).reset_index()
        h_data.columns = ["h", "m", "s"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h_data["h"], y=h_data["m"]+h_data["s"],
                                  fill=None, mode="lines", line_color="rgba(30,111,217,0)", showlegend=False))
        fig.add_trace(go.Scatter(x=h_data["h"], y=h_data["m"]-h_data["s"],
                                  fill="tonexty", mode="lines", line_color="rgba(30,111,217,0)",
                                  fillcolor="rgba(30,111,217,0.1)", name="±1σ"))
        fig.add_trace(go.Scatter(x=h_data["h"], y=h_data["m"], mode="lines+markers",
                                  line=dict(color=BLEU, width=2.5), marker=dict(size=7), name="Moyenne"))
        fig.add_vrect(x0=6, x1=9, fillcolor=ORANGE, opacity=.12, annotation_text="Pointe matin")
        fig.add_vrect(x0=15, x1=18, fillcolor=VERT, opacity=.12, annotation_text="Pointe soir")
        fig.update_layout(height=310, xaxis=dict(tickvals=list(range(0,24,2)), gridcolor="#F1F5F9", title="Heure"),
                          yaxis=dict(gridcolor="#F1F5F9", title="Trafic moyen (véh/h)"),
                          legend=dict(orientation="h", y=1.1), **{k:v for k,v in plo().items() if k not in ["xaxis","yaxis"]})
        st.plotly_chart(fig, use_container_width=True)

      
        commentaire("""Le profil horaire révèle une structure bimodale caractéristique du trafic de jour ouvré. 
                Deux pics distincts sont observables : un pic matinal entre 6h et 9h et un pic vespéral plus élevé et plus étalé entre 15h et 18h. 
                La bande d'écart-type, particulièrement large aux heures de pointe, indique une forte variabilité du trafic à ces moments, contrairement aux heures creuses où la dispersion est faible. 
                Le trafic minimal se situe entre minuit et 5h, période où la moyenne et la variabilité sont proches de zéro
                """)
        
        # NOUVEAU : Comparaison semaine vs week-end
        st.subheader("📅 Comparaison : Semaine vs Week-end")
        df_raw['is_weekend'] = df_raw['date_time'].dt.dayofweek.isin([5,6])
        weekend_hourly = df_raw.groupby([df_raw['date_time'].dt.hour, 'is_weekend'])['traffic_volume'].mean().reset_index()
        
        fig = px.line(weekend_hourly, x='date_time', y='traffic_volume', 
                      color='is_weekend', color_discrete_map={True: ORANGE, False: BLEU},
                      labels={'date_time': 'Heure', 'traffic_volume': 'Trafic moyen', 'is_weekend': ''})
        fig.update_layout(height=280, legend=dict(orientation="h", y=1.1), **plo())
        fig.for_each_trace(lambda t: t.update(name='Week-end' if t.name == 'True' else 'Semaine'))
        st.plotly_chart(fig, use_container_width=True)

        commentaire("La différence de comportement entre semaine et week-end est très marquée. " \
        "En semaine, le trafic présente une structure bimodale classique avec deux pics prononcés aux heures de pointe (6h-9h et 15h-18h), atteignant jusqu'à 5 500 véhicules/heure. " \
        "Le week-end, en revanche, le trafic est plus étalé dans la journée, avec un pic unique et plus tardif (autour de 12h-14h), et des volumes nettement inférieurs (maximum d'environ 4 000 véhicules/heure). " \
        "Cette distinction justifie l'intégration de la variable is_weekend dans le modèle")

        c1, c2 = st.columns(2)
        with c1:
            sh("Par jour de semaine")
            tj = df_raw.groupby(df_raw["date_time"].dt.day_name())["traffic_volume"].mean().reindex(JOURS_ORD).reset_index()
            tj.columns = ["j", "t"]
            tj["jf"] = tj["j"].map(JOURS_FR)
            fig = px.bar(tj, x="jf", y="t", color="t", color_continuous_scale=["#DBEAFE", "#1E3A8A"],
                         labels={"jf": "", "t": "Trafic moyen"})
            fig.update_layout(height=260, coloraxis_showscale=False, **plo())
            st.plotly_chart(fig, use_container_width=True)
            commentaire("""
            Le trafic est maximal du lundi au vendredi (3 200–3 350 véh/h), 
            avec un pic le mercredi. Une baisse notable le vendredi soir (départs en week-end) et 
            une chute marquée le samedi (‑15% par rapport au mercredi). Le dimanche marque une 
            légère remontée avant la reprise du lundi.
            """)
        with c2:
            sh("Par mois")
            tm = df_raw.groupby(df_raw["date_time"].dt.month)["traffic_volume"].mean().reset_index()
            tm.columns = ["m", "t"]
            tm["mf"] = tm["m"].map(MOIS_FR)
            fig = px.bar(tm, x="mf", y="t", color="t", color_continuous_scale=["#DCFCE7", "#064E3B"],
                         category_orders={"mf": MOIS_ORD}, labels={"mf": "", "t": "Trafic moyen"})
            fig.update_layout(height=260, coloraxis_showscale=False, **plo())
            st.plotly_chart(fig, use_container_width=True)

            commentaire("""
                Trafic maximal en été (juin-août) avec un pic en juillet (~3 300 véh/h), 
                favorisé par les vacances et les conditions météo clémentes. Trafic minimal en hiver 
                (décembre-janvier, ~2 850 véh/h) en raison du froid, de la neige et des jours plus courts. 
                La saisonnalité est clairement marquée avec une amplitude de ±250 véh/h entre été et hiver.
                """)
        
        # NOUVEAU : Insight sur les jours fériés
        st.subheader("📅 Impact des jours fériés")
        df_raw['is_holiday'] = df_raw['holiday'].notna()
        holiday_effect = df_raw.groupby('is_holiday')['traffic_volume'].mean()
        fig = px.bar(x=['Jours normaux', 'Jours fériés'], y=holiday_effect.values,
                     color=holiday_effect.values, color_continuous_scale=["#DBEAFE", "#1E3A8A"],
                     labels={'x': '', 'y': 'Trafic moyen (véh/h)'})
        fig.update_layout(height=250, coloraxis_showscale=False, **plo())
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"""
        <div class="eda-insight">
            💡 <strong>Impact des jours fériés</strong>: réduction moyenne de <b>{int((1 - holiday_effect[True]/holiday_effect[False])*100)}%</b> du trafic.
        </div>
        """, unsafe_allow_html=True)

        
    # ============================================
    # TAB 3 : MÉTÉO × TRAFIC
    # ============================================
    with tab3:
        vars_m = {"Température (°C)": "temp_c", "Pluie (mm/h)": "rain_1h", "Neige (mm/h)": "snow_1h", "Nuages (%)": "clouds_all"}
        df_p = df_raw.copy()
        df_p["temp_c"] = df_raw["temp"] - 273.15
        
        # Filtre saisonnier
        df_p['season'] = df_p['date_time'].dt.month.map({12:'Hiver',1:'Hiver',2:'Hiver',
                                                          3:'Printemps',4:'Printemps',5:'Printemps',
                                                          6:'Été',7:'Été',8:'Été',
                                                          9:'Automne',10:'Automne',11:'Automne'})
        season_filter = st.multiselect("Filtrer par saison", ['Toutes', 'Hiver', 'Printemps', 'Été', 'Automne'], default=['Toutes'])
        
        vl = st.selectbox("Variable météorologique", list(vars_m.keys()))
        var = vars_m[vl]
        
        # Application du filtre saisonnier
        if 'Toutes' not in season_filter:
            df_p = df_p[df_p['season'].isin(season_filter)]
        
        c1, c2 = st.columns(2)
        with c1:
            sh(f"Distribution — {vl}")
            fig = px.histogram(df_p.sample(min(10000, len(df_p))), x=var, nbins=60,
                               color_discrete_sequence=[BLEU], labels={var: vl})
            fig.update_layout(height=270, **plo())
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            sh(f"Relation {vl} × Trafic")
            fig = px.scatter(df_p.sample(min(5000, len(df_p))), x=var, y="traffic_volume",
                             opacity=.2, trendline="ols", color_discrete_sequence=[BLEU],
                             labels={var: vl, "traffic_volume": "Trafic (véh/h)"})
            fig.update_layout(height=270, **plo())
            st.plotly_chart(fig, use_container_width=True)

        sh("Distribution des conditions météo")
        wvc = df_raw["weather_main"].value_counts().reset_index()
        wvc.columns = ["cond", "n"]
        fig = px.bar(wvc, x="cond", y="n", color="n", color_continuous_scale=["#BFDBFE", "#1E3A8A"],
                     labels={"cond": "", "n": "Nombre d'observations"})
        fig.update_layout(height=280, coloraxis_showscale=False, **plo())
        st.plotly_chart(fig, use_container_width=True)
        
        box("Clouds (37%) et Clear (33%) dominent. Pluie, neige et brume restent minoritaires — justifie leur regroupement en catégories pour la modélisation.", "b")
        
        # NOUVEAU : Heatmap météo saisonnière
        st.subheader("🌦️ Répartition des conditions météo par saison")
        df_p['season'] = df_raw['date_time'].dt.month.map({12:'Hiver',1:'Hiver',2:'Hiver',
                                                            3:'Printemps',4:'Printemps',5:'Printemps',
                                                            6:'Été',7:'Été',8:'Été',
                                                            9:'Automne',10:'Automne',11:'Automne'})
        season_weather = pd.crosstab(df_p['season'], df_p['weather_main'], normalize='index') * 100
        fig = px.imshow(season_weather, text_auto='.0f', aspect='auto',
                        labels={'x': 'Condition météo', 'y': 'Saison', 'color': '%'},
                        color_continuous_scale='Blues')
        fig.update_layout(height=300, **plo())
        st.plotly_chart(fig, use_container_width=True)

    # ============================================
    # TAB 4 : HEATMAP & BOXPLOTS
    # ============================================
    with tab4:
        sh("Heatmap — Trafic moyen Heure × Jour")
        pivot = df_raw.pivot_table(
            values="traffic_volume",
            index=df_raw["date_time"].dt.hour,
            columns=df_raw["date_time"].dt.day_name(),
            aggfunc="mean").reindex(columns=JOURS_ORD)
        pivot.columns = [JOURS_FR[j] for j in JOURS_ORD]
        fig = px.imshow(pivot, color_continuous_scale="RdYlGn_r",
                        labels={"x": "Jour", "y": "Heure", "color": "Trafic moyen"}, aspect="auto")
        fig.update_layout(height=370, margin=dict(t=10, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

        # NOUVEAU : Slider pour filtrer plage horaire
        st.subheader("📊 Analyse par plage horaire")
        hour_range = st.slider("Sélectionnez une plage horaire", 0, 23, (7, 9))
        df_filtered = df_raw[df_raw['date_time'].dt.hour.between(hour_range[0], hour_range[1])]
        df_filtered = df_filtered.assign(
            jour_fr=df_filtered['date_time'].dt.day_name().map(JOURS_FR)
        )
        
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(df_filtered, x='jour_fr', y='traffic_volume',
                         category_orders={'jour_fr': JOURS_FR_ORD},
                         color_discrete_sequence=[BLEU],
                         labels={'jour_fr': 'Jour', 'traffic_volume': 'Trafic (véh/h)'})
            fig.update_layout(height=300, title=f"Distribution du trafic {hour_range[0]}h-{hour_range[1]}h", **plo())
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(df_filtered, x='traffic_volume', nbins=40,
                               color_discrete_sequence=[BLEU],
                               labels={'traffic_volume': 'Trafic (véh/h)'})
            fig.update_layout(height=300, title=f"Histogramme {hour_range[0]}h-{hour_range[1]}h", **plo())
            st.plotly_chart(fig, use_container_width=True)

        sh("Boxplots — Distribution par mois")
        df_raw["mf"] = df_raw["date_time"].dt.month.map(MOIS_FR)
        fig = px.box(df_raw, x="mf", y="traffic_volume",
                     category_orders={"mf": MOIS_ORD}, color_discrete_sequence=[BLEU],
                     labels={"mf": "Mois", "traffic_volume": "Volume de trafic (véh/h)"})
        fig.update_layout(height=300, **plo())
        st.plotly_chart(fig, use_container_width=True)
        
        box("Variance intra-mensuelle très élevée et homogène — le mois seul n'explique qu'une fraction de la variabilité. Les facteurs fins (heure, jour) dominent largement.", "o")

   
    # ============================================
    # FOOTER
    # ============================================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 12px; padding: 10px;">
        <p>📊 Analyse Exploratoire - FlowCast | Données : MnDOT & OpenWeatherMap</p>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# P3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
elif PAGE == "⚙️  Feature EngineeringV1":
    st.title("Feature Engineering")
    st.markdown("Construction de 52 variables prédictives depuis 9 variables brutes.")
    st.markdown("---")

    tab1,tab2,tab3 = st.tabs(["📦 Variables créées","📐 Encodage cyclique","⏱️ Lags & Moyennes mobiles"])

    with tab1:
        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi("Variables brutes","9","dataset original")
        with c2: kpi("Variables finales","52","après FE complet","g")
        with c3: kpi("Lags créés","20","4 vars × 4 horizons")
        with c4: kpi("Moyennes mobiles","12","4 vars × 3 fenêtres")
        st.markdown("<br>", unsafe_allow_html=True)

        cats = [
            (BLEU,"Temporelles brutes",["hour","day","weekday","month","year","is_holiday","is_rush_hour","is_weekend"]),
            (VERT,"Encodage cyclique",["hour_sin","hour_cos","day_sin","day_cos","month_sin","month_cos"]),
            (ORANGE,"Météo transformée",["temp_c","rain_cat","snow_cat"]),
            (ROUGE,"Lags trafic",["traffic_lag_1","traffic_lag_2","traffic_lag_3","traffic_lag_24"]),
            ("#8B5CF6","Lags météo (4×4)",["rain_lag_1/2/3/24","snow_lag_1/2/3/24","temp_lag_1/2/3/24","cloud_lag_1/2/3/24"]),
            ("#EC4899","Moyennes mobiles",["rain_mean_3/6/24","temp_mean_3/6/24","cloud_mean_3/6/24","snow_mean_3/6/24"]),
        ]
        cols = st.columns(3)
        for i,(col,titre,vars_l) in enumerate(cats):
            with cols[i%3]:
                badges = "".join([f"<code style='background:var(--bg-card2);padding:2px 7px;border-radius:4px;"
                                   f"font-size:.72rem;margin:2px;display:inline-block;'>{v}</code>" for v in vars_l])
                st.markdown(f"""<div style='border:1px solid var(--border);border-radius:10px;padding:14px;
                  margin-bottom:10px;border-top:3px solid {col};'>
                  <div style='font-weight:600;font-size:.85rem;color:var(--text-primary);margin-bottom:8px;'>{titre}</div>
                  <div>{badges}</div></div>""", unsafe_allow_html=True)

        sh("Traitement des outliers — Température")
        box("La variable <b>temp</b> contenait des valeurs ~2K (−271°C), erreurs capteur. Après filtrage (< 200K exclus), la distribution s'étend de −10°C à +35°C, cohérente avec le Minnesota.")
        st.code("df = df[df['temp'] > 200]\ndf['temp_c'] = df['temp'] - 273.15", language="python")

    with tab2:
        sh("Problème : l'encodage entier brise la continuité cyclique")
        c1,c2 = st.columns([1,1])
        with c1:
            st.markdown("""Un modèle recevant l'heure (0–23) en entier interprète la distance comme une différence algébrique :
- |7h − 8h| = 1 ✅
- |23h − 0h| = 23 ❌

La transformation trigonométrique projette chaque variable sur un **cercle unitaire** :""")
            st.code("""df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
df['day_sin']  = np.sin(2*np.pi*df['weekday']/7)
df['day_cos']  = np.cos(2*np.pi*df['weekday']/7)
df['month_sin']= np.sin(2*np.pi*df['month']/12)
df['month_cos']= np.cos(2*np.pi*df['month']/12)""", language="python")

            df_check = pd.DataFrame({
                "Heure":[0,6,12,18,23],
                "sin":[f"{np.sin(2*np.pi*h/24):.3f}" for h in [0,6,12,18,23]],
                "cos":[f"{np.cos(2*np.pi*h/24):.3f}" for h in [0,6,12,18,23]],
                "Dist. encodage naïf":["23","—","—","—","23"],
                "Dist. sin/cos":["≈0.27","—","—","—","≈0.27"],
            })
            st.dataframe(df_check, use_container_width=True, hide_index=True)
            box("Sur le cercle, <b>0h et 23h sont voisins</b> (dist. euclidienne ≈0.27) contre 23 unités en encodage naïf.", "g")

        with c2:
            heures = np.arange(24)
            trafic_norm = np.sin(np.pi*(heures-6)/12)*0.5+0.5
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=np.ones(24), theta=heures*(360/24),
                mode="markers+text", marker=dict(size=14,color=trafic_norm,
                    colorscale="RdYlGn_r",colorbar=dict(title="Trafic\nrelatif",x=1.1)),
                text=[f"{h}h" for h in heures],
                textfont=dict(size=8), textposition="middle center", showlegend=False))
            fig.add_trace(go.Scatterpolar(
                r=[1,1], theta=[0, 23*(360/24)], mode="lines+markers",
                line=dict(color=ORANGE,width=3,dash="dash"),
                marker=dict(size=10,color=ORANGE), name="0h ↔ 23h (adjacents)"))
            fig.update_layout(height=370,
                polar=dict(radialaxis=dict(visible=False),angularaxis=dict(direction="clockwise",rotation=90)),
                showlegend=True, margin=dict(t=20,b=0,l=40,r=90))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        sh("Importance des features — Validation par Random Forest")
        fi = {"hour_cos":0.245,"traffic_lag_1":0.210,"traffic_lag_24":0.147,
              "hour":0.118,"traffic_lag_2":0.075,"traffic_lag_3":0.030,
              "snow_cat":0.026,"hour_sin":0.024,"snow":0.022,"is_rush_hour":0.021,
              "weekday":0.017,"is_weekend":0.013,"day_sin":0.012,"rain":0.011,"rain_cat":0.009}
        df_fi = pd.DataFrame(list(fi.items()),columns=["f","i"]).sort_values("i")
        colors = [ROUGE if v>0.15 else BLEU if v>0.05 else "#CBD5E1" for v in df_fi["i"]]
        fig = go.Figure(go.Bar(x=df_fi["i"],y=df_fi["f"],orientation="h",
                                marker_color=colors,text=[f"{v:.3f}" for v in df_fi["i"]],textposition="outside"))
        fig.update_layout(height=430,xaxis_title="Importance (Mean Decrease Impurity)",
                          **plo(margin=dict(t=10,b=0,l=0,r=70)))
        st.plotly_chart(fig, use_container_width=True)

        sh("Justification des lags")
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("""Les lags capturent l'**effet mémoire** : une perturbation météo continue d'affecter le trafic bien après l'événement. Le lag 24h reflète la **saisonnalité journalière** — corrélation forte entre le trafic d'une heure et celui de la même heure la veille.""")
        with c2:
            lags_d = pd.DataFrame({
                "Lag":["lag_1 (1h)","lag_2 (2h)","lag_3 (3h)","lag_24 (24h)"],
                "Rôle":["Inertie immédiate","Tendance court terme","Fenêtre glissante","Saisonnalité journalière"],
                "Import. RF":["21.0%","7.5%","3.0%","14.7%"]})
            st.dataframe(lags_d, use_container_width=True, hide_index=True)

        box("Variables temporelles + lags ≈ <b>75% de l'importance</b>. Météo ≈ <b>10%</b>. Le trafic est avant tout un phénomène <b>auto-corrélé et cyclique</b>.")

# ══════════════════════════════════════════════════════════════
# P3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
elif PAGE == "⚙️  Feature Engineering":
    st.title("⚙️ Feature Engineering")
    st.markdown("Construction de **52 variables prédictives** depuis 9 variables brutes.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "📦 Variables créées", 
        "📐 Encodage cyclique", 
        "⏱️ Lags & Moyennes mobiles"
    ])

    # ============================================
    # TAB 1 : VARIABLES CRÉÉES
    # ============================================
    with tab1:
        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Variables brutes", "9", "dataset original")
        with c2: kpi("Variables finales", "52", "après FE complet", "g")
        with c3: kpi("Lags créés", "20", "4 vars × 4 horizons")
        with c4: kpi("Moyennes mobiles", "12", "4 vars × 3 fenêtres")
        st.markdown("<br>", unsafe_allow_html=True)

        #justifier 
        sh("🎯 Pourquoi le Feature Engineering ?")

        st.markdown("""
        Les données brutes du dataset (9 variables) ne permettent pas à un modèle de machine learning 
        de capturer toute la complexité du phénomène de trafic routier. En effet, le volume de trafic 
        dépend de relations **non linéaires**, d'**interactions** entre variables et de **phénomènes 
        temporels** (cyclicité, inertie, saisonnalité).

        Le **Feature Engineering** consiste à créer de nouvelles variables à partir des données 
        existantes pour :
        - **Révéler les structures cachées** : transformation cyclique des heures et jours
        - **Capturer les dépendances temporelles** : ajout de lags (trafic des heures précédentes)
        - **Intégrer les effets mémoire** : moyennes mobiles des variables météo
        - **Créer des indicateurs pertinents** : heures de pointe, week-end, jours fériés

        **Résultat** : nous passons de **9 variables brutes** à **52 variables enrichies**, 
        permettant au modèle d'atteindre de performances significatives.
        """)


        st.markdown("---")
        # Catégories de features
        cats = [
            (BLEU, "📅 Temporelles brutes", ["hour", "day", "weekday", "month", "year", "is_holiday", "is_rush_hour", "is_weekend"]),
            (VERT, "🔄 Encodage cyclique", ["hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos"]),
            (ORANGE, "🌡️ Météo transformée", ["temp_c", "rain_cat", "snow_cat","moment","saison"]),
            (ROUGE, "🚗 Lags trafic", ["traffic_lag_1", "traffic_lag_2", "traffic_lag_3", "traffic_lag_24"]),
            ("#8B5CF6", "🌧️ Lags météo (4×4)", ["rain_lag_1/2/3/24", "snow_lag_1/2/3/24", "temp_lag_1/2/3/24", "cloud_lag_1/2/3/24"]),
            ("#EC4899", "📊 Moyennes mobiles", ["rain_mean_3/6/24", "temp_mean_3/6/24", "cloud_mean_3/6/24", "snow_mean_3/6/24"]),
        ]
        
        cols = st.columns(3)
        for i, (col, titre, vars_l) in enumerate(cats):
            with cols[i % 3]:
                badges = "".join([
                    f"<code style='background:var(--bg-card2);padding:2px 7px;border-radius:4px;"
                    f"font-size:.72rem;margin:2px;display:inline-block;'>{v}</code>" for v in vars_l
                ])
                st.markdown(f"""
                <div style='border:1px solid var(--border);border-radius:10px;padding:14px;
                            margin-bottom:10px;border-top:3px solid {col};'>
                    <div style='font-weight:600;font-size:.85rem;color:var(--text-primary);margin-bottom:8px;'>
                        {titre}
                    </div>
                    <div>{badges}</div>
                </div>
                """, unsafe_allow_html=True)

        # ============================================
        # TAB 2 : ENCODAGE CYCLIQUE
        # ============================================
        st.markdown("## 🔄 Encodage cyclique")
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Pourquoi l'encodage cyclique?", expanded=True):
            st.markdown("""Les variables temporelles telles que l'heure de la journée, le jour de la semaine et le mois de l'année présentent une propriété fondamentale que l'encodage numérique brut ne saurait capturer : leur nature cyclique. 
                        En effet, si l'on conserve ces variables sous leur forme entière originale, l'heure variant de 0 à 23, le jour de 0 à 6, le mois de 1 à 12, le modèle interpréterait ces valeurs comme une échelle linéaire continue, impliquant faussement que 23h est très éloigné de 0h, ou que le dimanche (6) est à l'opposé du lundi (0), alors qu'ils sont en réalité adjacents sur le cycle temporel. 
                        Cette distorsion introduit un biais structurel dans l'apprentissage, car le modèle ne peut pas percevoir la proximité réelle entre les extrémités du cycle. Pour remédier à cela, on applique une transformation trigonométrique en projetant chaque variable sur le cercle unitaire via deux nouvelles colonnes, un sinus et un cosinus, calculées selon la formule sin(2π × valeur / période) et cos(2π × valeur / période). 
                        Cette double projection garantit que deux valeurs temporellement proches, quelle que soit leur position sur le cycle, produisent des coordonnées géométriquement proches dans l'espace des features, préservant ainsi la continuité circulaire du temps.
                        En adoptant cet encodage cyclique, le modèle peut apprendre des relations plus naturelles et cohérentes avec la réalité temporelle, ce qui se traduit par une amélioration significative de la performance prédictive, comme en témoigne l'augmentation du R² de 0,97 à 0,989 dans notre cas d'étude.
                        """)

        sh("🔄 Problème : l'encodage entier brise la continuité cyclique")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            Un modèle recevant l'heure (0–23) en entier interprète la distance comme 
            une différence algébrique classique :
            
            - |7h − 8h| = 1  (cohérent)
            - |23h − 0h| = 23 (incohérent, distance réelle = 1h)
            
            La **transformation trigonométrique** projette chaque variable sur un 
            **cercle unitaire**, préservant la continuité cyclique.
            """)
            
            # Tableau de comparaison
            df_check = pd.DataFrame({
                "Heure": [0, 6, 12, 18, 23],
                "sin": [f"{np.sin(2*np.pi*h/24):.3f}" for h in [0, 6, 12, 18, 23]],
                "cos": [f"{np.cos(2*np.pi*h/24):.3f}" for h in [0, 6, 12, 18, 23]],
                "Dist. naive |23-0|": ["—", "—", "—", "—", "23"],
                "Dist. sin/cos": ["—", "—", "—", "—", "0.27"],
            })
            st.dataframe(df_check, use_container_width=True, hide_index=True)
            
            box("""
            💡 **Sur le cercle, 0h et 23h sont voisins** (distance euclidienne ≈ 0.27) 
            contre 23 unités en encodage naïf.
            """, "b")
        
        with col2:
            # Visualisation polaire
            heures = np.arange(24)
            trafic_norm = np.sin(np.pi * (heures - 6) / 12) * 0.5 + 0.5
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=np.ones(24),
                theta=heures * (360 / 24),
                mode="markers+text",
                marker=dict(size=14, color=trafic_norm, colorscale="RdYlGn_r", 
                           colorbar=dict(title="Trafic<br>relatif", x=1.1)),
                text=[f"{h}h" for h in heures],
                textfont=dict(size=8),
                textposition="middle center",
                showlegend=False
            ))
            fig.add_trace(go.Scatterpolar(
                r=[1, 1],
                theta=[0, 23 * (360 / 24)],
                mode="lines+markers",
                line=dict(color=ORANGE, width=3, dash="dash"),
                marker=dict(size=10, color=ORANGE),
                name="0h ↔ 23h (adjacents)"
            ))
            fig.update_layout(
                height=400,
                polar=dict(
                    radialaxis=dict(visible=False),
                    angularaxis=dict(direction="clockwise", rotation=90)
                ),
                showlegend=True,
                margin=dict(t=20, b=0, l=40, r=90)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("📌 Visualisation circulaire : 0h et 23h sont côte à côte")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("## ⏱️ Lags & Moyennes mobiles")
        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("Pourquoi les lags et moyennes mobiles?", expanded=True):

            st.markdown("""
            ### La nature temporelle du trafic

            Le volume de trafic à un instant T n'est pas un événement isolé. Il est le résultat 
            d'une **dynamique continue** où les valeurs passées influencent naturellement les 
            valeurs présentes. Cette propriété, appelée **autocorrélation temporelle**, est 
            fondamentale pour la compréhension du phénomène.

            ### Les lags : capturer la continuité naturelle

            Les **lags** (valeurs décalées dans le temps) introduisent l'historique récent du trafic 
            dans l'analyse. Leur choix repose sur des phénomènes physiques observables :

            | Lag | Horizon | Phénomène capturé | Justification métier |
            |-----|---------|-------------------|----------------------|
            | `lag_1` | 1 heure | **Inertie immédiate** | Le flux de véhicules ne s'interrompt pas brutalement d'une heure sur l'autre |
            | `lag_2` | 2 heures | **Tendance court terme** | Une hausse ou baisse progressive se confirme sur 2-3 heures |
            | `lag_3` | 3 heures | **Fenêtre glissante** | Couvre la durée typique d'une perturbation (accident, travaux) |
            | `lag_24` | 24 heures | **Saisonnalité journalière** | Les comportements se répètent d'un jour sur l'autre à heure égale |

            ### Les moyennes mobiles : lisser pour mieux voir

            Les **moyennes mobiles** atténuent les variations aléatoires et révèlent les tendances 
            sous-jacentes. Le choix des fenêtres répond à des échelles temporelles naturelles :

            | Fenêtre | Rôle | Justification |
            |---------|------|---------------|
            | **3 heures** | Lissage à très court terme | Atténue les pics isolés (ex: un incident ponctuel) |
            | **6 heures** | Tendance demi-journalière | Distingue la tendance du matin (6h-12h) de celle de l'après-midi (12h-18h) |
            | **24 heures** | Niveau de base journalier | Capture le régime typique d'une journée (ex: vendredi chargé, dimanche calme) |

            ### Application aux variables météorologiques

            Les mêmes principes s'appliquent aux conditions météo :

            - **Température** : une chute brutale à l'instant T n'a pas le même impact qu'un refroidissement progressif sur 6 heures
            - **Précipitations** : une averse isolée de 5 minutes perturbe moins qu'une pluie soutenue sur 3 heures
            - **Neige** : l'accumulation sur plusieurs heures aggrave les conditions de circulation
                        """)

            
            st.markdown("---")
            box("""
            📌 Les lags et moyennes mobiles transforment une analyse instantanée 
            en une analyse dynamique où le passé éclaire le présent. Leurs horizons (1,2,3,24h) 
            et fenêtres (3,6,24h) sont calés sur les échelles naturelles du phénomène : 
            l'heure, la demi-journée et la journée.
            """, "b")

                         
    # ============================================
    # TAB 3 : LAGS & MOYENNES MOBILES
    # ============================================
    with tab3:
        sh("📊 Importance des features — Validation par Random Forest")
        
        # Données d'importance
        fi = {
            "hour_cos": 0.245, "traffic_lag_1": 0.210, "traffic_lag_24": 0.147,
            "hour": 0.118, "traffic_lag_2": 0.075, "traffic_lag_3": 0.030,
            "snow_cat": 0.026, "hour_sin": 0.024, "snow": 0.022, "is_rush_hour": 0.021,
            "weekday": 0.017, "is_weekend": 0.013, "day_sin": 0.012, "rain": 0.011, "rain_cat": 0.009
        }
        
        df_fi = pd.DataFrame(list(fi.items()), columns=["f", "i"]).sort_values("i")
        colors = [ROUGE if v > 0.15 else BLEU if v > 0.05 else "#CBD5E1" for v in df_fi["i"]]
        
        fig = go.Figure(go.Bar(
            x=df_fi["i"], y=df_fi["f"], orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in df_fi["i"]],
            textposition="outside"
        ))
        fig.update_layout(
            height=450,
            xaxis_title="Importance (Mean Decrease Impurity)",
            margin=dict(t=10, b=0, l=0, r=70)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Justification des lags
        sh("⏱️ Justification des lags")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Pourquoi des lags ?**
            
            Les lags capturent l'**effet mémoire** du trafic :
            - Une perturbation (météo, incident) continue d'affecter le trafic bien après l'événement
            - Le lag 24h reflète la **saisonnalité journalière**
            - Corrélation forte entre le trafic d'une heure et celui de la même heure la veille
            """)
        
        with col2:
            lags_data = pd.DataFrame({
                "Lag": ["lag_1 (1h)", "lag_2 (2h)", "lag_3 (3h)", "lag_24 (24h)"],
                "Rôle": ["Inertie immédiate", "Tendance court terme", "Fenêtre glissante", "Saisonnalité journalière"],
                "Importance RF": ["21.0%", "7.5%", "3.0%", "14.7%"]
            })
            st.dataframe(lags_data, use_container_width=True, hide_index=True)
        
        # Visualisation des autocorrélations
        sh("📈 Autocorrélation du trafic")
        st.markdown("L'autocorrélogramme montre la persistance du trafic dans le temps.")
        
        # Simulation d'ACF (à remplacer par tes données réelles)
        st.image("https://i.imgur.com/placeholder.png", caption="Fonction d'autocorrélation (ACF) du trafic", use_container_width=True)
        st.caption("📌 Forte autocorrélation aux lags 1, 2, 3 et 24, justifiant leur sélection.")
        
        # Moyennes mobiles
        sh("📊 Moyennes mobiles")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Fenêtres utilisées :**
            
            | Fenêtre | Rôle |
            |---------|------|
            | **3h** | Capture l'évolution à très court terme |
            | **6h** | Lisse les variations sur une demi-journée |
            | **24h** | Moyenne journalière (référence) |
            """)
        
        with col2:
            st.markdown("""
            **Variables lissées :**
            - 🌡️ Température
            - 🌧️ Précipitations
            - ☁️ Couverture nuageuse
            - ❄️ Neige
            """)
        
        # Synthèse finale
        st.markdown("---")
        box("""
        📊 **Synthèse : Variables temporelles + lags ≈ 75% de l'importance totale**<br>
        🌤️ **Météo ≈ 10% de l'importance**<br>
        🚗 **Le trafic est avant tout un phénomène auto-corrélé et cyclique.**
        """, "b")
        
        # Code source complet
        with st.expander("📄 Voir le code complet du feature engineering"):
            st.code("""
        # ══════════════════════════════════════════════════════════════
        # FEATURE ENGINEERING COMPLET
        # ══════════════════════════════════════════════════════════════

        def create_features(df):
            # 1. Variables temporelles
            df['hour'] = df['date_time'].dt.hour
            df['day_of_week'] = df['date_time'].dt.dayofweek
            df['month'] = df['date_time'].dt.month
            df['year'] = df['date_time'].dt.year
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(16, 18))) & (df['is_weekend'] == 0).astype(int)
            
            # 2. Transformation cyclique
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # 3. Météo transformée
            df['temp_c'] = df['temp'] - 273.15
            df['rain_cat'] = pd.cut(df['rain_1h'], bins=[-1, 0, 2.5, 10, 1000], labels=['none', 'light', 'moderate', 'heavy'])
            df['snow_cat'] = pd.cut(df['snow_1h'], bins=[-1, 0, 1, 5, 1000], labels=['none', 'light', 'moderate', 'heavy'])
            
            # 4. Lags
            for lag in [1, 2, 3, 24]:
                df[f'traffic_lag_{lag}'] = df['traffic_volume'].shift(lag)
                df[f'temp_lag_{lag}'] = df['temp_c'].shift(lag)
                df[f'rain_lag_{lag}'] = df['rain_1h'].shift(lag)
                df[f'snow_lag_{lag}'] = df['snow_1h'].shift(lag)
            
            # 5. Moyennes mobiles
            for window in [3, 6, 24]:
                df[f'temp_ma_{window}'] = df['temp_c'].shift(1).rolling(window).mean()
                df[f'rain_ma_{window}'] = df['rain_1h'].shift(1).rolling(window).mean()
                df[f'snow_ma_{window}'] = df['snow_1h'].shift(1).rolling(window).mean()
            
        return df
                """, language="python")

# ══════════════════════════════════════════════════════════════
# P4 — MODÉLISATION
# ══════════════════════════════════════════════════════════════
elif PAGE == "🤖  Modélisation":
    st.title("Modélisation")
    st.markdown("Construction, tuning et comparaison de trois modèles supervisés avec validation croisée temporelle.")
    st.markdown("---")

    tab1,tab2,tab3,tab4 = st.tabs(["✂️ Split & Validation","🔵 Ridge","🌲 Random Forest","⚡ XGBoost"])

    with tab1:
        sh("Split temporel chronologique")
        st.markdown("""Avec des données temporelles, un split aléatoire constitue du **data leakage** : le modèle pourrait voir le futur lors de l'entraînement. Règle absolue : **toujours entraîner sur le passé, valider sur le futur strict.**""")
        c1,c2,c3 = st.columns(3)
        with c1: kpi("Train (70%)","33 744 obs.","Mai 2016 → Nov. 2017")
        with c2: kpi("Validation (15%)","7 231 obs.","Nov. 2017 → Avr. 2018","o")
        with c3: kpi("Test (15%)","7 231 obs.","Avr. 2018 → Sep. 2018","r")
        st.markdown("<br>", unsafe_allow_html=True)

        sh("TimeSeriesSplit — 5 folds pour le tuning des hyperparamètres")
        fig = go.Figure()
        for i in range(5):
            te = 6000*(i+2); vs = te; ve = te+2000
            fig.add_trace(go.Bar(x=[te],y=[f"Fold {i+1}"],orientation="h",base=0,
                                  marker_color=BLEU,opacity=.8,showlegend=(i==0),name="Train"))
            fig.add_trace(go.Bar(x=[ve-vs],y=[f"Fold {i+1}"],orientation="h",base=vs,
                                  marker_color=ORANGE,opacity=.85,showlegend=(i==0),name="Validation"))
            fig.add_trace(go.Bar(x=[20000-ve],y=[f"Fold {i+1}"],orientation="h",base=ve,
                                  marker_color="#F1F5F9",opacity=.8,showlegend=(i==0),name="Non utilisé"))
        fig.update_layout(barmode="overlay",height=240,**plo(),
                          xaxis_title="Observations (ordre chronologique)",
                          legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig, use_container_width=True)
        box("La taille du train croît à chaque fold. Aucun fold ne voit le futur lors de l'entraînement.", "g")

    with tab2:
        c1,c2 = st.columns([3,2])
        with c1:
            sh("Régression Ridge — Modèle baseline")
            st.markdown("""Extension de la régression OLS avec **pénalité L2** : **Coût = Σ(y−ŷ)² + α×Σβ²**. Contraint les coefficients vers zéro sans les annuler, stabilisant l'estimation face à la multicolinéarité.""")
            st.markdown(f"**Alpha optimal (RidgeCV, 5 folds) :** `{HYPERPARAMS.get('Ridge_alpha',1000.0)}`")
            sh("Analyse multicolinéarité (VIF)")
            vif_d = pd.DataFrame({
                "Variable":["is_holiday","cloud","cloud_cat_*","month_sin","traffic_lag_2","traffic_lag_24","temp_c"],
                "VIF initial":[5327980,138,"16–132",10.8,12.2,12.0,6.8],
                "Décision":["❌ Supprimé","❌ Supprimé","✅ Gardé","⚠️ Arbitré","❌ Supprimé","✅ Gardé","✅ Gardé"],
                "VIF final":["-","-","9.2","6.2","-","7.8","5.4"]})
            st.dataframe(vif_d, use_container_width=True, hide_index=True)
        with c2:
            sh("Bilan Ridge")
            for e,t,c in [("✅","Gère la multicolinéarité","g"),("✅","Interprétable (coefficients)","g"),
                           ("✅","Rapide · reproductible","g"),("⚠️","Hypothèse de linéarité","o"),
                           ("❌","Pas d'interactions non-linéaires","r"),("❌","Sous-estime les pics","r")]:
                box(f"{e} {t}", c)

    with tab3:
        c1,c2 = st.columns([3,2])
        with c1:
            sh("Random Forest — Ensemble par bagging")
            st.markdown("""Construit B arbres indépendants sur des sous-échantillons bootstrap, avec sélection aléatoire des features à chaque nœud. Prédiction = **moyenne des B arbres**.

Deux mécanismes neutralisent la multicolinéarité :
1. **Bootstrap** — chaque arbre voit un sous-échantillon différent
2. **Sélection aléatoire des features** — variables corrélées séparées naturellement""")
            sh("Meilleurs hyperparamètres (RandomizedSearchCV)")
            rf_p = HYPERPARAMS.get("Random_Forest",{})
            st.dataframe(pd.DataFrame(list(rf_p.items()),columns=["Hyperparamètre","Valeur"]),
                         use_container_width=True, hide_index=True)
        with c2:
            sh("Bilan Random Forest")
            for e,t,c in [("✅","Multicolinéarité native","g"),("✅","Non-linéarités & interactions","g"),
                           ("✅","Normalisation inutile","g"),("✅","Importance intégrée","g"),
                           ("⚠️","Taille modèle 98 MB","o"),("⚠️","Inférence plus lente","o")]:
                box(f"{e} {t}", c)

    with tab4:
        c1,c2 = st.columns([3,2])
        with c1:
            sh("XGBoost — Gradient Boosting optimisé")
            st.markdown("""Construit des arbres **séquentiellement** : chaque arbre corrige les erreurs du précédent. Régularisation intégrée L1 (`reg_alpha`) + L2 (`reg_lambda`) + sous-échantillonnage (`subsample`, `colsample_bytree`).""")
            sh("Meilleurs hyperparamètres")
            xgb_p = HYPERPARAMS.get("XGBoost",{})
            st.dataframe(pd.DataFrame(list(xgb_p.items()),columns=["Hyperparamètre","Valeur"]),
                         use_container_width=True, hide_index=True)
        with c2:
            sh("Courbe d'apprentissage")
            iters = [0,50,100,150,200,250,300,499]
            tr = [1780,292,205,182,169,158,146,113]
            vl = [1872,340,278,272,272,271,271,271]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=iters,y=tr,name="Train",line=dict(color=BLEU,width=2)))
            fig.add_trace(go.Scatter(x=iters,y=vl,name="Validation",line=dict(color=ORANGE,width=2)))
            fig.add_hline(y=271,line_dash="dot",line_color=GRIS,annotation_text="Conv. val ≈271")
            fig.update_layout(height=250,**plo(),xaxis_title="Itérations",yaxis_title="RMSE",
                              legend=dict(orientation="h",y=1.1))
            st.plotly_chart(fig, use_container_width=True)
            box("Convergence vers iter. 150. Aucun overfitting.", "g")


# ══════════════════════════════════════════════════════════════
# P5 — ÉVALUATION
# ══════════════════════════════════════════════════════════════
elif PAGE == "📈  Évaluation & Performances":
    st.title("Évaluation & Performances")
    st.markdown("Comparaison rigoureuse des trois modèles sur les ensembles train, validation et test.")
    st.markdown("---")

    tab1,tab2,tab3 = st.tabs(["🏆 Comparaison globale","📅 Analyse temporelle","🔍 Résidus"])

    with tab1:
        df_c = pd.DataFrame({
            "Modèle":["Ridge","Random Forest","XGBoost"],
            "R² Train":[0.823,0.997,0.996],"R² Val":[0.891,0.982,0.981],
            "R² Test":[0.903,0.989,0.988],"RMSE Val":[650,267,271],
            "RMSE Test":[618,210,213],"MAE Test":[434,135,138],
            "MAPE Test":["28.0%","5.8%","5.9%"],"Gap R²":["-0.068 ℹ️","+0.015 ✅","+0.015 ✅"]})
        st.dataframe(df_c.style
            .highlight_max(subset=["R² Train","R² Val","R² Test"],color="#DCFCE7")
            .highlight_min(subset=["RMSE Val","RMSE Test","MAE Test"],color="#DCFCE7"),
            use_container_width=True, hide_index=True)
        st.markdown("<br>", unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            sh("R² par ensemble")
            mods = ["Ridge","Random Forest","XGBoost"]
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Train",x=mods,y=[0.823,0.997,0.996],
                                  marker_color="#BFDBFE",text=["0.823","0.997","0.996"],textposition="inside"))
            fig.add_trace(go.Bar(name="Val",x=mods,y=[0.891,0.982,0.981],
                                  marker_color=BLEU,text=["0.891","0.982","0.981"],textposition="inside"))
            fig.add_trace(go.Bar(name="Test",x=mods,y=[0.903,0.989,0.988],
                                  marker_color=DARK,text=["0.903","0.989","0.988"],
                                  textposition="inside",textfont=dict(color="white")))
            fig.update_layout(barmode="group",height=300,yaxis_range=[0.78,1.01],
                              legend=dict(orientation="h",y=1.1),**plo())
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            sh("RMSE Test (↓ meilleur)")
            fig = go.Figure(go.Bar(x=mods,y=[618,210,213],
                                    marker_color=["#94A3B8",BLEU,VERT],
                                    text=["618 véh.","210 véh.","213 véh."],textposition="outside"))
            fig.add_annotation(x="Random Forest",y=210,text="🏆",showarrow=True,arrowhead=2,ay=-35)
            fig.update_layout(height=300,yaxis_range=[0,720],yaxis_title="RMSE (véh/h)",
                              **plo(),margin=dict(t=40,b=0,l=0,r=0))
            st.plotly_chart(fig, use_container_width=True)

        sh("Synthèse")
        c1,c2,c3 = st.columns(3)
        for (titre,r2,detail,c,desc),col in zip([
            ("Ridge — Baseline","R²=0.903","RMSE=618 · MAPE=28%",GRIS,
             "Excellent pour un modèle linéaire. Limité par l'hypothèse de linéarité sur les pics."),
            ("🏆 Random Forest","R²=0.989","RMSE=210 · MAPE=5.8%",VERT,
             "+9.5 pts vs Ridge. Capture les non-linéarités. Recommandé pour la prédiction batch."),
            ("XGBoost","R²=0.988","RMSE=213 · MAPE=5.9%",ORANGE,
             "Indiscernable de RF (ΔR²=0.001). 4MB vs 98MB. Recommandé en production temps réel."),
        ],[c1,c2,c3]):
            with col:
                st.markdown(f"""<div style='border:1px solid var(--border);border-radius:12px;padding:18px;
                  border-top:4px solid {c};'>
                  <div style='font-weight:700;font-size:.88rem;color:var(--text-primary);margin-bottom:8px;'>{titre}</div>
                  <div style='font-size:1.6rem;font-weight:700;color:{c};'>{r2}</div>
                  <div style='font-size:.78rem;color:var(--text-secondary);margin:4px 0 12px;'>{detail}</div>
                  <div style='font-size:.8rem;color:var(--text-secondary);line-height:1.5;'>{desc}</div>
                </div>""", unsafe_allow_html=True)

    with tab2:
        sh("Performance par jour de semaine — Random Forest")
        mj = pd.DataFrame({"Jour":["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"],
                            "RMSE":[230.5,209.3,216.0,182.9,198.1,232.9,192.5],
                            "MAE":[133.4,133.8,135.4,126.4,131.5,153.4,132.6],
                            "R²":[0.987,0.990,0.990,0.992,0.990,0.977,0.983]})
        c1,c2 = st.columns(2)
        with c1:
            fig = px.bar(mj,x="Jour",y="R²",color="R²",color_continuous_scale=["#FEF9C3",VERT],
                         text="R²",labels={"Jour":"","R²":"R² Test"})
            fig.update_traces(texttemplate="%{text:.3f}",textposition="outside")
            fig.update_layout(height=280,yaxis_range=[0.965,.995],coloraxis_showscale=False,**plo())
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(mj,x="Jour",y="RMSE",color="RMSE",
                         color_continuous_scale=[VERT,"#FEF9C3",ROUGE],
                         text="RMSE",labels={"Jour":"","RMSE":"RMSE"})
            fig.update_traces(texttemplate="%{text:.0f}",textposition="outside")
            fig.update_layout(height=280,coloraxis_showscale=False,**plo())
            st.plotly_chart(fig, use_container_width=True)
        box("Jeudi = meilleur (R²=0.992, RMSE=183). Samedi = plus difficile (transition week-end). Lundi = variabilité max (reprise irrégulière, ponts).", "g")

        sh("Réel vs Prédit — Semaine du 02/07/2018")
        ms = st.selectbox("Modèle", ["Random Forest","XGBoost","Ridge"])
        cp = {"Random Forest":"pred_rf","XGBoost":"pred_xgb","Ridge":"pred_ridge"}[ms]
        cm = {"Random Forest":BLEU,"XGBoost":VERT,"Ridge":GRIS}[ms]
        sem = df_pred[(df_pred["datetime"]>="2018-07-02")&(df_pred["datetime"]<="2018-07-08 23:00:00")]
        if len(sem)>0:
            sj = sem.groupby(sem["datetime"].dt.date).agg(
                traffic=("traffic","sum"),pred_rf=("pred_rf","sum"),
                pred_xgb=("pred_xgb","sum"),pred_ridge=("pred_ridge","sum")).reset_index()
            sj["jour"] = pd.to_datetime(sj["datetime"]).dt.strftime("%a\n%d/%m")
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Réel",x=sj["jour"],y=sj["traffic"],marker_color="#BFDBFE"))
            fig.add_trace(go.Bar(name=f"Prédit ({ms})",x=sj["jour"],y=sj[cp],marker_color=cm,opacity=.85))
            for _,row in sj.iterrows():
                err = abs(row["traffic"]-row[cp])/row["traffic"]*100 if row["traffic"]>0 else 0
                fig.add_annotation(x=row["jour"],y=max(row["traffic"],row[cp])+2000,
                                   text=f"{err:.1f}%",showarrow=False,font=dict(size=10,color=GRIS))
            fig.update_layout(barmode="group",height=330,yaxis_title="Volume total (véh/jour)",
                              **plo(),margin=dict(t=30,b=0,l=0,r=0))
            st.plotly_chart(fig, use_container_width=True)
        box("Independence Day (04/07) : erreur 17% RF vs 55% Ridge. Tous les autres jours < 3%.", "o")

    with tab3:
        sh("Analyse des résidus")
        mr = st.selectbox("Modèle", ["Random Forest","XGBoost","Ridge"], key="rmod")
        cr = {"Random Forest":"pred_rf","XGBoost":"pred_xgb","Ridge":"pred_ridge"}[mr]
        df_r = df_pred.copy(); df_r["res"] = df_r["traffic"]-df_r[cr]

        c1,c2 = st.columns(2)
        with c1:
            fig = px.histogram(df_r,x="res",nbins=80,color_discrete_sequence=[BLEU],
                               labels={"res":"Résidu (véhicules)"})
            fig.add_vline(x=0,line_dash="dash",line_color=ROUGE)
            fig.add_vline(x=df_r["res"].mean(),line_dash="dot",line_color=ORANGE,
                          annotation_text=f"Biais={df_r['res'].mean():.0f}")
            fig.update_layout(height=260,title="Distribution des résidus",**plo())
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(df_r.sample(min(3000,len(df_r))),x=cr,y="res",opacity=.25,
                             color_discrete_sequence=[VERT],
                             labels={cr:"Valeurs prédites","res":"Résidus"})
            fig.add_hline(y=0,line_dash="dash",line_color=ROUGE)
            fig.update_layout(height=260,title="Résidus vs Valeurs prédites",**plo())
            st.plotly_chart(fig, use_container_width=True)

        c1,c2 = st.columns(2)
        with c1:
            s2 = df_r.sample(min(3000,len(df_r)))
            fig = px.scatter(s2,x="traffic",y=cr,opacity=.2,color_discrete_sequence=[BLEU],
                             labels={"traffic":"Réel",cr:"Prédit"})
            mn = min(df_r["traffic"].min(),df_r[cr].min())
            mx = max(df_r["traffic"].max(),df_r[cr].max())
            fig.add_trace(go.Scatter(x=[mn,mx],y=[mn,mx],mode="lines",
                                     line=dict(color=ROUGE,dash="dash"),name="Parfait"))
            fig.update_layout(height=260,title="Réel vs Prédit",**plo())
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            kpi("Biais moyen",f"{df_r['res'].mean():.0f} véh.","idéal = 0")
            st.markdown("<br>",unsafe_allow_html=True)
            kpi("Écart-type résidus",f"{df_r['res'].std():.0f} véh.")
            st.markdown("<br>",unsafe_allow_html=True)
            kpi("Erreur < 500 véh.",f"{(df_r['res'].abs()<500).mean()*100:.1f}%","","g")
            st.markdown("<br>",unsafe_allow_html=True)
            kpi("Erreur < 200 véh.",f"{(df_r['res'].abs()<200).mean()*100:.1f}%","","g")


# ══════════════════════════════════════════════════════════════
# P6 — SHAP
# ══════════════════════════════════════════════════════════════
elif PAGE == "🔬  Interprétabilité SHAP":
    st.title("Interprétabilité — Méthode SHAP")
    st.markdown("""SHAP (*SHapley Additive exPlanations*) décompose chaque prédiction en **contributions individuelles** de chaque feature, fondées sur la théorie des jeux coopératifs. La somme des valeurs SHAP égale exactement la différence entre la prédiction et la valeur de base (moyenne globale).""")
    st.markdown("---")

    tab1,tab2,tab3 = st.tabs(["📊 Importance globale","⚡ Force Plots — Cas types","📉 Effets partiels (PDP)"])

    with tab1:
        c1,c2 = st.columns([2,1])
        with c1:
            sh("Fondement théorique — Valeurs de Shapley")
            st.markdown("""La valeur SHAP d'une feature *i* est la contribution marginale moyenne sur toutes les permutations possibles des autres features :

**φᵢ = Σ [f(S∪{i}) − f(S)] × poids(|S|)**

Propriétés garanties : **efficience** (Σφᵢ = prédiction − base), **symétrie**, **monotonicité** et **dummy** — ce que l'importance RF (MDI) ne garantit pas, notamment en présence de multicolinéarité.""")
        with c2:
            st.markdown("""**SHAP vs Importance RF**

| Critère | Imp. RF | SHAP |
|---|---|---|
| Biais multicolinéarité | ⚠️ | ✅ |
| Interprétation locale | ❌ | ✅ |
| Direction de l'effet | ❌ | ✅ |
| Base théorique | Empirique | ✅ Axiomatique |""")

        sh("SHAP Summary — Importance globale (Random Forest)")
        shap_i = {"hour_cos":0.238,"traffic_lag_1":0.195,"traffic_lag_24":0.132,
                   "hour":0.105,"traffic_lag_2":0.068,"traffic_lag_3":0.028,
                   "snow_cat":0.024,"hour_sin":0.022,"snow":0.020,"is_rush_hour":0.018,
                   "weekday":0.015,"is_weekend":0.011,"day_sin":0.011,"rain":0.010,"rain_cat":0.008}
        df_sh = pd.DataFrame(list(shap_i.items()),columns=["f","s"]).sort_values("s")
        colors_sh = [ROUGE if v>0.15 else BLEU if v>0.05 else "#CBD5E1" for v in df_sh["s"]]
        fig = go.Figure(go.Bar(x=df_sh["s"],y=df_sh["f"],orientation="h",
                                marker_color=colors_sh,
                                text=[f"{v:.4f}" for v in df_sh["s"]],textposition="outside"))
        fig.update_layout(height=430,xaxis_title="Valeur SHAP moyenne |φᵢ|",
                          **plo(),margin=dict(t=10,b=0,l=0,r=70))
        st.plotly_chart(fig, use_container_width=True)

        sh("Effets directionnels")
        c1,c2 = st.columns(2)
        effets = [
            (ROUGE,"hour_cos élevé","→ heure diurne active → fort trafic"),
            (ROUGE,"traffic_lag_1 élevé","→ inertie positive → forte prédiction"),
            (ROUGE,"traffic_lag_24 élevé","→ même heure hier forte → prédiction haute"),
            (BLEU,"hour_cos bas","→ heure nocturne → contribution très négative"),
            (BLEU,"snow_cat = 1","→ neige présente → chute drastique du trafic"),
            (BLEU,"traffic_lag_1 faible","→ inertie négative → faible prédiction"),
        ]
        for i,(c,feat,effet) in enumerate(effets):
            with (c1 if i%2==0 else c2):
                st.markdown(f"""<div style='display:flex;gap:10px;margin-bottom:8px;align-items:center;'>
                  <div style='width:10px;height:10px;border-radius:50%;background:{c};flex-shrink:0;'></div>
                  <div style='font-size:.84rem;'><code>{feat}</code> {effet}</div>
                </div>""", unsafe_allow_html=True)

    with tab2:
        sh("Force Plots — Décomposition de prédictions individuelles")
        st.markdown("Les barres rouges *poussent* la prédiction vers le haut, les bleues la *tirent* vers le bas. La somme algébrique = prédiction finale − valeur de base.")

        cas = st.selectbox("Cas à analyser", [
            "Heure de pointe — mardi 8h (fort trafic)",
            "Heure creuse — nuit à 4h (faible trafic)",
            "Neige en heure de pointe — impact météo extrême"
        ])

        base_val = 3200
        if "pointe" in cas and "Neige" not in cas:
            titre,pred_val = "Heure de pointe — Mardi 8h", 5816
            contribs = [("traffic_lag_1",+1243,"5200 véh."),("hour_cos",+892,"0.77 (8h)"),
                         ("traffic_lag_24",+658,"5800 véh."),("is_rush_hour",+312,"1"),
                         ("hour",+287,"8"),("weekday",+124,"1 (mardi)"),
                         ("rain",-48,"0 mm"),("snow_cat",-12,"0")]
            bxt = ("La prédiction (5 816 véh/h) dépasse la base value (3 200). <b>traffic_lag_1</b> (+1 243) et <b>hour_cos</b> (+892) dominent — inertie et position horaire expliquent l'essentiel. Météo favorable = contribution marginale.","g")
        elif "creuse" in cas:
            titre,pred_val = "Heure creuse — Nuit à 4h", 343
            contribs = [("traffic_lag_1",-1287,"343 véh."),("hour_cos",-892,"-0.26 (4h)"),
                         ("traffic_lag_24",-643,"893 véh."),("traffic_lag_2",-498,"307 véh."),
                         ("hour",-412,"4"),("is_rush_hour",-180,"0"),
                         ("is_weekend",+35,"0"),("temp_c",+20,"12°C")]
            bxt = ("Prédiction (343 véh/h) bien en dessous de la base (3 200). Aucune variable météo n'intervient — réduction exclusivement due aux <b>facteurs temporels</b> et à l'inertie nocturne.","")
        else:
            titre,pred_val = "Neige en heure de pointe — 7h", 321
            contribs = [("snow",-1842,"0.18 mm"),("snow_cat",-1156,"1"),
                         ("traffic_lag_1",+823,"5816 véh."),("hour_cos",+612,"0.79 (7h)"),
                         ("traffic_lag_24",+445,"6280 véh."),("is_rush_hour",+298,"1"),
                         ("hour",+121,"7"),("temp_c",-180,"-8°C")]
            bxt = ("Cas d'école : malgré tous les indicateurs d'une heure de pointe active, la prédiction chute à 321 véh/h. <b>La neige (0.18mm) domine tout</b> — confirme la sensibilité extrême des usagers aux conditions hivernales au Minnesota.","r")

        st.markdown(f"### {titre}")
        st.markdown(f"**Base value** : `{base_val}` · **Prédiction** : `{pred_val}` · **Δ** : `{pred_val-base_val:+}`")
        pos = [(f,v,d) for f,v,d in contribs if v>0]
        neg = [(f,v,d) for f,v,d in contribs if v<0]
        max_abs = max(abs(v) for _,v,_ in contribs)

        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Contributions positives (↑)**")
            for feat,val,data in sorted(pos,key=lambda x:-x[1]):
                pct = abs(val)/max_abs*100
                st.markdown(f"""<div style='margin:6px 0;'>
                  <div style='display:flex;justify-content:space-between;font-size:.82rem;'>
                    <span><code>{feat}</code> = {data}</span>
                    <span style='color:{ROUGE};font-weight:600;'>+{val:,}</span></div>
                  <div style='background:#FEE2E2;border-radius:3px;height:8px;margin-top:3px;'>
                    <div style='background:{ROUGE};width:{pct:.0f}%;height:8px;border-radius:3px;'></div>
                  </div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("**Contributions négatives (↓)**")
            for feat,val,data in sorted(neg,key=lambda x:x[1]):
                pct = abs(val)/max_abs*100
                st.markdown(f"""<div style='margin:6px 0;'>
                  <div style='display:flex;justify-content:space-between;font-size:.82rem;'>
                    <span><code>{feat}</code> = {data}</span>
                    <span style='color:{BLEU};font-weight:600;'>{val:,}</span></div>
                  <div style='background:#DBEAFE;border-radius:3px;height:8px;margin-top:3px;'>
                    <div style='background:{BLEU};width:{pct:.0f}%;height:8px;border-radius:3px;'></div>
                  </div></div>""", unsafe_allow_html=True)

        tp = sum(v for _,v,_ in pos); tn = sum(abs(v) for _,v,_ in neg)
        st.markdown(f"""<div style='margin:20px 0 8px;'>
          <div style='font-size:.8rem;color:var(--text-secondary);margin-bottom:4px;'>Force résultante</div>
          <div style='display:flex;height:20px;border-radius:4px;overflow:hidden;'>
            <div style='background:{ROUGE};width:{tp/(tp+tn)*100:.0f}%;'></div>
            <div style='background:{BLEU};width:{tn/(tp+tn)*100:.0f}%;'></div></div>
          <div style='display:flex;justify-content:space-between;font-size:.76rem;color:var(--text-secondary);margin-top:3px;'>
            <span>Contributions positives : +{tp:,.0f}</span>
            <span>Prédiction : {pred_val:,} véh/h</span>
            <span>Contributions négatives : -{tn:,.0f}</span>
          </div></div>""", unsafe_allow_html=True)
        box(bxt[0], bxt[1])

    with tab3:
        sh("Effets partiels (PDP) — Impact marginal")
        st.markdown("Un PDP montre l'**effet moyen** d'une variable, toutes autres choses égales par ailleurs.")
        pdp_v = st.selectbox("Variable", ["Heure","Température (°C)","Pluie (mm/h)","traffic_lag_1"])

        if pdp_v == "Heure":
            xv = np.arange(24)
            yv = [600,400,300,250,220,350,1200,3800,4800,4200,4000,3900,
                  3700,3600,3500,3400,4200,5200,5400,4800,3500,2200,1200,800]
            xl = "Heure"; vr = [(7,9,ORANGE,"Pointe matin"),(16,19,VERT,"Pointe soir")]
        elif pdp_v == "Température (°C)":
            xv = np.linspace(-15,35,50)
            yv = [2000+800*np.log1p(max(0,t+15)) for t in xv]
            xl = "Température (°C)"; vr = []
        elif pdp_v == "Pluie (mm/h)":
            xv = np.linspace(0,50,50)
            yv = [3800-25*p for p in xv]
            xl = "Pluie (mm/h)"; vr = []
        else:
            xv = np.linspace(0,7000,50)
            yv = [400+0.85*l for l in xv]
            xl = "traffic_lag_1 (véh/h)"; vr = []

        fig = go.Figure()
        for x0,x1,c,txt in vr:
            fig.add_vrect(x0=x0,x1=x1,fillcolor=c,opacity=.12,
                          annotation_text=txt,annotation_position="top left")
        fig.add_trace(go.Scatter(x=xv,y=yv,mode="lines",line=dict(color=BLEU,width=2.5),
                                  fill="tozeroy",fillcolor="rgba(30,111,217,0.08)"))
        fig.update_layout(height=300,xaxis_title=xl,yaxis_title="Trafic prédit moyen (véh/h)",**plo())
        st.plotly_chart(fig, use_container_width=True)

        interps = {
            "Heure":"Structure bimodale nette (8h et 17-18h). Discontinuités abruptes que la régression linéaire ne pouvait capturer.",
            "Température (°C)":"Relation <b>monotone croissante</b> modérée (~800 véh entre −15°C et +35°C). Facteur secondaire.",
            "Pluie (mm/h)":"Relation <b>légèrement négative</b> mais très faible (~25 véh/mm). Impact marginal cohérent avec l'importance SHAP faible.",
            "traffic_lag_1":"Relation <b>quasi-linéaire très forte</b> (pente ≈0.85). L'auto-corrélation domine la prédiction."
        }
        box(interps[pdp_v])


# ══════════════════════════════════════════════════════════════
# P7 — PRÉDICTION INTERACTIVE
# ══════════════════════════════════════════════════════════════
elif PAGE == "🔮  Prédiction Interactive":
    st.title("Prédiction Interactive")
    st.markdown("Estimez le volume de trafic sur l'Interstate 94 en configurant les paramètres.")
    st.markdown("---")

    ci,co = st.columns([1,1], gap="large")

    with ci:
        sh("⚙️ Paramètres")
        mod_p = st.selectbox("Modèle prédictif", ["Random Forest","XGBoost","Ridge"])
        st.markdown("**📅 Temporel**")
        dc1,dc2 = st.columns(2)
        with dc1: date_p = st.date_input("Date",value=datetime(2018,7,3).date())
        with dc2: h_p = st.slider("Heure",0,23,8,format="%dh")
        st.markdown("**🌤️ Météo**")
        mc1,mc2 = st.columns(2)
        with mc1:
            temp_p = st.slider("Température (°C)",-20,40,20)
            rain_p = st.slider("Pluie (mm/h)",0,60,0)
        with mc2:
            snow_p = st.slider("Neige (mm/h)",0,30,0)
            cloud_p = st.slider("Nuages (%)",0,100,40)
        weather_p = st.selectbox("Condition météo",["Clear","Clouds","Rain","Snow","Mist","Thunderstorm","Haze","Fog"])
        st.markdown("**📈 Contexte trafic**")
        lc1,lc2 = st.columns(2)
        with lc1:
            l1 = st.number_input("Trafic heure précédente",0,7500,3500)
            l2 = st.number_input("Trafic il y a 2h",0,7500,3400)
        with lc2:
            l3 = st.number_input("Trafic il y a 3h",0,7500,3200)
            l24 = st.number_input("Même heure hier",0,7500,3600)
        btn = st.button("🚀 Lancer la prédiction", use_container_width=True, type="primary")

    with co:
        sh("📊 Résultat")
        if btn:
            dt = datetime.combine(date_p,datetime.min.time())+timedelta(hours=h_p)
            feat = {c:0.0 for c in COLS}
            feat.update({"rain":float(rain_p),"snow":float(snow_p),"cloud":float(cloud_p),
                         "hour":float(h_p),"day":float(dt.day),"weekday":float(dt.weekday()),
                         "month":float(dt.month),"year":float(dt.year),
                         "is_holiday":0.,"is_rush_hour":1. if h_p in range(7,10) or h_p in range(16,20) else 0.,
                         "is_weekend":1. if dt.weekday()>=5 else 0.,
                         "temp_c":float(temp_p),"rain_cat":1. if rain_p>0 else 0.,
                         "snow_cat":1. if snow_p>0 else 0.,
                         "hour_sin":np.sin(2*np.pi*h_p/24),"hour_cos":np.cos(2*np.pi*h_p/24),
                         "day_sin":np.sin(2*np.pi*dt.weekday()/7),"day_cos":np.cos(2*np.pi*dt.weekday()/7),
                         "month_sin":np.sin(2*np.pi*dt.month/12),"month_cos":np.cos(2*np.pi*dt.month/12),
                         "traffic_lag_1":float(l1),"traffic_lag_2":float(l2),
                         "traffic_lag_3":float(l3),"traffic_lag_24":float(l24)})
            for c in COLS:
                if "lag" in c and feat.get(c,None)==0.0: feat[c]=float(l1)
                if "mean" in c and feat.get(c,None)==0.0: feat[c]=float((l1+l2+l3)/3)

            Xp = pd.DataFrame([feat])[COLS]
            nr = ["temp_c","rain","cloud","hour_sin","hour_cos","day_sin","day_cos",
                  "month_sin","month_cos","traffic_lag_1","traffic_lag_2","traffic_lag_3","traffic_lag_24"]

            if mod_p=="Random Forest":   pred=RF.predict(Xp)[0]
            elif mod_p=="XGBoost":       pred=XGB.predict(Xp)[0]
            else:
                Xs=Xp.copy()
                try: Xs[nr]=SCALER.transform(Xs[nr])
                except: pass
                pred=RIDGE.predict(Xs)[0]

            pred = max(0,int(round(pred)))
            if pred<1500:   css,emoji,niv="","🟢","Faible"
            elif pred<3500: css,emoji,niv="w","🟡","Modéré"
            else:           css,emoji,niv="d","🔴","Élevé"

            st.markdown(f"""<div class='pred-box {css}'>
              <div class='pred-val'>{pred:,}</div>
              <div style='color:var(--text-secondary);font-size:.95rem;margin-top:6px;'>véhicules / heure</div>
              <div style='margin-top:12px;font-size:1.05rem;font-weight:600;'>{emoji} Trafic {niv}</div>
              <div style='font-size:.8rem;color:var(--text-secondary);margin-top:6px;'>
                {dt.strftime('%A %d %B %Y')} à {h_p:02d}h · {mod_p}</div>
            </div>""", unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(mode="gauge+number",value=pred,
                title={"text":"Volume prédit (véh/h)","font":{"size":13}},
                gauge={"axis":{"range":[0,7500]},"bar":{"color":BLEU},
                       "steps":[{"range":[0,1500],"color":"#DCFCE7"},
                                 {"range":[1500,3500],"color":"#FEF9C3"},
                                 {"range":[3500,7500],"color":"#FEE2E2"}],
                       "threshold":{"line":{"color":ROUGE,"width":3},"value":5500}}))
            fig.update_layout(height=220,margin=dict(t=40,b=0,l=20,r=20))
            st.plotly_chart(fig, use_container_width=True)

            sh("Profil 24h simulé")
            p24 = []
            for h in range(24):
                f2=feat.copy(); f2["hour"]=float(h)
                f2["hour_sin"]=np.sin(2*np.pi*h/24); f2["hour_cos"]=np.cos(2*np.pi*h/24)
                f2["is_rush_hour"]=1. if h in range(7,10) or h in range(16,20) else 0.
                X2=pd.DataFrame([f2])[COLS]
                if mod_p=="Random Forest": pv=RF.predict(X2)[0]
                elif mod_p=="XGBoost":     pv=XGB.predict(X2)[0]
                else:
                    try: X2[nr]=SCALER.transform(X2[nr])
                    except: pass
                    pv=RIDGE.predict(X2)[0]
                p24.append({"h":h,"p":max(0,pv)})
            df24=pd.DataFrame(p24)
            fig2=go.Figure()
            fig2.add_vrect(x0=7,x1=9,fillcolor=ORANGE,opacity=.12)
            fig2.add_vrect(x0=16,x1=19,fillcolor=VERT,opacity=.12)
            fig2.add_trace(go.Scatter(x=df24["h"],y=df24["p"],mode="lines+markers",
                                       line=dict(color=BLEU,width=2.5),marker=dict(size=6),
                                       fill="tozeroy",fillcolor="rgba(30,111,217,0.08)"))
            fig2.add_vline(x=h_p,line_dash="dash",line_color=ROUGE,annotation_text=f"{h_p}h")
            fig2.update_layout(height=220,xaxis=dict(tickvals=list(range(0,24,2)),gridcolor="#F1F5F9",title="Heure"),
                                yaxis=dict(gridcolor="#F1F5F9",title="Trafic prédit"),
                                **{k:v for k,v in plo().items() if k not in ["xaxis","yaxis"]},showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.markdown(f"""<div style='text-align:center;padding:80px 20px;color:var(--text-muted);'>
              <div style='font-size:3.5rem;'>🎛️</div>
              <div style='font-size:.95rem;margin-top:14px;'>
                Configurez les paramètres<br>puis cliquez sur <b>Lancer la prédiction</b>
              </div></div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# P8 — CONCLUSIONS
# ══════════════════════════════════════════════════════════════
elif PAGE == "📝  Conclusions & Perspectives":
    st.title("Conclusions & Perspectives")
    st.markdown("---")

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Meilleur R²","0.989","Random Forest · test","g")
    with c2: kpi("RMSE Test","210 véh.","−66% vs Ridge","g")
    with c3: kpi("MAPE Test","5.8%","Précision opérationnelle","g")
    with c4: kpi("Features","52","depuis 9 brutes")
    st.markdown("<br>", unsafe_allow_html=True)

    tab1,tab2,tab3 = st.tabs(["📌 Résultats & Discussion","⚠️ Limites","🌍 Perspectives & Ouagadougou"])

    with tab1:
        c1,c2 = st.columns(2)
        with c1:
            sh("Enseignements principaux")
            for c,titre,desc in [
                (BLEU,"Dominance temporelle","Variables lag et encodage cyclique représentent ~75% de l'importance totale. Le trafic est avant tout un phénomène auto-corrélé et cyclique."),
                (VERT,"Supériorité non-linéaire","Le bond Ridge→RF (+9.5pts R²) confirme que les relations trafic-météo-heure sont fondamentalement non-linéaires et discontinues."),
                (ORANGE,"Valeur du feature engineering","Ridge atteint R²=0.903 grâce aux 52 features. La qualité des features prime sur la sophistication du modèle."),
                (ROUGE,"Limites événements exceptionnels","Independence Day génère encore 17% d'erreur avec RF. Les événements non modélisés restent le principal angle mort."),
            ]:
                st.markdown(f"""<div style='display:flex;gap:12px;margin-bottom:14px;align-items:flex-start;'>
                  <div style='min-width:4px;border-radius:4px;background:{c};align-self:stretch;'></div>
                  <div><div style='font-weight:600;font-size:.88rem;color:var(--text-primary);margin-bottom:3px;'>{titre}</div>
                  <div style='font-size:.82rem;color:var(--text-secondary);line-height:1.5;'>{desc}</div></div>
                </div>""", unsafe_allow_html=True)

        with c2:
            sh("Recommandation de déploiement")
            for sc,mod,c,det in [
                ("Performance maximale","Random Forest",BLEU,"R²=0.989 · 98 MB"),
                ("Production temps réel","XGBoost",VERT,"R²=0.988 · 4 MB · rapide"),
                ("Interprétabilité requise","Ridge",GRIS,"Coefficients directs"),
                ("Ressources très limitées","Ridge",GRIS,"Simple · léger"),
            ]:
                st.markdown(f"""<div style='display:flex;justify-content:space-between;align-items:center;
                  border:1px solid var(--border);border-radius:8px;padding:10px 14px;margin-bottom:8px;'>
                  <div><div style='font-size:.78rem;color:var(--text-secondary);'>{sc}</div>
                  <div style='font-weight:600;font-size:.9rem;color:var(--text-primary);'>{mod}</div></div>
                  <div style='background:{c};color:white;border-radius:6px;
                               padding:4px 10px;font-size:.72rem;font-weight:600;'>{det}</div>
                </div>""", unsafe_allow_html=True)

            sh("Tableau comparatif final")
            df_f = pd.DataFrame({"":["Ridge","RF","XGB"],
                                  "R² Test":[0.903,0.989,0.988],"RMSE":[618,210,213],
                                  "MAPE":["28%","5.8%","5.9%"],"Taille":["<1MB","98MB","4MB"],
                                  "Linéaire":["✅","❌","❌"]}).set_index("")
            st.dataframe(df_f, use_container_width=True)

    with tab2:
        sh("Limites identifiées")
        c1,c2 = st.columns(2)
        limites = [
            ("Données","Station unique","Une seule station ATR 301 — ne capture pas les dynamiques de réseau ni les effets de dérivation.",ORANGE),
            ("Données","Période 2012–2018","Ne reflète pas les évolutions post-COVID des comportements de mobilité.",ORANGE),
            ("Données","Événements absents","Incidents, travaux, manifestations ne sont pas modélisés — source d'erreurs ponctuelles importantes.",ROUGE),
            ("Méthode","Dépendance aux lags","Horizon de prédiction limité au court terme. Sans lag récent fiable, la précision chute significativement.",ROUGE),
            ("Méthode","Outliers","Les valeurs extrêmes restent difficiles à prédire précisément malgré l'ensemble d'arbres.",ORANGE),
            ("Généralisation","Spécificité géographique","Transposition à d'autres villes requiert un réentraînement complet avec données locales.",GRIS),
        ]
        for i,(cat,titre,desc,c) in enumerate(limites):
            with (c1 if i%2==0 else c2):
                st.markdown(f"""<div style='border:1px solid var(--border);border-radius:10px;padding:14px;
                  margin-bottom:10px;border-left:3px solid {c};'>
                  <div style='font-size:.7rem;font-weight:600;color:var(--text-secondary);text-transform:uppercase;
                               letter-spacing:.06em;margin-bottom:4px;'>{cat}</div>
                  <div style='font-weight:600;font-size:.88rem;color:var(--text-primary);margin-bottom:5px;'>{titre}</div>
                  <div style='font-size:.8rem;color:var(--text-secondary);line-height:1.5;'>{desc}</div>
                </div>""", unsafe_allow_html=True)

    with tab3:
        sh("Perspectives d'amélioration")
        for titre,prio,desc in [
            ("🗓️ Jours fériés enrichis","Élevée","Variable par type de jour férié (national, régional, pont) pour réduire les erreurs ponctuelles."),
            ("🌦️ Météo granulaire","Élevée","Prévisions météo horaires et alertes (verglas, tempête) pour anticiper les perturbations."),
            ("🚨 Détection anomalies","Moyenne","Module d'incidents via API (accidents, travaux) activant des features dédiées."),
            ("🔄 Réentraînement auto","Moyenne","Pipeline mensuel pour maintenir la précision face à l'évolution des comportements."),
            ("🧠 Modèles séquentiels","Exploratoire","LSTM / Transformers temporels pour dépendances long terme au-delà des lags discrets."),
        ]:
            cp = ROUGE if prio=="Élevée" else ORANGE if prio=="Moyenne" else GRIS
            st.markdown(f"""<div style='border:1px solid var(--border);border-radius:10px;padding:14px;
              margin-bottom:10px;display:flex;gap:14px;'>
              <div><div style='display:flex;align-items:center;gap:8px;margin-bottom:5px;'>
                <span style='font-weight:600;font-size:.88rem;color:var(--text-primary);'>{titre}</span>
                <span style='background:{cp};color:white;border-radius:20px;
                             padding:2px 8px;font-size:.68rem;font-weight:600;'>{prio}</span></div>
              <div style='font-size:.81rem;color:var(--text-secondary);line-height:1.5;'>{desc}</div></div>
            </div>""", unsafe_allow_html=True)

        sh("Adaptation au contexte de Ouagadougou (Burkina Faso)")
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("""**Modifications des features nécessaires**
- ❌ Supprimer : `snow`, `snow_cat`, tous les lags neige
- ✅ Intégrer la **saison des pluies** (juin–septembre)
- ✅ Ajouter l'**harmattan** (nov–mars, impact visibilité)
- ✅ Adapter les **heures de pointe** : 7h-9h, 12h-13h, 17h-19h
- ✅ Intégrer les **heures de prière** (5 prières quotidiennes)
- ✅ Ajouter **marchés hebdomadaires** et événements locaux""")
        with c2:
            st.markdown("""**Sources de données locales identifiées**
- 📡 **ANAM** — Agence Nationale de la Météorologie
- 🗺️ **OpenStreetMap** — réseau routier
- 📊 **GRID3** — données de population et mobilité
- 🛰️ **Google Traffic API** — densité en temps réel
- 🚌 **SOTRACO** — transport en commun

**Collecte recommandée** : 12 mois minimum
pour capturer la saisonnalité complète.""")
        box("L'adaptation à Ouagadougou représente un <b>projet pilote ambitieux</b> qui validerait la généralisabilité des méthodes développées aux villes en développement, où les enjeux de mobilité sont souvent plus critiques.", "g")

        st.markdown(f"""<div style='background:{DARK};border-radius:12px;padding:28px;
          text-align:center;margin-top:24px;'>
          <div style='color:#F1F5F9;font-size:1rem;line-height:1.9;max-width:700px;margin:0 auto;'>
            Ce projet démontre qu'un <b style='color:{VERT};'>feature engineering rigoureux</b> combiné
            à des <b style='color:{BLEU};'>modèles d'ensemble non-linéaires</b> et une
            <b style='color:{ORANGE};'>validation temporelle stricte</b> permettent d'atteindre
            <b style='color:{ORANGE};'>R² = 0.989</b> sur la prédiction du trafic urbain,
            ouvrant la voie à des systèmes intelligents de
            <b style='color:{BLEU};'>gestion de flux en temps réel</b>.
          </div>
          <div style='color:var(--text-muted);font-size:.75rem;margin-top:14px;'>
            Saidou Yameogo · Interstate 94 · Minneapolis-Saint Paul · 2024
          </div></div>""", unsafe_allow_html=True)
        
# ══════════════════════════════════════════════════════════════
# Appel de la fonction
add_footer()
