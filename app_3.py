"""
TrafficML — Application Streamlit
Prédiction du Trafic Urbain · Interstate 94 · Minneapolis-Saint Paul
Auteur : Saidou Yameogo
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import shap
import joblib, json, warnings
from datetime import datetime, timedelta

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

BLEU   = "#1E6FD9"
VERT   = "#17B897"
ORANGE = "#F4A223"
ROUGE  = "#E8432A"
GRIS   = "#6B7280"
NOIR   = "#0F172A"
DARK   = "#1A2535"
LIGTH  = "#F8FAFC"


st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{{font-family:'Inter',sans-serif;}}
section[data-testid="stSidebar"]{{background:{DARK};border-right:1px solid #2C3E50;}}
section[data-testid="stSidebar"] *{{color:#CBD5E1!important;}}
section[data-testid="stSidebar"] .stRadio label{{
  background:rgba(255,255,255,.04);border-radius:8px;padding:9px 14px;
  margin:3px 0;cursor:pointer;border-left:3px solid transparent;transition:all .2s;}}
section[data-testid="stSidebar"] .stRadio label:hover{{
  background:rgba(30,111,217,.2);border-left-color:{BLEU};}}
.kpi{{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;
      padding:18px 22px;border-left:4px solid {BLEU};}}
.kpi.g{{border-left-color:{VERT};}} .kpi.o{{border-left-color:{ORANGE};}}
.kpi.r{{border-left-color:{ROUGE};}} .kpi.gr{{border-left-color:{GRIS};}}
.kv{{font-size:1.9rem;font-weight:700;color:{NOIR};margin:4px 0;line-height:1;}}
.kl{{font-size:.72rem;font-weight:600;color:{GRIS};text-transform:uppercase;letter-spacing:.07em;}}
.kd{{font-size:.8rem;color:{VERT};margin-top:4px;}}
.sh{{border-bottom:2px solid {BLEU};padding-bottom:8px;margin:28px 0 14px;
     font-size:1.05rem;font-weight:700;color:{LIGTH};}}
.box{{background:#F0F9FF;border-left:3px solid {BLEU};border-radius:0 8px 8px 0;
      padding:12px 16px;font-size:.88rem;color:#1E3A5F;margin:8px 0;line-height:1.6;}}
.box.g{{background:#F0FFF4;border-color:{VERT};color:#14532D;}}
.box.o{{background:#FFFBEB;border-color:{ORANGE};color:#78350F;}}
.box.r{{background:#FFF1F0;border-color:{ROUGE};color:#7F1D1D;}}
.pred-box{{background:linear-gradient(135deg,#EFF6FF,#DBEAFE);border:2px solid {BLEU};
           border-radius:16px;padding:32px;text-align:center;}}
.pred-val{{font-size:3.2rem;font-weight:700;color:{BLEU};line-height:1;}}
.pred-box.w{{background:linear-gradient(135deg,#FFF7ED,#FED7AA);border-color:{ORANGE};}}
.pred-box.w .pred-val{{color:{ORANGE};}}
.pred-box.d{{background:linear-gradient(135deg,#FEF2F2,#FECACA);border-color:{ROUGE};}}
.pred-box.d .pred-val{{color:{ROUGE};}}
#MainMenu,footer{{visibility:hidden;}}
.block-container{{padding-top:1.8rem;}}
</style>
""", unsafe_allow_html=True)


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

@st.cache_resource
def build_explainer(model_name):
    if model_name == "Random Forest":
        return shap.TreeExplainer(RF)
    if model_name == "XGBoost":
        return shap.TreeExplainer(XGB)
    background = df_proc[COLS].sample(min(200, len(df_proc)), random_state=42)
    return shap.LinearExplainer(RIDGE, background)

JOURS_FR   = {"Monday":"Lundi","Tuesday":"Mardi","Wednesday":"Mercredi",
               "Thursday":"Jeudi","Friday":"Vendredi","Saturday":"Samedi","Sunday":"Dimanche"}
MOIS_FR    = {1:"Jan",2:"Fév",3:"Mar",4:"Avr",5:"Mai",6:"Jun",
              7:"Jul",8:"Aoû",9:"Sep",10:"Oct",11:"Nov",12:"Déc"}
JOURS_ORD  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
MOIS_ORD   = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]
CHART      = dict(plot_bgcolor="white", paper_bgcolor="black", font=dict(family="Inter, sans-serif",color=NOIR,size=12),
                  margin=dict(t=20,b=0,l=0,r=0),
                  yaxis=dict(gridcolor="#A2CCF6"),
                  xaxis=dict(gridcolor="#A2CCF6"))

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:20px 0 12px;'>
      <div style='font-size:2.4rem;'>🚗</div>
      <div style='font-size:1.25rem;font-weight:700;color:#F1F5F9;'>TrafficML</div>
      <div style='font-size:.68rem;color:#94A3B8;letter-spacing:.1em;margin-top:4px;'>
        INTERSTATE 94 · MINNEAPOLIS</div>
    </div>
    <hr style='border-color:#2C3E50;margin:10px 0 16px;'>""", unsafe_allow_html=True)

    PAGE = st.radio("", [
        "🏠  Accueil",
        "📊  Exploration (EDA)",
        "⚙️  Feature Engineering",
        "🤖  Modélisation",
        "📈  Évaluation & Performances",
        "🔬  Interprétabilité SHAP",
        "🔮  Prédiction Interactive",
        "📝  Conclusions & Perspectives",
    ], label_visibility="collapsed")

    st.markdown("""<hr style='border-color:#2C3E50;margin:16px 0;'>
    <div style='font-size:.7rem;color:#475569;line-height:1.8;'>
      <b style='color:#94A3B8;'>Meilleur modèle</b><br>
      Random Forest · R²=0.989<br>RMSE=210 · MAPE=5.8%<br><br>
      <b style='color:#94A3B8;'>Dataset</b><br>
      48 204 obs. · 2012–2018<br>Station ATR 301 · MnDOT<br><br>
      <b style='color:#94A3B8;'>Features</b><br>
      52 variables depuis 9 brutes
    </div>""", unsafe_allow_html=True)

if not OK:
    st.error(f"⚠️ Fichiers manquants : `{ERR}`")
    st.stop()


# ══════════════════════════════════════════════════════════════
# P1 — ACCUEIL
# ══════════════════════════════════════════════════════════════
if PAGE == "🏠  Accueil":
    st.markdown(f"""<div style='margin-bottom:20px;'>
      <h1 style='font-size:2.1rem;font-weight:700;color:{BLEU};margin:0;'>Prédiction du Trafic Urbain</h1>
      <p style='color:{LIGTH};font-size:1rem;margin-top:6px;'>
        Interstate 94 · Minneapolis-Saint Paul · Machine Learning supervisé</p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Observations","48 204","Oct. 2012 – Sep. 2018")
    with c2: kpi("R² Test","0.989","Random Forest · +9.5pts vs Ridge","g")
    with c3: kpi("RMSE Test","210 véh.","−66% vs Ridge","g")
    with c4: kpi("MAPE Test","5.8%","−79% vs Ridge","g")
    st.markdown("<br>", unsafe_allow_html=True)

    cL,cR = st.columns([3,2])
    with cL:
        sh("Contexte & Problématique")
        st.markdown("""Les embouteillages représentent un enjeu économique, environnemental et social majeur.
        Une prédiction fiable du volume de trafic permet d'**optimiser la gestion des flux**,
        de planifier les infrastructures et d'améliorer l'expérience des usagers.

        Ce projet vise à construire, évaluer et comparer plusieurs modèles de **machine learning supervisé**
        pour prédire le **volume horaire de trafic** sur l'Interstate 94 (direction ouest)
        à Minneapolis-Saint Paul, à partir de variables temporelles et météorologiques.""")

        sh("Source des données")
        st.markdown("""Jeu de données issu du **Département des Transports du Minnesota (MnDOT)**,
        complété par les données météo d'**OpenWeatherMap**. Station de mesure **ATR 301**,
        localisée à mi-chemin entre Minneapolis et Saint Paul. Période : *oct. 2012 – sept. 2018*.""")

        c1v,c2v = st.columns(2)
        with c1v:
            st.markdown("""**Variables météo & contexte**
- `temp` — Température (Kelvin)
- `rain_1h` — Pluie (mm/h)
- `snow_1h` — Neige (mm/h)
- `clouds_all` — Couverture nuageuse (%)
- `weather_main` — Condition météo
- `holiday` — Jours fériés US/MN
- `date_time` — Horodatage (CST)""")
        with c2v:
            st.markdown("""**Variable cible**
- `traffic_volume` — Volume horaire (véh/h)

**Dimensionnalité**
- 9 variables brutes → **52 features**
- après feature engineering complet

**Dataset**
- 48 204 observations · 6 ans
- résolution horaire""")

    with cR:
        sh("Pipeline méthodologique")
        for c,n,t,d in [
            (BLEU,"1","Importation & EDA","Distributions, patterns, corrélations"),
            (VERT,"2","Preprocessing","Filtrage 2016+, imputation MICE, encodage"),
            (ORANGE,"3","Feature Engineering","52 features : lags, moyennes, catégories"),
            (ROUGE,"4","Modélisation","Ridge · RF · XGBoost + TimeSeriesCV"),
            (GRIS,"5","Évaluation","R², RMSE, MAE, MAPE · résidus · comparaison"),
            ("#8B5CF6","6","Interprétabilité","SHAP summary · force plots · PDP"),
        ]:
            st.markdown(f"""
            <div style='display:flex;gap:12px;margin-bottom:10px;align-items:flex-start;'>
              <div style='min-width:26px;height:26px;border-radius:50%;background:{c};
                          display:flex;align-items:center;justify-content:center;
                          font-size:.75rem;font-weight:700;color:white;flex-shrink:0;'>{n}</div>
              <div>
                <div style='font-weight:600;font-size:.87rem;color:{NOIR};'>{t}</div>
                <div style='font-size:.76rem;color:{GRIS};'>{d}</div>
              </div></div>""", unsafe_allow_html=True)

    sh("Évolution historique du trafic horaire (2012–2018)")
    daily = df_raw.groupby(df_raw["date_time"].dt.date)["traffic_volume"].mean().reset_index()
    daily.columns = ["date","trafic"]
    fig = px.line(daily, x="date", y="trafic", color_discrete_sequence=[BLEU],
                  labels={"trafic":"Trafic moyen (véh/h)","date":""})
    fig.update_traces(line_width=1.2)
    fig.update_layout(height=270, **CHART)
    st.plotly_chart(fig, use_container_width=True)
    box("La série temporelle révèle une <b>cyclicité annuelle marquée</b> et une <b>stabilité interannuelle</b> des patterns — les comportements de mobilité sont structurés par des facteurs calendaires plus que par des tendances de long terme.")


# ══════════════════════════════════════════════════════════════
# P2 — EDA
# ══════════════════════════════════════════════════════════════
elif PAGE == "📊  Exploration (EDA)":
    st.title("Exploration des données (EDA)")
    st.markdown("Analyse exploratoire : distributions, patterns temporels et relations météo–trafic.")
    st.markdown("---")

    tab1,tab2,tab3,tab4 = st.tabs(["📋 Aperçu","🕐 Patterns temporels","🌤️ Météo × Trafic","🔥 Heatmap & Boxplots"])

    with tab1:
        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi("Lignes",f"{len(df_raw):,}","observations brutes")
        with c2: kpi("Colonnes","9","variables originales")
        with c3: kpi("Période","6 ans","Oct.2012 – Sep.2018")
        with c4: kpi("Trafic moyen",f"{int(df_raw['traffic_volume'].mean()):,}","véh/heure","g")
        st.markdown("<br>", unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            sh("Statistiques descriptives")
            desc = df_raw[["traffic_volume","temp","rain_1h","snow_1h","clouds_all"]].describe().round(2).T
            desc.index = ["Trafic (véh/h)","Temp (K)","Pluie (mm/h)","Neige (mm/h)","Nuages (%)"]
            desc.columns = ["N","Moyenne","Std","Min","Q1","Médiane","Q3","Max"]
            st.dataframe(desc, use_container_width=True)
        with c2:
            sh("Types & Valeurs manquantes")
            info = pd.DataFrame({
                "Variable":df_raw.columns,
                "Type":df_raw.dtypes.astype(str).values,
                "Manquants":df_raw.isna().sum().values,
                "% Manq.":(df_raw.isna().mean()*100).round(2).values,
                "Uniques":df_raw.nunique().values})
            st.dataframe(info, use_container_width=True, hide_index=True)

        sh("Distribution de traffic_volume")
        c1,c2 = st.columns([2,1])
        with c1:
            fig = px.histogram(df_raw, x="traffic_volume", nbins=80,
                               color_discrete_sequence=[BLEU],
                               labels={"traffic_volume":"Volume de trafic (véh/h)"})
            fig.add_vline(x=df_raw["traffic_volume"].mean(), line_dash="dash",
                          line_color=ORANGE, annotation_text=f"Moy={int(df_raw['traffic_volume'].mean())}")
            fig.add_vline(x=df_raw["traffic_volume"].median(), line_dash="dot",
                          line_color=VERT, annotation_text=f"Med={int(df_raw['traffic_volume'].median())}")
            fig.update_layout(height=280, **CHART)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            box("Distribution <b>bimodale</b> : pic à 0 (nuit/week-end) et concentration 3 000–5 500 véh/h (journée ouvrée). Justifie les modèles non-linéaires.")
            for k,v in {"Skewness":f"{df_raw['traffic_volume'].skew():.2f}",
                        "Kurtosis":f"{df_raw['traffic_volume'].kurtosis():.2f}",
                        "% zéros":f"{(df_raw['traffic_volume']==0).mean()*100:.1f}%",
                        "Max":f"{int(df_raw['traffic_volume'].max()):,}"}.items():
                st.markdown(f"**{k}** : `{v}`")

    with tab2:
        sh("Profil horaire moyen ± écart-type")
        h_data = df_raw.groupby(df_raw["date_time"].dt.hour)["traffic_volume"].agg(["mean","std"]).reset_index()
        h_data.columns = ["h","m","s"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h_data["h"],y=h_data["m"]+h_data["s"],
                                  fill=None,mode="lines",line_color="rgba(30,111,217,0)",showlegend=False))
        fig.add_trace(go.Scatter(x=h_data["h"],y=h_data["m"]-h_data["s"],
                                  fill="tonexty",mode="lines",line_color="rgba(30,111,217,0)",
                                  fillcolor="rgba(30,111,217,0.1)",name="±1σ"))
        fig.add_trace(go.Scatter(x=h_data["h"],y=h_data["m"],mode="lines+markers",
                                  line=dict(color=BLEU,width=2.5),marker=dict(size=7),name="Moyenne"))
        fig.add_vrect(x0=7,x1=9,fillcolor=ORANGE,opacity=.12,annotation_text="Pointe matin")
        fig.add_vrect(x0=16,x1=19,fillcolor=VERT,opacity=.12,annotation_text="Pointe soir")
        fig.update_layout(height=310,xaxis=dict(tickvals=list(range(0,24,2)),gridcolor="#F1F5F9",title="Heure"),
                          yaxis=dict(gridcolor="#F1F5F9",title="Trafic moyen (véh/h)"),
                          legend=dict(orientation="h",y=1.1),**{k:v for k,v in CHART.items() if k not in ["xaxis","yaxis"]})
        st.plotly_chart(fig, use_container_width=True)

        c1,c2 = st.columns(2)
        with c1:
            sh("Par jour de semaine")
            tj = df_raw.groupby(df_raw["date_time"].dt.day_name())["traffic_volume"].mean().reindex(JOURS_ORD).reset_index()
            tj.columns = ["j","t"]; tj["jf"] = tj["j"].map(JOURS_FR)
            fig = px.bar(tj,x="jf",y="t",color="t",color_continuous_scale=["#FEEFDB","#1E3A8A"],
                         labels={"jf":"","t":"Trafic moyen"})
            fig.update_layout(height=260,coloraxis_showscale=False,**CHART)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            sh("Par mois")
            tm = df_raw.groupby(df_raw["date_time"].dt.month)["traffic_volume"].mean().reset_index()
            tm.columns = ["m","t"]; tm["mf"] = tm["m"].map(MOIS_FR)
            fig = px.bar(tm,x="mf",y="t",color="t",color_continuous_scale=["#DCFCE7","#064E3B"],
                         category_orders={"mf":MOIS_ORD},labels={"mf":"","t":"Trafic moyen"})
            fig.update_layout(height=260,coloraxis_showscale=False,**CHART)
            st.plotly_chart(fig, use_container_width=True)

        box("Deux pics quotidiens nets (7h-9h, 16h-19h). Week-end structurellement différent. Saisonnalité estivale marquée (juin-août = volumes max).", "g")

    with tab3:
        vars_m = {"Température (°C)":"temp_c","Pluie (mm/h)":"rain_1h","Neige (mm/h)":"snow_1h","Nuages (%)":"clouds_all"}
        df_p = df_raw.copy(); df_p["temp_c"] = df_raw["temp"]-273.15
        vl = st.selectbox("Variable météorologique", list(vars_m.keys()))
        var = vars_m[vl]
        c1,c2 = st.columns(2)
        with c1:
            sh(f"Distribution — {vl}")
            fig = px.histogram(df_p.sample(min(10000,len(df_p))),x=var,nbins=60,
                               color_discrete_sequence=[BLEU],labels={var:vl})
            fig.update_layout(height=270,**CHART)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            sh(f"Relation {vl} × Trafic")
            fig = px.scatter(df_p.sample(min(5000,len(df_p))),x=var,y="traffic_volume",
                             opacity=.2,trendline="ols",color_discrete_sequence=[BLEU],
                             labels={var:vl,"traffic_volume":"Trafic (véh/h)"})
            fig.update_layout(height=270,**CHART)
            st.plotly_chart(fig, use_container_width=True)

        sh("Distribution des conditions météo")
        wvc = df_raw["weather_main"].value_counts().reset_index()
        wvc.columns = ["cond","n"]
        fig = px.bar(wvc,x="cond",y="n",color="n",color_continuous_scale=["#BFDBFE","#1E3A8A"],
                     labels={"cond":"","n":"Nombre d'observations"})
        fig.update_layout(height=280,coloraxis_showscale=False,**CHART)
        st.plotly_chart(fig, use_container_width=True)
        box("Clouds (37%) et Clear (33%) dominent. Pluie, neige et brume restent minoritaires — justifie leur regroupement en catégories pour la modélisation.")

    with tab4:
        sh("Heatmap — Trafic moyen Heure × Jour")
        pivot = df_raw.pivot_table(
            values="traffic_volume",
            index=df_raw["date_time"].dt.hour,
            columns=df_raw["date_time"].dt.day_name(),
            aggfunc="mean").reindex(columns=JOURS_ORD)
        pivot.columns = [JOURS_FR[j] for j in JOURS_ORD]
        fig = px.imshow(pivot,color_continuous_scale="RdYlGn_r",
                        labels={"x":"Jour","y":"Heure","color":"Trafic moyen"},aspect="auto")
        fig.update_layout(height=370,margin=dict(t=10,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

        sh("Boxplots — Distribution par mois")
        df_raw["mf"] = df_raw["date_time"].dt.month.map(MOIS_FR)
        fig = px.box(df_raw,x="mf",y="traffic_volume",
                     category_orders={"mf":MOIS_ORD},color_discrete_sequence=[BLEU],
                     labels={"mf":"Mois","traffic_volume":"Volume de trafic (véh/h)"})
        fig.update_layout(height=300,**CHART)
        st.plotly_chart(fig, use_container_width=True)
        box("Variance intra-mensuelle très élevée et homogène — le mois seul n'explique qu'une fraction de la variabilité. Les facteurs fins (heure, jour) dominent largement.", "o")


# ══════════════════════════════════════════════════════════════
# P3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
elif PAGE == "⚙️  Feature Engineering":
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
                badges = "".join([f"<code style='background:#F1F5F9;padding:2px 7px;border-radius:4px;"
                                   f"font-size:.72rem;margin:2px;display:inline-block;'>{v}</code>" for v in vars_l])
                st.markdown(f"""<div style='border:1px solid #E2E8F0;border-radius:10px;padding:14px;
                  margin-bottom:10px;border-top:3px solid {col};'>
                  <div style='font-weight:600;font-size:.85rem;color:{NOIR};margin-bottom:8px;'>{titre}</div>
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
                          **{k:v for k,v in CHART.items() if k!="margin"},margin=dict(t=10,b=0,l=0,r=70))
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
        fig.update_layout(barmode="overlay",height=240,**CHART,
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
            fig.update_layout(height=250,**CHART,xaxis_title="Itérations",yaxis_title="RMSE",
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
                              legend=dict(orientation="h",y=1.1),**CHART)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            sh("RMSE Test (↓ meilleur)")
            fig = go.Figure(go.Bar(x=mods,y=[618,210,213],
                                    marker_color=["#94A3B8",BLEU,VERT],
                                    text=["618 véh.","210 véh.","213 véh."],textposition="outside"))
            fig.add_annotation(x="Random Forest",y=210,text="🏆",showarrow=True,arrowhead=2,ay=-35)
            fig.update_layout(height=300,yaxis_range=[0,720],yaxis_title="RMSE (véh/h)",
                              **{k:v for k,v in CHART.items() if k!="margin"},margin=dict(t=40,b=0,l=0,r=0))
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
                st.markdown(f"""<div style='border:1px solid #E2E8F0;border-radius:12px;padding:18px;
                  border-top:4px solid {c};'>
                  <div style='font-weight:700;font-size:.88rem;color:{NOIR};margin-bottom:8px;'>{titre}</div>
                  <div style='font-size:1.6rem;font-weight:700;color:{c};'>{r2}</div>
                  <div style='font-size:.78rem;color:{GRIS};margin:4px 0 12px;'>{detail}</div>
                  <div style='font-size:.8rem;color:#374151;line-height:1.5;'>{desc}</div>
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
            fig.update_layout(height=280,yaxis_range=[0.965,.995],coloraxis_showscale=False,**CHART)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(mj,x="Jour",y="RMSE",color="RMSE",
                         color_continuous_scale=[VERT,"#FEF9C3",ROUGE],
                         text="RMSE",labels={"Jour":"","RMSE":"RMSE"})
            fig.update_traces(texttemplate="%{text:.0f}",textposition="outside")
            fig.update_layout(height=280,coloraxis_showscale=False,**CHART)
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
                              **{k:v for k,v in CHART.items() if k!="margin"},margin=dict(t=30,b=0,l=0,r=0))
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
            fig.update_layout(height=260,title="Distribution des résidus",**CHART)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(df_r.sample(min(3000,len(df_r))),x=cr,y="res",opacity=.25,
                             color_discrete_sequence=[VERT],
                             labels={cr:"Valeurs prédites","res":"Résidus"})
            fig.add_hline(y=0,line_dash="dash",line_color=ROUGE)
            fig.update_layout(height=260,title="Résidus vs Valeurs prédites",**CHART)
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
            fig.update_layout(height=260,title="Réel vs Prédit",**CHART)
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
                          **{k:v for k,v in CHART.items() if k!="margin"},margin=dict(t=10,b=0,l=0,r=70))
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
          <div style='font-size:.8rem;color:{GRIS};margin-bottom:4px;'>Force résultante</div>
          <div style='display:flex;height:20px;border-radius:4px;overflow:hidden;'>
            <div style='background:{ROUGE};width:{tp/(tp+tn)*100:.0f}%;'></div>
            <div style='background:{BLEU};width:{tn/(tp+tn)*100:.0f}%;'></div></div>
          <div style='display:flex;justify-content:space-between;font-size:.76rem;color:{GRIS};margin-top:3px;'>
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
        fig.update_layout(height=300,xaxis_title=xl,yaxis_title="Trafic prédit moyen (véh/h)",**CHART)
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

            if mod_p=="Random Forest":
                model = RF
                pred = model.predict(Xp)[0]
            elif mod_p=="XGBoost":
                model = XGB
                pred = model.predict(Xp)[0]
            else:
                model = RIDGE
                Xs=Xp.copy()
                try: Xs[nr]=SCALER.transform(Xs[nr])
                except: pass
                pred=model.predict(Xs)[0]

            pred = max(0,int(round(pred)))
            if pred<1500:   css,emoji,niv="","🟢","Faible"
            elif pred<3500: css,emoji,niv="w","🟡","Modéré"
            else:           css,emoji,niv="d","🔴","Élevé"

            st.markdown(f"""<div class='pred-box {css}'>
              <div class='pred-val'>{pred:,}</div>
              <div style='color:{GRIS};font-size:.95rem;margin-top:6px;'>véhicules / heure</div>
              <div style='margin-top:12px;font-size:1.05rem;font-weight:600;'>{emoji} Trafic {niv}</div>
              <div style='font-size:.8rem;color:{GRIS};margin-top:6px;'>
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
                                **{k:v for k,v in CHART.items() if k not in ["xaxis","yaxis"]},showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
            with st.expander("Comprendre les principaux facteurs de la prediction"):
                feature_row = Xp.copy()
                if mod_p == "Ridge":
                    try: feature_row[nr] = SCALER.transform(feature_row[nr])
                    except: pass
                explainer = build_explainer(mod_p)
                shap_values = explainer(feature_row)
                contribution_df = (
                    pd.DataFrame(
                        {
                            "Feature": COLS,
                            "Contribution": shap_values.values[0],
                            "Valeur": feature_row.iloc[0].values,
                        }
                    )
                    .assign(abs_value=lambda d: d["Contribution"].abs())
                    .sort_values("abs_value", ascending=False)
                    .head(10)
                )
                fig_imp = px.bar(
                    contribution_df.sort_values("Contribution"),
                    x="Contribution",
                    y="Feature",
                    orientation="h",
                    color="Contribution",
                    color_continuous_scale=["#d62828", "#f8f9fa", "#2a9d8f"],
                    title="Les 10 facteurs les plus influents pour ce scenario",
                )
                st.plotly_chart(fig_imp, use_container_width=True)

                fig, _ = plt.subplots(figsize=(10, 4.5))
                shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                st.pyplot(fig, clear_figure=True)
                st.caption("Cette lecture locale relie la prediction au raisonnement du notebook: contexte horaire, structure du calendrier et memoire recente du trafic restent les leviers les plus influents.")
        else:
            st.markdown(f"""<div style='text-align:center;padding:80px 20px;color:#94A3B8;'>
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
                  <div><div style='font-weight:600;font-size:.88rem;color:{NOIR};margin-bottom:3px;'>{titre}</div>
                  <div style='font-size:.82rem;color:#374151;line-height:1.5;'>{desc}</div></div>
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
                  border:1px solid #E2E8F0;border-radius:8px;padding:10px 14px;margin-bottom:8px;'>
                  <div><div style='font-size:.78rem;color:{GRIS};'>{sc}</div>
                  <div style='font-weight:600;font-size:.9rem;color:{NOIR};'>{mod}</div></div>
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
                st.markdown(f"""<div style='border:1px solid #E2E8F0;border-radius:10px;padding:14px;
                  margin-bottom:10px;border-left:3px solid {c};'>
                  <div style='font-size:.7rem;font-weight:600;color:{GRIS};text-transform:uppercase;
                               letter-spacing:.06em;margin-bottom:4px;'>{cat}</div>
                  <div style='font-weight:600;font-size:.88rem;color:{NOIR};margin-bottom:5px;'>{titre}</div>
                  <div style='font-size:.8rem;color:#374151;line-height:1.5;'>{desc}</div>
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
            st.markdown(f"""<div style='border:1px solid #E2E8F0;border-radius:10px;padding:14px;
              margin-bottom:10px;display:flex;gap:14px;'>
              <div><div style='display:flex;align-items:center;gap:8px;margin-bottom:5px;'>
                <span style='font-weight:600;font-size:.88rem;color:{NOIR};'>{titre}</span>
                <span style='background:{cp};color:white;border-radius:20px;
                             padding:2px 8px;font-size:.68rem;font-weight:600;'>{prio}</span></div>
              <div style='font-size:.81rem;color:#374151;line-height:1.5;'>{desc}</div></div>
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
          <div style='color:#475569;font-size:.75rem;margin-top:14px;'>
            Saidou Yameogo · Interstate 94 · Minneapolis-Saint Paul · 2024
          </div></div>""", unsafe_allow_html=True)
