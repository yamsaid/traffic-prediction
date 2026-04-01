# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import shap
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ============================================
# CONFIGURATION GÉNÉRALE
# ============================================
st.set_page_config(
    page_title="Prédiction du Trafic Urbain",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2C3E50;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #3498DB;
    }
    .prediction-box {
        background-color: #EBF5FB;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        border: 2px solid #3498DB;
    }
    .good-pred  { border-color: #2ECC71; background-color: #EAFAF1; }
    .mid-pred   { border-color: #F39C12; background-color: #FEF9E7; }
    .high-pred  { border-color: #E74C3C; background-color: #FDEDEC; }
</style>
""", unsafe_allow_html=True)

# ============================================
# CHARGEMENT DES RESSOURCES
# ============================================
@st.cache_resource
def charger_modele():
    model  = joblib.load("models/xgboost_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

@st.cache_data
def charger_donnees():
    df = pd.read_csv("data/data_raw.csv")
    df.rename(
    columns={
        "rain_1h": "rain",
        "snow_1h": "snow",
        "clouds_all": "cloud",
        "weather_main": "weather",
        "traffic_volume": "traffic",
        "date_time": "datetime"
    }, inplace=True
    )
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

@st.cache_resource
def charger_explainer(model, X_sample):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values

model, scaler = charger_modele()
df            = charger_donnees()

# ============================================
# SIDEBAR — NAVIGATION
# ============================================
st.sidebar.image("assets/logo.png", use_column_width=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Accueil",
     "📊 Exploration",
     "🔮 Prédiction",
     "🗺️ Carte",
     "📈 Performance modèle"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**À propos**")
st.sidebar.info(
    "Application de prédiction du volume "
    "de trafic sur l'Interstate 94 (Minneapolis). "
    "Modèle : Random Forest | R² = 0.989"
)

# ============================================
# PAGE 1 — ACCUEIL
# ============================================
if page == "🏠 Accueil":

    st.markdown('<p class="main-header">🚗 Prédiction du Trafic Urbain</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    # KPIs globaux
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Observations", f"{len(df):,}", "Dataset complet")
    with col2:
        st.metric("R² Test", "0.989", "+9.5% vs Ridge")
    with col3:
        st.metric("RMSE Test", "210 véh.", "-66% vs Ridge")
    with col4:
        st.metric("MAPE Test", "5.8%", "-79% vs Ridge")

    st.markdown("---")

    # Description du projet
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.subheader("📋 Objectif du projet")
        st.write("""
        Cette application prédit le volume horaire de trafic
        sur l'Interstate 94 en fonction des conditions météorologiques
        et temporelles. Elle permet de :
        - **Explorer** les patterns historiques du trafic
        - **Prédire** le volume pour des conditions données
        - **Comprendre** les facteurs influençant le trafic (SHAP)
        - **Visualiser** la distribution spatiale du trafic
        """)

    with col_right:
        st.subheader("🏆 Comparaison des modèles")
        df_comp = pd.DataFrame({
            "Modèle" : ["Ridge", "Random Forest", "XGBoost"],
            "R² Test": [0.903, 0.989, 0.988],
            "RMSE"   : [618, 210, 213]
        })
        fig_comp = px.bar(
            df_comp, x="Modèle", y="R² Test",
            color="Modèle",
            color_discrete_sequence=["#AED6F1","#2ECC71","#F39C12"],
            text="R² Test"
        )
        fig_comp.update_traces(textposition="outside")
        fig_comp.update_layout(
            showlegend=False,
            yaxis_range=[0.8, 1.0],
            height=300
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # Aperçu temporel
    st.subheader("📅 Évolution historique du trafic")
    df_daily = df.groupby(
        df["datetime"].dt.date
    )["traffic"].mean().reset_index()
    df_daily.columns = ["date", "trafic_moyen"]

    fig_hist = px.line(
        df_daily, x="date", y="trafic_moyen",
        title="Trafic moyen journalier — 2016 à 2018",
        labels={"trafic_moyen": "Trafic moyen (véh/h)",
                "date": "Date"},
        color_discrete_sequence=["#3498DB"]
    )
    fig_hist.update_layout(height=350)
    st.plotly_chart(fig_hist, use_container_width=True)

# ============================================
# PAGE 2 — EXPLORATION
# ============================================
elif page == "📊 Exploration":

    st.title("📊 Exploration des données")
    st.markdown("---")

    # Filtres
    col1, col2, col3 = st.columns(3)
    with col1:
        annee = st.selectbox(
            "Année",
            sorted(df["datetime"].dt.year.unique())
        )
    with col2:
        mois = st.selectbox(
            "Mois",
            range(1, 13),
            format_func=lambda x: [
                "Jan","Fév","Mar","Avr","Mai","Jun",
                "Jul","Aoû","Sep","Oct","Nov","Déc"
            ][x-1]
        )
    with col3:
        variable = st.selectbox(
            "Variable météo",
            ["temp", "rain_1h", "snow_1h", "clouds_all"]
        )

    df_filtre = df[
        (df["datetime"].dt.year  == annee) &
        (df["datetime"].dt.month == mois)
    ]

    # Graphiques exploration
    col_a, col_b = st.columns(2)

    with col_a:
        # Trafic par heure
        trafic_heure = df_filtre.groupby(
            df_filtre["datetime"].dt.hour
        )["traffic"].mean().reset_index()
        trafic_heure.columns = ["heure", "trafic"]

        fig_heure = px.line(
            trafic_heure, x="heure", y="trafic",
            title="Profil horaire moyen",
            markers=True,
            color_discrete_sequence=["#3498DB"]
        )
        fig_heure.add_vrect(
            x0=7, x1=9,
            fillcolor="orange", opacity=0.2,
            annotation_text="Pointe matin"
        )
        fig_heure.add_vrect(
            x0=16, x1=19,
            fillcolor="green", opacity=0.2,
            annotation_text="Pointe soir"
        )
        st.plotly_chart(fig_heure, use_container_width=True)

    with col_b:
        # Trafic par jour de semaine
        jours = ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"]
        trafic_jour = df_filtre.groupby(
            df_filtre["datetime"].dt.dayofweek
        )["traffic"].mean().reset_index()
        trafic_jour.columns = ["jour", "trafic"]
        trafic_jour["jour_nom"] = trafic_jour["jour"].map(
            dict(enumerate(jours))
        )

        fig_jour = px.bar(
            trafic_jour, x="jour_nom", y="trafic",
            title="Trafic moyen par jour",
            color="trafic",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_jour, use_container_width=True)

    # Corrélation météo × trafic
    st.subheader(f"🌡️ Relation {variable} × trafic")
    fig_corr = px.scatter(
        df_filtre.sample(min(2000, len(df_filtre))),
        x=variable, y="traffic",
        opacity=0.3,
        trendline="ols",
        title=f"Relation {variable} vs trafic",
        color_discrete_sequence=["#3498DB"]
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Heatmap heure × jour
    st.subheader("🔥 Heatmap trafic — Heure × Jour")
    pivot = df_filtre.pivot_table(
        values="traffic",
        index=df_filtre["datetime"].dt.hour,
        columns=df_filtre["datetime"].dt.dayofweek,
        aggfunc="mean"
    )
    pivot.columns = jours

    fig_heat = px.imshow(
        pivot,
        title="Volume de trafic moyen — Heure × Jour",
        labels={"x": "Jour", "y": "Heure",
                "color": "Trafic moyen"},
        color_continuous_scale="RdYlGn_r",
        aspect="auto"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ============================================
# PAGE 3 — PRÉDICTION
# ============================================
elif page == "🔮 Prédiction":

    st.title("🔮 Prédiction du volume de trafic")
    st.markdown("---")

    col_inputs, col_result = st.columns([1, 1])

    with col_inputs:
        st.subheader("⚙️ Paramètres de prédiction")

        # Paramètres temporels
        st.markdown("**📅 Paramètres temporels**")
        date_pred = st.date_input(
            "Date",
            value=datetime(2018, 7, 3)
        )
        heure_pred = st.slider(
            "Heure de la journée",
            0, 23, 8,
            format="%dh"
        )

        st.markdown("**🌤️ Conditions météo**")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            temp   = st.slider("Température (°C)", -20, 40, 20)
            pluie  = st.slider("Pluie (mm)",         0, 60,  0)
        with col_m2:
            neige  = st.slider("Neige (mm)",          0, 30,  0)
            nuages = st.slider("Nuages (%)",           0,100, 40)

        weather = st.selectbox(
            "Conditions météo",
            ["Clear", "Clouds", "Rain",
             "Snow", "Mist", "Thunderstorm"]
        )

        trafic_lag_1  = st.number_input(
            "Trafic heure précédente (lag_1)",
            0, 7500, 3000
        )
        trafic_lag_24 = st.number_input(
            "Trafic même heure hier (lag_24)",
            0, 7500, 3200
        )

        predict_btn = st.button(
            "🚀 Lancer la prédiction",
            use_container_width=True
        )

    with col_result:
        st.subheader("📊 Résultats")

        if predict_btn:
            # Construction du vecteur de features
            dt = datetime.combine(date_pred,
                                   datetime.min.time()) + \
                 timedelta(hours=heure_pred)

            features = {
                "hour"            : heure_pred,
                "temp_c"          : temp,
                "rain"            : pluie,
                "snow"            : neige,
                "cloud"           : nuages,
                "hour_sin"        : np.sin(2*np.pi*heure_pred/24),
                "hour_cos"        : np.cos(2*np.pi*heure_pred/24),
                "day_sin"         : np.sin(2*np.pi*dt.weekday()/7),
                "day_cos"         : np.cos(2*np.pi*dt.weekday()/7),
                "month_sin"       : np.sin(2*np.pi*dt.month/12),
                "month_cos"       : np.cos(2*np.pi*dt.month/12),
                "traffic_lag_1"   : trafic_lag_1,
                "traffic_lag_24"  : trafic_lag_24,
                "is_rush_hour"    : 1 if heure_pred in range(7,10) or
                                        heure_pred in range(16,20)
                                    else 0,
                "is_weekend"      : 1 if dt.weekday() >= 5 else 0,
                "weekday"         : dt.weekday(),
                "weather_cat_Rain": 1 if weather=="Rain" else 0,
                "weather_cat_Clouds":1 if weather=="Clouds" else 0,
                "weather_cat_bad" : 1 if weather in
                                    ["Snow","Thunderstorm","Fog"]
                                    else 0,
            }

            X_pred = pd.DataFrame([features])

            # Prédiction
            prediction = model.predict(X_pred)[0]
            prediction = max(0, round(prediction))

            # Niveau de trafic
            if prediction < 1500:
                niveau = "Faible"
                couleur = "good-pred"
                emoji  = "🟢"
            elif prediction < 3500:
                niveau = "Modéré"
                couleur = "mid-pred"
                emoji  = "🟡"
            else:
                niveau = "Élevé"
                couleur = "high-pred"
                emoji  = "🔴"

            # Affichage résultat
            st.markdown(f"""
            <div class="prediction-box {couleur}">
                <h1>{emoji} {prediction:,} véhicules/heure</h1>
                <h3>Niveau de trafic : {niveau}</h3>
                <p>📅 {dt.strftime('%A %d %B %Y à %Hh00')}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # Jauge de trafic
            fig_gauge = go.Figure(go.Indicator(
                mode  = "gauge+number+delta",
                value = prediction,
                title = {"text": "Volume prédit (véh/h)"},
                delta = {"reference": 3200,
                         "label": "vs moyenne"},
                gauge = {
                    "axis" : {"range": [0, 7500]},
                    "bar"  : {"color": "#3498DB"},
                    "steps": [
                        {"range": [0,    1500],
                         "color": "#EAFAF1"},
                        {"range": [1500, 3500],
                         "color": "#FEF9E7"},
                        {"range": [3500, 7500],
                         "color": "#FDEDEC"}
                    ],
                    "threshold": {
                        "line" : {"color": "red", "width": 4},
                        "value": 5000
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # SHAP local
            st.subheader("🔍 Explication SHAP")
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_pred)

            fig_shap, ax = plt.subplots(figsize=(10, 4))
            shap.waterfall_plot(
                shap.Explanation(
                    values        = shap_vals[0],
                    base_values   = explainer.expected_value,
                    data          = X_pred.iloc[0],
                    feature_names = X_pred.columns.tolist()
                ),
                show=False
            )
            st.pyplot(fig_shap)

            # Profil horaire simulé
            st.subheader("📈 Prédictions sur 24h")
            predictions_24h = []
            for h in range(24):
                f = features.copy()
                f["hour"]     = h
                f["hour_sin"] = np.sin(2*np.pi*h/24)
                f["hour_cos"] = np.cos(2*np.pi*h/24)
                f["is_rush_hour"] = 1 if h in range(7,10) or \
                                         h in range(16,20) else 0
                pred = max(0, model.predict(
                    pd.DataFrame([f]))[0])
                predictions_24h.append(
                    {"heure": h, "prediction": pred}
                )

            df_24h = pd.DataFrame(predictions_24h)
            fig_24h = px.line(
                df_24h, x="heure", y="prediction",
                title="Profil de trafic prédit sur 24h",
                markers=True,
                color_discrete_sequence=["#E74C3C"]
            )
            fig_24h.add_vline(
                x=heure_pred,
                line_dash="dash",
                line_color="blue",
                annotation_text=f"Heure sélectionnée ({heure_pred}h)"
            )
            fig_24h.add_vrect(
                x0=7, x1=9,
                fillcolor="orange", opacity=0.15
            )
            fig_24h.add_vrect(
                x0=16, x1=19,
                fillcolor="green", opacity=0.15
            )
            st.plotly_chart(fig_24h, use_container_width=True)

        else:
            st.info(
                "👈 Configurez les paramètres et cliquez sur "
                "'Lancer la prédiction'"
            )

# ============================================
# PAGE 4 — CARTE
# ============================================
elif page == "🗺️ Carte":

    st.title("🗺️ Visualisation géographique")
    st.markdown("---")
    st.info(
        "Localisation : Interstate 94, Minneapolis, Minnesota, USA"
    )

    # Carte de base avec Plotly
    fig_map = go.Figure(go.Scattermapbox(
        lat  = [44.9778],
        lon  = [-93.2650],
        mode = "markers+text",
        marker = go.scattermapbox.Marker(
            size  = 20,
            color = "#E74C3C"
        ),
        text      = ["Interstate 94 — Station de mesure"],
        textposition = "top right"
    ))

    fig_map.update_layout(
        mapbox = dict(
            style  = "open-street-map",
            center = dict(lat=44.9778, lon=-93.2650),
            zoom   = 11
        ),
        height = 500,
        margin = {"r":0,"t":0,"l":0,"b":0}
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Distribution horaire par zone
    st.subheader("📊 Distribution du trafic par période")
    col1, col2 = st.columns(2)

    with col1:
        trafic_saison = df.groupby(
            df["datetime"].dt.month
        )["traffic"].mean().reset_index()
        trafic_saison.columns = ["mois", "trafic"]
        mois_labels = ["Jan","Fév","Mar","Avr","Mai","Jun",
                       "Jul","Aoû","Sep","Oct","Nov","Déc"]
        trafic_saison["mois_nom"] = trafic_saison["mois"].map(
            dict(enumerate(mois_labels, 1))
        )
        fig_s = px.bar(
            trafic_saison,
            x="mois_nom", y="trafic",
            title="Trafic moyen par mois",
            color="trafic",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_s, use_container_width=True)

    with col2:
        fig_box = px.box(
            df.assign(
                heure=df["datetime"].dt.hour
            ).sample(5000),
            x="heure",
            y="traffic",
            title="Distribution du trafic par heure",
            color_discrete_sequence=["#3498DB"]
        )
        st.plotly_chart(fig_box, use_container_width=True)

# ============================================
# PAGE 5 — PERFORMANCE MODÈLE
# ============================================
elif page == "📈 Performance modèle":

    st.title("📈 Performance des modèles")
    st.markdown("---")

    # Tableau comparatif
    st.subheader("🏆 Comparaison des modèles")
    df_perf = pd.DataFrame({
        "Modèle"    : ["Ridge", "Random Forest", "XGBoost"],
        "R² Train"  : [0.823, 0.997, 0.996],
        "R² Val"    : [0.891, 0.982, 0.981],
        "R² Test"   : [0.903, 0.989, 0.988],
        "RMSE Test" : [618,   210,   213],
        "MAE Test"  : [434,   135,   138],
        "MAPE Test" : ["28.0%", "5.8%", "5.9%"]
    })
    st.dataframe(
        df_perf.style.highlight_max(
            subset=["R² Test"],
            color="#EAFAF1"
        ).highlight_min(
            subset=["RMSE Test", "MAE Test"],
            color="#EAFAF1"
        ),
        use_container_width=True
    )

    # Graphiques performance
    col1, col2 = st.columns(2)
    with col1:
        fig_r2 = px.bar(
            df_perf, x="Modèle", y="R² Test",
            title="R² Test par modèle",
            color="Modèle",
            color_discrete_sequence=[
                "#AED6F1","#2ECC71","#F39C12"
            ],
            text="R² Test"
        )
        fig_r2.update_traces(textposition="outside")
        fig_r2.update_layout(yaxis_range=[0.85, 1.0])
        st.plotly_chart(fig_r2, use_container_width=True)

    with col2:
        fig_rmse = px.bar(
            df_perf, x="Modèle", y="RMSE Test",
            title="RMSE Test par modèle (↓ meilleur)",
            color="Modèle",
            color_discrete_sequence=[
                "#AED6F1","#2ECC71","#F39C12"
            ],
            text="RMSE Test"
        )
        fig_rmse.update_traces(textposition="outside")
        st.plotly_chart(fig_rmse, use_container_width=True)
