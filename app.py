from datetime import date, datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st


st.set_page_config(
    page_title="Trafic urbain Minneapolis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

MONTH_LABELS = {
    1: "Jan",
    2: "Fev",
    3: "Mar",
    4: "Avr",
    5: "Mai",
    6: "Juin",
    7: "Juil",
    8: "Aout",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}
DAY_LABELS = {
    0: "Lun",
    1: "Mar",
    2: "Mer",
    3: "Jeu",
    4: "Ven",
    5: "Sam",
    6: "Dim",
}
SCENARIOS = {
    "Trajet domicile-travail": {
        "hour": 8,
        "temp_c": 12.0,
        "rain": 0.0,
        "snow": 0.0,
        "cloud": 55.0,
    },
    "Nuit calme": {
        "hour": 2,
        "temp_c": 8.0,
        "rain": 0.0,
        "snow": 0.0,
        "cloud": 15.0,
    },
    "Episode neigeux": {
        "hour": 17,
        "temp_c": -6.0,
        "rain": 0.0,
        "snow": 2.0,
        "cloud": 90.0,
    },
    "Orage en pointe": {
        "hour": 18,
        "temp_c": 24.0,
        "rain": 6.0,
        "snow": 0.0,
        "cloud": 98.0,
    },
}

st.markdown(
    """
    <style>
        .hero {
            padding: 1.25rem 1.4rem;
            border-radius: 20px;
            background: linear-gradient(135deg, #102542 0%, #1f4e79 55%, #f0a04b 100%);
            color: white;
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.6rem;
        }
        .hero p {
            margin: 0.5rem 0 0 0;
            font-size: 1rem;
            max-width: 760px;
        }
        .insight-card {
            border: 1px solid rgba(16, 37, 66, 0.12);
            border-radius: 16px;
            padding: 1rem;
            background: #f8fafc;
            min-height: 150px;
        }
        .prediction-box {
            border-radius: 18px;
            padding: 1.25rem 1.4rem;
            color: #102542;
            background: #eef4f8;
            border-left: 8px solid #1f4e79;
        }
        .prediction-low {
            background: #edf8f1;
            border-left-color: #2b9348;
        }
        .prediction-mid {
            background: #fff6e8;
            border-left-color: #f8961e;
        }
        .prediction-high {
            background: #fdeceb;
            border-left-color: #d62828;
        }
        .small-note {
            font-size: 0.92rem;
            color: #4b5563;
        }
        .method-card {
            border-radius: 16px;
            padding: 1rem 1.1rem;
            background: white;
            border: 1px solid rgba(16, 37, 66, 0.12);
            box-shadow: 0 10px 24px rgba(16, 37, 66, 0.06);
        }
        .callout {
            border-radius: 16px;
            padding: 1rem 1.1rem;
            background: #fff7e6;
            border-left: 6px solid #f0a04b;
            color: #3f3f46;
        }
        .page-banner {
            padding: 0.95rem 1.15rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(16, 37, 66, 0.06), rgba(240, 160, 75, 0.18));
            border: 1px solid rgba(16, 37, 66, 0.08);
            margin-bottom: 1rem;
        }
        .page-banner h2 {
            margin: 0 0 0.25rem 0;
            color: #102542;
        }
        .page-banner p {
            margin: 0;
            color: #475569;
        }
        .mini-card {
            border-radius: 14px;
            padding: 0.85rem 1rem;
            background: #ffffff;
            border: 1px solid rgba(16, 37, 66, 0.08);
        }
        .section-chip {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 999px;
            background: #eef4f8;
            color: #1f4e79;
            font-size: 0.82rem;
            margin-bottom: 0.55rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
    model = joblib.load("models/xgboost_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    metrics = pd.read_json("models/metriques.json")
    hyperparameters = pd.read_json("models/hyperparameters.json")
    return model, feature_columns, metrics, hyperparameters


@st.cache_data
def load_data():
    raw = pd.read_csv("data/data_raw.csv")
    raw = raw.rename(
        columns={
            "rain_1h": "rain",
            "snow_1h": "snow",
            "clouds_all": "cloud",
            "weather_main": "weather",
            "traffic_volume": "traffic",
            "date_time": "datetime",
        }
    )
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw["temp_c"] = raw["temp"] - 273.15
    raw["weekday"] = raw["datetime"].dt.weekday
    raw["month"] = raw["datetime"].dt.month
    raw["hour"] = raw["datetime"].dt.hour
    raw["day"] = raw["datetime"].dt.day

    processed = pd.read_csv("data/data_processed.csv")
    processed["datetime"] = pd.to_datetime(processed["datetime"])

    predictions = pd.read_csv("data/predictions_test.csv")
    predictions["datetime"] = pd.to_datetime(predictions["datetime"])
    return raw, processed, predictions


@st.cache_data
def build_reference_tables(processed_df):
    group_hour_weekday = (
        processed_df.groupby(["month", "weekday", "hour"], dropna=False)
        .median(numeric_only=True)
        .reset_index()
    )
    group_hour = processed_df.groupby("hour", dropna=False).median(numeric_only=True).reset_index()
    global_defaults = processed_df.median(numeric_only=True).to_dict()
    return group_hour_weekday, group_hour, global_defaults


@st.cache_resource
def build_explainer(_model):
    return shap.TreeExplainer(_model)


def metric_value(metrics_df, model_name, split, metric):
    return metrics_df.loc[split, model_name][metric]


def to_rain_category(value):
    return int(value > 0)


def to_snow_category(value):
    return int(value > 0)


def lookup_reference_row(month, weekday, hour, tables):
    group_hour_weekday, group_hour, global_defaults = tables
    exact = group_hour_weekday[
        (group_hour_weekday["month"] == month)
        & (group_hour_weekday["weekday"] == weekday)
        & (group_hour_weekday["hour"] == hour)
    ]
    if not exact.empty:
        return exact.iloc[0].to_dict()
    partial = group_hour[group_hour["hour"] == hour]
    if not partial.empty:
        return partial.iloc[0].to_dict()
    return global_defaults


def build_feature_row(target_dt, inputs, feature_columns, reference_tables):
    month = target_dt.month
    weekday = target_dt.weekday()
    hour = target_dt.hour
    day = target_dt.day
    base = lookup_reference_row(month, weekday, hour, reference_tables)

    features = {column: float(base.get(column, 0.0)) for column in feature_columns}
    features.update(
        {
            "rain": float(inputs["rain"]),
            "snow": float(inputs["snow"]),
            "cloud": float(inputs["cloud"]),
            "hour": float(hour),
            "day": float(day),
            "weekday": float(weekday),
            "month": float(month),
            "year": float(target_dt.year),
            "is_holiday": float(inputs["is_holiday"]),
            "is_rush_hour": float(hour in {7, 8, 9, 16, 17, 18, 19}),
            "is_weekend": float(weekday >= 5),
            "temp_c": float(inputs["temp_c"]),
            "rain_cat": float(to_rain_category(inputs["rain"])),
            "snow_cat": float(to_snow_category(inputs["snow"])),
            "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
            "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
            "day_sin": float(np.sin(2 * np.pi * weekday / 7)),
            "day_cos": float(np.cos(2 * np.pi * weekday / 7)),
            "month_sin": float(np.sin(2 * np.pi * month / 12)),
            "month_cos": float(np.cos(2 * np.pi * month / 12)),
            "traffic_lag_1": float(inputs["traffic_lag_1"]),
            "traffic_lag_2": float(inputs["traffic_lag_2"]),
            "traffic_lag_3": float(inputs["traffic_lag_3"]),
            "traffic_lag_24": float(inputs["traffic_lag_24"]),
        }
    )

    for column in ["rain", "snow", "temp_c", "cloud"]:
        current_value = features[column]
        for lag in [1, 2, 3, 24]:
            lag_name = f"{column}_lag_{lag}"
            if lag_name in features:
                features[lag_name] = current_value

    for window in [3, 6, 24]:
        for column in ["rain", "snow", "temp_c", "cloud"]:
            mean_name = f"{column}_mean_{window}"
            if mean_name in features:
                features[mean_name] = features[column]

    return pd.DataFrame([[features[column] for column in feature_columns]], columns=feature_columns)


def classify_prediction(value):
    if value < 1500:
        return "Faible", "prediction-low"
    if value < 3500:
        return "Modere", "prediction-mid"
    return "Eleve", "prediction-high"


def format_number(value):
    return f"{value:,.0f}".replace(",", " ")


def render_page_banner(title, subtitle):
    st.markdown(
        f"""
        <div class="page-banner">
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_notebook_insights(raw_df, processed_df):
    rush_mask = raw_df["hour"].isin([7, 8, 9, 16, 17, 18, 19])
    weekend_mask = raw_df["weekday"] >= 5
    rain_mask = raw_df["rain"] > 0
    snow_mask = raw_df["snow"] > 0
    return {
        "rush_hour_delta": raw_df.loc[rush_mask, "traffic"].mean() - raw_df.loc[~rush_mask, "traffic"].mean(),
        "weekend_delta": raw_df.loc[~weekend_mask, "traffic"].mean() - raw_df.loc[weekend_mask, "traffic"].mean(),
        "rain_delta": raw_df.loc[rain_mask, "traffic"].mean() - raw_df.loc[~rain_mask, "traffic"].mean(),
        "snow_share": 100 * snow_mask.mean(),
        "top_feature_block": processed_df[["traffic_lag_1", "traffic_lag_24", "hour", "weekday"]].corr(numeric_only=True)["traffic_lag_1"].drop("traffic_lag_1").abs().idxmax(),
    }


def make_input_warnings(inputs, target_dt):
    warnings_list = []
    if inputs["snow"] > 0 and inputs["temp_c"] > 8:
        warnings_list.append("Presence de neige avec temperature elevee: scenario peu frequent dans les donnees.")
    if inputs["rain"] > 12:
        warnings_list.append("Pluie tres forte: la prediction est plus incertaine car ce type d'episode est plus rare.")
    if abs(inputs["traffic_lag_1"] - inputs["traffic_lag_24"]) > 3500:
        warnings_list.append("Les lags saisis sont tres eloignes l'un de l'autre; verifiez que le scenario est coherent.")
    if target_dt.weekday() >= 5 and target_dt.hour in {7, 8, 9}:
        warnings_list.append("Heure de pointe choisie pendant le week-end: le trafic reel est souvent plus faible qu'en semaine.")
    return warnings_list


def make_24h_profile(base_dt, inputs, feature_columns, tables, model):
    rows = []
    for hour in range(24):
        target_dt = datetime.combine(base_dt.date(), datetime.min.time()).replace(
            year=base_dt.year,
            month=base_dt.month,
            day=base_dt.day,
            hour=hour,
        )
        row = build_feature_row(target_dt, inputs, feature_columns, tables)
        pred = max(0, float(model.predict(row)[0]))
        rows.append({"heure": hour, "prediction": round(pred)})
    return pd.DataFrame(rows)


model, feature_columns, metrics_df, hyperparameters_df = load_assets()
raw_df, processed_df, predictions_df = load_data()
reference_tables = build_reference_tables(processed_df)
notebook_insights = get_notebook_insights(raw_df, processed_df)

st.sidebar.image("assets/logo.png", use_container_width=True)
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Aller a",
    ["Vue d'ensemble", "Exploration", "Prediction", "Performance"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption("Du notebook d'analyse a une application de demonstration data")
st.sidebar.info(
    "Modele presente dans l'app : XGBoost\n\n"
    f"R2 test : {metric_value(metrics_df, 'XGBoost', 'test', 'R2'):.3f}\n\n"
    f"RMSE test : {metric_value(metrics_df, 'XGBoost', 'test', 'RMSE'):.0f} veh/h"
)
st.sidebar.markdown("### Guide rapide")
st.sidebar.caption("1. Comprendre le projet et ses resultats")
st.sidebar.caption("2. Explorer les dynamiques du trafic")
st.sidebar.caption("3. Tester un scenario de prediction")
st.sidebar.caption("4. Examiner les performances du modele")

if page == "Vue d'ensemble":
    st.markdown(
        """
        <div class="hero">
            <h1>Prediction du trafic urbain a Minneapolis</h1>
            <p>Cette application transforme un notebook de data science en demonstration interactive : comprendre les donnees, expliquer les choix de modelisation et simuler des scenarios de trafic de maniere lisible.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observations", f"{len(raw_df):,}".replace(",", " "))
    col2.metric("Periode", f"{raw_df['datetime'].dt.year.min()}-{raw_df['datetime'].dt.year.max()}")
    col3.metric("R2 test XGBoost", f"{metric_value(metrics_df, 'XGBoost', 'test', 'R2'):.3f}")
    col4.metric("RMSE test XGBoost", f"{metric_value(metrics_df, 'XGBoost', 'test', 'RMSE'):.0f} veh/h")

    overview_tab, method_tab, insight_tab = st.tabs(["Projet", "Methodologie", "Insights"])

    with overview_tab:
        left, right = st.columns([1.5, 1])
        with left:
            st.subheader("En une minute")
            st.write(
                "Le projet suit une demarche data complete: exploration des donnees, creation de variables temporelles et meteorologiques, construction de lags et de moyennes mobiles, puis comparaison de plusieurs modeles de regression."
            )
            st.write(
                "L'objectif de l'application n'est pas seulement de predire un volume de trafic, mais aussi de rendre visibles les enseignements du notebook: ce qui structure le trafic, ce que le modele apprend, et dans quelles limites il faut lire la prediction."
            )
            st.markdown(
                """
                <div class="callout">
                    <strong>Pourquoi cette application est utile</strong><br>
                    Elle sert a la fois de support de restitution pour un projet data, d'exemple de deploiement Streamlit, et de demonstrateur metier pour la prevision des congestions horaires.
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            comparison = pd.DataFrame(
                {
                    "Modele": ["Ridge", "Random Forest", "XGBoost"],
                    "R2 test": [
                        metric_value(metrics_df, "Ridge", "test", "R2"),
                        metric_value(metrics_df, "Random_Forest", "test", "R2"),
                        metric_value(metrics_df, "XGBoost", "test", "R2"),
                    ],
                    "RMSE test": [
                        metric_value(metrics_df, "Ridge", "test", "RMSE"),
                        metric_value(metrics_df, "Random_Forest", "test", "RMSE"),
                        metric_value(metrics_df, "XGBoost", "test", "RMSE"),
                    ],
                }
            )
            fig = px.bar(
                comparison,
                x="Modele",
                y="R2 test",
                color="Modele",
                color_discrete_sequence=["#9ec5fe", "#52b788", "#f4a261"],
                text="R2 test",
            )
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig.update_layout(height=320, showlegend=False, yaxis_range=[0.85, 1.01])
            st.plotly_chart(fig, use_container_width=True)

    with method_tab:
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown("<div class='method-card'><strong>1. Comprendre les donnees</strong><br>Identifier les rythmes du trafic, verifier la qualite du jeu de donnees et poser les premieres hypotheses metier.</div>", unsafe_allow_html=True)
        m2.markdown("<div class='method-card'><strong>2. Construire le signal</strong><br>Ajouter les variables calendaires, les cycles, les lags et les moyennes mobiles qui donnent de la memoire au modele.</div>", unsafe_allow_html=True)
        m3.markdown("<div class='method-card'><strong>3. Choisir le bon modele</strong><br>Comparer les approches lineaires et non lineaires sur des metriques train, validation et test.</div>", unsafe_allow_html=True)
        m4.markdown("<div class='method-card'><strong>4. Expliquer les resultats</strong><br>Relier la prediction aux facteurs qui la portent, avec des graphiques de performance et une lecture locale SHAP.</div>", unsafe_allow_html=True)
        st.caption("Cette section condense le notebook en recit de projet: on y voit la methode, la logique de modelisation et la valeur metier sans devoir relire toutes les cellules.")

    with insight_tab:
        i1, i2, i3 = st.columns(3)
        i1.metric("Hausse moyenne en pointe", f"{notebook_insights['rush_hour_delta']:+.0f} veh/h")
        i2.metric("Ecart semaine / week-end", f"{notebook_insights['weekend_delta']:+.0f} veh/h")
        i3.metric("Heures avec neige", f"{notebook_insights['snow_share']:.1f}%")
        st.write(
            "Le message central du notebook est clair: le trafic est d'abord organise par le temps, puis affine par le contexte meteorologique. Les variables de memoire recente jouent un role majeur pour retrouver cette dynamique."
        )

    daily = raw_df.groupby(raw_df["datetime"].dt.date)["traffic"].mean().reset_index()
    daily.columns = ["date", "traffic_mean"]
    fig_daily = px.line(
        daily,
        x="date",
        y="traffic_mean",
        color_discrete_sequence=["#1f4e79"],
        title="Evolution du trafic moyen journalier",
    )
    fig_daily.update_layout(height=360)
    st.plotly_chart(fig_daily, use_container_width=True)

    a, b, c = st.columns(3)
    a.markdown(
        """
        <div class="insight-card">
            <h4>Ce que racontent les donnees</h4>
            <p>Les heures de pointe, le rythme semaine / week-end et la saisonnalite expliquent la plus grande partie du comportement du trafic.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    b.markdown(
        """
        <div class="insight-card">
            <h4>Ce que retient le modele</h4>
            <p>Les lags trafic, les transformations cycliques et les variables meteorologiques enrichies donnent au modele le contexte dont il a besoin pour predire.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c.markdown(
        """
        <div class="insight-card">
            <h4>Ce que montre l'application</h4>
            <p>Une version deployable du projet: navigation guidee, prediction interpretable, comparaisons de scenarios et restitution claire des performances.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "Exploration":
    render_page_banner(
        "Exploration des donnees",
        "Cette page sert a relire les enseignements du notebook sous forme interactive: quand le trafic monte, comment le calendrier le structure, et ou la meteo vient deformer cette logique.",
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        year = st.selectbox("Annee", sorted(raw_df["datetime"].dt.year.unique()), index=3)
    with col2:
        month = st.selectbox("Mois", list(MONTH_LABELS.keys()), format_func=lambda m: MONTH_LABELS[m], index=6)
    with col3:
        variable = st.selectbox("Variable meteo", ["temp_c", "rain", "snow", "cloud"])
    with col4:
        calendar_view = st.selectbox("Calendrier", ["Tous les jours", "Semaine", "Week-end"])

    filtered = raw_df[(raw_df["datetime"].dt.year == year) & (raw_df["datetime"].dt.month == month)].copy()
    if calendar_view == "Semaine":
        filtered = filtered[filtered["weekday"] < 5]
    elif calendar_view == "Week-end":
        filtered = filtered[filtered["weekday"] >= 5]

    if filtered.empty:
        st.warning("Aucune donnee disponible pour cette selection.")
    elif "traffic" not in filtered.columns:
        st.error("La colonne 'traffic' est introuvable dans les donnees chargees. Rechargez l'application pour vider le cache Streamlit.")
    else:
        temporal_tab, weather_tab, calendar_tab = st.tabs(["Temporel", "Meteo", "Calendrier"])

        with temporal_tab:
            left, right = st.columns(2)
            hourly = filtered.groupby("hour")["traffic"].mean().reset_index()
            day_profile = filtered.groupby("weekday")["traffic"].mean().reset_index()
            day_profile["jour"] = day_profile["weekday"].map(DAY_LABELS)

            with left:
                fig_hour = px.line(
                    hourly,
                    x="hour",
                    y="traffic",
                    markers=True,
                    color_discrete_sequence=["#1f4e79"],
                    title="Profil horaire moyen",
                )
                fig_hour.add_vrect(x0=7, x1=9, fillcolor="#f4a261", opacity=0.18, line_width=0)
                fig_hour.add_vrect(x0=16, x1=19, fillcolor="#2a9d8f", opacity=0.15, line_width=0)
                st.plotly_chart(fig_hour, use_container_width=True)
            with right:
                fig_days = px.bar(
                    day_profile,
                    x="jour",
                    y="traffic",
                    color="traffic",
                    color_continuous_scale="Blues",
                    title="Trafic moyen par jour",
                )
                st.plotly_chart(fig_days, use_container_width=True)
            st.caption("Lecture guidee: le rythme journalier est le premier signal fort. Sur les jours ouvres, deux pics se dessinent nettement aux heures de pointe.")

        with weather_tab:
            sample_size = min(2000, len(filtered))
            weather_left, weather_right = st.columns([1.15, 0.85])
            with weather_left:
                fig_scatter = px.scatter(
                    filtered.sample(sample_size, random_state=42),
                    x=variable,
                    y="traffic",
                    opacity=0.35,
                    trendline="ols",
                    color_discrete_sequence=["#f4a261"],
                    title=f"Relation {variable} vs trafic",
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            with weather_right:
                weather_summary = (
                    filtered.groupby("weather")["traffic"]
                    .agg(["mean", "count"])
                    .reset_index()
                    .sort_values("mean", ascending=False)
                    .head(8)
                )
                weather_summary.columns = ["Meteo", "Trafic moyen", "Volume"]
                st.dataframe(weather_summary, use_container_width=True, hide_index=True)
            st.caption("Lecture guidee: la meteo ne pilote pas seule le trafic, mais elle l'infléchit, surtout lors des episodes les moins frequents.")

        with calendar_tab:
            heat = filtered.pivot_table(
                values="traffic",
                index="hour",
                columns="weekday",
                aggfunc="mean",
            )
            heat = heat.rename(columns=DAY_LABELS)
            fig_heat = px.imshow(
                heat,
                aspect="auto",
                color_continuous_scale="YlOrRd",
                title="Heatmap trafic moyen par heure et jour",
                labels={"x": "Jour", "y": "Heure", "color": "Trafic moyen"},
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            holiday_view = (
                filtered.assign(type_jour=np.where(filtered["holiday"].fillna("") == "", "Jour standard", "Jour ferie"))
                .groupby("type_jour")["traffic"]
                .mean()
                .reset_index()
            )
            fig_holiday = px.bar(
                holiday_view,
                x="type_jour",
                y="traffic",
                color="type_jour",
                color_discrete_sequence=["#1f4e79", "#f0a04b"],
                title="Trafic moyen selon le type de jour",
            )
            st.plotly_chart(fig_holiday, use_container_width=True)
            st.caption("Lecture guidee: apres l'heure, le calendrier est le second grand organisateur du trafic, avec un contraste net entre semaine, week-end et jours feries.")

elif page == "Prediction":
    render_page_banner(
        "Prediction du volume de trafic",
        "La prediction ne vaut que si le scenario est lisible. Cette page aide a formuler une situation plausible, a la comparer a une reference, puis a comprendre ce qui fait monter ou baisser le trafic estime.",
    )

    default_scenario = st.selectbox("Scenario de depart", list(SCENARIOS.keys()))
    scenario = SCENARIOS[default_scenario]

    col_inputs, col_output = st.columns([1, 1.2])
    with col_inputs:
        st.markdown("<div class='section-chip'>Construire un scenario</div>", unsafe_allow_html=True)
        target_date = st.date_input("Date", value=date(2018, 7, 3))
        hour = st.slider("Heure", 0, 23, int(scenario["hour"]))

        c1, c2 = st.columns(2)
        with c1:
            temp_c = st.slider("Temperature (C)", -25.0, 40.0, float(scenario["temp_c"]), 0.5)
            rain = st.slider("Pluie sur 1h (mm)", 0.0, 20.0, float(scenario["rain"]), 0.1)
            traffic_lag_1 = st.number_input("Trafic heure precedente", 0, 8000, 2800, 50)
            traffic_lag_2 = st.number_input("Trafic il y a 2h", 0, 8000, 2600, 50)
        with c2:
            snow = st.slider("Neige sur 1h (mm)", 0.0, 10.0, float(scenario["snow"]), 0.1)
            cloud = st.slider("Nuages (%)", 0.0, 100.0, float(scenario["cloud"]), 1.0)
            traffic_lag_3 = st.number_input("Trafic il y a 3h", 0, 8000, 2400, 50)
            traffic_lag_24 = st.number_input("Trafic meme heure hier", 0, 8000, 3000, 50)

        is_holiday = st.toggle("Jour ferie", value=False)

        user_inputs = {
            "temp_c": temp_c,
            "rain": rain,
            "snow": snow,
            "cloud": cloud,
            "traffic_lag_1": traffic_lag_1,
            "traffic_lag_2": traffic_lag_2,
            "traffic_lag_3": traffic_lag_3,
            "traffic_lag_24": traffic_lag_24,
            "is_holiday": int(is_holiday),
        }
        target_dt = datetime.combine(target_date, datetime.min.time()).replace(hour=hour)
        feature_row = build_feature_row(target_dt, user_inputs, feature_columns, reference_tables)
        prediction = max(0, float(model.predict(feature_row)[0]))
        input_warnings = make_input_warnings(user_inputs, target_dt)

    with col_output:
        st.markdown("<div class='section-chip'>Lire la prediction</div>", unsafe_allow_html=True)
        level, css_class = classify_prediction(prediction)
        st.markdown(
            f"""
            <div class="prediction-box {css_class}">
                <h2 style="margin:0;">{prediction:,.0f} vehicules par heure</h2>
                <p style="margin:0.4rem 0 0 0;"><strong>Lecture rapide :</strong> trafic {level.lower()}</p>
                <p style="margin:0.4rem 0 0 0;">{target_dt.strftime('%Y-%m-%d a %Hh00')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        avg_reference = float(processed_df["traffic"].mean())
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                delta={"reference": avg_reference},
                gauge={
                    "axis": {"range": [0, 8000]},
                    "bar": {"color": "#1f4e79"},
                    "steps": [
                        {"range": [0, 1500], "color": "#edf8f1"},
                        {"range": [1500, 3500], "color": "#fff6e8"},
                        {"range": [3500, 8000], "color": "#fdeceb"},
                    ],
                },
                title={"text": "Volume predit"},
            )
        )
        gauge.update_layout(height=310, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(gauge, use_container_width=True)

        st.markdown(
            "<p class='small-note'>Pour rester compatible avec le pipeline du notebook, certaines variables de retard et de moyenne mobile sont completees automatiquement a partir des profils historiques du jeu de donnees traite.</p>",
            unsafe_allow_html=True,
        )
        if input_warnings:
            for warning_message in input_warnings:
                st.warning(warning_message)

        baseline_inputs = {
            "temp_c": float(processed_df["temp_c"].median()),
            "rain": 0.0,
            "snow": 0.0,
            "cloud": float(processed_df["cloud"].median()),
            "traffic_lag_1": float(processed_df["traffic_lag_1"].median()),
            "traffic_lag_2": float(processed_df["traffic_lag_2"].median()),
            "traffic_lag_3": float(processed_df["traffic_lag_3"].median()),
            "traffic_lag_24": float(processed_df["traffic_lag_24"].median()),
            "is_holiday": 0,
        }
        baseline_row = build_feature_row(target_dt, baseline_inputs, feature_columns, reference_tables)
        baseline_prediction = max(0, float(model.predict(baseline_row)[0]))
        m1, m2, m3 = st.columns(3)
        m1.metric("Ecart vs reference mediane", f"{prediction - baseline_prediction:+.0f} veh/h", help="Comparaison avec un scenario median sur la meme date et la meme heure.")
        m2.metric("Trafic horaire moyen", f"{avg_reference:.0f} veh/h")
        m3.metric("Lecture operationnelle", "Surveillance renforcee" if prediction >= 3500 else "Situation reguliere")

    profile_24h = make_24h_profile(target_dt, user_inputs, feature_columns, reference_tables, model)
    fig_profile = px.line(
        profile_24h,
        x="heure",
        y="prediction",
        markers=True,
        color_discrete_sequence=["#d62828"],
        title="Projection sur 24 heures pour la meme journee",
    )
    fig_profile.add_vline(x=hour, line_dash="dash", line_color="#1f4e79")
    fig_profile.add_vrect(x0=7, x1=9, fillcolor="#f4a261", opacity=0.15, line_width=0)
    fig_profile.add_vrect(x0=16, x1=19, fillcolor="#2a9d8f", opacity=0.14, line_width=0)
    st.plotly_chart(fig_profile, use_container_width=True)

    compare_col1, compare_col2 = st.columns(2)
    with compare_col1:
        st.markdown("<div class='mini-card'><strong>Comment renseigner les lags</strong><br><br>Les lags representent la memoire recente du trafic. Sans historique metier precis, utilisez des valeurs proches pour <code>lag_1</code>, <code>lag_2</code> et <code>lag_3</code>, puis servez-vous de <code>lag_24</code> comme point de comparaison avec la meme heure la veille.</div>", unsafe_allow_html=True)
    with compare_col2:
        scenario_compare_name = st.selectbox("Comparer avec un autre scenario", list(SCENARIOS.keys()), index=1)
        compare_scenario = SCENARIOS[scenario_compare_name]
        compare_inputs = {
            "temp_c": compare_scenario["temp_c"],
            "rain": compare_scenario["rain"],
            "snow": compare_scenario["snow"],
            "cloud": compare_scenario["cloud"],
            "traffic_lag_1": user_inputs["traffic_lag_1"],
            "traffic_lag_2": user_inputs["traffic_lag_2"],
            "traffic_lag_3": user_inputs["traffic_lag_3"],
            "traffic_lag_24": user_inputs["traffic_lag_24"],
            "is_holiday": user_inputs["is_holiday"],
        }
        compare_dt = target_dt.replace(hour=int(compare_scenario["hour"]))
        compare_row = build_feature_row(compare_dt, compare_inputs, feature_columns, reference_tables)
        compare_prediction = max(0, float(model.predict(compare_row)[0]))
        st.markdown("<div class='section-chip'>Mettre en perspective</div>", unsafe_allow_html=True)
        st.metric("Prediction du scenario compare", f"{compare_prediction:.0f} veh/h", f"{compare_prediction - prediction:+.0f} vs scenario courant")

    with st.expander("Comprendre les principaux facteurs de la prediction"):
        explainer = build_explainer(model)
        shap_values = explainer(feature_row)
        contribution_df = (
            pd.DataFrame(
                {
                    "Feature": feature_columns,
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

elif page == "Performance":
    render_page_banner(
        "Performance et validation du modele",
        "Cette page relie la partie experimentation du notebook a la version applicative: quel modele performe le mieux, pourquoi XGBoost reste un bon choix de demonstration, et comment lire ses erreurs.",
    )

    comparison = pd.DataFrame(
        [
            {
                "Modele": "Ridge",
                "R2 train": metric_value(metrics_df, "Ridge", "train", "R2"),
                "R2 val": metric_value(metrics_df, "Ridge", "val", "R2"),
                "R2 test": metric_value(metrics_df, "Ridge", "test", "R2"),
                "RMSE test": metric_value(metrics_df, "Ridge", "test", "RMSE"),
                "MAE test": metric_value(metrics_df, "Ridge", "test", "MAE"),
                "MAPE test": metric_value(metrics_df, "Ridge", "test", "MAPE"),
            },
            {
                "Modele": "Random Forest",
                "R2 train": metric_value(metrics_df, "Random_Forest", "train", "R2"),
                "R2 val": metric_value(metrics_df, "Random_Forest", "val", "R2"),
                "R2 test": metric_value(metrics_df, "Random_Forest", "test", "R2"),
                "RMSE test": metric_value(metrics_df, "Random_Forest", "test", "RMSE"),
                "MAE test": metric_value(metrics_df, "Random_Forest", "test", "MAE"),
                "MAPE test": metric_value(metrics_df, "Random_Forest", "test", "MAPE"),
            },
            {
                "Modele": "XGBoost",
                "R2 train": metric_value(metrics_df, "XGBoost", "train", "R2"),
                "R2 val": metric_value(metrics_df, "XGBoost", "val", "R2"),
                "R2 test": metric_value(metrics_df, "XGBoost", "test", "R2"),
                "RMSE test": metric_value(metrics_df, "XGBoost", "test", "RMSE"),
                "MAE test": metric_value(metrics_df, "XGBoost", "test", "MAE"),
                "MAPE test": metric_value(metrics_df, "XGBoost", "test", "MAPE"),
            },
        ]
    )
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    perf_metrics = st.columns(3)
    perf_metrics[0].metric("Meilleure performance brute", "Random Forest", "R2 test 0.989")
    perf_metrics[1].metric("Modele montre dans l'app", "XGBoost", "RMSE 213")
    perf_metrics[2].metric("Ecart RF vs XGB", "2 veh/h", "ecart tres faible")

    col1, col2 = st.columns(2)
    with col1:
        fig_r2 = px.bar(
            comparison,
            x="Modele",
            y="R2 test",
            color="Modele",
            text="R2 test",
            color_discrete_sequence=["#9ec5fe", "#52b788", "#f4a261"],
            title="Comparaison des scores R2 sur le jeu de test",
        )
        fig_r2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_r2.update_layout(yaxis_range=[0.85, 1.01], showlegend=False)
        st.plotly_chart(fig_r2, use_container_width=True)
    with col2:
        fig_rmse = px.bar(
            comparison,
            x="Modele",
            y="RMSE test",
            color="Modele",
            text="RMSE test",
            color_discrete_sequence=["#9ec5fe", "#52b788", "#f4a261"],
            title="Comparaison du RMSE sur le jeu de test",
        )
        fig_rmse.update_traces(textposition="outside")
        fig_rmse.update_layout(showlegend=False)
        st.plotly_chart(fig_rmse, use_container_width=True)

    pred_chart = predictions_df.copy()
    pred_chart["Erreur absolue XGBoost"] = (pred_chart["traffic"] - pred_chart["pred_xgb"]).abs()
    pred_chart["Residual_xgb"] = pred_chart["traffic"] - pred_chart["pred_xgb"]
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=pred_chart["datetime"], y=pred_chart["traffic"], mode="lines", name="Reel", line=dict(color="#102542")))
    fig_pred.add_trace(go.Scatter(x=pred_chart["datetime"], y=pred_chart["pred_xgb"], mode="lines", name="Predit XGBoost", line=dict(color="#d62828")))
    fig_pred.update_layout(title="Reel vs predit sur un extrait du jeu de test", height=360)
    st.plotly_chart(fig_pred, use_container_width=True)

    perf_left, perf_right = st.columns(2)
    with perf_left:
        fig_residual = px.histogram(
            pred_chart,
            x="Residual_xgb",
            nbins=40,
            color_discrete_sequence=["#1f4e79"],
            title="Distribution des residus du modele XGBoost",
        )
        st.plotly_chart(fig_residual, use_container_width=True)
    with perf_right:
        fig_error = px.scatter(
            pred_chart,
            x="traffic",
            y="Erreur absolue XGBoost",
            opacity=0.5,
            color_discrete_sequence=["#f0a04b"],
            title="Erreur absolue selon le trafic observe",
        )
        st.plotly_chart(fig_error, use_container_width=True)

    strengths, limits = st.columns(2)
    with strengths:
        st.subheader("Ce que le projet fait bien")
        st.write("Le modele final restitue tres bien les rythmes temporels du trafic et reste solide sur le jeu de test.")
        st.write("L'application raconte maintenant le projet de bout en bout: donnees, methode, prediction, comparaison et interpretation.")
    with limits:
        st.subheader("Ce qu'il faut garder en tete")
        st.write("Le jeu de donnees provient d'une station unique: on decrit tres bien ce contexte local, mais pas tout un reseau routier.")
        st.write("Les variables de retard doivent rester plausibles; dans une application metier reelle, elles seraient idealement alimentees automatiquement.")

    with st.expander("Voir les hyperparametres sauvegardes"):
        st.json(hyperparameters_df.to_dict())
