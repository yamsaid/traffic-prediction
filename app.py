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

st.sidebar.image("assets/logo.png", use_container_width=True)
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Aller a",
    ["Vue d'ensemble", "Exploration", "Prediction", "Performance"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption("Projet notebook -> pipeline ML -> application Streamlit")
st.sidebar.info(
    "Modele deploye : XGBoost\n\n"
    f"R2 test: {metric_value(metrics_df, 'XGBoost', 'test', 'R2'):.3f}\n\n"
    f"RMSE test: {metric_value(metrics_df, 'XGBoost', 'test', 'RMSE'):.0f}"
)

if page == "Vue d'ensemble":
    st.markdown(
        """
        <div class="hero">
            <h1>Prediction du trafic urbain</h1>
            <p>Application construite a partir du notebook d'analyse du trafic de l'Interstate 94 a Minneapolis, avec exploration, prediction et lecture des performances du modele final.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observations", f"{len(raw_df):,}".replace(",", " "))
    col2.metric("Periode", f"{raw_df['datetime'].dt.year.min()}-{raw_df['datetime'].dt.year.max()}")
    col3.metric("R2 test XGBoost", f"{metric_value(metrics_df, 'XGBoost', 'test', 'R2'):.3f}")
    col4.metric("RMSE test XGBoost", f"{metric_value(metrics_df, 'XGBoost', 'test', 'RMSE'):.0f} veh/h")

    left, right = st.columns([1.5, 1])
    with left:
        st.subheader("Lecture rapide du projet")
        st.write(
            "Le notebook suit une progression complete: exploration des donnees, feature engineering temporel et meteorologique, construction des lags et moyennes mobiles, puis comparaison Ridge / Random Forest / XGBoost."
        )
        st.write(
            "Le modele final sauvegarde dans l'application est XGBoost. L'app a ete alignee sur les vraies features du pipeline afin que la prediction utilise bien les colonnes attendues par le modele."
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
            <h4>Ce que montre le notebook</h4>
            <p>Les effets temporels dominent: heures de pointe, jours de semaine et saisonnalite structurent l'essentiel du signal.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    b.markdown(
        """
        <div class="insight-card">
            <h4>Ce qui compte pour le modele</h4>
            <p>Les lags trafic, les transformations cycliques et les variables meteorologiques enrichies portent la prediction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c.markdown(
        """
        <div class="insight-card">
            <h4>Ce qui a ete ameliore</h4>
            <p>Encodage nettoye, navigation clarifiee, prediction alignee sur 52 features, resume des performances et saisie plus guidee.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "Exploration":
    st.title("Exploration des donnees")
    col1, col2, col3 = st.columns(3)
    with col1:
        year = st.selectbox("Annee", sorted(raw_df["datetime"].dt.year.unique()), index=3)
    with col2:
        month = st.selectbox("Mois", list(MONTH_LABELS.keys()), format_func=lambda m: MONTH_LABELS[m], index=6)
    with col3:
        variable = st.selectbox("Variable meteo", ["temp_c", "rain", "snow", "cloud"])

    filtered = raw_df[(raw_df["datetime"].dt.year == year) & (raw_df["datetime"].dt.month == month)]
    if filtered.empty:
        st.warning("Aucune donnee disponible pour cette selection.")
    elif "traffic" not in filtered.columns:
        st.error("La colonne 'traffic' est introuvable dans les donnees chargees. Rechargez l'application pour vider le cache Streamlit.")
    else:
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

        sample_size = min(2000, len(filtered))
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

        st.caption(
            "Lecture projet: le notebook confirme une structure tres forte autour des heures de pointe et des jours ouvres, tandis que la meteo agit surtout comme facteur modulant."
        )

elif page == "Prediction":
    st.title("Prediction du volume de trafic")

    default_scenario = st.selectbox("Scenario de depart", list(SCENARIOS.keys()))
    scenario = SCENARIOS[default_scenario]

    col_inputs, col_output = st.columns([1, 1.2])
    with col_inputs:
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

    with col_output:
        level, css_class = classify_prediction(prediction)
        st.markdown(
            f"""
            <div class="prediction-box {css_class}">
                <h2 style="margin:0;">{prediction:,.0f} vehicules / heure</h2>
                <p style="margin:0.4rem 0 0 0;"><strong>Niveau estime:</strong> {level}</p>
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
            "<p class='small-note'>Les variables de retard et moyennes mobiles non saisies manuellement sont completees a partir des profils historiques du jeu de donnees traite pour rester compatibles avec les 52 features du modele.</p>",
            unsafe_allow_html=True,
        )

    profile_24h = make_24h_profile(target_dt, user_inputs, feature_columns, reference_tables, model)
    fig_profile = px.line(
        profile_24h,
        x="heure",
        y="prediction",
        markers=True,
        color_discrete_sequence=["#d62828"],
        title="Simulation sur 24 heures pour la meme journee",
    )
    fig_profile.add_vline(x=hour, line_dash="dash", line_color="#1f4e79")
    fig_profile.add_vrect(x0=7, x1=9, fillcolor="#f4a261", opacity=0.15, line_width=0)
    fig_profile.add_vrect(x0=16, x1=19, fillcolor="#2a9d8f", opacity=0.14, line_width=0)
    st.plotly_chart(fig_profile, use_container_width=True)

    with st.expander("Voir les principales contributions SHAP"):
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
            title="Top 10 des contributions locales",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        fig, _ = plt.subplots(figsize=(10, 4.5))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig, clear_figure=True)

elif page == "Performance":
    st.title("Performance et validation du modele")

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

    col1, col2 = st.columns(2)
    with col1:
        fig_r2 = px.bar(
            comparison,
            x="Modele",
            y="R2 test",
            color="Modele",
            text="R2 test",
            color_discrete_sequence=["#9ec5fe", "#52b788", "#f4a261"],
            title="Comparaison du R2 test",
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
            title="Comparaison du RMSE test",
        )
        fig_rmse.update_traces(textposition="outside")
        fig_rmse.update_layout(showlegend=False)
        st.plotly_chart(fig_rmse, use_container_width=True)

    pred_chart = predictions_df.copy()
    pred_chart["Erreur absolue XGBoost"] = (pred_chart["traffic"] - pred_chart["pred_xgb"]).abs()
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=pred_chart["datetime"], y=pred_chart["traffic"], mode="lines", name="Reel", line=dict(color="#102542")))
    fig_pred.add_trace(go.Scatter(x=pred_chart["datetime"], y=pred_chart["pred_xgb"], mode="lines", name="Predit XGBoost", line=dict(color="#d62828")))
    fig_pred.update_layout(title="Extrait des predictions de test", height=360)
    st.plotly_chart(fig_pred, use_container_width=True)

    with st.expander("Hyperparametres sauvegardes"):
        st.json(hyperparameters_df.to_dict())
