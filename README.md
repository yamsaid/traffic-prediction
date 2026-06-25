![header](https://capsule-render.vercel.app/api?type=rect&color=0:16213e,100:0f3460&height=180&text=📡%20TrafficML%20-%20Prédiction%20du%20trafic%20urbain&fontSize=20&fontColor=ffffff&desc=XGBoost%20|%20Streamlit%20|%20SHAP%20|%20Data%20Science&descSize=15&descAlignY=75)

<p align="center">

<img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"/>
<img src="https://img.shields.io/badge/Python-3.11-blue.svg?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-Modélisation-orange?style=for-the-badge"/>
<img src="https://img.shields.io/badge/SHAP-Interprétabilité-purple?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Domaine-Transport%20Urbain-red?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Statut-Terminé-success?style=for-the-badge"/>

</p>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4facfe,100:00f2fe&height=3" width="100%"/>

<p align="center">
  <a href="#">
    <img src="https://img.shields.io/badge/🇫🇷%20Français-2d6a4f?style=for-the-badge" alt="Version Française"/>
  </a>

  <a href="README_EN.md">
    <img src="https://img.shields.io/badge/🇬🇧%20English-1d3557?style=for-the-badge" alt="English Version"/>
  </a>
</p>

# Résumé

*Ce projet présente **TrafficML**, une solution avancée de prédiction du volume de trafic horaire pour l'Interstate 94 (Minneapolis, États-Unis). Développée dans le cadre de la formation Africa TechUp Tour 2025, cette application utilise un modèle XGBoost optimisé pour des prédictions robustes et en temps quasi réel. Elle démontre une expertise en modélisation prédictive, ingénierie des fonctionnalités, interprétabilité des modèles (SHAP) et déploiement d'applications de données via Streamlit.*

### 🚀 Principaux résultats

✔ Modèle XGBoost expliquant **98,8%** de la variance du trafic (R²)

✔ Erreur relative moyenne (MAPE) de **5,95%**, garantissant une grande fiabilité

✔ Erreur absolue moyenne (RMSE) de **213 véhicules/heure**

✔ Modèle **18 fois plus léger** que Random Forest pour une performance équivalente

✔ Application interactive Streamlit pour l'exploration, l'analyse et la prédiction du trafic

✔ Interprétabilité des modèles grâce aux techniques SHAP

**Compétences mobilisées :** Machine Learning, Modélisation Prédictive, XGBoost, Interprétabilité (XAI, SHAP), Streamlit, Ingénierie des Fonctionnalités, Déploiement d'Applications de Données, Python, Pandas, Scikit-learn, Visualisation de Données.

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4facfe,100:00f2fe&height=3" width="100%"/>

# 📌 Contexte

La gestion du trafic urbain est un enjeu majeur pour les villes modernes, confrontées à la congestion, à la pollution et à la perte de temps. La capacité à prédire précisément le volume de trafic permet d'optimiser la planification urbaine, de réguler les flux et d'améliorer la sécurité routière. L'Interstate 94, un axe vital reliant Minneapolis à Saint Paul, est particulièrement sujette à ces défis.

Ce projet s'inscrit dans une démarche d'innovation pour fournir des outils d'aide à la décision basés sur l'analyse de données massives et l'intelligence artificielle, afin de rendre les infrastructures urbaines plus intelligentes et réactives.

> 💡 **Problématique :**
> Comment développer un modèle de prédiction du trafic urbain qui soit à la fois performant, interprétable et facilement déployable pour soutenir la gestion urbaine en temps quasi réel ?

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4facfe,100:00f2fe&height=3" width="100%"/>

# 🎯 Objectifs

### Objectifs Techniques et Analytiques

*   **Développement de Modèles Prédictifs** : Concevoir et entraîner des modèles de Machine Learning capables de prédire le volume de trafic horaire en exploitant des variables temporelles et météorologiques.
*   **Évaluation Comparative des Modèles** : Réaliser une étude comparative rigoureuse de plusieurs approches de modélisation (Ridge Regression, Random Forest, XGBoost) afin d'identifier la solution la plus performante et la plus efficiente.
*   **Interprétabilité des Modèles (XAI)** : Appliquer les techniques SHAP (SHapley Additive exPlanations) pour interpréter les prédictions du modèle retenu, offrant une transparence essentielle sur les facteurs influençant le trafic.
*   **Déploiement d'Application Interactive** : Développer une interface utilisateur intuitive et interactive via Streamlit, permettant aux utilisateurs d'explorer les données, de simuler des scénarios et de visualiser les prédictions.

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4facfe,100:00f2fe&height=3" width="100%"/>

# 🗂️ Données

<table>

<tr>

<td width="35%" valign="top">

<h3 align="center">Sources</h3>

| Élément | Description |
|----------|------------|
| Source principale | UCI Machine Learning Repository (ID 492) |
| Données de trafic | Metro Interstate Traffic Volume (Interstate 94) |
| Données météorologiques | OpenWeatherMap |
| Période | Octobre 2012 à Septembre 2018 |
| Fréquence | Horaires |
| Type d'analyse | Séries temporelles, Régression |

</td>

<td width="65%" valign="top">

<h3 align="center">Variables retenues (exemples)</h3>

| Variable | Description |
|-----------|------------|
| `date_time` | Date et heure de l'observation |
| `traffic_volume` | Volume de trafic horaire (variable cible) |
| `temp` | Température en Kelvin |
| `rain_1h` | Précipitations sur la dernière heure |
| `snow_1h` | Chutes de neige sur la dernière heure |
| `clouds_all` | Couverture nuageuse en pourcentage |
| `weather_main` | Description générale du temps |
| `holiday` | Indicateur de jour férié |
| `day_of_week` | Jour de la semaine |
| `hour` | Heure de la journée |

</td>

</tr>

</table>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4facfe,100:00f2fe&height=3" width="100%"/>

# 🔬 Méthodologie

```
Collecte des données (UCI, OpenWeatherMap)
        │
        ▼
Nettoyage et Prétraitement des Données
        │
        ▼
Ingénierie des Fonctionnalités
• Variables temporelles (heure, jour, mois, année)
• Variables météorologiques agrégées
• Indicateurs de jours fériés
        │
        ▼
Analyse Exploratoire des Données (EDA)
• Visualisations des tendances de trafic
• Corrélations avec les variables météorologiques
        │
        ▼
Sélection et Entraînement des Modèles
• Ridge Regression
• Random Forest
• XGBoost (modèle retenu)
        │
        ▼
Évaluation et Optimisation des Modèles
• Métriques (R², RMSE, MAE, MAPE)
• Validation croisée
• Optimisation des hyperparamètres
        │
        ▼
Interprétabilité du Modèle (SHAP)
• SHAP global (importance des caractéristiques)
• SHAP local (explication des prédictions individuelles)
        │
        ▼
Déploiement de l'Application Streamlit
• Interface utilisateur interactive
• Module de prédiction en temps réel
```

### Étapes réalisées

#### 1. Collecte et Prétraitement des Données
Les données brutes de trafic et météorologiques ont été collectées, nettoyées et fusionnées. Des valeurs manquantes ont été gérées et les types de données ajustés.

#### 2. Ingénierie des Fonctionnalités
Création de 52 variables explicatives à partir des données brutes, incluant des caractéristiques temporelles (heure, jour de la semaine, mois, année) et des indicateurs météorologiques (température, précipitations, nuages).

#### 3. Analyse Exploratoire des Données (EDA)
Des visualisations et statistiques descriptives ont été utilisées pour comprendre les tendances du trafic, les variations saisonnières et l'influence des conditions météorologiques.

#### 4. Modélisation Prédictive
Trois modèles de régression (Ridge, Random Forest, XGBoost) ont été entraînés et évalués. XGBoost a été sélectionné pour son équilibre optimal entre performance et efficacité.

#### 5. Interprétabilité avec SHAP
Les valeurs SHAP ont été calculées pour expliquer les prédictions du modèle XGBoost, identifiant les facteurs les plus influents sur le volume de trafic.

#### 6. Déploiement Streamlit
Une application web interactive a été développée avec Streamlit, permettant aux utilisateurs d'interagir avec le modèle et de visualiser les prédictions.

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4facfe,100:00f2fe&height=3" width="100%"/>

# 🛠️ Stack technique

<p align="center">

<img src="https://img.shields.io/badge/Python-Langage-3776AB?style=flat-square&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-Application%20Web-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-Modélisation-orange?style=flat-square&logo=xgboost&logoColor=white"/>
<img src="https://img.shields.io/badge/SHAP-Interprétabilité-purple?style=flat-square&logo=jupyter&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-Analyse%20de%20Données-150458?style=flat-square&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/NumPy-Calcul%20Numérique-013243?style=flat-square&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/Matplotlib-Visualisation-CB3C2B?style=flat-square&logo=matplotlib&logoColor=white"/>
<img src="https://img.shields.io/badge/Seaborn-Visualisation%20Statistique-0077B6?style=flat-square&logo=seaborn&logoColor=white"/>
<img src="https://img.shields.io/badge/Plotly-Visualisation%20Interactive-238C23?style=flat-square&logo=plotly&logoColor=white"/>
<img src="https://img.shields.io/badge/Joblib-Sauvegarde%20Modèles-557799?style=flat-square"/>

</p>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4facfe,100:00f2fe&height=3" width="100%"/>

# 🧠 Comparaison et Sélection des Modèles

Une analyse comparative approfondie a été menée sur trois modèles de régression pour la prédiction du trafic. Le tableau ci-dessous résume leurs performances et caractéristiques :

| Modèle | R² | RMSE (véhicules/heure) | MAE (véhicules/heure) | MAPE (%) | Taille du Modèle |
|--------|-----|------------------------|-----------------------|----------|------------------|
| **XGBoost** | 0,988 | 213 | 138 | 5,95 | ~5 Mo |
| Random Forest | 0,989 | 210 | 135 | 5,80 | ~90 Mo |
| Ridge | 0,903 | 617 | 450 | 28,00 | < 1 Mo |

**Modèle Retenu** : Le modèle **XGBoost** a été choisi pour le déploiement final. Bien que le Random Forest présente des métriques légèrement supérieures, XGBoost offre un **compromis performance/légèreté** nettement plus avantageux, étant **18 fois plus léger** (5 Mo contre 90 Mo) tout en maintenant une performance quasi identique. Cette décision stratégique est cruciale pour l'optimisation des ressources et la rapidité de déploiement en production.

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4facfe,100:00f2fe&height=3" width="100%"/>

# 📊 Fonctionnalités de l'Application Streamlit

L'application TrafficML offre une suite complète de fonctionnalités pour explorer, analyser et prédire le trafic :

*   **🏠 Accueil** : Une introduction concise au projet, présentant les objectifs et les métriques de performance clés du modèle.
  
*   **📊 Exploration des Données (EDA)** : Des visualisations interactives et des statistiques descriptives pour une compréhension approfondie des données brutes et de leurs distributions.
  
*   **⚙️ Ingénierie des Fonctionnalités (Feature Engineering)** : Une section détaillée expliquant la création des 52 variables utilisées pour l'entraînement du modèle, incluant des caractéristiques temporelles et météorologiques.
  
*   **🤖 Modélisation** : Une vue comparative des performances des trois modèles évalués (Ridge, Random Forest, XGBoost), soulignant les forces et faiblesses de chacun.
  
*   **📈 Évaluation et Diagnostics** : Des graphiques et des métriques pour évaluer la robustesse du modèle, incluant l'analyse des résidus et des courbes de performance.
  
*   **🔬 Interprétabilité SHAP** : Des visualisations SHAP globales et locales pour expliquer comment chaque caractéristique contribue aux prédictions du modèle, augmentant la transparence et la confiance.
  
*   **🔮 Prédiction Interactive** : Un module permettant aux utilisateurs de simuler des conditions spécifiques (date, heure, météo) et d'obtenir des prédictions de trafic en temps réel.
  
*   **📝 Conclusions et Perspectives** : Une discussion sur les limites actuelles du modèle et les pistes d'amélioration futures pour le projet.

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4facfe,100:00f2fe&height=3" width="100%"/>

# 📈 Résultats Clés

Ce projet a permis d'atteindre des résultats significatifs :

*   **98,8 %** de la variance du trafic expliquée (R²), démontrant une modélisation extrêmement précise.
*   **5,95 %** d'erreur relative moyenne (MAPE), indiquant une grande fiabilité des prédictions pour la planification.
*   **213 véhicules/heure** d'erreur absolue moyenne (RMSE), fournissant une mesure concrète de la précision des prédictions.
*   **18 fois plus léger** que Random Forest (5 Mo vs 90 Mo), optimisant les coûts de déploiement et la rapidité d'exécution.

Ces résultats soulignent la capacité du modèle à fournir des informations précieuses pour la gestion du trafic urbain, la planification des infrastructures et la réduction des embouteillages. L'application Streamlit rend ces informations accessibles et interactives pour les décideurs et le public.

## Captures 

### Page d'accueil

![alt text](assets/accueil1.png)

![alt text](assets/accueil2.png)

### Comparaison des modèles

![alt text](assets/comparaison.png)

### Prediction - Random Forest

![alt text](assets/pred_rf.png)

### Prediction - Modèle Ridge

![alt text](assets/pred_rid.png)

### Evaluation du modèle XGBOOST

![alt text](assets/res_xgboost.png)

### Courbe d'apprentissage de XGBOOST

![alt text](assets/courbe_xgb.png)

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4facfe,100:00f2fe&height=3" width="100%"/>

# 📁 Structure du Projet

```
traffic-prediction/
│
├── app.py                    # Application principale Streamlit
├── requirements.txt          # Dépendances Python
├── runtime.txt               # Version Python (3.11)
├── models/                   # Modèles sauvegardés (XGBoost, Ridge, Random Forest, Scaler, Feature Columns, Métriques)
│   ├── xgboost_model.pkl
│   ├── ridge_model.pkl
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   └── metriques.json
│
├── data/                     # Données brutes et prétraitées
│   ├── data_raw.csv
│   └── data_processed.csv
│
├── assets/                   # Ressources statiques (CSS, images)
│   ├── style.css
│   └── images/

```

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4facfe,100:00f2fe&height=3" width="100%"/>

# 🚀 Installation et Lancement Local

### Prérequis

*   Python 3.11
*   pip (gestionnaire de paquets Python)

### Étapes d'Installation

1.  **Cloner le dépôt GitHub**

    ```bash
    git clone [TON_LIEN_GITHUB_DEPOT]
    cd traffic-prediction
    ```

2.  **Créer et Activer un Environnement Virtuel**

    ```bash
    python -m venv venv
    source venv/bin/activate  # Sur Windows : venv\Scripts\activate
    ```

3.  **Installer les Dépendances du Projet**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Lancer l'Application Streamlit**

    ```bash
    streamlit run app.py
    ```

    L'application sera accessible via votre navigateur à l'adresse : `http://localhost:8501`

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4facfe,100:00f2fe&height=3" width="100%"/>

# ☁️ Déploiement sur Streamlit Cloud

Le projet est déployé et accessible publiquement sur Streamlit Cloud. Voici les étapes pour un déploiement similaire :

1.  **Pousser le Projet sur GitHub**

    Assurez-vous que votre dépôt GitHub est à jour avec les dernières modifications du projet.

    ```bash
    git add .
    git commit -m "Initial commit"
    git push origin main
    ```

2.  **Connecter à Streamlit Cloud**
    *   Naviguez vers [share.streamlit.io](https://share.streamlit.io).
    *   Cliquez sur "New app".
    *   Sélectionnez le dépôt GitHub correspondant à ce projet.
    *   Choisissez la branche `main`.
    *   Définissez `app.py` comme fichier principal de l'application.
    *   Cliquez sur "Deploy".

3.  **Configuration Spécifique**
    *   **Version Python** : 3.11 (automatiquement détectée via `runtime.txt`).
    *   **Secrets** : Aucun secret n'est requis pour le fonctionnement de cette application.

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4facfe,100:00f2fe&height=3" width="100%"/>

# 👨‍💻 Auteur

**Saïdou YAMEOGO**  
Data Scientist | Africa TechUp Tour 2025


*   📧 saidouyameogo3@gmail.com
*   🔗 [LinkedIn](https://www.linkedin.com/in/saidou-yameogo-1684b6336)
*   🐙 [GitHub](https://github.com/yamsaid)
*   🌐 [Application Streamlit](https://trafficml-smartcity.streamlit.app)

---

# 🙏 Remerciements

Je tiens à exprimer ma gratitude aux entités suivantes pour leur soutien et la mise à disposition des ressources essentielles à la réalisation de ce projet :

*   **Africa TechUp Tour 2025** pour la formation et l'encadrement.
*   **Minnesota Department of Transportation (MnDOT)** pour les données de trafic.
*   **OpenWeatherMap** pour les données météorologiques.
*   **UCI Machine Learning Repository** pour la mise à disposition du dataset (ID 492).

---

# 📝 Licence

Ce projet est distribué sous la [licence MIT](LICENSE). Pour plus de détails, veuillez consulter le fichier `LICENSE` inclus dans ce dépôt.

---

# 🔗 Liens Utiles et Références

*   [Application TrafficML sur Streamlit Cloud](https://trafficml-smartcity.streamlit.app)
*   [Documentation Streamlit](https://docs.streamlit.io/)
*   [Documentation SHAP](https://shap.readthedocs.io/)
*   [Documentation XGBoost](https://xgboost.readthedocs.io/)
*   [Dataset UCI - Metro Interstate Traffic Volume](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume)

---

<div align="center">
  <sub>© 2026 — Africa TechUp Tour | Projet fil rouge — Data Scientist</sub>
</div>

![footer](https://capsule-render.vercel.app/api?type=waving\&color=0:16213e,100:0f3460\&height=100\&section=footer)
