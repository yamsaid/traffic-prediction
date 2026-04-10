## README.md - TrafficML : Prédiction intelligente du trafic urbain


# 📡 TrafficML - Prédiction du trafic urbain

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📌 Présentation

**TrafficML** est une application de prédiction du volume de trafic horaire sur l'Interstate 94, l'axe autoroutier reliant Minneapolis à Saint Paul dans le Minnesota (États-Unis). Développée dans le cadre du projet fil rouge de la formation **Africa TechUp Tour 2025 (Option Data Scientist)**, l'application utilise un modèle XGBoost optimisé pour fournir des prédictions en temps réel.

### 🎯 Objectifs

- Prédire le volume de trafic horaire à partir de variables temporelles et météorologiques
- Comparer plusieurs approches de modélisation (Ridge, Random Forest, XGBoost)
- Interpréter les prédictions via SHAP
- Fournir une interface interactive intuitive

### 📊 Performances du modèle retenu (XGBoost)

| Métrique | Valeur |
|----------|--------|
| **R²** | 0,988 |
| **RMSE** | 213 véhicules/heure |
| **MAE** | 138 véhicules/heure |
| **MAPE** | 5,95 % |

---

## 📁 Structure du projet

```
traffic-prediction/
│
├── app.py                    # Application principale Streamlit
├── requirements.txt          # Dépendances Python
├── runtime.txt               # Version Python (3.11)
├── models/                   # Modèles sauvegardés
│   ├── xgboost_model.pkl
│   ├── ridge_model.pkl
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   └── metriques.json
│
├── data/                     # Données
│   ├── data_raw.csv
│   └── data_processed.csv
│
├── assets/                   # Ressources statiques
│   ├── style.css
│   └── images/

```

---

## 🚀 Installation et lancement local

### Prérequis

- Python 3.11
- pip

### Étapes d'installation

1. **Cloner le dépôt**

```bash
git clone https://github.com/yamsaid/traffic-prediction.git
cd traffic-prediction
```

2. **Créer un environnement virtuel**

```bash
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

4. **Lancer l'application**

```bash
streamlit run app.py
```

L'application sera accessible à l'adresse : `http://localhost:8501`

---

## ☁️ Déploiement sur Streamlit Cloud

1. **Pousser le projet sur GitHub**

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Connecter Streamlit Cloud**
   - Aller sur [share.streamlit.io](https://share.streamlit.io)
   - Cliquer sur "New app"
   - Sélectionner le dépôt GitHub
   - Choisir la branche `main`
   - Définir `app.py` comme fichier principal
   - Cliquer sur "Deploy"

3. **Configuration**
   - Python version : 3.11 (via `runtime.txt`)
   - Secrets : non requis pour cette application

---

## 📦 Dépendances principales

| Bibliothèque | Version | Rôle |
|--------------|---------|------|
| streamlit | 1.35.0 | Interface utilisateur |
| pandas | 2.1.4 | Manipulation des données |
| numpy | 1.26.2 | Calculs numériques |
| matplotlib | 3.8.2 | Visualisations |
| seaborn | 0.13.0 | Graphiques statistiques |
| plotly | 5.18.0 | Graphiques interactifs |
| scikit-learn | 1.3.2 | Modélisation |
| xgboost | 2.0.3 | Modèle XGBoost |
| shap | 0.44.0 | Interprétabilité |
| joblib | 1.3.2 | Sauvegarde des modèles |

---

## 🧠 Modèles disponibles

| Modèle | R² | RMSE | MAPE | Taille |
|--------|-----|------|------|--------|
| **XGBoost** | 0,988 | 213 | 5,95 % | ~5 Mo |
| Random Forest | 0,989 | 210 | 5,8 % | ~90 Mo |
| Ridge | 0,903 | 617 | 28,0 % | < 1 Mo |

**Modèle retenu** : XGBoost (meilleur compromis performance/légèreté)

---

## 📊 Fonctionnalités

- **🏠 Accueil** : Présentation du projet et métriques clés
- **📊 Exploration (EDA)** : Analyse exploratoire des données
- **⚙️ Feature Engineering** : Détail des 52 variables créées
- **🤖 Modélisation** : Comparaison des trois modèles
- **📈 Évaluation** : Performances et diagnostics
- **🔬 SHAP** : Interprétabilité globale et locale
- **🔮 Prédiction** : Simulation interactive
- **📝 Conclusions** : Limites et perspectives

---

## 📈 Résultats clés

- **98,8 %** de la variance du trafic expliquée (R²)
- **5,95 %** d'erreur relative moyenne (MAPE)
- **213 véhicules/heure** d'erreur absolue (RMSE)
- **18 fois plus léger** que Random Forest (5 Mo vs 90 Mo)

---

## 👨‍💻 Auteur

**Saïdou YAMEOGO**  
Data Scientist — Africa TechUp Tour 2025

- 📧 saidouyameogo3@gmail.com.com
- 🔗 [LinkedIn](www.linkedin.com/in/saidou-yameogo-1684b6336)
- 🐙 [GitHub](https://github.com/yamsaid)

---

## 🙏 Remerciements

- **Africa TechUp Tour 2025** pour la formation
- **Minnesota Department of Transportation (MnDOT)** pour les données de trafic
- **OpenWeatherMap** pour les données météorologiques
- **UCI Machine Learning Repository** pour le dataset (ID 492)

---

## 📝 License

Ce projet est distribué sous license MIT. Voir le fichier `LICENSE` pour plus d'informations.

---

## 🔗 Liens utiles

- [Documentation Streamlit](https://docs.streamlit.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Dataset UCI](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume)
- [L'application streamlit](https://trafficml-smartcity.streamlit.app)

---

<div align="center">
  <sub>© 2026 — Africa TechUp Tour | Projet fil rouge — Data Scientist</sub>
</div>
