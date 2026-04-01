# TrafficML — Application Streamlit

## Structure attendue

```
projet/
├── app.py
├── requirements.txt
├── models/
│   ├── random_forest_model.pkl
│   ├── ridge_model.pkl
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   ├── metriques.json
│   └── hyperparameters.json
└── data/
    ├── data_raw.csv
    ├── data_processed.csv
    └── predictions_test.csv
```

## Lancement local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Déploiement Streamlit Cloud

1. Pusher sur GitHub
2. https://share.streamlit.io → New app
3. Sélectionner repo → branch main → app.py
4. Deploy
