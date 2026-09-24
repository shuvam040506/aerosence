# AeroSense: Air Quality + PM2.5 Predictor (Flask)

A single-page Flask app that lets you search any location, view its current weather (Open-Meteo), and instantly predicts PM2.5 using a lightweight hybrid ML model (RandomForest + ExtraTrees averaged with log1p target transform). Dark, sleek UI with modern CSS.

- No API keys required (Open-Meteo geocoding + forecast APIs)
- Hybrid model trained on synthetic data by default; can be retrained on real data
- Works even without a trained model via a simple heuristic fallback

## Features

- Location search -> geocode to lat/lon -> fetch current weather
- Prediction using features: temperature, humidity, apparent temp, precipitation, wind speed, surface pressure, UV index, dew point, PM10
- Dark, modern UI with animated skeletons, badges, and scales

## Project structure

- `app.py` — Flask server and APIs
- `templates/index.html` — SPA page
- `static/style.css` — Modern dark theme CSS
- `static/app.js` — Client logic (fetch weather, call predictor, render)
- `ml/train_hybrid.py` — Train a hybrid RandomForest + ExtraTrees (averaged) model with feature engineering and log target transform
- `ml/pm25_pipeline.pkl` — Saved full pipeline (feature engineering + hybrid + log target transform)
	(Legacy: `pm25_hybrid.pkl` + `scaler.pkl` may exist but are not required now.)
- `requirements.txt` — Python deps

## Quick start (Windows PowerShell)

1) Install dependencies:

```powershell
pip install -r requirements.txt
```

2) (Optional) Train the model to replace heuristic with the hybrid predictor (creates `pm25_pipeline.pkl`):

```powershell
python .\ml\train_hybrid.py
```

This creates `ml/pm25_pipeline.pkl`.

3) Run the app:

```powershell
python .\app.py
```

Open http://127.0.0.1:5000 in your browser. Search a location and see weather + PM2.5 prediction.

## Using a real dataset

- Replace the synthetic data generation in `ml/train_hybrid.py` with code to load your CSV.
- Keep the `FEATURES` list aligned with your columns and the app.
- Re-run the training.

## Notes

- The app uses a heuristic fallback when no model is present, so it always functions.
- Predictions are illustrative and not medical advice.
