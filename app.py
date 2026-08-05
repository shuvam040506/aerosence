from flask import Flask, render_template, request, jsonify
import requests
import joblib
import numpy as np
from pathlib import Path

app = Flask(__name__)

MODEL_PATH = Path(__file__).with_name('ml') / 'pm25_hybrid.pkl'
SCALER_PATH = Path(__file__).with_name('ml') / 'scaler.pkl'
PIPELINE_PATH = Path(__file__).with_name('ml') / 'pm25_pipeline.pkl'

# Lazy-loaded globals
_model = None
_scaler = None
_pipeline = None


# Provide a compatible class so joblib can unpickle older models trained with this type
class Hybrid:
    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b
    def fit(self, X, y):
        # Not used in app runtime
        if self.a is not None:
            self.a.fit(X, y)
        if self.b is not None:
            self.b.fit(X, y)
        return self
    def predict(self, X):
        pa = self.a.predict(X) if self.a is not None else 0
        pb = self.b.predict(X) if self.b is not None else 0
        return 0.5 * (np.array(pa) + np.array(pb))


def load_artifacts():
    global _pipeline
    # Prefer and only load the safe pipeline. Avoid legacy objects that fail to unpickle.
    if _pipeline is None and PIPELINE_PATH.exists():
        try:
            _pipeline = joblib.load(PIPELINE_PATH)
            print('[artifacts] loaded pipeline')
        except Exception as e:
            print('[artifacts] failed to load pipeline:', e)
            _pipeline = None


@app.route('/')
def index():
    return render_template('index.html')


@app.get('/api/geocode')
def api_geocode():
    """Return top 5 geocoding matches for a query (to disambiguate same-name cities)."""
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify({"error": "Missing q"}), 400
    geo = requests.get(
        'https://geocoding-api.open-meteo.com/v1/search',
        params={'name': q, 'count': 5, 'language': 'en', 'format': 'json'},
        timeout=15
    ).json()
    results = geo.get('results') or []
    items = []
    for r in results:
        items.append({
            'name': r.get('name'),
            'country': r.get('country'),
            'admin1': r.get('admin1'),
            'latitude': r.get('latitude'),
            'longitude': r.get('longitude'),
            'timezone': r.get('timezone')
        })
    return jsonify({'results': items})


@app.get('/api/weather')
def api_weather():
    """Fetch weather by place name (q) or coordinates (lat, lon); merges air-quality."""
    q = (request.args.get('q') or '').strip()
    lat_str = request.args.get('lat')
    lon_str = request.args.get('lon')

    g = None
    if lat_str and lon_str:
        try:
            lat = float(lat_str)
            lon = float(lon_str)
            g = {'name': request.args.get('label') or None, 'country': None, 'latitude': lat, 'longitude': lon}
        except Exception:
            return jsonify({"error": "Invalid lat/lon"}), 400
    else:
        if not q:
            return jsonify({"error": "Missing q or lat/lon"}), 400
        # Geocode name -> lat/lon
        geo = requests.get(
            'https://geocoding-api.open-meteo.com/v1/search',
            params={'name': q, 'count': 1, 'language': 'en', 'format': 'json'},
            timeout=15
        ).json()
        if not geo.get('results'):
            return jsonify({"error": "Location not found"}), 404
        g = geo['results'][0]
        lat, lon = g['latitude'], g['longitude']

    # 2) Weather forecast/current from Open-Meteo
    weather = requests.get(
        'https://api.open-meteo.com/v1/forecast',
        params={
            'latitude': lat,
            'longitude': lon,
            'current': 'temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,wind_speed_10m,wind_direction_10m,surface_pressure,weather_code,is_day',
            'hourly': 'uv_index,precipitation,relative_humidity_2m,dew_point_2m,surface_pressure,weather_code,wind_speed_10m',
            'timezone': 'auto'
        },
        timeout=20
    ).json()

    # 3) Air quality (pm10, pm2_5, NO2, SO2, O3) from Open-Meteo Air Quality API
    try:
        aq = requests.get(
            'https://air-quality-api.open-meteo.com/v1/air-quality',
            params={
                'latitude': lat,
                'longitude': lon,
                'hourly': 'pm10,pm2_5,nitrogen_dioxide,sulphur_dioxide,ozone',
                'timezone': 'auto'
            },
            timeout=20
        ).json()
        if isinstance(weather.get('hourly'), dict) and isinstance(aq.get('hourly'), dict):
            # Merge pm10 and pm2_5 arrays into weather.hourly
            for key in ('pm10', 'pm2_5', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone'):
                if key in aq['hourly']:
                    weather['hourly'][key] = aq['hourly'][key]
            # If weather.hourly has no time, borrow from AQ
            if 'time' not in weather['hourly'] and 'time' in aq['hourly']:
                weather['hourly']['time'] = aq['hourly']['time']
    except Exception:
        # Non-fatal; continue without AQ merge
        pass

    return jsonify({
        'geocoding': {
            'name': (g.get('name') if isinstance(g, dict) else None),
            'country': (g.get('country') if isinstance(g, dict) else None),
            'timezone': weather.get('timezone'),
            'latitude': lat,
            'longitude': lon
        },
        'weather': weather
    })


@app.post('/api/predict')
def api_predict():
    """Predict PM2.5 using weather features from client."""
    # Only try to load the pipeline (safe). Legacy model load is skipped to avoid unpickle issues.
    global _pipeline
    if _pipeline is None and PIPELINE_PATH.exists():
        try:
            _pipeline = joblib.load(PIPELINE_PATH)
            print('[artifacts] loaded pipeline (on-demand)')
        except Exception as e:
            print('[artifacts] pipeline load failed in request:', e)
            _pipeline = None

    data = request.get_json(force=True)
    # Expected fields (numeric)
    feats = [
        'temperature_2m', 'relative_humidity_2m', 'apparent_temperature', 'precipitation',
        'wind_speed_10m', 'surface_pressure', 'uv_index', 'dew_point_2m', 'pm10'
    ]
    try:
        def safe_float(val, default=0.0):
            if val is None:
                return default
            try:
                return float(val)
            except Exception:
                return default

        x_list = [safe_float(data.get(f), 0.0) for f in feats]
        x = np.array([x_list], dtype=float)
    except Exception:
        return jsonify({"error": "Invalid input"}), 400

    # If we have full pipeline, use it directly
    if _pipeline is not None:
        y = float(_pipeline.predict(x)[0])
        try:
            print('[predict] pipeline input:', dict(zip(feats, x_list)), '->', y)
        except Exception:
            pass
        return jsonify({'pm25': round(max(0.0, y), 2), 'model': 'pipeline_hgb_logtarget'})

    # Else heuristic so app always works
    pm10_v = float(data.get('pm10', 0) or 0)
    rh_v = float(data.get('relative_humidity_2m', 50) or 50)
    temp_v = float(data.get('temperature_2m', 20) or 20)
    pred = 0.45 * pm10_v * (1 + (rh_v - 50) / 200) * (1 - (temp_v - 20) / 400)
    try:
        print('[predict] heuristic input:', dict(zip(feats, x_list)), '->', pred)
    except Exception:
        pass
    return jsonify({'pm25': round(max(0.0, pred), 2), 'model': 'heuristic'})

    # Note: Legacy separate scaler+model path removed to avoid unpickle issues in this demo.


if __name__ == '__main__':
    app.run(debug=True)
