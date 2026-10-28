"""API demostrativa de inferencia; no apta para diagnóstico clínico."""
from pathlib import Path
import json
import math
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

BASE = Path(__file__).resolve().parents[1]
metadata = json.loads((BASE / 'model' / 'metadata.json').read_text(encoding='utf-8'))
model = joblib.load(BASE / 'model' / 'model.joblib')  # Archivo confiable generado localmente, nunca de usuarios.
app = FastAPI(title='Wisconsin ML Demo API', version='1.0.0', description='Demostración académica; NO utilizar para decisiones médicas.')

class PredictRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    features: dict[str, float]

@app.get('/')
def home():
    return FileResponse(BASE / 'app' / 'static' / 'index.html')

@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': True}

@app.get('/api/model-info')
def model_info():
    return metadata

@app.post('/api/predict')
def predict(payload: PredictRequest):
    names = metadata['features']
    keys = set(payload.features)
    if keys != set(names):
        raise HTTPException(status_code=422, detail={'missing': sorted(set(names) - keys), 'unexpected': sorted(keys - set(names))})
    values = [payload.features[k] for k in names]
    if not all(math.isfinite(x) for x in values):
        raise HTTPException(status_code=422, detail='Todos los valores deben ser finitos.')
    probs = model.predict_proba(np.array([values], dtype=float))[0]
    benign = float(probs[1])
    result = int(benign >= metadata['threshold'])
    return {'prediction': result, 'label': metadata['class_mapping'][str(result)], 'probability_benign': benign, 'probability_malignant': float(probs[0]), 'threshold': metadata['threshold'], 'disclaimer': 'Demostración académica. No usar para diagnóstico médico.'}
