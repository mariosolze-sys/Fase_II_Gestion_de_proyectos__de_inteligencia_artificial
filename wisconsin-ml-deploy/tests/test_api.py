import math
from fastapi.testclient import TestClient
from app.main import app, metadata

client = TestClient(app)
def test_home():
    r=client.get('/'); assert r.status_code==200 and 'Clasificador Wisconsin' in r.text
def test_health():
    assert client.get('/health').json()['model_loaded'] is True
def test_info():
    r=client.get('/api/model-info');assert r.status_code==200 and len(r.json()['features'])==30
def test_predict_sample():
    r=client.post('/api/predict',json={'features':metadata['sample']}); assert r.status_code==200
    result=r.json(); assert result['prediction'] in [0,1]; assert math.isclose(result['probability_benign']+result['probability_malignant'],1,abs_tol=1e-7)
def test_missing_feature():
    example=metadata['sample'].copy();example.pop(next(iter(example)))
    r=client.post('/api/predict',json={'features':example});assert r.status_code==422 and r.json()['detail']['missing']
def test_unexpected_feature():
    example=metadata['sample'].copy();example['extra']=1
    assert client.post('/api/predict',json={'features':example}).status_code==422
def test_invalid_type():
    example=metadata['sample'].copy();example[next(iter(example))]='non-number'
    assert client.post('/api/predict',json={'features':example}).status_code==422
def test_empty_body():
    assert client.post('/api/predict',json={}).status_code==422
