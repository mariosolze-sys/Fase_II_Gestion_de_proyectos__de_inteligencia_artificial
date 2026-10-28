"""Entrena un modelo demostrativo reproducible con Wisconsin Breast Cancer."""
from pathlib import Path
import json
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
def main():
    dataset = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=42, stratify=dataset.target)
    pipeline = Pipeline([('scale', StandardScaler()), ('classifier', LogisticRegression(max_iter=2000, random_state=42))])
    pipeline.fit(X_train, y_train)
    scores = pipeline.predict_proba(X_test)[:, 1]
    predictions = (scores >= 0.5).astype(int)
    out = ROOT / 'model'; out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out / 'model.joblib')
    meta = {'dataset': 'sklearn.datasets.load_breast_cancer', 'features': list(dataset.feature_names), 'class_mapping': {'0': 'maligno', '1': 'benigno'}, 'positive_class': 'benigno', 'threshold': 0.5, 'test_size': 0.2, 'random_state': 42, 'sample': {name: float(value) for name, value in zip(dataset.feature_names, X_test[0])}, 'metrics': {'accuracy': float(accuracy_score(y_test, predictions)), 'f1_benign': float(f1_score(y_test, predictions)), 'recall_benign': float(recall_score(y_test, predictions)), 'roc_auc_benign': float(roc_auc_score(y_test, scores))}, 'test_rows': len(y_test)}
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(meta['metrics'], indent=2))
if __name__ == '__main__': main()
