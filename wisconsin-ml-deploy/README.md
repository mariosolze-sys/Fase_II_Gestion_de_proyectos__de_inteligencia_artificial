# Wisconsin ML Demo | Docker + FastAPI + Web

Aplicación académica reproducible para demostrar frontend ↔ API ↔ modelo de ML en un mismo contenedor. **No es un dispositivo médico ni debe utilizarse para diagnóstico.**

## Arquitectura

```text
Navegador (HTML + JavaScript)
   | GET /api/model-info / POST /api/predict
   v
FastAPI (uvicorn, puerto 8080)
   | lectura local
   v
Pipeline joblib: StandardScaler + LogisticRegression
```

## Contenido

- `app/main.py`: API, validaciones y servido de interfaz
- `app/static/index.html`: frontend responsive
- `model/model.joblib`: artefacto serializado reproducible
- `model/metadata.json`: orden de variables, mapeo de clases, ejemplo y métricas
- `scripts/train_model.py`: entrenamiento con partición estratificada 80/20, semilla 42
- `tests/test_api.py`: pruebas funcionales y casos de error
- `Dockerfile`, `compose.yaml`, `.dockerignore`: ejecución en contenedor
- `docs/`: manual y validación en formato Word

## Inicio rápido con Docker

Requisito: Docker Desktop o Docker Engine con Compose.

```bash
docker build -t wisconsin-ml-demo:1.0 .
docker run --rm -p 8080:8080 --name wisconsin-demo wisconsin-ml-demo:1.0
```

Visita: http://localhost:8080 (frontend), http://localhost:8080/docs (Swagger), http://localhost:8080/health (salud).

Alternativa:

```bash
docker compose up --build -d
docker compose ps
docker compose logs --tail=50
docker compose down
```

## Inicio local sin Docker

Python 3.12 recomendado.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python scripts/train_model.py   # Opcional si quieres regenerar modelo y metadata
python -m uvicorn app.main:app --reload --port 8080
```

## Pruebas

```bash
python -m pytest -q
curl -i http://localhost:8080/health
curl -i http://localhost:8080/api/model-info
```

Para probar predicción con el ejemplo completo sin escribir 30 números:

```bash
python -c "import json,urllib.request; m=json.load(open('model/metadata.json')); req=urllib.request.Request('http://localhost:8080/api/predict',data=json.dumps({'features':m['sample']}).encode(),headers={'Content-Type':'application/json'}); print(urllib.request.urlopen(req).read().decode())"
```

### API

| Endpoint | Método | Función |
|---|---|---|
| `/` | GET | Interfaz HTML |
| `/health` | GET | Estado de aplicación y carga del modelo |
| `/api/model-info` | GET | 30 features, clase objetivo, ejemplo, métricas |
| `/api/predict` | POST | Infiere con `{"features": { ...30 variables... }}` |
| `/docs` | GET | Documentación interactiva OpenAPI |

El endpoint de predicción devuelve clase (`0`: maligno, `1`: benigno) y probabilidades. Solicitudes incompletas, variables extra o tipos incompatibles producen HTTP 422.

## Publicar en GitHub

1. Crea en https://github.com/new un repositorio vacío llamado `wisconsin-ml-deploy` (sin README inicial).
2. Desde esta carpeta ejecuta:

```bash
git init
git add .
git commit -m "Proyecto ML desplegable con Docker y FastAPI"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/wisconsin-ml-deploy.git
git push -u origin main
```

Reemplaza `TU_USUARIO`; autentícate mediante el método que GitHub te indique. Verifica que aparezcan `Dockerfile`, `app/`, `model/`, `tests/` y `docs/`.

## Cloud Run (Google Cloud)

Requisitos: cuenta y proyecto Google Cloud con facturación habilitada, `gcloud` instalado e iniciado, APIs `run.googleapis.com`, `artifactregistry.googleapis.com`, `cloudbuild.googleapis.com` habilitadas y permisos IAM apropiados.

```bash
export PROJECT_ID="TU_PROYECTO"
export REGION="us-central1"
gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
gcloud artifacts repositories create wisconsin-demo --repository-format=docker --location="$REGION" --description="Demo academica ML"
gcloud auth configure-docker "$REGION-docker.pkg.dev"
export IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/wisconsin-demo/api:1.0"
docker build -t "$IMAGE" .
docker push "$IMAGE"
gcloud run deploy wisconsin-demo --image "$IMAGE" --region "$REGION" --port 8080 --allow-unauthenticated --memory 1Gi
```

**Nota:** Con Docker en ARM (Apple Silicon) y despliegue en Cloud Run x86_64, construye `linux/amd64` con `docker buildx build --platform linux/amd64 -t "$IMAGE" --push .` en vez de `docker build` + `docker push`. Verifica autorizaciones de IAM antes de habilitar acceso público. Una API pública necesita controles de abuso y seguridad antes de usarse en producción.

Verificación: abre la URL que devuelve Cloud Run y consulta `URL/health` y `URL/docs`. Borra el servicio para evitar cargos: `gcloud run services delete wisconsin-demo --region "$REGION"` (y limpia imágenes si procede).

## Limitaciones y reproducibilidad

- Entrenamiento desde cero con scikit-learn Wisconsin, `random_state=42`.
- `model.joblib` debe tratarse como archivo confiable; joblib/pickle no deben cargar archivos de terceros.
- Las estadísticas de clasificación son sobre el dataset de prueba, no una validación clínica.
- No se recopilan ni almacenan datos de pacientes.
- La compilación Docker y despliegue Cloud no se han validado desde el entorno de creación; debe adjuntarse evidencia real tras ejecutarlos.
- **Uso de IA:** un asistente de IA apoyó el desarrollo de código y borradores de documentación; pruebas ejecutadas y limitaciones se reportan por separado. Copilot es opcional, no se afirma haberlo utilizado.

Referencias: https://fastapi.tiangolo.com/deployment/docker/ y https://docs.cloud.google.com/run/docs/deploying
