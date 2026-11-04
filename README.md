# Customer Churn ML API

Production-oriented extension of the existing Customer Churn Prediction project.

## Architecture

Client → FastAPI REST API → input validation → serialized encoders → Random Forest → churn probability → risk level

## 1. Put the trained artifacts in `model/`

Copy the existing project artifacts:

- `churn_model.pkl`
- `label_encoders.pkl`

into:

```text
model/
├── churn_model.pkl
└── label_encoders.pkl
```

These files are intentionally not included in this starter package.

## 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

## 4. Start the API

```powershell
uvicorn app.main:app --reload
```

Open:

- Swagger UI: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health
- Metadata: http://127.0.0.1:8000/metadata

## 5. Run tests

```powershell
pytest -q
```

## 6. Docker

Build:

```powershell
docker build -t customer-churn-api .
```

Run:

```powershell
docker run --rm -p 8000:8000 customer-churn-api
```

## Important

The API intentionally does not contain a fallback prediction. If the trained model is unavailable, `/predict` returns HTTP 503. This keeps the API honest and prevents fake predictions from being presented as ML output.
