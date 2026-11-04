from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def get_valid_payload():
    return {
        "customer_id": "7590-VHVEG",
        "count": 1,
        "country": "United States",
        "state": "California",
        "city": "Los Angeles",
        "zip_code": 90001,
        "lat_long": "33.99,-118.09",
        "latitude": 33.99,
        "longitude": -118.09,
        "gender": "Female",
        "senior_citizen": "No",
        "partner": "Yes",
        "dependents": "No",
        "tenure_months": 12,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "DSL",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 50.0,
        "total_charges": 600.0,
        "cltv": 5000.0,
    }


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["service"] == "customer-churn-ml-api"


def test_metadata_endpoint():
    response = client.get("/metadata")

    assert response.status_code == 200

    data = response.json()

    assert data["model_type"] == "RandomForestClassifier"
    assert data["feature_count"] == 27
    assert data["encoder_count"] > 0


def test_valid_prediction():
    response = client.post(
        "/predict",
        json=get_valid_payload()
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]
    assert data["label"] in ["Churn", "No Churn"]
    assert 0 <= data["churn_probability"] <= 1
    assert data["risk_level"] in ["Low", "Medium", "High"]


def test_invalid_gender():
    payload = get_valid_payload()
    payload["gender"] = "SomethingElse"

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422


def test_negative_tenure():
    payload = get_valid_payload()
    payload["tenure_months"] = -5

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422


def test_extra_field_rejected():
    payload = get_valid_payload()
    payload["fake_field"] = "test"

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422


def test_missing_required_field():
    payload = get_valid_payload()
    del payload["tenure_months"]

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422