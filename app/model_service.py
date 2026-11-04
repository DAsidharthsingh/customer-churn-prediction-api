from pathlib import Path
from typing import Any
import joblib
import pandas as pd


MODEL_PATH = Path("model/churn_model.pkl")
ENCODER_PATH = Path("model/label_encoders.pkl")


class ModelService:
    def __init__(self, model_path: Path = MODEL_PATH, encoder_path: Path = ENCODER_PATH):
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.model = None
        self.encoders = {}
        self.load_error = None

    def load(self) -> None:
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Missing model artifact: {self.model_path}. "
                    "Copy your existing churn_model.pkl into model/."
                )
            if not self.encoder_path.exists():
                raise FileNotFoundError(
                    f"Missing encoder artifact: {self.encoder_path}. "
                    "Copy your existing label_encoders.pkl into model/."
                )

            self.model = joblib.load(self.model_path)
            self.encoders = joblib.load(self.encoder_path)
            self.load_error = None
        except Exception as exc:
            self.model = None
            self.encoders = {}
            self.load_error = str(exc)

    @property
    def ready(self) -> bool:
        return self.model is not None

    def _raw_frame(self, payload: dict[str, Any]) -> pd.DataFrame:
        # Convert API field names back to the exact column names used by the
        # existing Streamlit application/model.
        mapping = {
            "customer_id": "CustomerID",
            "count": "Count",
            "country": "Country",
            "state": "State",
            "city": "City",
            "zip_code": "Zip Code",
            "lat_long": "Lat Long",
            "latitude": "Latitude",
            "longitude": "Longitude",
            "gender": "Gender",
            "senior_citizen": "Senior Citizen",
            "partner": "Partner",
            "dependents": "Dependents",
            "tenure_months": "Tenure Months",
            "phone_service": "Phone Service",
            "multiple_lines": "Multiple Lines",
            "internet_service": "Internet Service",
            "online_security": "Online Security",
            "online_backup": "Online Backup",
            "device_protection": "Device Protection",
            "tech_support": "Tech Support",
            "streaming_tv": "Streaming TV",
            "streaming_movies": "Streaming Movies",
            "contract": "Contract",
            "paperless_billing": "Paperless Billing",
            "payment_method": "Payment Method",
            "monthly_charges": "Monthly Charges",
            "total_charges": "Total Charges",
            "cltv": "CLTV",
        }
        row = {mapping[k]: v for k, v in payload.items()}
        return pd.DataFrame([row])

    def preprocess(self, payload: dict[str, Any]) -> pd.DataFrame:
        if not self.ready:
            raise RuntimeError(self.load_error or "Model is not loaded.")

        frame = self._raw_frame(payload)

        # Preserve the exact feature order learned during training.
        if hasattr(self.model, "feature_names_in_"):
            expected = list(self.model.feature_names_in_)
            frame = frame.drop(
                columns=[c for c in frame.columns if c not in expected],
                errors="ignore",
            )

            for col in expected:
                if col not in frame.columns:
                    frame[col] = 0

            frame = frame[expected]

        # Reuse the serialized encoders from the original project so runtime
        # inference matches the training representation.
        for col in frame.columns:
            if col in self.encoders:
                try:
                    frame[col] = self.encoders[col].transform(frame[col])
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid categorical value for '{col}'. "
                        "Use one of the values accepted by the trained encoder."
                    ) from exc
            elif frame[col].dtype == "object":
                frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)

        return frame

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        frame = self.preprocess(payload)

        prediction = int(self.model.predict(frame)[0])

        if not hasattr(self.model, "predict_proba"):
            raise RuntimeError("Loaded model does not expose predict_proba().")

        probability = float(self.model.predict_proba(frame)[0][1])

        if probability < 0.30:
            risk = "Low"
        elif probability < 0.70:
            risk = "Medium"
        else:
            risk = "High"

        return {
            "prediction": prediction,
            "label": "Churn" if prediction == 1 else "No Churn",
            "churn_probability": round(probability, 6),
            "risk_level": risk,
        }
