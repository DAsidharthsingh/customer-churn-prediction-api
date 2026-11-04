from typing import Literal

from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    customer_id: str = Field(..., alias="CustomerID")
    count: int = Field(..., ge=1, alias="Count")
    country: str = Field(..., alias="Country")
    state: str = Field(..., alias="State")
    city: str = Field(..., alias="City")
    zip_code: int = Field(..., ge=0, alias="Zip Code")
    lat_long: str = Field(..., alias="Lat Long")
    latitude: float = Field(..., alias="Latitude")
    longitude: float = Field(..., alias="Longitude")

    gender: Literal["Male", "Female"] = Field(..., alias="Gender")
    senior_citizen: Literal["No", "Yes"] = Field(..., alias="Senior Citizen")
    partner: Literal["No", "Yes"] = Field(..., alias="Partner")
    dependents: Literal["No", "Yes"] = Field(..., alias="Dependents")

    tenure_months: int = Field(
        ..., ge=0, le=100, alias="Tenure Months"
    )

    phone_service: Literal["No", "Yes"] = Field(
        ..., alias="Phone Service"
    )

    multiple_lines: Literal[
        "No phone service", "No", "Yes"
    ] = Field(..., alias="Multiple Lines")

    internet_service: Literal[
        "DSL", "Fiber optic", "No"
    ] = Field(..., alias="Internet Service")

    online_security: Literal[
        "No internet service", "No", "Yes"
    ] = Field(..., alias="Online Security")

    online_backup: Literal[
        "No internet service", "No", "Yes"
    ] = Field(..., alias="Online Backup")

    device_protection: Literal[
        "No internet service", "No", "Yes"
    ] = Field(..., alias="Device Protection")

    tech_support: Literal[
        "No internet service", "No", "Yes"
    ] = Field(..., alias="Tech Support")

    streaming_tv: Literal[
        "No internet service", "No", "Yes"
    ] = Field(..., alias="Streaming TV")

    streaming_movies: Literal[
        "No internet service", "No", "Yes"
    ] = Field(..., alias="Streaming Movies")

    contract: Literal[
        "Month-to-month", "One year", "Two year"
    ] = Field(..., alias="Contract")

    paperless_billing: Literal[
        "No", "Yes"
    ] = Field(..., alias="Paperless Billing")

    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., alias="Payment Method")

    monthly_charges: float = Field(
        ..., ge=0, le=150, alias="Monthly Charges"
    )

    total_charges: float = Field(
        ..., ge=0, le=20000, alias="Total Charges"
    )

    cltv: int = Field(
        ..., ge=0, le=20000, alias="CLTV"
    )

    model_config = {
        "populate_by_name": True,
        "extra": "forbid",
    }