""" Tests """

from unittest.mock import patch, MagicMock
import numpy as np
from fastapi.testclient import TestClient

# Mock the model before importing the app
mock_model = MagicMock()
mock_model.predict.return_value = np.array([1])

with patch("joblib.load", return_value=mock_model):
    from app.main import app  # Import after patching

client = TestClient(app)


def test_root():
    """ root tests """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}


def test_predict_success():
    """ tests prediction method """
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] == 1


def test_predict_validation_error():
    """ tests validation error """
    payload = {
        "sepal_length": "invalid",  # wrong type
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422  # FastAPI validation error
