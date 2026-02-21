""" Main """
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model once at startup
model = joblib.load("./model.pkl")

class IrisInput(BaseModel):
    """ Input pydantic model """
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
async def root():
    """ Greeting """
    return {"Hello": "World"}

@app.post("/predict")
async def predict(data: IrisInput):
    """ Predictions """
    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    prediction = model.predict(features)[0]

    return {"prediction": int(prediction)}
