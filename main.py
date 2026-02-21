from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model once at startup
model = joblib.load("./artifacts/model.pkl")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
async def root():
    return {"Hello": "World"}

@app.post("/predict")
async def predict(data: IrisInput):
    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])
    
    prediction = model.predict(features)[0]

    return {"prediction": int(prediction)}
