
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# Load models
model_CA = pickle.load(open("gb_model.pkl", "rb"))
model_DEF = pickle.load(open("gb_model1.pkl", "rb"))

app = FastAPI()

class TyreInput(BaseModel):
    tyre_type: str
    inflation_pressure: float
    normal_load: float

@app.post("/predict")
def predict_characteristics(data: TyreInput):
    tt = 1 if data.tyre_type.lower() == "tubeless" else 0
    psi = data.inflation_pressure * 0.14
    query = np.array([[tt, data.normal_load, psi]])

    predicted_CA = model_CA.predict(query)[0] * 1_000_000
    predicted_DEF = model_DEF.predict(query)[0]

    return {
        "contact_area_mm2": round(predicted_CA, 2),
        "tyre_deflection_mm": round(predicted_DEF, 2)
    }
