from fastapi import FastAPI
import pickle
from pydantic import BaseModel


class LeadClient(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float


app = FastAPI()


with open("pipeline_v1.bin", "rb") as f_in:
    model = pickle.load(f_in)


@app.post("/predict")
def predict(client: LeadClient):
    sample = client.dict()
    pred = model.predict_proba([sample])[0, 1]
    return {"prediction": float(pred)}
