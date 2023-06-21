from fastapi import FastAPI
from pydantic import BaseModel
from app.model import generate_summary

app = FastAPI()


class TextIn(BaseModel):
    url: str


class InferenceOut(BaseModel):
    summary: str


@app.get("/")
def home():
    return {"health check": "OK", 
            "model_version": model_version}


@app.post("/summarize", response_model=InferenceOut)
def predict(payload: TextIn):
    summary, time = generate_summary(payload.url)
    return {"summary": summary,
            "inference_time": time}

