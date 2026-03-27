from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

MODEL_PATH = "models/anli_r2_roberta"

label_map = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
}

# Load model once at startup
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

class InputData(BaseModel):
    premise: str
    hypothesis: str

@app.get("/")
def home():
    return {"message": "ANLI NLI Model API is running"}

@app.post("/predict")
def predict(data: InputData):
    inputs = tokenizer(
        data.premise,
        data.hypothesis,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=192
    )

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

    return {"prediction": label_map[pred]}
