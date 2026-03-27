from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "models/anli_r2_roberta"

label_map = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def predict(premise: str, hypothesis: str) -> str:
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=192
    )

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

    return label_map[pred]

if __name__ == "__main__":
    premise = "A man is speaking into a microphone."
    hypothesis = "A person is giving a speech."
    prediction = predict(premise, hypothesis)

    print("Premise:", premise)
    print("Hypothesis:", hypothesis)
    print("Prediction:", prediction)
