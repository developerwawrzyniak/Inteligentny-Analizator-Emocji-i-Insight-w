from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL = "j-hartmann/emotion-english-distilroberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


def predict_emotion(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
    # map zgodnie z dokumentacją modelu
    return probs
