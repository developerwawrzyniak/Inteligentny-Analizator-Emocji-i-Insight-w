from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


def predict_sentiment(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
    labels = ["negative", "neutral", "positive"]
    return dict(zip(labels, probs.tolist()))
