import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================
# LOAD MODELS
# ==========================
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)

labels = ["negative", "neutral", "positive"]


def predict_sentiment(text):
    """Return sentiment label + probabilities."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]

    sentiment = labels[probs.argmax()]

    return sentiment, float(probs[0]), float(probs[1]), float(probs[2])


# ==========================
# MAIN PIPELINE
# ==========================


def run(input_path, output_path):
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    sentiments = []
    neg_p = []
    neu_p = []
    pos_p = []

    print("Running sentiment analysis...")

    for text in df["clean_text"]:
        sentiment, p_neg, p_neu, p_pos = predict_sentiment(text)
        sentiments.append(sentiment)
        neg_p.append(p_neg)
        neu_p.append(p_neu)
        pos_p.append(p_pos)

    df["sentiment"] = sentiments
    df["prob_negative"] = neg_p
    df["prob_neutral"] = neu_p
    df["prob_positive"] = pos_p

    print(f"Saving output to {output_path}")
    df.to_csv(output_path, index=False)


# ==========================
# CLI
# ==========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run(args.input, args.output)
