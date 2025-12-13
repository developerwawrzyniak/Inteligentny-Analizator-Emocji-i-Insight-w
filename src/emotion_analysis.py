import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL)
    model.eval()

    id2label = model.config.id2label
    return tokenizer, model, id2label


def predict_emotion(text, tokenizer, model, id2label):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1).squeeze()
    emotion_idx = int(torch.argmax(probs))

    emotion_label = id2label[emotion_idx]
    emotion_score = float(probs[emotion_idx])

    return emotion_label, round(emotion_score, 3)


def run(input_path, output_path):
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    if "clean_text" not in df.columns:
        raise ValueError("Input CSV must contain 'clean_text' column")

    tokenizer, model, id2label = load_model()

    emotions = []
    scores = []

    print("Running emotion analysis...")
    for text in df["clean_text"].fillna("").astype(str):
        emotion, score = predict_emotion(text, tokenizer, model, id2label)
        emotions.append(emotion)
        scores.append(score)

    df["dominant_emotion"] = emotions
    df["emotion_score"] = scores

    output_cols = [
        "clean_text",
        "topic_id",
        "topic_summary",
        "dominant_emotion",
        "emotion_score",
    ]

    print(f"Saving output to {output_path}")
    df[output_cols].to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run(args.input, args.output)
