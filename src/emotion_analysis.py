import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"


def run(input_path, output_path):
    print(f"Loading data: {input_path}")
    df = pd.read_csv(input_path)

    if "clean_text" not in df.columns:
        raise ValueError("Input CSV must contain 'clean_text' column.")

    print("Loading emotion model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    emotions = model.config.id2label
    emotion_scores = {label: [] for label in emotions.values()}
    dominant_emotions = []

    print("Running emotion detection...")
    for text in tqdm(df["clean_text"].astype(str)):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).squeeze()

        scores = probs.tolist()
        for i, label in emotions.items():
            emotion_scores[label].append(scores[i])

        dominant_emotions.append(emotions[int(probs.argmax())])

    for emotion, values in emotion_scores.items():
        df[f"emotion_{emotion}"] = values

    df["dominant_emotion"] = dominant_emotions

    print(f"Saving output to: {output_path}")
    df.to_csv(output_path, index=False)
    print("✅ Emotion analysis completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion Analysis")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run(args.input, args.output)
