import pandas as pd
import argparse
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# ensure lexicon exists
nltk.download("vader_lexicon", quiet=True)

sia = SentimentIntensityAnalyzer()


def detect_emotion(text: str):
    score = sia.polarity_scores(text)["compound"]

    if score >= 0.4:
        emotion = "joy"
    elif score <= -0.4:
        emotion = "anger"
    else:
        emotion = "neutral"

    intensity = round(abs(score), 2)
    return emotion, intensity


def run(input_path: str, output_path: str):
    print(f"Loading file: {input_path}")
    df = pd.read_csv(input_path)

    required_cols = {"clean_text", "topic_id", "topic_keywords"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols}")

    print("Analyzing emotions...")
    emotions = df["clean_text"].astype(str).apply(detect_emotion)

    df["emotion"] = emotions.apply(lambda x: x[0])
    df["intensity"] = emotions.apply(lambda x: x[1])

    output_cols = ["clean_text", "topic_id", "topic_keywords", "emotion", "intensity"]

    print(f"Saving output to: {output_path}")
    df[output_cols].to_csv(output_path, index=False)

    print("Emotion analysis completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()
    run(args.input, args.output)
