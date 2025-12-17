import argparse
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")


def sentiment_to_emotion(score):
    if score >= 0.4:
        return "joy"
    elif score >= 0.1:
        return "positive"
    elif score <= -0.4:
        return "anger"
    elif score <= -0.1:
        return "sadness"
    else:
        return "neutral"


def run(input_path, output_path):
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    sia = SentimentIntensityAnalyzer()

    emotions = []
    intensities = []

    for text in df["clean_text"]:
        score = sia.polarity_scores(str(text))["compound"]
        emotions.append(sentiment_to_emotion(score))
        intensities.append(abs(score))

    df["dominant_emotion"] = emotions
    df["emotion_intensity"] = intensities

    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run(args.input, args.output)
