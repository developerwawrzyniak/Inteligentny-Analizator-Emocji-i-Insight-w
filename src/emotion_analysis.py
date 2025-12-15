import argparse
import pandas as pd
from transformers import pipeline

# -------------------------
# EMOTION MODEL SETUP
# -------------------------
# Tutaj używamy gotowego pipeline Hugging Face do analizy emocji
# Możesz zmienić model np. "j-hartmann/emotion-english-distilroberta-base"
emotion_analyzer = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True,
)

EMOTIONS = ["joy", "anger", "sadness", "fear", "surprise", "neutral"]


def predict_emotion(text):
    """
    Zwraca dominującą emocję i jej score (0-1) dla pojedynczego tekstu.
    """
    try:
        result = emotion_analyzer(text)
        # result to lista słowników z 'label' i 'score'
        result = result[0]  # pipeline zwraca listę list
        # Wybierz label z największym score
        best = max(result, key=lambda x: x["score"])
        emotion_label = best["label"].lower()
        score = float(best["score"])
        # Jeśli label nie jest w naszej liście EMOTIONS, ustaw jako neutral
        if emotion_label not in EMOTIONS:
            emotion_label = "neutral"
        return emotion_label, score
    except Exception:
        # W przypadku problemu ustaw neutral
        return "neutral", 0.0


# -------------------------
# RUN FUNCTION
# -------------------------
def run(input_path, output_path):
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    # Sprawdzenie wymaganych kolumn
    if (
        "clean_text" not in df.columns
        or "topic_id" not in df.columns
        or "topic_summary" not in df.columns
    ):
        raise ValueError(
            "Input CSV must contain 'clean_text', 'topic_id', and 'topic_summary' columns."
        )

    print("Running emotion analysis...")

    dominant_emotions = []
    emotion_scores = []

    for text in df["clean_text"].tolist():
        emotion, score = predict_emotion(str(text))
        dominant_emotions.append(emotion)
        emotion_scores.append(score)

    df["dominant_emotion"] = dominant_emotions
    df["emotion_score"] = emotion_scores

    print(f"Saving output to {output_path}")
    df.to_csv(output_path, index=False)
    print("Done.")


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    run(args.input, args.output)
