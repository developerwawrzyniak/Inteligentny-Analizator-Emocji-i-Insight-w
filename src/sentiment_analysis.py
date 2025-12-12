import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm


def run(input_path, output_path):
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    if "clean_text" not in df.columns:
        raise ValueError("Input CSV must contain a 'clean_text' column.")

    print("Loading sentiment model...")
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        truncation=True,  # <--- DODANE
        max_length=256,  # <--- DODANE
        batch_size=16,
        device=-1,
    )

    print("Running sentiment analysis...")

    sentiments = []
    scores = []

    for out in tqdm(sentiment_pipe(df["clean_text"].tolist()), total=len(df)):
        sentiments.append(out["label"])
        scores.append(out["score"])

    df["sentiment"] = sentiments
    df["sentiment_score"] = scores

    print(f"Saving output to {output_path}")
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    run(args.input, args.output)
