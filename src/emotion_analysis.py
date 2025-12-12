import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm


def run(input_path, output_path):
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    if "clean_text" not in df.columns:
        raise ValueError("Input CSV must contain a 'clean_text' column.")

    print("Loading emotion model...")
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    emotion_pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=256,
        batch_size=16,
        return_all_scores=True,
    )

    emotions = []
    emotion_scores = []

    print("Running emotion analysis...")

    for out in tqdm(emotion_pipe(df["clean_text"].tolist()), total=len(df)):
        # out to lista dictów z emocjami i score
        sorted_out = sorted(out, key=lambda x: x["score"], reverse=True)
        emotions.append(sorted_out[0]["label"])
        emotion_scores.append({d["label"]: d["score"] for d in sorted_out})

    df["emotion"] = emotions
    df["emotion_scores"] = emotion_scores

    print(f"Saving output to {output_path}")
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    run(args.input, args.output)
