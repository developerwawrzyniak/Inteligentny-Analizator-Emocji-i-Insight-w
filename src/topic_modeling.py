import argparse
import pandas as pd
from bertopic import BERTopic
from tqdm import tqdm
import umap.umap_ as umap


def run(input_path, output_path):
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    if "clean_text" not in df.columns:
        raise ValueError("Input CSV must contain a 'clean_text' column.")

    print("Creating BERTopic model...")
    topic_model = BERTopic(language="english", calculate_probabilities=True)

    print("Fitting topics...")
    topics, probs = topic_model.fit_transform(df["clean_text"].tolist())

    df["topic_id"] = topics
    topic_names = [topic_model.get_topic(i) for i in topics]

    # Tworzymy czytelne nazwy tematów
    topic_labels = []
    for i in topics:
        t = topic_model.get_topic(i)
        if t:
            words = ", ".join([w[0] for w in t[:5]])
            topic_labels.append(words)
        else:
            topic_labels.append("Other")
    df["topic_name"] = topic_labels

    print(f"Saving output to {output_path}")
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    run(args.input, args.output)
