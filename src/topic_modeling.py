import argparse
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# -------------------------
# TEXT CLEANING
# -------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # keep letters only
    text = re.sub(r"\s+", " ", text).strip()  # normalize spaces
    return text


# -------------------------
# MAIN PIPELINE
# -------------------------
def run(input_path, output_path, n_topics=5):
    print(f"Loading input file: {input_path}")
    df = pd.read_csv(input_path)

    # Detect text column
    if "clean_text" in df.columns:
        print("Using existing 'clean_text' column.")
        texts = df["clean_text"].astype(str).tolist()
    elif "review_text" in df.columns:
        print("Cleaning text from 'review_text' column...")
        df["clean_text"] = df["review_text"].astype(str).apply(clean_text)
        texts = df["clean_text"].tolist()
    else:
        raise ValueError(
            "Input CSV must contain either 'clean_text' or 'review_text' column."
        )

    if len(texts) < n_topics * 2:
        raise ValueError(
            f"Not enough documents ({len(texts)}) for {n_topics} topics. "
            f"Add more rows or reduce --n_topics."
        )

    # -------------------------
    # VECTORIZE
    # -------------------------
    print("Vectorizing text data...")
    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words="english")
    X = vectorizer.fit_transform(texts)

    # -------------------------
    # LDA
    # -------------------------
    print(f"Fitting LDA model with {n_topics} topics...")
    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=42, learning_method="batch"
    )
    lda.fit(X)

    # -------------------------
    # ASSIGN TOPICS
    # -------------------------
    print("Assigning topics to documents...")
    topic_ids = lda.transform(X).argmax(axis=1)
    df["topic_id"] = topic_ids

    # -------------------------
    # TOPIC LABELS
    # -------------------------
    words = vectorizer.get_feature_names_out()
    topic_labels = {}

    for topic_idx, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[-5:][::-1]]
        topic_labels[topic_idx] = ", ".join(top_words)

    # Assign keywords and a human-readable short label for each document
    df["topic_keywords"] = [topic_labels[i] for i in topic_ids]
    # Keep the detailed keywords as the topic summary and create a short human label
    df["topic_summary"] = df["topic_keywords"]
    df["topic_label"] = df["topic_keywords"].apply(label_topic)

    # -------------------------
    # SAVE
    # -------------------------
    print(f"Saving output to: {output_path}")
    df.to_csv(output_path, index=False)
    print("✅ Topic modeling completed successfully!")


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Topic Modeling using LDA (Python 3.14 compatible)"
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n_topics", type=int, default=5)

    args = parser.parse_args()
    run(args.input, args.output, args.n_topics)


def label_topic(keywords: str) -> str:
    if "eating" in keywords:
        return "Food & Lifestyle"
    if "intelligence" in keywords:
        return "AI & Technology"
    return "General Feedback"
