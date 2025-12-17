import pandas as pd
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def run(input_path: str, output_path: str, n_topics: int):
    print(f"Loading input file: {input_path}")
    df = pd.read_csv(input_path)

    if "clean_text" not in df.columns:
        raise ValueError("Input file must contain 'clean_text' column")

    texts = df["clean_text"].astype(str)

    print("Vectorizing text data...")
    vectorizer = CountVectorizer(stop_words="english", max_df=0.95, min_df=1)
    X = vectorizer.fit_transform(texts)

    print(f"Fitting LDA model with {n_topics} topics...")
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    topic_probs = lda.fit_transform(X)

    print("Assigning topics to documents...")
    df["topic_id"] = topic_probs.argmax(axis=1)

    print("Extracting topic keywords...")
    feature_names = vectorizer.get_feature_names_out()

    topic_keywords = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-4:-1]]
        topic_keywords[topic_idx] = ", ".join(top_words)

    df["topic_keywords"] = df["topic_id"].map(topic_keywords)

    output_df = df[["clean_text", "topic_id", "topic_keywords"]]

    print(f"Saving output to: {output_path}")
    output_df.to_csv(output_path, index=False)

    print("Topic modeling completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_topics", type=int, default=3)

    args = parser.parse_args()
    run(args.input, args.output, args.n_topics)
