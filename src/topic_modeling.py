import argparse
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def run(input_path, output_path, n_topics=3):
    print(f"Loading input file: {input_path}")
    df = pd.read_csv(input_path)

    texts = df["clean_text"].astype(str).tolist()

    vectorizer = CountVectorizer(stop_words="english", max_df=0.9, min_df=1)
    X = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    topic_ids = lda.transform(X).argmax(axis=1)
    df["topic_id"] = topic_ids

    words = vectorizer.get_feature_names_out()

    def topic_label(topic_weights, top_n=3):
        top_words = sorted([words[i] for i in topic_weights.argsort()[-top_n:]])
        return ", ".join(top_words)

    labels = {i: topic_label(topic) for i, topic in enumerate(lda.components_)}

    df["topic_label"] = df["topic_id"].map(labels)

    df[["clean_text", "topic_id", "topic_label"]].to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_topics", type=int, default=3)
    args = parser.parse_args()

    run(args.input, args.output, args.n_topics)
