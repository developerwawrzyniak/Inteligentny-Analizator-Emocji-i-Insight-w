import argparse
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def run(input_path, output_path, n_topics=5):
    print(f"Loading input file: {input_path}")
    df = pd.read_csv(input_path)

    if "clean_text" not in df.columns:
        raise ValueError("Input CSV must contain a 'clean_text' column.")

    texts = df["clean_text"].tolist()

    print("Vectorizing text data...")
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
    X = vectorizer.fit_transform(texts)

    print(f"Fitting LDA model with {n_topics} topics...")
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    print("Assigning topics to documents...")
    topic_ids = lda.transform(X).argmax(axis=1)
    df["topic_id"] = topic_ids

    # Pobranie głównych słów dla każdego tematu
    words = vectorizer.get_feature_names_out()
    topic_words = []
    for topic in lda.components_:
        top_words = [words[i] for i in topic.argsort()[-5:][::-1]]  # top 5 słów
        topic_words.append(", ".join(top_words))

    df["topic_summary"] = [topic_words[i] for i in topic_ids]

    print(f"Saving output to: {output_path}")
    df.to_csv(output_path, index=False)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Topic Modeling with LDA (scikit-learn)"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save output CSV file"
    )
    parser.add_argument(
        "--n_topics", type=int, default=5, help="Number of topics to extract"
    )
    args = parser.parse_args()

    run(args.input, args.output, args.n_topics)
