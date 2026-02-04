import pandas as pd
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def generate_topic_names(lda_model, vectorizer, n_words=3):
    """Generate human-readable topic names from top words."""
    feature_names = vectorizer.get_feature_names_out()
    topic_names = {}

    for topic_idx, topic in enumerate(lda_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
        topic_names[topic_idx] = " & ".join(word.title() for word in top_words)

    return topic_names


def process_topics(df: pd.DataFrame, n_topics: int = 3) -> pd.DataFrame:
    """Process a DataFrame and add topic assignments.

    Args:
        df: DataFrame with 'clean_text' column
        n_topics: Number of topics to extract

    Returns:
        DataFrame with added 'topic_id' and 'topic' columns
    """
    if "clean_text" not in df.columns:
        raise ValueError("DataFrame must contain 'clean_text' column")

    texts = df["clean_text"].astype(str)

    # Vectorize text
    vectorizer = CountVectorizer(stop_words="english", max_df=0.95, min_df=1)
    X = vectorizer.fit_transform(texts)

    # Fit LDA model
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    topic_probs = lda.fit_transform(X)

    # Generate topic names from model
    topic_names = generate_topic_names(lda, vectorizer)

    # Assign topics
    df = df.copy()
    df["topic_id"] = topic_probs.argmax(axis=1)
    df["topic"] = df["topic_id"].map(topic_names)

    return df


def run(input_path: str, output_path: str, n_topics: int):
    """Run topic modeling from file to file (CLI interface)."""
    print(f"Loading input file: {input_path}")
    df = pd.read_csv(input_path)

    df = process_topics(df, n_topics)

    output_df = df[["clean_text", "topic_id", "topic"]]

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
