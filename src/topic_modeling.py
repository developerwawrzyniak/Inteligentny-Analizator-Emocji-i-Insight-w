import pandas as pd
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Keyword-to-category mapping for better topic names
CATEGORY_KEYWORDS = {
    "Performance & Speed": ["fast", "slow", "performance", "speed", "quick", "responsive", "lag", "smooth"],
    "Battery & Power": ["battery", "charge", "charging", "power", "drains", "lasts", "dies"],
    "Build Quality": ["quality", "build", "cheap", "premium", "durable", "sturdy", "flimsy", "solid"],
    "Price & Value": ["price", "expensive", "cheap", "value", "worth", "cost", "money", "affordable"],
    "Sound & Audio": ["sound", "audio", "noise", "loud", "quiet", "bass", "volume", "speaker"],
    "Display & Screen": ["display", "screen", "resolution", "bright", "colors", "touch"],
    "Comfort & Design": ["comfortable", "comfort", "wear", "lightweight", "heavy", "design", "stylish", "fit"],
    "Reliability & Durability": ["reliable", "stopped", "broke", "working", "failed", "lasted", "durable"],
    "Features & Functions": ["features", "functions", "works", "useful", "easy", "app", "settings"],
    "Customer Experience": ["disappointed", "amazing", "excellent", "terrible", "love", "hate", "recommend"],
    "Camera & Photos": ["camera", "photo", "photos", "picture", "pictures", "video"],
    "Heating & Temperature": ["heat", "heats", "hot", "warm", "cool", "temperature", "overheating"],
}


def get_topic_category(top_words):
    """Map top words to a meaningful category name."""
    top_words_lower = [w.lower() for w in top_words]

    # Score each category based on keyword matches
    category_scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for word in top_words_lower if any(kw in word or word in kw for kw in keywords))
        if score > 0:
            category_scores[category] = score

    # Return best matching category or create name from top words
    if category_scores:
        best_category = max(category_scores, key=category_scores.get)
        return best_category

    # Fallback: create readable name from top 2 meaningful words
    meaningful_words = [w for w in top_words[:2] if len(w) > 3]
    if meaningful_words:
        return " & ".join(word.title() for word in meaningful_words[:2])

    return f"Topic {top_words[0].title()}"


def generate_topic_names(lda_model, vectorizer, n_words=8):
    """Generate human-readable topic names from top words."""
    feature_names = vectorizer.get_feature_names_out()
    topic_names = {}
    used_categories = set()

    for topic_idx, topic in enumerate(lda_model.components_):
        # Get more top words for better category matching
        top_word_indices = topic.argsort()[:-n_words - 1:-1]
        top_words = [feature_names[i] for i in top_word_indices]

        # Get category name
        category = get_topic_category(top_words)

        # Ensure unique names by adding suffix if needed
        original_category = category
        suffix = 1
        while category in used_categories:
            suffix += 1
            category = f"{original_category} ({suffix})"

        used_categories.add(category)
        topic_names[topic_idx] = category

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

    # Vectorize text with better settings
    vectorizer = CountVectorizer(
        stop_words="english",
        max_df=0.9,
        min_df=1,
        ngram_range=(1, 2),  # Include bigrams for better context
    )
    X = vectorizer.fit_transform(texts)

    # Fit LDA model
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=20,
        learning_method="online",
    )
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
