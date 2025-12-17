import pandas as pd


INSIGHT_TEMPLATES = {
    "joy": "Customers feel positive about {topic}",
    "anger": "Customers are frustrated with {topic}",
    "neutral": "Customers feel neutral about {topic}",
}


def generate_insight(topic: str, emotion: str) -> str:
    return INSIGHT_TEMPLATES.get(
        emotion, "Customers have mixed feelings about {topic}"
    ).format(topic=topic)


def run(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    total_reviews = len(df)

    grouped = df.groupby("topic_keywords")

    rows = []

    for topic, g in grouped:
        topic_count = len(g)
        percent_reviews = round((topic_count / total_reviews) * 100)

        dominant_emotion = g["emotion"].value_counts().idxmax()

        avg_intensity = round(
            g[g["emotion"] == dominant_emotion]["intensity"].mean(), 2
        )

        insight = generate_insight(topic, dominant_emotion)

        rows.append(
            {
                "topic": topic,
                "dominant_emotion": dominant_emotion,
                "avg_intensity": avg_intensity,
                "percent_reviews": percent_reviews,
                "example_insight": insight,
            }
        )

    result = pd.DataFrame(rows).sort_values(by="percent_reviews", ascending=False)

    result.to_csv(output_path, index=False)
    print(f"Saved topic emotion insights to {output_path}")


if __name__ == "__main__":
    run("data/with_emotions.csv", "data/topic_emotion_insights.csv")
