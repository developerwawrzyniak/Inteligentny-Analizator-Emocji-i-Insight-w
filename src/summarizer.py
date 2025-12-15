import pandas as pd

INSIGHT_TEMPLATES = {
    "anger": "Customers are frustrated by {topic}",
    "sadness": "Customers feel disappointed with {topic}",
    "joy": "Customers are satisfied with {topic}",
    "fear": "Customers are concerned about {topic}",
    "surprise": "Customers are surprised by {topic}",
    "neutral": "Customers perceive {topic} as informational rather than emotional",
}


def generate_insight(topic: str, emotion: str) -> str:
    template = INSIGHT_TEMPLATES.get(
        emotion, "Customers express mixed feelings about {topic}"
    )
    return template.format(topic=topic)


def run(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    df["example_insight"] = df.apply(
        lambda row: generate_insight(row["topic_summary"], row["dominant_emotion"]),
        axis=1,
    )

    df.to_csv(output_path, index=False)
    print(f"Saved summarized insights to {output_path}")


if __name__ == "__main__":
    run("data/topic_emotion_table.csv", "data/topic_emotion_insights.csv")
