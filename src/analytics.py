import pandas as pd


def build_topic_emotion_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates emotions per topic in a human-readable way
    """

    grouped = (
        df.groupby(["topic_summary", "dominant_emotion"])
        .agg(
            avg_intensity=("emotion_score", "mean"),
            reviews_count=("clean_text", "count"),
        )
        .reset_index()
    )

    total_per_topic = grouped.groupby("topic_summary")["reviews_count"].transform("sum")
    grouped["percentage"] = (grouped["reviews_count"] / total_per_topic * 100).round(2)

    return grouped.sort_values(["topic_summary", "percentage"], ascending=[True, False])
