import pandas as pd


def build_topic_emotion_table(df):
    # Work on a copy so we don't mutate caller data
    df_local = df.copy()

    total_reviews = len(df_local)

    # Normalize possible emotion column names from different pipeline versions
    if "emotion_intensity" not in df_local.columns:
        if "emotion_score" in df_local.columns:
            df_local["emotion_intensity"] = df_local["emotion_score"]
        else:
            # No emotion info available — create placeholder NA column so groupby still works
            df_local["emotion_intensity"] = pd.NA

    # Ensure dominant_emotion exists
    if "dominant_emotion" not in df_local.columns:
        df_local["dominant_emotion"] = "unknown"

    grouped = (
        df_local.groupby(["topic_label", "dominant_emotion"])
        .agg(avg_intensity=("emotion_intensity", "mean"), count=("clean_text", "count"))
        .reset_index()
    )

    # % reviews per topic-emotion
    grouped["percentage"] = (grouped["count"] / total_reviews * 100).round(0)

    # Human-friendly display with '%' sign (no decimals)
    grouped["percentage_display"] = (
        grouped["percentage"].fillna(0).apply(lambda x: f"{x:.0f}%")
    )

    # wybieramy DOMINUJĄCĄ emocję per topic
    idx = grouped.groupby("topic_label")["count"].idxmax()
    final = grouped.loc[idx].copy()

    # Human-friendly Avg Intensity display with two decimals (use comma as decimal separator)
    final["avg_intensity_display"] = final["avg_intensity"].apply(
        lambda x: f"{x:.2f}".replace(".", ",") if pd.notna(x) else "N/A"
    )

    # Rule-based textual insight; handle missing emotion gracefully
    def _insight(r):
        if r["dominant_emotion"] in (None, "", "unknown"):
            return f"No emotion data available for {r['topic_label']}"
        return f"Customers feel {r['dominant_emotion']} about {r['topic_label']}"

    final["example_insight"] = final.apply(_insight, axis=1)

    # Drop raw numeric columns so only formatted display columns are returned
    final = final.drop(columns=["avg_intensity", "percentage"], errors=True)

    return final.rename(
        columns={
            "topic_label": "Topic",
            "dominant_emotion": "Dominant Emotion",
            "avg_intensity_display": "Avg Intensity",
            "percentage_display": "Reviews %",
            "example_insight": "Example Insight",
        }
    )
