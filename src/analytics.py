import pandas as pd


def build_topic_emotion_table(df):
    """
    Tworzy tabelę: topic_summary × dominant_emotion z:
      - avg_intensity: średnia score emocji
      - reviews_count: liczba recenzji w tym topic/emocja
      - percentage: % recenzji danej emocji w danym topic
    """
    # Najpierw wybieramy tylko potrzebne kolumny
    df = df[["topic_summary", "dominant_emotion", "emotion_score"]].copy()

    # Grupowanie po topic i emocji
    grouped = (
        df.groupby(["topic_summary", "dominant_emotion"])
        .agg(
            avg_intensity=("emotion_score", "mean"),
            reviews_count=("dominant_emotion", "count"),
        )
        .reset_index()
    )

    # Dodanie procentu
    total_per_topic = grouped.groupby("topic_summary")["reviews_count"].transform("sum")
    grouped["percentage"] = (grouped["reviews_count"] / total_per_topic * 100).round(2)

    # Sortowanie po topic i procent
    grouped = grouped.sort_values(
        ["topic_summary", "percentage"], ascending=[True, False]
    )

    return grouped
