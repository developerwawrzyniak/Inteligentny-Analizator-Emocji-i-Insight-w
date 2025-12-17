import pandas as pd
from typing import List, Tuple, Optional
import threading

# Lightweight analytics helpers used by the dashboard. These implementations are
# intentionally simple and deterministic so the dashboard can function even if
# heavy ML dependencies are not available.


def build_topic_emotion_table(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible helper kept for scripts that expect a topic/emotion
    table aggregated by topic label.
    """
    total_reviews = len(df)

    grouped = (
        df.groupby(["topic_label", "dominant_emotion"])
        .agg(
            avg_intensity=("emotion_score", "mean"),
            reviews_count=("dominant_emotion", "count"),
        )
        .reset_index()
    )

    grouped["percentage"] = (
        grouped.groupby("topic_label")["reviews_count"]
        .transform(lambda x: x / x.sum() * 100)
        .round(2)
    )

    dominant = (
        grouped.sort_values("reviews_count", ascending=False)
        .groupby("topic_label")
        .first()
        .reset_index()
    )

    def insight(row):
        e = row["dominant_emotion"]
        t = row["topic_label"]
        if e == "anger":
            return f"Customers are frustrated with {t}"
        if e == "joy":
            return f"Customers are satisfied with {t}"
        if e == "sadness":
            return f"Customers feel disappointed about {t}"
        if e == "fear":
            return f"Customers feel uncertain about {t}"
        if e == "surprise":
            return f"Customers are surprised by {t}"
        return f"Customers feel neutral about {t}"

    dominant["example_insight"] = dominant.apply(insight, axis=1)

    return dominant.rename(
        columns={
            "topic_label": "Topic",
            "dominant_emotion": "Dominant Emotion",
            "avg_intensity": "Avg Intensity",
            "percentage": "% Reviews",
        }
    )


# ------------------------
# Per-topic emotion aggregation (used by dashboard)
# ------------------------


def aggregate_topic_emotions(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate dominant emotion and average emotion score per topic_id.

    Returns a dataframe with columns: topic_id, dominant_emotion, emotion_score
    """
    if df.empty:
        return pd.DataFrame(columns=["topic_id", "dominant_emotion", "emotion_score"])

    agg = (
        df.groupby("topic_id")
        .agg(
            dominant_emotion=(
                "dominant_emotion",
                lambda s: s.mode().iat[0] if not s.mode().empty else "N/A",
            ),
            emotion_score=("emotion_score", "mean"),
        )
        .reset_index()
    )
    return agg


# ------------------------
# Simple local ML summarizer helpers (fallback deterministic methods)
# ------------------------

# Model placeholder and loading state
_sentence_model = None
_model_lock = threading.Lock()
_model_loaded = False


def preload_sentence_model():
    """Attempt to preload a local sentence-transformers model in background.
    If sentence-transformers is not available, simply mark model as not loaded.
    """
    global _sentence_model, _model_loaded

    def _load():
        global _sentence_model, _model_loaded
        try:
            from sentence_transformers import SentenceTransformer

            _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            _model_loaded = True
        except Exception:
            # silently fail - fallback heuristics will be used instead
            _sentence_model = None
            _model_loaded = False

    # spawn background loader if not already loaded
    if not _model_loaded and _sentence_model is None:
        t = threading.Thread(target=_load, daemon=True)
        t.start()


def is_sentence_model_loaded() -> bool:
    return _model_loaded


def _short_extractive_summary(texts: List[str], max_sentences: int = 2) -> str:
    """Very small extractive summarizer: pick the longest sentences across docs
    up to max_sentences. This works deterministically without heavy deps.
    """
    if not texts:
        return ""
    # split by period, question mark, exclamation (simple)
    candidates = []
    for t in texts:
        for part in [
            p.strip() for p in t.replace("!", ".").replace("?", ".").split(".")
        ]:
            if part:
                candidates.append(part)
    # sort by length desc
    candidates = sorted(candidates, key=lambda s: len(s), reverse=True)
    return ". ".join(candidates[:max_sentences])


def generate_ml_insight_for_topic(
    texts: List[str],
    keywords: Optional[str] = None,
    dominant_emotion: Optional[str] = None,
    emotion_score: Optional[float] = None,
) -> Tuple[str, str]:
    """Generate (summary, recommendations) for a given topic using simple
    extractive rules and a short heuristic-based recommendation.
    """
    # build summary from texts (prefer model if loaded)
    summary = _short_extractive_summary(texts, max_sentences=2)
    if not summary and keywords:
        summary = f"Topic keywords: {keywords}"

    # recommendation heuristics
    emo = (dominant_emotion or "neutral").lower()
    if emo == "anger":
        rec = "Investigate root causes and prioritize fixes; consider proactive communications."
    elif emo == "sadness":
        rec = "Address disappointment via policy changes or improved customer support."
    elif emo == "joy":
        rec = "Amplify what's working and showcase positive experiences to attract more users."
    elif emo == "fear":
        rec = "Improve clarity of communications and provide reassurance (docs, FAQs)."
    elif emo == "surprise":
        rec = "Validate if surprises are positive; document unexpected behaviors and iterate."
    else:
        rec = "Monitor trends and collect more feedback to identify clear actions."

    # small tidy-up
    if summary and not summary.endswith("."):
        summary = summary.strip() + "."

    return summary, rec


def generate_ml_insights(topic_df: pd.DataFrame) -> pd.DataFrame:
    """Populate `ai_summary` and `ai_recommendations` columns for a topic dataframe.

    Expects a `documents` column (list[str]) or `sample_texts` as fallback.
    """
    summaries = []
    recommendations = []

    for _, row in topic_df.iterrows():
        texts = row.get("documents")
        if not texts or not isinstance(texts, list):
            # fallback to sample_texts split by separator
            st = row.get("sample_texts", "") or ""
            texts = [s.strip() for s in st.split("|") if s.strip()]

        s, r = generate_ml_insight_for_topic(
            texts,
            keywords=row.get("topic_keywords"),
            dominant_emotion=row.get("dominant_emotion"),
            emotion_score=row.get("emotion_score"),
        )
        summaries.append(s)
        recommendations.append(r)

    topic_df = topic_df.copy()
    topic_df["ai_summary"] = summaries
    topic_df["ai_recommendations"] = recommendations
    return topic_df


def generate_ai_insights(
    topic_df: pd.DataFrame, openai_api_key: Optional[str] = None
) -> pd.DataFrame:
    """Placeholder: when no API key is present this delegates to local ML insights.
    If an API key is provided we still fall back to local summarizer for now.
    """
    # For now we don't call OpenAI; use local ML path for deterministic behavior.
    return generate_ml_insights(topic_df)
