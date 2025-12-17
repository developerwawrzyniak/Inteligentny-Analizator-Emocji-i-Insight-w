import pandas as pd


import os
import requests
import threading


def aggregate_topic_emotions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates emotions per topic.
    Returns one row per topic with dominant emotion and its score.
    """

    # 1. Średnia emocji per topic + emotion
    grouped = (
        df.groupby(["topic_id", "topic_summary", "dominant_emotion"])
        .agg(
            avg_emotion_score=("emotion_score", "mean"),
            documents_count=("clean_text", "count"),
        )
        .reset_index()
    )

    # 2. Wybierz dominującą emocję per topic
    idx = grouped.groupby("topic_id")["avg_emotion_score"].idxmax()
    topic_emotions = grouped.loc[idx].reset_index(drop=True)

    # 3. Czytelne nazwy kolumn
    topic_emotions = topic_emotions.rename(
        columns={"avg_emotion_score": "emotion_score"}
    )

    return topic_emotions[
        [
            "topic_id",
            "topic_summary",
            "dominant_emotion",
            "emotion_score",
            "documents_count",
        ]
    ]


# -------------------------
# AI INSIGHT HELPERS
# -------------------------
def _call_openai_chat(prompt: str, api_key: str, model: str = "gpt-4o-mini") -> str:
    """
    Minimal OpenAI chat call using the HTTP API (requests).
    Returns response text or raises an exception.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise business analyst that summarizes topic clusters for clients in a short, actionable way.",
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 200,
        "temperature": 0.2,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    # Extract text from choices
    return data["choices"][0]["message"]["content"].strip()


def generate_ai_insights(
    topic_df: pd.DataFrame,
    openai_api_key: str | None = None,
    model: str = "gpt-4o-mini",
) -> pd.DataFrame:
    """
    For each row in topic_df, generate an AI summary suitable for clients.
    If openai_api_key is provided (or present in environment), use the OpenAI chat API, otherwise fall back to a safe template-based summary.
    Adds/returns a new column `ai_insight`.
    """

    key = openai_api_key or os.getenv("OPENAI_API_KEY")

    def _fallback_summary(row):
        kws = row.get("topic_keywords") or row.get("topic_summary") or ""
        emo = row.get("dominant_emotion") or "N/A"
        score = row.get("emotion_score")
        emotion_str = (
            f" (emotion {emo} {score:.2f})" if pd.notna(score) and emo != "N/A" else ""
        )
        return f"Topic about {kws}.{emotion_str} {int(row.get('documents_count', 0))} documents. Suggested action: investigate top keywords and collect feedback."

    summaries = []
    recommendations = []
    for _, row in topic_df.iterrows():
        try:
            if key:
                prompt = (
                    f"Provide a concise (1-2 sentence) client-facing summary for the following topic.\n"
                    f"Keywords: {row.get('topic_keywords') or row.get('topic_summary')}\n"
                    f"Sample texts: {row.get('sample_texts') or ''}\n"
                    f"Dominant emotion: {row.get('dominant_emotion') or 'N/A'} with score {row.get('emotion_score') or 'N/A'}\n"
                    f"Number of documents: {int(row.get('documents_count', 0))}\n"
                    "Also include up to 3 short actionable recommendations (comma-separated). Keep it short and business-friendly."
                )
                ai_text = _call_openai_chat(prompt, key, model=model)
                # Small cleanup
                ai_text = " ".join(ai_text.split())
            else:
                ai_text = _fallback_summary(row)
        except Exception:
            ai_text = _fallback_summary(row)

        # Split AI text into summary + recommendations if possible
        if "recommend" in ai_text.lower():
            # naive split on 'recommend' token
            parts = (
                ai_text.split("Recommendations:")
                if "Recommendations:" in ai_text
                else ai_text.split("recommend")
            )
            summary = parts[0].strip()
            recs = parts[1].strip() if len(parts) > 1 else ""
        else:
            summary = ai_text
            recs = ""

        summaries.append(summary)
        recommendations.append(recs)

    topic_df = topic_df.copy()
    topic_df["ai_summary"] = summaries
    topic_df["ai_recommendations"] = recommendations
    return topic_df


# -------------------------
# ML-BASED INSIGHTS (LOCAL)
# -------------------------
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    _ST_MODEL = None
    _ST_MODEL_LOADING = False
    _ST_MODEL_LOCK = threading.Lock()

    def _init_st_model():
        global _ST_MODEL, _ST_MODEL_LOADING
        try:
            with _ST_MODEL_LOCK:
                _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        finally:
            _ST_MODEL_LOADING = False

    def preload_sentence_model():
        """Start loading the sentence-transformers model in a background thread."""
        global _ST_MODEL_LOADING
        if _ST_MODEL is not None:
            return
        if _ST_MODEL_LOADING:
            return
        _ST_MODEL_LOADING = True
        t = threading.Thread(target=_init_st_model, daemon=True)
        t.start()

    def is_sentence_model_loaded() -> bool:
        return _ST_MODEL is not None
except Exception:
    # sentence-transformers not available
    _ST_MODEL = None

    def preload_sentence_model():
        return

    def is_sentence_model_loaded() -> bool:
        return False


def _split_sentences(text: str) -> list:
    import re

    if not isinstance(text, str):
        return []
    # naive split on sentence enders
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    # filter very short parts
    parts = [p.strip() for p in parts if len(p.strip()) > 20]
    return parts


def generate_ml_insight_for_topic(
    texts: list,
    keywords: str = "",
    dominant_emotion: str = "N/A",
    emotion_score: float | None = None,
    n_sentences: int = 2,
) -> tuple[str, str]:
    """
    Generate an extractive, local ML-derived summary for a single topic using sentence-transformers.
    Returns a tuple (summary, recommendations) where summary is 1-2 concise sentences and recommendations is a short semicolon-separated string.
    """

    if not texts:
        return "No documents available for this topic.", "No recommendations."

    # collect sentences from texts
    sentences = []
    for t in texts:
        sentences.extend(_split_sentences(t))

    if not sentences:
        # fall back to short join
        summary = " ".join(texts[:2])
    else:
        # if model available use embeddings
        if _ST_MODEL is not None:
            try:
                embeds = _ST_MODEL.encode(sentences, convert_to_numpy=True)
                centroid = np.mean(embeds, axis=0, keepdims=True)
                sims = (embeds @ centroid.T).squeeze() / (
                    np.linalg.norm(embeds, axis=1) * np.linalg.norm(centroid)
                )
                top_idx = sims.argsort()[::-1][:n_sentences]
                top_sentences = [sentences[i] for i in top_idx]
                summary = " ".join(top_sentences)
            except Exception:
                summary = " ".join(sentences[:n_sentences])
        else:
            summary = " ".join(sentences[:n_sentences])

    # keep the summary concise
    summary = summary.strip()
    if len(summary) > 220:
        summary = summary[:217].rstrip() + "..."

    # Recommendations based on emotion and keywords
    recs = []
    emo = (dominant_emotion or "").lower()
    if emo == "joy":
        recs.append("Amplify positive feedback in marketing")
    elif emo in ("anger", "fear"):
        recs.append("Investigate root causes and collect detailed feedback")
    elif emo == "sadness":
        recs.append("Provide empathetic communication and support resources")
    elif emo == "surprise":
        recs.append("Explore surprising feedback and replicate successful elements")
    else:
        recs.append("Monitor sentiment and collect more feedback")

    kw = (keywords or "").lower()
    if "eating" in kw or "food" in kw:
        recs.append("Offer product tips or educational content about healthy eating")
    if "intelligence" in kw or "ai" in kw:
        recs.append("Produce clear explanatory content or demos for AI features")
    if "change" in kw or "new" in kw:
        recs.append("Communicate upcoming changes clearly to users")

    # deduplicate and keep top 3
    recs_unique = []
    for r in recs:
        if r not in recs_unique:
            recs_unique.append(r)
    recs = recs_unique[:3]

    rec_text = "; ".join(recs) if recs else "No immediate action suggested."

    return summary, rec_text


def generate_ml_insights(topic_df: pd.DataFrame, n_sentences: int = 2) -> pd.DataFrame:
    """
    Generate ML-based insights for each topic row and add `ai_insight` column (overwrites existing).
    """
    summaries = []
    recommendations = []
    for _, row in topic_df.iterrows():
        # Prefer full documents list if available (better summaries), else fall back to sample_texts
        texts = []
        if row.get("documents"):
            texts = row.get("documents")
        elif row.get("sample_texts"):
            # split sample_texts by ' | '
            texts = [
                s.strip() for s in str(row.get("sample_texts")).split("|") if s.strip()
            ]
        # if still empty, return small list using topic summary
        if not texts:
            texts = [str(row.get("topic_summary", ""))]
        summary, recs = generate_ml_insight_for_topic(
            texts,
            keywords=row.get("topic_keywords", ""),
            dominant_emotion=row.get("dominant_emotion", "N/A"),
            emotion_score=row.get("emotion_score"),
            n_sentences=n_sentences,
        )
        summaries.append(summary)
        recommendations.append(recs)
    td = topic_df.copy()
    td["ai_summary"] = summaries
    td["ai_recommendations"] = recommendations
    return td
