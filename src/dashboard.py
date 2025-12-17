import os
import pandas as pd
from dash import Dash, dcc, html, dash_table, Input, Output, State
import plotly.express as px
from analytics import (
    aggregate_topic_emotions,
    generate_ml_insights,
    generate_ml_insight_for_topic,
    is_sentence_model_loaded,
    preload_sentence_model,
    generate_ai_insights,
)
from pathlib import Path

# Data paths (project root / data)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
INPUT_CSV = DATA_DIR / "input.csv"
OUTPUT_CSV = DATA_DIR / "output.csv"

# Load data (require output.csv to compute topics)
if not OUTPUT_CSV.exists():
    raise FileNotFoundError(f"Missing {OUTPUT_CSV}. Run topic modeling first.")

df_out = pd.read_csv(OUTPUT_CSV)

# Aggregate topics: counts and percentages
# Group by topic_id only to avoid inconsistent 'topic_summary' values; pick the first summary/label per topic
# Group by topic_id and compute document counts; safely pull first values for optional columns
topic_group = (
    df_out.groupby("topic_id")
    .agg(documents_count=("clean_text", "count"))
    .reset_index()
)

# Add topic_summary if present
if "topic_summary" in df_out.columns:
    topic_summary = df_out.groupby("topic_id")["topic_summary"].first().reset_index()
    topic_group = topic_group.merge(topic_summary, on="topic_id", how="left")
else:
    topic_group["topic_summary"] = ""

# Add topic_label if present; otherwise fall back to topic_summary
if "topic_label" in df_out.columns:
    topic_label = df_out.groupby("topic_id")["topic_label"].first().reset_index()
    topic_group = topic_group.merge(topic_label, on="topic_id", how="left")
else:
    topic_group["topic_label"] = topic_group["topic_summary"].copy()

# Add topic_keywords if present; otherwise fall back to topic_summary
if "topic_keywords" in df_out.columns:
    topic_keywords = df_out.groupby("topic_id")["topic_keywords"].first().reset_index()
    topic_group = topic_group.merge(topic_keywords, on="topic_id", how="left")
else:
    topic_group["topic_keywords"] = topic_group["topic_summary"].copy()

total_docs = topic_group["documents_count"].sum()
if total_docs > 0:
    topic_group["percentage"] = (topic_group["documents_count"] / total_docs) * 100
else:
    topic_group["percentage"] = 0.0

# Add sample texts (up to 3 per topic) and the full documents per topic
samples = (
    df_out.groupby("topic_id")["clean_text"]
    .apply(lambda s: " | ".join(s.astype(str).head(3).tolist()))
    .reset_index(name="sample_texts")
)

# full documents list for better ML summaries
documents = (
    df_out.groupby("topic_id")["clean_text"]
    .apply(lambda s: s.astype(str).tolist())
    .reset_index(name="documents")
)

topic_df = topic_group.merge(samples, on="topic_id", how="left")
topic_df = topic_df.merge(documents, on="topic_id", how="left")

# If emotion columns exist, compute per-topic emotion summary using existing helper
if "dominant_emotion" in df_out.columns and "emotion_score" in df_out.columns:
    try:
        emotion_summary = aggregate_topic_emotions(df_out)[
            ["topic_id", "dominant_emotion", "emotion_score"]
        ]
        # Merge by topic_id only; avoid relying on documents_count (may differ depending on grouping)
        topic_df = topic_df.merge(emotion_summary, on=["topic_id"], how="left")
        topic_df["dominant_emotion"] = topic_df["dominant_emotion"].fillna("N/A")
    except Exception:
        # If aggregate fails, ignore and continue
        topic_df["dominant_emotion"] = "N/A"
        topic_df["emotion_score"] = None
else:
    topic_df["dominant_emotion"] = "N/A"
    topic_df["emotion_score"] = None

# Ensure topic_label exists (short human label) and keywords exist
if "topic_label" not in topic_df.columns:
    topic_df["topic_label"] = topic_df.get("topic_summary", "")
if "topic_keywords" not in topic_df.columns:
    topic_df["topic_keywords"] = topic_df.get("topic_summary", "")

# Sort by percentage desc
topic_df = topic_df.sort_values(by="percentage", ascending=False).reset_index(drop=True)

# Build dashboard
app = Dash(__name__)

# Callbacks for ML regeneration and model status


@app.callback(
    Output("topics-table", "data"),
    Output("regen-status", "children"),
    Input("regen-ml-btn", "n_clicks"),
    State("topic-select", "value"),
    State("topics-table", "data"),
)
def regenerate_ml(n_clicks, topic_id, table_data):
    """Regenerate a single topic's AI insight using local ML method and update the table data."""
    if not n_clicks:
        return table_data, ""
    if topic_id is None:
        return table_data, "No topic selected."

    df_table = pd.DataFrame(table_data)
    # find original texts for that topic from df_out
    texts = (
        df_out[df_out["topic_id"] == int(topic_id)]["clean_text"].astype(str).tolist()
    )
    if not texts:
        return table_data, f"No documents for topic {topic_id}."

    # fetch contextual fields
    row = df_table[df_table["topic_id"] == int(topic_id)].iloc[0]
    keywords = row.get("topic_keywords", "")
    emo = row.get("dominant_emotion", "N/A")
    score = row.get("emotion_score")

    # Ensure model is preloaded
    preload_sentence_model()

    try:
        new_summary, new_recs = generate_ml_insight_for_topic(
            texts, keywords=keywords, dominant_emotion=emo, emotion_score=score
        )
        df_table.loc[df_table["topic_id"] == int(topic_id), "ai_summary"] = new_summary
        df_table.loc[df_table["topic_id"] == int(topic_id), "ai_recommendations"] = (
            new_recs
        )
        status = f"Regenerated insight for topic {topic_id}."
    except Exception as e:
        status = f"Failed to generate insight: {e}"

    return df_table.to_dict("records"), status


@app.callback(
    Output("model-status", "children"), Input("model-check-interval", "n_intervals")
)
def update_model_status(n):
    return "Model loaded" if is_sentence_model_loaded() else "Model loading..."


# Prepare display columns for DataTable
# truncate samples for preview
topic_df["samples_preview"] = (
    topic_df.get("sample_texts", "")
    .fillna("")
    .apply(lambda s: s if len(s) <= 200 else s[:197] + "...")
)
# create a short preview of full documents (first up to 5)
if "documents" in topic_df.columns:
    topic_df["documents_preview"] = topic_df["documents"].apply(
        lambda docs: " | ".join(docs[:5]) if isinstance(docs, list) and docs else ""
    )
else:
    topic_df["documents_preview"] = ""

# friendly strings
topic_df["percentage_display"] = topic_df["percentage"].apply(lambda v: f"{v:.1f}%")
topic_df["emotion_score_display"] = topic_df["emotion_score"].apply(
    lambda v: f"{v:.2f}" if pd.notna(v) else "N/A"
)


# Generate AI-based insights (use local ML method by default; only call OpenAI if key present)
openai_key = os.getenv("OPENAI_API_KEY")

# Prefer local ML insights to avoid external API by default
# Start preloading model in background if local ML is used
preload_sentence_model()

try:
    if openai_key:
        from analytics import generate_ai_insights

        topic_df = generate_ai_insights(topic_df, openai_api_key=openai_key)
    else:
        topic_df = generate_ml_insights(topic_df)
except Exception:
    # On failure, ensure ai_summary/ai_recommendations exist with fallback text
    topic_df["ai_summary"] = topic_df.apply(
        lambda r: f"Topic {r.get('topic_label')}: {r.get('topic_keywords')}", axis=1
    )
    topic_df["ai_recommendations"] = ""

# Ensure keywords column exists
if "topic_keywords" not in topic_df.columns:
    topic_df["topic_keywords"] = ""

# Define DataTable columns (use human-friendly topic label, and show ML-generated insight)
table_columns = [
    {"name": "Topic ID", "id": "topic_id"},
    {"name": "Topic", "id": "topic_label"},
    {"name": "Keywords", "id": "topic_keywords"},
    {"name": "Share", "id": "percentage_display"},
    {"name": "Documents", "id": "documents_count", "type": "numeric"},
    {"name": "Top Documents", "id": "documents_preview"},
    {"name": "Dominant Emotion", "id": "dominant_emotion"},
    {"name": "Emotion Score", "id": "emotion_score_display"},
    {"name": "Samples", "id": "samples_preview"},
    {"name": "Summary", "id": "ai_summary"},
    {"name": "Recommendations", "id": "ai_recommendations"},
]

# Tooltips (show full keywords, summary and documents preview on hover)
tooltip_data = []
for row in topic_df.to_dict("records"):
    td = {}
    td["topic_keywords"] = {"value": row.get("topic_keywords", ""), "type": "text"}
    td["ai_summary"] = {"value": row.get("ai_summary", ""), "type": "text"}
    td["ai_recommendations"] = {
        "value": row.get("ai_recommendations", ""),
        "type": "text",
    }
    td["documents_preview"] = {
        "value": row.get("documents_preview", ""),
        "type": "text",
    }
    tooltip_data.append(td)

# Pie chart (dark theme)
pie_fig = px.pie(
    topic_df,
    values="percentage",
    names="topic_label",
    title="Topic Share",
    hover_data=["percentage", "documents_count", "topic_keywords"],
    template="plotly_dark",
)
pie_fig.update_layout(
    paper_bgcolor="#000",
    plot_bgcolor="#000",
    font_color="#fff",
    title={"font": {"size": 20, "color": "#fff"}},
)

# Controls for ML-based regeneration
topic_options = [
    {"label": f"{r.topic_label} ({r.topic_id})", "value": int(r.topic_id)}
    for r in topic_df[["topic_id", "topic_label"]].itertuples()
]

app.layout = html.Div(
    [
        html.H1("Customer Insights Dashboard"),
        dcc.Graph(figure=pie_fig, config={"displayModeBar": True}),
        html.Div(
            [
                html.Label("Select topic to regenerate ML insight:"),
                dcc.Dropdown(
                    id="topic-select",
                    options=topic_options,
                    value=topic_options[0]["value"] if topic_options else None,
                    clearable=False,
                    style={"width": "60%"},
                ),
                html.Button(
                    "Regenerate ML Insight",
                    id="regen-ml-btn",
                    n_clicks=0,
                    style={"marginLeft": "12px"},
                ),
                html.Span(
                    id="regen-status", style={"marginLeft": "12px", "color": "#ffd700"}
                ),
                html.Div(
                    [
                        html.Span("Model: "),
                        html.Span(
                            id="model-status",
                            style={"color": "#7CFC00", "marginLeft": "6px"},
                        ),
                    ],
                    style={"display": "inline-block", "marginLeft": "24px"},
                ),
                dcc.Interval(id="model-check-interval", interval=2000, n_intervals=0),
            ],
            style={"marginTop": "12px", "marginBottom": "12px"},
        ),
        html.H3("Topic → Emotion & Topic Insights"),
        html.Div(
            dash_table.DataTable(
                id="topics-table",
                columns=table_columns,
                data=topic_df.drop(columns=["documents"]).to_dict("records"),
                page_size=10,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto", "backgroundColor": "#000"},
                style_cell={
                    "textAlign": "left",
                    "whiteSpace": "normal",
                    "height": "auto",
                    "backgroundColor": "#000",
                    "color": "#fff",
                    "border": "1px solid #222",
                },
                style_header={
                    "fontWeight": "bold",
                    "backgroundColor": "#111",
                    "color": "#fff",
                },
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "#0a0a0a"},
                    {"if": {"row_index": "even"}, "backgroundColor": "#000000"},
                ],
                tooltip_data=tooltip_data,
                tooltip_duration=None,
            ),
            style={"width": "100%"},
        ),
    ],
    style={
        "backgroundColor": "#000",
        "color": "#fff",
        "minHeight": "100vh",
        "padding": "20px",
    },
)

if __name__ == "__main__":
    app.run(debug=True)
