import pandas as pd
from dash import Dash, dcc, html, Input, Output
from dash import dash_table
import plotly.express as px

from analytics import build_topic_emotion_table


# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("data/output_with_emotions.csv")
topic_emotion_df = build_topic_emotion_table(df)

app = Dash(__name__)


# -------------------------
# LAYOUT
# -------------------------
app.layout = html.Div(
    [
        html.H1("Customer Insights Dashboard"),
        dcc.Dropdown(
            id="topic-filter",
            options=[
                {"label": t, "value": t}
                for t in sorted(df["topic_summary"].dropna().unique())
            ],
            placeholder="Filter by topic",
            clearable=True,
        ),
        dcc.Graph(id="emotion-bar"),
        dcc.Graph(id="topic-distribution"),
        html.H3("Topic × Emotion Breakdown"),
        dash_table.DataTable(
            id="topic-emotion-table",
            columns=[
                {"name": "Topic", "id": "topic_summary"},
                {"name": "Emotion", "id": "dominant_emotion"},
                {"name": "Avg Intensity", "id": "avg_intensity"},
                {"name": "% Reviews", "id": "percentage"},
            ],
            page_size=8,
            style_cell={"textAlign": "left"},
            sort_action="native",
        ),
    ],
    style={"width": "85%", "margin": "auto"},
)


# -------------------------
# CALLBACK
# -------------------------
@app.callback(
    Output("emotion-bar", "figure"),
    Output("topic-distribution", "figure"),
    Output("topic-emotion-table", "data"),
    Input("topic-filter", "value"),
)
def update_dashboard(selected_topic):
    if selected_topic:
        dff = topic_emotion_df[topic_emotion_df["topic_summary"] == selected_topic]
        raw = df[df["topic_summary"] == selected_topic]
    else:
        dff = topic_emotion_df
        raw = df

    # Emotion bar
    emotion_fig = px.bar(
        dff,
        x="dominant_emotion",
        y="percentage",
        color="dominant_emotion",
        title="Emotion Distribution (%)",
    )

    # Topic distribution
    topic_fig = px.pie(
        raw,
        names="topic_summary",
        title="Topic Share",
    )

    return (
        emotion_fig,
        topic_fig,
        dff.sort_values("percentage", ascending=False).to_dict("records"),
    )


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
