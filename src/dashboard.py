import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from dash import dash_table

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("data/emotion_reviews.csv")

app = Dash(__name__)

# -------------------------
# LAYOUT
# -------------------------
app.layout = html.Div(
    [
        html.H1("Customer Insights Dashboard"),
        dcc.Dropdown(
            options=[
                {"label": t, "value": t}
                for t in sorted(df["topic_summary"].dropna().unique())
            ],
            placeholder="Filter by topic",
            id="topic-filter",
        ),
        dcc.Graph(id="emotion-bar"),
        dcc.Graph(id="topic-distribution"),
        html.H3("Customer Reviews"),
        dash_table.DataTable(
            id="table",
            columns=[
                {"name": "Review", "id": "review_text"},
                {"name": "Emotion", "id": "dominant_emotion"},
                {"name": "Topic", "id": "topic_summary"},
            ],
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "8px", "whiteSpace": "normal"},
        ),
    ]
)


# -------------------------
# CALLBACK
# -------------------------
@app.callback(
    Output("emotion-bar", "figure"),
    Output("topic-distribution", "figure"),
    Output("table", "data"),
    Input("topic-filter", "value"),
)
def update_dashboard(topic):
    dff = df if topic is None else df[df["topic_summary"] == topic]

    emotion_fig = px.histogram(dff, x="dominant_emotion", title="Dominant Emotions")

    topic_fig = px.pie(dff, names="topic_summary", title="Topic Distribution")

    return emotion_fig, topic_fig, dff.to_dict("records")


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
