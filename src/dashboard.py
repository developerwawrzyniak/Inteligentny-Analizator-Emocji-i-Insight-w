import pandas as pd
import dash
from dash import html, dcc
import plotly.express as px

# =========================
# Load data
# =========================
DATA_PATH = "data/topic_emotion_insights.csv"
df = pd.read_csv(DATA_PATH)

# =========================
# Formatting for table
# =========================
df["avg_intensity"] = df["avg_intensity"].round(2)
df["percent_reviews"] = df["percent_reviews"].astype(int)

# =========================
# Charts
# =========================

emotion_chart = px.bar(
    df,
    x="dominant_emotion",
    y="percent_reviews",
    color="dominant_emotion",
    title="Emotion Distribution by Topic (%)",
)

topic_chart = px.bar(
    df, x="topic", y="percent_reviews", title="Topic Importance (% of Reviews)"
)

# =========================
# Dash App
# =========================
app = dash.Dash(__name__)

app.layout = html.Div(
    style={"padding": "20px", "fontFamily": "Arial"},
    children=[
        html.H1("Customer Feedback – Topic & Emotion Insights"),
        html.Hr(),
        html.H2("Overview"),
        dcc.Graph(figure=emotion_chart),
        dcc.Graph(figure=topic_chart),
        html.Hr(),
        html.H2("Topic – Emotion Table"),
        html.Table(
            style={"width": "100%", "borderCollapse": "collapse", "marginTop": "20px"},
            children=[
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Topic"),
                            html.Th("Dominant Emotion"),
                            html.Th("Avg Intensity"),
                            html.Th("% Reviews"),
                            html.Th("Insight"),
                        ]
                    )
                ),
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td(row["topic"]),
                                html.Td(row["dominant_emotion"]),
                                html.Td(f"{row['avg_intensity']:.2f}"),
                                html.Td(f"{row['percent_reviews']}%"),
                                html.Td(row["example_insight"]),
                            ]
                        )
                        for _, row in df.iterrows()
                    ]
                ),
            ],
        ),
    ],
)

# =========================
# Run
# =========================
if __name__ == "__main__":
    app.run(debug=True)
