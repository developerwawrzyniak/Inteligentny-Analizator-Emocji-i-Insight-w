import io
import base64

import pandas as pd
import dash
from dash import html, dcc, dash_table, callback_context
from dash.dependencies import Input, Output, State
import plotly.express as px

from src.topic_modeling import process_topics
from src.emotion_analysis import process_emotions

# =========================
# Dash App
# =========================
app = dash.Dash(__name__)

app.layout = html.Div(
    style={
        "padding": "20px",
        "fontFamily": "Arial",
        "backgroundColor": "#1a1a1a",
        "color": "white",
        "minHeight": "100vh",
    },
    children=[
        html.H1("Customer Feedback - Topic & Emotion Analyzer"),
        html.Hr(),

        # File Upload Section
        html.Div([
            html.H3("Upload your data"),
            html.P("Upload a CSV file with a 'clean_text' or 'text' column containing comments to analyze."),
            dcc.Upload(
                id="upload-data",
                children=html.Div([
                    "Drag and Drop or ",
                    html.A("Select a CSV File", style={"color": "#4dabf7", "cursor": "pointer"})
                ]),
                style={
                    "width": "100%",
                    "height": "80px",
                    "lineHeight": "80px",
                    "borderWidth": "2px",
                    "borderStyle": "dashed",
                    "borderRadius": "10px",
                    "borderColor": "#4dabf7",
                    "textAlign": "center",
                    "marginBottom": "20px",
                    "backgroundColor": "#2a2a2a",
                },
                multiple=False,
            ),
            html.Div([
                html.Label("Number of topics: ", style={"marginRight": "10px"}),
                dcc.Input(
                    id="n-topics",
                    type="number",
                    value=3,
                    min=2,
                    max=10,
                    style={"width": "60px", "marginRight": "20px"}
                ),
                html.Button(
                    "Analyze",
                    id="analyze-button",
                    n_clicks=0,
                    style={
                        "backgroundColor": "#4dabf7",
                        "color": "white",
                        "border": "none",
                        "padding": "10px 20px",
                        "borderRadius": "5px",
                        "cursor": "pointer",
                    }
                ),
            ], style={"marginBottom": "20px"}),
        ]),

        # Status message
        html.Div(id="status-message", style={"marginBottom": "20px", "color": "#ffd43b"}),

        # Results Section (hidden until data is processed)
        html.Div(id="results-section", style={"display": "none"}, children=[
            html.Hr(),
            html.H2("Analysis Results"),

            # Charts
            html.Div([
                html.Div([
                    dcc.Graph(id="emotion-chart"),
                ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),
                html.Div([
                    dcc.Graph(id="topic-chart"),
                ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),
            ]),

            # Summary Table
            html.H3("Topic-Emotion Summary"),
            html.Div(id="summary-table"),

            # Detailed Data Table
            html.H3("Detailed Results", style={"marginTop": "30px"}),
            html.Div(id="data-table"),
        ]),

        # Store for processed data
        dcc.Store(id="processed-data"),
    ],
)


def parse_csv(contents, filename):
    """Parse uploaded CSV file."""
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    try:
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    except Exception as e:
        return None, f"Error reading file: {str(e)}"

    # Find text column
    text_col = None
    for col in ["clean_text", "text", "comment", "review", "content"]:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        return None, "CSV must contain a text column (clean_text, text, comment, review, or content)"

    # Rename to clean_text for consistency
    if text_col != "clean_text":
        df["clean_text"] = df[text_col]

    return df, None


def aggregate_results(df):
    """Aggregate results by topic and emotion."""
    total_reviews = len(df)

    rows = []
    for topic in df["topic"].unique():
        topic_df = df[df["topic"] == topic]
        topic_count = len(topic_df)
        percent_reviews = round((topic_count / total_reviews) * 100)

        # Get dominant emotion
        dominant_emotion = topic_df["emotion"].value_counts().idxmax()

        # Get average intensity for dominant emotion
        emotion_df = topic_df[topic_df["emotion"] == dominant_emotion]
        avg_intensity = round(emotion_df["intensity"].mean(), 2)

        # Generate insight
        if dominant_emotion == "joy":
            insight = f"Customers feel positive about {topic}"
        elif dominant_emotion == "anger":
            insight = f"Customers are frustrated with {topic}"
        else:
            insight = f"Customers have neutral feelings about {topic}"

        rows.append({
            "topic": topic,
            "dominant_emotion": dominant_emotion,
            "avg_intensity": avg_intensity,
            "percent_reviews": percent_reviews,
            "insight": insight,
        })

    return pd.DataFrame(rows).sort_values(by="percent_reviews", ascending=False)


@app.callback(
    [
        Output("processed-data", "data"),
        Output("status-message", "children"),
        Output("results-section", "style"),
        Output("emotion-chart", "figure"),
        Output("topic-chart", "figure"),
        Output("summary-table", "children"),
        Output("data-table", "children"),
    ],
    [Input("analyze-button", "n_clicks")],
    [
        State("upload-data", "contents"),
        State("upload-data", "filename"),
        State("n-topics", "value"),
    ],
    prevent_initial_call=True,
)
def analyze_data(n_clicks, contents, filename, n_topics):
    """Process uploaded data and return analysis results."""

    empty_fig = px.bar(title="No data")
    empty_fig.update_layout(template="plotly_dark")
    hidden_style = {"display": "none"}

    if contents is None:
        return None, "Please upload a CSV file first.", hidden_style, empty_fig, empty_fig, "", ""

    # Parse CSV
    df, error = parse_csv(contents, filename)
    if error:
        return None, error, hidden_style, empty_fig, empty_fig, "", ""

    if len(df) == 0:
        return None, "The uploaded file is empty.", hidden_style, empty_fig, empty_fig, "", ""

    try:
        # Process topics
        df = process_topics(df, n_topics or 3)

        # Process emotions
        df = process_emotions(df)

        # Aggregate results
        summary_df = aggregate_results(df)

    except Exception as e:
        return None, f"Error during analysis: {str(e)}", hidden_style, empty_fig, empty_fig, "", ""

    # Create charts
    emotion_chart = px.bar(
        summary_df,
        x="dominant_emotion",
        y="percent_reviews",
        color="dominant_emotion",
        title="Emotion Distribution (%)",
        template="plotly_dark",
    )

    topic_chart = px.bar(
        summary_df,
        x="topic",
        y="percent_reviews",
        color="topic",
        title="Topic Distribution (%)",
        template="plotly_dark",
    )

    # Create summary table
    summary_table = dash_table.DataTable(
        data=summary_df.to_dict("records"),
        columns=[
            {"name": "Topic", "id": "topic"},
            {"name": "Dominant Emotion", "id": "dominant_emotion"},
            {"name": "Avg Intensity", "id": "avg_intensity"},
            {"name": "% Reviews", "id": "percent_reviews"},
            {"name": "Insight", "id": "insight"},
        ],
        style_header={
            "backgroundColor": "#2a2a2a",
            "color": "white",
            "fontWeight": "bold",
        },
        style_cell={
            "backgroundColor": "#1a1a1a",
            "color": "white",
            "textAlign": "left",
            "padding": "10px",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#252525"},
        ],
    )

    # Create detailed data table
    detail_cols = ["clean_text", "topic", "emotion", "intensity"]
    data_table = dash_table.DataTable(
        data=df[detail_cols].to_dict("records"),
        columns=[{"name": col.replace("_", " ").title(), "id": col} for col in detail_cols],
        style_header={
            "backgroundColor": "#2a2a2a",
            "color": "white",
            "fontWeight": "bold",
        },
        style_cell={
            "backgroundColor": "#1a1a1a",
            "color": "white",
            "textAlign": "left",
            "padding": "10px",
            "maxWidth": "400px",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#252525"},
        ],
        page_size=10,
    )

    visible_style = {"display": "block"}
    status = f"Successfully analyzed {len(df)} comments from '{filename}'!"

    return df.to_json(), status, visible_style, emotion_chart, topic_chart, summary_table, data_table


# =========================
# Run
# =========================
if __name__ == "__main__":
    app.run(debug=True)
