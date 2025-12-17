import pandas as pd
from dash import Dash, html
from dash import dash_table
from analytics import build_topic_emotion_table

df = pd.read_csv("data/output_with_emotions.csv")
table_df = build_topic_emotion_table(df)

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Topic → Emotion Insights"),
        dash_table.DataTable(
            data=table_df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in table_df.columns],
            style_table={"width": "90%"},
            style_cell={"textAlign": "left"},
            page_size=10,
        ),
    ]
)

if __name__ == "__main__":
    app.run(debug=True)
