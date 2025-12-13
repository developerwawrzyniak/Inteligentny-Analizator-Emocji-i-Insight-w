import pandas as pd
from openai import OpenAI

client = OpenAI()


def generate_summary(csv_path):
    df = pd.read_csv(csv_path)

    sample = df[["topic_summary", "dominant_emotion", "review_text"]].head(50)

    prompt = f"""
Analyze the following customer feedback.

Data:
{sample.to_string(index=False)}

Tasks:
1. What customers like the most
2. Main complaints
3. Top 3 actionable improvements

Return bullet points.
"""

    response = client.chat.completions.create(
        model="gpt-4.1", messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    summary = generate_summary("data/emotion_reviews.csv")
    print("\n📊 AI SUMMARY:\n")
    print(summary)
