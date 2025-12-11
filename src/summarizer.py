# Przykład z OpenAI-like client
from openai import OpenAI

client = OpenAI()


def generate_insights(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4.1", messages=[{"role": "user", "content": prompt}], max_tokens=500
    )
    return resp.choices[0].message["content"]
