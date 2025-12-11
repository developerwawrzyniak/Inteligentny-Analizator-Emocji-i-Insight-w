"""Preprocessing tekstu: czyszczenie, detekcja języka, opcjonalne tłumaczenie"""

import pandas as pd
from langdetect import detect
import re

EMOJI_PATTERN = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = EMOJI_PATTERN.sub("", s)
    s = s.replace("\n", " ").strip()
    return s


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["clean_text"] = df["review_text"].astype(str).apply(clean_text)
    df["lang"] = df["clean_text"].apply(lambda t: detect(t) if t else "unknown")
    df.to_csv(args.output, index=False)
    print("Preprocessing done ->", args.output)
