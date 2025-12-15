import argparse
import shutil
from pathlib import Path
import sys

from src.topic_modeling import run as run_topic
from src.emotion_analysis import run as run_emotion
from src.dashboard import app

DATA_DIR = Path(__file__).resolve().parents[0] / "data"
DEFAULT_INPUT = DATA_DIR / "input.csv"
DEFAULT_OUTPUT = DATA_DIR / "output.csv"
DEFAULT_OUTPUT_EMO = DATA_DIR / "output_with_emotions.csv"


def main():
    parser = argparse.ArgumentParser(description="Prepare data and run dashboard")
    parser.add_argument(
        "--input", default=str(DEFAULT_INPUT), help="Path to input CSV (clean_text)"
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to output CSV to write topics",
    )
    parser.add_argument("--n_topics", type=int, default=5, help="Number of LDA topics")
    parser.add_argument(
        "--with-emotions",
        action="store_true",
        help="Run emotion analysis and use its output in the dashboard",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for Dash app")
    parser.add_argument("--port", type=int, default=8050, help="Port for Dash app")
    parser.add_argument("--debug", action="store_true", help="Run Dash in debug mode")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    # 1) Run topic modeling
    print("Running topic modeling...")
    try:
        run_topic(str(input_path), str(output_path), args.n_topics)
    except Exception as e:
        print("Topic modeling failed:", e)
        sys.exit(1)

    # 2) Optional: run emotion analysis and overwrite output with emotion-augmented file
    if args.with_emotions:
        emo_out = DEFAULT_OUTPUT_EMO
        print("Running emotion analysis...")
        try:
            run_emotion(str(output_path), str(emo_out))
            # Use emotion file as dashboard input
            shutil.copyfile(str(emo_out), str(output_path))
            print(f"Emotion-augmented data copied to {output_path}")
        except Exception as e:
            print("Emotion analysis failed:", e)
            # continue without emotions

    # 3) Launch dashboard
    print(f"Starting dashboard at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
