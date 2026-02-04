import argparse
from src.dashboard import app


def main():
    parser = argparse.ArgumentParser(description="Run the Emotion & Topic Analyzer Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host for Dash app")
    parser.add_argument("--port", type=int, default=8050, help="Port for Dash app")
    parser.add_argument("--debug", action="store_true", help="Run Dash in debug mode")

    args = parser.parse_args()

    print(f"Starting dashboard at http://{args.host}:{args.port}")
    print("Upload a CSV file with text data to analyze emotions and topics.")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
