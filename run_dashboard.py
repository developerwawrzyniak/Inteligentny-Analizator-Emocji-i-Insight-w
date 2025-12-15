from src.dashboard import app

if __name__ == "__main__":
    # Use modern Dash API
    app.run(host="0.0.0.0", port=8050, debug=True)
