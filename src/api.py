from fastapi import FastAPI
import pandas as pd

app = FastAPI()


@app.get("/insights")
def get_insights():
    # zwróć wcześniej wygenerowane insights z pliku/DB
    return {"status": "ok", "insights": "..."}
