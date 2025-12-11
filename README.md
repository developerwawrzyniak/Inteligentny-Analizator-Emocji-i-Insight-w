# Inteligentny Analizator Emocji i Insightów z Opinii Klientów

Gotowy do wdrożenia README oraz szkielet kodu dla projektu **customer-emotions-ai**. Znajdziesz tu instrukcje uruchomienia, opis architektury oraz minimalne implementacje plików, które pozwolą szybko wystartować i rozwijać projekt.

---

## Spis treści

1. Opis projektu
2. Funkcjonalności
3. Wymagania
4. Szybki start
5. Struktura repozytorium
6. Opis modułów
7. Przykładowe komendy
8. Rozszerzenia i deployment
9. Licencja

---

## 1. Opis projektu

Projekt ma na celu zbudowanie end‑to‑end systemu do analizy opinii klientów: zbieranie danych (CSV/API/web), preprocessing, klasyfikacja sentymentu i emocji, modelowanie tematów, generowanie insightów przez LLM, interaktywny dashboard oraz automatyczna generacja raportu PDF.

Cel: projekt portfolio pokazujący umiejętności Data Scientist + AI Engineer.

---

## 2. Funkcjonalności

* Import recenzji (CSV, API, scraper)
* Czyszczenie i normalizacja tekstu
* Wykrywanie języka i opcjonalne tłumaczenie
* Analiza sentymentu i emocji (transformery)
* Topic modeling (BERTopic / LDA fallback)
* Generacja insightów z LLM
* Interaktywny dashboard (Dash + Plotly)
* API (FastAPI) do zwracania insightów i filtrów
* Eksport raportu PDF

---

## 3. Wymagania

* Python 3.10+
* Zależności w `requirements.txt` (patrz sekcja poniżej)
* GPU przydatne do trenowania / inferencji transformera
* Klucz do API (opcjonalnie): OpenAI / inny provider LLM

Przykładowe zależności (plik `requirements.txt` dostarczony w repo):

```
pandas
numpy
scikit-learn
transformers
torch
langdetect
googletrans==4.0.0-rc1
bertopic
sentence-transformers
umap-learn
hdbscan
dash
plotly
fastapi
uvicorn
reportlab
beautifulsoup4
requests
python-multipart
pytest
```

---

## 4. Szybki start

1. Sklonuj repo:

   ```bash
   git clone <repo-url>
   cd customer-emotions-ai
   ```
2. Stwórz i aktywuj virtualenv:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # linux/mac
   .venv\Scripts\activate     # windows
   ```
3. Zainstaluj zależności:

   ```bash
   pip install -r requirements.txt
   ```
4. Uruchom preprocessing i analizę przykładowych danych (notebooki lub skrypty w `src/`):

   ```bash
   python src/preprocessing.py --input data/raw_reviews.csv --output data/processed_reviews.csv
   ```
5. Uruchom dashboard:

   ```bash
   python run_dashboard.py
   ```

---

## 5. Struktura repozytorium

```
customer-emotions-ai/
├── data/
│   ├── raw_reviews.csv
│   ├── processed_reviews.csv
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_sentiment_model.ipynb
│   ├── 03_topic_modeling.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── sentiment_analysis.py
│   ├── emotion_analysis.py
│   ├── topic_modeling.py
│   ├── summarizer.py
│   ├── dashboard.py
│   ├── api.py
│
├── report/
│   ├── example_report.pdf
│   ├── generate_report.py
│
├── requirements.txt
├── README.md
└── run_dashboard.py
```

---

## 6. Opis modułów i minimalne skeletony

> Pliki poniżej są minimalnymi punktami startowymi; w repo warto dodać pełne logowanie, walidację, testy i CI.

### src/preprocessing.py

```python
"""Preprocessing tekstu: czyszczenie, detekcja języka, opcjonalne tłumaczenie"""
import pandas as pd
from langdetect import detect
import re

EMOJI_PATTERN = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = EMOJI_PATTERN.sub('', s)
    s = s.replace('\n', ' ').strip()
    return s

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df['clean_text'] = df['review_text'].astype(str).apply(clean_text)
    df['lang'] = df['clean_text'].apply(lambda t: detect(t) if t else 'unknown')
    df.to_csv(args.output, index=False)
    print('Preprocessing done ->', args.output)
```

### src/sentiment_analysis.py

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def predict_sentiment(text: str):
    inputs = tokenizer(text, return_tensors='pt', truncation=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
    labels = ['negative','neutral','positive']
    return dict(zip(labels, probs.tolist()))
```

### src/emotion_analysis.py

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL = 'j-hartmann/emotion-english-distilroberta-base'

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def predict_emotion(text: str):
    inputs = tokenizer(text, return_tensors='pt', truncation=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
    # map zgodnie z dokumentacją modelu
    return probs
```

### src/topic_modeling.py (BERTopic skeleton)

```python
from bertopic import BERTopic

def fit_bertopic(docs):
    topic_model = BERTopic(language='english')
    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics, probs
```

### src/summarizer.py (LLM wrapper)

```python
# Przykład z OpenAI-like client
from openai import OpenAI
client = OpenAI()

def generate_insights(prompt: str) -> str:
    resp = client.chat.completions.create(
        model='gpt-4.1',
        messages=[{'role':'user','content': prompt}],
        max_tokens=500
    )
    return resp.choices[0].message['content']
```

### src/dashboard.py (Dash skeleton)

```python
import dash
from dash import html, dcc

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Customer Emotions Dashboard'),
    dcc.Graph(id='sentiment-pie'),
])

if __name__ == '__main__':
    app.run_server(debug=True)
```

### src/api.py (FastAPI skeleton)

```python
from fastapi import FastAPI
import pandas as pd
app = FastAPI()

@app.get('/insights')
def get_insights():
    # zwróć wcześniej wygenerowane insights z pliku/DB
    return {'status':'ok','insights':'...' }
```

### report/generate_report.py

```python
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def generate(output='report.pdf', summary_text=''):
    doc = SimpleDocTemplate(output)
    styles = getSampleStyleSheet()
    story = [Paragraph('Summary', styles['Heading1']), Paragraph(summary_text, styles['BodyText'])]
    doc.build(story)
```

### run_dashboard.py

```python
from src.dashboard import app

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050)
```

---

## 7. Przykładowe komendy

* Preprocess: `python src/preprocessing.py --input data/raw_reviews.csv --output data/processed_reviews.csv`
* Fit topics (notebook): `python -c "from src.topic_modeling import fit_bertopic; ..."`
* Run dashboard: `python run_dashboard.py`
* Run API: `uvicorn src.api:app --reload --port 8000`
* Generate report: `python report/generate_report.py --output report/example_report.pdf`

---

## 8. Rozszerzenia i deployment

* Dodaj Dockerfile i docker-compose (app + worker + redis + db)
* Użyj Celery do asynchronicznych zadań (topic modeling, LLM calls)
* CI/CD: GitHub Actions do testów i deploya
* Skalowanie: przenieś inferencję modeli do dedykowanego serwisu z GPU
* Bezpieczeństwo: ukryj klucze API w CI/CD secrets

---

## 9. Licencja

MIT — dostosuj do własnych potrzeb.

---

### Co dalej?

W pliku zawarłem kompletne README i minimalne implementacje plików startowych. Jeśli chcesz, mogę:

* wygenerować archiwum ZIP z plikami projektu,
* napisać pełną implementację konkretnego modułu (np. preprocessing.py z testami),
* przygotować Dockerfile i plik docker-compose,
* rozwinąć dashboard o konkretne wykresy i filtry.

Napisz, co chcesz teraz zrobić — przygotuję to od razu.

