# Train Your Own AI — Kulturnatten26

Kiosk demo: pick "Digits" (MNIST) or "Drawings" (Quick Draw), draw, configure
a small MLP, hit Retrain, watch it learn.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pre-download MNIST once (also happens automatically on first Retrain, but do
this ahead of time if venue wifi is unreliable):

```
python -c "from server.data import get_data_loaders; get_data_loaders()"
```

Drawings mode fetches Quick Draw category slices on demand (a small range
request per category, not the full multi-100MB files - see
`server/quickdraw_data.py`) and caches them under `data/quickdraw/`, the
first time each category is picked. There's no pre-warm-everything script
yet - if venue wifi on the night is a concern, that's an easy thing to add
once caching itself is proven out (ask for it).

## Run

```
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000/` in a browser. Configure the layers, click
Retrain, and start drawing.
