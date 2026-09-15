# Train Your Own AI — Kulturnatten26

Kiosk demo: pick "Digits" (MNIST) or "Drawings" (Quick Draw), draw, configure
a small MLP, hit Retrain, watch it learn.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The kiosk is meant to run fully offline on the night - the server prewarms
both MNIST (`data/MNIST/`) and every curated Quick Draw category (a bounded
~4MB-per-category range request each, not the full multi-100MB files - see
`server/quickdraw_data.py`) into `data/quickdraw/` on startup, before it
starts accepting connections. So: **start the server at least once while
you still have internet** (e.g. the day before, or during setup) and let it
finish - watch the console for `MNIST prewarm done: ...`, then
`Quick Draw prewarm done: ... already cached, 0 failed`, then
`Application startup complete`. Everything is then cached on disk for good;
any later start, including on the actual event day with no network at all,
finds it all already local and skips straight past it in a couple of
seconds.

## Run

```
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000/` in a browser. Configure the layers, click
Retrain, and start drawing.
