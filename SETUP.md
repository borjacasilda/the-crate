# Setup guide (macOS)

This walks you through running **The Crate** on your own machine, in plain steps.
It is written for macOS; Linux is nearly identical. Plan for about 10–15 minutes,
plus download time for the optional AI/ML models.

The only hard requirement is **Docker** (for the database). Everything else is
either bundled by the installer or optional.

---

## What you need

| Tool | Why | Install |
|------|-----|---------|
| **uv** | Runs the project and manages Python | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Docker Desktop** | Runs the PostgreSQL database | <https://www.docker.com/products/docker-desktop/> |
| **git** | Clone the repository | `brew install git` (or Xcode tools) |
| Ollama *(optional)* | The local AI assistant | <https://ollama.com/download> |

You do **not** need to install Python yourself — `uv` fetches the right version
(3.11) automatically.

If you do not have Homebrew yet and want it:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

---

## Step 1 — Get the code

```bash
git clone https://github.com/borjacasilda/the-crate.git
cd the-crate
```

## Step 2 — Install the project

```bash
uv sync
```

This creates a local `.venv` and installs every dependency from `uv.lock`
(Essentia, FastAPI, etc.). It takes a couple of minutes the first time.

## Step 3 — Set your configuration

Copy the template and open it:

```bash
cp .env.example .env
nano .env        # or open it in any editor
```

You only have to change **one thing** to get started: set
`POSTGRES_PASSWORD` to any password you like. Everything else has working
defaults. (The optional sections — Discogs, Ollama — can stay commented out for now.)

## Step 4 — Start the database

Make sure Docker Desktop is running, then:

```bash
docker compose up -d
```

This starts PostgreSQL with the `pgvector` extension in the background. The Crate
creates its tables automatically the first time it connects — you do not run any
SQL yourself.

> Already using port 5432? Set a different `DB_PORT` in `.env` (e.g. `5433`) and
> run `docker compose up -d` again.

## Step 5 — Start The Crate

```bash
uv run uvicorn api:app --host 127.0.0.1 --port 8000
```

Then open **<http://127.0.0.1:8000/crates>** in your browser. You should see the
collection page with an empty **Vinyl Collection** (your master library).

> Use `127.0.0.1`, not `localhost` — on some setups `localhost` collides with
> other local services.

## Step 6 — Add some music

From the collection, open **Vinyl Collection → Crate management → Import files**
and pick a few `.wav`, `.mp3`, or `.flac` files. The Crate standardises and
analyses each track (about 30–60 seconds per track the first time, because it runs
the full audio analysis). Once a track shows a BPM and key, it is ready.

You now have a working setup. The sections below are optional upgrades.

---

## Optional — better analysis (Essentia ML models)

Out of the box The Crate uses classic signal analysis (tempo, key, energy). To
unlock mood, genre, and "sounds-like" recommendations, download the pretrained
models once:

```bash
uv run python analyze.py download
```

This fetches a few GB of model files into `models/` (kept out of git). After it
finishes, re-analyse your tracks so they reach the higher pipeline levels.

## Optional — the AI assistant (Ollama)

The in-app assistant (the **ASK** tab) runs a local language model through Ollama.

1. Install and start Ollama (`ollama serve`, or just open the app).
2. Pull the default model: `ollama pull qwen3:4b-instruct`.
3. Reload any page and open **ASK** — the assistant is now online.

Nothing is sent to the cloud; the model runs locally.

## Optional — Discogs enrichment (cover art, labels)

1. Create a token at <https://www.discogs.com/settings/developers> →
   *Generate new token* (a personal access token is enough).
2. In `.env`, uncomment and fill `DISCOGS_ACCESS_TOKEN`.
3. Restart the server. The **Enrich** button on a crate now pulls cover art,
   label, year, and styles.

---

## Verify your setup

```bash
uv run python scripts/verify.py
```

It checks your Python version, that `.env` exists, that dependencies are installed,
and whether the database is reachable.

## Troubleshooting

**`docker: command not found` / database won't start**
Open Docker Desktop and wait until it says "running", then `docker compose up -d` again.

**`connection refused` / the app can't reach the database**
The database container may still be starting. Wait a few seconds, or check it with
`docker compose ps`. Confirm `DB_PORT` in `.env` matches the published port.

**Port 8000 is already in use**
Start the server on another port: `uv run uvicorn api:app --host 127.0.0.1 --port 8010`.

**Changed a `.py` file and nothing changed in the app**
The API caches at startup — stop the server (Ctrl-C) and start it again. Edits to
files under `web/` only need a browser refresh.

**The assistant says it is offline**
Ollama isn't running or the model isn't pulled. Run `ollama serve` and
`ollama pull qwen3:4b-instruct`.

**Apple Silicon (M1/M2/M3) install issues**
`uv` installs native wheels automatically; if a dependency fails, re-run `uv sync`.

Still stuck? Open an issue on GitHub with the output of `uv run python scripts/verify.py`.
