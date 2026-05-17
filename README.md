# AI-Powered Financial Advisory & Risk Intelligence System

An integrated platform combining time-series forecasting (ARIMA/LSTM), NLP sentiment
analysis, RAG-based financial Q&A, risk profiling, and portfolio optimization, exposed
through a FastAPI service and a Streamlit dashboard.

## Architecture at a Glance

```
                        ┌────────────────────────┐
                        │   Streamlit Dashboard  │
                        │  (visualization + UX)  │
                        └──────────┬─────────────┘
                                   │ HTTPS
                        ┌──────────▼─────────────┐
                        │      FastAPI API       │
                        │  (auth, orchestration) │
                        └──────────┬─────────────┘
                                   │
        ┌────────────┬─────────────┼─────────────┬────────────┐
        │            │             │             │            │
   ┌────▼────┐  ┌────▼─────┐  ┌────▼────┐  ┌─────▼────┐  ┌────▼────┐
   │Forecast │  │Sentiment │  │  RAG    │  │ Risk &   │  │ Data    │
   │ARIMA/LSTM│  │  NLP    │  │LangChain│  │Portfolio │  │ Layer   │
   └─────────┘  └──────────┘  │ + FAISS │  └──────────┘  │(Yahoo,  │
                              └─────────┘                │ News)   │
                                   │                     └─────────┘
                              ┌────▼────┐
                              │ MongoDB │
                              │ (state) │
                              └─────────┘
```

## Repository Layout

```
financial-advisory-system/
├── backend/                  FastAPI service (API layer + orchestration)
│   ├── main.py               App entry, lifespan, middleware, router mount
│   ├── core/                 Settings, logging, exceptions, security
│   ├── db/                   MongoDB async client + repositories
│   ├── api/v1/endpoints/     Route handlers per module
│   ├── schemas/              Pydantic request/response contracts
│   ├── services/             Business-logic orchestration (calls ml_engine)
│   └── middleware/           Request logging, CORS, auth gates
│
├── ml_engine/                Pure-Python ML/NLP/RAG layer (no web framework)
│   ├── forecasting/          ARIMA + LSTM models behind a Forecaster interface
│   ├── sentiment/            News ingestion + NLTK/transformer pipelines
│   ├── risk/                 Risk scoring + Markowitz portfolio optimization
│   ├── rag/                  LangChain ingestion, FAISS store, retriever, chain
│   ├── data/                 yfinance, news APIs, synthetic user generators
│   └── utils/                Caching, metrics, IO helpers
│
├── dashboard/                Streamlit app (calls the FastAPI service)
│   ├── app.py                Landing page + auth shell
│   ├── pages/                One page per module
│   └── components/           Reusable charts, auth widgets, API client
│
├── notebooks/                EDA and modeling experiments (Jupyter)
├── tests/                    pytest tests mirroring src tree
├── scripts/                  CLIs for ingestion, training, DB seeding
├── data/                     Local raw/processed/vectorstore (gitignored)
├── models/                   Trained model artifacts (gitignored)
├── docs/                     Architecture notes, API reference, roadmap
└── docker-compose.yml        FastAPI + MongoDB + Streamlit for local dev
```

The two non-obvious choices worth calling out:

`backend/` and `ml_engine/` are separate top-level packages. The backend imports
from `ml_engine`, but `ml_engine` knows nothing about FastAPI. This lets us reuse
the modeling layer from notebooks, batch scripts, or another service later
without dragging in web dependencies.

`services/` sits between `api/v1/endpoints/` and `ml_engine/`. Route handlers
stay thin: parse the request, call a service, return the response. Services hold
the orchestration logic (cache lookups, model selection, persistence). This is
the seam most beginners skip and regret six months in.

## Quickstart

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in MongoDB URI, OpenAI key, etc.

# 2. MongoDB (Docker is easiest)
docker compose up -d mongo

# 3. Backend
uvicorn backend.main:app --reload --port 8000
# → OpenAPI docs at http://localhost:8000/docs

# 4. Dashboard (separate terminal)
streamlit run dashboard/app.py
```

## Roadmap

See `docs/roadmap.md` for the sequenced build plan.
