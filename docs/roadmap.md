# Execution Roadmap

This roadmap sequences the work so each phase produces something demonstrable.
The general principle: get a thin end-to-end slice working before deepening any
single module.

## Phase 0 — Foundation (✅ delivered with this scaffold)

| Deliverable | Status |
|---|---|
| Repository layout, FastAPI app, MongoDB connection | ✅ |
| Pydantic schemas + service interfaces | ✅ |
| RAG plumbing (FAISS + LangChain) | ✅ |
| Streamlit shell + API client | ✅ |
| Forecaster / Sentiment / Risk interfaces | ✅ |
| Dockerized local dev stack | ✅ |

Smoke test: `uvicorn backend.main:app --reload` → `GET /api/v1/health` returns
`{"status": "ok"}` with `mongo_ok: true`.

## Phase 1 — Data Layer (1–2 weeks)

Establish reliable, repeatable data acquisition. Skip this and every downstream
metric will be junk.

1. **Yahoo Finance ingest** (`ml_engine/data/yahoo_finance.py` — partly done).
   Add caching to `data/raw/prices/{ticker}.parquet`; expire after 1 trading day.
2. **News fetcher** (`ml_engine/data/news_api.py`). Use NewsAPI or Finnhub free
   tier. Persist articles to MongoDB with a unique `(ticker, url)` index to dedupe.
3. **Synthetic users** (`ml_engine/data/synthetic_users.py`). Generate ~1000
   profiles using `faker` to seed the risk-profiler training set.
4. **EDA notebook** (`notebooks/01_data_exploration.ipynb`). Document the shape
   and idiosyncrasies of each source — missing values, market closures,
   timezone handling.

Exit criterion: a single script can refresh prices + news for a watchlist
of 20 tickers in under 60 seconds, cached.

## Phase 2 — Forecasting (2–3 weeks)

Build ARIMA first (cheap baseline) → LSTM (improvement) → ensemble.

### 2A. ARIMA baseline

1. In `notebooks/02_arima_baseline.ipynb`:
   - Plot ACF/PACF, run ADF stationarity tests.
   - Compare hand-picked `(p,d,q)` against `pmdarima.auto_arima`.
   - Walk-forward backtest: train on rolling window, evaluate RMSE / MAPE.
2. Promote the validated configuration into
   `ml_engine/forecasting/arima_model.py` (interface already defined).
3. Wire `ForecastingService.forecast()` to call ARIMA when `model_type ∈ {"arima","auto"}`.
4. Persist trained models under `models/arima/{ticker}.pkl`.

### 2B. LSTM model

1. `notebooks/03_lstm_modeling.ipynb` — preprocess with MinMaxScaler, build
   sliding windows, try 1- and 2-layer LSTM with dropout. Use early stopping
   on validation loss.
2. Implement `LSTMForecaster.fit/predict/save/load` (skeleton in place).
3. Use **MC-dropout** at inference for confidence intervals — cleaner than
   bootstrap given the cost of retraining.
4. Compare ARIMA vs. LSTM on the same walk-forward backtest. Record results in
   `docs/experiments.md`.

### 2C. Ensemble / Auto

1. Add `ForecastEnsemble` that blends ARIMA + LSTM via inverse-RMSE weights.
2. `model_type="auto"` routes to the best-performing model per ticker, decided
   at training time and stored alongside the artifact.

### 2D. Training pipeline

1. `scripts/train_forecaster.py` — CLI taking `--ticker AAPL --model arima` and
   writing artifacts + metrics to `models/` + MongoDB.
2. Schedule a nightly retrain via cron / GitHub Actions / Airflow once
   stable.

Exit criterion: `POST /api/v1/forecasting/predict {"ticker": "AAPL", "horizon_days": 30}`
returns a 30-day forecast with CIs in under 2 seconds (cached) / 30 seconds (cold).

## Phase 3 — Sentiment (1–2 weeks)

1. Implement `ml_engine/sentiment/news_fetcher.py` — pull headlines + bodies
   for a ticker in a date range; cache to MongoDB.
2. Implement the FinBERT pipeline in `analyzer.py` (skeleton in place). Run
   the heavy model load **once** at app startup; cache the pipeline on
   `SentimentService`.
3. Aggregate: per article → daily rolling sentiment → ticker-level summary
   with distribution counts.
4. Wire `SentimentService.analyze()` and surface in the Streamlit page with a
   time-series chart of daily sentiment.

Exit criterion: dashboard renders a 30-day sentiment chart for any S&P 500
ticker within 10 seconds (first run) or sub-second (cached).

## Phase 4 — Risk + Portfolio (1–2 weeks)

1. Promote the rule-based scorer in `ml_engine/risk/profiler.py` into the
   `RiskService.profile()` path. Add unit tests for boundary cases.
2. Implement `ml_engine/risk/portfolio_optimizer.py` using PyPortfolioOpt's
   `EfficientFrontier` (max_sharpe / min_volatility / efficient_return).
3. Compose: risk score → target equity allocation → optimizer constrained to
   that target. Save the optimized portfolio to MongoDB per user.

Exit criterion: given a questionnaire response and a watchlist, the API
returns an allocation that respects the user's risk band.

## Phase 5 — RAG hardening (parallel, ongoing)

The scaffold gives you a working chain. Productionizing it is its own track:

- Seed a curated corpus (latest 10-Ks for top 50 tickers + a glossary doc).
- Add **citations** — already wired into the prompt; verify retrieval metadata
  flows back to the response.
- Add **MMR** retrieval to reduce redundancy on overlapping filings.
- Add a **reranker** (Cohere or a small cross-encoder) once recall is good but
  precision is wobbly.
- Track **groundedness** with `langsmith` or a manual eval set — guard against
  hallucinated numbers.

## Phase 6 — Integration & UX (1 week)

1. Cross-module integration: an "Advisor Summary" endpoint that combines
   forecast + sentiment + risk-aware allocation for a watchlist.
2. Streamlit polish: caching with `st.cache_data`, loading skeletons, error
   states, mobile-friendly layout.
3. Auth: JWT login on `/api/v1/auth/*` (placeholder in `core/security.py`).
   Hook into the Streamlit sidebar.

## Phase 7 — Quality & Deployment (1–2 weeks)

1. **Tests**: minimum 70% coverage across `backend/services/` and
   `ml_engine/*`. Use `pytest` + `pytest-asyncio` + `mongomock-motor` for the DB.
2. **CI**: `.github/workflows/ci.yml` running ruff + mypy + pytest on every PR.
3. **Containerize**: finalize Dockerfiles under `docker/`, push to a registry.
4. **Deploy**: backend on Fly.io / Render / AWS App Runner; dashboard on
   Streamlit Community Cloud or the same container host.

## Engineering Practices to Hold the Line

- One module per PR. Don't merge a 3000-line "add forecasting" change.
- Every notebook has a sibling Python module — notebooks for exploration only.
- Mock external services (Yahoo, OpenAI) in tests via `httpx.MockTransport`.
- Pin model artifacts by ticker + training date. Never overwrite silently.
- Log every external call (RAG query, model invocation) with a request id so
  you can replay any user session.
