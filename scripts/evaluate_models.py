"""
Reproducible evaluation harness for the README "Results" section.

Runs three benchmarks on the same set of tickers using ~2 years of history:

1. **Forecasting accuracy** (per ticker, walk-forward):
   Train ARIMA + LSTM on the first N-H days, forecast the last H days, and
   compare against actuals using MAPE and RMSE.

2. **Portfolio backtest** (out-of-sample):
   Split the ticker basket history 70/30. Fit max-Sharpe weights on the
   training window. Compute realized Sharpe on the test window against two
   baselines — equal-weight (1/N) and SPY buy-and-hold.

3. **Sentiment signal** (optional — needs NewsAPI or Finnhub key):
   Correlate daily aggregated FinBERT sentiment with next-day log returns
   to give a directional reading on whether the signal is informative.

Output: a console table plus a clean markdown table at the end suitable for
pasting directly into the README. Total runtime ~3-5 minutes (LSTM training
dominates).

Usage:
    # Default (AAPL, MSFT, GOOGL on 2 years of data, 30-day forecast holdout)
    python scripts/evaluate_models.py

    # Custom tickers + skip the slow LSTM
    python scripts/evaluate_models.py --tickers AAPL,NVDA --skip-lstm

    # Save the markdown table to a file for easy README embedding
    python scripts/evaluate_models.py --output docs/eval_results.md
"""
from __future__ import annotations

# ─── sys.path bootstrap ───────────────────────────────────────────────────
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ──────────────────────────────────────────────────────────────────────────

import argparse
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from backend.core.logging import configure_logging
from ml_engine.data.yahoo_finance import fetch_close_series


# ─── Result containers ────────────────────────────────────────────────────


@dataclass
class ForecastEval:
    ticker: str
    model: str
    rmse: float
    mape: float
    runtime_s: float
    n_train: int
    n_test: int


@dataclass
class PortfolioEval:
    name: str
    realized_return_ann: float   # Annualized return on the test window
    realized_vol_ann: float      # Annualized realized vol on the test window
    realized_sharpe: float       # Excess return / vol (rf=0.02)
    final_value: float           # $10,000 invested at test start


@dataclass
class EvalReport:
    """Top-level container — printed at the end of the run."""

    forecast: list[ForecastEval] = field(default_factory=list)
    portfolio: list[PortfolioEval] = field(default_factory=list)
    sentiment_note: str | None = None
    timestamp: str = ""
    tickers: list[str] = field(default_factory=list)


# ─── Metric helpers ───────────────────────────────────────────────────────


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error in percent."""
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def _annualize_return(daily_log_returns: pd.Series) -> float:
    return float(daily_log_returns.mean() * 252)


def _annualize_vol(daily_log_returns: pd.Series) -> float:
    return float(daily_log_returns.std(ddof=1) * np.sqrt(252))


def _sharpe(daily_log_returns: pd.Series, rf: float = 0.02) -> float:
    ann_ret = _annualize_return(daily_log_returns)
    ann_vol = _annualize_vol(daily_log_returns)
    return (ann_ret - rf) / ann_vol if ann_vol > 0 else 0.0


# ─── Forecast benchmark ───────────────────────────────────────────────────


def evaluate_forecaster(
    ticker: str,
    lookback_days: int,
    horizon: int,
    skip_lstm: bool,
) -> list[ForecastEval]:
    """Walk-forward eval: train on [start, end-horizon), forecast last `horizon` days."""
    results: list[ForecastEval] = []

    series = fetch_close_series(ticker, lookback_days=lookback_days)
    if len(series) < horizon + 100:
        logger.warning(f"[{ticker}] Insufficient history ({len(series)} days); skipping")
        return results

    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:]
    actuals = test.values
    n_train, n_test = len(train), len(test)

    logger.info(
        f"[{ticker}] Train window: {n_train} days "
        f"({train.index[0].date()} → {train.index[-1].date()}); "
        f"Test window: {n_test} days ({test.index[0].date()} → {test.index[-1].date()})"
    )

    # ── ARIMA ────────────────────────────────────────────────────────────
    try:
        from ml_engine.forecasting.arima_model import ARIMAForecaster

        t0 = time.time()
        # Fixed (5,1,0) is fast and works well for daily close prices; auto
        # would also work but adds 30-60s of grid search per ticker.
        f = ARIMAForecaster(order=(5, 1, 0))
        f.fit(train)
        preds, _, _ = f.predict(horizon=horizon, confidence=0.95)
        runtime = time.time() - t0

        results.append(ForecastEval(
            ticker=ticker,
            model="ARIMA(5,1,0)",
            rmse=_rmse(actuals, preds),
            mape=_mape(actuals, preds),
            runtime_s=runtime,
            n_train=n_train,
            n_test=n_test,
        ))
        logger.info(
            f"[{ticker}] ARIMA: RMSE={results[-1].rmse:.4f} "
            f"MAPE={results[-1].mape:.2f}% ({runtime:.1f}s)"
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[{ticker}] ARIMA failed: {exc}")

    # ── LSTM ─────────────────────────────────────────────────────────────
    if not skip_lstm:
        try:
            from ml_engine.forecasting.lstm_model import LSTMForecaster

            t0 = time.time()
            # Modest epochs to keep runtime reasonable; EarlyStopping usually
            # kicks in well before the cap anyway.
            f = LSTMForecaster(lookback=60, epochs=30, batch_size=32, mc_samples=20)
            f.fit(train)
            preds, _, _ = f.predict(horizon=horizon, confidence=0.95)
            runtime = time.time() - t0

            results.append(ForecastEval(
                ticker=ticker,
                model="LSTM(64,32) +MC-Dropout",
                rmse=_rmse(actuals, preds),
                mape=_mape(actuals, preds),
                runtime_s=runtime,
                n_train=n_train,
                n_test=n_test,
            ))
            logger.info(
                f"[{ticker}] LSTM:  RMSE={results[-1].rmse:.4f} "
                f"MAPE={results[-1].mape:.2f}% ({runtime:.1f}s)"
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[{ticker}] LSTM failed: {exc}")

    return results


# ─── Portfolio backtest ───────────────────────────────────────────────────


def evaluate_portfolio(
    tickers: list[str],
    lookback_days: int,
    train_pct: float = 0.7,
    budget: float = 10_000.0,
    rf: float = 0.02,
) -> list[PortfolioEval]:
    """Out-of-sample portfolio backtest.

    Split history at ``train_pct``. Fit max-Sharpe weights on the training
    window; evaluate realized return / vol / Sharpe on the test window for
    three strategies: max-Sharpe, equal-weight, SPY buy-and-hold.
    """
    from ml_engine.portfolio import optimizer as PO

    logger.info(f"Portfolio backtest: {tickers} + SPY, {lookback_days}d total")
    panel = PO.fetch_price_panel(tickers + ["SPY"], lookback_days=lookback_days)

    split_idx = int(len(panel) * train_pct)
    train_panel = panel.iloc[:split_idx].drop(columns=["SPY"])
    test_panel = panel.iloc[split_idx:]
    logger.info(
        f"Split at index {split_idx}: train {train_panel.index[0].date()}→{train_panel.index[-1].date()}, "
        f"test {test_panel.index[0].date()}→{test_panel.index[-1].date()}"
    )

    # ── Strategy 1: max-Sharpe on training window ─────────────────────────
    opt = PO.optimize(train_panel, objective="max_sharpe", risk_free_rate=rf)

    # ── Compute realized test-window daily log returns for each strategy ─
    test_log_returns = np.log(test_panel / test_panel.shift(1)).dropna()

    # Max-Sharpe: weighted sum of test returns
    weights = pd.Series(opt.weights).reindex(train_panel.columns).fillna(0.0)
    weights /= weights.sum() if weights.sum() > 0 else 1.0   # safety
    optimal_rets = (test_log_returns[train_panel.columns] * weights.values).sum(axis=1)

    # Equal-weight on the same basket
    n = len(train_panel.columns)
    eq_weights = pd.Series(1.0 / n, index=train_panel.columns)
    equal_rets = (test_log_returns[train_panel.columns] * eq_weights.values).sum(axis=1)

    # SPY buy-and-hold
    spy_rets = test_log_returns["SPY"]

    results: list[PortfolioEval] = []
    for name, rets in [
        ("Max-Sharpe portfolio", optimal_rets),
        ("Equal-weight (1/N)", equal_rets),
        ("SPY buy-and-hold", spy_rets),
    ]:
        cumulative_log = rets.sum()
        final_value = budget * float(np.exp(cumulative_log))
        results.append(PortfolioEval(
            name=name,
            realized_return_ann=_annualize_return(rets),
            realized_vol_ann=_annualize_vol(rets),
            realized_sharpe=_sharpe(rets, rf=rf),
            final_value=final_value,
        ))

    for r in results:
        logger.info(
            f"  {r.name:24s} ann.return={r.realized_return_ann:7.2%}  "
            f"ann.vol={r.realized_vol_ann:6.2%}  Sharpe={r.realized_sharpe:5.2f}  "
            f"final=${r.final_value:,.0f}"
        )
    return results


# ─── Sentiment signal (best-effort) ───────────────────────────────────────


def evaluate_sentiment(tickers: list[str], lookback_days: int = 30) -> str:
    """Optional: correlate FinBERT daily-aggregated sentiment with next-day log returns.

    Returns a one-paragraph note suitable for the README. Returns a placeholder
    note if no news API keys are configured.
    """
    from backend.core.config import settings

    if not (settings.news_api_key or settings.finnhub_api_key):
        return (
            "_Sentiment correlation skipped — neither `NEWS_API_KEY` nor "
            "`FINNHUB_API_KEY` is configured. yfinance fallback typically returns "
            "only ~10 headlines per ticker, too few for a stable correlation reading._"
        )

    try:
        from ml_engine.data.news_fetcher import fetch_news
        from ml_engine.sentiment.analyzer import FinBERTAnalyzer
    except Exception as exc:  # noqa: BLE001
        return f"_Sentiment eval skipped — analyzer not importable: {exc}_"

    analyzer = FinBERTAnalyzer()
    correlations: list[tuple[str, int, float]] = []

    for ticker in tickers:
        try:
            articles = fetch_news(ticker, max_articles=50, days_back=lookback_days)
            if len(articles) < 5:
                correlations.append((ticker, len(articles), float("nan")))
                continue

            df = pd.DataFrame(articles)
            df["published_at"] = pd.to_datetime(df["published_at"]).dt.tz_localize(None)
            df["date"] = df["published_at"].dt.date

            # FinBERT returns SentimentScore(label, score) where score ∈ [-1, 1]
            scores = analyzer.score_batch([a["title"] for a in articles])
            df["signed"] = [s.score for s in scores]
            daily_sent = df.groupby("date")["signed"].mean()

            # Align with next-day returns
            prices = fetch_close_series(ticker, lookback_days=lookback_days + 5)
            log_ret = np.log(prices / prices.shift(1)).dropna()
            log_ret.index = log_ret.index.date  # type: ignore[attr-defined]
            next_day_ret = log_ret.shift(-1).dropna()

            joined = pd.DataFrame({"sent": daily_sent, "next_ret": next_day_ret}).dropna()
            if len(joined) < 5:
                correlations.append((ticker, len(articles), float("nan")))
                continue

            corr = float(joined["sent"].corr(joined["next_ret"]))
            correlations.append((ticker, len(articles), corr))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[{ticker}] sentiment correlation failed: {exc}")
            correlations.append((ticker, 0, float("nan")))

    rows = []
    for t, n, c in correlations:
        if np.isnan(c):
            rows.append(f"- **{t}**: {n} headlines (insufficient or correlation undefined)")
        else:
            rows.append(f"- **{t}**: {n} headlines, corr(sentiment, next-day return) = `{c:+.3f}`")

    return (
        f"FinBERT daily-aggregated sentiment vs next-day log returns over "
        f"the last {lookback_days} days:\n\n" + "\n".join(rows) + (
            "\n\n_Note: correlations on small samples are noisy. A value near 0 "
            "doesn't mean sentiment is useless — it means at this aggregation "
            "level it's not directly predictive. Real-world strategies typically "
            "combine sentiment with technicals rather than using it standalone._"
        )
    )


# ─── Report rendering ─────────────────────────────────────────────────────


def render_markdown(report: EvalReport) -> str:
    """Render the report as markdown for README embedding."""
    lines: list[str] = []
    lines.append(f"_Generated {report.timestamp} on tickers: `{', '.join(report.tickers)}`._\n")

    # ── Forecasting table ───────────────────────────────────────────────
    if report.forecast:
        lines.append("### Forecasting Accuracy (30-day walk-forward holdout)\n")
        lines.append("| Ticker | Model | RMSE ($) | MAPE | Train size | Runtime |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for r in report.forecast:
            lines.append(
                f"| {r.ticker} | {r.model} | {r.rmse:.3f} | {r.mape:.2f}% | "
                f"{r.n_train} | {r.runtime_s:.1f}s |"
            )
        lines.append("")
        lines.append(
            "Walk-forward methodology: trained on history up to T-30, forecasted "
            "the next 30 trading days, scored against actuals. Lower RMSE / MAPE = better.\n"
        )

    # ── Portfolio table ─────────────────────────────────────────────────
    if report.portfolio:
        lines.append("### Portfolio Backtest (out-of-sample, last 30% of window)\n")
        lines.append("| Strategy | Realized Return (ann.) | Realized Vol (ann.) | Sharpe | $10k → |")
        lines.append("|---|---:|---:|---:|---:|")
        for p in report.portfolio:
            lines.append(
                f"| {p.name} | {p.realized_return_ann:.2%} | {p.realized_vol_ann:.2%} | "
                f"{p.realized_sharpe:.2f} | ${p.final_value:,.0f} |"
            )
        lines.append("")
        lines.append(
            "Max-Sharpe weights fitted on the first 70% of the window; "
            "all three strategies evaluated on the held-out 30%. Risk-free rate = 2%. "
            "A Sharpe above the equal-weight baseline indicates the optimizer "
            "added real value out-of-sample (often it doesn't — that's the famous "
            "estimation-error problem in mean-variance optimization).\n"
        )

    # ── Sentiment note ──────────────────────────────────────────────────
    if report.sentiment_note:
        lines.append("### Sentiment Signal\n")
        lines.append(report.sentiment_note)
        lines.append("")

    return "\n".join(lines)


def render_console(report: EvalReport) -> str:
    """Plain-text version for the terminal."""
    out = ["\n" + "=" * 70, "EVALUATION REPORT", "=" * 70]
    out.append(f"Generated: {report.timestamp}")
    out.append(f"Tickers:   {', '.join(report.tickers)}\n")

    if report.forecast:
        out.append("Forecasting accuracy (30-day walk-forward):")
        out.append(f"  {'Ticker':<8} {'Model':<26} {'RMSE':>8} {'MAPE':>8} {'Runtime':>9}")
        out.append("  " + "-" * 62)
        for r in report.forecast:
            out.append(
                f"  {r.ticker:<8} {r.model:<26} {r.rmse:>8.3f} "
                f"{r.mape:>7.2f}% {r.runtime_s:>8.1f}s"
            )
        out.append("")

    if report.portfolio:
        out.append("Portfolio backtest (out-of-sample):")
        out.append(f"  {'Strategy':<24} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} {'$10k →':>12}")
        out.append("  " + "-" * 64)
        for p in report.portfolio:
            out.append(
                f"  {p.name:<24} {p.realized_return_ann:>7.2%} "
                f"{p.realized_vol_ann:>7.2%} {p.realized_sharpe:>8.2f} "
                f"${p.final_value:>10,.0f}"
            )
        out.append("")

    return "\n".join(out)


# ─── CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible model evaluation for the README.")
    parser.add_argument(
        "--tickers",
        type=str,
        default="AAPL,MSFT,GOOGL",
        help="Comma-separated tickers (default: AAPL,MSFT,GOOGL)",
    )
    parser.add_argument("--lookback-days", type=int, default=730)
    parser.add_argument("--horizon", type=int, default=30, help="Forecast holdout in days")
    parser.add_argument("--skip-lstm", action="store_true", help="Skip LSTM (faster)")
    parser.add_argument("--skip-portfolio", action="store_true")
    parser.add_argument("--skip-sentiment", action="store_true")
    parser.add_argument("--output", type=Path, help="Save markdown to this path")
    args = parser.parse_args()

    configure_logging()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    report = EvalReport(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        tickers=tickers,
    )

    # ── 1. Forecasting ──────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 1/3: Forecasting benchmark")
    logger.info("=" * 70)
    for t in tickers:
        report.forecast.extend(evaluate_forecaster(
            ticker=t,
            lookback_days=args.lookback_days,
            horizon=args.horizon,
            skip_lstm=args.skip_lstm,
        ))

    # ── 2. Portfolio ─────────────────────────────────────────────────────
    if not args.skip_portfolio and len(tickers) >= 2:
        logger.info("=" * 70)
        logger.info("STEP 2/3: Portfolio backtest")
        logger.info("=" * 70)
        report.portfolio = evaluate_portfolio(
            tickers=tickers,
            lookback_days=args.lookback_days,
        )

    # ── 3. Sentiment ─────────────────────────────────────────────────────
    if not args.skip_sentiment:
        logger.info("=" * 70)
        logger.info("STEP 3/3: Sentiment signal")
        logger.info("=" * 70)
        report.sentiment_note = evaluate_sentiment(tickers=tickers)

    # ── Render ───────────────────────────────────────────────────────────
    print(render_console(report))

    md = render_markdown(report)
    print("\n" + "=" * 70)
    print("MARKDOWN (paste into README Results section):")
    print("=" * 70)
    print(md)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md, encoding="utf-8")
        logger.info(f"Markdown saved to {args.output}")


if __name__ == "__main__":
    main()
