_Generated 2026-05-21 17:34 on tickers: `AAPL, MSFT, GOOGL`._

### Forecasting Accuracy (30-day walk-forward holdout)

| Ticker | Model | RMSE ($) | MAPE | Train size | Runtime |
|---|---|---:|---:|---:|---:|
| AAPL | ARIMA(5,1,0) | 24.726 | 7.03% | 471 | 2.1s |
| AAPL | LSTM(64,32) +MC-Dropout | 36.203 | 11.20% | 471 | 32.5s |
| MSFT | ARIMA(5,1,0) | 41.420 | 9.34% | 221 | 0.1s |
| MSFT | LSTM(64,32) +MC-Dropout | 32.850 | 7.39% | 221 | 14.1s |
| GOOGL | ARIMA(5,1,0) | 49.580 | 10.65% | 221 | 0.1s |
| GOOGL | LSTM(64,32) +MC-Dropout | 72.924 | 17.80% | 221 | 14.1s |

Walk-forward methodology: trained on history up to T-30, forecasted the next 30 trading days, scored against actuals. Lower RMSE / MAPE = better.

### Portfolio Backtest (out-of-sample, last 30% of window)

| Strategy | Realized Return (ann.) | Realized Vol (ann.) | Sharpe | $10k → |
|---|---:|---:|---:|---:|
| Max-Sharpe portfolio | 41.49% | 31.00% | 1.27 | $11,314 |
| Equal-weight (1/N) | 26.38% | 19.70% | 1.24 | $10,817 |
| SPY buy-and-hold | 22.13% | 14.63% | 1.38 | $10,681 |

Max-Sharpe weights fitted on the first 70% of the window; all three strategies evaluated on the held-out 30%. Risk-free rate = 2%. A Sharpe above the equal-weight baseline indicates the optimizer added real value out-of-sample (often it doesn't — that's the famous estimation-error problem in mean-variance optimization).

### Sentiment Signal

FinBERT daily-aggregated sentiment vs next-day log returns over the last 30 days:

- **AAPL**: 50 headlines, corr(sentiment, next-day return) = `-0.042`
- **MSFT**: 50 headlines, corr(sentiment, next-day return) = `+0.470`
- **GOOGL**: 45 headlines, corr(sentiment, next-day return) = `-0.334`

_Note: correlations on small samples are noisy. A value near 0 doesn't mean sentiment is useless — it means at this aggregation level it's not directly predictive. Real-world strategies typically combine sentiment with technicals rather than using it standalone._
