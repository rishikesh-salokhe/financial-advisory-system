# Dashboard Screenshots

These images are embedded in the project README. To refresh them, run the
dashboard locally (`streamlit run dashboard/app.py`), capture each page with
the suggested inputs, and save to this directory with the filenames below.

PNG, ~1600×900 ideal. Use a clean default Streamlit theme.

| File | Page | Suggested inputs |
|---|---|---|
| `home.png` | Landing page | Default landing view |
| `forecasting.png` | 📈 Forecasting | `AAPL` or `NVDA`, horizon 30, model = LSTM |
| `sentiment.png` | 📰 Sentiment | `NVDA` or `TSLA`, 30-day window |
| `risk.png` | ⚠️ Risk | `AAPL`, benchmark SPY, 2-year lookback |
| `advisor.png` | 💬 Advisor | Question: "Compare the revenue concentration risks of Apple and Microsoft." |
| `portfolio.png` | 📈 Portfolio | `AAPL,MSFT,GOOGL,NVDA,AMZN`, 5-year, max-Sharpe, $10k — capture with frontier visible |
