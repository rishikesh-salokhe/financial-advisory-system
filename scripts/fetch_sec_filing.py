"""
Programmatic SEC filing downloader.

Modern EDGAR routes most user-facing links through a JavaScript "XBRL Viewer"
wrapper that returns a tiny HTML shim instead of the actual filing document.
This script bypasses the viewer by calling SEC's structured data API
(``data.sec.gov``) to find the filing and downloading the primary document
file directly.

Usage:
    python scripts/fetch_sec_filing.py --ticker MSFT
    python scripts/fetch_sec_filing.py --ticker TSLA --form 10-Q
    python scripts/fetch_sec_filing.py --ticker AAPL --out data/raw/

The output file lands in ``data/raw/`` by default as an ``.htm`` (the inline
XBRL filing — our ingester handles ``.htm`` natively after the Phase 6 update).

SEC etiquette
-------------
The SEC requires every API client to identify itself in the User-Agent header
with a contact email. Set ``SEC_USER_AGENT`` in your ``.env`` to a real value
like ``"FinancialAdvisorProject your.email@example.com"`` — without it we fall
back to a generic project string, which works but is more likely to be
rate-limited if SEC tightens enforcement.
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
import os
import time
from typing import Any

import requests
from loguru import logger

from backend.core.logging import configure_logging


# Ticker → CIK mapping for the most-requested names. The full CIK lookup
# table lives at https://www.sec.gov/files/company_tickers.json but
# hardcoding the popular handful keeps the script self-contained.
CIK_BY_TICKER: dict[str, str] = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "GOOG": "0001652044",
    "AMZN": "0001018724",
    "META": "0001326801",
    "NVDA": "0001045810",
    "TSLA": "0001318605",
    "BRK.A": "0001067983",
    "BRK.B": "0001067983",
    "JPM": "0000019617",
    "V": "0001403161",
    "MA": "0001141391",
    "JNJ": "0000200406",
    "WMT": "0000104169",
    "UNH": "0000731766",
    "HD": "0000354950",
    "PG": "0000080424",
    "XOM": "0000034088",
    "BAC": "0000070858",
    "DIS": "0001744489",
    "NFLX": "0001065280",
    "PYPL": "0001633917",
    "ORCL": "0001341439",
    "INTC": "0000050863",
    "AMD": "0000002488",
    "CRM": "0001108524",
    "ADBE": "0000796343",
    "CSCO": "0000858877",
    "PEP": "0000077476",
    "KO": "0000021344",
    "MCD": "0000063908",
    "NKE": "0000320187",
    "BA": "0000012927",
    "GS": "0000886982",
    "IBM": "0000051143",
    "T": "0000732717",
    "VZ": "0000732712",
    "F": "0000037996",
    "GM": "0001467858",
    "SPY": "0000884394",
}


def _user_agent() -> str:
    """Return a valid SEC User-Agent string."""
    custom = os.environ.get("SEC_USER_AGENT", "").strip()
    if custom:
        return custom
    # Default — works but is generic; users should set their own.
    return "FinancialAdvisorProject research@example.com"


def _resolve_cik(ticker: str) -> str:
    """Return the zero-padded CIK for ``ticker``.

    Falls back to SEC's official company_tickers.json if the ticker isn't in
    our static map.
    """
    t = ticker.upper()
    if t in CIK_BY_TICKER:
        return CIK_BY_TICKER[t]

    logger.info(f"CIK for {t} not in static map — falling back to SEC company_tickers.json")
    headers = {"User-Agent": _user_agent()}
    resp = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=15)
    resp.raise_for_status()
    payload: dict[str, Any] = resp.json()
    for _, entry in payload.items():
        if str(entry.get("ticker", "")).upper() == t:
            cik_int = int(entry["cik_str"])
            return f"{cik_int:010d}"
    raise ValueError(f"Could not resolve CIK for ticker {t!r}")


def _find_latest_filing(cik: str, form: str) -> dict[str, str]:
    """Hit the submissions API and return metadata for the latest matching filing."""
    headers = {"User-Agent": _user_agent()}
    sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    logger.debug(f"GET {sub_url}")
    resp = requests.get(sub_url, headers=headers, timeout=15)
    resp.raise_for_status()
    sub: dict[str, Any] = resp.json()

    recent = sub["filings"]["recent"]
    forms: list[str] = recent["form"]
    accessions: list[str] = recent["accessionNumber"]
    primary_docs: list[str] = recent["primaryDocument"]
    filing_dates: list[str] = recent["filingDate"]
    report_dates: list[str] = recent["reportDate"]

    # ``recent`` is already sorted newest-first.
    for i, f in enumerate(forms):
        if f == form:
            return {
                "accession": accessions[i],
                "primary_doc": primary_docs[i],
                "filing_date": filing_dates[i],
                "report_date": report_dates[i],
                "company_name": sub.get("name", ""),
            }
    raise ValueError(f"No {form} filing found in recent submissions for CIK {cik}")


def _download_document(cik: str, accession: str, primary_doc: str, out_path: Path) -> int:
    """Download the primary filing document and return bytes written."""
    accession_no_dashes = accession.replace("-", "")
    cik_int = int(cik)  # the Archives URL uses the unpadded CIK
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_no_dashes}/{primary_doc}"
    logger.info(f"Downloading: {url}")

    headers = {"User-Agent": _user_agent()}
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    return len(resp.content)


def fetch(ticker: str, form: str, out_dir: Path) -> Path:
    """Download the latest ``form`` filing for ``ticker`` and return the local path."""
    cik = _resolve_cik(ticker)
    meta = _find_latest_filing(cik, form)
    logger.info(
        f"Found {form} for {ticker.upper()} ({meta['company_name']}): "
        f"accession={meta['accession']}, filing_date={meta['filing_date']}, "
        f"report_date={meta['report_date']}"
    )

    suffix = Path(meta["primary_doc"]).suffix or ".htm"
    safe_form = form.replace("/", "_")
    out_path = out_dir / f"{ticker.upper()}_{safe_form}_{meta['report_date']}{suffix}"
    bytes_written = _download_document(cik, meta["accession"], meta["primary_doc"], out_path)

    logger.info(f"Saved {out_path} ({bytes_written:,} bytes)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a recent SEC filing for a ticker directly from EDGAR.",
    )
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g. MSFT, TSLA)")
    parser.add_argument(
        "--form",
        default="10-K",
        help="Filing form (default: 10-K). Common values: 10-K, 10-Q, 8-K, 20-F",
    )
    parser.add_argument(
        "--out",
        default="data/raw",
        help="Output directory (default: data/raw)",
    )
    args = parser.parse_args()

    configure_logging()
    out_dir = Path(args.out).resolve()
    out_path = fetch(args.ticker, args.form, out_dir)

    # Be polite to SEC — small pause if invoked in a loop by a wrapper script.
    time.sleep(0.2)
    print(f"\n✓ Filing saved to: {out_path}")


if __name__ == "__main__":
    main()
