"""
Batch-ingest documents into the FAISS vector store.

Usage:
    python scripts/ingest_documents.py data/raw/
    python scripts/ingest_documents.py data/raw/AAPL_10K.pdf data/raw/MSFT_10K.pdf
    python scripts/ingest_documents.py data/raw/ --reset    # wipe index first

The script delegates to ``ml_engine.rag.ingestion.ingest_paths`` so it shares
the exact same pipeline the FastAPI ``/rag/ingest`` endpoint uses. Embeddings
are computed once and persisted to ``data/vectorstore/`` — the API picks up
the new index on the next query (the chain is cached lazily and reset when
``ingest`` is called from the service).

Cost notes
----------
With OpenAI's ``text-embedding-3-small`` (~$0.02 per 1M tokens), a typical
10-K filing of ~150K tokens costs roughly $0.003 to embed. Three filings
should run under $0.02. The chunking ratio is ~1 chunk per 1000 chars.
"""
from __future__ import annotations

# ─── sys.path bootstrap ───────────────────────────────────────────────────
# Allow ``python scripts/ingest_documents.py`` to find the project packages
# without the user having to set PYTHONPATH.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ──────────────────────────────────────────────────────────────────────────

import argparse
import time

from loguru import logger

from backend.core.logging import configure_logging
from ml_engine.rag.ingestion import ingest_paths
from ml_engine.rag.vectorstore import VectorStoreManager


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documents (PDF/TXT/MD/CSV) into the FAISS vectorstore.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more file or directory paths to ingest",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't recurse into subdirectories when given a folder",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the existing vectorstore before ingesting (destructive)",
    )
    args = parser.parse_args()

    configure_logging()

    # Validate paths up front so we fail fast on typos.
    missing: list[str] = []
    for raw in args.paths:
        if not Path(raw).expanduser().resolve().exists():
            missing.append(raw)
    if missing:
        for m in missing:
            logger.error(f"Path does not exist: {m}")
        sys.exit(2)

    # Optional destructive reset (useful when re-ingesting after schema changes).
    if args.reset:
        logger.warning("--reset specified; dropping existing vectorstore")
        VectorStoreManager().reset()

    logger.info(f"Starting ingestion of {len(args.paths)} path(s)")
    started = time.perf_counter()
    stats = ingest_paths(args.paths, recursive=not args.no_recursive)
    elapsed = time.perf_counter() - started

    logger.info(
        f"Done in {elapsed:.1f}s — "
        f"documents={stats['documents_added']}, "
        f"chunks={stats['chunks_added']}, "
        f"vectorstore_size={stats['vectorstore_size']}"
    )


if __name__ == "__main__":
    main()
