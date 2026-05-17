"""
CLI entry-point for ingesting a folder of documents into the RAG vectorstore.

Example:
    python -m scripts.ingest_documents ./data/raw/10ks --recursive
"""
from __future__ import annotations

import argparse
import sys

from loguru import logger

from ml_engine.rag.ingestion import ingest_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into the FAISS vectorstore.")
    parser.add_argument("paths", nargs="+", help="Files or directories to ingest")
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recurse into subdirectories (default: True)",
    )
    args = parser.parse_args()

    stats = ingest_paths(args.paths, recursive=args.recursive)
    logger.info(f"Ingestion complete: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
