"""
Document ingestion for the RAG pipeline.

Supported sources out of the box:
  - .txt / .md   → TextLoader
  - .pdf         → PyPDFLoader  (install ``pypdf`` to enable)
  - .csv         → CSVLoader
  - .html        → UnstructuredHTMLLoader (optional, requires ``unstructured``)

Add new file types by mapping them in ``LOADER_REGISTRY``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from backend.core.config import settings
from ml_engine.rag.vectorstore import VectorStoreManager


# ─── Loader registry ──────────────────────────────────────────────────────


LoaderFactory = Callable[[str], "BaseLoaderLike"]  # noqa: F821 (typing alias only)


def _txt(path: str):
    return TextLoader(path, encoding="utf-8")


def _csv(path: str):
    return CSVLoader(path, encoding="utf-8")


def _pdf(path: str):
    return PyPDFLoader(path)


LOADER_REGISTRY: dict[str, LoaderFactory] = {
    ".txt": _txt,
    ".md": _txt,
    ".pdf": _pdf,
    ".csv": _csv,
}


def _loader_for(path: Path):
    factory = LOADER_REGISTRY.get(path.suffix.lower())
    if factory is None:
        raise ValueError(f"Unsupported file type: {path.suffix} ({path})")
    return factory(str(path))


# ─── Splitter ─────────────────────────────────────────────────────────────


def get_splitter() -> RecursiveCharacterTextSplitter:
    """Recursive character splitter tuned for financial prose.

    The defaults (1000 / 150) work well for 10-K / news article chunks. Use a
    larger window for long-form filings, smaller for tabular data.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


# ─── Public API ───────────────────────────────────────────────────────────


def load_documents(paths: Iterable[str], recursive: bool = True) -> list[Document]:
    """Load raw documents (pre-split) from a list of files or directories."""
    docs: list[Document] = []
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        if not p.exists():
            logger.warning(f"Skipping non-existent path: {p}")
            continue

        if p.is_dir():
            for ext, factory in LOADER_REGISTRY.items():
                glob = f"**/*{ext}" if recursive else f"*{ext}"
                loader = DirectoryLoader(
                    str(p),
                    glob=glob,
                    loader_cls=lambda path, _f=factory: _f(path),  # type: ignore[misc]
                    show_progress=False,
                    use_multithreading=False,
                )
                docs.extend(loader.load())
        else:
            docs.extend(_loader_for(p).load())

    logger.info(f"Loaded {len(docs)} raw document(s)")
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = get_splitter()
    chunks = splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunk(s) "
                f"(chunk_size={settings.rag_chunk_size}, overlap={settings.rag_chunk_overlap})")
    return chunks


def ingest_paths(paths: list[str], recursive: bool = True) -> dict[str, int]:
    """Full ingest pipeline: load → split → embed → persist.

    Returns counters suitable for the API response.
    """
    docs = load_documents(paths, recursive=recursive)
    chunks = split_documents(docs)
    manager = VectorStoreManager()
    added = manager.add_documents(chunks)
    return {
        "documents_added": len(docs),
        "chunks_added": added,
        "vectorstore_size": manager.size(),
    }
