"""
RAG advisor — chat interface backed by the FAISS-indexed knowledge base.

Asks ``POST /api/v1/rag/query`` on each user message and renders the answer
with source citations. Conversation history is kept in ``st.session_state``
so the chat persists across reruns (Streamlit reruns the script on every
interaction, but session_state survives).

This page does NOT send the chat history to the backend on each turn — the
backend chain is currently stateless single-turn. The history is purely a UI
convenience that the user sees. Multi-turn context would require extending
the chain with a ConversationBufferMemory + reformulating each question
against prior turns; that's a Phase 6B enhancement.
"""
from __future__ import annotations

# ─── sys.path bootstrap ───────────────────────────────────────────────────
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ──────────────────────────────────────────────────────────────────────────

import uuid

import streamlit as st

from dashboard.components.api_client import APIError, get_client

st.set_page_config(page_title="Advisor", page_icon="💬", layout="wide")

st.title("💬 AI Financial Advisor")
st.caption(
    "Ask questions grounded in your indexed knowledge base (10-K filings, "
    "research reports, news). Answers cite their sources — if the corpus "
    "doesn't contain the answer, the model will say so rather than hallucinate. "
    "Not investment advice."
)


# ─── Session bootstrap ────────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[dict[role, content, sources?]]


# ─── Sidebar controls ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("Advisor settings")
    top_k = st.slider("Chunks to retrieve (top-k)", 2, 10, 6, step=1)
    st.caption("How many chunks the retriever pulls per question. More = richer context, slower + more tokens.")

    use_mmr = st.toggle(
        "Diversify retrieved chunks (MMR)",
        value=True,
        help=(
            "Maximum Marginal Relevance — actively penalizes redundancy "
            "across retrieved chunks. Best for comparison questions that "
            "should pull from multiple documents."
        ),
    )

    st.divider()

    if st.button("🗑️  Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.markdown("### Ingest")
    st.caption(
        "To add documents, drop PDFs/TXT/MD into `data/raw/` and run:\n\n"
        "```\npython scripts/ingest_documents.py data/raw/\n```\n\n"
        "The advisor automatically picks up new content."
    )


# ─── Render conversation history ──────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # If this assistant turn has sources, render them as expanders.
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                for i, src in enumerate(msg["sources"], start=1):
                    source_name = src.get("source", "unknown")
                    score = src.get("score")
                    score_str = f" • similarity={score:.3f}" if score is not None else ""
                    page = src.get("metadata", {}).get("page")
                    page_str = f" • page {page}" if page is not None else ""
                    st.markdown(f"**[{i}] {Path(source_name).name}**{page_str}{score_str}")
                    # Show first ~500 chars of the chunk
                    snippet = src.get("content", "")
                    if len(snippet) > 500:
                        snippet = snippet[:500] + "…"
                    st.markdown(f"> {snippet}")
                    if i < len(msg["sources"]):
                        st.divider()


# ─── Chat input ───────────────────────────────────────────────────────────

user_input = st.chat_input("Ask about a company's risk factors, revenue drivers, executive comments...")
if user_input:
    # 1. Append user message immediately so it renders before the API call.
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Call the backend and render the assistant reply.
    client = get_client()
    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            try:
                payload = client.rag_query({
                    "question": user_input,
                    "top_k": top_k,
                    "use_mmr": use_mmr,
                    "session_id": st.session_state.session_id,
                })
            except APIError as exc:
                err_msg = f"❌ Backend error ({exc.status_code}): {exc.message}"
                if exc.status_code == 500 and "OPENAI_API_KEY" in (exc.message or ""):
                    err_msg += (
                        "\n\n**Fix:** add `OPENAI_API_KEY=sk-...` to your `.env` "
                        "and restart the backend."
                    )
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
                st.stop()

        answer = payload["answer"]
        sources = payload.get("sources", [])

        st.markdown(answer)

        if sources:
            with st.expander(f"📚 Sources ({len(sources)})"):
                for i, src in enumerate(sources, start=1):
                    source_name = src.get("source", "unknown")
                    score = src.get("score")
                    score_str = f" • similarity={score:.3f}" if score is not None else ""
                    page = src.get("metadata", {}).get("page")
                    page_str = f" • page {page}" if page is not None else ""
                    st.markdown(f"**[{i}] {Path(source_name).name}**{page_str}{score_str}")
                    snippet = src.get("content", "")
                    if len(snippet) > 500:
                        snippet = snippet[:500] + "…"
                    st.markdown(f"> {snippet}")
                    if i < len(sources):
                        st.divider()

        # Token + model footer
        tokens = payload.get("tokens_used")
        model = payload.get("model", "?")
        footer_parts = [f"model: `{model}`"]
        if tokens:
            footer_parts.append(f"tokens: {tokens}")
        st.caption(" • ".join(footer_parts))

    # 3. Save assistant turn to history (so it persists across reruns).
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
