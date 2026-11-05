"""
The RAG engine: everything between "user asks a question" and "app returns an answer".

Flow for one question:
    1. Embed the question     -> a vector (list of 384 numbers) that captures its meaning
    2. Search the FAISS index -> the 3 chunks whose vectors are closest to the question's
    3. Build a prompt         -> instructions + those 3 chunks + the question
    4. Call the LLM           -> it answers using ONLY the chunks (or says "I don't know")

ingest.py (run once, ahead of time) creates the index this file searches.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"
INDEX_FILE = INDEX_DIR / "faiss.index"
CHUNKS_FILE = INDEX_DIR / "chunks.json"

# Small, fast embedding model that runs locally (no API key, no cost).
# It turns any text into a 384-number vector. Similar meaning -> similar vectors.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 3  # how many chunks to send to the LLM

# Chunks scoring below this are treated as "not relevant". Scores are cosine
# similarity: 1.0 = same meaning, ~0 = unrelated. If even the best chunk is below
# this, we skip the LLM and answer "I don't know" straight away.
MIN_SCORE = 0.25

NOT_FOUND = "I don't know based on the documents."

SYSTEM_PROMPT = f"""You answer questions using ONLY the context provided below.
Rules:
- If the context does not contain the answer, reply exactly: "{NOT_FOUND}"
- Do not use outside knowledge, even if you know the answer.
- Keep answers short and factual.
- After the answer, list the sources you used in square brackets, e.g. [web-projects.md]."""


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load the embedding model once and reuse it.

    Loading takes a few seconds, so doing it on every request would make each
    question slow. lru_cache keeps the loaded model in memory after the first call.
    """
    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    """Turn a list of texts into a matrix of vectors (one row per text).

    normalize_embeddings=True scales every vector to length 1. That makes the
    dot product of two vectors equal to their cosine similarity, which is what
    the FAISS index below compares.
    """
    vectors = get_embedder().encode(texts, normalize_embeddings=True)
    return np.asarray(vectors, dtype="float32")  # FAISS requires float32


@lru_cache(maxsize=1)
def load_index() -> tuple[faiss.Index, list[dict]]:
    """Load the FAISS index and the chunk texts saved by ingest.py.

    FAISS only stores vectors (numbers), not text. chunks.json keeps the text and
    source file for each vector, in the same order, so position i in the index
    matches chunks[i].
    """
    if not INDEX_FILE.exists() or not CHUNKS_FILE.exists():
        raise FileNotFoundError("Index not found. Run `python ingest.py` first.")
    index = faiss.read_index(str(INDEX_FILE))
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    return index, chunks


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """Find the k chunks most similar in meaning to the question."""
    index, chunks = load_index()
    query_vector = embed([question])
    scores, positions = index.search(query_vector, k)

    results = []
    for score, pos in zip(scores[0], positions[0]):
        if pos == -1:  # FAISS returns -1 when there are fewer than k chunks
            continue
        results.append({**chunks[pos], "score": round(float(score), 3)})
    return results


def build_prompt(question: str, sources: list[dict]) -> str:
    """Put the retrieved chunks and the question into one message for the LLM."""
    context = "\n\n".join(
        f"[{s['source']}] {s['heading']}\n{s['text']}" for s in sources
    )
    return f"Context:\n{context}\n\nQuestion: {question}"


def get_llm_client() -> OpenAI | None:
    """Create an LLM client from .env settings, or None if no key is set.

    Uses the `openai` library, but LLM_BASE_URL lets it talk to any
    OpenAI-compatible API (OpenAI itself, Google Gemini, Groq, ...).
    """
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        return None
    if not os.getenv("LLM_MODEL"):
        raise RuntimeError("LLM_API_KEY is set but LLM_MODEL is missing in .env")
    base_url = os.getenv("LLM_BASE_URL") or None  # None = OpenAI's default URL
    return OpenAI(api_key=api_key, base_url=base_url)


def generate(question: str, sources: list[dict], client: OpenAI) -> str:
    """Ask the LLM to answer using only the retrieved chunks."""
    response = client.chat.completions.create(
        model=os.environ["LLM_MODEL"],  # set in .env, e.g. a small Gemini or OpenAI model
        temperature=0,  # 0 = least creative, most consistent; good for factual answers
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, sources)},
        ],
    )
    return response.choices[0].message.content.strip()


def answer(question: str) -> dict:
    """The full RAG pipeline: retrieve, then (if possible) generate."""
    sources = retrieve(question)
    relevant = [s for s in sources if s["score"] >= MIN_SCORE]

    if not relevant:
        # Nothing in the documents is close to the question: don't even ask the LLM.
        return {"answer": NOT_FOUND, "mode": "no-match", "sources": sources}

    client = get_llm_client()
    if client is None:
        # No API key: still useful, we show what retrieval found.
        return {"answer": None, "mode": "retrieval-only", "sources": relevant}

    return {"answer": generate(question, relevant, client), "mode": "rag", "sources": relevant}
