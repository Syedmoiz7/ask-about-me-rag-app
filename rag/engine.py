"""
The RAG engine: everything between "user asks a question" and "app returns an answer".

Flow for one question:
    1. Embed the question     -> a vector (list of 384 numbers) that captures its meaning
    2. Hybrid search          -> score chunks by meaning (FAISS) + exact keyword matches,
                                 keep the 3 best
    3. Build a prompt         -> instructions + those 3 chunks + the question
    4. Call the LLM           -> it answers using ONLY the chunks (or says "I don't know")

ingest.py (run once, ahead of time) creates the index this file searches.
"""

import json
import os
import re
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
CANDIDATES = 20  # how many chunks FAISS returns before re-scoring with keywords

# Hybrid search: final score = vector score + KEYWORD_WEIGHT * keyword score.
# The embedding model understands meaning ("built with" ~ "tech stack") but not
# proper names it never saw in training ("Kelsey", "ioMoVo"). Exact keyword
# matching covers that gap. 0.3 lets a full keyword match lift a weak vector
# score above MIN_SCORE without letting keywords alone dominate.
KEYWORD_WEIGHT = 0.3

# Chunks scoring below this are treated as "not relevant". If even the best
# chunk is below it, we skip the LLM and answer "I don't know" straight away.
MIN_SCORE = 0.25

# Common words that carry no topic, so they shouldn't count as keyword matches.
STOPWORDS = set("""
a an and are as at be by can did do does for from has have he her his how i in is it
its me my of on or she so tell that the their them they this to was what when where
which who why will with you your about any
""".split())

NOT_FOUND = "I don't know based on the documents."

SYSTEM_PROMPT = f"""You answer questions using ONLY the context provided below.
Rules:
- If the input is just a keyword or topic (e.g. "Kelsey"), summarize what the context says about it.
- If the context does not contain the answer, reply exactly: "{NOT_FOUND}"
- Do not use outside knowledge, even if you know the answer.
- Keep answers short and factual.
- Each context chunk starts with a number like [1]. Cite the chunks you used with those
  numbers right after the sentence they support, e.g. "It uses PostgreSQL [1]."
- Only cite chunks you actually used."""


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


def keywords(text: str) -> set[str]:
    """Lowercase words in the text, minus stopwords: 'Who is Kelsey?' -> {'kelsey'}."""
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def keyword_score(question_words: set[str], chunk: dict) -> float:
    """Fraction of the question's keywords that appear in the chunk (0.0 to 1.0)."""
    if not question_words:
        return 0.0
    chunk_words = keywords(f"{chunk['heading']} {chunk['text']}")
    return len(question_words & chunk_words) / len(question_words)


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """Hybrid search: find the k best chunks by meaning + keyword overlap."""
    index, chunks = load_index()
    question_words = keywords(question)

    # 1. Vector search: FAISS returns candidates ranked by meaning.
    scores, positions = index.search(embed([question]), min(CANDIDATES, index.ntotal))
    vector_scores = {int(pos): float(s) for s, pos in zip(scores[0], positions[0]) if pos != -1}

    # 2. Keyword search: also include any chunk that shares a keyword with the
    #    question, even if FAISS ranked it low (e.g. a project name).
    candidates = set(vector_scores) | {
        i for i, c in enumerate(chunks) if keyword_score(question_words, c) > 0
    }

    # 3. Combine both scores and keep the best k.
    results = []
    for i in candidates:
        vec = vector_scores.get(i, 0.0)
        kw = keyword_score(question_words, chunks[i])
        results.append({
            **chunks[i],
            "score": round(vec + KEYWORD_WEIGHT * kw, 3),
            "vector_score": round(vec, 3),
            "keyword_score": round(kw, 3),
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]


def build_prompt(question: str, sources: list[dict]) -> str:
    """Put the numbered chunks and the question into one message for the LLM.

    Numbering the chunks ([1], [2], [3]) lets the LLM cite exactly which ones it
    used, so the app can show only those as sources.
    """
    context = "\n\n".join(
        f"[{n}] ({s['source']} > {s['heading']})\n{s['text']}"
        for n, s in enumerate(sources, start=1)
    )
    return f"Context:\n{context}\n\nQuestion: {question}"


def cited_sources(answer_text: str, sources: list[dict]) -> list[dict]:
    """Return only the chunks the LLM cited, e.g. "[1]" or "[1, 3]" in the answer.

    Each returned chunk gets a "ref" number matching its citation in the answer.
    If the LLM cited nothing, we can't tell what it used, so all chunks are returned.
    """
    numbers = {int(n) for group in re.findall(r"\[([\d,\s]+)\]", answer_text)
               for n in re.findall(r"\d+", group)}
    numbered = [{**s, "ref": n} for n, s in enumerate(sources, start=1)]
    cited = [s for s in numbered if s["ref"] in numbers]
    return cited or numbered


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
        # The closest chunks are still returned (as "closest") so you can see why,
        # but they are NOT sources: nothing was used to answer.
        return {"answer": NOT_FOUND, "mode": "no-match", "sources": [], "closest": sources}

    client = get_llm_client()
    if client is None:
        # No API key: still useful, we show what retrieval found.
        return {"answer": None, "mode": "retrieval-only", "sources": relevant}

    text = generate(question, relevant, client)

    if text.startswith(NOT_FOUND):
        # The chunks were related, but the LLM found no answer in them (e.g.
        # "Does Moiz know Kubernetes?"). Nothing was used, so there are no sources.
        return {"answer": NOT_FOUND, "mode": "no-answer", "sources": [], "closest": relevant}

    return {"answer": text, "mode": "rag", "sources": cited_sources(text, relevant)}
