# Ask My Docs

A small **RAG (Retrieval-Augmented Generation)** app built with **Python, Django, FAISS,
sentence-transformers and an LLM API**. Ask a question about the documents in `data/`. The app
finds the most relevant passages and has an LLM answer **using only those passages**, showing
the sources it used. If the documents don't contain the answer, it says so instead of guessing.

The sample documents describe my own projects and tech stack, so you can ask things like
*"What was the Kelsey Scholarship Portal built with?"* or *"Which mobile apps has Moiz built?"*

## Architecture

```
 INGEST (run once: python ingest.py)
 ───────────────────────────────────
 data/*.md ──► split into chunks ──► embed each chunk ──► FAISS index + chunks.json
               (by markdown heading)  (all-MiniLM-L6-v2,     (saved in index/)
                                       384-dim vectors)

 QUESTION (POST /ask)
 ────────────────────
 question ──► hybrid search: ─────────► top 3 ──► score ≥ 0.25? ──no──► "I don't know
              meaning (FAISS vectors)    chunks            │              based on the
              + exact keyword matches                     yes             documents."
                                                              ▼
                                   prompt = rules + 3 chunks + question
                                                              ▼
                                   LLM (OpenAI-compatible API) ──► answer + sources
```

**Why RAG?** An LLM doesn't know about my private documents, and it can make things up. RAG
fixes both: we *retrieve* the relevant text first, then the LLM *generates* an answer grounded
in that text. Showing the source chunks lets the user check the answer.

## How it works

| Step | Where | What happens |
|---|---|---|
| Chunking | `ingest.py` | Each markdown file is split on headings, so each chunk covers one topic. Each chunk's heading includes the document title (e.g. "About Moiz Tahir > Education"), which raised the score for "Where did Moiz study?" from 0.15 to 0.42 in testing. Sections over 800 characters are split again with 100 characters of overlap so nothing is lost at the cut. |
| Embedding | `rag/engine.py` → `embed()` | `all-MiniLM-L6-v2` turns text into a 384-number vector. Texts with similar meaning get similar vectors, even with different words ("built with" ≈ "tech stack"). It runs locally: free, no API key. |
| Indexing | `ingest.py` | Vectors are normalized and stored in a FAISS `IndexFlatIP`. With normalized vectors, inner product = cosine similarity. "Flat" = exact brute-force search, ideal for a small dataset. FAISS stores only numbers, so `chunks.json` keeps the matching text in the same order. |
| Retrieval (hybrid) | `engine.retrieve()` | The question is embedded with the same model and FAISS ranks chunks by meaning. Each chunk also gets a keyword score: the share of the question's words (minus stopwords like "the", "is") that appear in it. Final score = vector score + 0.3 × keyword score; the top 3 are kept. |
| Guardrail | `engine.answer()` | If even the best chunk scores below `0.25`, the app answers "I don't know" without calling the LLM. |
| Generation | `engine.generate()` | The numbered chunks ([1], [2], [3]) and the question go to the LLM with strict rules: use only the context, cite chunk numbers, otherwise reply "I don't know based on the documents." `temperature=0` keeps answers consistent. |
| Citations | `engine.cited_sources()` | The app reads the citation numbers in the answer and returns only those chunks as sources, so retrieved-but-unused chunks aren't shown as sources. |
| API | `rag/views.py` | `POST /ask` validates the input, calls the engine and returns JSON. Errors return clear messages with proper HTTP status codes (400, 502, 503). |

### Design decisions

- **Why hybrid search?** Embedding models understand meaning but not proper names they never
  saw in training. In testing, the one-word query "Kelsey" scored only 0.246 on vectors alone
  (below the cutoff), and "Ask It World" matched the *Education* chunk instead of the app.
  Adding keyword matching fixed both (Kelsey → 0.546, Ask It World → correct chunk), while
  off-topic questions like "pizza" still score under 0.1 and are blocked. This is the same
  idea production systems use when they combine BM25 keyword search with vector search.
- **Only cited chunks are shown as sources.** Retrieval returns up to 3 chunks, but the LLM
  doesn't always use all of them. Showing only cited ones keeps sources honest. In the UI they
  start collapsed under the answer and expand to show the full text and score breakdown.
- **Two safety layers against made-up answers.** The score cutoff blocks clearly unrelated
  questions without calling the LLM. Near-misses that pass it (e.g. "Does Moiz know
  Kubernetes?", or "what time is it" matching "real-time") are caught by the prompt, and the
  LLM answers "I don't know based on the documents."

- **Engine separate from Django.** `rag/engine.py` has no Django code, so the RAG logic can be
  tested or reused (CLI, another framework) on its own. Views only deal with HTTP.
- **Models load once.** The embedding model and index are cached in memory (`lru_cache`) after
  the first request. Loading them on every request would add several seconds per question.
- **Any OpenAI-compatible LLM.** It uses the `openai` library, and `LLM_BASE_URL` in `.env`
  switches providers (OpenAI, Google Gemini, Groq, ...) without code changes.
- **Works without a key.** With no `LLM_API_KEY`, it runs in *retrieval-only* mode and returns
  the matching chunks. That's useful for testing retrieval on its own.
- **No database.** The data lives in the FAISS index, so Django runs with a minimal config
  (no ORM, admin or auth).

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/Syedmoiz7/ask-my-docs.git
cd ask-my-docs
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # includes PyTorch, so this can take a few minutes

cp .env.example .env                # then add your API key (optional)
python ingest.py                    # builds the index (downloads the ~90 MB embedding model once)
python manage.py runserver
```

Open http://127.0.0.1:8000, or call the API directly:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the Kelsey Scholarship Portal built with?"}'
```

Example response (the exact answer wording depends on the LLM):

```json
{
  "answer": "The Kelsey Scholarship Portal was built with Next.js, Node.js, Express, PostgreSQL, WebSockets, JWT authentication, and Tailwind CSS [1].",
  "mode": "rag",
  "sources": [
    {"ref": 1, "source": "web-projects.md", "heading": "Web projects > Kelsey Scholarship Portal", "text": "...", "score": 0.862, "vector_score": 0.562, "keyword_score": 1.0}
  ]
}
```

To use your own documents, put `.md` files in `data/` and run `python ingest.py` again.

## Limitations and next steps

- **Similarity threshold is hand-tuned.** `0.25` works for these documents; a larger dataset
  would need an evaluation set of questions to tune it properly.
- **Top 3 isn't enough for "list everything" questions.** "Which mobile apps has Moiz built?"
  matches 3 chunks, but the apps are spread across more. Fixes: a higher or dynamic `k`, or
  summary chunks per document.
- **Simple keyword matching.** Every keyword counts equally. A real system would use BM25,
  which gives rare words ("Kelsey") more weight than common ones ("time").
- **Flat index.** Brute-force search is exact but slows down past roughly a million vectors.
  At that scale I'd use an approximate index (FAISS IVF/HNSW) or a vector database such as
  Qdrant or Pinecone.
- **Markdown only.** Next step: PDF support, with the same pipeline after text extraction.
- **No conversation memory.** Each question is independent.
- **Possible upgrades:** BM25 keyword scoring, re-ranking retrieved chunks, streaming
  responses, and exposing the search as an MCP tool so AI agents can query the documents.
