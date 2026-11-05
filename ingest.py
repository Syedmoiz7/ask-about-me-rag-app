"""
Ingest pipeline: turn the markdown files in data/ into a searchable FAISS index.

Run it once, and again whenever the documents change:
    python ingest.py

Steps:
    1. Load    every .md file in data/
    2. Chunk   split each file into small pieces (one per markdown section)
    3. Embed   turn every chunk into a vector with the local embedding model
    4. Index   put the vectors in FAISS and save the index + chunk texts to index/
"""

import json
import re

import faiss

from rag.engine import CHUNKS_FILE, DATA_DIR, INDEX_DIR, INDEX_FILE, embed

# Chunks longer than this get split further. Small chunks keep retrieval precise:
# the LLM receives only the relevant part, not a whole document.
MAX_CHARS = 800
OVERLAP = 100  # characters repeated between split pieces so a sentence cut in half isn't lost


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown on headings (#, ##, ...) into (heading, body) pairs.

    Splitting on headings keeps each chunk about one topic, which gives better
    search results than cutting the text at arbitrary positions.
    """
    sections = []
    heading, lines = "", []
    for line in text.splitlines():
        if re.match(r"^#{1,6}\s", line):
            if "".join(lines).strip():
                sections.append((heading, "\n".join(lines).strip()))
            heading, lines = line.lstrip("#").strip(), []
        else:
            lines.append(line)
    if "".join(lines).strip():
        sections.append((heading, "\n".join(lines).strip()))
    return sections


def split_long_text(text: str) -> list[str]:
    """Cut text longer than MAX_CHARS into overlapping windows."""
    if len(text) <= MAX_CHARS:
        return [text]
    pieces, start = [], 0
    while start < len(text):
        pieces.append(text[start : start + MAX_CHARS])
        start += MAX_CHARS - OVERLAP
    return pieces


def load_chunks() -> list[dict]:
    """Read every markdown file and return a flat list of chunks with their source."""
    chunks = []
    for path in sorted(DATA_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        # The document title (first "# " heading) gives every chunk context:
        # "Education" alone is vague, "About Moiz Tahir > Education" is clear.
        title = next((l[2:].strip() for l in text.splitlines() if l.startswith("# ")), path.stem)
        for heading, body in split_into_sections(text):
            full_heading = title if heading in ("", title) else f"{title} > {heading}"
            for piece in split_long_text(body):
                chunks.append({"source": path.name, "heading": full_heading, "text": piece})
    return chunks


def main() -> None:
    chunks = load_chunks()
    if not chunks:
        raise SystemExit(f"No markdown files found in {DATA_DIR}")
    print(f"Loaded {len(chunks)} chunks from {DATA_DIR.name}/")

    # Include the heading in what we embed: "Kelsey Scholarship Portal" in the
    # heading helps match questions that mention the project by name.
    vectors = embed([f"{c['heading']}\n{c['text']}" for c in chunks])
    print(f"Embedded chunks into vectors of size {vectors.shape[1]}")

    # IndexFlatIP = compare the question against every vector using the inner
    # (dot) product. Our vectors are normalized, so this equals cosine similarity.
    # "Flat" means exact, brute-force search: perfect for a few hundred chunks.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    CHUNKS_FILE.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    print(f"Saved index with {index.ntotal} vectors to {INDEX_DIR.name}/")


if __name__ == "__main__":
    main()
