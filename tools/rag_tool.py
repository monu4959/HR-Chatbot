"""
tools/rag_tool.py

Semantic search using sentence-transformers/all-MiniLM-L6-v2 (local, free)
stored in ChromaDB. CRAG scoring filters low-relevance results.
"""

import os
import re
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from langchain_core.tools import tool
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

DATA_PATH   = os.getenv("DATA_PATH",   "./data/annotated_results.csv")
CHROMA_DIR  = os.getenv("CHROMA_PATH", "./chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"
COLL_NAME   = "recognition_records"
TOP_K       = 20
CRAG_THRESHOLD = float(os.getenv("CRAG_THRESHOLD", "0.25"))

_collection  = None
_embedder    = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_collection():
    global _collection
    if _collection is not None:
        return _collection

    import chromadb
    client    = chromadb.PersistentClient(path=str(CHROMA_DIR))
    existing  = [c.name for c in client.list_collections()]
    data_hash = hashlib.sha256(Path(DATA_PATH).read_bytes()).hexdigest()[:16]

    if COLL_NAME in existing:
        coll = client.get_collection(name=COLL_NAME)
        if (coll.metadata or {}).get("data_hash") == data_hash:
            _collection = coll
            return _collection
        print("CSV changed since the index was built — rebuilding ChromaDB index...")
        client.delete_collection(name=COLL_NAME)

    # Build index from CSV (once per CSV version)
    print("Building ChromaDB index (one-time)...")
    df = pd.read_csv(DATA_PATH)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("").astype(str)

    embedder = _get_embedder()
    chunks = (
        "Award: "       + df["award_title"]       + " | " +
        "Recipient: "   + df["recipient_title"]   + " | " +
        "Nominator: "   + df["nominator_title"]   + " | " +
        "Category: "    + df["category_name"]     + " | " +
        "Subcategory: " + df["subcategory_name"]  + " | " +
        "Message: "     + df["message"]
    ).tolist()

    embeddings = embedder.encode(chunks, show_progress_bar=True).tolist()
    metadatas  = [
        {
            "category_name":    row["category_name"],
            "subcategory_name": row["subcategory_name"],
            "recipient_title":  row["recipient_title"],
            "nominator_title":  row["nominator_title"],
            "award_title":      row["award_title"],
        }
        for _, row in df.iterrows()
    ]

    _collection = client.create_collection(
        name=COLL_NAME,
        metadata={"hnsw:space": "cosine", "data_hash": data_hash},
    )
    batch = 100
    for i in range(0, len(chunks), batch):
        _collection.add(
            ids        = [str(j) for j in range(i, min(i + batch, len(chunks)))],
            documents  = chunks[i:i+batch],
            embeddings = embeddings[i:i+batch],
            metadatas  = metadatas[i:i+batch],
        )
    print(f"ChromaDB index built with {_collection.count()} records.")
    return _collection


# Words that say *how* to search, not *what* to search for — they must not count as relevance evidence
_FILLER = {
    "find", "show", "give", "list", "get", "tell", "what", "which", "who", "where", "when", "how",
    "the", "and", "for", "with", "about", "any", "all", "some", "that", "this", "those", "these",
    "from", "into", "are", "was", "were", "has", "have", "had", "did", "does", "can", "could",
    "messages", "message", "recognitions", "recognition", "awards", "award", "mention", "mentions",
    "mentioning", "related", "involving", "people", "person", "someone", "examples", "example",
    "please", "named", "times", "there",
}


def _content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _FILLER]


def _stem(word: str) -> str:
    """Crude prefix stem so 'mentorship' matches 'mentoring' and 'migrations' matches 'migrated'."""
    return word[:6] if len(word) > 6 else word


def _keyword_overlap(query: str, doc: str) -> float:
    words = set(_content_words(query))
    if not words:
        return 0.0
    d = doc.lower()
    return sum(1 for w in words if _stem(w) in d) / len(words)


@traceable(name="crag_score")
def _crag_score(query: str, docs: list[str], distances: list[float]) -> list[float]:
    """CRAG relevance: half embedding similarity (1 - cosine distance), half content-keyword overlap."""
    return [round(0.5 * (1 - dist) + 0.5 * _keyword_overlap(query, doc), 3)
            for doc, dist in zip(docs, distances)]


@tool
def semantic_search(query: str) -> str:
    """
    Use this tool to search recognition messages semantically.
    Best for finding recognitions about specific themes, topics, behaviors,
    or when the user wants to find similar messages or examples.

    Examples:
    - Find recognitions about cross-team collaboration
    - Show me awards related to mentorship
    - Find messages about saving time or reducing costs
    - What recognitions mention NPS or customer satisfaction?
    - Find awards involving a person named Sarah
    """
    collection = _get_collection()
    embedder   = _get_embedder()

    query_vec = embedder.encode([query])[0].tolist()

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not docs:
        return f"SEARCH COMPLETE: 0 results found for '{query}'."

    # CRAG scoring
    crag_scores = _crag_score(query, docs, distances)
    scored      = list(zip(docs, metadatas, distances, crag_scores))
    passing     = [(d, m, dist, s) for d, m, dist, s in scored if s >= CRAG_THRESHOLD]
    low_quality = len(scored) - len(passing)

    # Retry with just the content words if too few results pass
    if len(passing) < 3:
        broad = " ".join(_content_words(query))
        if broad and broad != query.lower():
            retry = collection.query(
                query_embeddings=[embedder.encode([broad])[0].tolist()],
                n_results=TOP_K,
                include=["documents", "metadatas", "distances"],
            )
            r_docs  = retry["documents"][0]
            r_metas = retry["metadatas"][0]
            r_dists = retry["distances"][0]
            r_crag  = _crag_score(query, r_docs, r_dists)
            retry_passing = [(d, m, dist, s) for d, m, dist, s in zip(r_docs, r_metas, r_dists, r_crag) if s >= CRAG_THRESHOLD]
            if len(retry_passing) > len(passing):
                passing = retry_passing

    if not passing:
        passing = [(docs[0], metadatas[0], distances[0], crag_scores[0])]

    avg_score = round(sum(s for _, _, _, s in passing) / len(passing), 3)
    quality   = "High" if avg_score >= 0.6 else "Medium" if avg_score >= 0.35 else "Low"

    output = (
        f"SEARCH COMPLETE: Found {len(passing)} relevant recognitions for '{query}' "
        f"(CRAG quality: {quality}, avg score: {avg_score}). "
        f"Present ALL of these to the user:\n\n"
    )

    for i, (doc, meta, dist, score) in enumerate(passing, 1):
        msg_preview = doc.split("Message:")[-1].strip()[:400] if "Message:" in doc else doc[:400]
        output += f"RESULT {i} [relevance: {round(score, 3)}]:\n"
        output += f"  Recipient: {meta.get('recipient_title', 'N/A')}\n"
        output += f"  Nominator: {meta.get('nominator_title', 'N/A')}\n"
        output += f"  Category: {meta.get('category_name', 'N/A')} > {meta.get('subcategory_name', 'N/A')}\n"
        output += f"  Message: {msg_preview}\n\n"

    if low_quality > 0:
        output += f"Note: {low_quality} additional results were below the relevance threshold and excluded.\n"

    output += f"INSTRUCTION: Present all {len(passing)} results above to the user."
    return output
