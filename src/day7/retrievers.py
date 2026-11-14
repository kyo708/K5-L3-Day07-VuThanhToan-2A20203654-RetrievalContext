"""Retrieval methods for Day 7. All return the same result schema:

    {id, text, source_id, source, page, metadata, score, rank, method}

Methods: dense (Day 6 pgvector + e5-base), BM25 (syllable or diacritic-folded
tokens), hybrid RRF (dense + BM25), and cross-encoder rerank on top of any base.
None of them sees benchmark gold.
"""
from __future__ import annotations

import math
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

FILTER_KEYS = {"chapter", "page"}


class Retriever(Protocol):
    name: str

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        ...


def _check_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    filters = dict(filters or {})
    bad = set(filters) - FILTER_KEYS
    if bad:
        raise ValueError(f"filter không hỗ trợ: {sorted(bad)}")
    return filters


def _passes(chunk: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, value in filters.items():
        actual = chunk["page"] if key == "page" else chunk["metadata"].get(key)
        if actual != value:
            return False
    return True


def _ranked(rows: list[tuple[float, dict[str, Any]]], method: str, top_k: int) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda r: (-r[0], r[1]["id"]))[:top_k]
    return [{**c, "score": round(float(s), 6), "rank": i, "method": method} for i, (s, c) in enumerate(rows, 1)]


# ---------------------------------------------------------------- dense (Day 6)
class DenseRetriever:
    """Exact cosine search in the Day 6 pgvector table, query embedded with the same e5 model."""

    def __init__(self, chunks: list[dict[str, Any]], strategy: str = "B",
                 database_url: str | None = None, model_name: str | None = None) -> None:
        from dotenv import load_dotenv
        from day6.embeddings import SentenceTransformerEmbedder
        from day6.store import PgVectorStore

        load_dotenv(ROOT / ".env")
        self.name = "dense"
        self.strategy = strategy
        self.embedder = SentenceTransformerEmbedder(model_name or os.environ["EMBEDDING_MODEL"])
        self.store = PgVectorStore(database_url or os.environ["DATABASE_URL"], self.embedder.dimensions)
        self.by_id = {c["id"]: c for c in chunks}
        indexed = self.store.count(strategy)
        if indexed != len(chunks):
            raise RuntimeError(f"vector store có {indexed} chunk strategy {strategy}, JSON có {len(chunks)}")

    def search(self, query, *, filters=None, top_k=3):
        filters = _check_filters(filters)
        rows = self.store.search(self.strategy, self.embedder.embed_query(query), top_k=top_k, filters=filters)
        out = []
        for r in rows:
            chunk = self.by_id[r["chunk_id"]]  # same IDs as Day 6 table; KeyError = index mismatch
            out.append({**chunk, "score": r["score"], "rank": r["rank"], "method": self.name})
        return out


# ---------------------------------------------------------------- lexical
def fold(text: str) -> str:
    """Remove Vietnamese diacritics (đ -> d) to tolerate OCR tone-mark errors."""
    text = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def tokenize(text: str, *, folded: bool = False, bigrams: bool = False) -> list[str]:
    text = unicodedata.normalize("NFC", text).casefold()
    if folded:
        text = fold(text)
    tokens = re.findall(r"\w+", text)
    if bigrams:  # Vietnamese words are often two syllables ("hợp đồng", "tiền lương")
        tokens = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    return tokens


class BM25Retriever:
    """Okapi BM25 over syllable tokens; `folded=True` strips diacritics first."""

    def __init__(self, chunks: list[dict[str, Any]], *, folded: bool = False, bigrams: bool = False,
                 k1: float = 1.5, b: float = 0.75) -> None:
        self.name = "bm25" + ("_fold" if folded else "") + ("_bi" if bigrams else "")
        self.chunks, self.folded, self.bigrams, self.k1, self.b = chunks, folded, bigrams, k1, b
        self.tf = [Counter(tokenize(c["text"], folded=folded, bigrams=bigrams)) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avgdl = sum(self.len) / len(self.len)
        df = Counter(term for t in self.tf for term in t)
        n = len(chunks)
        self.idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}

    def search(self, query, *, filters=None, top_k=3):
        filters = _check_filters(filters)
        terms = [t for t in tokenize(query, folded=self.folded, bigrams=self.bigrams) if t in self.idf]
        rows = []
        for chunk, tf, dl in zip(self.chunks, self.tf, self.len):
            if not _passes(chunk, filters):
                continue
            score = 0.0
            for term in terms:
                f = tf.get(term)
                if f:
                    score += self.idf[term] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if score > 0:
                rows.append((score, chunk))
        return _ranked(rows, self.name, top_k)


# ---------------------------------------------------------------- hybrid
class HybridRRF:
    """Reciprocal Rank Fusion: score = sum 1/(k + rank) over retrievers (rank-based, no score scaling)."""

    def __init__(self, retrievers: list[Retriever], *, k: int = 60, candidates: int = 30) -> None:
        self.name = "hybrid_rrf(" + "+".join(r.name for r in retrievers) + ")"
        self.retrievers, self.k, self.candidates = retrievers, k, candidates

    def search(self, query, *, filters=None, top_k=3):
        fused: dict[str, float] = {}
        chunks: dict[str, dict[str, Any]] = {}
        for retriever in self.retrievers:
            for r in retriever.search(query, filters=filters, top_k=self.candidates):
                fused[r["id"]] = fused.get(r["id"], 0.0) + 1.0 / (self.k + r["rank"])
                chunks[r["id"]] = {k: v for k, v in r.items() if k not in ("score", "rank", "method")}
        return _ranked([(s, chunks[i]) for i, s in fused.items()], self.name, top_k)


# ---------------------------------------------------------------- rerank
class CrossEncoderReranker:
    """Take `candidates` from a base retriever and re-score (query, passage) with a cross-encoder."""

    def __init__(self, base: Retriever, model_name: str = "BAAI/bge-reranker-v2-m3", *,
                 candidates: int = 20, device: str = "cpu", max_length: int = 512) -> None:
        from sentence_transformers import CrossEncoder

        self.name = f"rerank({base.name})"
        self.base, self.candidates, self.model_name = base, candidates, model_name
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)

    def search(self, query, *, filters=None, top_k=3):
        pool = self.base.search(query, filters=filters, top_k=self.candidates)
        if not pool:
            return []
        scores = self.model.predict([(query, c["text"]) for c in pool], batch_size=8, show_progress_bar=False)
        rows = [(float(s), {k: v for k, v in c.items() if k not in ("score", "rank", "method")} | {"base_rank": c["rank"]})
                for s, c in zip(scores, pool)]
        return _ranked(rows, self.name, top_k)
