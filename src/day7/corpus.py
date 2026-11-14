"""Corpus, benchmark loading and evidence matching for Day 7.

Gold evidence is only used here, by the evaluator. Retrievers and the agent never
import these helpers, so gold cannot leak into retrieval or generation.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
DAY6_DIR = ROOT / "starter" / "data" / "day6"
QUESTIONS_PATH = ROOT / "data" / "benchmark" / "questions.json"


def norm(text: str) -> str:
    """NFC + casefold + collapse whitespace; used only for evidence matching."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().casefold()


def load_chunks(strategy: str = "B") -> list[dict[str, Any]]:
    """Day 6 chunks converted to the Day 7 schema (same IDs as pgvector `chunks_768`)."""
    path = DAY6_DIR / f"chunks_{strategy}.json"
    chunks = json.loads(path.read_text(encoding="utf-8"))
    for order, chunk in enumerate(chunks):
        chunk["metadata"]["order"] = order  # reading order, used for neighbour expansion
    return chunks


def load_questions(split: str | None = None, path: Path = QUESTIONS_PATH) -> list[dict[str, Any]]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    return [q for q in questions if split is None or q["split"] == split]


def public_question(q: dict[str, Any]) -> dict[str, Any]:
    """What a retriever/agent may see: the question and a user-provided filter, never gold."""
    return {"id": q["id"], "question": q["question"], "filter": q.get("filter")}


def evidence_covered(chunks: Iterable[dict[str, Any]], gold: list[dict[str, Any]]) -> bool:
    """True if some gold alternative has every required quote inside a chunk of the right source/page."""
    chunks = list(chunks)
    for alt in gold:
        pool = [norm(c["text"]) for c in chunks
                if c["source_id"] == alt["source_id"] and c["page"] == alt["page"]]
        if pool and all(any(norm(quote) in text for text in pool) for quote in alt["evidence"]):
            return True
    return False


def validate_benchmark(questions: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> list[str]:
    """Return problems: duplicate IDs, bad split, gold quote not locatable in this index."""
    problems, seen = [], set()
    for q in questions:
        if q["id"] in seen:
            problems.append(f"{q['id']}: duplicate id")
        seen.add(q["id"])
        if q["split"] not in ("dev", "test"):
            problems.append(f"{q['id']}: bad split")
        if q["answerable"]:
            for alt in q["gold"]:
                page_chunks = [c for c in chunks if c["page"] == alt["page"] and c["source_id"] == alt["source_id"]]
                for quote in alt["evidence"]:
                    if not any(norm(quote) in norm(c["text"]) for c in page_chunks):
                        problems.append(f"{q['id']}: quote not inside a single chunk on page {alt['page']}: {quote[:50]}")
        elif not q.get("absence_reason"):
            problems.append(f"{q['id']}: unanswerable without absence_reason")
    return problems
