"""Context selection: dedupe, keep rank order, optional neighbour chunks, char budget."""
from __future__ import annotations

from typing import Any


def format_block(chunk: dict[str, Any]) -> str:
    md = chunk.get("metadata", {})
    article = f" | Điều {md['article']}" if md.get("article") else ""
    return f"[{chunk['id']} | {chunk['source']} | trang {chunk['page']}{article}]\n{chunk['text']}"


def pack_context(results: list[dict[str, Any]], *, budget: int = 3000,
                 neighbours: int = 0, corpus: list[dict[str, Any]] | None = None) -> tuple[str, list[str]]:
    """Return (context text, chunk IDs in context).

    Results are taken in rank order, duplicates dropped. With `neighbours=n`, the n
    chunks before/after each hit (reading order) are inserted right after it — this
    recovers evidence that the chunker split across a boundary. A block that would
    exceed `budget` characters is skipped; smaller later blocks may still fit.
    """
    by_order = {c["metadata"]["order"]: c for c in corpus or []}
    ordered: list[dict[str, Any]] = []
    for r in results:
        ordered.append(r)
        if neighbours and corpus:
            o = r["metadata"]["order"]
            ordered += [by_order[i] for i in range(o - neighbours, o + neighbours + 1) if i != o and i in by_order]
    blocks, ids, used = [], [], 0
    for c in ordered:
        if c["id"] in ids:
            continue
        block = format_block(c)
        extra = len(block) + (2 if blocks else 0)
        if used + extra <= budget:
            blocks.append(block)
            ids.append(c["id"])
            used += extra
    return "\n\n".join(blocks), ids
