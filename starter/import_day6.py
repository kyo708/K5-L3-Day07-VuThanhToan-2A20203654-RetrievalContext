"""Chuyển chunk Day 6 (results/chunks.jsonl) sang schema chunk Day 7 (JSON array).

Chỉ đổi tên trường; không đổi text, ID hay trang. `id` giữ nguyên `chunk_id` Day 6
nên khớp khóa chính trong bảng pgvector `chunks_768`.

    python starter/import_day6.py --day6-chunks starter/data/day6/chunks.jsonl \
        --manifest starter/data/day6/manifest.json --output starter/data/day6
"""
import argparse
import json
from collections import Counter
from pathlib import Path

DOCUMENT_ID = "blld-2019"
# Giá trị `source` trùng cột `source` trong vector store Day 6 (đường dẫn tương đối của starter Day 6).
SOURCE = "starter/data/blld.pdf"


def convert(row: dict, version: str | None) -> dict:
    return {
        "id": row["chunk_id"],
        "text": row["text"],
        "source_id": DOCUMENT_ID,
        "source": SOURCE,
        "page": row["page"],
        "metadata": {
            "strategy": row["strategy"],
            "chapter": row.get("chapter"),
            "article": row.get("article"),
            "start": row.get("start"),
            "char_count": row.get("char_count"),
            "version": version,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--day6-chunks", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text(encoding="utf-8"))
    src = next(s for s in manifest["sources"] if s["source_id"] == DOCUMENT_ID)
    version = src.get("version")
    rows = [json.loads(line) for line in a.day6_chunks.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [r["chunk_id"] for r in rows]
    dup = [k for k, n in Counter(ids).items() if n > 1]
    if dup:
        raise ValueError(f"chunk_id trùng: {dup[:5]}")
    a.output.mkdir(parents=True, exist_ok=True)
    by_strategy: dict[str, list[dict]] = {}
    for r in rows:
        by_strategy.setdefault(r["strategy"], []).append(convert(r, version))
    for strategy, chunks in sorted(by_strategy.items()):
        out = a.output / f"chunks_{strategy}.json"
        out.write_text(json.dumps(chunks, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        pages = {c["page"] for c in chunks}
        print(f"{out}: {len(chunks)} chunks, {len(pages)} trang")


if __name__ == "__main__":
    main()
