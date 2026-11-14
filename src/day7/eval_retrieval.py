"""Benchmark retrieval + context configs on one split (no LLM).

    python -m day7.eval_retrieval --split dev --configs all
    python -m day7.eval_retrieval --split test --configs dense_k3,<cấu hình đã chốt>   # only after locking

Run from repo root with PYTHONPATH=src. Writes results/retrieval/<split>/<config>.json and
results/retrieval/<split>/summary.json. Gold is used only to score, after retrieval.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from .context import pack_context
from .corpus import ROOT, evidence_covered, load_chunks, load_questions, public_question
from .retrievers import BM25Retriever, CrossEncoderReranker, DenseRetriever, HybridRRF

OUT = ROOT / "results" / "retrieval"

# name -> (description, retriever key, top_k used for context, context budget chars, neighbours)
CONFIGS: dict[str, tuple[str, str, int, int, int]] = {
    "dense_k3":         ("Dense e5-base trên pgvector Day 6, index B (baseline)", "dense", 3, 3000, 0),
    "bm25_k3":          ("BM25 âm tiết có dấu", "bm25", 3, 3000, 0),
    "bm25fold_k3":      ("BM25 bỏ dấu + bigram (giả thuyết: chịu lỗi dấu OCR)", "bm25fold", 3, 3000, 0),
    "hybrid_fold_k3":   ("RRF(dense, BM25 bỏ dấu), k=60, 30 candidate mỗi nhánh", "hybrid_fold", 3, 3000, 0),
    "hybrid_k3":        ("RRF(dense, BM25 có dấu), k=60, 30 candidate mỗi nhánh", "hybrid", 3, 3000, 0),
    "rerank_k3":        ("Hybrid RRF lấy 20 candidate -> bge-reranker-v2-m3", "rerank", 3, 3000, 0),
    "dense_k5":         ("Dense, top-5 vào context, budget 5000", "dense", 5, 5000, 0),
    "dense_k3_nb":      ("Dense top-3 + 1 chunk lân cận mỗi bên, budget 6000", "dense", 3, 6000, 1),
    "rerank_k3_nb":     ("Rerank top-3 + 1 chunk lân cận mỗi bên, budget 6000", "rerank", 3, 6000, 1),
    "hybrid_k3_nb":     ("Hybrid RRF top-3 + 1 chunk lân cận mỗi bên, budget 6000", "hybrid", 3, 6000, 1),
}


def build(keys: set[str], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    made: dict[str, Any] = {}
    if keys & {"dense", "hybrid", "hybrid_fold", "rerank"}:
        made["dense"] = DenseRetriever(chunks)
    if keys & {"bm25", "hybrid", "rerank"}:
        made["bm25"] = BM25Retriever(chunks)
    if keys & {"bm25fold", "hybrid_fold"}:
        made["bm25fold"] = BM25Retriever(chunks, folded=True, bigrams=True)
    if keys & {"hybrid", "rerank"}:
        made["hybrid"] = HybridRRF([made["dense"], made["bm25"]])
    if "hybrid_fold" in keys:
        made["hybrid_fold"] = HybridRRF([made["dense"], made["bm25fold"]])
    if "rerank" in keys:
        made["rerank"] = CrossEncoderReranker(made["hybrid"], candidates=20)
    return made


def slim(r: dict[str, Any]) -> dict[str, Any]:
    return {"id": r["id"], "page": r["page"], "article": r["metadata"].get("article"),
            "chapter": r["metadata"].get("chapter"), "score": r["score"], "rank": r["rank"],
            "base_rank": r.get("base_rank"), "text_preview": r["text"][:160]}


def run_config(name: str, retriever, questions, chunks) -> dict[str, Any]:
    desc, _, k, budget, nb = CONFIGS[name]
    traces = []
    for q in questions:
        pq = public_question(q)
        t0 = time.perf_counter()
        error = None
        try:
            results = retriever.search(pq["question"], filters=pq["filter"], top_k=10)
        except Exception as exc:  # record, do not replace with fake results
            results, error = [], f"{type(exc).__name__}: {exc}"
        ms = (time.perf_counter() - t0) * 1000
        context, ids = pack_context(results[:k], budget=budget, neighbours=nb, corpus=chunks)
        row = {"query_id": q["id"], "answerable": q["answerable"], "type": q["type"],
               "question": q["question"], "filter": pq["filter"], "error": error,
               "elapsed_ms": round(ms, 1), "results_top10": [slim(r) for r in results],
               "context_ids": ids, "context_chars": len(context)}
        if q["answerable"]:
            in_ctx = [c for c in chunks if c["id"] in set(ids)]
            ranks = [i for i in range(1, len(results) + 1) if evidence_covered(results[:i], q["gold"])]
            row.update(hit1=evidence_covered(results[:1], q["gold"]),
                       hit3=evidence_covered(results[:3], q["gold"]),
                       hit10=evidence_covered(results[:10], q["gold"]),
                       first_full_rank=ranks[0] if ranks else None,
                       context_evidence=evidence_covered(in_ctx, q["gold"]),
                       gold_pages=[g["page"] for g in q["gold"]])
        traces.append(row)
    ans = [t for t in traces if t["answerable"]]
    n = len(ans)

    def frac(key):
        v = sum(bool(t[key]) for t in ans)
        return {"n": v, "N": n, "rate": round(v / n, 3) if n else None}

    summary = {"config": name, "description": desc, "method": retriever.name, "context_top_k": k,
               "budget_chars": budget, "neighbours": nb, "split": questions[0]["split"],
               "n_questions": len(traces), "n_answerable": n,
               "hit1": frac("hit1"), "hit3": frac("hit3"), "hit10": frac("hit10"),
               "mrr10": round(sum(1 / t["first_full_rank"] for t in ans if t["first_full_rank"]) / n, 3) if n else None,
               "context_evidence_rate": frac("context_evidence"),
               "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces)),
               "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 1),
               "errors": sum(t["error"] is not None for t in traces), "cost_usd": 0.0,
               "misses_at3": [t["query_id"] for t in ans if not t["hit3"]]}
    return {"summary": summary, "traces": traces}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", required=True, choices=("dev", "test"))
    p.add_argument("--configs", default="all")
    p.add_argument("--strategy", default="B")
    a = p.parse_args(argv)
    names = list(CONFIGS) if a.configs == "all" else a.configs.split(",")
    chunks = load_chunks(a.strategy)
    questions = load_questions(a.split)
    t0 = time.perf_counter()
    retrievers = build({CONFIGS[n][1] for n in names}, chunks)
    load_s = time.perf_counter() - t0
    out = OUT / a.split
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name in names:
        res = run_config(name, retrievers[CONFIGS[name][1]], questions, chunks)
        (out / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        s = res["summary"]
        summaries.append(s)
        print(f"{name:22s} hit1 {s['hit1']['n']}/{s['hit1']['N']}  hit3 {s['hit3']['n']}/{s['hit3']['N']}  "
              f"hit10 {s['hit10']['n']}/{s['hit10']['N']}  mrr {s['mrr10']}  ctx {s['context_evidence_rate']['n']}/"
              f"{s['context_evidence_rate']['N']}  chars {s['avg_context_chars']}  {s['median_ms']}ms  miss@3 {s['misses_at3']}")
    meta = {"split": a.split, "index_strategy": a.strategy, "model_load_seconds": round(load_s, 1),
            "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "summaries": summaries}
    path = out / "summary.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"summaries": []}
    keep = [s for s in old["summaries"] if s["config"] not in names]
    meta["summaries"] = keep + summaries
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
