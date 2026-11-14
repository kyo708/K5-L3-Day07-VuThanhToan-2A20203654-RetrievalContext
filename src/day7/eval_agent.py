"""Run the tool-calling agent with one or more OpenRouter models on one split.

    python -m day7.eval_agent --list
    python -m day7.eval_agent --split dev --agents gemma-4-31b-free --limit 3          # smoke test
    python -m day7.eval_agent --split dev --agents all --retriever hybrid
    python -m day7.eval_agent --split dev --agents openai/gpt-4o-mini                  # any raw OpenRouter id (paid ids cost money)

Run from repo root with PYTHONPATH=src and OPENROUTER_API_KEY in .env. Writes
results/agent_models/<split>/<agent>.json (full traces) and summary.json there.

Retriever, tools, prompt, top-k and budget are identical for every agent; only the
model changes. The agent only sees `public_question` (question + user filter). Gold
is used afterwards for the automatic Context Evidence Rate only. These numbers are
NOT answer quality: citations still need a human audit (starter/agent_metrics.py).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

from .agents import OPENROUTER_AGENTS, get_spec, make_agent
from .corpus import ROOT, evidence_covered, load_chunks, load_questions, public_question
from .eval_retrieval import build

OUT = ROOT / "results" / "agent_models"
RETRIEVERS = ("dense", "bm25", "bm25fold", "hybrid", "hybrid_fold", "rerank")


def run_one(agent, q: dict[str, Any], chunks_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pq = public_question(q)
    t0 = time.perf_counter()
    try:
        out = agent.run(pq["question"], filters=pq["filter"])
    except Exception as exc:  # provider / vector store error: record it, never fake an answer
        error = f"{type(exc).__name__}: {exc}"
        body = getattr(exc, "body", None)
        meta = body.get("metadata") if isinstance(body, dict) else None
        raw = meta.get("raw") if isinstance(meta, dict) else None  # OpenRouter's real cause
        if raw:
            error += f" | {raw}"
        out = {"status": "error", "answer": None, "citations": [], "stop_reason": "exception",
               "error": error, "events": [], "context_ids": [], "llm_calls": None,
               "tool_calls": None, "usage": {"prompt_tokens": 0, "completion_tokens": 0},
               "elapsed_s": round(time.perf_counter() - t0, 2)}
    row = {"query_id": q["id"], "answerable": q["answerable"], "type": q.get("type"),
           "question": q["question"], "filter": pq["filter"], **out}
    if q["answerable"] and out["status"] != "error":
        ctx = [chunks_by_id[i] for i in out["context_ids"] if i in chunks_by_id]
        row["context_evidence"] = evidence_covered(ctx, q["gold"])
    return row


def summarize(name: str, spec, traces: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    ok = [t for t in traces if t["status"] != "error"]
    ans = [t for t in ok if t["answerable"]]
    unans = [t for t in ok if not t["answerable"]]

    def frac(n: int, N: int) -> dict[str, Any]:
        return {"n": n, "N": N, "rate": round(n / N, 3) if N else None}

    def mean(key: str):
        vals = [t[key] for t in ok if t.get(key) is not None]
        return round(statistics.mean(vals), 2) if vals else None

    reasons = sorted({t["stop_reason"] for t in traces})
    return {"agent": name, "model": spec.model, "provider": "openrouter", "note": spec.note, **cfg,
            "n_questions": len(traces), "errors": len(traces) - len(ok),
            "answered_on_answerable": frac(sum(t["status"] == "answered" for t in ans), len(ans)),
            "abstain_on_unanswerable": frac(sum(t["status"] == "abstain" for t in unans), len(unans)),
            "context_evidence_rate": frac(sum(bool(t.get("context_evidence")) for t in ans), len(ans)),
            "stop_reasons": {r: sum(t["stop_reason"] == r for t in traces) for r in reasons},
            "mean_llm_calls": mean("llm_calls"), "mean_tool_calls": mean("tool_calls"),
            "prompt_tokens": sum(t["usage"]["prompt_tokens"] for t in traces),
            "completion_tokens": sum(t["usage"]["completion_tokens"] for t in traces),
            "cost_usd": None,  # not computed; check the OpenRouter dashboard
            "median_s": round(statistics.median(t["elapsed_s"] for t in ok), 2) if ok else None,
            "error_ids": [t["query_id"] for t in traces if t["status"] == "error"]}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="liệt kê agent đã đăng ký rồi thoát")
    p.add_argument("--split", choices=("dev", "test"))
    p.add_argument("--agents", default="gemma-4-31b-free",
                   help="'all', tên agent, hoặc model id OpenRouter, cách nhau bởi dấu phẩy")
    p.add_argument("--retriever", default="hybrid", choices=RETRIEVERS)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--budget", type=int, default=6000)
    p.add_argument("--neighbours", type=int, default=1)
    p.add_argument("--max-tool-calls", type=int, default=6)
    p.add_argument("--strategy", default="B")
    p.add_argument("--limit", type=int, default=None, help="chỉ chạy N câu đầu (chạy thử)")
    a = p.parse_args(argv)

    if a.list:
        for s in OPENROUTER_AGENTS.values():
            print(f"{s.name:20s} {s.model:34s} {s.note}")
        return
    if not a.split:
        p.error("cần --split (hoặc --list)")

    names = list(OPENROUTER_AGENTS) if a.agents == "all" else [n.strip() for n in a.agents.split(",") if n.strip()]
    specs = [get_spec(n) for n in names]  # fail fast on typos before loading models

    chunks = load_chunks(a.strategy)
    chunks_by_id = {c["id"]: c for c in chunks}
    questions = load_questions(a.split)[: a.limit]
    retriever = build({a.retriever}, chunks)[a.retriever]
    cfg = {"split": a.split, "retriever": retriever.name, "top_k": a.top_k, "budget_chars": a.budget,
           "neighbours": a.neighbours, "max_tool_calls": a.max_tool_calls, "index_strategy": a.strategy,
           "limit": a.limit}

    out = OUT / a.split
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, spec in zip(names, specs):
        agent = make_agent(name, retriever, chunks, top_k=a.top_k, budget=a.budget,
                           neighbours=a.neighbours, max_tool_calls=a.max_tool_calls)
        traces = []
        for i, q in enumerate(questions, 1):
            row = run_one(agent, q, chunks_by_id)
            traces.append(row)
            print(f"  [{spec.name}] {i}/{len(questions)} {q['id']}: {row['status']} ({row['stop_reason']}) "
                  f"{row['elapsed_s']}s" + (f"  {row['error']}" if row.get("error") else ""), flush=True)
        s = summarize(spec.name, spec, traces, cfg)
        (out / f"{spec.name}.json").write_text(json.dumps({"summary": s, "traces": traces}, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
        summaries.append(s)
        print(f"{spec.name:20s} answered {s['answered_on_answerable']['n']}/{s['answered_on_answerable']['N']}  "
              f"abstain {s['abstain_on_unanswerable']['n']}/{s['abstain_on_unanswerable']['N']}  "
              f"ctx {s['context_evidence_rate']['n']}/{s['context_evidence_rate']['N']}  errors {s['errors']}  "
              f"llm {s['mean_llm_calls']}  tools {s['mean_tool_calls']}  {s['median_s']}s")

    path = out / "summary.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"summaries": []}
    done = {s["agent"] for s in summaries}
    meta = {"split": a.split, "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "summaries": [s for s in old["summaries"] if s["agent"] not in done] + summaries}
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
