"""Run the real agent on one split and save traces in the format of starter/agent_metrics.py.

    python -m day7.run_agent --split dev --retriever dense --neighbours 0 --tag baseline
    python -m day7.run_agent --split dev --retriever rerank --neighbours 1 --tag selected

Outputs results/agent/<split>/<tag>/{traces.json,cases.json,run.json}. Labels
(reviews.json) are written separately after reading each answer against the source.
"""
from __future__ import annotations

import argparse
import json
import time

from .agent import Agent, make_client
from .corpus import ROOT, load_chunks, load_questions, public_question
from .eval_retrieval import build

PRICE_PER_MTOK = {"input": 0.0, "output": 0.0}  # free tier / local model; edit if a paid model is used


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", required=True, choices=("dev", "test"))
    p.add_argument("--retriever", default="dense", choices=("dense", "bm25", "hybrid", "rerank"))
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--neighbours", type=int, default=0)
    p.add_argument("--budget", type=int, default=6000)
    p.add_argument("--max-tool-calls", type=int, default=6)
    p.add_argument("--tag", required=True)
    p.add_argument("--only", default="", help="comma-separated question IDs (debug)")
    a = p.parse_args(argv)

    chunks = load_chunks("B")
    retriever = build({a.retriever}, chunks)[a.retriever]
    client, provider, model = make_client()
    agent = Agent(retriever, chunks, top_k=a.top_k, budget=a.budget, neighbours=a.neighbours,
                  max_tool_calls=a.max_tool_calls, client=client, model=model, provider=provider)
    questions = load_questions(a.split)
    if a.only:
        questions = [q for q in questions if q["id"] in a.only.split(",")]
    out = ROOT / "results" / "agent" / a.split / a.tag
    out.mkdir(parents=True, exist_ok=True)
    traces = []
    for q in questions:
        pq = public_question(q)  # gold never reaches the agent
        try:
            r = agent.run(pq["question"], filters=pq["filter"])
        except Exception as exc:  # provider/vector-store error: logged, not replaced by a fake answer
            r = {"status": "error", "answer": None, "citations": [], "events": [],
                 "error": f"{type(exc).__name__}: {exc}", "model": model, "provider": provider}
        r = {"query_id": q["id"], "question": q["question"], "filter": pq["filter"], **r}
        traces.append(r)
        print(f"{q['id']} {r['status']:8s} tools={r.get('tool_calls')} {r.get('elapsed_s')}s  "
              f"{(r.get('answer') or r.get('error') or '')[:110]!r}", flush=True)
        (out / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1), encoding="utf-8")
    cases = [{"id": q["id"], "split": q["split"], "answerable": q["answerable"]} for q in questions]
    (out / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = [t for t in traces if t["status"] != "error"]
    tok_in = sum(t["usage"]["prompt_tokens"] for t in ok)
    tok_out = sum(t["usage"]["completion_tokens"] for t in ok)
    run = {"split": a.split, "tag": a.tag, "provider": provider, "model": model, "retriever": retriever.name,
           "top_k": a.top_k, "neighbours": a.neighbours, "budget_chars": a.budget, "max_tool_calls": a.max_tool_calls,
           "temperature": 0.0, "n": len(traces), "errors": len(traces) - len(ok),
           "prompt_tokens": tok_in, "completion_tokens": tok_out,
           "cost_usd": round(tok_in / 1e6 * PRICE_PER_MTOK["input"] + tok_out / 1e6 * PRICE_PER_MTOK["output"], 6),
           "total_elapsed_s": round(sum(t.get("elapsed_s") or 0 for t in traces), 1),
           "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(run, ensure_ascii=False))


if __name__ == "__main__":
    main()
