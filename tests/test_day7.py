"""Offline checks for src/day7 (fake LLM client; measures code behaviour, not model quality)."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from day7.agent import Agent  # noqa: E402
from day7.context import pack_context  # noqa: E402
from day7.corpus import evidence_covered, public_question  # noqa: E402
from day7.retrievers import BM25Retriever, HybridRRF, fold  # noqa: E402


def chunk(i, text, page=1, chapter=1, order=0):
    return {"id": i, "text": text, "source_id": "doc", "source": "doc.pdf", "page": page,
            "metadata": {"chapter": chapter, "article": None, "order": order}}


CORPUS = [chunk("c0", "Điều 1. Phạm vi điều chỉnh của bộ luật", order=0),
          chunk("c1", "Người lao động được nghỉ Tết Âm lịch: 05 ngày", page=2, chapter=7, order=1),
          chunk("c2", "Thời gian thử việc không quá 60 ngày", page=3, chapter=3, order=2)]


class Static:
    name = "static"

    def __init__(self, rows): self.rows = rows

    def search(self, query, *, filters=None, top_k=3): return [dict(r, rank=i + 1, score=1.0) for i, r in enumerate(self.rows[:top_k])]


def call(name, args, cid="t1"):
    return NS(id=cid, function=NS(name=name, arguments=json.dumps(args, ensure_ascii=False)))


class FakeClient:
    def __init__(self, script):
        self.script, self.calls = iter(script), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        tcs = next(self.script)
        return NS(usage=NS(prompt_tokens=10, completion_tokens=5),
                  choices=[NS(message=NS(content=None, tool_calls=tcs))])


def agent(script, rows=CORPUS[1:2], **kw):
    return Agent(Static(rows), CORPUS, client=FakeClient(script), model="fake", provider="fake", neighbours=0, **kw)


class RetrievalTests(unittest.TestCase):
    def test_bm25_and_filter(self):
        bm = BM25Retriever(CORPUS)
        self.assertEqual(bm.search("nghỉ Tết", top_k=1)[0]["id"], "c1")
        self.assertEqual(bm.search("nghỉ Tết", filters={"chapter": 3}), [])
        with self.assertRaises(ValueError):
            bm.search("x", filters={"gold": 1})

    def test_fold_handles_ocr_tone_errors(self):
        self.assertEqual(fold("tiên lương"), fold("tiền lương"))
        self.assertEqual(BM25Retriever(CORPUS, folded=True).search("thu viec", top_k=1)[0]["id"], "c2")

    def test_rrf_is_rank_based(self):
        a, b = Static([CORPUS[1], CORPUS[2]]), Static([CORPUS[2], CORPUS[0]])
        out = HybridRRF([a, b]).search("q", top_k=3)
        self.assertEqual(out[0]["id"], "c2")  # appears in both lists
        self.assertEqual([r["rank"] for r in out], [1, 2, 3])

    def test_neighbours_and_budget(self):
        ctx, ids = pack_context([dict(CORPUS[1])], budget=10_000, neighbours=1, corpus=CORPUS)
        self.assertEqual(ids, ["c1", "c0", "c2"])
        _, ids = pack_context([dict(CORPUS[1])], budget=20)
        self.assertEqual(ids, [])

    def test_evidence_requires_all_quotes_on_gold_page(self):
        gold = [{"source_id": "doc", "page": 2, "evidence": ["Tết Âm lịch", "05 ngày"]}]
        self.assertTrue(evidence_covered([CORPUS[1]], gold))
        self.assertFalse(evidence_covered([CORPUS[2]], gold))
        self.assertNotIn("gold", public_question({"id": "x", "question": "q", "gold": gold}))


class AgentTests(unittest.TestCase):
    def test_answer_with_valid_citation(self):
        a = agent([[call("search", {"query": "tết"})],
                   [call("final_answer", {"status": "answered", "answer": "05 ngày [c1]", "citations": ["c1"]})]])
        r = a.run("Tết nghỉ mấy ngày?")
        self.assertEqual((r["status"], r["citations"], r["tool_calls"], r["llm_calls"]), ("answered", ["c1"], 1, 2))
        self.assertEqual(r["usage"], {"prompt_tokens": 20, "completion_tokens": 10})

    def test_citation_not_in_context_abstains(self):
        a = agent([[call("search", {"query": "tết"})],
                   [call("final_answer", {"status": "answered", "answer": "60 ngày [c2]", "citations": ["c2"]})]])
        r = a.run("q")
        self.assertEqual((r["status"], r["stop_reason"]), ("abstain", "invalid_citations"))

    def test_answer_without_search_has_no_context(self):
        a = agent([[call("final_answer", {"status": "answered", "answer": "05 ngày [c1]", "citations": ["c1"]})]])
        self.assertEqual(a.run("q")["status"], "abstain")

    def test_tool_limit_forces_final(self):
        script = [[call("search", {"query": f"q{i}"})] for i in range(2)]
        script.append([call("final_answer", {"status": "abstain", "answer": "Không đủ bằng chứng", "citations": []})])
        a = agent(script, max_tool_calls=2)
        r = a.run("q")
        self.assertEqual((r["status"], r["tool_calls"]), ("abstain", 2))
        self.assertNotEqual(a.client.calls[-1]["tool_choice"], "auto")

    def test_tool_error_is_logged_and_returned_to_model(self):
        a = agent([[call("read_neighbours", {"chunk_id": "missing"})],
                   [call("final_answer", {"status": "abstain", "answer": "Không đủ", "citations": []})]])
        r = a.run("q")
        self.assertEqual(r["events"][1]["event"], "tool_error")
        tool_msgs = [m for m in a.client.calls[-1]["messages"] if m["role"] == "tool"]
        self.assertIn("Lỗi tool", tool_msgs[-1]["content"])

    def test_filter_passed_from_user_not_model(self):
        rows = []

        class Spy(Static):
            def search(self, query, *, filters=None, top_k=3):
                rows.append(filters)
                return []

        a = Agent(Spy([]), CORPUS, client=FakeClient([[call("search", {"query": "x"})],
                                                      [call("final_answer", {"status": "abstain", "answer": "-", "citations": []})]]),
                  model="m", provider="p", neighbours=0)
        a.run("q", filters={"chapter": 7})
        self.assertEqual(rows, [{"chapter": 7}])


if __name__ == "__main__":
    unittest.main()
