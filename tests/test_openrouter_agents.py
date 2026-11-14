"""Mock tests for the OpenRouter agent registry and runner. No network call is made;
they check code paths only, not model quality."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from day7.agents import OPENROUTER_AGENTS, AgentSpec, OpenRouterClient, get_spec, make_agent  # noqa: E402
from day7.eval_agent import run_one, summarize  # noqa: E402

FREE = "gemma-4-31b-free"
CHUNK = {"id": "c1", "text": "Thời gian thử việc không quá 60 ngày.", "source_id": "blld", "source": "blld.pdf",
         "page": 7, "metadata": {"order": 0, "article": 25}}


def _call(name, args, cid="t1"):
    return NS(id=cid, function=NS(name=name, arguments=json.dumps(args, ensure_ascii=False)))


class FakeInner:
    """Scripted OpenAI-compatible client: search once, then answer citing c1."""

    def __init__(self):
        self.requests = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        calls = [_call("search", {"query": "thử việc"})] if len(self.requests) == 1 else \
            [_call("final_answer", {"status": "answered", "answer": "Không quá 60 ngày [c1].", "citations": ["c1"]})]
        return NS(usage=NS(prompt_tokens=100, completion_tokens=10),
                  choices=[NS(message=NS(content=None, tool_calls=calls))])


class FakeRetriever:
    name = "fake"

    def search(self, query, *, filters=None, top_k=3):
        return [{**CHUNK, "score": 1.0, "rank": 1}]


class RegistryTest(unittest.TestCase):
    def test_registry_names(self):
        self.assertIn(FREE, OPENROUTER_AGENTS)
        for name, s in OPENROUTER_AGENTS.items():
            self.assertEqual(name, s.name)

    def test_raw_model_ids(self):
        self.assertEqual(get_spec("openai/gpt-4o-mini").model, "openai/gpt-4o-mini")
        self.assertEqual(get_spec("google/gemma-4-26b-a4b-it:free").model, "google/gemma-4-26b-a4b-it:free")
        with self.assertRaises(KeyError):
            get_spec("khong-co")


class ClientTest(unittest.TestCase):
    def test_extra_body(self):
        inner = FakeInner()
        spec = AgentSpec("r", "x/y:free", "test", extra_body={"reasoning": {"effort": "low"}})
        OpenRouterClient(spec, inner=inner).chat.completions.create(model="m", messages=[])
        body = inner.requests[0]["extra_body"]
        self.assertTrue(body["provider"]["require_parameters"])
        self.assertEqual(body["reasoning"], {"effort": "low"})


class RunnerTest(unittest.TestCase):
    def test_run_one_and_summary(self):
        inner = FakeInner()
        agent = make_agent(FREE, FakeRetriever(), [CHUNK], inner_client=inner, neighbours=0)
        q = {"id": "q1", "answerable": True, "type": "fact", "question": "Thử việc bao lâu?",
             "gold": [{"source_id": "blld", "page": 7, "evidence": ["không quá 60 ngày"]}]}
        row = run_one(agent, q, {"c1": CHUNK})
        self.assertEqual(row["status"], "answered")
        self.assertEqual(row["citations"], ["c1"])
        self.assertTrue(row["context_evidence"])
        self.assertEqual(inner.requests[0]["model"], "google/gemma-4-31b-it:free")
        self.assertNotIn("gold", json.dumps(inner.requests[0]["messages"], ensure_ascii=False))

        class Boom:
            name = "boom"

            def search(self, *a, **k):
                raise RuntimeError("db down")

        bad = make_agent(FREE, Boom(), [CHUNK], inner_client=FakeInner(), neighbours=0)
        err = run_one(bad, {**q, "id": "q2"}, {"c1": CHUNK})
        # tool error is reported to the model; the scripted model cites c1, which never entered context
        self.assertEqual(err["status"], "abstain")
        self.assertEqual(err["stop_reason"], "invalid_citations")

        s = summarize(FREE, get_spec(FREE), [row, err], {"split": "dev"})
        self.assertEqual(s["answered_on_answerable"], {"n": 1, "N": 2, "rate": 0.5})
        self.assertEqual(s["context_evidence_rate"]["n"], 1)
        self.assertEqual(s["errors"], 0)
        self.assertIsNone(s["cost_usd"])


if __name__ == "__main__":
    unittest.main()
