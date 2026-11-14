"""Named agents, one per OpenRouter model, all sharing the same tool loop (`agent.Agent`).

    python -m day7.eval_agent --list
    python -m day7.eval_agent --split dev --agents gemma-4-31b-free

Every agent uses OPENROUTER_API_KEY from .env. Only the model changes between
agents, so differences come from the model, not from retriever, tools, prompt or
context budget. Any OpenRouter model id can be passed raw; non-`:free` models
cost money (billed to the OpenRouter account), `:free` ones are rate-limited.

Model IDs checked against https://openrouter.ai/api/v1/models (`supported_parameters`
contains "tools") on 2026-11-14.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .agent import Agent
from .corpus import ROOT

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class AgentSpec:
    name: str                 # short key used on the command line and in result file names
    model: str                # OpenRouter model id
    note: str
    temperature: float = 0.0
    extra_body: dict[str, Any] = field(default_factory=dict)  # OpenRouter-specific request fields


OPENROUTER_AGENTS: dict[str, AgentSpec] = {s.name: s for s in [
    AgentSpec("gemma-4-31b-free", "google/gemma-4-31b-it:free",
              "Miễn phí, giới hạn rate; hỗ trợ tool calling"),
]}


def get_spec(name: str) -> AgentSpec:
    """Registered agent by name, or an ad-hoc one when `name` is a raw OpenRouter model id."""
    if name in OPENROUTER_AGENTS:
        return OPENROUTER_AGENTS[name]
    if "/" in name:
        return AgentSpec(name.replace("/", "__").replace(":", "_"), name, "model id tự nhập")
    raise KeyError(f"agent không tồn tại: {name}. Có: {', '.join(OPENROUTER_AGENTS)}")


class _Completions:
    def __init__(self, owner: "OpenRouterClient") -> None:
        self._owner = owner

    def create(self, **kwargs):
        return self._owner._create(**kwargs)


class _Chat:
    def __init__(self, owner: "OpenRouterClient") -> None:
        self.completions = _Completions(owner)


class OpenRouterClient:
    """Thin wrapper over an OpenAI-compatible client pointed at OpenRouter.

    Adds per-agent `extra_body` and asks OpenRouter to route only to providers that
    support every parameter we send (tools, tool_choice).
    """

    def __init__(self, spec: AgentSpec, inner=None) -> None:
        self.spec = spec
        if inner is None:
            from dotenv import load_dotenv
            from openai import OpenAI

            load_dotenv(ROOT / ".env")
            key = os.environ.get("OPENROUTER_API_KEY", "")
            if not key:
                raise RuntimeError("Thiếu OPENROUTER_API_KEY trong .env")
            inner = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key, max_retries=3, timeout=120)
        self.inner = inner
        self.chat = _Chat(self)

    def _create(self, **kwargs):
        body = {"provider": {"require_parameters": True}}
        body.update(self.spec.extra_body)
        body.update(kwargs.pop("extra_body", None) or {})
        return self.inner.chat.completions.create(**kwargs, extra_body=body)


def make_agent(name: str, retriever, corpus, *, inner_client=None, **agent_kwargs) -> Agent:
    """Build the shared tool-loop Agent for one OpenRouter model."""
    spec = get_spec(name)
    client = OpenRouterClient(spec, inner=inner_client)
    return Agent(retriever, corpus, client=client, model=spec.model, provider="openrouter",
                 temperature=spec.temperature, **agent_kwargs)
