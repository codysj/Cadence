from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from core import OPENROUTER_BASE_URL, make_client


@pytest.fixture
def openai_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake_openai(**kwargs: object) -> str:
        calls.append(kwargs)
        return "client"

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=fake_openai))
    return calls


def test_openrouter_key_routes_to_openrouter(
    monkeypatch: pytest.MonkeyPatch, openai_calls: list[dict[str, object]]
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    assert make_client() == "client"
    assert openai_calls == [{"base_url": OPENROUTER_BASE_URL, "api_key": "test-key"}]


def test_without_openrouter_key_uses_openai_defaults(
    monkeypatch: pytest.MonkeyPatch, openai_calls: list[dict[str, object]]
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert make_client() == "client"
    assert openai_calls == [{}]
