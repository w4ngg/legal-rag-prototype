from __future__ import annotations

"""LLM interface and deterministic fake used by tests and local smoke runs."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..domain import LLMAnswer


def _required_prompt(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


@dataclass(frozen=True, slots=True)
class LLMCall:
    system_prompt: str
    user_prompt: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "system_prompt",
            _required_prompt(self.system_prompt, "system_prompt"),
        )
        object.__setattr__(
            self,
            "user_prompt",
            _required_prompt(self.user_prompt, "user_prompt"),
        )


@runtime_checkable
class LLMClient(Protocol):
    @property
    def model(self) -> str:
        """Configured model identifier used before the first provider call."""
        ...

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMAnswer:
        """Generate one structured legal answer."""
        ...


class FakeLLMClient:
    """Return a fixed answer while recording every prompt sent by the caller."""

    def __init__(self, response: LLMAnswer) -> None:
        self.response = response
        self.calls: list[LLMCall] = []

    @property
    def model(self) -> str:
        return self.response.model

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMAnswer:
        self.calls.append(LLMCall(system_prompt, user_prompt))
        return self.response
