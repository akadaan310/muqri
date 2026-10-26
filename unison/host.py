"""Opt-in reversible capability routing: legacy by default.

Legacy behavior (including recitation scoring) MUST be passed as a callable.
Unison research modules never automatically replace recitation verdicts.
Shadow evaluation is opt-in only for pure, side-effect-free capabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

Mode = Literal["legacy", "shadow", "unison"]


@dataclass(frozen=True, slots=True)
class RunReceipt:
    capability: str
    mode: str
    result: Any
    shadow: Any | None = None
    warning: str | None = None


class UnisonHost:
    def __init__(self) -> None:
        self.mode: Mode = "legacy"
        self.legacy: dict[str, Callable] = {}
        self.unison: dict[str, Callable] = {}
        self.pure: set[str] = set()

    def register(self, capability: str, *, legacy: Callable,
                 unison: Callable, pure: bool = False) -> None:
        if capability in self.legacy or not capability:
            raise ValueError("Duplicate or unnamed capability")
        self.legacy[capability] = legacy
        self.unison[capability] = unison
        if pure:
            self.pure.add(capability)

    def set_mode(self, mode: Mode) -> None:
        if mode not in {"legacy", "shadow", "unison"}:
            raise ValueError("Unknown capability dispatch mode")
        self.mode = mode

    def rollback(self) -> None:
        """Revert DISPATCH, not underlying external state."""
        self.mode = "legacy"

    def invoke(self, capability: str, *args, **kwargs) -> RunReceipt:
        if capability not in self.legacy:
            raise KeyError(f"Unregistered capability {capability}")
        if self.mode == "legacy":
            return RunReceipt(capability, self.mode,
                              self.legacy[capability](*args, **kwargs))
        if self.mode == "unison":
            return RunReceipt(capability, self.mode,
                              self.unison[capability](*args, **kwargs))
        if capability not in self.pure:
            raise PermissionError("Shadow mode restricted to registered pure operations")
        result = self.legacy[capability](*args, **kwargs)
        try:
            shadow = self.unison[capability](*args, **kwargs)
            warning = None if shadow == result else "OUTPUTS_DIVERGE: no promotion performed"
        except Exception as exc:
            shadow = None
            warning = f"SHADOW_FAILED: {type(exc).__name__}: {exc}"
        return RunReceipt(capability, self.mode, result, shadow, warning)
