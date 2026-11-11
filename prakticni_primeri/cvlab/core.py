"""The whole library API: parameter specs, outputs, and the example registry."""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Callable, Union

import numpy as np
from matplotlib.figure import Figure

# An output is an image (gray or BGR uint8/float ndarray), a matplotlib Figure, or text.
Output = Union[np.ndarray, Figure, str]
Outputs = dict[str, Output]


# --------------------------------------------------------------------------- params


@dataclass(frozen=True)
class Param:
    name: str
    default: Any
    label: str = ""  # shown in GUI; falls back to name
    help: str = ""

    @property
    def display(self) -> str:
        return self.label or self.name

    def coerce(self, value: Any) -> Any:
        return value

    def describe(self) -> str:
        return f"default={self.default!r}"


@dataclass(frozen=True)
class Float(Param):
    min: float = 0.0
    max: float = 1.0
    step: float = 0.1

    def coerce(self, value: Any) -> float:
        return float(min(max(float(value), self.min), self.max))

    def describe(self) -> str:
        return f"float [{self.min}, {self.max}] step {self.step}, default={self.default}"


@dataclass(frozen=True)
class Int(Param):
    min: int = 0
    max: int = 100
    step: int = 1

    def coerce(self, value: Any) -> int:
        return int(min(max(int(float(value)), self.min), self.max))

    def describe(self) -> str:
        return f"int [{self.min}, {self.max}] step {self.step}, default={self.default}"


@dataclass(frozen=True)
class Choice(Param):
    options: tuple[str, ...] = ()

    def coerce(self, value: Any) -> str:
        value = str(value)
        if value not in self.options:
            raise ValueError(f"{self.name}: {value!r} not in {self.options}")
        return value

    def describe(self) -> str:
        return f"one of {list(self.options)}, default={self.default!r}"


@dataclass(frozen=True)
class Bool(Param):
    def coerce(self, value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "da"}
        return bool(value)

    def describe(self) -> str:
        return f"bool, default={self.default}"


# --------------------------------------------------------------------------- examples


@dataclass
class Example:
    qid: int
    title: str
    fn: Callable[..., Outputs]
    params: list[Param] = field(default_factory=list)
    description: str = ""

    @property
    def key(self) -> str:
        return f"q{self.qid:02d}"

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}

    def run(self, **overrides: Any) -> Outputs:
        specs = {p.name: p for p in self.params}
        unknown = set(overrides) - set(specs)
        if unknown:
            raise KeyError(f"{self.key}: unknown params {sorted(unknown)}")
        values = {name: spec.coerce(overrides.get(name, spec.default)) for name, spec in specs.items()}
        return self.fn(**values)


REGISTRY: dict[str, Example] = {}


def example(qid: int, title: str, params: list[Param] | None = None, description: str = ""):
    """Decorator: register `fn(**params) -> Outputs` as the example for question `qid`."""

    def deco(fn: Callable[..., Outputs]) -> Callable[..., Outputs]:
        ex = Example(qid, title, fn, list(params or []), description or (fn.__doc__ or "").strip())
        if ex.key in REGISTRY:
            raise ValueError(f"duplicate example {ex.key}")
        REGISTRY[ex.key] = ex
        return fn

    return deco


def load_examples() -> dict[str, Example]:
    """Import every module in cvlab.examples so they self-register; return sorted registry."""
    import cvlab.examples as pkg

    for mod in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"{pkg.__name__}.{mod.name}")
    return dict(sorted(REGISTRY.items()))
