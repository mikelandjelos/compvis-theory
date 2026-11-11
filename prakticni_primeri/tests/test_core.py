import numpy as np
import pytest
from matplotlib.figure import Figure

from cvlab.cli import main as cli_main
from cvlab.core import REGISTRY, Bool, Choice, Example, Float, Int, load_examples

EXAMPLES = load_examples()


# ---------------------------------------------------------------- params / Example plumbing


def test_param_coercion():
    assert Float("a", 0.5, min=0, max=1).coerce("2") == 1.0
    assert Int("b", 3, min=1, max=9).coerce("4.7") == 4
    assert Bool("c", False).coerce("true") is True
    assert Choice("d", "x", options=("x", "y")).coerce("y") == "y"
    with pytest.raises(ValueError):
        Choice("d", "x", options=("x", "y")).coerce("z")


def test_example_run_fills_defaults_and_rejects_unknown():
    ex = Example(99, "t", lambda a, b: {"sum": str(a + b)}, [Int("a", 1), Int("b", 2)])
    assert ex.key == "q99"
    assert ex.run() == {"sum": "3"}
    assert ex.run(b=10) == {"sum": "11"}
    with pytest.raises(KeyError):
        ex.run(c=1)


def test_registry_is_sorted():
    assert list(EXAMPLES) == sorted(REGISTRY)


# ---------------------------------------------------------------- every registered example


@pytest.mark.parametrize("key", list(EXAMPLES))
def test_example_runs_with_defaults(key):
    outputs = EXAMPLES[key].run()
    assert outputs, "example returned no outputs"
    for name, value in outputs.items():
        assert isinstance(name, str)
        assert isinstance(value, (np.ndarray, Figure, str)), f"{name}: {type(value)}"
        if isinstance(value, np.ndarray):
            assert value.ndim in (2, 3) and value.size > 0


@pytest.mark.parametrize("key", list(EXAMPLES))
def test_every_choice_option_runs(key):
    ex = EXAMPLES[key]
    for p in ex.params:
        if isinstance(p, Choice):
            for opt in p.options:
                ex.run(**{p.name: opt})


# ---------------------------------------------------------------- CLI host


def test_cli_list(capsys):
    assert cli_main(["list"]) == 0


def test_cli_run_all(tmp_path):
    assert cli_main(["run", "all", "-o", str(tmp_path)]) == 0
