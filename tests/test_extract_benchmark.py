"""The benchmark JobSpec extractor must not spend LLM calls on JDs that already have a spec."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "extract_benchmark_jobspecs",
    Path(__file__).resolve().parents[1] / "scripts" / "extract_benchmark_jobspecs.py")
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)


def test_pending_skips_existing_specs():
    assert script.pending(["a", "b", "c"], {"a": {}, "c": {}}) == ["b"]


def test_pending_nothing_to_do_when_all_present():
    assert script.pending(["a", "b"], {"a": {}, "b": {}}) == []


def test_pending_force_reextracts_everything():
    assert script.pending(["a", "b"], {"a": {}}, force=True) == ["a", "b"]
