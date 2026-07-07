"""Shared JSON I/O helpers."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Union


def append_jsonl(record: dict, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_strip_callables(record), ensure_ascii=False) + "\n")


def save_jsonl(records: list, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(_strip_callables(r), ensure_ascii=False) + "\n")


def load_jsonl(path: Union[str, Path]) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(obj: Any, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_strip_callables(obj), f, ensure_ascii=False, indent=2)


def load_json(path: Union[str, Path]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _strip_callables(obj):
    """JSON cannot serialize lambdas. Drop the verifier closure during dump.
    The verifier is reconstructed at load time via ground_truth + answer_type."""
    if isinstance(obj, dict):
        return {k: _strip_callables(v) for k, v in obj.items() if not callable(v)}
    if isinstance(obj, list):
        return [_strip_callables(x) for x in obj]
    return obj
