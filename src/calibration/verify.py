"""Answer extraction + verification.

Reconstructs the verifier from (ground_truth, answer_type) — no need to keep
Python lambdas in the JSON dataset.
"""
from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Answer extraction from raw model output
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def extract_answer(text: str) -> str:
    """Extract the candidate answer from a model completion.

    Strategy:
    1. Strip <think>...</think> blocks emitted by R1-Distill / Qwen3-thinking.
    2. Take the LAST \\boxed{...} match.
    3. Fall back to last non-empty line.
    """
    t = _THINK_TAG_RE.sub("", text or "")
    boxed = _BOXED_RE.findall(t)
    if boxed:
        return boxed[-1].strip()

    # Heuristic fallbacks
    lines = [ln.strip() for ln in t.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    last = lines[-1]
    # Drop common prefixes
    for prefix in ("Answer:", "answer:", "Final answer:", "ANSWER:"):
        if last.startswith(prefix):
            last = last[len(prefix):].strip()
    return last


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _normalise_int(s: str) -> int | None:
    """Try hard to parse an integer out of `s`."""
    s = s.strip().strip(".").strip()
    if not s:
        return None

    # Clean answer-only thousands separators, e.g. "1,234". Do not globally
    # remove commas/spaces: "(0, 2, 7)" must not become "027".
    if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+", s):
        try:
            return int(s.replace(",", ""))
        except ValueError:
            return None

    # Exact integer string.
    if re.fullmatch(r"[-+]?\d+", s):
        try:
            return int(s)
        except ValueError:
            return None

    nums = re.findall(r"[-+]?\d+", s)
    if len(nums) != 1:
        # Multiple integers in an unboxed fallback line are ambiguous. Treat as
        # unparsable rather than accidentally converting an edge tuple/list into
        # the correct answer.
        return None
    try:
        return int(nums[0])
    except ValueError:
        return None


def verify(
    prediction: str,
    ground_truth,
    answer_type: str,
    *,
    ground_truth_raw: str | None = None,
) -> bool:
    """Compare a model's predicted answer against the recorded ground truth.

    Args:
        prediction: raw extracted answer string.
        ground_truth: the canonical answer (int or str).
        answer_type: 'integer' | 'yes_no' | 'yes_unknown' | 'math'.
        ground_truth_raw: optional un-normalized reference (preferred for math).
    """
    if prediction is None:
        return False
    p = str(prediction).strip()

    if answer_type == "integer":
        from src.calibration.math_grade import strip_answer_string

        p = strip_answer_string(p)
        pi = _normalise_int(p)
        try:
            gt = int(ground_truth)
        except (TypeError, ValueError):
            return False
        return pi is not None and pi == gt

    if answer_type == "yes_no":
        p_low = p.lower()
        gt_low = str(ground_truth).strip().lower()
        # Match by token presence to be permissive of wrappers like "Yes."
        for tok in ("yes", "no"):
            if re.search(rf"\b{tok}\b", p_low):
                return tok == gt_low
        return False

    if answer_type == "yes_unknown":
        p_low = p.lower()
        gt_low = str(ground_truth).strip().lower()
        # Be careful: "yes" is a substring of "Yes I am unknown" — use word boundaries
        has_yes = bool(re.search(r"\byes\b", p_low))
        has_unknown = bool(re.search(r"\bunknown\b", p_low))
        if has_yes and not has_unknown:
            return gt_low == "yes"
        if has_unknown and not has_yes:
            return gt_low == "unknown"
        # Ambiguous or none -> treat as wrong
        return False

    if answer_type == "math":
        from src.calibration.math_grade import math_equal, strip_answer_string

        ref = str(ground_truth_raw if ground_truth_raw is not None else ground_truth)
        if math_equal(p, ref):
            return True
        pn = strip_answer_string(p).strip().lower()
        gn = strip_answer_string(ref).strip().lower()
        if pn and gn and pn == gn:
            return True
        return False

    # Unknown answer_type: fallback to string equality (case-insensitive).
    return p.lower() == str(ground_truth).strip().lower()
