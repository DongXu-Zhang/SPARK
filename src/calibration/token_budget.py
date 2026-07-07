"""Adaptive max_tokens estimator.

Why adaptive? A 1500-problem dataset has length-bimodal needs:
  * 1-step counting: ~1500 tokens of CoT is plenty
  * 25-node MST  : 8000+ tokens of step-by-step Kruskal

Using a single global ``max_tokens`` forces a bad tradeoff:
  * Big (12288): KV cache is dominated by long-tail seqs; vLLM preempts
    short seqs all the time; throughput collapses.
  * Small (4096): hard problems get truncated mid-CoT; ``finish_reason='length'``
    floods the dataset with false negatives.

Solution: per-prompt SamplingParams, with ``max_tokens`` set from the
problem's structural parameters. vLLM accepts a list of SamplingParams
of the same length as the prompt list (one per prompt).
"""
from __future__ import annotations

from typing import Optional


# Hard-coded budgets per (domain, complexity-knob). Numbers come from
# inspecting our sanity logs: median tokens used by Qwen3-4B on similar
# problems, scaled up by 1.5x to leave headroom for slower-than-typical CoT.

# Floor and ceiling — never go below the floor (model needs room for any
# reasonable CoT) or above the ceiling (KV cache safety).
_MIN_BUDGET = 1024
_MAX_BUDGET = 12288


def _budget_symbolic(params: dict) -> int:
    """Symbolic compose: budget grows with operation count."""
    n_steps = int(params.get("n_steps", 3))
    # Empirical: Qwen3-4B uses ~300-500 tokens per step + ~800 setup
    # Scale up x1.5 for safety
    return 1024 + n_steps * 600


def _budget_logical(params: dict) -> int:
    """Logical inference: budget grows with depth + distractors."""
    depth = int(params.get("depth", 3))
    n_distractors = int(params.get("n_distractors", 0))
    # Each chain step ~400 tokens of CoT, distractors ~50 each
    return 1024 + depth * 500 + n_distractors * 60


def _budget_algorithmic(params: dict) -> int:
    """Algorithmic reasoning: budget depends on n_nodes AND algorithm class."""
    n_nodes = int(params.get("n_nodes", 10))
    algo = str(params.get("algorithm", ""))
    weighted = bool(params.get("weighted", False))

    # Cheap algorithms: linear scan over edges
    cheap = {"node_degree_max", "is_bipartite", "has_cycle", "connectivity"}
    # Medium algorithms: small graph traversal
    medium = {"shortest_path_length", "graph_diameter"}
    # Expensive algorithms: full graph reasoning
    expensive = {"min_spanning_tree_weight"}

    if algo in cheap:
        per_node = 80
        base = 1024
    elif algo in medium:
        per_node = 250
        base = 2048
    elif algo in expensive:
        per_node = 350
        base = 2048
    else:
        per_node = 200
        base = 1500

    weight_bonus = 1.3 if weighted else 1.0
    return int(base + n_nodes * per_node * weight_bonus)


_DOMAIN_DISPATCH = {
    "symbolic_compose": _budget_symbolic,
    "logical_inference": _budget_logical,
    "algorithmic_reasoning": _budget_algorithmic,
}


def estimate_max_tokens(
    problem: dict,
    floor: int = _MIN_BUDGET,
    ceiling: int = _MAX_BUDGET,
    fallback: int = 4096,
) -> int:
    """Estimate the token budget for one problem.

    Parameters
    ----------
    problem : dict — must contain ``domain`` and ``structural_params``.
    floor : minimum tokens (1024 default — even trivial problems need room).
    ceiling : maximum tokens (12288 default — KV cache safety cap).
    fallback : returned if domain is unknown.

    Returns
    -------
    An integer in [floor, ceiling].
    """
    domain = problem.get("domain", "")
    params = problem.get("structural_params", {}) or {}

    fn = _DOMAIN_DISPATCH.get(domain)
    if fn is None:
        budget = fallback
    else:
        try:
            budget = fn(params)
        except (TypeError, ValueError, KeyError):
            budget = fallback

    return max(floor, min(ceiling, int(budget)))


def summarize_budget_distribution(problems: list) -> dict:
    """For diagnostic logging — what's the budget distribution over a dataset?

    Returns a dict like:
        {"n": 4500, "min": 1024, "median": 3200, "p90": 6500,
         "max": 11200, "mean": 3850, "total": ...}
    """
    budgets = [estimate_max_tokens(p) for p in problems]
    if not budgets:
        return {"n": 0}
    sorted_b = sorted(budgets)
    n = len(sorted_b)
    return {
        "n": n,
        "min": sorted_b[0],
        "p10": sorted_b[int(n * 0.10)],
        "median": sorted_b[n // 2],
        "p90": sorted_b[int(n * 0.90)],
        "max": sorted_b[-1],
        "mean": sum(sorted_b) / n,
        "total": sum(sorted_b),
    }
