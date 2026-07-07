"""Symbolic Compose generator.

Each problem = [integer range] + [chain of K operations].
Operations belong to three families: filter, transform, aggregate.
The chain MUST end with an aggregate so the answer is a single integer.
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.utils.io import save_jsonl


# ---------------------------------------------------------------------------
# Operation pool
# ---------------------------------------------------------------------------

@dataclass
class Op:
    name: str
    family: str         # 'filter' | 'transform' | 'aggregate'
    nl: str             # natural-language template
    fn: Callable        # operation
    complexity: float   # in [0, 1], used in d_structural


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    return all(n % i for i in range(3, int(math.isqrt(n)) + 1, 2))


FILTERS: list[Op] = [
    Op("even", "filter", "keep only the even numbers",
       lambda xs: [x for x in xs if x % 2 == 0], 0.10),
    Op("odd", "filter", "keep only the odd numbers",
       lambda xs: [x for x in xs if x % 2 == 1], 0.10),
    Op("div_by_3", "filter", "keep numbers divisible by 3",
       lambda xs: [x for x in xs if x % 3 == 0], 0.15),
    Op("div_by_5", "filter", "keep numbers divisible by 5",
       lambda xs: [x for x in xs if x % 5 == 0], 0.15),
    Op("greater_than_50", "filter", "keep numbers strictly greater than 50",
       lambda xs: [x for x in xs if x > 50], 0.10),
    Op("less_than_30", "filter", "keep numbers strictly less than 30",
       lambda xs: [x for x in xs if x < 30], 0.10),
    Op("multiple_of_7", "filter", "keep numbers that are multiples of 7",
       lambda xs: [x for x in xs if x % 7 == 0], 0.20),
    Op("perfect_square", "filter", "keep only perfect squares",
       lambda xs: [x for x in xs if x >= 0 and int(math.isqrt(x)) ** 2 == x], 0.30),
    Op("prime", "filter", "keep only prime numbers",
       lambda xs: [x for x in xs if _is_prime(x)], 0.45),
    Op("palindrome", "filter", "keep numbers whose decimal representation is a palindrome",
       lambda xs: [x for x in xs if str(x) == str(x)[::-1]], 0.45),
    Op("digit_count_2", "filter", "keep numbers with exactly two decimal digits",
       lambda xs: [x for x in xs if 10 <= x <= 99], 0.20),
]

TRANSFORMS: list[Op] = [
    Op("square", "transform", "replace each number with its square",
       lambda xs: [x * x for x in xs], 0.10),
    Op("plus_3", "transform", "add 3 to each number",
       lambda xs: [x + 3 for x in xs], 0.05),
    Op("times_2", "transform", "double each number",
       lambda xs: [x * 2 for x in xs], 0.05),
    Op("mod_7", "transform", "replace each number by its remainder modulo 7",
       lambda xs: [x % 7 for x in xs], 0.20),
    Op("mod_11", "transform", "replace each number by its remainder modulo 11",
       lambda xs: [x % 11 for x in xs], 0.25),
    Op("digit_sum", "transform", "replace each number by the sum of its decimal digits",
       lambda xs: [sum(int(c) for c in str(abs(x))) for x in xs], 0.30),
    Op("xor_5", "transform", "replace each number by its bitwise XOR with 5",
       lambda xs: [x ^ 5 for x in xs], 0.40),
    Op("and_15", "transform", "replace each number by its bitwise AND with 15",
       lambda xs: [x & 15 for x in xs], 0.40),
    Op("negate", "transform", "negate each number",
       lambda xs: [-x for x in xs], 0.05),
    Op("abs_diff_50", "transform", "replace each number with the absolute difference between it and 50",
       lambda xs: [abs(x - 50) for x in xs], 0.20),
]


def _safe_aggregate(fn, xs):
    if not xs:
        return 0
    return fn(xs)


AGGREGATES: list[Op] = [
    Op("count", "aggregate", "report how many numbers remain",
       lambda xs: len(xs), 0.10),
    Op("sum", "aggregate", "report their sum",
       lambda xs: _safe_aggregate(sum, xs), 0.15),
    Op("min", "aggregate", "report the smallest value",
       lambda xs: _safe_aggregate(min, xs), 0.10),
    Op("max", "aggregate", "report the largest value",
       lambda xs: _safe_aggregate(max, xs), 0.10),
    Op("mean_floor", "aggregate", "report the floor of their mean",
       lambda xs: int(_safe_aggregate(sum, xs) / max(1, len(xs))), 0.20),
    Op("median_low", "aggregate",
       "report the median (use the lower of the two middles when count is even)",
       lambda xs: int(statistics.median_low(xs)) if xs else 0, 0.25),
    Op("mode_smallest", "aggregate",
       "report the most frequent value (smallest among ties)",
       lambda xs: (min(x for x in xs if xs.count(x) == max(map(xs.count, xs))) if xs else 0),
       0.40),
    Op("range_span", "aggregate", "report (max - min)",
       lambda xs: (_safe_aggregate(max, xs) - _safe_aggregate(min, xs)) if xs else 0, 0.20),
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _structural_difficulty(n_steps: int, ops: list[Op], range_high: int) -> float:
    """Return d_structural in [0, 1]."""
    op_complexity = sum(o.complexity for o in ops) / max(1, len(ops))
    range_term = min(1.0, math.log10(max(2, range_high)) / math.log10(500))
    has_bitwise = any("xor" in o.name or "and_" in o.name for o in ops)
    d = (
        0.40 * (n_steps / 7.0)
        + 0.25 * op_complexity
        + 0.20 * range_term
        + 0.15 * (1.0 if has_bitwise else 0.0)
    )
    return float(min(1.0, max(0.0, d)))


def _target_bin_counts(n: int, n_bins: int = 5) -> list[int]:
    base, extra = divmod(n, n_bins)
    return [base + (1 if i < extra else 0) for i in range(n_bins)]


def _sample_ops_by_target(rng: random.Random, pool: list[Op], n: int,
                          target_d: float | None) -> list[Op]:
    if n <= 0:
        return []
    if target_d is None:
        return rng.sample(pool, n)

    ordered = sorted(pool, key=lambda op: op.complexity)
    if target_d < 0.30:
        candidates = ordered[: max(n, len(ordered) // 2)]
    elif target_d > 0.70:
        candidates = ordered[-max(n, len(ordered) // 2):]
    else:
        candidates = pool
    return rng.sample(candidates, n)


def generate_one(seed: int, target_d: float | None = None) -> dict:
    rng = random.Random(seed)

    # 1. step count (main difficulty knob)
    if target_d is None:
        n_steps = rng.randint(1, 7)
    elif target_d < 0.20:
        n_steps = 1
    elif target_d < 0.40:
        n_steps = rng.choice([2, 3])
    elif target_d < 0.60:
        n_steps = rng.choice([3, 4, 5])
    elif target_d < 0.80:
        n_steps = rng.choice([5, 6])
    else:
        n_steps = 7

    # 2. value range
    range_low = 1
    if target_d is None:
        range_high = rng.choice([20, 50, 100, 200, 500])
    elif target_d < 0.25:
        range_high = rng.choice([20, 50])
    elif target_d > 0.75:
        range_high = 500
    else:
        range_high = rng.choice([50, 100, 200])

    # 3. build operation chain: 1-2 filters, 0-(n_steps-1) transforms, 1 aggregate
    if n_steps == 1:
        n_filter, n_transform = 0, 0
    else:
        n_filter = rng.randint(1, min(2, n_steps - 1))
        n_transform = max(0, n_steps - n_filter - 1)

    filters = _sample_ops_by_target(rng, FILTERS, n_filter, target_d)
    transforms = _sample_ops_by_target(rng, TRANSFORMS, n_transform, target_d)
    if target_d is not None and target_d > 0.75 and n_transform > 0:
        bitwise = [op for op in TRANSFORMS if op.name in ("xor_5", "and_15")]
        if not any(op in bitwise for op in transforms):
            transforms[0] = rng.choice(bitwise)
    aggregate = _sample_ops_by_target(rng, AGGREGATES, 1, target_d)[0]
    chain = filters + transforms + [aggregate]

    # 4. compute ground truth
    xs = list(range(range_low, range_high + 1))
    answer = None
    try:
        for op in chain[:-1]:
            xs = op.fn(xs)
        answer = int(chain[-1].fn(xs))
    except Exception:
        # rare: regenerate with a new seed
        return generate_one(seed + 7919, target_d)

    # 5. natural-language prompt
    parts = [f"Consider the integers from {range_low} to {range_high}, inclusive."]
    if len(chain) == 1:
        parts.append(f"Then, {chain[0].nl}.")
    else:
        for i, op in enumerate(chain[:-1]):
            parts.append(f"Step {i + 1}: {op.nl}.")
        parts.append(f"Finally, {chain[-1].nl}.")
    parts.append("Provide the final integer answer inside \\boxed{}.")
    prompt = " ".join(parts)

    d_s = _structural_difficulty(n_steps, chain, range_high)

    return {
        "id": f"FRONT-SYM-{seed:06d}",
        "domain": "symbolic_compose",
        "prompt": prompt,
        "ground_truth": answer,
        "answer_type": "integer",
        "d_structural": round(d_s, 4),
        "structural_params": {
            "n_steps": n_steps,
            "range": [range_low, range_high],
            "op_chain": [op.name for op in chain],
            "op_families": [op.family for op in chain],
        },
        "generation_seed": seed,
    }


def generate_dataset(n: int, seed_base: int = 100000) -> list[dict]:
    """Generate n problems with near-uniform d_structural bins."""
    out = []
    wanted = _target_bin_counts(n)
    counts = [0, 0, 0, 0, 0]
    attempts = 0
    max_attempts = max(1000, n * 80)
    while len(out) < n and attempts < max_attempts:
        needed = [i for i, c in enumerate(counts) if c < wanted[i]]
        target_bin = min(needed, key=lambda i: counts[i] / max(1, wanted[i]))
        target_d = (target_bin + 0.5) / 5.0
        prob = generate_one(seed_base + attempts, target_d=target_d)
        actual_bin = min(4, int(prob["d_structural"] * 5))
        if actual_bin == target_bin:
            out.append(prob)
            counts[actual_bin] += 1
        attempts += 1
    if len(out) < n:
        raise RuntimeError(f"Could only generate {len(out)}/{n} symbolic problems "
                           f"with balanced difficulty bins: {counts}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed_base", type=int, default=100000)
    ap.add_argument("--out", type=str, default="data/raw/symbolic.jsonl")
    args = ap.parse_args()

    problems = generate_dataset(args.n, args.seed_base)
    save_jsonl(problems, args.out)
    print(f"[symbolic] wrote {len(problems)} problems → {args.out}")
    # quick sanity: distribution of d_structural
    bins = [0, 0, 0, 0, 0]
    for p in problems:
        b = min(4, int(p["d_structural"] * 5))
        bins[b] += 1
    print(f"[symbolic] d_structural bin counts: {bins}")


if __name__ == "__main__":
    main()
