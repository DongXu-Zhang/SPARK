"""End-to-end smoke test on 30 problems per domain.

Runs entirely on CPU — no GPU, no vLLM required. Verifies that:
  1. all three generators produce valid problems
  2. ground-truth verifiers accept the canonical answer
  3. d_structural distribution roughly covers [0,1]
  4. JSONL serialization round-trips cleanly

Run from the project root:
    python scripts/sanity_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when run as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from src.calibration.verify import verify
from src.generators import symbolic, logical, algorithmic
from src.utils.io import load_jsonl, save_jsonl


N_PER_DOMAIN = 30
SANITY_DIR = ROOT / "data" / "sanity"


def _bucket_distribution(problems, key="d_structural", n_bins=5):
    bins = [0] * n_bins
    for p in problems:
        b = min(n_bins - 1, int(p[key] * n_bins))
        bins[b] += 1
    return bins


def _check_verifier(problems, label: str):
    fail = 0
    for p in problems:
        # Canonical answer must verify true.
        gt_str = str(p["ground_truth"])
        if not verify(gt_str, p["ground_truth"], p["answer_type"]):
            print(f"  ✗ verifier rejects own ground truth: {p['id']} "
                  f"(gt={p['ground_truth']!r}, type={p['answer_type']})")
            fail += 1
    print(f"  verifier self-check ({label}): "
          f"{len(problems) - fail}/{len(problems)} pass")
    return fail == 0


def _show_examples(problems, label: str, k: int = 3):
    print(f"\n  --- {label}: 3 random examples ---")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(problems), size=min(k, len(problems)), replace=False)
    for i in idx:
        p = problems[int(i)]
        print(f"  [{p['id']}  d_s={p['d_structural']:.2f}]")
        print(f"  Q: {p['prompt'][:280]}{'...' if len(p['prompt']) > 280 else ''}")
        print(f"  A: {p['ground_truth']}")
        print()


def run_domain(name: str, gen_fn, seed_base: int):
    print(f"\n=== {name} ===")
    problems = gen_fn(N_PER_DOMAIN, seed_base=seed_base)
    print(f"  generated {len(problems)} problems")

    out_path = SANITY_DIR / f"{name}.jsonl"
    save_jsonl(problems, out_path)
    reloaded = load_jsonl(out_path)
    assert len(reloaded) == len(problems), "round-trip mismatch"

    bins = _bucket_distribution(problems)
    print(f"  d_structural bin counts (5 bins): {bins}")
    if min(bins) == 0:
        print(f"  NOTE: some d_structural bins are empty for {name} "
              f"(expected for n=30; full 1500 should fill all 5).")

    ok = _check_verifier(problems, name)
    _show_examples(problems, name)
    return ok


def main():
    SANITY_DIR.mkdir(parents=True, exist_ok=True)
    print(f"writing sanity outputs to {SANITY_DIR}")

    results = []
    results.append(("symbolic_compose",
                    run_domain("symbolic", symbolic.generate_dataset, 100000)))
    results.append(("logical_inference",
                    run_domain("logical", logical.generate_dataset, 200000)))
    results.append(("algorithmic_reasoning",
                    run_domain("algorithmic", algorithmic.generate_dataset, 300000)))

    print("\n=== summary ===")
    for name, ok in results:
        print(f"  {name}: {'OK' if ok else 'FAIL'}")
    if all(ok for _, ok in results):
        print("\nAll sanity checks passed. You can now run scripts/generate_all.py.")
        sys.exit(0)
    else:
        print("\nSome checks failed; please inspect the offending problems above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
