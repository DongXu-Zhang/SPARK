"""Logical Inference generator.

Each problem = facts + Horn-clause rules + distractors + a query.
Ground truth via forward chaining over an instantiated propositional KB.

Reasoning depth (1-6) is the dominant difficulty knob: the answer requires
chaining `depth` rule applications.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from src.utils.io import save_jsonl


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

UNARY_PREDS = [
    "tall", "fast", "kind", "happy", "curious", "careful", "clever",
    "brave", "quiet", "noisy", "patient", "creative", "fit", "tired",
    "honest", "thoughtful",
]
SUBJECTS = [
    "Alice", "Bob", "Carol", "Dan", "Eve", "Finn", "Grace", "Henry",
    "Ivy", "Jack",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Atom:
    pred: str
    subj: str

    def to_text(self) -> str:
        return f"{self.subj} is {self.pred}"


@dataclass
class Rule:
    premises: tuple[Atom, ...]
    conclusion: Atom

    def to_text(self) -> str:
        if len(self.premises) == 1:
            prem = self.premises[0].to_text()
        else:
            prem = " and ".join(p.to_text() for p in self.premises)
        return f"If {prem}, then {self.conclusion.to_text()}."


# ---------------------------------------------------------------------------
# Forward chaining
# ---------------------------------------------------------------------------

def forward_chain(facts: list[Atom], rules: list[Rule], max_iter: int = 50) -> set[Atom]:
    known = set(facts)
    for _ in range(max_iter):
        added = set()
        for rule in rules:
            if rule.conclusion in known:
                continue
            if all(p in known for p in rule.premises):
                added.add(rule.conclusion)
        if not added:
            break
        known |= added
    return known


def _target_bin_counts(n: int, n_bins: int = 5) -> list[int]:
    base, extra = divmod(n, n_bins)
    return [base + (1 if i < extra else 0) for i in range(n_bins)]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_one(seed: int, target_d: float | None = None) -> dict:
    rng = random.Random(seed)

    if target_d is None:
        depth = rng.randint(1, 6)
        n_distractors = rng.randint(0, 10)
    elif target_d < 0.20:
        depth = 1
        n_distractors = rng.randint(0, 2)
    elif target_d < 0.40:
        depth = rng.choice([2, 3])
        n_distractors = rng.randint(2, 5)
    elif target_d < 0.60:
        depth = rng.choice([3, 4])
        n_distractors = rng.randint(4, 7)
    elif target_d < 0.80:
        depth = rng.choice([4, 5])
        n_distractors = rng.randint(7, 10)
    else:
        depth = 6
        n_distractors = rng.randint(10, 12)

    n_subjects = rng.randint(3, 5)
    subjects = rng.sample(SUBJECTS, n_subjects)
    target_subj = subjects[0]

    # Sample `depth + 1` distinct predicates for the inference chain.
    # Plus a few extras to use as distractors.
    n_pred_total = depth + 1 + rng.randint(3, 6)
    pred_pool = rng.sample(UNARY_PREDS, min(n_pred_total, len(UNARY_PREDS)))
    chain_preds = pred_pool[: depth + 1]
    extra_preds = pred_pool[depth + 1:]

    # Build the canonical chain: P0(target) -> P1(target) -> ... -> P_depth(target)
    facts: list[Atom] = [Atom(chain_preds[0], target_subj)]
    rules: list[Rule] = []
    for i in range(depth):
        # Mix single- and two-premise rules to vary surface form.
        if i > 0 and rng.random() < 0.4 and extra_preds:
            # Two-premise rule. Add the auxiliary fact too.
            aux_pred = rng.choice(extra_preds)
            aux_fact = Atom(aux_pred, target_subj)
            facts.append(aux_fact)
            rules.append(Rule(
                premises=(Atom(chain_preds[i], target_subj), aux_fact),
                conclusion=Atom(chain_preds[i + 1], target_subj),
            ))
        else:
            rules.append(Rule(
                premises=(Atom(chain_preds[i], target_subj),),
                conclusion=Atom(chain_preds[i + 1], target_subj),
            ))

    # Inject distractors (unrelated facts and unrelated rules).
    for _ in range(n_distractors):
        if rng.random() < 0.5 and extra_preds:
            s = rng.choice([s for s in subjects if s != target_subj] or subjects)
            p = rng.choice(extra_preds)
            facts.append(Atom(p, s))
        elif len(extra_preds) >= 2:
            p1, p2 = rng.sample(extra_preds, 2)
            s = rng.choice(subjects)
            rules.append(Rule(premises=(Atom(p1, s),), conclusion=Atom(p2, s)))

    derived = forward_chain(facts, rules)

    # Balance positive and unknown queries. Pick the query after deriving the
    # closure so negative examples are genuinely not entailed.
    want_positive = (seed % 2 == 0)
    positive_query = Atom(chain_preds[depth], target_subj)
    if want_positive:
        query = positive_query
    else:
        known_target_preds = {a.pred for a in derived if a.subj == target_subj}
        neg_preds = [p for p in UNARY_PREDS if p not in known_target_preds]
        query = Atom(rng.choice(neg_preds), target_subj) if neg_preds else positive_query
    answer = "Yes" if query in derived else "Unknown"

    rng.shuffle(facts)
    rng.shuffle(rules)
    facts_text = " ".join(f.to_text() + "." for f in facts)
    rules_text = " ".join(r.to_text() for r in rules)
    prompt = (
        f"Facts: {facts_text}\n"
        f"Rules: {rules_text}\n"
        f"Question: Is it true that {query.to_text()}? "
        f"Answer with exactly one of: Yes, Unknown. "
        f"Put your answer inside \\boxed{{}}."
    )

    d_s = (
        0.55 * (depth / 6.0)
        + 0.25 * (n_distractors / 10.0)
        + 0.20 * (any(len(r.premises) == 2 for r in rules))
    )
    d_s = float(min(1.0, max(0.0, d_s)))

    return {
        "id": f"FRONT-LOG-{seed:06d}",
        "domain": "logical_inference",
        "prompt": prompt,
        "ground_truth": answer,
        "answer_type": "yes_unknown",
        "d_structural": round(d_s, 4),
        "structural_params": {
            "depth": depth,
            "n_distractors": n_distractors,
            "n_facts": len(facts),
            "n_rules": len(rules),
            "two_premise_rules": sum(1 for r in rules if len(r.premises) == 2),
        },
        "generation_seed": seed,
    }


def generate_dataset(n: int, seed_base: int = 200000) -> list[dict]:
    out = []
    wanted = _target_bin_counts(n)
    counts = [0, 0, 0, 0, 0]
    attempts = 0
    max_attempts = max(1000, n * 80)
    while len(out) < n and attempts < max_attempts:
        needed = [i for i, c in enumerate(counts) if c < wanted[i]]
        target_bin = min(needed, key=lambda i: counts[i] / max(1, wanted[i]))
        target_d = (target_bin + 0.5) / 5.0
        try:
            prob = generate_one(seed_base + attempts, target_d=target_d)
            actual_bin = min(4, int(prob["d_structural"] * 5))
            if actual_bin == target_bin:
                out.append(prob)
                counts[actual_bin] += 1
        except Exception:
            pass
        attempts += 1
    if len(out) < n:
        raise RuntimeError(f"Could only generate {len(out)}/{n} logical problems "
                           f"with balanced difficulty bins: {counts}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed_base", type=int, default=200000)
    ap.add_argument("--out", type=str, default="data/raw/logical.jsonl")
    args = ap.parse_args()

    problems = generate_dataset(args.n, args.seed_base)
    save_jsonl(problems, args.out)
    print(f"[logical] wrote {len(problems)} problems → {args.out}")

    yes_count = sum(1 for p in problems if p["ground_truth"] == "Yes")
    bins = [0, 0, 0, 0, 0]
    for p in problems:
        b = min(4, int(p["d_structural"] * 5))
        bins[b] += 1
    print(f"[logical] Yes: {yes_count}, Unknown: {len(problems) - yes_count}")
    print(f"[logical] d_structural bin counts: {bins}")


if __name__ == "__main__":
    main()
