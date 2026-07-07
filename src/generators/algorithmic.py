"""Algorithmic Reasoning generator.

Each problem = a small graph (4-25 nodes) + a question solved by a textbook
graph algorithm. Ground truth via networkx.

Difficulty depends on: n_nodes, edge density, algorithm complexity, weighting.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import networkx as nx

from src.utils.io import save_jsonl


# (algorithm_name, complexity_in_[0,1])
ALGO_POOL = [
    ("node_degree_max", 0.10),
    ("connectivity", 0.15),
    ("shortest_path_length", 0.25),
    ("has_cycle", 0.25),
    ("is_bipartite", 0.30),
    ("graph_diameter", 0.45),
    ("min_spanning_tree_weight", 0.55),
]


def _target_bin_counts(n: int, n_bins: int = 5) -> list[int]:
    base, extra = divmod(n, n_bins)
    return [base + (1 if i < extra else 0) for i in range(n_bins)]


def _ensure_connected(G: nx.Graph) -> nx.Graph:
    """Return the largest connected component, relabeled 0..n-1."""
    if nx.is_connected(G):
        return G
    cc = max(nx.connected_components(G), key=len)
    H = G.subgraph(cc).copy()
    return nx.convert_node_labels_to_integers(H, first_label=0)


def _format_graph(G: nx.Graph, weighted: bool) -> str:
    nodes = sorted(G.nodes())
    if weighted:
        edges = sorted([(u, v, int(G[u][v]["weight"])) for u, v in G.edges()])
        return f"Nodes: {nodes}; Edges (u, v, weight): {edges}"
    edges = sorted([tuple(sorted(e)) for e in G.edges()])
    return f"Nodes: {nodes}; Edges: {edges}"


def generate_one(seed: int, target_d: float | None = None) -> dict:
    rng = random.Random(seed)

    if target_d is None:
        n_nodes = rng.randint(4, 25)
        algo_name, algo_comp = rng.choice(ALGO_POOL)
        density = rng.uniform(0.2, 0.7)
    elif target_d < 0.20:
        n_nodes = rng.randint(4, 7)
        algo_name, algo_comp = rng.choice([
            ("node_degree_max", 0.10),
            ("connectivity", 0.15),
        ])
        density = rng.uniform(0.15, 0.30)
    elif target_d < 0.40:
        n_nodes = rng.randint(7, 12)
        algo_name, algo_comp = rng.choice([
            ("connectivity", 0.15),
            ("has_cycle", 0.25),
            ("is_bipartite", 0.30),
        ])
        density = rng.uniform(0.25, 0.40)
    elif target_d < 0.60:
        n_nodes = rng.randint(12, 18)
        algo_name, algo_comp = rng.choice([
            ("shortest_path_length", 0.25),
            ("has_cycle", 0.25),
            ("is_bipartite", 0.30),
            ("graph_diameter", 0.45),
        ])
        density = rng.uniform(0.35, 0.55)
    elif target_d < 0.80:
        n_nodes = rng.randint(18, 24)
        algo_name, algo_comp = rng.choice([
            ("shortest_path_length", 0.25),
            ("graph_diameter", 0.45),
            ("min_spanning_tree_weight", 0.55),
        ])
        density = rng.uniform(0.45, 0.65)
    else:
        n_nodes = rng.randint(23, 25)
        algo_name, algo_comp = ("min_spanning_tree_weight", 0.55)
        density = rng.uniform(0.62, 0.72)

    weighted = algo_name in ("shortest_path_length", "min_spanning_tree_weight")

    # Sample a graph; for connectivity-style algos, use the largest CC.
    G = nx.gnp_random_graph(n_nodes, density, seed=seed)
    if weighted:
        for u, v in G.edges():
            G[u][v]["weight"] = rng.randint(1, 10)

    needs_connected = algo_name in (
        "shortest_path_length", "min_spanning_tree_weight", "graph_diameter"
    )
    if needs_connected:
        G = _ensure_connected(G)
        if G.number_of_nodes() < 3:
            # Degenerate: regenerate with a different seed.
            return generate_one(seed + 7919, target_d)
        if weighted:
            for u, v in G.edges():
                if "weight" not in G[u][v]:
                    G[u][v]["weight"] = rng.randint(1, 10)

    n_eff = G.number_of_nodes()
    nodes = sorted(G.nodes())

    # Ground truth + query
    if algo_name == "shortest_path_length":
        src, tgt = rng.sample(nodes, 2)
        answer = nx.shortest_path_length(G, src, tgt, weight="weight")
        query = f"What is the shortest path length (sum of weights) from node {src} to node {tgt}?"

    elif algo_name == "connectivity":
        # Use full graph (not the CC restriction).
        G_full = nx.gnp_random_graph(n_nodes, density, seed=seed)
        if G_full.number_of_nodes() < 2:
            return generate_one(seed + 7919, target_d)
        u, v = rng.sample(list(G_full.nodes()), 2)
        answer = "Yes" if nx.has_path(G_full, u, v) else "No"
        G = G_full
        query = f"Is there a path from node {u} to node {v}? Answer Yes or No."

    elif algo_name == "has_cycle":
        try:
            nx.find_cycle(G, orientation="ignore")
            answer = "Yes"
        except nx.NetworkXNoCycle:
            answer = "No"
        query = "Does the graph contain a cycle? Answer Yes or No."

    elif algo_name == "graph_diameter":
        answer = nx.diameter(G)
        query = "What is the diameter of the graph (maximum shortest-path distance between any two nodes)?"

    elif algo_name == "is_bipartite":
        answer = "Yes" if nx.is_bipartite(G) else "No"
        query = "Is the graph bipartite? Answer Yes or No."

    elif algo_name == "min_spanning_tree_weight":
        T = nx.minimum_spanning_tree(G)
        answer = sum(int(d["weight"]) for _, _, d in T.edges(data=True))
        query = "What is the total weight of a minimum spanning tree?"

    elif algo_name == "node_degree_max":
        if G.number_of_edges() == 0:
            return generate_one(seed + 7919, target_d)
        answer = max(dict(G.degree()).values())
        query = "What is the maximum degree across all nodes?"

    else:
        raise ValueError(f"Unknown algo {algo_name}")

    graph_text = _format_graph(G, weighted)
    prompt = (
        f"Graph: {graph_text}\n"
        f"Question: {query} Put your final answer inside \\boxed{{}}."
    )

    is_int_answer = isinstance(answer, int)
    answer_str = str(answer)

    d_s = (
        0.35 * (n_eff / 25.0)
        + 0.20 * density
        + 0.30 * algo_comp
        + 0.15 * (1.0 if weighted else 0.0)
    )
    d_s = float(min(1.0, max(0.0, d_s)))

    return {
        "id": f"FRONT-ALG-{seed:06d}",
        "domain": "algorithmic_reasoning",
        "prompt": prompt,
        "ground_truth": answer_str,
        "answer_type": "integer" if is_int_answer else "yes_no",
        "d_structural": round(d_s, 4),
        "structural_params": {
            "n_nodes": n_eff,
            "density": round(density, 3),
            "algorithm": algo_name,
            "weighted": weighted,
            "n_edges": G.number_of_edges(),
        },
        "generation_seed": seed,
    }


def generate_dataset(n: int, seed_base: int = 300000) -> list[dict]:
    out = []
    wanted = _target_bin_counts(n)
    counts = [0, 0, 0, 0, 0]
    attempts = 0
    max_attempts = max(1000, n * 100)
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
        except Exception as e:  # noqa
            pass
        attempts += 1
    if len(out) < n:
        raise RuntimeError(f"Could only generate {len(out)}/{n} algorithmic problems "
                           f"with balanced difficulty bins: {counts}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed_base", type=int, default=300000)
    ap.add_argument("--out", type=str, default="data/raw/algorithmic.jsonl")
    args = ap.parse_args()

    problems = generate_dataset(args.n, args.seed_base)
    save_jsonl(problems, args.out)
    print(f"[algorithmic] wrote {len(problems)} problems → {args.out}")

    by_algo = {}
    bins = [0, 0, 0, 0, 0]
    for p in problems:
        a = p["structural_params"]["algorithm"]
        by_algo[a] = by_algo.get(a, 0) + 1
        bins[min(4, int(p["d_structural"] * 5))] += 1
    print(f"[algorithmic] per-algo counts: {by_algo}")
    print(f"[algorithmic] d_structural bin counts: {bins}")


if __name__ == "__main__":
    main()
