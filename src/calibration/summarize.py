"""Aggregate per-(model, domain) pass rates from data/calibrated/*.jsonl.

File-name convention produced by run_eval.py + eval_one_model.sh:
    <model>__<domain>.jsonl    (e.g. Qwen3-4B__symbolic.jsonl)
Files without a `__` separator are treated as having tag "default" so that
legacy runs are still aggregated.

Outputs:
    data/calibrated/SUMMARY.md   (human-readable Markdown table + failure samples)
    data/calibrated/summary.csv  (machine-readable, opens in Excel)

Usage:
    python -m src.calibration.summarize
    python -m src.calibration.summarize --dir data/calibrated --show_failures 3
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Tuple


# Preferred display order; unknown domains are appended alphabetically.
DOMAIN_ORDER = ["symbolic", "logical", "algorithmic"]


def parse_filename(stem: str) -> Tuple[str, str]:
    """'Qwen3-4B__symbolic' -> ('Qwen3-4B', 'symbolic')

    Stems without the '__' separator fall back to ('<stem>', 'default'),
    so legacy outputs from before run_eval.py grew the --tag flag still load.
    """
    if "__" in stem:
        model, tag = stem.rsplit("__", 1)
        return model, tag
    return stem, "default"


def load_jsonl(path: Path) -> list:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _sort_domains(found: set) -> list:
    """Domains in DOMAIN_ORDER come first (in that order), unknowns appended."""
    known = [d for d in DOMAIN_ORDER if d in found]
    unknown = sorted(d for d in found if d not in DOMAIN_ORDER)
    return known + unknown


def _format_cell(c: int, n: int) -> str:
    if n == 0:
        return "—"
    return f"{c / n:.3f} ({c}/{n})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/calibrated",
                    help="Directory holding <model>__<domain>.jsonl files.")
    ap.add_argument("--out_md", default=None,
                    help="Markdown output path (default: <dir>/SUMMARY.md).")
    ap.add_argument("--out_csv", default=None,
                    help="CSV output path (default: <dir>/summary.csv).")
    ap.add_argument("--show_failures", type=int, default=2,
                    help="How many failed problem IDs to list per cell in Markdown.")
    args = ap.parse_args()

    base = Path(args.dir)
    if not base.exists():
        print(f"[summarize] ERROR: directory not found: {base}")
        return
    out_md = Path(args.out_md) if args.out_md else base / "SUMMARY.md"
    out_csv = Path(args.out_csv) if args.out_csv else base / "summary.csv"

    # --- load all jsonl files ---
    cell_records: dict = {}             # (model, domain) -> list[dict]
    for jp in sorted(base.glob("*.jsonl")):
        model, dom = parse_filename(jp.stem)
        cell_records[(model, dom)] = load_jsonl(jp)

    if not cell_records:
        print(f"[summarize] no .jsonl files found in {base}")
        return

    # --- compute table ---
    models = sorted({m for (m, _) in cell_records.keys()})
    domains = _sort_domains({d for (_, d) in cell_records.keys()})

    # cell_stats[(model, domain)] = (correct, total)
    cell_stats: dict = {}
    for key, recs in cell_records.items():
        c = sum(int(r.get("correct", 0)) for r in recs)
        n = len(recs)
        cell_stats[key] = (c, n)

    # --- write CSV ---
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["model"] + domains + ["overall_correct", "overall_total", "overall_rate"]
        w.writerow(header)
        for m in models:
            row = [m]
            tot_c, tot_n = 0, 0
            for d in domains:
                if (m, d) in cell_stats:
                    c, n = cell_stats[(m, d)]
                    row.append(f"{c / n:.4f}" if n else "")
                    tot_c += c
                    tot_n += n
                else:
                    row.append("")
            row.append(tot_c)
            row.append(tot_n)
            row.append(f"{tot_c / tot_n:.4f}" if tot_n else "")
            w.writerow(row)

    # --- write Markdown ---
    md_lines = []
    md_lines.append("# Calibration summary")
    md_lines.append("")
    md_lines.append(f"Source directory: `{base}`")
    md_lines.append("")

    # main table
    header = ["Model"] + [d.capitalize() for d in domains] + ["Overall"]
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for m in models:
        row = [m]
        tot_c, tot_n = 0, 0
        for d in domains:
            if (m, d) in cell_stats:
                c, n = cell_stats[(m, d)]
                row.append(_format_cell(c, n))
                tot_c += c
                tot_n += n
            else:
                row.append("—")
        if tot_n:
            row.append(f"**{tot_c / tot_n:.3f}** ({tot_c}/{tot_n})")
        else:
            row.append("—")
        md_lines.append("| " + " | ".join(row) + " |")
    md_lines.append("")

    # failure samples
    if args.show_failures > 0:
        md_lines.append("## Failed problems (sample)")
        md_lines.append("")
        any_fail = False
        for m in models:
            for d in domains:
                recs = cell_records.get((m, d), [])
                fails = [r for r in recs if int(r.get("correct", 0)) == 0]
                if not fails:
                    continue
                any_fail = True
                md_lines.append(f"### `{m}` × `{d}`  ({len(fails)} fails)")
                md_lines.append("")
                for r in fails[: args.show_failures]:
                    pred = (r.get("pred", "") or "").replace("\n", " ").replace("|", "\\|")[:120]
                    md_lines.append(f"- **{r.get('id', '?')}**  pred: `{pred}`")
                md_lines.append("")
        if not any_fail:
            md_lines.append("(no failures recorded across all cells)")
            md_lines.append("")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # --- console summary ---
    print(f"[summarize] {len(cell_records)} (model, domain) cell(s) loaded "
          f"from {base}")
    print(f"[summarize] models  : {models}")
    print(f"[summarize] domains : {domains}")
    print()

    # plain-text table
    model_w = max(5, max(len(m) for m in models)) + 2
    col_w = 16
    head_line = "model".ljust(model_w) + "".join(d[:col_w - 2].ljust(col_w) for d in domains) + "overall"
    print(head_line)
    print("-" * len(head_line))
    for m in models:
        line = m.ljust(model_w)
        tot_c, tot_n = 0, 0
        for d in domains:
            if (m, d) in cell_stats:
                c, n = cell_stats[(m, d)]
                cell = f"{c / n:.2f} ({c}/{n})" if n else "—"
                line += cell.ljust(col_w)
                tot_c += c
                tot_n += n
            else:
                line += "—".ljust(col_w)
        line += f"{tot_c / tot_n:.2f} ({tot_c}/{tot_n})" if tot_n else "—"
        print(line)
    print()
    print(f"[summarize] Markdown -> {out_md}")
    print(f"[summarize] CSV      -> {out_csv}")


if __name__ == "__main__":
    main()
