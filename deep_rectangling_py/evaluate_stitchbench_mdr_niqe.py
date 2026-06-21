from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .mdr_niqe import finite_mean

DEFAULT_BASELINE_ROOT = (
    r"C:\Users\22499\Documents\GitHub\OBJ-GSP\experiments\phase1_depth_loss"
    r"\runs\depth_gsp_v5_planarity035"
)


def parse_float(value: Any) -> float:
    if value in ("", None):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.5f}"


def pct(value: float) -> str:
    return "" if not math.isfinite(value) else f"{100.0 * value:.1f}%"


def rel_gap(candidate_value: float, baseline_value: float) -> float:
    if not math.isfinite(candidate_value) or not math.isfinite(baseline_value) or baseline_value == 0:
        return math.nan
    return candidate_value / baseline_value - 1.0


def load_per_pair(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["mdr_rmse"] = parse_float(row.get("mdr_rmse"))
            row["niqe"] = parse_float(row.get("niqe"))
            rows[row["dataset"]] = row
    return rows


def is_ok(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("status") == "ok" and math.isfinite(parse_float(row.get("mdr_rmse"))) and math.isfinite(parse_float(row.get("niqe"))))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(value) if isinstance(value, float) else value for key, value in row.items()})


def compare_to_baseline(output_root: Path, candidate_rows: list[dict[str, Any]], baseline_root: Path, baseline_name: str, candidate_name: str) -> None:
    baseline = load_per_pair(baseline_root / "per_pair.csv")
    candidate = {row["dataset"]: row for row in candidate_rows}
    names = sorted(set(baseline) | set(candidate))
    rows = []
    common_ok = []
    for name in names:
        base = baseline.get(name)
        cand = candidate.get(name)
        base_ok = is_ok(base)
        cand_ok = is_ok(cand)
        if base_ok and cand_ok:
            common_ok.append(name)
        base_mdr = parse_float(base.get("mdr_rmse")) if base else math.nan
        cand_mdr = parse_float(cand.get("mdr_rmse")) if cand else math.nan
        base_niqe = parse_float(base.get("niqe")) if base else math.nan
        cand_niqe = parse_float(cand.get("niqe")) if cand else math.nan
        mdr_delta = cand_mdr - base_mdr if base_ok and cand_ok else math.nan
        niqe_delta = cand_niqe - base_niqe if base_ok and cand_ok else math.nan
        rows.append(
            {
                "dataset": name,
                "category": (cand or base or {}).get("category", ""),
                "baseline_status": base.get("status", "missing") if base else "missing",
                "candidate_status": cand.get("status", "missing") if cand else "missing",
                "baseline_mdr": base_mdr,
                "candidate_mdr": cand_mdr,
                "mdr_delta": mdr_delta,
                "mdr_rel_gap": rel_gap(cand_mdr, base_mdr) if base_ok and cand_ok else math.nan,
                "mdr_better": int(mdr_delta < 0) if math.isfinite(mdr_delta) else "",
                "baseline_niqe": base_niqe,
                "candidate_niqe": cand_niqe,
                "niqe_delta": niqe_delta,
                "niqe_rel_gap": rel_gap(cand_niqe, base_niqe) if base_ok and cand_ok else math.nan,
                "niqe_better": int(niqe_delta < 0) if math.isfinite(niqe_delta) else "",
                "both_better": int(mdr_delta < 0 and niqe_delta < 0) if math.isfinite(mdr_delta) and math.isfinite(niqe_delta) else "",
                "candidate_result_image": cand.get("result_image", "") if cand else "",
                "baseline_result_image": base.get("result_image", "") if base else "",
            }
        )
    write_csv(output_root / "method_pair_comparison.csv", rows)

    by_category: dict[str, list[str]] = defaultdict(list)
    for name in common_ok:
        by_category[(candidate.get(name) or baseline.get(name)).get("category", "")].append(name)
    category_rows = []
    for category in sorted(by_category):
        category_names = by_category[category]
        both_better = [
            name
            for name in category_names
            if parse_float(candidate[name]["mdr_rmse"]) < parse_float(baseline[name]["mdr_rmse"])
            and parse_float(candidate[name]["niqe"]) < parse_float(baseline[name]["niqe"])
        ]
        category_rows.append(
            {
                "category": category,
                "common_ok_count": len(category_names),
                "candidate_mdr_mean": finite_mean([parse_float(candidate[name]["mdr_rmse"]) for name in category_names]),
                "baseline_mdr_mean": finite_mean([parse_float(baseline[name]["mdr_rmse"]) for name in category_names]),
                "candidate_niqe_mean": finite_mean([parse_float(candidate[name]["niqe"]) for name in category_names]),
                "baseline_niqe_mean": finite_mean([parse_float(baseline[name]["niqe"]) for name in category_names]),
                "both_better_count": len(both_better),
                "both_better_rate": len(both_better) / len(category_names),
            }
        )
    write_csv(output_root / "method_category_comparison.csv", category_rows)

    common_candidate_mdr = finite_mean([parse_float(candidate[name]["mdr_rmse"]) for name in common_ok])
    common_baseline_mdr = finite_mean([parse_float(baseline[name]["mdr_rmse"]) for name in common_ok])
    common_candidate_niqe = finite_mean([parse_float(candidate[name]["niqe"]) for name in common_ok])
    common_baseline_niqe = finite_mean([parse_float(baseline[name]["niqe"]) for name in common_ok])
    report = [
        f"# {candidate_name} vs {baseline_name}",
        "",
        f"- Common successful datasets: {len(common_ok)}",
        f"- Common mean MDR/RMSE: {candidate_name} {fmt(common_candidate_mdr)} vs {baseline_name} {fmt(common_baseline_mdr)}",
        f"- Common mean MDR relative gap: {pct(rel_gap(common_candidate_mdr, common_baseline_mdr))}",
        f"- Common mean NIQE: {candidate_name} {fmt(common_candidate_niqe)} vs {baseline_name} {fmt(common_baseline_niqe)}",
        f"- Common mean NIQE relative gap: {pct(rel_gap(common_candidate_niqe, common_baseline_niqe))}",
        "",
        "NIQE uses pyiqa on final panorama images. DeepRectangling MDR is overlap mapping RMSE; baseline MDR is OBJ-GSP C++ mesh RMSE.",
    ]
    (output_root / "method_comparison.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare DeepRectangling MDR/NIQE tables against OBJ-GSP baseline.")
    parser.add_argument("--candidate-root", default="outputs/stitchbench_general")
    parser.add_argument("--baseline-root", default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output-root", default="outputs/deeprect_mdr_niqe")
    parser.add_argument("--baseline-name", default="Depth-GSP-v5")
    parser.add_argument("--candidate-name", default="OpenCV+DeepRectangling")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_rows = load_per_pair(Path(args.candidate_root) / "per_pair.csv")
    rows = list(candidate_rows.values())
    write_csv(output_root / "per_pair.csv", rows)
    compare_to_baseline(output_root, rows, Path(args.baseline_root), args.baseline_name, args.candidate_name)
    print(f"Wrote comparison tables to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
