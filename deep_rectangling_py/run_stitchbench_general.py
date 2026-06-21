from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from .io_utils import list_images, read_bgr, save_bgr, write_json
from .mdr_niqe import (
    CATEGORIES,
    DEFAULT_MANIFEST,
    category_for,
    compute_mdr_rmse,
    compute_niqe,
    finite_mean,
    load_niqe_metric,
)
from .stitch import stitch_pair

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TF_SCRIPT = REPO_ROOT / "Codes_for_Arbitrary_Resolution" / "batch_inference.py"
DEFAULT_CHECKPOINT = REPO_ROOT / "Codes" / "checkpoints" / "pretrained_model" / "model.ckpt-100000"


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_float(value: Any) -> Any:
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.5f}"
    return value


def _mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask


MAX_RECTANGLE_SIDE = 2048


def _maybe_downscale_for_rectangling(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side <= MAX_RECTANGLE_SIDE:
        return image
    scale = MAX_RECTANGLE_SIDE / float(long_side)
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _prepare_tf_input(stitched_path: Path, mask_path: Path, scene_out: Path) -> tuple[Path, Path]:
    tf_input_dir = scene_out / "tf_input"
    tf_mask_dir = scene_out / "tf_mask"
    tf_input_dir.mkdir(parents=True, exist_ok=True)
    tf_mask_dir.mkdir(parents=True, exist_ok=True)
    input_copy = tf_input_dir / "00001.jpg"
    mask_copy = tf_mask_dir / "00001.jpg"
    stitched = _maybe_downscale_for_rectangling(read_bgr(stitched_path))
    mask = _maybe_downscale_for_rectangling(read_bgr(mask_path))
    save_bgr(input_copy, stitched)
    save_bgr(mask_copy, mask)
    return input_copy, mask_copy


def _run_rectangling(jobs: list[dict[str, str]], checkpoint: Path, skip_existing: bool = True) -> tuple[int, int]:
    import importlib.util

    ar_dir = REPO_ROOT / "Codes_for_Arbitrary_Resolution"
    if str(ar_dir) not in sys.path:
        sys.path.insert(0, str(ar_dir))
    spec = importlib.util.spec_from_file_location("dr_batch_inference", DEFAULT_TF_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {DEFAULT_TF_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run_single_job = module.run_single_job

    ok = 0
    failed = 0
    for job in jobs:
        output_path = Path(job["output"])
        if skip_existing and output_path.exists():
            print(f"skip existing {output_path}")
            ok += 1
            continue
        success, message = run_single_job(
            {
                "input": str(Path(job["input"]).resolve()),
                "mask": str(Path(job["mask"]).resolve()),
                "output": str(output_path.resolve()),
            },
            checkpoint.resolve(),
        )
        print(message)
        if success:
            ok += 1
        else:
            failed += 1
    print(f"rectangling summary: ok={ok}, failed={failed}")
    return ok, failed


def stage_existing_scene(scene_dir: Path, out_root: Path, category: str, pending_jobs: list[dict[str, str]]) -> dict[str, Any]:
    scene_name = scene_dir.name
    scene_out = out_root / scene_name
    images = list_images(scene_dir)
    if len(images) < 2:
        raise ValueError(f"Scene has fewer than two images: {scene_dir}")
    img_path1, img_path2 = images[:2]
    stitched_path = scene_out / "stitched.png"
    mask_path = scene_out / "mask.png"
    if not stitched_path.exists() or not mask_path.exists():
        raise FileNotFoundError(f"Missing stitched outputs for {scene_name}")
    rectangled_path = scene_out / "rectangled.png"
    if not rectangled_path.exists():
        input_copy, mask_copy = _prepare_tf_input(stitched_path, mask_path, scene_out)
        pending_jobs.append({"scene": scene_name, "input": str(input_copy), "mask": str(mask_copy), "output": str(rectangled_path)})
    return {
        "dataset": scene_name,
        "category": category,
        "image1": str(img_path1),
        "image2": str(img_path2),
        "stitched_path": str(stitched_path),
        "result_image": str(rectangled_path),
        "_img1": read_bgr(img_path1),
        "_img2": read_bgr(img_path2),
    }
def process_scene(
    scene_dir: Path,
    out_root: Path,
    category: str,
    pending_jobs: list[dict[str, str]],
    allow_stitch_fallback: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    scene_name = scene_dir.name
    scene_out = out_root / scene_name
    images = list_images(scene_dir)
    if len(images) < 2:
        raise ValueError(f"Scene has fewer than two images: {scene_dir}")

    img_path1, img_path2 = images[:2]
    img1 = read_bgr(img_path1)
    img2 = read_bgr(img_path2)
    stitched, mask, stitch_method = stitch_pair(img1, img2, allow_fallback=allow_stitch_fallback)

    stitched_path = scene_out / "stitched.png"
    mask_path = scene_out / "mask.png"
    save_bgr(stitched_path, stitched)
    save_bgr(mask_path, _mask_to_bgr(mask))

    rectangled_path = scene_out / "rectangled.png"
    if not rectangled_path.exists():
        input_copy, mask_copy = _prepare_tf_input(stitched_path, mask_path, scene_out)
        pending_jobs.append({"scene": scene_name, "input": str(input_copy), "mask": str(mask_copy), "output": str(rectangled_path)})

    return {
        "dataset": scene_name,
        "category": category,
        "image1": str(img_path1),
        "image2": str(img_path2),
        "stitched_path": str(stitched_path),
        "stitch_method": stitch_method,
        "result_image": str(rectangled_path),
        "elapsed_stitch_sec": time.perf_counter() - started,
        "_img1": img1,
        "_img2": img2,
    }


def finalize_metrics(rows: list[dict[str, Any]], niqe_metric) -> None:
    for row in rows:
        rectangled_path = Path(row["result_image"])
        if not rectangled_path.exists():
            if row.get("status") != "failed" or not row.get("error"):
                row.update({"status": "failed", "error": "Rectangling output missing.", "mdr_rmse": math.nan, "niqe": math.nan})
            continue
        try:
            mdr = compute_mdr_rmse(row.pop("_img1"), row.pop("_img2"), read_bgr(rectangled_path))
            niqe = compute_niqe(niqe_metric, rectangled_path)
            row.update(
                {
                    "status": "ok",
                    "error": "",
                    "mdr_rmse": mdr["mdr_rmse"],
                    "warping_residual_avg": mdr["warping_residual_avg"],
                    "warping_residual_sd": mdr["warping_residual_sd"],
                    "niqe": niqe,
                    "metric_note": mdr["metric_note"],
                }
            )
        except Exception as exc:
            row.pop("_img1", None)
            row.pop("_img2", None)
            row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "mdr_rmse": math.nan, "niqe": math.nan})


def write_by_category(out_root: Path, rows: list[dict[str, Any]]) -> None:
    category_rows = []
    for category in CATEGORIES:
        group = [row for row in rows if row.get("category") == category]
        if not group:
            continue
        ok_group = [row for row in group if row.get("status") == "ok"]
        mdr_values = [float(row["mdr_rmse"]) for row in ok_group if math.isfinite(float(row.get("mdr_rmse", math.nan)))]
        niqe_values = [float(row["niqe"]) for row in ok_group if math.isfinite(float(row.get("niqe", math.nan)))]
        category_rows.append(
            {
                "category": category,
                "total_count": len(group),
                "valid_mdr_count": len(mdr_values),
                "valid_niqe_count": len(niqe_values),
                "mdr_rmse_mean": finite_mean(mdr_values),
                "warping_residual_avg_mean": finite_mean([float(row.get("warping_residual_avg", math.nan)) for row in ok_group]),
                "warping_residual_sd_mean": finite_mean([float(row.get("warping_residual_sd", math.nan)) for row in ok_group]),
                "niqe_mean": finite_mean(niqe_values),
            }
        )
    _write_csv(
        out_root / "by_category.csv",
        [{k: _format_float(v) for k, v in row.items()} for row in category_rows],
        ["category", "total_count", "valid_mdr_count", "valid_niqe_count", "mdr_rmse_mean", "warping_residual_avg_mean", "warping_residual_sd_mean", "niqe_mean"],
    )


def write_report(out_root: Path, rows: list[dict[str, Any]]) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    mean_mdr = finite_mean([float(r["mdr_rmse"]) for r in ok_rows]) if ok_rows else math.nan
    mean_niqe = finite_mean([float(r["niqe"]) for r in ok_rows]) if ok_rows else math.nan
    lines = [
        "# DeepRectangling StitchBench General MDR/NIQE Report",
        "",
        "Pipeline: OpenCV Stitcher + DeepRectangling rectangling.",
        "",
        f"- Total datasets: {len(rows)}",
        f"- Valid MDR/NIQE datasets: {len(ok_rows)}",
        f"- Failed datasets: {len(rows) - len(ok_rows)}",
        f"- Mean MDR/RMSE: {mean_mdr:.5f}" if ok_rows else "- Mean MDR/RMSE: n/a",
        f"- Mean NIQE: {mean_niqe:.5f}" if ok_rows else "- Mean NIQE: n/a",
        "",
        "MDR is overlap mapping RMSE in final rectangled panorama coordinates. NIQE uses pyiqa on `rectangled.png`, matching OBJ-GSP evaluation style.",
        "",
        "## Failed",
        "",
    ]
    failed = [row for row in rows if row.get("status") != "ok"]
    lines.extend(f"- {row['dataset']}: {row.get('error', row.get('status'))}" for row in failed) if failed else lines.append("- None")
    (out_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OpenCV+DeepRectangling on StitchBench General with MDR/NIQE.")
    parser.add_argument("--dataset", default=r"D:\StitchBench\General")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default="outputs/stitchbench_general")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--scene", action="append", default=None)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--tf-python", default=sys.executable)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--gpu", default=os.environ.get("DEEPRECT_GPU", "0"))
    parser.add_argument("--skip-rectangling", action="store_true")
    parser.add_argument("--skip-stitch", action="store_true", help="Reuse existing stitched.png/mask.png in output folders.")
    parser.add_argument(
        "--allow-stitch-fallback",
        action="store_true",
        help="If OpenCV Stitcher fails, retry with SIFT homography stitching.",
    )
    parser.add_argument(
        "--retry-missing-stitch",
        action="store_true",
        help="Only process scenes without stitched.png; implies --allow-stitch-fallback.",
    )
    return parser


def _load_existing_rows(out_root: Path) -> dict[str, dict[str, Any]]:
    per_pair_path = out_root / "per_pair.csv"
    if not per_pair_path.exists():
        return {}
    with per_pair_path.open(newline="", encoding="utf-8") as handle:
        return {row["dataset"]: row for row in csv.DictReader(handle)}


def _merge_rows(existing: dict[str, dict[str, Any]], updated: list[dict[str, Any]], manifest_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    merged = dict(existing)
    for row in updated:
        merged[row["dataset"]] = {k: v for k, v in row.items() if not k.startswith("_")}
    order = [row["dataset"] for row in manifest_rows]
    return [merged[name] for name in order if name in merged]


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")
    args = build_arg_parser().parse_args(argv)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    full_manifest = _load_manifest(manifest_path)
    manifest_rows = list(full_manifest)
    if args.retry_missing_stitch:
        args.allow_stitch_fallback = True
        manifest_rows = [
            row for row in manifest_rows if not (out_root / row["dataset"] / "stitched.png").exists()
        ]
    if args.scene:
        wanted = set(args.scene)
        manifest_rows = [row for row in manifest_rows if row["dataset"] in wanted]
    if args.limit and args.limit > 0:
        manifest_rows = manifest_rows[: args.limit]
    existing_rows = _load_existing_rows(out_root) if args.retry_missing_stitch else {}

    niqe_metric, _ = load_niqe_metric(args.device)
    pending_jobs: list[dict[str, str]] = []
    staged: list[dict[str, Any]] = []

    for row in tqdm(manifest_rows, desc="StitchBench scenes"):
        dataset = row["dataset"]
        scene_dir = Path(row["data_dir"])
        category = row.get("category") or category_for(dataset) or ""
        try:
            if args.skip_stitch:
                staged.append(stage_existing_scene(scene_dir, out_root, category, pending_jobs))
            else:
                staged.append(
                    process_scene(
                        scene_dir,
                        out_root,
                        category,
                        pending_jobs,
                        allow_stitch_fallback=args.allow_stitch_fallback,
                    )
                )
        except Exception as exc:
            failed = {
                "dataset": dataset,
                "category": category,
                "result_image": str(out_root / dataset / "rectangled.png"),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "mdr_rmse": math.nan,
                "warping_residual_avg": math.nan,
                "warping_residual_sd": math.nan,
                "niqe": math.nan,
            }
            write_json(out_root / dataset / "metrics.json", {**failed, "traceback": traceback.format_exc()})
            staged.append(failed)
            if args.stop_on_error:
                raise

    if not args.skip_rectangling and pending_jobs:
        os.environ["DEEPRECT_GPU"] = args.gpu
        _run_rectangling(pending_jobs, Path(args.checkpoint), skip_existing=True)

    finalize_metrics(staged, niqe_metric)

    per_pair_fieldnames = [
        "dataset",
        "category",
        "result_image",
        "stitch_method",
        "mdr_rmse",
        "warping_residual_avg",
        "warping_residual_sd",
        "niqe",
        "status",
        "error",
        "metric_note",
    ]
    per_pair_rows = [{key: row.get(key, "") for key in per_pair_fieldnames} for row in staged]
    if existing_rows:
        per_pair_rows = _merge_rows(existing_rows, per_pair_rows, full_manifest)
        per_pair_rows = [{key: row.get(key, "") for key in per_pair_fieldnames} for row in per_pair_rows]
    for row in per_pair_rows:
        write_json(out_root / row["dataset"] / "metrics.json", row)

    _write_csv(out_root / "per_pair.csv", [{k: _format_float(v) for k, v in row.items()} for row in per_pair_rows], per_pair_fieldnames)
    write_by_category(out_root, per_pair_rows)
    write_report(out_root, per_pair_rows)

    ok_rows = [row for row in per_pair_rows if row.get("status") == "ok"]
    write_json(
        out_root / "summary.json",
        {
            "dataset": args.dataset,
            "manifest": args.manifest,
            "out": str(out_root),
            "pipeline": "opencv_stitcher+deep_rectangling",
            "total": len(per_pair_rows),
            "ok": len(ok_rows),
            "failed": len(per_pair_rows) - len(ok_rows),
            "mean_mdr_rmse": finite_mean([float(r["mdr_rmse"]) for r in ok_rows]) if ok_rows else None,
            "mean_niqe": finite_mean([float(r["niqe"]) for r in ok_rows]) if ok_rows else None,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
