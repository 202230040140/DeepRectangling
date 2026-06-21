"""Run OpenCV Stitcher + DeepRectangling on HD3D two-view pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from tqdm import tqdm

from .hd3d_eval import (
    PER_PAIR_FIELDS,
    SUMMARY_FIELDS,
    evaluate_raw,
    load_existing_per_pair,
    load_lpips_metric,
    load_niqe_metric,
    summarize,
    write_csv,
    write_report,
)
from .io_utils import read_bgr, save_bgr, write_json
from .stitch import stitch_pair

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TF_SCRIPT = REPO_ROOT / "Codes_for_Arbitrary_Resolution" / "batch_inference.py"
DEFAULT_CHECKPOINT = REPO_ROOT / "Codes" / "checkpoints" / "pretrained_model" / "model.ckpt-100000"
METHOD_NAME = "DeepRectangling"
MAX_RECTANGLE_SIDE = 2048


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def backup_once(path: Path, suffix: str = ".before_DeepRectangling") -> None:
    if not path.exists():
        return
    backup = path.with_name(f"{path.stem}{suffix}{path.suffix}")
    if not backup.exists():
        shutil.copy2(path, backup)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}


def _mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask


def _maybe_downscale(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side <= MAX_RECTANGLE_SIDE:
        return image
    scale = MAX_RECTANGLE_SIDE / float(long_side)
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _load_rectangling_runner() -> Callable[[dict[str, str], Path], tuple[bool, str]]:
    ar_dir = REPO_ROOT / "Codes_for_Arbitrary_Resolution"
    if str(ar_dir) not in sys.path:
        sys.path.insert(0, str(ar_dir))
    import importlib.util

    spec = importlib.util.spec_from_file_location("dr_batch_inference", DEFAULT_TF_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {DEFAULT_TF_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_single_job


def _prepare_tf_input(stitched: np.ndarray, mask: np.ndarray, work_dir: Path) -> tuple[Path, Path, Path]:
    tf_input_dir = work_dir / "tf_input"
    tf_mask_dir = work_dir / "tf_mask"
    tf_input_dir.mkdir(parents=True, exist_ok=True)
    tf_mask_dir.mkdir(parents=True, exist_ok=True)
    input_copy = tf_input_dir / "00001.jpg"
    mask_copy = tf_mask_dir / "00001.jpg"
    save_bgr(input_copy, _maybe_downscale(stitched))
    save_bgr(mask_copy, _maybe_downscale(mask))
    rectangled_path = work_dir / "rectangled.png"
    return input_copy, mask_copy, rectangled_path


def base_row(row: dict[str, str], out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "scene": row["scene"],
        "pair_id": row["pair_id"],
        "pair_name": row["pair_name"],
        "method": METHOD_NAME,
        "status": "failed",
        "failure_reason": "",
        "mdr": math.nan,
        "niqe": math.nan,
        "psnr": math.nan,
        "ssim": math.nan,
        "lpips": math.nan,
        "rmse": math.nan,
        "runtime_seconds": math.nan,
        "valid_ratio": math.nan,
        "alignment_matcher": "",
        "alignment_matches": "",
        "alignment_inliers": "",
        "valid_mask_strategy": "",
        "lpips_max_side": args.lpips_max_side,
        "raw_path": str(out_dir / "raw.png"),
        "aligned_path": "",
        "valid_mask_path": "",
        "gt_path": row["gt_path"],
        "cpp_mdr": math.nan,
        "cpp_warping_residual_avg": math.nan,
        "cpp_warping_residual_sd": math.nan,
        "gt_width": "",
        "gt_height": "",
    }


def process_pair(
    row: dict[str, str],
    result_root: Path,
    work_root: Path,
    checkpoint: Path,
    run_single_job: Callable[[dict[str, str], Path], tuple[bool, str]],
    niqe_metric,
    lpips_metric,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir = result_root / row["scene"] / f"pair_{row['pair_id']}" / METHOD_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.png"
    status_path = out_dir / "method_status.json"
    metrics_path = out_dir / "metrics.json"

    cached = load_json(status_path)
    if not args.force and cached.get("success") and raw_path.exists() and metrics_path.exists():
        metrics = load_json(metrics_path)
        return {key: metrics.get(key, "") for key in PER_PAIR_FIELDS}

    work_dir = work_root / METHOD_NAME / row["pair_name"]
    work_dir.mkdir(parents=True, exist_ok=True)
    result = base_row(row, out_dir, args)
    status: dict[str, Any] = {
        "method": METHOD_NAME,
        "pair_name": row["pair_name"],
        "success": False,
        "runtime_seconds": None,
        "failure_reason": "",
    }

    try:
        left_path = Path(row["left_source"])
        right_path = Path(row["right_source"])
        img1 = read_bgr(left_path)
        img2 = read_bgr(right_path)
        stitched, mask, stitch_method = stitch_pair(img1, img2, allow_fallback=args.allow_stitch_fallback)
        save_bgr(work_dir / "stitched.png", stitched)
        save_bgr(work_dir / "mask.png", _mask_to_bgr(mask))

        input_copy, mask_copy, rectangled_path = _prepare_tf_input(stitched, mask, work_dir)
        success, message = run_single_job(
            {
                "input": str(input_copy.resolve()),
                "mask": str(mask_copy.resolve()),
                "output": str(rectangled_path.resolve()),
            },
            checkpoint.resolve(),
        )
        if not success:
            raise RuntimeError(message)

        rectangled = read_bgr(rectangled_path)
        cv2.imwrite(str(raw_path), rectangled)
        eval_info = evaluate_raw(raw_path, Path(row["gt_path"]), out_dir, niqe_metric, lpips_metric, args)
        runtime = time.perf_counter() - started
        result.update(eval_info)
        result.update(
            {
                "status": "success",
                "failure_reason": "",
                "runtime_seconds": runtime,
                "raw_path": str(raw_path),
            }
        )
        status.update(
            {
                "success": True,
                "runtime_seconds": runtime,
                "stitch_method": stitch_method,
                "rectangling_message": message,
                "raw_path": str(raw_path),
                "failure_reason": "",
            }
        )
    except Exception as exc:
        runtime = time.perf_counter() - started
        result["failure_reason"] = str(exc)
        result["runtime_seconds"] = runtime
        status.update(
            {
                "success": False,
                "failure_reason": str(exc),
                "runtime_seconds": runtime,
                "failure_traceback": traceback.format_exc(),
            }
        )

    write_json(status_path, status)
    write_json(metrics_path, result)
    return {key: result.get(key, "") for key in PER_PAIR_FIELDS}


def update_top_level_reports(result_root: Path, method: str, new_rows: list[dict[str, Any]]) -> None:
    per_pair_path = result_root / "per_pair_metrics.csv"
    summary_path = result_root / "summary_all.csv"
    report_path = result_root / "report.md"
    for path in (per_pair_path, summary_path, report_path):
        backup_once(path)
    existing_rows = load_existing_per_pair(per_pair_path, method)
    rows = existing_rows + new_rows
    write_csv(per_pair_path, rows, PER_PAIR_FIELDS)
    summary_rows = summarize(rows)
    write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    write_report(report_path, summary_rows, rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OpenCV+DeepRectangling on HD3D two-view pairs.")
    parser.add_argument("--manifest", default=r"D:\HD3D_Result\_work\manifest.csv")
    parser.add_argument("--result-root", default=r"D:\HD3D_Result")
    parser.add_argument("--work-root", default=r"D:\HD3D_Result\_work")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", default=os.environ.get("DEEPRECT_GPU", "0"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--scene", action="append")
    parser.add_argument("--pair", action="append")
    parser.add_argument("--allow-stitch-fallback", action="store_true", default=True)
    parser.add_argument("--no-stitch-fallback", action="store_false", dest="allow_stitch_fallback")
    parser.add_argument("--feature-max-side", type=int, default=1800)
    parser.add_argument("--min-alignment-inliers", type=int, default=12)
    parser.add_argument("--min-valid-ratio", type=float, default=0.05)
    parser.add_argument("--min-niqe-side", type=int, default=96)
    parser.add_argument("--valid-black-threshold", type=int, default=5)
    parser.add_argument("--lpips-max-side", type=int, default=1024)
    return parser


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")
    args = build_parser().parse_args(argv)
    manifest = read_manifest(Path(args.manifest))
    if args.scene:
        wanted = set(args.scene)
        manifest = [row for row in manifest if row["scene"] in wanted]
    if args.pair:
        wanted_pairs = set(args.pair)
        manifest = [row for row in manifest if row["pair_id"] in wanted_pairs]

    os.environ["DEEPRECT_GPU"] = args.gpu
    run_single_job = _load_rectangling_runner()
    niqe_metric, metric_device = load_niqe_metric(args.device)
    lpips_metric, _ = load_lpips_metric(metric_device)
    checkpoint = Path(args.checkpoint)

    rows = []
    for row in tqdm(manifest, desc="HD3D DeepRectangling"):
        result = process_pair(
            row,
            Path(args.result_root),
            Path(args.work_root),
            checkpoint,
            run_single_job,
            niqe_metric,
            lpips_metric,
            args,
        )
        rows.append(result)
        print(f"{row['pair_name']} {METHOD_NAME}: {result['status']}")

    if not args.skip_summary:
        update_top_level_reports(Path(args.result_root), METHOD_NAME, rows)
        print(f"Updated {Path(args.result_root) / 'per_pair_metrics.csv'}")
        print(f"Updated {Path(args.result_root) / 'summary_all.csv'}")
        print(f"Updated {Path(args.result_root) / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
