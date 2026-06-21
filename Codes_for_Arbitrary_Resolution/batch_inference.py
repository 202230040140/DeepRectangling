"""Batch rectangling inference for arbitrary-resolution inputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from deep_rectangling_py.tf_compat import enable_legacy_tf1

enable_legacy_tf1()

import tensorflow as tf  # noqa: E402

from model import RectanglingNetwork  # noqa: E402
from utils import load  # noqa: E402

import constant  # noqa: E402

os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("DEEPRECT_GPU", constant.GPU)


def _load_pair(input_path: Path, mask_path: Path) -> np.ndarray:
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR).astype(np.float32)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR).astype(np.float32)
    if image is None or mask is None:
        raise FileNotFoundError(f"Missing input or mask: {input_path}, {mask_path}")
    image = (image / 127.5) - 1.0
    mask = (mask / 127.5) - 1.0
    return np.concatenate([image, mask], axis=2)


def _build_graph():
    batch_size = 1
    inputs = tf.compat.v1.placeholder(shape=[batch_size, None, None, 6], dtype=tf.float32)
    input_image = inputs[..., 0:3]
    input_mask = inputs[..., 3:6]
    with tf.compat.v1.variable_scope("generator", reuse=None):
        warp_image_final = RectanglingNetwork(input_image, input_mask)
    return inputs, warp_image_final


def run_single_job(job: dict[str, str], checkpoint: Path) -> tuple[bool, str]:
    tf.compat.v1.reset_default_graph()
    inputs_tensor, warp_tensor = _build_graph()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    try:
        with tf.compat.v1.Session(config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            loader = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables())
            load(loader, sess, str(checkpoint))
            clip = np.expand_dims(_load_pair(Path(job["input"]), Path(job["mask"])), axis=0)
            warp = sess.run(warp_tensor, feed_dict={inputs_tensor: clip})
            image = np.clip((warp[0] + 1.0) * 127.5, 0, 255).astype(np.uint8)
            output_path = Path(job["output"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), image)
            return True, f"saved {output_path}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _run_jobs(jobs: list[dict[str, str]], checkpoint: Path) -> tuple[int, int]:
    ok = 0
    failed = 0
    for job in jobs:
        success, message = run_single_job(job, checkpoint)
        print(message)
        if success:
            ok += 1
        else:
            failed += 1
    return ok, failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run DeepRectangling on prepared stitched/mask pairs.")
    parser.add_argument("--manifest", required=True, help="JSON list of {input, mask, output} jobs.")
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "Codes" / "checkpoints" / "pretrained_model" / "model.ckpt-100000"),
        help="TensorFlow checkpoint prefix.",
    )
    parser.add_argument("--gpu", default=constant.GPU, help="CUDA visible device index.")
    args = parser.parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    jobs = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    if not jobs:
        print("No jobs in manifest.")
        return 0
    ok, failed = _run_jobs(jobs, Path(args.checkpoint))
    print(f"rectangling done: ok={ok}, failed={failed}")
    return 0 if ok > 0 or failed == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
