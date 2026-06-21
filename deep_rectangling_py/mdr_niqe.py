from __future__ import annotations

import math
from pathlib import Path
from statistics import mean
from typing import Optional

import cv2
import numpy as np

CATEGORIES = (
    "OBJ-GSP",
    "AANAP",
    "APAP",
    "CAVE",
    "DFW",
    "DHW",
    "GES",
    "LPC",
    "REW",
    "SEAGULL",
    "SVA",
    "SPHP",
)

DEFAULT_MANIFEST = r"D:\StitchBench_Result\_shared\manifest.csv"


def category_for(dataset: str) -> Optional[str]:
    for category in CATEGORIES:
        if dataset.startswith(category):
            return category
    return None


def canvas_valid_mask(image: np.ndarray, black_threshold: int = 5) -> np.ndarray:
    near_black = (np.max(image, axis=2) <= black_threshold).astype(np.uint8)
    flood = near_black.copy()
    height, width = flood.shape
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)

    def fill_if_background(x: int, y: int) -> None:
        if flood[y, x] == 1:
            cv2.floodFill(flood, flood_mask, (x, y), 2)

    for x in range(width):
        fill_if_background(x, 0)
        fill_if_background(x, height - 1)
    for y in range(height):
        fill_if_background(0, y)
        fill_if_background(width - 1, y)
    return (~(flood == 2)).astype(np.uint8) * 255


def _match_points(img1: np.ndarray, img2: np.ndarray, max_features: int = 10000) -> tuple[np.ndarray, np.ndarray]:
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=max_features)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        raise RuntimeError("Insufficient SIFT features for MDR.")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(des1, des2, k=2)
    good = [first for first, second in pairs if first.distance < 0.75 * second.distance]
    if len(good) < 4:
        raise RuntimeError(f"Insufficient SIFT matches for MDR ({len(good)}).")
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2


def _estimate_image_to_pano(image: np.ndarray, pano: np.ndarray, max_side: int = 1800, min_inliers: int = 12):
    mask = canvas_valid_mask(pano)
    h, w = pano.shape[:2]
    scale = 1.0
    long_side = max(h, w)
    if long_side > max_side:
        scale = max_side / float(long_side)
        pano_small = cv2.resize(pano, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        mask_small = cv2.resize(mask, (pano_small.shape[1], pano_small.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        pano_small = pano
        mask_small = mask

    ih, iw = image.shape[:2]
    image_scale = 1.0
    image_long = max(ih, iw)
    if image_long > max_side:
        image_scale = max_side / float(image_long)
        image_small = cv2.resize(image, None, fx=image_scale, fy=image_scale, interpolation=cv2.INTER_AREA)
    else:
        image_small = image

    kp_img, kp_pano = _match_points(image_small, pano_small)
    kp_img = kp_img / image_scale
    kp_pano = kp_pano / scale
    homography, inlier_mask = cv2.findHomography(
        kp_img.reshape(-1, 1, 2), kp_pano.reshape(-1, 1, 2), cv2.RANSAC, 5.0, maxIters=8000
    )
    if homography is None or inlier_mask is None:
        raise RuntimeError("Failed to estimate image-to-panorama homography for MDR.")
    inliers = inlier_mask.ravel().astype(bool)
    if int(inliers.sum()) < min_inliers:
        raise RuntimeError(f"Insufficient inliers for MDR ({int(inliers.sum())}).")
    return homography, int(inliers.sum()), len(kp_img)


def compute_mdr_rmse(img1: np.ndarray, img2: np.ndarray, panorama: np.ndarray) -> dict[str, float | int | str]:
    """Two-view overlap mapping RMSE in panorama coordinates (OBJ-GSP-compatible MDR role)."""
    pts1, pts2 = _match_points(img1, img2)
    homography1, inliers1, _ = _estimate_image_to_pano(img1, panorama)
    homography2, inliers2, _ = _estimate_image_to_pano(img2, panorama)
    warped1 = cv2.perspectiveTransform(pts1.reshape(-1, 1, 2), homography1).reshape(-1, 2)
    warped2 = cv2.perspectiveTransform(pts2.reshape(-1, 1, 2), homography2).reshape(-1, 2)
    residuals = np.linalg.norm(warped1 - warped2, axis=1)
    rmse = float(math.sqrt(float(np.mean(residuals * residuals))))
    return {
        "mdr_rmse": rmse,
        "warping_residual_avg": float(np.mean(residuals)),
        "warping_residual_sd": float(np.std(residuals)),
        "mdr_matches": len(residuals),
        "mdr_inliers_img1": inliers1,
        "mdr_inliers_img2": inliers2,
        "metric_note": "Two-view overlap mapping RMSE in panorama coordinates; comparable role to OBJ-GSP mesh RMSE.",
    }


def load_niqe_metric(device: str = "cuda"):
    import pyiqa
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    return pyiqa.create_metric("niqe", device=device), device


def compute_niqe(metric, image_path: Path) -> float:
    if not image_path.exists():
        return math.nan
    try:
        score = metric(str(image_path))
    except Exception:
        import torch

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return math.nan
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        score = metric(tensor)
    return float(score.detach().cpu().item()) if hasattr(score, "detach") else float(score)


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return mean(finite) if finite else math.nan
