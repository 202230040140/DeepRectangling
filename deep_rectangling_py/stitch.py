from __future__ import annotations

import os

import cv2
import numpy as np

cv2.ocl.setUseOpenCL(False)
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")


def _mask_from_panorama(pano: np.ndarray, threshold: int = 8) -> np.ndarray:
    gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY) if pano.ndim == 3 else pano
    mask = (gray > threshold).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _homography_stitch(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=8000)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        raise RuntimeError("Not enough SIFT features for homography stitching.")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        raise RuntimeError(f"Not enough good matches for homography stitching ({len(good)}).")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, inliers = cv2.findHomography(pts2, pts1, cv2.RANSAC, 4.0)
    if homography is None or inliers is None or int(inliers.sum()) < 4:
        raise RuntimeError("Homography estimation failed.")

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners2, homography)
    all_corners = np.concatenate(
        (np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2), warped_corners), axis=0
    )
    xmin, ymin = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    xmax, ymax = np.ceil(all_corners.max(axis=0).ravel()).astype(int)
    translation = np.array([[1.0, 0.0, -xmin], [0.0, 1.0, -ymin], [0.0, 0.0, 1.0]], dtype=np.float64)
    canvas_w = int(xmax - xmin)
    canvas_h = int(ymax - ymin)
    if canvas_w <= 0 or canvas_h <= 0:
        raise RuntimeError("Invalid stitched canvas size.")

    warped2 = cv2.warpPerspective(img2, translation @ homography, (canvas_w, canvas_h))
    base = np.zeros_like(warped2)
    base[-ymin : -ymin + h1, -xmin : -xmin + w1] = img1
    mask1 = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    mask1[-ymin : -ymin + h1, -xmin : -xmin + w1] = 255
    mask2 = (cv2.cvtColor(warped2, cv2.COLOR_BGR2GRAY) > 8).astype(np.uint8) * 255
    overlap = cv2.bitwise_and(mask1, mask2)
    only1 = cv2.bitwise_and(mask1, cv2.bitwise_not(mask2))
    only2 = cv2.bitwise_and(mask2, cv2.bitwise_not(mask1))
    result = np.zeros_like(warped2)
    result[only1.astype(bool)] = base[only1.astype(bool)]
    result[only2.astype(bool)] = warped2[only2.astype(bool)]
    if overlap.any():
        alpha = 0.5
        overlap_bool = overlap.astype(bool)
        blended = alpha * base[overlap_bool].astype(np.float32) + (1.0 - alpha) * warped2[overlap_bool].astype(np.float32)
        result[overlap_bool] = np.clip(blended, 0, 255).astype(np.uint8)
    mask = _mask_from_panorama(result)
    return result, mask


def stitch_pair_opencv(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, pano = stitcher.stitch([img1, img2])
    if status != cv2.Stitcher_OK or pano is None or pano.size == 0:
        raise RuntimeError(f"OpenCV Stitcher failed with status={status}")
    mask = _mask_from_panorama(pano)
    if int(mask.sum()) == 0:
        raise RuntimeError("OpenCV Stitcher produced an empty panorama mask.")
    return pano, mask, "opencv_stitcher"


def stitch_pair(img1: np.ndarray, img2: np.ndarray, allow_fallback: bool = False) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        return stitch_pair_opencv(img1, img2)
    except RuntimeError:
        if not allow_fallback:
            raise
        pano, mask = _homography_stitch(img1, img2)
        return pano, mask, "homography_fallback"
