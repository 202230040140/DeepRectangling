from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def natural_key(path: str | Path) -> list[object]:
    text = Path(path).name
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def list_images(folder: str | Path) -> list[Path]:
    return sorted(
        [
            p
            for p in Path(folder).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and "_depth_" not in p.stem.lower()
        ],
        key=natural_key,
    )


def iter_scene_dirs(dataset: str | Path) -> Iterable[Path]:
    dataset = Path(dataset)
    if list_images(dataset):
        yield dataset
        return
    yield from sorted([p for p in dataset.iterdir() if p.is_dir()], key=natural_key)


def read_bgr(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def save_bgr(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
