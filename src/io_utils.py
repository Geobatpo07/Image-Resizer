from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import cv2


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_image_file(p: Path) -> bool:
    return p.is_file() and (p.suffix.lower() in SUPPORTED_EXTS)


def discover_images(root: Path) -> List[Path]:
    images: List[Path] = []
    for ext in SUPPORTED_EXTS:
        images.extend(root.rglob(f"*{ext}"))
        images.extend(root.rglob(f"*{ext.upper()}"))
    # Deduplicate preserving order
    seen = set()
    uniq: List[Path] = []
    for p in images:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def make_output_path(inp: Path, input_root: Path, output_dir: Path, keep_structure: bool) -> Path:
    stem = inp.stem + "_resized"
    out_name = stem + inp.suffix.lower()
    if keep_structure:
        try:
            rel = inp.parent.relative_to(input_root)
        except Exception:
            rel = Path(".")
        return output_dir / rel / out_name
    else:
        return output_dir / out_name


def load_image_cv2(path: str, as_float32: bool = True) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Impossible de charger le fichier: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if as_float32:
        rgb = rgb.astype(np.float32) / 255.0
    return rgb


def save_image(path: Path, image: np.ndarray) -> None:
    img_u8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    ensure_dir(path.parent)
    if img_u8.ndim == 3:
        bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    else:
        bgr = img_u8
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise IOError(f"Echec d'écriture: {path}")
