"""
Pipeline de redimensionnement inspiré du notebook `resize_modern_bspline.ipynb`.

Fonctionnalités principales:
- Traite un fichier image unique OU tout un dossier d'images (avec option pour conserver l'arborescence).
- Écrit les résultats dans un dossier de transformation.
- Utilise les dossiers `data` et `models` par défaut.
- Implémente un pipeline moderne combinant diffusion anisotrope, pyramide laplacienne
  et interpolation par B-splines adaptatives.

Dépendances attendues (voir requirements): numpy, scipy, opencv-python-headless, pillow

Exemples d'utilisation (PowerShell):
  - Traiter tout le dossier data/input → data/transformed
      python pipeline.py --input data\input --output data\transformed --scale-x 2 --scale-y 2

  - Traiter un seul fichier et écraser la sortie s'il existe déjà
      python pipeline.py -i data\input\photo.jpg -o data\transformed -x 1.5 -y 1.5 --overwrite

  - Conserver la structure de sous-dossiers sous `output`
      python pipeline.py -i data\input -o data\transformed --keep-structure

Par défaut:
  - input:  data\input
  - output: data\transformed
  - models: models (un fichier JSON des paramètres utilisés y est écrit)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Tuple, List

import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline

# -----------------------------
# Utilitaires fichiers/chemins
# -----------------------------
SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def is_image_file(p: Path) -> bool:
    return p.is_file() and (p.suffix.lower() in SUPPORTED_EXTS)

def discover_images(root: Path) -> List[Path]:
    images: List[Path] = []
    for ext in SUPPORTED_EXTS:
        images.extend(root.rglob(f"*{ext}"))
        images.extend(root.rglob(f"*{ext.upper()}"))
    # Supprime doublons tout en conservant l'ordre
    seen = set()
    uniq: List[Path] = []
    for p in images:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq

def make_output_path(inp: Path, input_root: Path, output_dir: Path, keep_structure: bool) -> Path:
    # Construit le chemin de sortie: même nom + suffixe '_resized' avant l'extension
    stem = inp.stem + '_resized'
    out_name = stem + inp.suffix.lower()
    if keep_structure:
        try:
            rel = inp.parent.relative_to(input_root)
        except Exception:
            rel = Path('.')
        return output_dir / rel / out_name
    else:
        return output_dir / out_name

# -----------------------------
# I/O image
# -----------------------------

def load_image_cv2(path: str, as_float32: bool = True) -> np.ndarray:
    """Charge une image via OpenCV et retourne un tableau RGB dans [0, 1]."""
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

# ---------------------------------------------
# Diffusion anisotrope (Perona-Malik)
# ---------------------------------------------

def anisotropic_diffusion_gray(
    image: np.ndarray,
    niter: int = 10,
    kappa: float = 30.0,
    gamma: float = 0.15,
    option: int = 1,
) -> np.ndarray:
    """
    Diffusion anisotrope de Perona Malik sur une image niveau de gris.
    - niter : nombre d'itérations
    - kappa : seuil de conduction (sensibilité aux gradients)
    - gamma : pas de temps (<= ~0.25 pour stabilité en 2D)
    - option : 1 (exp) ou 2 (1 / (1 + (g/kappa)^2))
    """
    img = image.astype(np.float32)
    for _ in range(niter):
        north = np.zeros_like(img)
        south = np.zeros_like(img)
        east = np.zeros_like(img)
        west = np.zeros_like(img)

        north[1:, :] = img[1:, :] - img[:-1, :]
        south[:-1, :] = img[:-1, :] - img[1:, :]
        east[:, :-1] = img[:, :-1] - img[:, 1:]
        west[:, 1:] = img[:, 1:] - img[:, :-1]

        if option == 1:
            cN = np.exp(-(north / kappa) ** 2.0)
            cS = np.exp(-(south / kappa) ** 2.0)
            cE = np.exp(-(east / kappa) ** 2.0)
            cW = np.exp(-(west / kappa) ** 2.0)
        else:
            cN = 1.0 / (1.0 + (north / kappa) ** 2.0)
            cS = 1.0 / (1.0 + (south / kappa) ** 2.0)
            cE = 1.0 / (1.0 + (east / kappa) ** 2.0)
            cW = 1.0 / (1.0 + (west / kappa) ** 2.0)

        img = img + gamma * (cN * north + cS * south + cE * east + cW * west)

    return np.clip(img, 0.0, 1.0)

def anisotropic_diffusion_color(
    image_rgb: np.ndarray,
    niter: int = 10,
    kappa: float = 30.0,
    gamma: float = 0.15,
    option: int = 1,
) -> np.ndarray:
    """Applique la diffusion anisotrope sur la luminance (Lab) pour préserver la couleur."""
    img = np.clip(image_rgb.astype(np.float32), 0.0, 1.0)
    img_8u = (img * 255.0).astype(np.uint8)
    lab = cv2.cvtColor(img_8u, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a, b = cv2.split(lab)
    L_norm = L / 255.0

    L_f = anisotropic_diffusion_gray(L_norm, niter=niter, kappa=kappa, gamma=gamma, option=option)
    L_f = np.clip(L_f, 0.0, 1.0) * 255.0

    lab_f = cv2.merge([L_f, a, b])
    rgb_f = cv2.cvtColor(lab_f.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return rgb_f

# ---------------------------------------------
# Pyramides gaussienne et laplacienne
# ---------------------------------------------

def build_gaussian_pyramid(image: np.ndarray, levels: int = 4) -> list:
    assert levels >= 1
    pyramid = [image.astype(np.float32)]
    for _ in range(1, levels):
        img = cv2.pyrDown(pyramid[-1])
        pyramid.append(img)
    return pyramid

def build_laplacian_pyramid(gaussian_pyr: list) -> list:
    lap_pyr = []
    for i in range(len(gaussian_pyr) - 1):
        current = gaussian_pyr[i]
        next_img = gaussian_pyr[i + 1]
        size = (current.shape[1], current.shape[0])
        up = cv2.pyrUp(next_img, dstsize=size)
        lap = current - up
        lap_pyr.append(lap)
    lap_pyr.append(gaussian_pyr[-1])
    return lap_pyr

def reconstruct_from_laplacian(laplacian_pyr: list) -> np.ndarray:
    current = laplacian_pyr[-1]
    for level in range(len(laplacian_pyr) - 2, -1, -1):
        size = (laplacian_pyr[level].shape[1], laplacian_pyr[level].shape[0])
        up = cv2.pyrUp(current, dstsize=size)
        current = up + laplacian_pyr[level]
    return np.clip(current, 0.0, 1.0)

# ---------------------------------------------
# Interpolation B-splines (standard + adaptative)
# ---------------------------------------------

def bspline_resize_channel(
    channel: np.ndarray,
    scale_x: float,
    scale_y: float,
    order: int = 3,
) -> np.ndarray:
    h, w = channel.shape
    y = np.arange(h, dtype=np.float32)
    x = np.arange(w, dtype=np.float32)

    spline = RectBivariateSpline(y, x, channel.astype(np.float32), kx=order, ky=order, s=0.0)

    new_h = max(int(round(h * scale_y)), 1)
    new_w = max(int(round(w * scale_x)), 1)
    y_new = np.linspace(0.0, float(h - 1), new_h)
    x_new = np.linspace(0.0, float(w - 1), new_w)

    out = spline(y_new, x_new)
    return out.astype(np.float32)

def bspline_resize(
    image: np.ndarray,
    scale_x: float,
    scale_y: float,
    order: int = 3,
) -> np.ndarray:
    img = image.astype(np.float32)
    if img.ndim == 2:
        return np.clip(bspline_resize_channel(img, scale_x, scale_y, order=order), 0.0, 1.0)
    channels = []
    for c in range(img.shape[2]):
        ch = bspline_resize_channel(img[:, :, c], scale_x, scale_y, order=order)
        channels.append(ch)
    out = np.stack(channels, axis=-1)
    return np.clip(out, 0.0, 1.0)

def compute_local_energy(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    energy = np.sqrt(gx * gx + gy * gy)
    energy = energy / (energy.max() + 1e-8)
    return energy

def bspline_resize_adaptive(
    image: np.ndarray,
    scale_x: float,
    scale_y: float,
    order_low: int = 1,
    order_high: int = 3,
    beta: float = 2.0,
) -> np.ndarray:
    img = np.clip(image.astype(np.float32), 0.0, 1.0)
    if img.ndim == 2:
        img_rgb = np.stack([img, img, img], axis=-1)
    else:
        img_rgb = img

    gray = cv2.cvtColor((img_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    energy = compute_local_energy(gray)

    base_low = bspline_resize(img_rgb, scale_x, scale_y, order=order_low)
    base_high = bspline_resize(img_rgb, scale_x, scale_y, order=order_high)

    out_h, out_w = base_low.shape[:2]
    energy_resized = cv2.resize(energy, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    e = energy_resized
    e = e / (e.max() + 1e-8)

    weights = e ** beta
    if base_low.ndim == 3:
        weights = weights[..., np.newaxis]

    result = weights * base_high + (1.0 - weights) * base_low
    return np.clip(result, 0.0, 1.0)

# ---------------------------------------------
# Pipeline complet
# ---------------------------------------------

def resize_modern(
    image: np.ndarray,
    scale_x: float,
    scale_y: float,
    spline_order: int = 3,
    diffusion_iter: int = 10,
    pyr_levels: int = 4,
    kappa: float = 30.0,
    gamma: float = 0.15,
    option: int = 1,
    detail_boost: float = 1.0,
    sharpen: bool = True,
) -> np.ndarray:
    """
    Pipeline de redimensionnement combinant:
      1) diffusion anisotrope (Perona-Malik),
      2) pyramide laplacienne multi-résolution,
      3) interpolation finale par B-splines adaptatives.
    """
    img = np.clip(image.astype(np.float32), 0.0, 1.0)

    # 1) Denoising edge-preserving
    denoised = anisotropic_diffusion_color(
        img, niter=diffusion_iter, kappa=kappa, gamma=gamma, option=option
    )

    # 2) Pyramide laplacienne
    gaussian_pyr = build_gaussian_pyramid(denoised, levels=pyr_levels)
    laplacian_pyr = build_laplacian_pyramid(gaussian_pyr)

    # 3) Reconstruction progressive jusqu'à la taille d'origine
    current = laplacian_pyr[-1]
    for level in range(len(laplacian_pyr) - 2, -1, -1):
        target_h, target_w = laplacian_pyr[level].shape[:2]
        scale_x_level = float(target_w) / float(current.shape[1])
        scale_y_level = float(target_h) / float(current.shape[0])

        current = bspline_resize(current, scale_x_level, scale_y_level, order=spline_order)
        current = current + detail_boost * laplacian_pyr[level]
        current = np.clip(current, 0.0, 1.0)

    reconstructed = current

    # 4) Redimensionnement final (cible) via B-splines adaptatives
    resized = bspline_resize_adaptive(
        reconstructed,
        scale_x=scale_x,
        scale_y=scale_y,
        order_low=1,
        order_high=spline_order,
        beta=2.0,
    )

    # 5) Unsharp masking léger
    if sharpen:
        resized_8u = (np.clip(resized, 0.0, 1.0) * 255.0).astype(np.uint8)
        blurred = cv2.GaussianBlur(resized_8u, (0, 0), sigmaX=1.0, sigmaY=1.0)
        sharpened = cv2.addWeighted(resized_8u, 1.3, blurred, -0.3, 0)
        resized = sharpened.astype(np.float32) / 255.0

    return np.clip(resized, 0.0, 1.0)

# ---------------------------------------------
# Orchestration batch
# ---------------------------------------------

def process_one_image(input_image_path: Path, scale_x: float, scale_y: float,
                      spline_order: int = 3, diffusion_iter: int = 10, pyr_levels: int = 4,
                      kappa: float = 30.0, gamma: float = 0.15, option: int = 1,
                      detail_boost: float = 1.0, sharpen: bool = True) -> np.ndarray:
    img = load_image_cv2(str(input_image_path), as_float32=True)
    out = resize_modern(
        img,
        scale_x=scale_x,
        scale_y=scale_y,
        spline_order=spline_order,
        diffusion_iter=diffusion_iter,
        pyr_levels=pyr_levels,
        kappa=kappa,
        gamma=gamma,
        option=option,
        detail_boost=detail_boost,
        sharpen=sharpen,
    )
    return out

def run_pipeline_on_path(
    input_path: str,
    output_dir: str,
    scale_x: float,
    scale_y: float,
    keep_structure: bool = True,
    overwrite: bool = False,
    models_dir: str | None = None,
    spline_order: int = 3,
    diffusion_iter: int = 10,
    pyr_levels: int = 4,
    kappa: float = 30.0,
    gamma: float = 0.15,
    option: int = 1,
    detail_boost: float = 1.0,
    sharpen: bool = True,
) -> None:
    """Exécute le pipeline sur un fichier ou dossier d'images."""
    if not input_path:
        print("--input est vide. Fournissez un fichier ou un dossier.")
        return

    inp = Path(input_path)
    out_dir = Path(output_dir)
    ensure_dir(out_dir)

    files: List[Path] = []
    if inp.is_file():
        if is_image_file(inp):
            files = [inp]
        else:
            print(f"Le fichier n'est pas une image supportée: {inp}")
            return
    elif inp.is_dir():
        files = discover_images(inp)
        if not files:
            print(f"Aucune image trouvée dans: {inp}")
            return
    else:
        print(f"Chemin introuvable: {inp}")
        return

    # Sauvegarde des paramètres dans models_dir
    if models_dir:
        models_path = Path(models_dir)
        ensure_dir(models_path)
        params = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input": str(inp.resolve()),
            "output": str(out_dir.resolve()),
            "scale_x": scale_x,
            "scale_y": scale_y,
            "keep_structure": keep_structure,
            "overwrite": overwrite,
            "algo": {
                "spline_order": spline_order,
                "diffusion_iter": diffusion_iter,
                "pyr_levels": pyr_levels,
                "kappa": kappa,
                "gamma": gamma,
                "option": option,
                "detail_boost": detail_boost,
                "sharpen": sharpen,
            },
        }
        with (models_path / "resize_params.json").open("w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)

    t0 = time.time()
    ok_count = 0
    fail = []
    input_root = inp if inp.is_dir() else inp.parent

    for f in files:
        try:
            out_path = make_output_path(f, input_root, out_dir, keep_structure)
            if out_path.exists() and not overwrite:
                print(f"Déjà présent, on saute (OVERWRITE=False): {out_path}")
                ok_count += 1
                continue
            img_out = process_one_image(
                f,
                scale_x=scale_x,
                scale_y=scale_y,
                spline_order=spline_order,
                diffusion_iter=diffusion_iter,
                pyr_levels=pyr_levels,
                kappa=kappa,
                gamma=gamma,
                option=option,
                detail_boost=detail_boost,
                sharpen=sharpen,
            )
            save_image(out_path, img_out)
            ok_count += 1
        except Exception as e:
            fail.append((str(f), str(e)))
            print(f"[ERREUR] {f}: {e}")

    dt = time.time() - t0
    print(f"Terminé. {ok_count}/{len(files)} réussites en {dt:.2f}s. Sortie: {out_dir}")
    if fail:
        print("Échecs:")
        for name, msg in fail:
            print(" -", name, "->", msg)

# ---------------------------------------------
# CLI
# ---------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pipeline de redimensionnement haute qualité (B-splines, diffusion anisotrope, pyramides)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", default=str(Path("data") / "input"), help="Chemin d'entrée (fichier image ou dossier)")
    p.add_argument("--output", "-o", default=str(Path("data") / "transformed"), help="Dossier de sortie")
    p.add_argument("--models", default=str(Path("models")), help="Dossier models pour sauvegarder les paramètres du run")

    p.add_argument("--scale-x", "-x", type=float, default=2.0, help="Facteur d'échelle horizontal")
    p.add_argument("--scale-y", "-y", type=float, default=2.0, help="Facteur d'échelle vertical")

    p.add_argument("--keep-structure", action="store_true", help="Préserver la structure des sous-dossiers")
    p.add_argument("--overwrite", action="store_true", help="Écraser les fichiers de sortie existants")

    # Paramètres algorithmiques
    p.add_argument("--spline-order", type=int, default=3, help="Ordre des B-splines (3 = cubique)")
    p.add_argument("--diffusion-iter", type=int, default=10, help="Itérations de diffusion anisotrope")
    p.add_argument("--pyr-levels", type=int, default=4, help="Nombre de niveaux de pyramide")
    p.add_argument("--kappa", type=float, default=30.0, help="Seuil de conduction (Perona-Malik)")
    p.add_argument("--gamma", type=float, default=0.15, help="Pas de temps (Perona-Malik)")
    p.add_argument("--option", type=int, default=1, choices=[1, 2], help="Forme de conduction (1=exp, 2=ratio)")
    p.add_argument("--detail-boost", type=float, default=1.0, help="Gain appliqué aux bandes laplaciennes")
    p.add_argument("--no-sharpen", action="store_true", help="Désactiver le rehaussement final")

    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Crée les dossiers data/ et models/ si absents
    ensure_dir(Path("data") / "input")
    ensure_dir(Path("data") / "transformed")
    ensure_dir(Path(args.models))

    run_pipeline_on_path(
        input_path=args.input,
        output_dir=args.output,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
        keep_structure=args.keep_structure,
        overwrite=args.overwrite,
        models_dir=args.models,
        spline_order=args.spline_order,
        diffusion_iter=args.diffusion_iter,
        pyr_levels=args.pyr_levels,
        kappa=args.kappa,
        gamma=args.gamma,
        option=args.option,
        detail_boost=args.detail_boost,
        sharpen=(not args.no_sharpen),
    )


if __name__ == "__main__":
    main()
