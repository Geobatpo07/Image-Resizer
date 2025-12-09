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
      uv run python pipeline.py --input data\input --output data\transformed --scale-x 2 --scale-y 2

  - Traiter un seul fichier et écraser la sortie s'il existe déjà
      uv run python pipeline.py -i data\input\photo.jpg -o data\transformed -x 1.5 -y 1.5 --overwrite

  - Conserver la structure de sous-dossiers sous `output`
      uv run python pipeline.py -i data\input -o data\transformed --keep-structure

Par défaut:
  - input:  data\input
  - output: data\transformed
  - models: models (un fichier JSON des paramètres utilisés y est écrit)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List
from src import (
    ensure_dir as s_ensure_dir,
    is_image_file as s_is_image_file,
    discover_images as s_discover_images,
    make_output_path as s_make_output_path,
    save_image as s_save_image,
    process_one_image as s_process_one_image,
)
from src import save_params_json as s_save_params_json, append_run_metrics as s_append_run_metrics

# Les implémentations utilitaires et de traitement d'image ont été déplacées sous src/.

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
    compression_factor: float = 1.0,
) -> None:
    """Exécute le pipeline sur un fichier ou dossier d'images."""
    if not input_path:
        print("--input est vide. Fournissez un fichier ou un dossier.")
        return

    inp = Path(input_path)
    out_dir = Path(output_dir)
    s_ensure_dir(out_dir)

    files: List[Path] = []
    if inp.is_file():
        if s_is_image_file(inp):
            files = [inp]
        else:
            print(f"Le fichier n'est pas une image supportée: {inp}")
            return
    elif inp.is_dir():
        files = s_discover_images(inp)
        if not files:
            print(f"Aucune image trouvée dans: {inp}")
            return
    else:
        print(f"Chemin introuvable: {inp}")
        return

    # Avertissement si un facteur de compression est utilisé avec un scale != 1
    if compression_factor is not None and compression_factor > 1.0 and (abs(scale_x - 1.0) > 1e-6 or abs(scale_y - 1.0) > 1e-6):
        print("[INFO] --compression-factor > 1.0 actif: la taille finale sera égale à celle de l'entrée. Les paramètres --scale-x/--scale-y sont ignorés dans ce mode.")

    # Sauvegarde des paramètres dans models_dir
    if models_dir:
        params = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input": str(inp.resolve()),
            "output": str(out_dir.resolve()),
            "scale_x": scale_x,
            "scale_y": scale_y,
            "compression_factor": compression_factor,
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
        try:
            s_save_params_json(models_dir, params)
        except Exception as e:
            print(f"[WARN] Impossible d'écrire resize_params.json: {e}")

    t0 = time.time()
    ok_count = 0
    fail = []
    files_metrics = []
    input_root = inp if inp.is_dir() else inp.parent

    for f in files:
        try:
            out_path = s_make_output_path(f, input_root, out_dir, keep_structure)
            input_size = f.stat().st_size if f.exists() else None
            if out_path.exists() and not overwrite:
                print(f"Déjà présent, on saute (OVERWRITE=False): {out_path}")
                ok_count += 1
                files_metrics.append({
                    "input": str(f),
                    "output": str(out_path),
                    "input_size": input_size,
                    "output_size": out_path.stat().st_size if out_path.exists() else None,
                    "status": "skipped",
                })
                continue
            img_out = s_process_one_image(
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
                compression_factor=compression_factor,
            )
            s_save_image(out_path, img_out)
            ok_count += 1
            files_metrics.append({
                "input": str(f),
                "output": str(out_path),
                "input_size": input_size,
                "output_size": out_path.stat().st_size if out_path.exists() else None,
                "status": "processed",
            })
        except Exception as e:
            fail.append((str(f), str(e)))
            print(f"[ERREUR] {f}: {e}")
            files_metrics.append({
                "input": str(f),
                "output": str(out_path) if 'out_path' in locals() else None,
                "input_size": f.stat().st_size if f.exists() else None,
                "output_size": None,
                "status": "failed",
                "error": str(e),
            })

    dt = time.time() - t0
    print(f"Terminé. {ok_count}/{len(files)} réussites en {dt:.2f}s. Sortie: {out_dir}")
    if fail:
        print("Échecs:")
        for name, msg in fail:
            print(" -", name, "->", msg)

    # Sauvegarde des métriques d'exécution avec historique dans models_dir/metrics.json
    if models_dir:
        try:
            run_metrics = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "input": str(inp.resolve()),
                "output": str(out_dir.resolve()),
                "params": {
                    "scale_x": scale_x,
                    "scale_y": scale_y,
                    "compression_factor": compression_factor,
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
                },
                "stats": {
                    "total_files": len(files),
                    "success_count": ok_count,
                    "fail_count": len(fail),
                    "duration_seconds": round(dt, 3),
                    "avg_seconds_per_file": round(dt / len(files), 4) if files else None,
                },
                "failures": [
                    {"file": name, "error": msg} for name, msg in fail
                ],
                "files": files_metrics,
            }
            s_append_run_metrics(models_dir, run_metrics)
        except Exception as e:
            print(f"[WARN] Impossible d'écrire les métriques: {e}")

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

    p.add_argument("--scale-x", "-x", type=float, default=1.0, help="Facteur d'échelle horizontal")
    p.add_argument("--scale-y", "-y", type=float, default=1.0, help="Facteur d'échelle vertical")

    p.add_argument("--compression-factor", type=float, default=1.0,
                   help="Facteur de compression perceptuelle à taille constante (>1.0 = downscale interne puis upscale intelligent; la sortie garde la même taille que l'entrée)")

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
    s_ensure_dir(Path("data") / "input")
    s_ensure_dir(Path("data") / "transformed")
    s_ensure_dir(Path(args.models))

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
        compression_factor=args.compression_factor,
    )


if __name__ == "__main__":
    main()
