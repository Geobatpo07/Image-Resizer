# Image_Resizer — Pipeline de redimensionnement haute qualité (B‑splines, diffusion anisotrope, pyramides)

Ce projet fournit un pipeline moderne de redimensionnement d’images, inspiré du notebook `resize_modern_bspline.ipynb`. Il combine diffusion anisotrope (Perona–Malik), pyramides laplaciennes et interpolation par B‑splines (avec mode adaptatif) pour produire des agrandissements nets, naturels et robustes, avec moins d’artefacts que les méthodes classiques (bilinéaire/bicubique).

## Fonctionnalités
- Redimensionnement de haute qualité via une combinaison de techniques complémentaires:
  - Diffusion anisotrope (edge‑preserving) pour atténuer le bruit tout en préservant les contours
  - Pyramides laplaciennes multi‑résolution pour contrôler les détails à plusieurs échelles
  - Interpolation par B‑splines (standard et adaptative) pour une reconstruction lisse et précise
- Traitement d’un fichier unique ou d’un dossier complet (batch), avec option pour préserver l’arborescence
- Interface en ligne de commande riche (via `argparse`)
- Dossiers par défaut:
  - `data/input` (entrée), `data/transformed` (sortie)
  - `models` pour journaliser les paramètres de chaque exécution (`models/resize_params.json`)

## Prérequis
- Python 3.9+ recommandé
- Gestion du projet avec `uv` (au lieu de `pip` classique)
- Le projet fournit `pyproject.toml` et `uv.lock` pour des installations reproductibles

### Installer `uv` (Windows/PowerShell)
- Recommandé (via pipx):
```powershell
pipx install uv
```
- Alternative (via pip):
```powershell
python -m pip install --upgrade pip
pip install uv
```

### Installer les dépendances avec `uv`
Depuis la racine du projet:
```powershell
uv sync
```
- `uv sync` lit `pyproject.toml` et `uv.lock`, crée un environnement virtuel (par défaut `.venv/`) et installe les dépendances.

Si vous préférez une étape explicite de création d'environnement:
```powershell
uv venv
uv sync
```

Remarque: un `requirements.txt` est présent pour compatibilité, mais avec `uv` il n'est pas nécessaire. Pour une installation sans `uv`, vous pouvez toujours faire:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Structure du projet
```
Image_Resizer/
├─ pipeline.py                 # Script CLI principal (pipeline complet)
├─ main.py                     # Conversion de notebook_json.json -> .ipynb
├─ resize_modern_bspline.ipynb # Notebook d’origine (ou généré)
├─ notebook_json.json          # Source JSON du notebook
├─ README.md                   # Ce fichier
├─ requirements.txt            # (Optionnel) dépendances
├─ uv.lock / pyproject.toml    # Métadonnées projet (si utilisées)
└─ data/
   ├─ input/                   # Images d’entrée (créé auto si absent)
   └─ transformed/             # Images de sortie (créé auto si absent)
```

## Utilisation rapide (CLI)
Le script principal est `pipeline.py`. Il crée automatiquement les dossiers `data/input`, `data/transformed` et `models` si nécessaires.

Avec `uv` (recommandé), exécutez les commandes via l'environnement géré par `uv`:

- Dossier → dossier (préserve la structure des sous‑dossiers):

```powershell
uv run python pipeline.py --input data\input --output data\transformed --scale-x 2 --scale-y 2 --keep-structure
```

- Fichier unique, écrasement autorisé:

```powershell
uv run python pipeline.py -i data\input\photo.jpg -o data\transformed -x 1.5 -y 1.5 --overwrite
```

Alternative sans `uv` (si vous avez activé manuellement votre venv):

```powershell
python pipeline.py --input data\input --output data\transformed --scale-x 2 --scale-y 2 --keep-structure
```

Les images de sortie sont sauvegardées avec le suffixe `_resized` en conservant l’extension d’origine.

## Options de la ligne de commande
Toutes les options exposées par `pipeline.py`:

```text
--input, -i           Chemin d'entrée (fichier image ou dossier). Défaut: data/input
--output, -o          Dossier de sortie. Défaut: data/transformed
--models              Dossier models pour sauvegarder les paramètres du run. Défaut: models

--scale-x, -x         Facteur d'échelle horizontal (float). Défaut: 2.0
--scale-y, -y         Facteur d'échelle vertical (float). Défaut: 2.0

--keep-structure      Préserver la structure des sous-dossiers
--overwrite           Écraser les fichiers de sortie existants

# Paramètres algorithmiques
--spline-order        Ordre des B-splines (3 = cubique). Défaut: 3
--diffusion-iter      Itérations de diffusion anisotrope. Défaut: 10
--pyr-levels          Nombre de niveaux de pyramide. Défaut: 4
--kappa               Seuil de conduction (Perona-Malik). Défaut: 30.0
--gamma               Pas de temps (Perona-Malik). Défaut: 0.15
--option              Forme de conduction (1=exp, 2=ratio). Défaut: 1
--detail-boost        Gain appliqué aux bandes laplaciennes. Défaut: 1.0
--no-sharpen          Désactiver le rehaussement final (unsharp masking)
```

## Détails du pipeline (`resize_modern`)
Le cœur du pipeline suit ces étapes:
1) Filtrage anisotrope (Perona–Malik) pour débruiter en préservant les contours
2) Construction d’une pyramide laplacienne multi‑résolution
3) Reconstruction progressive avec contrôle du niveau de détail
4) Redimensionnement final par B‑splines adaptatives (pondération par énergie locale)
5) Rehaussement optionnel de la netteté

Fonctions clés (implémentées dans `pipeline.py`):
- `anisotropic_diffusion_gray`, `anisotropic_diffusion_color`
- `build_gaussian_pyramid`, `build_laplacian_pyramid`, `reconstruct_from_laplacian`
- `bspline_resize_channel`, `bspline_resize`, `compute_local_energy`, `bspline_resize_adaptive`
- `resize_modern` (pipeline complet)
- `run_pipeline_on_path` (traitement fichier/dossier)

## Conseils et dépannage
- Si la mémoire est insuffisante pour de très grands agrandissements, réduisez `--pyr-levels` et/ou `--diffusion-iter`.
- Si l’image finale paraît trop douce, augmentez `--detail-boost` légèrement (ex: 1.2–1.4) et laissez le sharpen activé (ne pas mettre `--no-sharpen`).
- Si des halos apparaissent, diminuez `--detail-boost` (ex: 0.8–1.0) ou activez `--no-sharpen`.
- Les paramètres de diffusion (`--kappa`, `--gamma`, `--option`) influencent le lissage près des contours; commencez par les valeurs par défaut puis ajustez finement.

## Licence
Ce projet est distribué sous la licence MIT. Voir le fichier `LICENSE` à la racine du dépôt.

## Auteur
Développé par Geovany LAGUERRE — Data Science & Analytics Engineer
