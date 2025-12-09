from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

from .io_utils import ensure_dir


def _load_json_list(path: Path) -> List[Any]:
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = [data]
            return data
        except Exception:
            return []
    return []


def _append_json_list(path: Path, entry: Any) -> None:
    hist = _load_json_list(path)
    hist.append(entry)
    with path.open("w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)


def save_params_json(models_dir: str | Path, params: dict) -> None:
    models_path = Path(models_dir)
    ensure_dir(models_path)
    json_file = models_path / "resize_params.json"
    _append_json_list(json_file, params)


def append_run_metrics(models_dir: str | Path, run_metrics: dict) -> None:
    models_path = Path(models_dir)
    ensure_dir(models_path)
    metrics_file = models_path / "metrics.json"
    _append_json_list(metrics_file, run_metrics)
