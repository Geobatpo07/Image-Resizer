import json
from pathlib import Path

# Convertit le fichier JSON de notebook en fichier .ipynb valide
# Entrée: notebook_json.json (contenu déjà fourni dans le dépôt)
# Sortie: resize_modern_bspline.ipynb

def convert_json_to_ipynb(
    input_path: str = "notebook_json.json",
    output_path: str = "resize_modern_bspline.ipynb",
) -> None:
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Fichier introuvable: {input_file}")

    with input_file.open("r", encoding="utf-8") as f_in:
        notebook_obj = json.load(f_in)

    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(notebook_obj, f_out, ensure_ascii=False, indent=2)

    print(f"Notebook écrit: {output_path}")


if __name__ == "__main__":
    convert_json_to_ipynb()
