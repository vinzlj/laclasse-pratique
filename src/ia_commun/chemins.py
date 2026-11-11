"""Chemins du projet : racine, dossier de données, checkpoints, figures."""

from __future__ import annotations

import os
from pathlib import Path


def racine_projet() -> Path:
    """Remonte depuis ce fichier jusqu'au dossier qui contient `pyproject.toml`."""
    ici = Path(__file__).resolve()
    for parent in ici.parents:
        if (parent / "pyproject.toml").exists() and (parent / "src" / "ia_commun").exists():
            return parent
    # Paquet installé ailleurs (cas improbable ici) : on prend le dossier courant
    return Path.cwd()


def data_dir(sous_dossier: str | None = None) -> Path:
    """Dossier des données (ignoré par git), créé au besoin.

    Ordre de priorité : variable `DATA_DIR` (posée par le Makefile), sinon
    `<racine du projet>/data`. `sous_dossier` (« mnist », « cifar10 »…) est ajouté.
    """
    base = Path(os.environ.get("DATA_DIR") or racine_projet() / "data")
    chemin = base / sous_dossier if sous_dossier else base
    chemin.mkdir(parents=True, exist_ok=True)
    return chemin


def checkpoints_dir() -> Path:
    """`data/checkpoints/` : modèles entraînés partagés entre exercices (ex. 20 → 28)."""
    return data_dir("checkpoints")
