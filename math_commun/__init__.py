"""Paquet commun des notebooks du parcours de maths.

Graine, mode smoke, style des figures et sauvegarde, palette, chemins. Le paquet n’est pas
installé : ``make nb`` et ``make smoke-nb`` placent la racine du dépôt dans ``PYTHONPATH``
(``marimo edit`` lancé par ``make nb`` aussi). Le gabarit
(``notebooks/_gabarit/notebook.py``) garde un repli local aux mêmes noms et signatures.
"""

from __future__ import annotations

import os
import random
import sys
from collections.abc import Sequence
from pathlib import Path

__all__ = [
    "PALETTE", "RACINE", "NOTEBOOKS", "is_smoke", "seed_everything", "style_commun", "sauvegarder_figure",
]

#: Racine du dépôt et dossier des notebooks.
RACINE = Path(__file__).resolve().parent.parent
NOTEBOOKS = RACINE / "notebooks" / "maths"

#: Couleurs des séries, accordées à celles du site (--mp-serie-1 à 3 de custom.css).
PALETTE = ("#2563eb", "#d97706", "#059669", "#db2777", "#7c3aed", "#0891b2")


def is_smoke(argv: Sequence[str] | None = None) -> bool:
    """Mode smoke : option ``--smoke`` ou variable d’environnement ``MATH_SMOKE=1``."""
    argv = sys.argv if argv is None else argv
    return "--smoke" in argv or os.environ.get("MATH_SMOKE", "") == "1"


def seed_everything(seed: int | None = None) -> int:
    """Fixe les graines de ``random`` et de numpy (``SEED`` de l’environnement, 42 par défaut)."""
    seed = int(os.environ.get("SEED", 42)) if seed is None else int(seed)
    random.seed(seed)
    try:
        import numpy as np
    except ModuleNotFoundError:  # numpy absent : seule la graine de random est fixée
        return seed
    np.random.seed(seed)
    return seed


def style_commun() -> None:
    """Style matplotlib commun aux figures des notebooks."""
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams.update({
        "figure.figsize": (7, 4.2), "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "axes.prop_cycle": cycler(color=list(PALETTE)),
    })


def sauvegarder_figure(fig, nom: str, dossier: str | Path = "figures", formats: Sequence[str] = ("png",)) -> list[Path]:
    """Enregistre ``fig`` sous ``dossier/nom.<format>`` ; renvoie les chemins écrits."""
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    chemins = [dossier / f"{nom}.{fmt}" for fmt in formats]
    for chemin in chemins:
        fig.savefig(chemin)
    return chemins
