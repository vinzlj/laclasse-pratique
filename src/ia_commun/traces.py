"""Style commun des figures matplotlib et sauvegarde.

Usage :
    from ia_commun import style_commun, sauvegarder_figure
    style_commun()
    fig, ax = plt.subplots()
    ...
    sauvegarder_figure(fig, "courbe-loss", dossier=Path(__file__).parent / "figures")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Pas de choix de backend ici : le Makefile exporte MPLBACKEND=Agg pour les modes
# run et smoke (aucune fenêtre), et marimo gère lui-même l'affichage dans l'éditeur.
import matplotlib.pyplot as plt

# Palette sobre et lisible par les daltoniens (Okabe-Ito)
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]


def style_commun() -> None:
    """Applique le style partagé par tous les exercices (taille, grille, couleurs)."""
    plt.rcParams.update(
        {
            "figure.figsize": (7, 4.2),
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
            "font.size": 11,
            "axes.titlesize": 12,
            "legend.frameon": False,
            "axes.formatter.use_locale": False,
        }
    )


def sauvegarder_figure(fig, nom: str, dossier: str | Path = "figures",
                       formats: tuple[str, ...] = ("png",)) -> list[Path]:
    """Enregistre `fig` sous `dossier/nom.<format>` et renvoie les chemins écrits.

    Les figures d'exercice vont dans `exercices/<num>/figures/` (versionnées seulement
    si le README les montre). Les figures du site sont produites par
    `site/scripts/figures/` en SVG.
    """
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    chemins = []
    for fmt in formats:
        chemin = dossier / f"{nom}.{fmt}"
        fig.savefig(chemin)
        chemins.append(chemin)
    return chemins
