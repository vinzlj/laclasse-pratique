"""ia_commun : utilitaires partagés par tous les exercices du parcours.

Import typique en tête de notebook :

    from ia_commun import get_device, seed_everything, is_smoke, data_dir

Les sous-modules lourds (données, mini-GPT) s'importent explicitement :

    from ia_commun.donnees import CIFAR10, charger_tinyshakespeare
    from ia_commun.augmentations import AugmentationSimCLR
    from ia_commun.mini_gpt import MiniGPT, ConfigGPT
"""

from .chemins import checkpoints_dir, data_dir, racine_projet
from .materiel import get_device, is_smoke, seed_everything
from .traces import PALETTE, sauvegarder_figure, style_commun

__all__ = [
    "PALETTE",
    "checkpoints_dir",
    "data_dir",
    "get_device",
    "is_smoke",
    "racine_projet",
    "sauvegarder_figure",
    "seed_everything",
    "style_commun",
]
