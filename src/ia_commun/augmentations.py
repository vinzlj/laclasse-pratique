"""Augmentations d'images appliquées IMAGE PAR IMAGE.

Piège vérifié en phase 1 : une transformation torchvision v2 appliquée à un *batch*
(tenseur N×C×H×W) tire les mêmes paramètres aléatoires pour toutes les images du
batch (même recadrage, même retournement). On applique donc les transformations
dans `Dataset.__getitem__`, sur une seule image à la fois.

Toutes les transformations sont des classes définies au niveau du module, donc
sérialisables pour les workers du DataLoader (mode *spawn* sur macOS).
"""

from __future__ import annotations

import torch
from torchvision.transforms import v2

# Moyenne et écart-type par canal de CIFAR-10 (jeu d'entraînement)
MOYENNE_CIFAR10 = (0.4914, 0.4822, 0.4465)
ECART_TYPE_CIFAR10 = (0.2470, 0.2435, 0.2616)


def _vers_float_normalise() -> list:
    return [
        v2.ToDtype(torch.float32, scale=True),  # uint8 [0, 255] → float32 [0, 1]
        v2.Normalize(MOYENNE_CIFAR10, ECART_TYPE_CIFAR10),
    ]


_NORMALISATION = v2.Compose(_vers_float_normalise())


def normaliser_cifar10(x: torch.Tensor) -> torch.Tensor:
    """Image uint8 3×32×32 → float32 normalisée (sans augmentation : évaluation)."""
    return _NORMALISATION(x)


def denormaliser_cifar10(x: torch.Tensor) -> torch.Tensor:
    """Inverse de la normalisation, pour afficher une image (valeurs ramenées dans [0, 1])."""
    m = torch.tensor(MOYENNE_CIFAR10, device=x.device).view(-1, 1, 1)
    s = torch.tensor(ECART_TYPE_CIFAR10, device=x.device).view(-1, 1, 1)
    return (x * s + m).clamp(0, 1)


class AugmentationClassique:
    """Recadrage aléatoire (marge de 4 pixels) + retournement horizontal (ch. 13)."""

    def __init__(self, taille: int = 32, marge: int = 4):
        self.t = v2.Compose(
            [v2.RandomCrop(taille, padding=marge, padding_mode="reflect"),
             v2.RandomHorizontalFlip(),
             *_vers_float_normalise()]
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.t(x)


class AugmentationSimCLR:
    """Augmentations de SimCLR (Chen et al., 2020, arXiv:2002.05709) adaptées à 32×32.

    Recadrage redimensionné (échelle 0,2 à 1), retournement, distorsion de couleur
    (probabilité 0,8), niveaux de gris (0,2). Pas de flou gaussien : inutile en 32×32,
    comme dans l'expérience CIFAR-10 de l'article.
    """

    def __init__(self, taille: int = 32, force: float = 0.5):
        self.t = v2.Compose(
            [v2.RandomResizedCrop(taille, scale=(0.2, 1.0), antialias=True),
             v2.RandomHorizontalFlip(),
             v2.RandomApply([v2.ColorJitter(0.8 * force, 0.8 * force, 0.8 * force, 0.2 * force)],
                            p=0.8),
             v2.RandomGrayscale(p=0.2),
             *_vers_float_normalise()]
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.t(x)
