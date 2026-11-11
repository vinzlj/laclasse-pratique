"""Choix du périphérique de calcul, graines aléatoires et mode « smoke ».

Pièges MPS (Apple Silicon) à garder en tête :
- pas de float64 sur MPS : `torch.from_numpy(tableau_float64).to("mps")` échoue ;
  convertir d'abord en float32 (`tableau.astype(np.float32)`) ;
- `PYTORCH_ENABLE_MPS_FALLBACK=1` (exporté par le Makefile) renvoie sur le CPU les
  opérations non implémentées sur MPS au lieu de lever une erreur ;
- « même graine = même courbe à quelques décimales près, pas au bit près » : MPS
  n'est pas déterministe au bit près ;
- pas de `torch.compile` : on reste en mode eager.
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch

GRAINE_PAR_DEFAUT = 42


def get_device(prefere: str | None = None, verbeux: bool = True) -> torch.device:
    """Renvoie le meilleur périphérique disponible : MPS (GPU Apple), sinon CPU.

    `prefere` force un choix ("cpu", "mps", "cuda") ; la variable d'environnement
    `IA_DEVICE` fait de même (utile pour comparer CPU et MPS sans toucher au code).
    CUDA est accepté pour que le même code tourne sur Colab ou Kaggle.
    """
    choix = (prefere or os.environ.get("IA_DEVICE") or "").lower()
    if choix:
        device = torch.device(choix)
        raison = "imposé par l'utilisateur"
    elif torch.cuda.is_available():
        device, raison = torch.device("cuda"), "GPU NVIDIA détecté"
    elif torch.backends.mps.is_available():
        device, raison = torch.device("mps"), "GPU Apple (Metal Performance Shaders)"
    else:
        device = torch.device("cpu")
        if torch.backends.mps.is_built():
            raison = "MPS compilé dans torch mais indisponible (macOS trop ancien ?) : repli CPU"
        else:
            raison = "aucun GPU utilisable : calcul sur CPU (plus lent mais correct)"
    if verbeux:
        print(f"[ia_commun] Périphérique : {device} ({raison})")
    return device


def seed_everything(seed: int | None = None) -> int:
    """Fixe les graines de `random`, NumPy et PyTorch (CPU, MPS, CUDA).

    Sans argument, lit la variable d'environnement `SEED` (posée par le Makefile),
    sinon 42. Renvoie la graine utilisée, pour l'afficher dans le notebook.
    """
    if seed is None:
        seed = int(os.environ.get("SEED", GRAINE_PAR_DEFAUT))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # couvre aussi CUDA et MPS dans les versions récentes
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def is_smoke(argv: list[str] | None = None) -> bool:
    """Vrai si l'exercice est lancé en mode test rapide (< 30 s).

    Lancé par `make ex NUM=… MODE=smoke` ou `make smoke`, qui passent `--smoke`
    au script. La variable d'environnement `IA_SMOKE=1` a le même effet (pratique
    dans l'éditeur marimo, qui ne transmet pas d'arguments).
    """
    argv = sys.argv if argv is None else argv
    return "--smoke" in argv or os.environ.get("IA_SMOKE", "") == "1"
