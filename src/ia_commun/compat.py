"""Contournements de plateforme mesurés sur Mac Apple Silicon.

torch et faiss-cpu embarquent chacun leur propre libomp. Sur macOS arm64 (torch 2.14.1,
faiss-cpu 1.15.1), une recherche faiss multithread dans un processus qui a importé torch
plante (segfault, « OMP: System error #22 ») ; pas d'« OMP Error #15 », et
KMP_DUPLICATE_LIB_OK=TRUE n'y change rien.

Ce qui marche (testé) : importer torch AVANT faiss, puis limiter faiss à un thread.
Pour les petits corpus du cours (quelques milliers de vecteurs), la perte est négligeable.
"""

from __future__ import annotations

import sys


def importer_faiss():
    """Importe faiss de façon sûre à côté de torch. Usage : `faiss = importer_faiss()`."""
    if "faiss" in sys.modules and "torch" not in sys.modules:
        raise RuntimeError(
            "faiss a été importé avant torch : importez torch d'abord (ou utilisez "
            "importer_faiss() dès le début), sinon la recherche faiss plante sur macOS."
        )
    # L'ordre compte : torch AVANT faiss. L'instruction intercalée empêche un trieur
    # d'imports (ruff, isort) de les inverser.
    import torch

    _ = torch.__version__
    import faiss

    faiss.omp_set_num_threads(1)
    return faiss
