"""Téléchargeurs de jeux de données avec cache dans `data/`, et classes Dataset partagées.

Pourquoi ces classes vivent ici et pas dans les notebooks : sur macOS, les workers
du DataLoader (`num_workers > 0`) sont lancés en mode *spawn* et doivent pouvoir
importer la classe Dataset et ses transformations. Une classe définie dans une
cellule marimo (donc dans une fonction) ne peut pas être sérialisée.

Sources :
- MNIST via torchvision ;
- CIFAR-10 depuis Hugging Face (`uoft-cs/cifar10`, parquet), bien plus rapide que le
  serveur de Toronto ; converti une fois en tableaux `.npy` lus en *memory map* ;
- TinyShakespeare depuis le dépôt char-rnn de Karpathy (1 115 394 octets).
"""

from __future__ import annotations

import contextlib
import fcntl
import io
import os
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .chemins import data_dir

URL_TINYSHAKESPEARE = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
TAILLE_TINYSHAKESPEARE = 1_115_394
DEPOT_CIFAR10 = "uoft-cs/cifar10"
CLASSES_CIFAR10 = (
    "avion", "automobile", "oiseau", "chat", "cerf",
    "chien", "grenouille", "cheval", "bateau", "camion",
)


# ---------------------------------------------------------------- outils de cache


@contextlib.contextmanager
def _verrou_telechargement(dossier: Path, nom: str) -> Iterator[None]:
    """Verrou de fichier (flock) autour d'un premier téléchargement.

    Plusieurs lots (ou git worktrees partageant `DATA_DIR`) peuvent lancer le même
    téléchargement en même temps : un seul processus télécharge, les autres attendent
    puis trouvent le cache rempli. Le verrou est libéré par le système si le
    processus meurt (pas de verrou orphelin).
    """
    dossier.mkdir(parents=True, exist_ok=True)
    with open(dossier / f".{nom}.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _temporaire(chemin: Path) -> Path:
    """Nom temporaire propre au processus, dans le même dossier (pour un `replace` atomique)."""
    return chemin.with_name(f"{chemin.name}.{os.getpid()}.part")


def _ecrire_atomique(chemin: Path, contenu: bytes) -> None:
    tmp = _temporaire(chemin)
    tmp.write_bytes(contenu)
    tmp.replace(chemin)  # pas de fichier à moitié écrit dans le cache


def _sauver_npy_atomique(chemin: Path, tableau: np.ndarray) -> None:
    tmp = _temporaire(chemin)
    with open(tmp, "wb") as f:  # objet fichier : np.save n'ajoute pas « .npy » au nom
        np.save(f, tableau)
    tmp.replace(chemin)


# ---------------------------------------------------------------- téléchargeurs


def telecharger_tinyshakespeare() -> Path:
    """Télécharge TinyShakespeare dans `data/input.txt` (une seule fois) et renvoie le chemin."""
    chemin = data_dir() / "input.txt"

    def present() -> bool:
        return chemin.exists() and chemin.stat().st_size == TAILLE_TINYSHAKESPEARE

    if present():
        return chemin
    with _verrou_telechargement(chemin.parent, "tinyshakespeare"):
        if present():  # téléchargé par un autre processus pendant l'attente
            return chemin
        print(f"[ia_commun] Téléchargement de TinyShakespeare → {chemin}")
        requete = urllib.request.Request(URL_TINYSHAKESPEARE, headers={"User-Agent": "ia-experiments"})
        with urllib.request.urlopen(requete, timeout=60) as reponse:
            contenu = reponse.read()
        _ecrire_atomique(chemin, contenu)
    return chemin


def charger_tinyshakespeare() -> str:
    """Renvoie le texte complet de TinyShakespeare (téléchargé au besoin)."""
    return telecharger_tinyshakespeare().read_text(encoding="utf-8")


def telecharger_mnist(train: bool = True):
    """Renvoie le Dataset torchvision MNIST (images PIL, étiquettes int), cache `data/mnist/`.

    Passer ensuite `transform=` vous-même, ou utiliser `mnist_tenseurs()`.
    """
    from torchvision import datasets

    racine = data_dir("mnist")
    # torchvision n'écrit pas de façon atomique : premier téléchargement sous verrou.
    with _verrou_telechargement(racine, "mnist"):
        return datasets.MNIST(root=str(racine), train=train, download=True)


def mnist_tenseurs(train: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """MNIST sous forme de tenseurs : images float32 (N, 1, 28, 28) dans [0, 1], étiquettes int64."""
    ds = telecharger_mnist(train)
    x = ds.data.unsqueeze(1).to(torch.float32) / 255.0
    return x, ds.targets.clone().to(torch.int64)


def telecharger_cifar10() -> Path:
    """Télécharge CIFAR-10 depuis le Hub et le convertit en `.npy` dans `data/cifar10/`.

    Fichiers produits : `{train,test}_images.npy` (uint8, N×32×32×3) et
    `{train,test}_labels.npy` (int64). Renvoie le dossier.
    """
    dossier = data_dir("cifar10")
    attendus = [dossier / f"{s}_{k}.npy" for s in ("train", "test") for k in ("images", "labels")]
    if all(p.exists() for p in attendus):
        return dossier

    with _verrou_telechargement(dossier, "cifar10"):
        if all(p.exists() for p in attendus):  # converti par un autre processus pendant l'attente
            return dossier
        _convertir_cifar10(dossier)
    return dossier


def _convertir_cifar10(dossier: Path) -> None:
    """Télécharge les parquet de CIFAR-10 et écrit les `.npy` (appelé sous verrou)."""
    import shutil

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    print("[ia_commun] Téléchargement de CIFAR-10 (Hugging Face, ~120 Mo)…")
    for split in ("train", "test"):
        fichier = hf_hub_download(
            DEPOT_CIFAR10,
            f"plain_text/{split}-00000-of-00001.parquet",
            repo_type="dataset",
            local_dir=dossier / "brut",
        )
        table = pq.read_table(fichier)
        colonne_img = "img" if "img" in table.column_names else table.column_names[0]
        images = np.stack(
            [np.asarray(Image.open(io.BytesIO(d["bytes"])).convert("RGB"))
             for d in table.column(colonne_img).to_pylist()]
        ).astype(np.uint8)
        labels = np.asarray(table.column("label").to_pylist(), dtype=np.int64)
        # Écritures atomiques : un lecteur ne voit jamais un .npy tronqué.
        _sauver_npy_atomique(dossier / f"{split}_labels.npy", labels)
        _sauver_npy_atomique(dossier / f"{split}_images.npy", images)
        print(f"[ia_commun]   {split} : {images.shape[0]} images")
    # Les parquet ne servent plus : les .npy suffisent (gain de ~120 Mo)
    shutil.rmtree(dossier / "brut", ignore_errors=True)


# ---------------------------------------------------------------- Datasets


class CIFAR10(Dataset):
    """CIFAR-10 lu depuis les `.npy` en *memory map* (léger à envoyer aux workers).

    `transform` reçoit **une seule image** (tenseur uint8 3×32×32) : les paramètres
    aléatoires sont donc tirés image par image (voir `augmentations.py`).
    Sans `transform`, renvoie l'image normalisée en float32.
    `indices` permet de travailler sur un sous-ensemble (mode smoke, SimCLR 25 000 images).
    """

    def __init__(self, split: str = "train", transform=None, indices=None):
        if split not in ("train", "test"):
            raise ValueError("split doit valoir 'train' ou 'test'")
        self.dossier = telecharger_cifar10()
        self.split = split
        self.transform = transform
        self.labels = np.load(self.dossier / f"{split}_labels.npy")
        self.indices = np.arange(len(self.labels)) if indices is None else np.asarray(indices)
        self._images = None  # ouvert à la demande dans chaque processus

    @property
    def images(self) -> np.ndarray:
        if self._images is None:
            self._images = np.load(self.dossier / f"{self.split}_images.npy", mmap_mode="r")
        return self._images

    def __getstate__(self):
        # On n'envoie pas le memory map aux workers : chacun le rouvre.
        etat = self.__dict__.copy()
        etat["_images"] = None
        return etat

    def __len__(self) -> int:
        return len(self.indices)

    def image_brute(self, i: int) -> torch.Tensor:
        """Image i en uint8, format C×H×W (convention torchvision)."""
        j = int(self.indices[i])
        return torch.from_numpy(np.array(self.images[j])).permute(2, 0, 1).contiguous()

    def __getitem__(self, i: int):
        from .augmentations import normaliser_cifar10

        x = self.image_brute(i)
        x = self.transform(x) if self.transform is not None else normaliser_cifar10(x)
        return x, int(self.labels[int(self.indices[i])])


class PairesVues(Dataset):
    """Enveloppe pour l'apprentissage contrastif (SimCLR, ch. 23) : deux vues augmentées.

    Renvoie `(vue1, vue2, etiquette)`. L'étiquette ne sert qu'à la sonde linéaire.
    """

    def __init__(self, base: CIFAR10, transform):
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        x = self.base.image_brute(i)
        y = int(self.base.labels[int(self.base.indices[i])])
        return self.transform(x), self.transform(x), y


class TexteCaracteres(Dataset):
    """Texte découpé en fenêtres de `taille_bloc` caractères pour un modèle de langage.

    `__getitem__(i)` renvoie `(x, y)` où y est x décalé d'un caractère (token suivant).
    """

    def __init__(self, ids: torch.Tensor, taille_bloc: int):
        if len(ids) <= taille_bloc:
            raise ValueError("Texte plus court que la fenêtre")
        self.ids = ids
        self.taille_bloc = taille_bloc

    def __len__(self) -> int:
        return len(self.ids) - self.taille_bloc

    def __getitem__(self, i: int):
        morceau = self.ids[i : i + self.taille_bloc + 1]
        return morceau[:-1], morceau[1:]
