"""Tests rapides de ia_commun (sans réseau, < 15 s). Lancer : `uv run pytest`."""

from __future__ import annotations

import os
import pickle

import numpy as np
import pytest
import torch

import ia_commun
from ia_commun import data_dir, get_device, is_smoke, seed_everything
from ia_commun.augmentations import AugmentationClassique, AugmentationSimCLR
from ia_commun.mini_gpt import (
    ConfigGPT,
    MiniGPT,
    TokeniseurCaracteres,
    entrainer,
    estimer_loss,
    lot_aleatoire,
)


@pytest.fixture
def faux_cifar(tmp_path, monkeypatch):
    """Faux CIFAR-10 (40 images aléatoires) dans un DATA_DIR temporaire : pas de téléchargement."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    d = tmp_path / "cifar10"
    d.mkdir()
    rng = np.random.default_rng(0)
    for split, n in (("train", 40), ("test", 10)):
        np.save(d / f"{split}_images.npy", rng.integers(0, 256, (n, 32, 32, 3), dtype=np.uint8))
        np.save(d / f"{split}_labels.npy", rng.integers(0, 10, n).astype(np.int64))
    return d


def test_get_device_renvoie_un_device(capsys):
    dev = get_device()
    assert dev.type in {"mps", "cpu", "cuda"}
    assert "Périphérique" in capsys.readouterr().out
    assert get_device("cpu", verbeux=False).type == "cpu"


def test_seed_everything_reproductible(monkeypatch):
    monkeypatch.delenv("SEED", raising=False)
    assert seed_everything() == 42
    a = (torch.rand(3), np.random.rand(3))
    seed_everything(42)
    b = (torch.rand(3), np.random.rand(3))
    assert torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    monkeypatch.setenv("SEED", "7")
    assert seed_everything() == 7


def test_is_smoke(monkeypatch):
    monkeypatch.delenv("IA_SMOKE", raising=False)
    assert is_smoke(["notebook.py", "--smoke"])
    assert not is_smoke(["notebook.py"])
    monkeypatch.setenv("IA_SMOKE", "1")
    assert is_smoke(["notebook.py"])


def test_data_dir_suit_la_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "donnees"))
    assert data_dir("mnist") == tmp_path / "donnees" / "mnist"
    assert (tmp_path / "donnees" / "mnist").is_dir()


def test_racine_projet():
    assert (ia_commun.racine_projet() / "pyproject.toml").exists()


def test_cifar10_dataset_et_pickle(faux_cifar):
    from ia_commun.donnees import CIFAR10, PairesVues

    ds = CIFAR10("train", transform=AugmentationClassique(), indices=range(10))
    assert len(ds) == 10
    x, y = ds[0]
    assert x.shape == (3, 32, 32) and x.dtype == torch.float32 and 0 <= y < 10
    _ = ds.images  # ouvre le memory map…
    clone = pickle.loads(pickle.dumps(ds))  # …qui ne doit pas être sérialisé
    assert clone._images is None and clone[0][0].shape == (3, 32, 32)
    v1, v2, _ = PairesVues(ds, AugmentationSimCLR())[3]
    assert v1.shape == v2.shape == (3, 32, 32)
    assert not torch.equal(v1, v2)  # deux vues tirées indépendamment


def test_augmentation_par_image_tire_des_parametres_differents():
    """Le piège du batch : ici chaque appel tire ses propres paramètres."""
    torch.manual_seed(0)
    img = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
    aug = AugmentationSimCLR()
    sorties = [aug(img) for _ in range(6)]
    assert len({round(s.sum().item(), 3) for s in sorties}) > 1


def test_dataloader_workers_spawn(faux_cifar):
    """Les workers en mode spawn (défaut macOS) doivent pouvoir recevoir le Dataset."""
    from torch.utils.data import DataLoader

    from ia_commun.donnees import CIFAR10

    ds = CIFAR10("train", transform=AugmentationClassique())
    dl = DataLoader(ds, batch_size=8, num_workers=2, multiprocessing_context="spawn")
    xb, yb = next(iter(dl))
    assert xb.shape == (8, 3, 32, 32) and yb.shape == (8,)


def test_texte_caracteres():
    from ia_commun.donnees import TexteCaracteres

    tok = TokeniseurCaracteres("abcabc")
    ids = torch.tensor(tok.encoder("abcabcabc"))
    ds = TexteCaracteres(ids, taille_bloc=4)
    x, y = ds[1]
    assert tok.decoder(x) == "bcab" and tok.decoder(y) == "cabc"


def test_mini_gpt_apprend_un_motif():
    seed_everything(0)
    texte = "le chat dort. " * 200
    tok = TokeniseurCaracteres(texte)
    donnees = torch.tensor(tok.encoder(texte), dtype=torch.long)
    cfg = ConfigGPT(taille_vocab=tok.taille_vocab, taille_bloc=16, n_couches=2, n_tetes=2,
                    n_embd=32, dropout=0.0)
    modele = MiniGPT(cfg)
    assert modele.nb_parametres() > 0
    x, y = lot_aleatoire(donnees, 16, 8, "cpu")
    logits, loss = modele(x, y)
    assert logits.shape == (8, 16, tok.taille_vocab)
    assert abs(loss.item() - np.log(tok.taille_vocab)) < 0.5  # loss initiale ≈ ln(V)
    historique = entrainer(modele, donnees, n_pas=150, device="cpu", taille_lot=16, lr=3e-3)
    assert historique[-1] < historique[0] / 3
    assert estimer_loss(modele, donnees, 16, 8, "cpu", n_lots=3) < 1.0
    g = torch.Generator().manual_seed(0)
    sortie = modele.generer(torch.tensor([tok.encoder("le ")]), 10, generateur=g, top_k=3)
    assert sortie.shape == (1, 13)


def test_mini_gpt_sauvegarde(tmp_path):
    cfg = ConfigGPT(taille_vocab=10, taille_bloc=8, n_couches=1, n_tetes=1, n_embd=8)
    m = MiniGPT(cfg)
    m.sauvegarder(tmp_path / "m.pt", vocab="abc")
    m2, extra = MiniGPT.charger(tmp_path / "m.pt")
    assert extra["vocab"] == "abc"
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))


def test_style_et_sauvegarde_figure(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ia_commun import sauvegarder_figure, style_commun

    style_commun()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    chemins = sauvegarder_figure(fig, "essai", tmp_path, formats=("png", "svg"))
    assert all(p.exists() and p.stat().st_size > 0 for p in chemins)
    plt.close(fig)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS absent")
def test_mini_gpt_dropout_sur_mps():
    """Régression : SDPA + dropout n'est pas implémenté sur MPS (torch 2.14.1)."""
    cfg = ConfigGPT(taille_vocab=11, taille_bloc=8, n_couches=1, n_tetes=2, n_embd=16, dropout=0.1)
    m = MiniGPT(cfg).to("mps").train()
    idx = torch.randint(0, 11, (2, 8), device="mps")
    _, loss = m(idx, idx)
    loss.backward()
    with torch.no_grad():
        m(idx)  # mode train sans gradient : chemin qui plantait
    assert m.generer(idx[:, :2], 3).shape == (2, 5) and m.training


def test_importer_faiss_a_cote_de_torch():
    """Contournement libomp : recherche faiss dans un processus qui utilise torch."""
    import subprocess
    import sys

    code = ("from ia_commun.compat import importer_faiss; faiss = importer_faiss(); import torch, numpy as np;"
            "x = torch.randn(500, 32); (x @ x.T).sum().item();"
            "i = faiss.IndexFlatL2(32); b = np.random.rand(1000, 32).astype('float32'); i.add(b);"
            "print(i.search(b[:50], 3)[1][0, 0])")
    r = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.strip() == "0"


def test_cifar10_premier_telechargement_concurrent(tmp_path):
    """3 processus demandent CIFAR-10 en même temps : une seule conversion, .npy complets."""
    import subprocess
    import sys

    code = (
        "import time, numpy as np, ia_commun.donnees as d\n"
        "def faux(dossier):\n"
        "    with open(dossier / 'appels', 'a') as f: f.write('x')\n"
        "    time.sleep(0.5)\n"
        "    for s in ('train', 'test'):\n"
        "        d._sauver_npy_atomique(dossier / f'{s}_labels.npy', np.arange(3))\n"
        "        d._sauver_npy_atomique(dossier / f'{s}_images.npy', np.zeros((3, 32, 32, 3), np.uint8))\n"
        "d._convertir_cifar10 = faux\n"
        "print(len(d.CIFAR10('train')))\n"
    )
    env = {**os.environ, "DATA_DIR": str(tmp_path)}
    procs = [subprocess.Popen([sys.executable, "-c", code], env=env, stdout=subprocess.PIPE, text=True)
             for _ in range(3)]
    sorties = [p.communicate(timeout=120)[0].strip() for p in procs]
    assert all(p.returncode == 0 for p in procs)
    assert sorties == ["3", "3", "3"]
    assert (tmp_path / "cifar10" / "appels").read_text() == "x"  # une seule conversion
    assert not list((tmp_path / "cifar10").glob("*.part"))
