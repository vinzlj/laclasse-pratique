# Gabarit d'exercice local (marimo). Copier ce dossier en exercices/ia/NN-nom-court/.
#
# Trois façons de l'exécuter :
#   make ex NUM=_gabarit              → éditeur marimo (interactif, bouton « Entraîner »)
#   make ex NUM=_gabarit MODE=run     → script complet (le bouton est considéré comme cliqué)
#   make ex NUM=_gabarit MODE=smoke   → test rapide, < 30 s, utilisé par `make smoke`
#
# Règles : graine 42 (seed_everything), périphérique via get_device(), classes Dataset
# et augmentations dans src/ia_commun (jamais dans une cellule si num_workers > 0),
# figure sauvegardée dans figures/, entraînement derrière mo.ui.run_button().

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch import nn

    from ia_commun import get_device, is_smoke, sauvegarder_figure, seed_everything, style_commun

    return (
        get_device,
        is_smoke,
        mo,
        nn,
        np,
        plt,
        sauvegarder_figure,
        seed_everything,
        style_commun,
        time,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Gabarit : classer deux demi-lunes avec un petit MLP

    **Objectif.** Montrer la structure attendue d'un exercice, pas enseigner quelque chose :
    configuration, données, modèle, entraînement derrière un bouton, figure, bilan chiffré.

    La loss optimisée est l'entropie croisée binaire
    $\mathcal{L} = -\frac{1}{N}\sum_i \big[y_i \log \hat y_i + (1-y_i)\log(1-\hat y_i)\big]$.
    """)


@app.cell
def _(get_device, is_smoke, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    device = get_device()
    style_commun()
    config = dict(
        n_points=400 if SMOKE else 4000,
        n_epochs=20 if SMOKE else 300,
        lr=1e-2,
        cachee=32,
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SMOKE, config, device


@app.cell
def _(config, np, torch):
    # --- Données : synthétiques ici (aucun téléchargement). Un vrai exercice utilise
    # ia_commun.donnees (MNIST, CIFAR10, TinyShakespeare), cache dans data/.
    from sklearn.datasets import make_moons

    X_np, y_np = make_moons(n_samples=config["n_points"], noise=0.2, random_state=42)
    # float32 obligatoire : MPS ne connaît pas le float64 de NumPy
    X = torch.from_numpy(X_np.astype(np.float32))
    y = torch.from_numpy(y_np.astype(np.float32))
    return X, X_np, y, y_np


@app.cell
def _(config, nn):
    def creer_modele():
        return nn.Sequential(
            nn.Linear(2, config["cachee"]), nn.ReLU(),
            nn.Linear(config["cachee"], config["cachee"]), nn.ReLU(),
            nn.Linear(config["cachee"], 1),
        )

    return (creer_modele,)


@app.cell
def _(mo):
    # Le bouton empêche l'entraînement de se relancer à chaque modification d'une cellule amont.
    bouton = mo.ui.run_button(label="Entraîner")
    bouton
    return (bouton,)


@app.cell
def _(X, bouton, config, creer_modele, device, mo, nn, time, torch, y):
    # En script (make ex MODE=run|smoke), il n'y a personne pour cliquer : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Entraîner » pour lancer l'apprentissage.*"))

    modele = creer_modele().to(device)
    optim = torch.optim.Adam(modele.parameters(), lr=config["lr"])
    critere = nn.BCEWithLogitsLoss()
    Xd, yd = X.to(device), y.to(device)
    pertes = []
    t0 = time.perf_counter()
    for _epoch in range(config["n_epochs"]):
        loss = critere(modele(Xd).squeeze(1), yd)
        optim.zero_grad()
        loss.backward()
        optim.step()
        pertes.append(loss.item())
    duree = time.perf_counter() - t0
    with torch.no_grad():
        precision = ((modele(Xd).squeeze(1) > 0).float() == yd).float().mean().item()
    print(f"Entraînement : {duree:.1f} s, loss finale {pertes[-1]:.3f}, précision {precision:.1%}")
    return duree, modele, pertes, precision


@app.cell
def _(
    SMOKE,
    X_np,
    device,
    mo,
    modele,
    np,
    pertes,
    plt,
    sauvegarder_figure,
    torch,
    y_np,
):
    # --- Figure : courbe de loss + frontière de décision, sauvegardée dans figures/ ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(pertes)
    ax1.set(xlabel="époque", ylabel="loss (entropie croisée)", title="Courbe d'apprentissage")
    xx, yy = np.meshgrid(np.linspace(-2, 3, 150), np.linspace(-1.5, 2, 150))
    grille = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()].astype(np.float32)).to(device)
    with torch.no_grad():
        zz = torch.sigmoid(modele(grille)).cpu().numpy().reshape(xx.shape)
    ax2.contourf(xx, yy, zz, levels=20, cmap="RdBu", alpha=0.6)
    ax2.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap="RdBu", s=8, edgecolors="k", linewidths=0.2)
    ax2.set(title="Frontière de décision")
    dossier = mo.notebook_dir() / "figures"
    # En smoke, on écrit à côté pour ne pas écraser la figure du run complet
    chemins = sauvegarder_figure(fig, "frontiere-smoke" if SMOKE else "frontiere", dossier)
    print(f"Figure : {chemins[0]}")
    fig


@app.cell
def _(SMOKE, duree, mo, precision):
    # --- Bilan : seuil chiffré du README. En smoke, on vérifie seulement que tout tourne.
    SEUIL = 0.95
    if not SMOKE and precision < SEUIL:
        raise SystemExit(f"Précision {precision:.1%} sous le seuil attendu de {SEUIL:.0%}")
    mo.md(f"**Bilan** : précision {precision:.1%} (seuil {SEUIL:.0%}), {duree:.1f} s d'entraînement.")


if __name__ == "__main__":
    app.run()
