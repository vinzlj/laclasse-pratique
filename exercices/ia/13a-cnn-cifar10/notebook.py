# Exercice 13a : réseaux convolutifs (CNN), de la convolution en NumPy à CIFAR-10.
#
#   make ex NUM=13a              → éditeur marimo (bouton « Entraîner »)
#   make ex NUM=13a MODE=run     → script complet, vérifie les seuils du README
#   make ex NUM=13a MODE=smoke   → test rapide (< 30 s)
#   IA_RESNET18=1 make ex NUM=13a MODE=run   → ajoute l'option ResNet-18 (≈ 1 min 30 s par époque, mesure de la phase 1)
#
# Partie A en NumPy (convolution, pooling, équivariance), vérifiée contre PyTorch.
# Parties B et C en PyTorch sur CIFAR-10 (GPU Apple via MPS si disponible).
# Les augmentations aléatoires sont appliquées IMAGE PAR IMAGE dans le Dataset
# (ia_commun.donnees.CIFAR10 + ia_commun.augmentations.AugmentationClassique), jamais
# sur un lot entier : une transformation v2 appliquée à un lot tire les mêmes paramètres
# pour toutes les images.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from numpy.lib.stride_tricks import sliding_window_view
    from scipy import signal
    from torch import nn

    from ia_commun import get_device, is_smoke, sauvegarder_figure, seed_everything, style_commun

    return (
        F,
        get_device,
        is_smoke,
        json,
        mo,
        nn,
        np,
        os,
        plt,
        sauvegarder_figure,
        seed_everything,
        signal,
        sliding_window_view,
        style_commun,
        time,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Exercice 13a : réseaux convolutifs (CNN)

    Un MLP traite une image 32×32 en couleur comme une liste de 3 072 nombres : il ne sait
    pas que deux pixels voisins sont liés, ni qu'un chat reste un chat décalé de trois
    pixels. Une couche de **convolution** applique le même petit filtre (3×3, 5×5) à
    toutes les positions de l'image : peu de paramètres, et un motif appris à un endroit
    est reconnu partout.

    Plan :

    - **A. La convolution à la main (NumPy).** Écrire l'opération (boucles, puis version
      vectorisée), la comparer à `torch.nn.functional.conv2d`, appliquer des filtres
      classiques (flou, Sobel, Laplacien), calculer tailles de sortie et champs récepteurs,
      vérifier l'**équivariance par translation** et compter les paramètres.
    - **B. MLP contre CNN sur CIFAR-10 (PyTorch).** Même budget d'entraînement pour un MLP,
      un petit CNN, et le même CNN avec **augmentation de données** (recadrage aléatoire et
      retournement horizontal, tirés image par image). Puis tester les trois modèles sur des
      images retournées et décalées.
    - **C. Ce que le CNN a appris.** Les 32 filtres de la première couche, et des cartes
      **Grad-CAM** : quelles régions de l'image ont pesé dans la décision ?

    **Perte optimisée** (parties B et C) : l'entropie croisée multiclasse
    $\mathcal{L} = -\frac1N\sum_i \log \hat p_{i,y_i}$ (chapitre 01). La partie A n'optimise
    rien : les filtres y sont écrits à la main.
    """)


@app.cell
def _(get_device, is_smoke, os, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    device = get_device()
    style_commun()
    config = dict(
        n_train=2000 if SMOKE else 50000,   # CIFAR-10 complet : 50 000 images d'entraînement
        n_test=500 if SMOKE else 10000,
        epoques=1 if SMOKE else 15,         # même budget pour les trois modèles
        lot=128,
        lr_max=2e-3,                        # AdamW + taux « one cycle » (montée puis descente)
        weight_decay=5e-4,
        workers=2 if SMOKE else 6,          # DataLoader de l'entraînement avec augmentation
        decalage=4,                         # pixels, pour le test de robustesse
        n_gradcam=8,
        # Option ResNet-18 (≈ 1 min 30 s par époque sur M1 Pro, mesure de la phase 1) : IA_RESNET18=1
        resnet18=os.environ.get("IA_RESNET18", "0") == "1",
        epoques_resnet=1 if SMOKE else 5,
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, périphérique {device}")
    print(config)
    return SEED, SMOKE, config, device


@app.cell
def _(mo):
    mo.md(r"""
    ## A. La convolution à la main

    Une couche de convolution fait glisser un **noyau** $K$ de taille $k\times k$ sur
    l'image $X$ et calcule, à chaque position, la somme des produits terme à terme :

    $$Y[i,j] = \sum_{a=0}^{k-1}\sum_{b=0}^{k-1} K[a,b]\;X[i+a-p,\; j+b-p]$$

    ($p$ = marge de zéros ajoutée autour de l'image, *padding*). En toute rigueur c'est une
    **corrélation croisée** : la convolution des mathématiciens retourne d'abord le noyau.
    Les bibliothèques d'apprentissage profond gardent le nom « convolution » ; comme le
    noyau est appris, le retournement ne change rien.

    L'image de test est synthétique (64×64, niveaux de gris dans [0, 1]) : un dégradé
    horizontal en fond, un carré clair, un disque, des rayures verticales et des rayures
    horizontales. La démo du chapitre utilise exactement la même.
    """)


@app.cell
def _(np):
    def image_formes(n=64):
        """Image de test 64×64 (identique à la démo `Convolution` du chapitre 13)."""
        i, j = np.mgrid[0:n, 0:n]
        img = 0.2 + 0.3 * j / (n - 1)                       # dégradé horizontal
        img = np.where((i >= 6) & (i < 26) & (j >= 6) & (j < 26), 0.9, img)           # carré
        img = np.where((i - 17) ** 2 + (j - 46) ** 2 <= 90, 0.7, img)               # disque
        zone_v = (i >= 36) & (i < 58) & (j >= 6) & (j < 28)
        img = np.where(zone_v, np.where((j // 3) % 2 == 1, 0.95, 0.1), img)           # rayures verticales
        zone_h = (i >= 36) & (i < 58) & (j >= 36) & (j < 58)
        img = np.where(zone_h, np.where((i // 3) % 2 == 1, 0.95, 0.1), img)           # rayures horizontales
        return img.astype(np.float64)

    # Noyaux classiques (mêmes valeurs que les préréglages de la démo)
    NOYAUX = {
        "identité": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        "flou (moyenne)": [[1 / 9] * 3] * 3,
        "flou gaussien": [[1 / 16, 2 / 16, 1 / 16], [2 / 16, 4 / 16, 2 / 16], [1 / 16, 2 / 16, 1 / 16]],
        "netteté": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        "Sobel x (bords verticaux)": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        "Sobel y (bords horizontaux)": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        "Laplacien": [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        "relief": [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
    }
    NOYAUX = {nom: np.array(k, dtype=np.float64) for nom, k in NOYAUX.items()}
    IMG = image_formes()
    return IMG, NOYAUX


@app.cell
def _(np, sliding_window_view):
    def correlation_boucles(x, k, pad=0, pas=1):
        """Version naïve, quatre boucles imbriquées : lisible, lente."""
        x = np.pad(x, pad)                       # marge de zéros
        kh, kw = k.shape
        h = (x.shape[0] - kh) // pas + 1
        w = (x.shape[1] - kw) // pas + 1
        y = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                s = 0.0
                for a in range(kh):
                    for b in range(kw):
                        s += k[a, b] * x[i * pas + a, j * pas + b]
                y[i, j] = s
        return y

    def correlation(x, k, pad=0, pas=1):
        """Version vectorisée : on découpe toutes les fenêtres k×k d'un coup, puis un
        produit terme à terme et une somme (c'est ce que fait im2col dans les bibliothèques)."""
        x = np.pad(x, pad)
        fenetres = sliding_window_view(x, k.shape)[::pas, ::pas]   # (h, w, kh, kw), sans copie
        return np.einsum("hwab,ab->hw", fenetres, k)

    def max_pooling(x, taille=2):
        """Max pooling taille×taille, pas = taille (bords qui dépassent ignorés)."""
        h, w = x.shape[0] // taille, x.shape[1] // taille
        return x[: h * taille, : w * taille].reshape(h, taille, w, taille).max(axis=(1, 3))

    def taille_sortie(n, k, p, s):
        """Taille de sortie d'une convolution : ⌊(n + 2p − k) / s⌋ + 1."""
        return (n + 2 * p - k) // s + 1

    return correlation, correlation_boucles, max_pooling, taille_sortie


@app.cell
def _(F, IMG, NOYAUX, correlation, correlation_boucles, np, signal, time, torch):
    # --- A1. Trois implémentations, un seul résultat ---
    _k = NOYAUX["Sobel x (bords verticaux)"]
    _t0 = time.perf_counter()
    _y_boucles = correlation_boucles(IMG, _k, pad=1)
    _t_boucles = time.perf_counter() - _t0
    _t0 = time.perf_counter()
    _y_vect = correlation(IMG, _k, pad=1)
    _t_vect = time.perf_counter() - _t0
    # PyTorch : entrée (lot, canaux, H, W), poids (sorties, entrées, kh, kw)
    _y_torch = F.conv2d(torch.from_numpy(IMG)[None, None], torch.from_numpy(_k)[None, None],
                        padding=1)[0, 0].numpy()
    ecart_torch = float(np.abs(_y_vect - _y_torch).max())
    ecart_boucles = float(np.abs(_y_vect - _y_boucles).max())
    # SciPy distingue les deux opérations : correlate2d = notre calcul, convolve2d retourne le noyau
    ecart_correlate = float(np.abs(signal.correlate2d(IMG, _k, mode="same") - _y_vect).max())
    ecart_convolve = float(np.abs(signal.convolve2d(IMG, _k, mode="same") - _y_vect).max())
    ecart_convolve_flou = float(np.abs(signal.convolve2d(IMG, NOYAUX["flou gaussien"], mode="same")
                                       - correlation(IMG, NOYAUX["flou gaussien"], pad=1)).max())
    print(f"Boucles : {_t_boucles * 1000:.0f} ms ; vectorisée : {_t_vect * 1000:.2f} ms "
          f"(× {_t_boucles / _t_vect:.0f})")
    print(f"Écart max vectorisée / boucles : {ecart_boucles:.1e} ; / torch.conv2d : {ecart_torch:.1e} ; "
          f"/ scipy.correlate2d : {ecart_correlate:.1e}")
    print(f"scipy.convolve2d (noyau retourné) avec Sobel x : écart {ecart_convolve:.2f} "
          f"(le signe s'inverse) ; avec le flou gaussien, symétrique : {ecart_convolve_flou:.1e}")

    # Calcul détaillé en un pixel (ligne 15, colonne 6 : bord gauche du carré), comme dans la démo
    _i, _j = 15, 6
    _patch = np.pad(IMG, 1)[_i: _i + 3, _j: _j + 3]
    print(f"\nPixel (ligne {_i}, colonne {_j}), fenêtre 3×3 de l'image :\n{np.round(_patch, 4)}")
    for _nom in ["Sobel x (bords verticaux)", "Sobel y (bords horizontaux)", "flou (moyenne)", "Laplacien"]:
        print(f"  {_nom:<28} → {float((_patch * NOYAUX[_nom]).sum()):+.4f}")
    valeur_sobel_pixel = float((_patch * _k).sum())
    return ecart_boucles, ecart_convolve, ecart_torch, valeur_sobel_pixel


@app.cell
def _(IMG, NOYAUX, SMOKE, correlation, mo, plt, sauvegarder_figure):
    # --- A2. Huit filtres classiques sur l'image de test ---
    _fig, _axes = plt.subplots(3, 3, figsize=(9, 9.4))
    _axes[0, 0].imshow(IMG, cmap="gray", vmin=0, vmax=1)
    _axes[0, 0].set_title("image (64×64)")
    for _ax, (_nom, _k) in zip(_axes.flat[1:], NOYAUX.items()):
        _y = correlation(IMG, _k, pad=1)
        if abs(_k.sum() - 1) < 1e-9:          # noyau qui conserve la luminosité : gris [0, 1]
            _ax.imshow(_y.clip(0, 1), cmap="gray", vmin=0, vmax=1)
        else:                                 # noyau de somme nulle : signe en couleur
            _m = abs(_y).max()
            _ax.imshow(_y, cmap="RdBu_r", vmin=-_m, vmax=_m)
        _ax.set_title(_nom, fontsize=10)
    for _ax in _axes.flat:
        _ax.set_xticks([])
        _ax.set_yticks([])
        _ax.grid(False)
    _fig.tight_layout()
    sauvegarder_figure(_fig, "filtres-classiques-smoke" if SMOKE else "filtres-classiques",
                       mo.notebook_dir() / "figures")
    _fig


@app.cell
def _(IMG, NOYAUX, correlation, max_pooling, np, taille_sortie):
    # --- A3. Tailles de sortie : formule contre calcul réel ---
    for _n, _k, _p, _s in [(64, 3, 0, 1), (64, 3, 1, 1), (64, 5, 2, 1), (64, 3, 1, 2), (32, 5, 0, 1)]:
        _reel = correlation(np.zeros((_n, _n)), np.ones((_k, _k)), pad=_p, pas=_s).shape[0]
        print(f"n={_n}, k={_k}, p={_p}, s={_s} : formule {taille_sortie(_n, _k, _p, _s)}, calcul {_reel}")

    # --- A4. Équivariance par translation ---
    # On décale l'image de (3, 5) pixels. Convolution puis décalage = décalage puis convolution,
    # sauf près des bords (ce qui sort d'un côté n'entre pas de l'autre).
    _k = NOYAUX["Sobel x (bords verticaux)"]
    _dec = lambda x: np.roll(x, (3, 5), axis=(0, 1))  # noqa: E731
    _a = correlation(_dec(IMG), _k, pad=1)
    _b = _dec(correlation(IMG, _k, pad=1))
    _interieur = (slice(8, -8), slice(8, -8))
    ecart_equivariance = float(np.abs(_a - _b)[_interieur].max())
    print(f"\nconv(décalée) − décalée(conv), à l'intérieur : {ecart_equivariance:.1e} (équivariance)")
    # Max pooling 2×2 : un décalage d'UN pixel change la sortie (invariance seulement locale)
    _r = np.abs(correlation(IMG, _k, pad=1))
    _p1 = max_pooling(_r)
    _p2 = max_pooling(np.roll(_r, 1, axis=1))
    part_pooling_change = float((np.abs(_p1 - _p2) > 1e-9).mean())
    # Maximum global : invariant au décalage
    ecart_global = float(abs(_r.max() - np.roll(_r, (3, 5), axis=(0, 1)).max()))
    print(f"Max pooling 2×2 après un décalage d'un pixel : {part_pooling_change:.0%} des sorties changent")
    print(f"Maximum global après décalage : écart {ecart_global:.1e} (invariance)")

    # --- A5. Nombre de paramètres : couche dense contre couche de convolution ---
    # Entrée 32×32×3, sortie 32×32×32 (même taille, 32 « canaux »)
    params_dense = (32 * 32 * 3) * (32 * 32 * 32) + 32 * 32 * 32
    # Avec biais (nn.Conv2d par défaut). Le CNN de la partie B n'en met pas (la BatchNorm
    # qui suit en tient lieu) : sa première couche a donc 2 400 poids.
    params_conv = 32 * 3 * 5 * 5 + 32
    print(f"\nCouche dense 3 072 → 32 768 : {params_dense:,} paramètres".replace(",", " "))
    print(f"Convolution 5×5, 3 → 32 canaux, avec biais : {params_conv:,} paramètres "
          f"(× {params_dense / params_conv:,.0f} de moins)".replace(",", " "))

    # --- A6. Champ récepteur du CNN de la partie B ---
    # r ← r + (k − 1) × saut ; saut ← saut × pas (le pooling 2×2 double le saut)
    _couches = [("conv 5×5", 5, 1), ("conv 3×3", 3, 1), ("max pool 2×2", 2, 2),
                ("conv 3×3", 3, 1), ("conv 3×3", 3, 1), ("max pool 2×2", 2, 2),
                ("conv 3×3", 3, 1), ("conv 3×3", 3, 1)]
    _r, _saut = 1, 1
    print("\nChamp récepteur (pixels de l'image vus par un neurone) :")
    for _nom, _kk, _s in _couches:
        _r += (_kk - 1) * _saut
        _saut *= _s
        print(f"  après {_nom:<13} : {_r}×{_r}")
    champ_recepteur = _r
    return champ_recepteur, ecart_equivariance, params_conv, params_dense, part_pooling_change


@app.cell
def _(mo):
    mo.md(r"""
    ## B. MLP contre CNN sur CIFAR-10

    CIFAR-10 : 60 000 images 32×32 en couleur, 10 classes (avion, automobile, oiseau, chat,
    cerf, chien, grenouille, cheval, bateau, camion), 50 000 pour l'entraînement et 10 000
    pour le test. Le hasard donne 10 %.

    | Modèle | Architecture | Paramètres |
    |---|---|---|
    | MLP | 3 072 → 512 → 256 → 10 (ReLU) | ≈ 1,7 M |
    | CNN | 6 convolutions (32, 32, 64, 64, 128, 128 canaux) + 2 max pooling + moyenne globale + couche linéaire | ≈ 0,29 M |
    | CNN + augmentation | le même, entraîné sur des images recadrées et retournées au hasard | ≈ 0,29 M |

    Chaque convolution est suivie d'une normalisation par lot (BatchNorm, chapitre 12) et
    d'une ReLU (chapitre 07). Même optimiseur (AdamW, chapitre 02), même nombre d'époques.

    **Augmentation de données** : à chaque époque, chaque image d'entraînement est recadrée
    au hasard (marge de 4 pixels par réflexion, puis découpe 32×32) et retournée
    horizontalement une fois sur deux. Le réseau ne voit jamais deux fois exactement la
    même image : on lui apprend que ces transformations ne changent pas la classe.
    """)


@app.cell
def _(config, np, torch):
    from ia_commun.augmentations import (
        ECART_TYPE_CIFAR10,
        MOYENNE_CIFAR10,
        AugmentationClassique,
        denormaliser_cifar10,
    )
    from ia_commun.donnees import CIFAR10, CLASSES_CIFAR10

    # Sous-ensembles fixes (mode smoke : quelques milliers d'images)
    _rng = np.random.default_rng(42)
    idx_train = np.sort(_rng.permutation(50000)[: config["n_train"]])
    idx_test = np.arange(config["n_test"])

    def charger_tenseurs(ds):
        """Toutes les images normalisées (float32) et les étiquettes, en mémoire.
        Même calcul que normaliser_cifar10 (uint8 → [0, 1] → moyenne et écart type par
        canal), mais sur tout le tableau d'un coup."""
        X = torch.from_numpy(np.array(ds.images[ds.indices])).permute(0, 3, 1, 2).float() / 255
        X = (X - torch.tensor(MOYENNE_CIFAR10).view(1, 3, 1, 1)) / torch.tensor(ECART_TYPE_CIFAR10).view(1, 3, 1, 1)
        y = torch.from_numpy(ds.labels[ds.indices]).long()
        return X.contiguous(), y

    ds_train = CIFAR10("train", indices=idx_train)
    ds_test = CIFAR10("test", indices=idx_test)
    X_train, y_train = charger_tenseurs(ds_train)
    X_test, y_test = charger_tenseurs(ds_test)
    # Le même jeu d'entraînement, mais augmenté image par image dans __getitem__
    ds_train_aug = CIFAR10("train", transform=AugmentationClassique(), indices=idx_train)
    print(f"Entraînement {tuple(X_train.shape)}, test {tuple(X_test.shape)}")
    return (
        CLASSES_CIFAR10,
        X_test,
        X_train,
        denormaliser_cifar10,
        ds_test,
        ds_train_aug,
        y_test,
        y_train,
    )


@app.cell
def _(nn):
    def creer_mlp():
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )

    def bloc(c_in, c_out, k=3):
        # Convolution (sans biais : la BatchNorm qui suit en a un) + BatchNorm + ReLU
        return nn.Sequential(nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False),
                             nn.BatchNorm2d(c_out), nn.ReLU())

    class PetitCNN(nn.Module):
        """6 convolutions, 2 max pooling, moyenne globale, couche linéaire."""

        def __init__(self):
            super().__init__()
            self.extraire = nn.Sequential(
                bloc(3, 32, 5), bloc(32, 32), nn.MaxPool2d(2),      # 32×32 → 16×16
                bloc(32, 64), bloc(64, 64), nn.MaxPool2d(2),        # 16×16 → 8×8
                bloc(64, 128), bloc(128, 128),                      # cartes 128×8×8 (Grad-CAM)
            )
            self.classer = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10))

        def forward(self, x):
            return self.classer(self.extraire(x))

    def creer_cnn():
        return PetitCNN()

    def nb_parametres(m):
        return sum(p.numel() for p in m.parameters())

    print(f"MLP : {nb_parametres(creer_mlp()):,} paramètres ; CNN : {nb_parametres(creer_cnn()):,}"
          .replace(",", " "))
    return creer_cnn, creer_mlp, nb_parametres


@app.cell
def _(mo):
    # Le bouton empêche l'entraînement de se relancer à chaque modification d'une cellule amont.
    bouton = mo.ui.run_button(label="Entraîner (≈ 5 min)")
    bouton
    return (bouton,)


@app.cell
def _(F, X_test, config, device, time, torch):
    def evaluer(modele, X, y, lot=1000):
        """Exactitude (fraction de bonnes réponses) sur des tenseurs en mémoire."""
        modele.eval()
        bons = 0
        with torch.no_grad():
            for i in range(0, len(X), lot):
                xb = X[i: i + lot].to(device)
                bons += (modele(xb).argmax(1).cpu() == y[i: i + lot]).sum().item()
        return bons / len(X)

    def entrainer(modele, lots_par_epoque, iter_lots, nom, X_eval, y_eval, epoques=None):
        """Boucle commune : AdamW, taux « one cycle », entropie croisée.
        `iter_lots()` renvoie un itérable de (x, y) pour une époque."""
        epoques = epoques or config["epoques"]
        modele = modele.to(device)
        optim = torch.optim.AdamW(modele.parameters(), lr=config["lr_max"],
                                  weight_decay=config["weight_decay"])
        sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=config["lr_max"],
                                                    total_steps=epoques * lots_par_epoque)
        hist = {"perte": [], "prec_train": [], "prec_test": [], "duree": []}
        t0 = time.perf_counter()
        for ep in range(epoques):
            modele.train()
            somme, bons, n = 0.0, 0, 0
            for xb, yb in iter_lots():
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                logits = modele(xb)
                perte = F.cross_entropy(logits, yb)
                optim.zero_grad(set_to_none=True)
                perte.backward()
                optim.step()
                sched.step()
                somme += perte.item() * len(yb)
                bons += (logits.argmax(1) == yb).sum().item()
                n += len(yb)
            hist["perte"].append(somme / n)
            hist["prec_train"].append(bons / n)
            hist["prec_test"].append(evaluer(modele, X_eval, y_eval))
            hist["duree"].append(time.perf_counter() - t0)
            print(f"  {nom} époque {ep + 1:2d} : perte {somme / n:.3f}, train {bons / n:.1%}, "
                  f"test {hist['prec_test'][-1]:.1%} ({hist['duree'][-1]:.0f} s)")
        return modele, hist

    _ = X_test  # dépendance explicite : les données sont chargées avant
    return entrainer, evaluer


@app.cell
def _(
    SEED,
    X_test,
    X_train,
    bouton,
    config,
    creer_cnn,
    creer_mlp,
    ds_train_aug,
    entrainer,
    mo,
    time,
    torch,
    y_test,
    y_train,
):
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Entraîner » pour lancer les trois entraînements.*"))

    def lots_memoire(X, y, lot, graine):
        """Mélange à chaque époque des tenseurs déjà en mémoire (pas d'augmentation)."""
        g = torch.Generator().manual_seed(graine)

        def iterer():
            perm = torch.randperm(len(X), generator=g)
            for i in range(0, len(X), lot):
                j = perm[i: i + lot]
                yield X[j], y[j]
        return iterer

    _nb_lots = (len(X_train) + config["lot"] - 1) // config["lot"]
    _t_total = time.perf_counter()
    resultats = {}

    print("MLP, sans augmentation")
    torch.manual_seed(SEED)
    mlp, resultats["MLP"] = entrainer(creer_mlp(), _nb_lots,
                                      lots_memoire(X_train, y_train, config["lot"], SEED),
                                      "MLP", X_test, y_test)

    print("CNN, sans augmentation")
    torch.manual_seed(SEED)
    cnn, resultats["CNN"] = entrainer(creer_cnn(), _nb_lots,
                                      lots_memoire(X_train, y_train, config["lot"], SEED),
                                      "CNN", X_test, y_test)

    print("CNN, avec augmentation (recadrage + retournement, image par image)")
    # Les augmentations tournent dans des processus séparés (workers) pendant que le GPU calcule
    _chargeur = torch.utils.data.DataLoader(
        ds_train_aug, batch_size=config["lot"], shuffle=True, drop_last=False,
        num_workers=config["workers"], persistent_workers=True,
        generator=torch.Generator().manual_seed(SEED))
    torch.manual_seed(SEED)
    cnn_aug, resultats["CNN + augmentation"] = entrainer(creer_cnn(), len(_chargeur),
                                                         lambda: _chargeur, "CNN+aug",
                                                         X_test, y_test)
    duree_entrainement = time.perf_counter() - _t_total
    print(f"\nEntraînements : {duree_entrainement:.0f} s au total")
    return cnn, cnn_aug, duree_entrainement, mlp, resultats


@app.cell
def _(SMOKE, mo, plt, resultats, sauvegarder_figure):
    # --- Courbes : exactitude en entraînement (pointillés) et en test (trait plein) ---
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for _c, (_nom, _h) in zip(["#D55E00", "#0072B2", "#009E73"], resultats.items()):
        _ep = range(1, len(_h["perte"]) + 1)
        _ax1.plot(_ep, _h["perte"], color=_c, label=_nom)
        _ax2.plot(_ep, [100 * v for v in _h["prec_test"]], color=_c, label=f"{_nom} (test)")
        _ax2.plot(_ep, [100 * v for v in _h["prec_train"]], color=_c, ls=":", label=f"{_nom} (entr.)")
    _ax1.set(xlabel="époque", ylabel="perte d'entraînement", title="Entropie croisée")
    _ax2.set(xlabel="époque", ylabel="exactitude (%)", title="Exactitude : test (plein), entraînement (pointillés)")
    _ax1.legend()
    _ax2.legend(fontsize=8, ncol=1)
    _fig.tight_layout()
    sauvegarder_figure(_fig, "courbes-smoke" if SMOKE else "courbes", mo.notebook_dir() / "figures")
    _fig


@app.cell
def _(X_test, cnn, cnn_aug, config, evaluer, mlp, nb_parametres, torch, y_test):
    # --- B2. Robustesse : images retournées, images décalées ---
    _d = config["decalage"]

    def decaler(X, dx):
        """Décale les images de dx pixels vers la droite (marge par réflexion, comme l'augmentation)."""
        Xp = torch.nn.functional.pad(X, (_d, _d, 0, 0), mode="reflect")
        return Xp[..., _d - dx: _d - dx + 32]

    robustesse = {}
    for _nom, _m in [("MLP", mlp), ("CNN", cnn), ("CNN + augmentation", cnn_aug)]:
        robustesse[_nom] = {
            "parametres": nb_parametres(_m),
            "test": evaluer(_m, X_test, y_test),
            "retournees": evaluer(_m, torch.flip(X_test, dims=[3]), y_test),
            "decalees": evaluer(_m, decaler(X_test, _d), y_test),
        }
    print(f"{'modèle':<20} {'param.':>9} {'test':>7} {'retournées':>11} {'décalées ' + str(_d) + ' px':>14}")
    for _nom, _r in robustesse.items():
        print(f"{_nom:<20} {_r['parametres']:>9,} {_r['test']:>7.1%} {_r['retournees']:>11.1%} "
              f"{_r['decalees']:>14.1%}".replace(",", " "))
    return (robustesse,)


@app.cell
def _(mo):
    mo.md(r"""
    ## C. Ce que le CNN a appris

    **Filtres de la première couche.** Les 32 noyaux 5×5×3 du CNN avec augmentation,
    chacun ramené dans [0, 1] pour l'affichage. On y cherche des détecteurs de bords
    orientés et des taches de couleur, comme dans AlexNet.

    **Grad-CAM** (Selvaraju et al., 2017). Pour une image et une classe $c$, on prend les
    $K = 128$ cartes $A^k$ (8×8) de la dernière convolution et le gradient du score
    $y^c$ par rapport à chacune. Le poids d'une carte est la moyenne de son gradient,
    $\alpha_k = \frac{1}{64}\sum_{i,j} \partial y^c / \partial A^k_{ij}$, et la carte de
    chaleur vaut $\mathrm{ReLU}\big(\sum_k \alpha_k A^k\big)$, agrandie à 32×32. Elle
    montre les régions qui poussent le score de la classe vers le haut.
    """)


@app.cell
def _(F, X_test, cnn_aug, config, device, torch, y_test):
    def grad_cam(modele, x, classe=None):
        """Grad-CAM pour un lot x (B, 3, 32, 32). Classe = prédite si None.
        Renvoie (cartes B×32×32 dans [0, 1], classes utilisées, probabilités)."""
        modele.eval()
        A = modele.extraire(x)                 # (B, 128, 8, 8)
        A.retain_grad()
        logits = modele.classer(A)
        c = logits.argmax(1) if classe is None else classe
        logits.gather(1, c[:, None]).sum().backward()
        alpha = A.grad.mean(dim=(2, 3), keepdim=True)          # (B, 128, 1, 1)
        cam = F.relu((alpha * A).sum(1, keepdim=True))         # (B, 1, 8, 8)
        cam = F.interpolate(cam, size=32, mode="bilinear", align_corners=False)[:, 0]
        cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        return cam.detach().cpu(), c.cpu(), logits.softmax(1).detach().cpu()

    # Les premières images de test bien classées, une par classe autant que possible
    with torch.no_grad():
        _pred = cnn_aug.eval()(X_test[:500].to(device)).argmax(1).cpu()
    _choisis, _vues = [], set()
    for _i in range(500):
        if _pred[_i] == y_test[_i] and int(y_test[_i]) not in _vues:
            _choisis.append(_i)
            _vues.add(int(y_test[_i]))
        if len(_choisis) == config["n_gradcam"]:
            break
    idx_gradcam = _choisis
    cartes, classes_gradcam, probas_gradcam = grad_cam(cnn_aug, X_test[idx_gradcam].to(device))
    print("Images retenues :", idx_gradcam)
    return cartes, classes_gradcam, idx_gradcam, probas_gradcam


@app.cell
def _(
    CLASSES_CIFAR10,
    SMOKE,
    X_test,
    cartes,
    classes_gradcam,
    cnn_aug,
    denormaliser_cifar10,
    idx_gradcam,
    mo,
    np,
    plt,
    probas_gradcam,
    sauvegarder_figure,
):
    # --- Filtres de la première couche ---
    filtres = cnn_aug.extraire[0][0].weight.detach().cpu().numpy()      # (32, 3, 5, 5)
    _fig, _axes = plt.subplots(4, 8, figsize=(8, 4.3))
    for _ax, _w in zip(_axes.flat, filtres):
        _w = _w.transpose(1, 2, 0)
        _ax.imshow((_w - _w.min()) / (_w.max() - _w.min() + 1e-8), interpolation="nearest")
        _ax.axis("off")
    _fig.suptitle("32 filtres 5×5×3 de la première couche (CNN + augmentation)")
    sauvegarder_figure(_fig, "filtres-appris-smoke" if SMOKE else "filtres-appris",
                       mo.notebook_dir() / "figures")

    # --- Grad-CAM ---
    images_gradcam = np.stack([denormaliser_cifar10(X_test[i]).permute(1, 2, 0).numpy()
                               for i in idx_gradcam])
    _n = len(idx_gradcam)
    _fig2, _axes2 = plt.subplots(2, _n, figsize=(1.6 * _n, 3.6), squeeze=False)
    for _k in range(_n):
        _c = int(classes_gradcam[_k])
        _axes2[0, _k].imshow(images_gradcam[_k], interpolation="nearest")
        _axes2[0, _k].set_title(f"{CLASSES_CIFAR10[_c]}\n{probas_gradcam[_k, _c]:.0%}", fontsize=9)
        _axes2[1, _k].imshow(images_gradcam[_k], interpolation="nearest")
        _axes2[1, _k].imshow(cartes[_k], cmap="inferno", alpha=0.55, vmin=0, vmax=1)
        for _a in (_axes2[0, _k], _axes2[1, _k]):
            _a.axis("off")
    _fig2.suptitle("Grad-CAM : régions qui font monter le score de la classe prédite")
    _fig2.tight_layout(rect=(0, 0, 1, 0.9))   # place pour le titre au-dessus des sous-titres
    sauvegarder_figure(_fig2, "gradcam-smoke" if SMOKE else "gradcam", mo.notebook_dir() / "figures")
    mo.vstack([_fig, _fig2])
    return filtres, images_gradcam


@app.cell
def _(
    SEED,
    X_test,
    config,
    ds_train_aug,
    entrainer,
    mo,
    torch,
    torchvision_resnet18,
    y_test,
):
    # --- Option : ResNet-18 adapté à CIFAR-10 (IA_RESNET18=1) ---
    resultats_resnet = None
    if config["resnet18"]:
        _m = torchvision_resnet18(num_classes=10)
        # Images 32×32 : première convolution 3×3 de pas 1, pas de max pooling initial
        _m.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1, bias=False)
        _m.maxpool = torch.nn.Identity()
        _chargeur = torch.utils.data.DataLoader(
            ds_train_aug, batch_size=256, shuffle=True, num_workers=config["workers"],
            persistent_workers=True, generator=torch.Generator().manual_seed(SEED))
        torch.manual_seed(SEED)
        print(f"ResNet-18 : {sum(p.numel() for p in _m.parameters()):,} paramètres".replace(",", " "))
        _, resultats_resnet = entrainer(_m, len(_chargeur), lambda: _chargeur, "ResNet-18",
                                        X_test, y_test, epoques=config["epoques_resnet"])
    mo.md("Option ResNet-18 : " + ("lancée." if config["resnet18"] else
                                   "non lancée (relancer avec `IA_RESNET18=1`)."))
    return (resultats_resnet,)


@app.cell
def _():
    from torchvision.models import resnet18 as torchvision_resnet18

    return (torchvision_resnet18,)


@app.cell
def _(
    SMOKE,
    cartes,
    champ_recepteur,
    classes_gradcam,
    duree_entrainement,
    ecart_boucles,
    ecart_convolve,
    ecart_equivariance,
    ecart_torch,
    filtres,
    idx_gradcam,
    images_gradcam,
    json,
    mo,
    np,
    params_conv,
    params_dense,
    part_pooling_change,
    probas_gradcam,
    resultats,
    resultats_resnet,
    robustesse,
    valeur_sobel_pixel,
    y_test,
):
    # --- Bilan : seuils du README (vérifiés en mode complet seulement) ---
    _r = robustesse
    bilan = {
        "duree_entrainement_s": round(duree_entrainement, 1),
        "ecart_torch": ecart_torch, "ecart_boucles": ecart_boucles, "ecart_convolve": ecart_convolve,
        "ecart_equivariance": ecart_equivariance, "part_pooling_change": part_pooling_change,
        "valeur_sobel_pixel": valeur_sobel_pixel, "champ_recepteur": champ_recepteur,
        "params_dense": params_dense, "params_conv": params_conv,
        "robustesse": _r, "courbes": resultats, "resnet18": resultats_resnet,
    }
    print(json.dumps({k: v for k, v in bilan.items() if k not in ("courbes",)}, indent=1,
                     ensure_ascii=False, default=float))
    # Résultats bruts pour les figures du site (site/scripts/figures/ch13_*.py)
    _dossier = mo.notebook_dir() / "figures"
    np.savez_compressed(
        _dossier / ("resultats-smoke.npz" if SMOKE else "resultats.npz"),
        bilan=json.dumps(bilan, ensure_ascii=False),
        filtres=filtres.astype(np.float32),
        images=(images_gradcam * 255).round().astype(np.uint8),
        cartes=cartes.numpy().astype(np.float16),
        classes=classes_gradcam.numpy(), vraies=y_test[idx_gradcam].numpy(),
        probas=probas_gradcam.numpy().astype(np.float32),
    )

    SEUILS = [
        ("convolution NumPy = torch.conv2d (écart < 1e-12)", ecart_torch < 1e-12),
        ("équivariance par translation (écart < 1e-12)", ecart_equivariance < 1e-12),
        ("MLP : exactitude test ≥ 50 %", _r["MLP"]["test"] >= 0.50),
        ("CNN : exactitude test ≥ 80 %", _r["CNN"]["test"] >= 0.80),
        ("CNN + augmentation : exactitude test ≥ 86 %", _r["CNN + augmentation"]["test"] >= 0.86),
        ("l'augmentation gagne au moins 2 points", _r["CNN + augmentation"]["test"] - _r["CNN"]["test"] >= 0.02),
        ("MLP : perd au moins 7 points sur les images décalées",
         _r["MLP"]["test"] - _r["MLP"]["decalees"] >= 0.07),
        ("CNN + augmentation : perd moins de 2 points sur les images retournées",
         _r["CNN + augmentation"]["test"] - _r["CNN + augmentation"]["retournees"] < 0.02),
    ]
    _echecs = [nom for nom, ok in SEUILS if not ok]
    for _nom, _ok in SEUILS:
        print(("OK    " if _ok else "ÉCHEC ") + _nom)
    if not SMOKE and _echecs:
        raise SystemExit("Seuils non atteints : " + " ; ".join(_echecs))
    mo.md(f"**Bilan** : MLP {_r['MLP']['test']:.1%}, CNN {_r['CNN']['test']:.1%}, "
          f"CNN + augmentation {_r['CNN + augmentation']['test']:.1%} en test ; "
          f"{duree_entrainement:.0f} s d'entraînement.")


if __name__ == "__main__":
    app.run()
