# Exercice 26 : reconstruire les pixels (MAE) ou prédire la représentation (JEPA) ?
#
#   make ex NUM=26              → éditeur marimo (bouton « Entraîner »)
#   make ex NUM=26 MODE=run     → script complet, vérifie les seuils du README
#   make ex NUM=26 MODE=smoke   → test rapide (< 30 s)
#
# Partie A en NumPy (pertes Barlow Twins et VICReg calculées à la main sur trois nuages de
# vecteurs), parties B à E en PyTorch sur MNIST (GPU Apple via MPS si disponible) : un petit
# MAE, une mini-JEPA à encodeur cible EMA, puis la même JEPA sans EMA ni arrêt du gradient
# (elle s'effondre) et avec un régulariseur de type VICReg (elle ne s'effondre plus).

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import math
    import os
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from ia_commun import get_device, is_smoke, sauvegarder_figure, seed_everything, style_commun
    from ia_commun.donnees import mnist_tenseurs

    return (
        F,
        get_device,
        is_smoke,
        json,
        math,
        os,
        mnist_tenseurs,
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
    # Exercice 26 : reconstruire les pixels ou prédire la représentation ?

    Deux façons d'apprendre sans étiquette à partir d'une image dont on cache une partie :

    - **MAE** (autoencodeur masqué, He et al., 2021) : on cache 75 % des patchs, l'encodeur ne
      voit que le reste, un décodeur **reconstruit les pixels** cachés. Perte : erreur
      quadratique moyenne (MSE) sur les pixels des patchs masqués.
    - **JEPA** (architecture prédictive à plongement joint, ici dans l'esprit d'I-JEPA,
      Assran et al., 2023) : un encodeur de **contexte** voit une partie de l'image, un
      **prédicteur** devine la **représentation** que l'encodeur **cible** donne des blocs
      cachés. Perte : MSE entre vecteurs latents, jamais entre pixels. L'encodeur cible est
      une moyenne mobile exponentielle (EMA) de l'encodeur de contexte, et le gradient ne le
      traverse pas.

    Le piège de la JEPA : si la cible et la prédiction peuvent bouger ensemble, la solution la
    plus simple est de tout envoyer sur la même constante (perte nulle, représentation
    inutile). C'est l'**effondrement**. On le provoque exprès, puis on l'évite par une autre
    voie, un régulariseur de variance et de covariance (VICReg).

    - **A. NumPy, sans apprentissage.** Les pertes Barlow Twins et VICReg, calculées à la main
      sur un nuage sain, un nuage effondré et un nuage redondant.
    - **B. Données et briques.** MNIST découpé en 4 × 4 patchs de 7 × 7 pixels ; un petit
      Transformer (chapitre 20) sert d'encodeur.
    - **C. MAE.** Masquage aléatoire de 75 %, reconstruction des pixels.
    - **D. JEPA.** Masquage par blocs, prédiction en espace latent ; trois variantes : EMA +
      arrêt du gradient (la recette d'I-JEPA), ni l'un ni l'autre (effondrement), ni l'un ni
      l'autre mais avec un régulariseur VICReg.
    - **E. Ce que valent les représentations.** Sondes linéaires (toutes les étiquettes, puis
      100 par classe) et k plus proches voisins, comparées aux pixels bruts et à un encodeur
      au hasard ; indicateur d'effondrement.

    **Pertes optimisées.** MAE : $\frac{1}{|M|}\sum_{i \in M}\lVert x_i - \hat x_i\rVert^2$ sur
    les patchs masqués $M$. JEPA : $\frac{1}{|T|}\sum_{j \in T}\lVert s_j - \hat s_j\rVert^2$,
    $s_j$ = représentation cible (normalisée) du patch $j$, $\hat s_j$ = prédiction. Variante
    régularisée : la même, plus les termes de variance et de covariance de VICReg. Sondes :
    entropie croisée. Partie A : rien n'est appris, on évalue des pertes.
    """)


@app.cell
def _(get_device, is_smoke, json, os, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    device = get_device()
    style_commun()
    config = dict(
        # B. données : sous-ensemble d'entraînement, test complet
        n_train=1024 if SMOKE else 20000,
        n_test=1000 if SMOKE else 10000,
        patch=7,                # 28 / 7 = 4 → grille 4 × 4 = 16 patchs de 49 pixels
        # Encodeur (identique pour MAE et JEPA) : petit Transformer
        dim=128, profondeur=4, tetes=4, mlp=256,
        # Décodeur MAE et prédicteur JEPA : plus étroits
        dim_tete=64, profondeur_tete=2,
        # Entraînement
        epoques=1 if SMOKE else 15, lot=256, lr=1e-3, wd=0.05, echauffement=0.1,
        # C. MAE
        ratio_masque=0.75,
        # D. JEPA : 3 blocs cibles de 12 à 25 % de l'image chacun, contexte = le reste
        # (I-JEPA : 4 blocs de 15 à 20 % sur une grille de 14 × 14 ou plus ; notre grille de
        # 4 × 4 impose des blocs d'au moins un patch, soit 6 % de l'image)
        n_blocs=3, echelle_bloc=(0.12, 0.25), ratio_bloc=(0.75, 1.5),
        ema=(0.996, 1.0),       # moment EMA, de 0,996 à 1 (calendrier d'I-JEPA)
        vicreg_var=1.0, vicreg_cov=0.04,
        # E. évaluation
        n_peu=100,              # étiquettes par classe pour la sonde « peu d'étiquettes »
        k_voisins=20, n_indic=1024,
        pos_echelle=0.5,
    )
    # Pour expérimenter sans modifier le fichier : IA26_CONFIG='{"epoques": 20}' make ex NUM=26 MODE=run
    config.update(json.loads(os.environ.get("IA26_CONFIG", "{}")))
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, périphérique {device}")
    print(config)
    return SEED, SMOKE, config, device


@app.cell
def _(mo):
    mo.md(r"""
    ## A. NumPy : Barlow Twins et VICReg à la main

    On fabrique trois lots de $N = 512$ paires de vues, plongées en $d = 8$ dimensions :

    1. **sain** : chaque image a un vecteur aléatoire, ses deux vues en sont deux copies
       légèrement bruitées ;
    2. **effondré** : toutes les images ont le même vecteur (à un bruit infime près) ;
    3. **redondant** : les huit dimensions ne sont que deux facteurs recopiés quatre fois.

    Pour chacun, on calcule l'**alignement seul** (MSE entre les deux vues : ce qu'une JEPA
    naïve minimise), la perte **Barlow Twins**
    $\sum_i (1 - C_{ii})^2 + \lambda \sum_i \sum_{j \neq i} C_{ij}^2$ avec $C$ la matrice de
    corrélation croisée entre les deux vues, et les trois termes de **VICReg** (invariance,
    variance, covariance).
    """)


@app.cell
def _(np):
    # --- A. Trois lots de vues, puis les pertes calculées à la main ---
    def correlation_croisee(za, zb):
        """C[i, j] = corrélation entre la dimension i de la vue A et la dimension j de la vue B,
        calculée sur le lot (chaque dimension est centrée et réduite)."""
        za = (za - za.mean(0)) / (za.std(0) + 1e-9)
        zb = (zb - zb.mean(0)) / (zb.std(0) + 1e-9)
        return za.T @ zb / len(za)

    def perte_barlow(za, zb, lam=0.005):
        """Barlow Twins (Zbontar et al., 2021) : diagonale de C vers 1 (invariance),
        hors diagonale vers 0 (réduction de redondance). λ = 0,005 comme dans l'article."""
        c = correlation_croisee(za, zb)
        diag = np.diag(c)
        hors = c - np.diag(diag)
        return float(((1 - diag) ** 2).sum()), float(lam * (hors**2).sum()), c

    def termes_vicreg(za, zb, gamma=1.0, eps=1e-4):
        """VICReg (Bardes, Ponce et LeCun, 2021) : invariance (MSE entre vues), variance
        (charnière : chaque dimension doit garder un écart-type ≥ γ), covariance (somme des
        covariances hors diagonale au carré, divisée par d). Variance et covariance sont ici
        MOYENNÉES sur les deux vues ; l'article les additionne (μ[v(Z) + v(Z')]), ce qui les doublerait."""
        d = za.shape[1]
        inv = float(((za - zb) ** 2).mean())
        var, cov = 0.0, 0.0
        for z in (za, zb):
            ecart = np.sqrt(z.var(0) + eps)
            var += float(np.maximum(0.0, gamma - ecart).mean()) / 2
            zc = z - z.mean(0)
            m = zc.T @ zc / (len(z) - 1)
            cov += float(((m - np.diag(np.diag(m))) ** 2).sum() / d) / 2
        return inv, var, cov

    _rng = np.random.default_rng(42)
    _n, _d = 512, 8
    _base = _rng.normal(size=(_n, _d))
    _facteurs = _rng.normal(size=(_n, 2))
    lots_a = {
        "sain": _base,
        "effondré": np.ones((_n, _d)) * 0.5,
        "redondant": np.repeat(_facteurs, 4, axis=1),
    }
    resultats_a = {}
    matrices_c = {}
    for _nom, _z in lots_a.items():
        # Deux vues = deux petits bruits indépendants (comme deux augmentations d'une même image)
        _za = _z + 0.1 * _rng.normal(size=_z.shape) * (0.01 if _nom == "effondré" else 1)
        _zb = _z + 0.1 * _rng.normal(size=_z.shape) * (0.01 if _nom == "effondré" else 1)
        _diag, _hors, _c = perte_barlow(_za, _zb)
        _inv, _var, _cov = termes_vicreg(_za, _zb)
        resultats_a[_nom] = dict(alignement=float(((_za - _zb) ** 2).mean()),
                                 barlow_diag=_diag, barlow_hors=_hors,
                                 barlow=_diag + _hors,
                                 vicreg_inv=_inv, vicreg_var=_var, vicreg_cov=_cov,
                                 vicreg=25 * _inv + 25 * _var + _cov)
        matrices_c[_nom] = _c
    print(f"{'lot':<10} {'alignement':>10} {'Barlow diag':>11} {'hors diag':>9} "
          f"{'VIC inv':>8} {'var':>6} {'cov':>6} {'VICReg':>7}")
    for _nom, _r in resultats_a.items():
        print(f"{_nom:<10} {_r['alignement']:>10.4f} {_r['barlow_diag']:>11.3f} {_r['barlow_hors']:>9.3f} "
              f"{_r['vicreg_inv']:>8.4f} {_r['vicreg_var']:>6.3f} {_r['vicreg_cov']:>6.3f} {_r['vicreg']:>7.2f}")
    return matrices_c, resultats_a


@app.cell
def _(mo, resultats_a):
    _s, _e, _r = resultats_a["sain"], resultats_a["effondré"], resultats_a["redondant"]
    mo.md(rf"""
    **Lecture.** L'alignement seul est **minimal pour le lot effondré**
    ({_e['alignement']:.5f} contre {_s['alignement']:.4f} pour le lot sain) : une perte qui ne
    regarde que l'accord entre les deux vues préfère l'effondrement. Barlow Twins le pénalise
    par sa diagonale (il ne reste qu'un bruit infime, indépendant d'une vue à l'autre : les
    corrélations valent environ 0, terme {_e['barlow_diag']:.1f}, soit presque $d = 8$) et VICReg par son terme de variance ({_e['vicreg_var']:.2f}). Le lot redondant
    passe l'alignement et la variance, mais pas la **covariance** : Barlow hors diagonale
    {_r['barlow_hors']:.3f} contre {_s['barlow_hors']:.4f}, VICReg covariance
    {_r['vicreg_cov']:.2f} contre {_s['vicreg_cov']:.3f}. C'est l'**effondrement dimensionnel** :
    huit dimensions qui n'en valent que deux.
    """)


@app.cell
def _(mo):
    mo.md(r"""
    ## B. Données et briques communes

    Chaque image 28 × 28 devient une suite de 16 patchs de 7 × 7 = 49 pixels, comme un ViT
    (chapitre 24). L'encodeur est un petit Transformer : projection linéaire de chaque patch,
    position ajoutée (sinus et cosinus en 2D, fixes), quatre blocs d'attention. Il est
    **le même** pour le MAE et pour la JEPA : seule la tâche change.
    """)


@app.cell
def _(config, mnist_tenseurs, torch):
    # --- B1. MNIST : 20 000 images d'entraînement tirées au hasard (graine 42), test complet ---
    _x_tr, _y_tr = mnist_tenseurs(train=True)
    _x_te, _y_te = mnist_tenseurs(train=False)
    _g = torch.Generator().manual_seed(42)
    _idx = torch.randperm(len(_x_tr), generator=_g)[: config["n_train"]]
    x_train, y_train = _x_tr[_idx], _y_tr[_idx]
    x_test, y_test = _x_te[: config["n_test"]], _y_te[: config["n_test"]]
    print(f"Entraînement : {tuple(x_train.shape)}, test : {tuple(x_test.shape)}")
    return x_test, x_train, y_test, y_train


@app.cell
def _(F, config, math, nn, np, torch):
    # --- B2. Briques : découpage en patchs, positions 2D, encodeur Transformer ---
    P = config["patch"]
    GRILLE = 28 // P            # 7
    N_PATCHS = GRILLE * GRILLE  # 16

    def en_patchs(x):
        """(B, 1, 28, 28) → (B, 16, 49), patchs lus ligne par ligne."""
        b = x.shape[0]
        x = x.reshape(b, GRILLE, P, GRILLE, P).permute(0, 1, 3, 2, 4)
        return x.reshape(b, N_PATCHS, P * P)

    def depuis_patchs(p):
        """(B, 16, 49) → (B, 28, 28)."""
        b = p.shape[0]
        p = p.reshape(b, GRILLE, GRILLE, P, P).permute(0, 1, 3, 2, 4)
        return p.reshape(b, 28, 28)

    def positions_2d(dim):
        """Plongement de position sinus/cosinus 2D fixe (MAE, I-JEPA) : (16, dim)."""
        quart = dim // 4
        freq = 1.0 / (10000 ** (np.arange(quart) / quart))
        lignes, colonnes = np.meshgrid(np.arange(GRILLE), np.arange(GRILLE), indexing="ij")
        morceaux = []
        for coord in (lignes.ravel(), colonnes.ravel()):
            angle = coord[:, None] * freq[None, :]
            morceaux += [np.sin(angle), np.cos(angle)]
        return torch.from_numpy(np.concatenate(morceaux, axis=1).astype(np.float32))

    def pile_transformer(dim, profondeur, tetes, mlp):
        couche = nn.TransformerEncoderLayer(dim, tetes, mlp, dropout=0.0, activation="gelu",
                                            batch_first=True, norm_first=True)
        return nn.TransformerEncoder(couche, profondeur, enable_nested_tensor=False)

    def selection(t, idx):
        """t : (B, N, D), idx : (B, k) entiers → (B, k, D)."""
        return torch.gather(t, 1, idx.unsqueeze(-1).expand(-1, -1, t.shape[-1]))

    class Encodeur(nn.Module):
        """Patchs → vecteurs. `idx` : indices des patchs visibles (None = tous)."""

        def __init__(self):
            super().__init__()
            d = config["dim"]
            self.projection = nn.Linear(P * P, d)
            self.register_buffer("pos", positions_2d(d))
            self.blocs = pile_transformer(d, config["profondeur"], config["tetes"], config["mlp"])
            self.norme = nn.LayerNorm(d)

        def forward(self, patchs, idx=None):
            # Pixels centrés-réduits (moyenne et écart-type de MNIST) : sans cela, le contenu du
            # patch pèse peu face au plongement de position, et la cible JEPA ne code presque que
            # la position (solution triviale).
            h = self.projection((patchs - 0.1307) / 0.3081) + config["pos_echelle"] * self.pos
            if idx is not None:
                h = selection(h, idx)
            return self.norme(self.blocs(h))

    class TeteMasquee(nn.Module):
        """Décodeur MAE ou prédicteur JEPA : même forme. Reçoit les vecteurs du contexte et des
        « jetons de masque » (un vecteur appris + la position à deviner), renvoie une sortie
        par position à deviner : 49 pixels (MAE) ou un vecteur latent de dimension `dim` (JEPA)."""

        def __init__(self, sortie):
            super().__init__()
            d, dt = config["dim"], config["dim_tete"]
            self.entree = nn.Linear(d, dt)
            self.jeton_masque = nn.Parameter(torch.zeros(1, 1, dt))
            nn.init.normal_(self.jeton_masque, std=0.02)
            self.register_buffer("pos", positions_2d(dt))
            self.blocs = pile_transformer(dt, config["profondeur_tete"], config["tetes"], 2 * dt)
            self.norme = nn.LayerNorm(dt)
            self.sortie = nn.Linear(dt, sortie)

        def forward(self, h_contexte, idx_contexte, idx_cible):
            b = h_contexte.shape[0]
            pos = self.pos.unsqueeze(0).expand(b, -1, -1)
            ctx = self.entree(h_contexte) + selection(pos, idx_contexte)
            cib = self.jeton_masque.expand(b, idx_cible.shape[1], -1) + selection(pos, idx_cible)
            z = self.norme(self.blocs(torch.cat([ctx, cib], dim=1)))
            return self.sortie(z[:, ctx.shape[1]:])

    def calendrier_lr(pas, total):
        """Échauffement linéaire puis décroissance en cosinus (facteur multiplicatif)."""
        ech = max(1, int(config["echauffement"] * total))
        if pas < ech:
            return (pas + 1) / ech
        return 0.5 * (1 + math.cos(math.pi * (pas - ech) / max(1, total - ech)))

    def caracteristiques(encodeur, x, device, lot=1000, mode="moyenne"):
        """Représentation d'une image entière vue par l'encodeur gelé.
        « moyenne » : moyenne des vecteurs de patchs (protocole de MAE et d'I-JEPA) : la
        disposition spatiale est perdue, le vecteur doit coder la forme lui-même.
        « concat » : les vecteurs de patchs mis bout à bout (garde la disposition spatiale)."""
        encodeur.eval()
        sorties = []
        with torch.no_grad():
            for i in range(0, len(x), lot):
                h = encodeur(en_patchs(x[i:i + lot].to(device)))
                sorties.append((h.flatten(1) if mode == "concat" else h.mean(1)).cpu())
        return torch.cat(sorties)

    def indicateur_effondrement(f):
        """Dispersion : écart-type d'une image à l'autre, moyenné sur les dimensions. 0 = toutes
        les images ont la même représentation (effondrement). Les sorties de l'encodeur passent
        par une LayerNorm : l'échelle est comparable d'un modèle à l'autre."""
        return float(f.std(0).mean())

    def rang_effectif(f):
        """exp(entropie des valeurs singulières normalisées) des caractéristiques centrées :
        nombre de directions réellement utilisées (Roy et Vetterli, 2007)."""
        s = torch.linalg.svdvals(f - f.mean(0))
        p = s / s.sum()
        return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))

    print(f"{N_PATCHS} patchs de {P}×{P} pixels par image")
    return (
        Encodeur,
        GRILLE,
        N_PATCHS,
        P,
        TeteMasquee,
        calendrier_lr,
        caracteristiques,
        depuis_patchs,
        en_patchs,
        indicateur_effondrement,
        rang_effectif,
        selection,
    )


@app.cell
def _(GRILLE, N_PATCHS, config, np, torch):
    # --- B3. Les deux masquages ---
    def masque_aleatoire(b, generateur):
        """MAE : pour chaque image, 75 % des patchs tirés au hasard sont cachés."""
        n_garde = int(round(N_PATCHS * (1 - config["ratio_masque"])))
        bruit = torch.rand(b, N_PATCHS, generator=generateur)
        ordre = bruit.argsort(dim=1)
        return ordre[:, :n_garde], ordre[:, n_garde:]          # (visibles, cachés)

    def masque_blocs(rng):
        """JEPA (dans l'esprit d'I-JEPA) : 3 blocs rectangulaires cibles de 12 à 25 % de
        l'image ; le contexte est tout le reste. Un même masque pour tout le lot (simplification :
        I-JEPA tire un masque par image et prédit chaque bloc séparément)."""
        cible = np.zeros((GRILLE, GRILLE), dtype=bool)
        for _ in range(config["n_blocs"]):
            aire = rng.uniform(*config["echelle_bloc"]) * N_PATCHS
            ratio = np.exp(rng.uniform(np.log(config["ratio_bloc"][0]), np.log(config["ratio_bloc"][1])))
            h = int(np.clip(round(np.sqrt(aire * ratio)), 1, GRILLE - 1))
            w = int(np.clip(round(np.sqrt(aire / ratio)), 1, GRILLE - 1))
            haut, gauche = rng.integers(0, GRILLE - h + 1), rng.integers(0, GRILLE - w + 1)
            cible[haut:haut + h, gauche:gauche + w] = True
        idx_cible = np.flatnonzero(cible.ravel())
        idx_contexte = np.flatnonzero(~cible.ravel())
        if len(idx_contexte) < 8:            # garder au moins 8 patchs de contexte
            return masque_blocs(rng)
        return torch.from_numpy(idx_contexte), torch.from_numpy(idx_cible)

    _rng = np.random.default_rng(42)
    _c, _t = masque_blocs(_rng)
    print(f"Exemple de masque par blocs : {len(_t)} patchs cibles, {len(_c)} de contexte")
    return masque_aleatoire, masque_blocs


@app.cell
def _(mo):
    # Le bouton empêche l'entraînement de se relancer à chaque modification d'une cellule amont.
    bouton = mo.ui.run_button(label="Entraîner")
    bouton
    return (bouton,)


@app.cell
def _(mo):
    mo.md(r"""
    ## C. MAE : reconstruire les pixels cachés
    """)


@app.cell
def _(
    Encodeur,
    F,
    N_PATCHS,
    TeteMasquee,
    bouton,
    calendrier_lr,
    caracteristiques,
    config,
    device,
    en_patchs,
    indicateur_effondrement,
    masque_aleatoire,
    mo,
    selection,
    time,
    torch,
    x_test,
    x_train,
):
    # En script (make ex MODE=run|smoke), il n'y a personne pour cliquer : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Entraîner » pour lancer les quatre entraînements.*"))

    def entrainer_mae():
        torch.manual_seed(42)
        enc = Encodeur().to(device)
        dec = TeteMasquee(sortie=config["patch"] ** 2).to(device)
        params = list(enc.parameters()) + list(dec.parameters())
        opt = torch.optim.AdamW(params, lr=config["lr"], weight_decay=config["wd"])
        n_lots = len(x_train) // config["lot"]
        total = n_lots * config["epoques"]
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: calendrier_lr(s, total))
        gen = torch.Generator().manual_seed(42)
        hist = dict(perte=[], indic=[indicateur_effondrement(
            caracteristiques(enc, x_test[: config["n_indic"]], device))])
        for _ep in range(config["epoques"]):
            enc.train(); dec.train()
            ordre = torch.randperm(len(x_train), generator=gen)
            for i in range(n_lots):
                patchs = en_patchs(x_train[ordre[i * config["lot"]:(i + 1) * config["lot"]]].to(device))
                vis, cach = masque_aleatoire(patchs.shape[0], gen)
                vis, cach = vis.to(device), cach.to(device)
                pred = dec(enc(patchs, vis), vis, cach)                 # pixels prédits des patchs cachés
                perte = F.mse_loss(pred, selection(patchs, cach))      # MSE sur les seuls patchs cachés
                opt.zero_grad(); perte.backward(); opt.step(); sch.step()
                hist["perte"].append(perte.item())
            hist["indic"].append(indicateur_effondrement(
                caracteristiques(enc, x_test[: config["n_indic"]], device)))
            print(f"  MAE époque {_ep + 1} : perte {sum(hist['perte'][-n_lots:]) / n_lots:.4f}, "
                  f"indicateur {hist['indic'][-1]:.2f}")
        return enc, dec, hist

    _t0 = time.perf_counter()
    mae_enc, mae_dec, hist_mae = entrainer_mae()
    duree_mae = time.perf_counter() - _t0
    print(f"MAE : {duree_mae:.1f} s, {sum(p.numel() for p in mae_enc.parameters()):,} paramètres "
          f"d'encodeur ; référence : prédire 0 partout donne une MSE de "
          f"{float((en_patchs(x_train[:2000]) ** 2).mean()):.4f}, prédire la moyenne "
          f"{float(en_patchs(x_train[:2000]).var()):.4f} ({N_PATCHS} patchs)")
    return duree_mae, hist_mae, mae_dec, mae_enc


@app.cell
def _(mo):
    mo.md(r"""
    ## D. JEPA : prédire la représentation des blocs cachés

    Trois variantes, même encodeur, même prédicteur, même masquage :

    | Variante | Cible | Ce qui empêche l'effondrement |
    |---|---|---|
    | **JEPA (EMA)** | encodeur cible = moyenne mobile de l'encodeur de contexte, sans gradient | l'asymétrie (EMA + arrêt du gradient + prédicteur), comme BYOL et I-JEPA |
    | **JEPA sans EMA** | le même encodeur, et le gradient passe aussi par la cible | rien |
    | **JEPA + VICReg** | le même encodeur, gradient partout | termes de variance et de covariance sur les vecteurs de l'encodeur |
    """)


@app.cell
def _(
    Encodeur,
    F,
    TeteMasquee,
    bouton,
    calendrier_lr,
    caracteristiques,
    config,
    copy_module,
    device,
    en_patchs,
    indicateur_effondrement,
    masque_blocs,
    mo,
    np,
    selection,
    time,
    torch,
    x_test,
    x_train,
):
    mo.stop(mo.running_in_notebook() and not bouton.value, None)

    def penalite_vicreg(h):
        """Variance (charnière à 1) et covariance de VICReg sur des vecteurs (n, d)."""
        h = h.reshape(-1, h.shape[-1])
        ecart = torch.sqrt(h.var(0) + 1e-4)
        var = F.relu(1 - ecart).mean()
        hc = h - h.mean(0)
        cov = hc.T @ hc / (len(h) - 1)
        hors = cov - torch.diag(torch.diag(cov))
        return config["vicreg_var"] * var + config["vicreg_cov"] * (hors**2).sum() / h.shape[-1]

    def entrainer_jepa(variante):
        """variante : 'ema' (I-JEPA), 'sans_ema' (effondrement), 'vicreg' (régularisée)."""
        torch.manual_seed(42)
        rng = np.random.default_rng(42)
        enc = Encodeur().to(device)
        pred = TeteMasquee(sortie=config["dim"]).to(device)
        cible = None
        if variante == "ema":
            cible = copy_module(enc)                # copie, mise à jour par EMA seulement
            for p in cible.parameters():
                p.requires_grad_(False)
        params = list(enc.parameters()) + list(pred.parameters())
        opt = torch.optim.AdamW(params, lr=config["lr"], weight_decay=config["wd"])
        n_lots = len(x_train) // config["lot"]
        total = n_lots * config["epoques"]
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: calendrier_lr(s, total))
        gen = torch.Generator().manual_seed(42)
        a_evaluer = cible if cible is not None else enc     # I-JEPA évalue l'encodeur cible
        hist = dict(perte=[], indic=[indicateur_effondrement(
            caracteristiques(a_evaluer, x_test[: config["n_indic"]], device))])
        pas = 0
        for _ep in range(config["epoques"]):
            enc.train(); pred.train()
            ordre = torch.randperm(len(x_train), generator=gen)
            for i in range(n_lots):
                patchs = en_patchs(x_train[ordre[i * config["lot"]:(i + 1) * config["lot"]]].to(device))
                b = patchs.shape[0]
                ctx, cib = masque_blocs(rng)
                ctx = ctx.to(device).unsqueeze(0).expand(b, -1)
                cib = cib.to(device).unsqueeze(0).expand(b, -1)
                # Cible : représentation de l'image ENTIÈRE, normalisée (LayerNorm sans paramètre),
                # puis restreinte aux patchs cachés.
                if variante == "ema":
                    with torch.no_grad():                         # arrêt du gradient
                        s = F.layer_norm(cible(patchs), (config["dim"],))
                else:
                    h_tout = enc(patchs)                          # le gradient passe par la cible
                    s = F.layer_norm(h_tout, (config["dim"],))
                s_cachee = selection(s, cib)
                s_pred = pred(enc(patchs, ctx), ctx, cib)        # le contexte ne voit pas les cibles
                perte_pred = F.mse_loss(s_pred, s_cachee)
                perte = perte_pred
                if variante == "vicreg":
                    perte = perte + penalite_vicreg(h_tout.mean(1))   # sur la représentation de chaque image
                opt.zero_grad(); perte.backward(); opt.step(); sch.step()
                if variante == "ema":
                    # EMA : ξ ← m ξ + (1 − m) θ, m monte linéairement de 0,996 à 1
                    m = config["ema"][0] + (config["ema"][1] - config["ema"][0]) * pas / max(1, total - 1)
                    with torch.no_grad():
                        for pc, pe in zip(cible.parameters(), enc.parameters()):
                            pc.mul_(m).add_(pe.detach(), alpha=1 - m)
                pas += 1
                hist["perte"].append(perte_pred.item())
            hist["indic"].append(indicateur_effondrement(
                caracteristiques(a_evaluer, x_test[: config["n_indic"]], device)))
            print(f"  JEPA {variante} époque {_ep + 1} : perte de prédiction "
                  f"{sum(hist['perte'][-n_lots:]) / n_lots:.4f}, indicateur {hist['indic'][-1]:.2f}")
        return enc, pred, a_evaluer, hist

    jepa = {}
    durees_jepa = {}
    for _v in ("ema", "sans_ema", "vicreg"):
        _t0 = time.perf_counter()
        jepa[_v] = entrainer_jepa(_v)
        durees_jepa[_v] = time.perf_counter() - _t0
        print(f"JEPA {_v} : {durees_jepa[_v]:.1f} s")
    return durees_jepa, jepa


@app.cell
def _():
    import copy

    def copy_module(m):
        """Copie profonde d'un module (poids et tampons) : l'encodeur cible de départ."""
        return copy.deepcopy(m)

    return (copy_module,)


@app.cell
def _(mo):
    mo.md(r"""
    ## E. Ce que valent les représentations

    Les étiquettes servent **seulement ici**. Pour chaque encodeur gelé, la représentation d'une
    image est la moyenne de ses 16 vecteurs de patchs (protocole de MAE et d'I-JEPA). On mesure :

    - une **sonde linéaire** (régression logistique) entraînée sur les 20 000 images, puis sur
      100 images par classe seulement ;
    - un **k plus proches voisins** (k = 20, similarité cosinus) ;
    - la **dispersion** (écart-type d'une image à l'autre, moyenné sur les 128 dimensions :
      0 = effondrement) et le **rang effectif** (nombre de directions réellement utilisées) ;
    - pour comparaison, une sonde linéaire sur les 16 vecteurs **mis bout à bout** (2 048
      nombres, disposition spatiale conservée).

    Repères : les **pixels bruts** (784 nombres) et le même encodeur **au hasard**.
    """)


@app.cell
def _(
    Encodeur,
    F,
    caracteristiques,
    config,
    device,
    indicateur_effondrement,
    jepa,
    mae_enc,
    rang_effectif,
    time,
    torch,
    x_test,
    x_train,
    y_test,
    y_train,
):
    def sonde_lineaire(f_tr, y_tr, f_te, y_te, pas=500):
        """Régression logistique (entropie croisée) sur caractéristiques gelées et standardisées."""
        moy, ec = f_tr.mean(0), f_tr.std(0) + 1e-6
        a, b = ((f_tr - moy) / ec).to(device), ((f_te - moy) / ec).to(device)
        torch.manual_seed(42)
        lin = torch.nn.Linear(a.shape[1], 10).to(device)
        opt = torch.optim.Adam(lin.parameters(), lr=0.01, weight_decay=1e-4)
        ya = y_tr.to(device)
        for _ in range(pas):
            perte = F.cross_entropy(lin(a), ya)
            opt.zero_grad(); perte.backward(); opt.step()
        with torch.no_grad():
            return float((lin(b).argmax(1).cpu() == y_te).float().mean())

    def knn(f_tr, y_tr, f_te, y_te, k):
        """Vote des k plus proches voisins (similarité cosinus), par paquets de 1 000."""
        a = F.normalize(f_tr.to(device), dim=1)
        justes = 0
        for i in range(0, len(f_te), 1000):
            b = F.normalize(f_te[i:i + 1000].to(device), dim=1)
            voisins = (b @ a.T).topk(k, dim=1).indices.cpu()
            votes = torch.nn.functional.one_hot(y_tr[voisins], 10).sum(1)
            justes += int((votes.argmax(1) == y_te[i:i + 1000]).sum())
        return justes / len(f_te)

    # Sous-ensemble « peu d'étiquettes » : les 100 premières images de chaque classe
    _g = torch.Generator().manual_seed(42)
    _perm = torch.randperm(len(y_train), generator=_g)
    idx_peu = torch.cat([_perm[y_train[_perm] == c][: config["n_peu"]] for c in range(10)])

    _t0 = time.perf_counter()
    torch.manual_seed(42)
    _hasard = Encodeur().to(device)
    encodeurs = {
        "Pixels bruts": None,
        "Encodeur au hasard": _hasard,
        "MAE": mae_enc,
        "JEPA (EMA)": jepa["ema"][2],
        "JEPA sans EMA": jepa["sans_ema"][2],
        "JEPA + VICReg": jepa["vicreg"][2],
    }
    evaluation = {}
    feats = {}
    for _nom, _enc in encodeurs.items():
        if _enc is None:
            _ftr, _fte = x_train.reshape(len(x_train), -1), x_test.reshape(len(x_test), -1)
            _ctr, _cte = _ftr, _fte
        else:
            _ftr = caracteristiques(_enc, x_train, device)
            _fte = caracteristiques(_enc, x_test, device)
            _ctr = caracteristiques(_enc, x_train, device, mode="concat")
            _cte = caracteristiques(_enc, x_test, device, mode="concat")
        feats[_nom] = _fte
        evaluation[_nom] = dict(
            sonde=sonde_lineaire(_ftr, y_train, _fte, y_test),
            sonde_peu=sonde_lineaire(_ftr[idx_peu], y_train[idx_peu], _fte, y_test),
            knn=knn(_ftr, y_train, _fte, y_test, config["k_voisins"]),
            indic=indicateur_effondrement(_fte[: config["n_indic"]]),
            rang=rang_effectif(_fte[:2000]),
            sonde_concat=sonde_lineaire(_ctr, y_train, _cte, y_test),
        )
    duree_eval = time.perf_counter() - _t0
    print(f"{'représentation':<20} {'sonde':>7} {'sonde 100/cl.':>13} {'kNN':>7} {'dispersion':>10} "
          f"{'rang':>6} {'sonde concat':>12}")
    for _nom, _e in evaluation.items():
        print(f"{_nom:<20} {_e['sonde']:>7.1%} {_e['sonde_peu']:>13.1%} {_e['knn']:>7.1%} "
              f"{_e['indic']:>10.3f} {_e['rang']:>6.1f} {_e['sonde_concat']:>12.1%}")
    print(f"Évaluation : {duree_eval:.1f} s")
    return duree_eval, evaluation, feats


@app.cell
def _(evaluation, mo):
    _e = evaluation
    _h, _m, _j = _e["Encodeur au hasard"], _e["MAE"], _e["JEPA (EMA)"]
    mo.md(rf"""
    **Lecture.** Le MAE améliore la représentation moyenne par rapport au même réseau au hasard
    (sonde {_m['sonde']:.1%} contre {_h['sonde']:.1%}, kNN {_m['knn']:.1%} contre {_h['knn']:.1%}).
    La JEPA à EMA ne s'effondre pas (dispersion {_j['indic']:.2f} contre
    {_e['JEPA sans EMA']['indic']:.3f} sans EMA) et prédit des représentations qui dépendent de
    l'image (partie F), mais **à cette échelle** sa représentation moyenne ne classe pas mieux
    les chiffres que le hasard (sonde {_j['sonde']:.1%}). Les gains d'I-JEPA sur MAE sont mesurés
    sur ImageNet, avec des ViT de centaines de millions de paramètres et des centaines
    d'époques ; ils ne se reproduisent pas en trois minutes sur MNIST. Dernière colonne : mis
    bout à bout, les vecteurs d'un encodeur **au hasard** donnent déjà {_h['sonde_concat']:.1%},
    plus que les pixels bruts ({_e['Pixels bruts']['sonde']:.1%}) : sur MNIST, une sonde trop
    riche ne mesure plus l'apprentissage.
    """)


@app.cell
def _(mo):
    mo.md(r"""
    ## F. Regarder ce que chaque modèle prédit

    Pour trois chiffres de test : le MAE reçoit 25 % des patchs et **dessine** le reste ; la
    JEPA reçoit le contexte d'un masque par blocs et prédit un **vecteur** par patch caché. Pour
    rendre ces vecteurs visibles, on les projette sur les trois premières composantes
    principales des représentations cibles (ACP), traduites en couleurs rouge, vert, bleu : deux
    patchs de même couleur ont des représentations proches. La JEPA ne produit **aucun pixel**.
    """)


@app.cell
def _(
    F,
    GRILLE,
    N_PATCHS,
    config,
    depuis_patchs,
    device,
    en_patchs,
    jepa,
    mae_dec,
    mae_enc,
    masque_aleatoire,
    np,
    selection,
    torch,
    x_test,
):
    # --- F. Données de la démo MaskingCompare et des figures du chapitre ---
    CHIFFRES = [3, 2, 1] if len(x_test) > 3 else [0, 1, 2]   # indices de test : un 0, un 1, un 2
    _x = x_test[CHIFFRES]
    _patchs = en_patchs(_x.to(device))
    _gen = torch.Generator().manual_seed(7)
    _vis, _cach = masque_aleatoire(len(CHIFFRES), _gen)
    mae_enc.eval(); mae_dec.eval()
    with torch.no_grad():
        _pred = mae_dec(mae_enc(_patchs, _vis.to(device)), _vis.to(device), _cach.to(device))
        _recon = _patchs.clone()
        _recon.scatter_(1, _cach.to(device).unsqueeze(-1).expand(-1, -1, _patchs.shape[-1]), _pred)
        mae_mse = F.mse_loss(_pred, selection(_patchs, _cach.to(device)), reduction="none").mean((1, 2)).cpu()
    demo_masque_mae = np.zeros((len(CHIFFRES), N_PATCHS), dtype=bool)
    for _i in range(len(CHIFFRES)):
        demo_masque_mae[_i, _cach[_i].numpy()] = True
    demo_recon_mae = depuis_patchs(_recon).clamp(0, 1).cpu().numpy()

    # Masque de démonstration fixe (le même pour les trois chiffres) : un bloc 2 × 2 au centre
    # et un bloc 1 × 2 en haut à droite, soit 6 patchs cibles et 10 de contexte (grille 4 × 4).
    _cible = np.zeros((GRILLE, GRILLE), dtype=bool)
    _m = GRILLE // 4
    _cible[_m:3 * _m, _m:3 * _m] = True
    _cible[0:_m, 2 * _m:4 * _m] = True
    demo_masque_jepa = _cible.ravel()
    _idx_cib = torch.from_numpy(np.flatnonzero(demo_masque_jepa)).to(device)
    _idx_ctx = torch.from_numpy(np.flatnonzero(~demo_masque_jepa)).to(device)
    _enc, _predicteur, _cible_enc, _ = jepa["ema"]
    _enc.eval(); _predicteur.eval(); _cible_enc.eval()
    with torch.no_grad():
        _b = len(CHIFFRES)
        _s = F.layer_norm(_cible_enc(_patchs), (config["dim"],))          # (3, 16, D) cibles
        _sp = _predicteur(_enc(_patchs, _idx_ctx.expand(_b, -1)), _idx_ctx.expand(_b, -1),
                          _idx_cib.expand(_b, -1))                          # (3, k, D) prédictions
        # ACP des représentations cibles de 500 images de test → 3 composantes
        _ref = F.layer_norm(_cible_enc(en_patchs(x_test[:500].to(device))), (config["dim"],))
        _ref = _ref.reshape(-1, config["dim"]).cpu()
        _moy = _ref.mean(0)
        _u, _sv, _v = torch.linalg.svd(_ref - _moy, full_matrices=False)
        _axes = _v[:3].T                                                    # (D, 3)
        _proj_ref = (_ref - _moy) @ _axes
        _bas, _haut = _proj_ref.quantile(0.02, dim=0), _proj_ref.quantile(0.98, dim=0)

        def _couleurs(t):
            return (((t.cpu() - _moy) @ _axes - _bas) / (_haut - _bas)).clamp(0, 1).numpy()

        demo_rgb_cible = _couleurs(_s.reshape(-1, config["dim"])).reshape(_b, N_PATCHS, 3)
        demo_rgb_pred = _couleurs(_sp.reshape(-1, config["dim"])).reshape(_b, -1, 3)
        _s_cach = _s[:, _idx_cib]
        demo_cos_pred = F.cosine_similarity(_sp, _s_cach, dim=-1).mean(1).cpu().numpy()
        # Repère : prédire pour chaque patch caché la représentation moyenne (sur 500 images)
        _moy_pos = _ref.reshape(500, N_PATCHS, -1).mean(0).to(device)[_idx_cib]
        demo_cos_moy = F.cosine_similarity(_moy_pos.unsqueeze(0).expand(_b, -1, -1), _s_cach,
                                           dim=-1).mean(1).cpu().numpy()
        jepa_mse = F.mse_loss(_sp, _s_cach, reduction="none").mean((1, 2)).cpu().numpy()
        variance_acp = float((_sv[:3] ** 2).sum() / (_sv**2).sum())
    demo_images = _x[:, 0].numpy()
    print("Chiffres :", CHIFFRES, "MSE MAE (pixels cachés) :", np.round(mae_mse.numpy(), 4))
    print("JEPA : cosinus prédiction/cible par chiffre", np.round(demo_cos_pred, 3),
          "; repère « moyenne par position »", np.round(demo_cos_moy, 3))
    print(f"ACP : 3 composantes = {variance_acp:.1%} de la variance des représentations cibles")
    return (
        CHIFFRES,
        demo_cos_moy,
        demo_cos_pred,
        demo_images,
        demo_masque_jepa,
        demo_masque_mae,
        demo_recon_mae,
        demo_rgb_cible,
        demo_rgb_pred,
        jepa_mse,
        mae_mse,
        variance_acp,
    )


@app.cell
def _(
    GRILLE,
    P,
    SMOKE,
    demo_images,
    demo_masque_jepa,
    demo_masque_mae,
    demo_recon_mae,
    demo_rgb_cible,
    demo_rgb_pred,
    evaluation,
    hist_mae,
    jepa,
    mo,
    np,
    plt,
    sauvegarder_figure,
):
    # --- Figure de l'exercice : courbes, indicateur, sondes, et ce que chaque modèle prédit ---
    _couleurs = {"MAE": "#E69F00", "JEPA (EMA)": "#0072B2", "JEPA sans EMA": "#D55E00",
                 "JEPA + VICReg": "#009E73", "Encodeur au hasard": "#7F7F7F", "Pixels bruts": "#000000",
                 "JEPA (EMA), contexte": "#56B4E9"}
    _hist = {"MAE": hist_mae, "JEPA (EMA)": jepa["ema"][3], "JEPA sans EMA": jepa["sans_ema"][3],
             "JEPA + VICReg": jepa["vicreg"][3]}
    fig = plt.figure(figsize=(13, 8))
    _g = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    _ax = fig.add_subplot(_g[0, 0])
    for _n, _h in _hist.items():
        _p = np.array(_h["perte"])
        _l = np.convolve(_p, np.ones(10) / 10, mode="valid") if len(_p) > 20 else _p
        _ax.plot(_l, color=_couleurs[_n], label=_n)
    _ax.set(yscale="log", xlabel="pas", ylabel="perte (moyenne glissante)", title="Pertes d'entraînement")
    _ax.legend(fontsize=8)
    _ax = fig.add_subplot(_g[0, 1])
    for _n, _h in _hist.items():
        _ax.plot(_h["indic"], "o-", color=_couleurs[_n], label=_n)
    _ax.set(xlabel="époque", ylabel="écart-type d'une image à l'autre", ylim=(0, None),
            title="Dispersion (0 = effondrement)")
    _ax = fig.add_subplot(_g[0, 2])
    _noms = list(evaluation)
    _x = np.arange(len(_noms))
    _ax.bar(_x - 0.2, [100 * evaluation[n]["sonde"] for n in _noms], 0.4,
            color=[_couleurs[n] for n in _noms], label="sonde, toutes étiquettes")
    _ax.bar(_x + 0.2, [100 * evaluation[n]["sonde_peu"] for n in _noms], 0.4,
            color=[_couleurs[n] for n in _noms], hatch="//", edgecolor="white", label="sonde, 100 par classe")
    _ax.set_xticks(_x, [n.replace(" ", "\n", 1) for n in _noms], fontsize=7)
    _ax.set(ylim=(0, 100), ylabel="exactitude en test (%)", title="Sondes linéaires")
    _ax.legend(fontsize=7, loc="lower right")
    # Ligne du bas : chiffre, masque MAE, reconstruction MAE, couleurs JEPA cible / prédites
    _sub = _g[1, :].subgridspec(len(demo_images), 5)
    for _i in range(len(demo_images)):
        _masque = np.repeat(np.repeat(demo_masque_mae[_i].reshape(GRILLE, GRILLE), P, 0), P, 1)
        _vu = np.where(_masque, 0.5, demo_images[_i])
        _rgb_c = demo_rgb_cible[_i].reshape(GRILLE, GRILLE, 3)
        _rgb_p = np.full((GRILLE, GRILLE, 3), 0.85)
        _rgb_p[demo_masque_jepa.reshape(GRILLE, GRILLE)] = demo_rgb_pred[_i]
        _ctx = np.where(np.repeat(np.repeat(demo_masque_jepa.reshape(GRILLE, GRILLE), P, 0), P, 1), 0.5, demo_images[_i])
        for _j, (_img, _t) in enumerate([(demo_images[_i], "image"), (_vu, "vu par le MAE"),
                                         (demo_recon_mae[_i], "pixels reconstruits"),
                                         (_rgb_c, "JEPA : cibles (ACP)"),
                                         (_rgb_p, "JEPA : prédites")]):
            _a = fig.add_subplot(_sub[_i, _j])
            _a.imshow(_img, cmap="gray" if _img.ndim == 2 else None, vmin=0, vmax=1, interpolation="nearest")
            _a.set_xticks([]); _a.set_yticks([]); _a.grid(False)
            if _i == 0:
                _a.set_title(_t, fontsize=9)
    fig.tight_layout()
    _chemins = sauvegarder_figure(fig, "mae-jepa-smoke" if SMOKE else "mae-jepa", mo.notebook_dir() / "figures")
    print(f"Figure : {_chemins[0]}")
    fig


@app.cell
def _(
    CHIFFRES,
    SMOKE,
    config,
    demo_cos_moy,
    demo_cos_pred,
    demo_images,
    demo_masque_jepa,
    demo_masque_mae,
    demo_recon_mae,
    demo_rgb_cible,
    demo_rgb_pred,
    duree_eval,
    duree_mae,
    durees_jepa,
    evaluation,
    hist_mae,
    jepa,
    jepa_mse,
    json,
    mae_mse,
    matrices_c,
    mo,
    np,
    resultats_a,
    variance_acp,
):
    # --- Résultats bruts pour les figures du site (site/scripts/figures/ch26_*.py) et la démo ---
    bilan = dict(
        config={k: (list(v) if isinstance(v, tuple) else v) for k, v in config.items()},
        partie_a=resultats_a,
        evaluation=evaluation,
        historiques={"MAE": hist_mae, "JEPA (EMA)": jepa["ema"][3],
                     "JEPA sans EMA": jepa["sans_ema"][3], "JEPA + VICReg": jepa["vicreg"][3]},
        durees=dict(mae=duree_mae, **{f"jepa_{k}": v for k, v in durees_jepa.items()}, evaluation=duree_eval),
        demo=dict(chiffres=CHIFFRES, mse_mae=mae_mse.tolist(), mse_jepa=jepa_mse.tolist(),
                  cos_pred=demo_cos_pred.tolist(), cos_moyenne=demo_cos_moy.tolist(),
                  variance_acp=variance_acp),
    )
    _chemin = mo.notebook_dir() / "figures" / ("resultats-smoke.npz" if SMOKE else "resultats.npz")
    np.savez_compressed(
        _chemin, bilan=json.dumps(bilan, ensure_ascii=False),
        images=demo_images.astype(np.float32), recon_mae=demo_recon_mae.astype(np.float32),
        masque_mae=demo_masque_mae, masque_jepa=demo_masque_jepa,
        rgb_cible=demo_rgb_cible.astype(np.float32), rgb_pred=demo_rgb_pred.astype(np.float32),
        **{f"c_{k}": v.astype(np.float32) for k, v in
           {"sain": matrices_c["sain"], "effondre": matrices_c["effondré"],
            "redondant": matrices_c["redondant"]}.items()},
    )
    print(f"Résultats : {_chemin}")
    return


@app.cell
def _(
    SMOKE,
    demo_cos_moy,
    demo_cos_pred,
    duree_eval,
    duree_mae,
    durees_jepa,
    evaluation,
    jepa,
    mo,
    np,
    resultats_a,
):
    # --- Bilan : seuils chiffrés du README. En smoke, on vérifie seulement que tout tourne.
    _e = evaluation
    _verifs = {
        "A : l'alignement seul préfère le lot effondré":
            resultats_a["effondré"]["alignement"] < resultats_a["sain"]["alignement"],
        "A : Barlow et VICReg pénalisent l'effondré plus que le sain":
            resultats_a["effondré"]["barlow"] > 5 * resultats_a["sain"]["barlow"]
            and resultats_a["effondré"]["vicreg"] > 5 * resultats_a["sain"]["vicreg"],
        "MAE : sonde linéaire ≥ encodeur au hasard + 3 points":
            _e["MAE"]["sonde"] >= _e["Encodeur au hasard"]["sonde"] + 0.03,
        "MAE : kNN ≥ encodeur au hasard + 5 points":
            _e["MAE"]["knn"] >= _e["Encodeur au hasard"]["knn"] + 0.05,
        "JEPA (EMA) : pas effondrée (dispersion ≥ celle de l'encodeur au hasard)":
            _e["JEPA (EMA)"]["indic"] >= _e["Encodeur au hasard"]["indic"],
        "JEPA (EMA) : prédit mieux que la moyenne par position (cosinus + 0,1)":
            float(np.mean(demo_cos_pred)) >= float(np.mean(demo_cos_moy)) + 0.1,
        "JEPA sans EMA : effondrée (dispersion ≤ 10 % de celle du hasard)":
            _e["JEPA sans EMA"]["indic"] <= 0.1 * _e["Encodeur au hasard"]["indic"],
        "JEPA sans EMA : perte de prédiction finale ≤ 0,01 (solution triviale)":
            float(np.mean(jepa["sans_ema"][3]["perte"][-20:])) <= 0.01,
        "JEPA + VICReg : pas effondrée (dispersion ≥ celle du hasard)":
            _e["JEPA + VICReg"]["indic"] >= _e["Encodeur au hasard"]["indic"],
    }
    _total = duree_mae + sum(durees_jepa.values()) + duree_eval
    _lignes = "\n".join(f"- {'✅' if ok else '❌'} {k}" for k, ok in _verifs.items())
    if not SMOKE and not all(_verifs.values()):
        raise SystemExit("Seuils non atteints :\n" + _lignes)
    mo.md(f"**Bilan** ({_total:.0f} s d'entraînement et d'évaluation)\n\n{_lignes}")


if __name__ == "__main__":
    app.run()
