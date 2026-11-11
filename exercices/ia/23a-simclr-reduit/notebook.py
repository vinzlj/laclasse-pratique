# Exercice 23a : apprentissage contrastif, d'InfoNCE en NumPy à SimCLR réduit sur CIFAR-10.
#
#   make ex NUM=23a              → éditeur marimo (bouton « Entraîner »)
#   make ex NUM=23a MODE=run     → script complet, vérifie les seuils du README
#   make ex NUM=23a MODE=smoke   → test rapide (< 30 s)
#
# Partie A en NumPy : InfoNCE (NT-Xent), température, perte sigmoïde de SigLIP, Barlow Twins,
# et ce que chaque perte « voit » d'un encodeur effondré. Vérifiée contre PyTorch.
# Partie B en PyTorch (MPS) : SimCLR réduit (petit CNN, 25 000 images sans étiquette),
# Barlow Twins avec le même encodeur, la même augmentation et la même sonde, et un
# entraînement « alignement seul » qui s'effondre. Sonde linéaire, peu d'étiquettes, t-SNE.
#
# Les deux vues augmentées sont tirées IMAGE PAR IMAGE dans les processus de chargement
# (ia_commun.donnees.PairesVues + ia_commun.augmentations.AugmentationSimCLR) : une
# transformation v2 appliquée à un lot tirerait le même recadrage pour toutes les images.

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
        style_commun,
        time,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Exercice 23a : apprentissage contrastif (SimCLR réduit)

    Un réseau de classification apprend parce qu'on lui donne la bonne réponse. Ici, **aucune
    étiquette** : on prend une image, on en tire deux versions déformées (recadrage, couleurs,
    retournement), et on demande à l'encodeur de les envoyer au même endroit de l'espace des
    représentations, **et loin des autres images du lot**. Les étiquettes ne servent qu'à la
    fin, pour mesurer ce que vaut la représentation apprise : une simple régression logistique
    (la *sonde linéaire*) posée sur l'encodeur gelé.

    **Perte optimisée.** InfoNCE dans sa version SimCLR (NT-Xent). Pour une paire positive
    $(i, j)$ d'un lot de $N$ images, donc $2N$ vues :

    $$\ell_{i,j} = -\log \frac{\exp(\mathrm{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(\mathrm{sim}(z_i, z_k)/\tau)}$$

    où $\mathrm{sim}$ est la similarité cosinus et $\tau$ la température. C'est une entropie
    croisée : la « bonne classe » de la vue $i$ est sa jumelle $j$, parmi $2N - 1$ candidates.
    En comparaison : **Barlow Twins** (même encodeur, aucun négatif) et l'**alignement seul**
    $\lVert z_i - z_j \rVert^2$, qui s'effondre.

    - **Partie A (NumPy)** : les pertes à la main sur des vecteurs choisis, vérifiées contre PyTorch.
    - **Partie B (PyTorch)** : SimCLR et Barlow Twins sur 25 000 images de CIFAR-10, sonde linéaire,
      peu d'étiquettes, t-SNE, effondrement.
    """)


@app.cell
def _(get_device, is_smoke, os, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    device = get_device()
    style_commun()
    config = dict(
        n_images=512 if SMOKE else 25000,   # images SANS étiquette pour l'auto-supervisé
        epoques=1 if SMOKE else int(os.environ.get("IA_EPOQUES", "15")),
        epoques_alignement=1 if SMOKE else 3,  # l'alignement seul s'effondre vite
        lot=256 if SMOKE else 512,           # N images → 2N vues → 2N − 1 candidates par vue
        tau=float(os.environ.get("IA_TAU", "0.5")),  # température (SimCLR, annexe B.9 : 0,5 sur CIFAR-10)
        lambda_barlow=5e-3,                  # poids hors diagonale (Barlow Twins, article : 5·10⁻³)
        lr_max=2e-3,                         # AdamW + taux « one cycle », comme l'exercice 13a
        weight_decay=1e-4,
        dim_h=128,                           # représentation h (sortie de l'encodeur)
        dim_cachee=512,                      # tête de projection 128 → 512 → 128
        dim_z=128,
        workers=2 if SMOKE else 6,
        n_sonde=1000 if SMOKE else 50000,    # étiquettes utilisées par la sonde linéaire
        n_test=1000 if SMOKE else 10000,
        n_rares=(100, 500),                  # 10 puis 50 étiquettes par classe (« peu d'étiquettes »)
        epoques_supervise=5 if SMOKE else 100,
        n_tsne=300 if SMOKE else 2000,
        n_mesure=512 if SMOKE else 1024,     # images de test fixes pour mesurer l'effondrement
    )
    # Variante (IA_TAU ≠ 0,5 ou IA_EPOQUES posé) : figures et résultats sous un autre nom,
    # pour ne pas écraser ceux du run de référence (relus par les figures du site), et
    # pas de vérification des seuils (ils valent pour les réglages par défaut).
    VARIANTE = os.environ.get("IA_TAU", "0.5") != "0.5" or "IA_EPOQUES" in os.environ
    SUFFIXE = "-smoke" if SMOKE else (f"-tau{config['tau']}-ep{config['epoques']}" if VARIANTE else "")
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, périphérique {device}"
          + (f", variante {SUFFIXE}" if VARIANTE and not SMOKE else ""))
    print(config)
    return SEED, SMOKE, SUFFIXE, VARIANTE, config, device


@app.cell
def _(mo):
    mo.md(r"""
    ## Partie A : les pertes à la main (NumPy)

    Trois images, deux vues chacune, des représentations en 2 dimensions placées à la main
    sur le cercle unité (l'angle suffit). Les deux vues d'une même image sont proches
    (20° à 30° d'écart), les images sont à 120° les unes des autres.
    """)


@app.cell
def _(np):
    def normaliser(z):
        """Ramène chaque ligne sur la sphère unité (la similarité devient le cosinus)."""
        return z / np.linalg.norm(z, axis=1, keepdims=True)

    def infonce_numpy(z1, z2, tau):
        """NT-Xent de SimCLR, écrit comme la formule : moyenne de ℓ(i, j) sur les 2N vues.

        z1[k] et z2[k] sont les deux vues de l'image k. Renvoie (perte, probabilités) où
        probabilités[i] est la softmax de la vue i sur ses 2N − 1 candidates (diagonale à 0).
        """
        z = normaliser(np.concatenate([z1, z2]))          # 2N vues
        n2 = len(z)
        n = n2 // 2
        s = z @ z.T / tau                                 # similarités cosinus / τ
        np.fill_diagonal(s, -np.inf)                      # k ≠ i : une vue n'est pas sa propre candidate
        s = s - s.max(axis=1, keepdims=True)              # stabilité numérique (ne change pas la softmax)
        p = np.exp(s)
        p = p / p.sum(axis=1, keepdims=True)
        jumelle = np.r_[np.arange(n, n2), np.arange(0, n)]  # la vue i a pour positive i ± N
        pertes = -np.log(p[np.arange(n2), jumelle])
        return pertes.mean(), p

    def siglip_numpy(x, y, t=10.0, b=-10.0):
        """Perte sigmoïde de SigLIP (Zhai et al., 2023, algorithme 1) : chaque couple (i, j)
        est une classification binaire, +1 si i et j vont ensemble, −1 sinon. Pas de softmax :
        aucune normalisation sur le lot. t = 10 et b = −10 : initialisation de l'article."""
        x, y = normaliser(x), normaliser(y)
        logits = x @ y.T * t + b
        etiquettes = 2 * np.eye(len(x)) - 1
        # −log σ(u) = log(1 + e^(−u)), écrit avec logaddexp pour la stabilité
        return np.logaddexp(0, -etiquettes * logits).sum() / len(x)

    def barlow_numpy(z1, z2, lam=5e-3):
        """Barlow Twins (Zbontar et al., 2021) : matrice de corrélation croisée C (d × d)
        entre les deux vues, poussée vers l'identité. Diagonale = invariance, hors
        diagonale = redondance. Aucun négatif."""
        def standardiser(z):
            return (z - z.mean(0)) / (z.std(0) + 1e-5)   # comme une BatchNorm sans paramètres
        c = standardiser(z1).T @ standardiser(z2) / len(z1)
        diag = np.diag(c)
        return ((1 - diag) ** 2).sum() + lam * ((c ** 2).sum() - (diag ** 2).sum())

    def alignement_numpy(z1, z2):
        """Alignement seul : distance au carré entre vues normalisées (= 2 − 2 cos)."""
        return ((normaliser(z1) - normaliser(z2)) ** 2).sum(1).mean()

    def sur_cercle(degres):
        a = np.deg2rad(np.asarray(degres, dtype=np.float64))
        return np.stack([np.cos(a), np.sin(a)], axis=1)

    # Vue 1 et vue 2 de trois images : chat (0° et 20°), avion (120° et 150°), camion (240° et 215°)
    Z1_main = sur_cercle([0, 120, 240])
    Z2_main = sur_cercle([20, 150, 215])
    return (
        Z1_main,
        Z2_main,
        alignement_numpy,
        barlow_numpy,
        infonce_numpy,
        normaliser,
        siglip_numpy,
    )


@app.cell
def _(F, Z1_main, Z2_main, infonce_numpy, normaliser, np, siglip_numpy, torch):
    # --- A1. InfoNCE pas à pas, sur 3 images (6 vues) ---
    _z = normaliser(np.concatenate([Z1_main, Z2_main]))
    cosinus_main = _z @ _z.T
    print("Similarités cosinus des 6 vues (lignes et colonnes : chat1, avion1, camion1, chat2, avion2, camion2)")
    print(np.array2string(cosinus_main, precision=3, suppress_small=True))

    infonce_main = {}
    for _tau in (1.0, 0.5, 0.1):
        _perte, _p = infonce_numpy(Z1_main, Z2_main, _tau)
        infonce_main[_tau] = dict(perte=float(_perte), p_chat=_p[0].tolist())
        print(f"τ = {_tau:>4} : perte {_perte:.4f} ; softmax de la vue chat1 sur "
              f"[avion1, camion1, chat2, avion2, camion2] = "
              f"{np.round(np.delete(_p[0], 0), 3).tolist()}")

    # Le même calcul en PyTorch, écrit comme dans la partie B : une entropie croisée dont
    # la « classe » de la vue i est l'indice de sa jumelle.
    def nt_xent(z1, z2, tau):
        z = F.normalize(torch.cat([z1, z2]), dim=1)
        n = z1.shape[0]
        sim = z @ z.T / tau
        masque = torch.eye(2 * n, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(masque, float("-inf"))      # k ≠ i
        cibles = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
        return F.cross_entropy(sim, cibles)

    _t1, _t2 = torch.from_numpy(Z1_main), torch.from_numpy(Z2_main)
    ecart_numpy_torch = max(
        abs(infonce_numpy(Z1_main, Z2_main, _tau)[0] - nt_xent(_t1, _t2, _tau).item())
        for _tau in (1.0, 0.5, 0.1)
    )
    # SigLIP : même exemple, vue 1 jouant les « images » et vue 2 les « textes »
    _x, _y = torch.from_numpy(normaliser(Z1_main)), torch.from_numpy(normaliser(Z2_main))
    _etiq = 2 * torch.eye(3, dtype=torch.float64) - 1
    _siglip_torch = -F.logsigmoid(_etiq * (_x @ _y.T * 10.0 - 10.0)).sum() / 3
    siglip_main = float(siglip_numpy(Z1_main, Z2_main))
    ecart_numpy_torch = max(ecart_numpy_torch, abs(siglip_main - _siglip_torch.item()))
    print(f"SigLIP (t = 10, b = −10) : perte {siglip_main:.4f}")
    print(f"Écart maximal NumPy / PyTorch (InfoNCE, SigLIP) : {ecart_numpy_torch:.1e}")
    return cosinus_main, ecart_numpy_torch, infonce_main, nt_xent, siglip_main


@app.cell
def _(np):
    # --- A2. La température : sur qui se porte le gradient ? ---
    # Une ligne de similarités : la positive à 0,8, six négatives de plus en plus faciles.
    # ∂ℓ/∂s_k = (p_k − 1[k = positive]) / τ : chaque négative est repoussée en proportion de
    # sa probabilité p_k. La part de la plus difficile (0,7) dit où va l'effort.
    s_pos, s_neg = 0.8, np.array([0.7, 0.5, 0.2, 0.0, -0.3, -0.5])
    temperature = []
    for _tau in (1.0, 0.5, 0.2, 0.1, 0.05):
        _logits = np.r_[s_pos, s_neg] / _tau
        _p = np.exp(_logits - _logits.max())
        _p /= _p.sum()
        temperature.append(dict(
            tau=_tau, perte=float(-np.log(_p[0])), p_positive=float(_p[0]),
            part_plus_difficile=float(_p[1] / _p[1:].sum()),
            part_plus_facile=float(_p[-1] / _p[1:].sum()),
        ))
        print(f"τ = {_tau:<4} : perte {temperature[-1]['perte']:.3f}, p(positive) {_p[0]:.3f}, "
              f"part du gradient des négatives sur la plus difficile {temperature[-1]['part_plus_difficile']:.0%}, "
              f"sur la plus facile {temperature[-1]['part_plus_facile']:.1%}")
    return (temperature,)


@app.cell
def _(alignement_numpy, barlow_numpy, infonce_numpy, np):
    # --- A3. Ce que chaque perte « voit » d'un encodeur effondré ---
    # 256 images, représentations en 128 dimensions, trois situations :
    #   bonnes : deux vues proches, images différentes bien séparées ;
    #   indépendantes : les deux vues n'ont rien à voir (encodeur au hasard) ;
    #   effondrées : tout le monde reçoit le même vecteur.
    _rng = np.random.default_rng(42)
    _n, _d = 256, 128
    _base = _rng.normal(size=(_n, _d))
    situations = {
        "bonnes": (_base + 0.3 * _rng.normal(size=(_n, _d)), _base + 0.3 * _rng.normal(size=(_n, _d))),
        "indépendantes": (_rng.normal(size=(_n, _d)), _rng.normal(size=(_n, _d))),
        "effondrées": (np.ones((_n, _d)), np.ones((_n, _d))),
    }
    effondrement_numpy = {}
    for _nom, (_a, _b) in situations.items():
        effondrement_numpy[_nom] = dict(
            alignement=float(alignement_numpy(_a, _b)),
            infonce=float(infonce_numpy(_a, _b, 0.5)[0]),
            barlow=float(barlow_numpy(_a, _b)),
        )
        print(f"{_nom:>14} : alignement {effondrement_numpy[_nom]['alignement']:.3f}, "
              f"InfoNCE (τ = 0,5) {effondrement_numpy[_nom]['infonce']:.3f}, "
              f"Barlow Twins {effondrement_numpy[_nom]['barlow']:.2f}")
    print(f"Repère : log(2N − 1) = log({2 * _n - 1}) = {np.log(2 * _n - 1):.3f} "
          "(InfoNCE quand toutes les candidates se valent)")
    return (effondrement_numpy,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Partie B : SimCLR réduit sur CIFAR-10 (PyTorch)

    **Données.** 25 000 images de CIFAR-10 tirées au hasard, **sans leurs étiquettes**. Chaque
    image donne deux vues par l'augmentation de SimCLR (recadrage redimensionné à 20-100 % de
    la surface, retournement, distorsion de couleur avec une probabilité de 0,8, niveaux de
    gris avec 0,2). **Encodeur** : le petit CNN de l'exercice 13a (6 convolutions), qui sort
    une représentation $h$ de 128 nombres. **Tête de projection** : 128 → 512 → 128, la sortie
    $z$ entre dans la perte ; on la jette après l'entraînement (SimCLR, section 4.2).

    **Évaluation.** On gèle l'encodeur, on calcule $h$ pour les 50 000 images d'entraînement
    (avec leurs étiquettes cette fois) et on entraîne une régression logistique dessus.
    Son exactitude sur les 10 000 images de test mesure la qualité de la représentation.
    """)


@app.cell
def _(SEED, config, np, torch):
    from ia_commun.augmentations import AugmentationClassique, AugmentationSimCLR, normaliser_cifar10
    from ia_commun.donnees import CIFAR10, CLASSES_CIFAR10, PairesVues

    _complet = CIFAR10("train")
    _rng = np.random.default_rng(SEED)
    indices_ssl = np.sort(_rng.choice(len(_complet), size=config["n_images"], replace=False))
    paires = PairesVues(CIFAR10("train", indices=indices_ssl), AugmentationSimCLR())
    chargeur_paires = torch.utils.data.DataLoader(
        paires, batch_size=config["lot"], shuffle=True, drop_last=True,
        num_workers=config["workers"], persistent_workers=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    # Images brutes (uint8) pour la sonde et le test : normalisées lot par lot au moment de
    # calculer les représentations (50 000 images en float32 occuperaient 600 Mo).
    _ind_sonde = np.sort(_rng.choice(len(_complet), size=config["n_sonde"], replace=False))
    X_sonde = torch.from_numpy(np.asarray(_complet.images[_ind_sonde])).permute(0, 3, 1, 2).contiguous()
    y_sonde = torch.from_numpy(_complet.labels[_ind_sonde])
    _test = CIFAR10("test")
    X_test = torch.from_numpy(np.asarray(_test.images[: config["n_test"]])).permute(0, 3, 1, 2).contiguous()
    y_test = torch.from_numpy(_test.labels[: config["n_test"]])

    # Scénario « peu d'étiquettes » : 10 puis 50 images étiquetées par classe, prises dans la
    # sonde (les 100 sont incluses dans les 500)
    _tirage = [_rng.choice(np.flatnonzero(y_sonde.numpy() == c), size=max(config["n_rares"]) // 10, replace=False)
               for c in range(10)]
    ind_rares = {n: np.concatenate([t[: n // 10] for t in _tirage]) for n in config["n_rares"]}
    print(f"{len(paires)} images sans étiquette ({len(chargeur_paires)} lots de {config['lot']} par époque), "
          f"sonde : {len(y_sonde)} images étiquetées, test : {len(y_test)}, peu d'étiquettes : {list(ind_rares)}")
    return (
        AugmentationClassique,
        AugmentationSimCLR,
        CIFAR10,
        CLASSES_CIFAR10,
        X_sonde,
        X_test,
        chargeur_paires,
        ind_rares,
        indices_ssl,
        normaliser_cifar10,
        paires,
        y_sonde,
        y_test,
    )


@app.cell
def _(config, nn):
    def bloc(c_in, c_out, k=3):
        return nn.Sequential(nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False),
                             nn.BatchNorm2d(c_out), nn.ReLU())

    class Encodeur(nn.Module):
        """Le petit CNN de l'exercice 13a, sans sa couche de classification : image → h (128)."""

        def __init__(self):
            super().__init__()
            self.extraire = nn.Sequential(
                bloc(3, 32, 5), bloc(32, 32), nn.MaxPool2d(2),      # 32×32 → 16×16
                bloc(32, 64), bloc(64, 64), nn.MaxPool2d(2),        # 16×16 → 8×8
                bloc(64, 128), bloc(128, 128),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),              # → h, 128 nombres
            )

        def forward(self, x):
            return self.extraire(x)

    class ModeleSSL(nn.Module):
        """Encodeur f + tête de projection g (SimCLR) : renvoie h = f(x) et z = g(h)."""

        def __init__(self):
            super().__init__()
            self.f = Encodeur()
            self.g = nn.Sequential(
                nn.Linear(config["dim_h"], config["dim_cachee"]), nn.BatchNorm1d(config["dim_cachee"]),
                nn.ReLU(), nn.Linear(config["dim_cachee"], config["dim_z"]),
            )

        def forward(self, x):
            h = self.f(x)
            return h, self.g(h)

    def nb_parametres(m):
        return sum(p.numel() for p in m.parameters())

    _m = ModeleSSL()
    print(f"Encodeur : {nb_parametres(_m.f):,} paramètres ; tête de projection : "
          f"{nb_parametres(_m.g):,}".replace(",", " "))
    return Encodeur, ModeleSSL, nb_parametres


@app.cell
def _(F, config, nt_xent, torch):
    # --- Les trois pertes de la partie B (mêmes formules que la partie A, en PyTorch) ---
    def perte_barlow(z1, z2, lam=config["lambda_barlow"]):
        n = z1.shape[0]
        z1n = (z1 - z1.mean(0)) / (z1.std(0, unbiased=False) + 1e-5)
        z2n = (z2 - z2.mean(0)) / (z2.std(0, unbiased=False) + 1e-5)
        c = z1n.T @ z2n / n                                 # corrélation croisée d × d
        diag = torch.diagonal(c)
        return ((1 - diag) ** 2).sum() + lam * ((c ** 2).sum() - (diag ** 2).sum())

    def perte_alignement(z1, z2):
        return (F.normalize(z1, dim=1) - F.normalize(z2, dim=1)).pow(2).sum(1).mean()

    PERTES = {
        "SimCLR (InfoNCE)": lambda z1, z2: nt_xent(z1, z2, config["tau"]),
        "Barlow Twins": perte_barlow,
        "Alignement seul": perte_alignement,
    }
    return (PERTES,)


@app.cell
def _(F, SEED, X_test, config, device, torch):
    from ia_commun.augmentations import ECART_TYPE_CIFAR10, MOYENNE_CIFAR10

    _moy = torch.tensor(MOYENNE_CIFAR10, device=device).view(1, 3, 1, 1)
    _et = torch.tensor(ECART_TYPE_CIFAR10, device=device).view(1, 3, 1, 1)

    def normaliser_lot(x_uint8):
        """Lot d'images uint8 → float32 normalisé, sur le GPU (même calcul que normaliser_cifar10)."""
        return (x_uint8.to(device).float() / 255 - _moy) / _et

    @torch.no_grad()
    def representations(modele, X, lot=1000, sortie="h"):
        """h (ou z) pour des images uint8, encodeur en mode évaluation, sans augmentation."""
        modele.eval()
        morceaux = []
        for i in range(0, len(X), lot):
            h, z = modele(normaliser_lot(X[i : i + lot]))
            morceaux.append((h if sortie == "h" else z).float().cpu())
        return torch.cat(morceaux)

    @torch.no_grad()
    def mesure_effondrement(modele):
        """Écart-type par dimension de z/‖z‖ sur des images de test fixes, moyenné, × √d.

        Repère de SimSiam (Chen et He, 2020) : 0 si toutes les sorties sont identiques
        (effondrement), environ 1/√d si elles sont réparties uniformément sur la sphère ;
        multiplié par √d, l'indicateur vaut donc 0 (effondré) à ≈ 1 (bien réparti).
        """
        z = F.normalize(representations(modele, X_test[: config["n_mesure"]], sortie="z"), dim=1)
        return float(z.std(0).mean() * z.shape[1] ** 0.5)

    def sonde_lineaire(h_train, y_train, h_eval, y_eval, pas=300, decroissance=1e-4):
        """Régression logistique sur représentations gelées (standardisées), lot complet, Adam."""
        moy, et = h_train.mean(0), h_train.std(0) + 1e-6
        a = ((h_train - moy) / et).to(device)
        b = ((h_eval - moy) / et).to(device)
        ya = y_train.to(device)
        torch.manual_seed(SEED)
        lin = torch.nn.Linear(a.shape[1], 10).to(device)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=decroissance)
        for _ in range(pas):
            perte = F.cross_entropy(lin(a), ya)
            opt.zero_grad()
            perte.backward()
            opt.step()
        with torch.no_grad():
            return (lin(b).argmax(1).cpu() == y_eval).float().mean().item()

    @torch.no_grad()
    def knn(h_train, y_train, h_eval, y_eval, k=20):
        """k plus proches voisins (cosinus), vote majoritaire : la géométrie brute de l'espace."""
        a = F.normalize(h_train, dim=1).to(device)
        b = F.normalize(h_eval, dim=1).to(device)
        ya = y_train.to(device)
        # par paquets de 2 000 requêtes : la matrice 10 000 × 50 000 entière pèserait 2 Go
        voisins = torch.cat([(b[i : i + 2000] @ a.T).topk(k, dim=1).indices for i in range(0, len(b), 2000)])
        pred = torch.mode(ya[voisins].cpu(), dim=1).values   # mode : pas d'équivalent MPS, calcul sur CPU
        return (pred == y_eval).float().mean().item()

    return knn, mesure_effondrement, normaliser_lot, representations, sonde_lineaire


@app.cell
def _(mo):
    # Le bouton empêche l'entraînement de se relancer à chaque modification d'une cellule amont.
    bouton = mo.ui.run_button(label="Entraîner (≈ 7 min)")
    bouton
    return (bouton,)


@app.cell
def _(
    ModeleSSL,
    PERTES,
    SEED,
    bouton,
    chargeur_paires,
    config,
    device,
    mesure_effondrement,
    mo,
    time,
    torch,
):
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Entraîner » pour lancer l'apprentissage (≈ 7 min sur M1 Pro).*"))

    def entrainer_ssl(nom, epoques, depart=None):
        """Même encodeur, même augmentation, même optimiseur : seule la perte change.
        `depart` : modèle dont on recopie les poids (sinon, poids aléatoires)."""
        torch.manual_seed(SEED)
        modele = ModeleSSL().to(device)
        if depart is not None:
            modele.load_state_dict(depart.state_dict())
        perte_fn = PERTES[nom]
        optim = torch.optim.AdamW(modele.parameters(), lr=config["lr_max"],
                                  weight_decay=config["weight_decay"])
        sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=config["lr_max"],
                                                    total_steps=epoques * len(chargeur_paires))
        hist = {"perte": [], "ecart_type": [mesure_effondrement(modele)], "duree": []}
        t0 = time.perf_counter()
        for ep in range(epoques):
            modele.train()
            somme, n = 0.0, 0
            for v1, v2, _y in chargeur_paires:            # l'étiquette _y n'est PAS utilisée
                v = torch.cat([v1, v2]).to(device, non_blocking=True)
                _h, z = modele(v)                         # les 2N vues en un seul passage
                z1, z2 = z.chunk(2)
                perte = perte_fn(z1, z2)
                optim.zero_grad(set_to_none=True)
                perte.backward()
                optim.step()
                sched.step()
                somme += perte.item()
                n += 1
            hist["perte"].append(somme / n)
            hist["ecart_type"].append(mesure_effondrement(modele))
            hist["duree"].append(time.perf_counter() - t0)
            print(f"  {nom} époque {ep + 1:2d} : perte {somme / n:.4f}, "
                  f"écart-type × √d {hist['ecart_type'][-1]:.3f} ({hist['duree'][-1]:.0f} s)")
        return modele, hist

    t_debut = time.perf_counter()
    torch.manual_seed(SEED)
    modele_hasard = ModeleSSL().to(device)                # même architecture, jamais entraînée
    modeles, historiques = {"Encodeur au hasard": modele_hasard}, {}
    for _nom, _ep in (("SimCLR (InfoNCE)", config["epoques"]),
                      ("Barlow Twins", config["epoques"])):
        print(f"{_nom} : {_ep} époque(s)")
        modeles[_nom], historiques[_nom] = entrainer_ssl(_nom, _ep)
    # Alignement seul, EN PARTANT de l'encodeur SimCLR entraîné : on retire les négatifs à une
    # représentation saine et on regarde ce qu'elle devient.
    print(f"Alignement seul, depuis SimCLR : {config['epoques_alignement']} époque(s)")
    modeles["Alignement seul"], historiques["Alignement seul"] = entrainer_ssl(
        "Alignement seul", config["epoques_alignement"], depart=modeles["SimCLR (InfoNCE)"])
    duree_ssl = time.perf_counter() - t_debut
    print(f"Entraînements auto-supervisés : {duree_ssl:.0f} s")
    return duree_ssl, historiques, modeles, t_debut


@app.cell
def _(
    X_sonde,
    X_test,
    config,
    historiques,
    knn,
    mesure_effondrement,
    modeles,
    np,
    representations,
    sonde_lineaire,
    time,
    y_sonde,
    y_test,
):
    # --- Sonde linéaire (toutes les étiquettes de la sonde) et kNN, pour chaque encodeur ---
    _t0 = time.perf_counter()
    reps = {}
    evaluation = {}
    for _nom, _m in modeles.items():
        _ht, _he = representations(_m, X_sonde), representations(_m, X_test)
        reps[_nom] = (_ht, _he)
        evaluation[_nom] = dict(
            sonde=sonde_lineaire(_ht, y_sonde, _he, y_test),
            knn=knn(_ht, y_sonde, _he, y_test),
            ecart_type=mesure_effondrement(_m),
            # Rang effectif de h : combien de directions portent 99 % de la variance ?
            dims_99=int(np.searchsorted(np.cumsum(np.sort(np.linalg.eigvalsh(np.cov(_he.numpy().T)))[::-1])
                                        / np.trace(np.cov(_he.numpy().T)), 0.99) + 1),
            perte_finale=historiques[_nom]["perte"][-1] if _nom in historiques else None,
        )
        print(f"{_nom:>20} : sonde linéaire {evaluation[_nom]['sonde']:.1%}, kNN {evaluation[_nom]['knn']:.1%}, "
              f"écart-type × √d {evaluation[_nom]['ecart_type']:.3f}, "
              f"{evaluation[_nom]['dims_99']} dimensions de h pour 99 % de la variance")
    duree_sondes = time.perf_counter() - _t0
    print(f"Sondes : {duree_sondes:.0f} s (repère supervisé, exercice 13a : même CNN, 50 000 étiquettes, 84,6 % ; "
          f"87,4 % avec augmentation)")
    _ = config
    return duree_sondes, evaluation, reps


@app.cell
def _(
    AugmentationClassique,
    Encodeur,
    F,
    SEED,
    X_sonde,
    X_test,
    config,
    device,
    ind_rares,
    nn,
    normaliser_lot,
    reps,
    sonde_lineaire,
    time,
    torch,
    y_sonde,
    y_test,
):
    # --- Peu d'étiquettes : 100 puis 500 images étiquetées (10 puis 50 par classe) ---
    # (a) sonde linéaire sur chaque encodeur gelé, entraînée sur ces images seulement ;
    # (b) le même CNN entraîné de zéro, en supervisé, sur ces images (augmentation du ch. 13).
    _t0 = time.perf_counter()
    peu_etiquettes = {}
    for _n, _idx in ind_rares.items():
        _ind = torch.from_numpy(_idx)
        _res = {}
        for _nom in ("Encodeur au hasard", "SimCLR (InfoNCE)", "Barlow Twins"):
            _ht, _he = reps[_nom]
            _res[_nom] = sonde_lineaire(_ht[_ind], y_sonde[_ind], _he, y_test, decroissance=1e-3)

        torch.manual_seed(SEED)
        _sup = nn.Sequential(Encodeur(), nn.Linear(config["dim_h"], 10)).to(device)
        _aug = AugmentationClassique()                    # recadrage + retournement, image par image
        _Xr, _yr = X_sonde[_ind], y_sonde[_ind]
        _lot = 100
        # même nombre de pas de gradient quelle que soit la taille : 500 pas
        _epoques = config["epoques_supervise"] * 5 * _lot // len(_yr)
        _opt = torch.optim.AdamW(_sup.parameters(), lr=config["lr_max"], weight_decay=5e-4)
        _sched = torch.optim.lr_scheduler.OneCycleLR(
            _opt, max_lr=config["lr_max"], total_steps=_epoques * (len(_yr) // _lot))
        _g = torch.Generator().manual_seed(SEED)
        for _ep in range(_epoques):
            _sup.train()
            _perm = torch.randperm(len(_yr), generator=_g)
            for _i in range(0, len(_yr) - _lot + 1, _lot):
                _b = _perm[_i : _i + _lot]
                _xb = torch.stack([_aug(_Xr[j]) for j in _b]).to(device)
                _perte = F.cross_entropy(_sup(_xb), _yr[_b].to(device))
                _opt.zero_grad(set_to_none=True)
                _perte.backward()
                _opt.step()
                _sched.step()
        _sup.eval()
        _bons = 0
        with torch.no_grad():
            for _i in range(0, len(X_test), 1000):
                _bons += (_sup(normaliser_lot(X_test[_i : _i + 1000])).argmax(1).cpu()
                          == y_test[_i : _i + 1000]).sum().item()
        _res["CNN supervisé de zéro"] = _bons / len(y_test)
        peu_etiquettes[_n] = _res
        for _nom, _v in _res.items():
            print(f"{_n} étiquettes, {_nom:>22} : {_v:.1%} en test")
    duree_peu = time.perf_counter() - _t0
    print(f"({duree_peu:.0f} s)")
    return duree_peu, peu_etiquettes


@app.cell
def _(
    AugmentationSimCLR,
    F,
    SEED,
    X_test,
    config,
    device,
    modeles,
    normaliser_cifar10,
    np,
    reps,
    time,
    torch,
    y_test,
):
    # --- t-SNE des représentations h (avant / après) et matrice de similarités ---
    from sklearn.manifold import TSNE

    _t0 = time.perf_counter()
    tsne = {}
    for _nom in ("Encodeur au hasard", "SimCLR (InfoNCE)"):
        _h = F.normalize(reps[_nom][1][: config["n_tsne"]], dim=1).numpy()
        tsne[_nom] = TSNE(n_components=2, perplexity=30, init="pca",
                          random_state=SEED).fit_transform(_h).astype(np.float32)
    etiquettes_tsne = y_test[: config["n_tsne"]].numpy()

    # Huit images de test de classes différentes, deux vues augmentées chacune.
    torch.manual_seed(SEED)
    _aug = AugmentationSimCLR()
    _choix = [int(np.flatnonzero(y_test.numpy() == c)[0]) for c in range(8)]
    _v1 = torch.stack([_aug(X_test[i]) for i in _choix])
    _v2 = torch.stack([_aug(X_test[i]) for i in _choix])
    vues_exemple = torch.cat([_v1, _v2])                  # 16 vues : 8 premières, puis 8 jumelles
    classes_exemple = [int(y_test[i]) for i in _choix]
    originaux_exemple = X_test[_choix].permute(0, 2, 3, 1).numpy()   # uint8, pour la figure du site

    @torch.no_grad()
    def similarites(modele, vues):
        modele.eval()
        _hh, z = modele(vues.to(device))
        z = F.normalize(z, dim=1)
        return (z @ z.T).cpu().numpy()

    sim_exemple = {nom: similarites(modeles[nom], vues_exemple)
                   for nom in ("Encodeur au hasard", "SimCLR (InfoNCE)")}

    # Tâche prétexte sur 512 images de test jamais vues : la jumelle est-elle la plus proche
    # des 1 023 autres vues ? (exactitude « contrastive » top-1 ; le hasard donne 1/1 023)
    torch.manual_seed(SEED + 1)
    _imgs = X_test[:512]
    _w = torch.cat([torch.stack([_aug(x) for x in _imgs]), torch.stack([_aug(x) for x in _imgs])])
    jumelle_top1 = {}
    for _nom, _m in modeles.items():
        _s = similarites(_m, _w)
        np.fill_diagonal(_s, -np.inf)
        _cible = np.r_[np.arange(512, 1024), np.arange(0, 512)]
        jumelle_top1[_nom] = float((_s.argmax(1) == _cible).mean())
        print(f"{_nom:>20} : la jumelle est la plus proche dans {jumelle_top1[_nom]:.1%} des cas")

    _diag = np.eye(16, dtype=bool)
    _pos = np.roll(np.eye(16, dtype=bool), 8, axis=1)
    for _nom, _s in sim_exemple.items():
        print(f"{_nom:>20} : cosinus moyen entre jumelles {_s[_pos].mean():.2f}, "
              f"entre images différentes {_s[~_diag & ~_pos].mean():.2f}")
    duree_tsne = time.perf_counter() - _t0
    print(f"t-SNE et similarités : {duree_tsne:.0f} s")
    _ = normaliser_cifar10
    return (
        classes_exemple,
        duree_tsne,
        originaux_exemple,
        etiquettes_tsne,
        jumelle_top1,
        sim_exemple,
        tsne,
        vues_exemple,
    )


@app.cell
def _(
    CLASSES_CIFAR10,
    SUFFIXE,
    evaluation,
    historiques,
    mo,
    np,
    peu_etiquettes,
    plt,
    sauvegarder_figure,
):
    # --- Figure 1 : courbes de perte et indicateur d'effondrement ---
    _coul = {"SimCLR (InfoNCE)": "#0072B2", "Barlow Twins": "#009E73", "Alignement seul": "#D55E00"}
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 4))
    for _nom, _h in historiques.items():
        _ep = np.arange(1, len(_h["perte"]) + 1)
        _a1.plot(_ep, np.array(_h["perte"]) / _h["perte"][0], "o-", color=_coul[_nom], label=_nom)
        _a2.plot(np.arange(0, len(_h["ecart_type"])), _h["ecart_type"], "o-", color=_coul[_nom], label=_nom)
    _a1.set(xlabel="époque", ylabel="perte / perte de la 1re époque", title="Chaque méthode fait baisser SA perte")
    _a2.axhline(1, color="gray", ls=":", lw=1)
    _a2.set(xlabel="époque (0 = avant entraînement)", ylabel="écart-type de z/‖z‖ × √d",
            title="Effondrement : 0 = toutes les sorties identiques", ylim=(0, 1.1))
    _a1.legend()
    _a2.legend()
    _ch = sauvegarder_figure(_fig, "courbes" + SUFFIXE, mo.notebook_dir() / "figures")

    # --- Figure 2 : ce que vaut chaque représentation ---
    _fig2, _ax = plt.subplots(figsize=(9, 4))
    _noms = list(evaluation)
    _x = np.arange(len(_noms))
    _ax.bar(_x - 0.27, [100 * evaluation[n]["sonde"] for n in _noms], 0.27, label="sonde linéaire (toutes les étiquettes)")
    _ax.bar(_x, [100 * evaluation[n]["knn"] for n in _noms], 0.27, label="kNN (k = 20)")
    _ax.bar(_x + 0.27, [100 * peu_etiquettes[500].get(n, np.nan) for n in _noms], 0.27, label="sonde, 500 étiquettes")
    _ax.axhline(100 * peu_etiquettes[500]["CNN supervisé de zéro"], color="#D55E00", ls="--", lw=1.2,
                label="CNN supervisé de zéro, 500 étiquettes")
    _ax.axhline(10, color="gray", ls=":", lw=1)
    _ax.set_xticks(_x, _noms)
    _ax.set(ylabel="exactitude en test (%)", ylim=(0, 100), title="CIFAR-10, 10 classes, encodeur gelé")
    _ax.legend(fontsize=8, loc="upper right")
    _ch2 = sauvegarder_figure(_fig2, "sondes" + SUFFIXE, mo.notebook_dir() / "figures")
    print(f"Figures : {_ch[0]}, {_ch2[0]}")
    _ = CLASSES_CIFAR10
    _fig


@app.cell
def _(
    CLASSES_CIFAR10,
    SUFFIXE,
    classes_exemple,
    etiquettes_tsne,
    mo,
    np,
    plt,
    sauvegarder_figure,
    sim_exemple,
    tsne,
):
    # --- Figure 3 : t-SNE avant / après ; figure 4 : matrice de similarités ---
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 5))
    _cmap = plt.get_cmap("tab10")
    for _ax, (_nom, _xy) in zip(_axes, tsne.items()):
        for _c in range(10):
            _m = etiquettes_tsne == _c
            _ax.scatter(_xy[_m, 0], _xy[_m, 1], s=5, color=_cmap(_c), label=CLASSES_CIFAR10[_c])
        _ax.set(title=_nom, xticks=[], yticks=[])
    _axes[1].legend(markerscale=3, fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
    _ch = sauvegarder_figure(_fig, "tsne" + SUFFIXE, mo.notebook_dir() / "figures")

    _fig2, _axes2 = plt.subplots(1, 2, figsize=(11, 5))
    _lib = [f"{CLASSES_CIFAR10[c]}·{k}" for k in (1, 2) for c in classes_exemple]
    for _ax, (_nom, _s) in zip(_axes2, sim_exemple.items()):
        _im = _ax.imshow(_s, vmin=-1, vmax=1, cmap="RdBu_r")
        _ax.set_xticks(range(16), _lib, rotation=90, fontsize=7)
        _ax.set_yticks(range(16), _lib, fontsize=7)
        _ax.grid(False)
        _ax.set_title(_nom)
    _fig2.colorbar(_im, ax=_axes2, shrink=0.8, label="similarité cosinus de z")
    _ch2 = sauvegarder_figure(_fig2, "similarites" + SUFFIXE, mo.notebook_dir() / "figures")
    print(f"Figures : {_ch[0]}, {_ch2[0]}")
    _ = np
    _fig


@app.cell
def _(
    SMOKE,
    SUFFIXE,
    VARIANTE,
    config,
    duree_peu,
    duree_ssl,
    duree_sondes,
    duree_tsne,
    ecart_numpy_torch,
    effondrement_numpy,
    etiquettes_tsne,
    evaluation,
    historiques,
    infonce_main,
    json,
    jumelle_top1,
    mo,
    np,
    peu_etiquettes,
    siglip_main,
    sim_exemple,
    temperature,
    tsne,
    vues_exemple,
    classes_exemple,
    cosinus_main,
    originaux_exemple,
    t_debut,
    time,
):
    # --- Bilan : seuils du README (vérifiés en mode complet seulement) ---
    from ia_commun.augmentations import denormaliser_cifar10

    _sim = evaluation["SimCLR (InfoNCE)"]
    _bar = evaluation["Barlow Twins"]
    _ali = evaluation["Alignement seul"]
    _has = evaluation["Encodeur au hasard"]
    _n2 = 2 * config["lot"] - 1
    bilan = dict(
        config=config,
        numpy_torch=ecart_numpy_torch,
        infonce_main=infonce_main,
        siglip_main=siglip_main,
        temperature=temperature,
        effondrement_numpy=effondrement_numpy,
        evaluation=evaluation,
        peu_etiquettes=peu_etiquettes,
        jumelle_top1=jumelle_top1,
        historiques=historiques,
        borne_info_mutuelle=dict(
            candidates=_n2, log_candidates=float(np.log(_n2)),
            perte_finale=_sim["perte_finale"], borne=float(np.log(_n2) - _sim["perte_finale"])),
        durees=dict(ssl=duree_ssl, sondes=duree_sondes, peu=duree_peu, tsne=duree_tsne,
                    total_depuis_entrainement=time.perf_counter() - t_debut),
    )
    print(f"Borne InfoNCE : I(vue 1 ; vue 2) ≥ log({_n2}) − perte = {np.log(_n2):.3f} − "
          f"{_sim['perte_finale']:.3f} = {bilan['borne_info_mutuelle']['borne']:.3f} nat "
          f"(plafond log({_n2}) = {np.log(_n2):.2f})")
    _dossier = mo.notebook_dir() / "figures"
    _dossier.mkdir(exist_ok=True)
    _vues = (denormaliser_cifar10(vues_exemple).permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
    np.savez_compressed(
        _dossier / f"resultats{SUFFIXE}.npz",
        bilan=json.dumps(bilan, ensure_ascii=False),
        tsne_hasard=tsne["Encodeur au hasard"], tsne_simclr=tsne["SimCLR (InfoNCE)"],
        etiquettes_tsne=etiquettes_tsne,
        sim_hasard=sim_exemple["Encodeur au hasard"], sim_simclr=sim_exemple["SimCLR (InfoNCE)"],
        vues=_vues, originaux=originaux_exemple, classes_vues=np.array(classes_exemple),
        cosinus_main=cosinus_main,
    )

    SEUILS = [
        ("NumPy = PyTorch (InfoNCE, SigLIP)", ecart_numpy_torch, "≤", 1e-9),
        ("sonde SimCLR", _sim["sonde"], "≥", 0.60),
        ("sonde Barlow Twins", _bar["sonde"], "≥", 0.55),
        ("gain SimCLR sur l'encodeur au hasard (points)", 100 * (_sim["sonde"] - _has["sonde"]), "≥", 10),
        ("écart-type × √d, SimCLR (pas d'effondrement)", _sim["ecart_type"], "≥", 0.5),
        ("écart-type × √d, Barlow Twins (pas d'effondrement)", _bar["ecart_type"], "≥", 0.5),
        ("écart-type × √d, alignement seul (effondrement)", _ali["ecart_type"], "≤", 0.1),
        ("100 étiquettes : SimCLR − supervisé de zéro (points)",
         100 * (peu_etiquettes[100]["SimCLR (InfoNCE)"] - peu_etiquettes[100]["CNN supervisé de zéro"]), "≥", 5),
        ("500 étiquettes : SimCLR − supervisé de zéro (points)",
         100 * (peu_etiquettes[500]["SimCLR (InfoNCE)"] - peu_etiquettes[500]["CNN supervisé de zéro"]), "≥", 0),
    ]
    _lignes, _echecs = [], []
    for _nom, _v, _op, _s in SEUILS:
        _ok = _v <= _s if _op == "≤" else _v >= _s
        _lignes.append(f"| {_nom} | {_v:.4g} | {_op} {_s:g} | {'oui' if _ok else 'NON'} |")
        if not _ok:
            _echecs.append(_nom)
    print("\n".join(_lignes))
    if not SMOKE and not VARIANTE and _echecs:
        raise SystemExit(f"Seuils non atteints : {', '.join(_echecs)}")
    mo.md("**Bilan**\n\n| Contrôle | Valeur | Seuil | Atteint |\n|---|---|---|---|\n" + "\n".join(_lignes))


if __name__ == "__main__":
    app.run()
