# Exercice 39a : se fier (ou non) à une prédiction. Calibration, incertitude, entrées hors
# distribution et exemples adversariaux, sur le petit CNN CIFAR-10 du chapitre 13.
#
#   make ex NUM=39a              → éditeur marimo (bouton « Entraîner »)
#   make ex NUM=39a MODE=run     → script complet, vérifie les seuils du README
#   make ex NUM=39a MODE=smoke   → test rapide (< 30 s)
#
# Partie A en NumPy : ECE, diagramme de fiabilité, ajustement de la température, prédiction
# conforme, AUROC, et l'argument « linéaire » de l'attaque FGSM, chaque fois vérifiés contre
# une bibliothèque (PyTorch, scikit-learn). Parties B à F en PyTorch sur CIFAR-10 (MPS).
# Le CNN est celui de l'exercice 13a, en version courte (8 époques, sans augmentation),
# entraîné trois fois avec trois graines (ensemble profond).

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import math
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
        math,
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
    # Exercice 39a : quand se fier à une prédiction ?

    Un classifieur renvoie une classe **et** une probabilité. La classe peut être juste ou
    fausse ; la probabilité, elle, devrait dire *à quel point* se fier à la classe. Ce
    notebook mesure si c'est le cas, et ce qu'on peut faire quand ça ne l'est pas.

    - **A. Les outils en NumPy** : erreur de calibration (ECE), diagramme de fiabilité,
      mise à l'échelle par la température, prédiction conforme, AUROC, et l'argument
      linéaire de Goodfellow et al. (2014) qui explique les exemples adversariaux.
    - **B. Trois CNN sur CIFAR-10** (le petit CNN de l'exercice 13a, 8 époques, sans
      augmentation), chacun avec sa graine : un modèle seul et un **ensemble profond**.
    - **C. Calibration** : ECE et log-vraisemblance négative (NLL) avant et après la
      température, sur le test et sur des images bruitées (décalage de distribution).
    - **D. Prédiction conforme** : des *ensembles* de classes qui contiennent la bonne
      classe 90 % du temps, garantie comprise.
    - **E. Hors distribution** : le modèle reconnaît-il qu'un chiffre MNIST ou du bruit
      n'est pas une image CIFAR-10 ?
    - **F. Exemples adversariaux** : FGSM et PGD, des perturbations invisibles qui font
      tomber l'exactitude, et leur transfert d'un modèle à l'autre.

    **Pertes.** L'entraînement minimise l'entropie croisée (chapitre 01), qui est aussi la
    NLL. La température est choisie en minimisant la même NLL sur des données mises de côté
    (un seul paramètre). L'ECE n'est **pas** une perte : c'est une métrique (non
    différentiable, dépend du découpage en intervalles). FGSM et PGD **maximisent**
    l'entropie croisée, par rapport à l'image et non aux poids.
    """)


@app.cell
def _(get_device, is_smoke, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    device = get_device()
    style_commun()
    config = dict(
        n_train=2000 if SMOKE else 45000,   # entraînement ; les 5 000 autres images = calibration
        n_calib=500 if SMOKE else 5000,     # sert à la température ET à la prédiction conforme
        n_test=500 if SMOKE else 10000,
        n_modeles=2 if SMOKE else 3,        # taille de l'ensemble profond (graines 42, 43, 44)
        epoques=1 if SMOKE else 8,
        lot=128,
        lr_max=2e-3,                        # AdamW + « one cycle », comme l'exercice 13a
        weight_decay=5e-4,
        n_intervalles=15,                   # intervalles de confiance de l'ECE (Guo et al. 2017)
        alpha=0.10,                         # prédiction conforme : 90 % de couverture visée
        sigma_bruit=0.10,                   # décalage de distribution : bruit gaussien (pixels dans [0, 1])
        n_ood=500 if SMOKE else 2000,       # images MNIST et images de bruit
        n_attaque=200 if SMOKE else 1000,   # images de test attaquées
        epsilons=[0, 1, 2, 4, 8],           # amplitude de l'attaque, en niveaux de gris sur 255
        pas_pgd=10,
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, périphérique {device}")
    print(config)
    return SEED, SMOKE, config, device


@app.cell
def _(mo):
    mo.md(r"""
    ## A. Les outils, en NumPy

    **Calibration.** Un modèle est *calibré* si, parmi toutes les prédictions faites avec
    une confiance de 80 %, 80 % sont justes. On range les prédictions en $M$ intervalles
    de confiance $B_1, \dots, B_M$ (ici 15 de largeur égale) et on compare, dans chacun,
    l'exactitude moyenne à la confiance moyenne :

    $$\mathrm{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n}\,\bigl|\,\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\,\bigr|$$

    **Température.** On divise les logits par un scalaire $T > 0$ avant le softmax :
    $\hat p = \mathrm{softmax}(z / T)$. $T > 1$ aplatit les probabilités, $T < 1$ les
    durcit ; la classe prédite (l'argmax) ne change pas, donc l'exactitude non plus. $T$
    est choisi en minimisant la NLL sur un jeu de **calibration** que le modèle n'a pas vu.

    **Prédiction conforme** (version « split »). Sur $n$ exemples de calibration, score de
    non-conformité $s_i = 1 - \hat p_{y_i}(x_i)$ ; $\hat q$ = le quantile empirique de
    niveau $\lceil (n+1)(1-\alpha) \rceil / n$ des $s_i$. Pour une nouvelle image,
    l'ensemble prédit contient toutes les classes $k$ telles que $1 - \hat p_k \le \hat q$.
    Si calibration et test sont échangeables, la bonne classe est dans l'ensemble avec une
    probabilité **au moins** $1 - \alpha$, quel que soit le modèle.

    **AUROC.** Probabilité qu'un exemple positif tiré au hasard ait un score plus haut
    qu'un négatif tiré au hasard (0,5 = hasard, 1 = séparation parfaite).
    """)


@app.cell
def _(math, np):
    def softmax(z, T=1.0):
        """Softmax stable des logits z (n, K), divisés par la température T."""
        z = np.asarray(z, dtype=np.float64) / T
        z = z - z.max(axis=1, keepdims=True)     # évite exp(1000) = inf
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def nll(p, y):
        """Log-vraisemblance négative moyenne = entropie croisée (chapitre 01)."""
        return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())

    def intervalles_fiabilite(p, y, M=15):
        """Pour chaque intervalle de confiance : effectif, confiance moyenne, exactitude."""
        conf = p.max(axis=1)
        juste = (p.argmax(axis=1) == y).astype(np.float64)
        bornes = np.linspace(0, 1, M + 1)
        # intervalle m = ]bornes[m], bornes[m+1]] ; une confiance de 1/K tombe dans le premier
        m = np.clip(np.searchsorted(bornes, conf, side="left") - 1, 0, M - 1)
        n_m = np.bincount(m, minlength=M)
        conf_m = np.bincount(m, weights=conf, minlength=M) / np.maximum(n_m, 1)
        acc_m = np.bincount(m, weights=juste, minlength=M) / np.maximum(n_m, 1)
        return n_m, conf_m, acc_m

    def ece(p, y, M=15):
        n_m, conf_m, acc_m = intervalles_fiabilite(p, y, M)
        return float((n_m / n_m.sum() * np.abs(acc_m - conf_m)).sum())

    def ajuster_temperature(z, y):
        """Température qui minimise la NLL, par recherche du nombre d'or sur log T.
        La NLL est convexe en 1/T (log-somme-exp d'une fonction linéaire) : un seul minimum."""
        f = lambda logT: nll(softmax(z, math.exp(logT)), y)
        a, b = math.log(0.05), math.log(20.0)
        r = (math.sqrt(5) - 1) / 2
        c, d = b - r * (b - a), a + r * (b - a)
        for _ in range(80):                       # 80 itérations : intervalle × 0,618⁸⁰
            if f(c) < f(d):
                b, d = d, c
                c = b - r * (b - a)
            else:
                a, c = c, d
                d = a + r * (b - a)
        return math.exp((a + b) / 2)

    def seuil_conforme(p_calib, y_calib, alpha):
        """Quantile corrigé des scores 1 − p(vraie classe) sur le jeu de calibration."""
        s = 1 - p_calib[np.arange(len(y_calib)), y_calib]
        n = len(s)
        niveau = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
        return float(np.quantile(s, niveau, method="higher"))

    def ensembles_conformes(p, q):
        """Masque (n, K) : la classe k est dans l'ensemble si 1 − p_k ≤ q."""
        return (1 - p) <= q

    def auroc(score_pos, score_neg):
        """AUROC par la statistique de Mann-Whitney (rangs moyens en cas d'égalité)."""
        s = np.concatenate([score_pos, score_neg])
        ordre = s.argsort(kind="mergesort")
        rangs = np.empty(len(s))
        rangs[ordre] = np.arange(1, len(s) + 1)
        # égalités : rang moyen du groupe
        _, inverse, compte = np.unique(s, return_inverse=True, return_counts=True)
        somme = np.bincount(inverse, weights=rangs)
        rangs = (somme / compte)[inverse]
        n1, n0 = len(score_pos), len(score_neg)
        return float((rangs[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    return (
        ajuster_temperature,
        auroc,
        ece,
        ensembles_conformes,
        intervalles_fiabilite,
        nll,
        seuil_conforme,
        softmax,
    )


@app.cell
def _(F, ajuster_temperature, auroc, ece, ensembles_conformes, nll, np, seuil_conforme, softmax, torch):
    # --- A1. Un exemple à la main : 10 prédictions, 4 intervalles (vérifiable au crayon) ---
    _p_max = np.array([0.95, 0.92, 0.91, 0.90, 0.85, 0.80, 0.70, 0.65, 0.60, 0.55])
    _juste = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    # Probabilités à 2 classes : la classe 0 est prédite avec la confiance p_max (≥ 0,5) ;
    # la vraie classe vaut 0 si la prédiction est juste, 1 sinon.
    _p2 = np.stack([_p_max, 1 - _p_max], axis=1)
    _y2 = np.where(_juste == 1, 0, 1)
    ece_main = ece(_p2, _y2, M=4)                  # intervalles ]0;0,25] ]0,25;0,5] ]0,5;0,75] ]0,75;1]
    print("A1. Dix prédictions, quatre intervalles de confiance :")
    print("  ]0,75 ; 1] : 6 prédictions, confiance moyenne 0,888, exactitude 4/6 = 0,667, écart 0,222")
    print("  ]0,5 ; 0,75] : 4 prédictions, confiance moyenne 0,625, exactitude 2/4 = 0,500, écart 0,125")
    print(f"  ECE = 6/10 × 0,222 + 4/10 × 0,125 = {ece_main:.4f}")

    # --- A2. Vérifications contre les bibliothèques sur des logits aléatoires ---
    _rng = np.random.default_rng(42)
    _z = _rng.normal(0, 3, size=(2000, 10))
    _y = np.array([_rng.choice(10, p=softmax(_z[i:i + 1], 1.5)[0]) for i in range(2000)])
    ecart_nll_torch = abs(nll(softmax(_z), _y) - float(F.cross_entropy(torch.tensor(_z), torch.tensor(_y))))
    # Ces étiquettes sont tirées avec T = 1,5 : la température retrouvée doit être proche de 1,5
    T_jouet = ajuster_temperature(_z, _y)
    # Même ajustement « à la manière des bibliothèques » : L-BFGS de PyTorch sur log T
    _logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    _opt = torch.optim.LBFGS([_logT], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")
    _zt, _yt = torch.tensor(_z), torch.tensor(_y)

    def _ferme():
        _opt.zero_grad()
        _perte = F.cross_entropy(_zt / _logT.exp(), _yt)
        _perte.backward()
        return _perte

    _opt.step(_ferme)
    T_lbfgs = float(_logT.detach().exp())
    from sklearn.metrics import roc_auc_score

    _pos, _neg = _rng.normal(1, 1, 500), _rng.normal(0, 1, 700)
    _pos[:50] = 0.0
    _neg[:50] = 0.0                                 # quelques égalités, pour le rang moyen
    ecart_auroc = abs(auroc(_pos, _neg) - roc_auc_score(np.r_[np.ones(500), np.zeros(700)], np.r_[_pos, _neg]))
    # Conforme : sur des données échangeables, la couverture vaut ≈ 1 − α
    _pc = softmax(_z[:1000], 1.0)
    _q = seuil_conforme(_pc, _y[:1000], 0.1)
    couverture_jouet = float(ensembles_conformes(softmax(_z[1000:]), _q)[np.arange(1000), _y[1000:]].mean())
    print(f"\nA2. NLL NumPy − F.cross_entropy : {ecart_nll_torch:.1e}")
    print(f"    Température retrouvée (vraie : 1,5) : nombre d'or {T_jouet:.4f}, L-BFGS PyTorch {T_lbfgs:.4f}")
    print(f"    AUROC NumPy − sklearn.roc_auc_score : {ecart_auroc:.1e}")
    print(f"    Prédiction conforme, α = 0,1 : couverture {couverture_jouet:.1%} sur 1 000 exemples jouets")
    return T_jouet, T_lbfgs, couverture_jouet, ece_main, ecart_auroc, ecart_nll_torch


@app.cell
def _(np):
    # --- A3. L'argument linéaire de Goodfellow et al. (2014) ---
    # Score linéaire w·x. Perturbation η = ε · signe(w), donc chaque pixel bouge d'au plus ε
    # (invisible si ε = 2/255). Le score bouge de w·η = ε · Σ|w_i| = ε · n · moyenne(|w|) :
    # proportionnel à la DIMENSION n. Une image CIFAR-10 a n = 3 × 32 × 32 = 3 072 valeurs.
    variation_lineaire = {}
    for _n in [10, 100, 3072, 150528]:              # 150 528 = 3 × 224 × 224 (ImageNet)
        _w = np.full(_n, 0.04)                     # tous les poids à 0,04 : calcul vérifiable à la main
        _eps = 2 / 255
        variation_lineaire[_n] = float(_eps * np.abs(_w).sum())   # = ε × n × 0,04
    print("A3. Variation du score linéaire w·x pour ε = 2/255 par pixel (|w| = 0,04) :")
    for _n, _v in variation_lineaire.items():
        print(f"    n = {_n:>7,} valeurs → score décalé de {_v:8.3f}".replace(",", " "))
    return (variation_lineaire,)


@app.cell
def _(mo):
    mo.md(r"""
    ## B. Trois CNN sur CIFAR-10

    Le petit CNN de l'exercice 13a (6 convolutions, BatchNorm, ReLU, 290 282 paramètres),
    en version courte : 8 époques, **sans** augmentation de données, sur 45 000 images ;
    les 5 000 autres images d'entraînement forment le **jeu de calibration**. Les images
    restent en pixels dans $[0, 1]$ ; la normalisation est la première couche du modèle,
    pour que les attaques de la partie F travaillent directement sur les pixels.
    """)


@app.cell
def _(config, np, torch):
    from ia_commun.augmentations import ECART_TYPE_CIFAR10, MOYENNE_CIFAR10
    from ia_commun.donnees import CIFAR10, CLASSES_CIFAR10, mnist_tenseurs

    def pixels(ds):
        """Images en float32 dans [0, 1], format (N, 3, 32, 32), et étiquettes."""
        X = torch.from_numpy(np.array(ds.images[ds.indices])).permute(0, 3, 1, 2).float() / 255
        return X.contiguous(), torch.from_numpy(ds.labels[ds.indices]).long()

    _perm = np.random.default_rng(42).permutation(50000)
    _idx_train = np.sort(_perm[: config["n_train"]])
    _idx_calib = np.sort(_perm[45000: 45000 + config["n_calib"]])   # jamais vues à l'entraînement
    X_train, y_train = pixels(CIFAR10("train", indices=_idx_train))
    X_calib, y_calib = pixels(CIFAR10("train", indices=_idx_calib))
    X_test, y_test = pixels(CIFAR10("test", indices=np.arange(config["n_test"])))

    # Hors distribution : chiffres MNIST (28×28 → marge de 2 pixels → 32×32, gris → 3 canaux)
    _Xm, _ = mnist_tenseurs(train=False)
    X_mnist = torch.nn.functional.pad(_Xm[: config["n_ood"]], (2, 2, 2, 2)).repeat(1, 3, 1, 1)
    # … et du bruit uniforme
    X_bruit_ood = torch.rand(config["n_ood"], 3, 32, 32, generator=torch.Generator().manual_seed(42))
    # Décalage de distribution : les images de test avec un bruit gaussien d'écart type σ
    X_test_bruit = (X_test + config["sigma_bruit"] * torch.randn(
        X_test.shape, generator=torch.Generator().manual_seed(42))).clamp(0, 1)
    MOY = torch.tensor(MOYENNE_CIFAR10).view(1, 3, 1, 1)
    ECT = torch.tensor(ECART_TYPE_CIFAR10).view(1, 3, 1, 1)
    print(f"Entraînement {tuple(X_train.shape)}, calibration {tuple(X_calib.shape)}, "
          f"test {tuple(X_test.shape)}, MNIST {tuple(X_mnist.shape)}")
    return (
        CLASSES_CIFAR10,
        ECT,
        MOY,
        X_bruit_ood,
        X_calib,
        X_mnist,
        X_test,
        X_test_bruit,
        X_train,
        y_calib,
        y_test,
        y_train,
    )


@app.cell
def _(ECT, MOY, nn):
    def bloc(c_in, c_out, k=3):
        return nn.Sequential(nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False),
                             nn.BatchNorm2d(c_out), nn.ReLU())

    class PetitCNN(nn.Module):
        """Le CNN de l'exercice 13a, précédé de la normalisation par canal."""

        def __init__(self):
            super().__init__()
            self.register_buffer("moy", MOY.clone())
            self.register_buffer("ect", ECT.clone())
            self.extraire = nn.Sequential(
                bloc(3, 32, 5), bloc(32, 32), nn.MaxPool2d(2),
                bloc(32, 64), bloc(64, 64), nn.MaxPool2d(2),
                bloc(64, 128), bloc(128, 128),
            )
            self.classer = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10))

        def forward(self, x):                 # x : pixels dans [0, 1]
            return self.classer(self.extraire((x - self.moy) / self.ect))

    return (PetitCNN,)


@app.cell
def _(mo):
    bouton = mo.ui.run_button(label="Entraîner les trois CNN (≈ 4 à 5 min)")
    bouton
    return (bouton,)


@app.cell
def _(
    F,
    PetitCNN,
    SEED,
    X_train,
    bouton,
    config,
    device,
    mo,
    time,
    torch,
    y_train,
):
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Entraîner » pour lancer les entraînements.*"))

    def entrainer(graine):
        torch.manual_seed(graine)
        modele = PetitCNN().to(device)
        optim = torch.optim.AdamW(modele.parameters(), lr=config["lr_max"],
                                  weight_decay=config["weight_decay"])
        n_lots = (len(X_train) + config["lot"] - 1) // config["lot"]
        sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=config["lr_max"],
                                                    total_steps=config["epoques"] * n_lots)
        g = torch.Generator().manual_seed(graine)
        for ep in range(config["epoques"]):
            modele.train()
            perm = torch.randperm(len(X_train), generator=g)
            somme, bons = 0.0, 0
            for i in range(0, len(X_train), config["lot"]):
                j = perm[i: i + config["lot"]]
                xb, yb = X_train[j].to(device), y_train[j].to(device)
                logits = modele(xb)
                perte = F.cross_entropy(logits, yb)
                optim.zero_grad(set_to_none=True)
                perte.backward()
                optim.step()
                sched.step()
                somme += perte.item() * len(j)
                bons += (logits.argmax(1) == yb).sum().item()
            print(f"  graine {graine}, époque {ep + 1} : perte {somme / len(X_train):.3f}, "
                  f"exactitude d'entraînement {bons / len(X_train):.1%}")
        return modele.eval()

    _t0 = time.perf_counter()
    modeles = []
    for _k in range(config["n_modeles"]):
        modeles.append(entrainer(SEED + _k))
    duree_entrainement = time.perf_counter() - _t0
    print(f"Entraînements : {duree_entrainement:.0f} s")
    return duree_entrainement, modeles


@app.cell
def _(
    X_bruit_ood,
    X_calib,
    X_mnist,
    X_test,
    X_test_bruit,
    device,
    modeles,
    np,
    torch,
):
    def logits_de(modele, X, lot=1000):
        """Logits (NumPy float64) d'un modèle sur des tenseurs en mémoire."""
        sorties = []
        with torch.no_grad():
            for i in range(0, len(X), lot):
                sorties.append(modele(X[i: i + lot].to(device)).float().cpu())
        return torch.cat(sorties).double().numpy()

    # logits[jeu] : tableau (n_modeles, n, 10)
    jeux = {"calib": X_calib, "test": X_test, "test_bruit": X_test_bruit,
            "mnist": X_mnist, "bruit": X_bruit_ood}
    logits = {nom: np.stack([logits_de(m, X) for m in modeles]) for nom, X in jeux.items()}
    print({nom: v.shape for nom, v in logits.items()})
    return logits, logits_de


@app.cell
def _(mo):
    mo.md(r"""
    ## C. Calibration : le modèle est-il trop sûr de lui ?

    Trois façons d'obtenir des probabilités : le **modèle seul** (graine 42), le même
    après **température** ajustée sur le jeu de calibration, et l'**ensemble** (moyenne
    des probabilités des trois modèles, Lakshminarayanan et al., 2017). On compare sur le
    test d'origine et sur le test bruité (σ = 0,1 sur des pixels dans [0, 1]).
    """)


@app.cell
def _(ajuster_temperature, config, ece, intervalles_fiabilite, logits, nll, np, softmax, y_calib, y_test):
    _yc, _yt = y_calib.numpy(), y_test.numpy()
    T = ajuster_temperature(logits["calib"][0], _yc)

    def probas(jeu, methode):
        if methode == "seul":
            return softmax(logits[jeu][0])
        if methode == "température":
            return softmax(logits[jeu][0], T)
        return np.mean([softmax(z) for z in logits[jeu]], axis=0)    # ensemble

    calibration = {}
    for _jeu in ["test", "test_bruit"]:
        for _m in ["seul", "température", "ensemble"]:
            _p = probas(_jeu, _m)
            calibration[f"{_jeu}/{_m}"] = {
                "exactitude": float((_p.argmax(1) == _yt).mean()),
                "confiance": float(_p.max(1).mean()),
                "ece": ece(_p, _yt, config["n_intervalles"]),
                "nll": nll(_p, _yt),
            }
    # 10 intervalles pour les diagrammes (plus lisibles que 15)
    fiabilite = {f"{j}/{m}": [v.tolist() for v in intervalles_fiabilite(probas(j, m), _yt, 10)]
                 for j in ["test", "test_bruit"] for m in ["seul", "température", "ensemble"]}
    print(f"Température ajustée sur {len(_yc)} images de calibration : T = {T:.3f}")
    print(f"{'jeu / méthode':<26} {'exact.':>7} {'conf.':>7} {'ECE':>7} {'NLL':>7}")
    for _k, _v in calibration.items():
        print(f"{_k:<26} {_v['exactitude']:>7.1%} {_v['confiance']:>7.1%} {_v['ece']:>7.2%} {_v['nll']:>7.3f}")
    return T, calibration, fiabilite, probas


@app.cell
def _(SMOKE, calibration, fiabilite, mo, np, plt, sauvegarder_figure):
    # --- Diagramme de fiabilité (10 intervalles pour la lisibilité) ---
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    _centres = (np.arange(10) + 0.5) / 10
    for _ax, _m in zip(_axes, ["seul", "température", "ensemble"]):
        _n, _conf, _acc = (np.array(v) for v in fiabilite["test/" + _m])
        _ok = _n > 0
        _ax.bar(_centres[_ok], _acc[_ok], width=0.1, edgecolor="k", lw=0.5, alpha=0.8, label="exactitude")
        _ax.plot([0, 1], [0, 1], "k--", lw=1, label="calibration parfaite")
        _ax.set(title=f"{_m} : ECE {calibration['test/' + _m]['ece']:.1%}",
                xlabel="confiance", xlim=(0, 1), ylim=(0, 1))
    _axes[0].set_ylabel("exactitude dans l'intervalle")
    _axes[0].legend(loc="upper left", fontsize=9)
    _fig.tight_layout()
    sauvegarder_figure(_fig, "fiabilite-smoke" if SMOKE else "fiabilite", mo.notebook_dir() / "figures")
    _fig


@app.cell
def _(mo):
    mo.md(r"""
    ## D. Prédiction conforme : un ensemble de classes garanti

    Au lieu d'une classe, on renvoie l'ensemble des classes dont la probabilité dépasse
    $1 - \hat q$. Le seuil $\hat q$ est réglé sur le jeu de calibration pour que la bonne
    classe soit dans l'ensemble au moins 90 % du temps. On utilise les probabilités après
    température (la garantie tient avec n'importe quelles probabilités, mais de meilleures
    probabilités donnent des ensembles plus petits).
    """)


@app.cell
def _(T, config, ensembles_conformes, logits, np, seuil_conforme, softmax, y_calib, y_test):
    _yc, _yt = y_calib.numpy(), y_test.numpy()
    _pc = softmax(logits["calib"][0], T)
    _pt = softmax(logits["test"][0], T)
    q_conforme = seuil_conforme(_pc, _yc, config["alpha"])
    _ens = ensembles_conformes(_pt, q_conforme)
    tailles = _ens.sum(1)
    _couvre = _ens[np.arange(len(_yt)), _yt]
    _juste = _pt.argmax(1) == _yt
    _pb = softmax(logits["test_bruit"][0], T)
    _ens_b = ensembles_conformes(_pb, q_conforme)
    conforme = {
        "q": q_conforme,
        "seuil_proba": 1 - q_conforme,
        "couverture": float(_couvre.mean()),
        "taille_moyenne": float(tailles.mean()),
        "repartition_tailles": np.bincount(tailles, minlength=11).tolist(),
        "taille_si_juste": float(tailles[_juste].mean()),
        "taille_si_faux": float(tailles[~_juste].mean()),
        "couverture_bruit": float(_ens_b[np.arange(len(_yt)), _yt].mean()),
        "taille_bruit": float(_ens_b.sum(1).mean()),
        "vides": int((tailles == 0).sum()),
    }
    print(f"Seuil : q = {q_conforme:.4f}, soit garder les classes de probabilité ≥ {1 - q_conforme:.4f}")
    print(f"Couverture sur le test : {conforme['couverture']:.1%} (visée : ≥ {1 - config['alpha']:.0%})")
    print(f"Taille moyenne de l'ensemble : {conforme['taille_moyenne']:.2f} classe(s) ; "
          f"{conforme['taille_si_juste']:.2f} quand l'argmax est juste, {conforme['taille_si_faux']:.2f} quand il est faux")
    print(f"Répartition des tailles (0 à 10) : {conforme['repartition_tailles']}")
    print(f"Test bruité (même seuil, données plus échangeables) : couverture {conforme['couverture_bruit']:.1%}, "
          f"taille {conforme['taille_bruit']:.2f}")
    return (conforme,)


@app.cell
def _(mo):
    mo.md(r"""
    ## E. Hors distribution : reconnaître ce qu'on n'a jamais vu

    Le modèle n'a vu que des photos des 10 classes de CIFAR-10. On lui montre des chiffres
    MNIST et du bruit uniforme. Il répondra forcément une des 10 classes ; la question est
    de savoir si sa **confiance** (la probabilité maximale, ou *MSP*, référence de Hendrycks
    et Gimpel, 2016) est plus basse que sur de vraies images CIFAR-10. On mesure la
    séparation par l'AUROC (positif = image CIFAR-10).
    """)


@app.cell
def _(T, auroc, logits, np, softmax):
    hors_distribution = {}
    msp = {}
    for _m in ["seul", "ensemble"]:
        def _p(jeu, m=_m):
            if m == "seul":
                return softmax(logits[jeu][0], T)
            return np.mean([softmax(z) for z in logits[jeu]], axis=0)
        _in = _p("test").max(1)
        msp[_m] = {"cifar10": _in}
        for _jeu in ["mnist", "bruit"]:
            _out = _p(_jeu).max(1)
            msp[_m][_jeu] = _out
            hors_distribution[f"{_m}/{_jeu}"] = {
                "auroc": auroc(_in, _out),
                "msp_moyenne": float(_out.mean()),
                "part_confiance_90": float((_out >= 0.9).mean()),
                "classe_majoritaire": int(np.bincount(_p(_jeu).argmax(1), minlength=10).argmax()),
                "part_classe_majoritaire": float(np.bincount(_p(_jeu).argmax(1), minlength=10).max() / len(_out)),
            }
        hors_distribution[f"{_m}/cifar10"] = {"msp_moyenne": float(_in.mean()),
                                              "part_confiance_90": float((_in >= 0.9).mean())}
    for _k, _v in hors_distribution.items():
        print(_k, {a: round(b, 4) if isinstance(b, float) else b for a, b in _v.items()})
    return hors_distribution, msp


@app.cell
def _(CLASSES_CIFAR10, SMOKE, hors_distribution, mo, msp, np, plt, sauvegarder_figure):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    _bins = np.linspace(0.1, 1, 37)
    for _ax, _m in zip(_axes, ["seul", "ensemble"]):
        for _jeu, _c, _lib in [("cifar10", "#0072B2", "CIFAR-10 (test)"), ("mnist", "#D55E00", "MNIST"),
                               ("bruit", "#009E73", "bruit uniforme")]:
            _ax.hist(msp[_m][_jeu], bins=_bins, density=True, alpha=0.55, color=_c, label=_lib)
        _ax.set(title=f"{_m} : AUROC MNIST {hors_distribution[_m + '/mnist']['auroc']:.2f}, "
                      f"bruit {hors_distribution[_m + '/bruit']['auroc']:.2f}",
                xlabel="probabilité maximale (MSP)")
    _axes[0].set_ylabel("densité")
    _axes[0].legend(fontsize=9)
    _fig.tight_layout()
    sauvegarder_figure(_fig, "hors-distribution-smoke" if SMOKE else "hors-distribution",
                       mo.notebook_dir() / "figures")
    print("Classe la plus prédite pour MNIST (modèle seul) :",
          CLASSES_CIFAR10[hors_distribution["seul/mnist"]["classe_majoritaire"]])
    _fig


@app.cell
def _(mo):
    mo.md(r"""
    ## F. Exemples adversariaux : FGSM et PGD

    **FGSM** (Goodfellow et al., 2014) : un seul pas, dans la direction qui fait le plus
    monter la perte, chaque pixel bougeant d'exactement $\varepsilon$ :
    $x' = \mathrm{clip}_{[0,1]}\bigl(x + \varepsilon\,\mathrm{signe}(\nabla_x \mathcal{L}(x, y))\bigr)$.

    **PGD** (Madry et al., 2017) : départ aléatoire dans la boule
    $\|x' - x\|_\infty \le \varepsilon$, puis 10 petits pas de FGSM (taille
    $\varepsilon/4$), chacun suivi d'une **projection** dans la boule. Plus coûteux,
    beaucoup plus fort.

    $\varepsilon$ est exprimé en niveaux de gris : $\varepsilon = 8/255$ veut dire que
    chaque pixel bouge d'au plus 8 niveaux sur 255. Les attaques visent le modèle seul ;
    on teste aussi les images produites sur les deux autres modèles (**transfert**).
    """)


@app.cell
def _(F, torch):
    def fgsm(modele, x, y, eps):
        """Un pas de signe du gradient de la perte par rapport à l'IMAGE."""
        x = x.clone().requires_grad_(True)
        perte = F.cross_entropy(modele(x), y)
        (grad,) = torch.autograd.grad(perte, x)
        return (x + eps * grad.sign()).clamp(0, 1).detach()

    def pgd(modele, x, y, eps, pas, n_pas, generateur):
        """Montée de gradient projetée dans la boule L∞ de rayon eps, départ aléatoire."""
        bruit = (torch.rand(x.shape, generator=generateur) * 2 - 1).to(x.device)
        xa = (x + eps * bruit).clamp(0, 1)
        for _ in range(n_pas):
            xa.requires_grad_(True)
            perte = F.cross_entropy(modele(xa), y)
            (grad,) = torch.autograd.grad(perte, xa)
            xa = xa.detach() + pas * grad.sign()
            xa = torch.min(torch.max(xa, x - eps), x + eps).clamp(0, 1)   # projection
        return xa.detach()

    return fgsm, pgd


@app.cell
def _(F, X_test, config, device, fgsm, modeles, pgd, time, torch, y_test):
    _n = config["n_attaque"]
    _x, _y = X_test[:_n].to(device), y_test[:_n].to(device)
    _cible = modeles[0]
    _gen = torch.Generator().manual_seed(42)

    def evaluer_attaque(xa):
        with torch.no_grad():
            sortie = _cible(xa)
            res = {"exactitude": float((sortie.argmax(1) == _y).float().mean()),
                   "perte": float(F.cross_entropy(sortie, _y)),
                   "confiance": float(sortie.softmax(1).max(1).values.mean())}
            # transfert : mêmes images, autres modèles
            res["transfert"] = [float((m(xa).argmax(1) == _y).float().mean()) for m in modeles[1:]]
        return res

    _t0 = time.perf_counter()
    attaques = {"fgsm": {}, "pgd": {}}
    images_adv = {}
    for _e in config["epsilons"]:
        _eps = _e / 255
        _xf = fgsm(_cible, _x, _y, _eps) if _e > 0 else _x
        _xp = pgd(_cible, _x, _y, _eps, _eps / 4, config["pas_pgd"], _gen) if _e > 0 else _x
        attaques["fgsm"][_e] = evaluer_attaque(_xf)
        attaques["pgd"][_e] = evaluer_attaque(_xp)
        images_adv[_e] = (_xf[:64].cpu(), _xp[:64].cpu())
        print(f"ε = {_e}/255 : FGSM exactitude {attaques['fgsm'][_e]['exactitude']:.1%} "
              f"(perte {attaques['fgsm'][_e]['perte']:.2f}, transfert {[f'{v:.1%}' for v in attaques['fgsm'][_e]['transfert']]}) ; "
              f"PGD {attaques['pgd'][_e]['exactitude']:.1%} (perte {attaques['pgd'][_e]['perte']:.2f}, "
              f"confiance {attaques['pgd'][_e]['confiance']:.1%}, transfert {[f'{v:.1%}' for v in attaques['pgd'][_e]['transfert']]})")
    duree_attaques = time.perf_counter() - _t0
    print(f"Attaques : {duree_attaques:.1f} s")
    return attaques, duree_attaques, images_adv


@app.cell
def _(CLASSES_CIFAR10, SMOKE, X_test, device, images_adv, mo, modeles, np, plt, sauvegarder_figure, torch, y_test):
    # --- Une image, sa perturbation (amplifiée), l'image attaquée (PGD, ε = 4/255) ---
    _e = 4
    _xp = images_adv[_e][1]
    with torch.no_grad():
        _p0 = modeles[0](X_test[:64].to(device)).softmax(1).cpu()
        _pa = modeles[0](_xp.to(device)).softmax(1).cpu()
    # première image bien classée avec confiance ≥ 90 % et retournée par l'attaque
    _cand = [i for i in range(64) if _p0[i].argmax() == y_test[i] and _p0[i].max() >= 0.9
             and _pa[i].argmax() != y_test[i]]
    idx_exemple = _cand[0] if _cand else 0
    _i = idx_exemple
    _x0 = X_test[_i].permute(1, 2, 0).numpy()
    _xa = _xp[_i].permute(1, 2, 0).numpy()
    _d = _xa - _x0
    exemple_adv = {
        "indice": _i,
        "vraie": CLASSES_CIFAR10[int(y_test[_i])],
        "avant": (CLASSES_CIFAR10[int(_p0[_i].argmax())], float(_p0[_i].max())),
        "apres": (CLASSES_CIFAR10[int(_pa[_i].argmax())], float(_pa[_i].max())),
        "ecart_max_niveaux": float(np.abs(_d).max() * 255),
    }
    _fig, _axes = plt.subplots(1, 3, figsize=(9, 3.4))
    _axes[0].imshow(_x0, interpolation="nearest")
    _axes[0].set_title(f"{exemple_adv['avant'][0]} {exemple_adv['avant'][1]:.0%}")
    _axes[1].imshow(0.5 + _d / (2 * np.abs(_d).max() + 1e-12), interpolation="nearest")
    _axes[1].set_title(f"perturbation × {0.5 / (np.abs(_d).max() + 1e-12):.0f}")
    _axes[2].imshow(_xa, interpolation="nearest")
    _axes[2].set_title(f"{exemple_adv['apres'][0]} {exemple_adv['apres'][1]:.0%}")
    for _a in _axes:
        _a.axis("off")
    _fig.tight_layout()
    sauvegarder_figure(_fig, "exemple-adversarial-smoke" if SMOKE else "exemple-adversarial",
                       mo.notebook_dir() / "figures")
    print(exemple_adv)
    _fig
    return exemple_adv, idx_exemple


@app.cell
def _(SMOKE, attaques, config, mo, plt, sauvegarder_figure):
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    _eps = config["epsilons"]
    for _nom, _c in [("fgsm", "#E69F00"), ("pgd", "#D55E00")]:
        _ax.plot(_eps, [100 * attaques[_nom][e]["exactitude"] for e in _eps], "o-", color=_c,
                 label=f"{_nom.upper()} (modèle attaqué)")
        _ax.plot(_eps, [100 * attaques[_nom][e]["transfert"][0] for e in _eps], "o:", color=_c,
                 label=f"{_nom.upper()} transféré à un autre modèle")
    _ax.set(xlabel="ε (niveaux de gris sur 255)", ylabel="exactitude (%)", ylim=(0, 100),
            title="Exactitude sous attaque")
    _ax.legend(fontsize=9)
    sauvegarder_figure(_fig, "attaques-smoke" if SMOKE else "attaques", mo.notebook_dir() / "figures")
    _fig


@app.cell
def _(
    SMOKE,
    T,
    T_jouet,
    T_lbfgs,
    X_test,
    attaques,
    calibration,
    config,
    conforme,
    couverture_jouet,
    duree_attaques,
    duree_entrainement,
    ece_main,
    ecart_auroc,
    ecart_nll_torch,
    exemple_adv,
    fiabilite,
    hors_distribution,
    idx_exemple,
    images_adv,
    json,
    mo,
    msp,
    np,
    variation_lineaire,
):
    # --- Bilan : seuils du README (vérifiés en mode complet seulement) ---
    _c = calibration
    _e8 = max(config["epsilons"])
    bilan = {
        "duree_entrainement_s": round(duree_entrainement, 1), "duree_attaques_s": round(duree_attaques, 1),
        "ece_main": ece_main, "ecart_nll_torch": ecart_nll_torch, "T_jouet": T_jouet, "T_lbfgs": T_lbfgs,
        "ecart_auroc": ecart_auroc, "couverture_jouet": couverture_jouet,
        "variation_lineaire": variation_lineaire, "T": T, "calibration": _c, "fiabilite": fiabilite,
        "conforme": conforme, "hors_distribution": hors_distribution, "attaques": attaques,
        "exemple_adv": exemple_adv,
    }
    print(json.dumps(bilan, indent=1, ensure_ascii=False, default=float))
    # Résultats bruts pour les figures du site (site/scripts/figures/ch39_*.py)
    _x0 = X_test[idx_exemple].permute(1, 2, 0).numpy()
    _xf = images_adv[4][0][idx_exemple].permute(1, 2, 0).numpy()
    _xp = images_adv[4][1][idx_exemple].permute(1, 2, 0).numpy()
    np.savez_compressed(
        mo.notebook_dir() / "figures" / ("resultats-smoke.npz" if SMOKE else "resultats.npz"),
        bilan=json.dumps(bilan, ensure_ascii=False, default=float),
        image=(_x0 * 255).round().astype(np.uint8),
        image_fgsm=(_xf * 255).round().astype(np.uint8),
        image_pgd=(_xp * 255).round().astype(np.uint8),
        msp_seul_cifar=msp["seul"]["cifar10"].astype(np.float32),
        msp_seul_mnist=msp["seul"]["mnist"].astype(np.float32),
        msp_seul_bruit=msp["seul"]["bruit"].astype(np.float32),
        msp_ens_cifar=msp["ensemble"]["cifar10"].astype(np.float32),
        msp_ens_mnist=msp["ensemble"]["mnist"].astype(np.float32),
        msp_ens_bruit=msp["ensemble"]["bruit"].astype(np.float32),
    )

    SEUILS = [
        ("NLL NumPy = F.cross_entropy (écart < 1e-9)", ecart_nll_torch < 1e-9),
        ("température jouet : nombre d'or = L-BFGS à 1e-3 près", abs(T_jouet - T_lbfgs) < 1e-3),
        ("AUROC NumPy = scikit-learn (écart < 1e-12)", ecart_auroc < 1e-12),
        ("modèle seul : exactitude test ≥ 80 %", _c["test/seul"]["exactitude"] >= 0.80),
        ("modèle seul un peu trop sûr de lui : confiance > exactitude et T > 1",
         _c["test/seul"]["confiance"] > _c["test/seul"]["exactitude"] and T > 1),
        ("température : ECE divisée par 2 au moins et ≤ 1,5 %",
         _c["test/température"]["ece"] <= min(0.015, _c["test/seul"]["ece"] / 2)),
        ("température : NLL en baisse", _c["test/température"]["nll"] < _c["test/seul"]["nll"]),
        ("ensemble : exactitude ≥ seul + 1,5 point",
         _c["test/ensemble"]["exactitude"] >= _c["test/seul"]["exactitude"] + 0.015),
        ("bruit : ECE du modèle seul ≥ 20 %", _c["test_bruit/seul"]["ece"] >= 0.20),
        ("bruit : la température ne répare pas (ECE ≥ 15 %)", _c["test_bruit/température"]["ece"] >= 0.15),
        ("bruit : l'ensemble est mieux calibré que le modèle seul",
         _c["test_bruit/ensemble"]["ece"] < _c["test_bruit/seul"]["ece"]),
        ("conforme : couverture test entre 89 et 91 %", 0.89 <= conforme["couverture"] <= 0.91),
        ("hors distribution : AUROC MNIST ≥ 0,65 (modèle seul)", hors_distribution["seul/mnist"]["auroc"] >= 0.65),
        ("bruit uniforme : ≥ 30 % des images à confiance ≥ 90 % (modèle seul)",
         hors_distribution["seul/bruit"]["part_confiance_90"] >= 0.30),
        (f"FGSM ε = {_e8}/255 : exactitude ≤ 25 %", attaques["fgsm"][_e8]["exactitude"] <= 0.25),
        (f"PGD ε = {_e8}/255 : exactitude ≤ 2 %", attaques["pgd"][_e8]["exactitude"] <= 0.02),
        ("PGD plus fort que FGSM à chaque ε > 0",
         all(attaques["pgd"][e]["exactitude"] <= attaques["fgsm"][e]["exactitude"] for e in config["epsilons"] if e > 0)),
    ]
    _echecs = [nom for nom, ok in SEUILS if not ok]
    for _nom, _ok in SEUILS:
        print(("OK    " if _ok else "ÉCHEC ") + _nom)
    if not SMOKE and _echecs:
        raise SystemExit("Seuils non atteints : " + " ; ".join(_echecs))
    mo.md(f"**Bilan** : ECE {_c['test/seul']['ece']:.1%} → {_c['test/température']['ece']:.1%} "
          f"(T = {T:.2f}) ; couverture conforme {conforme['couverture']:.1%} ; "
          f"PGD ε = {_e8}/255 : {attaques['pgd'][_e8]['exactitude']:.1%}.")


if __name__ == "__main__":
    app.run()
