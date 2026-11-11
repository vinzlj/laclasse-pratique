# Exercice 27 : pourquoi le deep learning généralise, la partie empirique.
#
#   make ex NUM=27              → éditeur marimo (interactif, boutons « Lancer »)
#   make ex NUM=27 MODE=run     → run complet en script (mesure de la durée)
#   make ex NUM=27 MODE=smoke   → test rapide, < 30 s
#
# Quatre expériences, dans l'ordre du chapitre 27 :
#   A. un MLP mémorise des étiquettes tirées au hasard (Zhang et al., 2016) ;
#   B. double descente avec des features aléatoires, en NumPy (Belkin et al., 2018) ;
#   C. régularisation implicite : la descente de gradient choisit l'interpolant de norme minimale ;
#   D. grokking sur l'addition modulo 113 (Power et al., 2022 ; réglages de Nanda et al., 2023).
# Les résultats chiffrés sont écrits dans figures/resultats.npz (lu par les figures du site).

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
    from torch import nn

    from ia_commun import get_device, is_smoke, sauvegarder_figure, seed_everything, style_commun
    from ia_commun.donnees import mnist_tenseurs

    return (
        get_device,
        is_smoke,
        json,
        math,
        mnist_tenseurs,
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
    # Exercice 27 : pourquoi le deep learning généralise (la partie empirique)

    Le chapitre 10 a donné des bornes de généralisation qui grandissent avec la taille de la
    classe de modèles. Un réseau qui a plus de paramètres que d'exemples devrait donc
    surapprendre. Ici, on mesure ce qui se passe vraiment, en quatre expériences :

    - **A. Mémoriser du bruit.** Un MLP atteint 100 % d'exactitude d'entraînement sur MNIST avec
      les vraies étiquettes… et aussi avec des étiquettes tirées au hasard. La capacité du modèle
      ne peut donc pas, à elle seule, expliquer qu'il généralise dans le premier cas.
    - **B. Double descente** (NumPy, from scratch). Régression sur $p$ features aléatoires :
      l'erreur de test monte quand $p$ approche le nombre d'exemples $n$, explose à $p = n$,
      puis redescend sous le meilleur modèle « classique » quand $p \gg n$.
    - **C. Régularisation implicite.** Au-delà de $p = n$, une infinité de solutions donnent une
      erreur d'entraînement nulle. La descente de gradient partie de zéro choisit celle de norme
      minimale ; partie d'ailleurs, elle en choisit une autre, qui généralise moins bien.
    - **D. Grokking.** Un Transformer d'une couche apprend par cœur 30 % des sommes
      $a + b \bmod 113$ en quelques centaines de pas, puis, des milliers de pas plus tard,
      généralise d'un coup aux 70 % restants.

    **Pertes optimisées.** A : entropie croisée. B et C : erreur quadratique (MSE) sur les
    étiquettes codées en one-hot, minimisée exactement (moindres carrés) en B, par descente de
    gradient en C. D : entropie croisée, avec un weight decay fort ($\lambda = 1$, AdamW).
    """)


@app.cell
def _(get_device, is_smoke, json, os, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    device = get_device()
    style_commun()
    config = dict(
        # A. étiquettes aléatoires : MLP 784-512-512-10, Adam
        a_n=1000 if SMOKE else 5000,          # images d'entraînement
        a_epoques=8 if SMOKE else 100,
        a_lot=128, a_lr=1e-3, a_cachee=512,
        a_fractions=[0.0, 1.0] if SMOKE else [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        a_epoques_fractions=8 if SMOKE else 60,
        # B. double descente : n images, p features ReLU aléatoires
        b_n=300 if SMOKE else 1000,
        b_n_test=1000 if SMOKE else 10000,
        b_p=[30, 100, 200, 280, 300, 320, 400, 1000, 3000] if SMOKE else
            [50, 100, 200, 300, 400, 500, 600, 700, 800, 850, 900, 940, 970, 990, 1000,
             1010, 1030, 1060, 1100, 1150, 1200, 1300, 1500, 1700, 2000, 2500, 3000,
             4000, 5000, 7000, 10000, 15000, 20000],
        b_lambdas=[1e-3, 1e-1],               # ridge explicite, pour comparaison
        b_p_fixe=300 if SMOKE else 1000,      # B2 : p fixé, n varie (« plus de données peut nuire »)
        b_n_liste=[100, 200, 290, 300, 310, 400, 600] if SMOKE else
            [200, 400, 600, 800, 900, 950, 980, 1000, 1020, 1050, 1100, 1200, 1500, 2000, 3000, 4000],
        # C. régularisation implicite
        c_p=1200 if SMOKE else 4000,
        c_pas=200 if SMOKE else 1500,
        # initialisations aléatoires : écart-type = facteur / √p (facteur 1 : échelle usuelle
        # d'une couche linéaire à p entrées ; 100 et 300 : départs volontairement grands)
        c_facteurs_init=[1.0, 100.0, 300.0],
        # D. grokking (Nanda et al. 2023 : p = 113, 30 % des paires, 1 couche, d = 128,
        # 4 têtes, MLP 512, pas de LayerNorm, AdamW lr 1e-3, weight decay 1, lot complet)
        d_p=113, d_fraction=0.3,
        d_dim=128, d_tetes=4, d_mlp=512,
        d_lr=1e-3, d_wd=1.0, d_betas=(0.9, 0.98), d_echauffement=10,
        d_pas=300 if SMOKE else 15000,
        d_eval_tous=50 if SMOKE else 100,
    )
    # Pour expérimenter sans modifier le fichier : IA27_CONFIG='{"d_wd": 0}' make ex NUM=27 MODE=run
    config.update(json.loads(os.environ.get("IA27_CONFIG", "{}")))
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, périphérique {device}")
    print(config)
    return SEED, SMOKE, config, device


@app.cell
def _(mnist_tenseurs, np, torch):
    # --- Données MNIST (cache data/mnist/), aplaties en vecteurs de 784 pixels dans [0, 1] ---
    _x_tr, _y_tr = mnist_tenseurs(train=True)
    _x_te, _y_te = mnist_tenseurs(train=False)
    mnist_x = _x_tr.reshape(len(_x_tr), -1)
    mnist_y = _y_tr
    mnist_xt = _x_te.reshape(len(_x_te), -1)
    mnist_yt = _y_te
    # Un ordre aléatoire fixe des images d'entraînement, partagé par A, B et C
    ordre = torch.randperm(len(mnist_x), generator=torch.Generator().manual_seed(42))
    print(f"MNIST : {len(mnist_x)} images d'entraînement, {len(mnist_xt)} de test")
    _ = np
    return mnist_x, mnist_xt, mnist_y, mnist_yt, ordre


@app.cell
def _(mo):
    mo.md(r"""
    ## A. Un réseau mémorise des étiquettes tirées au hasard

    Même MLP (784 → 512 → 512 → 10, 670 000 paramètres environ), même optimiseur, trois jeux
    d'entraînement de même taille :

    1. **vraies étiquettes** ;
    2. **étiquettes aléatoires** : chaque image reçoit un chiffre tiré uniformément, sans lien
       avec l'image ;
    3. **pixels aléatoires** : les images sont remplacées par du bruit gaussien (même moyenne et
       même écart-type que MNIST), avec les vraies étiquettes.

    Puis on remplace une fraction croissante des étiquettes (0 %, 20 %… 100 %) par du hasard et
    on mesure l'exactitude sur le jeu de test (vraies images, vraies étiquettes).
    """)


@app.cell
def _(mo):
    bouton = mo.ui.run_button(label="Lancer les quatre expériences")
    bouton
    return (bouton,)


@app.cell
def _(
    bouton,
    config,
    device,
    mnist_x,
    mnist_xt,
    mnist_y,
    mnist_yt,
    mo,
    nn,
    ordre,
    seed_everything,
    time,
    torch,
):
    # En script (make ex MODE=run|smoke), personne ne clique : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer les quatre expériences ».*"))

    def entrainer_mlp(x, y, epoques, x_test=None, y_test=None):
        """Entraîne un MLP neuf (graine 42) ; renvoie l'historique par époque."""
        seed_everything(42)
        _c = config["a_cachee"]
        modele = nn.Sequential(nn.Linear(784, _c), nn.ReLU(), nn.Linear(_c, _c), nn.ReLU(),
                               nn.Linear(_c, 10)).to(device)
        optim = torch.optim.Adam(modele.parameters(), lr=config["a_lr"])
        x, y = x.to(device), y.to(device)
        hist = {"perte": [], "precision": [], "test": []}
        n = len(x)
        for _ep in range(epoques):
            modele.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, config["a_lot"]):
                j = perm[i:i + config["a_lot"]]
                perte = nn.functional.cross_entropy(modele(x[j]), y[j])
                optim.zero_grad()
                perte.backward()
                optim.step()
            modele.eval()
            with torch.no_grad():
                logits = modele(x)
                hist["perte"].append(nn.functional.cross_entropy(logits, y).item())
                hist["precision"].append((logits.argmax(1) == y).float().mean().item())
                if x_test is not None:
                    hist["test"].append((modele(x_test).argmax(1) == y_test).float().mean().item())
        return hist

    _t0 = time.perf_counter()
    _g = torch.Generator().manual_seed(42)
    _idx = ordre[:config["a_n"]]
    _x, _y = mnist_x[_idx], mnist_y[_idx]
    _xt, _yt = mnist_xt.to(device), mnist_yt.to(device)
    _moy, _et = _x.mean(), _x.std()
    _y_alea = torch.randint(0, 10, (len(_y),), generator=_g)
    _x_bruit = torch.randn(_x.shape, generator=_g) * _et + _moy
    # Pour les pixels aléatoires, le « test » est un nouveau bruit : il n'y a rien à généraliser.
    _xt_bruit = (torch.randn((len(_xt), 784), generator=_g) * _et + _moy).to(device)
    conditions_a = {
        "vraies étiquettes": (_x, _y, _xt, _yt),
        "étiquettes aléatoires": (_x, _y_alea, _xt, _yt),
        "pixels aléatoires": (_x_bruit, _y, _xt_bruit, _yt),
    }
    historiques_a = {}
    for _nom, (_xa, _ya, _xta, _yta) in conditions_a.items():
        historiques_a[_nom] = entrainer_mlp(_xa, _ya, config["a_epoques"], _xta, _yta)
        _h = historiques_a[_nom]
        _ep100 = next((i + 1 for i, v in enumerate(_h["precision"]) if v >= 1.0), None)
        _quand = "non atteint" if _ep100 is None else f"atteint à l'époque {_ep100}"
        print(f"A. {_nom:22s} : exactitude d'entraînement {_h['precision'][-1]:.1%} "
              f"(100 % {_quand}), test {_h['test'][-1]:.1%}")

    # Fraction d'étiquettes remplacées par du hasard → exactitude de test
    fractions_a, tests_a, trains_a = [], [], []
    for _f in config["a_fractions"]:
        _yf = _y.clone()
        _m = torch.rand(len(_y), generator=_g) < _f
        _yf[_m] = torch.randint(0, 10, (int(_m.sum()),), generator=_g)
        _h = entrainer_mlp(_x, _yf, config["a_epoques_fractions"], _xt, _yt)
        fractions_a.append(_f)
        tests_a.append(_h["test"][-1])
        trains_a.append(_h["precision"][-1])
        print(f"A. {_f:.0%} d'étiquettes aléatoires : entraînement {_h['precision'][-1]:.1%}, "
              f"test {_h['test'][-1]:.1%}")
    duree_a = time.perf_counter() - _t0
    print(f"Partie A : {duree_a:.1f} s")
    return duree_a, fractions_a, historiques_a, tests_a, trains_a


@app.cell
def _(mo):
    mo.md(r"""
    ## B. Double descente avec des features aléatoires (NumPy)

    On tire une fois pour toutes une matrice $W \in \mathbb R^{784 \times p_{\max}}$ de nombres
    gaussiens. Une image $x$ devient le vecteur de $p$ features
    $\phi(x) = \mathrm{ReLU}(W_{:, 1..p}^\top x)/\sqrt p$ : un réseau à une couche cachée dont
    **seule la dernière couche est apprise**. Avec la matrice $\Phi$ ($n \times p$) des features
    et les étiquettes one-hot $Y$ ($n \times 10$), on cherche $B$ ($p \times 10$) qui minimise
    $\lVert \Phi B - Y \rVert^2$.

    - $p < n$ : une seule solution, les moindres carrés.
    - $p \ge n$ : une infinité de solutions avec une erreur d'entraînement nulle. On prend celle
      de **plus petite norme**, $B = \Phi^+ Y$ (pseudo-inverse), que renvoie `np.linalg.lstsq`.
    - Comparaison : ridge, $B = (\Phi^\top\Phi + \lambda I)^{-1}\Phi^\top Y$.
    """)


@app.cell
def _(config, mnist_x, mnist_xt, mnist_y, mnist_yt, np, ordre, time):
    _t0 = time.perf_counter()
    _rng = np.random.default_rng(42)
    # Données centrées (moyenne des images d'entraînement retirée), en float64 pour lstsq
    _pool = ordre.numpy()
    _x_tout = mnist_x.numpy()[_pool].astype(np.float64)
    _y_tout = mnist_y.numpy()[_pool]
    _moy = _x_tout[:config["b_n"]].mean(0)
    _it = _rng.choice(len(mnist_xt), config["b_n_test"], replace=False)
    _xt = mnist_xt.numpy()[_it].astype(np.float64) - _moy
    _yt = mnist_yt.numpy()[_it]
    _p_max = max(max(config["b_p"]), config["b_p_fixe"], config["c_p"])
    W_features = _rng.normal(size=(784, _p_max)) / np.sqrt(784)

    def features(x, p):
        """φ(x) = ReLU(Wᵀx)/√p : les p premières colonnes de W, toujours les mêmes."""
        return np.maximum(x @ W_features[:, :p], 0.0) / np.sqrt(p)

    def ajuster(F, Y, lam=0.0):
        """Moindres carrés de norme minimale (lam = 0) ou ridge (lam > 0)."""
        n, p = F.shape
        if lam == 0.0:
            return np.linalg.lstsq(F, Y, rcond=None)[0]
        if p <= n:
            return np.linalg.solve(F.T @ F + lam * np.eye(p), F.T @ Y)
        # forme duale, moins chère quand p > n : Fᵀ (F Fᵀ + λI)⁻¹ Y
        return F.T @ np.linalg.solve(F @ F.T + lam * np.eye(n), Y)

    def evaluer(F, Y, Ft, yt, B):
        """MSE d'entraînement, MSE de test (somme sur les 10 sorties), erreur de classification de test."""
        _Yt = np.eye(10)[yt]
        _pt = Ft @ B
        return (float(((F @ B - Y) ** 2).sum(1).mean()), float(((_pt - _Yt) ** 2).sum(1).mean()),
                float((_pt.argmax(1) != yt).mean()))

    # B1 : n fixé, p varie (double descente « en fonction du modèle »)
    _x = _x_tout[:config["b_n"]] - _moy
    _Y = np.eye(10)[_y_tout[:config["b_n"]]]
    courbe_b = {k: [] for k in ["p", "train", "test", "erreur", "norme"]}
    ridge_b = {lam: {"test": [], "erreur": []} for lam in config["b_lambdas"]}
    for _p in config["b_p"]:
        _F, _Ft = features(_x, _p), features(_xt, _p)
        _B = ajuster(_F, _Y)
        _tr, _te, _er = evaluer(_F, _Y, _Ft, _yt, _B)
        for _k, _v in zip(["p", "train", "test", "erreur", "norme"], [_p, _tr, _te, _er, np.linalg.norm(_B)]):
            courbe_b[_k].append(float(_v))
        for _lam in config["b_lambdas"]:
            _, _te_r, _er_r = evaluer(_F, _Y, _Ft, _yt, ajuster(_F, _Y, _lam))
            ridge_b[_lam]["test"].append(_te_r)
            ridge_b[_lam]["erreur"].append(_er_r)
    courbe_b = {k: np.array(v) for k, v in courbe_b.items()}
    for _p, _tr, _te, _er, _no in zip(*[courbe_b[k] for k in ["p", "train", "test", "erreur", "norme"]]):
        print(f"B1. p = {int(_p):6d} : MSE train {_tr:.3f}, test {_te:9.3f}, erreur {_er:.1%}, ‖B‖ = {_no:8.1f}")

    # B2 : p fixé, n varie (double descente « en fonction des données »)
    courbe_b2 = {"n": [], "test": [], "erreur": []}
    for _n in config["b_n_liste"]:
        _xn = _x_tout[:_n] - _moy
        _Yn = np.eye(10)[_y_tout[:_n]]
        _F, _Ft = features(_xn, config["b_p_fixe"]), features(_xt, config["b_p_fixe"])
        _, _te, _er = evaluer(_F, _Yn, _Ft, _yt, ajuster(_F, _Yn))
        courbe_b2["n"].append(_n)
        courbe_b2["test"].append(_te)
        courbe_b2["erreur"].append(_er)
        print(f"B2. p = {config['b_p_fixe']}, n = {_n:5d} : MSE test {_te:9.3f}, erreur {_er:.1%}")
    courbe_b2 = {k: np.array(v) for k, v in courbe_b2.items()}
    duree_b = time.perf_counter() - _t0
    print(f"Partie B : {duree_b:.1f} s")
    donnees_c = dict(x=_x, Y=_Y, xt=_xt, yt=_yt)
    return (
        ajuster,
        courbe_b,
        courbe_b2,
        donnees_c,
        duree_b,
        evaluer,
        features,
        ridge_b,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## C. Régularisation implicite : quel interpolant la descente de gradient choisit-elle ?

    Avec $p = 4\,000$ features pour $n = 1\,000$ images, l'équation $\Phi B = Y$ a une infinité
    de solutions. La descente de gradient sur la MSE fait
    $B \leftarrow B - \eta\, \Phi^\top(\Phi B - Y)/n$ : chaque mise à jour est une combinaison
    des lignes de $\Phi$. La partie de $B$ orthogonale à ces lignes (le **noyau** de $\Phi$) ne
    bouge donc jamais. Partie de $B_0 = 0$, la descente converge vers $\Phi^+ Y$, l'interpolant
    de norme minimale ; partie d'un $B_0$ aléatoire, elle converge vers
    $\Phi^+ Y + P_{\text{noyau}} B_0$, qui interpole aussi mais garde le bruit de départ.
    """)


@app.cell
def _(ajuster, config, donnees_c, evaluer, features, np, time):
    _t0 = time.perf_counter()
    _F = features(donnees_c["x"], config["c_p"])
    _Ft = features(donnees_c["xt"], config["c_p"])
    _Y, _yt = donnees_c["Y"], donnees_c["yt"]
    _n = len(_F)
    _B_min = ajuster(_F, _Y)                       # interpolant de norme minimale
    # Projecteur sur le noyau de Φ, via la SVD : P = I − V Vᵀ (V : directions des lignes)
    _U, _S, _Vt = np.linalg.svd(_F, full_matrices=False)
    _V = _Vt[_S > _S[0] * 1e-10].T
    _eta = _n / _S[0] ** 2                          # pas sûr : η < 2n / σ_max²
    _rng = np.random.default_rng(42)
    _bruit = _rng.normal(size=(config["c_p"], 10)) / np.sqrt(config["c_p"])
    _departs = [("zéro", np.zeros((config["c_p"], 10)))]
    _departs += [(f"aléatoire ×{_f:g}", _f * _bruit) for _f in config["c_facteurs_init"]]
    resultats_c = {}
    for _nom, _B0 in _departs:
        _B = _B0.copy()
        _hist = {"pas": [], "train": [], "distance": [], "noyau": []}
        _noyau0 = _B0 - _V @ (_V.T @ _B0)
        for _s in range(config["c_pas"] + 1):
            if _s % max(1, config["c_pas"] // 30) == 0 or _s == config["c_pas"]:
                _hist["pas"].append(_s)
                _hist["train"].append(float(((_F @ _B - _Y) ** 2).sum(1).mean()))
                _hist["distance"].append(float(np.linalg.norm(_B - _B_min) / np.linalg.norm(_B_min)))
                # écart de la composante « noyau » de B à celle de B0 : reste nul (machine)
                _hist["noyau"].append(float(np.linalg.norm((_B - _V @ (_V.T @ _B)) - _noyau0)))
            _B -= _eta * _F.T @ (_F @ _B - _Y) / _n
        # Limite exacte de la descente (pas infini) : B* + composante noyau de B0
        _B_lim = _B_min + _noyau0
        _, _te_lim, _er_lim = evaluer(_F, _Y, _Ft, _yt, _B_lim)
        _tr_T, _te_T, _er_T = evaluer(_F, _Y, _Ft, _yt, _B)
        resultats_c[_nom] = dict(hist={k: np.array(v) for k, v in _hist.items()},
                                 norme_limite=float(np.linalg.norm(_B_lim)),
                                 test_limite=_te_lim, erreur_limite=_er_lim,
                                 train_T=_tr_T, test_T=_te_T, erreur_T=_er_T,
                                 noyau_max=float(max(_hist["noyau"])))
        print(f"C. départ {_nom:9s} : après {config['c_pas']} pas, MSE train {_tr_T:.4f}, "
              f"test {_te_T:.3f}, erreur {_er_T:.1%}, distance relative à Φ⁺Y {_hist['distance'][-1]:.3f}, "
              f"composante noyau déplacée de {max(_hist['noyau']):.1e}")
        print(f"   limite exacte : ‖B‖ = {np.linalg.norm(_B_lim):.1f}, MSE test {_te_lim:.3f}, erreur {_er_lim:.1%}")
    resultats_c["norme_min"] = float(np.linalg.norm(_B_min))
    duree_c = time.perf_counter() - _t0
    print(f"‖Φ⁺Y‖ = {resultats_c['norme_min']:.1f} ; partie C : {duree_c:.1f} s")
    return duree_c, resultats_c


@app.cell
def _(mo):
    mo.md(r"""
    ## D. Grokking : généraliser longtemps après avoir mémorisé

    Tâche : prédire $c = (a + b) \bmod 113$ à partir de la séquence de trois tokens
    « $a$ $b$ = ». Il y a $113^2 = 12\,769$ paires ; on en montre 30 % (3 830) au modèle, on
    teste sur les 8 939 autres. Le modèle est un Transformer d'**une** couche (attention à 4 têtes
    puis MLP), sans LayerNorm, entraîné sur le lot complet par AdamW avec un weight decay de 1,
    comme dans Nanda et al. (2023). On suit l'exactitude d'entraînement et de test, et la norme
    des poids.
    """)


@app.cell
def _(config, device, math, nn, np, seed_everything, time, torch):
    seed_everything(42)
    _P = config["d_p"]
    _a = torch.arange(_P).repeat_interleave(_P)
    _b = torch.arange(_P).repeat(_P)
    _X = torch.stack([_a, _b, torch.full_like(_a, _P)], dim=1)   # le token _P joue « = »
    _Y = (_a + _b) % _P
    _perm = torch.randperm(_P * _P, generator=torch.Generator().manual_seed(42))
    _ntr = int(config["d_fraction"] * _P * _P)
    _Xtr, _Ytr = _X[_perm[:_ntr]].to(device), _Y[_perm[:_ntr]].to(device)
    _Xte, _Yte = _X[_perm[_ntr:]].to(device), _Y[_perm[_ntr:]].to(device)
    # Paires de test dont la paire « miroir » (b, a) est dans l'entraînement : un modèle qui a
    # seulement appris la commutativité a + b = b + a les réussit sans savoir additionner.
    _dans_train = torch.zeros(_P * _P, dtype=torch.bool)
    _dans_train[_perm[:_ntr]] = True
    _idx_te = _perm[_ntr:]
    miroir_d = _dans_train[(_idx_te % _P) * _P + _idx_te // _P].to(device)
    print(f"D. {miroir_d.float().mean().item():.1%} des paires de test ont leur miroir (b, a) dans l'entraînement")

    class TransformerUneCouche(nn.Module):
        """Une couche d'attention (4 têtes) puis un MLP ReLU, connexions résiduelles, sans LayerNorm."""

        def __init__(self, vocab, d, tetes, d_mlp):
            super().__init__()
            self.tetes = tetes
            self.plongement = nn.Embedding(vocab, d)
            self.position = nn.Parameter(torch.randn(3, d) / math.sqrt(d))
            self.qkv = nn.Linear(d, 3 * d, bias=False)
            self.sortie_attention = nn.Linear(d, d, bias=False)
            self.mlp = nn.Sequential(nn.Linear(d, d_mlp), nn.ReLU(), nn.Linear(d_mlp, d))
            self.lecture = nn.Linear(d, _P, bias=False)

        def forward(self, x):
            z = self.plongement(x) + self.position
            n, t, d = z.shape
            q, k, v = self.qkv(z).view(n, t, 3, self.tetes, d // self.tetes).permute(2, 0, 3, 1, 4)
            scores = q @ k.transpose(-1, -2) / math.sqrt(d // self.tetes)
            masque = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), 1)
            attention = scores.masked_fill(masque, float("-inf")).softmax(-1)
            z = z + self.sortie_attention((attention @ v).transpose(1, 2).reshape(n, t, d))
            z = z + self.mlp(z)
            return self.lecture(z[:, -1])        # prédiction lue sur le token « = »

    modele_d = TransformerUneCouche(_P + 1, config["d_dim"], config["d_tetes"], config["d_mlp"]).to(device)
    n_param_d = sum(p.numel() for p in modele_d.parameters())
    _optim = torch.optim.AdamW(modele_d.parameters(), lr=config["d_lr"], weight_decay=config["d_wd"],
                               betas=tuple(config["d_betas"]))
    _sched = torch.optim.lr_scheduler.LambdaLR(_optim, lambda s: min(1.0, (s + 1) / config["d_echauffement"]))
    historique_d = {k: [] for k in ["pas", "perte_train", "perte_test", "acc_train", "acc_test", "norme",
                                     "acc_test_miroir", "acc_test_sans_miroir"]}
    _t0 = time.perf_counter()
    for _s in range(config["d_pas"] + 1):
        # évaluation tous les d_eval_tous pas, plus serrée au début (la mémorisation est rapide)
        if _s % config["d_eval_tous"] == 0 or (_s < 1000 and _s % 20 == 0) or _s == config["d_pas"]:
            with torch.no_grad():
                _lt, _lte = modele_d(_Xtr), modele_d(_Xte)
                _vals = [_s, nn.functional.cross_entropy(_lt, _Ytr).item(),
                         nn.functional.cross_entropy(_lte, _Yte).item(),
                         (_lt.argmax(1) == _Ytr).float().mean().item(),
                         (_lte.argmax(1) == _Yte).float().mean().item(),
                         torch.sqrt(sum((p ** 2).sum() for p in modele_d.parameters())).item()]
                _ok = _lte.argmax(1) == _Yte
                _vals += [_ok[miroir_d].float().mean().item(), _ok[~miroir_d].float().mean().item()]
            for _k, _v in zip(historique_d, _vals):
                historique_d[_k].append(_v)
            if _s % (config["d_eval_tous"] * 20) == 0 or _s == config["d_pas"]:  # affichage
                print(f"D. pas {_s:6d} ({time.perf_counter() - _t0:5.0f} s) : exactitude train "
                      f"{_vals[3]:.1%}, test {_vals[4]:.1%} ; perte test {_vals[2]:.3f} ; ‖θ‖ = {_vals[5]:.1f}")
        if _s == config["d_pas"]:
            break
        _perte = nn.functional.cross_entropy(modele_d(_Xtr), _Ytr)
        _optim.zero_grad()
        _perte.backward()
        _optim.step()
        _sched.step()
    duree_d = time.perf_counter() - _t0
    historique_d = {k: np.array(v) for k, v in historique_d.items()}
    print(f"Partie D : {duree_d:.1f} s ({1000 * duree_d / config['d_pas']:.1f} ms par pas), "
          f"{n_param_d} paramètres, {_ntr} paires d'entraînement")
    return duree_d, historique_d, n_param_d


@app.cell
def _(config, historique_d, np):
    # Repères du grokking : premier pas où l'entraînement (resp. le test) dépasse 99 % (resp. 95 %)
    def premier_pas(serie, seuil):
        _i = np.flatnonzero(historique_d[serie] >= seuil)
        return int(historique_d["pas"][_i[0]]) if len(_i) else None

    pas_memorise = premier_pas("acc_train", 0.99)
    pas_generalise = premier_pas("acc_test", 0.95)
    # Au milieu du plateau : que réussit le modèle parmi les paires de test ?
    _i = int(np.argmin(np.abs(historique_d["pas"] - config["d_pas"] // 3)))
    plateau_d = dict(pas=int(historique_d["pas"][_i]),
                     miroir=float(historique_d["acc_test_miroir"][_i]),
                     sans_miroir=float(historique_d["acc_test_sans_miroir"][_i]))
    print(f"D. au pas {plateau_d['pas']} : exactitude de test {plateau_d['miroir']:.1%} sur les paires dont le "
          f"miroir est connu, {plateau_d['sans_miroir']:.1%} sur les autres (hasard : {1 / config['d_p']:.1%})")
    _txt_m = "non atteinte" if pas_memorise is None else f"au pas {pas_memorise}"
    _txt_g = "non atteinte" if pas_generalise is None else f"au pas {pas_generalise}"
    print(f"D. mémorisation (train ≥ 99 %) {_txt_m} ; généralisation (test ≥ 95 %) {_txt_g}")
    return pas_generalise, pas_memorise, plateau_d


@app.cell
def _(
    SMOKE,
    config,
    courbe_b,
    courbe_b2,
    fractions_a,
    historique_d,
    historiques_a,
    mo,
    np,
    plt,
    resultats_c,
    ridge_b,
    sauvegarder_figure,
    tests_a,
):
    # --- Figure de l'exercice : les quatre expériences ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    _ax = axes[0, 0]
    for _nom, _h in historiques_a.items():
        _ax.plot(np.arange(1, len(_h["perte"]) + 1), _h["perte"], label=_nom)
    _ax.set(yscale="log", xlabel="époque", ylabel="perte d'entraînement",
            title="A. Mémoriser : vraies étiquettes, hasard, bruit")
    _ax.legend(fontsize=8, loc="lower left")
    _ins = _ax.inset_axes([0.62, 0.55, 0.35, 0.4])
    _ins.plot(np.array(fractions_a) * 100, np.array(tests_a) * 100, "o-", color="0.3")
    _ins.set_xlabel("% d'étiquettes aléatoires", fontsize=7)
    _ins.set_ylabel("exactitude test (%)", fontsize=7)
    _ins.tick_params(labelsize=7)

    _ax = axes[0, 1]
    _p = courbe_b["p"]
    _ax.plot(_p, courbe_b["erreur"] * 100, "o-", ms=3, label="norme minimale (λ = 0)")
    for _lam, _r in ridge_b.items():
        _ax.plot(_p, np.array(_r["erreur"]) * 100, "--", label=f"ridge λ = {_lam:g}")
    _ax.axvline(config["b_n"], color="0.6", lw=0.8)
    _ax.set(xscale="log", xlabel="p (nombre de features)", ylabel="erreur de classification de test (%)",
            title=f"B. Double descente (n = {config['b_n']})")
    _ax.legend(fontsize=8)

    _ax = axes[1, 0]
    for _nom, _r in resultats_c.items():
        if isinstance(_r, dict):
            _ax.plot(_r["hist"]["pas"], _r["hist"]["distance"], label=f"départ {_nom}")
    _ax.set(yscale="log", xlabel="pas de descente de gradient", ylabel="‖B − Φ⁺Y‖ / ‖Φ⁺Y‖",
            title="C. Vers quel interpolant converge-t-on ?")
    _ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(0.0, 0.62))
    _ins2 = _ax.inset_axes([0.6, 0.3, 0.36, 0.32])
    _ins2.plot(courbe_b2["n"], courbe_b2["test"], "o-", ms=3, color="0.3")
    _ins2.set(yscale="log")
    _ins2.set_xlabel(f"n (p = {config['b_p_fixe']})", fontsize=7)
    _ins2.set_ylabel("MSE test", fontsize=7)
    _ins2.tick_params(labelsize=7)

    _ax = axes[1, 1]
    _ax.plot(historique_d["pas"], historique_d["acc_train"] * 100, label="entraînement")
    _ax.plot(historique_d["pas"], historique_d["acc_test"] * 100, label="test")
    _ax2 = _ax.twinx()
    _ax2.plot(historique_d["pas"], historique_d["norme"], color="0.5", lw=0.8, ls="--")
    _ax2.set_ylabel("norme des poids (tirets)")
    _ax2.grid(False)
    _ax.set(xlabel="pas", ylabel="exactitude (%)", title="D. Grokking, addition mod 113")
    _ax.legend(fontsize=8, loc="center left")
    fig.tight_layout()
    _chemins = sauvegarder_figure(fig, "generalisation-smoke" if SMOKE else "generalisation",
                                  mo.notebook_dir() / "figures")
    print(f"Figure : {_chemins[0]}")
    fig


@app.cell
def _(
    SMOKE,
    config,
    courbe_b,
    courbe_b2,
    duree_a,
    duree_b,
    duree_c,
    duree_d,
    fractions_a,
    historique_d,
    historiques_a,
    json,
    mo,
    n_param_d,
    np,
    pas_generalise,
    pas_memorise,
    plateau_d,
    resultats_c,
    ridge_b,
    tests_a,
    trains_a,
):
    # --- Sauvegarde des résultats (lus par site/scripts/figures/ch27_*.py) ---
    _bilan = dict(
        config={k: v for k, v in config.items()},
        durees=dict(a=duree_a, b=duree_b, c=duree_c, d=duree_d),
        n_param_d=n_param_d, pas_memorise=pas_memorise, pas_generalise=pas_generalise,
        plateau_d=plateau_d,
        c={k: ({kk: vv for kk, vv in v.items() if kk != "hist"} if isinstance(v, dict) else v)
           for k, v in resultats_c.items()},
    )
    _tableaux = {}
    for _i, (_nom, _h) in enumerate(historiques_a.items()):
        for _k, _v in _h.items():
            _tableaux[f"a{_i}_{_k}"] = np.array(_v)
    _tableaux.update(a_fractions=np.array(fractions_a), a_tests=np.array(tests_a), a_trains=np.array(trains_a))
    _tableaux.update({f"b_{k}": v for k, v in courbe_b.items()})
    _tableaux.update({f"b_ridge_{_j}_{k}": np.array(v) for _j, (_lam, _r) in enumerate(ridge_b.items())
                      for k, v in _r.items()})
    _tableaux.update({f"b2_{k}": v for k, v in courbe_b2.items()})
    _noms_c = [k for k, v in resultats_c.items() if isinstance(v, dict)]
    for _j, _nom in enumerate(_noms_c):
        _tableaux.update({f"c{_j}_{k}": v for k, v in resultats_c[_nom]["hist"].items()})
    _bilan["noms_c"] = _noms_c
    _tableaux.update({f"d_{k}": v for k, v in historique_d.items()})
    _bilan["noms_a"] = list(historiques_a)
    _bilan["lambdas"] = list(ridge_b)
    _chemin = mo.notebook_dir() / "figures" / ("resultats-smoke.npz" if SMOKE else "resultats.npz")
    np.savez_compressed(_chemin, bilan=json.dumps(_bilan, ensure_ascii=False), **_tableaux)
    print(f"Résultats : {_chemin}")
    return


@app.cell
def _(
    SMOKE,
    config,
    courbe_b,
    duree_a,
    duree_b,
    duree_c,
    duree_d,
    historiques_a,
    mo,
    np,
    pas_generalise,
    pas_memorise,
    plateau_d,
    resultats_c,
    ridge_b,
):
    # --- Bilan : seuils chiffrés du README. En smoke, on vérifie seulement que tout tourne.
    _ha = historiques_a
    _p = courbe_b["p"]
    _classique = _p <= config["b_n"] // 2                  # modèles nettement sous le seuil
    _i_seuil = int(np.argmin(np.abs(_p - config["b_n"])))  # p = n
    _err_classique = float(courbe_b["erreur"][_classique].min())
    _lam_petit = min(ridge_b)
    _c_grand = resultats_c[f"aléatoire ×{max(config['c_facteurs_init']):g}"]
    _verifs = {
        "A : vraies étiquettes, exactitude d'entraînement 100 %": _ha["vraies étiquettes"]["precision"][-1] >= 1.0,
        "A : étiquettes aléatoires, exactitude d'entraînement ≥ 99 %": _ha["étiquettes aléatoires"]["precision"][-1] >= 0.99,
        "A : pixels aléatoires, exactitude d'entraînement ≥ 99 %": _ha["pixels aléatoires"]["precision"][-1] >= 0.99,
        "A : vraies étiquettes, exactitude de test ≥ 90 %": _ha["vraies étiquettes"]["test"][-1] >= 0.90,
        "A : étiquettes aléatoires, exactitude de test ≤ 15 %": _ha["étiquettes aléatoires"]["test"][-1] <= 0.15,
        "B : pic à p = n (MSE test ≥ 10 × le minimum classique)":
            courbe_b["test"][_i_seuil] >= 10 * courbe_b["test"][_classique].min(),
        "B : MSE d'entraînement ≈ 0 pour p ≥ n (≤ 1e-6)": float(courbe_b["train"][_p >= config["b_n"]].max()) <= 1e-6,
        "B : le plus grand modèle bat le meilleur modèle classique (erreur de test)":
            float(courbe_b["erreur"][-1]) < _err_classique,
        f"B : ridge λ = {_lam_petit:g} efface le pic (erreur à p = n ≤ 20 %)":
            ridge_b[_lam_petit]["erreur"][_i_seuil] <= 0.20,
        "C : la composante « noyau » ne bouge pas (≤ 1e-8)":
            max(v["noyau_max"] for v in resultats_c.values() if isinstance(v, dict)) <= 1e-8,
        "C : départ ×1 (échelle usuelle) → même limite que le départ zéro (erreur à 0,5 point près)":
            abs(resultats_c["aléatoire ×1"]["erreur_limite"] - resultats_c["zéro"]["erreur_limite"]) <= 0.005,
        "C : plus grand départ aléatoire → interpolant de norme ≥ 2 × la norme minimale":
            _c_grand["norme_limite"] >= 2 * resultats_c["norme_min"],
        "C : plus grand départ aléatoire → erreur de test plus haute d'au moins 5 points":
            _c_grand["erreur_limite"] >= resultats_c["zéro"]["erreur_limite"] + 0.05,
        "D : mémorisation (train ≥ 99 %) avant le pas 2 000": pas_memorise is not None and pas_memorise <= 2000,
        "D : généralisation (test ≥ 95 %) atteinte": pas_generalise is not None,
        "D : au plateau, ≥ 50 % de réussite sur les paires « miroir », ≤ 5 % sur les autres":
            plateau_d["miroir"] >= 0.5 and plateau_d["sans_miroir"] <= 0.05,
        "D : au moins 5 000 pas entre mémorisation et généralisation":
            pas_memorise is not None and pas_generalise is not None and pas_generalise - pas_memorise >= 5000,
    }
    _lignes = "\n".join(f"- {'✅' if ok else '❌'} {k}" for k, ok in _verifs.items())
    _total = duree_a + duree_b + duree_c + duree_d
    if not SMOKE and not all(_verifs.values()):
        raise SystemExit("Seuils non atteints :\n" + _lignes)
    mo.md(f"**Bilan** ({_total:.0f} s de calcul : A {duree_a:.0f} s, B {duree_b:.0f} s, "
          f"C {duree_c:.0f} s, D {duree_d:.0f} s)\n\n{_lignes}")


if __name__ == "__main__":
    app.run()
