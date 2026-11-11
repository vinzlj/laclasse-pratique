# Exercice 12 : rendre les réseaux profonds entraînables (initialisation, dropout,
# BatchNorm, pénalité L2, label smoothing, mixup).
#
#   make ex NUM=12              → éditeur marimo (bouton « Entraîner »)
#   make ex NUM=12 MODE=run     → script complet, vérifie les seuils du README
#   make ex NUM=12 MODE=smoke   → test rapide (< 30 s)
#
# Parties A et B en NumPy (aucun entraînement : on propage et on compare à PyTorch),
# parties C et D en PyTorch sur MNIST (GPU Apple via MPS si disponible).

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import copy
    import json
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
        copy,
        get_device,
        is_smoke,
        json,
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
    # Exercice 12 : rendre les réseaux profonds entraînables

    Deux problèmes empêchaient d'entraîner des réseaux profonds avant 2010-2015 :

    1. **le signal s'éteint ou explose** en traversant les couches, à l'aller (activations)
       comme au retour (gradients) : un réseau de 20 couches mal initialisé n'apprend rien ;
    2. **le réseau surapprend** : il récite ses exemples d'entraînement et se trompe sur les
       autres.

    Plan :

    - **A. Initialisation (NumPy).** Propager un lot dans 30 couches avec cinq couples
      activation / initialisation ; mesurer l'écart type des activations et la norme du
      gradient couche par couche.
    - **B. Les mécanismes à la main (NumPy), vérifiés contre PyTorch** : dropout inversé,
      normalisation par lot (BatchNorm), label smoothing, mixup.
    - **C. Régularisation sur MNIST (PyTorch).** Le même MLP 784-256-256-10 sur 5 000 images
      seulement (pour qu'il surapprenne vite) : sans rien, avec dropout, avec BatchNorm, avec
      les deux, puis pénalité L2, label smoothing et mixup ; trois graines par configuration.
    - **D. Profondeur (PyTorch).** Un MLP de 20 couches cachées, six variantes
      d'initialisation et de normalisation : lesquelles apprennent ?

    **Perte optimisée** : l'entropie croisée
    $\mathcal{L} = -\frac1N\sum_i \log \hat p_{i,y_i}$ ; avec la pénalité L2, on ajoute
    $\lambda \sum_W \lVert W\rVert^2$ (poids des couches linéaires, pas les biais) ; le label
    smoothing remplace la cible « un seul 1 » par $1-\varepsilon$ sur la bonne classe et
    $\varepsilon/K$ réparti sur les $K$ classes ; mixup mélange deux images et leurs deux
    cibles avec le même coefficient $\lambda$.
    """)


@app.cell
def _(get_device, is_smoke, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    device = get_device()
    style_commun()
    config = dict(
        # Partie A : réseau NumPy
        couches_a=30, largeur_a=256, lot_a=512,
        # Partie C : MLP 784-256-256-10
        n_train=1000 if SMOKE else 5000,   # sous-ensemble d'entraînement (surapprentissage visible)
        n_val=10000,                        # validation : 10 000 autres images du jeu d'entraînement
        epoques_c=2 if SMOKE else 30,
        graines_c=[SEED] if SMOKE else [SEED, SEED + 1, SEED + 2],
        lot=128, lr=1e-3,
        p_dropout=0.5, lambda_l2=5e-4, epsilon_ls=0.1, alpha_mixup=0.2,
        masques_mc=5 if SMOKE else 100,     # passes avec dropout actif pour la moyenne d'ensemble
        # Partie D : MLP profond
        couches_d=20, largeur_d=256,
        epoques_d=1 if SMOKE else 5,
        lr_d=0.01,                          # SGD avec momentum 0,9 (Adam masquerait le problème)
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, périphérique {device}")
    print(config)
    return SEED, SMOKE, config, device


@app.cell
def _(mo):
    mo.md(r"""
    ## A. Initialisation : le signal traverse-t-il 30 couches ?

    Chaque couche calcule $h_{l} = f(h_{l-1} W_l)$ avec $W_l$ de taille $n \times n$
    ($n = 256$) tirée selon une loi normale centrée d'écart type $\sigma$. La variance de
    $z = h W$ vaut $n\,\sigma^2\,\mathrm{Var}(h)$ : elle est conservée si
    $\sigma^2 = 1/n$ (**Xavier / Glorot** pour tanh) ; la ReLU met la moitié des sorties à
    zéro, il faut doubler : $\sigma^2 = 2/n$ (**He**).

    On envoie un lot de 512 vecteurs gaussiens, puis on rétropropage un gradient aléatoire
    depuis la sortie pour mesurer la norme du gradient des poids de chaque couche.
    """)


@app.cell
def _(SEED, config, np):
    def propager(activation, init, couches, largeur, lot, graine):
        """Aller-retour dans un MLP NumPy sans biais. Renvoie, couche par couche (index 0 =
        première couche) : l'écart type des activations, la norme du gradient qui redescend
        ‖∂L/∂z_l‖ (z_l = préactivation) et la norme du gradient des poids ‖∂L/∂W_l‖."""
        rng = np.random.default_rng(graine)
        ecarts_types = {"petite": 0.01, "grande": 1.0,
                        "xavier": np.sqrt(1 / largeur), "he": np.sqrt(2 / largeur)}
        sigma = ecarts_types[init]
        h = rng.standard_normal((lot, largeur))
        poids, entrees, preacts = [], [], []
        for _ in range(couches):
            W = rng.standard_normal((largeur, largeur)) * sigma
            z = h @ W
            entrees.append(h)
            preacts.append(z)
            poids.append(W)
            h = np.tanh(z) if activation == "tanh" else np.maximum(z, 0.0)
        std_activations = [float(np.tanh(z).std() if activation == "tanh" else np.maximum(z, 0).std())
                           for z in preacts]
        # Retour : gradient aléatoire en sortie, règle de la chaîne couche par couche
        g = rng.standard_normal(h.shape) / lot
        normes, signal = [0.0] * couches, [0.0] * couches
        for l in reversed(range(couches)):
            derivee = 1 - np.tanh(preacts[l]) ** 2 if activation == "tanh" else (preacts[l] > 0)
            dz = g * derivee
            signal[l] = float(np.linalg.norm(dz))                    # ‖∂L/∂z_l‖
            normes[l] = float(np.linalg.norm(entrees[l].T @ dz))   # ‖∂L/∂W_l‖
            g = dz @ poids[l].T
        return std_activations, signal, normes

    variantes_a = [("tanh", "petite"), ("tanh", "grande"), ("tanh", "xavier"),
                   ("relu", "xavier"), ("relu", "he")]
    resultats_a = {}
    for _act, _init in variantes_a:
        resultats_a[f"{_act} + {_init}"] = propager(_act, _init, config["couches_a"],
                                                    config["largeur_a"], config["lot_a"], SEED)
    print(f"{'variante':16s} {'σ act. c.1':>11s} {'c.10':>10s} {'c.30':>10s}   "
          f"{'‖∂L/∂z‖ c.1':>12s} {'c.30':>10s}   {'‖∂L/∂W‖ c.1':>12s} {'c.30':>10s}")
    for _nom, (_std, _sig, _gn) in resultats_a.items():
        print(f"{_nom:16s} {_std[0]:11.3g} {_std[9]:10.3g} {_std[-1]:10.3g}   "
              f"{_sig[0]:12.3g} {_sig[-1]:10.3g}   {_gn[0]:12.3g} {_gn[-1]:10.3g}")
    return (resultats_a,)


@app.cell
def _(mo):
    mo.md(r"""
    ## B. Les mécanismes à la main, puis la bibliothèque

    - **Dropout inversé** : à l'entraînement, chaque neurone est mis à zéro avec la
      probabilité $p$ et les survivants sont multipliés par $1/(1-p)$, pour que l'espérance
      de la sortie ne change pas ; à l'inférence, on ne touche à rien.
    - **BatchNorm** : $\hat z = (z - \mu_B)/\sqrt{\sigma_B^2 + \epsilon}$ puis
      $y = \gamma \hat z + \beta$, avec $\mu_B, \sigma_B^2$ calculés **sur le lot** à
      l'entraînement, et des moyennes glissantes (momentum 0,1) à l'inférence.
    - **Label smoothing** : cible $(1-\varepsilon)\,\text{one-hot} + \varepsilon/K$.
    - **Mixup** : $\tilde x = \lambda x_i + (1-\lambda) x_j$, perte
      $\lambda\,\mathcal L(\tilde x, y_i) + (1-\lambda)\,\mathcal L(\tilde x, y_j)$,
      $\lambda \sim \mathrm{Beta}(\alpha, \alpha)$.
    """)


@app.cell
def _(F, SEED, config, nn, np, torch):
    _rng = np.random.default_rng(SEED)
    verif_b = {}

    # --- Dropout inversé ---------------------------------------------------------
    def dropout_numpy(h, p, rng, entrainement=True):
        if not entrainement:
            return h
        masque = rng.random(h.shape) >= p          # True = neurone conservé
        return h * masque / (1 - p)

    _h = np.abs(_rng.standard_normal((2000, 256)))   # activations positives, comme après ReLU
    _d = dropout_numpy(_h, config["p_dropout"], _rng)
    verif_b["dropout_frac_zero_numpy"] = float((_d == 0).mean())
    verif_b["dropout_ecart_moyenne"] = float(abs(_d.mean() / _h.mean() - 1))
    _couche = nn.Dropout(config["p_dropout"]).train()
    _dt = _couche(torch.from_numpy(_h.astype(np.float32))).numpy()
    verif_b["dropout_frac_zero_torch"] = float((_dt == 0).mean())
    # Les survivants valent exactement h / (1 - p) chez PyTorch aussi
    _surv = _dt != 0
    verif_b["dropout_facteur_torch"] = float(np.median(_dt[_surv] / _h[_surv]))
    _couche.eval()
    verif_b["dropout_eval_identite"] = bool(np.allclose(_couche(torch.from_numpy(_h.astype(np.float32))).numpy(), _h.astype(np.float32)))

    # --- BatchNorm (entraînement puis inférence) ---------------------------------
    def batchnorm_numpy(z, gamma, beta, etat, entrainement, momentum=0.1, eps=1e-5):
        if entrainement:
            mu, var = z.mean(axis=0), z.var(axis=0)            # variance biaisée pour normaliser
            n = z.shape[0]
            # PyTorch met à jour la moyenne glissante avec la variance NON biaisée
            etat["moyenne"] = (1 - momentum) * etat["moyenne"] + momentum * mu
            etat["variance"] = (1 - momentum) * etat["variance"] + momentum * var * n / (n - 1)
        else:
            mu, var = etat["moyenne"], etat["variance"]
        return gamma * (z - mu) / np.sqrt(var + eps) + beta

    _z = (_rng.standard_normal((64, 32)) * 3 + 5).astype(np.float32)   # moyenne 5, écart type 3
    _gamma = _rng.standard_normal(32).astype(np.float32)
    _beta = _rng.standard_normal(32).astype(np.float32)
    _etat = {"moyenne": np.zeros(32), "variance": np.ones(32)}
    _bn = nn.BatchNorm1d(32)
    with torch.no_grad():
        _bn.weight.copy_(torch.from_numpy(_gamma))
        _bn.bias.copy_(torch.from_numpy(_beta))
    _bn.train()
    _y_np = batchnorm_numpy(_z, _gamma, _beta, _etat, True)
    _y_t = _bn(torch.from_numpy(_z)).detach().numpy()
    verif_b["bn_ecart_entrainement"] = float(np.abs(_y_np - _y_t).max())
    _zhat = (_z - _z.mean(0)) / np.sqrt(_z.var(0) + 1e-5)
    verif_b["bn_moyenne_sortie_normalisee"] = float(np.abs(_zhat.mean(0)).max())
    verif_b["bn_ecart_type_sortie_normalisee"] = float(_zhat.std(0).mean())
    _bn.eval()
    _z2 = (_rng.standard_normal((8, 32)) * 3 + 5).astype(np.float32)
    verif_b["bn_ecart_inference"] = float(np.abs(
        batchnorm_numpy(_z2, _gamma, _beta, _etat, False) - _bn(torch.from_numpy(_z2)).detach().numpy()).max())

    # --- Label smoothing ---------------------------------------------------------
    def entropie_croisee_lissee(logits, y, eps):
        K = logits.shape[1]
        logp = logits - logits.max(axis=1, keepdims=True)
        logp = logp - np.log(np.exp(logp).sum(axis=1, keepdims=True))   # log-softmax stable
        cible = np.full_like(logp, eps / K)
        cible[np.arange(len(y)), y] += 1 - eps
        return float(-(cible * logp).sum(axis=1).mean()), cible

    _logits = _rng.standard_normal((16, 10)).astype(np.float32) * 3
    _y = _rng.integers(0, 10, 16)
    _ls_np, _cible = entropie_croisee_lissee(_logits, _y, config["epsilon_ls"])
    _ls_t = F.cross_entropy(torch.from_numpy(_logits), torch.from_numpy(_y),
                            label_smoothing=config["epsilon_ls"]).item()
    verif_b["ls_ecart"] = abs(_ls_np - _ls_t)
    verif_b["ls_cible_exemple"] = [round(float(v), 3) for v in _cible[0]]
    # Plus petite perte possible pour UN exemple : l'entropie de la cible lissée (> 0)
    verif_b["ls_perte_minimale"] = float(-(_cible[0] * np.log(_cible[0])).sum())

    # --- Mixup -------------------------------------------------------------------
    def mixup_numpy(x, y_onehot, alpha, rng):
        lam = rng.beta(alpha, alpha)
        j = rng.permutation(len(x))
        return lam * x + (1 - lam) * x[j], lam * y_onehot + (1 - lam) * y_onehot[j], lam

    _x = _rng.random((8, 784))
    _yo = np.eye(10)[_rng.integers(0, 10, 8)]
    _xm, _ym, _lam = mixup_numpy(_x, _yo, config["alpha_mixup"], _rng)
    verif_b["mixup_lambda"] = float(_lam)
    verif_b["mixup_somme_cibles"] = float(_ym.sum(axis=1).mean())
    _lams = _rng.beta(config["alpha_mixup"], config["alpha_mixup"], 100_000)
    verif_b["mixup_part_lambda_extreme"] = float(((_lams < 0.1) | (_lams > 0.9)).mean())

    for _k, _v in verif_b.items():
        print(f"{_k:34s} {_v}")
    return (verif_b,)


@app.cell
def _(config, device, mnist_tenseurs, torch):
    # --- Données : MNIST (cache data/mnist/), 5 000 images d'entraînement seulement ---
    _Xtr, _ytr = mnist_tenseurs(train=True)
    _Xte, _yte = mnist_tenseurs(train=False)
    _perm = torch.randperm(60000, generator=torch.Generator().manual_seed(42))
    _itr = _perm[: config["n_train"]]
    _iva = _perm[60000 - config["n_val"]:]                  # aucune image commune avec _itr
    donnees = {
        "train": (_Xtr[_itr].view(-1, 784).to(device), _ytr[_itr].to(device)),
        "val": (_Xtr[_iva].view(-1, 784).to(device), _ytr[_iva].to(device)),
        "test": (_Xte.view(-1, 784).to(device), _yte.to(device)),
    }
    for _k, (_x, _y) in donnees.items():
        print(f"{_k:5s} {tuple(_x.shape)}  classes {torch.bincount(_y, minlength=10).tolist()}")
    return (donnees,)


@app.cell
def _(nn):
    def mlp(dropout=0.0, batchnorm=False, cachee=256):
        """MLP 784-256-256-10. Ordre par couche : Linéaire → BatchNorm → ReLU → Dropout."""
        couches, d = [], 784
        for _ in range(2):
            couches.append(nn.Linear(d, cachee))
            if batchnorm:
                couches.append(nn.BatchNorm1d(cachee))
            couches.append(nn.ReLU())
            if dropout:
                couches.append(nn.Dropout(dropout))
            d = cachee
        couches.append(nn.Linear(d, 10))
        return nn.Sequential(*couches)

    def mlp_profond(activation, init, batchnorm=False, couches=20, largeur=256):
        """MLP de `couches` couches cachées. init : petite (σ = 0,01), xavier, he ou defaut
        (initialisation par défaut de PyTorch : uniforme sur ±1/√n, soit σ² = 1/(3n))."""
        modules, d = [], 784
        for _ in range(couches):
            lin = nn.Linear(d, largeur)
            if init == "petite":
                nn.init.normal_(lin.weight, 0.0, 0.01)
            elif init == "xavier":
                nn.init.xavier_normal_(lin.weight)
            elif init == "he":
                nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
            if init != "defaut":
                nn.init.zeros_(lin.bias)
            modules.append(lin)
            if batchnorm:
                modules.append(nn.BatchNorm1d(largeur))
            modules.append(nn.Tanh() if activation == "tanh" else nn.ReLU())
            d = largeur
        modules.append(nn.Linear(d, 10))
        return nn.Sequential(*modules)

    return mlp, mlp_profond


@app.cell
def _(mo):
    # Le bouton empêche l'entraînement de se relancer à chaque modification d'une cellule amont.
    bouton = mo.ui.run_button(label="Entraîner")
    bouton
    return (bouton,)


@app.cell
def _(F, bouton, config, copy, donnees, mlp, mo, np, seed_everything, time, torch):
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Entraîner » pour lancer les parties C et D.*"))

    @torch.no_grad()
    def evaluer(modele, X, y):
        modele.eval()                     # dropout coupé, BatchNorm sur ses moyennes glissantes
        sortie = modele(X)
        modele.train()
        return F.cross_entropy(sortie, y).item(), (sortie.argmax(1) == y).float().mean().item()

    def entrainer(fabrique, graine, l2=0.0, ls=0.0, mixup=0.0):
        """Adam, lots de 128, `epoques_c` époques. Renvoie l'historique par époque,
        le modèle final et celui de la meilleure époque de validation (arrêt précoce)."""
        seed_everything(graine)
        rng = np.random.default_rng(graine)           # tirages de λ pour mixup
        modele = fabrique().to(donnees["train"][0].device)
        optim = torch.optim.Adam(modele.parameters(), lr=config["lr"])
        Xa, ya = donnees["train"]
        n = len(ya)
        gen = torch.Generator().manual_seed(graine)
        histo = {k: [] for k in ("loss_train", "acc_train", "loss_val", "acc_val")}
        meilleur, meilleure_loss = None, float("inf")
        for _ep in range(config["epoques_c"]):
            ordre = torch.randperm(n, generator=gen).to(ya.device)
            for i in range(0, n, config["lot"]):
                idx = ordre[i:i + config["lot"]]
                x, y = Xa[idx], ya[idx]
                if mixup:
                    lam = float(rng.beta(mixup, mixup))
                    j = torch.randperm(len(idx), generator=gen).to(ya.device)
                    sortie = modele(lam * x + (1 - lam) * x[j])
                    perte = lam * F.cross_entropy(sortie, y) + (1 - lam) * F.cross_entropy(sortie, y[j])
                else:
                    perte = F.cross_entropy(modele(x), y, label_smoothing=ls)
                if l2:
                    # Pénalité L2 sur les matrices de poids (pas sur les biais ni sur BatchNorm)
                    perte = perte + l2 * sum((p ** 2).sum() for p in modele.parameters() if p.dim() > 1)
                optim.zero_grad()
                perte.backward()
                optim.step()
            lt, at = evaluer(modele, Xa, ya)
            lv, av = evaluer(modele, *donnees["val"])
            for k, v in zip(histo, (lt, at, lv, av)):
                histo[k].append(v)
            if lv < meilleure_loss:
                meilleure_loss, meilleur = lv, copy.deepcopy(modele)
        return histo, modele, meilleur

    configurations_c = {
        "sans régularisation": (lambda: mlp(), {}),
        "dropout": (lambda: mlp(dropout=config["p_dropout"]), {}),
        "BatchNorm": (lambda: mlp(batchnorm=True), {}),
        "dropout + BatchNorm": (lambda: mlp(dropout=config["p_dropout"], batchnorm=True), {}),
        "pénalité L2": (lambda: mlp(), {"l2": config["lambda_l2"]}),
        "label smoothing": (lambda: mlp(), {"ls": config["epsilon_ls"]}),
        "mixup": (lambda: mlp(), {"mixup": config["alpha_mixup"]}),
    }
    resultats_c, modeles_c = {}, {}
    _t0 = time.perf_counter()
    for _nom, (_fab, _kw) in configurations_c.items():
        _runs = []
        for _g in config["graines_c"]:
            _h, _m, _best = entrainer(_fab, _g, **_kw)
            _runs.append({"histo": _h,
                          "test": evaluer(_m, *donnees["test"])[1],
                          "test_arret_precoce": evaluer(_best, *donnees["test"])[1],
                          "epoque_min_val": int(np.argmin(_h["loss_val"])) + 1})
            if _g == config["graines_c"][0]:
                modeles_c[_nom] = {"final": _m, "arret_precoce": _best}
        _moy = {k: np.mean([r["histo"][k] for r in _runs], axis=0).tolist() for k in _runs[0]["histo"]}
        resultats_c[_nom] = {
            "courbes": _moy,
            "test": float(np.mean([r["test"] for r in _runs])),
            "test_std": float(np.std([r["test"] for r in _runs], ddof=1)) if len(_runs) > 1 else 0.0,  # écart type d'échantillon
            "test_arret_precoce": float(np.mean([r["test_arret_precoce"] for r in _runs])),
            "epoque_min_val": [r["epoque_min_val"] for r in _runs],
        }
        _r = resultats_c[_nom]
        print(f"{_nom:20s} train {_moy['acc_train'][-1]:.2%} (loss {_moy['loss_train'][-1]:.3f}) | "
              f"val {_moy['acc_val'][-1]:.2%} (loss {_moy['loss_val'][-1]:.3f}, min {min(_moy['loss_val']):.3f} "
              f"à l'époque {_r['epoque_min_val']}) | test {_r['test']:.2%} ± {_r['test_std']:.2%}, "
              f"arrêt précoce {_r['test_arret_precoce']:.2%}", flush=True)
    duree_c = time.perf_counter() - _t0
    print(f"Partie C : {duree_c:.1f} s")
    return duree_c, evaluer, modeles_c, resultats_c


@app.cell
def _(config, donnees, evaluer, modeles_c, torch):
    # --- Dropout à l'inférence : eval() contre train(), et moyenne de sous-réseaux ---
    _m = modeles_c["dropout"]["final"]
    _Xv, _yv = donnees["val"]
    inference_dropout = {"acc_eval": evaluer(_m, _Xv, _yv)[1]}
    with torch.no_grad():
        _m.train()                                     # dropout ACTIF : un sous-réseau tiré au hasard
        _p1, _p2 = _m(_Xv).argmax(1), _m(_Xv).argmax(1)
        inference_dropout["acc_un_sous_reseau"] = (_p1 == _yv).float().mean().item()
        inference_dropout["desaccord_deux_passes"] = (_p1 != _p2).float().mean().item()
        # Moyenne des probabilités de nombreux sous-réseaux (ensemble « Monte-Carlo »)
        _proba = sum(torch.softmax(_m(_Xv), 1) for _ in range(config["masques_mc"])) / config["masques_mc"]
        _m.eval()
        _p_eval = _m(_Xv).argmax(1)
    inference_dropout["acc_moyenne_sous_reseaux"] = (_proba.argmax(1) == _yv).float().mean().item()
    inference_dropout["accord_moyenne_vs_eval"] = (_proba.argmax(1) == _p_eval).float().mean().item()
    for _k, _v in inference_dropout.items():
        print(f"{_k:26s} {_v:.2%}")

    # --- Pourquoi la perte de validation remonte-t-elle ? Modèle sans régularisation,
    # meilleure époque de validation contre dernière époque : erreurs et confiance.
    confiance_erreurs = {}
    for _quand, _mod in modeles_c["sans régularisation"].items():
        _mod.eval()
        with torch.no_grad():
            _out = _mod(_Xv)
        _mod.train()
        _pertes = torch.nn.functional.cross_entropy(_out, _yv, reduction="none")
        _ok = _out.argmax(1) == _yv
        confiance_erreurs[_quand] = {
            "perte_val": _pertes.mean().item(),
            "erreurs": int((~_ok).sum().item()),
            "perte_moyenne_par_erreur": _pertes[~_ok].mean().item(),
            "confiance_moyenne_sur_erreurs": torch.softmax(_out, 1).max(1).values[~_ok].mean().item(),
            "part_de_la_perte_due_aux_erreurs": (_pertes[~_ok].sum() / _pertes.sum()).item(),
        }
        print(_quand, {k: round(v, 3) for k, v in confiance_erreurs[_quand].items()})
    return confiance_erreurs, inference_dropout


@app.cell
def _(F, config, donnees, mlp_profond, nn, seed_everything, time, torch):
    # --- Partie D : 20 couches cachées, SGD + momentum, 5 époques ---
    variantes_d = {
        "tanh + petite (σ = 0,01)": ("tanh", "petite", False),
        "tanh + Xavier": ("tanh", "xavier", False),
        "ReLU + Xavier": ("relu", "xavier", False),
        "ReLU + défaut PyTorch": ("relu", "defaut", False),
        "ReLU + He": ("relu", "he", False),
        "ReLU + défaut PyTorch + BatchNorm": ("relu", "defaut", True),
    }
    resultats_d = {}
    _Xa, _ya = donnees["train"]
    _Xv, _yv = donnees["val"]
    _t0 = time.perf_counter()
    for _nom, (_act, _init, _bn) in variantes_d.items():
        seed_everything(42)
        _m = mlp_profond(_act, _init, _bn, config["couches_d"], config["largeur_d"]).to(_ya.device)
        _opt = torch.optim.SGD(_m.parameters(), lr=config["lr_d"], momentum=0.9)
        _gen = torch.Generator().manual_seed(42)
        _pertes, _grad = [], None
        for _ep in range(config["epoques_d"]):
            _ordre = torch.randperm(len(_ya), generator=_gen).to(_ya.device)
            for _i in range(0, len(_ya), config["lot"]):
                _idx = _ordre[_i:_i + config["lot"]]
                _perte = F.cross_entropy(_m(_Xa[_idx]), _ya[_idx])
                _opt.zero_grad()
                _perte.backward()
                if _grad is None:   # norme du gradient au tout premier pas, couche par couche
                    _grad = [l.weight.grad.norm().item() for l in _m if isinstance(l, nn.Linear)][:-1]
                _opt.step()
                _pertes.append(_perte.item())
        _m.eval()
        with torch.no_grad():
            _acc = (_m(_Xv).argmax(1) == _yv).float().mean().item()
        resultats_d[_nom] = {"pertes": _pertes, "grad_couches": _grad, "acc_val": _acc}
        print(f"{_nom:34s} ‖grad‖ couche 1 : {_grad[0]:9.2e}, couche 20 : {_grad[-1]:9.2e} | "
              f"perte {_pertes[0]:.3f} → {sum(_pertes[-10:]) / 10:.3f} | val {_acc:.2%}", flush=True)
    duree_d = time.perf_counter() - _t0
    # Vérification de l'initialisation par défaut : σ·√n ≈ 1/√3 ≈ 0,577
    _w = nn.Linear(256, 256).weight
    std_defaut = float(_w.detach().std() * 256 ** 0.5)
    print(f"Partie D : {duree_d:.1f} s ; init. par défaut de PyTorch : σ·√n = {std_defaut:.3f}")
    return duree_d, resultats_d, std_defaut


@app.cell
def _(
    SMOKE,
    confiance_erreurs,
    config,
    json,
    mo,
    np,
    plt,
    resultats_a,
    resultats_c,
    resultats_d,
    sauvegarder_figure,
    inference_dropout,
    verif_b,
):
    # --- Figures (PNG pour le notebook) + résultats bruts (JSON lu par les figures du site) ---
    dossier = mo.notebook_dir() / "figures"
    suffixe = "-smoke" if SMOKE else ""

    # 1. Partie A : écart type des activations et norme du gradient par couche
    fig_a, (axa1, axa2) = plt.subplots(1, 2, figsize=(11, 4))
    _couches = np.arange(1, config["couches_a"] + 1)
    for _nom, (_std, _sig, _gn) in resultats_a.items():
        axa1.semilogy(_couches, _std, marker=".", label=_nom)
        axa2.semilogy(_couches, _sig, marker=".", label=_nom)
    axa1.set(xlabel="couche", ylabel="écart type des activations", title="Aller : activations")
    axa2.set(xlabel="couche", ylabel="‖∂L/∂z‖", title="Retour : gradient qui redescend")
    axa2.legend(fontsize=8)
    sauvegarder_figure(fig_a, "initialisation" + suffixe, dossier)

    # 2. Partie C : courbes moyennes sur les graines
    fig_c, axes_c = plt.subplots(1, 3, figsize=(15, 4.2))
    _ep = np.arange(1, config["epoques_c"] + 1)
    for _nom, _r in resultats_c.items():
        _ls = "-" if _nom in ("sans régularisation", "dropout", "BatchNorm", "dropout + BatchNorm") else "--"
        axes_c[0].plot(_ep, _r["courbes"]["loss_train"], _ls, label=_nom)
        axes_c[1].plot(_ep, _r["courbes"]["loss_val"], _ls, label=_nom)
        axes_c[2].plot(_ep, _r["courbes"]["acc_val"], _ls, label=_nom)
    axes_c[0].set(yscale="log", xlabel="époque", ylabel="perte (entraînement)", title="Entraînement")
    axes_c[1].set(xlabel="époque", ylabel="perte (validation)", title="Validation")
    axes_c[2].set(xlabel="époque", ylabel="précision (validation)", title="Précision de validation")
    axes_c[2].legend(fontsize=8)
    sauvegarder_figure(fig_c, "regularisation" + suffixe, dossier)

    # 3. Partie D : perte d'entraînement du MLP profond, pas par pas
    fig_d, axd = plt.subplots(figsize=(8, 4.2))
    for _nom, _r in resultats_d.items():
        _p = np.convolve(_r["pertes"], np.ones(5) / 5, mode="valid")   # moyenne glissante sur 5 pas
        _style = ":" if _r["acc_val"] < 0.2 else "-"     # pointillés : réseaux qui n'apprennent rien
        axd.plot(_p, _style, label=f"{_nom} ({_r['acc_val']:.0%})")
    axd.set(xlabel="pas de SGD", ylabel="perte (entraînement)", title="MLP de 20 couches cachées")
    axd.legend(fontsize=8)
    sauvegarder_figure(fig_d, "profondeur" + suffixe, dossier)

    if not SMOKE:
        _json = {"config": config, "partie_a": {k: {"std": v[0], "signal": v[1], "grad_poids": v[2]} for k, v in resultats_a.items()},
                 "partie_b": verif_b, "partie_c": resultats_c, "partie_d": resultats_d,
                 "inference_dropout": inference_dropout, "confiance_erreurs": confiance_erreurs}
        (dossier / "resultats.json").write_text(json.dumps(_json, ensure_ascii=False, indent=1))
        print(f"Résultats bruts : {dossier / 'resultats.json'}")
    mo.vstack([fig_a, fig_c, fig_d])


@app.cell
def _(
    SMOKE,
    confiance_erreurs,
    duree_c,
    duree_d,
    inference_dropout,
    mo,
    resultats_a,
    resultats_c,
    resultats_d,
    std_defaut,
    verif_b,
):
    # --- Bilan : seuils du README. En smoke, on vérifie seulement que tout tourne. ---
    _base = resultats_c["sans régularisation"]
    _cb = _base["courbes"]
    seuils = {
        "A. tanh + petite : σ couche 30 < 1e-10": resultats_a["tanh + petite"][0][-1] < 1e-10,
        "A. tanh + petite : ‖∂L/∂z‖ couche 1 < 1e-10 (gradient éteint)": resultats_a["tanh + petite"][1][0] < 1e-10,
        "A. tanh + grande : ‖∂L/∂z‖ c.1 / c.30 > 1e6 (gradient explosif)": resultats_a["tanh + grande"][1][0] / resultats_a["tanh + grande"][1][-1] > 1e6,
        "A. ReLU + He : ‖∂L/∂z‖ c.1 / c.30 entre 0,1 et 10": 0.1 < resultats_a["relu + he"][1][0] / resultats_a["relu + he"][1][-1] < 10,
        "A. ReLU + Xavier : σ couche 30 < 1e-3": resultats_a["relu + xavier"][0][-1] < 1e-3,
        "A. ReLU + He : σ couche 30 > 0,1": resultats_a["relu + he"][0][-1] > 0.1,
        "B. dropout : 49 à 51 % de zéros (NumPy et PyTorch)": all(0.49 < verif_b[k] < 0.51 for k in ("dropout_frac_zero_numpy", "dropout_frac_zero_torch")),
        "B. dropout : moyenne conservée à 1 % près": verif_b["dropout_ecart_moyenne"] < 0.01,
        "B. BatchNorm NumPy = PyTorch (écart < 1e-4)": max(verif_b["bn_ecart_entrainement"], verif_b["bn_ecart_inference"]) < 1e-4,
        "B. label smoothing NumPy = PyTorch (écart < 1e-5)": verif_b["ls_ecart"] < 1e-5,
        "C. sans régularisation : perte val finale ≥ 1,15 × son minimum": _cb["loss_val"][-1] >= 1.15 * min(_cb["loss_val"]),
        "C. dropout, BatchNorm, les deux : test ≥ base + 0,4 point": all(
            resultats_c[k]["test"] >= _base["test"] + 0.004 for k in ("dropout", "BatchNorm", "dropout + BatchNorm")),
        "C. label smoothing : test ≥ base + 0,4 point": resultats_c["label smoothing"]["test"] >= _base["test"] + 0.004,
        "C. sans régularisation : moins d'erreurs mais plus sûr de lui à la fin qu'à l'arrêt précoce": (
            confiance_erreurs["final"]["erreurs"] <= confiance_erreurs["arret_precoce"]["erreurs"]
            and confiance_erreurs["final"]["confiance_moyenne_sur_erreurs"] > confiance_erreurs["arret_precoce"]["confiance_moyenne_sur_erreurs"]),
        "C. dropout : eval() et moyenne de sous-réseaux d'accord à ≥ 98 %": inference_dropout["accord_moyenne_vs_eval"] >= 0.98,
        "D. tanh petite, ReLU Xavier, ReLU défaut : val < 20 %": all(
            resultats_d[k]["acc_val"] < 0.2 for k in ("tanh + petite (σ = 0,01)", "ReLU + Xavier", "ReLU + défaut PyTorch")),
        "D. tanh Xavier, ReLU He, BatchNorm : val > 60 %": all(
            resultats_d[k]["acc_val"] > 0.6 for k in ("tanh + Xavier", "ReLU + He", "ReLU + défaut PyTorch + BatchNorm")),
        "D. init. par défaut de PyTorch : σ·√n ≈ 0,577": abs(std_defaut - 3 ** -0.5) < 0.02,
    }
    _lignes = "\n".join(f"- {'✅' if ok else '❌'} {nom}" for nom, ok in seuils.items())
    _echecs = [nom for nom, ok in seuils.items() if not ok]
    print(_lignes)
    if SMOKE:
        print("(mode smoke : seuils non vérifiés, des ❌ sont attendus)")
    if not SMOKE and _echecs:
        raise SystemExit("Seuils non atteints :\n" + "\n".join(_echecs))
    _note = "\n\n*Mode smoke : seuils non vérifiés, des ❌ sont attendus.*" if SMOKE else ""
    mo.md(f"**Bilan** ({duree_c + duree_d:.0f} s d'entraînement)\n\n{_lignes}{_note}")


if __name__ == "__main__":
    app.run()
