# Exercice 07 : rétropropagation, MLP et fonctions d'activation (chapitre 07).
#
#   make ex NUM=07              → éditeur marimo (bouton « Entraîner » pour MNIST)
#   make ex NUM=07 MODE=run     → script complet, vérifie les seuils du README
#   make ex NUM=07 MODE=smoke   → test rapide (< 30 s)
#
# Progression : la rétropropagation est d'abord faite « à la main » sur un réseau
# 2-2-1 (chaque nombre affiché), puis écrite une fois pour toutes en numpy pour un MLP
# de taille quelconque, contrôlée par différences finies, utilisée sur le XOR bruité,
# sur un réseau profond (sigmoïde contre tanh contre ReLU) et sur MNIST. Enfin le même
# modèle est écrit en PyTorch : autograd donne exactement les mêmes gradients.

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

    from ia_commun import (
        PALETTE,
        get_device,
        is_smoke,
        sauvegarder_figure,
        seed_everything,
        style_commun,
    )

    return (
        PALETTE,
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
    # Exercice 07 : rétropropagation, MLP et fonctions d'activation

    **Objectif.** Calculer soi-même le gradient de la perte par rapport à **tous** les
    poids d'un réseau multicouche, en une passe avant et une passe arrière, vérifier ce
    calcul par différences finies, puis constater que PyTorch (autograd) fait
    exactement le même calcul.

    **Ce qu'on minimise.** L'entropie croisée (chapitre 01) : binaire
    $\mathcal{L} = -\big[y \log \hat y + (1-y)\log(1-\hat y)\big]$ pour le réseau 2-2-1,
    multiclasse $\mathcal{L} = -\frac1N\sum_i \log \operatorname{softmax}(z_i)_{y_i}$
    ensuite. Avec ces pertes, l'erreur en sortie a une forme très simple :
    $\delta = \hat y - y$.

    Plan : (1) un réseau 2-2-1 à la main ; (2) un MLP numpy de taille quelconque et le
    contrôle du gradient ; (3) le XOR bruité ; (4) la profondeur : sigmoïde, tanh,
    ReLU ; (5) MNIST en numpy ; (6) le même modèle en PyTorch.
    """)


@app.cell
def _(get_device, is_smoke, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    device = get_device()
    style_commun()
    config = dict(
        # partie 3 : XOR bruité, descente de gradient sur tout le lot
        n_xor=400,
        pas_xor=300 if SMOKE else 3000,
        lr_xor=0.5,
        # partie 4 : réseau profond (8 couches cachées de 32 neurones)
        profondeur=8,
        largeur=32,
        pas_profond=300 if SMOKE else 2000,
        lr_profond=0.05,
        # parties 5 et 6 : MLP 784-256-256-10 sur MNIST, SGD par mini-lots
        n_mnist=5000 if SMOKE else 60000,
        epoques_mnist=1 if SMOKE else 10,
        lot_mnist=64,
        lr_mnist=0.1,
        cachees_mnist=(256, 256),
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SEED, SMOKE, config, device


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Un réseau 2-2-1 à la main

    Deux entrées, deux neurones cachés (sigmoïde), une sortie (sigmoïde), perte
    d'entropie croisée binaire. Tous les biais valent 0 pour garder des calculs courts.

    ```text
    x = (1, 0)  ──W1──►  z1 = x·W1  ──σ──►  h  ──W2──►  z2 = h·W2  ──σ──►  ŷ  ──►  ℒ(ŷ, y = 1)
    ```

    Passe arrière, de la sortie vers l'entrée (règle de la chaîne) :

    - $\delta_2 = \partial\mathcal{L}/\partial z_2 = \hat y - y$ (sigmoïde + entropie croisée se simplifient) ;
    - $\partial\mathcal{L}/\partial W_2 = h \,\delta_2$ ;
    - $\delta_1 = \partial\mathcal{L}/\partial z_1 = (W_2\,\delta_2) \odot h\,(1-h)$, car $\sigma'(z) = \sigma(z)(1-\sigma(z))$ ;
    - $\partial\mathcal{L}/\partial W_1 = x^\top \delta_1$ : le gradient d'un poids = activation **en amont** × erreur **en aval**.
    """)


@app.cell
def _(np):
    # --- Réseau 2-2-1 : chaque étape est écrite et affichée ---
    def sigmoide(z):
        return 1.0 / (1.0 + np.exp(-z))

    x_ex = np.array([1.0, 0.0])
    y_ex = 1.0
    W1_ex = np.array([[0.5, -0.5],   # poids partant de x1 vers h1, h2
                      [0.3, 0.8]])   # poids partant de x2 vers h1, h2
    W2_ex = np.array([1.0, -1.0])    # poids de h1, h2 vers la sortie

    def passe_221(W1, W2):
        z1 = x_ex @ W1
        h = sigmoide(z1)
        z2 = h @ W2
        y_chap = sigmoide(z2)
        perte = -(y_ex * np.log(y_chap) + (1 - y_ex) * np.log(1 - y_chap))
        return z1, h, z2, y_chap, perte

    z1_ex, h_ex, z2_ex, ychap_ex, perte_ex = passe_221(W1_ex, W2_ex)
    # Passe arrière
    delta2_ex = ychap_ex - y_ex
    gW2_ex = h_ex * delta2_ex
    delta1_ex = (W2_ex * delta2_ex) * h_ex * (1 - h_ex)
    gW1_ex = np.outer(x_ex, delta1_ex) + 0.0  # « + 0.0 » : affiche 0 au lieu de −0
    # Un pas de descente de gradient (taux 1) et la nouvelle perte
    perte_apres_ex = passe_221(W1_ex - 1.0 * gW1_ex, W2_ex - 1.0 * gW2_ex)[4]

    def _f(v):
        return np.array2string(np.asarray(v), precision=4, floatmode="fixed")

    print("Passe avant")
    print(f"  z1 = x·W1     = {_f(z1_ex)}")
    print(f"  h  = σ(z1)    = {_f(h_ex)}")
    print(f"  z2 = h·W2     = {z2_ex:.4f}")
    print(f"  ŷ  = σ(z2)    = {ychap_ex:.4f}")
    print(f"  ℒ  = −ln ŷ    = {perte_ex:.4f}")
    print("Passe arrière")
    print(f"  δ2 = ŷ − y    = {delta2_ex:.4f}")
    print(f"  ∂ℒ/∂W2 = h δ2 = {_f(gW2_ex)}")
    print(f"  σ'(z1) = h(1−h) = {_f(h_ex * (1 - h_ex))}")
    print(f"  δ1 = W2 δ2 ⊙ σ'(z1) = {_f(delta1_ex)}")
    print(f"  ∂ℒ/∂W1 = xᵀ δ1 =\n{_f(gW1_ex)}")
    print(f"Après un pas (taux 1) : ℒ passe de {perte_ex:.4f} à {perte_apres_ex:.4f}")
    return W1_ex, W2_ex, gW1_ex, gW2_ex, perte_apres_ex, perte_ex, x_ex, y_ex


@app.cell
def _(W1_ex, W2_ex, gW1_ex, gW2_ex, np, torch, x_ex, y_ex):
    # --- Le même calcul par autograd : PyTorch enregistre le graphe pendant la passe
    # avant, puis le parcourt à l'envers quand on appelle .backward().
    _W1 = torch.tensor(W1_ex, requires_grad=True)
    _W2 = torch.tensor(W2_ex, requires_grad=True)
    _x = torch.tensor(x_ex)
    _ychap = torch.sigmoid(torch.sigmoid(_x @ _W1) @ _W2)
    _perte = torch.nn.functional.binary_cross_entropy(_ychap, torch.tensor(y_ex, dtype=torch.float64))
    _perte.backward()

    # Le graphe de calcul, lu depuis la perte (chaque nœud connaît sa dérivée locale)
    _noeud, _chaine = _perte.grad_fn, []
    while _noeud is not None:
        _chaine.append(type(_noeud).__name__)
        _suivants = [n for n, _ in _noeud.next_functions if n is not None
                     and type(n).__name__ != "AccumulateGrad"]
        _noeud = _suivants[0] if _suivants else None
    print("Graphe (de la perte vers l'entrée) :", " ← ".join(_chaine))
    ecart_autograd_221 = max(np.abs(_W1.grad.numpy() - gW1_ex).max(),
                             np.abs(_W2.grad.numpy() - gW2_ex).max())
    print(f"Écart max entre nos gradients et ceux d'autograd : {ecart_autograd_221:.1e}")
    return (ecart_autograd_221,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Un MLP numpy de taille quelconque

    Le calcul du 2-2-1 se généralise couche par couche. Notations : $a_0 = X$ (un lot de
    $N$ exemples en lignes), puis pour chaque couche $\ell$ :
    $z_\ell = a_{\ell-1} W_\ell + b_\ell$ et $a_\ell = f(z_\ell)$ (pas d'activation sur la
    dernière couche, qui donne les logits). Passe arrière, en partant de
    $\delta_L = (\operatorname{softmax}(z_L) - Y)/N$ :

    $$\frac{\partial\mathcal{L}}{\partial W_\ell} = a_{\ell-1}^\top \delta_\ell,\qquad
    \frac{\partial\mathcal{L}}{\partial b_\ell} = \textstyle\sum_i \delta_{\ell,i},\qquad
    \delta_{\ell-1} = (\delta_\ell W_\ell^\top) \odot f'(z_{\ell-1}).$$

    Coût : une passe arrière coûte environ deux passes avant, quel que soit le nombre de
    poids. Les différences finies, elles, demandent deux passes avant **par poids**.
    """)


@app.cell
def _(np):
    # --- Activations : (fonction, dérivée). La dérivée reçoit z et a = f(z) ---
    ACTIVATIONS = {
        "sigmoïde": (lambda z: 1.0 / (1.0 + np.exp(-z)), lambda z, a: a * (1.0 - a)),
        "tanh": (np.tanh, lambda z, a: 1.0 - a * a),
        "ReLU": (lambda z: np.maximum(z, 0.0), lambda z, a: (z > 0).astype(z.dtype)),
    }

    def initialiser(tailles, rng, dtype=np.float64):
        """Poids ~ N(0, 1/n_entrée), biais nuls (même règle pour toutes les activations :
        le choix de l'initialisation selon l'activation est le sujet du chapitre 12)."""
        return [[(rng.standard_normal((n_in, n_out)) / np.sqrt(n_in)).astype(dtype),
                 np.zeros(n_out, dtype=dtype)]
                for n_in, n_out in zip(tailles[:-1], tailles[1:])]

    def propager(params, X, act):
        """Passe avant. Renvoie les logits et le cache (z, a) de chaque couche."""
        f = ACTIVATIONS[act][0]
        cache = [(None, X)]
        a = X
        for i, (W, b) in enumerate(params):
            z = a @ W + b
            a = z if i == len(params) - 1 else f(z)
            cache.append((z, a))
        return a, cache

    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)  # stabilité numérique, ne change rien au résultat
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def entropie_croisee(logits, y):
        """Perte moyenne −log softmax(z)_y, calculée de façon stable (log-sum-exp)."""
        m = logits.max(axis=1, keepdims=True)
        log_z = m[:, 0] + np.log(np.exp(logits - m).sum(axis=1))
        return float(np.mean(log_z - logits[np.arange(len(y)), y]))

    def retropropager(params, cache, y, act, oublier_derivee=False):
        """Passe arrière : gradients de l'entropie croisée moyenne pour tous les poids.

        `oublier_derivee=True` simule une erreur classique (on oublie f'(z)) : sert à voir
        que le contrôle par différences finies la détecte."""
        df = ACTIVATIONS[act][1]
        logits = cache[-1][0]
        delta = softmax(logits)
        delta[np.arange(len(y)), y] -= 1.0
        delta /= len(y)                                  # δ_L = (softmax − Y) / N
        grads = [None] * len(params)
        for i in range(len(params) - 1, -1, -1):
            a_prec = cache[i][1]
            grads[i] = [a_prec.T @ delta, delta.sum(axis=0)]
            if i > 0:
                z, a = cache[i]
                delta = delta @ params[i][0].T            # on remonte à travers W
                if not oublier_derivee:
                    delta = delta * df(z, a)              # … puis à travers l'activation
        return grads

    def exactitude(params, X, y, act):
        return float(np.mean(propager(params, X, act)[0].argmax(axis=1) == y))

    return ACTIVATIONS, entropie_croisee, initialiser, exactitude, propager, retropropager


@app.cell
def _(entropie_croisee, initialiser, np, propager, retropropager):
    # --- Contrôle du gradient par différences finies centrées ---
    # Petit réseau 3-5-4-3, 6 exemples, float64 : on compare CHAQUE dérivée partielle.
    def gradient_numerique(params, X, y, act, eps=1e-6):
        grads = []
        for W, b in params:
            g_couche = []
            for tableau in (W, b):
                g = np.zeros_like(tableau)
                for idx in np.ndindex(tableau.shape):
                    ancien = tableau[idx]
                    tableau[idx] = ancien + eps
                    plus = entropie_croisee(propager(params, X, act)[0], y)
                    tableau[idx] = ancien - eps
                    moins = entropie_croisee(propager(params, X, act)[0], y)
                    tableau[idx] = ancien
                    g[idx] = (plus - moins) / (2 * eps)
                g_couche.append(g)
            grads.append(g_couche)
        return grads

    def erreur_relative(g1, g2):
        a = np.concatenate([t.ravel() for c in g1 for t in c])
        b = np.concatenate([t.ravel() for c in g2 for t in c])
        return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b)))

    _rng = np.random.default_rng(42)
    _X = _rng.standard_normal((6, 3))
    _y = _rng.integers(0, 3, 6)
    controle_gradient = {}
    for _act in ["sigmoïde", "tanh", "ReLU"]:
        _p = initialiser([3, 5, 4, 3], np.random.default_rng(42))
        _num = gradient_numerique(_p, _X, _y, _act)
        _ana = retropropager(_p, propager(_p, _X, _act)[1], _y, _act)
        _bug = retropropager(_p, propager(_p, _X, _act)[1], _y, _act, oublier_derivee=True)
        controle_gradient[_act] = (erreur_relative(_num, _ana), erreur_relative(_num, _bug))
        _n = sum(t.size for c in _p for t in c)
        print(f"{_act:9s} ({_n} paramètres) : erreur relative {controle_gradient[_act][0]:.1e}"
              f" ; avec f'(z) oubliée : {controle_gradient[_act][1]:.2f}")
    return (controle_gradient,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Le XOR bruité : ce que le perceptron ne savait pas faire

    Quatre nuages de points : classe 0 autour de $(-1,-1)$ et $(1,1)$, classe 1 autour
    de $(-1,1)$ et $(1,-1)$ (bruit gaussien d'écart-type 0,35). Aucune droite ne sépare
    les deux classes (chapitre 06). On compare un modèle **sans couche cachée** (une
    régression logistique, frontière droite) à un MLP 2-16-2 (tanh), entraînés de la même
    façon : descente de gradient sur tout le lot.
    """)


@app.cell
def _(
    config,
    entropie_croisee,
    initialiser,
    np,
    exactitude,
    propager,
    retropropager,
):
    def xor_bruite(n, rng, bruit=0.35):
        centres = np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]], dtype=float)
        k = rng.integers(0, 4, n)
        return centres[k] + bruit * rng.standard_normal((n, 2)), (k >= 2).astype(int)

    def entrainer_plein_lot(tailles, act, X, y, pas, lr, graine=42):
        """Descente de gradient simple sur tout le lot ; renvoie params et courbe de perte."""
        params = initialiser(tailles, np.random.default_rng(graine))
        pertes = []
        for _ in range(pas):
            logits, cache = propager(params, X, act)
            pertes.append(entropie_croisee(logits, y))
            for (W, b), (gW, gb) in zip(params, retropropager(params, cache, y, act)):
                W -= lr * gW
                b -= lr * gb
        return params, pertes

    _rng = np.random.default_rng(42)
    X_xor, y_xor = xor_bruite(config["n_xor"], _rng)
    X_xor_test, y_xor_test = xor_bruite(2000, _rng)
    xor_modeles = {}
    for _nom, _tailles in [("sans couche cachée", [2, 2]), ("MLP 2-16-2 (tanh)", [2, 16, 2])]:
        _p, _pertes = entrainer_plein_lot(_tailles, "tanh", X_xor, y_xor,
                                          config["pas_xor"], config["lr_xor"])
        xor_modeles[_nom] = dict(params=_p, pertes=_pertes,
                                 exactitude=exactitude(_p, X_xor_test, y_xor_test, "tanh"))
        print(f"{_nom:20s} : perte finale {_pertes[-1]:.3f}, "
              f"exactitude test {xor_modeles[_nom]['exactitude']:.1%}")
    return X_xor, entrainer_plein_lot, xor_modeles, y_xor


@app.cell
def _(
    PALETTE,
    SMOKE,
    X_xor,
    mo,
    np,
    plt,
    propager,
    sauvegarder_figure,
    xor_modeles,
    y_xor,
):
    # --- Figure : frontières de décision des deux modèles ---
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4.4))
    _xx, _yy = np.meshgrid(np.linspace(-2.5, 2.5, 200), np.linspace(-2.5, 2.5, 200))
    _grille = np.c_[_xx.ravel(), _yy.ravel()]
    for _ax, (_nom, _m) in zip(_axes, xor_modeles.items()):
        _proba = propager(_m["params"], _grille, "tanh")[0]
        _proba = np.exp(_proba - _proba.max(1, keepdims=True))
        _proba = (_proba[:, 1] / _proba.sum(1)).reshape(_xx.shape)
        _ax.contourf(_xx, _yy, _proba, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.45)
        _ax.contour(_xx, _yy, _proba, levels=[0.5], colors="k", linewidths=1.2)
        for _c, _coul in [(0, PALETTE[0]), (1, PALETTE[3])]:
            _sel = y_xor == _c
            _ax.scatter(X_xor[_sel, 0], X_xor[_sel, 1], s=10, color=_coul,
                        edgecolors="white", linewidths=0.3, label=f"classe {_c}")
        _ax.set(title=f"{_nom} : {100 * _m['exactitude']:.1f} % en test".replace(".", ","), xlabel="x₁", ylabel="x₂",
                aspect="equal")
    _axes[0].legend(loc="upper right", frameon=True)
    _fig.tight_layout()
    _chemin = sauvegarder_figure(_fig, "xor-smoke" if SMOKE else "xor", mo.notebook_dir() / "figures")
    print(f"Figure : {_chemin[0]}")
    _fig


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. La profondeur : sigmoïde, tanh, ReLU

    Même XOR, mais un réseau de **8 couches cachées** de 32 neurones, avec la même
    initialisation pour les trois activations. Deux mesures :

    1. à l'initialisation, la norme du gradient de chaque matrice de poids, de la
       première couche à la dernière : pendant la passe arrière, le signal d'erreur est
       multiplié à chaque couche par $f'(z)$, or $\sigma'(z) \le 0{,}25$ ;
    2. la courbe de perte pendant l'entraînement (même taux, même nombre de pas).
    """)


@app.cell
def _(
    X_xor,
    config,
    entrainer_plein_lot,
    initialiser,
    np,
    exactitude,
    propager,
    retropropager,
    y_xor,
):
    _tailles = [2] + [config["largeur"]] * config["profondeur"] + [2]
    profond = {}
    for _act in ["sigmoïde", "tanh", "ReLU"]:
        _p0 = initialiser(_tailles, np.random.default_rng(42))
        _g0 = retropropager(_p0, propager(_p0, X_xor, _act)[1], y_xor, _act)
        _normes = [float(np.linalg.norm(gW)) for gW, _ in _g0]
        _p, _pertes = entrainer_plein_lot(_tailles, _act, X_xor, y_xor,
                                          config["pas_profond"], config["lr_profond"])
        profond[_act] = dict(normes=_normes, pertes=_pertes,
                             exactitude=exactitude(_p, X_xor, y_xor, _act))
        print(f"{_act:9s} : gradient couche 1 / couche {len(_normes)} = "
              f"{_normes[0] / _normes[-1]:.1e} ; perte {_pertes[0]:.3f} → {_pertes[-1]:.3f} ; "
              f"exactitude (entraînement) {profond[_act]['exactitude']:.1%}")
    return (profond,)


@app.cell
def _(PALETTE, SMOKE, mo, np, plt, profond, sauvegarder_figure):
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 4))
    _couleurs = {"sigmoïde": PALETTE[3], "tanh": PALETTE[1], "ReLU": PALETTE[0]}
    for _act, _r in profond.items():
        _a1.semilogy(np.arange(1, len(_r["normes"]) + 1), _r["normes"], "o-",
                     color=_couleurs[_act], label=_act)
        _a2.plot(_r["pertes"], color=_couleurs[_act], label=_act)
    _a1.set(xlabel="couche (1 = la plus proche de l'entrée)", ylabel="‖∂ℒ/∂W‖ (échelle log)",
            title="Gradient par couche, à l'initialisation")
    _a2.axhline(np.log(2), color="0.5", ls=":", lw=1)
    _a2.text(len(next(iter(profond.values()))["pertes"]) * 0.55, np.log(2) + 0.02,
             "ln 2 : réponse au hasard", fontsize=9, color="0.4")
    _a2.set(xlabel="pas de descente de gradient", ylabel="entropie croisée",
            title="Entraînement, 8 couches cachées")
    _a1.legend()
    _a2.legend()
    _fig.tight_layout()
    _chemin = sauvegarder_figure(_fig, "profondeur-smoke" if SMOKE else "profondeur",
                                 mo.notebook_dir() / "figures")
    print(f"Figure : {_chemin[0]}")
    _fig


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. MNIST en numpy

    Le même code, sans rien changer à `propager` ni `retropropager`, sur un vrai jeu :
    MLP 784-256-256-10 (ReLU), SGD par mini-lots de 64, taux 0,1, en float32 (moitié moins
    de mémoire que float64, et assez de chiffres significatifs pour apprendre).
    """)


@app.cell
def _(mo):
    bouton = mo.ui.run_button(label="Entraîner (MNIST, numpy puis PyTorch)")
    bouton
    return (bouton,)


@app.cell
def _(config, np):
    from ia_commun.donnees import mnist_tenseurs

    _x, _y = mnist_tenseurs(train=True)
    X_train = _x[: config["n_mnist"]].reshape(-1, 784).numpy().astype(np.float32)
    y_train = _y[: config["n_mnist"]].numpy()
    _x, _y = mnist_tenseurs(train=False)
    X_test = _x.reshape(-1, 784).numpy().astype(np.float32)
    y_test = _y.numpy()
    print(f"MNIST : {len(X_train)} images d'entraînement, {len(X_test)} de test, pixels dans [0, 1]")
    return X_test, X_train, y_test, y_train


@app.cell
def _(
    X_test,
    X_train,
    bouton,
    config,
    entropie_croisee,
    initialiser,
    mo,
    np,
    exactitude,
    propager,
    retropropager,
    time,
    y_test,
    y_train,
):
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Entraîner » pour lancer MNIST.*"))

    _tailles = [784, *config["cachees_mnist"], 10]
    params_numpy = initialiser(_tailles, np.random.default_rng(42), dtype=np.float32)
    _rng = np.random.default_rng(42)
    _lr, _lot = np.float32(config["lr_mnist"]), config["lot_mnist"]
    histo_numpy = dict(perte=[], exactitude_test=[], duree=[])
    for _ep in range(config["epoques_mnist"]):
        _t0 = time.perf_counter()
        _ordre = _rng.permutation(len(X_train))
        _somme = 0.0
        for _deb in range(0, len(_ordre), _lot):
            _idx = _ordre[_deb:_deb + _lot]
            _logits, _cache = propager(params_numpy, X_train[_idx], "ReLU")
            _somme += entropie_croisee(_logits, y_train[_idx]) * len(_idx)
            for (_W, _b), (_gW, _gb) in zip(params_numpy,
                                            retropropager(params_numpy, _cache, y_train[_idx], "ReLU")):
                _W -= _lr * _gW
                _b -= _lr * _gb
        histo_numpy["duree"].append(time.perf_counter() - _t0)
        histo_numpy["perte"].append(_somme / len(_ordre))
        histo_numpy["exactitude_test"].append(exactitude(params_numpy, X_test, y_test, "ReLU"))
        print(f"numpy, époque {_ep + 1} : perte {histo_numpy['perte'][-1]:.4f}, "
              f"exactitude test {histo_numpy['exactitude_test'][-1]:.2%}, "
              f"{histo_numpy['duree'][-1]:.2f} s")
    return (histo_numpy,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Le même modèle en PyTorch

    D'abord une vérification : mêmes poids, même lot, en float64 sur CPU ; les gradients
    d'autograd doivent coïncider avec ceux de `retropropager` à l'arrondi près. Ensuite
    l'entraînement complet, sur le GPU si disponible. Différence de code : la passe
    arrière tient en une ligne, `perte.backward()`.
    """)


@app.cell
def _(X_train, config, histo_numpy, initialiser, nn, np, propager, retropropager, torch, y_train):
    # histo_numpy en argument : cette cellule attend la fin de la partie 5 (et le bouton).
    _ = histo_numpy

    def creer_mlp():
        _couches, _n = [], 784
        for _h in config["cachees_mnist"]:
            _couches += [nn.Linear(_n, _h), nn.ReLU()]
            _n = _h
        return nn.Sequential(*_couches, nn.Linear(_n, 10))

    _p = initialiser([784, *config["cachees_mnist"], 10], np.random.default_rng(7))
    _Xb, _yb = X_train[:64].astype(np.float64), y_train[:64]
    _g_numpy = retropropager(_p, propager(_p, _Xb, "ReLU")[1], _yb, "ReLU")

    _m = creer_mlp().double()
    with torch.no_grad():
        for _lin, (_W, _b) in zip([c for c in _m if isinstance(c, nn.Linear)], _p):
            _lin.weight.copy_(torch.from_numpy(_W.T))   # nn.Linear range W transposée
            _lin.bias.copy_(torch.from_numpy(_b))
    _perte = nn.functional.cross_entropy(_m(torch.from_numpy(_Xb)), torch.from_numpy(_yb))
    _perte.backward()
    ecart_autograd_mnist = max(
        float(np.abs(_lin.weight.grad.numpy().T - gW).max())
        for _lin, (gW, _) in zip([c for c in _m if isinstance(c, nn.Linear)], _g_numpy))
    print(f"Gradients numpy contre autograd ({sum(t.size for c in _p for t in c):,} poids) : "
          f"écart max {ecart_autograd_mnist:.1e}".replace(",", " "))
    return creer_mlp, ecart_autograd_mnist


@app.cell
def _(X_test, X_train, config, creer_mlp, device, nn, time, torch, y_test, y_train):
    torch.manual_seed(42)
    modele = creer_mlp().to(device)
    _optim = torch.optim.SGD(modele.parameters(), lr=config["lr_mnist"])
    _Xtr = torch.from_numpy(X_train).to(device)
    _ytr = torch.from_numpy(y_train).to(device)
    _Xte = torch.from_numpy(X_test).to(device)
    _yte = torch.from_numpy(y_test).to(device)
    _gen = torch.Generator().manual_seed(42)
    _lot = config["lot_mnist"]
    histo_torch = dict(perte=[], exactitude_test=[], duree=[])
    for _ep in range(config["epoques_mnist"]):
        _t0 = time.perf_counter()
        _ordre = torch.randperm(len(_Xtr), generator=_gen).to(device)
        _somme = torch.zeros((), device=device)
        for _deb in range(0, len(_ordre), _lot):
            _idx = _ordre[_deb:_deb + _lot]
            _perte = nn.functional.cross_entropy(modele(_Xtr[_idx]), _ytr[_idx])
            _optim.zero_grad()
            _perte.backward()          # toute la passe arrière
            _optim.step()
            _somme += _perte.detach() * len(_idx)
        with torch.no_grad():
            _prec = (modele(_Xte).argmax(1) == _yte).float().mean().item()
        histo_torch["perte"].append(_somme.item() / len(_ordre))
        histo_torch["duree"].append(time.perf_counter() - _t0)
        histo_torch["exactitude_test"].append(_prec)
        print(f"PyTorch ({device}), époque {_ep + 1} : perte {histo_torch['perte'][-1]:.4f}, "
              f"exactitude test {_prec:.2%}, {histo_torch['duree'][-1]:.2f} s")
    return (histo_torch,)


@app.cell
def _(PALETTE, SMOKE, histo_numpy, histo_torch, mo, np, plt, sauvegarder_figure):
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 4))
    _ep = np.arange(1, len(histo_numpy["perte"]) + 1)
    for _h, _nom, _c, _ls in [(histo_numpy, "numpy (à la main)", PALETTE[0], "-"),
                              (histo_torch, "PyTorch (autograd)", PALETTE[1], "--")]:
        _a1.plot(_ep, _h["perte"], _ls, marker="o", color=_c, label=_nom)
        _a2.plot(_ep, 100 * np.array(_h["exactitude_test"]), _ls, marker="o", color=_c, label=_nom)
    _a1.set(xlabel="époque", ylabel="entropie croisée (entraînement)", title="Perte")
    _a2.set(xlabel="époque", ylabel="exactitude en test (%)", title="Exactitude")
    _a1.legend()
    _fig.tight_layout()
    _chemin = sauvegarder_figure(_fig, "mnist-smoke" if SMOKE else "mnist", mo.notebook_dir() / "figures")
    print(f"Figure : {_chemin[0]}")
    _fig


@app.cell
def _(
    SMOKE,
    controle_gradient,
    ecart_autograd_221,
    ecart_autograd_mnist,
    histo_numpy,
    histo_torch,
    mo,
    np,
    perte_apres_ex,
    perte_ex,
    profond,
    xor_modeles,
):
    # --- Bilan : seuils du README. En smoke, on vérifie seulement que tout tourne. ---
    _prec_lin = xor_modeles["sans couche cachée"]["exactitude"]
    _prec_mlp = xor_modeles["MLP 2-16-2 (tanh)"]["exactitude"]
    _ratio = {a: r["normes"][0] / r["normes"][-1] for a, r in profond.items()}
    controles = [
        ("2-2-1 : la perte baisse après un pas", perte_apres_ex < perte_ex),
        ("2-2-1 : écart avec autograd < 1e-12", ecart_autograd_221 < 1e-12),
        ("différences finies : erreur relative < 1e-7 (3 activations)",
         all(v[0] < 1e-7 for v in controle_gradient.values())),
        ("différences finies : f'(z) oubliée détectée (> 1e-2)",
         all(v[1] > 1e-2 for v in controle_gradient.values())),
        ("XOR : sans couche cachée ≤ 60 %", _prec_lin <= 0.60),
        ("XOR : MLP ≥ 97 %", _prec_mlp >= 0.97),
        ("profond : gradient couche 1 / couche 9 en sigmoïde < 1e-4", _ratio["sigmoïde"] < 1e-4),
        ("profond : ce rapport > 1e-2 en ReLU et en tanh",
         _ratio["ReLU"] > 1e-2 and _ratio["tanh"] > 1e-2),
        ("profond : sigmoïde bloquée (perte finale > 0,65), ReLU et tanh < 0,1",
         profond["sigmoïde"]["pertes"][-1] > 0.65
         and max(profond["ReLU"]["pertes"][-1], profond["tanh"]["pertes"][-1]) < 0.1),
        ("MNIST numpy ≥ 97,5 % en test", histo_numpy["exactitude_test"][-1] >= 0.975),
        ("MNIST PyTorch ≥ 97,5 % en test", histo_torch["exactitude_test"][-1] >= 0.975),
        ("numpy contre autograd (MNIST, float64) : écart < 1e-10", ecart_autograd_mnist < 1e-10),
    ]
    _lignes = "\n".join(f"| {nom} | {'oui' if ok else ('non vérifié (smoke)' if SMOKE else '**NON**')} |"
                         for nom, ok in controles)
    print("\n".join(f"[{'ok' if ok else ('non vérifié (smoke)' if SMOKE else 'ÉCHEC')}] {nom}"
                    for nom, ok in controles))
    print(f"MNIST : numpy {np.mean(histo_numpy['duree']):.2f} s/époque, "
          f"PyTorch {np.mean(histo_torch['duree']):.2f} s/époque")
    _echecs = [nom for nom, ok in controles if not ok]
    if not SMOKE and _echecs:
        raise SystemExit("Seuils non atteints : " + " ; ".join(_echecs))
    mo.md(f"**Bilan** ({'smoke : seuils non vérifiés' if SMOKE else 'mode complet'})\n\n"
          f"| Contrôle | Atteint |\n|---|---|\n{_lignes}")


if __name__ == "__main__":
    app.run()
