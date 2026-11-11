# Exercice 01 : la loss choisie change le modèle appris.
#
#   make ex NUM=01              → éditeur marimo (interactif, bouton « Lancer »)
#   make ex NUM=01 MODE=run     → script complet (vérifie les seuils du README)
#   make ex NUM=01 MODE=smoke   → test rapide, < 30 s
#
# Tout est en numpy « from scratch », puis comparé à scikit-learn et à PyTorch.
# Aucun téléchargement, calcul sur CPU (quelques secondes).

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from ia_commun import PALETTE, is_smoke, sauvegarder_figure, seed_everything, style_commun

    return (
        PALETTE,
        is_smoke,
        mo,
        np,
        plt,
        sauvegarder_figure,
        seed_everything,
        style_commun,
        time,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Exercice 01 : même données, autre loss, autre modèle

    **Objectif.** Constater qu'à données et modèle identiques, changer la fonction de
    perte change le résultat, et comprendre pourquoi en regardant la **dérivée** de
    chaque perte (le « poids » qu'elle donne à chaque point).

    1. **Modèle constant** : la meilleure constante au sens de la MSE est la moyenne,
       au sens de la MAE la médiane.
    2. **Régression** avec 10 % de points aberrants : MSE, MAE et Huber en numpy, puis
       scikit-learn.
    3. **Classification** : moindres carrés, log-loss (régression logistique) et hinge
       (SVM linéaire), avec et sans points « trop bien classés ».
    4. **Softmax, température, entropie croisée, KL** : vérification du gradient
       $\partial \mathcal{L}/\partial z = p - y$ et comparaison avec PyTorch.

    Les losses optimisées sont précisément l'objet de l'exercice :
    $\ell_{\text{MSE}} = r^2$, $\ell_{\text{MAE}} = |r|$,
    $\ell_{\text{Huber}} = \tfrac12 r^2$ si $|r| \le \delta$, sinon $\delta(|r| - \tfrac12\delta)$
    (résidu $r = \hat y - y$) ; $\ell_{\log} = \log(1 + e^{-m})$ et
    $\ell_{\text{hinge}} = \max(0, 1 - m)$ (marge $m = y\,f(x)$, $y \in \{-1, +1\}$).
    """)


@app.cell
def _(is_smoke, seed_everything, style_commun, time):
    T0 = time.perf_counter()  # chronomètre de l'ensemble des calculs
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        n_regression=40 if SMOKE else 100,     # points d'entraînement (régression)
        part_aberrants=0.10,                   # 10 % de points aberrants
        decalage_aberrants=-25.0,              # capteur défaillant : mesure trop basse de 25
        delta_huber=1.0,                       # seuil de Huber (même valeur que torch par défaut)
        n_classe=30 if SMOKE else 100,         # points par classe (classification)
        n_lointains=6 if SMOKE else 20,        # points +1 très loin, du BON côté
        n_test=200 if SMOKE else 1000,         # points de test par classe / de régression
        iterations=300 if SMOKE else 5000,     # pas de descente de gradient
        lam=1e-2,                              # pénalité L2 (logistique et hinge)
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SEED, SMOKE, T0, config


@app.cell
def _(mo):
    # Le bouton évite de tout relancer à chaque modification dans l'éditeur.
    bouton = mo.ui.run_button(label="Lancer les calculs")
    bouton
    return (bouton,)


@app.cell
def _(bouton, mo, np):
    # ===================== 1. Le modèle le plus simple : une constante ==================
    # En script (make ex MODE=run|smoke), personne ne clique : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer les calculs ».*"))

    # Salaires mensuels (euros) d'une équipe de 11 personnes, dont la dirigeante.
    salaires = np.array([2000, 2100, 2300, 2400, 2500, 2600, 2700, 2900, 3100, 3300, 25000.0])
    candidats = np.arange(0, 30001, 1.0)          # constantes c essayées, au centime près inutile
    residus = candidats[:, None] - salaires[None, :]

    def huber(r, delta):
        """Quadratique près de 0, linéaire au-delà de delta : continue et dérivable."""
        a = np.abs(r)
        return np.where(a <= delta, 0.5 * r**2, delta * (a - 0.5 * delta))

    pertes_constante = {
        "MSE": (residus**2).mean(axis=1),
        "MAE": np.abs(residus).mean(axis=1),
        "Huber (δ = 1 000 €)": huber(residus, 1000.0).mean(axis=1),
    }
    meilleures_constantes = {nom: candidats[p.argmin()] for nom, p in pertes_constante.items()}
    for _nom, _c in meilleures_constantes.items():
        print(f"{_nom:22s} → meilleure constante {_c:8.0f} €")
    print(f"Moyenne {salaires.mean():.0f} €, médiane {np.median(salaires):.0f} €")
    return huber, meilleures_constantes, salaires


@app.cell
def _(config, np):
    # ===================== 2. Régression avec points aberrants ==========================
    def donnees_regression(n, rng, part_aberrants=0.0, decalage=0.0):
        """y = 2x + 1 + bruit gaussien ; les points les plus à droite peuvent être faussés."""
        x = rng.uniform(0, 10, n)
        y = 2 * x + 1 + rng.normal(0, 1, n)
        k = round(part_aberrants * n)
        if k:
            # Les k plus grands x : un capteur qui sature en haut de gamme, par exemple.
            y[np.argsort(x)[-k:]] += decalage
        return x, y

    rng_reg = np.random.default_rng(42)
    x_reg, y_reg = donnees_regression(config["n_regression"], rng_reg,
                                      config["part_aberrants"], config["decalage_aberrants"])
    x_test_reg, y_test_reg = donnees_regression(config["n_test"], rng_reg)  # propres
    return x_reg, x_test_reg, y_reg, y_test_reg


@app.cell
def _(config, huber, np, x_reg, y_reg):
    # Descente de gradient « from scratch » sur un modèle affine y ≈ a·x + b.
    # On centre-réduit x pour que la descente converge vite, puis on revient à l'échelle.
    mu, sigma = x_reg.mean(), x_reg.std()
    A = np.c_[(x_reg - mu) / sigma, np.ones_like(x_reg)]   # colonnes : x normalisé, 1

    # Dérivée de chaque perte par rapport à la prédiction : c'est elle qui « tire » la droite.
    derivees = {
        "MSE": lambda r: 2 * r,                                    # proportionnelle au résidu
        "MAE": lambda r: np.sign(r),                               # ±1, quel que soit le résidu
        "Huber": lambda r: np.clip(r, -config["delta_huber"], config["delta_huber"]),  # bornée
    }
    valeurs = {
        "MSE": lambda r: r**2,
        "MAE": lambda r: np.abs(r),
        "Huber": lambda r: huber(r, config["delta_huber"]),
    }

    def ajuster(nom):
        w = np.zeros(2)
        for t in range(config["iterations"]):
            r = A @ w - y_reg
            g = A.T @ derivees[nom](r) / len(y_reg)
            # Pas décroissant pour la MAE (sous-gradient), constant sinon.
            lr = 1.0 / np.sqrt(t + 1) if nom == "MAE" else 0.4
            w -= lr * g
        pente = w[0] / sigma
        return pente, w[1] - pente * mu

    droites = {nom: ajuster(nom) for nom in derivees}
    # Contrôle : la MSE a aussi une solution exacte (moindres carrés).
    exacte = np.linalg.lstsq(np.c_[x_reg, np.ones_like(x_reg)], y_reg, rcond=None)[0]
    print(f"MSE exacte (lstsq) : pente {exacte[0]:.3f}, ordonnée {exacte[1]:.3f}")
    for _nom, (_a, _b) in droites.items():
        print(f"{_nom:6s} (descente)  : pente {_a:.3f}, ordonnée {_b:.3f}   (vraie : 2 et 1)")
    return droites, exacte, valeurs


@app.cell
def _(droites, np, valeurs, x_reg, x_test_reg, y_reg, y_test_reg):
    # Loss contre métrique : chaque modèle est le meilleur pour SA loss d'entraînement,
    # mais la métrique qui compte ici est l'erreur sur des données propres.
    tableau_reg = {}
    for _nom, (_a, _b) in droites.items():
        r_train = _a * x_reg + _b - y_reg
        r_test = _a * x_test_reg + _b - y_test_reg
        tableau_reg[_nom] = {
            "pente": _a,
            **{f"train {k}": valeurs[k](r_train).mean() for k in valeurs},
            "test MAE (propre)": np.abs(r_test).mean(),
        }
    print(f"{'modèle':8s}" + "".join(f"{k:>20s}" for k in next(iter(tableau_reg.values()))))
    for _nom, _ligne in tableau_reg.items():
        print(f"{_nom:8s}" + "".join(f"{v:20.3f}" for v in _ligne.values()))
    return (tableau_reg,)


@app.cell
def _(droites, np, x_reg, y_reg):
    # Même chose avec scikit-learn.
    from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor

    X1 = x_reg[:, None]
    sk_reg = {
        "MSE": LinearRegression().fit(X1, y_reg),
        # La régression quantile à 50 % minimise la MAE (alpha=0 : sans pénalité).
        "MAE": QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs").fit(X1, y_reg),
        # HuberRegressor estime aussi l'échelle du bruit : son seuil epsilon=1,35 porte sur
        # des résidus divisés par cette échelle. Ce n'est donc pas exactement notre δ = 1.
        "Huber": HuberRegressor(epsilon=1.35, alpha=0.0).fit(X1, y_reg),
    }
    ecarts_sklearn = {}
    for _nom, _m in sk_reg.items():
        _a = float(_m.coef_[0])
        ecarts_sklearn[_nom] = abs(_a - droites[_nom][0])
        print(f"{_nom:6s} scikit-learn : pente {_a:.3f}  (numpy : {droites[_nom][0]:.3f})")
    _ = np
    return (ecarts_sklearn,)


@app.cell
def _(config, np):
    # ===================== 3. Classification : moindres carrés, log-loss, hinge =========
    def deux_nuages(n, rng):
        """Deux nuages gaussiens, classe -1 autour de (-2, 0), classe +1 autour de (2, 0)."""
        Xm = rng.normal([-2.0, 0.0], 1.0, (n, 2))
        Xp = rng.normal([2.0, 0.0], 1.0, (n, 2))
        return np.r_[Xm, Xp], np.r_[-np.ones(n), np.ones(n)]

    rng_cl = np.random.default_rng(42)
    X_cl, y_cl = deux_nuages(config["n_classe"], rng_cl)
    X_test_cl, y_test_cl = deux_nuages(config["n_test"], rng_cl)
    # Points de classe +1 TRÈS loin du bon côté : ils sont « trop bien classés ».
    X_loin = rng_cl.normal([12.0, 3.0], 0.7, (config["n_lointains"], 2))
    X_cl_loin = np.r_[X_cl, X_loin]
    y_cl_loin = np.r_[y_cl, np.ones(config["n_lointains"])]
    return X_cl, X_cl_loin, X_loin, X_test_cl, y_cl, y_cl_loin, y_test_cl


@app.cell
def _(config, np):
    def avec_biais(X):
        return np.c_[X, np.ones(len(X))]

    # Poids de chaque point dans le gradient = -dℓ/df (signe compris) en fonction de la marge
    # m = y·f(x). C'est TOUTE la différence entre les trois méthodes.
    poids_classif = {
        # (f - y)² = (m - 1)² quand y = ±1 : un point « trop bien classé » (m > 1) est PÉNALISÉ
        "moindres carrés": lambda m: -2 * (m - 1),
        "log-loss": lambda m: 1 / (1 + np.exp(np.clip(m, -30, 30))),   # σ(-m), tend vers 0
        "hinge": lambda m: (m < 1).astype(float),                      # 0 au-delà de la marge
    }

    def entrainer(X, y, nom):
        """Modèle linéaire f(x) = w·x + b entraîné par descente de gradient."""
        if nom == "moindres carrés":
            # Solution exacte, sans pénalité (la régression linéaire appliquée aux étiquettes ±1).
            return np.linalg.lstsq(avec_biais(X), y, rcond=None)[0]
        Ab = avec_biais(X)
        w = np.zeros(3)
        for t in range(config["iterations"]):
            m = y * (Ab @ w)
            g = -(Ab * (y * poids_classif[nom](m))[:, None]).mean(axis=0)
            g[:2] += config["lam"] * w[:2]       # pénalité L2, pas sur le biais
            lr = 0.5 / np.sqrt(t + 1) if nom == "hinge" else 0.1
            w -= lr * g
        return w

    def exactitude(w, X, y):
        return float((np.sign(avec_biais(X) @ w) == y).mean())

    return avec_biais, entrainer, poids_classif, exactitude


@app.cell
def _(
    X_cl,
    X_cl_loin,
    X_loin,
    X_test_cl,
    avec_biais,
    entrainer,
    np,
    poids_classif,
    exactitude,
    y_cl,
    y_cl_loin,
    y_test_cl,
):
    modeles_cl = {}
    exactitudes_cl = {}
    for _jeu, (_X, _y) in {"propre": (X_cl, y_cl), "avec lointains": (X_cl_loin, y_cl_loin)}.items():
        for _nom in poids_classif:
            _w = entrainer(_X, _y, _nom)
            modeles_cl[(_jeu, _nom)] = _w
            exactitudes_cl[(_jeu, _nom)] = exactitude(_w, X_test_cl, y_test_cl)

    # Force = -dℓ/df, signe compris : négative = pousse ces points vers la frontière.
    print(f"{'perte':16s}{'propre':>10s}{'avec lointains':>17s}   force moyenne des points lointains")
    poids_lointains = {}
    for _nom in poids_classif:
        _w = modeles_cl[("avec lointains", _nom)]
        _m = avec_biais(X_loin) @ _w           # y = +1 : marge = f(x)
        poids_lointains[_nom] = float(poids_classif[_nom](_m).mean())
        print(f"{_nom:16s}{exactitudes_cl[('propre', _nom)]:>10.1%}"
              f"{exactitudes_cl[('avec lointains', _nom)]:>17.1%}   {poids_lointains[_nom]:+.2e}")

    # Vecteurs de support : seuls les points de marge < 1 comptent pour la hinge.
    _w = modeles_cl[("avec lointains", "hinge")]
    n_support = int((y_cl_loin * (avec_biais(X_cl_loin) @ _w) < 1).sum())
    print(f"hinge : {n_support} points sur {len(y_cl_loin)} ont une marge < 1 (les autres ne comptent pas)")
    return modeles_cl, n_support, poids_lointains, exactitudes_cl


@app.cell
def _(X_cl_loin, X_test_cl, config, np, exactitudes_cl, y_cl_loin, y_test_cl):
    # Mêmes modèles avec scikit-learn (même pénalité : C = 1 / (λ n)).
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    _C = 1 / (config["lam"] * len(y_cl_loin))
    sk_cl = {
        "log-loss": LogisticRegression(C=_C).fit(X_cl_loin, y_cl_loin),
        "hinge": SVC(kernel="linear", C=_C).fit(X_cl_loin, y_cl_loin),
    }
    for _nom, _m in sk_cl.items():
        _p = float((_m.predict(X_test_cl) == y_test_cl).mean())
        print(f"{_nom:9s} scikit-learn : exactitude {_p:.1%}  (numpy : {exactitudes_cl[('avec lointains', _nom)]:.1%})")
    n_support_svc = int(sk_cl["hinge"].n_support_.sum())
    print(f"SVC : {n_support_svc} vecteurs de support")
    _ = np
    return (n_support_svc,)


@app.cell
def _(np):
    # ===================== 4. Softmax, température, entropie croisée, KL ================
    def softmax(z, T=1.0):
        """Softmax stable : on retire le max avant l'exponentielle (même calcul que la démo)."""
        z = np.asarray(z, dtype=float) / T
        e = np.exp(z - z.max())
        return e / e.sum()

    def entropie(p):
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())          # en nats

    logits = np.array([2.0, 1.0, 0.1, -1.0])
    for _T in (0.5, 1.0, 2.0):
        _p = softmax(logits, _T)
        print(f"T = {_T:3.1f} : p = {np.round(_p, 3)}, entropie {entropie(_p):.3f} nats")

    # Entropie croisée avec la classe 0 pour cible, et son gradient par rapport aux logits.
    cible = 0
    def entropie_croisee(z):
        return -np.log(softmax(z)[cible])

    y_un = np.eye(len(logits))[cible]
    grad_formule = softmax(logits) - y_un          # p - y
    eps = 1e-6
    grad_numerique = np.array([
        (entropie_croisee(logits + eps * e) - entropie_croisee(logits - eps * e)) / (2 * eps)
        for e in np.eye(len(logits))
    ])
    ecart_gradient = float(np.abs(grad_formule - grad_numerique).max())
    print(f"CE = {entropie_croisee(logits):.4f} ; gradient p - y = {np.round(grad_formule, 4)} ;"
          f" écart avec les différences finies {ecart_gradient:.1e}")

    # KL n'est pas symétrique.
    P = softmax(logits)
    Q = np.full(4, 0.25)
    kl_pq = float((P * np.log(P / Q)).sum())
    kl_qp = float((Q * np.log(Q / P)).sum())
    print(f"KL(P‖Q) = {kl_pq:.4f}, KL(Q‖P) = {kl_qp:.4f} (P = softmax, Q = uniforme)")
    return P, Q, ecart_gradient, entropie, kl_pq, logits, softmax


@app.cell
def _(P, Q, config, huber, kl_pq, logits, np, softmax):
    # Les mêmes pertes dans PyTorch : on vérifie que nos formules donnent les mêmes valeurs.
    import torch
    import torch.nn.functional as F

    _rng = np.random.default_rng(42)
    _pred, _vrai = _rng.normal(size=50), _rng.normal(size=50)
    _pt, _vt = torch.tensor(_pred), torch.tensor(_vrai)
    _z, _yb = _rng.normal(size=50), (_rng.random(50) > 0.5).astype(float)
    comparaisons = {
        "MSE": (np.mean((_pred - _vrai) ** 2), F.mse_loss(_pt, _vt).item()),
        "MAE": (np.mean(np.abs(_pred - _vrai)), F.l1_loss(_pt, _vt).item()),
        "Huber": (huber(_pred - _vrai, config["delta_huber"]).mean(),
                  F.huber_loss(_pt, _vt, delta=config["delta_huber"]).item()),
        # BCE sur logits : log(1 + e^{-z}) si y = 1, log(1 + e^{z}) si y = 0
        "BCE (logits)": (np.mean(np.logaddexp(0, -_z) * _yb + np.logaddexp(0, _z) * (1 - _yb)),
                         F.binary_cross_entropy_with_logits(torch.tensor(_z), torch.tensor(_yb)).item()),
        "CE (softmax)": (-np.log(softmax(logits)[0]),
                         F.cross_entropy(torch.tensor(logits)[None], torch.tensor([0])).item()),
        # kl_div attend log Q en entrée et P en cible : il calcule KL(P‖Q)
        "KL": (kl_pq, F.kl_div(torch.tensor(np.log(Q)), torch.tensor(P), reduction="sum").item()),
    }
    ecart_torch = max(abs(a - b) for a, b in comparaisons.values())
    for _nom, (_a, _b) in comparaisons.items():
        print(f"{_nom:13s} numpy {_a:.6f}   torch {_b:.6f}")
    print(f"écart maximal : {ecart_torch:.1e}")
    return (ecart_torch,)


@app.cell
def _(
    PALETTE,
    SMOKE,
    droites,
    mo,
    np,
    plt,
    sauvegarder_figure,
    x_reg,
    y_reg,
):
    # --- Figure 1 : les trois droites sur les données avec points aberrants ---
    fig_reg, ax_reg = plt.subplots(figsize=(7, 4.2))
    ax_reg.scatter(x_reg, y_reg, s=14, color="0.45", label="données (10 % aberrantes)")
    _xs = np.linspace(0, 10, 2)
    ax_reg.plot(_xs, 2 * _xs + 1, color="0.2", ls=":", lw=1.5, label="vraie droite y = 2x + 1")
    for _i, (_nom, (_a, _b)) in enumerate(droites.items()):
        # MAE en tirets : sa droite est presque confondue avec celle de Huber
        ax_reg.plot(_xs, _a * _xs + _b, color=PALETTE[_i], lw=2,
                    ls="--" if _nom == "MAE" else "-", zorder=4 if _nom == "MAE" else 3,
                    label=f"{_nom} : pente {_a:.2f}".replace(".", ","))
    ax_reg.set(xlabel="x", ylabel="y", title="Même modèle affine, trois pertes")
    ax_reg.legend(loc="upper left")
    _chemin = sauvegarder_figure(fig_reg, "regression-aberrants" + ("-smoke" if SMOKE else ""),
                                 mo.notebook_dir() / "figures")
    print(f"Figure : {_chemin[0]}")
    fig_reg


@app.cell
def _(
    PALETTE,
    SMOKE,
    X_cl_loin,
    mo,
    modeles_cl,
    np,
    plt,
    poids_classif,
    exactitudes_cl,
    sauvegarder_figure,
    y_cl_loin,
):
    # --- Figure 2 : frontières de décision, avec les points lointains ---
    fig_cl, ax_cl = plt.subplots(figsize=(8, 4.2))
    _moins = y_cl_loin < 0
    ax_cl.scatter(*X_cl_loin[_moins].T, s=12, marker="o", color="0.55", label="classe −1")
    ax_cl.scatter(*X_cl_loin[~_moins].T, s=16, marker="+", color="0.15", label="classe +1")
    _x1 = np.linspace(-6, 14, 2)
    for _i, _nom in enumerate(poids_classif):
        _w = modeles_cl[("avec lointains", _nom)]
        # frontière w0·x1 + w1·x2 + b = 0  →  x2 = -(w0·x1 + b) / w1
        ax_cl.plot(_x1, -(_w[0] * _x1 + _w[2]) / _w[1], color=PALETTE[_i], lw=2,
                   label=f"{_nom} ({exactitudes_cl[('avec lointains', _nom)] * 100:.0f} % en test)")
    ax_cl.set(xlim=(-6, 14), ylim=(-5, 7), xlabel="x₁", ylabel="x₂",
              title="Frontières apprises avec 20 points « trop bien classés »")
    ax_cl.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9)
    _chemin = sauvegarder_figure(fig_cl, "classification-pertes" + ("-smoke" if SMOKE else ""),
                                 mo.notebook_dir() / "figures")
    print(f"Figure : {_chemin[0]}")
    fig_cl


@app.cell
def _(
    SMOKE,
    T0,
    droites,
    ecart_gradient,
    ecart_torch,
    ecarts_sklearn,
    meilleures_constantes,
    mo,
    np,
    exactitudes_cl,
    salaires,
    time,
):
    # --- Bilan : seuils du README. En smoke, on vérifie seulement que tout tourne. ---
    verifs = {
        "constante MSE = moyenne (à 1 € près)":
            abs(meilleures_constantes["MSE"] - salaires.mean()) <= 1,
        "constante MAE = médiane":
            meilleures_constantes["MAE"] == np.median(salaires),
        "pente MSE faussée (écart à 2 ≥ 0,5)": abs(droites["MSE"][0] - 2) >= 0.5,
        "pente MAE juste (écart à 2 ≤ 0,15)": abs(droites["MAE"][0] - 2) <= 0.15,
        "pente Huber juste (écart à 2 ≤ 0,15)": abs(droites["Huber"][0] - 2) <= 0.15,
        "numpy ≈ scikit-learn, MSE et MAE (pente à 0,02 près)":
            max(ecarts_sklearn["MSE"], ecarts_sklearn["MAE"]) <= 0.02,
        "moindres carrés : perte ≥ 5 points d'exactitude avec les lointains":
            exactitudes_cl[("propre", "moindres carrés")]
            - exactitudes_cl[("avec lointains", "moindres carrés")] >= 0.05,
        "log-loss et hinge : perte ≤ 1 point":
            all(exactitudes_cl[("propre", n)] - exactitudes_cl[("avec lointains", n)] <= 0.01
                for n in ("log-loss", "hinge")),
        "gradient de la CE = p − y (écart ≤ 1e-6)": ecart_gradient <= 1e-6,
        "numpy = PyTorch (écart ≤ 1e-6)": ecart_torch <= 1e-6,
    }
    for _k, _ok in verifs.items():
        _etat = "ok" if _ok else ("non vérifié en smoke" if SMOKE else "ÉCHEC")
        print(f"[{_etat}] {_k}")
    if not SMOKE and not all(verifs.values()):
        raise SystemExit("Seuils non atteints : " + ", ".join(k for k, ok in verifs.items() if not ok))
    duree_totale = time.perf_counter() - T0
    print(f"Durée des calculs : {duree_totale:.1f} s")
    mo.md(f"**Bilan** : {sum(verifs.values())}/{len(verifs)} vérifications réussies "
          f"en {duree_totale:.1f} s"
          + (" (mode smoke : données réduites, seuils non vérifiés)." if SMOKE else "."))


if __name__ == "__main__":
    app.run()
