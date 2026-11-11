# Exercice 00 : régression linéaire, descente de gradient, Ridge, validation croisée.
#
#   make ex NUM=00              → éditeur marimo (bouton « Lancer les calculs »)
#   make ex NUM=00 MODE=run     → script complet, vérifie les seuils du README
#   make ex NUM=00 MODE=smoke   → test rapide (< 30 s)
#
# Tout est en NumPy « à la main », puis vérifié avec scikit-learn. Aucun GPU : le
# problème est minuscule (quelques dizaines de points), ce qui permet de tout voir.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from ia_commun import is_smoke, sauvegarder_figure, seed_everything, style_commun

    return is_smoke, mo, np, plt, sauvegarder_figure, seed_everything, style_commun, time


@app.cell
def _(mo):
    mo.md(r"""
    # Exercice 00 : apprendre, c'est minimiser une perte

    Fil conducteur : la **régression linéaire**. On cherche une droite $\hat y = b + w\,x$
    qui prédit le temps de chargement d'une page web ($y$, en secondes) à partir de son
    poids ($x$, en Mo). Les données sont synthétiques : on connaît la vraie droite
    ($b = 0{,}8$, $w = 1{,}5$) et le bruit (écart-type $0{,}3$ s).

    **Perte optimisée** : l'erreur quadratique moyenne (MSE)
    $$\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^{N} \big(\hat y_i - y_i\big)^2,
    \qquad \theta = (b, w).$$

    Plan :

    - **A.** Résoudre exactement (formule fermée), puis retrouver la même solution par
      **descente de gradient**, et voir ce qui se passe quand le pas $\eta$ est trop grand.
    - **B.** Passer à une courbe (polynôme de degré 12 sur 20 points) : **surapprentissage**,
      puis **Ridge** (pénalité L2, ou estimateur MAP avec un a priori gaussien),
      **Lasso** (L1) et **arrêt précoce**.
    - **C.** Choisir la force de la pénalité par **validation croisée à 5 plis**, d'abord
      à la main, puis avec scikit-learn, et mesurer le résultat sur un jeu de test jamais vu.
    """)


@app.cell
def _(is_smoke, np, seed_everything, style_commun):
    # --- Configuration : le SEUL endroit où le mode smoke réduit le travail ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        # Partie A : descente de gradient sur la droite
        n_points_a=40,
        etas=[0.05, 0.3, 0.42, 0.45],      # pas testés (le pas critique vaut ≈ 0,43 ici)
        theta0=(2.0, 0.0),                  # départ : droite plate à 2 s (b = 2, w = 0)
        n_iter_a=300 if SMOKE else 3000,
        # Partie B : polynôme
        n_train=20,
        n_val=20,                           # jeu de validation de l'arrêt précoce
        n_test=1000,
        degre=12,
        degres=list(range(0, 16)),
        lambdas=list(10.0 ** np.arange(-8, 2.01, 1.0 if SMOKE else 0.25)),   # grille de λ
        n_iter_arret=20_000 if SMOKE else 500_000,
        # Partie C
        k_plis=5,
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}")
    print({k: v for k, v in config.items() if k != "lambdas"}, f"{len(config['lambdas'])} valeurs de λ")
    return SEED, SMOKE, config


@app.cell
def _(mo):
    mo.md(r"""
    ## A. La droite : formule fermée puis descente de gradient

    Avec la matrice de conception $X$ (une colonne de 1 pour $b$, une colonne $x$ pour $w$),
    la MSE s'écrit $\mathcal{L}(\theta) = \frac{1}{N}\lVert X\theta - y\rVert^2$. Son
    gradient est $\nabla\mathcal{L} = \frac{2}{N}X^\top(X\theta - y)$ ; il s'annule en
    $\theta^\star = (X^\top X)^{-1}X^\top y$ : ce sont les **équations normales**
    (Legendre 1805, Gauss).
    """)


@app.cell
def _(SEED, config, np):
    # --- Données A : 40 pages web, poids x ∈ [0, 2] Mo, temps y = 0,8 + 1,5 x + bruit ---
    # (mêmes données que la démo GradientDescent et la figure ch00-regression-residus)
    _rng = np.random.default_rng(SEED)
    x_a = _rng.uniform(0.0, 2.0, config["n_points_a"])
    y_a = 0.8 + 1.5 * x_a + _rng.normal(0.0, 0.3, config["n_points_a"])
    X_a = np.column_stack([np.ones_like(x_a), x_a])   # matrice de conception N × 2
    print(f"X_a : {X_a.shape}, y_a : {y_a.shape}")
    return X_a, x_a, y_a


@app.cell
def _(X_a, np, y_a):
    def mse(theta, X, y):
        """Erreur quadratique moyenne de la prédiction X @ theta."""
        r = X @ theta - y
        return float(r @ r / len(y))

    def gradient_mse(theta, X, y):
        """Gradient exact de la MSE : (2/N) Xᵀ (Xθ − y)."""
        return 2.0 / len(y) * X.T @ (X @ theta - y)

    # Formule fermée : on résout (XᵀX) θ = Xᵀy (plus stable que d'inverser XᵀX)
    theta_ferme = np.linalg.solve(X_a.T @ X_a, X_a.T @ y_a)
    mse_ferme = mse(theta_ferme, X_a, y_a)

    # Vérifier son gradient par différences finies : réflexe à prendre avant toute descente
    _theta_test = np.array([0.3, -0.7])
    _h = 1e-6
    _g_num = np.array([(mse(_theta_test + _h * e, X_a, y_a) - mse(_theta_test - _h * e, X_a, y_a)) / (2 * _h)
                       for e in np.eye(2)])
    ecart_gradient = float(np.abs(_g_num - gradient_mse(_theta_test, X_a, y_a)).max())

    # La MSE est une cuvette quadratique : sa courbure est la hessienne H = (2/N) XᵀX.
    # La descente de gradient converge si et seulement si η < 2 / λ_max(H).
    hessienne = 2.0 / len(y_a) * X_a.T @ X_a
    valeurs_propres = np.linalg.eigvalsh(hessienne)
    eta_critique = 2.0 / valeurs_propres.max()

    print(f"Formule fermée : b = {theta_ferme[0]:.4f}, w = {theta_ferme[1]:.4f}, MSE = {mse_ferme:.4f}")
    print(f"Gradient analytique contre différences finies : écart max {ecart_gradient:.1e}")
    print(f"Valeurs propres de la hessienne : {valeurs_propres.round(3)} "
          f"(conditionnement {valeurs_propres.max() / valeurs_propres.min():.1f})")
    print(f"Pas critique η_c = 2 / λ_max = {eta_critique:.3f}")
    return ecart_gradient, eta_critique, gradient_mse, mse, mse_ferme, theta_ferme, valeurs_propres


@app.cell
def _(mo):
    # Le bouton évite de tout relancer à chaque modification d'une cellule en amont.
    bouton = mo.ui.run_button(label="Lancer les calculs")
    bouton
    return (bouton,)


@app.cell
def _(X_a, bouton, config, gradient_mse, mo, mse, np, theta_ferme, time, y_a):
    # En script (make ex MODE=run|smoke), personne ne clique : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer les calculs ».*"))

    t0_total = time.perf_counter()

    def descente(eta, n_iter, theta0):
        """Descente de gradient « batch » : θ ← θ − η ∇L(θ). Renvoie trajectoire et pertes."""
        theta = np.array(theta0, dtype=float)
        trajectoire, pertes = [theta.copy()], [mse(theta, X_a, y_a)]
        for _ in range(n_iter):
            theta = theta - eta * gradient_mse(theta, X_a, y_a)
            trajectoire.append(theta.copy())
            pertes.append(mse(theta, X_a, y_a))
            if not np.isfinite(pertes[-1]) or pertes[-1] > 1e8:   # divergence : on arrête
                break
        return np.array(trajectoire), np.array(pertes)

    resultats_gd = {}
    for _eta in config["etas"]:
        _traj, _pertes = descente(_eta, config["n_iter_a"], config["theta0"])
        _ecarts = np.abs(_traj - theta_ferme).max(axis=1)
        _atteint = np.flatnonzero(_ecarts < 1e-6)
        resultats_gd[_eta] = dict(
            trajectoire=_traj, pertes=_pertes,
            ecart_final=float(_ecarts[-1]),
            iterations_1e6=int(_atteint[0]) if _atteint.size else None,
            diverge=bool(_pertes[-1] > 1e6 or not np.isfinite(_pertes[-1])),
        )
        _r = resultats_gd[_eta]
        _etat = ("DIVERGE" if _r["diverge"] else
                 f"|θ − θ*| < 1e-6 en {_r['iterations_1e6']} itérations" if _r["iterations_1e6"] is not None
                 else f"écart encore {_r['ecart_final']:.1e}")
        print(f"η = {_eta:<5} : {len(_pertes) - 1:>5} itérations, perte finale {_pertes[-1]:.4g}, {_etat}")
    return descente, resultats_gd, t0_total


@app.cell
def _(X_a, theta_ferme, x_a, y_a):
    # --- Vérification avec scikit-learn : même modèle, même solution ---
    from sklearn.linear_model import LinearRegression

    _reg = LinearRegression().fit(x_a.reshape(-1, 1), y_a)
    theta_sklearn = (float(_reg.intercept_), float(_reg.coef_[0]))
    ecart_sklearn = max(abs(theta_sklearn[0] - theta_ferme[0]), abs(theta_sklearn[1] - theta_ferme[1]))
    r2_a = float(_reg.score(x_a.reshape(-1, 1), y_a))
    print(f"scikit-learn : b = {theta_sklearn[0]:.4f}, w = {theta_sklearn[1]:.4f} "
          f"(écart à la formule fermée {ecart_sklearn:.1e}), R² = {r2_a:.3f}")
    _ = X_a
    return ecart_sklearn, r2_a


@app.cell
def _(
    SMOKE,
    X_a,
    config,
    eta_critique,
    mo,
    mse,
    np,
    plt,
    resultats_gd,
    sauvegarder_figure,
    theta_ferme,
    y_a,
):
    # --- Figure A : lignes de niveau de la MSE et trajectoires, puis courbes de perte ---
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    _B, _W = np.meshgrid(np.linspace(-0.6, 2.4, 200), np.linspace(-0.4, 3.2, 200))
    _Z = np.array([[mse(np.array([b, w]), X_a, y_a) for b, w in zip(rb, rw)] for rb, rw in zip(_B, _W)])
    _ax1.contour(_B, _W, _Z, levels=np.geomspace(0.08, 20, 14), colors="0.6", linewidths=0.7)
    for _eta, _r in resultats_gd.items():
        # On ne trace que le début : 40 pas, ou 12 pour la trajectoire qui diverge
        _t = _r["trajectoire"][: 12 if _r["diverge"] else 40]
        _ax1.plot(_t[:, 0], _t[:, 1], ".-", ms=3, lw=1, label=f"η = {_eta}")
        # abscisse k + 1 pour pouvoir passer l'axe des itérations en échelle log
        _ax2.loglog(np.arange(1, len(_r["pertes"]) + 1), _r["pertes"], label=f"η = {_eta}" + (" (diverge)" if _r["diverge"] else ""))
    _ax1.plot(*theta_ferme, "k*", ms=12, label="θ* (formule fermée)")
    _ax1.set(xlim=(-0.6, 2.4), ylim=(-0.4, 3.2), xlabel="b (ordonnée à l'origine)", ylabel="w (pente)",
             title=f"Trajectoires depuis (b, w) = {tuple(config['theta0'])}, η_c = {eta_critique:.3f}")
    _ax1.legend(fontsize=8, loc="upper right")
    _ax2.axhline(mse(theta_ferme, X_a, y_a), color="k", ls="--", lw=0.8, label="MSE minimale")
    _ax2.set(xlabel="itération + 1 (échelle log)", ylabel="MSE (échelle log)", ylim=(0.04, 1e3), xlim=(1, None),
             title="Courbes de perte")
    _ax2.legend(fontsize=8)
    _chemin = sauvegarder_figure(_fig, "descente-smoke" if SMOKE else "descente", mo.notebook_dir() / "figures")
    print(f"Figure : {_chemin[0]}")
    _fig


@app.cell
def _(mo):
    mo.md(r"""
    ## B. La courbe : surapprentissage, Ridge, Lasso, arrêt précoce

    Vingt points tirés de $y = \sin(2\pi x) + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, 0{,}3^2)$.
    Le modèle reste **linéaire en ses paramètres** : on ajoute seulement des colonnes
    $u, u^2, \dots, u^d$ (avec $u = 2x - 1 \in [-1, 1]$ pour garder des nombres raisonnables).

    **Ridge** minimise $\sum_i (\hat y_i - y_i)^2 + \lambda \lVert w \rVert^2$ (l'ordonnée à
    l'origine n'est pas pénalisée). C'est aussi l'estimateur **MAP** quand on suppose un bruit
    gaussien d'écart-type $\sigma$ et un a priori $w_j \sim \mathcal{N}(0, \tau^2)$ :
    $\lambda = \sigma^2 / \tau^2$. Sans pénalité ($\lambda = 0$), on retrouve le **maximum de
    vraisemblance** (MLE), c'est-à-dire les moindres carrés.
    """)


@app.cell
def _(SEED, config, np):
    # --- Données B : trois tirages indépendants (entraînement, validation, test) ---
    def tirer_sinus(rng, n):
        x = rng.uniform(0.0, 1.0, n)
        return x, np.sin(2 * np.pi * x) + rng.normal(0.0, 0.3, n)

    _rng = np.random.default_rng(SEED)
    x_tr, y_tr = tirer_sinus(_rng, config["n_train"])
    x_va, y_va = tirer_sinus(_rng, config["n_val"])
    x_te, y_te = tirer_sinus(_rng, config["n_test"])

    def polynome(x, degre):
        """Colonnes 1, u, u², …, u^degre avec u = 2x − 1 ∈ [−1, 1]."""
        return np.vander(2 * x - 1, degre + 1, increasing=True)

    def ridge(A, y, lam):
        """Ridge en formule fermée : (AᵀA + λP) θ = Aᵀy, P = identité sauf pour la constante."""
        P = np.eye(A.shape[1])
        P[0, 0] = 0.0
        return np.linalg.solve(A.T @ A + lam * P, A.T @ y)

    def mse_np(A, theta, y):
        return float(np.mean((A @ theta - y) ** 2))

    print(f"entraînement {len(x_tr)}, validation {len(x_va)}, test {len(x_te)} points ; "
          f"variance du bruit 0,09 (MSE minimale atteignable)")
    return mse_np, polynome, ridge, x_te, x_tr, x_va, y_te, y_tr, y_va


@app.cell
def _(bouton, config, mo, mse_np, np, polynome, x_te, x_tr, y_te, y_tr):
    mo.stop(mo.running_in_notebook() and not bouton.value)
    # --- B1. Maximum de vraisemblance (moindres carrés) pour chaque degré ---
    courbe_degres = []
    for _d in config["degres"]:
        _A = polynome(x_tr, _d)
        _theta = np.linalg.lstsq(_A, y_tr, rcond=None)[0]
        courbe_degres.append((_d, mse_np(_A, _theta, y_tr), mse_np(polynome(x_te, _d), _theta, y_te),
                              float(np.abs(_theta).max())))
    print("degré | MSE entraînement | MSE test | plus grand |θ_j|")
    for _d, _tr, _te, _m in courbe_degres:
        print(f"{_d:>5} | {_tr:>16.4f} | {_te:>8.4g} | {_m:.4g}")
    return (courbe_degres,)


@app.cell
def _(config, mse_np, np, polynome, ridge, x_te, x_tr, y_te, y_tr):
    # --- B2. Degré 12 : MLE (λ = 0) contre Ridge / MAP sur une grille de λ ---
    _d = config["degre"]
    A_tr, A_te = polynome(x_tr, _d), polynome(x_te, _d)
    theta_mle = np.linalg.lstsq(A_tr, y_tr, rcond=None)[0]
    mse_test_mle = mse_np(A_te, theta_mle, y_te)
    courbe_lambdas = []
    for _lam in config["lambdas"]:
        _th = ridge(A_tr, y_tr, _lam)
        courbe_lambdas.append((_lam, mse_np(A_tr, _th, y_tr), mse_np(A_te, _th, y_te), float(np.linalg.norm(_th[1:]))))
    print(f"MLE degré {_d} : MSE entraînement {mse_np(A_tr, theta_mle, y_tr):.4f}, "
          f"MSE test {mse_test_mle:.4g}, ‖w‖ = {np.linalg.norm(theta_mle[1:]):.4g}")
    for _lam, _tr, _te, _n in courbe_lambdas[:: max(1, len(courbe_lambdas) // 10)]:
        print(f"  λ = {_lam:8.0e} : MSE entraînement {_tr:.4f}, test {_te:.4f}, ‖w‖ = {_n:.3g}")
    # Lecture MAP : avec σ = 0,3 et un a priori w_j ~ N(0, τ²), λ = σ²/τ²
    for _tau in (1.0, 10.0):
        print(f"  a priori τ = {_tau:>4} → λ = σ²/τ² = {0.3**2 / _tau**2:.0e}")
    return A_te, A_tr, courbe_lambdas, mse_test_mle, theta_mle


@app.cell
def _(A_te, A_tr, np, y_te, y_tr):
    # --- B3. Lasso (pénalité L1) : des coefficients exactement nuls ---
    import warnings

    from sklearn.linear_model import Lasso

    resultats_lasso = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # avertissements de convergence pour les α minuscules
        for _alpha in (1e-4, 1e-3, 1e-2):
            # scikit-learn minimise (1 / 2N) Σ (ŷ − y)² + α Σ |w_j| ; la colonne de 1 est retirée
            _las = Lasso(alpha=_alpha, max_iter=200_000, tol=1e-6).fit(A_tr[:, 1:], y_tr)
            _nuls = int(np.sum(np.abs(_las.coef_) < 1e-10))
            _te = float(np.mean((_las.predict(A_te[:, 1:]) - y_te) ** 2))
            resultats_lasso.append((_alpha, _nuls, _te))
            print(f"Lasso α = {_alpha:.0e} : {_nuls} coefficients nuls sur {A_tr.shape[1] - 1}, MSE test {_te:.4f}")
    return (resultats_lasso,)


@app.cell
def _(A_tr, config, mse_np, np, polynome, x_te, x_va, y_te, y_tr, y_va):
    # --- B4. Arrêt précoce : descente de gradient sur la MSE SANS pénalité, degré 12 ---
    # On surveille l'erreur de validation et on garde les paramètres du meilleur moment.
    _A_va, _A_te = polynome(x_va, config["degre"]), polynome(x_te, config["degre"])
    _H = 2.0 / len(y_tr) * A_tr.T @ A_tr
    _eta = 1.0 / np.linalg.eigvalsh(_H).max()          # pas sûr : moitié du pas critique
    _theta = np.zeros(A_tr.shape[1])
    _points = set(np.unique(np.geomspace(1, config["n_iter_arret"], 300).astype(int)))
    historique_arret = []                                  # (itération, MSE entr., val, test)
    meilleur = (np.inf, 0, None)
    for _k in range(1, config["n_iter_arret"] + 1):
        _theta -= _eta * (2.0 / len(y_tr)) * A_tr.T @ (A_tr @ _theta - y_tr)
        if _k in _points:
            _v = mse_np(_A_va, _theta, y_va)
            historique_arret.append((_k, mse_np(A_tr, _theta, y_tr), _v, mse_np(_A_te, _theta, y_te)))
            if _v < meilleur[0]:
                meilleur = (_v, _k, mse_np(_A_te, _theta, y_te))
    arret = dict(iteration=meilleur[1], mse_val=meilleur[0], mse_test=meilleur[2],
                 mse_test_finale=historique_arret[-1][3], eta=_eta)
    print(f"η = {_eta:.3f} ; meilleure validation à l'itération {arret['iteration']} "
          f"(MSE val {arret['mse_val']:.4f}, test {arret['mse_test']:.4f}) ; "
          f"après {config['n_iter_arret']} itérations : MSE test {arret['mse_test_finale']:.4f}")
    return arret, historique_arret


@app.cell
def _(mo):
    mo.md(r"""
    ## C. Choisir λ par validation croisée à 5 plis

    On découpe les 20 points d'entraînement en 5 plis de 4. Pour chaque $\lambda$, on entraîne
    5 fois sur 4 plis et on mesure la MSE sur le pli laissé de côté ; la moyenne des 5 mesures
    est le **score de validation croisée**. Le jeu de test (1 000 points) ne sert qu'à la fin.
    """)


@app.cell
def _(A_tr, SEED, config, mse_np, np, ridge, y_tr):
    # --- C1. Validation croisée « à la main » ---
    _n = len(y_tr)
    _perm = np.random.default_rng(SEED).permutation(_n)
    plis = [(np.setdiff1d(_perm, p), p) for p in np.array_split(_perm, config["k_plis"])]

    cv_main = []
    for _lam in config["lambdas"]:
        _scores = [mse_np(A_tr[_va], ridge(A_tr[_ap], y_tr[_ap], _lam), y_tr[_va]) for _ap, _va in plis]
        cv_main.append(float(np.mean(_scores)))
    cv_main = np.array(cv_main)
    lambda_cv = float(config["lambdas"][int(np.argmin(cv_main))])
    print(f"λ choisi par validation croisée (à la main) : {lambda_cv:.1e} (MSE CV {cv_main.min():.4f})")
    return cv_main, lambda_cv, plis


@app.cell
def _(A_te, A_tr, config, cv_main, lambda_cv, np, plis, ridge, y_te, y_tr):
    # --- C2. La même chose avec scikit-learn, sur les MÊMES plis ---
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import cross_val_score

    # Ridge de scikit-learn minimise Σ (ŷ − y)² + α ‖w‖² et ne pénalise pas la constante :
    # c'est exactement notre formule, avec α = λ. On retire la colonne de 1 (fit_intercept).
    cv_sklearn = np.array([
        -cross_val_score(Ridge(alpha=_lam), A_tr[:, 1:], y_tr, cv=plis, scoring="neg_mean_squared_error").mean()
        for _lam in config["lambdas"]
    ])
    ecart_cv = float(np.abs(cv_sklearn - cv_main).max() / cv_main.max())

    # Modèle final : Ridge avec λ choisi, réentraîné sur les 20 points, évalué UNE fois sur le test
    theta_final = ridge(A_tr, y_tr, lambda_cv)
    _pred = A_te @ theta_final
    mse_test_final = float(np.mean((_pred - y_te) ** 2))
    r2_main = 1.0 - np.sum((_pred - y_te) ** 2) / np.sum((y_te - y_te.mean()) ** 2)
    _sk = Ridge(alpha=lambda_cv).fit(A_tr[:, 1:], y_tr)
    r2_sklearn = float(r2_score(y_te, _sk.predict(A_te[:, 1:])))
    ecart_r2 = abs(r2_sklearn - float(r2_main))
    print(f"Écart relatif max entre CV à la main et scikit-learn : {ecart_cv:.1e}")
    print(f"Test : MSE {mse_test_final:.4f} (scikit-learn {mean_squared_error(y_te, _sk.predict(A_te[:, 1:])):.4f}), "
          f"R² {float(r2_main):.3f} (scikit-learn {r2_sklearn:.3f})")
    return cv_sklearn, ecart_cv, ecart_r2, mse_test_final, r2_main, theta_final


@app.cell
def _(
    SMOKE,
    arret,
    config,
    courbe_degres,
    courbe_lambdas,
    cv_main,
    historique_arret,
    lambda_cv,
    mo,
    np,
    plt,
    polynome,
    sauvegarder_figure,
    theta_final,
    theta_mle,
    x_tr,
    y_tr,
):
    # --- Figure B/C : quatre panneaux ---
    _fig, _axes = plt.subplots(2, 2, figsize=(11, 8))
    (_a1, _a2), (_a3, _a4) = _axes
    _deg = np.array(courbe_degres)
    _a1.semilogy(_deg[:, 0], _deg[:, 1], "o-", label="entraînement")
    _a1.semilogy(_deg[:, 0], _deg[:, 2], "s-", label="test")
    _a1.axhline(0.09, color="k", ls=":", lw=0.8, label="bruit (σ² = 0,09)")
    _a1.set(xlabel="degré du polynôme", ylabel="MSE (log)", title="MLE : l'erreur de test explose")
    _a1.legend(fontsize=8)

    _lam = np.array(courbe_lambdas)
    _a2.loglog(_lam[:, 0], _lam[:, 1], label="entraînement")
    _a2.loglog(_lam[:, 0], _lam[:, 2], label="test")
    _a2.loglog(_lam[:, 0], cv_main, "--", label="validation croisée (5 plis)")
    _a2.axvline(lambda_cv, color="k", lw=0.8, ls=":", label=f"λ choisi = {lambda_cv:.0e}")
    _a2.set(xlabel="λ (force de la pénalité L2)", ylabel="MSE (log)", title=f"Ridge, degré {config['degre']}")
    _a2.legend(fontsize=8)

    _xs = np.linspace(0, 1, 400)
    _A = polynome(_xs, config["degre"])
    _a3.plot(_xs, np.sin(2 * np.pi * _xs), color="0.5", lw=1, label="vraie fonction")
    _a3.plot(_xs, _A @ theta_mle, label="MLE (λ = 0)")
    _a3.plot(_xs, _A @ theta_final, label=f"Ridge (λ = {lambda_cv:.0e})")
    _a3.scatter(x_tr, y_tr, color="k", s=15, zorder=3, label="20 points d'entraînement")
    _a3.set(ylim=(-2, 2), xlabel="x", ylabel="y", title="Mêmes 20 points, deux modèles")
    _a3.legend(fontsize=8)

    _h = np.array(historique_arret)
    _a4.loglog(_h[:, 0], _h[:, 1], label="entraînement")
    _a4.loglog(_h[:, 0], _h[:, 2], label="validation")
    _a4.loglog(_h[:, 0], _h[:, 3], label="test")
    _a4.axvline(arret["iteration"], color="k", lw=0.8, ls=":", label=f"arrêt précoce ({arret['iteration']})")
    _a4.set(xlabel="itération de descente de gradient", ylabel="MSE (log)", title="Arrêt précoce (sans pénalité)")
    _a4.legend(fontsize=8)
    _fig.tight_layout()
    _chemin = sauvegarder_figure(_fig, "regularisation-smoke" if SMOKE else "regularisation",
                                 mo.notebook_dir() / "figures")
    print(f"Figure : {_chemin[0]}")
    _fig


@app.cell
def _(
    SMOKE,
    arret,
    ecart_cv,
    ecart_gradient,
    ecart_r2,
    ecart_sklearn,
    mo,
    mse_test_final,
    mse_test_mle,
    r2_main,
    resultats_gd,
    t0_total,
    time,
):
    # --- Bilan : seuils du README. En smoke, on vérifie seulement que tout tourne. ---
    duree = time.perf_counter() - t0_total
    controles = {
        "gradient analytique = différences finies (écart < 1e-6)": ecart_gradient < 1e-6,
        "η = 0,05 et 0,3 : |θ − θ*| < 1e-6": all(resultats_gd[e]["iterations_1e6"] is not None for e in (0.05, 0.3)),
        "η = 0,45 (> η_c) : divergence": resultats_gd[0.45]["diverge"],
        "scikit-learn = formule fermée (écart < 1e-8)": ecart_sklearn < 1e-8,
        "Ridge choisi par CV : MSE test ≤ 0,16": mse_test_final <= 0.16,
        "MLE degré 12 : MSE test ≥ 10 × celle de Ridge": mse_test_mle >= 10 * mse_test_final,
        "R² test ≥ 0,70": r2_main >= 0.70,
        "CV à la main = scikit-learn (écart relatif < 1e-6)": ecart_cv < 1e-6 and ecart_r2 < 1e-6,
        "arrêt précoce : test au meilleur moment < test final": arret["mse_test"] < arret["mse_test_finale"],
    }
    for _nom, _ok in controles.items():
        # En smoke (moins d'itérations, grille de λ grossière), un seuil manqué n'est pas une erreur
        _etat = "ok" if _ok else ("smoke : non vérifié" if SMOKE else "ÉCHEC")
        print(f"[{_etat}] {_nom}")
    print(f"Durée des calculs : {duree:.1f} s")
    if not SMOKE and not all(controles.values()):
        raise SystemExit("Au moins un seuil du README n'est pas atteint (voir ci-dessus).")
    mo.md(f"**Bilan{' (smoke)' if SMOKE else ''}** : {sum(controles.values())}/{len(controles)} contrôles réussis, "
          f"MSE test Ridge {mse_test_final:.3f} contre {mse_test_mle:.3g} sans pénalité, {duree:.1f} s.")


if __name__ == "__main__":
    app.run()
