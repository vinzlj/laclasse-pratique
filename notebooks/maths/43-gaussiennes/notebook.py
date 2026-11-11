# Notebook 43-gaussiennes (chapitre 43, Plusieurs variables aléatoires : loi jointe,
# covariance, espérance conditionnelle, normale multivariée).
#
# Tirer une normale à deux dimensions par x = μ + Lz (L de Cholesky), comparer la covariance
# empirique à Σ, tracer les ellipses et les axes propres (np.linalg.eigh) et compter les
# points dans chaque ellipse ; vérifier par simulation la somme de gaussiennes indépendantes,
# une covariance nulle sans indépendance (Y = X²), la règle de la chaîne sur un petit modèle de
# bigrammes, et la densité d’un flot affine (histogramme de x = 2z + 1 contre f_X).
#
#   make nb NUM=43              → éditeur marimo (bouton « Lancer le calcul »)
#   make nb NUM=43 MODE=run     → script complet
#   make nb NUM=43 MODE=smoke   → test rapide (moins de 30 s)

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import sympy as sp
    from math_commun import is_smoke, sauvegarder_figure, seed_everything, style_commun

    return is_smoke, math, mo, np, plt, sauvegarder_figure, seed_everything, sp, style_commun, time


@app.cell
def _(mo):
    mo.md(r"""
    # Gaussiennes à plusieurs dimensions, covariance et règle de la chaîne

    **Chapitre 43, sections 3, 6, 7, 8 et 9.** Cinq vérifications par simulation :

    1. **Tirer** $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu} \,;\, \Sigma)$ par
       $\mathbf{x} = \boldsymbol{\mu} + L\mathbf{z}$, où $\mathbf{z} \sim \mathcal{N}(\mathbf{0} \,;\, I)$
       et $LL^\top = \Sigma$ (factorisation de Cholesky) : d’après la section 6,
       $\mathrm{Cov}(L\mathbf{z}) = LIL^\top = \Sigma$. On compare la covariance du nuage à
       $\Sigma$, on trace les ellipses et les axes propres, et on compte les points dans les
       ellipses $c = 1$ et $c = 4$ (proportions attendues $1 - e^{-c/2}$ en dimension 2).
    2. **Sommer** deux normales indépendantes : moyenne $\mu_1 + \mu_2$, variance
       $\sigma_1^2 + \sigma_2^2$ (section 8).
    3. **Covariance nulle sans indépendance** : $X$ uniforme sur $\{-1 \,;\, 0 \,;\, 1\}$ et
       $Y = X^2$ (section 6).
    4. **Règle de la chaîne** sur un modèle de bigrammes : la fréquence d’une phrase tirée
       approche le produit des probabilités du mot suivant (section 3).
    5. **Flot affine** : l’histogramme de $x = 2z + 1$ suit $f_X(x) = \frac{1}{2}\varphi\big(\frac{x - 1}{2}\big)$,
       et $f_X(3) \approx 0{,}121$ (section 9).
    """)


@app.cell
def _(is_smoke, np, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        n=40_000 if SMOKE else 400_000,          # tirages par expérience
        n_phrases=40_000 if SMOKE else 400_000,  # phrases tirées du modèle de bigrammes
        tol_cov=0.03 if SMOKE else 0.01,         # écart maximal toléré entre covariances
        tol_prop=0.015 if SMOKE else 0.005,      # écart toléré sur les proportions
    )
    rng = np.random.default_rng(SEED)
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SMOKE, config, rng


@app.cell
def _(mo):
    bouton = mo.ui.run_button(label="Lancer le calcul")
    bouton  # noqa: B018 (marimo affiche la dernière expression)
    return (bouton,)


@app.cell
def _(bouton, config, math, mo, np, rng, time):
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer le calcul ».*"))
    t0 = time.perf_counter()
    n = config["n"]

    # 1. Normale à deux dimensions par Cholesky
    mu = np.array([1.0, 2.0])
    sigma = np.array([[1.0, 0.6], [0.6, 1.0]])
    chol = np.linalg.cholesky(sigma)
    z = rng.standard_normal((n, 2))
    pts = mu + z @ chol.T
    cov_emp = np.cov(pts.T, bias=True)
    ecart_cov = float(np.abs(cov_emp - sigma).max())
    valeurs, vecteurs = np.linalg.eigh(sigma)
    centres = pts - mu
    q = np.einsum("ij,jk,ik->i", centres, np.linalg.inv(sigma), centres)
    prop = {c: float(np.mean(q <= c)) for c in (1.0, 4.0)}
    attendu = {c: 1 - math.exp(-c / 2) for c in (1.0, 4.0)}

    # 2. Somme de deux normales indépendantes : N(1 ; 1) + N(2 ; 1,5²)
    s = rng.normal(1.0, 1.0, n) + rng.normal(2.0, 1.5, n)
    moy_s, var_s = float(s.mean()), float(s.var())
    dans_un_sigma = float(np.mean(np.abs(s - 3.0) <= math.sqrt(3.25)))

    # 3. Covariance nulle sans indépendance
    xu = rng.choice([-1, 0, 1], n)
    yu = xu**2
    cov_u = float(np.mean(xu * yu) - xu.mean() * yu.mean())
    p_joint = float(np.mean((xu == 0) & (yu == 1)))
    p_produit = float(np.mean(xu == 0) * np.mean(yu == 1))

    # 4. Règle de la chaîne : modèle de bigrammes (début, mot 2 sachant mot 1, mot 3 sachant mot 2)
    mots = ["le", "chat", "chien", "dort", "mange"]
    debut = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    transition = np.array([
        [0.0, 0.6, 0.4, 0.0, 0.0],   # après « le »
        [0.0, 0.0, 0.0, 0.7, 0.3],   # après « chat »
        [0.0, 0.0, 0.0, 0.2, 0.8],   # après « chien »
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
    ])
    m = config["n_phrases"]
    w1 = rng.choice(5, m, p=debut)
    u2 = rng.random(m)
    w2 = (u2[:, None] > np.cumsum(transition[w1], axis=1)).sum(axis=1)
    u3 = rng.random(m)
    w3 = (u3[:, None] > np.cumsum(transition[w2], axis=1)).sum(axis=1)
    cible = (0, 1, 3)  # « le chat dort »
    p_chaine = debut[0] * transition[0, 1] * transition[1, 3]
    freq = float(np.mean((w1 == cible[0]) & (w2 == cible[1]) & (w3 == cible[2])))
    phrase = " ".join(mots[i] for i in cible)

    # 5. Flot affine x = 2z + 1 : densité empirique près de 3
    xs = 2 * rng.standard_normal(n) + 1
    largeur = 0.1
    dens_emp = float(np.mean(np.abs(xs - 3) <= largeur / 2) / largeur)
    f3 = 0.5 * math.exp(-0.5) / math.sqrt(2 * math.pi)
    log_f3 = math.log(f3)

    duree = time.perf_counter() - t0
    print(f"Calcul : {duree:.2f} s")
    print(f"1. covariance du nuage {np.round(cov_emp, 4).tolist()}, écart max {ecart_cov:.4f} ; "
          f"valeurs propres {valeurs.round(4).tolist()} ; dans c = 1 : {prop[1.0]:.4f} "
          f"(attendu {attendu[1.0]:.4f}), dans c = 4 : {prop[4.0]:.4f} (attendu {attendu[4.0]:.4f})")
    print(f"2. somme : moyenne {moy_s:.4f} (3), variance {var_s:.4f} (3,25), "
          f"dans moyenne ± écart-type : {dans_un_sigma:.4f} (0,6827)")
    print(f"3. Y = X² : covariance {cov_u:.4f}, P(X = 0, Y = 1) = {p_joint:.4f} contre "
          f"P(X = 0) P(Y = 1) = {p_produit:.4f} (2/9 ≈ 0,2222)")
    print(f"4. « {phrase} » : règle de la chaîne {p_chaine:.4f}, fréquence observée {freq:.4f}")
    print(f"5. flot : densité empirique en 3 {dens_emp:.4f}, f_X(3) = {f3:.4f}, ln f_X(3) = {log_f3:.4f}")
    return (attendu, cov_u, dans_un_sigma, dens_emp, duree, ecart_cov, f3, freq, mu, p_chaine,
            p_joint, p_produit, prop, pts, sigma, valeurs, var_s, moy_s, vecteurs, xs)


@app.cell
def _(sp):
    # --- Vérifications exactes (sympy) ---
    _r, _c, _y = sp.symbols("r c y", positive=True)
    _S = sp.Matrix([[1, sp.Rational(3, 5)], [sp.Rational(3, 5), 1]])
    assert _S.det() == sp.Rational(16, 25)
    assert set(_S.eigenvals()) == {sp.Rational(8, 5), sp.Rational(2, 5)}
    _x1, _x2 = sp.symbols("_x1 _x2")
    _v = sp.Matrix([_x1, _x2])
    _forme = sp.expand((_v.T * _S.inv() * _v)[0])
    assert sp.simplify(_forme - (_x1**2 - sp.Rational(6, 5) * _x1 * _x2 + _x2**2) / sp.Rational(16, 25)) == 0
    # Proportion dans l’ellipse c en dimension 2 : ∫₀^√c r e^(−r²/2) dr = 1 − e^(−c/2)
    _proportion = sp.integrate(_r * sp.exp(-_r**2 / 2), (_r, 0, sp.sqrt(_c)))
    assert sp.simplify(_proportion - (1 - sp.exp(-_c / 2))) == 0
    # Densité du flot affine en 3
    _f_y = sp.exp(-((_y - 1) / 2) ** 2 / 2) / (2 * sp.sqrt(2 * sp.pi))
    assert sp.simplify(_f_y.subs(_y, 3) - sp.exp(-sp.Rational(1, 2)) / (2 * sp.sqrt(2 * sp.pi))) == 0
    assert sp.simplify(sp.integrate(_f_y, (_y, -sp.oo, sp.oo)) - 1) == 0
    print("sympy : det Σ = 16/25, valeurs propres 8/5 et 2/5, forme quadratique, 1 − e^(−c/2), "
          f"f_Y(3) = {sp.N(_f_y.subs(_y, 3), 6)}")
    return


@app.cell
def _(SMOKE, f3, mo, mu, np, plt, pts, sauvegarder_figure, sigma, valeurs, vecteurs, xs):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4.4))
    _ax1.scatter(pts[:2000, 0], pts[:2000, 1], s=3, alpha=0.4, label="2 000 premiers tirages")
    _t = np.linspace(0, 2 * np.pi, 300)
    _cercle = np.vstack([np.cos(_t), np.sin(_t)])
    for _c, _style in ((1, "-"), (4, "--")):
        _ell = mu[:, None] + vecteurs @ np.diag(np.sqrt(_c * valeurs)) @ _cercle
        _ax1.plot(_ell[0], _ell[1], _style, color="C1", label=f"ellipse c = {_c}")
    for _k in range(2):
        _d = vecteurs[:, _k] * np.sqrt(4 * valeurs[_k])
        _ax1.plot([mu[0] - _d[0], mu[0] + _d[0]], [mu[1] - _d[1], mu[1] + _d[1]], ":", color="k")
    _ax1.set(aspect="equal", title=f"Loi normale N(μ ; Σ), Σ = {sigma.tolist()}", xlabel="x₁", ylabel="x₂")
    _ax1.legend(fontsize=8, loc="upper left")
    _grille = np.linspace(-7, 9, 400)
    _ax2.hist(xs, bins=120, range=(-7, 9), density=True, alpha=0.5, label="x = 2z + 1, histogramme")
    _ax2.plot(_grille, 0.5 * np.exp(-((_grille - 1) / 2) ** 2 / 2) / np.sqrt(2 * np.pi), color="C1",
             label="f_X(x) = φ((x − 1)/2) / 2")
    _ax2.plot([3], [f3], "ko")
    _ax2.set(title="Flot affine : densité de x = 2z + 1", xlabel="x")
    _ax2.legend(fontsize=8)
    _chemins = sauvegarder_figure(_fig, "gaussiennes-smoke" if SMOKE else "gaussiennes", mo.notebook_dir() / "figures")
    print(f"Figure : {_chemins[0].name}")
    _fig  # noqa: B018 (marimo affiche la dernière expression)
    return


@app.cell
def _(attendu, config, cov_u, dans_un_sigma, dens_emp, duree, ecart_cov, f3, freq, mo, moy_s,
      p_chaine, p_joint, p_produit, prop, var_s):
    # --- Bilan : seuils du README, vérifiés aussi en smoke (tolérances plus larges) ---
    _tc, _tp = config["tol_cov"], config["tol_prop"]
    _echecs = []
    if ecart_cov > _tc:
        _echecs.append(f"covariance du nuage à {ecart_cov:.4f} de Σ (> {_tc})")
    for _c in (1.0, 4.0):
        if abs(prop[_c] - attendu[_c]) > _tp:
            _echecs.append(f"proportion dans c = {_c:g} : {prop[_c]:.4f} contre {attendu[_c]:.4f}")
    if abs(moy_s - 3) > 0.03 or abs(var_s - 3.25) > 0.1:
        _echecs.append(f"somme : moyenne {moy_s:.3f}, variance {var_s:.3f}")
    if abs(dans_un_sigma - 0.6827) > _tp:
        _echecs.append(f"somme : {dans_un_sigma:.4f} dans moyenne ± écart-type")
    if abs(cov_u) > _tc or p_joint != 0.0 or abs(p_produit - 2 / 9) > _tp:
        _echecs.append(f"Y = X² : covariance {cov_u:.4f}, {p_joint} contre {p_produit:.4f}")
    if abs(freq - p_chaine) > _tp:
        _echecs.append(f"règle de la chaîne : {freq:.4f} contre {p_chaine:.4f}")
    if abs(dens_emp - f3) > 3 * _tp:
        _echecs.append(f"flot : densité {dens_emp:.4f} contre {f3:.4f}")
    if _echecs:
        raise SystemExit("Bilan : ÉCHEC : " + " ; ".join(_echecs))
    print(f"Bilan : OK (écart de covariance {ecart_cov:.4f}, ellipses {prop[1.0]:.3f} et {prop[4.0]:.3f}, "
          f"somme {moy_s:.3f} et {var_s:.3f}, Cov(X, X²) = {cov_u:.4f}, chaîne {freq:.4f} contre "
          f"{p_chaine:.2f}, flot {dens_emp:.3f} contre {f3:.3f}, {duree:.2f} s)")
    mo.md(f"**Bilan** : covariance à {ecart_cov:.4f} près ; {prop[1.0]:.3f} et {prop[4.0]:.3f} des points "
          f"dans les ellipses c = 1 et c = 4 ; somme de variance {var_s:.3f} ; « le chat dort » "
          f"observée {freq:.4f} fois pour {p_chaine:.2f} prévue.")
    return


if __name__ == "__main__":
    app.run()
