# Notebook 42-tcl (chapitre 42, Lois continues, loi normale, théorème central limite).
#
# Le théorème central limite, mesuré : la moyenne de n tirages d’une loi très dissymétrique
# (exponentielle, dé, Bernoulli rare), une fois centrée et réduite, a une fonction de
# répartition de plus en plus proche de Φ quand n grandit (n = 1, 5, 30, 100) ; couverture
# EXACTE de l’intervalle p̂ ± 1,96 SE selon n et p (sommes de probabilités binomiales, sans
# tirage) ; Φ et ses quantiles calculés avec statistics.NormalDist (bibliothèque standard).
#
#   make nb NUM=42              → éditeur marimo (bouton « Lancer le calcul »)
#   make nb NUM=42 MODE=run     → script complet
#   make nb NUM=42 MODE=smoke   → test rapide (moins de 30 s)

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import time
    from statistics import NormalDist

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import sympy as sp
    from math_commun import is_smoke, sauvegarder_figure, seed_everything, style_commun

    return NormalDist, is_smoke, math, mo, np, plt, sauvegarder_figure, seed_everything, sp, style_commun, time


@app.cell
def _(mo):
    mo.md(r"""
    # Le théorème central limite, mesuré

    **Chapitre 42, sections 5 et 6.** Pour $X_1, \dots, X_n$ i.i.d. d’espérance $\mu$ et
    d’écart-type $\sigma$, la moyenne centrée réduite
    $Z_n = \dfrac{\bar{X}_n - \mu}{\sigma/\sqrt{n}}$ suit approximativement la loi
    $\mathcal{N}(0 \,;\, 1)$ quand $n$ est grand. Trois expériences :

    1. **La cloche apparaît.** Pour trois lois très différentes d’une cloche (exponentielle de
       paramètre 1, dé équilibré, Bernoulli de paramètre 0,05), on tire $R$ moyennes de $n$
       valeurs, pour $n = 1, 5, 30, 100$. On mesure l’écart maximal $D_n$ entre la fonction de
       répartition observée de $Z_n$ et $\Phi$ : il doit diminuer quand $n$ grandit.
    2. **L’intervalle à 1,96.** Pour une proportion $p$ et $n$ tirages, on calcule
       **exactement** (par les probabilités de la loi binomiale, sans simulation) la
       probabilité que $\hat{p} \pm 1{,}96\sqrt{\hat{p}(1 - \hat{p})/n}$ contienne $p$. Elle
       est proche de 0,95 quand $n$ est grand et $p$ pas trop petit, et bien plus faible
       sinon : c’est « ce que le TCL ne dit pas ».
    3. **Φ et ses quantiles**, par `statistics.NormalDist` (bibliothèque standard de Python ;
       avec scipy, ce seraient `scipy.stats.norm.cdf` et `scipy.stats.norm.ppf`), comparés à
       la règle 68-95-99,7 et aux quantiles 1,645, 1,96 et 2,576 du cours.
    """)


@app.cell
def _(is_smoke, np, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        tailles=[1, 5, 30, 100],                      # n valeurs par moyenne
        repetitions=4_000 if SMOKE else 40_000,       # R moyennes par taille et par loi
        p_rare=0.05,                                  # Bernoulli rare
        tailles_ic=[20, 50, 100, 1_000],              # n de l’intervalle p̂ ± 1,96 SE
        p_ic=[0.5, 0.1, 0.02],
    )
    rng = np.random.default_rng(SEED)
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SMOKE, config, rng


@app.cell
def _(NormalDist, sp):
    # --- Vérification exacte (sympy) : φ a une _aire 1, une espérance 0 et une _variance 1 ---
    _z = sp.symbols("z", real=True)
    _phi = sp.exp(-_z**2 / 2) / sp.sqrt(2 * sp.pi)
    _aire = sp.integrate(_phi, (_z, -sp.oo, sp.oo))
    _moyenne = sp.integrate(_z * _phi, (_z, -sp.oo, sp.oo))
    _variance = sp.integrate(_z**2 * _phi, (_z, -sp.oo, sp.oo))
    assert (_aire, _moyenne, _variance) == (1, 0, 1), (_aire, _moyenne, _variance)
    # Exponentielle de paramètre λ : _aire 1, espérance 1/λ, _variance 1/λ²
    _x, _lam = sp.symbols("x lambda", positive=True)
    _f = _lam * sp.exp(-_lam * _x)
    _e1 = sp.integrate(_x * _f, (_x, 0, sp.oo))
    _e2 = sp.integrate(_x**2 * _f, (_x, 0, sp.oo))
    assert sp.integrate(_f, (_x, 0, sp.oo)) == 1 and sp.simplify(_e1 - 1 / _lam) == 0
    assert sp.simplify(_e2 - _e1**2 - 1 / _lam**2) == 0
    print(f"sympy : aire de φ = {_aire}, E(Z) = {_moyenne}, Var(Z) = {_variance} ; "
          f"exponentielle : E = {_e1}, Var = {sp.simplify(_e2 - _e1**2)}")
    Z = NormalDist()
    return (Z,)


@app.cell
def _(mo):
    # Le bouton empêche le calcul de se relancer à chaque modification d’une cellule amont.
    bouton = mo.ui.run_button(label="Lancer le calcul")
    bouton  # noqa: B018 (marimo affiche la dernière expression)
    return (bouton,)


@app.cell
def _(Z, bouton, config, math, mo, np, rng, time):
    # En script (make nb MODE=run|smoke), personne ne clique : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer le calcul ».*"))

    _t0 = time.perf_counter()
    _R = config["repetitions"]
    _p_rare = config["p_rare"]
    # Trois lois : (nom, tirage de R × n valeurs, μ, σ)
    _lois = {
        "exponentielle": (lambda taille: rng.exponential(1.0, size=taille), 1.0, 1.0),
        "dé": (lambda taille: rng.integers(1, 7, size=taille), 3.5, math.sqrt(35 / 12)),
        "Bernoulli rare": (lambda taille: rng.random(taille) < _p_rare, _p_rare, math.sqrt(_p_rare * (1 - _p_rare))),
    }
    _grille = np.linspace(-3, 3, 241)
    _Phi_grille = np.array([Z.cdf(g) for g in _grille])
    distances = {}
    echantillons = {}
    for _nom, (_tirer, _mu, _sigma) in _lois.items():
        distances[_nom] = []
        for _n in config["tailles"]:
            _moyennes = _tirer((_R, _n)).mean(axis=1)
            _zn = (_moyennes - _mu) / (_sigma / math.sqrt(_n))
            _zn.sort()
            # Fonction de répartition observée sur la _grille, des deux côtés des sauts
            _haut = np.searchsorted(_zn, _grille, side="right") / _R
            _bas = np.searchsorted(_zn, _grille, side="left") / _R
            _d = float(np.max(np.maximum(np.abs(_haut - _Phi_grille), np.abs(_bas - _Phi_grille))))
            distances[_nom].append(_d)
            if _nom == "exponentielle":
                echantillons[_n] = _zn

    # Couverture exacte de _p̂ ± 1,96 SE : somme des P(K = _k) pour les _k dont l’intervalle contient _p
    def couverture(_n, _p):
        _total = 0.0
        for _k in range(_n + 1):
            _ph = _k / _n
            _marge = 1.96 * math.sqrt(_ph * (1 - _ph) / _n)
            if _ph - _marge <= _p <= _ph + _marge:
                _total += math.comb(_n, _k) * _p**_k * (1 - _p) ** (_n - _k)
        return _total

    couvertures = {_p: [couverture(_n, _p) for _n in config["tailles_ic"]] for _p in config["p_ic"]}

    # Φ et quantiles
    regle = [Z.cdf(_k) - Z.cdf(-_k) for _k in (1, 2, 3)]
    quantiles = [Z.inv_cdf(0.95), Z.inv_cdf(0.975), Z.inv_cdf(0.995)]
    duree = time.perf_counter() - _t0

    print(f"Calcul : {duree:.2f} s")
    for _nom, _ds in distances.items():
        print(f"  {_nom:>15} : D_n = " + ", ".join(
            f"{_d:.3f} (n = {_n})" for _n, _d in zip(config["tailles"], _ds, strict=True)))
    for _p, _cs in couvertures.items():
        print(f"  couverture de p̂ ± 1,96 SE pour p = {_p} : " + ", ".join(
            f"{_c:.3f} (n = {_n})" for _n, _c in zip(config["tailles_ic"], _cs, strict=True)))
    print("  règle 68-95-99,7 : " + ", ".join(f"{r:.4f}" for r in regle))
    print("  quantiles 0,95 ; 0,975 ; 0,995 : " + ", ".join(f"{q:.4f}" for q in quantiles))
    return couvertures, distances, duree, echantillons, quantiles, regle


@app.cell
def _(SMOKE, Z, config, couvertures, echantillons, mo, np, plt, sauvegarder_figure):
    # --- Figure, sauvegardée dans figures/ (suffixe -smoke : _n’écrase pas celle du run complet) ---
    _fig, _axes = plt.subplots(1, 5, figsize=(19, 3.8))
    _z = np.linspace(-4, 4, 400)
    _densite = np.array([Z.pdf(v) for v in _z])
    for _ax, _n in zip(_axes[:4], config["tailles"], strict=True):
        _ax.hist(echantillons[_n], bins=np.linspace(-4, 4, 41), density=True, alpha=0.7,
                 label="moyennes centrées réduites")
        _ax.plot(_z, _densite, "r-", lw=2, label="densité φ")
        _ax.set(title=f"exponentielle, n = {_n}", xlim=(-4, 4))
    _axes[0].legend(fontsize=8)
    _ax = _axes[4]
    for _p, _cs in couvertures.items():
        _ax.semilogx(config["tailles_ic"], _cs, "o-", label=f"p = {str(_p).replace('.', ',')}")
    _ax.axhline(0.95, color="k", ls="--", lw=0.8)
    _ax.set(xlabel="n", ylabel="couverture exacte", title="p̂ ± 1,96 SE", ylim=(0, 1))
    _ax.legend(fontsize=8)
    _nom = "tcl-smoke" if SMOKE else "tcl"
    _chemins = sauvegarder_figure(_fig, _nom, mo.notebook_dir() / "figures")
    print(f"Figure : {_chemins[0].name}")
    _fig  # noqa: B018 (marimo affiche la dernière expression)
    return


@app.cell
def _(SMOKE, couvertures, distances, duree, mo, quantiles, regle):
    # --- Bilan : seuils chiffrés du README, vérifiés aussi en smoke (tolérance plus large) ---
    _echecs = []
    _seuil = 0.05 if SMOKE else 0.03
    for _nom, _ds in distances.items():
        # La Bernoulli rare est discrète et très dissymétrique (n p = 5 pour n = 100) : ses
        # sauts restent grands, l’écart à Φ diminue mais reste loin des deux autres lois.
        if _nom != "Bernoulli rare" and _ds[-1] > _seuil:
            _echecs.append(f"{_nom} : D_100 = {_ds[-1]:.3f} (attendu ≤ {_seuil})")
        if _nom == "Bernoulli rare" and _ds[-1] < 0.08:
            _echecs.append(f"{_nom} : D_100 = {_ds[-1]:.3f} (attendu ≥ 0,08 : la cloche n’est pas encore là)")
        if _ds[-1] > _ds[0] / 3:
            _echecs.append(f"{_nom} : D_100 = {_ds[-1]:.3f} pas trois fois plus petit que D_1 = {_ds[0]:.3f}")
    _c_bon = couvertures[0.5][-1]
    _c_rare = couvertures[0.02][0]
    if abs(_c_bon - 0.95) > 0.01:
        _echecs.append(f"couverture pour p = 0,5 et n = 1 000 : {_c_bon:.4f} (attendue 0,95 ± 0,01)")
    if _c_rare > 0.8:
        _echecs.append(f"couverture pour p = 0,02 et n = 20 : {_c_rare:.4f} (attendue < 0,8)")
    _attendus_regle = (0.6827, 0.9545, 0.9973)
    _attendus_q = (1.645, 1.96, 2.576)
    if any(abs(_r - _a) > 5e-5 for _r, _a in zip(regle, _attendus_regle, strict=True)):
        _echecs.append(f"règle 68-95-99,7 : {regle}")
    if any(abs(_q - _a) > 5e-4 for _q, _a in zip(quantiles, _attendus_q, strict=True)):
        _echecs.append(f"quantiles : {quantiles}")
    if _echecs:
        raise SystemExit("Bilan : ÉCHEC : " + " ; ".join(_echecs))
    _d_exp = distances["exponentielle"]
    print(f"Bilan : OK (exponentielle D_1 = {_d_exp[0]:.3f} → D_100 = {_d_exp[-1]:.3f} ; couverture "
          f"{_c_bon:.4f} pour p = 0,5 et n = 1 000, {_c_rare:.4f} pour p = 0,02 et n = 20 ; {duree:.2f} s)")

    def _fr(_x, _d):
        return f"{_x:.{_d}f}".replace(".", ",")

    mo.md(f"**Bilan** : pour l’exponentielle, l’écart à Φ passe de {_fr(_d_exp[0], 3)} ($n = 1$) à "
          f"{_fr(_d_exp[-1], 3)} ($n = 100$) ; l’intervalle à 1,96 couvre $p$ dans {_fr(100 * _c_bon, 1)} % des "
          f"cas pour $p = 0{{,}}5$ et $n = 1\\,000$, mais dans {_fr(100 * _c_rare, 1)} % seulement pour "
          f"$p = 0{{,}}02$ et $n = 20$.")
    return


if __name__ == "__main__":
    app.run()
