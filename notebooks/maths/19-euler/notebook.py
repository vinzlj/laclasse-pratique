# Notebook 19-euler (chapitre 19, Équations différentielles et méthode d’Euler).
#
# Simuler une équation différentielle pas à pas : y_{n+1} = y_n + h f(t_n, y_n). On compare
# le calcul d’Euler à la solution exacte pour le café du cours, T′ = −0,1 (T − 20), et pour
# le modèle logistique y′ = 0,8 y (1 − y/8) ; on mesure l’erreur en fonction du pas h
# (pente 1 en échelle log-log : erreur proportionnelle à h) ; puis on montre ce qui se passe
# quand h dépasse 2/|a| : le calcul oscille, puis explose.
#
#   make nb NUM=19              → éditeur marimo (bouton « Lancer le calcul »)
#   make nb NUM=19 MODE=run     → script complet
#   make nb NUM=19 MODE=smoke   → test rapide (moins de 30 s)

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import sympy as sp
    from math_commun import is_smoke, sauvegarder_figure, seed_everything, style_commun

    return is_smoke, mo, np, plt, sauvegarder_figure, seed_everything, sp, style_commun, time


@app.cell
def _(mo):
    mo.md(r"""
    # Méthode d’Euler : pas, erreur et stabilité

    **Chapitre 19, sections 3 à 5.** Pour une équation $y' = f(t, y)$ avec $y(0) = y_0$, la
    méthode d’Euler avance par pas de durée $h$ en suivant la tangente :

    $$t_{n+1} = t_n + h, \qquad y_{n+1} = y_n + h\,f(t_n, y_n).$$

    On l’applique à deux équations dont on connaît la solution exacte :

    - le **café** du cours, $T' = -0{,}1\,(T - 20)$, $T(0) = 90$, de solution
      $T(t) = 20 + 70\,\mathrm{e}^{-0{,}1\,t}$, jusqu’à $t = 30$ minutes ;
    - le **modèle logistique** $y' = 0{,}8\,y\,(1 - y/8)$, $y(0) = 0{,}5$, de solution
      $y(t) = \dfrac{8}{1 + 15\,\mathrm{e}^{-0{,}8\,t}}$, jusqu’à $t = 10$.

    On mesure l’erreur finale pour des pas de plus en plus petits, puis on fait varier le
    pas du café autour de $\dfrac{2}{|a|} = 20$ minutes.
    """)


@app.cell
def _(is_smoke, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)  # aucun tirage au hasard ici ; graine posée par convention
    style_commun()
    config = dict(
        divisions=range(1, 9) if SMOKE else range(1, 15),  # nombre de pas N = 2¹ … 2⁸ ou 2¹⁴ (× base)
        pas_stabilite=[5, 15, 20, 25],                      # pas du café autour de 2/|a| = 20
        pas_explosion=12,                                   # nombre de pas pour h = 25
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SMOKE, config


@app.cell
def _(mo):
    # Le bouton empêche le calcul de se relancer à chaque modification d’une cellule amont.
    bouton = mo.ui.run_button(label="Lancer le calcul")
    bouton  # noqa: B018 (marimo affiche la dernière expression)
    return (bouton,)


@app.cell
def _(bouton, config, mo, np, time):
    # En script (make nb MODE=run|smoke), personne ne clique : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer le calcul ».*"))

    t0 = time.perf_counter()

    def euler(f, y0, T, n):
        """n pas d’Euler de longueur h = T/n ; renvoie les instants et les valeurs."""
        h = T / n
        ts = np.linspace(0.0, T, n + 1)
        ys = np.empty(n + 1)
        ys[0] = y0
        for k in range(n):
            ys[k + 1] = ys[k] + h * f(ts[k], ys[k])
        return ts, ys

    cas = {
        "café": (lambda t, y: -0.1 * (y - 20), 90.0, 30.0, lambda t: 20 + 70 * np.exp(-0.1 * t), 6),
        "logistique": (lambda t, y: 0.8 * y * (1 - y / 8), 0.5, 10.0,
                       lambda t: 8 / (1 + 15 * np.exp(-0.8 * t)), 10),
    }
    erreurs, pas = {}, {}
    for _nom, (_f, _y0, _T, _exacte, _base) in cas.items():
        _ns = np.array([_base * 2**p for p in config["divisions"]])
        pas[_nom] = _T / _ns
        erreurs[_nom] = np.array([abs(euler(_f, _y0, _T, int(_n))[1][-1] - _exacte(_T)) for _n in _ns])
    # Pente de l’erreur en log-log : 1 attendu (erreur proportionnelle à h)
    pentes = {_nom: float(np.polyfit(np.log10(pas[_nom]), np.log10(erreurs[_nom]), 1)[0]) for _nom in cas}

    # Valeurs du cours
    cafe = cas["café"][0]
    cours = {
        "café h = 5, t = 30": euler(cafe, 90.0, 30.0, 6)[1][-1],
        "café h = 1, t = 10": euler(cafe, 90.0, 10.0, 10)[1][-1],
        "y' = y, h = 0,1": euler(lambda t, y: y, 1.0, 1.0, 10)[1][-1],
        "y' = y, h = 0,01": euler(lambda t, y: y, 1.0, 1.0, 100)[1][-1],
        "logistique, un pas de 0,5": euler(cas["logistique"][0], 0.5, 0.5, 1)[1][-1],
    }

    # Stabilité : café avec des pas autour de 2/|a| = 20
    trajectoires = {}
    for _h in config["pas_stabilite"]:
        _n = config["pas_explosion"] if _h == 25 else max(1, int(round(120 / _h)))
        trajectoires[_h] = euler(cafe, 90.0, _h * _n, _n)
    duree = time.perf_counter() - t0

    print(f"Calcul : {duree:.3f} s")
    for _nom in cas:
        print(f"  {_nom:10s} pente log-log {pentes[_nom]:+.3f} ; erreurs {np.round(erreurs[_nom][:4], 4)} …")
    for _k, _v in cours.items():
        print(f"  {_k} : {_v:.6f}")
    for _h, (_ts, _ys) in trajectoires.items():
        print(f"  café, h = {_h} : {np.round(_ys[:6], 3)} …")
    return cours, duree, erreurs, pas, pentes, trajectoires


@app.cell
def _(sp):
    # --- Vérification exacte (sympy) : solutions du cours et suite d’Euler pour y′ = ay ---
    _t, _h, _n = sp.symbols("t h n", positive=True)
    _T = 20 + 70 * sp.exp(-sp.Rational(1, 10) * _t)
    assert sp.simplify(sp.diff(_T, _t) - (-sp.Rational(1, 10) * (_T - 20))) == 0
    _y = 8 / (1 + 15 * sp.exp(-sp.Rational(4, 5) * _t))
    assert sp.simplify(sp.diff(_y, _t) - sp.Rational(4, 5) * _y * (1 - _y / 8)) == 0
    assert sp.simplify(_y.subs(_t, 0) - sp.Rational(1, 2)) == 0
    # Un pas d’Euler sur l’écart au café : multiplie par 1 − 0,1 h ; stable si |1 − 0,1 h| < 1
    _fact = 1 - sp.Rational(1, 10) * _h
    assert sp.solve_univariate_inequality(sp.Abs(_fact) < 1, _h, relational=False) == sp.Interval.open(0, 20)
    print("sympy : solutions du café et de la logistique vérifiées, stabilité 0 < h < 20")
    return


@app.cell
def _(SMOKE, erreurs, mo, np, pas, plt, sauvegarder_figure, trajectoires):
    # --- Figure : erreur selon h (log-log) et trajectoires du café selon le pas ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for _nom, _style in zip(erreurs, ["o-", "s-"], strict=True):
        ax1.loglog(pas[_nom], erreurs[_nom], _style, ms=3, label=_nom)
    ax1.set(xlabel="pas h", ylabel="erreur finale", title="Erreur d’Euler : proportionnelle à h")
    ax1.legend(fontsize=9)
    _tt = np.linspace(0, 60, 200)
    ax2.plot(_tt, 20 + 70 * np.exp(-0.1 * _tt), color="black", lw=2, label="solution exacte")
    for _h, (_ts, _ys) in trajectoires.items():
        _m = _ts <= 60
        ax2.plot(_ts[_m], _ys[_m], "o-", ms=3, label=f"Euler, h = {_h}")
    ax2.axhline(20, color="gray", lw=1, ls="--")
    ax2.set(xlabel="t (min)", ylabel="T (°C)", ylim=(-120, 200), title="Café : stable si h < 20")
    ax2.legend(fontsize=8)
    chemins = sauvegarder_figure(fig, "euler-smoke" if SMOKE else "euler", mo.notebook_dir() / "figures")
    print(f"Figure : {chemins[0].name}")
    fig  # noqa: B018 (marimo affiche la dernière expression)
    return


@app.cell
def _(SMOKE, cours, duree, mo, pentes, trajectoires):
    # --- Bilan : seuils chiffrés du README, vérifiés aussi en smoke ---
    tol = 0.1
    echecs = []
    for _nom, _p in pentes.items():
        if abs(_p - 1.0) > tol:
            echecs.append(f"{_nom} : pente {_p:.3f} (attendue 1 ± {tol})")
    attendues = {
        "café h = 5, t = 30": 21.09375,
        "café h = 1, t = 10": 20 + 70 * 0.9**10,
        "y' = y, h = 0,1": 1.1**10,
        "y' = y, h = 0,01": 1.01**100,
        "logistique, un pas de 0,5": 0.6875,
    }
    for _k, _v in attendues.items():
        if abs(cours[_k] - _v) > 1e-9:
            echecs.append(f"{_k} : {cours[_k]} au lieu de {_v}")
    _y15 = trajectoires[15][1]
    _y20 = trajectoires[20][1]
    _y25 = trajectoires[25][1]
    if not abs(_y15[-1] - 20) < 1:
        echecs.append(f"h = 15 : l’oscillation devrait s’amortir (dernière valeur {_y15[-1]:.3f})")
    if not (abs(_y20[2] - 90) < 1e-9 and abs(_y20[1] + 50) < 1e-9):
        echecs.append(f"h = 20 : oscillation 90 ; −50 attendue, obtenu {_y20[:3]}")
    if not abs(_y25[-1]) > 1000:
        echecs.append(f"h = 25 : explosion attendue (|T| > 1 000), obtenu {_y25[-1]:.1f}")
    if echecs:
        raise SystemExit("Bilan : ÉCHEC : " + " ; ".join(echecs))
    print(f"Bilan : OK (pentes {', '.join(f'{k} {v:.3f}' for k, v in pentes.items())} à ±{tol}, "
          f"valeurs du cours retrouvées, h = 25 : |T| = {abs(_y25[-1]):.0f} après {len(_y25) - 1} pas, "
          f"{duree:.3f} s{', smoke' if SMOKE else ''})")
    mo.md("**Bilan** : l’erreur d’Euler est proportionnelle au pas $h$ (pente $1$). Pour le café, "
          "un pas plus petit que $20$ minutes converge, $h = 20$ oscille sans fin entre $90$ et $-50$, "
          "et $h = 25$ explose.")
    return


if __name__ == "__main__":
    app.run()
