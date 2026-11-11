# Notebook 13-newton-taylor (chapitre 13, Étude de fonctions, optimisation, convexité, Taylor).
#
# Méthode de Newton pour √2 : le nombre de décimales justes double à chaque pas (comparé à la
# dichotomie) ; polynômes de Taylor d’ordre 1 et 2 de √(1 + h) : erreurs en h² et en h³ ;
# un départ qui fait tourner Newton en rond (x³ − 2x + 2 depuis 0) et la proportion de
# départs au hasard qui convergent.
#
#   make nb NUM=13              → éditeur marimo (bouton « Lancer le calcul »)
#   make nb NUM=13 MODE=run     → script complet
#   make nb NUM=13 MODE=smoke   → test rapide (moins de 30 s)

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import time
    from fractions import Fraction

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import sympy as sp
    from math_commun import is_smoke, sauvegarder_figure, seed_everything, style_commun

    return Fraction, is_smoke, math, mo, np, plt, sauvegarder_figure, seed_everything, sp, style_commun, time


@app.cell
def _(mo):
    mo.md(r"""
    # Newton et Taylor : remplacer une courbe par sa tangente, ou par une parabole

    **Chapitre 13, sections 6 et 7.** Trois expériences :

    1. **Newton pour $\sqrt{2}$.** On résout $x^2 - 2 = 0$ par $x_{n+1} = x_n - \dfrac{x_n^2 - 2}{2x_n}$
       à partir de $x_0 = 2$, en fractions exactes, et on compte les décimales justes à chaque pas.
       On compare avec la **dichotomie**, qui coupe l’intervalle en deux à chaque pas.
    2. **Taylor pour $\sqrt{1 + h}$.** On compare l’ordre 1, $1 + \dfrac{h}{2}$, et l’ordre 2,
       $1 + \dfrac{h}{2} - \dfrac{h^2}{8}$, pour des $h$ de plus en plus petits : quand $h$ est
       divisé par 10, l’erreur est divisée par 100 (ordre 1) ou par 1 000 (ordre 2).
    3. **Quand Newton échoue.** Pour $f(x) = x^3 - 2x + 2$, le départ $x_0 = 0$ donne
       $0, 1, 0, 1, \ldots$ ; on mesure la part des départs tirés au hasard dans $[-3 ; 3]$ qui
       atteignent quand même la racine.
    """)


@app.cell
def _(is_smoke, np, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        pas_newton=6,                        # itérations de Newton pour √2 (exactes, rapides)
        n_h=30 if SMOKE else 400,            # nombre de pas h entre 10⁻⁴ et 10⁻⁰ʼ⁵ pour Taylor
        n_departs=500 if SMOKE else 50_000,  # départs au hasard pour x³ − 2x + 2
        iterations=60,                       # itérations de Newton par départ
    )
    rng = np.random.default_rng(SEED)
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SMOKE, config, rng


@app.cell
def _(mo):
    # Le bouton empêche le calcul de se relancer à chaque modification d’une cellule amont.
    bouton = mo.ui.run_button(label="Lancer le calcul")
    bouton  # noqa: B018 (marimo affiche la dernière expression)
    return (bouton,)


@app.cell
def _(Fraction, bouton, config, math, mo, np, rng, sp, time):
    # En script (make nb MODE=run|smoke), personne ne clique : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer le calcul ».*"))

    t0 = time.perf_counter()
    racine2 = sp.sqrt(2)

    # 1. Newton pour √2, en fractions exactes ; erreur évaluée à 80 chiffres par sympy
    iteres = [Fraction(2)]
    for _ in range(config["pas_newton"]):
        _x = iteres[-1]
        iteres.append(_x - (_x * _x - 2) / (2 * _x))
    erreurs_newton = [abs(sp.N(sp.Rational(_x.numerator, _x.denominator) - racine2, 80)) for _x in iteres]
    decimales = [float(-sp.log(_e, 10)) for _e in erreurs_newton]
    # Rapport e(n+1) / e(n)² : tend vers 1/(2√2) près de la racine
    rapports = [float(erreurs_newton[_k + 1] / erreurs_newton[_k] ** 2) for _k in range(len(erreurs_newton) - 1)]
    # Dichotomie sur [1 ; 2] : nombre de pas pour une erreur inférieure à celle du dernier itéré
    cible = erreurs_newton[-1]
    pas_dichotomie = math.ceil(float(sp.log(1 / cible, 2)))

    # 2. Taylor pour √(1 + h)
    pas_h = np.logspace(-4, -0.5, config["n_h"])
    exacte = np.sqrt(1 + pas_h)
    err1 = np.abs(exacte - (1 + pas_h / 2))
    err2 = np.abs(exacte - (1 + pas_h / 2 - pas_h**2 / 8))
    zone = pas_h >= 1e-3  # au-dessous, les arrondis des flottants polluent err2 (≈ 10⁻¹³)
    pente1 = np.polyfit(np.log10(pas_h[zone]), np.log10(err1[zone]), 1)[0]
    pente2 = np.polyfit(np.log10(pas_h[zone]), np.log10(err2[zone]), 1)[0]
    majoration_ok = bool(np.all(err2 <= pas_h**3 / 16 * (1 + 1e-9) + 1e-15))

    # 3. Newton pour x³ − 2x + 2 : cycle depuis 0, puis départs au hasard
    def _pas(_x):
        return _x - (_x**3 - 2 * _x + 2) / (3 * _x**2 - 2)

    cycle = [Fraction(0)]
    for _ in range(4):
        cycle.append(_pas(cycle[-1]))
    racine_cubique = float(sp.nsolve(sp.Symbol("x") ** 3 - 2 * sp.Symbol("x") + 2, -1.7))
    departs = rng.uniform(-3, 3, config["n_departs"])
    _x = departs.copy()
    with np.errstate(all="ignore"):
        for _ in range(config["iterations"]):
            _x = _pas(_x)
    converge = np.abs(_x - racine_cubique) < 1e-10
    part_convergee = float(converge.mean())
    duree = time.perf_counter() - t0
    print(f"Calcul : {duree:.2f} s")
    print("Newton pour √2 : décimales justes " + ", ".join(f"{_d:.1f}" for _d in decimales))
    print(f"Rapports e(n+1)/e(n)² : {', '.join(f'{_r:.4f}' for _r in rapports)} (limite 1/(2√2) ≈ 0,3536)")
    print(f"Dichotomie : {pas_dichotomie} pas pour la précision atteinte par Newton en {config['pas_newton']}")
    print(f"Taylor : pentes log-log {pente1:.3f} (ordre 1) et {pente2:.3f} (ordre 2)")
    print(f"x³ − 2x + 2 depuis 0 : {', '.join(str(_c) for _c in cycle)} ; racine ≈ {racine_cubique:.6f} ; "
          f"part des départs qui convergent : {part_convergee:.3f}")
    return (cycle, decimales, duree, err1, err2, majoration_ok, part_convergee, pas_dichotomie, pas_h,
            pente1, pente2, racine_cubique, rapports)


@app.cell
def _(sp):
    # --- Vérification exacte (sympy) ---
    _h, _x = sp.symbols("h x", positive=True)
    # Développement de √(1 + h) : 1 + h/2 − h²/8 + h³/16 + …
    _serie = sp.series(sp.sqrt(1 + _h), _h, 0, 4).removeO()
    assert sp.expand(_serie - (1 + _h / 2 - _h**2 / 8 + _h**3 / 16)) == 0, _serie
    # Newton pour x² − 2 : x(n+1) − √2 = (x(n) − √2)² / (2 x(n)), exactement
    _suivant = _x - (_x**2 - 2) / (2 * _x)
    assert sp.simplify(_suivant - sp.sqrt(2) - (_x - sp.sqrt(2)) ** 2 / (2 * _x)) == 0
    # Carré de T₂ : 1 + h − h³/8 + h⁴/64 (exercice papier du chapitre)
    assert sp.expand((1 + _h / 2 - _h**2 / 8) ** 2 - (1 + _h - _h**3 / 8 + _h**4 / 64)) == 0
    print(f"sympy : √(1 + h) = {_serie} + O(h⁴) ; erreur de Newton au carré vérifiée")
    return


@app.cell
def _(SMOKE, decimales, err1, err2, mo, pas_h, plt, sauvegarder_figure):
    # --- Figure : à gauche les décimales justes de Newton, à droite les erreurs de Taylor ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(range(len(decimales)), decimales, "o-", label="Newton (tangentes)")
    ax1.plot(range(len(decimales)), [_n * 0.30103 for _n in range(len(decimales))], "s--",
             label="dichotomie (un bit par pas)")
    ax1.set(xlabel="pas n", ylabel="décimales justes", title="√2 : les décimales doublent")
    ax1.legend()
    ax2.loglog(pas_h, err1, label="ordre 1 : 1 + h/2")
    ax2.loglog(pas_h, err2, label="ordre 2 : 1 + h/2 − h²/8")
    ax2.loglog(pas_h, pas_h**3 / 16, ":", label="majoration h³/16")
    ax2.set(xlabel="h", ylabel="erreur", title="√(1 + h) : erreurs en h² et h³")
    ax2.legend(fontsize=9)
    _nom = "newton-taylor-smoke" if SMOKE else "newton-taylor"
    chemins = sauvegarder_figure(fig, _nom, mo.notebook_dir() / "figures")
    print(f"Figure : {chemins[0].name}")
    fig  # noqa: B018 (marimo affiche la dernière expression)
    return


@app.cell
def _(Fraction, SMOKE, cycle, decimales, duree, majoration_ok, mo, part_convergee, pas_dichotomie, pente1,
      pente2, rapports):
    # --- Bilan : seuils chiffrés du README, vérifiés aussi en smoke (tolérance plus large) ---
    tol = 0.15 if SMOKE else 0.05
    echecs = []
    if decimales[-1] < 40:
        echecs.append(f"Newton : {decimales[-1]:.1f} décimales après 6 pas (attendu ≥ 40)")
    if abs(rapports[-1] - 0.35355) > 0.001:
        echecs.append(f"rapport e(n+1)/e(n)² = {rapports[-1]:.4f} (attendu 0,3536 ± 0,001)")
    if pas_dichotomie < 100:
        echecs.append(f"dichotomie : {pas_dichotomie} pas (attendu ≥ 100)")
    if abs(pente1 - 2) > tol or abs(pente2 - 3) > tol:
        echecs.append(f"pentes {pente1:.2f} et {pente2:.2f} (attendues 2 et 3 ± {tol})")
    if not majoration_ok:
        echecs.append("erreur d’ordre 2 au-dessus de h³/16")
    if cycle != [Fraction(0), Fraction(1), Fraction(0), Fraction(1), Fraction(0)]:
        echecs.append(f"cycle attendu 0, 1, 0, 1, 0 : obtenu {cycle}")
    if not 0.5 <= part_convergee < 1:
        echecs.append(f"part des départs convergents {part_convergee:.3f} (attendue entre 0,5 et 1, 1 exclu)")
    if echecs:
        raise SystemExit("Bilan : ÉCHEC : " + " ; ".join(echecs))
    print(f"Bilan : OK (Newton : {decimales[-1]:.0f} décimales en 6 pas, rapport {rapports[-1]:.4f} ; "
          f"dichotomie : {pas_dichotomie} pas ; pentes {pente1:.2f} et {pente2:.2f} ; cycle 0, 1, 0, 1 ; "
          f"{100 * part_convergee:.1f} % des départs convergent ; {duree:.2f} s)")
    def _fr(_v, _d):
        return f"{_v:.{_d}f}".replace(".", ",")

    mo.md(f"**Bilan** : Newton atteint {decimales[-1]:.0f} décimales justes de √2 en 6 pas, quand la "
          f"dichotomie en demande {pas_dichotomie}. Erreurs de Taylor : pentes {_fr(pente1, 2)} et "
          f"{_fr(pente2, 2)}. Pour x³ − 2x + 2, {_fr(100 * part_convergee, 1)} % des départs au hasard "
          "trouvent la racine ; le départ 0 tourne en rond.")
    return


if __name__ == "__main__":
    app.run()
