# Notebook 12-derivee-numerique (chapitre 12, Dérivation).
#
# Approcher un nombre dérivé par un taux de variation : différence avant (erreur de
# l’ordre de h), différence centrée (erreur de l’ordre de h²), effondrement pour h trop
# petit (arrondis des flottants), puis ce que voient ces formules en un point anguleux
# (|x| et ReLU en 0).
#
#   make nb NUM=12              → éditeur marimo (bouton « Lancer le calcul »)
#   make nb NUM=12 MODE=run     → script complet
#   make nb NUM=12 MODE=smoke   → test rapide (moins de 30 s)

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
    # Dérivée numérique : approcher f′(a) par un taux de variation

    **Chapitre 12, sections 2 et 8.** Le nombre dérivé est la limite du taux de variation :
    $f'(a) = \lim_{h \to 0} \dfrac{f(a + h) - f(a)}{h}$. Une machine ne sait pas faire
    tendre $h$ vers $0$ : elle choisit un $h$ petit et calcule le taux. Deux questions :

    1. **Quelle erreur commet-on ?** On compare, pour $f = \sin$ (dont la dérivée est $\cos$) :
       - la différence **avant** $\dfrac{f(a + h) - f(a)}{h}$ ;
       - la différence **centrée** $\dfrac{f(a + h) - f(a - h)}{2h}$, la pente de la sécante
         entre $a - h$ et $a + h$.
    2. **Que se passe-t-il en un point anguleux ?** Pour $|x|$ et ReLU en $0$, les taux à
       gauche et à droite ne se rejoignent jamais, aussi petit que soit $h$.

    Pour $h$ trop petit, l’erreur remonte : $f(a + h)$ et $f(a)$ sont presque égaux, et leur
    différence perd les chiffres significatifs des flottants (chapitre 48).
    """)


@app.cell
def _(is_smoke, np, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        n_points=50 if SMOKE else 5_000,   # abscisses a tirées au hasard dans [−3 ; 3]
        n_pas=12 if SMOKE else 80,         # nombre de pas h entre 10⁻¹³ et 10⁻¹
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
def _(bouton, config, mo, np, rng, time):
    # En script (make nb MODE=run|smoke), personne ne clique : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer le calcul ».*"))

    t0 = time.perf_counter()
    a = rng.uniform(-3, 3, config["n_points"])
    pas = np.logspace(-13, -1, config["n_pas"])
    exacte = np.cos(a)
    # Erreur absolue maximale sur les abscisses tirées, pour chaque pas h
    err_avant = np.array([np.max(np.abs((np.sin(a + _h) - np.sin(a)) / _h - exacte)) for _h in pas])
    err_centree = np.array([np.max(np.abs((np.sin(a + _h) - np.sin(a - _h)) / (2 * _h) - exacte)) for _h in pas])
    # Pente en log-log là où l’erreur de méthode domine (h ≥ 10⁻³)
    zone = pas >= 1e-3
    pente_avant = np.polyfit(np.log10(pas[zone]), np.log10(err_avant[zone]), 1)[0]
    pente_centree = np.polyfit(np.log10(pas[zone]), np.log10(err_centree[zone]), 1)[0]

    # Point anguleux : taux à droite, à gauche et centré en 0, pour |x| et ReLU
    def _relu(x):
        return np.maximum(x, 0.0)

    anguleux = {}
    for _nom, _f in (("|x|", np.abs), ("ReLU", _relu)):
        _droite = np.array([(_f(_h) - _f(0.0)) / _h for _h in pas])
        _gauche = np.array([(_f(0.0) - _f(-_h)) / _h for _h in pas])
        _centre = np.array([(_f(_h) - _f(-_h)) / (2 * _h) for _h in pas])
        anguleux[_nom] = (_droite, _gauche, _centre)
    duree = time.perf_counter() - t0
    print(f"Calcul : {duree:.2f} s ; pentes log-log : avant {pente_avant:.2f}, centrée {pente_centree:.2f}")
    for _nom, (_d, _g, _c) in anguleux.items():
        print(f"{_nom} en 0 : taux à droite {_d.min():g} à {_d.max():g}, à gauche {_g.min():g} à {_g.max():g}, "
              f"centré {_c.min():g} à {_c.max():g}")
    return anguleux, duree, err_avant, err_centree, pas, pente_avant, pente_centree


@app.cell
def _(sp):
    # --- Vérification exacte (sympy) : les développements expliquent les deux pentes ---
    _x, _h = sp.symbols("x h")
    _f = sp.sin(_x)
    _avant = sp.series((_f.subs(_x, _x + _h) - _f) / _h - sp.diff(_f, _x), _h, 0, 2).removeO()
    _centree = sp.series((_f.subs(_x, _x + _h) - _f.subs(_x, _x - _h)) / (2 * _h) - sp.diff(_f, _x), _h, 0, 3).removeO()
    assert sp.simplify(_avant + _h * sp.sin(_x) / 2) == 0, _avant
    assert sp.simplify(_centree + _h**2 * sp.cos(_x) / 6) == 0, _centree
    # Pour x², le taux avant vaut exactement 2a + h (section 2 du chapitre)
    _a = sp.symbols("a")
    assert sp.simplify(((_a + _h) ** 2 - _a**2) / _h - (2 * _a + _h)) == 0
    print(f"sympy : erreur avant ≈ {_avant}, erreur centrée ≈ {_centree}")
    return


@app.cell
def _(SMOKE, anguleux, err_avant, err_centree, mo, pas, plt, sauvegarder_figure):
    # --- Figure : à gauche l’erreur selon h, à droite les taux en 0 pour |x| ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.loglog(pas, err_avant, "o-", ms=3, label="différence avant (ordre 1)")
    ax1.loglog(pas, err_centree, "s-", ms=3, label="différence centrée (ordre 2)")
    ax1.set(xlabel="pas h", ylabel="erreur maximale", title="Dérivée numérique de sin")
    ax1.legend(loc="upper center", fontsize=9)
    _d, _g, _c = anguleux["|x|"]
    ax2.semilogx(pas, _d, "o-", ms=3, label="taux à droite")
    ax2.semilogx(pas, _g, "s-", ms=3, label="taux à gauche")
    ax2.semilogx(pas, _c, "^-", ms=3, label="différence centrée")
    ax2.set(xlabel="pas h", ylabel="taux", ylim=(-1.5, 2.6), title="|x| en 0 : deux pentes, aucune limite")
    ax2.legend()
    chemins = sauvegarder_figure(fig, "erreur-pas-smoke" if SMOKE else "erreur-pas", mo.notebook_dir() / "figures")
    print(f"Figure : {chemins[0].name}")
    fig  # noqa: B018 (marimo affiche la dernière expression)
    return


@app.cell
def _(SMOKE, anguleux, duree, err_centree, mo, np, pas, pente_avant, pente_centree):
    # --- Bilan : seuils chiffrés du README, vérifiés aussi en smoke (tolérance plus large) ---
    tol = 0.3 if SMOKE else 0.1
    meilleure = float(err_centree.min())
    echecs = []
    if abs(pente_avant - 1) > tol:
        echecs.append(f"pente avant {pente_avant:.2f} (attendue 1 ± {tol})")
    if abs(pente_centree - 2) > tol:
        echecs.append(f"pente centrée {pente_centree:.2f} (attendue 2 ± {tol})")
    if meilleure > 1e-9:
        echecs.append(f"meilleure erreur centrée {meilleure:.1e} (attendue ≤ 1e-9)")
    attendus = {"|x|": (1.0, -1.0, 0.0), "ReLU": (1.0, 0.0, 0.5)}
    for _nom, (_dd, _gg, _cc) in attendus.items():
        _d, _g, _c = anguleux[_nom]
        if not (np.all(_d == _dd) and np.all(_g == _gg) and np.all(_c == _cc)):
            echecs.append(f"{_nom} en 0 : taux différents de {_dd}, {_gg}, {_cc}")
    if echecs:
        raise SystemExit("Bilan : ÉCHEC : " + " ; ".join(echecs))
    h_opt = float(pas[err_centree.argmin()])
    print(f"Bilan : OK (pentes {pente_avant:.2f} et {pente_centree:.2f}, "
          f"meilleure erreur centrée {meilleure:.1e} pour h = {h_opt:.0e}, "
          f"|x| en 0 : 1 et −1, ReLU en 0 : 1 et 0, {duree:.2f} s)")
    mo.md(f"**Bilan** : pentes {pente_avant:.2f} (avant) et {pente_centree:.2f} (centrée) ; "
          f"meilleure erreur centrée {meilleure:.1e} pour $h \\approx$ {h_opt:.0e}. "
          "En 0, |x| garde deux pentes, 1 et −1, et ReLU deux pentes, 1 et 0, pour tout $h$.")
    return


if __name__ == "__main__":
    app.run()
