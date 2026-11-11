# Gabarit de notebook marimo du parcours. Copier ce dossier en notebooks/NN-slug/
# (NN = numéro du chapitre, slug du plan), puis remplacer l’exemple.
#
# Trois façons de l’exécuter (notebooks/README.md) :
#   make nb NUM=_gabarit              → éditeur marimo (bouton « Lancer le calcul »)
#   make nb NUM=_gabarit MODE=run     → script complet : le bouton est considéré comme cliqué
#   make nb NUM=_gabarit MODE=smoke   → test rapide, moins de 30 s, utilisé par make smoke
#
# Structure imposée, dans cet ordre : imports, énoncé, configuration (seul endroit où le
# mode smoke réduit le travail), calcul derrière un bouton, vérification exacte par sympy,
# figure (suffixe -smoke en mode smoke), bilan chiffré qui lève SystemExit sous le seuil.
#
# Exemple : dérivée numérique de exp, différences avant et centrées (chapitre 12).

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys
    import time
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import sympy as sp

    try:
        # Paquet commun du parcours (§2.3) : à utiliser dès qu’il existe.
        from math_commun import is_smoke, sauvegarder_figure, seed_everything, style_commun
    except ModuleNotFoundError:
        # Repli local tant que math_commun n’est pas écrit : mêmes noms, mêmes signatures.
        import random

        def is_smoke(argv=None):
            argv = sys.argv if argv is None else argv
            return "--smoke" in argv or os.environ.get("MATH_SMOKE", "") == "1"

        def seed_everything(seed=None):
            seed = int(os.environ.get("SEED", 42)) if seed is None else seed
            random.seed(seed)
            np.random.seed(seed)
            return seed

        def style_commun():
            plt.rcParams.update({"figure.figsize": (7, 4.2), "savefig.dpi": 150, "savefig.bbox": "tight",
                                 "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                                 "axes.spines.right": False, "font.size": 11})

        def sauvegarder_figure(fig, nom, dossier="figures", formats=("png",)):
            dossier = Path(dossier)
            dossier.mkdir(parents=True, exist_ok=True)
            chemins = [dossier / f"{nom}.{fmt}" for fmt in formats]
            for chemin in chemins:
                fig.savefig(chemin)
            return chemins

    return is_smoke, mo, np, plt, sauvegarder_figure, seed_everything, sp, style_commun, time


@app.cell
def _(mo):
    mo.md(r"""
    # Gabarit : approcher une dérivée par un taux d’accroissement

    **Objectif.** Montrer la structure attendue d’un notebook, sur un exemple court du
    chapitre 12 : le nombre dérivé $f'(x)$ est la limite du taux d’accroissement
    $\dfrac{f(x + h) - f(x)}{h}$ quand $h$ tend vers $0$.

    On compare deux approximations pour $f = \exp$ (dont la dérivée est $\exp$ elle-même) :

    - différence **avant** : $\dfrac{f(x + h) - f(x)}{h}$, erreur de l’ordre de $h$ ;
    - différence **centrée** : $\dfrac{f(x + h) - f(x - h)}{2h}$, erreur de l’ordre de $h^2$.

    Pour $h$ trop petit, l’erreur d’arrondi des flottants reprend le dessus (chapitre 48).
    """)


@app.cell
def _(is_smoke, np, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        n_points=20 if SMOKE else 2_000,   # points x tirés au hasard dans [−2 ; 2]
        n_pas=8 if SMOKE else 60,          # nombre de pas h entre 10⁻¹² et 10⁻¹
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
            mo.md("*Cliquez sur « Lancer le calcul ».*"))

    t0 = time.perf_counter()
    xs = rng.uniform(-2, 2, config["n_points"])
    pas = np.logspace(-12, -1, config["n_pas"])
    exacte = np.exp(xs)
    # Erreur relative maximale sur les points tirés, pour chaque pas h
    err_avant = np.array([np.max(np.abs((np.exp(xs + h) - np.exp(xs)) / h - exacte) / exacte) for h in pas])
    err_centree = np.array([np.max(np.abs((np.exp(xs + h) - np.exp(xs - h)) / (2 * h) - exacte) / exacte)
                            for h in pas])
    # Pente en log-log sur la zone où l’erreur de méthode domine (h ≥ 10⁻³)
    zone = pas >= 1e-3
    pente_avant = np.polyfit(np.log10(pas[zone]), np.log10(err_avant[zone]), 1)[0]
    pente_centree = np.polyfit(np.log10(pas[zone]), np.log10(err_centree[zone]), 1)[0]
    duree = time.perf_counter() - t0
    print(f"Calcul : {duree:.2f} s ; pentes log-log : avant {pente_avant:.2f}, centrée {pente_centree:.2f}")
    return duree, err_avant, err_centree, pas, pente_avant, pente_centree


@app.cell
def _(sp):
    # --- Vérification exacte (sympy) : le développement de Taylor explique les deux pentes ---
    x, h = sp.symbols("x h")
    f = sp.exp(x)
    avant = sp.series((f.subs(x, x + h) - f) / h - sp.diff(f, x), h, 0, 2).removeO()
    centree = sp.series((f.subs(x, x + h) - f.subs(x, x - h)) / (2 * h) - sp.diff(f, x), h, 0, 3).removeO()
    assert sp.simplify(avant - h * sp.exp(x) / 2) == 0, avant
    assert sp.simplify(centree - h**2 * sp.exp(x) / 6) == 0, centree
    print(f"sympy : erreur avant ≈ {avant}, erreur centrée ≈ {centree}")
    return


@app.cell
def _(SMOKE, err_avant, err_centree, mo, pas, plt, sauvegarder_figure):
    # --- Figure, sauvegardée dans figures/ (suffixe -smoke : n’écrase pas celle du run complet) ---
    fig, ax = plt.subplots()
    ax.loglog(pas, err_avant, "o-", ms=3, label="différence avant (ordre 1)")
    ax.loglog(pas, err_centree, "s-", ms=3, label="différence centrée (ordre 2)")
    ax.set(xlabel="pas h", ylabel="erreur relative maximale", title="Dérivée numérique de exp")
    ax.legend()
    chemins = sauvegarder_figure(fig, "erreur-pas-smoke" if SMOKE else "erreur-pas", mo.notebook_dir() / "figures")
    print(f"Figure : {chemins[0].name}")
    fig  # noqa: B018 (marimo affiche la dernière expression)
    return


@app.cell
def _(SMOKE, duree, err_centree, mo, pas, pente_avant, pente_centree):
    # --- Bilan : seuils chiffrés du README. En smoke, ils sont vérifiés aussi (calcul exact),
    # avec une tolérance plus large sur les pentes (moins de points).
    tol = 0.3 if SMOKE else 0.1
    meilleure = float(err_centree.min())
    echecs = []
    if abs(pente_avant - 1) > tol:
        echecs.append(f"pente avant {pente_avant:.2f} (attendue 1 ± {tol})")
    if abs(pente_centree - 2) > tol:
        echecs.append(f"pente centrée {pente_centree:.2f} (attendue 2 ± {tol})")
    if meilleure > 1e-9:
        echecs.append(f"meilleure erreur centrée {meilleure:.1e} (attendue ≤ 1e-9)")
    if echecs:
        raise SystemExit("Bilan : ÉCHEC : " + " ; ".join(echecs))
    h_opt = float(pas[err_centree.argmin()])
    print(f"Bilan : OK (pentes {pente_avant:.2f} et {pente_centree:.2f}, "
          f"meilleure erreur centrée {meilleure:.1e} pour h = {h_opt:.0e}, {duree:.2f} s)")
    mo.md(f"**Bilan** : pentes {pente_avant:.2f} (avant) et {pente_centree:.2f} (centrée) ; "
          f"meilleure erreur centrée {meilleure:.1e} pour $h \\approx$ {h_opt:.0e}.")
    return


if __name__ == "__main__":
    app.run()
