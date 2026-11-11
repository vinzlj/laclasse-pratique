# Notebook 02-numpy-premiers-pas (chapitre 02, numpy pour les maths).
#
# Quatre réflexes du parcours, chacun mesuré ou vérifié :
#   1. vectoriser : une somme de carrés par une boucle Python, puis par numpy (temps mesuré) ;
#   2. fixer une graine : même graine, mêmes tirages ; autre graine, autres tirages ;
#   3. tracer : n, n², 1 000 n² et 2ⁿ en échelle linéaire puis logarithmique ;
#   4. vérifier avec sympy : la formule de la somme des carrés et le seuil n = 19.
#
#   make nb NUM=02              → éditeur marimo (bouton « Lancer le calcul »)
#   make nb NUM=02 MODE=run     → script complet
#   make nb NUM=02 MODE=smoke   → test rapide (moins de 30 s)

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
    # Premiers pas avec numpy : vectoriser, fixer une graine, tracer, vérifier

    **Chapitre 02, sections 1 à 7.** Ce notebook rejoue les quatre réflexes du chapitre :

    1. **Vectoriser.** On calcule $\sum_{i=0}^{N-1} i^2$ deux fois : avec une boucle Python,
       puis avec `np.sum(x * x)`. Même résultat, mais combien de fois plus vite ?
    2. **Fixer une graine.** Deux générateurs créés avec `np.random.default_rng(42)` donnent
       exactement les mêmes tirages de dé ; avec la graine 43, la suite change.
    3. **Tracer.** Les coûts $n$, $n^2$, $1\,000\,n^2$ et $2^n$ pour $n$ de 1 à 30, en
       échelle linéaire puis logarithmique.
    4. **Vérifier avec sympy.** La formule $\sum_{i=0}^{N-1} i^2 = \frac{(N-1)N(2N-1)}{6}$,
       et le plus petit $n$ tel que $2^n > 1\,000\,n^2$ (on attend $n = 19$).
    """)


@app.cell
def _(is_smoke, np, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        N=200_000 if SMOKE else 2_000_000,   # nombre de termes de la somme des carrés
        repetitions=2 if SMOKE else 5,       # on garde le meilleur temps de chaque méthode
        tirages=20,                          # tirages de dé comparés pour la graine
        n_max=30,                            # abscisse maximale du tracé des coûts
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, numpy {np.__version__}, config {config}")
    return SEED, SMOKE, config


@app.cell
def _(mo):
    # Le bouton empêche le calcul de se relancer à chaque modification d’une cellule amont.
    bouton = mo.ui.run_button(label="Lancer le calcul")
    bouton  # noqa: B018 (marimo affiche la dernière expression)
    return (bouton,)


@app.cell
def _(SEED, bouton, config, mo, np, time):
    # En script (make nb MODE=run|smoke), personne ne clique : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer le calcul ».*"))

    # 1. Boucle contre vectorisation : somme des carrés des entiers de 0 à N − 1
    _N = config["N"]
    x = np.arange(_N, dtype=np.int64)

    def somme_boucle(valeurs):
        total = 0
        for v in valeurs:          # un tour de boucle Python par élément
            total += v * v
        return total

    liste = x.tolist()             # entiers Python, comme dans du code sans numpy
    t_boucle, t_numpy = float("inf"), float("inf")
    for _ in range(config["repetitions"]):
        _t0 = time.perf_counter()
        s_boucle = somme_boucle(liste)
        t_boucle = min(t_boucle, time.perf_counter() - _t0)
        _t0 = time.perf_counter()
        s_numpy = int(np.sum(x * x))   # une seule instruction, exécutée en code compilé
        t_numpy = min(t_numpy, time.perf_counter() - _t0)
    acceleration = t_boucle / t_numpy

    # 2. Graines : deux générateurs de même graine, puis une autre graine
    de_a = np.random.default_rng(SEED).integers(1, 7, size=config["tirages"])
    de_b = np.random.default_rng(SEED).integers(1, 7, size=config["tirages"])
    de_c = np.random.default_rng(SEED + 1).integers(1, 7, size=config["tirages"])

    # 3. Coûts à tracer, et seuil à partir duquel 2ⁿ dépasse 1 000 n²
    n = np.arange(1, config["n_max"] + 1)
    couts = {"n": n.astype(float), "n²": n.astype(float) ** 2,
             "1 000 n²": 1000.0 * n**2, "2ⁿ": 2.0**n}
    seuil = int(n[np.argmax(2.0**n > 1000.0 * n**2)])

    print(f"Somme des carrés (N = {_N}) : boucle {t_boucle * 1e3:.1f} ms, numpy {t_numpy * 1e3:.2f} ms, "
          f"rapport {acceleration:.0f}")
    print(f"Graine {SEED} : {de_a.tolist()}")
    print(f"Graine {SEED} (2e générateur) : {de_b.tolist()}")
    print(f"Graine {SEED + 1} : {de_c.tolist()}")
    print(f"Plus petit n tel que 2ⁿ > 1 000 n² : {seuil}")
    return acceleration, couts, de_a, de_b, de_c, n, s_boucle, s_numpy, seuil, t_boucle, t_numpy


@app.cell
def _(config, sp):
    # --- Vérification exacte (sympy) ---
    _i, _N = sp.symbols("i N", integer=True, nonnegative=True)
    formule = sp.factor(sp.summation(_i**2, (_i, 0, _N - 1)))
    assert sp.simplify(formule - (_N - 1) * _N * (2 * _N - 1) / 6) == 0, formule
    attendu = int(formule.subs(_N, config["N"]))
    # Seuil : on cherche le premier entier n ≥ 1 avec 2ⁿ − 1000 n² > 0, en entiers exacts
    _k = 1
    while not 2**_k > 1000 * _k**2:
        _k += 1
    seuil_exact = _k
    print(f"sympy : somme des i² pour i de 0 à N − 1 = {formule} ; pour N = {config['N']} : {attendu}")
    print(f"Seuil exact : n = {seuil_exact} (2^{seuil_exact - 1} = {2**(seuil_exact - 1)} ≤ "
          f"{1000 * (seuil_exact - 1)**2}, 2^{seuil_exact} = {2**seuil_exact} > {1000 * seuil_exact**2})")
    return attendu, seuil_exact


@app.cell
def _(SMOKE, couts, mo, n, plt, sauvegarder_figure):
    # --- Figure : les mêmes courbes en échelle linéaire (gauche) et logarithmique (droite) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for _nom, _y in couts.items():
        ax1.plot(n, _y, label=_nom)
        ax2.semilogy(n, _y, label=_nom)
    ax1.set(xlabel="n", ylabel="nombre d’opérations", title="Échelle linéaire", ylim=(0, 1.2e6))
    ax2.set(xlabel="n", ylabel="nombre d’opérations", title="Échelle logarithmique")
    ax2.axvline(19, color="grey", ls=":", lw=1)
    ax1.legend()
    ax2.legend()
    chemins = sauvegarder_figure(fig, "couts-smoke" if SMOKE else "couts", mo.notebook_dir() / "figures")
    print(f"Figure : {chemins[0].name}")
    fig  # noqa: B018 (marimo affiche la dernière expression)
    return


@app.cell
def _(acceleration, attendu, de_a, de_b, de_c, mo, np, s_boucle, s_numpy, seuil, seuil_exact, t_boucle, t_numpy):
    # --- Bilan : seuils du README, vérifiés aussi en smoke ---
    echecs = []
    if not (s_boucle == s_numpy == attendu):
        echecs.append(f"sommes différentes : boucle {s_boucle}, numpy {s_numpy}, formule {attendu}")
    if acceleration < 5:
        echecs.append(f"numpy seulement {acceleration:.1f} fois plus rapide (attendu ≥ 5)")
    if not np.array_equal(de_a, de_b):
        echecs.append("même graine, tirages différents")
    if np.array_equal(de_a, de_c):
        echecs.append("graines différentes, tirages identiques")
    if not (seuil == seuil_exact == 19):
        echecs.append(f"seuil {seuil} (numpy), {seuil_exact} (exact), attendu 19")
    if echecs:
        raise SystemExit("Bilan : ÉCHEC : " + " ; ".join(echecs))
    print(f"Bilan : OK (somme {s_numpy}, numpy {acceleration:.0f} fois plus rapide : "
          f"{t_boucle * 1e3:.1f} ms contre {t_numpy * 1e3:.2f} ms ; graine reproductible ; seuil n = {seuil})")
    mo.md(f"**Bilan** : même somme par la boucle et par numpy, calculée environ {acceleration:.0f} fois "
          f"plus vite par numpy ; même graine, mêmes tirages ; $2^n$ dépasse $1\\,000\\,n^2$ à partir de "
          f"$n = {seuil}$.")
    return


if __name__ == "__main__":
    app.run()
