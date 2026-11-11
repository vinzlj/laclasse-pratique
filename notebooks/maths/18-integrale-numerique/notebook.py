# Notebook 18-integrale-numerique (chapitre 18, Intégration).
#
# Approcher une intégrale par des sommes : rectangles à gauche, à droite, au milieu et
# trapèzes, sur x², eˣ et 1/x ; erreur en fonction du nombre de pas (échelle log), qui
# baisse comme 1/n ou comme 1/n². Puis les gradients intégrés en une dimension : une somme
# de rectangles le long du segment qui va de la référence x′ à l’entrée x, et leur
# complétude IG = f(x) − f(x′).
#
#   make nb NUM=18              → éditeur marimo (bouton « Lancer le calcul »)
#   make nb NUM=18 MODE=run     → script complet
#   make nb NUM=18 MODE=smoke   → test rapide (moins de 30 s)

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
    # Intégrale numérique : des rectangles aux gradients intégrés

    **Chapitre 18, sections 2 et 8.** On découpe ${[a \,;\, b]}$ en $n$ bandes de largeur
    $h = \dfrac{b - a}{n}$ et on additionne des aires simples :

    - **rectangles à gauche** : hauteur $f(a + k h)$, pour $k = 0, \ldots, n - 1$ ;
    - **rectangles à droite** : hauteur $f(a + k h)$, pour $k = 1, \ldots, n$ ;
    - **rectangles au milieu** : hauteur $f\big(a + (k + 0{,}5) h\big)$ ;
    - **trapèzes** : la moyenne des sommes à gauche et à droite (on joint les points de la
      courbe par des segments).

    On compare l’erreur commise sur trois intégrales connues exactement :
    $\int_0^2 x^2\,\mathrm{d}x = \dfrac{8}{3}$, $\int_0^1 \mathrm{e}^{x}\,\mathrm{d}x = \mathrm{e} - 1$
    et $\int_1^2 \dfrac{\mathrm{d}x}{x} = \ln 2$.

    Puis on calcule des **gradients intégrés** en une dimension :
    $\mathrm{IG} = (x - x') \int_0^1 f'\big(x' + \alpha(x - x')\big)\,\mathrm{d}\alpha$,
    approché par $m$ rectangles en $\alpha$, et on vérifie que $\mathrm{IG}$ se rapproche de
    $f(x) - f(x')$ (complétude), y compris pour $f(x) = 1 - \mathrm{ReLU}(1 - x)$, dont la
    pente est nulle en $x = 2$.
    """)


@app.cell
def _(is_smoke, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)  # aucun tirage au hasard ici ; graine posée par convention
    style_commun()
    config = dict(
        puissances=range(1, 9) if SMOKE else range(1, 15),   # n = 2¹ … 2⁸ (smoke) ou 2¹⁴
        pas_ig=[4, 20, 100, 300] if SMOKE else [4, 10, 20, 50, 100, 300, 1000],
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

    def sommes(f, a, b, n):
        """Rectangles à gauche, à droite, au milieu, et trapèzes, avec n bandes."""
        h = (b - a) / n
        k = np.arange(n)
        gauche = h * f(a + k * h).sum()
        droite = h * f(a + (k + 1) * h).sum()
        milieu = h * f(a + (k + 0.5) * h).sum()
        return {"gauche": gauche, "droite": droite, "milieu": milieu, "trapèzes": (gauche + droite) / 2}

    cas = {
        "x² sur [0 ; 2]": (lambda x: x**2, 0.0, 2.0, 8 / 3),
        "eˣ sur [0 ; 1]": (np.exp, 0.0, 1.0, np.e - 1),
        "1/x sur [1 ; 2]": (lambda x: 1 / x, 1.0, 2.0, np.log(2)),
    }
    ns = np.array([2**p for p in config["puissances"]])
    erreurs = {}
    for _nom, (_f, _a, _b, _exacte) in cas.items():
        _lignes = [sommes(_f, _a, _b, int(_n)) for _n in ns]
        erreurs[_nom] = {m: np.array([abs(_l[m] - _exacte) for _l in _lignes]) for m in _lignes[0]}
    # Pente de l’erreur en log-log : −1 attendu (gauche, droite), −2 (milieu, trapèzes)
    pentes = {(_nom, m): float(np.polyfit(np.log10(ns), np.log10(e), 1)[0])
              for _nom, d in erreurs.items() for m, e in d.items()}
    # Valeurs du cours : x² sur [0 ; 2] avec n = 4 et n = 1 000 (rectangles à gauche)
    s4 = sommes(lambda x: x**2, 0.0, 2.0, 4)
    s1000 = sommes(lambda x: x**2, 0.0, 2.0, 1000)["gauche"]

    # --- Gradients intégrés en 1D : m rectangles à droite en α, x′ = 0 ---
    def gradient_integre(fprime, xref, x, m):
        alpha = np.arange(1, m + 1) / m
        return (x - xref) * fprime(xref + alpha * (x - xref)).mean()

    def relu_prime(z):
        return (z > 0).astype(float)

    ig_cas = {
        "x², x′ = 0, x = 3": (lambda t: 2 * t, 0.0, 3.0, 9.0),
        "1 − ReLU(1 − x), x′ = 0, x = 2": (lambda t: relu_prime(1 - t), 0.0, 2.0, 1.0),
    }
    ig = {_nom: np.array([gradient_integre(_fp, _xr, _x, m) for m in config["pas_ig"]])
          for _nom, (_fp, _xr, _x, _cible) in ig_cas.items()}
    duree = time.perf_counter() - t0
    print(f"Calcul : {duree:.3f} s")
    print(f"x² sur [0 ; 2], n = 4 : {s4} ; n = 1 000, gauche : {s1000:.6f}")
    for (_nom, m), p in pentes.items():
        print(f"  {_nom:16s} {m:9s} pente log-log {p:+.2f}")
    for _nom, v in ig.items():
        print(f"IG {_nom} : {dict(zip(config['pas_ig'], np.round(v, 4), strict=True))}")
    return duree, erreurs, ig, ig_cas, ns, pentes, s1000, s4


@app.cell
def _(sp):
    # --- Vérification exacte (sympy) : intégrales de référence et formules du cours ---
    _x, _n, _a = sp.symbols("x n alpha", positive=True)
    assert sp.integrate(_x**2, (_x, 0, 2)) == sp.Rational(8, 3)
    assert sp.integrate(sp.exp(_x), (_x, 0, 1)) == sp.E - 1
    assert sp.integrate(1 / _x, (_x, 1, 2)) == sp.log(2)
    # Rectangles à gauche pour x² sur [0 ; 2] : 8/3 − 4/n + 4/(3n²)
    _k = sp.symbols("k", integer=True, nonnegative=True)
    _sg = sp.summation((2 * _k / _n) ** 2 * 2 / _n, (_k, 0, _n - 1))
    assert sp.simplify(_sg - (sp.Rational(8, 3) - 4 / _n + sp.Rational(4, 3) / _n**2)) == 0
    # Complétude en 1D : (x − x′) ∫₀¹ f′(x′ + α(x − x′)) dα = f(x) − f(x′) pour f = x²
    _xr, _xx = sp.symbols("xr xx")
    _ig = (_xx - _xr) * sp.integrate(2 * (_xr + _a * (_xx - _xr)), (_a, 0, 1))
    assert sp.expand(_ig - (_xx**2 - _xr**2)) == 0
    print("sympy : 8/3, e − 1, ln 2, formule des rectangles et complétude vérifiées")
    return


@app.cell
def _(SMOKE, config, erreurs, ig, ig_cas, mo, ns, plt, sauvegarder_figure):
    # --- Figure : à gauche l’erreur selon n pour eˣ, à droite les gradients intégrés selon m ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for _m, _style in zip(erreurs["eˣ sur [0 ; 1]"], ["o-", "s-", "^-", "d-"], strict=True):
        ax1.loglog(ns, erreurs["eˣ sur [0 ; 1]"][_m], _style, ms=3, label=_m)
    ax1.set(xlabel="nombre de bandes n", ylabel="erreur", title="∫₀¹ eˣ dx : erreur selon n")
    ax1.legend(fontsize=9)
    for _nom, _v in ig.items():
        ax2.semilogx(config["pas_ig"], _v / ig_cas[_nom][3], "o-", ms=4, label=_nom)
    ax2.axhline(1, color="gray", lw=1, ls="--")
    ax2.set(xlabel="nombre de pas m", ylabel="IG / (f(x) − f(x′))", title="Complétude des gradients intégrés")
    ax2.legend(fontsize=8)
    chemins = sauvegarder_figure(fig, "integrale-numerique-smoke" if SMOKE else "integrale-numerique",
                                 mo.notebook_dir() / "figures")
    print(f"Figure : {chemins[0].name}")
    fig  # noqa: B018 (marimo affiche la dernière expression)
    return


@app.cell
def _(SMOKE, config, duree, ig, ig_cas, mo, pentes, s1000, s4):
    # --- Bilan : seuils chiffrés du README, vérifiés aussi en smoke ---
    tol = 0.15 if SMOKE else 0.1
    echecs = []
    for (_nom, _m), _p in pentes.items():
        _attendue = -1.0 if _m in ("gauche", "droite") else -2.0
        if abs(_p - _attendue) > tol:
            echecs.append(f"{_nom} {_m} : pente {_p:.2f} (attendue {_attendue:.0f} ± {tol})")
    if abs(s4["gauche"] - 1.75) > 1e-12 or abs(s4["droite"] - 3.75) > 1e-12 or abs(s4["milieu"] - 2.625) > 1e-12:
        echecs.append(f"sommes du cours pour n = 4 : {s4}")
    if abs(s1000 - 2.662668) > 1e-9:
        echecs.append(f"somme à gauche pour n = 1 000 : {s1000}")
    _m300 = config["pas_ig"].index(300)
    for _nom, _v in ig.items():
        _cible = ig_cas[_nom][3]
        _err = abs(_v[_m300] - _cible)
        if _err > 0.04:
            echecs.append(f"IG {_nom} avec 300 pas : écart {_err:.3g} (attendu ≤ 0,04)")
    if echecs:
        raise SystemExit("Bilan : ÉCHEC : " + " ; ".join(echecs))
    _ig300 = {k: float(v[_m300]) for k, v in ig.items()}
    print(f"Bilan : OK (pentes −1 et −2 à ±{tol}, sommes du cours 1,75 ; 3,75 ; 2,625 ; "
          f"2,662668, IG avec 300 pas : {_ig300}, {duree:.3f} s)")
    mo.md("**Bilan** : les rectangles à gauche et à droite ont une erreur en $1/n$ (pente $-1$), "
          "le milieu et les trapèzes en $1/n^2$ (pente $-2$). Avec 300 pas, les gradients "
          "intégrés retrouvent $f(x) - f(x')$ à moins de $0{,}04$ près, même là où la pente "
          "finale est nulle.")
    return


if __name__ == "__main__":
    app.run()
