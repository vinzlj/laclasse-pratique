# Notebook 16-softmax-temperature (chapitre 16, Les fonctions de l’IA).
#
# Softmax naïve contre softmax stable sur de grands scores (renvoi au chapitre 48),
# échantillonnage d’un « mot suivant » selon la température T (fréquences observées contre
# probabilités), et entropie de la loi en fonction de T, de 0 bit (T → 0) à log₂ 3 (T → +∞).
#
#   make nb NUM=16              → éditeur marimo (bouton « Lancer le calcul »)
#   make nb NUM=16 MODE=run     → script complet
#   make nb NUM=16 MODE=smoke   → test rapide (moins de 30 s)

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
    # Softmax et température : des scores aux probabilités, puis au tirage

    **Chapitre 16, section 5.** Un modèle de langage donne un score à chaque mot possible ;
    la softmax $\operatorname{softmax}(z / T)_i = \dfrac{\mathrm{e}^{z_i / T}}{\sum_j \mathrm{e}^{z_j / T}}$
    en fait des probabilités, et le mot suivant est **tiré au hasard** selon ces probabilités.
    Trois expériences, avec les scores $(2 ; 1 ; 0)$ de « canapé », « lit », « toit » :

    1. **Naïve contre stable.** Avec les scores $(1000 ; 999 ; 998)$, la formule directe
       déborde ($\mathrm{e}^{1000}$ n’est pas représentable en flottant) et donne `nan` ;
       en retranchant d’abord le plus grand score, on retrouve les probabilités de $(2 ; 1 ; 0)$.
    2. **Tirer selon $T$.** Pour plusieurs températures, on tire des milliers de mots et on
       compare les fréquences observées aux probabilités de la softmax.
    3. **Entropie selon $T$.** L’entropie $H = -\sum_i p_i \log_2 p_i$ (chapitre 46) mesure
       l’étalement de la loi : 0 bit quand tout est sur un mot, $\log_2 3 \approx 1{,}585$ bit
       pour trois mots équiprobables. On vérifie qu’elle croît avec $T$.
    """)


@app.cell
def _(is_smoke, np, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        scores=[2.0, 1.0, 0.0],                 # canapé, lit, toit
        mots=["canapé", "lit", "toit"],
        temperatures=[0.25, 0.5, 1.0, 2.0, 4.0],
        n_tirages=20_000 if SMOKE else 400_000,  # tirages par température
        n_T=60 if SMOKE else 400,                # points de la courbe d’entropie
    )
    rng = np.random.default_rng(SEED)
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SMOKE, config, rng


@app.cell
def _(np):
    def softmax_naive(z, T=1.0):
        _e = np.exp(np.asarray(z, dtype=float) / T)
        return _e / _e.sum()

    def softmax_stable(z, T=1.0):
        _u = np.asarray(z, dtype=float) / T
        _e = np.exp(_u - _u.max())   # le plus grand exposant vaut 0 : rien ne déborde
        return _e / _e.sum()

    def entropie_bits(p):
        p = np.asarray(p, dtype=float)
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    return entropie_bits, softmax_naive, softmax_stable


@app.cell
def _(np, softmax_naive, softmax_stable):
    # --- Expérience 1 : naïve contre stable. np.errstate masque l’avertissement de débordement :
    # on veut voir le résultat (nan), pas une trace d’erreur.
    grands = [1000.0, 999.0, 998.0]
    with np.errstate(over="ignore", invalid="ignore"):
        p_naive = softmax_naive(grands)
    p_stable = softmax_stable(grands)
    p_reference = softmax_naive([2.0, 1.0, 0.0])
    print(f"naïve  (1000 ; 999 ; 998) : {p_naive}")
    print(f"stable (1000 ; 999 ; 998) : {np.round(p_stable, 4)}")
    print(f"naïve  (2 ; 1 ; 0)        : {np.round(p_reference, 4)}")
    return p_naive, p_reference, p_stable


@app.cell
def _(mo):
    # Le bouton empêche le calcul de se relancer à chaque modification d’une cellule amont.
    bouton = mo.ui.run_button(label="Lancer le calcul")
    bouton  # noqa: B018 (marimo affiche la dernière expression)
    return (bouton,)


@app.cell
def _(bouton, config, entropie_bits, mo, np, rng, softmax_stable, time):
    # En script (make nb MODE=run|smoke), personne ne clique : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer le calcul ».*"))

    t0 = time.perf_counter()
    # --- Expérience 2 : tirages selon la température ---
    resultats = []
    for _T in config["temperatures"]:
        _p = softmax_stable(config["scores"], _T)
        _tirages = rng.choice(len(_p), size=config["n_tirages"], p=_p)
        _freq = np.bincount(_tirages, minlength=len(_p)) / config["n_tirages"]
        resultats.append(dict(T=_T, p=_p, freq=_freq, ecart=float(np.abs(_freq - _p).max())))
    # --- Expérience 3 : entropie selon T (échelle logarithmique de 0,05 à 50) ---
    Ts = np.logspace(np.log10(0.05), np.log10(50), config["n_T"])
    H = np.array([entropie_bits(softmax_stable(config["scores"], _T)) for _T in Ts])
    duree = time.perf_counter() - t0
    for _r in resultats:
        print(f"T = {_r['T']:<5} probabilités {np.round(_r['p'], 4)}  fréquences {np.round(_r['freq'], 4)}  "
              f"écart max {_r['ecart']:.4f}")
    print(f"Entropie : {H[0]:.4f} bit pour T = {Ts[0]:.2f}, {H[-1]:.4f} bit pour T = {Ts[-1]:.0f} ; "
          f"calcul {duree:.2f} s")
    return H, Ts, duree, resultats


@app.cell
def _(sp):
    # --- Vérification exacte (sympy) : invariance par translation et limite T → +∞ ---
    _z1, _z2, _z3, _c, _T = sp.symbols("z1 z2 z3 c T", real=True)

    def _sm(_zs, _i):
        return sp.exp(_zs[_i]) / sum(sp.exp(_v) for _v in _zs)

    _zs = [_z1, _z2, _z3]
    for _i in range(3):
        assert sp.simplify(_sm([_v + _c for _v in _zs], _i) - _sm(_zs, _i)) == 0
    _p0_T = _sm([sp.Integer(2) / _T, sp.Integer(1) / _T, sp.Integer(0)], 0)
    assert sp.limit(_p0_T, _T, sp.oo) == sp.Rational(1, 3)
    assert sp.limit(_p0_T, _T, 0, "+") == 1
    # Deux classes : softmax(z1, z2)_1 = σ(z1 − z2)
    assert sp.simplify(_sm([_z1, _z2], 0) - 1 / (1 + sp.exp(-(_z1 - _z2)))) == 0
    print("sympy : softmax(z + c) = softmax(z) ; p₀ → 1/3 quand T → +∞, → 1 quand T → 0 ; deux classes = σ(z₁ − z₂)")
    return


@app.cell
def _(H, SMOKE, Ts, config, mo, np, plt, resultats, sauvegarder_figure):
    # --- Figure (suffixe -smoke : n’écrase pas celle du run complet) ---
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    _largeur = 0.8 / len(resultats)
    _x = np.arange(len(config["mots"]))
    for _k, _r in enumerate(resultats):
        _etiquette = "T = " + f"{_r['T']:g}".replace(".", ",")
        _ax1.bar(_x + (_k - (len(resultats) - 1) / 2) * _largeur, _r["freq"], _largeur, label=_etiquette)
        _ax1.scatter(_x + (_k - (len(resultats) - 1) / 2) * _largeur, _r["p"], color="black", s=8, zorder=3)
    _ax1.set_xticks(_x, config["mots"])
    _ax1.set(ylabel="fréquence observée (points : probabilité)", title="Mot tiré selon la température")
    _ax1.legend(fontsize=9)
    _ax2.semilogx(Ts, H)
    _ax2.axhline(np.log2(3), ls="--", color="gray", lw=1)
    _ax2.text(Ts[0], np.log2(3) - 0.12, "log₂ 3 ≈ 1,585 bit", fontsize=9)
    _ax2.set(xlabel="température T (échelle logarithmique)", ylabel="entropie (bits)", title="Entropie selon T")
    _chemins = sauvegarder_figure(_fig, "softmax-temperature-smoke" if SMOKE else "softmax-temperature",
                                 mo.notebook_dir() / "figures")
    print(f"Figure : {_chemins[0].name}")
    _fig  # noqa: B018 (marimo affiche la dernière expression)
    return


@app.cell
def _(H, SMOKE, Ts, duree, mo, np, p_naive, p_reference, p_stable, resultats):
    # --- Bilan : seuils du README, vérifiés aussi en smoke (tolérance plus large : moins de tirages)
    _tol = 0.02 if SMOKE else 0.005
    _echecs = []
    if not np.isnan(p_naive).all():
        _echecs.append(f"softmax naïve de (1000 ; 999 ; 998) : {p_naive} (attendu : nan)")
    if np.abs(p_stable - p_reference).max() > 1e-12:
        _echecs.append("softmax stable de (1000 ; 999 ; 998) différente de celle de (2 ; 1 ; 0)")
    _pire = max(_r["ecart"] for _r in resultats)
    if _pire > _tol:
        _echecs.append(f"écart fréquence-probabilité {_pire:.4f} (attendu ≤ {_tol})")
    _p_T05 = next(_r["p"][0] for _r in resultats if _r["T"] == 0.5)
    if abs(_p_T05 - 0.8668) > 1e-4:
        _echecs.append(f"p(canapé) pour T = 0,5 : {_p_T05:.4f} (attendu 0,8668)")
    if np.any(np.diff(H) < -1e-12):
        _echecs.append("l’entropie n’est pas croissante en T")
    if H[0] > 0.01 or abs(H[-1] - np.log2(3)) > 0.01:
        _echecs.append(f"entropie aux bords {H[0]:.4f} et {H[-1]:.4f} (attendu ≈ 0 et ≈ 1,585)")
    if _echecs:
        raise SystemExit("Bilan : ÉCHEC : " + " ; ".join(_echecs))
    print(f"Bilan : OK (naïve nan, stable = (0,6652 ; 0,2447 ; 0,0900) ; écart max des tirages {_pire:.4f} ; "
          f"p(canapé) = {_p_T05:.4f} pour T = 0,5 ; entropie de {H[0]:.4f} à {H[-1]:.4f} bit, croissante ; "
          f"{duree:.2f} s)")
    mo.md(f"**Bilan** : softmax naïve `nan`, stable égale à celle de $(2 ; 1 ; 0)$ ; écart maximal "
          f"fréquence-probabilité {_pire:.4f} ; entropie croissante de {H[0]:.3f} à {H[-1]:.3f} bit "
          f"pour $T$ de {Ts[0]:.2f} à {Ts[-1]:.0f}.")
    return


if __name__ == "__main__":
    app.run()
