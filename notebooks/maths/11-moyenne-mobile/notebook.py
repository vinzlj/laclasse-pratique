# Notebook 11-moyenne-mobile (chapitre 11, Suites).
#
# La moyenne mobile exponentielle m(t) = β m(t−1) + (1 − β) x(t), partie de m(0) = 0 :
# lissage d’un signal bruité pour trois valeurs de β, formule fermée c(1 − βᵗ) sur un signal
# constant, correction de biais m(t) / (1 − βᵗ), et demi-vie lue dans un tableau.
#
#   make nb NUM=11              → éditeur marimo (bouton « Lancer le calcul »)
#   make nb NUM=11 MODE=run     → script complet
#   make nb NUM=11 MODE=smoke   → test rapide (moins de 30 s)

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
    # Moyenne mobile exponentielle : une suite arithmético-géométrique

    **Chapitre 11, section 6.** Un tableau de bord lisse un temps de réponse bruité, autour
    de 200 ms, en gardant une seule valeur mise à jour à chaque mesure $x_t$ :

    $$m_t = \beta\, m_{t-1} + (1 - \beta)\, x_t, \qquad m_0 = 0.$$

    Trois questions :

    1. **Lisser ou suivre ?** Plus $\beta$ est proche de 1, plus la courbe est lisse, mais plus
       elle réagit lentement à un changement de niveau.
    2. **Le défaut de départ.** Sur un signal constant $c$, la suite vaut exactement
       $c\,(1 - \beta^t)$ : bien trop bas au début. La correction de biais
       $\hat m_t = m_t / (1 - \beta^t)$, celle d’Adam, la remet au niveau $c$.
    3. **La demi-vie.** En combien de pas la moyenne comble-t-elle la moitié d’un écart ?
       C’est le premier $t$ tel que $\beta^t \le 0{,}5$, lu ici dans un tableau (la formule
       avec le logarithme arrive au chapitre 15).
    """)


@app.cell
def _(is_smoke, np, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        n_pas=120 if SMOKE else 400,      # nombre de mesures
        n_tirages=50 if SMOKE else 2_000,  # signaux bruités pour l’erreur moyenne
        niveau=200.0,                      # niveau du signal (ms)
        ecart_type=40.0,                   # bruit gaussien (ms)
        betas=(0.5, 0.9, 0.99),
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

    def ema(x, beta):
        """Moyenne mobile exponentielle partie de 0 : une boucle, une seule variable."""
        m = np.zeros(len(x) + 1)
        for t in range(1, len(x) + 1):
            m[t] = beta * m[t - 1] + (1 - beta) * x[t - 1]
        return m  # m[0] = 0, m[t] après la mesure t

    t0 = time.perf_counter()
    n, c = config["n_pas"], config["niveau"]
    t = np.arange(n + 1)
    # 1. Signal constant : la boucle contre la formule fermée c(1 − βᵗ)
    ecart_formule = max(float(np.max(np.abs(ema(np.full(n, c), _b) - c * (1 - _b**t)))) for _b in config["betas"])
    ecart_corrige = max(
        float(np.max(np.abs(ema(np.full(n, c), _b)[1:] / (1 - _b ** t[1:]) - c))) for _b in config["betas"]
    )
    # 2. Signal bruité : erreur moyenne des 10 premiers pas, sans et avec correction (β = 0,9)
    bruit = rng.normal(c, config["ecart_type"], size=(config["n_tirages"], n))
    erreurs_brutes, erreurs_corrigees = [], []
    for _x in bruit:
        _m = ema(_x, 0.9)[1:11]
        erreurs_brutes.append(np.mean(np.abs(_m - c)))
        erreurs_corrigees.append(np.mean(np.abs(_m / (1 - 0.9 ** np.arange(1, 11)) - c)))
    err_brute, err_corrigee = float(np.mean(erreurs_brutes)), float(np.mean(erreurs_corrigees))
    # 3. Un exemple de signal bruité, lissé avec les trois β
    exemple = bruit[0]
    lisses = {_b: ema(exemple, _b) for _b in config["betas"]}
    # 4. Demi-vie : premier t tel que βᵗ ≤ 0,5 (tableau)
    def demi_vie(beta):
        k, p = 0, 1.0
        while p > 0.5:
            p *= beta
            k += 1
        return k

    demi_vies = {_b: demi_vie(_b) for _b in config["betas"]}
    demi_vies[0.996] = demi_vie(0.996)
    duree = time.perf_counter() - t0
    print(f"Calcul : {duree:.2f} s")
    print(f"Signal constant : écart boucle/formule {ecart_formule:.1e}, écart corrigé/niveau {ecart_corrige:.1e}")
    print(f"β = 0,9, 10 premiers pas : erreur moyenne {err_brute:.1f} ms sans correction, {err_corrigee:.1f} ms avec")
    print("Demi-vies : " + ", ".join(f"β = {_b} : {_k} pas" for _b, _k in demi_vies.items()))
    print("Tableau β = 0,9 : " + " ; ".join(f"0,9^{_k} = {0.9**_k:.3f}" for _k in range(5, 9)))
    return demi_vies, duree, ecart_corrige, ecart_formule, err_brute, err_corrigee, exemple, lisses, t


@app.cell
def _(sp):
    # --- Vérification exacte (sympy) : la récurrence et la formule fermée coïncident ---
    _b, _c = sp.symbols("beta c", positive=True)
    _m = sp.Integer(0)
    for _t in range(1, 8):
        _m = sp.expand(_b * _m + (1 - _b) * _c)
        assert sp.simplify(_m - _c * (1 - _b**_t)) == 0, _t
        assert sp.simplify(_m / (1 - _b**_t) - _c) == 0, _t
    # Point fixe de m ↦ βm + (1 − β)c : c lui-même
    _l = sp.symbols("l")
    assert sp.solve(sp.Eq(_l, _b * _l + (1 - _b) * _c), _l) == [_c]
    print("sympy : mₜ = c(1 − βᵗ) et m̂ₜ = c pour t = 1 à 7 ; point fixe c")
    return


@app.cell
def _(SMOKE, config, exemple, lisses, mo, plt, sauvegarder_figure, t):
    # --- Figure : le signal bruité et ses trois moyennes mobiles ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t[1:], exemple, ".", ms=3, color="0.6", label="mesures")
    for _b, _m in lisses.items():
        ax.plot(t, _m, lw=2, label=f"β = {str(_b).replace('.', ',')}")
    ax.axhline(config["niveau"], color="k", lw=1, ls="--")
    ax.set(xlabel="pas t", ylabel="temps de réponse (ms)", title="Moyenne mobile exponentielle partie de 0")
    ax.legend(loc="lower right")
    _nom = "moyenne-mobile-smoke" if SMOKE else "moyenne-mobile"
    chemins = sauvegarder_figure(fig, _nom, mo.notebook_dir() / "figures")
    print(f"Figure : {chemins[0].name}")
    fig  # noqa: B018 (marimo affiche la dernière expression)
    return


@app.cell
def _(demi_vies, duree, ecart_corrige, ecart_formule, err_brute, err_corrigee, mo):
    # --- Bilan : seuils chiffrés du README, vérifiés aussi en smoke ---
    echecs = []
    if ecart_formule > 1e-9:
        echecs.append(f"écart boucle/formule {ecart_formule:.1e} (attendu ≤ 1e-9)")
    if ecart_corrige > 1e-9:
        echecs.append(f"écart corrigé/niveau {ecart_corrige:.1e} (attendu ≤ 1e-9)")
    if not err_corrigee < err_brute / 3:
        echecs.append(f"correction insuffisante : {err_corrigee:.1f} contre {err_brute:.1f} ms")
    attendues = {0.5: 1, 0.9: 7, 0.99: 69, 0.996: 173}
    for _b, _k in attendues.items():
        if demi_vies[_b] != _k:
            echecs.append(f"demi-vie pour β = {_b} : {demi_vies[_b]} (attendue {_k})")
    if echecs:
        raise SystemExit("Bilan : ÉCHEC : " + " ; ".join(echecs))
    print(f"Bilan : OK (formule fermée exacte, erreur des 10 premiers pas {err_brute:.1f} → {err_corrigee:.1f} ms "
          f"avec la correction, demi-vies 1, 7, 69 et 173 pas, {duree:.2f} s)")
    _fr = lambda v: f"{v:.1f}".replace(".", ",")  # noqa: E731
    mo.md(f"**Bilan** : sur les 10 premiers pas, l’erreur moyenne passe de {_fr(err_brute)} ms à "
          f"{_fr(err_corrigee)} ms avec la correction de biais. Demi-vies : 1 pas pour β = 0,5, "
          "7 pour 0,9, 69 pour 0,99, 173 pour 0,996.")
    return


if __name__ == "__main__":
    app.run()
