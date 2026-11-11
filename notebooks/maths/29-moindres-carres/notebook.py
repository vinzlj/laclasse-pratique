# Notebook 29-moindres-carres (chapitre 29, Statistique à deux variables : écart-type,
# centrer-réduire, moindres carrés, corrélation).
#
# Moindres carrés sur les cinq pages du cours (poids et temps de chargement), puis sur un jeu
# simulé (graine 42) : formules a = cov(x, y)/V(x) et b = ȳ − a x̄ comparées à np.polyfit,
# résidus de somme nulle, R² égal à r², effet d’un seul point aberrant sur r, et ajustement
# exponentiel par ln.
#
#   make nb NUM=29              → éditeur marimo (bouton « Lancer le calcul »)
#   make nb NUM=29 MODE=run     → script complet
#   make nb NUM=29 MODE=smoke   → test rapide (moins de 30 s)

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
    # Moindres carrés, corrélation, point aberrant

    **Chapitre 29, sections 1, 5 à 8.** Pour une série de points $(x_i \,;\, y_i)$, la droite
    des moindres carrés $y = ax + b$ rend minimale la somme des carrés des résidus
    $e_i = y_i - (a x_i + b)$. Le cours établit

    $$
    a = \frac{\mathrm{cov}(x, y)}{V(x)}, \qquad b = \bar{y} - a\,\bar{x},
    \qquad r = \frac{\mathrm{cov}(x, y)}{\sigma_x\,\sigma_y},
    $$

    variance et covariance divisées par $n$. Quatre expériences :

    1. **Les cinq pages du cours** (1 à 5 Mo, 1,1 à 3,8 s) : sympy redonne exactement
       $a = \frac{17}{25}$ et $b = \frac{23}{50}$, et `np.polyfit` trouve la même droite.
    2. **Un nuage simulé** de $n$ points autour de $y = 0{,}7x + 0{,}5$ (bruit gaussien,
       graine 42) : formules contre `np.polyfit`, somme des résidus, $R^2 = r^2$.
    3. **Un point aberrant** : on ajoute au nuage de la démo du cours un point $M$ que l’on
       promène sur une grille ; on cherche où $r$ tombe près de 0.
    4. **Un nuage exponentiel** (utilisateurs d’un service, mois 0 à 5) : la droite ajustée
       sur $\ln y$ donne $y \approx 120{,}9 \times 2{,}01^x$, alors qu’une droite ajustée
       directement sur $y$ prédit un nombre d’utilisateurs négatif au mois 0.
    """)


@app.cell
def _(is_smoke, np, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        n_simule=50 if SMOKE else 2_000,   # points du nuage simulé
        pas_grille=1.0 if SMOKE else 0.5,  # pas de la grille du point aberrant
        a_vrai=0.7,
        b_vrai=0.5,
        bruit=0.4,                         # écart-type du bruit gaussien
    )
    rng = np.random.default_rng(SEED)
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SMOKE, config, rng


@app.cell
def _(np):
    def ajuster(x, y):
        """Droite des moindres carrés par les formules du cours (divisions par n)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        xb, yb = x.mean(), y.mean()
        cov = np.mean((x - xb) * (y - yb))
        vx = np.mean((x - xb) ** 2)
        vy = np.mean((y - yb) ** 2)
        a = cov / vx
        b = yb - a * xb
        r = cov / np.sqrt(vx * vy)
        e = y - (a * x + b)
        r2 = 1 - np.sum(e**2) / np.sum((y - yb) ** 2)
        return dict(a=a, b=b, r=r, R2=r2, residus=e, cov=cov, vx=vx)

    return (ajuster,)


@app.cell
def _(sp):
    # --- Vérification exacte (sympy) sur les cinq pages du cours ---
    xs = [sp.Integer(k) for k in range(1, 6)]
    ys = [sp.Rational(v) for v in ("11/10", "19/10", "12/5", "33/10", "19/5")]
    n = len(xs)
    xb = sum(xs) / n
    yb = sum(ys) / n
    cov = sum((x - xb) * (y - yb) for x, y in zip(xs, ys, strict=True)) / n
    vx = sum((x - xb) ** 2 for x in xs) / n
    vy = sum((y - yb) ** 2 for y in ys) / n
    a_ex = cov / vx
    b_ex = yb - a_ex * xb
    sse = sum((y - a_ex * x - b_ex) ** 2 for x, y in zip(xs, ys, strict=True))
    sst = n * vy
    assert (xb, yb) == (3, sp.Rational(5, 2)), (xb, yb)
    assert (cov, vx) == (sp.Rational(34, 25), 2), (cov, vx)
    assert (a_ex, b_ex) == (sp.Rational(17, 25), sp.Rational(23, 50)), (a_ex, b_ex)
    assert sp.simplify(1 - sse / sst - cov**2 / (vx * vy)) == 0  # R² = r² exactement
    assert sse == sp.Rational(9, 250)
    print(f"sympy : G({xb} ; {yb}), cov = {cov}, V(x) = {vx}, a = {a_ex}, b = {b_ex}, "
          f"somme des carrés des résidus = {sse}, R² = r² = {sp.nsimplify(1 - sse / sst)}")
    return a_ex, b_ex


@app.cell
def _(mo):
    # Le bouton empêche le calcul de se relancer à chaque modification d’une cellule amont.
    bouton = mo.ui.run_button(label="Lancer le calcul")
    bouton  # noqa: B018 (marimo affiche la dernière expression)
    return (bouton,)


@app.cell
def _(a_ex, ajuster, b_ex, bouton, config, mo, np, rng, time):
    # En script (make nb MODE=run|smoke), personne ne clique : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer le calcul ».*"))

    t0 = time.perf_counter()
    # 1. Les cinq pages du cours
    pages_x = np.arange(1, 6, dtype=float)
    pages_y = np.array([1.1, 1.9, 2.4, 3.3, 3.8])
    pages = ajuster(pages_x, pages_y)
    pf_pages = np.polyfit(pages_x, pages_y, 1)
    ecart_pages = max(abs(pages["a"] - float(a_ex)), abs(pages["b"] - float(b_ex)),
                      abs(pf_pages[0] - pages["a"]), abs(pf_pages[1] - pages["b"]))

    # 2. Nuage simulé
    xs_sim = rng.uniform(0, 10, config["n_simule"])
    ys_sim = config["a_vrai"] * xs_sim + config["b_vrai"] + rng.normal(0, config["bruit"], config["n_simule"])
    sim = ajuster(xs_sim, ys_sim)
    pf_sim = np.polyfit(xs_sim, ys_sim, 1)
    ecart_sim = max(abs(pf_sim[0] - sim["a"]), abs(pf_sim[1] - sim["b"]))

    # 3. Point aberrant : nuage de la démo NuageAberrant, M sur une grille de [0 ; 10]²
    nuage = np.array([[1, 1.5], [2, 2], [3, 3.5], [4, 4], [5, 5.5], [6, 6]])
    r_nuage = ajuster(nuage[:, 0], nuage[:, 1])["r"]
    grille = np.arange(0, 10 + 1e-9, config["pas_grille"])
    r_grille = np.empty((len(grille), len(grille)))
    for i, ym in enumerate(grille):
        for j, xm in enumerate(grille):
            pts = np.vstack([nuage, [xm, ym]])
            r_grille[i, j] = ajuster(pts[:, 0], pts[:, 1])["r"]
    pres_de_zero = int(np.sum(np.abs(r_grille) <= 0.1))
    r_coin = ajuster(np.r_[nuage[:, 0], 0.0], np.r_[nuage[:, 1], 9.0])["r"]

    # 4. Nuage exponentiel : ln y contre y
    mois = np.arange(6, dtype=float)
    users = np.array([120, 250, 480, 1000, 1900, 4100], dtype=float)
    ajust_ln = ajuster(mois, np.log(users))
    ajust_lin = ajuster(mois, users)
    k_exp, q_exp = float(np.exp(ajust_ln["b"])), float(np.exp(ajust_ln["a"]))
    duree = time.perf_counter() - t0
    print(f"Pages : a = {pages['a']:.4f}, b = {pages['b']:.4f}, r = {pages['r']:.4f}, "
          f"R² = {pages['R2']:.4f}, écart à sympy et polyfit {ecart_pages:.1e}")
    print(f"Simulé ({config['n_simule']} points) : a = {sim['a']:.4f}, b = {sim['b']:.4f}, "
          f"r = {sim['r']:.4f}, |R² − r²| = {abs(sim['R2'] - sim['r'] ** 2):.1e}, "
          f"somme des résidus {np.sum(sim['residus']):.1e}")
    print(f"Point aberrant : r du nuage seul {r_nuage:.4f}, avec M(0 ; 9) {r_coin:.4f}, "
          f"{pres_de_zero} positions de la grille avec |r| ≤ 0,1")
    print(f"Exponentiel : ln y ≈ {ajust_ln['a']:.4f} x + {ajust_ln['b']:.4f}, y ≈ {k_exp:.1f} × {q_exp:.3f}^x, "
          f"R²(ln) = {ajust_ln['R2']:.4f} ; droite sur y : {ajust_lin['a']:.1f} x {ajust_lin['b']:+.1f}, "
          f"R² = {ajust_lin['R2']:.3f}")
    return (ajust_lin, ajust_ln, duree, ecart_pages, ecart_sim, grille, k_exp, mois, pages,
            pages_x, pages_y, pres_de_zero, q_exp, r_coin, r_grille, r_nuage, sim, users,
            xs_sim, ys_sim)


@app.cell
def _(SMOKE, ajust_lin, grille, k_exp, mo, mois, np, pages, pages_x, pages_y, plt, q_exp,
      r_grille, sauvegarder_figure, users):
    # --- Figure, sauvegardée dans figures/ (suffixe -smoke : n’écrase pas celle du run complet) ---
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    ax = axs[0]
    ax.plot(pages_x, pages_y, "o", label="pages mesurées")
    xx = np.linspace(0, 8, 50)
    ax.plot(xx, pages["a"] * xx + pages["b"], "-", label=f"y = {pages['a']:.2f}x + {pages['b']:.2f}")
    for xi, yi, ei in zip(pages_x, pages_y, pages["residus"], strict=True):
        ax.plot([xi, xi], [yi, yi - ei], "k:", lw=1)
    ax.set(xlabel="poids (Mo)", ylabel="temps (s)", title="Droite des moindres carrés")
    ax.legend()

    ax = axs[1]
    im = ax.imshow(np.abs(r_grille), origin="lower", extent=(grille[0], grille[-1], grille[0], grille[-1]),
                   cmap="viridis", vmin=0, vmax=1)
    ax.plot([1, 2, 3, 4, 5, 6], [1.5, 2, 3.5, 4, 5.5, 6], "wo", ms=4)
    ax.set(xlabel="x de M", ylabel="y de M", title="|r| selon la position de M")
    fig.colorbar(im, ax=ax)

    ax = axs[2]
    ax.plot(mois, users, "o", label="utilisateurs")
    mm = np.linspace(0, 5, 50)
    ax.plot(mm, k_exp * q_exp**mm, "-", label=f"{k_exp:.0f} × {q_exp:.2f}^x (ajustée sur ln y)")
    ax.plot(mm, ajust_lin["a"] * mm + ajust_lin["b"], "--", label="droite ajustée sur y")
    ax.set(xlabel="mois", ylabel="utilisateurs", title="Ajustement exponentiel")
    ax.legend()
    chemins = sauvegarder_figure(fig, "moindres-carres-smoke" if SMOKE else "moindres-carres",
                                 mo.notebook_dir() / "figures")
    print(f"Figure : {chemins[0].name}")
    fig  # noqa: B018 (marimo affiche la dernière expression)
    return


@app.cell
def _(SMOKE, ajust_lin, ajust_ln, duree, ecart_pages, ecart_sim, mo, np, pages, pres_de_zero,
      q_exp, r_coin, r_nuage, sim):
    # --- Bilan : seuils chiffrés du README, vérifiés aussi en smoke ---
    tol_a = 0.1 if SMOKE else 0.02
    echecs = []
    if ecart_pages > 1e-12:
        echecs.append(f"pages : écart aux valeurs exactes {ecart_pages:.1e}")
    if abs(pages["r"] - 0.9961) > 1e-4 or abs(pages["R2"] - pages["r"] ** 2) > 1e-12:
        echecs.append(f"pages : r = {pages['r']:.5f}, R² = {pages['R2']:.5f}")
    if ecart_sim > 1e-9:
        echecs.append(f"simulé : formules et polyfit diffèrent de {ecart_sim:.1e}")
    if abs(sim["a"] - 0.7) > tol_a:
        echecs.append(f"simulé : a = {sim['a']:.4f} (attendu 0,7 ± {tol_a})")
    if abs(np.sum(sim["residus"])) > 1e-8 or abs(sim["R2"] - sim["r"] ** 2) > 1e-12:
        echecs.append("simulé : résidus de somme non nulle ou R² ≠ r²")
    if not (r_nuage > 0.98 and abs(r_coin) <= 0.1 and pres_de_zero >= 1):
        echecs.append(f"point aberrant : r = {r_nuage:.4f}, {r_coin:.4f}, {pres_de_zero} positions")
    if not (abs(ajust_ln["a"] - 0.6993) < 1e-3 and ajust_ln["R2"] > 0.999 and ajust_lin["b"] < 0):
        echecs.append(f"exponentiel : a = {ajust_ln['a']:.4f}, R² = {ajust_ln['R2']:.4f}")
    if echecs:
        raise SystemExit("Bilan : ÉCHEC : " + " ; ".join(echecs))
    print(f"Bilan : OK (pages a = 0,68, b = 0,46, r = {pages['r']:.4f} ; simulé a = {sim['a']:.4f}, "
          f"r = {sim['r']:.4f} ; r passe de {r_nuage:.3f} à {r_coin:.3f} avec M(0 ; 9) ; "
          f"facteur mensuel {q_exp:.3f} ; {duree:.2f} s)")
    def fr(v, d):
        return f"{v:.{d}f}".replace(".", ",").replace("-", "−")

    mo.md(f"**Bilan** : droite des pages $y = 0{{,}}68x + 0{{,}}46$, $r \\approx$ {fr(pages['r'], 4)} ; "
          f"un seul point en $(0 \\,;\\, 9)$ fait passer $r$ de {fr(r_nuage, 3)} à {fr(r_coin, 3)} ; "
          f"le nombre d’utilisateurs est multiplié par environ {fr(q_exp, 2)} chaque mois.")
    return


if __name__ == "__main__":
    app.run()
