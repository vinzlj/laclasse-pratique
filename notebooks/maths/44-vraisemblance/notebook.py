# Notebook 44-vraisemblance (chapitre 44, Inférence : Bayes, vraisemblance, MAP, estimation).
#
# Tracer la log-vraisemblance d’une suite de Bernoulli et retrouver k/n numériquement ; mettre
# à jour une loi Bêta au fil des observations (densité écrite avec math.lgamma) ; mesurer le
# biais de la variance « divisée par n » ; le taux de couverture de x̄ ± 1,96 σ/√n ; un
# bootstrap de la moyenne et de la médiane ; la borne de Hoeffding face aux fréquences
# observées ; l’effondrement d’une gaussienne réestimée génération après génération.
#
#   make nb NUM=44              → éditeur marimo (bouton « Lancer le calcul »)
#   make nb NUM=44 MODE=run     → script complet
#   make nb NUM=44 MODE=smoke   → test rapide (moins de 30 s)

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import sympy as sp
    from math_commun import is_smoke, sauvegarder_figure, seed_everything, style_commun

    return is_smoke, math, mo, np, plt, sauvegarder_figure, seed_everything, sp, style_commun, time


@app.cell
def _(mo):
    mo.md(r"""
    # Vraisemblance, Bayes et estimateurs, par la simulation

    **Chapitre 44, sections 1 à 7.** Sept vérifications :

    1. **Maximum de vraisemblance** : sur 200 lancers d’une pièce de paramètre $p = 0{,}3$,
       la log-vraisemblance $\ell(p) = k \ln p + (n - k) \ln(1 - p)$, calculée sur une grille,
       culmine en $\frac{k}{n}$ (sections 1 et 2).
    2. **Conjugaison Bêta-Bernoulli** : partir de $\text{Bêta}(2 \,;\, 18)$ (taux habituel 10 %)
       et ajouter les clics un par un ; la moyenne a posteriori $\frac{a + k}{a + b + n}$ rejoint
       la fréquence observée (section 3). La densité est écrite avec `math.lgamma`, sans scipy.
    3. **Biais de $\hat\sigma^2$** : pour $n = 5$, la moyenne de $\frac{1}{n}\sum (x_i - \bar x)^2$
       sur de nombreux échantillons vaut environ $\frac{4}{5}\sigma^2$ (section 5).
    4. **Intervalle de confiance** : $\bar x \pm 1{,}96\,\frac{\sigma}{\sqrt n}$ contient $\mu$
       dans environ 95 % des échantillons (section 6).
    5. **Bootstrap** de la moyenne et de la médiane de dix temps de réponse (section 6).
    6. **Hoeffding** : la fréquence des écarts $|\hat R - R| > \gamma$ reste sous
       $2e^{-2N\gamma^2}$ (section 7).
    7. **Effondrement** d’une gaussienne réestimée par maximum de vraisemblance, génération
       après génération (section 5, et parcours IA, chapitre 37).
    """)


@app.cell
def _(is_smoke, np, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        n_echantillons=20_000 if SMOKE else 200_000,  # échantillons pour le biais et la couverture
        B=1_000 if SMOKE else 4_000,                   # rééchantillonnages du bootstrap
        repetitions=4_000 if SMOKE else 40_000,        # répétitions pour Hoeffding
        chaines=100 if SMOKE else 400,                 # chaînes de l’effondrement
        generations=300,                               # générations de l’effondrement
        tol_biais=0.01 if SMOKE else 0.004,            # écart toléré sur le rapport 4/5
        tol_couv=0.01 if SMOKE else 0.004,             # écart toléré sur la couverture
    )
    rng = np.random.default_rng(SEED)
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SMOKE, config, rng


@app.cell
def _(mo):
    bouton = mo.ui.run_button(label="Lancer le calcul")
    bouton  # noqa: B018 (marimo affiche la dernière expression)
    return (bouton,)


@app.cell
def _(bouton, config, math, mo, np, rng, time):
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer le calcul »."))
    t0 = time.perf_counter()

    # 1. Maximum de vraisemblance d’une Bernoulli, par une grille
    lancers = rng.random(200) < 0.3
    k, n = int(lancers.sum()), lancers.size
    grille = np.linspace(0.001, 0.999, 9_981)
    ell = k * np.log(grille) + (n - k) * np.log(1 - grille)
    p_grille = float(grille[np.argmax(ell)])

    # 2. Conjugaison Bêta-Bernoulli : a priori Bêta(2 ; 18), puis des clics de taux 0,3
    def densite_beta(p, a, b):
        log_c = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        return np.exp(log_c + (a - 1) * np.log(p) + (b - 1) * np.log(1 - p))

    clics = rng.random(500) < 0.3
    a0, b0 = 2, 18
    cumul = np.cumsum(clics)
    nb = np.arange(1, clics.size + 1)
    moyennes_post = (a0 + cumul) / (a0 + b0 + nb)
    aire_beta = float(np.sum(densite_beta(grille, a0 + cumul[-1], b0 + nb[-1] - cumul[-1])) * (grille[1] - grille[0]))

    # 3. Biais de la variance divisée par n (n = 5, σ² = 1)
    ech = rng.standard_normal((config["n_echantillons"], 5))
    rapport_biais = float(np.var(ech, axis=1).mean())          # ddof=0 : division par n
    rapport_sans_biais = float(np.var(ech, axis=1, ddof=1).mean())

    # 4. Couverture de x̄ ± 1,96 σ/√n (μ = 200, σ = 50, n = 25)
    mesures = rng.normal(200, 50, (config["n_echantillons"], 25))
    xbar = mesures.mean(axis=1)
    couverture = float(np.mean(np.abs(xbar - 200) <= 1.96 * 50 / 5))

    # 5. Bootstrap de dix temps de réponse
    donnees = np.array([120, 95, 140, 310, 105, 130, 98, 115, 160, 125], dtype=float)
    idx = rng.integers(0, donnees.size, (config["B"], donnees.size))
    boot_moy = donnees[idx].mean(axis=1)
    boot_med = np.median(donnees[idx], axis=1)
    ic_moy = np.percentile(boot_moy, [2.5, 97.5])
    ic_med = np.percentile(boot_med, [2.5, 97.5])
    ic_formule = donnees.mean() + np.array([-1.96, 1.96]) * donnees.std(ddof=1) / math.sqrt(donnees.size)

    # 6. Hoeffding : erreurs 0/1 de taux R = 0,2 sur N exemples
    N, R, gamma = 100, 0.2, 0.1
    erreurs = rng.random((config["repetitions"], N)) < R
    freq_ecart = float(np.mean(np.abs(erreurs.mean(axis=1) - R) > gamma))
    borne = 2 * math.exp(-2 * N * gamma**2)

    # 7. Effondrement : n = 100 points, réestimation à chaque génération
    def effondrement(ddof):
        sigmas = np.empty((config["chaines"], config["generations"] + 1))
        for c in range(config["chaines"]):
            mu, s = 0.0, 1.0
            sigmas[c, 0] = s
            for g in range(config["generations"]):
                x = rng.normal(mu, s, 100)
                mu, s = float(x.mean()), float(x.std(ddof=ddof))
                sigmas[c, g + 1] = s
        return np.median(sigmas, axis=0)

    med_mv = effondrement(0)
    med_sb = effondrement(1)
    duree = time.perf_counter() - t0
    print(f"1. k = {k} sur n = {n} : maximum de la grille en {p_grille:.3f}, k/n = {k / n:.3f}")
    print(f"2. a posteriori après 500 clics : moyenne {moyennes_post[-1]:.4f}, "
          f"fréquence {cumul[-1] / 500:.4f}, aire {aire_beta:.4f}")
    print(f"3. 𝔼[σ̂²] ≈ {rapport_biais:.4f} (attendu 0,8), 𝔼[s²] ≈ {rapport_sans_biais:.4f} (attendu 1)")
    print(f"4. couverture de l’intervalle à 95 % : {couverture:.4f}")
    print(f"5. bootstrap : moyenne [{ic_moy[0]:.1f} ; {ic_moy[1]:.1f}], "
          f"médiane [{ic_med[0]:.1f} ; {ic_med[1]:.1f}], "
          f"formule [{ic_formule[0]:.1f} ; {ic_formule[1]:.1f}]")
    print(f"6. fréquence des écarts > {gamma} : {freq_ecart:.4f}, borne de Hoeffding {borne:.4f}")
    print(f"7. σ médian après {config['generations']} générations : {med_mv[-1]:.4f} (n), {med_sb[-1]:.4f} (n − 1)")
    print(f"Calcul : {duree:.1f} s")
    return (aire_beta, borne, couverture, cumul, densite_beta, duree, ell, freq_ecart, grille, ic_med, ic_moy,
            k, med_mv, med_sb, moyennes_post, n, nb, p_grille, rapport_biais, rapport_sans_biais)


@app.cell
def _(sp):
    # --- Vérifications exactes (sympy) ---
    p, kk, nn = sp.symbols("p k n", positive=True)
    ell_s = kk * sp.log(p) + (nn - kk) * sp.log(1 - p)
    racine = sp.solve(sp.diff(ell_s, p), p)
    assert racine == [kk / nn], racine
    integrale = sp.integrate(p**7 * (1 - p) ** 3, (p, 0, 1))
    assert integrale == sp.Rational(1, 1320), integrale
    a, b = sp.symbols("a b", positive=True)
    moyenne_beta = sp.integrate(p * p**4 * (1 - p) ** 4, (p, 0, 1)) / sp.integrate(p**4 * (1 - p) ** 4, (p, 0, 1))
    assert moyenne_beta == sp.Rational(1, 2)
    gamma_s = sp.sqrt(sp.log(40) / 200)
    print(f"sympy : ℓ′(p) = 0 en p = {racine[0]} ; ∫ p⁷(1 − p)³ dp = {integrale} ; "
          f"moyenne de Bêta(5 ; 5) = {moyenne_beta} ; γ(N = 100, δ = 0,05) = {float(gamma_s):.4f}")
    return


@app.cell
def _(SMOKE, config, ell, grille, k, med_mv, med_sb, mo, moyennes_post, n, nb, np, plt, sauvegarder_figure):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(grille, ell)
    axes[0].axvline(k / n, linestyle="--", color="gray")
    axes[0].set(ylim=(ell.max() - 40, ell.max() + 3), xlabel="p", ylabel="ℓ(p)",
                title=f"Log-vraisemblance, {k} succès sur {n}")
    axes[1].plot(nb, moyennes_post, label="moyenne a posteriori")
    axes[1].axhline(0.3, linestyle="--", color="gray", label="taux réel 0,3")
    axes[1].axhline(0.1, linestyle=":", color="gray", label="moyenne a priori 0,1")
    axes[1].set(xlabel="nombre d’affichages", ylabel="taux estimé", title="Bêta(2 ; 18) mise à jour")
    axes[1].legend(fontsize=8)
    g = np.arange(config["generations"] + 1)
    axes[2].semilogy(g, med_mv, label="division par n")
    axes[2].semilogy(g, med_sb, label="division par n − 1")
    axes[2].set(xlabel="génération", ylabel="σ médian", title="Effondrement, n = 100")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    nom = "vraisemblance-smoke" if SMOKE else "vraisemblance"
    chemins = sauvegarder_figure(fig, nom, mo.notebook_dir() / "figures")
    print(f"Figure : {chemins[0].name}")
    fig  # noqa: B018 (marimo affiche la dernière expression)
    return


@app.cell
def _(aire_beta, borne, config, couverture, duree, freq_ecart, ic_med, ic_moy, k, math, med_mv, med_sb, mo,
      moyennes_post, n, p_grille, rapport_biais, rapport_sans_biais):
    # --- Bilan : seuils du README, vérifiés aussi en smoke ---
    echecs = []
    if abs(p_grille - k / n) > 0.001:
        echecs.append(f"maximum de la grille {p_grille:.4f} loin de k/n = {k / n:.4f}")
    if abs(aire_beta - 1) > 0.01:
        echecs.append(f"aire de la densité Bêta {aire_beta:.4f}")
    if abs(moyennes_post[-1] - 0.3) > 0.05:
        echecs.append(f"moyenne a posteriori finale {moyennes_post[-1]:.3f}")
    if abs(rapport_biais - 0.8) > config["tol_biais"] or abs(rapport_sans_biais - 1) > config["tol_biais"] * 1.25:
        echecs.append(f"biais : {rapport_biais:.4f} et {rapport_sans_biais:.4f}")
    if abs(couverture - 0.95) > config["tol_couv"]:
        echecs.append(f"couverture {couverture:.4f}")
    if not (ic_moy[0] < 139.8 < ic_moy[1] and ic_med[0] < 122.5 < ic_med[1]):
        echecs.append("intervalles bootstrap qui ne contiennent pas la statistique observée")
    if freq_ecart > borne:
        echecs.append(f"Hoeffding violé : {freq_ecart:.4f} > {borne:.4f}")
    # Dérive de ln σ̂² : environ −2/n par génération (division par n), −1/(n − 1) (division par n − 1)
    attendu_mv = math.exp(-config["generations"] / 100)
    attendu_sb = math.exp(-config["generations"] / (2 * 99))
    if not (attendu_mv / 1.6 < med_mv[-1] < attendu_mv * 1.6):
        echecs.append(f"effondrement (n) : {med_mv[-1]:.4f} pour ≈ {attendu_mv:.4f}")
    if not (attendu_sb / 1.6 < med_sb[-1] < attendu_sb * 1.6):
        echecs.append(f"effondrement (n − 1) : {med_sb[-1]:.4f} pour ≈ {attendu_sb:.4f}")
    if echecs:
        raise SystemExit("Bilan : ÉCHEC : " + " ; ".join(echecs))
    print(f"Bilan : OK (k/n retrouvé, biais {rapport_biais:.3f}, couverture {couverture:.3f}, "
          f"Hoeffding {freq_ecart:.4f} ≤ {borne:.4f}, σ médian {med_mv[-1]:.3f} et {med_sb[-1]:.3f}, {duree:.1f} s)")
    mo.md(f"**Bilan** : $\\mathbb{{E}}[\\hat\\sigma^2] \\approx$ {rapport_biais:.3f} (attendu 0,8) ; "
          f"couverture {couverture:.3f} ; écarts > γ : {freq_ecart:.4f}, sous la borne {borne:.4f}.")
    return


if __name__ == "__main__":
    app.run()
