# Notebook 45-monte-carlo (chapitre 45, Échantillonnage : Monte-Carlo, réparamétrisation,
# score, gradient de politique).
#
# Estimer π par des fléchettes et voir l’erreur en 1/√n ; tirer une loi exponentielle par la
# transformée inverse et estimer son 99ᵉ centile ; comparer l’estimation directe et
# l’échantillonnage préférentiel de P(Z > 3) ; tirer dans un mélange ; estimer un gradient par
# réparamétrisation et par l’astuce du logarithme ; faire tourner REINFORCE sur un bandit à
# deux bras, avec et sans ligne de base ; suivre la dynamique de Langevin sur un mélange.
#
#   make nb NUM=45              → éditeur marimo (bouton « Lancer le calcul »)
#   make nb NUM=45 MODE=run     → script complet
#   make nb NUM=45 MODE=smoke   → test rapide (moins de 30 s)

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
    # Tirer, estimer, dériver une moyenne

    **Chapitre 45, sections 1 à 7.** Sept expériences :

    1. **π par fléchettes** : l’erreur décroît comme $\frac{1}{\sqrt{n}}$, et la marge à 95 %
       $1{,}96\,\sigma/\sqrt{n}$ (avec $\sigma \approx 1{,}642$) contient l’erreur environ 95 fois
       sur 100 (section 1).
    2. **Transformée inverse** : $-200 \ln(1 - U)$ suit la loi exponentielle de moyenne 200 ;
       son 99ᵉ centile estimé approche $-200 \ln 0{,}01 \approx 921{,}0$ (section 2).
    3. **Échantillonnage préférentiel** de $P(Z > 3) \approx 0{,}00135$ : avec
       $q = \mathcal{N}(3 \,;\, 1)$, la variance est divisée par environ 218 (section 3).
    4. **Mélange** $0{,}7\,\mathcal{N}(0 \,;\, 1) + 0{,}3\,\mathcal{N}(4 \,;\, 1)$ : variance 4,36,
       contre 0,58 pour la combinaison $0{,}7\,X_1 + 0{,}3\,X_2$ (section 4).
    5. **Gradient de $\mathbb{E}[z^2]$ en $\mu$** pour $z \sim \mathcal{N}(1 \,;\, 4)$ : la
       réparamétrisation et l’astuce du logarithme donnent toutes deux 2 en moyenne, avec des
       variances très différentes (sections 5 et 6).
    6. **REINFORCE** sur un bandit à deux bras : la ligne de base ne change pas la moyenne du
       gradient, elle annule ici sa variance ; la politique apprend à jouer le bon bras
       (section 6).
    7. **Langevin** sur $0{,}7\,\mathcal{N}(-2 \,;\, 0{,}8^2) + 0{,}3\,\mathcal{N}(2 \,;\, 0{,}8^2)$,
       toutes les particules partant de $x = 4$ : sans bruit, elles se figent à droite ; avec
       le bruit, environ 70 % finissent à gauche (section 7).
    """)


@app.cell
def _(is_smoke, np, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        n_pi=100_000 if SMOKE else 1_000_000,          # fléchettes par simulation
        simulations_pi=20 if SMOKE else 100,            # simulations pour la couverture
        n_expo=100_000 if SMOKE else 1_000_000,         # tirages exponentiels
        n_rare=10_000,                                  # tirages par estimation de P(Z > 3)
        repetitions_rare=200 if SMOKE else 1_000,       # estimations répétées
        n_melange=200_000 if SMOKE else 1_000_000,      # tirages du mélange
        n_gradient=200_000 if SMOKE else 1_000_000,     # tirages pour les gradients
        pas_reinforce=2_000,                            # pas de REINFORCE
        particules=2_000,                               # particules de Langevin
        pas_langevin=1_000,                             # pas de Langevin
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

    # 1. π par fléchettes : erreur en fonction de n, couverture de la marge à 95 %
    p_pi = math.pi / 4
    sigma_pi = 4 * math.sqrt(p_pi * (1 - p_pi))
    tailles_pi = np.array([10**k for k in range(2, int(math.log10(config["n_pi"])) + 1)])
    erreurs_pi = np.empty((config["simulations_pi"], tailles_pi.size))
    for s in range(config["simulations_pi"]):
        xy = rng.random((config["n_pi"], 2))
        dedans = np.cumsum((xy**2).sum(axis=1) <= 1)
        erreurs_pi[s] = np.abs(4 * dedans[tailles_pi - 1] / tailles_pi - math.pi)
    marges_pi = 1.96 * sigma_pi / np.sqrt(tailles_pi)
    couverture_pi = float(np.mean(erreurs_pi <= marges_pi))
    # pente de log(erreur moyenne) en fonction de log(n)
    pente_pi = float(np.polyfit(np.log10(tailles_pi), np.log10(erreurs_pi.mean(axis=0)), 1)[0])

    # 2. Transformée inverse : exponentielle de moyenne 200, 99ᵉ centile
    u = rng.random(config["n_expo"])
    temps = -200 * np.log(1 - u)
    moyenne_expo = float(temps.mean())
    p99_estime = float(np.quantile(temps, 0.99))
    p99_exact = -200 * math.log(0.01)
    au_dela_p99 = float(np.mean(temps > p99_exact))

    # 3. P(Z > 3) : directe contre préférentielle (q = N(3 ; 1), w(x) = exp(−3x + 4,5))
    p_rare = 0.5 * math.erfc(3 / math.sqrt(2))
    directes, preferentielles = [], []
    for _ in range(config["repetitions_rare"]):
        z = rng.standard_normal(config["n_rare"])
        directes.append(float(np.mean(z > 3)))
        x = 3 + rng.standard_normal(config["n_rare"])
        preferentielles.append(float(np.mean((x > 3) * np.exp(-3 * x + 4.5))))
    directes, preferentielles = np.array(directes), np.array(preferentielles)
    ratio_variances = float(directes.var() / preferentielles.var())
    erreur_rel_directe = float(directes.std() / p_rare)
    erreur_rel_pref = float(preferentielles.std() / p_rare)

    # 4. Mélange 0,7 N(0 ; 1) + 0,3 N(4 ; 1) contre la combinaison 0,7 X1 + 0,3 X2
    n_m = config["n_melange"]
    k = (rng.random(n_m) > 0.7).astype(int)               # composante : 0 ou 1
    melange = np.array([0.0, 4.0])[k] + rng.standard_normal(n_m)
    combinaison = 0.7 * rng.standard_normal(n_m) + 0.3 * (4 + rng.standard_normal(n_m))
    stats_melange = (float(melange.mean()), float(melange.var()))
    stats_combinaison = (float(combinaison.mean()), float(combinaison.var()))

    # 5. Gradient de E[z²] par rapport à μ, z ~ N(μ ; σ²), μ = 1, σ = 2 : valeur exacte 2μ = 2
    mu, sigma = 1.0, 2.0
    eps = rng.standard_normal(config["n_gradient"])
    zz = mu + sigma * eps
    g_reparam = 2 * zz                                     # f'(μ + σ ε)
    g_log = zz**2 * (zz - mu) / sigma**2                   # f(z) × score en μ
    g_log_base = (zz**2 - (mu**2 + sigma**2)) * (zz - mu) / sigma**2   # ligne de base 𝔼[f]
    gradients = {nom: (float(g.mean()), float(g.var()))
                 for nom, g in (("réparamétrisation", g_reparam), ("logarithme", g_log),
                                ("logarithme, ligne de base", g_log_base))}

    # 6. REINFORCE, bandit à deux bras : π(A) = σ(θ), récompense 1 pour A, 0 pour B
    def sig(t):
        return 1 / (1 + math.exp(-t))

    estimations_bandit = {}
    for b in (0.0, 0.5):
        pa = sig(0.0)
        a_joue = rng.random(config["n_gradient"]) < pa
        g = np.where(a_joue, (1 - b) * (1 - pa), (0 - b) * (-pa))
        estimations_bandit[b] = (float(g.mean()), float(g.var()))
    theta, eta, b_courant = 0.0, 0.1, 0.0
    trace_pa = []
    for _ in range(config["pas_reinforce"]):
        pa = sig(theta)
        a = rng.random() < pa
        gain = 1.0 if a else 0.0
        score = (1 - pa) if a else -pa
        theta += eta * (gain - b_courant) * score
        b_courant += 0.05 * (gain - b_courant)          # ligne de base : moyenne mobile des gains
        trace_pa.append(sig(theta))

    # 7. Langevin sur le mélange 0,7 N(−2 ; 0,8²) + 0,3 N(2 ; 0,8²), départ x = 4
    poids, moyennes, ecarts = np.array([0.7, 0.3]), np.array([-2.0, 2.0]), np.array([0.8, 0.8])

    def score_melange(x):
        lp = np.log(poids) - 0.5 * ((x[:, None] - moyennes) / ecarts) ** 2 - np.log(ecarts)
        r = np.exp(lp - lp.max(axis=1, keepdims=True))
        r /= r.sum(axis=1, keepdims=True)
        return (r * (-(x[:, None] - moyennes) / ecarts**2)).sum(axis=1)

    eta_l = 0.1
    x_bruit = np.full(config["particules"], 4.0)
    x_sans = np.full(config["particules"], 4.0)
    part_gauche = {}
    for t in range(1, config["pas_langevin"] + 1):
        x_bruit = x_bruit + eta_l / 2 * score_melange(x_bruit) + math.sqrt(eta_l) * rng.standard_normal(x_bruit.size)
        x_sans = x_sans + eta_l / 2 * score_melange(x_sans)
        if t in (100, 500, 1000):
            part_gauche[t] = float(np.mean(x_bruit < 0))

    duree = time.perf_counter() - t0
    print(f"1. π : couverture de la marge à 95 % {couverture_pi:.3f}, pente log-log {pente_pi:.3f}")
    print(f"2. Exponentielle : moyenne {moyenne_expo:.2f}, p99 estimé {p99_estime:.1f} "
          f"(exact {p99_exact:.1f}), part au-delà {au_dela_p99:.4f}")
    print(f"3. P(Z > 3) = {p_rare:.6f} ; erreur relative directe {erreur_rel_directe:.3f}, "
          f"préférentielle {erreur_rel_pref:.4f}, variance divisée par {ratio_variances:.0f}")
    print(f"4. Mélange : moyenne {stats_melange[0]:.3f}, variance {stats_melange[1]:.3f} ; "
          f"combinaison : moyenne {stats_combinaison[0]:.3f}, variance {stats_combinaison[1]:.3f}")
    for nom, (m, v) in gradients.items():
        print(f"5. Gradient par {nom} : moyenne {m:.3f}, variance par tirage {v:.1f}")
    for b, (m, v) in estimations_bandit.items():
        print(f"6. Bandit, b = {b} : moyenne {m:.4f} (exact 0,25), variance {v:.4f}")
    print(f"6. REINFORCE : π(A) passe de 0,5 à {trace_pa[-1]:.3f} en {len(trace_pa)} pas")
    print(f"7. Langevin : part à gauche {part_gauche} ; sans bruit, position moyenne finale "
          f"{x_sans.mean():.3f}, part à gauche {np.mean(x_sans < 0):.3f}")
    print(f"Calcul : {duree:.1f} s")
    return (au_dela_p99, couverture_pi, duree, erreurs_pi, estimations_bandit, gradients, moyenne_expo,
            p99_estime, p99_exact, part_gauche, pente_pi, ratio_variances, stats_combinaison,
            stats_melange, tailles_pi, temps, trace_pa, x_bruit, x_sans, marges_pi)


@app.cell
def _(sp):
    def _verifier():
        # --- Vérifications exactes (sympy) ---
        from sympy.stats import E, Normal

        # Transformée inverse de l’exponentielle de moyenne m
        x, u = sp.symbols("x u", positive=True)
        m = sp.symbols("m", positive=True)
        F = 1 - sp.exp(-x / m)
        inverse = sp.solve(sp.Eq(F, u), x)
        assert len(inverse) == 1, inverse
        assert sp.simplify(F.subs(x, inverse[0]) - u) == 0, inverse
        for uv in (sp.Rational(1, 2), sp.Rational(19, 20)):
            assert abs(float(inverse[0].subs({m: 200, u: uv}) + 200 * sp.log(1 - uv))) < 1e-9
        # 𝔼[(U1 + … + U10)²] = 10/12 + 25
        t = sp.symbols("t")
        mgf = ((sp.exp(t) - 1) / t) ** 10
        serie = sp.series(mgf, t, 0, 5).removeO()
        e2, e4 = serie.coeff(t, 2) * 2, serie.coeff(t, 4) * 24
        assert e2 == sp.Rational(155, 6), e2
        # Mélange : variance 1 + 0,7 × 0,3 × 16
        assert sp.Rational(7, 10) * 1 + sp.Rational(3, 10) * 17 - sp.Rational(6, 5) ** 2 == sp.Rational(109, 25)
        # Réparamétrisation : 𝔼[2(μ + σε)ε] = 2σ
        mu, s = sp.symbols("mu sigma", real=True)
        eps = Normal("eps", 0, 1)
        assert sp.simplify(E(2 * (mu + s * eps) * eps) - 2 * s) == 0
        # Astuce du logarithme : 𝔼[z² (z − μ)/σ²] = 2μ pour z ~ N(μ ; σ²)
        s_pos = sp.symbols("s", positive=True)
        Z = Normal("Z", mu, s_pos)
        assert sp.simplify(E(Z**2 * (Z - mu) / s_pos**2) - 2 * mu) == 0
        # Score de la normale
        xx = sp.symbols("xx", real=True)
        dens = sp.exp(-(xx - mu) ** 2 / (2 * s_pos**2)) / (s_pos * sp.sqrt(2 * sp.pi))
        assert sp.simplify(sp.diff(sp.log(dens), xx) + (xx - mu) / s_pos**2) == 0
        print(f"sympy : F⁻¹(u) = {inverse[0]} ; 𝔼[S²] = {e2} ≈ {float(e2):.4f}, "
              f"écart-type de S² ≈ {float(sp.sqrt(e4 - e2**2)):.3f} ; variance du mélange 109/25 = 4,36 ; "
              f"réparamétrisation, astuce du logarithme et score de la normale vérifiés")
    _verifier()
    return


@app.cell
def _(SMOKE, erreurs_pi, marges_pi, mo, np, plt, sauvegarder_figure, tailles_pi, temps, trace_pa,
      x_bruit, x_sans):
    def _tracer():
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for e in erreurs_pi[:5]:
            axes[0].loglog(tailles_pi, np.maximum(e, 1e-6), color="0.6", linewidth=1)
        axes[0].loglog(tailles_pi, marges_pi, "o-", color="C1", label="marge 1,96 σ/√n")
        axes[0].set(xlabel="n", ylabel="|estimation − π|", title="π par fléchettes")
        axes[0].legend(fontsize=8)
        grille = np.linspace(0, 1200, 300)
        axes[1].hist(temps[temps < 1200], bins=60, density=True, alpha=0.6, label="−200 ln(1 − U)")
        axes[1].plot(grille, np.exp(-grille / 200) / 200, color="C1", label="densité exponentielle")
        axes[1].set(xlabel="temps (ms)", title="Transformée inverse")
        axes[1].legend(fontsize=8)
        xs = np.linspace(-5, 5, 400)
        dens = 0.7 * np.exp(-((xs + 2) / 0.8) ** 2 / 2) + 0.3 * np.exp(-((xs - 2) / 0.8) ** 2 / 2)
        dens /= 0.8 * np.sqrt(2 * np.pi)
        axes[2].hist(x_bruit, bins=60, density=True, alpha=0.6, label="Langevin, avec bruit")
        axes[2].plot(xs, dens, color="C1", label="densité du mélange")
        axes[2].axvline(float(np.mean(x_sans)), color="k", linestyle="--", label="sans bruit (moyenne)")
        axes[2].set(xlabel="x", title=f"Langevin, π(A) final du bandit {trace_pa[-1]:.2f}")
        axes[2].legend(fontsize=8)
        fig.tight_layout()
        nom_fig = "monte-carlo-smoke" if SMOKE else "monte-carlo"
        chemins = sauvegarder_figure(fig, nom_fig, mo.notebook_dir() / "figures")
        print(f"Figure : {chemins[0].name}")
        return fig
    _sortie = _tracer()
    _sortie  # noqa: B018 (marimo affiche la dernière expression)
    return


@app.cell
def _(au_dela_p99, couverture_pi, duree, estimations_bandit, gradients, mo, moyenne_expo, np,
      p99_estime, p99_exact, part_gauche, pente_pi, ratio_variances, stats_combinaison, stats_melange,
      trace_pa, x_sans):
    def _bilan():
        # --- Bilan : seuils du README, vérifiés aussi en smoke ---
        echecs = []
        if not 0.85 <= couverture_pi <= 1.0:
            echecs.append(f"couverture de la marge {couverture_pi:.3f}")
        if not -0.65 <= pente_pi <= -0.35:
            echecs.append(f"pente log-log {pente_pi:.3f}")
        if abs(moyenne_expo - 200) > 2:
            echecs.append(f"moyenne exponentielle {moyenne_expo:.2f}")
        if abs(p99_estime - p99_exact) / p99_exact > 0.03:
            echecs.append(f"p99 {p99_estime:.1f} contre {p99_exact:.1f}")
        if abs(au_dela_p99 - 0.01) > 0.002:
            echecs.append(f"part au-delà du p99 {au_dela_p99:.4f}")
        if not 120 <= ratio_variances <= 400:
            echecs.append(f"rapport des variances {ratio_variances:.0f} (attendu ≈ 218)")
        if abs(stats_melange[0] - 1.2) > 0.02 or abs(stats_melange[1] - 4.36) > 0.05:
            echecs.append(f"mélange {stats_melange}")
        if abs(stats_combinaison[0] - 1.2) > 0.02 or abs(stats_combinaison[1] - 0.58) > 0.02:
            echecs.append(f"combinaison {stats_combinaison}")
        m_r, v_r = gradients["réparamétrisation"]
        m_l, v_l = gradients["logarithme"]
        if abs(m_r - 2) > 0.05 or abs(m_l - 2) > 0.3 or abs(v_r - 16) > 0.5 or abs(v_l - 74.25) > 6:
            echecs.append(f"gradients {gradients}")
        (m0, v0), (m5, v5) = estimations_bandit[0.0], estimations_bandit[0.5]
        if abs(m0 - 0.25) > 0.002 or abs(m5 - 0.25) > 1e-12 or v5 > 1e-12 or abs(v0 - 0.0625) > 0.002:
            echecs.append(f"bandit {estimations_bandit}")
        if trace_pa[-1] < 0.95:
            echecs.append(f"REINFORCE : π(A) final {trace_pa[-1]:.3f}")
        if not (part_gauche[100] < 0.25 and part_gauche[500] > 0.5 and abs(part_gauche[1000] - 0.7) < 0.05):
            echecs.append(f"Langevin, part à gauche {part_gauche}")
        if abs(float(np.mean(x_sans)) - 2) > 0.1 or np.any(x_sans < 0):
            echecs.append(f"Langevin sans bruit : moyenne {np.mean(x_sans):.3f}")
        if echecs:
            raise SystemExit("Bilan : ÉCHEC : " + " ; ".join(echecs))
        print(f"Bilan : OK (couverture {couverture_pi:.2f}, pente {pente_pi:.2f}, p99 {p99_estime:.0f}, "
              f"variance ÷ {ratio_variances:.0f}, mélange {stats_melange[1]:.2f}, "
              f"Langevin {part_gauche[1000]:.2f} à gauche, {duree:.1f} s)")
        return mo.md(f"**Bilan** : erreur de Monte-Carlo en pente {pente_pi:.2f} (attendu −0,5) ; "
                     f"variance de $P(Z > 3)$ divisée par {ratio_variances:.0f} ; "
                     f"Langevin : {100 * part_gauche[1000]:.0f} % des particules dans la grosse bosse.")
    _sortie = _bilan()
    _sortie  # noqa: B018 (marimo affiche la dernière expression)
    return


if __name__ == "__main__":
    app.run()
