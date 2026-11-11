# Notebook 47-elbo-melange (chapitre 47, Information II : inégalité de Jensen, information
# mutuelle, ELBO).
#
# Vérifier l’inégalité de Jensen sur des tirages ; calculer une information mutuelle exacte
# et l’estimer sur des tirages ; faire tourner EM sur un mélange de deux gaussiennes à une
# dimension en suivant ln p et l’ELBO à chaque demi-étape ; comparer la formule fermée de la
# KL d’un « encodeur » gaussien à une somme de Riemann ; calculer la politique de Gibbs de
# trois réponses pour plusieurs β.
#
#   make nb NUM=47              → éditeur marimo (bouton « Lancer le calcul »)
#   make nb NUM=47 MODE=run     → script complet
#   make nb NUM=47 MODE=smoke   → test rapide (moins de 30 s)

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
    # Jensen, information mutuelle et ELBO, par le calcul

    **Chapitre 47, sections 1 à 7.** Cinq vérifications :

    1. **Jensen** : pour $X$ uniforme sur $[1 \,;\, 4]$, la moyenne des $\ln X$ tirés est sous
       $\ln$ de la moyenne, et l’écart approche la valeur exacte (section 2).
    2. **Information mutuelle** du tableau des pages lourdes et lentes, $(0{,}4 \,;\, 0{,}1 \,;\,
       0{,}1 \,;\, 0{,}4)$ : 0,278 bit exactement ; estimée sur $N$ tirages, elle s’en approche,
       et celle d’un tableau indépendant reste petite mais positive (section 4).
    3. **EM** sur un mélange de deux gaussiennes à une dimension : $\ln p$ ne baisse jamais,
       l’ELBO reste dessous et la touche après chaque étape E (sections 5 et 6).
    4. **KL d’un encodeur gaussien** contre $\mathcal{N}(0 \,;\, 1)$ : formule fermée
       $\frac{1}{2}(\mu^2 + \sigma^2 - 1 - \ln \sigma^2)$ et somme de Riemann (section 6).
    5. **Politique de Gibbs** sur trois réponses : $\pi^* \propto \pi_{\mathrm{ref}}\,e^{r/\beta}$
       bat des politiques tirées au hasard, et sa valeur est $\beta \ln Z$ (section 7).
    """)


@app.cell
def _(is_smoke, np, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        n_jensen=50_000 if SMOKE else 500_000,          # tirages pour Jensen
        tailles_mi=[100, 1_000, 10_000] if SMOKE else [100, 1_000, 10_000, 100_000],
        repetitions_mi=50 if SMOKE else 200,            # répétitions de chaque estimation
        n_points_em=300,                                # points du mélange
        iterations_em=30 if SMOKE else 100,             # itérations d’EM
        politiques=2_000 if SMOKE else 20_000,          # politiques tirées au hasard
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

    # 1. Jensen pour ln : X uniforme sur [1 ; 4]
    x = rng.uniform(1, 4, config["n_jensen"])
    ln_moyenne, moyenne_ln = math.log(x.mean()), float(np.log(x).mean())
    # Valeurs exactes : 𝔼[X] = 2,5 ; 𝔼[ln X] = (4 ln 4 − 4 − (1 ln 1 − 1))/3
    exact_moyenne_ln = (4 * math.log(4) - 3) / 3
    ecart_exact = math.log(2.5) - exact_moyenne_ln

    # 2. Information mutuelle (bits)
    def info_mutuelle(t):
        t = np.asarray(t, dtype=float)
        t = t / t.sum()
        produit = t.sum(axis=1, keepdims=True) * t.sum(axis=0, keepdims=True)
        masque = t > 0
        return float(np.sum(t[masque] * np.log2(t[masque] / produit[masque])))

    table = np.array([[0.4, 0.1], [0.1, 0.4]])
    independante = np.array([[0.12, 0.18], [0.28, 0.42]])
    i_exacte = info_mutuelle(table)
    estimations = {}
    for nom_t, t in (("dépendante", table), ("indépendante", independante)):
        moy = []
        for n_tir in config["tailles_mi"]:
            vals = [info_mutuelle(rng.multinomial(n_tir, t.ravel()).reshape(2, 2))
                    for _ in range(config["repetitions_mi"])]
            moy.append(float(np.mean(vals)))
        estimations[nom_t] = moy

    # 3. EM sur un mélange de deux gaussiennes (cinq paramètres)
    # Données tirées par un générateur à part (graine 47), identiques en smoke et en complet
    rng_em = np.random.default_rng(47)
    donnees = np.concatenate([rng_em.normal(-1.0, 1.0, 120), rng_em.normal(2.5, 1.0, 180)])

    def log_jointes(th):
        lj = []
        for w, m, s in ((th[0], th[1], th[2]), (1 - th[0], th[3], th[4])):
            lj.append(math.log(w) - 0.5 * np.log(2 * np.pi * s**2) - (donnees - m) ** 2 / (2 * s**2))
        return np.stack(lj, axis=1)

    def log_p(th):
        lj = log_jointes(th)
        return float(np.sum(np.logaddexp(lj[:, 0], lj[:, 1])))

    def elbo(q, th):
        return float(np.sum(q * (log_jointes(th) - np.log(q))))

    th = (0.5, 0.6, 1.5, 1.6, 1.5)
    q = np.full((donnees.size, 2), 0.5)
    trace_lp, trace_elbo, ecarts_e = [log_p(th)], [elbo(q, th)], []
    for _ in range(config["iterations_em"]):
        lj = log_jointes(th)
        q = np.exp(lj - np.logaddexp(lj[:, [0]], lj[:, [1]]))       # étape E
        trace_lp.append(log_p(th))
        trace_elbo.append(elbo(q, th))
        ecarts_e.append(trace_lp[-1] - trace_elbo[-1])
        n1, n2 = q[:, 0].sum(), q[:, 1].sum()                        # étape M
        m1, m2 = q[:, 0] @ donnees / n1, q[:, 1] @ donnees / n2
        s1 = math.sqrt(q[:, 0] @ (donnees - m1) ** 2 / n1)
        s2 = math.sqrt(q[:, 1] @ (donnees - m2) ** 2 / n2)
        th = (n1 / donnees.size, m1, s1, m2, s2)
        trace_lp.append(log_p(th))
        trace_elbo.append(elbo(q, th))
    trace_lp, trace_elbo = np.array(trace_lp), np.array(trace_elbo)

    # 4. KL d’un encodeur gaussien contre 𝒩(0 ; 1) : formule fermée et somme de Riemann
    def kl_fermee(mu, s):
        return 0.5 * (mu**2 + s**2 - 1 - math.log(s**2))

    def kl_riemann(mu, s):
        grille = np.linspace(mu - 12 * s, mu + 12 * s, 200_001)
        dx = grille[1] - grille[0]
        log_q = -0.5 * np.log(2 * np.pi * s**2) - (grille - mu) ** 2 / (2 * s**2)
        log_p0 = -0.5 * np.log(2 * np.pi) - grille**2 / 2
        return float(np.sum(np.exp(log_q) * (log_q - log_p0)) * dx)

    cas_kl = [(0, 1), (1, 1), (0, 0.5), (2, 0.3)]
    kl_comparaison = [(mu, s, kl_fermee(mu, s), kl_riemann(mu, s)) for mu, s in cas_kl]

    # 5. Politique de Gibbs sur trois réponses
    p_ref = np.array([0.5, 0.3, 0.2])
    r = np.array([0.0, 1.0, 2.0])

    def objectif(pi, beta):
        return float(pi @ r - beta * np.sum(pi * np.log(pi / p_ref)))

    gibbs = {}
    for beta in (0.1, 0.5, 1.0, 2.0, 10.0):
        poids = p_ref * np.exp(r / beta)
        z = float(poids.sum())
        pi_star = poids / z
        aleatoires = rng.dirichlet(np.ones(3), config["politiques"])
        meilleur_aleatoire = max(objectif(pi, beta) for pi in aleatoires)
        gibbs[beta] = (pi_star, beta * math.log(z), objectif(pi_star, beta), meilleur_aleatoire)

    duree = time.perf_counter() - t0
    print(f"1. Jensen : ln 𝔼[X] ≈ {ln_moyenne:.4f}, 𝔼[ln X] ≈ {moyenne_ln:.4f}, "
          f"écart {ln_moyenne - moyenne_ln:.4f} (exact {ecart_exact:.4f})")
    print(f"2. I exacte {i_exacte:.4f} bit ; estimations (dépendante) "
          f"{[round(v, 4) for v in estimations['dépendante']]}, "
          f"(indépendante) {[round(v, 5) for v in estimations['indépendante']]}")
    print(f"3. EM : ln p de {trace_lp[0]:.2f} à {trace_lp[-1]:.2f} ; plus grand écart après une "
          f"étape E {max(ecarts_e):.2e}")
    for mu, s, kf, kr in kl_comparaison:
        print(f"4. (μ ; σ) = ({mu} ; {s}) : formule {kf:.4f}, Riemann {kr:.4f}")
    for beta, (pi_star, val, obj, alea) in gibbs.items():
        print(f"5. β = {beta} : π* = {np.round(pi_star, 3)}, β ln Z = {val:.4f}, "
              f"J(π*) = {obj:.4f}, meilleure politique au hasard {alea:.4f}")
    print(f"Calcul : {duree:.1f} s")
    return (duree, ecart_exact, ecarts_e, estimations, gibbs, i_exacte, kl_comparaison, ln_moyenne,
            moyenne_ln, trace_elbo, trace_lp)


@app.cell
def _(sp):
    def _verifier():
        # --- Vérifications exactes (sympy) ---
        from sympy.stats import E, Normal

        m1, m2 = sp.symbols("mu1 mu2", real=True)
        s = sp.symbols("sigma", positive=True)
        X = Normal("X", m1, s)
        kl_meme_sigma = sp.simplify(E(((X - m2) ** 2 - (X - m1) ** 2) / (2 * s**2)))
        assert sp.simplify(kl_meme_sigma - (m1 - m2) ** 2 / (2 * s**2)) == 0, kl_meme_sigma
        # Cas général avec σ₂ = 1, μ₂ = 0 : formule du VAE
        mu, s1 = sp.symbols("mu s1", positive=True)
        general = sp.log(1 / s1) + (s1**2 + mu**2) / 2 - sp.Rational(1, 2)
        vae = sp.Rational(1, 2) * (mu**2 + s1**2 - 1 - sp.log(s1**2))
        assert sp.simplify(sp.expand_log(general - vae, force=True)) == 0
        # Gibbs : J(π*) = β ln Z sur l’exemple de l’ouverture (β = 1)
        z = sp.Rational(1, 2) * sp.E + sp.Rational(1, 2)
        pis = [sp.Rational(1, 2) * sp.E / z, sp.Rational(1, 2) / z]
        j = pis[0] * 1 - sum(p * sp.log(p / sp.Rational(1, 2)) for p in pis)
        assert abs(float(j - sp.log(z))) < 1e-12
        print(f"sympy : KL même σ = {kl_meme_sigma} ; formule du VAE retrouvée ; "
              f"π*(1) = {float(pis[0]):.4f}, J(π*) = ln Z = {float(sp.log(z)):.4f}")
    _verifier()
    return


@app.cell
def _(SMOKE, config, estimations, gibbs, i_exacte, mo, np, plt, sauvegarder_figure, trace_elbo, trace_lp):
    def _tracer():
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        k = min(9, trace_lp.size)
        axes[0].plot(range(k), trace_lp[:k], "o-", label="ln p(données | θ)")
        axes[0].plot(range(k), trace_elbo[:k], "s--", label="ELBO(q, θ)")
        axes[0].set(xlabel="demi-étape (départ, E1, M1, …)", ylabel="nats", title="EM : quatre premières itérations")
        axes[0].legend(fontsize=8)
        axes[1].semilogx(config["tailles_mi"], estimations["dépendante"], "o-", label="tableau dépendant")
        axes[1].semilogx(config["tailles_mi"], estimations["indépendante"], "s-", label="tableau indépendant")
        axes[1].axhline(i_exacte, linestyle="--", color="gray", label="valeur exacte 0,278")
        axes[1].set(xlabel="nombre de tirages", ylabel="bits", title="Information mutuelle estimée")
        axes[1].legend(fontsize=8)
        betas = sorted(gibbs)
        pis = np.array([gibbs[b][0] for b in betas])
        for j, nom in enumerate(["r = 0", "r = 1", "r = 2"]):
            axes[2].semilogx(betas, pis[:, j], "o-", label=nom)
        axes[2].set(xlabel="β", ylabel="π*(y)", title="Politique de Gibbs")
        axes[2].legend(fontsize=8)
        fig.tight_layout()
        nom_fig = "elbo-melange-smoke" if SMOKE else "elbo-melange"
        chemins = sauvegarder_figure(fig, nom_fig, mo.notebook_dir() / "figures")
        print(f"Figure : {chemins[0].name}")
        return fig
    _sortie = _tracer()
    _sortie  # noqa: B018 (marimo affiche la dernière expression)
    return


@app.cell
def _(duree, ecart_exact, ecarts_e, estimations, gibbs, i_exacte, kl_comparaison, ln_moyenne, mo,
      moyenne_ln, np, trace_elbo, trace_lp):
    def _bilan():
        # --- Bilan : seuils du README, vérifiés aussi en smoke ---
        echecs = []
        if not moyenne_ln < ln_moyenne:
            echecs.append("Jensen violé sur les tirages")
        if abs((ln_moyenne - moyenne_ln) - ecart_exact) > 0.005:
            echecs.append(f"écart de Jensen {ln_moyenne - moyenne_ln:.4f} loin de {ecart_exact:.4f}")
        if abs(i_exacte - 0.2781) > 1e-4:
            echecs.append(f"information mutuelle exacte {i_exacte:.4f}")
        if abs(estimations["dépendante"][-1] - i_exacte) > 0.005:
            echecs.append(f"estimation de I {estimations['dépendante'][-1]:.4f}")
        if not all(v >= 0 for v in estimations["indépendante"]) or estimations["indépendante"][-1] > 0.001:
            echecs.append(f"tableau indépendant : {estimations['indépendante']}")
        if np.any(np.diff(trace_lp) < -1e-9):
            echecs.append("ln p a baissé pendant EM")
        if np.any(trace_lp - trace_elbo < -1e-9):
            echecs.append("ELBO au-dessus de ln p")
        if max(ecarts_e) > 1e-9:
            echecs.append(f"écart après l’étape E {max(ecarts_e):.2e}")
        for mu, s, kf, kr in kl_comparaison:
            if abs(kf - kr) > 1e-4:
                echecs.append(f"KL ({mu} ; {s}) : {kf:.5f} contre {kr:.5f}")
        for beta, (_pi, val, obj, alea) in gibbs.items():
            if abs(val - obj) > 1e-9 or alea > obj + 1e-12:
                echecs.append(f"Gibbs β = {beta} : β ln Z {val:.5f}, J(π*) {obj:.5f}, hasard {alea:.5f}")
        if abs(gibbs[1.0][1] - 1.0272) > 1e-4:
            echecs.append(f"ln Z pour β = 1 : {gibbs[1.0][1]:.4f}")
        if echecs:
            raise SystemExit("Bilan : ÉCHEC : " + " ; ".join(echecs))
        print(f"Bilan : OK (Jensen {ln_moyenne - moyenne_ln:.4f}, I = {i_exacte:.4f} bit, "
              f"EM monotone, écart après E ≤ {max(ecarts_e):.1e}, KL fermée = Riemann, "
              f"Gibbs optimale, {duree:.1f} s)")
        return mo.md(f"**Bilan** : écart de Jensen {ln_moyenne - moyenne_ln:.4f} nat ; "
                     f"$I \\approx$ {i_exacte:.3f} bit ; "
              f"EM : $\\ln p$ de {trace_lp[0]:.1f} à {trace_lp[-1]:.1f}, sans jamais baisser.")
    _sortie = _bilan()
    _sortie  # noqa: B018 (marimo affiche la dernière expression)
    return


if __name__ == "__main__":
    app.run()
