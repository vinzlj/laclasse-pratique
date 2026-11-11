# Notebook 41-descente-gradient (chapitre 41, Optimisation : descente de gradient, SGD,
# conditionnement, flots, contraintes).
#
# La descente de gradient sur la perte de l’ouverture du chapitre (trois mesures de
# chargement de page) pour une gamme de pas : nombre de pas, rebonds, explosion au-delà du
# pas critique ; le bruit du gradient de mini-lot en 1/√B sur des données générées ; le
# momentum contre la descente simple sur la vallée ½(x² + κy²).
#
#   make nb NUM=41              → éditeur marimo (bouton « Lancer le calcul »)
#   make nb NUM=41 MODE=run     → script complet
#   make nb NUM=41 MODE=smoke   → test rapide (moins de 30 s)

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
    # Descente de gradient : le pas, le bruit, l’élan

    **Chapitre 41, sections 2 à 6.** Trois expériences :

    1. **Le pas critique** (sections 2 à 4) : la perte de l’ouverture,
       $L(a, b) = \frac{1}{3}(14a^2 + 12ab + 3b^2 - 46a - 20b + 38)$, descendue depuis
       $(0 \,;\, 0)$ avec des pas de 0,05 à 0,19. On compte les pas pour arriver à 0,001 du
       minimum $\frac{1}{18}$ ; le pas critique vaut $2/\lambda_{\max} \approx 0{,}1803$.
    2. **Le bruit du mini-lot** (section 5) : sur $N = 2\,000$ points générés, l’écart-type du
       gradient de mini-lot, mesuré sur de nombreux tirages, comparé à $\sigma/\sqrt{B}$.
    3. **Le momentum** (sections 4 et 6) : sur la vallée $\frac{1}{2}(x^2 + 16y^2)$, la descente
       avec le meilleur pas fixe contre le momentum $\eta = 0{,}16$, $\mu = 0{,}36$.
    """)


@app.cell
def _(is_smoke, np, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        pas=[0.05, 0.1, 0.17, 0.18, 0.19],   # pas testés sur la perte de l’ouverture
        n_donnees=2_000,                      # points générés pour le mini-lot
        tailles_lot=[1, 4, 16, 64],
        repetitions=2_000 if SMOKE else 50_000,  # tirages de mini-lots par taille
        kappa=16,
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
            mo.md("*Cliquez sur « Lancer le calcul ».*"))
    t0 = time.perf_counter()

    # 1. Perte de l’ouverture : L(w) = ½ wᵀHw + cᵀw + 38/3, H = (28/3 4 ; 4 2), c = (−46/3 ; −20/3)
    H = np.array([[28 / 3, 4.0], [4.0, 2.0]])
    c = np.array([-46 / 3, -20 / 3])
    L_min = 1 / 18

    def perte(w):
        return 0.5 * w @ H @ w + c @ w + 38 / 3

    traces = {}
    resultats_pas = {}
    for _eta in config["pas"]:
        _w = np.zeros(2)
        _trace = [perte(_w)]
        _resultat = None
        for _k in range(1, 3_000):
            _w = _w - _eta * (H @ _w + c)
            _v = perte(_w)
            _trace.append(_v)
            if _v - L_min < 1e-3:
                _resultat = ("arrivé", _k)
                break
            if _v > 1_000:
                _resultat = ("explosion", _k)
                break
        traces[_eta] = np.array(_trace)
        resultats_pas[_eta] = _resultat
    lam = np.linalg.eigvalsh(H)
    pas_critique = 2 / lam.max()

    # 2. Bruit du gradient de mini-lot, modèle t ≈ a p + b, au point (a ; b) = (0 ; 0)
    p = rng.uniform(0, 2, config["n_donnees"])
    t = 2 * p + 1 + rng.normal(0, 0.3, config["n_donnees"])
    w0 = np.array([0.0, 0.0])
    # gradient individuel de (a p + b − t)² : 2 (a p + b − t) (p ; 1)
    residus = w0[0] * p + w0[1] - t
    grads = 2 * residus[:, None] * np.stack([p, np.ones_like(p)], axis=1)
    sigma = grads.std(axis=0)          # écart-type d’un gradient individuel (tirage uniforme)
    ecarts = {}
    for _B in config["tailles_lot"]:
        _idx = rng.integers(0, config["n_donnees"], size=(config["repetitions"], _B))
        ecarts[_B] = grads[_idx].mean(axis=1).std(axis=0)
    biais = max(abs(grads[rng.integers(0, config["n_donnees"], size=(config["repetitions"], 16))]
                    .mean(axis=1).mean(axis=0) - grads.mean(axis=0)) / sigma)

    # 3. Momentum contre descente simple sur ½(x² + κ y²), départ (−4 ; 1)
    kappa = config["kappa"]

    def pas_pour_mille(eta, mu, n_max=5_000):
        w = np.array([-4.0, 1.0])
        b = np.zeros(2)
        n0 = np.linalg.norm(w)
        for k in range(1, n_max):
            g = np.array([w[0], kappa * w[1]])
            b = mu * b + g
            w = w - eta * b
            if np.linalg.norm(w) <= n0 / 1_000:
                return k
        return None

    def facteur(eta, mu):
        r = 0.0
        for lam_i in (1.0, kappa):
            r = max(r, max(abs(np.roots([1, -(1 + mu - eta * lam_i), mu]))))
        return float(r)

    eta_simple = 2 / (1 + kappa)
    momentum = dict(
        simple=(eta_simple, 0.0, facteur(eta_simple, 0.0), pas_pour_mille(eta_simple, 0.0)),
        elan=(0.16, 0.36, facteur(0.16, 0.36), pas_pour_mille(0.16, 0.36)),
    )
    regle = {cle: math.ceil(math.log(1_000) / math.log(1 / v[2])) for cle, v in momentum.items()}
    duree = time.perf_counter() - t0
    print(f"Calcul : {duree:.2f} s")
    print("Pas :", resultats_pas, f"; pas critique {pas_critique:.4f}")
    print("Écarts-types mesurés :", {_b: np.round(_e, 4) for _b, _e in ecarts.items()}, "σ =", np.round(sigma, 4))
    print("Momentum :", momentum, "règle du facteur :", regle)
    return (H, biais, duree, ecarts, momentum, pas_critique, regle, resultats_pas, sigma, traces)


@app.cell
def _(H, sp):
    # --- Vérification exacte (sympy) : minimum, hessienne et pas critique de la perte ---
    a, b = sp.symbols("a b")
    L = ((a + b - 2) ** 2 + (2 * a + b - 3) ** 2 + (3 * a + b - 5) ** 2) / 3
    sol = sp.solve([sp.diff(L, a), sp.diff(L, b)], [a, b])
    assert sol == {a: sp.Rational(3, 2), b: sp.Rational(1, 3)}, sol
    assert L.subs(sol) == sp.Rational(1, 18)
    Hs = sp.hessian(L, (a, b))
    assert Hs == sp.Matrix([[sp.Rational(28, 3), 4], [4, 2]])
    assert all(abs(float(x) - y) < 1e-12 for x, y in zip(Hs, H.flatten(), strict=True))
    lam_max = max(Hs.eigenvals())
    print(f"sympy : minimum {sol}, L = 1/18, hessienne {Hs.tolist()}, "
          f"λmax = {sp.nsimplify(lam_max)} ≈ {float(lam_max):.4f}, 2/λmax ≈ {float(2 / lam_max):.4f}")
    return


@app.cell
def _(SMOKE, config, ecarts, mo, np, plt, sauvegarder_figure, sigma, traces):
    # --- Figure : perte au fil des pas (échelle log) et écart-type du mini-lot contre B ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for _eta, _tr in traces.items():
        _n = min(len(_tr), 120)
        ax1.semilogy(np.arange(_n), np.maximum(_tr[:_n] - 1 / 18, 1e-6), label=f"η = {_eta:g}".replace(".", ","))
    ax1.set(xlabel="pas k", ylabel="L(w_k) − L(w*)", title="Perte de l’ouverture", ylim=(1e-4, 2e3))
    ax1.legend(fontsize=9)
    Bs = np.array(config["tailles_lot"])
    mesures = np.array([ecarts[_b][0] for _b in Bs])
    ax2.loglog(Bs, mesures, "o", label="mesuré (coordonnée a)")
    ax2.loglog(Bs, sigma[0] / np.sqrt(Bs), "-", label="σ/√B")
    ax2.set(xlabel="taille de lot B", ylabel="écart-type du gradient", title="Bruit du mini-lot")
    ax2.legend(fontsize=9)
    chemins = sauvegarder_figure(fig, "descente-gradient-smoke" if SMOKE else "descente-gradient",
                                 mo.notebook_dir() / "figures")
    print(f"Figure : {chemins[0].name}")
    fig  # noqa: B018 (marimo affiche la dernière expression)
    return


@app.cell
def _(SMOKE, biais, config, duree, ecarts, mo, momentum, pas_critique, regle, resultats_pas, sigma):
    # --- Bilan : seuils du README, vérifiés aussi en smoke (tolérance plus large sur le bruit) ---
    attendu = {0.05: ("arrivé", 99), 0.1: ("arrivé", 49), 0.17: ("arrivé", 41),
               0.18: ("arrivé", 1441), 0.19: ("explosion", 22)}
    tol = 0.08 if SMOKE else 0.02
    echecs = []
    for _eta, _res in attendu.items():
        if resultats_pas[_eta] != _res:
            echecs.append(f"η = {_eta} : {resultats_pas[_eta]} (attendu {_res})")
    if abs(pas_critique - 0.18029) > 1e-4:
        echecs.append(f"pas critique {pas_critique:.5f} (attendu 0,18029)")
    for _B in config["tailles_lot"]:
        _rapport = ecarts[_B] / (sigma / _B**0.5)
        if max(abs(_rapport - 1)) > tol:
            echecs.append(f"B = {_B} : écart-type / (σ/√B) = {_rapport.round(3)} (attendu 1 ± {tol})")
    if biais > tol:
        echecs.append(f"biais relatif du gradient de mini-lot {biais:.3f} (attendu < {tol})")
    if momentum["simple"][3] != 56 or momentum["elan"][3] != 19:
        echecs.append(f"pas pour diviser par 1 000 : {momentum['simple'][3]} et {momentum['elan'][3]} "
                      "(attendus 56 et 19)")
    if abs(momentum["simple"][2] - 15 / 17) > 1e-9 or abs(momentum["elan"][2] - 0.6) > 1e-6:
        echecs.append(f"facteurs {momentum['simple'][2]:.4f} et {momentum['elan'][2]:.4f} (attendus 15/17 et 0,6)")
    if regle != {"simple": 56, "elan": 14}:
        echecs.append(f"règle du facteur {regle} (attendue 56 et 14)")
    if echecs:
        raise SystemExit("Bilan : ÉCHEC : " + " ; ".join(echecs))
    r64 = (ecarts[16] / ecarts[64]).round(2)
    pc = f"{pas_critique:.4f}".replace(".", ",")
    r64s = " et ".join(f"{x:.2f}".replace(".", ",") for x in r64)
    print(f"Bilan : OK (pas 0,05 → 99, 0,1 → 49, 0,17 → 41, 0,18 → 1 441, 0,19 → explosion au pas 22 ; "
          f"pas critique {pc} ; écart-type B = 16 / B = 64 : {r64s} ; momentum 56 → 19 pas ; "
          f"{duree:.2f} s)".replace(".", ","))
    mo.md(f"**Bilan** : pas critique {pc} ; quadrupler le lot divise le bruit par "
          f"{r64s} ; momentum : {momentum['elan'][3]} pas au lieu de {momentum['simple'][3]}.")
    return


if __name__ == "__main__":
    app.run()
