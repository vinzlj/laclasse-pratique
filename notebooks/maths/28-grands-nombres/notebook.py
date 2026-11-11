# Notebook 28-grands-nombres (chapitre 28, Sommes de variables aléatoires, concentration,
# loi des grands nombres).
#
# Simulation de la loi des grands nombres : écart-type de la moyenne de n lancers de dé
# comparé à σ/√n (pente −1/2 en log-log), trajectoire d’une moyenne qui entre dans la bande
# 3,5 ± 2σ/√n, et fréquence observée d’un grand écart d’une fréquence de Bernoulli comparée
# aux majorants de Bienaymé-Tchebychev et de Hoeffding.
#
#   make nb NUM=28              → éditeur marimo (bouton « Lancer le calcul »)
#   make nb NUM=28 MODE=run     → script complet
#   make nb NUM=28 MODE=smoke   → test rapide (moins de 30 s)

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
    # La loi des grands nombres, simulée

    **Chapitre 28, sections 3 à 5 et 7.** Pour un échantillon i.i.d. d’espérance $\mu$ et de
    variance $\sigma^2$, la moyenne $M_n$ a pour espérance $\mu$ et pour écart-type
    $\sigma/\sqrt{n}$ ; l’inégalité de Bienaymé-Tchebychev majore
    $P(|M_n - \mu| \ge \varepsilon)$ par $\dfrac{\sigma^2}{n\varepsilon^2}$. Trois expériences :

    1. **La loi en $1/\sqrt{n}$.** Pour plusieurs tailles $n$, on répète $R$ fois l’expérience
       « lancer $n$ dés et faire la moyenne », puis on mesure l’écart-type des $R$ moyennes.
       On le compare à $\sigma/\sqrt{n}$, avec $\sigma^2 = 35/12$ (vérifié par sympy).
    2. **Une trajectoire.** Une seule suite de lancers : la moyenne après $n$ lancers entre
       dans la bande $3{,}5 \pm 2\sigma/\sqrt{n}$ et s’y maintient le plus souvent.
    3. **Les majorants.** Pour une pièce truquée ($p = 0{,}3$), on mesure la fréquence des
       échantillons dont la fréquence de pile s’écarte de $p$ d’au moins $\varepsilon = 0{,}05$,
       et on la compare aux majorants de Bienaymé-Tchebychev et de Hoeffding
       $2\,\mathrm{e}^{-2n\varepsilon^2}$ (encadré de la section 7).
    """)


@app.cell
def _(is_smoke, np, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        tailles=[10, 30, 100, 300, 1_000, 3_000] if SMOKE else [10, 30, 100, 300, 1_000, 3_000, 10_000],
        repetitions=400 if SMOKE else 4_000,          # R moyennes par taille n
        trajectoire=20_000,                           # lancers de la trajectoire unique
        p=0.3,                                        # pièce truquée de l’expérience 3
        epsilon=0.05,
        tailles_bernoulli=[50, 100, 200, 400, 800],
        repetitions_bernoulli=2_000 if SMOKE else 50_000,
    )
    rng = np.random.default_rng(SEED)
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SMOKE, config, rng


@app.cell
def _(sp):
    # --- Vérification exacte (sympy) : variance d’un dé et de la somme de deux dés ---
    faces = range(1, 7)
    mu_exact = sp.Rational(sum(faces), 6)
    var_exact = sp.Rational(sum(k * k for k in faces), 6) - mu_exact**2
    somme = [a + b for a in faces for b in faces]
    var_deux = sp.Rational(sum(s * s for s in somme), 36) - sp.Rational(sum(somme), 36) ** 2
    assert mu_exact == sp.Rational(7, 2) and var_exact == sp.Rational(35, 12), (mu_exact, var_exact)
    assert var_deux == 2 * var_exact == sp.Rational(35, 6), var_deux
    print(f"sympy : μ = {mu_exact}, σ² = {var_exact}, Var(somme de deux dés) = {var_deux}")
    MU, SIGMA = float(mu_exact), float(sp.sqrt(var_exact))
    return MU, SIGMA


@app.cell
def _(mo):
    # Le bouton empêche le calcul de se relancer à chaque modification d’une cellule amont.
    bouton = mo.ui.run_button(label="Lancer le calcul")
    bouton  # noqa: B018 (marimo affiche la dernière expression)
    return (bouton,)


@app.cell
def _(MU, SIGMA, bouton, config, mo, np, rng, time):
    # En script (make nb MODE=run|smoke), personne ne clique : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer le calcul ».*"))

    t0 = time.perf_counter()
    # Expérience 1 : écart-type des moyennes de n dés, R répétitions par taille
    tailles = np.array(config["tailles"])
    R = config["repetitions"]
    ecarts = []
    for n in tailles:
        # Somme de n dés = somme des faces ; par paquets pour borner la mémoire
        moyennes = np.empty(R)
        paquet = max(1, 2_000_000 // n)
        for debut in range(0, R, paquet):
            fin = min(R, debut + paquet)
            moyennes[debut:fin] = rng.integers(1, 7, size=(fin - debut, n)).mean(axis=1)
        ecarts.append(moyennes.std(ddof=1))
    ecarts = np.array(ecarts)
    theorique = SIGMA / np.sqrt(tailles)
    pente = np.polyfit(np.log10(tailles), np.log10(ecarts), 1)[0]

    # Expérience 2 : une trajectoire de la moyenne
    N = config["trajectoire"]
    lancers = rng.integers(1, 7, size=N)
    traj = np.cumsum(lancers) / np.arange(1, N + 1)
    ns = np.arange(1, N + 1)
    dans_bande = np.abs(traj - MU) <= 2 * SIGMA / np.sqrt(ns)
    part_bande = float(dans_bande[99:].mean())    # à partir de n = 100

    # Expérience 3 : grands écarts d’une fréquence de Bernoulli, contre les majorants
    p, eps = config["p"], config["epsilon"]
    tb = np.array(config["tailles_bernoulli"])
    Rb = config["repetitions_bernoulli"]
    observe = np.array([np.mean(np.abs(rng.binomial(n, p, size=Rb) / n - p) >= eps - 1e-12) for n in tb])
    bt = p * (1 - p) / (tb * eps**2)
    hoeffding = 2 * np.exp(-2 * tb * eps**2)
    duree = time.perf_counter() - t0
    print(f"Calcul : {duree:.2f} s ; pente log-log de l’écart-type : {pente:.3f}")
    for n, e, t in zip(tailles, ecarts, theorique, strict=True):
        print(f"  n = {n:>6} : écart-type mesuré {e:.4f}, σ/√n = {t:.4f}, rapport {e / t:.3f}")
    print(f"Trajectoire : moyenne finale {traj[-1]:.4f} après {N} lancers ; "
          f"dans la bande ± 2σ/√n pour {100 * part_bande:.1f} % des n ≥ 100")
    for n, o, b, h in zip(tb, observe, bt, hoeffding, strict=True):
        print(f"  Bernoulli n = {n:>4} : observé {o:.4f}, Bienaymé-Tchebychev {b:.4f}, Hoeffding {h:.4f}")
    return (bt, duree, ecarts, hoeffding, ns, observe, part_bande, pente, tailles, tb,
            theorique, traj)


@app.cell
def _(MU, SIGMA, SMOKE, bt, ecarts, hoeffding, mo, np, ns, observe, plt, sauvegarder_figure, tailles, tb,
      theorique, traj):
    # --- Figure, sauvegardée dans figures/ (suffixe -smoke : n’écrase pas celle du run complet) ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))
    ax1.loglog(tailles, ecarts, "o", label="écart-type mesuré de $M_n$")
    ax1.loglog(tailles, theorique, "-", label=r"$\sigma/\sqrt{n}$")
    ax1.set(xlabel="n (lancers par moyenne)", ylabel="écart-type", title="La loi en 1/√n")
    ax1.legend()
    ax2.semilogx(ns, traj, lw=1, label="moyenne des faces")
    ax2.semilogx(ns, MU + 2 * SIGMA / np.sqrt(ns), "k--", lw=0.8, label="bande ± 2σ/√n")
    ax2.semilogx(ns, MU - 2 * SIGMA / np.sqrt(ns), "k--", lw=0.8)
    ax2.set(xlabel="n", ylabel="moyenne", ylim=(1, 6), title="Une trajectoire")
    ax2.legend()
    ax3.semilogy(tb, observe, "o-", label="fréquence observée")
    ax3.semilogy(tb, np.minimum(bt, 1), "s--", label="Bienaymé-Tchebychev")
    ax3.semilogy(tb, np.minimum(hoeffding, 1), "^--", label="Hoeffding")
    ax3.set(xlabel="n", ylabel="P(|F − p| ≥ 0,05)", title="Les majorants")
    ax3.legend()
    nom = "grands-nombres-smoke" if SMOKE else "grands-nombres"
    chemins = sauvegarder_figure(fig, nom, mo.notebook_dir() / "figures")
    print(f"Figure : {chemins[0].name}")
    fig  # noqa: B018 (marimo affiche la dernière expression)
    return


@app.cell
def _(MU, SMOKE, bt, duree, ecarts, hoeffding, mo, np, observe, part_bande, pente, theorique, traj):
    # --- Bilan : seuils chiffrés du README, vérifiés aussi en smoke (tolérances plus larges) ---
    tol_rapport = 0.15 if SMOKE else 0.05
    tol_pente = 0.06 if SMOKE else 0.03
    rapports = ecarts / theorique
    echecs = []
    if np.max(np.abs(rapports - 1)) > tol_rapport:
        echecs.append(f"rapport écart-type / (σ/√n) hors de 1 ± {tol_rapport} : {np.round(rapports, 3)}")
    if abs(pente + 0.5) > tol_pente:
        echecs.append(f"pente {pente:.3f} (attendue −0,5 ± {tol_pente})")
    if abs(traj[-1] - MU) > 0.05:
        echecs.append(f"moyenne finale {traj[-1]:.4f} (attendue 3,5 ± 0,05)")
    if part_bande < 0.75:
        echecs.append(f"part de la trajectoire dans la bande {part_bande:.3f} (attendue ≥ 0,75)")
    marge = 0.01 if SMOKE else 0.002
    if np.any(observe > np.minimum(bt, 1) + marge) or np.any(observe > np.minimum(hoeffding, 1) + marge):
        echecs.append("une fréquence observée dépasse un majorant")
    if echecs:
        raise SystemExit("Bilan : ÉCHEC : " + " ; ".join(echecs))
    print(f"Bilan : OK (pente {pente:.3f}, rapports entre {rapports.min():.3f} et {rapports.max():.3f}, "
          f"moyenne finale {traj[-1]:.4f}, {100 * part_bande:.1f} % dans la bande, majorants respectés, {duree:.2f} s)")
    def fr(x, d):
        return f"{x:.{d}f}".replace(".", ",").replace("-", "−")

    mo.md(f"**Bilan** : pente {fr(pente, 3)} (attendue −0,5) ; moyenne finale {fr(traj[-1], 4)} ; "
          "majorants de Bienaymé-Tchebychev et de Hoeffding respectés.")
    return


if __name__ == "__main__":
    app.run()
