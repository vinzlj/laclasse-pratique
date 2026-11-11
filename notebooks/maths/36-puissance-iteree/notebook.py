# Notebook 36-puissance-iteree (chapitre 36, Valeurs propres, diagonalisation, matrices
# symétriques et formes quadratiques).
#
# La puissance itérée en numpy : sur la matrice 2 × 2 du cours, puis sur une matrice
# symétrique 50 × 50 (et 200 × 200 en mode complet), comparée à np.linalg.eigh ; vitesse de
# convergence selon le rapport |λ₂/λ₁| ; pas critique d’une descente de gradient sur
# ½ xᵀAx (η < 2/λ_max). Vérifications exactes par sympy. numpy seul (pas de scipy).
#
#   make nb NUM=36              → éditeur marimo (bouton « Lancer le calcul »)
#   make nb NUM=36 MODE=run     → script complet
#   make nb NUM=36 MODE=smoke   → test rapide (moins de 30 s)

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
    # La puissance itérée, et le pas critique d’une descente

    **Chapitre 36, sections 4, 5 et 7.** Pour une matrice $A$, la puissance itérée répète

    $$
    \mathbf{x}_{k+1} = \frac{A\mathbf{x}_k}{\lVert A\mathbf{x}_k \rVert},
    $$

    et le quotient de Rayleigh $\frac{\mathbf{x}_k^\top A\mathbf{x}_k}{\mathbf{x}_k^\top\mathbf{x}_k}$
    tend vers la valeur propre $\lambda_1$ de plus grand module.

    Quatre expériences :

    1. la matrice $\begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$ du cours, départ $(1 \,;\, 0)$ :
       les quotients $2$ ; $2{,}8$ ; $2{,}976$… tendent vers $3$ ;
    2. une matrice symétrique $50 \times 50$ tirée au hasard : la puissance itérée contre
       `np.linalg.eigh` ;
    3. la vitesse : l’écart à la direction propre est multiplié par $|\lambda_2/\lambda_1|$
       à chaque pas ; on compte les pas pour trois rapports, $0{,}5$ ; $0{,}9$ et $0{,}99$ ;
    4. la descente de gradient sur $\frac{1}{2}\mathbf{x}^\top A\mathbf{x}$ : elle converge
       pour $\eta$ juste sous $2/\lambda_{\max}$ et explose juste au-dessus.
    """)


@app.cell
def _(is_smoke, np, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        tailles=[50] if SMOKE else [50, 200],   # tailles des matrices symétriques de l’expérience 2
        pas_max=3_000 if SMOKE else 20_000,     # itérations au plus (expériences 2 et 3)
        n_descente=300 if SMOKE else 2_000,     # pas de la descente de l’expérience 4
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
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer le calcul ».*"))
    t0 = time.perf_counter()

    def puissance_iteree(A, x0, n_pas, tol=None):
        """Renvoie (x, quotients, nombre de pas) ; s’arrête si le quotient bouge de moins de tol."""
        x = x0 / np.linalg.norm(x0)
        quotients = [float(x @ A @ x)]
        for k in range(1, n_pas + 1):
            y = A @ x
            x = y / np.linalg.norm(y)
            quotients.append(float(x @ A @ x))
            if tol is not None and abs(quotients[-1] - quotients[-2]) < tol * abs(quotients[-1]):
                return x, quotients, k
        return x, quotients, n_pas

    # 1. La matrice du cours
    A2 = np.array([[2.0, 1.0], [1.0, 2.0]])
    x_cours, q_cours, _ = puissance_iteree(A2, np.array([1.0, 0.0]), 12)
    non_normalises = [np.linalg.matrix_power(A2, k) @ np.array([1, 0]) for k in range(5)]

    # 2. Matrices symétriques aléatoires contre np.linalg.eigh
    comparaisons = []
    for n in config["tailles"]:
        G = rng.normal(size=(n, n))
        S = (G + G.T) / 2
        valeurs, vecteurs = np.linalg.eigh(S)
        indice = int(np.argmax(np.abs(valeurs)))
        lam_ref, v_ref = valeurs[indice], vecteurs[:, indice]
        modules = np.sort(np.abs(valeurs))[::-1]
        x_n, q_n, pas_n = puissance_iteree(S, rng.normal(size=n), config["pas_max"], tol=1e-14)
        ecart_lambda = abs(q_n[-1] - lam_ref) / abs(lam_ref)
        ecart_direction = 1 - abs(float(x_n @ v_ref))
        comparaisons.append(dict(n=n, lam=float(lam_ref), rapport=float(modules[1] / modules[0]), pas=pas_n,
                                 ecart_lambda=float(ecart_lambda), ecart_direction=float(ecart_direction)))

    # 3. Vitesse selon |λ₂/λ₁| : matrice Q diag(1, r, r/2) Qᵀ en dimension 3, départ fixe
    Q, _r = np.linalg.qr(rng.normal(size=(3, 3)))
    depart = rng.normal(size=3)
    vitesses = {}
    courbes = {}
    for r in (0.5, 0.9, 0.99):
        M = Q @ np.diag([1.0, r, r / 2]) @ Q.T
        x = depart / np.linalg.norm(depart)
        angles = []
        for _k in range(config["pas_max"]):
            x = M @ x
            x /= np.linalg.norm(x)
            sinus = float(np.sqrt(max(0.0, 1 - (x @ Q[:, 0]) ** 2)))
            angles.append(sinus)
            if sinus < 1e-6:
                break
        vitesses[r] = len(angles)
        courbes[r] = angles
    # prévision : sinus ≈ C rᵏ, donc k ≈ log(1e-6 / C) / log r
    prevues = {}
    for r, angles in courbes.items():
        C = angles[0] / r
        prevues[r] = float(np.log(1e-6 / C) / np.log(r))

    # 4. Pas critique : A symétrique définie positive tirée au hasard, η = 0,99 et 1,01 fois 2/λ_max
    H = rng.normal(size=(20, 20))
    A_dp = H @ H.T / 20 + 0.1 * np.eye(20)
    lam_max = float(np.linalg.eigvalsh(A_dp)[-1])
    eta_c = 2 / lam_max
    normes = {}
    for facteur in (0.99, 1.01):
        x = rng.normal(size=20)
        suite = [float(np.linalg.norm(x))]
        for _k in range(config["n_descente"]):
            x = x - facteur * eta_c * (A_dp @ x)
            suite.append(float(np.linalg.norm(x)))
        normes[facteur] = suite
    # exemple du cours : ½(x² + 10y²), facteurs 1 − η et 1 − 10η
    cours_descente = {eta: (1 - eta, 1 - 10 * eta) for eta in (0.15, 0.2, 0.25)}

    duree = time.perf_counter() - t0
    print(f"Calcul : {duree:.2f} s")
    print("1. vecteurs non normalisés :", [v.tolist() for v in non_normalises])
    print("   quotients de Rayleigh :", [round(q, 6) for q in q_cours[:6]], "…", round(q_cours[-1], 10))
    for c in comparaisons:
        print(f"2. n = {c['n']} : λ₁ = {c['lam']:.6f}, |λ₂/λ₁| = {c['rapport']:.3f}, {c['pas']} pas, "
              f"écart relatif sur λ {c['ecart_lambda']:.1e}, sur la direction {c['ecart_direction']:.1e}")
    print(f"3. pas pour un sinus < 10⁻⁶ : mesurés {vitesses}, prévus {({r: round(p) for r, p in prevues.items()})}")
    print(f"4. λ_max = {lam_max:.4f}, 2/λ_max = {eta_c:.4f} ; norme finale à 0,99 η_c : {normes[0.99][-1]:.2e}, "
          f"à 1,01 η_c : {normes[1.01][-1]:.2e} ; cours : {cours_descente}")
    return comparaisons, courbes, cours_descente, duree, eta_c, normes, non_normalises, prevues, q_cours, vitesses


@app.cell
def _(sp):
    # --- Vérification exacte (sympy) : exemples du cours ---
    _lam = sp.symbols("lambda")
    _A = sp.Matrix([[2, 1], [1, 2]])
    assert sp.factor((_A - _lam * sp.eye(2)).det()) == (_lam - 3) * (_lam - 1)
    _quotients = []
    for _k in range(5):
        _v = _A**_k * sp.Matrix([1, 0])
        _quotients.append((_v.T * _A * _v)[0] / (_v.T * _v)[0])
    assert _quotients == [2, sp.Rational(14, 5), sp.Rational(122, 41), sp.Rational(1094, 365), sp.Rational(9842, 3281)]
    _B = sp.Matrix([[4, 1], [2, 3]])
    _P = sp.Matrix([[1, 1], [-2, 1]])
    _D = sp.diag(2, 5)
    assert _P * _D * _P.inv() == _B
    assert _B**10 == sp.Matrix([[6510758, 3254867], [6509734, 3255891]])
    _M = sp.Matrix([[sp.Rational(9, 10), sp.Rational(2, 10)], [sp.Rational(1, 10), sp.Rational(8, 10)]])
    assert set(_M.eigenvals()) == {1, sp.Rational(7, 10)}
    print(f"sympy : quotients de Rayleigh exacts {_quotients} ; A¹⁰ de (4 1 ; 2 3) vérifiée ; "
          f"valeurs propres des abonnements {sorted(_M.eigenvals())}")
    return


@app.cell
def _(SMOKE, courbes, eta_c, mo, normes, np, plt, sauvegarder_figure):
    # --- Figure, sauvegardée dans figures/ (suffixe -smoke en mode smoke) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for _r, _angles in courbes.items():
        ax1.semilogy(np.arange(1, len(_angles) + 1), _angles, label=f"|λ₂/λ₁| = {_r}".replace(".", ","))
    ax1.set(title="Puissance itérée : écart à la direction propre", xlabel="pas k",
            ylabel="sinus de l’angle (échelle log)", xscale="log")
    ax1.legend(fontsize=8)
    for _facteur, _suite in normes.items():
        ax2.semilogy(_suite, label=f"η = {_facteur} × 2/λ_max".replace(".", ","))
    ax2.set(title=f"Descente sur ½ xᵀAx (2/λ_max = {eta_c:.3f})".replace(".", ","), xlabel="pas",
            ylabel="‖x‖ (échelle log)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    chemins = sauvegarder_figure(fig, "puissance-iteree-smoke" if SMOKE else "puissance-iteree",
                                 mo.notebook_dir() / "figures")
    print(f"Figure : {chemins[0].name}")
    fig  # noqa: B018 (marimo affiche la dernière expression)
    return


@app.cell
def _(comparaisons, cours_descente, duree, mo, normes, prevues, q_cours, vitesses):
    # --- Bilan : seuils chiffrés du README, vérifiés aussi en smoke ---
    echecs = []
    if abs(q_cours[-1] - 3) > 1e-9:
        echecs.append(f"quotient après 12 pas {q_cours[-1]}")
    for _c in comparaisons:
        if _c["ecart_lambda"] > 1e-8:
            echecs.append(f"n = {_c['n']} : écart sur λ {_c['ecart_lambda']:.1e}")
    for _r, _k in vitesses.items():
        if not 0.8 * prevues[_r] <= _k <= 1.2 * prevues[_r] + 2:
            echecs.append(f"rapport {_r} : {_k} pas, {prevues[_r]:.0f} prévus")
    if not vitesses[0.5] < vitesses[0.9] < vitesses[0.99]:
        echecs.append(f"ordre des vitesses {vitesses}")
    if not (normes[0.99][-1] < 0.01 * normes[0.99][0] and normes[1.01][-1] > 10 * normes[1.01][0]):
        echecs.append(f"pas critique : {normes[0.99][-1]:.2e} / {normes[1.01][-1]:.2e}")
    if cours_descente[0.25][1] != -1.5 or abs(cours_descente[0.15][1] + 0.5) > 1e-12:
        echecs.append(f"facteurs du cours {cours_descente}")
    if echecs:
        raise SystemExit("Bilan : ÉCHEC : " + " ; ".join(echecs))
    _ecart = max(_c["ecart_lambda"] for _c in comparaisons)
    print(f"Bilan : OK (quotient {q_cours[-1]:.10f}, écart à eigh {_ecart:.1e}, "
          f"pas {vitesses}, {duree:.2f} s)")
    mo.md(f"**Bilan** : la puissance itérée retrouve λ₁ de `np.linalg.eigh` ; il faut {vitesses[0.5]}, "
          f"{vitesses[0.9]} et {vitesses[0.99]} pas pour les rapports 0,5 ; 0,9 et 0,99.")
    return


if __name__ == "__main__":
    app.run()
