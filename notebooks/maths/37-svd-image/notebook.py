# Notebook 37-svd-image (chapitre 37, SVD, approximation de rang faible, pseudo-inverse, ACP).
#
# Compresser une image en niveaux de gris générée par sa SVD : reconstruction de rang k,
# part de l’énergie gardée, erreur de Frobenius et erreur spectrale (théorème d’Eckart et
# Young), comparaison avec des approximations de rang k tirées au hasard. Puis l’ACP d’un
# jeu de « chiffres » 8 × 8 générés (aucun téléchargement) : SVD des données centrées
# contre vecteurs propres de la matrice de covariance C = (1/n) XcᵀXc. Vérifications exactes
# par sympy. numpy seul (pas de scipy, pas de scikit-learn).
#
#   make nb NUM=37              → éditeur marimo (bouton « Lancer le calcul »)
#   make nb NUM=37 MODE=run     → script complet
#   make nb NUM=37 MODE=smoke   → test rapide (moins de 30 s)

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
    # Compresser une image par la SVD, et l’ACP par la SVD

    **Chapitre 37, sections 3 et 6.** Toute matrice s’écrit $A = U\Sigma V^\top$, et la somme
    des $k$ premiers termes

    $$
    A_k = \sigma_1\,\mathbf{u}_1\mathbf{v}_1^\top + \dots + \sigma_k\,\mathbf{u}_k\mathbf{v}_k^\top
    $$

    est la meilleure approximation de rang $k$ (Eckart et Young) :
    $\lVert A - A_k \rVert_F = \sqrt{\sigma_{k+1}^2 + \dots}$ et $\lVert A - A_k \rVert_2 = \sigma_{k+1}$.

    Quatre expériences :

    1. une image en niveaux de gris générée (disque, bandes, dégradé, un peu de bruit) :
       reconstruction de rang $k$, nombres stockés $k(m + n + 1)$, énergie gardée et erreur ;
    2. Eckart et Young à l’épreuve : des approximations de rang $k$ « au hasard » (projection
       sur $k$ directions tirées au hasard) font toujours moins bien que $A_k$ ;
    3. l’ACP d’un jeu de chiffres $8 \times 8$ générés : la SVD du tableau centré donne les
       mêmes axes et les mêmes variances $\lambda_j = \sigma_j^2 / n$ que la matrice de
       covariance, et l’on compte les composantes qu’il faut pour 90 % de la variance ;
    4. sympy refait exactement la SVD de $\begin{pmatrix} 2 & 2 \\ -1 & 2 \end{pmatrix}$ du cours.
    """)


@app.cell
def _(is_smoke, np, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        taille=96 if SMOKE else 256,          # côté de l’image carrée générée
        rangs=[1, 2, 5, 10, 20, 50],          # rangs reconstruits et affichés
        n_hasard=20 if SMOKE else 200,        # approximations de rang k tirées au hasard
        n_chiffres=300 if SMOKE else 2_000,   # images 8 × 8 de l’expérience 3
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

    # 1. Image générée : dégradé, disque, bandes obliques, bruit léger ; valeurs dans [0, 1]
    n = config["taille"]
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1)
    image = 0.3 + 0.4 * xx
    image = np.where((xx - 0.35) ** 2 + (yy - 0.4) ** 2 < 0.06, 0.95, image)
    image = np.where(((xx + yy) * 6 % 1 < 0.35) & (yy > 0.65), 0.1, image)
    image = np.clip(image + 0.03 * rng.normal(size=image.shape), 0, 1)

    U, S, Vh = np.linalg.svd(image)
    energie_totale = float(np.sum(S**2))
    reconstructions, mesures = {}, []
    for k in config["rangs"]:
        A_k = (U[:, :k] * S[:k]) @ Vh[:k]
        reconstructions[k] = A_k
        err_F = float(np.linalg.norm(image - A_k, "fro"))
        err_2 = float(np.linalg.norm(image - A_k, 2))
        mesures.append(dict(
            k=k, stockes=k * (2 * n + 1), part_stockee=k * (2 * n + 1) / n**2,
            energie=float(np.sum(S[:k] ** 2)) / energie_totale,
            err_F=err_F, err_F_theorie=float(np.sqrt(np.sum(S[k:] ** 2))),
            err_2=err_2, err_2_theorie=float(S[k]),
            err_relative=err_F / float(np.linalg.norm(image, "fro")),
        ))
    ecart_eckart_young = max(max(abs(m["err_F"] - m["err_F_theorie"]), abs(m["err_2"] - m["err_2_theorie"]))
                             for m in mesures)

    # 2. Approximations de rang k au hasard : Q (n × k) orthonormée tirée au hasard, B = Q Qᵀ A
    #    (la meilleure matrice dont les colonnes sont dans l’espace engendré par Q)
    k_test = 10
    erreur_svd = float(np.sqrt(np.sum(S[k_test:] ** 2)))
    erreurs_hasard = []
    for _i in range(config["n_hasard"]):
        Q, _r = np.linalg.qr(rng.normal(size=(n, k_test)))
        erreurs_hasard.append(float(np.linalg.norm(image - Q @ (Q.T @ image), "fro")))
    # même chose, mais en partant de directions « presque bonnes » (U_k bruité)
    erreurs_proches = []
    for _i in range(config["n_hasard"]):
        Q, _r = np.linalg.qr(U[:, :k_test] + 0.05 * rng.normal(size=(n, k_test)))
        erreurs_proches.append(float(np.linalg.norm(image - Q @ (Q.T @ image), "fro")))

    # 3. ACP de « chiffres » 8 × 8 générés : 10 gabarits faits de traits, décalés et bruités
    def gabarit(segments):
        g = np.zeros((8, 8))
        for (l0, c0, l1, c1) in segments:
            for t in np.linspace(0, 1, 12):
                g[round(l0 + t * (l1 - l0)), round(c0 + t * (c1 - c0))] = 1.0
        return g

    traits = [
        [(1, 2, 1, 5), (1, 5, 6, 5), (6, 5, 6, 2), (6, 2, 1, 2)],  # 0
        [(1, 4, 6, 4)],                                            # 1
        [(1, 2, 1, 5), (1, 5, 3, 5), (3, 5, 6, 2), (6, 2, 6, 5)],  # 2
        [(1, 2, 1, 5), (1, 5, 6, 5), (3, 3, 3, 5), (6, 2, 6, 5)],  # 3
        [(1, 2, 4, 2), (4, 2, 4, 5), (1, 5, 6, 5)],                # 4
        [(1, 5, 1, 2), (1, 2, 3, 2), (3, 2, 3, 5), (3, 5, 6, 5), (6, 5, 6, 2)],  # 5
        [(1, 5, 1, 2), (1, 2, 6, 2), (6, 2, 6, 5), (6, 5, 3, 5), (3, 5, 3, 2)],  # 6
        [(1, 2, 1, 5), (1, 5, 6, 3)],                              # 7
        [(1, 2, 1, 5), (1, 5, 6, 5), (6, 5, 6, 2), (6, 2, 1, 2), (3, 2, 3, 5)],  # 8
        [(3, 5, 3, 2), (3, 2, 1, 2), (1, 2, 1, 5), (1, 5, 6, 5)],  # 9
    ]
    gabarits = [gabarit(s) for s in traits]
    lignes = []
    for _i in range(config["n_chiffres"]):
        g = gabarits[_i % 10]
        g = np.roll(g, rng.integers(-1, 2), axis=1)
        lignes.append((16 * g + rng.normal(scale=2.0, size=(8, 8))).ravel())
    X = np.array(lignes)
    n_ind = X.shape[0]
    Xc = X - X.mean(axis=0)
    _U, sing, Vh_acp = np.linalg.svd(Xc, full_matrices=False)
    lam_svd = sing**2 / n_ind
    C = Xc.T @ Xc / n_ind
    lam_C, vec_C = np.linalg.eigh(C)
    lam_C, vec_C = lam_C[::-1], vec_C[:, ::-1]
    ecart_lambda = float(np.max(np.abs(lam_svd - lam_C)) / lam_C[0])
    ecart_axes = float(max(1 - abs(Vh_acp[j] @ vec_C[:, j]) for j in range(5)))
    parts = np.cumsum(lam_svd) / np.sum(lam_svd)
    k90 = int(np.argmax(parts >= 0.9) + 1)
    scores = Xc @ Vh_acp[:2].T
    # convention n − 1 (numpy, scikit-learn) : mêmes parts
    lam_n1 = sing**2 / (n_ind - 1)
    ecart_parts_n1 = float(np.max(np.abs(np.cumsum(lam_n1) / np.sum(lam_n1) - parts)))

    duree = time.perf_counter() - t0
    print(f"Calcul : {duree:.2f} s")
    print(f"1. image {n} × {n}, σ₁ = {S[0]:.2f}, σ₁₀ = {S[9]:.3f}, σ₅₀ = {S[49]:.4f}")
    for m in mesures:
        print(f"   k = {m['k']:2d} : {m['stockes']:6d} nombres ({100 * m['part_stockee']:.1f} %), "
              f"énergie {100 * m['energie']:.3f} %, erreur relative {100 * m['err_relative']:.2f} %")
    print(f"   écart maximal à Eckart-Young (Frobenius et spectrale) : {ecart_eckart_young:.1e}")
    print(f"2. rang {k_test} : SVD {erreur_svd:.3f} ; au hasard, au mieux {min(erreurs_hasard):.3f} ; "
          f"directions proches de U_k, au mieux {min(erreurs_proches):.3f}")
    print(f"3. {n_ind} chiffres 8 × 8 : écart relatif σ²/n contre valeurs propres de C {ecart_lambda:.1e}, "
          f"axes {ecart_axes:.1e} ; part des 2 premiers axes {100 * parts[1]:.1f} %, {k90} axes pour 90 %")
    return (C, S, X, Xc, duree, ecart_axes, ecart_eckart_young, ecart_lambda, ecart_parts_n1, erreur_svd,
            erreurs_hasard, erreurs_proches, image, k90, k_test, mesures, parts, reconstructions, scores)


@app.cell
def _(sp):
    # --- Vérification exacte (sympy) : la SVD du cours ---
    _A = sp.Matrix([[2, 2], [-1, 2]])
    _AtA = _A.T * _A
    assert _AtA == sp.Matrix([[5, 2], [2, 8]])
    assert _AtA.eigenvals() == {9: 1, 4: 1}
    _U = sp.Matrix([[2, -1], [1, 2]]) / sp.sqrt(5)
    _V = sp.Matrix([[1, -2], [2, 1]]) / sp.sqrt(5)
    _S = sp.diag(3, 2)
    assert sp.simplify(_U * _S * _V.T - _A) == sp.zeros(2, 2)
    assert sp.simplify(_U.T * _U) == sp.eye(2) and sp.simplify(_V.T * _V) == sp.eye(2)
    assert _U.det() == 1 and _V.det() == 1
    _A1 = 3 * _U[:, 0] * _V[:, 0].T
    assert sp.simplify(_A1 - sp.Matrix([[6, 12], [3, 6]]) / 5) == sp.zeros(2, 2)
    assert sp.sqrt(sum(x**2 for x in (_A - _A1))) == 2
    _lam = sp.symbols("lambda")
    assert sp.discriminant((_A - _lam * sp.eye(2)).det(), _lam) == -8
    print("sympy : AᵀA = (5 2 ; 2 8), valeurs propres 9 et 4, A = UΣVᵀ exacte avec σ = 3 et 2, "
          "‖A − A₁‖_F = 2, aucune valeur propre réelle (discriminant −8)")
    return


@app.cell
def _(S, SMOKE, image, k90, mesures, mo, np, parts, plt, reconstructions, sauvegarder_figure):
    # --- Figure, sauvegardée dans figures/ (suffixe -smoke en mode smoke) ---
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    vues = [("Original", image)] + [(f"Rang {k}", reconstructions[k]) for k in (1, 5, 20)]
    for ax, (titre, img) in zip([axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]], vues, strict=True):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set(title=titre, xticks=[], yticks=[])
    ax = axes[1, 1]
    ks = np.arange(1, len(S) + 1)
    ax.semilogy(ks, S, label="valeurs singulières σₖ")
    ax.set(title="Valeurs singulières de l’image", xlabel="k", ylabel="σₖ (échelle log)")
    for _m in mesures:
        ax.annotate(f"{100 * _m['energie']:.1f} %".replace(".", ","), (_m["k"], S[_m["k"] - 1]), fontsize=7)
    ax = axes[1, 2]
    ax.bar(np.arange(1, 21), parts[:20] * 100, color="tab:blue", alpha=0.7)
    ax.axhline(90, color="tab:red", linestyle="--", label="90 %")
    ax.set(title=f"ACP des chiffres 8 × 8 : {k90} axes pour 90 %", xlabel="nombre k d’axes gardés",
           ylabel="part cumulée de la variance (%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    chemins = sauvegarder_figure(fig, "svd-image-smoke" if SMOKE else "svd-image", mo.notebook_dir() / "figures")
    print(f"Figure : {chemins[0].name}")
    fig  # noqa: B018 (marimo affiche la dernière expression)
    return


@app.cell
def _(duree, ecart_axes, ecart_eckart_young, ecart_lambda, ecart_parts_n1, erreur_svd, erreurs_hasard,
      erreurs_proches, k90, mesures, mo, parts):
    # --- Bilan : seuils chiffrés du README, vérifiés aussi en smoke ---
    echecs = []
    if ecart_eckart_young > 1e-8:
        echecs.append(f"Eckart-Young : écart {ecart_eckart_young:.1e}")
    if not all(e >= erreur_svd - 1e-9 for e in erreurs_hasard + erreurs_proches):
        echecs.append("une approximation au hasard bat la SVD")
    if min(erreurs_hasard) < 2 * erreur_svd:
        echecs.append(f"hasard trop proche de la SVD : {min(erreurs_hasard):.3f} contre {erreur_svd:.3f}")
    _m20 = next(m for m in mesures if m["k"] == 20)
    if _m20["energie"] < 0.99:
        echecs.append(f"énergie au rang 20 : {_m20['energie']:.4f}")
    _energies = [m["energie"] for m in mesures]
    if _energies != sorted(_energies):
        echecs.append("énergie non croissante en k")
    if ecart_lambda > 1e-10 or ecart_axes > 1e-8:
        echecs.append(f"ACP : SVD contre covariance {ecart_lambda:.1e} / {ecart_axes:.1e}")
    if ecart_parts_n1 > 1e-12:
        echecs.append(f"parts différentes avec n − 1 : {ecart_parts_n1:.1e}")
    if not 3 <= k90 <= 60 or parts[1] >= 0.9:
        echecs.append(f"ACP : {k90} axes pour 90 %, deux axes {parts[1]:.3f}")
    if echecs:
        raise SystemExit("Bilan : ÉCHEC : " + " ; ".join(echecs))
    print(f"Bilan : OK (Eckart-Young à {ecart_eckart_young:.1e} près, énergie au rang 20 "
          f"{100 * _m20['energie']:.2f} %, hasard ≥ {min(erreurs_hasard) / erreur_svd:.1f} fois l’erreur SVD, "
          f"ACP par SVD = covariance à {ecart_lambda:.1e} près, {k90} axes pour 90 %, {duree:.2f} s)")
    _energie_txt = f"{100 * _m20['energie']:.2f}".replace(".", ",")
    mo.md(f"**Bilan** : au rang 20, l’image garde {_energie_txt} % de son énergie ; "
          f"l’ACP des chiffres demande {k90} axes pour 90 % de la variance.")
    return


if __name__ == "__main__":
    app.run()
