# Notebook 32-tenseurs (chapitre 32, Matrices, produit matriciel, tenseurs et broadcasting).
#
# Les opérations du chapitre en numpy : le produit de la boutique (commandes @ tarifs), la
# règle des dimensions, l’ordre des produits et leur coût, @ contre * (produit terme à terme),
# la transposée d’un produit, le broadcasting (le cas utile, l’erreur, le piège silencieux),
# une couche dense sur un lot et une matrice orthogonale qui conserve les longueurs.
#
#   make nb NUM=32              → éditeur marimo (bouton « Lancer le calcul »)
#   make nb NUM=32 MODE=run     → script complet
#   make nb NUM=32 MODE=smoke   → test rapide (moins de 30 s)

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
    # Tenseurs en numpy : formes, @, *, transposée, broadcasting

    **Chapitre 32, sections 3 à 8.** Avant chaque calcul, écrivez la forme attendue du résultat ;
    le notebook la vérifie par un `assert`. Six expériences :

    1. **La boutique.** Quantités $(3, 2)$ `@` tarifs $(2, 2)$ donne un tableau $(3, 2)$ :
       montant et poids de chaque commande.
    2. **L’ordre des produits.** $(AB)\mathbf{x} = A(B\mathbf{x})$ par associativité, mais le
       premier coûte $n^3 + n^2$ multiplications et le second $2n^2$.
    3. **`@` contre `*`.** Le produit matriciel $AB$ et le produit terme à terme $A \odot B$
       n’ont rien à voir ; $(AB)^\top = B^\top A^\top$, pas $A^\top B^\top$.
    4. **Broadcasting.** $(3, 1) + (1, 4)$ donne $(3, 4)$ ; $(3, 4) + (3,)$ lève une erreur ;
       $(40,) - (40, 1)$ donne en silence un tableau $(40, 40)$ et une erreur quadratique fausse.
    5. **Couche dense sur un lot.** $Y = XW^\top + \mathbf{b}$, de forme $(B, 128)$ pour une
       couche $784 \to 128$ : $100\,480$ paramètres.
    6. **Matrice orthogonale.** $Q^\top Q = I$ et $\lVert Q\mathbf{x} \rVert_2 = \lVert \mathbf{x} \rVert_2$.
    """)


@app.cell
def _(is_smoke, np, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        n_ordre=300 if SMOKE else 1_000,          # taille des matrices de l’expérience 2
        tailles_cout=[10, 30, 100, 300, 1_000, 3_000],   # tailles n de la courbe des coûts
        lot=32 if SMOKE else 256,                 # exemples du lot de l’expérience 5
        couche=(784, 128),                        # entrées, sorties
        n_orthogonale=50 if SMOKE else 500,       # taille de la matrice orthogonale
        n_vecteurs=100 if SMOKE else 2_000,       # vecteurs dont on compare les normes
    )
    rng = np.random.default_rng(SEED)
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SMOKE, config, rng


@app.cell
def _(np):
    # --- Expérience 1 : la boutique (« Pourquoi ça compte ») ---
    commandes = np.array([[2, 1],      # Alice : 2 t-shirts, 1 mug
                          [0, 3],      # Bob : 3 mugs
                          [1, 2]])     # Chloé : 1 t-shirt, 2 mugs
    tarifs = np.array([[15, 0.2],      # t-shirt : 15 €, 0,2 kg
                       [8, 0.4]])      # mug : 8 €, 0,4 kg
    boutique = commandes @ tarifs
    assert boutique.shape == (3, 2)
    print("commandes @ tarifs =\n", boutique)
    try:
        tarifs @ commandes             # (2, 2) @ (3, 2) : dimensions intérieures 2 et 3
    except ValueError as _e:
        erreur_ordre = str(_e)
    print("tarifs @ commandes :", erreur_ordre)
    return boutique, erreur_ordre


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

    t0 = time.perf_counter()
    # --- Expérience 2 : (AB)x contre A(Bx) ---
    _n = config["n_ordre"]
    _A = rng.normal(size=(_n, _n))
    _B = rng.normal(size=(_n, _n))
    _x = rng.normal(size=_n)
    _t = time.perf_counter()
    _gauche = (_A @ _B) @ _x
    duree_gauche = time.perf_counter() - _t
    _t = time.perf_counter()
    _droite = _A @ (_B @ _x)
    duree_droite = time.perf_counter() - _t
    ordre = dict(
        n=_n,
        ecart=float(np.abs(_gauche - _droite).max() / np.abs(_droite).max()),
        mult_gauche=_n**3 + _n**2,
        mult_droite=2 * _n**2,
        duree_gauche=duree_gauche,
        duree_droite=duree_droite,
    )
    # --- Expérience 3 : @ contre *, transposée d’un produit ---
    M = np.array([[1, 2], [3, 4]])
    N = np.array([[5, 6], [7, 8]])
    produits = dict(matriciel=M @ N, terme_a_terme=M * N)
    _P = rng.integers(-5, 6, size=(2, 3))
    _R = rng.integers(-5, 6, size=(3, 4))
    transposee = dict(
        ok=bool(np.array_equal((_P @ _R).T, _R.T @ _P.T)),
        forme_AtBt=(_P.T.shape, _R.T.shape),
    )
    try:
        _P.T @ _R.T                    # (3, 2) @ (4, 3) : impossible
        transposee["AtBt"] = "existe"
    except ValueError:
        transposee["AtBt"] = "n’existe pas"
    # --- Expérience 4 : broadcasting ---
    _col = np.array([[0], [10], [20]])           # forme (3, 1)
    _lig = np.array([[1, 2, 3, 4]])              # forme (1, 4)
    table = _col + _lig
    try:
        np.ones((3, 4)) + np.arange(3)           # (3, 4) + (3,) : 4 et 3 incompatibles
        erreur_broadcast = "aucune"
    except ValueError as _e:
        erreur_broadcast = str(_e)
    _y = rng.normal(size=40)                     # cibles, forme (40,)
    _pred = (_y + rng.normal(scale=0.1, size=40))[:, None]   # prédictions, forme (40, 1)
    _diff = _y - _pred                           # piège : forme (40, 40)
    piege = dict(forme=_diff.shape,
                 mse_fausse=float((_diff**2).mean()),
                 mse_juste=float(((_y - _pred[:, 0]) ** 2).mean()))
    # --- Expérience 5 : couche dense sur un lot ---
    _n_in, _n_out = config["couche"]
    _B_lot = config["lot"]
    W = rng.normal(scale=1 / np.sqrt(_n_in), size=(_n_out, _n_in))    # (n_sortie, n_entrée)
    b = rng.normal(size=_n_out)                                        # (n_sortie,)
    X = rng.normal(size=(_B_lot, _n_in))                               # un exemple par ligne
    Y = X @ W.T + b                                                    # biais diffusé sur le lot
    _une_par_une = np.stack([W @ X[_i] + b for _i in range(_B_lot)])  # y = Wx + b, exemple par exemple
    couche = dict(forme=Y.shape, ecart=float(np.abs(Y - _une_par_une).max()),
                  parametres=W.size + b.size,
                  reseau=(128 * 784 + 128) + (10 * 128 + 10))
    # --- Expérience 6 : matrice orthogonale (décomposition QR d’une matrice au hasard) ---
    _Q, _ = np.linalg.qr(rng.normal(size=(config["n_orthogonale"], config["n_orthogonale"])))
    _V = rng.normal(size=(config["n_orthogonale"], config["n_vecteurs"]))
    orthogonale = dict(
        ecart_identite=float(np.abs(_Q.T @ _Q - np.eye(config["n_orthogonale"])).max()),
        ecart_normes=float(np.abs(np.linalg.norm(_Q @ _V, axis=0) - np.linalg.norm(_V, axis=0)).max()),
    )
    duree = time.perf_counter() - t0
    print(f"(AB)x contre A(Bx), n = {_n} : écart relatif {ordre['ecart']:.1e} ; "
          f"{ordre['mult_gauche']:,} contre {ordre['mult_droite']:,} multiplications ; "
          f"{duree_gauche * 1000:.1f} ms contre {duree_droite * 1000:.2f} ms")
    print("M @ N =\n", produits["matriciel"], "\nM * N =\n", produits["terme_a_terme"])
    print(f"(PR)ᵀ = RᵀPᵀ : {transposee['ok']} ; PᵀRᵀ {transposee['AtBt']} (formes {transposee['forme_AtBt']})")
    print("(3, 1) + (1, 4) =\n", table, "\n(3, 4) + (3,) :", erreur_broadcast)
    print(f"Piège : forme {piege['forme']}, MSE fausse {piege['mse_fausse']:.3f}, juste {piege['mse_juste']:.4f}")
    print(f"Couche : Y de forme {couche['forme']}, écart lot / boucle {couche['ecart']:.1e}, "
          f"{couche['parametres']:,} paramètres ; réseau 784 → 128 → 10 : {couche['reseau']:,}")
    print(f"Orthogonale : |QᵀQ − I| max {orthogonale['ecart_identite']:.1e}, "
          f"écart des normes {orthogonale['ecart_normes']:.1e} ; calcul {duree:.2f} s")
    return couche, duree, erreur_broadcast, ordre, orthogonale, piege, produits, table, transposee


@app.cell
def _(sp):
    # --- Vérification exacte (sympy) sur des matrices symboliques ---
    _A = sp.Matrix(2, 3, sp.symbols("a1:7"))
    _B = sp.Matrix(3, 2, sp.symbols("b1:7"))
    _C = sp.Matrix(2, 2, sp.symbols("c1:5"))
    assert sp.expand((_A * _B).T - _B.T * _A.T) == sp.zeros(2, 2)         # (AB)ᵀ = BᵀAᵀ
    assert sp.expand((_A * _B) * _C - _A * (_B * _C)) == sp.zeros(2, 2)   # associativité
    _u = sp.Matrix(sp.symbols("u1:4"))
    _v = sp.Matrix(sp.symbols("v1:3"))
    assert all((_u * _v.T)[i, j] == _u[i] * _v[j] for i in range(3) for j in range(2))   # (uvᵀ)ᵢⱼ = uᵢvⱼ
    _M = sp.Matrix([[1, 2], [0, 1]])
    _N = sp.Matrix([[1, 0], [3, 1]])
    assert _M * _N == sp.Matrix([[7, 2], [3, 1]]) and _N * _M == sp.Matrix([[1, 2], [3, 7]])
    _Q = sp.Matrix([[sp.Rational(3, 5), -sp.Rational(4, 5)], [sp.Rational(4, 5), sp.Rational(3, 5)]])
    assert _Q.T * _Q == sp.eye(2)
    print("sympy : (AB)ᵀ = BᵀAᵀ, (AB)C = A(BC), (uvᵀ)ᵢⱼ = uᵢvⱼ, AB ≠ BA sur l’exemple du cours, "
          "QᵀQ = I pour la matrice (3/5, −4/5 ; 4/5, 3/5)")
    return


@app.cell
def _(SMOKE, config, mo, np, plt, sauvegarder_figure, table):
    # --- Figure (suffixe -smoke : n’écrase pas celle du run complet) ---
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    _n = np.array(config["tailles_cout"], dtype=float)
    _ax1.loglog(_n, _n**3 + _n**2, marker="o", label="(AB)x : n³ + n²")
    _ax1.loglog(_n, 2 * _n**2, marker="s", label="A(Bx) : 2n²")
    _ax1.set(xlabel="taille n des matrices", ylabel="multiplications",
             title="Même résultat, coûts différents")
    _ax1.legend()
    _ax2.imshow(table, cmap="Blues")
    for (_i, _j), _val in np.ndenumerate(table):
        _ax2.text(_j, _i, str(_val), ha="center", va="center", color="white" if _val > 10 else "black")
    _ax2.set(xticks=range(4), yticks=range(3), title="(3, 1) + (1, 4) → (3, 4)")
    _ax2.grid(False)
    _chemins = sauvegarder_figure(_fig, "tenseurs-smoke" if SMOKE else "tenseurs",
                                 mo.notebook_dir() / "figures")
    print(f"Figure : {_chemins[0].name}")
    _fig  # noqa: B018 (marimo affiche la dernière expression)
    return


@app.cell
def _(boutique, config, couche, duree, erreur_broadcast, erreur_ordre, mo, np, ordre, orthogonale,
      piege, produits, table, transposee):
    # --- Bilan : seuils du README, vérifiés aussi en smoke ---
    _echecs = []
    if not np.allclose(boutique, [[38, 0.8], [24, 1.2], [31, 1.0]], atol=1e-12):
        _echecs.append(f"boutique {boutique.tolist()} (attendu 38 € 0,8 kg ; 24 € 1,2 kg ; 31 € 1 kg)")
    if "mismatch" not in erreur_ordre:
        _echecs.append(f"tarifs @ commandes aurait dû échouer : {erreur_ordre}")
    if ordre["ecart"] > 1e-10:
        _echecs.append(f"(AB)x et A(Bx) diffèrent : écart relatif {ordre['ecart']:.1e}")
    if ordre["mult_gauche"] < 100 * ordre["mult_droite"]:
        _echecs.append("rapport des coûts inférieur à 100")
    if not (np.array_equal(produits["matriciel"], [[19, 22], [43, 50]])
            and np.array_equal(produits["terme_a_terme"], [[5, 12], [21, 32]])):
        _echecs.append("produits @ et * de l’exemple du cours")
    if not transposee["ok"] or transposee["AtBt"] != "n’existe pas":
        _echecs.append(f"transposée d’un produit : {transposee}")
    if table.shape != (3, 4) or table[2, 3] != 24:
        _echecs.append(f"broadcasting (3, 1) + (1, 4) : forme {table.shape}")
    if "could not be broadcast" not in erreur_broadcast:
        _echecs.append(f"(3, 4) + (3,) aurait dû échouer : {erreur_broadcast}")
    if piege["forme"] != (40, 40) or piege["mse_fausse"] < 10 * piege["mse_juste"]:
        _echecs.append(f"piège (40,) − (40, 1) : {piege}")
    if couche["forme"] != (config["lot"], 128) or couche["ecart"] > 1e-10:
        _echecs.append(f"couche dense : forme {couche['forme']}, écart {couche['ecart']:.1e}")
    if couche["parametres"] != 100_480 or couche["reseau"] != 101_770:
        _echecs.append(f"paramètres {couche['parametres']} et {couche['reseau']} (attendu 100 480 et 101 770)")
    if orthogonale["ecart_identite"] > 1e-10 or orthogonale["ecart_normes"] > 1e-10:
        _echecs.append(f"matrice orthogonale : {orthogonale}")
    if _echecs:
        raise SystemExit("Bilan : ÉCHEC : " + " ; ".join(_echecs))
    print(f"Bilan : OK (boutique 38/0,8, 24/1,2, 31/1 ; (AB)x = A(Bx) à {ordre['ecart']:.0e} près, "
          f"{ordre['mult_gauche'] // ordre['mult_droite']} fois plus de multiplications ; @ et * ; (AB)ᵀ = BᵀAᵀ ; "
          f"broadcasting (3, 4), erreur et piège (40, 40) ; couche {couche['forme']}, 100 480 paramètres ; "
          f"QᵀQ = I ; {duree:.2f} s)")
    mo.md(f"**Bilan** : $(AB)\\mathbf{{x}}$ coûte {ordre['mult_gauche']:,} multiplications, "
          f"$A(B\\mathbf{{x}})$ seulement {ordre['mult_droite']:,} ; le piège $(40,) - (40, 1)$ donne une "
          f"erreur quadratique de {piege['mse_fausse']:.2f} au lieu de {piege['mse_juste']:.4f}.".replace(",", " "))
    return


if __name__ == "__main__":
    app.run()
