# Notebook 31-plongements-jouets (chapitre 31, Vecteurs de ℝⁿ, normes, similarité cosinus,
# projections).
#
# Des « plongements » jouets : des textes de trois thèmes résumés par neuf comptages de mots,
# de longueurs très différentes. Plus proche voisin par distance euclidienne contre similarité
# cosinus ; vecteurs normalisés (les deux classements coïncident) ; codage one-hot (toutes les
# distances valent √2) ; k-means écrit en numpy, dont l’inertie ne remonte jamais.
#
#   make nb NUM=31              → éditeur marimo (bouton « Lancer le calcul »)
#   make nb NUM=31 MODE=run     → script complet
#   make nb NUM=31 MODE=smoke   → test rapide (moins de 30 s)

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
    # Plongements jouets : distance ou cosinus ?

    **Chapitre 31, sections 5 et 6.** Chaque texte est résumé par un vecteur de $\R^9$ : le
    nombre de fois où il emploie neuf mots, trois par thème (Python, CSS, cuisine). Un texte
    long a de grands comptages, un texte court de petits : la **longueur** du vecteur dit la
    taille du texte, sa **direction** dit son thème. Quatre expériences :

    1. **Plus proche voisin.** On range chaque texte d’un grand lot de requêtes dans le thème
       de son plus proche voisin parmi 12 textes de référence, d’abord par la distance
       euclidienne $\lVert \mathbf{u} - \mathbf{v} \rVert_2$, puis par la similarité cosinus
       $\frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\lVert \mathbf{u} \rVert_2 \lVert \mathbf{v} \rVert_2}$.
    2. **Normaliser.** Pour des vecteurs de norme 1,
       $\lVert \mathbf{u} - \mathbf{v} \rVert_2^2 = 2 - 2\cos(\mathbf{u}, \mathbf{v})$ : la distance et le
       cosinus donnent alors le même voisin.
    3. **One-hot.** Le mot numéro $i$ d’un vocabulaire devient le vecteur $\mathbf{e}_i$ (un 1,
       des 0) : deux mots différents sont toujours à la distance $\sqrt 2$, de cosinus 0.
    4. **k-means.** Trois centres, deux étapes répétées (affecter, puis moyenner) : l’inertie
       $J = \sum_i \lVert \mathbf{x}_i - \boldsymbol{\mu}_{c(i)} \rVert_2^2$ ne remonte jamais.
    """)


@app.cell
def _(is_smoke, np, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        themes=["Python", "CSS", "cuisine"],
        dimension=9,                               # 3 mots par thème
        n_reference=4,                             # textes de référence par thème
        n_requetes=600 if SMOKE else 30_000,       # requêtes (toutes thèmes confondus)
        longueur=(0.2, 10.0),                      # facteur de longueur, tiré en échelle log
        bruit=0.6,
        vocabulaire=50 if SMOKE else 2_000,        # taille du vocabulaire one-hot
        n_kmeans_par_theme=60 if SMOKE else 2_000,
        iterations_kmeans=15,
    )
    rng = np.random.default_rng(SEED)
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SMOKE, config, rng


@app.cell
def _(config, np):
    # Profil moyen de chaque thème : ses trois mots fréquents (2), les autres rares (0,3).
    profils = np.full((3, config["dimension"]), 0.3)
    for _j in range(3):
        profils[_j, 3 * _j:3 * _j + 3] = 2.0

    def textes(rng, n_par_theme):
        """n_par_theme textes par thème : profil bruité (comptages ≥ 0) × longueur aléatoire."""
        _X, _y = [], []
        lo, hi = np.log(config["longueur"][0]), np.log(config["longueur"][1])
        for _j in range(3):
            for _ in range(n_par_theme):
                _L = np.exp(rng.uniform(lo, hi))
                _v = np.clip(profils[_j] + rng.normal(scale=config["bruit"], size=config["dimension"]), 0, None)
                if not _v.any():
                    _v[3 * _j] = 1.0   # jamais de vecteur nul : son cosinus n’existe pas
                _X.append(_L * _v)
                _y.append(_j)
        return np.array(_X), np.array(_y)

    def normaliser(X):
        return X / np.linalg.norm(X, axis=-1, keepdims=True)

    def k_means(X, k, iterations, rng):
        """Algorithme de Lloyd : centres de départ tirés parmi les points ; inertie à chaque étape."""
        _centres = X[rng.choice(len(X), size=k, replace=False)].copy()
        _inerties = []
        for _ in range(iterations):
            _d2 = ((X[:, None, :] - _centres[None, :, :]) ** 2).sum(axis=2)
            _c = _d2.argmin(axis=1)                                   # affecter
            _inerties.append(float(_d2[np.arange(len(X)), _c].sum()))
            for _j in range(k):                                       # moyenner
                if (_c == _j).any():
                    _centres[_j] = X[_c == _j].mean(axis=0)
        return _centres, _c, np.array(_inerties)

    return k_means, normaliser, textes


@app.cell
def _(np):
    # --- L’exemple du cours (« Pourquoi ça compte ») : comptages de python, css, matrice, recette
    A = np.array([4, 0, 2, 0.0])
    B = np.array([12, 0, 6, 0.0])
    C = np.array([0, 1, 0, 1.0])

    def cosinus(u, v):
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))

    blog = dict(dAB=float(np.linalg.norm(A - B)), dAC=float(np.linalg.norm(A - C)),
                cAB=cosinus(A, B), cAC=cosinus(A, C))
    print(f"d(A, B) = {blog['dAB']:.3f} ; d(A, C) = {blog['dAC']:.3f} ; "
          f"cos(A, B) = {blog['cAB']:.3f} ; cos(A, C) = {blog['cAC']:.3f}")
    return (blog,)


@app.cell
def _(mo):
    # Le bouton empêche le calcul de se relancer à chaque modification d’une cellule amont.
    bouton = mo.ui.run_button(label="Lancer le calcul")
    bouton  # noqa: B018 (marimo affiche la dernière expression)
    return (bouton,)


@app.cell
def _(bouton, config, k_means, mo, normaliser, np, rng, textes, time):
    # En script (make nb MODE=run|smoke), personne ne clique : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer le calcul ».*"))

    t0 = time.perf_counter()
    # --- Expérience 1 : plus proche voisin parmi 12 textes de référence ---
    R, yR = textes(rng, config["n_reference"])
    Q, yQ = textes(rng, config["n_requetes"] // 3)
    _d2 = (Q * Q).sum(1)[:, None] - 2 * Q @ R.T + (R * R).sum(1)[None, :]   # ‖q − r‖² développé
    voisin_dist = _d2.argmin(axis=1)
    _cos = normaliser(Q) @ normaliser(R).T
    voisin_cos = _cos.argmax(axis=1)
    justesse = dict(distance=float((yR[voisin_dist] == yQ).mean()),
                    cosinus=float((yR[voisin_cos] == yQ).mean()))
    # --- Expérience 2 : vecteurs normalisés, distance et cosinus donnent le même voisin ---
    Qn, Rn = normaliser(Q), normaliser(R)
    _d2n = ((Qn[:, None, :] - Rn[None, :, :]) ** 2).sum(axis=2)
    ecart_identite = float(np.abs(_d2n - (2 - 2 * (Qn @ Rn.T))).max())
    meme_voisin = bool((_d2n.argmin(axis=1) == voisin_cos).all())
    justesse["distance normalisée"] = float((yR[_d2n.argmin(axis=1)] == yQ).mean())
    # --- Expérience 3 : one-hot ---
    _V = config["vocabulaire"]
    _E = np.eye(_V)
    _G = _E @ _E.T                                                  # produits scalaires
    _D2 = np.diag(_G)[:, None] - 2 * _G + np.diag(_G)[None, :]
    _hors_diag = ~np.eye(_V, dtype=bool)
    onehot = dict(dmin=float(np.sqrt(_D2[_hors_diag].min())), dmax=float(np.sqrt(_D2[_hors_diag].max())),
                  cosmax=float(np.abs(_G[_hors_diag]).max()))
    # --- Expérience 4 : k-means sur les textes normalisés ---
    Xk, yk = textes(rng, config["n_kmeans_par_theme"])
    Xk = normaliser(Xk)
    centres, groupes, inerties = k_means(Xk, 3, config["iterations_kmeans"], rng)
    # Pureté : part des points dont le groupe a pour thème majoritaire leur propre thème.
    _majoritaire = {g: np.bincount(yk[groupes == g], minlength=3).argmax() for g in np.unique(groupes)}
    purete = float(np.mean([_majoritaire[g] == t for g, t in zip(groupes, yk, strict=True)]))
    duree = time.perf_counter() - t0
    print(f"Plus proche voisin : justesse {justesse}")
    print(f"Normalisés : écart max à 2 − 2 cos = {ecart_identite:.2e} ; même voisin : {meme_voisin}")
    print(f"One-hot ({_V} mots) : distances de {onehot['dmin']:.6f} à {onehot['dmax']:.6f} ; "
          f"|cos| max {onehot['cosmax']}")
    print(f"k-means : inertie {inerties[0]:.2f} → {inerties[-1]:.2f} en {len(inerties)} étapes ; pureté {purete:.3f} ; "
          f"calcul {duree:.2f} s")
    return duree, ecart_identite, inerties, justesse, meme_voisin, onehot, purete


@app.cell
def _(sp):
    # --- Vérification exacte (sympy), en dimension 3 ---
    _u = sp.Matrix(sp.symbols("u1:4", real=True))
    _v = sp.Matrix(sp.symbols("v1:4", real=True))

    def _ps(a, b):
        return (a.T * b)[0]

    # ‖u − v‖² = ‖u‖² − 2⟨u, v⟩ + ‖v‖²
    assert sp.expand(_ps(_u - _v, _u - _v) - (_ps(_u, _u) - 2 * _ps(_u, _v) + _ps(_v, _v))) == 0
    # Identité de Lagrange : ‖u‖²‖v‖² − ⟨u, v⟩² est une somme de carrés, d’où Cauchy-Schwarz.
    _carres = sum((_u[i] * _v[j] - _u[j] * _v[i]) ** 2 for i in range(3) for j in range(i + 1, 3))
    assert sp.expand(_ps(_u, _u) * _ps(_v, _v) - _ps(_u, _v) ** 2 - _carres) == 0
    # Projeté sur une droite : u − p(u) orthogonal à v.
    _p = (_ps(_u, _v) / _ps(_v, _v)) * _v
    assert sp.simplify(_ps(_u - _p, _v)) == 0
    print("sympy : ‖u − v‖² = ‖u‖² − 2⟨u, v⟩ + ‖v‖² ; ‖u‖²‖v‖² − ⟨u, v⟩² = Σ (uᵢvⱼ − uⱼvᵢ)² ≥ 0 ; "
          "⟨u − p(u), v⟩ = 0")
    return


@app.cell
def _(SMOKE, inerties, justesse, mo, plt, sauvegarder_figure):
    # --- Figure (suffixe -smoke : n’écrase pas celle du run complet) ---
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    _noms = list(justesse)
    _ax1.bar(range(len(_noms)), [100 * justesse[n] for n in _noms], color=["#d97706", "#2563eb", "#059669"])
    _ax1.set_xticks(range(len(_noms)), ["distance", "cosinus", "distance\n(normalisés)"])
    _ax1.set(ylabel="bons thèmes (%)", ylim=(0, 105), title="Plus proche voisin parmi 12 textes")
    for _i, _n in enumerate(_noms):
        _ax1.text(_i, 100 * justesse[_n] + 1.5, f"{100 * justesse[_n]:.1f} %".replace(".", ","), ha="center")
    _ax2.plot(range(1, len(inerties) + 1), inerties, marker="o")
    _ax2.set(xlabel="étape (affecter puis moyenner)", ylabel="inertie J", title="k-means : l’inertie ne remonte pas")
    _chemins = sauvegarder_figure(_fig, "plongements-jouets-smoke" if SMOKE else "plongements-jouets",
                                 mo.notebook_dir() / "figures")
    print(f"Figure : {_chemins[0].name}")
    _fig  # noqa: B018 (marimo affiche la dernière expression)
    return


@app.cell
def _(blog, duree, ecart_identite, inerties, justesse, meme_voisin, mo, np, onehot, purete):
    # --- Bilan : seuils du README, vérifiés aussi en smoke ---
    _echecs = []
    if abs(blog["dAB"] - np.sqrt(80)) > 1e-12 or abs(blog["dAC"] - np.sqrt(22)) > 1e-12:
        _echecs.append(f"distances du blog {blog['dAB']:.4f} et {blog['dAC']:.4f} (attendu √80 et √22)")
    if abs(blog["cAB"] - 1) > 1e-12 or abs(blog["cAC"]) > 1e-12:
        _echecs.append(f"cosinus du blog {blog['cAB']} et {blog['cAC']} (attendu 1 et 0)")
    if justesse["cosinus"] < 0.97:
        _echecs.append(f"justesse du cosinus {justesse['cosinus']:.3f} (attendu ≥ 0,97)")
    if justesse["cosinus"] - justesse["distance"] < 0.10:
        _echecs.append(f"écart cosinus − distance {justesse['cosinus'] - justesse['distance']:.3f} (attendu ≥ 0,10)")
    if not meme_voisin or ecart_identite > 1e-12:
        _echecs.append(f"vecteurs normalisés : même voisin {meme_voisin}, écart {ecart_identite:.1e}")
    _r2 = np.sqrt(2)
    if abs(onehot["dmin"] - _r2) > 1e-12 or abs(onehot["dmax"] - _r2) > 1e-12 or onehot["cosmax"] != 0:
        _echecs.append(f"one-hot : distances {onehot['dmin']} à {onehot['dmax']}, cos max {onehot['cosmax']}")
    if np.any(np.diff(inerties) > 1e-9):
        _echecs.append("l’inertie de k-means remonte")
    if purete < 0.95:
        _echecs.append(f"pureté de k-means {purete:.3f} (attendu ≥ 0,95)")
    if _echecs:
        raise SystemExit("Bilan : ÉCHEC : " + " ; ".join(_echecs))
    print(f"Bilan : OK (blog √80 ≈ 8,94 et √22 ≈ 4,69, cosinus 1 et 0 ; justesse distance "
          f"{justesse['distance']:.3f}, cosinus {justesse['cosinus']:.3f}, "
          f"normalisés {justesse['distance normalisée']:.3f} ; "
          f"one-hot √2 partout ; k-means {inerties[0]:.2f} → {inerties[-1]:.2f}, pureté {purete:.3f} ; {duree:.2f} s)")
    mo.md(f"**Bilan** : le plus proche voisin trouve le bon thème dans {100 * justesse['distance']:.1f} % des cas "
          f"par la distance, {100 * justesse['cosinus']:.1f} % par le cosinus ; one-hot : toutes les distances valent "
          f"$\\sqrt 2$ ; k-means : inertie de {inerties[0]:.2f} à {inerties[-1]:.2f}, pureté {100 * purete:.1f} %.")
    return


if __name__ == "__main__":
    app.run()
