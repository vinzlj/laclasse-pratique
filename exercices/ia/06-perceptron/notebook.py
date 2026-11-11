# Exercice 06 : neurone formel, règle de Hebb, perceptron, XOR.
#
#   make ex NUM=06              → éditeur marimo (interactif, bouton « Lancer »)
#   make ex NUM=06 MODE=run     → script complet (vérifie les seuils du README)
#   make ex NUM=06 MODE=smoke   → test rapide, < 30 s
#
# Tout est écrit en numpy « from scratch », puis comparé à scikit-learn
# (sklearn.linear_model.Perceptron). Aucun téléchargement, calcul sur CPU (quelques secondes).

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from ia_commun import PALETTE, is_smoke, sauvegarder_figure, seed_everything, style_commun

    return (
        PALETTE,
        is_smoke,
        mo,
        np,
        plt,
        sauvegarder_figure,
        seed_everything,
        style_commun,
        time,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Exercice 06 : neurone formel et perceptron

    **Objectif.** Programmer les premiers neurones artificiels et voir, chiffres à
    l'appui, ce qu'ils savent apprendre et ce qu'ils ne peuvent pas apprendre.

    1. **Neurone formel** (McCulloch et Pitts, 1943) : entrées binaires, seuil, entrée
       inhibitrice. On câble ET, OU, NON à la main : aucun apprentissage.
    2. **Règle de Hebb** (formalisation usuelle $\Delta w = \eta\, y\, x$) : elle renforce
       les connexions actives ensemble, **sans regarder si la sortie était juste**.
    3. **Perceptron** (Rosenblatt, 1957-1958) : on ne corrige les poids **que sur une
       erreur**, $w \leftarrow w + \eta\, y\, \tilde x$. Sur des données linéairement
       séparables, il converge en un nombre fini de corrections, borné par
       $(R/\gamma)^2$ (théorème de convergence). Même calcul avec scikit-learn.
    4. **XOR** : aucune droite ne sépare les quatre points ; le perceptron tourne en rond.
    5. **XOR avec deux couches câblées à la main** : une couche cachée (OU, ET) rend le
       problème séparable pour la couche de sortie.

    **Loss optimisée.** Le perceptron fait une descente de (sous-)gradient stochastique
    sur le **critère du perceptron** $\ell(w; x, y) = \max\big(0,\, -y\, w\cdot\tilde x\big)$,
    c'est-à-dire une perte charnière (hinge) **sans marge** : la hinge du chapitre 01 est
    $\max(0, 1 - y\, w\cdot \tilde x)$. Un point bien classé, même de justesse, coûte 0.
    Le neurone formel et la règle de Hebb n'optimisent **aucune** perte.

    Notations : $y \in \{-1, +1\}$, $\tilde x = (x_1, x_2, 1)$ (le 1 porte le biais),
    sortie $\operatorname{signe}(w \cdot \tilde x)$.
    """)


@app.cell
def _(is_smoke, seed_everything, style_commun, time):
    T0 = time.perf_counter()  # chronomètre de l'ensemble des calculs
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        n_classe=30 if SMOKE else 100,      # points par classe (données séparables)
        bande=0.1,                          # demi-largeur de la bande vide autour de la vraie frontière
        eta=1.0,                            # pas du perceptron (sans effet sur la convergence si w₀ = 0)
        epoques_max=1000,                   # garde-fou : on s'arrête dès qu'une époque est sans erreur
        epoques_xor=20 if SMOKE else 100,   # époques sur XOR (il ne s'arrête jamais de lui-même)
        tirages_droites=10_000 if SMOKE else 200_000,  # droites aléatoires essayées sur XOR
        bandes_marge=[0.05, 0.1, 0.25, 0.5, 1.0],      # partie « marge » : largeurs essayées
        jeux_par_bande=3 if SMOKE else 10,             # jeux de données par largeur
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SEED, SMOKE, T0, config


@app.cell
def _(mo):
    # Le bouton évite de tout relancer à chaque modification dans l'éditeur.
    bouton = mo.ui.run_button(label="Lancer les calculs")
    bouton
    return (bouton,)


@app.cell
def _(bouton, mo, np):
    # ===================== 1. Le neurone formel de McCulloch et Pitts (1943) ============
    # En script (make ex MODE=run|smoke), personne ne clique : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer les calculs ».*"))

    def neurone_mp(excitatrices, inhibitrices, seuil):
        """Neurone de McCulloch-Pitts : entrées 0/1, pas de poids appris.

        Il « s'allume » (sortie 1) si AUCUNE entrée inhibitrice n'est active et si le
        nombre d'entrées excitatrices actives atteint le seuil. Dans l'article de 1943,
        une seule synapse inhibitrice active suffit à empêcher la décharge.
        """
        if any(inhibitrices):
            return 0
        return int(sum(excitatrices) >= seuil)

    entrees_binaires = [(0, 0), (0, 1), (1, 0), (1, 1)]
    portes_mp = {
        "ET":  lambda a, b: neurone_mp([a, b], [], seuil=2),
        "OU":  lambda a, b: neurone_mp([a, b], [], seuil=1),
        # NON a : aucune entrée excitatrice, seuil 0 (toujours prêt), a est inhibitrice
        "NON a": lambda a, b: neurone_mp([], [a], seuil=0),
        # a ET NON b : a excite, b inhibe
        "a ET NON b": lambda a, b: neurone_mp([a], [b], seuil=1),
    }
    tables_mp = {nom: [porte(a, b) for a, b in entrees_binaires] for nom, porte in portes_mp.items()}
    tables_attendues = {
        "ET": [0, 0, 0, 1], "OU": [0, 1, 1, 1], "NON a": [1, 1, 0, 0], "a ET NON b": [0, 0, 1, 0],
    }
    print("Entrées (a, b) :", entrees_binaires)
    for _nom, _t in tables_mp.items():
        print(f"  {_nom:11s} → {_t}  (attendu {tables_attendues[_nom]})")
    mp_ok = tables_mp == tables_attendues
    return entrees_binaires, mp_ok, neurone_mp


@app.cell
def _(config, np):
    # ===================== Données : deux nuages linéairement séparables ================
    # Vraie frontière : x₁ + 2·x₂ = 6. On tire deux nuages gaussiens, on étiquette chaque
    # point selon son côté de la vraie frontière et on RETIRE les points à moins de
    # `bande` de celle-ci : les données sont séparables par construction, avec une marge.
    U_VRAI = np.array([1.0, 2.0])
    B_VRAI = -6.0

    def nuages_separables(n_classe, rng, bande):
        X, Y = [], []
        compte = {1: 0, -1: 0}
        while len(X) < 2 * n_classe:
            cote = 1 if rng.random() < 0.5 else -1
            centre = [3.0, 2.5] if cote > 0 else [1.0, 1.0]
            x = rng.normal(centre, [1.0, 0.8])
            distance = (U_VRAI @ x + B_VRAI) / np.linalg.norm(U_VRAI)  # distance signée
            if abs(distance) < bande:
                continue
            y = 1 if distance > 0 else -1
            if compte[y] >= n_classe:
                continue
            compte[y] += 1
            X.append(x)
            Y.append(y)
        return np.array(X), np.array(Y)

    X_sep, y_sep = nuages_separables(config["n_classe"], np.random.default_rng(42), config["bande"])
    print(f"{len(X_sep)} points, {int((y_sep > 0).sum())} de classe +1, "
          f"{int((y_sep < 0).sum())} de classe −1")
    return X_sep, nuages_separables, y_sep


@app.cell
def _(X_sep, np, y_sep):
    # ===================== 2. Règle de Hebb ==============================================
    # « Les neurones qui s'activent ensemble se connectent » : Δw = η·y·x̃ pour CHAQUE
    # exemple, qu'il soit bien ou mal classé. Hebb (1949) ne donne pas de formule ; celle-ci
    # est la formalisation usuelle, avec des entrées et sorties codées en ±1.
    def avec_biais(X):
        """Ajoute la colonne de 1 qui porte le biais : x̃ = (x₁, x₂, 1)."""
        return np.c_[X, np.ones(len(X))]

    def hebb(X, y, eta=1.0):
        return eta * (avec_biais(X) * y[:, None]).sum(axis=0)

    def precision(w, X, y):
        # signe(0) = 0 compte comme une erreur, comme dans la règle du perceptron (y·w·x̃ ≤ 0)
        return float((np.sign(avec_biais(X) @ w) == y).mean())

    # a) ET en codage bipolaire (−1 = faux, +1 = vrai) : Hebb trouve une bonne réponse
    X_et = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)
    y_et = np.array([-1, -1, -1, 1])
    w_hebb_et = hebb(X_et, y_et)
    prec_hebb_et = precision(w_hebb_et, X_et, y_et)
    # b) Deux nuages séparables : Hebb additionne tout, ne corrige rien
    w_hebb_sep = hebb(X_sep, y_sep)
    prec_hebb_sep = precision(w_hebb_sep, X_sep, y_sep)
    print(f"Hebb sur ET bipolaire : w = {w_hebb_et}, précision {prec_hebb_et:.0%}")
    print(f"Hebb sur les nuages   : w = {np.round(w_hebb_sep, 1)}, précision {prec_hebb_sep:.1%}")
    print("  → biais = Σy = 0 (classes équilibrées) : la droite passe par l'origine, loin des données.")
    return avec_biais, prec_hebb_et, prec_hebb_sep, precision, w_hebb_sep


@app.cell
def _(avec_biais, np):
    # ===================== 3. Le perceptron ==============================================
    def perceptron(X, y, eta=1.0, epoques_max=1000):
        """Règle du perceptron, exemples parcourus dans l'ordre, poids initiaux nuls.

        Renvoie les poids finaux, le nombre d'erreurs (= de corrections) par époque et
        l'historique des poids après chaque correction. S'arrête après une époque sans
        erreur : tous les exemples sont alors bien classés.
        """
        Xb = avec_biais(X)
        w = np.zeros(Xb.shape[1])
        historique = [w.copy()]
        erreurs_par_epoque = []
        for _ in range(epoques_max):
            erreurs = 0
            for xi, yi in zip(Xb, y):
                if yi * (w @ xi) <= 0:          # mal classé (ou sur la frontière)
                    w = w + eta * yi * xi       # on tire la frontière vers le bon côté
                    erreurs += 1
                    historique.append(w.copy())
            erreurs_par_epoque.append(erreurs)
            if erreurs == 0:
                break
        return w, erreurs_par_epoque, historique

    def sgd_critere_perceptron(X, y, eta=1.0, epoques_max=1000):
        """Même chose écrit comme une descente de sous-gradient stochastique.

        Perte d'un exemple : ℓ(w) = max(0, −y·w·x̃). Sous-gradient : −y·x̃ si y·w·x̃ ≤ 0,
        0 sinon. Pas : w ← w − η·∂ℓ. Si ce code donne les mêmes poids que `perceptron`,
        c'est que la règle de Rosenblatt EST une descente de gradient sur cette perte.
        """
        Xb = avec_biais(X)
        w = np.zeros(Xb.shape[1])
        for _ in range(epoques_max):
            perte_totale = 0.0
            for xi, yi in zip(Xb, y):
                perte_totale += max(0.0, -yi * (w @ xi))
                sous_gradient = -yi * xi if yi * (w @ xi) <= 0 else np.zeros_like(xi)
                w = w - eta * sous_gradient
            if np.all(y * (Xb @ w) > 0):
                break
        return w

    def critere(w, X, y):
        """Valeur moyenne du critère du perceptron sur un jeu de données."""
        return float(np.maximum(0, -y * (avec_biais(X) @ w)).mean())

    return critere, perceptron, sgd_critere_perceptron


@app.cell
def _(
    X_sep,
    avec_biais,
    config,
    critere,
    np,
    perceptron,
    precision,
    sgd_critere_perceptron,
    y_sep,
):
    # --- 3a. Entraînement sur les nuages séparables ---
    w_perc, erreurs_epoques, historique_w = perceptron(
        X_sep, y_sep, config["eta"], config["epoques_max"])
    n_corrections = len(historique_w) - 1
    prec_perc = precision(w_perc, X_sep, y_sep)
    criteres_historique = [critere(_w, X_sep, y_sep) for _w in historique_w]
    print(f"Perceptron : {n_corrections} corrections en {len(erreurs_epoques)} époques "
          f"(erreurs par époque : {erreurs_epoques})")
    print(f"  poids finaux w = {np.round(w_perc, 3)}, précision {prec_perc:.0%}")
    print(f"  frontière apprise : {w_perc[0]:.2f}·x₁ + {w_perc[1]:.2f}·x₂ {w_perc[2]:+.0f} = 0 "
          f"(vraie : x₁ + 2·x₂ − 6 = 0)")
    # Piège : en w = 0, le critère vaut déjà 0 (minimum trivial, tous les points sur la
    # frontière). La règle corrige aussi quand y·w·x̃ = 0, ce qui la fait sortir de ce point ;
    # la hinge (marge 1) n'a pas ce défaut.
    print(f"  critère du perceptron : {criteres_historique[0]:.3f} en w = 0 (minimum trivial), "
          f"{criteres_historique[1]:.3f} après la 1re correction, "
          f"maximum {max(criteres_historique):.3f}, {criteres_historique[-1]:.3f} à la fin")

    # --- 3b. Même calcul vu comme SGD sur le critère du perceptron ---
    w_sgd = sgd_critere_perceptron(X_sep, y_sep, config["eta"], config["epoques_max"])
    ecart_sgd = float(np.abs(w_sgd - w_perc).max())
    print(f"SGD sur max(0, −y·w·x̃) : w = {np.round(w_sgd, 3)}, écart au perceptron {ecart_sgd:.1e}")

    # --- 3c. Théorème de convergence (Novikoff) : corrections ≤ (R/γ)² ---
    # R = plus grande norme d'un x̃ ; γ = marge d'un séparateur unitaire u (y·u·x̃ ≥ γ pour
    # tout exemple). On prend pour u le séparateur de marge maximale (SVM linéaire à marge
    # dure, C très grand), normalisé dans l'espace augmenté (x₁, x₂, 1).
    from sklearn.svm import SVC

    def rayon_et_marge(X, y):
        Xb = avec_biais(X)
        R = float(np.linalg.norm(Xb, axis=1).max())
        svm = SVC(kernel="linear", C=1e8, tol=1e-6).fit(X, y)
        u = np.r_[svm.coef_[0], svm.intercept_[0]]
        u = u / np.linalg.norm(u)
        gamma = float((y * (Xb @ u)).min())
        return R, gamma

    R_sep, gamma_sep = rayon_et_marge(X_sep, y_sep)
    borne_sep = (R_sep / gamma_sep) ** 2
    print(f"Novikoff : R = {R_sep:.2f}, γ = {gamma_sep:.4f}, borne (R/γ)² = {borne_sep:.0f} ; "
          f"corrections observées : {n_corrections}")

    # La droite du perceptron sépare, mais frôle des points : distance (dans le plan x₁, x₂)
    # du point le plus proche, comparée à la marge de la droite du SVM.
    _dist = y_sep * (avec_biais(X_sep) @ w_perc) / np.linalg.norm(w_perc[:2])
    _svm = SVC(kernel="linear", C=1e8, tol=1e-6).fit(X_sep, y_sep)
    print(f"Point le plus proche de la frontière du perceptron : {_dist.min():.4f} ; "
          f"marge du SVM : {1 / np.linalg.norm(_svm.coef_[0]):.4f}")
    return (
        borne_sep,
        criteres_historique,
        ecart_sgd,
        erreurs_epoques,
        gamma_sep,
        historique_w,
        n_corrections,
        prec_perc,
        rayon_et_marge,
        w_perc,
    )


@app.cell
def _(X_sep, config, np, w_perc, y_sep):
    # --- 3d. La bibliothèque : scikit-learn fait exactement le même calcul ---
    # Perceptron de scikit-learn = SGDClassifier(loss="perceptron", eta0=1, pas constant,
    # sans pénalité). shuffle=False pour garder l'ordre des exemples ; tol=None pour ne pas
    # s'arrêter sur un critère de perte (il n'y a plus de mise à jour une fois convergé).
    from sklearn.linear_model import Perceptron

    sk_perc = Perceptron(eta0=config["eta"], shuffle=False, tol=None, penalty=None,
                         max_iter=config["epoques_max"]).fit(X_sep, y_sep)
    w_sklearn = np.r_[sk_perc.coef_[0], sk_perc.intercept_[0]]
    ecart_sklearn = float(np.abs(w_sklearn - w_perc).max())
    prec_sklearn = float(sk_perc.score(X_sep, y_sep))
    print(f"scikit-learn : w = {np.round(w_sklearn, 3)}, précision {prec_sklearn:.0%}, "
          f"écart à numpy {ecart_sklearn:.1e}")
    return Perceptron, ecart_sklearn, prec_sklearn


@app.cell
def _(config, np, nuages_separables, perceptron, rayon_et_marge):
    # --- 3e. Marge et nombre de corrections ---
    # Pour plusieurs largeurs de bande vide, plusieurs jeux de données (graines 42, 43…) :
    # nombre de corrections observé contre la borne (R/γ)².
    marges_exp = []   # (bande, graine, gamma, R, corrections, borne)
    for _bande in config["bandes_marge"]:
        for _k in range(config["jeux_par_bande"]):
            _X, _y = nuages_separables(config["n_classe"], np.random.default_rng(42 + _k), _bande)
            _w, _e, _h = perceptron(_X, _y, config["eta"], config["epoques_max"])
            _R, _g = rayon_et_marge(_X, _y)
            marges_exp.append((_bande, 42 + _k, _g, _R, len(_h) - 1, (_R / _g) ** 2))
    marges_exp = np.array(marges_exp)
    print("bande | γ moyen | corrections (moyenne, max) | borne (R/γ)² (moyenne)")
    for _bande in config["bandes_marge"]:
        _m = marges_exp[marges_exp[:, 0] == _bande]
        print(f"{_bande:5.2f} | {_m[:, 2].mean():.4f}  | {_m[:, 4].mean():7.1f} ({_m[:, 4].max():4.0f})"
              f"             | {_m[:, 5].mean():9.0f}")
    borne_respectee = bool(np.all(marges_exp[:, 4] <= marges_exp[:, 5]))
    print(f"Borne respectée sur les {len(marges_exp)} jeux : {borne_respectee}")
    return borne_respectee, marges_exp


@app.cell
def _(Perceptron, avec_biais, config, np, perceptron):
    # ===================== 4. XOR : l'échec du perceptron ================================
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_xor = np.array([-1, 1, 1, -1])   # vrai (+1) si exactement une entrée vaut 1

    _w, erreurs_xor, historique_xor = perceptron(X_xor, y_xor, 1.0, config["epoques_xor"])
    # Meilleure précision atteinte par l'un des poids visités pendant l'apprentissage
    _Xb = avec_biais(X_xor)
    prec_visitees = [float((np.sign(_Xb @ _h) == y_xor).mean()) for _h in historique_xor]
    meilleure_xor_perceptron = max(prec_visitees)
    # Les poids repassent-ils par une valeur déjà vue ? (cycle)
    _vus = {}
    cycle_xor = None
    for _i, _h in enumerate(historique_xor):
        _cle = tuple(np.round(_h, 9))
        if _cle in _vus:
            cycle_xor = (_vus[_cle], _i)
            break
        _vus[_cle] = _i
    print(f"XOR : erreurs par époque {erreurs_xor[:8]}… ({len(erreurs_xor)} époques, jamais 0)")
    print(f"  meilleure précision parmi les {len(historique_xor)} poids visités : "
          f"{meilleure_xor_perceptron:.0%}")
    if cycle_xor:
        print(f"  cycle : les poids après la correction {cycle_xor[1]} sont ceux d'après la "
              f"correction {cycle_xor[0]} ({historique_xor[cycle_xor[0]]}) : le perceptron tourne en rond")

    # Recherche brute : des droites tirées au hasard ne dépassent jamais 3 points sur 4
    _rng = np.random.default_rng(42)
    _W = _rng.normal(size=(config["tirages_droites"], 3))
    meilleure_xor_aleatoire = float((np.sign(_W @ _Xb.T) == y_xor).mean(axis=1).max())
    print(f"  {config['tirages_droites']} droites aléatoires : meilleure précision "
          f"{meilleure_xor_aleatoire:.0%}")

    # Et la bibliothèque ?
    _sk = Perceptron(eta0=1.0, shuffle=False, tol=None, penalty=None,
                     max_iter=config["epoques_xor"]).fit(X_xor, y_xor)
    prec_xor_sklearn = float(_sk.score(X_xor, y_xor))
    print(f"  scikit-learn sur XOR : précision {prec_xor_sklearn:.0%}")
    return (
        X_xor,
        cycle_xor,
        erreurs_xor,
        meilleure_xor_aleatoire,
        meilleure_xor_perceptron,
        prec_xor_sklearn,
        y_xor,
    )


@app.cell
def _(X_xor, avec_biais, entrees_binaires, neurone_mp, np, perceptron, precision, y_xor):
    # ===================== 5. XOR avec deux couches, câblées à la main ====================
    # Couche cachée : h₁ = OU(a, b), h₂ = ET(a, b) ; sortie : h₁ ET NON h₂.
    # Version « neurone formel » (seuils, entrée inhibitrice) puis version « perceptron »
    # (poids et biais, fonction échelon), qui servira au chapitre 07.
    def xor_mp(a, b):
        h1 = neurone_mp([a, b], [], seuil=1)   # OU
        h2 = neurone_mp([a, b], [], seuil=2)   # ET
        return neurone_mp([h1], [h2], seuil=1) # h₁ ET NON h₂

    table_xor_mp = [xor_mp(a, b) for a, b in entrees_binaires]

    def echelon(z):
        return (z > 0).astype(float)

    W1 = np.array([[1.0, 1.0], [1.0, 1.0]])   # une ligne par neurone caché
    b1 = np.array([-0.5, -1.5])               # OU : x₁ + x₂ > 0,5 ; ET : x₁ + x₂ > 1,5
    w2 = np.array([1.0, -1.0])                # sortie : h₁ − h₂ > 0,5
    b2 = -0.5
    H_xor = echelon(X_xor @ W1.T + b1)        # représentation cachée des 4 points
    sortie_xor = echelon(H_xor @ w2 + b2)
    table_xor_2c = [int(v) for v in sortie_xor]
    print("XOR attendu        :", [int(v > 0) for v in y_xor])
    print("XOR, neurones MP   :", table_xor_mp)
    print("XOR, deux couches  :", table_xor_2c)
    print("Représentation cachée (h₁, h₂) :",
          {tuple(map(int, x)): tuple(map(int, h)) for x, h in zip(X_xor, H_xor)})

    # Dans l'espace caché, (0,1) et (1,0) tombent au même endroit, (1,0) : le problème
    # devient séparable. Un perceptron entraîné SUR (h₁, h₂) converge.
    w_cache, erreurs_cache, historique_cache = perceptron(H_xor, y_xor, 1.0, 100)
    prec_cache = precision(w_cache, H_xor, y_xor)
    print(f"Perceptron sur (h₁, h₂) : {len(historique_cache) - 1} corrections, "
          f"{len(erreurs_cache)} époques, précision {prec_cache:.0%}, w = {w_cache}")
    return H_xor, W1, b1, prec_cache, table_xor_2c, table_xor_mp, w_cache


@app.cell
def _(
    PALETTE,
    SMOKE,
    X_sep,
    erreurs_epoques,
    historique_w,
    mo,
    np,
    plt,
    sauvegarder_figure,
    w_hebb_sep,
    y_sep,
):
    # ===================== Figure 1 : le perceptron sur données séparables =================
    fig_sep, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.3),
                                         gridspec_kw={"width_ratios": [1.5, 1]})
    _pos = y_sep > 0
    ax_a.scatter(*X_sep[~_pos].T, s=14, marker="o", color="0.55", label="classe −1")
    ax_a.scatter(*X_sep[_pos].T, s=22, marker="+", color="0.1", label="classe +1")
    _x1 = np.linspace(-2, 7, 2)
    _n = len(historique_w)
    for _i, _w in enumerate(historique_w[1:], start=1):
        if abs(_w[1]) < 1e-12:
            continue
        _final = _i == _n - 1
        ax_a.plot(_x1, -(_w[0] * _x1 + _w[2]) / _w[1],
                  color=PALETTE[0] if _final else PALETTE[1],
                  lw=2.2 if _final else 0.8, alpha=1 if _final else 0.15 + 0.6 * _i / _n,
                  label="frontière finale" if _final else None)
    ax_a.plot(_x1, (6 - _x1) / 2, color="0.3", ls=":", lw=1.5, label="vraie frontière")
    ax_a.plot(_x1, -(w_hebb_sep[0] * _x1 + w_hebb_sep[2]) / w_hebb_sep[1], color=PALETTE[3],
              ls="--", lw=1.5, label="règle de Hebb")
    ax_a.set(xlim=(-2, 7), ylim=(-2, 5.5), xlabel="x₁", ylabel="x₂",
             title=f"{_n - 1} corrections : la frontière se déplace (orange) puis se fige")
    ax_a.legend(fontsize=8, loc="upper right")
    ax_b.bar(np.arange(1, len(erreurs_epoques) + 1), erreurs_epoques, color=PALETTE[0])
    ax_b.set(xlabel="époque", ylabel="erreurs (= corrections)", title="Erreurs par époque")
    for _e, _v in enumerate(erreurs_epoques, start=1):
        ax_b.text(_e, _v + 0.5, str(_v), ha="center", fontsize=9)
    _chemin = sauvegarder_figure(fig_sep, "perceptron-separable" + ("-smoke" if SMOKE else ""),
                                 mo.notebook_dir() / "figures")
    print(f"Figure : {_chemin[0]}")
    fig_sep


@app.cell
def _(
    H_xor,
    PALETTE,
    SMOKE,
    W1,
    X_xor,
    b1,
    erreurs_xor,
    mo,
    np,
    plt,
    sauvegarder_figure,
    w_cache,
    y_xor,
):
    # ===================== Figure 2 : XOR, échec puis deux couches ========================
    fig_xor, (ax_x1, ax_x2, ax_x3) = plt.subplots(1, 3, figsize=(14, 4.4))
    fig_xor.subplots_adjust(wspace=0.35)
    ax_x1.plot(np.arange(1, len(erreurs_xor) + 1), erreurs_xor, color=PALETTE[3], marker=".")
    ax_x1.set(xlabel="époque", ylabel="erreurs", ylim=(-0.2, 4.5),
              title="XOR : 4 erreurs à chaque époque")

    def points(ax, P):
        for _p, _y in zip(P, y_xor):
            ax.scatter(*_p, s=120, marker="+" if _y > 0 else "o",
                       color="0.1" if _y > 0 else "0.55", zorder=3)

    _t = np.linspace(-0.5, 1.5, 2)
    points(ax_x2, X_xor)
    for _j, (_nom, _c) in enumerate([("h₁ = OU", PALETTE[0]), ("h₂ = ET", PALETTE[2])]):
        ax_x2.plot(_t, -(W1[_j, 0] * _t + b1[_j]) / W1[_j, 1], color=_c, lw=2, label=_nom)
    ax_x2.set(xlim=(-0.5, 1.5), ylim=(-0.5, 1.5), xlabel="x₁", ylabel="x₂", aspect="equal",
              title="Entrée : une droite par neurone caché")
    ax_x2.legend(fontsize=8, loc="upper right")
    points(ax_x3, H_xor + np.array([[0, 0], [-0.04, 0], [0.04, 0], [0, 0]]))  # (1,0) doublé
    ax_x3.plot(_t, -(w_cache[0] * _t + w_cache[2]) / w_cache[1], color=PALETTE[1], lw=2,
               label="droite apprise")
    ax_x3.annotate("(0,1) et (1,0)\nconfondus", xy=(1, 0), xytext=(0.95, 0.45), fontsize=8,
                   ha="center", arrowprops={"arrowstyle": "->", "color": "0.4"})
    ax_x3.set(xlim=(-0.5, 1.5), ylim=(-0.5, 1.5), xlabel="h₁", ylabel="h₂", aspect="equal",
              title="Espace caché : séparable")
    ax_x3.legend(fontsize=8, loc="upper left")
    _chemin = sauvegarder_figure(fig_xor, "xor" + ("-smoke" if SMOKE else ""),
                                 mo.notebook_dir() / "figures")
    print(f"Figure : {_chemin[0]}")
    fig_xor


@app.cell
def _(PALETTE, SMOKE, marges_exp, mo, plt, sauvegarder_figure):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    # ===================== Figure 3 : marge et nombre de corrections ======================
    fig_m, ax_m = plt.subplots(figsize=(7, 4.2))
    ax_m.scatter(marges_exp[:, 2], marges_exp[:, 4], color=PALETTE[0], s=18,
                 label="corrections observées (un point = un jeu de données)")
    ax_m.scatter(marges_exp[:, 2], marges_exp[:, 5], color=PALETTE[3], s=18, marker="x",
                 label="borne (R/γ)²")
    ax_m.set(xscale="log", yscale="log", xlabel="marge γ (espace augmenté)",
             ylabel="nombre de corrections", title="Plus la marge est petite, plus il corrige")
    # Graduations lisibles (pas de notation 10⁻² superposée)
    ax_m.xaxis.set_major_locator(FixedLocator([0.02, 0.05, 0.1, 0.2, 0.4]))
    ax_m.xaxis.set_minor_locator(NullLocator())
    ax_m.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}".replace(".", ",")))
    ax_m.legend(fontsize=8)
    _chemin = sauvegarder_figure(fig_m, "marge" + ("-smoke" if SMOKE else ""),
                                 mo.notebook_dir() / "figures")
    print(f"Figure : {_chemin[0]}")
    fig_m


@app.cell
def _(
    SMOKE,
    T0,
    borne_respectee,
    borne_sep,
    ecart_sgd,
    ecart_sklearn,
    meilleure_xor_aleatoire,
    meilleure_xor_perceptron,
    mo,
    mp_ok,
    n_corrections,
    prec_cache,
    prec_hebb_et,
    prec_hebb_sep,
    prec_perc,
    prec_sklearn,
    prec_xor_sklearn,
    table_xor_2c,
    table_xor_mp,
    time,
):
    # --- Bilan : seuils du README. En smoke, on vérifie seulement que tout tourne. ---
    _xor_attendu = [0, 1, 1, 0]
    verifs = {
        "neurone formel : ET, OU, NON, a ET NON b exacts": mp_ok,
        "Hebb : ET bipolaire à 100 %": prec_hebb_et == 1.0,
        "Hebb : nuages séparables sous 80 %": prec_hebb_sep < 0.80,
        "perceptron : 100 % sur les nuages séparables": prec_perc == 1.0,
        "perceptron : corrections ≤ borne (R/γ)²": n_corrections <= borne_sep,
        "toutes les bandes : corrections ≤ borne": borne_respectee,
        "perceptron = SGD sur max(0, −y·w·x̃) (écart ≤ 1e-9)": ecart_sgd <= 1e-9,
        "numpy = scikit-learn (écart ≤ 1e-9, précision 100 %)":
            ecart_sklearn <= 1e-9 and prec_sklearn == 1.0,
        "XOR : aucun poids visité ne dépasse 75 %": meilleure_xor_perceptron <= 0.75,
        "XOR : aucune droite aléatoire ne dépasse 75 %": meilleure_xor_aleatoire <= 0.75,
        "XOR : scikit-learn ne dépasse pas 75 %": prec_xor_sklearn <= 0.75,
        "XOR à deux couches (MP et échelon) exact": table_xor_mp == _xor_attendu
            and table_xor_2c == _xor_attendu,
        "perceptron sur l'espace caché : 100 %": prec_cache == 1.0,
    }
    for _k, _ok in verifs.items():
        print(f"[{'ok' if _ok else 'ÉCHEC'}] {_k}")
    if not SMOKE and not all(verifs.values()):
        raise SystemExit("Seuils non atteints : " + ", ".join(k for k, ok in verifs.items() if not ok))
    duree_totale = time.perf_counter() - T0
    print(f"Durée des calculs : {duree_totale:.1f} s")
    mo.md(f"**Bilan** : {sum(verifs.values())}/{len(verifs)} vérifications réussies "
          f"en {duree_totale:.1f} s.")


if __name__ == "__main__":
    app.run()
