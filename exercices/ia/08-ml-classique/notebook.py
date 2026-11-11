# Exercice 08 : apprentissage automatique « classique ».
#
#   make ex NUM=08              → éditeur marimo (interactif, bouton « Lancer »)
#   make ex NUM=08 MODE=run     → script complet (vérifie les seuils du README)
#   make ex NUM=08 MODE=smoke   → test rapide, < 30 s
#
# Progression : k plus proches voisins, souche d'arbre (Gini), k-means et ACP écrits
# « from scratch » en numpy, chacun comparé à scikit-learn ; puis comparaison de neuf
# modèles (sept familles) avec scikit-learn, validation croisée, recherche aléatoire
# d'hyperparamètres et t-SNE. Tout tourne sur CPU ; aucun téléchargement (les jeux
# `digits` et `breast_cancer` sont livrés avec scikit-learn).

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
    # Exercice 08 : l'apprentissage automatique « classique »

    **Objectif.** Programmer en numpy les briques de base du *machine learning* des
    années 1960 à 2000, les comparer à scikit-learn, puis mettre neuf modèles (sept familles)
    face à face sur les mêmes données.

    1. **k plus proches voisins (k-NN)** en numpy : le réglage de $k$ fait passer du
       surapprentissage ($k = 1$) au sous-apprentissage ($k$ grand). Même résultat que
       `KNeighborsClassifier`. Puis la **malédiction de la dimension**.
    2. **Souche d'arbre** (arbre de profondeur 1) : on cherche le meilleur seuil en
       minimisant l'impureté de Gini, et on retrouve le choix de `DecisionTreeClassifier`.
       Puis arbre profond contre forêt aléatoire.
    3. **Neuf modèles (sept familles) avec scikit-learn** : Bayes naïf, régression logistique, k-NN,
       arbre, forêt aléatoire, AdaBoost, *gradient boosting*, SVM linéaire et à noyau.
       Frontières sur deux lunes, validation croisée sur un jeu médical réel.
    4. **k-means** (algorithme de Lloyd) en numpy, comparé à `KMeans` ; minima locaux,
       et échec sur des groupes non convexes que DBSCAN sépare.
    5. **Validation croisée et recherche d'hyperparamètres** : grille contre tirage aléatoire.
    6. **ACP** en numpy (SVD) contre `PCA`, puis **t-SNE** et un de ses pièges.

    **Ce qui est optimisé** (une ligne par méthode ; fiches au chapitre 01) :
    k-NN : rien (on stocke les exemples) ; arbre : impureté de Gini
    $G = 1 - \sum_c p_c^2$, choisie gloutonnement nœud par nœud ; SVM : perte charnière
    $\max(0, 1 - y f(x))$ plus $\tfrac{1}{2}\lVert w\rVert^2$ ; *gradient boosting* et
    régression logistique : log-loss (entropie croisée binaire) ; k-means : inertie
    $\sum_i \lVert x_i - \mu_{c(i)} \rVert^2$ ; ACP : variance projetée (ou, c'est
    équivalent, erreur de reconstruction quadratique) ; t-SNE : divergence de
    Kullback-Leibler entre voisinages.
    """)


@app.cell
def _(is_smoke, seed_everything, style_commun, time):
    T0 = time.perf_counter()  # chronomètre de l'ensemble des calculs
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        n_train=200,                        # deux lunes : points d'entraînement (mêmes que la démo)
        n_test=1000,                        # points de test (mêmes que la démo)
        bruit_lunes=0.3,
        k_max=199,                          # k impairs de 1 à k_max
        dimensions=[1, 2, 10, 100, 1000],   # malédiction de la dimension
        n_uniformes=1000,                   # points tirés dans [0, 1]^d
        bruit_dims=[0, 2, 10, 50],          # dimensions de bruit ajoutées aux lunes
        profondeurs=list(range(1, 16)),
        n_arbres=50 if SMOKE else 300,      # forêt aléatoire
        n_graines_kmeans=5 if SMOKE else 50,
        n_iter_recherche=4 if SMOKE else 30,  # budget de la recherche d'hyperparamètres
        cv=3 if SMOKE else 5,               # plis de la validation croisée
        n_tsne=400 if SMOKE else None,      # None = les 1 797 images de digits
        perplexites=[30] if SMOKE else [5, 30, 100],
        tsne_iter=250 if SMOKE else 1000,
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
def _(bouton, config, mo, np):
    # ===================== Données : deux lunes entrelacées ==============================
    # En script (make ex MODE=run|smoke), personne ne clique : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer les calculs ».*"))
    from sklearn.datasets import make_moons

    # Arrondi à 4 décimales : ce sont exactement les points de la démo KnnBoundary du site.
    X_tr, y_tr = make_moons(config["n_train"], noise=config["bruit_lunes"], random_state=42)
    X_te, y_te = make_moons(config["n_test"], noise=config["bruit_lunes"], random_state=43)
    X_tr, X_te = np.round(X_tr, 4), np.round(X_te, 4)
    print(f"Deux lunes : {len(X_tr)} points d'entraînement, {len(X_te)} de test, "
          f"bruit gaussien d'écart-type {config['bruit_lunes']}")
    return X_te, X_tr, make_moons, y_te, y_tr


@app.cell
def _(X_te, X_tr, config, np, y_te, y_tr):
    # ===================== 1. k plus proches voisins, en numpy ===========================
    def distances2(Xq, Xr):
        """Distances euclidiennes AU CARRÉ entre chaque requête et chaque point de référence.

        La racine ne change pas l'ordre des voisins : inutile de la calculer.
        Forme (n_requêtes, n_références).
        """
        d = Xq[:, None, :] - Xr[None, :, :]
        return (d**2).sum(axis=-1)

    def voisins_tries(Xq, Xr):
        # Tri stable : à distance égale, le point d'indice le plus petit passe devant
        # (même règle que la démo du site).
        return np.argsort(distances2(Xq, Xr), axis=1, kind="stable")

    def knn_predire(ordre, y_ref, k):
        """Vote majoritaire parmi les k premiers voisins (classes 0/1, k impair : pas d'égalité)."""
        votes = y_ref[ordre[:, :k]].mean(axis=1)
        return (votes > 0.5).astype(int)

    ordre_tr = voisins_tries(X_tr, X_tr)   # chaque point d'entraînement est son propre 1er voisin
    ordre_te = voisins_tries(X_te, X_tr)
    ks = np.arange(1, config["k_max"] + 1, 2)
    err_tr = np.array([(knn_predire(ordre_tr, y_tr, k) != y_tr).mean() for k in ks])
    err_te = np.array([(knn_predire(ordre_te, y_tr, k) != y_te).mean() for k in ks])
    k_meilleur_test = int(ks[np.argmin(err_te)])
    for _k in [1, 5, 15, 25, 35, 51, 99, 151, 199]:
        if _k <= config["k_max"]:
            _i = int(np.where(ks == _k)[0][0])
            print(f"k = {_k:3d} : erreur d'entraînement {err_tr[_i]:6.1%}, de test {err_te[_i]:6.1%}")
    print(f"Meilleur k sur le test (on triche, pour la courbe) : {k_meilleur_test}, "
          f"erreur {err_te.min():.1%}")
    return err_te, err_tr, knn_predire, ks, ordre_te, voisins_tries


@app.cell
def _(X_tr, config, knn_predire, ks, np, voisins_tries, y_tr):
    # Choisir k SANS regarder le test : validation croisée à 5 plis sur l'entraînement.
    def choisir_k_cv(X, y, ks_candidats, n_plis, graine=42):
        rng = np.random.default_rng(graine)
        plis = np.array_split(rng.permutation(len(X)), n_plis)
        erreurs = np.zeros(len(ks_candidats))
        for i in range(n_plis):
            val = plis[i]
            app_ = np.concatenate([plis[j] for j in range(n_plis) if j != i])
            ordre = voisins_tries(X[val], X[app_])
            for a, k in enumerate(ks_candidats):
                if k <= len(app_):
                    erreurs[a] += (knn_predire(ordre, y[app_], k) != y[val]).mean() / n_plis
                else:
                    erreurs[a] = np.inf
        return int(ks_candidats[np.argmin(erreurs)]), erreurs

    k_cv, err_cv = choisir_k_cv(X_tr, y_tr, ks[ks <= 99], config["cv"])
    print(f"k choisi par validation croisée ({config['cv']} plis) : {k_cv}, "
          f"erreur de validation {err_cv.min():.1%}")
    return (k_cv,)


@app.cell
def _(X_te, X_tr, err_te, k_cv, knn_predire, ks, np, ordre_te, y_tr):
    # Même calcul avec scikit-learn : les prédictions doivent être identiques.
    from sklearn.neighbors import KNeighborsClassifier

    ecarts_sklearn = {}
    for _k in [1, 5, 25, k_cv, 99]:
        _p_sk = KNeighborsClassifier(n_neighbors=_k, algorithm="brute").fit(X_tr, y_tr).predict(X_te)
        ecarts_sklearn[_k] = int((_p_sk != knn_predire(ordre_te, y_tr, _k)).sum())
    err_te_kcv = float(err_te[int(np.where(ks == k_cv)[0][0])])
    print(f"Prédictions différentes numpy / scikit-learn (sur {len(X_te)} points) : {ecarts_sklearn}")
    print(f"Erreur de test avec k = {k_cv} (choisi par CV) : {err_te_kcv:.1%}")
    return KNeighborsClassifier, ecarts_sklearn, err_te_kcv


@app.cell
def _(X_tr, config, err_te, err_tr, k_cv, knn_predire, ks, np, plt, voisins_tries, y_tr):
    # Figure : frontières pour trois valeurs de k + courbe d'erreur en fonction de k.
    def grille_plan(n=200):
        gx, gy = np.meshgrid(np.linspace(-2.5, 3.5, n), np.linspace(-2.0, 2.5, n))
        return gx, gy, np.c_[gx.ravel(), gy.ravel()]

    _gx, _gy, _pts = grille_plan(160)
    _ordre_grille = voisins_tries(_pts, X_tr)
    fig_knn, _axes = plt.subplots(1, 4, figsize=(17, 4))
    for _ax, _k in zip(_axes[:3], [1, k_cv, config["k_max"]]):
        _z = knn_predire(_ordre_grille, y_tr, _k).reshape(_gx.shape)
        _ax.contourf(_gx, _gy, _z, levels=[-0.5, 0.5, 1.5], colors=["#9ecae9", "#f6c86b"], alpha=0.6)
        _ax.scatter(X_tr[:, 0], X_tr[:, 1], c=np.where(y_tr == 1, "#D55E00", "#0072B2"),
                    s=12, edgecolors="k", linewidths=0.3)
        _i = int(np.where(ks == _k)[0][0])
        _ax.set(title=f"k = {_k} : app. {err_tr[_i]:.0%}, test {err_te[_i]:.1%}", xticks=[], yticks=[])
    _axes[3].plot(ks, err_tr, label="entraînement")
    _axes[3].plot(ks, err_te, label="test (1 000 points)")
    _axes[3].axvline(k_cv, color="0.5", ls="--", lw=0.8, label=f"k choisi par CV = {k_cv}")
    _axes[3].set(xscale="log", xlabel="k (échelle log)", ylabel="taux d'erreur",
                 title="Biais-variance du k-NN")
    _axes[3].legend(fontsize=8)
    fig_knn.tight_layout()
    return fig_knn, grille_plan


@app.cell
def _(KNeighborsClassifier, X_te, X_tr, config, k_cv, np, y_te, y_tr):
    # ===================== Malédiction de la dimension ===================================
    # a) Dans [0, 1]^d, quand d grandit, le plus proche et le plus lointain voisin d'un
    #    point sont presque à la même distance : « proche » ne veut plus rien dire.
    _rng = np.random.default_rng(42)
    contraste = {}
    for _d in config["dimensions"]:
        _P = _rng.random((config["n_uniformes"], _d))
        _q = _rng.random(_d)
        _dist = np.sqrt(((_P - _q) ** 2).sum(axis=1))
        contraste[_d] = (float(_dist.min()), float(_dist.max()))
        print(f"d = {_d:5d} : plus proche {_dist.min():7.3f}, plus lointain {_dist.max():7.3f}, "
              f"rapport {_dist.max() / _dist.min():8.2f}")
    # b) Mêmes deux lunes, plus des dimensions de bruit pur (uniformes sur [-2, 2]) :
    #    elles noient les deux dimensions utiles dans le calcul des distances.
    from sklearn.ensemble import RandomForestClassifier as _RF
    err_bruit, err_bruit_foret = {}, {}
    for _D in config["bruit_dims"]:
        _Xtr = np.c_[X_tr, _rng.uniform(-2, 2, (len(X_tr), _D))]
        _Xte = np.c_[X_te, _rng.uniform(-2, 2, (len(X_te), _D))]
        err_bruit[_D] = 1 - KNeighborsClassifier(k_cv).fit(_Xtr, y_tr).score(_Xte, y_te)
        # Même test pour une forêt aléatoire : elle choisit ses variables coupe par coupe.
        err_bruit_foret[_D] = 1 - _RF(config["n_arbres"], random_state=42).fit(_Xtr, y_tr).score(_Xte, y_te)
        print(f"lunes + {_D:2d} dimensions de bruit : erreur de test du {k_cv}-NN {err_bruit[_D]:.1%}, "
              f"de la forêt aléatoire {err_bruit_foret[_D]:.1%}")
    return contraste, err_bruit, err_bruit_foret


@app.cell
def _(X_tr, np, y_tr):
    # ===================== 2. Souche d'arbre (profondeur 1) avec l'impureté de Gini ======
    def gini(y):
        """Impureté de Gini d'un nœud : 1 − Σ p_c². 0 = nœud pur ; 0,5 = moitié-moitié (2 classes)."""
        if len(y) == 0:
            return 0.0
        p = np.bincount(y, minlength=2) / len(y)
        return 1.0 - float((p**2).sum())

    def meilleure_coupe(X, y):
        """Essaie toutes les coupes « x_j ≤ s » (s au milieu de deux valeurs successives) et
        garde celle qui minimise l'impureté moyenne pondérée des deux enfants."""
        meilleure = (np.inf, None, None)
        for j in range(X.shape[1]):
            valeurs = np.unique(X[:, j])
            for s in (valeurs[:-1] + valeurs[1:]) / 2:
                gauche, droite = y[X[:, j] <= s], y[X[:, j] > s]
                imp = (len(gauche) * gini(gauche) + len(droite) * gini(droite)) / len(y)
                if imp < meilleure[0] - 1e-12:
                    meilleure = (imp, j, float(s))
        return meilleure

    imp_souche, j_souche, s_souche = meilleure_coupe(X_tr, y_tr)
    print(f"Gini à la racine : {gini(y_tr):.4f}")
    print(f"Meilleure coupe numpy : x{j_souche + 1} ≤ {s_souche:.4f}, Gini pondéré des enfants "
          f"{imp_souche:.4f} (gain {gini(y_tr) - imp_souche:.4f})")
    return gini, imp_souche, j_souche, s_souche


@app.cell
def _(X_te, X_tr, config, gini, imp_souche, j_souche, np, s_souche, y_te, y_tr):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    _souche = DecisionTreeClassifier(max_depth=1, random_state=42).fit(X_tr, y_tr)
    j_sk, s_sk = int(_souche.tree_.feature[0]), float(_souche.tree_.threshold[0])
    _imp_sk = float((_souche.tree_.impurity[1:] * _souche.tree_.n_node_samples[1:]).sum()
                    / _souche.tree_.n_node_samples[0])
    souche_ok = (j_sk == j_souche) and abs(s_sk - s_souche) < 1e-6 and abs(_imp_sk - imp_souche) < 1e-9
    print(f"scikit-learn       : x{j_sk + 1} ≤ {s_sk:.4f}, Gini pondéré {_imp_sk:.4f} "
          f"→ {'identique' if souche_ok else 'DIFFÉRENT'}")
    _ = gini

    # Profondeur de l'arbre : l'erreur d'entraînement tombe à 0, celle de test remonte.
    err_arbre = {}
    for _p in config["profondeurs"]:
        _a = DecisionTreeClassifier(max_depth=_p, random_state=42).fit(X_tr, y_tr)
        err_arbre[_p] = (1 - _a.score(X_tr, y_tr), 1 - _a.score(X_te, y_te))
    _plein = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)
    err_arbre_plein = (1 - _plein.score(X_tr, y_tr), 1 - _plein.score(X_te, y_te))
    _foret = RandomForestClassifier(config["n_arbres"], random_state=42).fit(X_tr, y_tr)
    err_foret = (1 - _foret.score(X_tr, y_tr), 1 - _foret.score(X_te, y_te))
    # Chaque arbre de la forêt, seul : parfait sur son tirage bootstrap, pas sur tout l'entraînement.
    _err_indiv = np.array([(np.mean(_t.predict(X_tr) != y_tr), np.mean(_t.predict(X_te) != y_te))
                           for _t in _foret.estimators_])
    _p_best = min(err_arbre, key=lambda p: err_arbre[p][1])
    print(f"Arbre profondeur 1 : test {err_arbre[1][1]:.1%} ; meilleure profondeur sur le test "
          f"{_p_best} ({err_arbre[_p_best][1]:.1%})")
    print(f"Arbre complet ({_plein.get_depth()} niveaux, {_plein.get_n_leaves()} feuilles) : "
          f"entraînement {err_arbre_plein[0]:.1%}, test {err_arbre_plein[1]:.1%}")
    print(f"Forêt aléatoire de {config['n_arbres']} arbres complets : entraînement {err_foret[0]:.1%}, "
          f"test {err_foret[1]:.1%}")
    print(f"  ses arbres pris un par un : entraînement {_err_indiv[:, 0].mean():.1%} en moyenne, "
          f"test {_err_indiv[:, 1].mean():.1%} en moyenne "
          f"(de {_err_indiv[:, 1].min():.1%} à {_err_indiv[:, 1].max():.1%})")
    return (
        DecisionTreeClassifier,
        RandomForestClassifier,
        err_arbre,
        err_arbre_plein,
        err_foret,
        souche_ok,
    )


@app.cell
def _(
    DecisionTreeClassifier,
    KNeighborsClassifier,
    RandomForestClassifier,
    X_te,
    X_tr,
    config,
    k_cv,
    time,
    y_te,
    y_tr,
):
    # ===================== 3. Neuf modèles (sept familles), mêmes données ================
    from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    def modeles(k_voisins):
        """Les modèles comparés. Ceux qui reposent sur des distances ou des produits
        scalaires (k-NN, logistique, SVM) reçoivent des variables standardisées."""
        std = StandardScaler
        return {
            "Bayes naïf (gaussien)": GaussianNB(),
            "Régression logistique": make_pipeline(std(), LogisticRegression(max_iter=5000)),
            f"{k_voisins}-NN": make_pipeline(std(), KNeighborsClassifier(k_voisins)),
            "Arbre (profondeur 4)": DecisionTreeClassifier(max_depth=4, random_state=42),
            "Forêt aléatoire": RandomForestClassifier(config["n_arbres"], random_state=42),
            "AdaBoost (souches)": AdaBoostClassifier(n_estimators=200, random_state=42),
            "Gradient boosting (HGB)": HistGradientBoostingClassifier(random_state=42),
            "SVM linéaire": make_pipeline(std(), SVC(kernel="linear", C=1.0)),
            "SVM noyau RBF": make_pipeline(std(), SVC(kernel="rbf", C=1.0, gamma="scale")),
        }

    resultats_lunes = {}
    modeles_lunes = {}
    for _nom, _m in modeles(k_cv).items():
        _t = time.perf_counter()
        _m.fit(X_tr, y_tr)
        _duree = time.perf_counter() - _t
        resultats_lunes[_nom] = (1 - _m.score(X_te, y_te), _duree)
        modeles_lunes[_nom] = _m
        print(f"{_nom:26s} : erreur de test {resultats_lunes[_nom][0]:6.1%}  ({_duree * 1000:6.1f} ms)")
    return SVC, StandardScaler, make_pipeline, modeles, modeles_lunes, resultats_lunes


@app.cell
def _(SVC, StandardScaler, X_tr, make_pipeline, np, y_tr):
    # SVM linéaire sur deux nuages séparables : marge et vecteurs de support.
    _rng = np.random.default_rng(42)
    X_sep = np.r_[_rng.normal([-1.5, -1.0], 0.7, (40, 2)), _rng.normal([1.5, 1.0], 0.7, (40, 2))]
    y_sep = np.r_[np.zeros(40, int), np.ones(40, int)]
    svm_sep = SVC(kernel="linear", C=1e6).fit(X_sep, y_sep)   # C très grand : marge « dure »
    _w = svm_sep.coef_[0]
    marge_sep = 2 / np.linalg.norm(_w)
    _f = svm_sep.decision_function(X_sep)
    _y_pm = 2 * y_sep - 1
    print(f"SVM linéaire, marge dure : {len(svm_sep.support_)} vecteurs de support sur {len(X_sep)} "
          f"points ; largeur de la marge 2/‖w‖ = {marge_sep:.3f}")
    print(f"  min y·f(x) = {(_y_pm * _f).min():.4f} (≈ 1 : les vecteurs de support sont SUR la marge)")
    # Sur les lunes, combien de points deviennent vecteurs de support avec le noyau RBF ?
    _rbf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")).fit(X_tr, y_tr)
    n_sv_rbf = int(_rbf[-1].support_.size)
    print(f"SVM RBF sur les lunes : {n_sv_rbf} vecteurs de support sur {len(X_tr)} points")
    return X_sep, marge_sep, n_sv_rbf, svm_sep, y_sep


@app.cell
def _(X_tr, grille_plan, modeles_lunes, np, plt, resultats_lunes, y_tr):
    # Figure : frontières de décision des neuf modèles.
    _gx, _gy, _pts = grille_plan(200)
    fig_frontieres, _axes = plt.subplots(3, 3, figsize=(12, 11))
    for _ax, (_nom, _m) in zip(_axes.ravel(), modeles_lunes.items()):
        _z = _m.predict_proba(_pts)[:, 1] if hasattr(_m, "predict_proba") and "SVM" not in _nom \
            else (_m.decision_function(_pts) > 0).astype(float)
        _ax.contourf(_gx, _gy, _z.reshape(_gx.shape), levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.55)
        _ax.contour(_gx, _gy, _z.reshape(_gx.shape), levels=[0.5], colors="k", linewidths=1)
        _ax.scatter(X_tr[:, 0], X_tr[:, 1], c=np.where(y_tr == 1, "#D55E00", "#0072B2"),
                    s=10, edgecolors="k", linewidths=0.3)
        _ax.set(title=f"{_nom}\nerreur de test {resultats_lunes[_nom][0]:.1%}", xticks=[], yticks=[])
    fig_frontieres.tight_layout()
    return (fig_frontieres,)


@app.cell
def _(config, modeles, np, time):
    # Validation croisée sur un vrai jeu tabulaire : cancer du sein (Wisconsin, 569 patientes,
    # 30 variables numériques, livré avec scikit-learn).
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X_bc, y_bc = load_breast_cancer(return_X_y=True)
    _plis = StratifiedKFold(config["cv"], shuffle=True, random_state=42)
    resultats_bc = {}
    for _nom, _m in modeles(5).items():
        _t = time.perf_counter()
        _s = cross_val_score(_m, X_bc, y_bc, cv=_plis)
        resultats_bc[_nom] = (float(_s.mean()), float(_s.std()), time.perf_counter() - _t)
        print(f"{_nom:26s} : exactitude {_s.mean():6.1%} ± {_s.std():.1%}  "
              f"({resultats_bc[_nom][2]:5.2f} s pour {config['cv']} plis)")
    _ = np
    return StratifiedKFold, resultats_bc


@app.cell
def _(make_moons, np):
    # ===================== 4. k-means (algorithme de Lloyd), en numpy ====================
    from itertools import pairwise

    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.datasets import make_blobs
    from sklearn.metrics import adjusted_rand_score

    K_KM = 5  # nombre de groupes (connu ici : données synthétiques)
    X_km, y_km = make_blobs(600, centers=K_KM, cluster_std=1.5, random_state=42)

    def kmeans_lloyd(X, centres, n_iter_max=100):
        """Alterne (1) affecter chaque point au centre le plus proche, (2) déplacer chaque
        centre au barycentre de ses points. L'inertie ne peut que baisser à chaque étape."""
        centres = centres.copy()
        inerties = []
        for _ in range(n_iter_max):
            d2 = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
            etiquettes = d2.argmin(axis=1)
            inerties.append(float(d2[np.arange(len(X)), etiquettes].sum()))
            nouveaux = np.array([X[etiquettes == c].mean(axis=0) if np.any(etiquettes == c)
                                 else centres[c] for c in range(len(centres))])
            if np.allclose(nouveaux, centres):
                break
            centres = nouveaux
        return centres, etiquettes, inerties

    # Initialisation « naïve » : K_KM points tirés au hasard parmi les données.
    _init = X_km[np.random.default_rng(42).choice(len(X_km), K_KM, replace=False)]
    centres_np, etiq_np, inerties_np = kmeans_lloyd(X_km, _init)
    _km = KMeans(K_KM, init=_init, n_init=1, algorithm="lloyd", tol=0).fit(X_km)
    _ordre = np.lexsort(centres_np.T[::-1]), np.lexsort(_km.cluster_centers_.T[::-1])
    ecart_centres = float(np.abs(centres_np[_ordre[0]] - _km.cluster_centers_[_ordre[1]]).max())
    inertie_monotone = all(b <= a + 1e-9 for a, b in pairwise(inerties_np))
    print(f"k-means numpy : {len(inerties_np)} itérations, inertie "
          f"{inerties_np[0]:.1f} → {inerties_np[-1]:.1f} (décroissante : {inertie_monotone})")
    print(f"scikit-learn (même initialisation) : inertie {_km.inertia_:.1f}, "
          f"écart max des centres {ecart_centres:.2e}")
    print(f"Accord avec les vrais groupes (ARI, 1 = parfait) : {adjusted_rand_score(y_km, etiq_np):.3f}")
    return (
        DBSCAN,
        KMeans,
        K_KM,
        X_km,
        adjusted_rand_score,
        centres_np,
        ecart_centres,
        etiq_np,
        inertie_monotone,
        inerties_np,
        kmeans_lloyd,
        y_km,
    )


@app.cell
def _(
    DBSCAN,
    KMeans,
    K_KM,
    X_km,
    adjusted_rand_score,
    config,
    kmeans_lloyd,
    make_moons,
    np,
):
    # Minima locaux : selon l'initialisation, Lloyd converge vers des solutions différentes.
    inerties_alea, inerties_pp = [], []
    for _g in range(config["n_graines_kmeans"]):
        _init = X_km[np.random.default_rng(_g).choice(len(X_km), K_KM, replace=False)]
        inerties_alea.append(kmeans_lloyd(X_km, _init)[2][-1])
        inerties_pp.append(KMeans(K_KM, init="k-means++", n_init=1, random_state=_g).fit(X_km).inertia_)
    inerties_alea, inerties_pp = np.array(inerties_alea), np.array(inerties_pp)
    _opt = min(inerties_alea.min(), inerties_pp.min())
    part_mauvais_alea = float((inerties_alea > _opt * 1.01).mean())
    part_mauvais_pp = float((inerties_pp > _opt * 1.01).mean())
    print(f"{config['n_graines_kmeans']} initialisations : solution à plus de 1 % de la meilleure "
          f"inertie ({_opt:.1f}) dans {part_mauvais_alea:.0%} des cas (points au hasard) "
          f"et {part_mauvais_pp:.0%} (k-means++) ; pire inertie {inerties_alea.max():.1f}")

    # Groupes non convexes : k-means coupe les lunes en deux par une droite, DBSCAN suit la densité.
    X_lunes_nettes, y_lunes_nettes = make_moons(400, noise=0.06, random_state=42)
    _e_km = KMeans(2, n_init=10, random_state=42).fit_predict(X_lunes_nettes)
    _e_db = DBSCAN(eps=0.2, min_samples=5).fit_predict(X_lunes_nettes)
    ari_km_lunes = adjusted_rand_score(y_lunes_nettes, _e_km)
    ari_db_lunes = adjusted_rand_score(y_lunes_nettes, _e_db)
    etiq_km_lunes, etiq_db_lunes = _e_km, _e_db
    print(f"Lunes peu bruitées : ARI k-means {ari_km_lunes:.3f}, DBSCAN {ari_db_lunes:.3f} "
          f"({len(set(_e_db) - {-1})} groupes, {int((_e_db == -1).sum())} points de bruit)")
    return (
        X_lunes_nettes,
        ari_db_lunes,
        ari_km_lunes,
        etiq_db_lunes,
        etiq_km_lunes,
        inerties_alea,
        inerties_pp,
        part_mauvais_alea,
        part_mauvais_pp,
    )


@app.cell
def _(
    PALETTE,
    X_km,
    X_lunes_nettes,
    centres_np,
    etiq_db_lunes,
    etiq_km_lunes,
    etiq_np,
    inerties_alea,
    inerties_np,
    inerties_pp,
    np,
    plt,
):
    # Figure k-means : groupes trouvés, inertie par itération, minima locaux, lunes.
    fig_kmeans, _axes = plt.subplots(1, 4, figsize=(17, 4))
    _cols = np.array(PALETTE[:6])
    _axes[0].scatter(X_km[:, 0], X_km[:, 1], c=_cols[etiq_np], s=8)
    _axes[0].scatter(centres_np[:, 0], centres_np[:, 1], c="k", marker="X", s=120)
    _axes[0].set(title=f"k-means numpy (k = {len(centres_np)})", xticks=[], yticks=[])
    _axes[1].plot(range(1, len(inerties_np) + 1), inerties_np, "o-")
    _axes[1].set(xlabel="itération", ylabel="inertie", title="L'inertie ne fait que baisser")
    _axes[2].hist([inerties_alea, inerties_pp], bins=20, label=["init. au hasard", "k-means++"])
    _axes[2].set(xlabel="inertie finale", ylabel="nombre de graines", title="Minima locaux")
    _axes[2].legend(fontsize=8)
    _axes[3].scatter(X_lunes_nettes[:, 0], X_lunes_nettes[:, 1],
                     c=np.where(etiq_db_lunes < 0, "0.6", _cols[np.clip(etiq_db_lunes, 0, 3)]), s=8)
    _frontiere = etiq_km_lunes != etiq_km_lunes[0]
    _axes[3].scatter(X_lunes_nettes[_frontiere, 0], X_lunes_nettes[_frontiere, 1],
                     facecolors="none", edgecolors="k", s=18, linewidths=0.4)
    _axes[3].set(title="Lunes : DBSCAN (couleurs)\net k-means (points cerclés = 2e groupe)",
                 xticks=[], yticks=[])
    fig_kmeans.tight_layout()
    return (fig_kmeans,)


@app.cell
def _(SVC, StratifiedKFold, config, np, time):
    # ===================== 5. Recherche d'hyperparamètres : grille contre hasard =========
    # SVM RBF sur digits (images 8×8 de chiffres manuscrits, 1 797 exemples). Deux
    # hyperparamètres : C (régularisation) et γ (largeur du noyau), sur une échelle log.
    from scipy.stats import loguniform
    from sklearn.datasets import load_digits
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

    X_dg, y_dg = load_digits(return_X_y=True)
    X_dg_tr, X_dg_te, y_dg_tr, y_dg_te = train_test_split(
        X_dg, y_dg, test_size=0.25, stratify=y_dg, random_state=42)
    _plis = StratifiedKFold(config["cv"], shuffle=True, random_state=42)
    _budget = config["n_iter_recherche"]
    # Grille : même budget, réparti en n_C × n_γ valeurs régulières en log.
    _n_c = 5 if _budget >= 30 else 2
    _n_g = _budget // _n_c
    _grille = {"C": np.logspace(-2, 3, _n_c), "gamma": np.logspace(-5, -1, _n_g)}
    recherches = {}
    for _nom, _cherche in [
        ("grille", GridSearchCV(SVC(), _grille, cv=_plis)),
        ("hasard", RandomizedSearchCV(SVC(), {"C": loguniform(1e-2, 1e3), "gamma": loguniform(1e-5, 1e-1)},
                                      n_iter=_budget, cv=_plis, random_state=42)),
    ]:
        _t = time.perf_counter()
        _cherche.fit(X_dg_tr, y_dg_tr)
        recherches[_nom] = dict(
            essais=len(_cherche.cv_results_["params"]), cv=float(_cherche.best_score_),
            test=float(_cherche.score(X_dg_te, y_dg_te)), params=_cherche.best_params_,
            duree=time.perf_counter() - _t,
            gammas=np.array([p["gamma"] for p in _cherche.cv_results_["params"]], float),
            Cs=np.array([p["C"] for p in _cherche.cv_results_["params"]], float),
            scores=_cherche.cv_results_["mean_test_score"],
        )
        _r = recherches[_nom]
        print(f"{_nom:6s} : {_r['essais']} essais, CV {_r['cv']:.2%}, test {_r['test']:.2%}, "
              f"C = {_r['params']['C']:.3g}, γ = {_r['params']['gamma']:.3g} ({_r['duree']:.1f} s) ; "
              f"{len(np.unique(_r['gammas']))} valeurs de γ distinctes essayées")
    return X_dg, recherches, y_dg


@app.cell
def _(X_dg, config, np, y_dg):
    # ===================== 6. ACP en numpy (SVD), comparée à scikit-learn ================
    from sklearn.decomposition import PCA

    def acp(X, n_composantes):
        """Centrer, puis décomposition en valeurs singulières : les lignes de Vt sont les
        directions de variance maximale ; S²/(n−1) donne la variance de chaque direction."""
        Xc = X - X.mean(axis=0)
        _U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        variance = S**2 / (len(X) - 1)
        return Xc @ Vt[:n_composantes].T, variance[:n_composantes] / variance.sum(), Vt[:n_composantes]

    Z_acp, part_var_np, _ = acp(X_dg, 2)
    _pca = PCA(2).fit(X_dg)
    ecart_acp = float(np.abs(part_var_np - _pca.explained_variance_ratio_).max())
    _, _cumul, _ = acp(X_dg, 64)
    n_90 = int(np.searchsorted(np.cumsum(_cumul), 0.90) + 1)
    print(f"ACP numpy : variance expliquée par 2 composantes {part_var_np.sum():.1%} "
          f"({part_var_np[0]:.1%} + {part_var_np[1]:.1%}) ; scikit-learn identique à {ecart_acp:.1e} près")
    print(f"Il faut {n_90} composantes sur 64 pour garder 90 % de la variance.")
    _ = config, y_dg
    return PCA, Z_acp, ecart_acp, n_90, part_var_np


@app.cell
def _(KNeighborsClassifier, X_dg, Z_acp, config, np, time, y_dg):
    # t-SNE : une carte 2D qui préserve les VOISINAGES, pas les distances.
    from sklearn.manifold import TSNE
    from sklearn.model_selection import cross_val_score as _cvs

    _n = config["n_tsne"] or len(X_dg)
    _idx = np.arange(_n)
    y_carte = y_dg[_idx]
    # Mesure commune : exactitude d'un 5-NN dans la carte 2D (validation croisée à 5 plis).
    # Elle dit si les chiffres d'une même classe restent voisins une fois projetés.
    def separation(Z):
        return float(_cvs(KNeighborsClassifier(5), Z, y_carte, cv=5).mean())

    sep_acp = separation(Z_acp[_idx])
    cartes_tsne, sep_tsne, durees_tsne = {}, {}, {}
    for _p in config["perplexites"]:
        _t = time.perf_counter()
        cartes_tsne[_p] = TSNE(perplexity=_p, max_iter=config["tsne_iter"], init="pca",
                               random_state=42).fit_transform(X_dg[_idx])
        durees_tsne[_p] = time.perf_counter() - _t
        sep_tsne[_p] = separation(cartes_tsne[_p])
    print(f"5-NN dans la carte 2D : ACP {sep_acp:.1%} ; t-SNE " +
          ", ".join(f"perplexité {p} : {s:.1%} ({durees_tsne[p]:.1f} s)" for p, s in sep_tsne.items()))
    return TSNE, cartes_tsne, sep_acp, sep_tsne, y_carte


@app.cell
def _(TSNE, config, np):
    # Piège de t-SNE : la TAILLE apparente des groupes ne veut rien dire.
    # Deux groupes gaussiens en dimension 10, l'un 10 fois plus étalé que l'autre.
    _rng = np.random.default_rng(42)
    _A = _rng.normal(0, 1, (250, 10))
    _B = _rng.normal(0, 10, (250, 10)) + 100
    _X = np.r_[_A, _B]
    _Z = TSNE(perplexity=30, max_iter=config["tsne_iter"], init="pca", random_state=42).fit_transform(_X)

    def etalement(P):
        return float(np.sqrt(((P - P.mean(axis=0)) ** 2).sum(axis=1).mean()))

    rapport_origine = etalement(_B) / etalement(_A)
    rapport_tsne = etalement(_Z[250:]) / etalement(_Z[:250])
    print(f"Étalement du groupe B / groupe A : {rapport_origine:.1f} dans l'espace d'origine, "
          f"{rapport_tsne:.2f} dans la carte t-SNE")
    return rapport_origine, rapport_tsne


@app.cell
def _(PALETTE, Z_acp, cartes_tsne, np, plt, sep_acp, sep_tsne, y_carte):
    # Figure : ACP contre t-SNE sur digits.
    _cmap = plt.get_cmap("tab10")
    _n = len(y_carte)
    fig_cartes, _axes = plt.subplots(1, 1 + len(cartes_tsne), figsize=(4.4 * (1 + len(cartes_tsne)), 4.2))
    _axes = np.atleast_1d(_axes)
    _axes[0].scatter(Z_acp[:_n, 0], Z_acp[:_n, 1], c=y_carte, cmap=_cmap, s=4)
    _axes[0].set(title=f"ACP (2 composantes)\n5-NN dans la carte : {sep_acp:.0%}", xticks=[], yticks=[])
    for _ax, (_p, _Z) in zip(_axes[1:], cartes_tsne.items()):
        _ax.scatter(_Z[:, 0], _Z[:, 1], c=y_carte, cmap=_cmap, s=4)
        _ax.set(title=f"t-SNE, perplexité {_p}\n5-NN dans la carte : {sep_tsne[_p]:.0%}",
                xticks=[], yticks=[])
    fig_cartes.tight_layout()
    _ = PALETTE
    return (fig_cartes,)


@app.cell
def _(
    SMOKE,
    contraste,
    fig_cartes,
    fig_frontieres,
    fig_kmeans,
    fig_knn,
    mo,
    recherches,
    plt,
    sauvegarder_figure,
):
    # Figure : grille contre hasard, puis sauvegarde de toutes les figures.
    fig_recherche, _axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for _ax, _nom in zip(_axes, ["grille", "hasard"]):
        _r = recherches[_nom]
        _sc = _ax.scatter(_r["Cs"], _r["gammas"], c=_r["scores"], cmap="viridis", vmin=0.1, vmax=1, s=40)
        _ax.scatter([_r["params"]["C"]], [_r["params"]["gamma"]], marker="*", s=250, c="#D55E00")
        _ax.set(xscale="log", yscale="log", xlabel="C", title=f"{_nom} : {_r['essais']} essais, "
                f"{len(set(_r['gammas'].round(12)))} valeurs de γ")
    _axes[0].set_ylabel("γ")
    fig_recherche.colorbar(_sc, ax=_axes, label="exactitude moyenne (CV)")
    _ = contraste

    _dossier = mo.notebook_dir() / "figures"
    _suffixe = "-smoke" if SMOKE else ""
    for _nom, _fig in [("knn", fig_knn), ("frontieres", fig_frontieres), ("kmeans", fig_kmeans),
                       ("recherche", fig_recherche), ("acp-tsne", fig_cartes)]:
        print("Figure :", sauvegarder_figure(_fig, _nom + _suffixe, _dossier)[0])


@app.cell
def _(
    SMOKE,
    T0,
    ari_db_lunes,
    ari_km_lunes,
    contraste,
    ecart_acp,
    ecart_centres,
    ecarts_sklearn,
    err_arbre,
    err_arbre_plein,
    err_bruit,
    err_foret,
    err_te,
    err_te_kcv,
    err_tr,
    inertie_monotone,
    mo,
    rapport_origine,
    rapport_tsne,
    recherches,
    resultats_bc,
    sep_acp,
    sep_tsne,
    souche_ok,
    time,
):
    # --- Bilan : seuils du README. En smoke, on vérifie seulement que tout tourne. ---
    _duree = time.perf_counter() - T0
    _d_max = max(contraste)
    verifs = {
        "k-NN numpy = scikit-learn (0 prédiction différente)": all(v == 0 for v in ecarts_sklearn.values()),
        "1-NN : erreur d'entraînement nulle": err_tr[0] == 0,
        "1-NN : test ≥ meilleur k + 3 points": err_te[0] >= err_te.min() + 0.03,
        "k = 199 : test ≥ meilleur k + 10 points": err_te[-1] >= err_te.min() + 0.10,
        "k choisi par CV : erreur de test ≤ 12 %": err_te_kcv <= 0.12,
        "dimension 1000 : plus lointain / plus proche ≤ 1,5": contraste[_d_max][1] / contraste[_d_max][0] <= 1.5,
        "50 dimensions de bruit : erreur ≥ +10 points": err_bruit[max(err_bruit)] >= err_bruit[0] + 0.10,
        "souche de Gini numpy = scikit-learn": souche_ok,
        "arbre complet : entraînement 0 %, test > profondeur optimale":
            err_arbre_plein[0] == 0 and err_arbre_plein[1] > min(e[1] for e in err_arbre.values()),
        "forêt : test ≤ arbre complet": err_foret[1] <= err_arbre_plein[1],
        "cancer du sein : SVM RBF ≥ 96 % en CV": resultats_bc["SVM noyau RBF"][0] >= 0.96,
        "cancer du sein : forêt et HGB ≥ 94 %": min(resultats_bc["Forêt aléatoire"][0],
                                                    resultats_bc["Gradient boosting (HGB)"][0]) >= 0.94,
        "k-means numpy = scikit-learn (centres à 1e-6)": ecart_centres <= 1e-6,
        "k-means : inertie décroissante": inertie_monotone,
        "lunes : DBSCAN ARI ≥ 0,95 et k-means ARI ≤ 0,6": ari_db_lunes >= 0.95 and ari_km_lunes <= 0.6,
        "digits : meilleur SVM ≥ 97 % sur le test": max(r["test"] for r in recherches.values()) >= 0.97,
        "ACP numpy = scikit-learn (1e-10)": ecart_acp <= 1e-10,
        "t-SNE : 5-NN ≥ ACP + 25 points": max(sep_tsne.values()) >= sep_acp + 0.25,
        "t-SNE : étalement 10× → moins de 3×": rapport_origine >= 8 and rapport_tsne <= 3,
    }
    for _k, _ok in verifs.items():
        print(f"  [{'ok' if _ok else 'ÉCHEC'}] {_k}")
    print(f"Durée totale des calculs : {_duree:.1f} s")
    if not SMOKE and not all(verifs.values()):
        raise SystemExit("Seuils non atteints : " + ", ".join(k for k, ok in verifs.items() if not ok))
    mo.md(f"**Bilan** : {sum(verifs.values())}/{len(verifs)} vérifications réussies, "
          f"{_duree:.0f} s de calcul.")


if __name__ == "__main__":
    app.run()
