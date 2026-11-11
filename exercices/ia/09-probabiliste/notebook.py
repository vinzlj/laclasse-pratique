# Exercice 09 : modèles probabilistes et variables latentes (chapitre 09).
#
#   make ex NUM=09              → éditeur marimo (interactif, curseur sur les itérations d'EM)
#   make ex NUM=09 MODE=run     → script complet (vérifie les seuils du README)
#   make ex NUM=09 MODE=smoke   → test rapide (< 30 s)
#
# Tout est écrit en numpy « from scratch », puis comparé à scikit-learn quand une
# implémentation de référence existe (GaussianMixture, KMeans). Aucun GPU, aucun
# téléchargement. Les algorithmes sont des fonctions « @app.function » : les scripts de
# figures du site (site/scripts/figures/ch09_*.py) les importent pour refaire EXACTEMENT
# les mêmes calculs.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import itertools
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Ellipse

    from ia_commun import PALETTE, is_smoke, sauvegarder_figure, seed_everything, style_commun


@app.cell
def _():
    mo.md(r"""
    # Exercice 09 : raisonner sous incertitude, deviner les causes cachées

    Cinq petites expériences, toutes en numpy :

    1. **Réseau bayésien** (pluie, arroseur, herbe mouillée) : inférence exacte par
       énumération, puis par échantillonnage avec rejet. Phénomène observé : l'**effet
       d'explication** (*explaining away*), isolé dans une variante où pluie et arroseur
       sont indépendants.
    2. **Mélange de gaussiennes et EM** : trois groupes de points dont on ignore
       l'étiquette. EM alterne « deviner les étiquettes (en probabilité) » et « réestimer
       les gaussiennes ». On suit la log-vraisemblance et l'**ELBO** à chaque itération,
       et on compare à k-means et à scikit-learn.
    3. **Modèle de Markov caché (HMM)** : la météo est cachée, on n'observe que la façon
       dont une collègue vient travailler. Algorithme forward, Viterbi, forward-backward,
       puis Baum-Welch (EM pour les HMM) pour réapprendre le modèle sans voir la météo.
    4. **MCMC** : Metropolis-Hastings sur une densité 2D à deux bosses, avec trois tailles
       de pas.
    5. **Inférence variationnelle** : la loi a posteriori d'une régression logistique
       bayésienne, calculée exactement sur une grille, par MCMC, puis approchée par une
       gaussienne qui maximise l'ELBO.

    **Ce qui est optimisé.** Parties 2 et 3 : la **log-vraisemblance** des observations
    $\log p(x \mid \theta)$ (on minimise donc la log-vraisemblance négative), à travers sa
    borne inférieure, l'**ELBO** :

    $$\log p(x \mid \theta) \;=\; \underbrace{\mathbb{E}_{q(z)}\!\left[\log \frac{p(x, z \mid \theta)}{q(z)}\right]}_{\text{ELBO}(q,\,\theta)} \;+\; \mathrm{KL}\big(q(z)\,\|\,p(z \mid x, \theta)\big).$$

    Partie 5 : l'ELBO elle-même, cette fois en fonction des paramètres d'une loi
    approchée $q$. Parties 1 et 4 : **aucune** perte, on calcule ou on échantillonne
    une loi, on n'ajuste rien.
    """)
    return


@app.cell
def _():
    T0 = time.perf_counter()  # chronomètre de l'ensemble des calculs
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        n_rejet=2_000 if SMOKE else 100_000,       # tirages du réseau bayésien
        n_par_groupe=100 if SMOKE else 300,        # points par composante du mélange
        n_inits_em=3 if SMOKE else 10,             # initialisations aléatoires d'EM
        t_hmm=200 if SMOKE else 365,               # jours simulés pour Viterbi
        t_baum_welch=1_000 if SMOKE else 5_000,    # jours pour réapprendre le HMM
        iter_baum_welch=15 if SMOKE else 200,
        n_inits_bw=2 if SMOKE else 3,
        n_mh=3_000 if SMOKE else 30_000,           # pas de Metropolis-Hastings (partie 4)
        n_mh_post=3_000 if SMOKE else 30_000,      # pas de MH sur la loi a posteriori (partie 5)
        n_pas_vi=300 if SMOKE else 3_000,          # pas d'Adam pour l'inférence variationnelle
        grille=120 if SMOKE else 400,              # résolution de la grille « exacte » (partie 5)
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SEED, SMOKE, T0, config


@app.cell
def _():
    mo.md(r"""
    ## 1. Réseau bayésien : pluie, arroseur, herbe mouillée

    Trois variables binaires. La pluie influence l'arroseur (on ne l'allume pas quand il
    pleut) et l'herbe ; l'arroseur influence l'herbe. Le graphe orienté dit quelles
    probabilités conditionnelles suffisent :

    $$P(P, A, H) = P(P)\;P(A \mid P)\;P(H \mid A, P)$$

    soit 1 + 2 + 4 = **7 nombres** au lieu des $2^3 - 1 = 7$ de la table complète. Le gain
    est nul à trois variables, mais à 30 variables binaires ayant au plus 3 parents, on
    passe de $2^{30} - 1 \approx 10^9$ nombres à au plus $30 \times 2^3 = 240$.

    Valeurs (celles du schéma de Wikipédia, `File:SimpleBayesNet.svg`) :
    $P(\text{pluie}) = 0{,}2$ ; $P(\text{arroseur} \mid \text{pluie}) = 0{,}01$,
    $P(\text{arroseur} \mid \neg\text{pluie}) = 0{,}4$ ; $P(\text{herbe} \mid A, P)$ =
    0 (ni l'un ni l'autre), 0,8 (pluie seule), 0,9 (arroseur seul), 0,99 (les deux).

    **Inférence exacte par énumération** : $P(P \mid H) = \sum_A P(P, A, H) / \sum_{P, A} P(P, A, H)$.
    Le coût est exponentiel en nombre de variables cachées ; c'est acceptable ici.
    """)
    return


@app.function
def jointe_arroseur(p_arroseur_si_pluie=0.01, p_arroseur_sans_pluie=0.4):
    """Loi jointe du réseau pluie → arroseur, (pluie, arroseur) → herbe.

    Par défaut, les valeurs du schéma Commons (l'arroseur dépend de la pluie). Avec
    p_arroseur_si_pluie = p_arroseur_sans_pluie, la flèche pluie → arroseur disparaît :
    pluie et arroseur sont indépendants a priori (variante « effet d'explication pur »).
    Renvoie un dict {(pluie, arroseur, herbe): probabilité} sur les 8 combinaisons.
    """
    p_pluie = 0.2
    p_arroseur = {True: p_arroseur_si_pluie, False: p_arroseur_sans_pluie}  # P(arroseur | pluie)
    p_herbe = {  # P(herbe mouillée | arroseur, pluie)
        (False, False): 0.0,
        (False, True): 0.8,
        (True, False): 0.9,
        (True, True): 0.99,
    }
    jointe = {}
    for p, a, h in itertools.product([True, False], repeat=3):
        pp = p_pluie if p else 1 - p_pluie
        pa = p_arroseur[p] if a else 1 - p_arroseur[p]
        ph = p_herbe[(a, p)] if h else 1 - p_herbe[(a, p)]
        jointe[(p, a, h)] = pp * pa * ph
    return jointe


@app.function
def probabilite(jointe, requete, evidence=None):
    """P(requête | évidence) par énumération de la loi jointe.

    requete, evidence : dict {"pluie" | "arroseur" | "herbe": bool}.
    """
    noms = ("pluie", "arroseur", "herbe")
    evidence = evidence or {}

    def compatible(valeurs, contraintes):
        return all(valeurs[noms.index(k)] == v for k, v in contraintes.items())

    num = sum(p for v, p in jointe.items() if compatible(v, {**evidence, **requete}))
    den = sum(p for v, p in jointe.items() if compatible(v, evidence))
    return num / den


@app.function
def echantillonner_arroseur(rng, n):
    """Échantillonnage ancestral : on tire les parents avant les enfants."""
    pluie = rng.random(n) < 0.2
    arroseur = rng.random(n) < np.where(pluie, 0.01, 0.4)
    p_h = np.select([arroseur & pluie, arroseur, pluie], [0.99, 0.9, 0.8], default=0.0)
    herbe = rng.random(n) < p_h
    return pluie, arroseur, herbe


@app.cell
def _(SEED, config):
    _j = jointe_arroseur()
    reseau = {
        "P(pluie)": probabilite(_j, {"pluie": True}),
        "P(herbe)": probabilite(_j, {"herbe": True}),
        "P(pluie | herbe)": probabilite(_j, {"pluie": True}, {"herbe": True}),
        "P(arroseur | herbe)": probabilite(_j, {"arroseur": True}, {"herbe": True}),
        "P(pluie | herbe, arroseur)": probabilite(_j, {"pluie": True}, {"herbe": True, "arroseur": True}),
        "P(pluie | herbe, pas d'arroseur)": probabilite(_j, {"pluie": True}, {"herbe": True, "arroseur": False}),
        # Sans regarder l'herbe : la flèche pluie → arroseur suffit à faire chuter la pluie
        "P(pluie | arroseur)": probabilite(_j, {"pluie": True}, {"arroseur": True}),
    }
    for _k, _v in reseau.items():
        print(f"{_k:<34} = {_v:.4f}")

    # Variante sans flèche pluie → arroseur (arroseur à 0,4 qu'il pleuve ou non) :
    # pluie et arroseur sont indépendants a priori, l'effet d'explication apparaît seul.
    _ji = jointe_arroseur(0.4, 0.4)
    reseau_indep = {
        "P(pluie)": probabilite(_ji, {"pluie": True}),
        "P(pluie | arroseur)": probabilite(_ji, {"pluie": True}, {"arroseur": True}),
        "P(pluie | herbe)": probabilite(_ji, {"pluie": True}, {"herbe": True}),
        "P(pluie | herbe, arroseur)": probabilite(_ji, {"pluie": True}, {"herbe": True, "arroseur": True}),
    }
    print("Variante à causes indépendantes (P(arroseur) = 0,4) :")
    for _k, _v in reseau_indep.items():
        print(f"  {_k:<32} = {_v:.4f}")

    # Échantillonnage avec rejet : on tire selon le réseau et on GARDE les tirages où
    # l'herbe est mouillée. Estimation de P(pluie | herbe) = proportion de pluie parmi eux.
    _rng = np.random.default_rng(SEED)
    _p, _a, _h = echantillonner_arroseur(_rng, config["n_rejet"])
    _tailles = np.unique(np.logspace(1, np.log10(config["n_rejet"]), 30).astype(int))
    rejet = dict(
        tailles=_tailles,
        estimations=np.array([_p[:n][_h[:n]].mean() if _h[:n].any() else np.nan for n in _tailles]),
        garde=_h.mean(),
        estimation=_p[_h].mean(),
    )
    print(f"Rejet : {config['n_rejet']} tirages, {rejet['garde']:.1%} gardés, "
          f"P(pluie | herbe) ≈ {rejet['estimation']:.4f} (exact {reseau['P(pluie | herbe)']:.4f})")
    return reseau, reseau_indep, rejet


@app.cell
def _():
    mo.md(r"""
    **À observer.** Voir l'herbe mouillée fait passer la pluie de 20 % à environ 36 %.
    Apprendre ensuite que l'arroseur tournait la fait **retomber** sous 1 %. Attention :
    dans ce réseau, l'arroseur dépend déjà de la pluie (il ne tourne presque jamais
    quand il pleut) ; sans même regarder l'herbe, $P(\text{pluie} \mid \text{arroseur})$
    vaut 0,6 %. La chute vient donc surtout de la flèche pluie → arroseur.

    La **variante à causes indépendantes** (arroseur à 0,4 qu'il pleuve ou non) isole
    l'**effet d'explication** : savoir l'arroseur en marche ne change rien à la pluie
    (20 %) ; l'herbe mouillée la fait monter à 37,8 % ; apprendre en plus que l'arroseur
    tournait la fait redescendre à 21,6 %, car l'arroseur « explique » l'herbe mouillée.
    Deux causes indépendantes deviennent dépendantes quand on observe leur effet commun.
    Ce raisonnement à rebours, d'un effet vers ses causes concurrentes, est exactement ce
    que des règles « SI … ALORS » du chapitre 05 ne savent pas faire proprement.

    L'échantillonnage avec rejet jette plus de la moitié des tirages (ceux où l'herbe est
    sèche). Avec une évidence rare (probabilité 0,001), il en jetterait 99,9 % : c'est ce
    qui motive les méthodes MCMC de la partie 4.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 2. Mélange de gaussiennes et algorithme EM

    Modèle génératif : pour chaque point, on tire une composante cachée
    $z \in \{1, \dots, K\}$ avec probabilité $\pi_z$, puis $x \sim \mathcal{N}(\mu_z, \Sigma_z)$.
    On ne voit que $x$. Maximiser directement
    $\log p(x \mid \theta) = \sum_i \log \sum_k \pi_k\, \mathcal{N}(x_i; \mu_k, \Sigma_k)$
    est difficile (un log de somme). EM contourne le problème :

    - **E (espérance)** : avec les paramètres actuels, calculer pour chaque point la
      **responsabilité** $r_{ik} = P(z_i = k \mid x_i, \theta)$, une étiquette « douce » ;
    - **M (maximisation)** : réestimer $\pi_k, \mu_k, \Sigma_k$ comme des moyennes
      pondérées par $r_{ik}$ (formules fermées).

    k-means est le cas limite où les responsabilités valent 0 ou 1 et où toutes les
    covariances sont identiques et sphériques : EM est un « k-means doux ».

    Le lien avec l'ELBO : après l'étape E, $q = p(z \mid x, \theta)$, donc KL = 0 et
    ELBO = log-vraisemblance. L'étape M fait monter l'ELBO ; la log-vraisemblance monte
    au moins autant, puisqu'elle vaut ELBO + KL avec KL ≥ 0. D'où la garantie : **la
    log-vraisemblance ne baisse jamais** d'une itération à l'autre. On le vérifie.
    """)
    return


@app.function
def donnees_melange(rng, n_par_groupe):
    """Trois groupes : deux bandes allongées parallèles et inclinées, un amas rond.

    Renvoie X (n, 2) et les vraies étiquettes (pour l'évaluation seulement).
    """
    angle = np.deg2rad(25)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    bande = rot @ np.diag([2.2**2, 0.3**2]) @ rot.T  # covariance allongée et inclinée
    specs = [
        ((0.0, 0.0), bande),
        ((-0.8, 1.7), bande),
        ((4.5, -2.5), np.diag([0.5**2, 0.5**2])),
    ]
    X, y = [], []
    for k, (m, c) in enumerate(specs):
        X.append(rng.multivariate_normal(m, c, size=n_par_groupe))
        y.append(np.full(n_par_groupe, k))
    return np.concatenate(X), np.concatenate(y)


@app.function
def log_gauss(X, mu, cov):
    """log N(x; mu, cov) pour chaque ligne de X (via la décomposition de Cholesky)."""
    d = X.shape[1]
    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, (X - mu).T)
    return -0.5 * np.sum(z**2, axis=0) - np.log(np.diag(L)).sum() - 0.5 * d * np.log(2 * np.pi)


@app.function
def logsumexp(a, axis):
    """log(sum(exp(a))) sans débordement numérique."""
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


@app.function
def log_joint_melange(X, pi, mu, cov):
    """Matrice (n, K) de log[π_k N(x_i; μ_k, Σ_k)]."""
    return np.stack([np.log(pi[k]) + log_gauss(X, mu[k], cov[k]) for k in range(len(pi))], axis=1)


@app.function
def em_melange(X, K, rng, n_iter=300, tol=1e-9):
    """EM pour un mélange de K gaussiennes à covariance pleine, from scratch.

    Initialisation : K points tirés au hasard comme moyennes, covariance globale des
    données pour chaque composante, poids uniformes.
    Renvoie un dict : paramètres finaux, responsabilités, et à chaque itération t :
    historique[t] = (π, μ, Σ) AVANT l'étape E de t, logv[t] = log p(X | θ_t),
    elbo[t] = ELBO(q_t, θ_{t+1}) mesurée juste après l'étape M de t.
    """
    n, d = X.shape
    mu = X[rng.choice(n, K, replace=False)].copy()
    cov = np.array([np.cov(X.T)] * K)
    pi = np.full(K, 1.0 / K)
    historique, logv, elbo, kl = [], [], [], []
    for t in range(n_iter):
        # --- Étape E : responsabilités r_ik = P(z_i = k | x_i, θ_t) ---
        lj = log_joint_melange(X, pi, mu, cov)
        lse = logsumexp(lj, axis=1)
        r = np.exp(lj - lse[:, None])
        historique.append((pi.copy(), mu.copy(), cov.copy()))
        logv.append(lse.sum())
        if t > 0 and logv[-1] - logv[-2] < tol * abs(logv[-1]):
            break
        # --- Étape M : moyennes pondérées par les responsabilités ---
        nk = r.sum(axis=0)
        pi = nk / n
        mu = (r.T @ X) / nk[:, None]
        cov = np.stack([
            ((r[:, k, None] * (X - mu[k])).T @ (X - mu[k])) / nk[k] + 1e-6 * np.eye(d)
            for k in range(K)
        ])
        # --- ELBO(q = r, θ_{t+1}) = Σ_i Σ_k r_ik [log π_k N(x_i | θ_k) − log r_ik] ---
        lj_new = log_joint_melange(X, pi, mu, cov)
        r_log_r = np.where(r > 0, r * np.log(np.where(r > 0, r, 1.0)), 0.0)
        elbo.append(np.sum(r * lj_new) - np.sum(r_log_r))
        # KL(q || p(z | x, θ_{t+1})) calculée directement, pour vérifier logv = ELBO + KL
        lpost_new = lj_new - logsumexp(lj_new, axis=1)[:, None]
        kl.append(np.sum(r_log_r) - np.sum(r * lpost_new))
    return dict(pi=pi, mu=mu, cov=cov, r=r, historique=historique,
                logv=np.array(logv), elbo=np.array(elbo), kl=np.array(kl))


@app.function
def associer_etiquettes(vrai, predit, K):
    """Meilleure permutation des étiquettes prédites (pour comparer à la vérité)."""
    meilleur = max(itertools.permutations(range(K)),
                   key=lambda p: np.mean(np.array(p)[predit] == vrai))
    return np.array(meilleur)[predit]


@app.function
def dessiner_ellipses(ax, mu, cov, couleurs, alpha=0.9):
    """Ellipses à 1 et 2 écarts-types de chaque composante."""
    for k in range(len(mu)):
        vals, vecs = np.linalg.eigh(cov[k])
        angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
        for nsig, ls in ((1, "-"), (2, "--")):
            ax.add_patch(Ellipse(mu[k], 2 * nsig * np.sqrt(vals[-1]), 2 * nsig * np.sqrt(vals[0]),
                                 angle=angle, fill=False, color=couleurs[k], lw=1.6, ls=ls, alpha=alpha))
        ax.plot(*mu[k], "x", color=couleurs[k], ms=9, mew=2.5)


@app.cell
def _(SEED, config):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture

    _rng = np.random.default_rng(SEED)
    X_mel, y_mel = donnees_melange(_rng, config["n_par_groupe"])

    # Plusieurs initialisations : EM ne trouve qu'un optimum LOCAL
    _t0 = time.perf_counter()
    runs_em = [em_melange(X_mel, 3, np.random.default_rng(SEED + i)) for i in range(config["n_inits_em"])]
    _t_em = time.perf_counter() - _t0
    em = max(runs_em, key=lambda r: r["logv"][-1])
    _pred_em = em["r"].argmax(axis=1)

    _km = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(X_mel)
    _gm = GaussianMixture(3, covariance_type="full", n_init=10, tol=1e-9, max_iter=1000,
                          random_state=SEED).fit(X_mel)
    res_melange = dict(
        n=len(X_mel),
        logv_finales=np.array([r["logv"][-1] for r in runs_em]),
        iterations=len(em["logv"]) - 1,  # logv[0] est la valeur initiale, avant toute itération
        ari_em=adjusted_rand_score(y_mel, _pred_em),
        ari_kmeans=adjusted_rand_score(y_mel, _km.labels_),
        ari_sklearn=adjusted_rand_score(y_mel, _gm.predict(X_mel)),
        logv_moy_em=em["logv"][-1] / len(X_mel),
        logv_moy_sklearn=_gm.score(X_mel),  # log-vraisemblance moyenne par point
        pred_kmeans=_km.labels_,
        pred_em=_pred_em,
        duree=_t_em,
    )
    _diff = np.diff(em["logv"])
    _pairs = em["logv"][1:]  # logv[t+1] doit être ≥ elbo[t] ≥ logv[t]
    res_melange["monotone"] = bool(np.all(_diff >= -1e-8))
    res_melange["elbo_sous_logv"] = bool(np.all(em["elbo"][: len(_pairs)] <= _pairs + 1e-8))
    res_melange["ecart_identite"] = float(np.max(np.abs(_pairs - (em["elbo"][: len(_pairs)] + em["kl"][: len(_pairs)]))))

    print(f"EM ({config['n_inits_em']} initialisations, {_t_em:.2f} s) : log-vraisemblances finales "
          f"{np.round(res_melange['logv_finales'], 1)}")
    print(f"Meilleure : {em['logv'][-1]:.2f} en {res_melange['iterations']} itérations "
          f"(log-vraisemblance initiale {em['logv'][0]:.1f})")
    print(f"Log-vraisemblance moyenne par point : EM numpy {res_melange['logv_moy_em']:.5f}, "
          f"scikit-learn {res_melange['logv_moy_sklearn']:.5f}")
    print(f"Indice de Rand ajusté (1 = partition parfaite) : EM {res_melange['ari_em']:.3f}, "
          f"scikit-learn GMM {res_melange['ari_sklearn']:.3f}, k-means {res_melange['ari_kmeans']:.3f}")
    print(f"Log-vraisemblance jamais en baisse : {res_melange['monotone']} ; ELBO ≤ log-vraisemblance : "
          f"{res_melange['elbo_sous_logv']} ; |logv − (ELBO + KL)| max = {res_melange['ecart_identite']:.1e}")
    return X_mel, em, res_melange, y_mel


@app.cell
def _(em):
    # Curseur : rejouer EM itération par itération dans l'éditeur.
    # En script, la valeur par défaut (dernière itération) est utilisée.
    curseur_em = mo.ui.slider(0, len(em["historique"]) - 1, value=len(em["historique"]) - 1,
                              label="Itération d'EM", show_value=True, full_width=True)
    curseur_em
    return (curseur_em,)


@app.cell
def _(X_mel, curseur_em, em):
    _t = curseur_em.value
    _pi, _mu, _cov = em["historique"][_t]
    _r = np.exp(log_joint_melange(X_mel, _pi, _mu, _cov)
                - logsumexp(log_joint_melange(X_mel, _pi, _mu, _cov), axis=1)[:, None])
    _fig, _ax = plt.subplots(figsize=(6, 4.5))
    # Couleur de chaque point = mélange des couleurs pondéré par ses responsabilités
    _rgb = np.array([plt.matplotlib.colors.to_rgb(c) for c in PALETTE[:3]])
    _ax.scatter(X_mel[:, 0], X_mel[:, 1], c=_r @ _rgb, s=8)
    dessiner_ellipses(_ax, _mu, _cov, PALETTE[:3])
    _ax.set(title=f"EM, itération {_t} : log-vraisemblance {em['logv'][_t]:.1f}", aspect="equal")
    _fig
    return


@app.cell
def _(SMOKE, X_mel, em, res_melange):
    # --- Figure de la partie 2 : instantanés d'EM, courbes, k-means contre EM ---
    _fig, _axes = plt.subplots(1, 5, figsize=(20, 4))
    _iters = sorted({0, 1, 3, len(em["historique"]) - 1})
    _rgb = np.array([plt.matplotlib.colors.to_rgb(c) for c in PALETTE[:3]])
    for _ax, _t in zip(_axes[:4], _iters):
        _pi, _mu, _cov = em["historique"][_t]
        _lj = log_joint_melange(X_mel, _pi, _mu, _cov)
        _r = np.exp(_lj - logsumexp(_lj, axis=1)[:, None])
        _ax.scatter(X_mel[:, 0], X_mel[:, 1], c=_r @ _rgb, s=5)
        dessiner_ellipses(_ax, _mu, _cov, PALETTE[:3])
        _ax.set(title=f"itération {_t}", aspect="equal", xlim=(-6, 7), ylim=(-5, 5))
    _ax = _axes[4]
    _ax.plot(em["logv"], "o-", ms=3, label="log-vraisemblance")
    _ax.plot(np.arange(len(em["elbo"])) + 0.5, em["elbo"], "s", ms=3, label="ELBO après l'étape M")
    _ax.set(xlabel="itération", title="Ne baisse jamais", ylim=(em["logv"][1] - 50, em["logv"][-1] + 30))
    _ax.legend()
    _fig.tight_layout()
    sauvegarder_figure(_fig, "em-smoke" if SMOKE else "em", mo.notebook_dir() / "figures")

    _fig2, (_a, _b) = plt.subplots(1, 2, figsize=(10, 4))
    _a.scatter(X_mel[:, 0], X_mel[:, 1], c=[PALETTE[i] for i in res_melange["pred_kmeans"]], s=5)
    _a.set(title=f"k-means (ARI {res_melange['ari_kmeans']:.2f})", aspect="equal")
    _b.scatter(X_mel[:, 0], X_mel[:, 1], c=[PALETTE[i] for i in res_melange["pred_em"]], s=5)
    dessiner_ellipses(_b, em["mu"], em["cov"], ["#444444"] * 3, alpha=0.6)
    _b.set(title=f"EM, mélange de gaussiennes (ARI {res_melange['ari_em']:.2f})", aspect="equal")
    _fig2.tight_layout()
    sauvegarder_figure(_fig2, "kmeans-em-smoke" if SMOKE else "kmeans-em", mo.notebook_dir() / "figures")
    _fig
    return


@app.cell
def _():
    mo.md(r"""
    ## 3. Modèle de Markov caché : la météo vue depuis le bureau

    Chaque jour, la météo $s_t \in$ {soleil, nuages, pluie} suit une **chaîne de Markov**
    (demain ne dépend que d'aujourd'hui, matrice de transition $A$). On ne la voit pas :
    on voit seulement comment une collègue vient travailler, $o_t \in$ {vélo, transports,
    télétravail}, tiré selon une loi d'émission $B$ qui dépend de la météo du jour.

    Trois questions, trois algorithmes, tous en $O(T K^2)$ au lieu de $O(K^T)$ :

    | Question | Algorithme | Idée |
    |---|---|---|
    | Quelle est la probabilité de cette suite d'observations ? | **forward** | somme sur les chemins, mise en commun jour après jour |
    | Quelle est la suite de météos la plus probable ? | **Viterbi** | même calcul avec un max au lieu de la somme, puis remontée |
    | Quelle est la météo la plus probable **chaque jour** ? | **forward-backward** | probabilité de chaque état sachant toute la séquence |
    | On ne connaît pas $A$ et $B$ | **Baum-Welch** | EM : forward-backward (E) puis comptages pondérés (M) |

    On vérifie forward et Viterbi contre une **énumération brute** des $3^8 = 6\,561$
    chemins possibles sur 8 jours.
    """)
    return


@app.function
def hmm_meteo():
    """Paramètres du HMM météo : π0 (3,), A (3, 3), B (3, 3)."""
    pi0 = np.array([0.5, 0.3, 0.2])
    A = np.array([  # A[i, j] = P(météo de demain = j | météo d'aujourd'hui = i)
        [0.80, 0.15, 0.05],   # soleil
        [0.20, 0.60, 0.20],   # nuages
        [0.10, 0.20, 0.70],   # pluie
    ])
    B = np.array([  # B[i, o] = P(observation o | météo i) ; o = vélo, transports, télétravail
        [0.75, 0.20, 0.05],
        [0.20, 0.60, 0.20],
        [0.05, 0.30, 0.65],
    ])
    return pi0, A, B


@app.function
def simuler_hmm(rng, pi0, A, B, T):
    """Tire une suite d'états cachés et d'observations."""
    s = np.empty(T, dtype=int)
    o = np.empty(T, dtype=int)
    s[0] = rng.choice(len(pi0), p=pi0)
    for t in range(T):
        if t > 0:
            s[t] = rng.choice(len(pi0), p=A[s[t - 1]])
        o[t] = rng.choice(B.shape[1], p=B[s[t]])
    return s, o


@app.function
def forward(obs, pi0, A, B):
    """Algorithme forward normalisé. Renvoie α̂ (T, K), facteurs c (T,), log p(obs).

    α̂_t(i) = P(s_t = i | o_1..o_t) ; c_t = P(o_t | o_1..o_{t-1}) ; log p(obs) = Σ log c_t.
    La normalisation à chaque pas évite que les probabilités deviennent nulles en machine.
    """
    T, K = len(obs), len(pi0)
    alpha = np.empty((T, K))
    c = np.empty(T)
    a = pi0 * B[:, obs[0]]
    for t in range(T):
        if t > 0:
            a = (alpha[t - 1] @ A) * B[:, obs[t]]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    return alpha, c, np.log(c).sum()


@app.function
def backward(obs, A, B, c):
    """Algorithme backward, avec les mêmes facteurs c que forward."""
    T, K = len(obs), A.shape[0]
    beta = np.empty((T, K))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (B[:, obs[t + 1]] * beta[t + 1])) / c[t + 1]
    return beta


@app.function
def viterbi(obs, pi0, A, B):
    """Chemin d'états le plus probable (en log). Renvoie (chemin, δ (T, K), ψ (T, K))."""
    T, K = len(obs), len(pi0)
    lA, lB = np.log(A), np.log(B)
    delta = np.empty((T, K))
    psi = np.zeros((T, K), dtype=int)
    delta[0] = np.log(pi0) + lB[:, obs[0]]
    for t in range(1, T):
        scores = delta[t - 1][:, None] + lA  # scores[i, j] : venir de i pour aller en j
        psi[t] = scores.argmax(axis=0)
        delta[t] = scores.max(axis=0) + lB[:, obs[t]]
    chemin = np.empty(T, dtype=int)
    chemin[-1] = delta[-1].argmax()
    for t in range(T - 1, 0, -1):  # remontée des pointeurs
        chemin[t - 1] = psi[t, chemin[t]]
    return chemin, delta, psi


@app.function
def force_brute_hmm(obs, pi0, A, B):
    """Énumère les K^T chemins : probabilité totale et meilleur chemin (référence)."""
    K, T = len(pi0), len(obs)
    chemins = np.array(list(itertools.product(range(K), repeat=T)))
    lp = np.log(pi0[chemins[:, 0]]) + np.log(B[chemins[:, 0], obs[0]])
    for t in range(1, T):
        lp += np.log(A[chemins[:, t - 1], chemins[:, t]]) + np.log(B[chemins[:, t], obs[t]])
    return logsumexp(lp, axis=0), chemins[lp.argmax()], len(chemins)


@app.function
def log_proba_chemin(chemin, obs, pi0, A, B):
    """log P(états = chemin, observations = obs)."""
    lp = np.log(pi0[chemin[0]]) + np.log(B[chemin[0], obs[0]])
    for t in range(1, len(obs)):
        lp += np.log(A[chemin[t - 1], chemin[t]]) + np.log(B[chemin[t], obs[t]])
    return lp


@app.function
def baum_welch(obs, K, M, rng, n_iter=300, tol=1e-9):
    """Baum-Welch (EM pour HMM) depuis une initialisation aléatoire. Renvoie (π0, A, B, logv)."""
    pi0 = rng.dirichlet(np.ones(K))
    A = rng.dirichlet(np.ones(K) * 5, size=K)
    B = rng.dirichlet(np.ones(M) * 5, size=K)
    onehot = np.eye(M)[obs]  # (T, M)
    logv = []
    for _ in range(n_iter):
        # Étape E : probabilités a posteriori des états (γ) et des transitions (ξ)
        alpha, c, ll = forward(obs, pi0, A, B)
        beta = backward(obs, A, B, c)
        logv.append(ll)
        if len(logv) > 1 and logv[-1] - logv[-2] < tol * abs(logv[-1]):
            break
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi = (alpha[:-1, :, None] * A[None] * (B[:, obs[1:]].T * beta[1:])[:, None, :]) / c[1:, None, None]
        # Étape M : comptages « doux » normalisés
        pi0 = gamma[0]
        A = xi.sum(axis=0) / xi.sum(axis=(0, 2))[:, None]
        B = (gamma.T @ onehot) / gamma.sum(axis=0)[:, None]
    return pi0, A, B, np.array(logv)


@app.cell
def _(SEED, config):
    pi0_h, A_h, B_h = hmm_meteo()
    _rng = np.random.default_rng(SEED)
    s_an, o_an = simuler_hmm(_rng, pi0_h, A_h, B_h, config["t_hmm"])

    # 1. Contrôle sur 8 jours : forward et Viterbi contre l'énumération des 3^8 chemins
    _o8 = o_an[:8]
    _ll_fb, _best_fb, _n_chemins = force_brute_hmm(_o8, pi0_h, A_h, B_h)
    _, _, _ll_fw = forward(_o8, pi0_h, A_h, B_h)
    _ch8, _, _ = viterbi(_o8, pi0_h, A_h, B_h)

    # 2. Décodage de l'année
    chemin_vit, _delta_vit, _psi_vit = viterbi(o_an, pi0_h, A_h, B_h)
    _alpha, _c, _ll = forward(o_an, pi0_h, A_h, B_h)
    _beta = backward(o_an, A_h, B_h, _c)
    gamma_an = _alpha * _beta
    gamma_an /= gamma_an.sum(axis=1, keepdims=True)
    # Décodage « naïf » jour par jour : météo la plus probable vu la seule observation du
    # jour, avec la loi stationnaire comme a priori (on ignore la dynamique)
    _w, _v = np.linalg.eig(A_h.T)
    _stat = np.real(_v[:, np.argmin(np.abs(_w - 1))])
    _stat /= _stat.sum()
    _naif = (B_h[:, o_an].T * _stat).argmax(axis=1)
    res_hmm = dict(
        n_chemins=_n_chemins,
        ll_brute=_ll_fb, ll_forward=_ll_fw,
        # Comparaison des PROBABILITÉS des chemins : deux chemins peuvent être ex aequo
        viterbi_ok=bool(abs(log_proba_chemin(_ch8, _o8, pi0_h, A_h, B_h)
                            - log_proba_chemin(_best_fb, _o8, pi0_h, A_h, B_h)) < 1e-9),
        meme_chemin=bool(np.array_equal(_ch8, _best_fb)),
        ll_an=_ll,
        acc_naif=np.mean(_naif == s_an),
        acc_viterbi=np.mean(chemin_vit == s_an),
        acc_marginal=np.mean(gamma_an.argmax(axis=1) == s_an),
        stationnaire=_stat,
    )
    print(f"8 jours : log p(obs) forward {_ll_fw:.6f}, force brute ({_n_chemins} chemins) {_ll_fb:.6f}")
    print(f"8 jours : chemin de Viterbi aussi probable que le meilleur de la force brute : "
          f"{res_hmm['viterbi_ok']} (chemin identique : {res_hmm['meme_chemin']})")
    print(f"{config['t_hmm']} jours : météo retrouvée {res_hmm['acc_naif']:.1%} (jour par jour, sans dynamique), "
          f"{res_hmm['acc_viterbi']:.1%} (Viterbi), {res_hmm['acc_marginal']:.1%} (forward-backward, état par état)")
    return A_h, B_h, chemin_vit, gamma_an, o_an, pi0_h, res_hmm, s_an


@app.cell
def _():
    # Baum-Welch est la partie la plus longue (≈ 15 s) : dans l'éditeur, elle attend ce clic.
    bouton_bw = mo.ui.run_button(label="Lancer Baum-Welch (≈ 15 s)")
    bouton_bw
    return (bouton_bw,)


@app.cell
def _(A_h, B_h, SEED, bouton_bw, config, pi0_h):
    # 3. Baum-Welch : réapprendre A et B à partir des seules observations
    mo.stop(mo.running_in_notebook() and not bouton_bw.value,
            mo.md("Cliquez sur **Lancer Baum-Welch** pour réapprendre le HMM (la suite du bilan en dépend)."))
    _rng = np.random.default_rng(SEED + 1)
    _s, _o = simuler_hmm(_rng, pi0_h, A_h, B_h, config["t_baum_welch"])
    _, _, _ll_vrai = forward(_o, pi0_h, A_h, B_h)
    _t0 = time.perf_counter()
    _runs = [baum_welch(_o, 3, 3, np.random.default_rng(SEED + 10 + i), config["iter_baum_welch"])
             for i in range(config["n_inits_bw"])]
    _t_bw = time.perf_counter() - _t0
    _pi_e, _A_e, _B_e, _logv = max(_runs, key=lambda r: r[3][-1])
    # Les états appris n'ont pas de nom : on cherche la permutation qui colle le mieux à B
    _perm = min(itertools.permutations(range(3)), key=lambda p: np.abs(_B_e[list(p)] - B_h).sum())
    _p = list(_perm)
    A_appris, B_appris = _A_e[np.ix_(_p, _p)], _B_e[_p]
    res_bw = dict(
        logv=_logv, ll_vrai=_ll_vrai, iterations=len(_logv), duree=_t_bw,
        logv_finales=np.array([r[3][-1] for r in _runs]),
        err_A=np.abs(A_appris - A_h).max(), err_B=np.abs(B_appris - B_h).max(),
        monotone=bool(np.all(np.diff(_logv) >= -1e-6)),
    )
    np.set_printoptions(precision=2, suppress=True)
    print(f"Baum-Welch sur {config['t_baum_welch']} jours, {config['n_inits_bw']} initialisations, "
          f"{_t_bw:.1f} s ; meilleure en {res_bw['iterations']} itérations")
    print(f"log p(obs) : modèle appris {_logv[-1]:.1f}, vrai modèle {_ll_vrai:.1f}")
    print(f"A appris :\n{A_appris}\nA vrai :\n{A_h}\nécart max {res_bw['err_A']:.3f}")
    print(f"B appris :\n{B_appris}\nB vrai :\n{B_h}\nécart max {res_bw['err_B']:.3f}")
    return A_appris, B_appris, res_bw


@app.cell
def _(SMOKE, chemin_vit, gamma_an, o_an, res_bw, s_an):
    # --- Figure de la partie 3 : 60 premiers jours, vérité, Viterbi, forward-backward ---
    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(14, 3.8), gridspec_kw=dict(width_ratios=[2.2, 1]))
    _n = min(60, len(s_an))
    _j = np.arange(_n)
    for _k, (_nom, _c) in enumerate(zip(["soleil", "nuages", "pluie"], PALETTE)):
        _a.fill_between(_j, 0, 0, color=_c, alpha=0.4, label=f"P({_nom} | tout)")
    _a.stackplot(_j, gamma_an[:_n].T, colors=PALETTE[:3], alpha=0.35, step="mid")
    _a.plot(_j, (2 - s_an[:_n]) / 2 * 0.8 + 0.1, "k.", ms=6, label="vraie météo")
    _a.plot(_j, (2 - chemin_vit[:_n]) / 2 * 0.8 + 0.1, "-", color=PALETTE[3], lw=1.2, label="Viterbi")
    _a.set(xlabel="jour", yticks=[0.1, 0.5, 0.9], yticklabels=["pluie", "nuages", "soleil"],
           title="HMM : vraie météo, chemin de Viterbi, probabilités a posteriori")
    _a.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    _b.plot(res_bw["logv"], label="modèle appris")
    _b.axhline(res_bw["ll_vrai"], color="k", ls="--", lw=1, label="vrai modèle")
    _b.set(xlabel="itération de Baum-Welch", ylabel="log p(observations)", title="Baum-Welch",
           ylim=(res_bw["logv"][0] - 50, res_bw["ll_vrai"] + 50))
    _b.legend()
    _fig.tight_layout()
    sauvegarder_figure(_fig, "hmm-smoke" if SMOKE else "hmm", mo.notebook_dir() / "figures")
    _fig
    return


@app.cell
def _():
    mo.md(r"""
    ## 4. MCMC : Metropolis-Hastings sur une densité à deux bosses

    On veut des échantillons d'une loi $p(x)$ dont on ne sait calculer que $\tilde p(x)$,
    à une constante près (c'est le cas de toute loi a posteriori : la constante est
    l'évidence $p(\text{données})$, une intégrale insoluble en grande dimension).

    **Metropolis-Hastings** construit une marche aléatoire dont la loi stationnaire est
    $p$ : depuis $x$, proposer $x' = x + \sigma\,\varepsilon$ ; l'accepter avec probabilité
    $\min\big(1, \tilde p(x') / \tilde p(x)\big)$, sinon rester en $x$ (et compter $x$
    une fois de plus). La constante inconnue se simplifie dans le rapport. Avec une
    proposition non symétrique, le rapport est corrigé par $q(x \mid x') / q(x' \mid x)$
    (la correction de Hastings) ; ici la proposition gaussienne est symétrique.

    Cible : mélange de deux gaussiennes, poids 0,3 en $(-2, -1)$ et 0,7 en $(2, 1)$,
    écart-type 0,7. Moyenne exacte : $(0{,}8\,;\,0{,}4)$. Départ volontairement loin, en
    $(-6, 6)$. Trois pas $\sigma$ : 0,1, 1,5 et 15.
    """)
    return


@app.function
def log_cible_bimodale(x):
    """log p̃(x) de la cible à deux bosses (x de forme (..., 2))."""
    m1, m2, s = np.array([-2.0, -1.0]), np.array([2.0, 1.0]), 0.7
    l1 = np.log(0.3) - np.sum((x - m1) ** 2, axis=-1) / (2 * s**2)
    l2 = np.log(0.7) - np.sum((x - m2) ** 2, axis=-1) / (2 * s**2)
    return np.logaddexp(l1, l2)


@app.function
def metropolis_hastings(log_cible, x0, pas, n, rng):
    """Metropolis-Hastings à marche aléatoire gaussienne. Renvoie (chaîne (n, d), taux d'acceptation)."""
    x = np.array(x0, dtype=float)
    lp = log_cible(x)
    chaine = np.empty((n, len(x)))
    acceptes = 0
    bruit = rng.standard_normal((n, len(x))) * pas
    u = np.log(rng.random(n))
    for i in range(n):
        x_prop = x + bruit[i]
        lp_prop = log_cible(x_prop)
        if u[i] < lp_prop - lp:  # accepter avec probabilité min(1, p̃(x')/p̃(x))
            x, lp = x_prop, lp_prop
            acceptes += 1
        chaine[i] = x
    return chaine, acceptes / n


@app.function
def taille_effective(x):
    """Taille d'échantillon effective (ESS) d'une chaîne 1D, estimateur de Geyer.

    n échantillons corrélés valent n / τ échantillons indépendants, avec
    τ = 1 + 2 Σ autocorrélations (somme arrêtée quand les paires deviennent négatives).
    """
    n = len(x)
    x = x - x.mean()
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    tau = -1.0
    for m in range(n // 2 - 1):
        paire = acf[2 * m] + acf[2 * m + 1]
        if paire <= 0:
            break
        tau += 2 * paire
    return n / max(tau, 1.0)


@app.cell
def _(SEED, config):
    _vrai_moy = 0.3 * np.array([-2.0, -1.0]) + 0.7 * np.array([2.0, 1.0])
    _rodage = config["n_mh"] // 10  # on jette les 10 % premiers pas (rodage, burn-in)
    chaines_mh = {}
    res_mh = {}
    for _i, _pas in enumerate((0.1, 1.5, 15.0)):
        _ch, _taux = metropolis_hastings(log_cible_bimodale, (-6.0, 6.0), _pas, config["n_mh"],
                                         np.random.default_rng(SEED + _i))
        chaines_mh[_pas] = _ch
        _x = _ch[_rodage:]
        res_mh[_pas] = dict(
            taux=_taux,
            moyenne=_x.mean(axis=0),
            erreur=float(np.linalg.norm(_x.mean(axis=0) - _vrai_moy)),
            part_droite=float(np.mean(_x[:, 0] > 0)),  # vraie valeur ≈ 0,7
            ess=taille_effective(_x[:, 0]),
        )
        _r = res_mh[_pas]
        print(f"pas {_pas:>4} : acceptation {_r['taux']:.1%}, moyenne ({_r['moyenne'][0]:+.2f} ; "
              f"{_r['moyenne'][1]:+.2f}), erreur {_r['erreur']:.2f}, part dans la bosse de droite "
              f"{_r['part_droite']:.1%} (vrai 70 %), ESS {_r['ess']:.0f} sur {len(_x)} pas")
    return chaines_mh, res_mh


@app.cell
def _(SMOKE, chaines_mh, res_mh):
    # --- Figure de la partie 4 : trois chaînes sur les courbes de niveau de la cible ---
    _g = np.linspace(-7, 7, 200)
    _gx, _gy = np.meshgrid(_g, _g)
    _z = np.exp(log_cible_bimodale(np.stack([_gx, _gy], axis=-1)))
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for _ax, (_pas, _ch) in zip(_axes, chaines_mh.items()):
        _ax.contour(_gx, _gy, _z, levels=6, colors="0.6", linewidths=0.8)
        _ax.plot(_ch[:2000, 0], _ch[:2000, 1], "-", lw=0.6, color=PALETTE[0], alpha=0.8)
        _ax.plot(*_ch[0], "o", color=PALETTE[3])
        _r = res_mh[_pas]
        _ax.set(title=f"pas σ = {_pas} : acceptation {_r['taux']:.0%}, ESS {_r['ess']:.0f}",
                aspect="equal", xlim=(-7, 7), ylim=(-7, 7))
    _fig.tight_layout()
    sauvegarder_figure(_fig, "mcmc-smoke" if SMOKE else "mcmc", mo.notebook_dir() / "figures")
    _fig
    return


@app.cell
def _():
    mo.md(r"""
    ## 5. Inférence variationnelle : remplacer l'intégrale par une optimisation

    Régression logistique **bayésienne** à deux paramètres $w = (w_0, w_1)$ :
    $P(y = 1 \mid x, w) = \sigma(w_0 + w_1 x)$, a priori $w \sim \mathcal{N}(0, 3^2 I)$,
    40 points. On veut la loi a posteriori $p(w \mid \mathcal{D})$, pas un seul $w$.

    - **Exacte** (possible seulement parce qu'il n'y a que 2 paramètres) : on évalue
      $\tilde p(w) = p(\mathcal{D} \mid w)\,p(w)$ sur une grille de $400 \times 400$ points ;
      la somme donne l'**évidence** $\log p(\mathcal{D})$.
    - **MCMC** : Metropolis-Hastings, comme en partie 4.
    - **Variationnelle** : on choisit une famille simple, ici des gaussiennes à
      covariance **diagonale** (« champ moyen ») $q(w) = \mathcal{N}(\mu, \operatorname{diag}(\sigma^2))$,
      et on maximise
      $\text{ELBO}(\mu, \sigma) = \mathbb{E}_q[\log \tilde p(w)] + H(q) = \log p(\mathcal{D}) - \mathrm{KL}(q \,\|\, p(w \mid \mathcal{D}))$.
      L'espérance est estimée avec 16 tirages $w = \mu + \sigma \odot \varepsilon$ par pas
      (astuce de reparamétrisation), et on monte le gradient avec Adam.

    Maximiser l'ELBO, c'est minimiser la KL vers la vraie loi a posteriori sans jamais
    calculer l'évidence. L'ELBO reste **sous** $\log p(\mathcal{D})$ ; l'écart est la KL.
    """)
    return


@app.function
def donnees_logistique(rng, n=40):
    """x ~ N(1,5 ; 1), y ~ Bernoulli(σ(−1 + 1,5 x)) : x non centré, d'où des paramètres corrélés."""
    x = rng.normal(1.5, 1.0, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1.0 + 1.5 * x)))).astype(float)
    return x, y


@app.function
def log_post_logistique(W, x, y, s_prior=3.0):
    """log p̃(w) = log p(D | w) + log p(w), normalisation de l'a priori comprise. W : (..., 2)."""
    z = W[..., :1] + W[..., 1:2] * x  # (..., n)
    log_vrais = np.sum(y * z - np.logaddexp(0.0, z), axis=-1)
    log_prior = -np.sum(W**2, axis=-1) / (2 * s_prior**2) - 2 * np.log(s_prior) - np.log(2 * np.pi)
    return log_vrais + log_prior


@app.function
def grad_log_post_logistique(W, x, y, s_prior=3.0):
    """Gradient de log p̃(w) par rapport à w, pour un lot W (S, 2)."""
    z = W[:, :1] + W[:, 1:2] * x  # (S, n)
    resid = y - 1 / (1 + np.exp(-z))
    return np.stack([resid.sum(axis=1), (resid * x).sum(axis=1)], axis=1) - W / s_prior**2


@app.function
def posterieure_grille(x, y, n=400, bornes=((-10.0, 4.0), (-1.5, 9.0))):
    """Loi a posteriori « exacte » sur une grille : évidence, moyenne, covariance, densité."""
    g0 = np.linspace(*bornes[0], n)
    g1 = np.linspace(*bornes[1], n)
    W = np.stack(np.meshgrid(g0, g1, indexing="ij"), axis=-1)  # (n, n, 2)
    lp = log_post_logistique(W, x, y)
    aire = (g0[1] - g0[0]) * (g1[1] - g1[0])
    log_evidence = logsumexp(lp.ravel(), axis=0) + np.log(aire)
    p = np.exp(lp - lp.max())
    p /= p.sum()
    moy = np.einsum("ijk,ij->k", W, p)
    c = W - moy
    cov = np.einsum("ijk,ijl,ij->kl", c, c, p)
    bord = p[0].sum() + p[-1].sum() + p[:, 0].sum() + p[:, -1].sum()  # masse sur le bord
    return dict(g0=g0, g1=g1, densite=p, log_evidence=log_evidence, moyenne=moy, cov=cov, bord=bord)


@app.function
def vi_champ_moyen(x, y, rng, n_pas=3000, S=16, lr=0.05):
    """Inférence variationnelle champ moyen par gradient reparamétré + Adam (numpy).

    Paramètres : μ (2,) et ρ = log σ (2,). Renvoie μ, σ et l'ELBO estimée à chaque pas.
    """
    mu, rho = np.zeros(2), np.zeros(2)
    m, v = np.zeros(4), np.zeros(4)
    b1, b2, eps_adam = 0.9, 0.999, 1e-8
    historique = []
    for t in range(1, n_pas + 1):
        sigma = np.exp(rho)
        eps = rng.standard_normal((S, 2))
        W = mu + sigma * eps                          # astuce de reparamétrisation
        g = grad_log_post_logistique(W, x, y)         # ∇_w log p̃(w), (S, 2)
        g_mu = g.mean(axis=0)
        g_rho = (g * eps * sigma).mean(axis=0) + 1.0  # +1 : dérivée de l'entropie Σ ρ
        elbo = log_post_logistique(W, x, y).mean() + rho.sum() + 1.0 + np.log(2 * np.pi)
        historique.append(elbo)
        # Adam en MONTÉE de gradient (on maximise l'ELBO)
        grad = np.concatenate([g_mu, g_rho])
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad**2
        pas = lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps_adam)
        mu, rho = mu + pas[:2], rho + pas[2:]
    return mu, np.exp(rho), np.array(historique)


@app.function
def elbo_estimee(mu, sigma, x, y, rng, S=200_000):
    """ELBO de q = N(μ, diag σ²) estimée par Monte-Carlo, avec son erreur type."""
    W = mu + sigma * rng.standard_normal((S, 2))
    termes = log_post_logistique(W, x, y) + np.log(sigma).sum() + 1.0 + np.log(2 * np.pi)
    return termes.mean(), termes.std() / np.sqrt(S)


@app.cell
def _(SEED, config):
    _rng = np.random.default_rng(SEED)
    x_log, y_log = donnees_logistique(_rng)
    post = posterieure_grille(x_log, y_log, n=config["grille"])
    _sd = np.sqrt(np.diag(post["cov"]))
    _corr = post["cov"][0, 1] / (_sd[0] * _sd[1])

    _t0 = time.perf_counter()
    _rodage = config["n_mh_post"] // 10
    _ch, _taux = metropolis_hastings(lambda w: log_post_logistique(w, x_log, y_log), (0.0, 0.0), 0.8,
                                     config["n_mh_post"], np.random.default_rng(SEED + 5))
    ech_mh = _ch[_rodage:]
    _t_mh = time.perf_counter() - _t0

    _t0 = time.perf_counter()
    mu_vi, sigma_vi, hist_vi = vi_champ_moyen(x_log, y_log, np.random.default_rng(SEED + 6),
                                              n_pas=config["n_pas_vi"])
    _t_vi = time.perf_counter() - _t0
    _elbo, _se = elbo_estimee(mu_vi, sigma_vi, x_log, y_log, np.random.default_rng(SEED + 7))

    res_vi = dict(
        log_evidence=post["log_evidence"], bord=post["bord"], corr=_corr, sd_exact=_sd,
        moy_exact=post["moyenne"],
        moy_mh=ech_mh.mean(axis=0), sd_mh=ech_mh.std(axis=0), taux_mh=_taux,
        ess_mh=min(taille_effective(ech_mh[:, 0]), taille_effective(ech_mh[:, 1])),
        moy_vi=mu_vi, sd_vi=sigma_vi, elbo=_elbo, elbo_se=_se, kl=post["log_evidence"] - _elbo,
        t_mh=_t_mh, t_vi=_t_vi,
    )
    res_vi["err_mh"] = float(np.max(np.abs(res_vi["moy_mh"] - post["moyenne"]) / _sd))
    res_vi["err_vi"] = float(np.max(np.abs(mu_vi - post["moyenne"]) / _sd))
    _f = lambda a: f"({a[0]:+.3f} ; {a[1]:+.3f})"  # noqa: E731
    print(f"Grille {config['grille']}² : log p(D) = {post['log_evidence']:.4f}, masse au bord "
          f"{post['bord']:.1e}, corrélation(w0, w1) a posteriori = {_corr:+.2f}")
    print(f"Moyenne a posteriori  exacte {_f(post['moyenne'])}  MH {_f(res_vi['moy_mh'])}  VI {_f(mu_vi)}")
    print(f"Écart-types           exacts {_f(_sd)}  MH {_f(res_vi['sd_mh'])}  VI {_f(sigma_vi)}")
    print(f"MH : acceptation {_taux:.1%}, ESS {res_vi['ess_mh']:.0f}, {_t_mh:.1f} s ; VI : {_t_vi:.1f} s")
    print(f"ELBO = {_elbo:.4f} ± {_se:.4f}  ≤  log p(D) = {post['log_evidence']:.4f} ; "
          f"écart = KL(q ‖ p) = {res_vi['kl']:.3f} nat")
    return ech_mh, hist_vi, mu_vi, post, res_vi, sigma_vi


@app.cell
def _(SMOKE, ech_mh, hist_vi, mu_vi, post, res_vi, sigma_vi):
    # --- Figure de la partie 5 : exacte, MCMC, VI ; courbe d'ELBO ---
    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(12, 4.6))
    _G0, _G1 = np.meshgrid(post["g0"], post["g1"], indexing="ij")
    _a.contour(_G0, _G1, post["densite"], levels=6, colors="0.4", linewidths=1)
    _a.plot(ech_mh[::20, 0], ech_mh[::20, 1], ".", ms=2, color=PALETTE[0], alpha=0.4)
    dessiner_ellipses(_a, mu_vi[None], np.diag(sigma_vi**2)[None], [PALETTE[3]])
    _a.set(xlabel="$w_0$ (biais)", ylabel="$w_1$ (pente)", xlim=(-8, 1), ylim=(0, 8),
           title="Gris : exacte ; points : MCMC ; rouge : VI champ moyen")
    _lisse = np.convolve(hist_vi, np.ones(50) / 50, mode="valid")
    _b.plot(np.arange(len(_lisse)) + 50, _lisse, label="ELBO (moyenne glissante)")
    _b.axhline(res_vi["log_evidence"], color="k", ls="--", lw=1, label="log p(D) (grille)")
    _b.set(xlabel="pas d'Adam", ylim=(res_vi["log_evidence"] - 6, res_vi["log_evidence"] + 1),
           title=f"L'ELBO monte et plafonne sous log p(D) (écart {res_vi['kl']:.2f})")
    _b.legend()
    _fig.tight_layout()
    sauvegarder_figure(_fig, "vi-smoke" if SMOKE else "vi", mo.notebook_dir() / "figures")
    _fig
    return


@app.cell
def _(T0, res_bw, res_hmm, res_melange, res_mh, res_vi, reseau, reseau_indep, rejet):
    # --- Bilan : seuils du README. En smoke, on vérifie seulement que tout tourne.
    verifs = {
        "Réseau de la figure : P(pluie | herbe, arroseur) < P(pluie) < P(pluie | herbe), et P(pluie | arroseur) < 0,01 (flèche pluie → arroseur)":
            reseau["P(pluie | herbe, arroseur)"] < reseau["P(pluie)"] < reseau["P(pluie | herbe)"]
            and reseau["P(pluie | arroseur)"] < 0.01,
        "Variante indépendante : effet d'explication, P(pluie | arroseur) = P(pluie) < P(pluie | herbe, arroseur) < P(pluie | herbe)":
            abs(reseau_indep["P(pluie | arroseur)"] - reseau_indep["P(pluie)"]) < 1e-12
            and reseau_indep["P(pluie)"] < reseau_indep["P(pluie | herbe, arroseur)"] < reseau_indep["P(pluie | herbe)"],
        "Réseau : rejet à ±0,01 de l'exact":
            abs(rejet["estimation"] - reseau["P(pluie | herbe)"]) < 0.01,
        "EM : log-vraisemblance jamais en baisse": res_melange["monotone"],
        "EM : ELBO ≤ log-vraisemblance, et égalité logv = ELBO + KL (à 1e-6)":
            res_melange["elbo_sous_logv"] and res_melange["ecart_identite"] < 1e-6,
        "EM : même log-vraisemblance moyenne que scikit-learn (à 1e-3)":
            abs(res_melange["logv_moy_em"] - res_melange["logv_moy_sklearn"]) < 1e-3,
        "EM : ARI ≥ 0,95": res_melange["ari_em"] >= 0.95,
        "k-means : ARI ≤ 0,8 (bandes allongées mal séparées)": res_melange["ari_kmeans"] <= 0.8,
        "HMM : forward = force brute (à 1e-9)": abs(res_hmm["ll_forward"] - res_hmm["ll_brute"]) < 1e-9,
        "HMM : chemin de Viterbi = probabilité max de la force brute": res_hmm["viterbi_ok"],
        "HMM : Viterbi bat le décodage jour par jour d'au moins 5 points":
            res_hmm["acc_viterbi"] >= res_hmm["acc_naif"] + 0.05,
        "Baum-Welch : log-vraisemblance jamais en baisse": res_bw["monotone"],
        "Baum-Welch : A et B retrouvés à ±0,1": res_bw["err_A"] <= 0.1 and res_bw["err_B"] <= 0.1,
        "MH : pas 1,5, moyenne à moins de 0,15 de la vraie": res_mh[1.5]["erreur"] < 0.15,
        "MH : pas 1,5, acceptation entre 20 % et 60 %": 0.2 <= res_mh[1.5]["taux"] <= 0.6,
        "MH : pas 0,1, acceptation > 90 % mais ESS ≤ 1/10 de celui du pas 1,5":
            res_mh[0.1]["taux"] > 0.9 and res_mh[0.1]["ess"] <= res_mh[1.5]["ess"] / 10,
        "MH : pas 15, acceptation < 5 %": res_mh[15.0]["taux"] < 0.05,
        "Grille : masse au bord < 1e-4": res_vi["bord"] < 1e-4,
        "MH a posteriori : moyenne à moins de 0,1 écart-type de l'exacte": res_vi["err_mh"] < 0.1,
        "VI : ELBO < log p(D) (au-delà de 3 erreurs types)": res_vi["kl"] > 3 * res_vi["elbo_se"],
        "VI : moyenne à moins de 0,25 écart-type de l'exacte": res_vi["err_vi"] < 0.25,
        "VI champ moyen : écarts-types sous-estimés (< 90 % des exacts)":
            bool(np.all(res_vi["sd_vi"] < 0.9 * res_vi["sd_exact"])),
    }
    for _k, _ok in verifs.items():
        print(("OK    " if _ok else "ÉCHEC ") + _k)
    duree_totale = time.perf_counter() - T0
    print(f"Durée totale des calculs : {duree_totale:.1f} s")
    return duree_totale, verifs


@app.cell
def _(SMOKE, duree_totale, verifs):
    _echecs = [k for k, ok in verifs.items() if not ok]
    if not SMOKE and _echecs:
        raise SystemExit("Seuils non atteints : " + " ; ".join(_echecs))
    mo.md(f"**Bilan** : {sum(verifs.values())} vérifications sur {len(verifs)} passent "
          f"({duree_totale:.1f} s de calcul).")
    return


if __name__ == "__main__":
    app.run()
