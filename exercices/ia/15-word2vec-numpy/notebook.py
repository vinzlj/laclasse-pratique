# Exercice 15 : plongements de mots avant les Transformers.
#
#   make ex NUM=15              → éditeur marimo (interactif, bouton « Entraîner »)
#   make ex NUM=15 MODE=run     → script complet (vérifie les seuils du README)
#   make ex NUM=15 MODE=smoke   → test rapide, < 30 s
#
# Progression, sur TinyShakespeare (≈ 200 000 mots) : one-hot, sac de mots et TF-IDF en
# numpy (vérifiés contre scikit-learn), LSA par décomposition spectrale (vérifiée contre
# TruncatedSVD), puis skip-gram avec échantillonnage négatif (word2vec) écrit à la main en
# numpy, gradient vérifié par l'autograd de PyTorch. Évaluation commune : voisins,
# test de groupes, analogies, projections 2D (ACP, t-SNE) et ce qu'elles déforment.
# Tout tourne sur CPU (numpy) ; seul téléchargement : TinyShakespeare (1,1 Mo, data/).

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import collections
    import json
    import re
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from ia_commun import PALETTE, is_smoke, sauvegarder_figure, seed_everything, style_commun
    from ia_commun.donnees import charger_tinyshakespeare

    return (
        PALETTE,
        charger_tinyshakespeare,
        collections,
        is_smoke,
        json,
        mo,
        np,
        plt,
        re,
        sauvegarder_figure,
        seed_everything,
        style_commun,
        time,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Exercice 15 : donner un vecteur à chaque mot

    **Objectif.** Construire, sur le même petit corpus, quatre façons de représenter un
    mot par un vecteur, de la plus naïve à word2vec, et mesurer ce que chacune capture.

    1. **One-hot** : un mot = une case à 1 dans un vecteur de la taille du vocabulaire.
       Tous les mots sont à la même distance les uns des autres.
    2. **Sac de mots et TF-IDF** : un mot (ou un document) = ses comptes de cooccurrence,
       pondérés par la rareté du mot. Moteur de recherche minimal.
    3. **LSA** (analyse sémantique latente) : on compresse la matrice mots × documents
       en 100 dimensions par décomposition en valeurs singulières.
    4. **Skip-gram avec échantillonnage négatif** (word2vec) : un petit réseau apprend à
       distinguer les vrais voisins d'un mot de mots tirés au hasard.

    **Ce qui est optimisé.** TF-IDF : rien (on compte). LSA : l'erreur de
    reconstruction quadratique de la matrice TF-IDF par une matrice de rang $k$ (meilleure
    approximation au sens de Frobenius, théorème d'Eckart-Young). Skip-gram : pour chaque
    paire (mot $w$, voisin $c$) et $K$ mots négatifs $n_1, \dots, n_K$ tirés au hasard,

    $$
    \mathcal{L} = -\log \sigma(\mathbf{v}_w \cdot \mathbf{u}_c)
                  - \sum_{k=1}^{K} \log \sigma(-\mathbf{v}_w \cdot \mathbf{u}_{n_k})
    $$

    une entropie croisée binaire : « vrai voisin » contre « bruit ».
    """)


@app.cell
def _(is_smoke, seed_everything, style_commun, time):
    T0 = time.perf_counter()  # chronomètre de l'ensemble des calculs
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        n_discours=1500 if SMOKE else None,  # None = les 7 222 répliques du corpus
        min_compte=3 if SMOKE else 5,         # mots plus rares : hors vocabulaire
        k_lsa=20 if SMOKE else 100,           # dimensions de la LSA
        dim=50,                               # dimension des plongements word2vec
        fenetre=4,                            # contexte : jusqu'à 4 mots de chaque côté
        k_negatifs=5,                         # mots négatifs par paire positive
        epoques=1 if SMOKE else 10,
        lr=0.025,                             # pas initial, décroît linéairement vers 0
        taille_lot=256,
        seuil_sous_ech=1e-3,                  # sous-échantillonnage des mots fréquents
        n_export=300 if SMOKE else 1000,      # mots exportés pour la démo et les projections
        tsne_iter=250 if SMOKE else 1000,
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SEED, SMOKE, T0, config


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Le corpus et un découpage naïf

    TinyShakespeare est une suite de répliques : un nom de personnage, deux points, puis
    le texte. On garde chaque réplique comme un **document** (pour TF-IDF et la LSA) et
    on la découpe en mots par une expression régulière : minuscules, lettres, une
    apostrophe interne autorisée (« i'll », « o'er »). C'est volontairement naïf : la
    ponctuation disparaît, « king's » et « king » sont deux mots différents, et tout mot
    vu moins de 5 fois est jeté. La tokenisation moderne (chapitre 19) traite ces cas.
    """)


@app.cell
def _(charger_tinyshakespeare, collections, config, np, re):
    _texte = charger_tinyshakespeare()
    _blocs = [b for b in _texte.split("\n\n") if b.strip()]
    if config["n_discours"]:
        _blocs = _blocs[: config["n_discours"]]
    # Première ligne d'un bloc = « NOM DU PERSONNAGE: » ; le reste = la réplique
    locuteurs = [b.split("\n")[0].rstrip(":") for b in _blocs]
    repliques = [" ".join(b.split("\n")[1:]) for b in _blocs]
    MOTIF = r"[a-z]+(?:'[a-z]+)?"
    docs_mots = [re.findall(MOTIF, r.lower()) for r in repliques]
    _tous = [m for d in docs_mots for m in d]
    comptes_bruts = collections.Counter(_tous)
    # Vocabulaire trié par fréquence décroissante : l'indice 0 est le mot le plus fréquent
    vocab = [m for m, n in comptes_bruts.most_common() if n >= config["min_compte"]]
    idx = {m: i for i, m in enumerate(vocab)}
    comptes = np.array([comptes_bruts[m] for m in vocab], dtype=np.float64)
    corpus = np.array([idx[m] for m in _tous if m in idx], dtype=np.int64)
    taux_hors_vocab = 1 - len(corpus) / len(_tous)
    print(f"{len(repliques)} répliques, {len(_tous)} mots, {len(comptes_bruts)} mots distincts")
    print(f"vocabulaire (≥ {config['min_compte']} occurrences) : {len(vocab)} mots ; "
          f"{taux_hors_vocab:.1%} des occurrences sont hors vocabulaire")
    print("les 15 plus fréquents :", vocab[:15])
    return (
        MOTIF,
        comptes,
        comptes_bruts,
        corpus,
        docs_mots,
        idx,
        locuteurs,
        repliques,
        taux_hors_vocab,
        vocab,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. One-hot : tous les mots à égale distance

    Le mot d'indice $i$ devient le vecteur $e_i$ (un 1 en position $i$, des 0 ailleurs).
    Le cosinus entre deux mots différents vaut toujours 0 : « king » est aussi loin de
    « queen » que de « three ». Rien n'est appris, rien n'est partagé.
    """)


@app.cell
def _(idx, np, vocab):
    def cosinus(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    _V = len(vocab)
    _onehot = lambda m: np.eye(1, _V, idx[m]).ravel()  # noqa: E731
    cos_onehot = {
        (a, b): cosinus(_onehot(a), _onehot(b))
        for a, b in [("king", "queen"), ("king", "three"), ("two", "three")]
        if a in idx and b in idx
    }
    print(f"dimension : {_V} ; cosinus one-hot : {cos_onehot}")
    return cos_onehot, cosinus


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Sac de mots et TF-IDF (numpy, puis scikit-learn)

    Matrice $X$ de taille (mots × répliques) : $X_{t,d}$ = nombre d'occurrences du mot $t$
    dans la réplique $d$, multiplié par l'**IDF** (*inverse document frequency*,
    Sparck Jones, 1972) :

    $$\mathrm{idf}(t) = \ln \frac{N}{\mathrm{df}(t)}$$

    où $N$ est le nombre de répliques et $\mathrm{df}(t)$ le nombre de répliques qui
    contiennent $t$. Un mot présent dans toutes les répliques pèserait 0 ; « the », présent
    dans 39 % des répliques, pèse six fois moins que « poison », qui est rare. Une **colonne** de $X$ représente une réplique (moteur de recherche) ; une
    **ligne** représente un mot par les répliques où il apparaît : c'est déjà un
    plongement, très creux (une dimension par réplique).

    scikit-learn ajoute 1 à l'IDF (`smooth_idf=False` donne $\ln(N/\mathrm{df}) + 1$) pour
    ne pas annuler les mots présents partout ; on vérifie qu'on retrouve sa matrice à
    cette convention près.
    """)


@app.cell
def _(MOTIF, docs_mots, idx, np, repliques, vocab):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    _V, _N = len(vocab), len(docs_mots)
    tf = np.zeros((_V, _N), dtype=np.float32)  # comptes bruts (sac de mots)
    for _j, _d in enumerate(docs_mots):
        for _m in _d:
            _i = idx.get(_m)
            if _i is not None:
                tf[_i, _j] += 1
    df = (tf > 0).sum(axis=1)
    idf = np.log(_N / df)
    tfidf = (tf * idf[:, None]).astype(np.float32)

    # Vérification : scikit-learn, même vocabulaire, même découpage, sans normalisation
    _sk = TfidfVectorizer(vocabulary=idx, token_pattern=MOTIF, smooth_idf=False, norm=None)
    _Xsk = _sk.fit_transform(repliques).T.toarray()  # (mots × répliques)
    ecart_tfidf = float(np.abs(tf * (idf[:, None] + 1) - _Xsk).max())
    print(f"écart max avec TfidfVectorizer (IDF + 1) : {ecart_tfidf:.2e}")

    # L'ordre des mots est perdu : deux phrases opposées ont le même sac de mots
    _cv = CountVectorizer(token_pattern=MOTIF)
    _sacs = _cv.fit_transform(["the dog bit the man", "the man bit the dog"]).toarray()
    sacs_identiques = bool((_sacs[0] == _sacs[1]).all())
    print(f"« the dog bit the man » et « the man bit the dog » : même sac de mots ? {sacs_identiques}")
    return df, ecart_tfidf, idf, sacs_identiques, tf, tfidf


@app.cell
def _(MOTIF, idf, idx, locuteurs, np, re, repliques, tfidf):
    # Moteur de recherche minimal : cosinus entre la requête et chaque réplique
    _docs = tfidf / np.maximum(np.linalg.norm(tfidf, axis=0, keepdims=True), 1e-12)

    def chercher(requete, k=3):
        q = np.zeros(len(idf), dtype=np.float32)
        for m in re.findall(MOTIF, requete.lower()):
            if m in idx:
                q[idx[m]] += idf[idx[m]]
        q /= max(np.linalg.norm(q), 1e-12)
        scores = q @ _docs
        meilleurs = np.argsort(-scores)[:k]
        return [(float(scores[j]), locuteurs[j], repliques[j][:90]) for j in meilleurs]

    recherches = {r: chercher(r) for r in ["crown king england", "poison sleep death", "love daughter"]}
    for _r, _res in recherches.items():
        print(f"\n« {_r} »")
        for _s, _loc, _txt in _res:
            print(f"  {_s:.3f}  {_loc} : {_txt}…")
    return (recherches,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. LSA : compresser la matrice mots × répliques

    Deerwester et al. (1990) approchent $X$ par $U_k \Sigma_k V_k^\top$ (décomposition en
    valeurs singulières tronquée à $k$ termes) et prennent les lignes de $U_k\Sigma_k$
    comme vecteurs de mots. Calcul « à la main » : les vecteurs singuliers à gauche de $X$
    sont les vecteurs propres de $XX^\top$ (mots × mots, cooccurrences pondérées), et
    $\sigma_i = \sqrt{\lambda_i}$. On compare ensuite à `TruncatedSVD` de scikit-learn.
    """)


@app.cell
def _(config, np, tfidf, time):
    from sklearn.decomposition import TruncatedSVD

    _t = time.perf_counter()
    _G = tfidf.astype(np.float64) @ tfidf.T.astype(np.float64)  # (mots × mots)
    _vp, _vecp = np.linalg.eigh(_G)  # valeurs propres croissantes
    _ordre = np.argsort(_vp)[::-1][: config["k_lsa"]]
    sigma_lsa = np.sqrt(np.maximum(_vp[_ordre], 0))
    lsa = (_vecp[:, _ordre] * sigma_lsa).astype(np.float32)  # U_k Σ_k : un vecteur par mot
    duree_lsa = time.perf_counter() - _t
    # Part de la « variance » (énergie de Frobenius) gardée par les k dimensions
    part_energie_lsa = float((sigma_lsa**2).sum() / _vp.clip(min=0).sum())

    _svd = TruncatedSVD(n_components=config["k_lsa"], algorithm="arpack", random_state=42)
    _svd.fit(tfidf.T)  # scikit-learn attend (échantillons × variables) = (répliques × mots)
    ecart_sigma = float(np.abs(np.sort(_svd.singular_values_)[::-1] - sigma_lsa).max() / sigma_lsa[0])
    print(f"LSA {config['k_lsa']} dimensions en {duree_lsa:.1f} s ; énergie gardée {part_energie_lsa:.1%} ; "
          f"écart relatif des valeurs singulières avec TruncatedSVD : {ecart_sigma:.1e}")
    return ecart_sigma, lsa, part_energie_lsa, sigma_lsa


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Skip-gram avec échantillonnage négatif, en numpy

    Mikolov et al. (2013). Chaque mot a **deux** vecteurs : $\mathbf{v}_w$ quand il est au
    centre, $\mathbf{u}_c$ quand il est dans le contexte. Pour chaque mot du corpus, on
    forme les paires (centre, voisin) à distance 1 à $b$ (avec $b$ tiré entre 1 et 4 :
    les voisins proches comptent plus souvent). Deux astuces de l'article :

    - **sous-échantillonnage** : un mot de fréquence relative $f$ est gardé avec
      probabilité $\min\big(1, \sqrt{t/f} + t/f\big)$, $t = 10^{-3}$ (formule du code
      C de word2vec) : « the » perd la plupart de ses occurrences ;
    - **négatifs** tirés selon la fréquence puissance 3/4 : les mots rares sont un peu
      plus souvent tirés qu'avec la fréquence brute.

    Gradient à la main ($s_+ = \sigma(\mathbf{v}_w\cdot\mathbf{u}_c)$,
    $s_k = \sigma(\mathbf{v}_w\cdot\mathbf{u}_{n_k})$) :

    $$
    \frac{\partial\mathcal{L}}{\partial \mathbf{v}_w} = (s_+ - 1)\,\mathbf{u}_c + \sum_k s_k\,\mathbf{u}_{n_k},
    \qquad
    \frac{\partial\mathcal{L}}{\partial \mathbf{u}_c} = (s_+ - 1)\,\mathbf{v}_w,
    \qquad
    \frac{\partial\mathcal{L}}{\partial \mathbf{u}_{n_k}} = s_k\,\mathbf{v}_w .
    $$

    Au départ, $\mathbf{u} = 0$ : tous les produits scalaires valent 0, $\sigma = 1/2$,
    et la perte vaut $(1 + K)\ln 2 = 6 \ln 2 \approx 4{,}16$.
    """)


@app.cell
def _(comptes, config, corpus, np):
    _f = comptes / comptes.sum()
    p_garder = np.minimum(1.0, np.sqrt(config["seuil_sous_ech"] / _f) + config["seuil_sous_ech"] / _f)
    _pneg = comptes**0.75
    cumul_negatifs = np.cumsum(_pneg / _pneg.sum())  # tirage par recherche dichotomique

    def sigmoide(x):
        return 1.0 / (1.0 + np.exp(-x))

    def paires_epoque(rng):
        """Sous-échantillonne le corpus puis renvoie les paires (centre, voisin) mélangées."""
        garde = corpus[rng.random(len(corpus)) < p_garder[corpus]]
        n = len(garde)
        b = rng.integers(1, config["fenetre"] + 1, n)  # fenêtre effective de chaque mot
        centres, voisins = [], []
        for d in range(1, config["fenetre"] + 1):
            i = np.nonzero(b[: n - d] >= d)[0]  # voisin à droite, distance d
            centres.append(garde[i]); voisins.append(garde[i + d])
            i = np.nonzero(b[d:] >= d)[0] + d   # voisin à gauche, distance d
            centres.append(garde[i]); voisins.append(garde[i - d])
        centres, voisins = np.concatenate(centres), np.concatenate(voisins)
        ordre = rng.permutation(len(centres))
        return centres[ordre], voisins[ordre], n

    def perte_et_gradients(W, C, w, o, neg):
        """Perte moyenne du lot et gradients par ligne (pour v_w, u_c et les u_négatifs)."""
        vw, uc, un = W[w], C[o], C[neg]                       # (B, D), (B, D), (B, K, D)
        sp = sigmoide((vw * uc).sum(1))                       # (B,)
        sn = sigmoide(np.einsum("bd,bkd->bk", vw, un))        # (B, K)
        perte = -(np.log(sp + 1e-10).sum() + np.log(1 - sn + 1e-10).sum()) / len(w)
        g_sp = (sp - 1)[:, None]
        g_vw = g_sp * uc + (sn[:, :, None] * un).sum(1)
        g_uc = g_sp * vw
        g_un = sn[:, :, None] * vw[:, None, :]
        return perte, g_vw, g_uc, g_un

    return cumul_negatifs, paires_epoque, perte_et_gradients


@app.cell
def _(SEED, comptes, config, cumul_negatifs, np, perte_et_gradients):
    # Vérification du gradient écrit à la main par l'autograd de PyTorch (CPU, float64)
    import torch

    _rng = np.random.default_rng(SEED)
    _V, _D, _B = len(comptes), config["dim"], 8
    _W = _rng.normal(0, 0.3, (_V, _D)); _C = _rng.normal(0, 0.3, (_V, _D))
    _w = _rng.integers(0, _V, _B); _o = _rng.integers(0, _V, _B)
    _neg = np.searchsorted(cumul_negatifs, _rng.random((_B, config["k_negatifs"])))
    _perte, _gvw, _guc, _gun = perte_et_gradients(_W, _C, _w, _o, _neg)

    _tvw = torch.tensor(_W[_w], requires_grad=True)
    _tuc = torch.tensor(_C[_o], requires_grad=True)
    _tun = torch.tensor(_C[_neg], requires_grad=True)
    _tperte = -(torch.nn.functional.logsigmoid((_tvw * _tuc).sum(1)).sum()
                + torch.nn.functional.logsigmoid(-torch.einsum("bd,bkd->bk", _tvw, _tun)).sum()) / _B
    _tperte.backward()
    # Notre perte est une moyenne sur le lot, nos gradients sont par exemple : facteur B
    ecart_gradient = float(max(np.abs(_gvw / _B - _tvw.grad.numpy()).max(),
                               np.abs(_guc / _B - _tuc.grad.numpy()).max(),
                               np.abs(_gun / _B - _tun.grad.numpy()).max()))
    print(f"perte numpy {_perte:.6f} / torch {_tperte.item():.6f} ; écart max des gradients {ecart_gradient:.1e}")
    return (ecart_gradient,)


@app.cell
def _(mo):
    bouton = mo.ui.run_button(label="Entraîner")
    bouton
    return (bouton,)


@app.cell
def _(
    SEED,
    bouton,
    comptes,
    config,
    cumul_negatifs,
    mo,
    np,
    paires_epoque,
    perte_et_gradients,
    time,
):
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Entraîner » pour lancer word2vec.*"))

    _rng = np.random.default_rng(SEED)
    _V, _D, _K, _B = len(comptes), config["dim"], config["k_negatifs"], config["taille_lot"]
    # Initialisation du code C de word2vec : v uniforme dans [-0,5/D, 0,5/D], u = 0
    W = (_rng.random((_V, _D)) - 0.5) / _D
    C = np.zeros((_V, _D))
    pertes_epoques, pertes_lots = [], []
    _t = time.perf_counter()
    _total = None
    for _ep in range(config["epoques"]):
        _centres, _voisins, _n = paires_epoque(_rng)
        _P = len(_centres)
        # approximation : nombre total de paires estimé d'après la 1re époque (le
        # sous-échantillonnage fait varier _P d'une époque à l'autre, le pas finit donc
        # légèrement avant ou après 0 ; sans effet visible sur les résultats)
        _total = _total or config["epoques"] * _P  # nombre total de paires prévues
        _pertes = []
        for _s in range(0, _P, _B):
            # pas qui décroît linéairement jusqu'à 0,01 % du pas initial
            _lr = config["lr"] * max(1e-4, 1 - (_ep * _P + _s) / _total)
            _w, _o = _centres[_s:_s + _B], _voisins[_s:_s + _B]
            _neg = np.searchsorted(cumul_negatifs, _rng.random((len(_w), _K)))
            _perte, _gvw, _guc, _gun = perte_et_gradients(W, C, _w, _o, _neg)
            # np.add.at additionne correctement quand un même mot apparaît plusieurs fois
            np.add.at(C, _o, -_lr * _guc)
            np.add.at(C, _neg.ravel(), -_lr * _gun.reshape(-1, _D))
            np.add.at(W, _w, -_lr * _gvw)
            _pertes.append(_perte)
        pertes_lots.extend(_pertes[:: max(1, len(_pertes) // 50)])
        pertes_epoques.append(float(np.mean(_pertes)))
        print(f"époque {_ep + 1:2d} : {_P} paires ({_n} mots gardés), perte moyenne {pertes_epoques[-1]:.3f}, "
              f"{time.perf_counter() - _t:.1f} s")
    duree_w2v = time.perf_counter() - _t
    perte_initiale = float(pertes_lots[0])
    return C, W, duree_w2v, perte_initiale, pertes_epoques, pertes_lots


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Comparer les représentations

    Trois tests, les mêmes pour TF-IDF (lignes de la matrice), LSA et word2vec
    (vecteurs $\mathbf{v}_w$), tous au cosinus :

    - **voisins** de quelques mots ;
    - **test de groupes** : 50 mots répartis en 5 groupes (famille, nombres, corps,
      titres, temps) ; on compte la part des mots dont le plus proche voisin, *parmi ces
      50*, est du même groupe (hasard : environ 20 %) ;
    - **analogies** « a est à b ce que c est à ? » : on cherche le mot le plus proche de
      $\mathbf{b} - \mathbf{a} + \mathbf{c}$ (vecteurs normalisés, a, b et c exclus),
      sur 16 analogies choisies à l'avance.
    """)


@app.cell
def _(W, idx, lsa, np, tfidf, vocab):
    GROUPES = {
        "famille": "father mother son daughter brother sister husband wife uncle cousin".split(),
        "nombres": "two three four five six seven ten twenty hundred thousand".split(),
        "corps": "eyes ears lips hand hands head heart tongue face arm".split(),
        "titres": "king queen prince duke lord lady earl sir madam master".split(),
        "temps": "day night morning morrow hour year years week month tomorrow".split(),
    }
    ANALOGIES = [
        ("man", "woman", "king", "queen"), ("he", "she", "his", "her"),
        ("he", "she", "him", "her"), ("father", "mother", "son", "daughter"),
        ("brother", "sister", "son", "daughter"), ("husband", "wife", "king", "queen"),
        ("lord", "lady", "king", "queen"), ("king", "queen", "prince", "princess"),
        ("i", "my", "he", "his"), ("i", "me", "he", "him"),
        ("we", "our", "they", "their"), ("thou", "thy", "you", "your"),
        ("eye", "eyes", "hand", "hands"), ("man", "men", "lord", "lords"),
        ("king", "kings", "lord", "lords"), ("brother", "brothers", "son", "sons"),
    ]
    analogies_dispo = [a for a in ANALOGIES if all(m in idx for m in a)]
    mots_groupes = [m for g in GROUPES.values() for m in g if m in idx]
    etiquettes_groupes = [g for g, l in GROUPES.items() for m in l if m in idx]

    def normer(E):
        E = np.asarray(E, dtype=np.float64)
        return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)

    def voisins(E, mot, k=8):
        En = normer(E)
        s = En @ En[idx[mot]]
        s[idx[mot]] = -np.inf
        o = np.argsort(-s)[:k]
        return [(vocab[i], float(s[i])) for i in o]

    def test_groupes(E):
        X = normer(np.asarray(E)[[idx[m] for m in mots_groupes]])
        S = X @ X.T
        np.fill_diagonal(S, -np.inf)
        pp = S.argmax(1)
        return float(np.mean([etiquettes_groupes[i] == etiquettes_groupes[j] for i, j in enumerate(pp)]))

    def analogie(En, a, b, c, k=5):
        """3CosAdd : mots les plus proches de b − a + c (a, b, c exclus)."""
        s = En @ (En[idx[b]] - En[idx[a]] + En[idx[c]])
        for m in (a, b, c):
            s[idx[m]] = -np.inf
        return [vocab[i] for i in np.argsort(-s)[:k]]

    def test_analogies(E):
        En = normer(E)
        rep = [analogie(En, a, b, c) for a, b, c, _d in analogies_dispo]
        top1 = sum(r[0] == d for r, (_a, _b, _c, d) in zip(rep, analogies_dispo))
        top5 = sum(d in r for r, (_a, _b, _c, d) in zip(rep, analogies_dispo))
        return top1, top5, [r[0] for r in rep]

    REPRESENTATIONS = {"TF-IDF": tfidf, "LSA": lsa, "word2vec": W}
    scores = {}
    for _nom, _E in REPRESENTATIONS.items():
        _t1, _t5, _rep = test_analogies(_E)
        scores[_nom] = {"dimension": int(np.asarray(_E).shape[1]), "groupes": test_groupes(_E),
                        "analogies_top1": _t1, "analogies_top5": _t5, "reponses": _rep}
        print(f"{_nom:9s} dim {scores[_nom]['dimension']:5d} | groupes {scores[_nom]['groupes']:.0%} | "
              f"analogies {_t1}/{len(analogies_dispo)} (top 5 : {_t5})")
    print("(one-hot : cosinus nul entre deux mots distincts, aucun voisin significatif)")

    MOTS_VOISINS = [m for m in ["king", "father", "three", "night", "eyes", "love", "sword", "rome"] if m in idx]
    table_voisins = {n: {m: voisins(E, m, 6) for m in MOTS_VOISINS} for n, E in REPRESENTATIONS.items()}
    for _m in MOTS_VOISINS:
        print(f"\n{_m}")
        for _n in REPRESENTATIONS:
            print(f"  {_n:9s}", ", ".join(v for v, _s in table_voisins[_n][_m]))
    return (
        GROUPES,
        MOTS_VOISINS,
        analogie,
        analogies_dispo,
        etiquettes_groupes,
        mots_groupes,
        normer,
        scores,
        table_voisins,
    )


@app.cell
def _(analogies_dispo, scores):
    # Détail des analogies : ce qui marche, ce qui rate
    for (_a, _b, _c, _d), _r_tfidf, _r_w2v in zip(analogies_dispo, scores["TF-IDF"]["reponses"],
                                                  scores["word2vec"]["reponses"]):
        print(f"{_a}:{_b} :: {_c}:? attendu {_d:9s} | TF-IDF {_r_tfidf:9s}"
              f"{'✓' if _r_tfidf == _d else ' '} | word2vec {_r_w2v:9s}{'✓' if _r_w2v == _d else ' '}")


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Projeter en 2D : ce qu'on voit et ce qu'on perd

    On garde les mots les plus fréquents (ceux de la démo `EmbeddingExplorer`), on
    normalise leurs vecteurs word2vec, puis on les projette en 2D par **ACP** (deux axes
    de variance maximale) et par **t-SNE** (qui cherche à préserver les voisinages ; ces
    deux méthodes sont présentées au chapitre 08). Mesure de la déformation : pour chaque
    mot, part de ses 10 plus proches voisins en 50 dimensions qui restent parmi ses 10
    plus proches voisins dans le plan.
    """)


@app.cell
def _(W, config, idx, mots_groupes, normer, np, time):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    _t = time.perf_counter()
    # Mots exportés : les plus fréquents, plus ceux des groupes (même s'ils sont rares)
    _ids = list(range(min(config["n_export"], len(W))))
    _ids += [idx[m] for m in mots_groupes if idx[m] not in set(_ids)]
    ids_export = np.array(_ids)
    E_export = normer(W[ids_export])
    _acp = PCA(n_components=2, random_state=42)
    proj_acp = _acp.fit_transform(E_export)
    variance_acp = float(_acp.explained_variance_ratio_.sum())
    proj_tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42,
                     max_iter=config["tsne_iter"]).fit_transform(E_export)

    def recouvrement(haute, basse, k=10):
        """Part moyenne des k plus proches voisins (en haute dimension) conservés en 2D."""
        def kpp(X, cos):
            if cos:
                S = X @ X.T
            else:
                S = -((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
            np.fill_diagonal(S, -np.inf)
            return np.argsort(-S, axis=1)[:, :k]
        a, b = kpp(haute, True), kpp(basse, False)
        return float(np.mean([len(set(a[i]) & set(b[i])) / k for i in range(len(a))]))

    recouvrement_acp = recouvrement(E_export, proj_acp)
    recouvrement_tsne = recouvrement(E_export, proj_tsne)
    print(f"{len(ids_export)} mots ; ACP : {variance_acp:.1%} de la variance sur 2 axes, "
          f"{recouvrement_acp:.1%} des 10 voisins conservés ; t-SNE : {recouvrement_tsne:.1%} "
          f"({time.perf_counter() - _t:.1f} s)")
    return (
        E_export,
        ids_export,
        proj_acp,
        proj_tsne,
        recouvrement_acp,
        recouvrement_tsne,
        variance_acp,
    )


@app.cell
def _(
    GROUPES,
    PALETTE,
    SMOKE,
    etiquettes_groupes,
    idx,
    ids_export,
    mo,
    mots_groupes,
    np,
    pertes_epoques,
    pertes_lots,
    plt,
    proj_acp,
    proj_tsne,
    recouvrement_acp,
    recouvrement_tsne,
    sauvegarder_figure,
    variance_acp,
):
    # --- Figures de l'exercice : courbe de perte et projections 2D ---
    _dossier = mo.notebook_dir() / "figures"
    _suffixe = "-smoke" if SMOKE else ""
    fig_perte, _ax = plt.subplots(figsize=(7, 3.8))
    _x = np.linspace(0, len(pertes_epoques), len(pertes_lots))
    _ax.plot(_x, pertes_lots, lw=0.8, alpha=0.6, label="lots (échantillon)")
    _ax.plot(np.arange(1, len(pertes_epoques) + 1), pertes_epoques, "o-", label="moyenne par époque")
    _ax.axhline(6 * np.log(2), ls=":", color="grey", label="6 ln 2 (départ)")
    _ax.set(xlabel="époque", ylabel="perte skip-gram (négatifs K = 5)", title="Apprentissage de word2vec")
    _ax.legend()
    sauvegarder_figure(fig_perte, "perte" + _suffixe, _dossier)

    _couleur = dict(zip(GROUPES, PALETTE))
    _pos = {i: k for k, i in enumerate(ids_export)}
    fig_proj, _axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for _a, _P, _titre in [(_axes[0], proj_acp, f"ACP : {variance_acp:.0%} de la variance, "
                                                  f"{recouvrement_acp:.0%} des voisins conservés"),
                           (_axes[1], proj_tsne, f"t-SNE : {recouvrement_tsne:.0%} des voisins conservés")]:
        _a.scatter(_P[:, 0], _P[:, 1], s=4, color="#BBBBBB")
        for _m, _g in zip(mots_groupes, etiquettes_groupes):
            _p = _P[_pos[idx[_m]]]
            _a.scatter(*_p, s=18, color=_couleur[_g])
            _a.annotate(_m, _p, fontsize=7, color=_couleur[_g], xytext=(2, 2), textcoords="offset points")
        _a.set(title=_titre, xticks=[], yticks=[])
    for _g, _c in _couleur.items():
        _axes[1].scatter([], [], color=_c, label=_g)
    _axes[1].legend(loc="lower right", fontsize=8)
    sauvegarder_figure(fig_proj, "projections" + _suffixe, _dossier)
    fig_proj


@app.cell
def _(
    E_export,
    MOTS_VOISINS,
    SMOKE,
    T0,
    W,
    analogies_dispo,
    comptes,
    config,
    cos_onehot,
    df,
    duree_w2v,
    ecart_gradient,
    ecart_sigma,
    ecart_tfidf,
    etiquettes_groupes,
    idx,
    ids_export,
    json,
    lsa,
    mo,
    mots_groupes,
    np,
    part_energie_lsa,
    perte_initiale,
    pertes_epoques,
    pertes_lots,
    proj_acp,
    proj_tsne,
    recherches,
    recouvrement_acp,
    recouvrement_tsne,
    sacs_identiques,
    scores,
    table_voisins,
    taux_hors_vocab,
    tfidf,
    time,
    variance_acp,
    vocab,
):
    # --- Bilan : seuils du README (vérifiés en mode complet seulement) ---
    duree_totale = time.perf_counter() - T0
    _w2v = scores["word2vec"]
    _n_an = len(analogies_dispo)
    bilan = {
        "taille_vocab": len(vocab), "taux_hors_vocab": taux_hors_vocab, "cos_onehot": {f"{a}/{b}": v for (a, b), v in cos_onehot.items()},
        "ecart_tfidf_sklearn": ecart_tfidf, "sacs_identiques": sacs_identiques,
        "ecart_sigma_lsa": ecart_sigma, "energie_lsa": part_energie_lsa,
        "ecart_gradient": ecart_gradient, "perte_initiale": perte_initiale,
        "pertes_epoques": pertes_epoques, "duree_w2v_s": round(duree_w2v, 1),
        "scores": scores, "n_analogies": _n_an, "analogies": analogies_dispo,
        "variance_acp": variance_acp, "recouvrement_acp": recouvrement_acp,
        "recouvrement_tsne": recouvrement_tsne,
        "recherches": recherches, "voisins": table_voisins, "duree_totale_s": round(duree_totale, 1),
    }
    print(json.dumps({k: v for k, v in bilan.items() if k not in ("voisins", "recherches")},
                     ensure_ascii=False, default=float))
    # Résultats bruts pour les figures du site et la démo (site/scripts/figures/ch15_*.py)
    _mots_fig = sorted(set(mots_groupes) | set(MOTS_VOISINS) | {m for a in analogies_dispo for m in a})
    _ids_fig = np.array([idx[m] for m in _mots_fig])
    np.savez_compressed(
        mo.notebook_dir() / "figures" / ("resultats-smoke.npz" if SMOKE else "resultats.npz"),
        bilan=json.dumps(bilan, ensure_ascii=False, default=float),
        vocab=np.array(vocab), comptes=comptes.astype(np.int32), df=df.astype(np.int32),
        w2v=W.astype(np.float16),  # float16 : fichier léger, précision suffisante pour les figures
        mots_fig=np.array(_mots_fig), tfidf_fig=tfidf[_ids_fig].astype(np.float16), lsa_fig=lsa[_ids_fig],
        ids_export=ids_export, export=E_export.astype(np.float16),
        proj_acp=proj_acp.astype(np.float32), proj_tsne=proj_tsne.astype(np.float32),
        mots_groupes=np.array(mots_groupes), etiquettes_groupes=np.array(etiquettes_groupes),
        pertes_lots=np.array(pertes_lots, dtype=np.float32),
    )

    SEUILS = [
        ("TF-IDF numpy = scikit-learn (écart < 1e-4)", ecart_tfidf < 1e-4),
        ("LSA : valeurs singulières = TruncatedSVD (écart relatif < 1e-6)", ecart_sigma < 1e-6),
        ("gradient skip-gram = autograd (écart < 1e-10)", ecart_gradient < 1e-10),
        ("perte initiale ≈ 6 ln 2 (à 0,01 près)", abs(perte_initiale - 6 * np.log(2)) < 0.01),
        ("perte de la dernière époque ≤ 2,60", pertes_epoques[-1] <= 2.60),
        ("word2vec : test de groupes ≥ 75 %", _w2v["groupes"] >= 0.75),
        (f"word2vec : au moins 6 analogies sur {_n_an} en top 1", _w2v["analogies_top1"] >= 6),
        ("word2vec fait au moins aussi bien que TF-IDF aux analogies",
         _w2v["analogies_top1"] >= scores["TF-IDF"]["analogies_top1"]),
        ("t-SNE conserve plus de voisins que l'ACP", recouvrement_tsne > recouvrement_acp),
    ]
    _echecs = [nom for nom, ok in SEUILS if not ok]
    for _nom, _ok in SEUILS:
        print(("OK    " if _ok else ("(smoke) " if SMOKE else "ÉCHEC ")) + _nom)
    if not SMOKE and _echecs:
        raise SystemExit("Seuils non atteints : " + " ; ".join(_echecs))
    mo.md(f"**Bilan** : word2vec {_w2v['groupes']:.0%} au test de groupes, "
          f"{_w2v['analogies_top1']}/{_n_an} analogies ; {duree_w2v:.0f} s d'entraînement, "
          f"{duree_totale:.0f} s au total.")


if __name__ == "__main__":
    app.run()
