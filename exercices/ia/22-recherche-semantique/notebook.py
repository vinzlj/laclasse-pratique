# Exercice 22 : embeddings de phrases et recherche vectorielle (chapitre 22).
#
#   make ex NUM=22              → éditeur marimo (bouton « Lancer »)
#   make ex NUM=22 MODE=run     → script complet, vérifie les seuils du README
#   make ex NUM=22 MODE=smoke   → test rapide (< 30 s)
#
# Progression :
#   1. Pourquoi le vecteur [CLS] d'un BERT brut est un mauvais embedding de phrase :
#      même architecture (MiniLM multilingue, 12 couches, 384 dimensions), avant et après
#      l'entraînement « phrases » de sentence-transformers, sur STS-B traduit en français.
#   2. Recherche sémantique (questions → passages) sur PIAF, en numpy : TF-IDF, modèle
#      « paraphrases », modèle multilingual-e5 avec et sans ses préfixes ; coût d'un
#      cross-encodeur estimé par la mesure.
#   3. Recherche approchée sur ~53 800 phrases de romans français (Gutenberg) : recherche
#      exacte en numpy, IVF et quantification de produit écrits en numpy, puis faiss
#      (Flat, HNSW, IVF, IVF-PQ) : rappel, latence, mémoire, passage à l'échelle.
#
# Aucun entraînement : on n'utilise que des modèles déjà entraînés (inférence sur MPS).
# faiss : toujours via ia_commun.compat.importer_faiss() (sinon plantage à côté de torch).

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import re
    import sys
    import time
    import urllib.request

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import hf_hub_download
    from scipy.stats import spearmanr
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Contournement (transformers 5.19) : quand marimo est chargé, transformers se croit dans
    # un notebook Jupyter et importe IPython (absent) en chargeant son Trainer, ce que fait
    # sentence-transformers. On masque marimo le temps de cet import, puis on le remet.
    _marimo = sys.modules.pop("marimo", None)
    try:
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModel
    finally:
        if _marimo is not None:
            sys.modules["marimo"] = _marimo

    from ia_commun import (
        PALETTE,
        data_dir,
        get_device,
        is_smoke,
        sauvegarder_figure,
        seed_everything,
        style_commun,
    )
    from ia_commun.compat import importer_faiss

    # torch est déjà importé : importer_faiss() limite faiss à un thread (PLATEFORME.md, test a)
    faiss = importer_faiss()
    return (
        AutoModel,
        PALETTE,
        SentenceTransformer,
        TfidfVectorizer,
        data_dir,
        faiss,
        get_device,
        hf_hub_download,
        is_smoke,
        json,
        mo,
        np,
        plt,
        pq,
        re,
        sauvegarder_figure,
        seed_everything,
        spearmanr,
        style_commun,
        time,
        torch,
        urllib,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Exercice 22 : embeddings de phrases et recherche vectorielle

    **Objectif.** Transformer une phrase en un vecteur tel que « proche » veuille dire
    « de même sens », puis retrouver en quelques dixièmes de milliseconde les voisins
    d'une requête parmi des dizaines de milliers de vecteurs.

    **Ce qu'on optimise.** Rien : aucun réseau n'est entraîné ici. Les modèles utilisés
    ont été entraînés par d'autres : e5 de façon contrastive (rapprocher les textes
    appariés, éloigner les autres du lot, chapitre 23) ; le modèle SBERT multilingue en
    imitant un modèle anglais (une phrase traduite doit tomber au même endroit que
    l'originale). On mesure seulement :

    - la **similarité cosinus** $\cos(u, v) = \frac{u \cdot v}{\|u\|\,\|v\|}$ ; sur des
      vecteurs normalisés ($\|u\| = 1$), c'est un simple produit scalaire ;
    - la **corrélation de Spearman** $\rho$ entre les cosinus et des notes humaines de
      similarité (0 à 5) : 1 = même classement que les humains ;
    - le **rappel@k** : proportion des requêtes dont la bonne réponse est parmi les k
      premiers résultats ; pour la recherche approchée, proportion des 10 vrais plus
      proches voisins retrouvés.

    Plan : (1) [CLS] brut contre embedding entraîné ; (2) recherche sémantique questions
    → passages ; (3) recherche exacte, IVF, quantification de produit, HNSW.
    """)


@app.cell
def _(get_device, is_smoke, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    device = get_device()
    style_commun()
    config = dict(
        n_sts=100 if SMOKE else None,          # paires STS-B (None = les 1 379 du test)
        n_contextes_piaf=40 if SMOKE else None,  # passages PIAF (None = les 761)
        n_cross=1 if SMOKE else 10,            # questions pour mesurer le coût d'un cross-encodeur
        # Gutenberg : 12 volumes (Hugo, Verne, Dumas) ; en smoke, un seul déjà en cache (ex. 19)
        livres=[800] if SMOKE else [17489, 17493, 17494, 17518, 17519, 800, 5097, 4791,
                                    17989, 17990, 17991, 17992],
        max_phrases=1500 if SMOKE else None,
        n_requetes=50 if SMOKE else 1000,     # phrases retirées de la base, servent de requêtes
        n_latence=30 if SMOKE else 300,        # requêtes chronométrées une par une
        k=10,                                  # on cherche les 10 plus proches voisins
        nlist=16 if SMOKE else 256,            # IVF : nombre de cellules (≈ √N)
        iters_kmeans=5 if SMOKE else 15,
        n_train_kmeans=2000 if SMOKE else 20000,
        pq_m=48,                               # PQ : 48 sous-vecteurs de 8 dimensions, 1 octet chacun
        n_requetes_pq=30 if SMOKE else 300,
        hnsw_M=32,
        hnsw_ef_construction=64,
        ef_search=[8, 16, 32] if SMOKE else [8, 16, 32, 64, 128, 256],
        nprobe=[1, 4] if SMOKE else [1, 2, 4, 8, 16, 32, 64],
        tailles=[500, 1000] if SMOKE else [2000, 8000, 32000],  # + la base entière
        batch=64,
    )
    # Révisions figées des modèles et jeux de données du Hub (reproductibilité)
    REV = dict(
        sbert=("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
               "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"),
        brut=("microsoft/Multilingual-MiniLM-L12-H384", "6e8c1ec6b4ec4e3fc6eb7d2cd834fcd582b61daf"),
        e5=("intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3"),
        sts=("PhilipMay/stsb_multi_mt", "3acaa3dd8c91649e0b8e627ffad891f059e47c8c"),
        piaf=("AgentPublic/piaf", None),
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, périphérique {device}")
    return REV, SEED, SMOKE, config, device


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Les trois modèles

    | Nom ici | Modèle (Hugging Face) | Ce que c'est |
    |---|---|---|
    | **brut** | `microsoft/Multilingual-MiniLM-L12-H384` (MIT) | Transformer encodeur pré-entraîné (modèle de langue masqué distillé), jamais entraîné à comparer des phrases |
    | **SBERT** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0) | **même architecture** (BertModel, 12 couches, 384 dimensions, vocabulaire de XLM-R), entraînée à rapprocher les paraphrases ; embedding = moyenne des vecteurs de tokens |
    | **e5** | `intfloat/multilingual-e5-small` (MIT) | même taille, entraîné pour la **recherche** ; chaque texte doit commencer par `query: ` ou `passage: ` |

    Téléchargement au premier lancement : ~1,4 Go au total, dans `data/hf/`.
    """)


@app.cell
def _(AutoModel, REV, SentenceTransformer, device, np, torch):
    # SBERT et e5 : la bibliothèque sentence-transformers fait tokenisation + réseau + pooling
    sbert = SentenceTransformer(REV["sbert"][0], revision=REV["sbert"][1], device=str(device))
    e5 = SentenceTransformer(REV["e5"][0], revision=REV["e5"][1], device=str(device))
    # Modèle brut : chargé avec transformers. Sa carte indique qu'il utilise le tokeniseur de
    # XLM-R ; celui de SBERT est le même (vocabulaire identique, vérifié), on le réutilise.
    tokeniseur = sbert.tokenizer
    brut = AutoModel.from_pretrained(REV["brut"][0], revision=REV["brut"][1]).to(device).eval()

    def encoder_brut(textes, batch=64, max_len=128):
        """Vecteur [CLS] (1er token) et moyenne des tokens de la dernière couche du modèle brut."""
        n = len(textes)
        cls = np.zeros((n, 384), np.float32)
        moy = np.zeros((n, 384), np.float32)
        ordre = np.argsort([len(t) for t in textes])  # lots de longueurs voisines : moins de remplissage
        for i in range(0, n, batch):
            idx = ordre[i:i + batch]
            enc = tokeniseur([textes[j] for j in idx], padding=True, truncation=True,
                             max_length=max_len, return_tensors="pt").to(device)
            with torch.no_grad():
                h = brut(**enc).last_hidden_state                  # (lot, tokens, 384)
            masque = enc["attention_mask"].unsqueeze(-1).float()   # 0 sur le remplissage
            cls[idx] = h[:, 0].float().cpu().numpy()
            moy[idx] = ((h * masque).sum(1) / masque.sum(1)).float().cpu().numpy()
        return cls, moy

    def encoder(modele, textes, batch=64):
        """Embeddings normalisés (norme 1) : le cosinus devient un produit scalaire."""
        return modele.encode(textes, batch_size=batch, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    def normaliser(X):
        return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)

    print(f"brut : {sum(p.numel() for p in brut.parameters()) / 1e6:.0f} M paramètres "
          f"(dont {brut.embeddings.word_embeddings.weight.numel() / 1e6:.0f} M pour la table des "
          f"{brut.config.vocab_size} tokens) ; SBERT et e5 : même taille")
    return brut, e5, encoder, encoder_brut, normaliser, sbert


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Pourquoi le [CLS] d'un BERT brut est un mauvais embedding

    STS-B (*Semantic Textual Similarity benchmark*) : des paires de phrases notées de 0
    (sans rapport) à 5 (même sens) par des humains ; ici sa traduction française
    automatique (DeepL), partie test, 1 379 paires. Pour chaque méthode, on calcule le
    cosinus de chaque paire et la corrélation de Spearman avec les notes humaines.

    On mesure aussi l'**anisotropie** : le cosinus moyen entre deux phrases **tirées au
    hasard**. S'il vaut 0,9, tous les vecteurs pointent dans la même direction et le
    cosinus ne distingue presque rien.
    """)


@app.cell
def _(REV, config, hf_hub_download, pq):
    _chemin = hf_hub_download(REV["sts"][0], "fr/test-00000-of-00001.parquet",
                              repo_type="dataset", revision=REV["sts"][1])
    _t = pq.read_table(_chemin).to_pylist()
    if config["n_sts"]:
        _t = _t[:config["n_sts"]]
    sts_a = [r["sentence1"] for r in _t]
    sts_b = [r["sentence2"] for r in _t]
    sts_note = [r["similarity_score"] for r in _t]
    print(f"STS-B fr (test) : {len(_t)} paires ; exemple : « {sts_a[0]} » / « {sts_b[0]} » "
          f"→ note {sts_note[0]:.1f}")
    return sts_a, sts_b, sts_note


@app.cell
def _(mo):
    bouton = mo.ui.run_button(label="Lancer les calculs")
    bouton
    return (bouton,)


@app.cell
def _(
    TfidfVectorizer,
    bouton,
    e5,
    encoder,
    encoder_brut,
    mo,
    normaliser,
    np,
    sbert,
    spearmanr,
    sts_a,
    sts_b,
    sts_note,
    time,
):
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer les calculs ».*"))
    _rng = np.random.default_rng(42)
    _perm = _rng.permutation(len(sts_b))  # paires « au hasard » pour l'anisotropie
    sts_cos, sts_rho, sts_hasard = {}, {}, {}

    # TF-IDF (chapitre 15) : un vecteur creux par phrase, un axe par mot du vocabulaire
    _tf = TfidfVectorizer(sublinear_tf=True).fit(sts_a + sts_b)
    _A, _B = _tf.transform(sts_a), _tf.transform(sts_b)       # déjà normalisés (norme L2)
    sts_cos["TF-IDF"] = np.asarray(_A.multiply(_B).sum(1)).ravel()
    sts_hasard["TF-IDF"] = float(np.asarray(_A.multiply(_B[_perm]).sum(1)).mean())

    _t0 = time.perf_counter()
    _cls_a, _moy_a = encoder_brut(sts_a)
    _cls_b, _moy_b = encoder_brut(sts_b)
    _embs = {
        "brut [CLS]": (normaliser(_cls_a), normaliser(_cls_b)),
        "brut moyenne": (normaliser(_moy_a), normaliser(_moy_b)),
        "SBERT": (encoder(sbert, sts_a), encoder(sbert, sts_b)),
        # tâche symétrique : la carte du modèle demande « query: » des deux côtés
        "e5": (encoder(e5, ["query: " + s for s in sts_a]), encoder(e5, ["query: " + s for s in sts_b])),
    }
    sts_duree = time.perf_counter() - _t0
    for _nom, (_ea, _eb) in _embs.items():
        sts_cos[_nom] = (_ea * _eb).sum(1)                      # cosinus = produit scalaire
        sts_hasard[_nom] = float((_ea * _eb[_perm]).sum(1).mean())
    for _nom, _c in sts_cos.items():
        sts_rho[_nom] = float(spearmanr(_c, sts_note).statistic)
    print(f"Encodage : {sts_duree:.1f} s pour 4 × {2 * len(sts_a)} phrases")
    print(f"{'méthode':<14} {'Spearman':>9} {'cos moyen (paires au hasard)':>30}")
    for _nom in sts_cos:
        print(f"{_nom:<14} {100 * sts_rho[_nom]:>8.1f}  {sts_hasard[_nom]:>29.3f}")
    return sts_cos, sts_hasard, sts_rho


@app.cell
def _(mo):
    mo.md(r"""
    **À lire** : le Spearman du [CLS] brut est faible, et ses cosinus sont tous proches
    les uns des autres (anisotropie). Le même réseau, après l'entraînement « phrases »,
    classe les paires presque comme les humains.

    ## 2. Recherche sémantique : questions → passages (PIAF)

    PIAF : 3 835 questions écrites par des bénévoles sur 761 paragraphes de Wikipédia en
    français. Chaque question a **un** paragraphe d'où vient sa réponse. Tâche : retrouver
    ce paragraphe parmi les 761. C'est une recherche **asymétrique** (question courte,
    passage long), la tâche du *retriever* d'un RAG (chapitre 32).

    La recherche elle-même tient en trois lignes de numpy : produit matriciel requêtes ×
    passages (vecteurs normalisés), puis les k meilleurs scores par ligne.
    """)


@app.cell
def _(REV, config, hf_hub_download, np, pq):
    _chemin = hf_hub_download(REV["piaf"][0], "plain_text/train-00000-of-00001.parquet",
                              repo_type="dataset", revision=REV["piaf"][1])
    _lignes = pq.read_table(_chemin, columns=["title", "context", "question"]).to_pylist()
    piaf_passages, _index = [], {}
    piaf_titres = []
    for _r in _lignes:
        if _r["context"] not in _index:
            _index[_r["context"]] = len(piaf_passages)
            piaf_passages.append(_r["context"])
            piaf_titres.append(_r["title"])
    if config["n_contextes_piaf"]:
        piaf_passages = piaf_passages[:config["n_contextes_piaf"]]
    piaf_questions = [r["question"] for r in _lignes if _index[r["context"]] < len(piaf_passages)]
    piaf_gold = np.array(
        [_index[r["context"]] for r in _lignes if _index[r["context"]] < len(piaf_passages)])
    print(f"PIAF : {len(piaf_questions)} questions, {len(piaf_passages)} passages, "
          f"{len(set(piaf_titres[:len(piaf_passages)]))} articles ; "
          f"passage moyen {sum(map(len, piaf_passages)) / len(piaf_passages):.0f} caractères")
    print(f"Exemple : « {piaf_questions[0]} »")
    return piaf_gold, piaf_passages, piaf_questions


@app.cell
def _(np):
    def topk(scores, k):
        """Indices des k plus grands scores de chaque ligne, triés (from scratch)."""
        k = min(k, scores.shape[1])
        idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]      # les k meilleurs, en vrac
        ordre = np.argsort(-np.take_along_axis(scores, idx, 1), axis=1)
        return np.take_along_axis(idx, ordre, 1)

    def evaluer(scores, gold):
        top = topk(scores, 10)
        rang = np.full(len(gold), np.inf)
        trouve = top == gold[:, None]
        rang[trouve.any(1)] = trouve.argmax(1)[trouve.any(1)] + 1
        return {"r@1": float((rang <= 1).mean()), "r@5": float((rang <= 5).mean()),
                "r@10": float((rang <= 10).mean()),
                "mrr@10": float(np.where(rang <= 10, 1 / rang, 0).mean())}

    return evaluer, topk


@app.cell
def _(
    TfidfVectorizer,
    e5,
    encoder,
    encoder_brut,
    evaluer,
    normaliser,
    np,
    piaf_gold,
    piaf_passages,
    piaf_questions,
    sbert,
    sts_rho,
    time,
):
    _ = sts_rho  # ordre d'exécution : partie 1 avant partie 2
    piaf_res, piaf_temps = {}, {}

    _tf = TfidfVectorizer(sublinear_tf=True).fit(piaf_passages)
    _t0 = time.perf_counter()
    piaf_res["TF-IDF"] = evaluer((_tf.transform(piaf_questions) @ _tf.transform(piaf_passages).T).toarray(), piaf_gold)
    piaf_temps["TF-IDF"] = time.perf_counter() - _t0

    _t0 = time.perf_counter()
    _, _qm = encoder_brut(piaf_questions)
    _, _pm = encoder_brut(piaf_passages, batch=32, max_len=256)
    piaf_res["brut moyenne"] = evaluer(normaliser(_qm) @ normaliser(_pm).T, piaf_gold)
    piaf_temps["brut moyenne"] = time.perf_counter() - _t0

    _t0 = time.perf_counter()
    _P_sbert = encoder(sbert, piaf_passages, batch=32)
    _Q_sbert = encoder(sbert, piaf_questions)
    piaf_res["SBERT"] = evaluer(_Q_sbert @ _P_sbert.T, piaf_gold)
    piaf_temps["SBERT"] = time.perf_counter() - _t0

    _t0 = time.perf_counter()
    _P = encoder(e5, ["passage: " + p for p in piaf_passages], batch=32)
    _Q = encoder(e5, ["query: " + q for q in piaf_questions])
    piaf_res["e5 (préfixes)"] = evaluer(_Q @ _P.T, piaf_gold)
    piaf_temps["e5 (préfixes)"] = time.perf_counter() - _t0
    _P0 = encoder(e5, piaf_passages, batch=32)
    _Q0 = encoder(e5, piaf_questions)
    piaf_res["e5 sans préfixe"] = evaluer(_Q0 @ _P0.T, piaf_gold)

    # Fusion naïve lexical + sémantique (moyenne des scores) : un avant-goût de l'hybride du ch. 32
    _s_tf = (_tf.transform(piaf_questions) @ _tf.transform(piaf_passages).T).toarray()
    piaf_res["TF-IDF + e5"] = evaluer(0.5 * _s_tf + 0.5 * (_Q @ _P.T), piaf_gold)

    print(f"{'méthode':<16} {'r@1':>6} {'r@5':>6} {'r@10':>6} {'MRR@10':>7}  temps")
    for _nom, _r in piaf_res.items():
        _tps = f"{piaf_temps[_nom]:.1f} s" if _nom in piaf_temps else ""
        print(f"{_nom:<16} {100 * _r['r@1']:>5.1f}% {100 * _r['r@5']:>5.1f}% {100 * _r['r@10']:>5.1f}% "
              f"{_r['mrr@10']:>7.3f}  {_tps}")
    piaf_exemple = (piaf_questions[0], int(np.argmax(_Q[0] @ _P.T)), int(piaf_gold[0]))
    return piaf_exemple, piaf_res


@app.cell
def _(mo):
    mo.md(r"""
    ### Bi-encodeur contre cross-encodeur : le coût

    Le **bi-encodeur** encode chaque passage **une fois** (hors ligne), puis chaque
    question une fois : 761 + 1 passages dans le réseau pour la première question, 1 pour
    les suivantes. Un **cross-encodeur** lit la question et le passage **ensemble** (il
    voit les deux textes dans la même attention, donc juge mieux) : il faut un passage
    dans le réseau **par paire** (question, passage), soit 761 par question.

    On ne charge pas de cross-encodeur (c'est l'affaire du chapitre 32) : on mesure le
    coût d'un passage dans un réseau **de même taille** (SBERT) de 761 textes
    « question + passage » : une borne basse de ce qu'il ferait, car SBERT tronque
    chaque texte à 128 tokens, alors qu'un vrai cross-encodeur en lirait jusqu'à 512.
    """)


@app.cell
def _(config, encoder, np, piaf_passages, piaf_questions, piaf_res, sbert, time):
    _ = piaf_res
    _q = piaf_questions[:config["n_cross"]]
    encoder(sbert, [_q[0] + " " + _p for _p in piaf_passages[:64]], batch=32)  # échauffement
    _t0 = time.perf_counter()
    for _question in _q:
        encoder(sbert, [_question + " " + _p for _p in piaf_passages], batch=32)
    cross_par_question = (time.perf_counter() - _t0) / len(_q)
    # Bi-encodeur, en ligne : encoder la question, puis un produit scalaire avec chacun des
    # passages, encodés une fois pour toutes (hors ligne, non chronométré)
    _P = encoder(sbert, piaf_passages, batch=32)
    _qb = piaf_questions[:5 * config["n_cross"]]
    encoder(sbert, [_qb[0]])  # échauffement
    _t0 = time.perf_counter()
    for _question in _qb:
        _v = encoder(sbert, [_question])
        np.argsort(-(_P @ _v[0]))[:10]
    bi_par_question = (time.perf_counter() - _t0) / len(_qb)
    print(f"Cross-encodeur (761 paires par question) : {cross_par_question:.2f} s par question")
    print(f"Bi-encodeur (1 encodage + produits scalaires) : {1000 * bi_par_question:.1f} ms par question "
          f"→ rapport ≈ {cross_par_question / bi_par_question:.0f}")
    return bi_par_question, cross_par_question


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Chercher vite parmi des dizaines de milliers de vecteurs

    Base : les phrases de 12 volumes de romans français du domaine public (Hugo, *Les
    Misérables* ; Verne, *Le Tour du monde en 80 jours*, *Vingt mille lieues sous les
    mers*, *Voyage au centre de la Terre* ; Dumas, *Le Comte de Monte-Cristo*), projet
    Gutenberg, encodées par SBERT. 1 000 phrases sont mises de côté comme requêtes.

    La **vérité** est la recherche exacte : les 10 plus proches voisins de chaque requête
    par produit scalaire avec **toute** la base. Les méthodes approchées sont jugées par
    leur **rappel@10** : combien de ces 10 vrais voisins elles retrouvent.
    """)


@app.cell
def _(config, data_dir, re, urllib):
    def telecharger_livre(numero):
        dossier = data_dir() / "gutenberg"
        chemin = dossier / f"pg{numero}.txt"
        if not chemin.exists():
            url = f"https://www.gutenberg.org/ebooks/{numero}.txt.utf-8"
            print(f"Téléchargement de {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "ia-experiments (exercice 22)"})
            with urllib.request.urlopen(req, timeout=60) as rep:
                contenu = rep.read()
            tmp = chemin.with_name(chemin.name + ".part")
            tmp.write_bytes(contenu)
            tmp.replace(chemin)  # pas de fichier à moitié écrit dans le cache
        brut_ = chemin.read_text(encoding="utf-8")
        return brut_[brut_.index("\n", brut_.index("*** START OF")):brut_.index("*** END OF")]

    def phrases_de(texte):
        texte = re.sub(r"\s+", " ", texte.replace("_", ""))
        # coupe après . ! ? … suivi d'une majuscule, d'un tiret ou d'un guillemet ouvrant
        morceaux = re.split(r"(?<=[.!?…])\s+(?=[«A-ZÀÂÉÈÊÎÔÛÇ—-])", texte)
        return [m.strip() for m in morceaux if 40 <= len(m.strip()) <= 400]

    _vues, phrases = set(), []
    for _n in config["livres"]:
        for _p in phrases_de(telecharger_livre(_n)):
            if _p not in _vues:          # doublons exacts retirés
                _vues.add(_p)
                phrases.append(_p)
    if config["max_phrases"]:
        phrases = phrases[:config["max_phrases"]]
    print(f"{len(phrases)} phrases de 40 à 400 caractères ({len(config['livres'])} volumes)")
    return (phrases,)


@app.cell
def _(bi_par_question, config, encoder, np, phrases, sbert, time):
    _ = bi_par_question
    _t0 = time.perf_counter()
    X_tout = encoder(sbert, phrases, batch=config["batch"])
    duree_encodage = time.perf_counter() - _t0
    _rng = np.random.default_rng(42)
    _perm = _rng.permutation(len(phrases))
    idx_req, idx_base = _perm[:config["n_requetes"]], np.sort(_perm[config["n_requetes"]:])
    Q, B = X_tout[idx_req], X_tout[idx_base]
    print(f"Encodage de {len(phrases)} phrases : {duree_encodage:.1f} s "
          f"({len(phrases) / duree_encodage:.0f} phrases/s) ; base {B.shape}, requêtes {Q.shape}")
    print(f"Mémoire de la base en float32 : {B.nbytes / 1e6:.1f} Mo ({B.shape[1] * 4} octets par vecteur)")
    return B, Q, X_tout, duree_encodage, idx_base


@app.cell
def _(TfidfVectorizer, X_tout, encoder, np, phrases, sbert):
    # Quelques requêtes écrites à la main, sans reprendre les mots du roman
    requetes_libres = [
        "Un ancien forçat dérobe l'argenterie du prêtre qui l'avait hébergé.",
        "Un gentleman anglais parie sa fortune qu'il fera le tour de la planète en moins de trois mois.",
        "Un prisonnier s'échappe en prenant la place d'un cadavre cousu dans un sac.",
        "Des savants descendent dans le cratère d'un volcan éteint d'Islande.",
    ]
    _tf = TfidfVectorizer(sublinear_tf=True).fit(phrases)
    _s_tf = (_tf.transform(requetes_libres) @ _tf.transform(phrases).T).toarray()
    _s_sb = encoder(sbert, requetes_libres) @ X_tout.T
    exemples_libres = []
    for _i, _r in enumerate(requetes_libres):
        print(f"\nRequête : {_r}")
        _ex = {"requete": _r}
        for _nom, _s in (("TF-IDF", _s_tf[_i]), ("SBERT", _s_sb[_i])):
            _top = np.argsort(-_s)[:3]
            _ex[_nom] = [(phrases[j], float(_s[j])) for j in _top]
            for _j in _top:
                print(f"  {_nom:<6} {_s[_j]:.2f}  {phrases[_j][:150]}")
        exemples_libres.append(_ex)
    return (exemples_libres,)


@app.cell
def _(B, Q, config, np, time, topk):
    # --- Recherche exacte en numpy : la vérité, et la référence de latence ---
    K = config["k"]
    _t0 = time.perf_counter()
    verite = topk(Q @ B.T, K)                      # (n_requetes, 10), par lots
    duree_exact_lot = time.perf_counter() - _t0

    def chrono_par_requete(fonction, n):
        """Latence moyenne en ms, requêtes envoyées UNE PAR UNE (comme un service en ligne)."""
        fonction(Q[0])  # échauffement
        t0 = time.perf_counter()
        for i in range(n):
            fonction(Q[i])
        return 1000 * (time.perf_counter() - t0) / n

    def exact_une(q, base=B):
        s = base @ q
        i = np.argpartition(-s, K)[:K]
        return i[np.argsort(-s[i])]

    lat_exact = chrono_par_requete(exact_une, config["n_latence"])

    def rappel(trouves, ref=None):
        ref = verite if ref is None else ref
        return float(np.mean([len(set(a) & set(b)) / K for a, b in zip(trouves, ref)]))

    print(f"Exacte numpy : {lat_exact:.2f} ms par requête ({B.shape[0]} produits scalaires de "
          f"dimension {B.shape[1]}) ; {1000 * duree_exact_lot / len(Q):.3f} ms par requête en lot")
    return K, chrono_par_requete, exact_une, lat_exact, rappel, verite


@app.cell
def _(mo):
    mo.md(r"""
    ### IVF en numpy : ne fouiller que quelques cellules

    *Inverted file* : on partitionne la base en `nlist` cellules par k-moyennes
    (chapitre 08). Chaque vecteur est rangé dans la liste de son centroïde le plus proche.
    À la recherche, on compare la requête aux centroïdes, puis seulement aux vecteurs des
    `nprobe` cellules les plus proches. Si un vrai voisin est tombé dans une cellule non
    visitée, il est perdu : c'est l'approximation.
    """)


@app.cell
def _(B, K, Q, config, np, rappel, time, verite):
    def kmeans_sphere(X, k, iters, rng):
        """k-moyennes sur la sphère : affectation par produit scalaire, centroïdes renormalisés."""
        C = X[rng.choice(len(X), k, replace=False)].copy()
        for _ in range(iters):
            a = np.argmax(X @ C.T, axis=1)
            for j in range(k):
                membres = X[a == j]
                if len(membres):
                    c = membres.sum(0)
                    C[j] = c / np.linalg.norm(c)
        return C

    _rng = np.random.default_rng(42)
    _t0 = time.perf_counter()
    _ech = B[_rng.choice(len(B), min(config["n_train_kmeans"], len(B)), replace=False)]
    centroides = kmeans_sphere(_ech, config["nlist"], config["iters_kmeans"], _rng)
    _affect = np.argmax(B @ centroides.T, axis=1)
    listes = [np.flatnonzero(_affect == j) for j in range(config["nlist"])]
    duree_ivf_np = time.perf_counter() - _t0
    _tailles = np.array([len(l) for l in listes])
    print(f"IVF numpy : {config['nlist']} cellules en {duree_ivf_np:.1f} s ; "
          f"taille des cellules : min {_tailles.min()}, médiane {int(np.median(_tailles))}, max {_tailles.max()}")

    def ivf_une(q, nprobe):
        cellules = np.argsort(-(centroides @ q))[:nprobe]
        cand = np.concatenate([listes[c] for c in cellules])
        s = B[cand] @ q
        i = np.argpartition(-s, min(K, len(s) - 1))[:K]
        return cand[i[np.argsort(-s[i])]], len(cand)

    ivf_numpy = []
    for _np_ in config["nprobe"]:
        _res = [ivf_une(q, _np_) for q in Q]
        _r = rappel([r[0] for r in _res])
        _frac = np.mean([r[1] for r in _res]) / len(B)
        ivf_numpy.append(dict(nprobe=_np_, rappel=_r, fraction=float(_frac)))
        print(f"  nprobe {_np_:>3} : rappel@10 {100 * _r:5.1f} % en comparant {100 * _frac:5.1f} % de la base")
    return (ivf_numpy,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Quantification de produit (PQ) en numpy : 32 fois moins de mémoire

    Un vecteur de 384 nombres float32 occupe 1 536 octets. On le coupe en 48 morceaux de
    8 dimensions ; pour chaque position de morceau, un k-moyennes apprend 256 « morceaux
    types » ; chaque morceau est remplacé par le numéro (1 octet) de son morceau type le
    plus proche. Un vecteur = **48 octets**.

    Pour chercher, on ne décompresse rien : on calcule une fois par requête la table des
    48 × 256 produits scalaires entre les morceaux de la requête et les morceaux types,
    puis le score approché d'un vecteur de la base est la somme de 48 lectures dans cette
    table (*asymmetric distance computation*, Jégou et al.).
    """)


@app.cell
def _(B, K, Q, config, np, rappel, time, verite):
    M, KS = config["pq_m"], 256
    DS = B.shape[1] // M                       # 8 dimensions par sous-vecteur

    def kmeans_l2(X, k, iters, rng):
        C = X[rng.choice(len(X), k, replace=False)].copy()
        for _ in range(iters):
            d = (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None]
            a = d.argmin(1)
            for j in range(k):
                m = X[a == j]
                if len(m):
                    C[j] = m.mean(0)
        return C

    _rng = np.random.default_rng(42)
    _t0 = time.perf_counter()
    _ech = B[_rng.choice(len(B), min(config["n_train_kmeans"], len(B)), replace=False)]
    livres_codes = np.stack([kmeans_l2(_ech[:, j * DS:(j + 1) * DS], KS, config["iters_kmeans"], _rng)
                             for j in range(M)])          # (48, 256, 8)
    codes = np.empty((len(B), M), np.uint8)
    for _j in range(M):
        _sub = B[:, _j * DS:(_j + 1) * DS]
        _C = livres_codes[_j]
        codes[:, _j] = ((_sub ** 2).sum(1)[:, None] - 2 * _sub @ _C.T + (_C ** 2).sum(1)[None]).argmin(1)
    duree_pq_np = time.perf_counter() - _t0
    print(f"PQ numpy : apprentissage et codage en {duree_pq_np:.1f} s ; "
          f"{codes.nbytes / 1e6:.2f} Mo de codes contre {B.nbytes / 1e6:.1f} Mo (÷{B.nbytes / codes.nbytes:.0f})")

    _cols = np.arange(M)

    def pq_scores(q):
        table = np.einsum("mkd,md->mk", livres_codes, q.reshape(M, DS))   # (48, 256)
        return table[_cols, codes].sum(1)                                  # 48 lectures par vecteur

    _n = config["n_requetes_pq"]
    _trouves, _trouves_rr = [], []
    _t0 = time.perf_counter()
    for _q in Q[:_n]:
        _s = pq_scores(_q)
        _i = np.argpartition(-_s, 100)[:100]          # 100 candidats selon PQ…
        _trouves.append(_i[np.argsort(-_s[_i])][:K])
        _e = B[_i] @ _q                                # …reclassés avec les vrais vecteurs
        _trouves_rr.append(_i[np.argsort(-_e)][:K])
    pq_numpy = dict(rappel=rappel(_trouves, verite[:_n]), rappel_reclasse=rappel(_trouves_rr, verite[:_n]),
                    octets=M, ms=1000 * (time.perf_counter() - _t0) / _n)
    print(f"PQ seul : rappel@10 {100 * pq_numpy['rappel']:.1f} % ; PQ puis reclassement des 100 premiers "
          f"avec les vecteurs exacts : {100 * pq_numpy['rappel_reclasse']:.1f} %")
    return (pq_numpy,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Les mêmes idées dans faiss, plus HNSW

    **HNSW** (*Hierarchical Navigable Small World*) : un graphe où chaque vecteur est relié
    à ~M voisins proches, empilé en couches de plus en plus clairsemées. La recherche part
    de la couche du haut, avance de voisin en voisin vers la requête (marche gloutonne),
    descend d'une couche, recommence. `efSearch` = taille de la liste de candidats gardés
    pendant la marche : plus grande, plus de rappel, plus lent.

    faiss est limité à **un thread** ici (contournement d'un conflit avec torch) ; numpy,
    lui, utilise la bibliothèque d'algèbre linéaire du Mac, multithread.
    """)


@app.cell
def _(B, K, Q, chrono_par_requete, config, exact_une, faiss, lat_exact, np, rappel, time):
    d = B.shape[1]
    IP = faiss.METRIC_INNER_PRODUCT

    def taille(index):
        return int(faiss.serialize_index(index).size)

    def courbe(index, nom, param, valeurs, regler):
        points = []
        for v in valeurs:
            regler(v)
            _, I = index.search(Q, K)
            lat = chrono_par_requete(lambda q: index.search(q[None], K), config["n_latence"])
            points.append(dict(param=v, rappel=rappel(I), ms=lat))
            print(f"  {nom} {param}={v:<4} rappel@10 {100 * points[-1]['rappel']:5.1f} %  {lat:.3f} ms")
        return points

    faiss_res = {}
    # 1. Flat : recherche exacte, vérifie qu'on retrouve la vérité numpy
    _flat = faiss.IndexFlatIP(d)
    _flat.add(B)
    _, _I = _flat.search(Q, K)
    faiss_res["flat"] = dict(rappel=rappel(_I), ms=chrono_par_requete(lambda q: _flat.search(q[None], K),
                                                                      config["n_latence"]),
                             octets=taille(_flat) / len(B))
    print(f"Flat (exacte, faiss) : rappel {100 * faiss_res['flat']['rappel']:.1f} %, "
          f"{faiss_res['flat']['ms']:.2f} ms ; exacte numpy : {lat_exact:.2f} ms")

    # 2. HNSW
    _t0 = time.perf_counter()
    hnsw = faiss.IndexHNSWFlat(d, config["hnsw_M"], IP)
    hnsw.hnsw.efConstruction = config["hnsw_ef_construction"]
    hnsw.add(B)
    _duree = time.perf_counter() - _t0
    _niveaux = faiss.vector_to_array(hnsw.hnsw.levels) - 1
    print(f"HNSW : construit en {_duree:.1f} s ; vecteurs par couche : "
          f"{[int((_niveaux >= c).sum()) for c in range(int(_niveaux.max()) + 1)]}")

    def _ef(v):
        hnsw.hnsw.efSearch = v

    faiss_res["hnsw"] = dict(points=courbe(hnsw, "HNSW", "efSearch", config["ef_search"], _ef),
                             construction_s=_duree, octets=taille(hnsw) / len(B),
                             couches=[int((_niveaux >= c).sum()) for c in range(int(_niveaux.max()) + 1)])

    # 3. IVF (même idée que la version numpy) puis IVF-PQ (IVF + codes PQ de 48 octets)
    for _nom, _fab in (("ivf", lambda qz: faiss.IndexIVFFlat(qz, d, config["nlist"], IP)),
                       ("ivfpq", lambda qz: faiss.IndexIVFPQ(qz, d, config["nlist"], config["pq_m"], 8, IP))):
        _qz = faiss.IndexFlatIP(d)
        _idx = _fab(_qz)
        _t0 = time.perf_counter()
        _idx.train(B)
        _idx.add(B)
        _duree = time.perf_counter() - _t0

        def _np_(v, _i=_idx):
            _i.nprobe = v

        faiss_res[_nom] = dict(points=courbe(_idx, _nom.upper(), "nprobe", config["nprobe"], _np_),
                               construction_s=_duree, octets=taille(_idx) / len(B))
    for _nom, _r in faiss_res.items():
        print(f"{_nom:<6} : {_r['octets']:.0f} octets par vecteur (index entier / N)")
    _ = exact_une, np
    return (faiss_res,)


@app.cell
def _(B, K, Q, chrono_par_requete, config, faiss, faiss_res, np, time, topk):
    # --- Passage à l'échelle : latence en fonction de la taille de la base ---
    _ = faiss_res
    echelle = []
    for _n in config["tailles"] + [len(B)]:
        _Bn = B[:_n]
        _v = topk(Q @ _Bn.T, K)

        def _ex(q, _b=_Bn):
            s = _b @ q
            return np.argpartition(-s, K)[:K]

        _lat_ex = chrono_par_requete(_ex, config["n_latence"])
        _h = faiss.IndexHNSWFlat(B.shape[1], config["hnsw_M"], faiss.METRIC_INNER_PRODUCT)
        _h.hnsw.efConstruction = config["hnsw_ef_construction"]
        _t0 = time.perf_counter()
        _h.add(_Bn)
        _construction = time.perf_counter() - _t0
        _h.hnsw.efSearch = 64
        _, _I = _h.search(Q, K)
        _r = float(np.mean([len(set(a) & set(b)) / K for a, b in zip(_I, _v)]))
        _lat_h = chrono_par_requete(lambda q, _i=_h: _i.search(q[None], K), config["n_latence"])
        echelle.append(dict(n=int(_n), exact_ms=_lat_ex, hnsw_ms=_lat_h, hnsw_rappel=_r,
                            construction_s=_construction))
        print(f"N = {_n:>6} : exacte {_lat_ex:.3f} ms, HNSW (efSearch 64) {_lat_h:.3f} ms, "
              f"rappel {100 * _r:.1f} %, construction {_construction:.1f} s")
    return (echelle,)


@app.cell
def _(
    PALETTE,
    SMOKE,
    echelle,
    faiss_res,
    ivf_numpy,
    lat_exact,
    mo,
    np,
    piaf_res,
    plt,
    sauvegarder_figure,
    sts_cos,
    sts_note,
    sts_rho,
):
    _dossier = mo.notebook_dir() / "figures"
    _suffixe = "-smoke" if SMOKE else ""

    # Figure 1 : cosinus contre note humaine (STS-B fr)
    fig_sts, _axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
    for _ax, _nom, _c in zip(_axes, ["brut [CLS]", "brut moyenne", "SBERT"], PALETTE):
        _ax.scatter(sts_note, sts_cos[_nom], s=5, alpha=0.4, color=_c)
        _ax.set(title=f"{_nom} : Spearman {100 * sts_rho[_nom]:.0f}", xlabel="note humaine (0 à 5)")
    _axes[0].set_ylabel("cosinus")
    fig_sts.tight_layout()
    sauvegarder_figure(fig_sts, "sts" + _suffixe, _dossier)

    # Figure 2 : PIAF
    fig_piaf, _ax = plt.subplots(figsize=(8, 3.8))
    _noms = list(piaf_res)
    _x = np.arange(len(_noms))
    for _k, (_m, _c) in enumerate((("r@1", PALETTE[0]), ("r@5", PALETTE[1]), ("r@10", PALETTE[2]))):
        _ax.bar(_x + (_k - 1) * 0.27, [100 * piaf_res[n][_m] for n in _noms], 0.27, color=_c, label=_m)
    _ax.set_xticks(_x, _noms, rotation=15)
    _ax.set(ylabel="rappel (%)", ylim=(0, 100), title="PIAF : retrouver le passage de chaque question")
    _ax.legend()
    sauvegarder_figure(fig_piaf, "piaf" + _suffixe, _dossier)

    # Figure 3 : rappel / latence, et passage à l'échelle
    fig_ann, (_a1, _a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for _nom, _c, _lib in (("hnsw", PALETTE[0], "HNSW (efSearch)"), ("ivf", PALETTE[1], "IVF (nprobe)"),
                           ("ivfpq", PALETTE[3], "IVF-PQ (nprobe)")):
        _p = faiss_res[_nom]["points"]
        _a1.plot([q["ms"] for q in _p], [100 * q["rappel"] for q in _p], "o-", color=_c, label=_lib)
    _a1.axvline(lat_exact, color="gray", ls="--", label="exacte numpy")
    _a1.set(xscale="log", xlabel="latence par requête (ms, échelle log)", ylabel="rappel@10 (%)",
            title="Rappel contre latence")
    _a1.legend(fontsize=8)
    _n = [e["n"] for e in echelle]
    _a2.plot(_n, [e["exact_ms"] for e in echelle], "o-", color="gray", label="exacte numpy")
    _a2.plot(_n, [e["hnsw_ms"] for e in echelle], "o-", color=PALETTE[0], label="HNSW (efSearch 64)")
    _a2.set(xscale="log", yscale="log", xlabel="taille de la base", ylabel="latence (ms)",
            title="Passage à l'échelle")
    _a2.legend(fontsize=8)
    fig_ann.tight_layout()
    sauvegarder_figure(fig_ann, "recherche-approchee" + _suffixe, _dossier)
    _ = ivf_numpy
    fig_ann


@app.cell
def _(
    B,
    SMOKE,
    cross_par_question,
    bi_par_question,
    duree_encodage,
    echelle,
    exemples_libres,
    faiss_res,
    ivf_numpy,
    json,
    lat_exact,
    mo,
    np,
    piaf_exemple,
    piaf_res,
    pq_numpy,
    sts_cos,
    sts_hasard,
    sts_note,
    sts_rho,
):
    # --- Bilan et seuils du README ---
    _hnsw = faiss_res["hnsw"]["points"]
    _hnsw_95 = [p for p in _hnsw if p["rappel"] >= 0.95]
    bilan = dict(
        sts_rho=sts_rho, sts_hasard=sts_hasard,
        piaf=piaf_res, cross_s=cross_par_question, bi_ms=1000 * bi_par_question,
        n_base=len(B), duree_encodage=duree_encodage, lat_exact_ms=lat_exact,
        ivf_numpy=ivf_numpy, pq_numpy=pq_numpy, faiss=faiss_res, echelle=echelle,
        exemples=exemples_libres, piaf_exemple=piaf_exemple,
    )
    _dossier = mo.notebook_dir() / "figures"
    np.savez_compressed(
        _dossier / ("resultats-smoke.npz" if SMOKE else "resultats.npz"),
        bilan=json.dumps(bilan, ensure_ascii=False),
        sts_note=np.array(sts_note, np.float32),
        **{f"sts_cos_{k.replace(' ', '_').replace('[', '').replace(']', '')}": np.asarray(v, np.float32)
           for k, v in sts_cos.items()},
    )
    seuils = {
        "SBERT Spearman ≥ [CLS] brut + 30 points": sts_rho["SBERT"] >= sts_rho["brut [CLS]"] + 0.30,
        "anisotropie : cos moyen au hasard [CLS] brut ≥ 0,8, SBERT ≤ 0,3":
            sts_hasard["brut [CLS]"] >= 0.8 and sts_hasard["SBERT"] <= 0.3,
        "PIAF : e5 avec préfixes r@5 ≥ 85 % et > e5 sans préfixe":
            piaf_res["e5 (préfixes)"]["r@5"] >= 0.85
            and piaf_res["e5 (préfixes)"]["r@5"] > piaf_res["e5 sans préfixe"]["r@5"],
        "faiss Flat = recherche exacte (rappel ≥ 99,9 %)": faiss_res["flat"]["rappel"] >= 0.999,
        "HNSW : rappel@10 ≥ 95 % au moins 10 fois plus vite que l'exacte":
            any(p["ms"] * 10 <= lat_exact for p in _hnsw_95),
        "PQ : 32 fois moins de mémoire, rappel après reclassement ≥ 90 %":
            pq_numpy["rappel_reclasse"] >= 0.90,
    }
    for _s, _ok in seuils.items():
        print(f"[{'OK' if _ok else 'ÉCHEC'}] {_s}")
    if not SMOKE and not all(seuils.values()):
        raise SystemExit("Seuil non atteint : " + ", ".join(s for s, ok in seuils.items() if not ok))
    mo.md("**Bilan** : " + " ; ".join(f"{s} : {'oui' if ok else 'non'}" for s, ok in seuils.items()))


if __name__ == "__main__":
    app.run()
