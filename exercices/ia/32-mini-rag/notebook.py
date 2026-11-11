# Exercice 32 : un mini-RAG sur le texte du cours (chapitre 32).
#
#   make ex NUM=32              → éditeur marimo (bouton « Lancer »)
#   make ex NUM=32 MODE=run     → script complet, vérifie les seuils du README
#   make ex NUM=32 MODE=smoke   → test rapide (< 30 s)
#
# Prérequis : Ollama lancé en local (application ou `ollama serve`) et le modèle téléchargé
# une fois : `ollama pull qwen3.5:2b` (≈ 3 Go). Les autres modèles (Hugging Face) sont
# téléchargés au premier lancement dans data/hf/.
#
# Progression :
#   1. Découper le corpus (chapitres 00 à 15 du cours, figés dans corpus/) en morceaux.
#   2. BM25 écrit en numpy, vérifié contre la bibliothèque bm25s.
#   3. Recherche dense : embeddings multilingual-e5 + index faiss.
#   4. Recherche hybride : fusion des deux listes par les rangs (RRF).
#   5. Reclassement par cross-encodeur : un modèle anglais contre un modèle multilingue.
#   6. Génération avec un LLM local (Ollama) : sans contexte, avec les passages
#      récupérés et des citations, avec des passages qui ne contiennent pas la réponse.
#   7. « Perdu au milieu » : la même réponse placée en tête, au milieu ou en fin de contexte.
#
# Aucun entraînement : tous les modèles sont figés. faiss : toujours via
# ia_commun.compat.importer_faiss() (sinon plantage à côté de torch).

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import re
    import sys
    import time
    import unicodedata
    from pathlib import Path

    import bm25s
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.sparse as sp
    import torch

    # Contournement (transformers 5.19) : quand marimo est chargé, transformers se croit dans
    # un notebook Jupyter et importe IPython (absent) en chargeant son Trainer, ce que fait
    # sentence-transformers. On masque marimo le temps de cet import, puis on le remet.
    _marimo = sys.modules.pop("marimo", None)
    try:
        from sentence_transformers import CrossEncoder, SentenceTransformer
    finally:
        if _marimo is not None:
            sys.modules["marimo"] = _marimo

    from ia_commun import (
        PALETTE,
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
        CrossEncoder,
        PALETTE,
        Path,
        SentenceTransformer,
        bm25s,
        faiss,
        get_device,
        is_smoke,
        json,
        mo,
        np,
        plt,
        re,
        sauvegarder_figure,
        seed_everything,
        sp,
        style_commun,
        time,
        torch,
        unicodedata,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Exercice 32 : un mini-RAG sur le texte du cours

    **Objectif.** Construire, étape par étape, une chaîne de génération augmentée par
    récupération (RAG) : retrouver dans le cours les passages qui répondent à une
    question, puis les donner à un petit LLM local pour qu'il réponde **en citant ses
    sources**. Mesurer chaque étage séparément.

    **Le corpus** : le texte des chapitres 00 à 15 de ce cours, figé dans `corpus/`
    (≈ 650 000 caractères). **Les questions** : 44 questions écrites à la main
    (`questions.json`), chacune avec le passage du corpus qui contient la réponse ; la
    moitié reprend des mots rares du passage (noms, nombres), l'autre moitié dit la même
    chose avec d'autres mots.

    **Ce qu'on optimise : rien.** Aucun réseau n'est entraîné. Les modèles utilisés ont
    été entraînés par d'autres : l'encodeur e5 de façon contrastive (InfoNCE, chapitre 23),
    les cross-encodeurs comme des classifieurs « pertinent / pas pertinent » (entropie
    croisée, chapitre 01), le LLM à prédire le token suivant (chapitre 20). On mesure :

    - le **rappel@k** : part des questions dont un morceau pertinent (qui contient le
      passage de référence) est parmi les k premiers résultats ;
    - le **MRR@10** : moyenne de $1/\text{rang}$ du premier morceau pertinent (0 s'il
      n'est pas dans les 10 premiers) ;
    - pour le LLM : la part de **réponses justes** (fragments attendus présents), la part
      de **« je ne sais pas »**, et la part de réponses justes dont une **citation** pointe
      vers un passage qui contient vraiment la réponse.
    """)


@app.cell
def _(get_device, is_smoke, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    device = get_device()
    style_commun()
    config = dict(
        chapitres=["04"] if SMOKE else None,  # None = les 16 chapitres du corpus
        n_questions=1 if SMOKE else None,                 # None = les 44 questions
        taille=1000,                       # taille visée d'un morceau (caractères)
        chevauchement=150,                 # recouvrement entre deux morceaux voisins
        tailles_comparees=[1000] if SMOKE else [300, 1000, 3000],  # effet de la taille des morceaux
        k1=1.5, b=0.75,                    # BM25 (valeurs par défaut de bm25s ; Lucene : k1 = 1,2)
        k_rrf=60,                          # RRF : constante de Cormack et al. (2009)
        n_candidats=5 if SMOKE else 10,    # candidats envoyés au reclassement
        # LLM local
        modele="qwen3.5:2b",
        num_ctx=8192,                      # fenêtre fixe : changer num_ctx recharge le modèle
        num_predict=60 if SMOKE else 100,  # réponses courtes (une ou deux phrases)
        k_contexte=3,                      # passages donnés au LLM (r@3 de la meilleure chaîne : 100 %)
        n_milieu=1 if SMOKE else 10,       # « perdu au milieu » : nombre de questions
        n_passages_milieu=2 if SMOKE else 8,   # passages par prompt dans cette expérience
    )
    # Révisions figées des modèles du Hub (reproductibilité ; licences vérifiées sur le Hub)
    REV = dict(
        e5=("intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3"),  # MIT
        ce_en=("cross-encoder/ms-marco-MiniLM-L6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a"),  # Apache-2.0
        ce_m3=("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"),  # Apache-2.0
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, périphérique {device}")
    return REV, SEED, SMOKE, config, device


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Le corpus, les questions et le découpage

    Un LLM ne lit pas un livre entier à chaque question, et un moteur de recherche ne
    renvoie pas un chapitre de 50 000 caractères : on **découpe** (*chunking*) le corpus en
    morceaux de quelques centaines de mots, qui seront les unités de recherche.

    Deux découpages :

    - **naïf** : des fenêtres de 1 000 caractères coupées n'importe où (au milieu d'un
      mot, d'un tableau, d'une phrase) ;
    - **par sections** : on coupe d'abord aux titres (`##`, `###`…), puis on remplit des
      morceaux d'environ 1 000 caractères phrase par phrase, avec 150 caractères de
      recouvrement ; chaque morceau commence par le **chemin de ses titres**
      (« 07. Rétropropagation… › Le mécanisme › 3. … ») : un morceau isolé garde ainsi
      son contexte.
    """)


@app.cell
def _(Path, config, json, mo, re):
    DOSSIER = mo.notebook_dir()

    def espaces(t: str) -> str:
        """Espaces insécables (fines ou non) → espace simple : le corpus en contient
        des milliers (« 8 192 », « : »), les questions et les LLM n'en mettent pas."""
        # Codes écrits en chiffres : un caractère insécable littéral ne survit pas à tous
        # les éditeurs (marimo le change en espace normale).
        return re.sub("[" + chr(0xA0) + chr(0x202F) + chr(0x2009) + "]", " ", t)

    documents = {}
    for _f in sorted(Path(DOSSIER / "corpus").glob("*.md")):
        if config["chapitres"] is None or _f.name[:2] in config["chapitres"]:
            documents[_f.name[:2]] = espaces(_f.read_text(encoding="utf-8"))
    _q = json.loads((DOSSIER / "questions.json").read_text(encoding="utf-8"))["questions"]
    questions = [dict(q, passage=espaces(q["passage"])) for q in _q
                 if q["chapitre"] in documents]
    if config["n_questions"]:
        questions = questions[: config["n_questions"]]
    # Chaque passage de référence doit exister, une seule fois, dans son chapitre
    for _q in questions:
        _n = documents[_q["chapitre"]].count(_q["passage"])
        if _n != 1:
            raise SystemExit(f"{_q['id']} : passage trouvé {_n} fois dans le chapitre "
                             f"{_q['chapitre']} (corpus régénéré ?) : {_q['passage']!r}")
    _types = {t: sum(q["type"] == t for q in questions) for t in ("lexicale", "reformulee")}
    print(f"{len(documents)} chapitres, {sum(map(len, documents.values())):,} caractères ; "
          f"{len(questions)} questions {_types}".replace(",", " "))
    return DOSSIER, documents, espaces, questions


@app.cell
def _(re):
    def decouper_naif(documents: dict, taille: int) -> list[dict]:
        """Fenêtres fixes de `taille` caractères, sans recouvrement ni respect du texte."""
        morceaux = []
        for ch, texte in documents.items():
            for i in range(0, len(texte), taille):
                morceaux.append({"chapitre": ch, "titre": "", "texte": texte[i:i + taille]})
        return morceaux

    def unites(texte: str) -> list[str]:
        """Phrases (et lignes de tableau ou de code) : les briques qu'on ne coupe pas."""
        sortie = []
        for bloc in re.split(r"\n\s*\n", texte):
            if bloc.lstrip().startswith(("|", "```", "-", ">")) or "\n|" in bloc:
                sortie.extend(ligne for ligne in bloc.split("\n") if ligne.strip())
            else:
                sortie.extend(re.split(r"(?<=[.!?…])\s+(?=[A-ZÀ-ÖØ-Þ«(])", bloc.strip()))
            sortie.append("\n")  # fin de paragraphe
        return [u for u in sortie if u.strip() or u == "\n"]

    def sections(texte: str) -> list[str]:
        """Coupe avant chaque titre Markdown, sauf dans les blocs de code (où « # » est un
        commentaire Python)."""
        morceaux, courant, dans_code = [], [], False
        for ligne in texte.split("\n"):
            if ligne.startswith("```"):
                dans_code = not dans_code
            if not dans_code and re.match(r"#{1,4} ", ligne) and courant:
                morceaux.append("\n".join(courant) + "\n")
                courant = []
            courant.append(ligne)
        morceaux.append("\n".join(courant) + "\n")
        return morceaux

    def decouper_sections(documents: dict, taille: int, chevauchement: int) -> list[dict]:
        """Coupe aux titres, puis remplit des morceaux d'environ `taille` caractères phrase
        par phrase ; le morceau suivant reprend les dernières phrases (≤ chevauchement)."""
        morceaux = []
        for ch, texte in documents.items():
            chemin = []  # titres en cours : [(niveau, titre), …]
            for section in sections(texte):
                m = re.match(r"(#{1,4}) (.*)\n", section)
                if m:
                    niveau = len(m.group(1))
                    chemin = [c for c in chemin if c[0] < niveau] + [(niveau, m.group(2).strip())]
                    section = section[m.end():]
                titre = " › ".join(t for _, t in chemin)
                courant, longueur = [], 0
                for u in unites(section):
                    if longueur + len(u) > taille and longueur > 0:
                        morceaux.append({"chapitre": ch, "titre": titre,
                                         "texte": " ".join(courant).replace(" \n ", "\n")})
                        garde, lg = [], 0
                        for v in reversed(courant):  # recouvrement : on reprend la fin
                            if lg + len(v) > chevauchement:
                                break
                            garde.insert(0, v)
                            lg += len(v)
                        courant, longueur = garde, lg
                    courant.append(u)
                    longueur += len(u)
                if any(u.strip() for u in courant):
                    morceaux.append({"chapitre": ch, "titre": titre,
                                     "texte": " ".join(courant).replace(" \n ", "\n")})
        for m in morceaux:
            m["texte"] = re.sub(r"\n{2,}", "\n", m["texte"]).strip()
            # Texte indexé : chemin des titres + contenu
            m["complet"] = (m["titre"] + "\n" + m["texte"]) if m["titre"] else m["texte"]
        return [m for m in morceaux if len(m["texte"]) > 20]

    return decouper_naif, decouper_sections


@app.cell
def _(config, decouper_naif, decouper_sections, documents, np, questions):
    def pertinents(morceaux: list[dict], q: dict) -> set[int]:
        """Indices des morceaux qui contiennent le passage de référence (ou, s'il est
        coupé par une frontière de morceau, ses 25 premiers ou 25 derniers caractères)."""
        p = q["passage"]
        cles = [p] if len(p) <= 30 else [p, p[:25], p[-25:]]
        return {i for i, m in enumerate(morceaux)
                if m["chapitre"] == q["chapitre"] and any(c in m["texte"] for c in cles)}

    decoupages = {"naïf 1000": decouper_naif(documents, config["taille"])}
    for _n in config["tailles_comparees"]:
        decoupages[f"sections {_n}"] = decouper_sections(documents, _n, config["chevauchement"]
                                                         if _n >= 1000 else _n // 6)
    for _m in decoupages["naïf 1000"]:
        _m["complet"] = _m["texte"]
    morceaux = decoupages[f"sections {config['taille']}"]   # découpage principal
    for _nom, _ms in decoupages.items():
        _l = np.array([len(m["complet"]) for m in _ms])
        _sans = sum(not pertinents(_ms, q) for q in questions)
        print(f"{_nom:>14} : {len(_ms):5d} morceaux, longueur médiane {int(np.median(_l))} "
              f"caractères (max {_l.max()}) ; questions sans morceau pertinent : {_sans}")
    print("\nExemple de morceau (découpage principal) :\n" + "-" * 60)
    print(morceaux[len(morceaux) // 3]["complet"][:700])
    return decoupages, morceaux, pertinents


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. BM25, écrit en numpy

    BM25 (Robertson et al., années 1990) note un morceau $d$ pour une requête $q$ en
    additionnant, pour chaque mot $t$ de la requête, un poids qui croît avec sa fréquence
    $f(t, d)$ dans le morceau, mais **sature**, et qui est d'autant plus fort que le mot est
    **rare** dans le corpus (idf) :

    $$
    \text{BM25}(q, d) = \sum_{t \in q} \text{idf}(t)\,
    \frac{f(t,d)\,(k_1 + 1)}{f(t,d) + k_1\left(1 - b + b\,\frac{|d|}{\overline{|d|}}\right)},
    \qquad \text{idf}(t) = \ln\!\left(1 + \frac{N - n_t + 0{,}5}{n_t + 0{,}5}\right)
    $$

    avec $N$ morceaux, $n_t$ morceaux qui contiennent $t$, $|d|$ la longueur du morceau en
    mots et $\overline{|d|}$ la longueur moyenne. $k_1 = 1{,}5$ règle la saturation, $b =
    0{,}75$ la pénalité des morceaux longs. C'est la variante dite « Lucene » de `bm25s`
    ($k_1 = 1{,}5$ est sa valeur par défaut ; Lucene utilise $k_1 = 1{,}2$), qui omet le facteur
    constant $(k_1 + 1)$ : tous les scores sont divisés par 2,5, aucun classement ne change.

    Tokenisation volontairement simple : minuscules, accents retirés, nombres recollés
    (« 8 192 » → « 8192 »), mots vides du français retirés ; pas de racinisation, donc
    « optimiseur » et « optimiseurs » sont deux mots différents.
    """)


@app.cell
def _(re, unicodedata):
    _mots = (
        "a au aux avec ce ces cet cette dans de des du elle en et eux il ils je la le les leur "
        "leurs lui ma mais me meme mes moi mon ne nos notre nous on ou par pas pour qu que qui "
        "sa se ses son sur ta te tes toi ton tu un une vos votre vous c d j l m n s t y est "
        "sont ete etre avoir ont plus quel quelle quels quelles comment combien pourquoi quoi "
        "dont si ni lequel laquelle cela ca tout tous toute toutes fait faire"
    )
    MOTS_VIDES = set(_mots.split())

    def sans_accents(t: str) -> str:
        return "".join(c for c in unicodedata.normalize("NFD", t)
                       if unicodedata.category(c) != "Mn")

    def tokeniser(texte: str) -> list[str]:
        t = sans_accents(texte.lower())
        t = re.sub(r"(?<=\d)[ ,.](?=\d{3}\b)", "", t)    # 8 192 → 8192 ; 1,000 → 1000
        return [m for m in re.findall(r"[a-z0-9]+", t) if m not in MOTS_VIDES]

    assert tokeniser("Combien de poids : 269 322 ?") == ["poids", "269322"]
    assert tokeniser("L’échauffement d'Adam") == ["echauffement", "adam"]
    return sans_accents, tokeniser


@app.cell
def _(np, sp, tokeniser):
    class BM25Numpy:
        """BM25 (variante Lucene) : toute la notation est précalculée dans une matrice creuse
        morceaux × vocabulaire ; noter une requête = additionner quelques colonnes."""

        def __init__(self, textes: list[str], k1: float = 1.5, b: float = 0.75):
            docs = [tokeniser(t) for t in textes]
            self.vocab = {}
            lignes, colonnes, valeurs = [], [], []
            for i, doc in enumerate(docs):
                if not doc:
                    continue
                ids, comptes = np.unique([self.vocab.setdefault(m, len(self.vocab)) for m in doc],
                                         return_counts=True)
                lignes += [i] * len(ids)
                colonnes += ids.tolist()
                valeurs += comptes.tolist()
            N, V = len(docs), len(self.vocab)
            tf = sp.csr_matrix((np.array(valeurs, dtype=np.float32), (lignes, colonnes)), shape=(N, V))
            longueur = np.array([len(d) for d in docs], dtype=np.float32)
            n_t = np.bincount(colonnes, minlength=V)                     # morceaux contenant t
            self.idf = np.log(1 + (N - n_t + 0.5) / (n_t + 0.5)).astype(np.float32)
            # Dénominateur par entrée non nulle : f + k1 (1 − b + b |d| / moyenne)
            norme = k1 * (1 - b + b * longueur / longueur.mean())
            tf = tf.tocoo()
            # Lucene (et bm25s) omettent le facteur constant (k1 + 1) de la formule d'origine :
            # il multiplie tous les scores par 2,5 sans changer aucun classement.
            poids = tf.data / (tf.data + norme[tf.row]) * self.idf[tf.col]
            self.W = sp.csc_matrix((poids, (tf.row, tf.col)), shape=(N, V))  # colonnes rapides
            self.N, self.V = N, V

        def scores(self, requete: str) -> np.ndarray:
            ids = sorted({self.vocab[m] for m in tokeniser(requete) if m in self.vocab})
            if not ids:
                return np.zeros(self.N, dtype=np.float32)
            return np.asarray(self.W[:, ids].sum(axis=1)).ravel()

    return (BM25Numpy,)


@app.cell
def _(BM25Numpy, bm25s, config, morceaux, np, questions, time, tokeniser):
    _t0 = time.perf_counter()
    bm25 = BM25Numpy([m["complet"] for m in morceaux], k1=config["k1"], b=config["b"])
    duree_index_bm25 = time.perf_counter() - _t0
    scores_bm25 = np.stack([bm25.scores(q["question"]) for q in questions])

    # Même calcul avec la bibliothèque bm25s (mêmes tokens) : les scores doivent coïncider
    _bib = bm25s.BM25(method="lucene", k1=config["k1"], b=config["b"])
    _bib.index([tokeniser(m["complet"]) for m in morceaux], show_progress=False)
    _sc_bib = np.stack([_bib.get_scores(sorted(set(tokeniser(q["question"])))) for q in questions])
    ecart_bm25s = float(np.abs(_sc_bib - scores_bm25).max())
    _top_np = np.argsort(-scores_bm25, axis=1)[:, :10]
    _top_bib = np.argsort(-_sc_bib, axis=1)[:, :10]
    accord_top10_bm25s = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(_top_np, _top_bib)]))
    print(f"Index BM25 numpy : {bm25.N} morceaux × {bm25.V} mots, construit en "
          f"{duree_index_bm25 * 1000:.0f} ms")
    print(f"Écart maximal avec bm25s : {ecart_bm25s:.2e} ; accord sur les 10 premiers : "
          f"{accord_top10_bm25s:.1%}")
    _i = int(np.argmax(scores_bm25[0]))
    print(f"\n« {questions[0]['question']} »\n→ meilleur morceau (score {scores_bm25[0, _i]:.2f}) : "
          f"{morceaux[_i]['complet'][:200]!r}")
    return accord_top10_bm25s, bm25, duree_index_bm25, ecart_bm25s, scores_bm25


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Recherche dense : embeddings et faiss

    On encode chaque morceau et chaque question avec `multilingual-e5-small` (384
    dimensions, le modèle de l'exercice 22), avec les préfixes demandés par sa carte
    (`passage: ` et `query: `), puis on cherche les plus grands produits scalaires entre
    vecteurs normalisés (cosinus) dans un index faiss. Sur quelques centaines de vecteurs,
    la recherche exacte (`IndexFlatIP`) prend une fraction de milliseconde : un index
    approché (HNSW, IVF, chapitre 22) ne servirait à rien ici.
    """)


@app.cell
def _(REV, SentenceTransformer, device, faiss, morceaux, np, questions, time):
    e5 = SentenceTransformer(REV["e5"][0], revision=REV["e5"][1], device=str(device))

    def encoder(textes: list[str], prefixe: str) -> np.ndarray:
        X = e5.encode([prefixe + t for t in textes], batch_size=32, convert_to_numpy=True,
                      normalize_embeddings=True, show_progress_bar=False)
        return X.astype(np.float32)

    def index_dense(ms: list[dict]):
        index = faiss.IndexFlatIP(e5.get_embedding_dimension())
        index.add(encoder([m["complet"] for m in ms], "passage: "))
        return index

    _t0 = time.perf_counter()
    index_e5 = index_dense(morceaux)
    duree_encodage = time.perf_counter() - _t0
    Q_e5 = encoder([q["question"] for q in questions], "query: ")
    _t0 = time.perf_counter()
    _sim, _idx = index_e5.search(Q_e5, index_e5.ntotal)       # tout le classement
    duree_recherche_dense = (time.perf_counter() - _t0) / len(questions)
    scores_dense = np.zeros((len(questions), index_e5.ntotal), dtype=np.float32)
    np.put_along_axis(scores_dense, _idx, _sim, axis=1)
    print(f"{index_e5.ntotal} morceaux encodés en {duree_encodage:.1f} s ; recherche exacte "
          f"faiss : {duree_recherche_dense * 1000:.2f} ms par question")
    return Q_e5, duree_encodage, encoder, index_dense, scores_dense


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Recherche hybride : la fusion par les rangs (RRF)

    BM25 et e5 donnent des scores sans commune mesure (de 0 à 40 pour l'un, de 0,7 à 0,9
    pour l'autre). Plutôt que de les normaliser, la **fusion réciproque des rangs**
    (*reciprocal rank fusion*, Cormack, Clarke et Büttcher, 2009) n'utilise que les rangs :

    $$
    \text{RRF}(d) = \sum_{\text{listes } L} \frac{1}{k + \text{rang}_L(d)}, \qquad k = 60
    $$

    Exemple : un morceau 1er pour BM25 et 8e pour e5 obtient $1/61 + 1/68 = 0{,}0311$ ; un
    morceau 3e dans les deux listes, $2/63 = 0{,}0317$ : être bien placé partout bat être
    premier d'un côté seulement. Le $k = 60$ aplatit les écarts entre les premiers rangs.
    """)


@app.cell
def _(config, np):
    def rangs(scores: np.ndarray) -> np.ndarray:
        """Rang (1 = meilleur) de chaque morceau, pour chaque question."""
        ordre = np.argsort(-scores, axis=1, kind="stable")
        r = np.empty_like(ordre)
        np.put_along_axis(r, ordre, np.arange(1, scores.shape[1] + 1)[None, :], axis=1)
        return r

    def rrf(*listes_scores: np.ndarray, k: int = config["k_rrf"]) -> np.ndarray:
        return sum(1.0 / (k + rangs(s)) for s in listes_scores)

    def minmax(s: np.ndarray) -> np.ndarray:
        lo, hi = s.min(axis=1, keepdims=True), s.max(axis=1, keepdims=True)
        return (s - lo) / np.maximum(hi - lo, 1e-9)

    assert abs(1 / 61 + 1 / 68 - 0.0311) < 1e-4 and abs(2 / 63 - 0.0317) < 1e-4
    return minmax, rrf


@app.cell
def _(morceaux, np, pertinents, questions):
    PERT = [pertinents(morceaux, q) for q in questions]

    def evaluer(scores: np.ndarray, pert=PERT, ks=(1, 3, 5, 10)) -> dict:
        """Rappel@k et MRR@10 à partir d'une matrice de scores questions × morceaux."""
        ordre = np.argsort(-scores, axis=1, kind="stable")
        premier = []
        for i, p in enumerate(pert):
            pos = np.flatnonzero(np.isin(ordre[i, :100], list(p)))
            premier.append(pos[0] + 1 if len(pos) else 10**6)
        premier = np.array(premier)
        res = {f"r@{k}": float(np.mean(premier <= k)) for k in ks}
        res["mrr@10"] = float(np.mean(np.where(premier <= 10, 1.0 / premier, 0.0)))
        res["premier"] = premier
        return res

    return PERT, evaluer


@app.cell
def _(evaluer, minmax, questions, rrf, scores_bm25, scores_dense):
    scores_hybride = rrf(scores_bm25, scores_dense)
    scores_moyenne = 0.5 * minmax(scores_bm25) + 0.5 * minmax(scores_dense)
    matrices = {"BM25": scores_bm25, "e5 (dense)": scores_dense,
                "moyenne des scores": scores_moyenne, "hybride RRF": scores_hybride}
    resultats = {
        "BM25": evaluer(scores_bm25),
        "e5 (dense)": evaluer(scores_dense),
        "moyenne des scores": evaluer(scores_moyenne),
        "hybride RRF": evaluer(scores_hybride),
    }
    TYPES = [q["type"] for q in questions]

    def afficher(res: dict) -> None:
        print(f"{'méthode':>28} | r@1    r@3    r@5    r@10  | MRR@10 | r@5 lexicales / reformulées")
        for nom, r in res.items():
            _lex = [p <= 5 for p, t in zip(r["premier"], TYPES) if t == "lexicale"]
            _ref = [p <= 5 for p, t in zip(r["premier"], TYPES) if t == "reformulee"]
            _l = f"{sum(_lex) / len(_lex):.0%}" if _lex else "–"
            _r = f"{sum(_ref) / len(_ref):.0%}" if _ref else "–"
            print(f"{nom:>28} | {r['r@1']:.1%} {r['r@3']:.1%} {r['r@5']:.1%} {r['r@10']:.1%} "
                  f"| {r['mrr@10']:.3f}  | {_l} / {_r}")

    afficher(resultats)
    return TYPES, afficher, matrices, resultats, scores_hybride


@app.cell
def _(mo):
    mo.md(r"""
    ### La taille des morceaux

    Même chaîne (BM25 + e5 + RRF), quatre découpages. Des morceaux trop petits perdent
    le contexte (la question parle de « l'exercice du chapitre 07 », le morceau ne dit plus
    de quel chapitre il s'agit) ; des morceaux trop grands diluent le passage utile et
    coûtent cher dans le prompt du LLM.
    """)


@app.cell
def _(
    Q_e5,
    bm25s,
    config,
    decoupages,
    evaluer,
    index_dense,
    morceaux,
    np,
    pertinents,
    questions,
    rrf,
    scores_bm25,
    scores_dense,
    tokeniser,
):
    resultats_tailles = {}
    for _nom, _ms in decoupages.items():
        if _ms is morceaux:  # découpage principal : déjà calculé plus haut
            _sb, _sd = scores_bm25, scores_dense
        else:
            _bib = bm25s.BM25(method="lucene", k1=config["k1"], b=config["b"])
            _bib.index([tokeniser(m["complet"]) for m in _ms], show_progress=False)
            _sb = np.stack([_bib.get_scores(sorted(set(tokeniser(q["question"])))) for q in questions])
            _idx = index_dense(_ms)
            _sim, _ids = _idx.search(Q_e5, _idx.ntotal)
            _sd = np.zeros((len(questions), _idx.ntotal), dtype=np.float32)
            np.put_along_axis(_sd, _ids, _sim, axis=1)
        _pert = [pertinents(_ms, q) for q in questions]
        _r = {"BM25": evaluer(_sb, _pert), "e5": evaluer(_sd, _pert),
              "RRF": evaluer(rrf(_sb, _sd), _pert)}
        _long = int(np.median([len(m["complet"]) for m in _ms]))
        resultats_tailles[_nom] = {"n": len(_ms), "longueur": _long,
                                   **{f"{k} r@5": v["r@5"] for k, v in _r.items()},
                                   "RRF r@1": _r["RRF"]["r@1"]}
        print(f"{_nom:>14} ({len(_ms):5d} morceaux, médiane {_long:5d} car.) : "
              f"r@5 BM25 {_r['BM25']['r@5']:.1%}, e5 {_r['e5']['r@5']:.1%}, "
              f"RRF {_r['RRF']['r@5']:.1%} ; RRF r@1 {_r['RRF']['r@1']:.1%}")
    return (resultats_tailles,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Reclassement par cross-encodeur

    Les deux recherches précédentes sont des **bi-encodeurs** ou des index de mots : la
    question et le morceau ne se « voient » jamais. Un **cross-encodeur** lit la paire
    « question + morceau » d'un bloc et sort un score de pertinence : bien plus précis,
    mais il faut un passage dans le réseau par paire (chapitre 22). On ne l'applique donc
    qu'aux 10 premiers candidats de la recherche hybride (le rappel@10 de
    la recherche hybride est déjà de 100 % : la bonne réponse est presque toujours dedans).

    Deux modèles :

    - `cross-encoder/ms-marco-MiniLM-L6-v2` : 6 couches, 22 M de paramètres, entraîné
      sur MS MARCO, en **anglais** seulement ;
    - `BAAI/bge-reranker-v2-m3` : 568 M de paramètres, **multilingue** (base XLM-RoBERTa).
    """)


@app.cell
def _(CrossEncoder, REV, SMOKE, config, device, morceaux, np, questions, scores_hybride, time):
    def reclasser(ce, n_candidats: int, base: np.ndarray):
        """Scores du cross-encodeur sur les n premiers candidats ; les autres morceaux
        gardent leur ordre, derrière."""
        nouveaux = base - base.max() - 1.0      # tous les non-candidats passent derrière
        ordre = np.argsort(-base, axis=1)[:, :n_candidats]
        t0 = time.perf_counter()
        for i, q in enumerate(questions):
            paires = [(q["question"], morceaux[j]["complet"]) for j in ordre[i]]
            s = ce.predict(paires, batch_size=16, show_progress_bar=False)
            nouveaux[i, ordre[i]] = 100.0 + np.asarray(s, dtype=np.float32)
        return nouveaux, (time.perf_counter() - t0) / len(questions)

    scores_rerank, temps_rerank = {}, {}
    for _cle, _nom in [("ce_en", "+ reclassement MiniLM (anglais)"),
                       ("ce_m3", "+ reclassement bge-m3 (multilingue)")]:
        # En smoke, le petit modèle remplace bge-m3 (2,3 Go à charger) : même code, plus court
        _cle = "ce_en" if SMOKE else _cle
        _ce = CrossEncoder(REV[_cle][0], revision=REV[_cle][1], device=str(device), max_length=512)
        scores_rerank[_nom], temps_rerank[_nom] = reclasser(_ce, config["n_candidats"],
                                                            scores_hybride)
        del _ce
        print(f"{_nom} : {temps_rerank[_nom] * 1000:.0f} ms par question "
              f"({config['n_candidats']} paires)")
    return scores_rerank, temps_rerank


@app.cell
def _(afficher, evaluer, resultats, scores_rerank):
    for _nom, _s in scores_rerank.items():
        resultats[_nom] = evaluer(_s)
    afficher(resultats)
    meilleur = max(resultats, key=lambda k: (resultats[k]["r@1"], resultats[k]["mrr@10"]))
    print(f"\nMeilleure chaîne (r@1) : {meilleur}")
    return (meilleur,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Génération : le LLM local, avec et sans passages

    Trois conditions, même question, même modèle (`qwen3.5:2b` dans Ollama, température
    0, graine 42, mode « réflexion » désactivé) :

    1. **sans contexte** (*closed-book*) : le modèle ne dispose que de ce qu'il a appris ;
    2. **RAG** : les 3 premiers morceaux de la meilleure chaîne, numérotés [1] à [3], avec
       la consigne de répondre seulement à partir d'eux et de citer ;
    3. **RAG sans la réponse** : les 3 meilleurs morceaux **qui ne contiennent pas** le
       passage de référence. Le bon comportement est « je ne sais pas ».

    Une réponse est **juste** si elle contient tous les fragments attendus de
    `questions.json` (après mise en minuscules, sans accents ni espaces). C'est un critère
    strict et simple ; il compte faux une réponse juste mais formulée autrement (« la
    moitié » au lieu de « 50 % » quand seule la seconde forme est prévue). La **fidélité**,
    au sens de RAGAS (Es et al., 2023), demanderait un LLM juge : on la remplace par une
    mesure vérifiable, la part de réponses justes dont au moins une citation [n] désigne un
    passage qui contient la réponse.
    """)


@app.cell
def _(config, mo):
    # --- Le LLM local : vérifier qu'Ollama répond et que le modèle est là ---
    import ollama

    client = ollama.Client()
    try:
        _presents = {m.model for m in client.list().models}
    except Exception as erreur:  # serveur absent : message clair plutôt qu'une trace
        raise SystemExit(
            "Ollama ne répond pas sur http://127.0.0.1:11434. Lancez l'application Ollama "
            f"(ou `ollama serve`), puis relancez. Détail : {erreur}"
        ) from erreur
    if config["modele"] not in _presents:
        raise SystemExit(f"Modèle {config['modele']} absent : `ollama pull {config['modele']}` "
                         f"(≈ 3 Go), puis relancez. Modèles présents : {sorted(_presents)}")
    mo.md(f"Ollama répond ; modèle `{config['modele']}` présent.")
    return (client,)


@app.cell
def _(SEED, client, config, re, sans_accents):
    CONSIGNE_SEUL = ("Tu réponds en français, en une ou deux phrases courtes. "
                     "Si tu ne connais pas la réponse, réponds exactement : Je ne sais pas.")
    CONSIGNE_RAG = ("Tu réponds en français, en une ou deux phrases courtes, UNIQUEMENT à partir "
                    "des passages fournis. Après l'information utilisée, cite le numéro du passage "
                    "entre crochets, par exemple [2]. Si les passages ne contiennent pas la réponse, "
                    "réponds exactement : Je ne sais pas.")

    def prompt_rag(question: str, passages: list[str]) -> str:
        blocs = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
        return f"Passages :\n\n{blocs}\n\nQuestion : {question}"

    def generer(consigne: str, contenu: str) -> dict:
        r = client.chat(
            model=config["modele"],
            messages=[{"role": "system", "content": consigne}, {"role": "user", "content": contenu}],
            think=False,
            options={"temperature": 0, "seed": SEED, "num_predict": config["num_predict"],
                     "num_ctx": config["num_ctx"]},
        )
        return {"texte": r.message.content.strip(), "tokens_prompt": r.prompt_eval_count or 0,
                "secondes": (r.total_duration or 0) / 1e9}

    def normaliser(t: str) -> str:
        t = sans_accents(t.lower())
        t = t.translate(str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹−–", "0123456789--"))
        return re.sub(r"[\s^{}\\$*]", "", t)  # \s couvre aussi les espaces insécables

    def juste(reponse: str, attendu: list[str]) -> bool:
        r = normaliser(reponse)
        return all(any(normaliser(v) in r for v in frag.split("|")) for frag in attendu)

    def ne_sait_pas(reponse: str) -> bool:
        return "nesaispas" in normaliser(reponse)

    def citations(reponse: str) -> set[int]:
        return {int(n) for n in re.findall(r"\[(\d{1,2})\]", reponse)}

    assert juste("Il a 269 322 poids.", ["269322"]) and not juste("Il a 269 poids.", ["269322"])
    assert juste("Environ 10^{120} variantes", ["10120"]) and juste("10¹²⁰", ["10120"])
    assert juste("La perte charnière (hinge).", ["charniere|hinge"])
    assert ne_sait_pas("Je ne sais pas.") and citations("Oui [2] et [4].") == {2, 4}
    return CONSIGNE_RAG, CONSIGNE_SEUL, citations, generer, juste, ne_sait_pas, prompt_rag


@app.cell
def _(mo):
    bouton = mo.ui.run_button(label="Lancer les générations")
    bouton
    return (bouton,)


@app.cell
def _(
    CONSIGNE_RAG,
    CONSIGNE_SEUL,
    PERT,
    SMOKE,
    bouton,
    citations,
    config,
    generer,
    juste,
    meilleur,
    mo,
    morceaux,
    ne_sait_pas,
    np,
    prompt_rag,
    questions,
    matrices,
    resultats,
    scores_rerank,
    time,
):
    # En script (make ex MODE=run|smoke), il n'y a personne pour cliquer : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer les générations ».*"))

    _ordre = np.argsort(-{**matrices, **scores_rerank}[meilleur], axis=1, kind="stable")
    K = config["k_contexte"]
    generations = []
    _t0 = time.perf_counter()
    for _i, _q in enumerate(questions):
        _top = list(_ordre[_i, :K])
        _faux = [j for j in _ordre[_i] if j not in PERT[_i]][:K]   # K meilleurs sans la réponse
        _seul = generer(CONSIGNE_SEUL, _q["question"])
        _rag = generer(CONSIGNE_RAG, prompt_rag(_q["question"], [morceaux[j]["complet"] for j in _top]))
        # En smoke, on saute la condition « sans la réponse » pour tenir sous 30 s
        _sans = ({"texte": "", "tokens_prompt": 0, "secondes": 0.0} if SMOKE else
                 generer(CONSIGNE_RAG, prompt_rag(_q["question"], [morceaux[j]["complet"] for j in _faux])))
        _cites = citations(_rag["texte"])
        _cites_justes = {c for c in _cites if 1 <= c <= K and _top[c - 1] in PERT[_i]}
        generations.append({
            "id": _q["id"], "type": _q["type"], "question": _q["question"],
            "pertinent_dans_contexte": bool(set(_top) & PERT[_i]),
            "seul": _seul["texte"], "seul_juste": juste(_seul["texte"], _q["attendu"]),
            "seul_nsp": ne_sait_pas(_seul["texte"]),
            "rag": _rag["texte"], "rag_juste": juste(_rag["texte"], _q["attendu"]),
            "rag_nsp": ne_sait_pas(_rag["texte"]),
            "rag_cite": bool(_cites), "rag_citation_juste": bool(_cites_justes),
            "sans": _sans["texte"], "sans_juste": juste(_sans["texte"], _q["attendu"]),
            "sans_nsp": ne_sait_pas(_sans["texte"]),
            "tokens_prompt_rag": _rag["tokens_prompt"], "secondes_rag": _rag["secondes"],
            "secondes_seul": _seul["secondes"],
        })
        print(f"{_q['id']} [{'V' if generations[-1]['seul_juste'] else '·'}"
              f"{'V' if generations[-1]['rag_juste'] else '·'}"
              f"{'V' if generations[-1]['sans_juste'] else '·'}] {_rag['texte'][:110]!r}")
    duree_generation = time.perf_counter() - _t0

    def _part(cle):
        return float(np.mean([g[cle] for g in generations]))

    bilan_llm = {c: _part(c) for c in ["seul_juste", "seul_nsp", "rag_juste", "rag_nsp", "rag_cite",
                                       "sans_juste", "sans_nsp", "pertinent_dans_contexte"]}
    _justes = [g for g in generations if g["rag_juste"]]
    bilan_llm["citation_juste_si_juste"] = (float(np.mean([g["rag_citation_juste"] for g in _justes]))
                                            if _justes else 0.0)
    bilan_llm["tokens_prompt_rag"] = float(np.mean([g["tokens_prompt_rag"] for g in generations]))
    bilan_llm["secondes_rag"] = float(np.mean([g["secondes_rag"] for g in generations]))
    bilan_llm["secondes_seul"] = float(np.mean([g["secondes_seul"] for g in generations]))
    print(f"\n{(2 if SMOKE else 3) * len(questions)} générations en {duree_generation:.0f} s")
    print(f"Sans contexte : {bilan_llm['seul_juste']:.0%} justes, {bilan_llm['seul_nsp']:.0%} « je ne sais pas »")
    print(f"RAG (top {K}, {resultats[meilleur]['r@5']:.0%} des contextes contiennent la réponse) : "
          f"{bilan_llm['rag_juste']:.0%} justes, {bilan_llm['rag_nsp']:.0%} « je ne sais pas », "
          f"{bilan_llm['rag_cite']:.0%} avec citation ; citation correcte pour "
          f"{bilan_llm['citation_juste_si_juste']:.0%} des réponses justes")
    print(f"RAG sans la réponse : {bilan_llm['sans_juste']:.0%} justes (hasard ou mémoire), "
          f"{bilan_llm['sans_nsp']:.0%} « je ne sais pas »")
    print(f"Coût moyen : {bilan_llm['tokens_prompt_rag']:.0f} tokens de prompt et "
          f"{bilan_llm['secondes_rag']:.1f} s par réponse RAG, {bilan_llm['secondes_seul']:.1f} s sans contexte")
    return K, bilan_llm, duree_generation, generations


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. « Perdu au milieu » : où placer le bon passage ?

    Liu et al. (2023) ont montré que des LLM retrouvent moins bien une information placée
    **au milieu** d'un long contexte qu'au début ou à la fin. On le teste à petite échelle :
    pour 10 questions, 8 passages (le morceau qui contient la réponse et les 7
    meilleurs morceaux qui ne la contiennent pas, donc des distracteurs plausibles), le bon
    morceau placé en position 1, 4 ou 8 (≈ 2 260 tokens de prompt).
    """)


@app.cell
def _(
    CONSIGNE_RAG,
    PERT,
    config,
    generer,
    generations,
    juste,
    morceaux,
    np,
    prompt_rag,
    questions,
    scores_hybride,
    time,
):
    _ = generations  # dépendance explicite : après les générations principales (même modèle chargé)
    _n = config["n_passages_milieu"]
    POSITIONS = sorted({1, (_n + 1) // 2, _n})
    _ordre = np.argsort(-scores_hybride, axis=1, kind="stable")
    # Questions retenues : les n_milieu premières dont un morceau pertinent existe
    _retenues = [i for i in range(len(questions)) if PERT[i]][: config["n_milieu"]]
    milieu = {p: [] for p in POSITIONS}
    tokens_milieu = []
    _t0 = time.perf_counter()
    for _i in _retenues:
        _bon = next(j for j in _ordre[_i] if j in PERT[_i])          # le mieux classé des pertinents
        _distr = [j for j in _ordre[_i] if j not in PERT[_i]][: _n - 1]
        for _p in POSITIONS:
            _liste = _distr[: _p - 1] + [_bon] + _distr[_p - 1:]
            _r = generer(CONSIGNE_RAG, prompt_rag(questions[_i]["question"],
                                                  [morceaux[j]["complet"] for j in _liste]))
            milieu[_p].append(juste(_r["texte"], questions[_i]["attendu"]))
            tokens_milieu.append(_r["tokens_prompt"])
    duree_milieu = time.perf_counter() - _t0
    bilan_milieu = {p: float(np.mean(v)) for p, v in milieu.items()}
    print(f"{len(_retenues)} questions × {len(POSITIONS)} positions, {_n} passages "
          f"(≈ {np.mean(tokens_milieu):.0f} tokens de prompt) en {duree_milieu:.0f} s")
    for _p, _v in bilan_milieu.items():
        print(f"  bon passage en position {_p:2d} : {_v:.0%} de réponses justes")
    return POSITIONS, bilan_milieu, duree_milieu, milieu, tokens_milieu


@app.cell
def _(
    DOSSIER,
    PALETTE,
    POSITIONS,
    SMOKE,
    bilan_llm,
    bilan_milieu,
    plt,
    resultats,
    resultats_tailles,
    sauvegarder_figure,
):
    # --- Figures : recherche (rappel@k), taille des morceaux, génération ---
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.2))
    _ax = _axes[0]
    for _c, (_nom, _r) in zip(PALETTE, resultats.items()):
        _ax.plot([1, 3, 5, 10], [_r[f"r@{k}"] * 100 for k in (1, 3, 5, 10)], "o-", color=_c, label=_nom)
    _ax.set(xlabel="k (morceaux renvoyés)", ylabel="rappel@k (%)", title="Recherche", xticks=[1, 3, 5, 10])
    _ax.legend(fontsize=8)
    _ax = _axes[1]
    _noms = list(resultats_tailles)
    for _j, (_cle, _c) in enumerate([("BM25 r@5", PALETTE[0]), ("e5 r@5", PALETTE[1]), ("RRF r@5", PALETTE[3])]):
        _ax.bar([x + 0.27 * (_j - 1) for x in range(len(_noms))],
                [resultats_tailles[n][_cle] * 100 for n in _noms], 0.27, color=_c, label=_cle)
    _ax.set_xticks(range(len(_noms)), [f"{n}\n({resultats_tailles[n]['n']} morceaux)" for n in _noms], fontsize=8)
    _ax.set(ylabel="rappel@5 (%)", title="Découpage")
    _ax.legend(fontsize=8)
    _ax = _axes[2]
    _barres = {"sans contexte": bilan_llm["seul_juste"], "RAG": bilan_llm["rag_juste"],
               "RAG sans\nla réponse": bilan_llm["sans_juste"]}
    _barres.update({f"bon passage\n{p}e sur {max(POSITIONS)}": v for p, v in bilan_milieu.items()})
    _ax.bar(range(len(_barres)), [v * 100 for v in _barres.values()],
            color=[PALETTE[7], PALETTE[2], PALETTE[3]] + [PALETTE[5]] * len(bilan_milieu))
    _ax.set_xticks(range(len(_barres)), list(_barres), fontsize=7, rotation=30, ha="right")
    _ax.set(ylabel="réponses justes (%)", title="Génération (qwen3.5:2b)", ylim=(0, 100))
    _fig.tight_layout()
    _chemins = sauvegarder_figure(_fig, "mini-rag-smoke" if SMOKE else "mini-rag", DOSSIER / "figures")
    print(f"Figure : {_chemins[0]}")
    _fig


@app.cell
def _(
    DOSSIER,
    K,
    POSITIONS,
    SMOKE,
    accord_top10_bm25s,
    bilan_llm,
    bilan_milieu,
    config,
    duree_encodage,
    duree_generation,
    duree_index_bm25,
    duree_milieu,
    ecart_bm25s,
    generations,
    json,
    meilleur,
    morceaux,
    questions,
    resultats,
    resultats_tailles,
    temps_rerank,
    tokens_milieu,
):
    # --- Résultats bruts, lus par les figures du site (site/scripts/figures/ch32_*.py) ---
    sortie = {
        "n_questions": len(questions), "n_morceaux": len(morceaux), "k_contexte": K,
        "types": {t: sum(q["type"] == t for q in questions) for t in ("lexicale", "reformulee")},
        "recherche": {n: {k: v for k, v in r.items() if k != "premier"} | {
            "premier": [int(x) for x in r["premier"]]} for n, r in resultats.items()},
        "types_questions": [q["type"] for q in questions],
        "tailles": resultats_tailles, "meilleur": meilleur,
        "bm25s": {"ecart_max": ecart_bm25s, "accord_top10": accord_top10_bm25s},
        "temps_rerank_s": temps_rerank, "duree_index_bm25_s": duree_index_bm25,
        "duree_encodage_s": duree_encodage,
        "llm": bilan_llm, "duree_generation_s": duree_generation,
        "milieu": {str(p): v for p, v in bilan_milieu.items()}, "positions": POSITIONS,
        "milieu_tokens": float(sum(tokens_milieu) / max(len(tokens_milieu), 1)),
        "duree_milieu_s": duree_milieu, "n_passages_milieu": config["n_passages_milieu"],
        "generations": generations,
    }
    _f = DOSSIER / "figures" / ("resultats-smoke.json" if SMOKE else "resultats.json")
    _f.write_text(json.dumps(sortie, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Résultats : {_f}")


@app.cell
def _(SMOKE, accord_top10_bm25s, bilan_llm, ecart_bm25s, mo, resultats):
    # --- Bilan : seuils du README. En smoke, on vérifie seulement que tout tourne. ---
    _r = resultats
    _hyb, _bm, _de = _r["hybride RRF"], _r["BM25"], _r["e5 (dense)"]
    _m3 = _r["+ reclassement bge-m3 (multilingue)"]
    seuils = [
        ("BM25 numpy = bm25s (écart < 1e-3, mêmes 10 premiers)",
         ecart_bm25s < 1e-3 and accord_top10_bm25s >= 0.99),
        ("RRF : rappel@5 ≥ 95 % et ≥ celui de BM25 et de e5",
         _hyb["r@5"] >= max(0.95, _bm["r@5"], _de["r@5"])),
        ("bge-m3 : rappel@1 ≥ 88 % et ≥ RRF + 5 points",
         _m3["r@1"] >= max(0.88, _hyb["r@1"] + 0.05)),
        ("RAG : réponses justes ≥ 70 % et ≥ sans contexte + 50 points",
         bilan_llm["rag_juste"] >= max(0.70, bilan_llm["seul_juste"] + 0.50)),
        ("RAG sans la réponse : « je ne sais pas » ≥ 35 %", bilan_llm["sans_nsp"] >= 0.35),
    ]
    _lignes = "\n".join(f"- {'OK' if ok else ('non vérifié (smoke)' if SMOKE else 'ÉCHEC')} : {nom}"
                         for nom, ok in seuils)
    print("Bilan :\n" + _lignes)
    if not SMOKE and not all(ok for _, ok in seuils):
        raise SystemExit("Seuils non atteints :\n" + _lignes)
    mo.md(f"**Bilan**\n\n{_lignes}")
    return (seuils,)


if __name__ == "__main__":
    app.run()
