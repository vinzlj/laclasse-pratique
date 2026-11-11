# Exercice 28a : les données et l'échelle, en miniature.
#
#   make ex NUM=28a              → éditeur marimo (bouton « Entraîner »)
#   make ex NUM=28a MODE=run     → script complet, vérifie les seuils du README
#   make ex NUM=28a MODE=smoke   → test rapide (< 30 s)
#
# Partie A (NumPy, CPU) : déduplication exacte puis approximative (MinHash + LSH, réglages
# de FineWeb), détection de contamination par 13-grammes (méthode de GPT-3).
# Partie B (MPS) : loi d'échelle sur cinq mini-GPT de 12 000 à 790 000 paramètres,
# frontière « compute-optimale », prédiction du plus gros modèle à partir des autres, et
# effet d'une fuite de données de validation selon la taille.
# Partie C (MPS) : mélange d'experts (MoE) à routage top-1, avec et sans perte
# d'équilibrage, contre un modèle dense de même calcul par token.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import re
    import time
    import zlib

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.optimize import curve_fit
    from torch import nn

    from ia_commun import get_device, is_smoke, sauvegarder_figure, seed_everything, style_commun
    from ia_commun.donnees import charger_tinyshakespeare
    from ia_commun.mini_gpt import ConfigGPT, MiniGPT, TokeniseurCaracteres, estimer_loss, lot_aleatoire

    return (
        ConfigGPT,
        F,
        MiniGPT,
        TokeniseurCaracteres,
        charger_tinyshakespeare,
        curve_fit,
        estimer_loss,
        get_device,
        is_smoke,
        lot_aleatoire,
        math,
        mo,
        nn,
        np,
        plt,
        re,
        sauvegarder_figure,
        seed_everything,
        style_commun,
        time,
        torch,
        zlib,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Exercice 28a : les données et l'échelle, en miniature

    Un LLM moderne, c'est d'abord **beaucoup de texte nettoyé** et **un budget de calcul**
    à répartir entre la taille du modèle et la quantité de données. On reproduit ici, sur
    TinyShakespeare (1,1 million de caractères) et des modèles de moins d'un million de
    paramètres, quatre gestes du pré-entraînement :

    - **A1. Dédupliquer.** Le web est plein de pages presque identiques. Un hachage exact ne
      les voit pas ; **MinHash** et le hachage sensible à la localité (LSH) les trouvent,
      avec les réglages de FineWeb (5-grammes de mots, 112 fonctions de hachage, 14 bandes
      de 8).
    - **A2. Repérer la contamination.** Des textes de l'ensemble de test qui se retrouvent dans
      les données d'entraînement faussent l'évaluation. On en glisse volontairement 100 et on
      les retrouve par recouvrement de 13-grammes de mots, comme l'a fait l'équipe de GPT-3.
    - **B. Loi d'échelle.** Cinq modèles de 12 000 à 790 000 paramètres, même recette, même
      nombre de tokens : la perte finale suit-elle une loi de puissance
      $L(N) = E + A\,N^{-\alpha}$ ? Peut-on **prédire** le plus gros modèle à partir des
      quatre autres ? Pour un budget de calcul $C \approx 6ND$ donné, quelle taille vaut-il
      mieux entraîner ?
    - **C. Mélange d'experts.** On remplace le MLP de chaque bloc par 4 experts dont un seul
      est activé par token, et on regarde ce que change la **perte d'équilibrage**.

    **Pertes optimisées.** Parties B et C : entropie croisée du token suivant
    $\mathcal{L} = -\frac{1}{T}\sum_t \log p_\theta(x_{t+1}\mid x_{\le t})$ (chapitres 01 et 20).
    Partie C, en plus : perte d'équilibrage du Switch Transformer,
    $\mathcal{L}_{\text{équil}} = \alpha\, N_e \sum_{i=1}^{N_e} f_i P_i$, où $f_i$ est la part des
    tokens envoyés à l'expert $i$ et $P_i$ la probabilité moyenne que le routeur lui donne.
    La partie A n'optimise rien : ce sont des algorithmes de comptage et de hachage.
    """)


@app.cell
def _(get_device, is_smoke, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    device = get_device()
    style_commun()
    config = dict(
        part_train=0.9,                      # même découpage que l'exercice 20a
        # A1. Déduplication approximative (réglages FineWeb, arXiv:2406.17557, annexe E.1)
        n_gramme_minhash=5, n_hachages=112, n_bandes=14, lignes_par_bande=8,
        mots_min_paragraphe=40,              # paragraphes assez longs pour avoir des 5-grammes
        n_variantes=60 if SMOKE else 400,    # copies modifiées injectées dans le corpus
        taux_modif_max=0.30,                 # part maximale de mots remplacés dans une variante
        # A2. Contamination (méthode de GPT-3 : recouvrement de 13-grammes de mots)
        n_gramme_contamination=13,
        n_fuite=100,                         # paragraphes de validation glissés dans l'entraînement
        mots_min_fuite=30,
        # B. Loi d'échelle : (couches, dimension, têtes)
        tailles=[(1, 32, 2), (2, 48, 2)] if SMOKE else
                [(1, 32, 2), (2, 48, 2), (2, 80, 4), (3, 96, 4), (4, 128, 4)],
        taille_bloc=128, taille_lot=32,      # 4 096 tokens (caractères) par pas
        n_pas=20 if SMOKE else 2000,         # 8,2 M tokens par modèle en mode complet
        lr_max=3e-3, lr_min=3e-4, pas_chauffe=5 if SMOKE else 100,
        eval_tous=10 if SMOKE else 100,
        n_lots_eval=2 if SMOKE else 10,
        # C. Mélange d'experts
        moe_couches=2, moe_dim=64, moe_tetes=4, moe_n_experts=4,
        moe_n_pas=8 if SMOKE else 1500,
        moe_coef_equilibrage=1e-2,           # α du Switch Transformer (arXiv:2101.03961)
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, périphérique {device}")
    print(config)
    return SEED, SMOKE, config, device


@app.cell
def _(TokeniseurCaracteres, charger_tinyshakespeare, config, re):
    texte = charger_tinyshakespeare()
    tok = TokeniseurCaracteres(texte)
    _n = int(config["part_train"] * len(texte))
    texte_train, texte_val = texte[:_n], texte[_n:]

    def mots(s):
        """Découpage en mots minimal : minuscules, lettres et apostrophes."""
        return re.findall(r"[a-z']+", s.lower())

    def paragraphes(s):
        """Un « document » = une réplique (blocs séparés par une ligne vide)."""
        return [p for p in s.split("\n\n") if p.strip()]

    print(f"{len(texte):,} caractères ; train {len(texte_train):,}, val {len(texte_val):,} ; "
          f"{len(paragraphes(texte))} répliques".replace(",", " "))
    return mots, paragraphes, texte, texte_train, texte_val, tok


@app.cell
def _(mo):
    mo.md(r"""
    ## A1. Déduplication : exacte, puis MinHash

    **Pourquoi.** Lee et al. (2021) ont montré que dédupliquer les corpus d'entraînement
    réduit la récitation mot pour mot et rend l'évaluation plus honnête. Sur le web, les
    doublons sont rarement exacts : bandeaux, dates, quelques mots qui changent.

    **Le jeu.** On prend les répliques de l'ensemble d'entraînement de plus de 40 mots, et on
    y injecte des **variantes** : des copies de répliques où une part $r$ des mots, tirée
    entre 0 et 30 %, est remplacée par des mots au hasard.

    **MinHash** (Broder, 1997). On représente un document par l'ensemble de ses 5-grammes de
    mots ; la similarité de Jaccard $J(A, B) = |A\cap B| / |A\cup B|$ mesure le recouvrement.
    Pour une fonction de hachage aléatoire $h$, $\Pr[\min h(A) = \min h(B)] = J(A, B)$ :
    112 hachages donnent une **signature** de 112 nombres, et la part de valeurs égales
    estime $J$. **LSH** : on coupe la signature en 14 bandes de 8 ; deux documents sont
    candidats s'ils ont une bande identique. Probabilité de détection :
    $P(J) = 1 - (1 - J^8)^{14}$, une marche autour de $J \approx 0{,}72$, sans jamais comparer
    toutes les paires.
    """)


@app.cell
def _(config, mots, np, paragraphes, texte_train, zlib):
    _rng = np.random.default_rng(42)
    # Documents de départ : répliques d'entraînement assez longues
    docs_origine = [p for p in paragraphes(texte_train) if len(mots(p)) >= config["mots_min_paragraphe"]]
    _vocab = sorted(set(mots(texte_train)))

    def varier(doc, taux, rng):
        """Copie quasi identique : chaque mot est remplacé avec la probabilité `taux`."""
        m = mots(doc)
        return " ".join(rng.choice(_vocab) if rng.random() < taux else w for w in m)

    _choisis = _rng.choice(len(docs_origine), size=config["n_variantes"], replace=False)
    taux_variantes = _rng.uniform(0.0, config["taux_modif_max"], size=config["n_variantes"])
    # Corpus = originaux (normalisés en mots) + variantes ; on garde qui est la copie de qui
    corpus = [" ".join(mots(d)) for d in docs_origine]
    paires_vraies = []
    for _k, (_i, _t) in enumerate(zip(_choisis, taux_variantes)):
        corpus.append(varier(docs_origine[_i], _t, _rng))
        paires_vraies.append((int(_i), len(corpus) - 1))

    # Déduplication exacte : hachage du texte entier
    _vus, doublons_exacts = {}, []
    for _j, _d in enumerate(corpus):
        _h = zlib.crc32(_d.encode())
        if _h in _vus:
            doublons_exacts.append((_vus[_h], _j))
        else:
            _vus[_h] = _j
    _variantes_exactes = sum(1 for a, b in paires_vraies if corpus[a] == corpus[b])
    print(f"{len(docs_origine)} répliques de ≥ {config['mots_min_paragraphe']} mots + "
          f"{config['n_variantes']} variantes = {len(corpus)} documents")
    print(f"Hachage exact : {len(doublons_exacts)} doublons trouvés, dont {_variantes_exactes} variantes "
          f"(celles où aucun mot n'a été tiré) sur {config['n_variantes']}")
    return corpus, docs_origine, doublons_exacts, paires_vraies, taux_variantes


@app.cell
def _(config, np, zlib):
    P_MERSENNE = (1 << 31) - 1   # nombre premier : h(x) = (a·x + b) mod p, a·x tient dans 64 bits

    def shingles(doc, n):
        """Ensemble des n-grammes de mots, chacun haché en un entier (crc32, reproductible)."""
        m = doc.split()
        return np.array(sorted({zlib.crc32(" ".join(m[i:i + n]).encode()) for i in range(len(m) - n + 1)}),
                        dtype=np.uint64)

    _rng = np.random.default_rng(42)
    _a = _rng.integers(1, P_MERSENNE, size=config["n_hachages"], dtype=np.uint64)
    _b = _rng.integers(0, P_MERSENNE, size=config["n_hachages"], dtype=np.uint64)

    def signature_minhash(sh):
        """112 minimums : pour chaque fonction de hachage, la plus petite valeur sur les shingles."""
        x = (sh % P_MERSENNE)[None, :]                       # (1, n_shingles)
        h = (_a[:, None] * x + _b[:, None]) % P_MERSENNE     # (112, n_shingles)
        return h.min(axis=1)

    def jaccard(s1, s2):
        inter = np.intersect1d(s1, s2, assume_unique=True).size
        return inter / (s1.size + s2.size - inter)

    def proba_lsh(j, b=config["n_bandes"], r=config["lignes_par_bande"]):
        """Probabilité qu'une paire de similarité j partage au moins une bande."""
        return 1 - (1 - np.asarray(j) ** r) ** b

    return jaccard, proba_lsh, shingles, signature_minhash


@app.cell
def _(
    config,
    corpus,
    jaccard,
    np,
    paires_vraies,
    proba_lsh,
    shingles,
    signature_minhash,
    time,
):
    _t0 = time.perf_counter()
    _sh = [shingles(d, config["n_gramme_minhash"]) for d in corpus]
    _sig = np.stack([signature_minhash(s) for s in _sh])     # (n_docs, 112)
    # LSH : un dictionnaire par bande, clé = les 8 valeurs de la bande
    _b, _r = config["n_bandes"], config["lignes_par_bande"]
    candidats = set()
    for _k in range(_b):
        _seaux = {}
        for _j, _bande in enumerate(map(bytes, _sig[:, _k * _r:(_k + 1) * _r])):
            _seaux.setdefault(_bande, []).append(_j)
        for _membres in _seaux.values():
            for _x in range(len(_membres)):
                for _y in range(_x + 1, len(_membres)):
                    candidats.add((_membres[_x], _membres[_y]))
    duree_minhash = time.perf_counter() - _t0

    # Pour chaque variante : vraie similarité, estimation MinHash, détection
    jaccard_vrai = np.array([jaccard(_sh[a], _sh[b]) for a, b in paires_vraies])
    jaccard_estime = np.array([(_sig[a] == _sig[b]).mean() for a, b in paires_vraies])
    detecte = np.array([(a, b) in candidats for a, b in paires_vraies])
    _vraies = set(paires_vraies)
    faux_positifs = [(a, b) for a, b in candidats if (a, b) not in _vraies]
    _jfp = [jaccard(_sh[a], _sh[b]) for a, b in faux_positifs]
    n_paires_total = len(corpus) * (len(corpus) - 1) // 2
    erreur_estimation = float(np.abs(jaccard_estime - jaccard_vrai).mean())
    rappel_haut = float(detecte[jaccard_vrai >= 0.8].mean()) if (jaccard_vrai >= 0.8).any() else float("nan")
    rappel_bas = float(detecte[jaccard_vrai < 0.5].mean()) if (jaccard_vrai < 0.5).any() else float("nan")
    print(f"MinHash + LSH : {duree_minhash:.1f} s pour {len(corpus)} documents ; "
          f"{len(candidats)} paires candidates sur {n_paires_total:,} possibles".replace(",", " "))
    print(f"Erreur moyenne |J estimé − J vrai| = {erreur_estimation:.3f}")
    print(f"Variantes détectées : {detecte.sum()}/{len(detecte)} ; J ≥ 0,8 : {rappel_haut:.0%} ; "
          f"J < 0,5 : {rappel_bas:.0%}")
    print(f"Paires candidates qui ne sont pas des variantes : {len(faux_positifs)}"
          + (f" (J vrai : {', '.join(f'{j:.2f}' for j in _jfp[:8])})" if _jfp else ""))
    for _lo, _hi in [(0, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.01)]:
        _m = (jaccard_vrai >= _lo) & (jaccard_vrai < _hi)
        if _m.any():
            print(f"  J dans [{_lo:.2f} ; {min(_hi, 1):.2f}[ : {_m.sum():3d} variantes, détectées {detecte[_m].mean():5.0%}, "
                  f"théorie {proba_lsh(jaccard_vrai[_m]).mean():5.0%}")
    return (
        detecte,
        duree_minhash,
        erreur_estimation,
        faux_positifs,
        jaccard_estime,
        jaccard_vrai,
        rappel_bas,
        rappel_haut,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## A2. Contamination : une fuite volontaire, retrouvée par 13-grammes

    On choisit 100 répliques de l'ensemble de **validation** et on les recopie à la fin des
    données d'entraînement : c'est ce qui arrive quand une page web cite les questions d'un
    test. Le détecteur de GPT-3 marque un exemple comme « potentiellement contaminé » s'il
    partage au moins un **13-gramme de mots** avec les données d'entraînement.

    Les modèles de la partie B seront entraînés sur ces données contaminées. On mesurera leur
    perte sur les 100 répliques qui ont fuité et sur 100 répliques témoins de même longueur
    restées propres : l'écart dit combien la fuite **gonfle** le score.
    """)


@app.cell
def _(config, mots, np, paragraphes, texte_train, texte_val):
    _rng = np.random.default_rng(42)
    _val_pars = [p for p in paragraphes(texte_val) if len(mots(p)) >= config["mots_min_fuite"]]
    _ordre = _rng.permutation(len(_val_pars))
    fuite = [_val_pars[i] for i in _ordre[:config["n_fuite"]]]
    temoins = [_val_pars[i] for i in _ordre[config["n_fuite"]:2 * config["n_fuite"]]]
    # Données d'entraînement contaminées : la fuite est recopiée une fois à la fin
    train_contamine = texte_train + "\n\n" + "\n\n".join(fuite) + "\n"
    # Validation propre : texte de validation sans les répliques qui ont fuité
    val_propre = texte_val
    for _p in fuite:
        val_propre = val_propre.replace(_p, "")

    def ngrammes(s, n):
        m = mots(s)
        return {tuple(m[i:i + n]) for i in range(len(m) - n + 1)}

    _n = config["n_gramme_contamination"]
    _index = ngrammes(train_contamine, _n)
    _drapeau = {p: bool(ngrammes(p, _n) & _index) for p in _val_pars}
    _fuite_set = set(fuite)
    rappel_contamination = float(np.mean([_drapeau[p] for p in fuite]))
    faux_contamination = sum(_drapeau[p] for p in _val_pars if p not in _fuite_set)
    # Avant la fuite : combien de répliques de validation partagent un 13-gramme avec le train ?
    _index_propre = ngrammes(texte_train, _n)
    contamination_naturelle = sum(bool(ngrammes(p, _n) & _index_propre) for p in _val_pars)
    print(f"{len(_val_pars)} répliques de validation de ≥ {config['mots_min_fuite']} mots ; "
          f"{len(fuite)} recopiées dans l'entraînement ({sum(map(len, fuite)):,} caractères)".replace(",", " "))
    print(f"Avant la fuite : {contamination_naturelle} répliques de validation partagent un {_n}-gramme avec le train")
    print(f"Après : détecteur 13-grammes, rappel {rappel_contamination:.0%} sur la fuite, "
          f"{faux_contamination} fausse(s) alerte(s) sur les {len(_val_pars) - len(fuite)} autres")
    return (
        contamination_naturelle,
        faux_contamination,
        fuite,
        rappel_contamination,
        temoins,
        train_contamine,
        val_propre,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## B. Loi d'échelle

    Cinq mini-GPT (ceux de l'exercice 20a, en plus petit, sans dropout), tous entraînés avec
    la **même recette** : 2 000 pas de 32 fenêtres de 128 caractères, soit
    $D \approx 8{,}2$ millions de tokens chacun, AdamW, taux qui monte pendant 100 pas puis
    décroît en cosinus. On compte $N$ = paramètres **hors plongements** (convention de
    Kaplan et al., 2020) et le calcul d'entraînement $C \approx 6ND$ opérations (2 pour le
    passage avant, 4 pour le passage arrière, par paramètre et par token).

    Trois questions :
    1. la perte finale suit-elle $L(N) = E + A\,N^{-\alpha}$ ?
    2. un ajustement sur les **quatre plus petits** modèles prédit-il le cinquième ?
    3. à budget $C$ fixé, quelle taille donne la perte la plus basse (la **frontière**) ?
    """)


@app.cell
def _(ConfigGPT, MiniGPT, config, device, math, tok, torch, train_contamine, val_propre):
    ids_train = torch.tensor(tok.encoder(train_contamine), dtype=torch.long)
    ids_val = torch.tensor(tok.encoder(val_propre), dtype=torch.long)
    V = tok.taille_vocab

    def creer_modele(couches, dim, tetes, graine=42):
        torch.manual_seed(graine)
        return MiniGPT(ConfigGPT(taille_vocab=V, taille_bloc=config["taille_bloc"], n_couches=couches,
                                 n_tetes=tetes, n_embd=dim, dropout=0.0)).to(device)

    def n_hors_plongements(m):
        """Paramètres des blocs et de la normalisation finale (sans plongements ni tête partagée)."""
        return sum(p.numel() for n, p in m.named_parameters() if n.startswith(("blocs.", "ln_final")))

    def taux(pas, n_pas):
        if pas < config["pas_chauffe"]:
            return config["lr_max"] * (pas + 1) / config["pas_chauffe"]
        r = min(1.0, (pas - config["pas_chauffe"]) / max(1, n_pas - config["pas_chauffe"]))
        return config["lr_min"] + 0.5 * (config["lr_max"] - config["lr_min"]) * (1 + math.cos(math.pi * r))

    for _c, _d, _h in config["tailles"]:
        _m = creer_modele(_c, _d, _h)
        print(f"  {_c} couche(s), {_d} dimensions, {_h} têtes : N = {n_hors_plongements(_m):>7,} hors plongements, "
              f"{_m.nb_parametres():>7,} au total".replace(",", " "))
    return V, creer_modele, ids_train, ids_val, n_hors_plongements, taux


@app.cell
def _(F, config, device, torch):
    @torch.no_grad()
    def perte_textes(modele, textes, tok):
        """Perte moyenne par caractère sur une liste de textes, fenêtres de 129 caractères sans recouvrement."""
        modele.eval()
        T = config["taille_bloc"]
        fenetres = []
        for s in textes:
            ids = tok.encoder(s)
            for i in range(0, len(ids) - 1, T):
                morceau = ids[i:i + T + 1]
                if len(morceau) >= 2:
                    fenetres.append(morceau)
        total, n = 0.0, 0
        for f in fenetres:
            x = torch.tensor([f[:-1]], device=device)
            y = torch.tensor([f[1:]], device=device)
            logits, _ = modele(x)
            total += F.cross_entropy(logits[0], y[0], reduction="sum").item()
            n += y.numel()
        return total / n

    return (perte_textes,)


@app.cell
def _(config, estimer_loss, lot_aleatoire, n_hors_plongements, taux, time, torch):
    def entrainer(modele, ids_train, ids_val, n_pas, journal="", perte_aux=None, coef_aux=0.0):
        """AdamW + cosinus. `perte_aux(modele)` : perte auxiliaire facultative (MoE)."""
        optim = torch.optim.AdamW(modele.parameters(), lr=config["lr_max"], betas=(0.9, 0.95), weight_decay=0.1)
        gen = torch.Generator().manual_seed(42)
        h = {"pas": [], "val": [], "tokens": [], "duree": 0.0}
        tokens_par_pas = config["taille_lot"] * config["taille_bloc"]
        dev = next(modele.parameters()).device
        t0 = time.perf_counter()
        modele.train()
        for pas in range(n_pas + 1):
            if pas % config["eval_tous"] == 0 or pas == n_pas:
                # mêmes lots de validation à chaque évaluation (générateur réinitialisé)
                v = estimer_loss(modele, ids_val, config["taille_bloc"], config["taille_lot"], dev,
                                 n_lots=config["n_lots_eval"], generateur=torch.Generator().manual_seed(2))
                h["pas"].append(pas)
                h["val"].append(v)
                h["tokens"].append(pas * tokens_par_pas)
                if pas % (5 * config["eval_tous"]) == 0 or pas == n_pas:
                    print(f"{journal}pas {pas:5d}  val {v:.3f}  ({time.perf_counter() - t0:.0f} s)")
            if pas == n_pas:
                break
            for g in optim.param_groups:
                g["lr"] = taux(pas, n_pas)
            x, y = lot_aleatoire(ids_train, config["taille_bloc"], config["taille_lot"], dev, gen)
            _, loss = modele(x, y)
            if perte_aux is not None and coef_aux > 0:
                loss = loss + coef_aux * perte_aux(modele)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
            optim.step()
        h["duree"] = time.perf_counter() - t0
        h["N"] = n_hors_plongements(modele)
        modele.eval()
        return h

    return (entrainer,)


@app.cell
def _(mo):
    bouton = mo.ui.run_button(label="Entraîner (≈ 3 min de loi d'échelle, puis ≈ 2 min de MoE, sur M1 Pro)")
    bouton
    return (bouton,)


@app.cell
def _(
    bouton,
    config,
    creer_modele,
    entrainer,
    fuite,
    ids_train,
    ids_val,
    mo,
    perte_textes,
    temoins,
    tok,
):
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Entraîner » pour lancer l'apprentissage.*"))
    resultats_echelle = []
    for _c, _d, _h in config["tailles"]:
        _m = creer_modele(_c, _d, _h)
        _hist = entrainer(_m, ids_train, ids_val, config["n_pas"], journal=f"[{_c}×{_d}] ")
        _hist.update(couches=_c, dim=_d, tetes=_h,
                     perte_fuite=perte_textes(_m, fuite, tok), perte_temoins=perte_textes(_m, temoins, tok))
        print(f"[{_c}×{_d}] N = {_hist['N']:,} ; val finale {_hist['val'][-1]:.3f} ; "
              f"fuite {_hist['perte_fuite']:.3f} contre témoins {_hist['perte_temoins']:.3f} ; "
              f"{_hist['duree']:.0f} s ({1000 * _hist['duree'] / config['n_pas']:.1f} ms/pas)".replace(",", " "))
        resultats_echelle.append(_hist)
    return (resultats_echelle,)


@app.cell
def _(SMOKE, config, curve_fit, np, resultats_echelle):
    N_tab = np.array([r["N"] for r in resultats_echelle], dtype=float)
    L_tab = np.array([r["val"][-1] for r in resultats_echelle])

    def loi(N, E, A, alpha):
        return E + A * N ** (-alpha)

    def ajuster(N, L):
        """Moindres carrés sur L(N) = E + A·N^-α ; E borné sous la plus petite perte observée."""
        p, _ = curve_fit(loi, N, L, p0=(L.min() - 0.3, 10.0, 0.3),
                         bounds=([0.0, 1e-3, 0.01], [L.min(), 1e4, 2.0]), maxfev=20000)
        return p

    if len(N_tab) >= 4:
        params_loi = ajuster(N_tab, L_tab)
        params_4 = ajuster(N_tab[:-1], L_tab[:-1])
        prediction_dernier = float(loi(N_tab[-1], *params_4))
    else:   # smoke : deux modèles ne suffisent pas pour trois paramètres
        params_loi = params_4 = np.array([np.nan, np.nan, np.nan])
        prediction_dernier = float("nan")
    erreur_prediction = abs(prediction_dernier - L_tab[-1])
    print("N hors plongements :", ", ".join(f"{int(n):,}".replace(",", " ") for n in N_tab))
    print("Perte de validation finale :", ", ".join(f"{l:.3f}" for l in L_tab))
    if not SMOKE:
        print(f"Ajustement sur les 5 : L(N) = {params_loi[0]:.3f} + {params_loi[1]:.2f} · N^-{params_loi[2]:.3f}")
        print(f"Ajustement sur les 4 plus petits : E = {params_4[0]:.3f}, A = {params_4[1]:.2f}, α = {params_4[2]:.3f} ; "
              f"prédiction pour N = {int(N_tab[-1]):,} : {prediction_dernier:.3f}, observé {L_tab[-1]:.3f} "
              f"(écart {erreur_prediction:.3f})".replace(",", " "))

    # Frontière : pour chaque budget C = 6·N·D, la plus basse des pertes atteintes (interpolation en log C)
    _grille = np.logspace(np.log10(6 * N_tab.min() * config["taille_lot"] * config["taille_bloc"] * config["eval_tous"]),
                          np.log10(6 * N_tab.max() * resultats_echelle[-1]["tokens"][-1]), 60)
    _courbes = []
    for _r in resultats_echelle:
        _C = 6 * _r["N"] * np.array(_r["tokens"][1:], dtype=float)
        _L = np.array(_r["val"][1:])
        _courbes.append(np.where((_grille >= _C[0]) & (_grille <= _C[-1]),
                                 np.interp(np.log(_grille), np.log(_C), _L), np.inf))
    _courbes = np.array(_courbes)
    frontiere_C = _grille[np.isfinite(_courbes.min(0))]
    frontiere_L = _courbes.min(0)[np.isfinite(_courbes.min(0))]
    frontiere_N = N_tab[_courbes.argmin(0)][np.isfinite(_courbes.min(0))]
    # Taille optimale par décade de calcul
    for _c in [1e11, 1e12, 1e13, 1e14]:
        _k = np.argmin(np.abs(np.log(frontiere_C) - np.log(_c))) if len(frontiere_C) else None
        if _k is not None and frontiere_C[0] <= _c <= frontiere_C[-1]:
            print(f"  C = {_c:.0e} : meilleur modèle N = {int(frontiere_N[_k]):,} "
                  f"(D = {_c / 6 / frontiere_N[_k] / 1e6:.1f} M tokens, D/N = {_c / 6 / frontiere_N[_k] ** 2:.0f}), "
                  f"perte {frontiere_L[_k]:.3f}".replace(",", " "))
    return (
        L_tab,
        N_tab,
        erreur_prediction,
        frontiere_C,
        frontiere_L,
        frontiere_N,
        loi,
        params_4,
        params_loi,
        prediction_dernier,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## C. Mélange d'experts (MoE) et perte d'équilibrage

    Un MoE remplace le MLP de chaque bloc par $N_e$ MLP « experts » et un **routeur**
    linéaire qui choisit, pour chaque token, le ou les experts à activer (ici **top-1**,
    comme le Switch Transformer). Le modèle a 4 fois plus de paramètres de MLP, mais chaque
    token ne passe que par un expert : **même calcul par token** que le modèle dense.

    Sans contrainte, le routeur peut s'effondrer : un expert reçoit presque tout, les autres
    ne servent à rien. La perte d'équilibrage $\alpha N_e \sum_i f_i P_i$ vaut $\alpha$
    quand la charge est uniforme ($f_i = P_i = 1/N_e$) et $\alpha N_e$ quand un seul expert
    prend tout ; elle pousse vers l'équilibre. Trois modèles, 1 500 pas chacun :
    dense ; MoE sans perte d'équilibrage ; MoE avec $\alpha = 10^{-2}$.
    """)


@app.cell
def _(F, nn, torch):
    class MoE(nn.Module):
        """Couche de mélange d'experts à routage top-1 (Switch Transformer).

        Chaque expert est un MLP n_embd → 4·n_embd → n_embd, comme le MLP dense du mini-GPT.
        La sortie d'un token est g · expert_i(x), où i = argmax du routeur et g sa probabilité
        (ce facteur laisse passer le gradient vers le routeur).
        """

        def __init__(self, n_embd, n_experts):
            super().__init__()
            self.n_experts = n_experts
            self.routeur = nn.Linear(n_embd, n_experts, bias=False)
            self.experts = nn.ModuleList(
                nn.Sequential(nn.Linear(n_embd, 4 * n_embd, bias=False), nn.GELU(),
                              nn.Linear(4 * n_embd, n_embd, bias=False))
                for _ in range(n_experts))
            self.perte_equilibrage = torch.tensor(0.0)
            self.charge = torch.zeros(n_experts)

        def forward(self, x):
            B, T, C = x.shape
            plat = x.reshape(-1, C)
            probas = F.softmax(self.routeur(plat), dim=-1)          # (n_tokens, n_experts)
            g, choix = probas.max(dim=-1)                           # top-1
            # Version pédagogique : chaque expert traite TOUS les tokens et on ne garde que sa part.
            # Le résultat est celui d'un vrai MoE ; un vrai MoE n'envoie à chaque expert que ses
            # tokens (c'est là que se fait l'économie de calcul), mais sur MPS l'indexation par
            # masque, de taille variable à chaque pas, est ici dix fois plus lente.
            sortie = torch.zeros_like(plat)
            for i, expert in enumerate(self.experts):
                m = (choix == i).unsqueeze(-1).to(plat.dtype)
                sortie = sortie + m * g[:, None] * expert(plat)
            # f_i : part des tokens envoyés à i (non dérivable) ; P_i : proba moyenne (dérivable)
            f = torch.bincount(choix, minlength=self.n_experts).float() / choix.numel()
            P = probas.mean(dim=0)
            self.perte_equilibrage = self.n_experts * (f * P).sum()
            self.charge = f.detach().cpu()
            return sortie.view(B, T, C)

    def perte_equilibrage_totale(modele):
        return sum(b.mlp.perte_equilibrage for b in modele.blocs)

    return MoE, perte_equilibrage_totale


@app.cell
def _(
    MoE,
    bouton,
    config,
    creer_modele,
    device,
    entrainer,
    ids_train,
    ids_val,
    lot_aleatoire,
    mo,
    perte_equilibrage_totale,
    torch,
):
    mo.stop(mo.running_in_notebook() and not bouton.value, mo.md("*En attente du bouton.*"))

    def creer_moe():
        m = creer_modele(config["moe_couches"], config["moe_dim"], config["moe_tetes"])
        torch.manual_seed(43)
        for b in m.blocs:
            b.mlp = MoE(config["moe_dim"], config["moe_n_experts"]).to(device)
            for p in b.mlp.parameters():
                torch.nn.init.normal_(p, std=0.02)
        return m

    @torch.no_grad()
    def charges(m):
        """Part des tokens envoyés à chaque expert, par couche, sur 8 lots de validation."""
        m.eval()
        g = torch.Generator().manual_seed(3)
        tot = torch.zeros(len(m.blocs), config["moe_n_experts"])
        for _ in range(8):
            x, _y = lot_aleatoire(ids_val, config["taille_bloc"], config["taille_lot"], device, g)
            m(x)
            tot += torch.stack([b.mlp.charge for b in m.blocs])
        return (tot / 8).numpy()

    resultats_moe = {}
    for _nom, _coef in [("dense", None), ("moe_sans", 0.0), ("moe_avec", config["moe_coef_equilibrage"])]:
        _m = creer_modele(config["moe_couches"], config["moe_dim"], config["moe_tetes"]) if _coef is None else creer_moe()
        _n_total = sum(p.numel() for p in _m.parameters())
        _hist = entrainer(_m, ids_train, ids_val, config["moe_n_pas"], journal=f"[{_nom}] ",
                          perte_aux=None if _coef is None else perte_equilibrage_totale, coef_aux=_coef or 0.0)
        _hist["n_total"] = _n_total
        if _coef is not None:
            _hist["charges"] = charges(_m)
            print(f"[{_nom}] charge des experts par couche :\n{_hist['charges'].round(3)}")
        print(f"[{_nom}] {_n_total:,} paramètres ; val finale {_hist['val'][-1]:.3f} ; {_hist['duree']:.0f} s".replace(",", " "))
        resultats_moe[_nom] = _hist
    return (resultats_moe,)


@app.cell
def _(
    SMOKE,
    detecte,
    frontiere_C,
    frontiere_L,
    jaccard_vrai,
    L_tab,
    loi,
    mo,
    N_tab,
    np,
    params_4,
    params_loi,
    plt,
    proba_lsh,
    resultats_echelle,
    resultats_moe,
    sauvegarder_figure,
):
    # --- Figure : quatre panneaux (dédup, échelle, contamination, MoE) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    # A1
    _ax = axes[0, 0]
    _j = np.linspace(0, 1, 200)
    _ax.plot(_j, proba_lsh(_j), color="0.4", ls="--", label="théorie 1 − (1 − J⁸)¹⁴")
    _bins = np.linspace(0, 1, 11)
    _idx = np.digitize(jaccard_vrai, _bins) - 1
    _cx = [(_bins[k] + _bins[k + 1]) / 2 for k in range(10) if (_idx == k).any()]
    _cy = [detecte[_idx == k].mean() for k in range(10) if (_idx == k).any()]
    _ax.plot(_cx, _cy, "o", ms=7, label="variantes détectées (par tranche de J)")
    _ax.set(xlabel="similarité de Jaccard vraie (5-grammes de mots)", ylabel="part détectée",
            title="A1. MinHash + LSH (14 bandes × 8)")
    _ax.legend(loc="upper left")
    # B : pertes en fonction du calcul
    _ax = axes[0, 1]
    for _r in resultats_echelle:
        _C = 6 * _r["N"] * np.array(_r["tokens"][1:], dtype=float)
        _ax.plot(_C, _r["val"][1:], label=f"N = {_r['N']:,}".replace(",", " "))
    if len(frontiere_C):
        _ax.plot(frontiere_C, frontiere_L, color="k", lw=2.5, alpha=0.35, label="frontière (meilleur à C fixé)")
    _ax.set(xscale="log", xlabel="calcul d'entraînement C ≈ 6·N·D (opérations)", ylabel="perte de validation",
            title="B. Pertes selon le calcul")
    _ax.legend(fontsize=9)
    # B : perte finale contre N + ajustements
    _ax = axes[1, 0]
    _ax.plot(N_tab, L_tab, "o", ms=8, label="perte finale observée")
    if not SMOKE:
        _n = np.logspace(np.log10(N_tab.min() / 1.5), np.log10(N_tab.max() * 3), 100)
        _ax.plot(_n, loi(_n, *params_loi), label=f"ajustement : {params_loi[0]:.2f} + {params_loi[1]:.1f}·N^-{params_loi[2]:.2f}")
        _ax.plot(_n, loi(_n, *params_4), ls=":", label="ajusté sur les 4 plus petits")
        _ax.axhline(params_loi[0], color="0.6", ls="--", lw=1, label=f"E = {params_loi[0]:.2f} (perte irréductible estimée)")
    _ax.plot(N_tab, [r["perte_fuite"] for r in resultats_echelle], "s", ms=7, label="perte sur la fuite (contaminée)")
    _ax.plot(N_tab, [r["perte_temoins"] for r in resultats_echelle], "^", ms=7, label="perte sur les témoins")
    _ax.set(xscale="log", xlabel="N (paramètres hors plongements)", ylabel="perte (nats par caractère)",
            title="B. Loi d'échelle et contamination")
    _ax.legend(fontsize=8)
    # C : MoE
    _ax = axes[1, 1]
    for _nom, _lib in [("dense", "dense"), ("moe_sans", "MoE 4 experts, sans équilibrage"),
                       ("moe_avec", "MoE 4 experts, α = 0,01")]:
        _h = resultats_moe[_nom]
        _ax.plot(_h["pas"][1:], _h["val"][1:], label=f"{_lib} ({_h['val'][-1]:.3f})")
    _ax.set(xlabel="pas", ylabel="perte de validation", title="C. Mélange d'experts")
    _ax.legend(fontsize=9)
    fig.tight_layout()
    _chemins = sauvegarder_figure(fig, "resultats-smoke" if SMOKE else "resultats", mo.notebook_dir() / "figures")
    print(f"Figure : {_chemins[0]}")
    fig


@app.cell
def _(
    SMOKE,
    contamination_naturelle,
    erreur_estimation,
    erreur_prediction,
    faux_contamination,
    faux_positifs,
    frontiere_C,
    frontiere_L,
    frontiere_N,
    jaccard_estime,
    jaccard_vrai,
    L_tab,
    mo,
    N_tab,
    np,
    params_4,
    params_loi,
    prediction_dernier,
    rappel_bas,
    rappel_contamination,
    rappel_haut,
    resultats_echelle,
    resultats_moe,
    taux_variantes,
    detecte,
):
    # --- Résultats bruts pour les figures du site (site/scripts/figures/ch28_*.py) ---
    _dossier = mo.notebook_dir() / "figures"
    _dossier.mkdir(exist_ok=True)
    _ech = {f"echelle_{k}": np.array([r[k] for r in resultats_echelle]) for k in ("N", "perte_fuite", "perte_temoins", "duree")}
    np.savez(_dossier / ("resultats-smoke.npz" if SMOKE else "resultats.npz"),
             jaccard_vrai=jaccard_vrai, jaccard_estime=jaccard_estime, detecte=detecte, taux_variantes=taux_variantes,
             n_faux_positifs=len(faux_positifs), contamination_naturelle=contamination_naturelle,
             rappel_contamination=rappel_contamination, faux_contamination=faux_contamination,
             N=N_tab, L=L_tab, params_loi=params_loi, params_4=params_4, prediction_dernier=prediction_dernier,
             echelle_tokens=np.array([r["tokens"] for r in resultats_echelle]),
             echelle_val=np.array([r["val"] for r in resultats_echelle]),
             frontiere_C=frontiere_C, frontiere_L=frontiere_L, frontiere_N=frontiere_N,
             moe_pas=np.array(resultats_moe["dense"]["pas"]),
             moe_val=np.array([resultats_moe[k]["val"] for k in ("dense", "moe_sans", "moe_avec")]),
             moe_charges_sans=resultats_moe["moe_sans"]["charges"], moe_charges_avec=resultats_moe["moe_avec"]["charges"],
             moe_n_total=np.array([resultats_moe[k]["n_total"] for k in ("dense", "moe_sans", "moe_avec")]),
             **_ech)
    print(f"Résultats bruts : {_dossier / ('resultats-smoke.npz' if SMOKE else 'resultats.npz')}")


@app.cell
def _(
    SMOKE,
    L_tab,
    erreur_estimation,
    erreur_prediction,
    faux_contamination,
    mo,
    rappel_bas,
    rappel_contamination,
    rappel_haut,
    resultats_echelle,
    resultats_moe,
):
    # --- Bilan : seuils du README, vérifiés en mode complet ---
    _ecarts = [r["perte_temoins"] - r["perte_fuite"] for r in resultats_echelle]
    _charge_max_sans = float(resultats_moe["moe_sans"]["charges"].max())
    _charge_max_avec = float(resultats_moe["moe_avec"]["charges"].max())
    seuils = {
        "MinHash : erreur moyenne d'estimation de J ≤ 0,05": erreur_estimation <= 0.05,
        "LSH : ≥ 95 % des variantes de J ≥ 0,8 détectées": rappel_haut >= 0.95,
        "LSH : ≤ 10 % des variantes de J < 0,5 détectées": rappel_bas <= 0.10,
        "contamination : rappel 100 %, aucune fausse alerte": rappel_contamination == 1.0 and faux_contamination == 0,
        "loi d'échelle : la perte baisse à chaque taille": bool(all(a > b for a, b in zip(L_tab, L_tab[1:]))),
        "loi d'échelle : plus gros modèle ≤ 1,75": L_tab[-1] <= 1.75,
        "prédiction du plus gros modèle à 0,05 près": erreur_prediction <= 0.05,
        "fuite : écart témoins − fuite croissant avec N (plus petit < plus gros)": _ecarts[0] < _ecarts[-1],
        "MoE avec équilibrage : aucun expert > 40 % des tokens": _charge_max_avec <= 0.40,
        "MoE avec équilibrage meilleur que dense": resultats_moe["moe_avec"]["val"][-1] < resultats_moe["dense"]["val"][-1],
    }
    for _k, _v in seuils.items():
        print(f"  [{'ok' if _v else 'ÉCHEC'}] {_k}")
    print("Écart témoins − fuite par taille :", ", ".join(f"{e:.3f}" for e in _ecarts))
    print(f"Charge maximale d'un expert : sans équilibrage {_charge_max_sans:.2f}, avec {_charge_max_avec:.2f}")
    _echecs = [k for k, v in seuils.items() if not v]
    if not SMOKE and _echecs:
        raise SystemExit("Seuils non atteints : " + " ; ".join(_echecs))
    mo.md(f"**Bilan** : {sum(seuils.values())}/{len(seuils)} seuils atteints"
          + (" (non vérifiés en smoke)" if SMOKE else "") + ".")


if __name__ == "__main__":
    app.run()
