# Exercice 19 : la tokenisation BPE, écrite à la main puis comparée à tiktoken.
#
#   make ex NUM=19              → éditeur marimo (interactif, bouton « Entraîner »)
#   make ex NUM=19 MODE=run     → script complet (vérifie les seuils du README)
#   make ex NUM=19 MODE=smoke   → test rapide, < 30 s
#
# Corpus : « Le tour du monde en quatre-vingts jours » de Jules Verne (1873) en français
# et dans la traduction anglaise de George M. Towle, téléchargés une fois depuis le
# projet Gutenberg (≈ 860 Ko, data/gutenberg/). Même histoire dans deux langues : on peut
# comparer ce que coûte chaque langue en tokens.
# Progression : caractères, octets et mots ; BPE sur octets en Python pur (version naïve),
# puis deux optimisations du comptage des paires (mêmes fusions, vérifié) ; encodage et
# décodage ; reproduction exacte de tiktoken (GPT-2) avec notre propre encodeur ;
# comparaison de tokenizers sur le français et l'anglais ; pourquoi compter les lettres
# est difficile pour un LLM. Tout tourne sur CPU, en Python pur (pas de GPU).

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import collections
    import hashlib
    import json
    import time
    import urllib.request

    import marimo as mo
    import matplotlib.pyplot as plt
    import regex  # module « regex » (dépendance de tiktoken) : il connaît \p{L}, \p{N}

    from ia_commun import (
        PALETTE,
        data_dir,
        is_smoke,
        sauvegarder_figure,
        seed_everything,
        style_commun,
    )

    return (
        PALETTE,
        collections,
        data_dir,
        hashlib,
        is_smoke,
        json,
        mo,
        plt,
        regex,
        sauvegarder_figure,
        seed_everything,
        style_commun,
        time,
        urllib,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Exercice 19 : découper un texte en tokens avec BPE

    **Objectif.** Écrire l'algorithme BPE (*byte pair encoding*) tel que l'utilisent
    GPT-2 et ses successeurs : partir des 256 octets possibles, fusionner à chaque étape
    la paire de symboles voisins la plus fréquente, et garder la liste des fusions. Puis
    l'accélérer, vérifier qu'il reproduit exactement le tokenizer de GPT-2, et mesurer ce
    que coûte le français par rapport à l'anglais.

    **Ce qui est optimisé.** Rien : BPE n'a **aucune fonction de perte**. Chaque fusion
    est un choix glouton (la paire la plus fréquente maintenant), sans retour en arrière
    ni critère global. (L'algorithme Unigram de Kudo, lui, maximise une vraisemblance
    par EM : voir « Pour aller plus loin » dans le README.)

    Plan :

    1. caractères, octets, mots : trois découpages naïfs et leurs défauts ;
    2. BPE naïf en Python pur ;
    3. deux optimisations du comptage des paires, qui donnent **les mêmes** fusions ;
    4. encoder, décoder, mesurer la compression ;
    5. notre encodeur avec les fusions de GPT-2 = `tiktoken`, au token près ;
    6. le français coûte plus cher ; compter les lettres de « strawberry ».
    """)


@app.cell
def _(is_smoke, seed_everything, style_commun, time):
    T0 = time.perf_counter()  # chronomètre de l'ensemble des calculs
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)  # BPE est déterministe ; la graine est fixée par principe
    style_commun()
    config = dict(
        part_test=0.2,                          # derniers 20 % de chaque livre : texte de test
        taille_naif=20_000 if SMOKE else 100_000,  # caractères pour la comparaison des 3 versions
        fusions_naif=60 if SMOKE else 256,
        fusions=300 if SMOKE else 2000,         # vocabulaire final : 256 + fusions
        jalons=[0, 10, 50, 100, 200, 500, 1000, 2000],  # nombres de fusions pour la courbe
        taille_tiktoken=30_000 if SMOKE else None,  # None = livres entiers
    )
    if SMOKE:
        config["jalons"] = [k for k in config["jalons"] if k <= config["fusions"]]
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SMOKE, T0, config


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Le corpus

    Deux fichiers texte du projet Gutenberg, téléchargés une fois dans `data/gutenberg/`.
    On retire l'en-tête et la licence de Gutenberg, on passe les fins de ligne en `\n`
    et on remplace les apostrophes et guillemets typographiques anglais (`’ ‘ “ ”`,
    3 octets chacun en UTF-8) par leurs équivalents ASCII : le texte français n'utilise
    que `'`, et l'anglais paierait sinon 3 octets par apostrophe. Les guillemets
    français `« »` (2 octets) restent : ils font partie de l'écriture du français.
    """)


@app.cell
def _(config, data_dir, hashlib, urllib):
    # Numéros des livres sur gutenberg.org : 800 (français), 103 (traduction de G. M. Towle)
    LIVRES = {"fr": 800, "en": 103}

    def telecharger_livre(numero: int) -> str:
        dossier = data_dir() / "gutenberg"
        dossier.mkdir(parents=True, exist_ok=True)
        chemin = dossier / f"pg{numero}.txt"
        if not chemin.exists():
            url = f"https://www.gutenberg.org/ebooks/{numero}.txt.utf-8"
            print(f"Téléchargement de {url} → {chemin}")
            requete = urllib.request.Request(url, headers={"User-Agent": "ia-experiments (exercice 19)"})
            with urllib.request.urlopen(requete, timeout=60) as reponse:
                contenu = reponse.read()
            tmp = chemin.with_name(chemin.name + ".part")
            tmp.write_bytes(contenu)
            tmp.replace(chemin)  # pas de fichier à moitié écrit dans le cache
        brut = chemin.read_text(encoding="utf-8")
        # Gutenberg met parfois ses fichiers à jour : on affiche l'empreinte pour comparer
        print(f"pg{numero}.txt : sha1 {hashlib.sha1(brut.encode()).hexdigest()[:12]}")
        return brut

    def nettoyer(brut: str) -> str:
        debut = brut.index("\n", brut.index("*** START OF"))
        fin = brut.index("*** END OF")
        texte = brut[debut:fin].replace("\r\n", "\n")
        # Apostrophes et guillemets anglais typographiques → ASCII (3 octets → 1) ;
        # les guillemets français « » restent : ils font partie du texte français.
        for typo, ascii_ in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"')):
            texte = texte.replace(typo, ascii_)
        return texte.strip()

    textes = {langue: nettoyer(telecharger_livre(n)) for langue, n in LIVRES.items()}
    # Découpage entraînement / test : le test est la FIN du livre (jamais vue par BPE)
    train, test = {}, {}
    for _l, _t in textes.items():
        _coupe = int(len(_t) * (1 - config["part_test"]))
        _coupe = _t.index("\n", _coupe)  # couper à une fin de ligne
        train[_l], test[_l] = _t[:_coupe], _t[_coupe:]
    for _l in textes:
        print(f"{_l} : {len(textes[_l]):,} caractères, {len(textes[_l].encode()):,} octets, "
              f"{len(textes[_l].split()):,} mots ; entraînement {len(train[_l]):,}, test {len(test[_l]):,}"
              .replace(",", " "))
    return test, textes, train


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Caractères, octets, mots

    Trois découpages évidents, trois défauts :

    - **caractères** : petit vocabulaire, mais des séquences longues (un token par
      lettre) et un vocabulaire qui n'est pas vraiment fermé (un caractère chinois ou un
      emoji jamais vu reste inconnu) ;
    - **octets UTF-8** : vocabulaire fermé de 256 symboles, rien n'est jamais inconnu ;
      mais « é » prend 2 octets, un emoji 4, et les séquences sont encore plus longues ;
    - **mots** : séquences courtes, mais vocabulaire énorme et des **mots inconnus** dans
      tout texte nouveau (noms propres, formes conjuguées rares).
    """)


@app.cell
def _(collections, regex, test, train):
    # Découpage en mots : lettres (avec accents) ou chiffres ; la ponctuation est un mot.
    MOT = r"\p{L}+|\p{N}+|[^\s\p{L}\p{N}]"
    naifs = {}
    for _l in train:
        _mots_train = regex.findall(MOT, train[_l])
        _mots_test = regex.findall(MOT, test[_l])
        _vocab_mots = collections.Counter(_mots_train)
        _inconnus = sum(1 for _m in _mots_test if _m not in _vocab_mots)
        naifs[_l] = dict(
            caracteres_vocab=len(set(train[_l])),
            caracteres_par_1000=1000.0,  # par définition
            octets_par_1000=1000 * len(test[_l].encode()) / len(test[_l]),
            mots_vocab=len(_vocab_mots),
            mots_par_1000=1000 * len(_mots_test) / len(test[_l]),
            mots_inconnus=_inconnus / len(_mots_test),
            exemples_inconnus=sorted({_m for _m in _mots_test if _m not in _vocab_mots
                                      and _m.isalpha()}, key=len)[-8:],
        )
        _n = naifs[_l]
        print(f"[{_l}] caractères : vocabulaire {_n['caracteres_vocab']}, 1 000 tokens / 1 000 car. ; "
              f"octets : 256, {_n['octets_par_1000']:.0f} tokens / 1 000 car. ; "
              f"mots : vocabulaire {_n['mots_vocab']}, {_n['mots_par_1000']:.0f} tokens / 1 000 car., "
              f"{_n['mots_inconnus']:.1%} de mots inconnus sur le test")
        print(f"     exemples de mots inconnus : {_n['exemples_inconnus']}")
    return (naifs,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. BPE naïf, en Python pur

    **Pré-découpage.** Comme GPT-2, on coupe d'abord le texte en morceaux avec une
    expression régulière (un mot précédé de son espace, un nombre, une ponctuation…).
    Les fusions ne traversent jamais la frontière d'un morceau : « chat. » ne
    deviendra pas un seul token avec le point. Puis chaque morceau devient sa suite
    d'octets UTF-8 (entiers de 0 à 255).

    **Boucle.** Pour la fusion $k$ (le nouveau symbole reçoit l'identifiant $256 + k$) :

    1. compter toutes les paires de symboles voisins dans tous les morceaux ;
    2. prendre la plus fréquente (à égalité, la plus petite paire d'identifiants, pour
       que le résultat soit reproductible) ;
    3. la remplacer partout par le nouveau symbole.

    C'est tout. Le coût : à chaque fusion, on relit **tout** le corpus.
    """)


@app.cell
def _(collections, regex):
    # Motif de pré-découpage de GPT-2 (fichier encoder.py du dépôt openai/gpt-2)
    MOTIF_GPT2 = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def pre_decouper(texte: str) -> list[str]:
        return regex.findall(MOTIF_GPT2, texte)

    def meilleure_paire(stats: dict) -> tuple[int, int]:
        """Paire la plus fréquente ; à égalité, la plus petite (a, b) : résultat déterministe."""
        return max(stats, key=lambda p: (stats[p], -p[0], -p[1]))

    def fusionner(seq: list[int], paire: tuple[int, int], nouveau: int) -> list[int]:
        """Remplace chaque occurrence de `paire` dans `seq` (de gauche à droite) par `nouveau`."""
        sortie, i = [], 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == paire[0] and seq[i + 1] == paire[1]:
                sortie.append(nouveau)
                i += 2
            else:
                sortie.append(seq[i])
                i += 1
        return sortie

    def bpe_naif(texte: str, n_fusions: int) -> list[tuple[tuple[int, int], int]]:
        """Renvoie la liste des fusions [((a, b), effectif), …] dans l'ordre où elles sont apprises."""
        morceaux = [list(m.encode("utf-8")) for m in pre_decouper(texte)]
        fusions = []
        for k in range(n_fusions):
            stats = collections.Counter()
            for m in morceaux:  # 1. compter TOUTES les paires, à chaque fusion
                for p in zip(m, m[1:]):
                    stats[p] += 1
            if not stats:
                break
            paire = meilleure_paire(stats)  # 2. la plus fréquente
            fusions.append((paire, stats[paire]))
            morceaux = [fusionner(m, paire, 256 + k) for m in morceaux]  # 3. remplacer
        return fusions

    print("exemple :", pre_decouper("Phileas Fogg n'avait pas 20 000 livres, mais 40 000 !"))
    return bpe_naif, fusionner, meilleure_paire, pre_decouper


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Optimiser le comptage des paires

    **Version 2, morceaux dédoublonnés.** Le corpus contient des milliers de fois le
    morceau « ␣de ». On ne garde que les morceaux **distincts** avec leur nombre
    d'occurrences, et chaque paire compte pour ce nombre. Même résultat, beaucoup moins
    de travail.

    **Version 3, mise à jour incrémentale.** Une fusion ne touche que les morceaux qui
    contiennent la paire fusionnée. On garde les effectifs de toutes les paires et un
    index « paire → morceaux qui la contiennent » ; après une fusion, on retire les
    paires des seuls morceaux modifiés et on ajoute leurs nouvelles paires. C'est
    l'idée des implémentations rapides (celle de Sennrich et al. en Python, et en Rust
    la bibliothèque `tokenizers` de Hugging Face).

    Les trois versions doivent donner **exactement** les mêmes fusions : c'est le test.
    """)


@app.cell
def _(collections, fusionner, meilleure_paire, pre_decouper):
    def bpe_dedoublonne(texte: str, n_fusions: int):
        freq = collections.Counter(pre_decouper(texte))
        morceaux = [list(m.encode("utf-8")) for m in freq]
        effectifs = list(freq.values())
        fusions = []
        for k in range(n_fusions):
            stats = collections.Counter()
            for m, c in zip(morceaux, effectifs):
                for p in zip(m, m[1:]):
                    stats[p] += c  # une paire compte autant de fois que son morceau
            if not stats:
                break
            paire = meilleure_paire(stats)
            fusions.append((paire, stats[paire]))
            morceaux = [fusionner(m, paire, 256 + k) if len(m) > 1 else m for m in morceaux]
        return fusions

    def bpe_incremental(texte: str, n_fusions: int):
        freq = collections.Counter(pre_decouper(texte))
        morceaux = [list(m.encode("utf-8")) for m in freq]
        effectifs = list(freq.values())
        stats = collections.Counter()
        index = collections.defaultdict(set)  # paire → indices des morceaux qui la contiennent
        for i, (m, c) in enumerate(zip(morceaux, effectifs)):
            for p in zip(m, m[1:]):
                stats[p] += c
                index[p].add(i)
        fusions = []
        for k in range(n_fusions):
            if not stats:
                break
            paire = meilleure_paire(stats)
            fusions.append((paire, stats[paire]))
            nouveau = 256 + k
            for i in list(index[paire]):  # seuls ces morceaux changent
                m, c = morceaux[i], effectifs[i]
                for p in zip(m, m[1:]):  # retirer les anciennes paires du morceau…
                    stats[p] -= c
                    if stats[p] == 0:
                        del stats[p]
                m = fusionner(m, paire, nouveau)
                morceaux[i] = m
                for p in zip(m, m[1:]):  # … et ajouter les nouvelles
                    stats[p] += c
                    index[p].add(i)  # (un index peut garder des morceaux périmés : sans effet)
            del index[paire]
        return fusions

    return bpe_dedoublonne, bpe_incremental


@app.cell
def _(mo):
    # Le bouton empêche les calculs longs de se relancer à chaque modification d'une cellule amont.
    bouton = mo.ui.run_button(label="Entraîner")
    bouton
    return (bouton,)


@app.cell
def _(bouton, bpe_dedoublonne, bpe_incremental, bpe_naif, config, mo, time, train):
    # En script (make ex MODE=run|smoke), il n'y a personne pour cliquer : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Entraîner » pour lancer l'apprentissage des fusions.*"))

    _extrait = train["fr"][: config["taille_naif"]]
    _n = config["fusions_naif"]
    chronos = {}
    _resultats = {}
    for _nom, _f in [("naïf", bpe_naif), ("dédoublonné", bpe_dedoublonne), ("incrémental", bpe_incremental)]:
        _t = time.perf_counter()
        _resultats[_nom] = _f(_extrait, _n)
        chronos[_nom] = time.perf_counter() - _t
        print(f"{_nom:>12} : {_n} fusions sur {len(_extrait.encode()):,} octets en {chronos[_nom]:.2f} s"
              .replace(",", " "))
    memes_fusions = _resultats["naïf"] == _resultats["dédoublonné"] == _resultats["incrémental"]
    acceleration = chronos["naïf"] / chronos["incrémental"]
    print(f"mêmes fusions dans les trois versions : {memes_fusions} ; "
          f"accélération naïf → incrémental : × {acceleration:.0f}")

    # Entraînement complet (version incrémentale) sur la partie « entraînement » de chaque livre
    fusions = {}
    chronos_complets = {}
    for _l in ("fr", "en"):
        _t = time.perf_counter()
        fusions[_l] = bpe_incremental(train[_l], config["fusions"])
        chronos_complets[_l] = time.perf_counter() - _t
        print(f"BPE-{_l} : {len(fusions[_l])} fusions sur {len(train[_l].encode()):,} octets en "
              f"{chronos_complets[_l]:.1f} s".replace(",", " "))
    return acceleration, chronos, chronos_complets, fusions, memes_fusions


@app.cell
def _(fusions):
    def table_octets(fus) -> dict[int, bytes]:
        """Identifiant → suite d'octets qu'il représente (256 octets de base + fusions)."""
        vocab = {i: bytes([i]) for i in range(256)}
        for k, ((a, b), _c) in enumerate(fus):
            vocab[256 + k] = vocab[a] + vocab[b]
        return vocab

    def lisible(b: bytes) -> str:
        """Affiche un token : texte si les octets forment de l'UTF-8 complet, sinon ⟨C3⟩."""
        try:
            return b.decode("utf-8").replace(" ", "␣").replace("\n", "↵")
        except UnicodeDecodeError:
            return "".join(f"⟨{x:02X}⟩" if x >= 128 else chr(x) for x in b).replace(" ", "␣")

    vocab_octets = {_l: table_octets(fusions[_l]) for _l in fusions}
    for _l in ("fr", "en"):
        _v = vocab_octets[_l]
        print(f"BPE-{_l}, 20 premières fusions :",
              " ".join(f"{lisible(_v[256 + k])}" for k in range(min(20, len(fusions[_l])))))
        print(f"BPE-{_l}, 10 dernières :",
              " ".join(f"{lisible(_v[256 + k])}" for k in range(len(fusions[_l]) - 10, len(fusions[_l]))))
    return lisible, table_octets, vocab_octets


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Encoder, décoder, mesurer

    **Encoder** un texte neuf : pré-découper, puis dans chaque morceau appliquer les
    fusions **dans l'ordre où elles ont été apprises** (tant qu'une paire présente a un
    rang, fusionner celle de plus petit rang). **Décoder** : concaténer les octets des
    tokens et lire l'UTF-8. Aucun texte ne peut produire de token inconnu : au pire,
    un caractère rare reste en octets.

    Mesure de la compression : nombre de tokens pour 1 000 caractères du texte de
    **test**, en fonction du nombre de fusions.
    """)


@app.cell
def _(fusionner, pre_decouper):
    def encoder_morceau(octets: bytes, rangs: dict) -> list[int]:
        """BPE sur un morceau : fusionne la paire de plus petit rang tant qu'il y en a une.

        `rangs` associe une paire (a, b) d'identifiants au nouvel identifiant ; comme les
        identifiants sont attribués dans l'ordre d'apprentissage, le plus petit est la
        fusion apprise le plus tôt.
        """
        seq = list(octets)
        while len(seq) > 1:
            # paires voisines qui ont une fusion : on garde celle apprise le plus tôt
            candidats = [(rangs[p], p) for p in zip(seq, seq[1:]) if p in rangs]
            if not candidats:
                break
            nouveau, paire = min(candidats)
            seq = fusionner(seq, paire, nouveau)  # la fonction de la partie 2
        return seq

    def rangs_depuis(fus, n: int | None = None) -> dict:
        """Les n premières fusions sous forme {(a, b): nouvel identifiant}."""
        return {p: 256 + k for k, (p, _c) in enumerate(fus[:n])}

    def encoder(texte: str, rangs: dict) -> list[int]:
        cache = {}  # un même morceau (« ␣de ») s'encode toujours pareil
        ids = []
        for m in pre_decouper(texte):
            if m not in cache:
                cache[m] = encoder_morceau(m.encode("utf-8"), rangs)
            ids.extend(cache[m])
        return ids

    def decoder(ids: list[int], vocab: dict[int, bytes]) -> str:
        return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

    return decoder, encoder, rangs_depuis


@app.cell
def _(config, decoder, encoder, fusions, lisible, rangs_depuis, test, vocab_octets):
    # Aller-retour exact, y compris sur des caractères jamais vus à l'entraînement
    _rangs_fr = rangs_depuis(fusions["fr"])
    _bizarre = "Ça coûte 12,50 € 🚂 — 東京 ; naïveté, œuvre."
    allers_retours = all(decoder(encoder(_t, _rangs_fr), vocab_octets["fr"]) == _t
                         for _t in (test["fr"], test["en"], _bizarre))
    print(f"décoder(encoder(texte)) == texte (test fr, test en, texte exotique) : {allers_retours}")
    print("texte exotique :", [lisible(vocab_octets["fr"][i]) for i in encoder(_bizarre, _rangs_fr)])

    # Compression : tokens pour 1 000 caractères de test, pour chaque nombre de fusions
    courbe = {}
    for _src in ("fr", "en"):  # langue du BPE
        for _cible in ("fr", "en"):  # langue du texte de test
            courbe[f"{_src}->{_cible}"] = [
                1000 * len(encoder(test[_cible], rangs_depuis(fusions[_src], k))) / len(test[_cible])
                for k in config["jalons"]
            ]
    for _cle, _v in courbe.items():
        print(f"BPE-{_cle.replace('->', ' sur texte ')} :",
              ", ".join(f"{k} fusions → {v:.0f}" for k, v in zip(config["jalons"], _v)))

    # Fusions successives sur une phrase de test (figure du chapitre)
    PHRASE = "Les voyageurs arrivèrent enfin à Londres."
    phrase_etapes = {
        k: [lisible(vocab_octets["fr"][i]) for i in encoder(PHRASE, rangs_depuis(fusions["fr"], k))]
        for k in config["jalons"]
    }
    for _k, _toks in phrase_etapes.items():
        print(f"{_k:>5} fusions, {len(_toks):>2} tokens : {' | '.join(_toks)}")
    return PHRASE, allers_retours, courbe, phrase_etapes


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Notre encodeur avec les fusions de GPT-2 = tiktoken

    `tiktoken` (OpenAI) publie les fusions de GPT-2 sous forme de **rangs** : chaque
    suite d'octets du vocabulaire a un numéro, et l'encodeur fusionne toujours la paire
    voisine dont la concaténation a le plus petit rang. C'est exactement notre
    `encoder_morceau`, à ceci près que les rangs portent sur des suites d'octets. On
    réécrit l'encodeur pour ces rangs et on compare, token par token, avec
    `tiktoken.get_encoding("gpt2")`. Les fichiers de rangs sont téléchargés une fois
    dans `data/tiktoken/`.
    """)


@app.cell
def _(SMOKE, pre_decouper, test):
    import tiktoken

    gpt2 = tiktoken.get_encoding("gpt2")
    _rangs = gpt2._mergeable_ranks  # dict : suite d'octets → rang (= identifiant du token)

    def encoder_gpt2_maison(texte: str) -> list[int]:
        ids = []
        for m in pre_decouper(texte):
            parts = [bytes([o]) for o in m.encode("utf-8")]
            while len(parts) > 1:
                # rang de chaque paire voisine fusionnée ; on fusionne la plus petite
                rangs_paires = [_rangs.get(parts[i] + parts[i + 1]) for i in range(len(parts) - 1)]
                valides = [(r, i) for i, r in enumerate(rangs_paires) if r is not None]
                if not valides:
                    break
                _rang, i = min(valides)
                parts = parts[:i] + [parts[i] + parts[i + 1]] + parts[i + 2:]
            ids.extend(_rangs[p] for p in parts)
        return ids

    _n = 5_000 if SMOKE else 20_000  # caractères de test de chaque langue
    _echantillon = test["fr"][:_n] + "\n" + test["en"][:_n]
    _a, _b = encoder_gpt2_maison(_echantillon), gpt2.encode(_echantillon)
    identique_tiktoken = _a == _b
    comparaison_tiktoken = {"caracteres": len(_echantillon), "tokens": len(_b)}
    print(f"{len(_b)} tokens ; notre encodeur == tiktoken gpt2 : {identique_tiktoken}")
    return comparaison_tiktoken, encoder_gpt2_maison, gpt2, identique_tiktoken, tiktoken


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Le français coûte plus cher

    Même histoire, deux langues. Pour chaque tokenizer, on compte les tokens du livre
    entier et la **fertilité** (tokens par mot, mot = suite séparée par des espaces).
    Tokenizers comparés : GPT-2 (`r50k_base`, 50 257 tokens, 2019), GPT-4
    (`cl100k_base`, 100 277), GPT-4o (`o200k_base`, 200 019), et nos deux BPE à
    2 256 symboles, appris l'un sur le français, l'autre sur l'anglais.
    """)


@app.cell
def _(config, encoder, fusions, gpt2, rangs_depuis, textes, tiktoken):
    _limite = config["taille_tiktoken"]
    _txt = {l: (t[:_limite] if _limite else t) for l, t in textes.items()}
    tokenizers = {
        "GPT-2 (r50k_base)": gpt2.encode,
        "GPT-4 (cl100k_base)": tiktoken.get_encoding("cl100k_base").encode,
        "GPT-4o (o200k_base)": tiktoken.get_encoding("o200k_base").encode,
        f"BPE-fr ({256 + len(fusions['fr'])})": lambda t: encoder(t, rangs_depuis(fusions["fr"])),
        f"BPE-en ({256 + len(fusions['en'])})": lambda t: encoder(t, rangs_depuis(fusions["en"])),
    }
    langues = {}
    ratio_octets = len(_txt["fr"].encode()) / len(_txt["en"].encode())
    print(f"le texte français fait {ratio_octets:.2f} fois la taille (en octets) de sa traduction")
    for _nom, _enc in tokenizers.items():
        _n = {l: len(_enc(t)) for l, t in _txt.items()}
        langues[_nom] = dict(
            tokens=_n,
            fertilite={l: _n[l] / len(_txt[l].split()) for l in _txt},
            par_1000_octets={l: 1000 * _n[l] / len(_txt[l].encode()) for l in _txt},
            ratio_fr_en=_n["fr"] / _n["en"],
        )
        # Le texte français est lui-même plus long (en octets) que sa traduction : on
        # sépare l'effet du texte de celui du tokenizer en divisant par le rapport des octets.
        langues[_nom]["surcout_tokenizer"] = langues[_nom]["ratio_fr_en"] / ratio_octets
        _f = langues[_nom]["fertilite"]
        print(f"{_nom:>22} : fr {_n['fr']:>7} tokens ({_f['fr']:.2f}/mot), en {_n['en']:>7} "
              f"({_f['en']:.2f}/mot) ; français / anglais = {langues[_nom]['ratio_fr_en']:.2f}, "
              f"à longueur égale en octets : {langues[_nom]['surcout_tokenizer']:.2f}")
    mots = {l: len(t.split()) for l, t in _txt.items()}
    print(f"mots (séparés par des espaces) : {mots} ; octets : "
          f"{ {l: len(t.encode()) for l, t in _txt.items()} }")
    return langues, mots, ratio_octets


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Pourquoi un LLM compte mal les lettres

    Le modèle ne voit pas des lettres mais des **identifiants** de tokens. « strawberry »
    est un ou quelques tokens ; le nombre de « r » n'est écrit nulle part dans ces
    identifiants : le modèle doit l'avoir appris par cœur, token par token.
    """)


@app.cell
def _(gpt2, tiktoken):
    _o200k = tiktoken.get_encoding("o200k_base")
    lettres = {}
    for _mot in (" strawberry", " anticonstitutionnellement", " 1234567"):
        lettres[_mot.strip()] = {
            "gpt2": [gpt2.decode([i]) for i in gpt2.encode(_mot)],
            "o200k": [_o200k.decode([i]) for i in _o200k.encode(_mot)],
            "ids_o200k": _o200k.encode(_mot),
        }
        print(f"{_mot!r} : GPT-2 {lettres[_mot.strip()]['gpt2']} ; GPT-4o {lettres[_mot.strip()]['o200k']} "
              f"= identifiants {lettres[_mot.strip()]['ids_o200k']}")
    return (lettres,)


@app.cell
def _(PALETTE, SMOKE, config, courbe, langues, mo, naifs, phrase_etapes, plt, sauvegarder_figure):
    # --- Figures de l'exercice (les figures du site sont redessinées par site/scripts/figures) ---
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    _styles = {"fr->fr": (PALETTE[0], "-"), "en->fr": (PALETTE[0], "--"),
               "en->en": (PALETTE[1], "-"), "fr->en": (PALETTE[1], "--")}
    for _cle, (_c, _ls) in _styles.items():
        _src, _cible = _cle.split("->")
        _ax1.plot([256 + k for k in config["jalons"]], courbe[_cle], _ls, color=_c, marker="o", ms=3,
                  label=f"BPE appris sur {_src}, texte {_cible}")
    for _l, _c, _dy in (("fr", PALETTE[0], 8), ("en", PALETTE[1], -14)):
        _ax1.scatter([naifs[_l]["mots_vocab"]], [naifs[_l]["mots_par_1000"]], color=_c, marker="s")
        _ax1.annotate(f"mots entiers ({_l}) : {naifs[_l]['mots_inconnus']:.1%} inconnus",
                      (naifs[_l]["mots_vocab"], naifs[_l]["mots_par_1000"]), fontsize=8,
                      xytext=(-8, _dy), textcoords="offset points", ha="right")
    _ax1.set(xscale="log", xlabel="taille du vocabulaire", ylabel="tokens pour 1 000 caractères (test)",
             title="Compression selon le nombre de fusions")
    _ax1.legend(fontsize=8)
    _noms = list(langues)
    _x = range(len(_noms))
    _ax2.bar([i - 0.2 for i in _x], [langues[n]["fertilite"]["fr"] for n in _noms], 0.4,
             label="français", color=PALETTE[0])
    _ax2.bar([i + 0.2 for i in _x], [langues[n]["fertilite"]["en"] for n in _noms], 0.4,
             label="anglais", color=PALETTE[1])
    _ax2.set_xticks(list(_x), [n.replace(" (", "\n(") for n in _noms], fontsize=8)
    _ax2.set(ylabel="tokens par mot", title="Même livre, deux langues")
    _ax2.legend()
    _dossier = mo.notebook_dir() / "figures"
    _ch = sauvegarder_figure(_fig, "compression-smoke" if SMOKE else "compression", _dossier)
    print(f"Figure : {_ch[0]}")
    print("\n".join(f"{k:>5} : {' | '.join(v)}" for k, v in phrase_etapes.items()))
    _fig


@app.cell
def _(
    PHRASE,
    SMOKE,
    T0,
    acceleration,
    allers_retours,
    chronos,
    chronos_complets,
    comparaison_tiktoken,
    config,
    courbe,
    fusions,
    identique_tiktoken,
    json,
    langues,
    lettres,
    lisible,
    memes_fusions,
    mo,
    mots,
    naifs,
    phrase_etapes,
    ratio_octets,
    time,
    vocab_octets,
):
    # --- Bilan : seuils chiffrés du README. En smoke, on vérifie seulement que tout tourne.
    duree_totale = time.perf_counter() - T0
    _gpt2 = langues["GPT-2 (r50k_base)"]
    _o200k = langues["GPT-4o (o200k_base)"]
    _nf = str(256 + len(fusions["fr"]))
    _bpe_fr = langues[f"BPE-fr ({_nf})"]
    _bpe_en = langues[f"BPE-en ({256 + len(fusions['en'])})"]
    bilan = dict(
        memes_fusions=memes_fusions,
        chronos_naif=chronos,
        acceleration=acceleration,
        chronos_complets=chronos_complets,
        allers_retours=allers_retours,
        identique_tiktoken=identique_tiktoken,
        comparaison_tiktoken=comparaison_tiktoken,
        jalons=config["jalons"],
        courbe=courbe,
        naifs=naifs,
        langues=langues,
        mots=mots,
        ratio_octets=ratio_octets,
        phrase=PHRASE,
        phrase_etapes={str(k): v for k, v in phrase_etapes.items()},
        premieres_fusions={l: [lisible(vocab_octets[l][256 + k]) for k in range(30)] for l in fusions},
        dernieres_fusions={l: [lisible(vocab_octets[l][256 + k])
                               for k in range(max(0, len(fusions[l]) - 10), len(fusions[l]))]
                           for l in fusions},
        # rang (1 = première fusion) des tokens de quelques noms propres du livre, ou None
        rangs_noms_propres={l: {_nom: next((k + 1 for k in range(len(fusions[l]))
                                             if vocab_octets[l][256 + k] == _nom.encode()), None)
                                for _nom in (" Phileas", " Fogg", " Passepartout", " Fix", " Londres", " London")}
                            for l in fusions},
        fusions_effectifs={l: [c for _p, c in fusions[l]] for l in fusions},
        lettres=lettres,
        duree_totale=duree_totale,
    )
    _seuils = [
        ("les trois versions de BPE donnent les mêmes fusions", memes_fusions),
        ("décoder(encoder(x)) == x", allers_retours),
        ("notre encodeur == tiktoken gpt2", identique_tiktoken),
        ("version incrémentale au moins 10 fois plus rapide que la naïve", acceleration >= 10),
        ("BPE-fr : ≤ 350 tokens pour 1 000 caractères de test français",
         courbe["fr->fr"][-1] <= 350),
        ("BPE-fr meilleur que BPE-en sur le français (≥ 15 % de tokens en moins)",
         _bpe_fr["tokens"]["fr"] <= 0.85 * _bpe_en["tokens"]["fr"]),
        ("GPT-2 : à longueur égale en octets, le français coûte ≥ 1,3 fois plus de tokens",
         _gpt2["surcout_tokenizer"] >= 1.3),
        ("GPT-4o réduit l'écart (surcoût du français plus petit que GPT-2)",
         _o200k["surcout_tokenizer"] < _gpt2["surcout_tokenizer"]),
    ]
    for _nom, _ok in _seuils:
        print(f"[{'ok' if _ok else 'ÉCHEC'}] {_nom}")
    _sortie = mo.notebook_dir() / "figures" / ("resultats-smoke.json" if SMOKE else "resultats.json")
    _sortie.write_text(json.dumps(bilan, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"Résultats : {_sortie} ; durée totale {duree_totale:.1f} s")
    _echecs = [n for n, ok in _seuils if not ok]
    if not SMOKE and _echecs:
        raise SystemExit(f"Seuils non atteints : {_echecs}")
    mo.md(f"**Bilan** : {sum(ok for _n, ok in _seuils)}/{len(_seuils)} seuils atteints, "
          f"{duree_totale:.1f} s de calcul.")


if __name__ == "__main__":
    app.run()
