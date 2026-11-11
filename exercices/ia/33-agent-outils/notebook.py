# Exercice 33 : un agent minimal avec deux outils, puis un LLM juge face au jugement humain.
#
#   make ex NUM=33              → éditeur marimo (bouton « Lancer l'agent et le juge »)
#   make ex NUM=33 MODE=run     → run complet en script (20 tâches, 20 réponses jugées), sous verrou GPU
#   make ex NUM=33 MODE=smoke   → test rapide (1 tâche, 2 réponses jugées, 1 paire), < 30 s
#
# Prérequis : Ollama lancé en local (l'application, ou `ollama serve`) et le modèle
# téléchargé une fois : `ollama pull qwen3.5:2b` (≈ 3 Go). Rien n'est entraîné : on
# interroge un petit LLM figé, avec ou sans outils, puis on lui fait noter des réponses.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import ast
    import json
    import math
    import operator
    import re
    import time
    import unicodedata
    from collections import Counter

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from ia_commun import PALETTE, is_smoke, sauvegarder_figure, seed_everything, style_commun

    return (
        Counter,
        PALETTE,
        ast,
        is_smoke,
        json,
        math,
        mo,
        np,
        operator,
        plt,
        re,
        sauvegarder_figure,
        seed_everything,
        style_commun,
        time,
        unicodedata,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Exercice 33 : un agent avec des outils, et un LLM qui juge

    **Partie A : l'agent.** Un petit LLM local (`qwen3.5:2b`, via Ollama) doit répondre à
    20 questions chiffrées de trois sortes :

    - **calcul** (« 3847 × 29 − 1234 ? ») : le modèle connaît la méthode mais se trompe
      sur les chiffres ;
    - **recherche** (« combien de stations ? ») : la réponse est dans une petite
      documentation (`corpus/`, dix fiches sur une coopérative de vélos **fictive**) que le
      modèle n'a jamais vue ;
    - **composée** : il faut chercher deux chiffres, puis calculer.

    Trois façons de faire, de la plus simple à la plus outillée :

    1. **sans outils** : le modèle raisonne par écrit puis donne sa réponse ;
    2. **ReAct écrit à la main** : on décrit deux outils dans la consigne
       (`calculatrice[…]`, `chercher[…]`), le modèle écrit « Pensée / Action », **notre
       code** lit l'action, exécute l'outil et renvoie « Observation : … » ; on boucle ;
    3. **appel de fonctions natif** (*function calling*) : on décrit les mêmes outils par un
       schéma JSON, le modèle renvoie un appel structuré que l'API d'Ollama sépare du texte.

    **Partie B : le juge.** Le même modèle note 20 réponses écrites à la main
    (`juge.json`), dont on connaît le verdict humain : on mesure l'accord avec le jugement
    humain, avec puis sans réponse de référence, puis la **cohérence** du juge quand on
    échange l'ordre de deux réponses à comparer (biais de position).

    **Fonction de perte : aucune.** Rien n'est entraîné : le modèle est figé (il a été
    pré-entraîné à prédire le token suivant, chapitre 20). Ce qui est mesuré : la part de
    réponses justes, le nombre d'appels au modèle, l'accord juge/humain.
    """)


@app.cell
def _(is_smoke, seed_everything, style_commun):
    # --- Configuration : seul endroit où le mode smoke réduit le travail ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        modele="qwen3.5:2b",
        taches=["M1"] if SMOKE else None,              # None = les 20 tâches
        n_reponses_jugees=2 if SMOKE else 20,          # réponses notées par le juge
        n_paires=1 if SMOKE else 10,                   # paires comparées dans les deux ordres
        max_appels=3 if SMOKE else 6,                  # appels au LLM par tâche, au plus
        tokens_sans_outils=96 if SMOKE else 400,       # budget du raisonnement sans outils
        tokens_etape=96 if SMOKE else 256,             # budget d'une étape d'agent
        temperature=0.0,                               # décodage glouton : runs reproductibles
        k_passages=3,                                  # fiches renvoyées par chercher[…]
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SEED, SMOKE, config


@app.cell
def _(config, json, mo):
    # --- Données : la documentation fictive, les tâches, les réponses à juger ---
    DOSSIER = mo.notebook_dir()
    documents = []
    for _chemin in sorted((DOSSIER / "corpus").glob("*.md")):
        _texte = _chemin.read_text(encoding="utf-8").strip()
        _titre = _texte.splitlines()[0].lstrip("# ").strip()
        documents.append({"fichier": _chemin.name, "titre": _titre, "texte": _texte})
    toutes_taches = json.loads((DOSSIER / "taches.json").read_text(encoding="utf-8"))
    taches = [t for t in toutes_taches if config["taches"] is None or t["id"] in config["taches"]]
    JUGE = json.loads((DOSSIER / "juge.json").read_text(encoding="utf-8"))
    taches_par_id = {t["id"]: t for t in toutes_taches}
    print(f"{len(documents)} fiches dans corpus/, {len(taches)} tâches, "
          f"{len(JUGE['reponses'])} réponses à juger, {len(JUGE['paires'])} paires")
    return DOSSIER, JUGE, documents, taches, taches_par_id


@app.cell
def _(Counter, config, documents, math, re, unicodedata):
    # --- Outil 1 : chercher[mots-clés], une recherche par mots-clés écrite à la main ---
    # Volontairement simple : on compte les mots de la requête présents dans chaque fiche,
    # pondérés par leur rareté (un mot présent partout, comme « Roulevent », ne départage
    # rien). Racinisation grossière : un mot est réduit à ses 4 premières lettres, pour que
    # « fondation » trouve « fondée » et « abonnés » trouve « abonnement ». C'est un TF-IDF
    # réduit à l'essentiel ; BM25, les embeddings et le reranking sont au chapitre 32 (RAG).
    MOTS_VIDES = {
        "le", "la", "les", "un", "une", "des", "du", "de", "d", "l", "a", "au", "aux", "et",
        "ou", "en", "est", "sont", "par", "pour", "sur", "dans", "avec", "sans", "ce", "cette",
        "ces", "son", "sa", "ses", "leur", "qui", "que", "quoi", "quel", "quelle", "combien",
        "comment", "il", "elle", "ils", "elles", "on", "se", "ne", "pas", "plus", "t", "fin",
    }

    def mots(texte: str) -> list[str]:
        # minuscules, accents retirés (« électrique » = « electrique »), mots vides retirés
        t = unicodedata.normalize("NFKD", texte.lower())
        t = "".join(c for c in t if not unicodedata.combining(c))
        return [m[:4] for m in re.findall(r"[a-z0-9]+", t) if m not in MOTS_VIDES]

    _comptes = [Counter(mots(d["texte"])) for d in documents]  # occurrences de chaque mot
    _df = Counter(m for c in _comptes for m in c)              # nb de fiches contenant le mot
    IDF = {m: math.log(len(documents) / n) for m, n in _df.items()}

    def chercher(requete: str) -> str:
        # score d'une fiche = somme, sur les mots de la requête qu'elle contient, de
        # (1 + log du nombre d'occurrences) × rareté du mot
        scores = [sum((1 + math.log(c[m])) * IDF[m] for m in set(mots(requete)) if m in c)
                  for c in _comptes]
        ordre = sorted(range(len(documents)), key=lambda i: -scores[i])[: config["k_passages"]]
        passages = [documents[i]["texte"].replace("\n\n", " ").replace("\n", " ")
                    for i in ordre if scores[i] > 0]
        return " | ".join(passages) if passages else "Aucun passage trouvé."

    print(chercher("nombre de stations")[:120], "…")
    return (chercher,)


@app.cell
def _(ast, operator, re):
    # --- Outil 2 : calculatrice[expression], sans eval() : on parcourt l'arbre syntaxique ---
    # Un LLM peut écrire n'importe quoi : on n'exécute que + − × ÷ puissance modulo, parenthèses,
    # nombres, et round/abs/min/max. Toute autre construction lève une erreur, renvoyée
    # au modèle comme observation (il peut alors corriger son appel).
    OPERATEURS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                  ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
                  ast.USub: operator.neg, ast.UAdd: operator.pos}
    FONCTIONS = {"round": round, "abs": abs, "min": min, "max": max}

    def _evaluer(n):
        if isinstance(n, ast.Expression):
            return _evaluer(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPERATEURS:
            g, d = _evaluer(n.left), _evaluer(n.right)
            if isinstance(n.op, ast.Pow) and abs(d) > 100:
                raise ValueError("exposant trop grand")
            return OPERATEURS[type(n.op)](g, d)
        if isinstance(n, ast.UnaryOp) and type(n.op) in OPERATEURS:
            return OPERATEURS[type(n.op)](_evaluer(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FONCTIONS:
            return FONCTIONS[n.func.id](*[_evaluer(a) for a in n.args])
        raise ValueError(f"construction non autorisée : {ast.dump(n)[:60]}")

    def calculatrice(expression: str) -> str:
        e = str(expression)
        # Écritures humaines courantes → Python : 3 847 → 3847, 17,5 → 17.5, × → *, 2^20 → 2**20
        e = (e.replace("×", "*").replace("÷", "/").replace("−", "-").replace("^", "**")
              .replace("€", "").replace(" ", " ").replace("\xa0", " "))
        e = re.sub(r"(?<=\d) (?=\d{3}\b)", "", e)
        e = re.sub(r"(?<=\d),(?=\d)", ".", e)
        e = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", e)    # 17.5 % → (17.5/100)
        try:
            valeur = _evaluer(ast.parse(e.strip(), mode="eval"))
        except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError,
                RecursionError) as erreur:  # renvoyé au modèle, qui peut réessayer
            return f"Erreur : {erreur}"
        return f"{valeur:.10g}" if isinstance(valeur, float) else str(valeur)

    # Tests vérifiables à la main
    assert calculatrice("3847 * 29 - 1234") == "110329"
    assert calculatrice("17,5 % * 2384") == "417.2"
    assert calculatrice("2^20") == "1048576"
    assert calculatrice("__import__('os')").startswith("Erreur")
    return (calculatrice,)


@app.cell
def _(re):
    # --- Lire le nombre final d'une réponse, et le comparer à l'attendu ---
    # Formats acceptés : 110329, 110 329 (espace fine ou insécable), 1,048,576, 27,62 ou 27.62.
    NOMBRE = r"-?(?:\d{1,3}(?:[  \xa0]\d{3})+|\d+)(?:[.,]\d+)*"

    def vers_float(s: str) -> float:
        s = re.sub(r"[  \xa0]", "", s)
        if "," in s and "." in s:          # 1,048,576.5 ou 1.048.576,5 : le dernier est décimal
            dec = "," if s.rfind(",") > s.rfind(".") else "."
            s = s.replace("." if dec == "," else ",", "").replace(",", ".")
        elif s.count(",") > 1:
            s = s.replace(",", "")         # 1,048,576 : séparateurs de milliers à l'anglaise
        else:
            s = s.replace(",", ".")        # 27,62 : virgule décimale
        if s.count(".") > 1:               # 1.048.576
            s = s.replace(".", "")
        return float(s)

    def lire_nombre(texte: str) -> float | None:
        """Premier nombre après « Réponse » (la dernière occurrence) ; sinon le dernier nombre."""
        t = texte.replace("**", "")
        marques = list(re.finditer(r"r[ée]ponse(?:\s+finale)?\s*:", t, flags=re.IGNORECASE))
        if marques:
            m = re.search(NOMBRE, t[marques[-1].end():])
            if m:
                return vers_float(m.group())
        tous = re.findall(NOMBRE, t)
        return vers_float(tous[-1]) if tous else None

    def est_juste(valeur: float | None, attendu: float) -> bool:
        # tolérance : un centime, plus 0,01 % de la valeur (arrondis)
        return valeur is not None and abs(valeur - attendu) <= 0.011 + 1e-4 * abs(attendu)

    assert lire_nombre("Le calcul donne 111 563 − 1 234.\nRéponse : 110 329") == 110329
    assert lire_nombre("Réponse finale : 27,62 %") == 27.62
    assert lire_nombre("**Réponse : 1,048,576**") == 1048576
    assert lire_nombre("Fin 2024, 43 stations.") == 43
    assert lire_nombre("je ne sais pas") is None
    return est_juste, lire_nombre


@app.cell
def _(config):
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
    print(f"Ollama répond ; modèle {config['modele']} présent.")
    return (client,)


@app.cell
def _(SEED, client, config):
    # --- Un appel au modèle, avec comptage du coût ---
    def appeler(messages, n_tokens, outils=None, stop=None):
        """think=False : pas de « réflexion » cachée (mode thinking de Qwen 3.5 désactivé) ;
        tout ce que le modèle produit est visible. Température 0 : décodage glouton."""
        options = {"temperature": config["temperature"], "seed": SEED, "num_predict": n_tokens}
        if stop:
            options["stop"] = stop
        r = client.chat(model=config["modele"], messages=messages, tools=outils,
                        think=False, options=options)
        cout = {"appels": 1, "tokens": r.eval_count or 0, "secondes": (r.eval_duration or 0) / 1e9,
                "tokens_lus": r.prompt_eval_count or 0}
        return r.message, cout

    def ajouter(total, cout):
        for k, v in cout.items():
            total[k] = total.get(k, 0) + v

    return ajouter, appeler


@app.cell
def _(ajouter, appeler, config, lire_nombre):
    # --- Stratégie 1 : sans outils ---
    CONSIGNE_SANS_OUTILS = (
        "Réponds à la question en français. Raisonne brièvement si besoin, puis termine par "
        "une ligne « Réponse : <nombre> » (le nombre seul, sans unité).")

    def sans_outils(question: str) -> dict:
        cout = {}
        msg, c = appeler([{"role": "system", "content": CONSIGNE_SANS_OUTILS},
                          {"role": "user", "content": question}], config["tokens_sans_outils"])
        ajouter(cout, c)
        return {"reponse": lire_nombre(msg.content), "trace": [msg.content], "cout": cout,
                "outils": []}

    return CONSIGNE_SANS_OUTILS, sans_outils


@app.cell
def _(ajouter, appeler, calculatrice, chercher, config, lire_nombre, re):
    # --- Stratégie 2 : ReAct écrit à la main (Yao et al., 2022) ---
    # Tout passe par du texte : la consigne décrit les outils et le format, avec un exemple
    # complet (few-shot) sur un autre sujet ; notre code lit « Action : outil[argument] ».
    CONSIGNE_REACT = """Tu réponds à des questions en t'aidant de deux outils :
- calculatrice[expression] : calcule une expression arithmétique, par exemple calculatrice[(12 + 3) * 4 / 7]
- chercher[mots-clés] : cherche dans la documentation de la coopérative Roulevent et renvoie les passages les plus pertinents

Procède une étape à la fois, avec ce format :
Pensée : ce que tu dois faire ensuite
Action : calculatrice[...] ou chercher[...]
Tu recevras alors une ligne « Observation : ... ». Recommence autant que nécessaire.
Quand tu connais la réponse, écris :
Pensée : ...
Réponse finale : <nombre seul>

Règles : n'invente jamais d'observation ; fais chaque calcul avec la calculatrice ; tout fait sur Roulevent vient de chercher.

Exemple (autre sujet) :
Question : Combien coûtent 3 cartes de bibliothèque de Lunel ?
Pensée : Je dois trouver le prix d'une carte.
Action : chercher[prix carte bibliothèque Lunel]
Observation : La carte annuelle de la bibliothèque de Lunel coûte 15,50 euros.
Pensée : Je multiplie le prix par 3.
Action : calculatrice[3 * 15.50]
Observation : 46.5
Pensée : J'ai la réponse.
Réponse finale : 46.5"""

    ACTION = re.compile(r"Action\s*:\s*(calculatrice|chercher)\s*\[(.*)\]", re.IGNORECASE)
    FINALE = re.compile(r"R[ée]ponse finale\s*:", re.IGNORECASE)
    OBSERVATION = re.compile(r"^\s*Observation\s*:", re.IGNORECASE | re.MULTILINE)
    OUTILS = {"calculatrice": calculatrice, "chercher": chercher}

    def react_texte(question: str) -> dict:
        messages = [{"role": "system", "content": CONSIGNE_REACT},
                    {"role": "user", "content": f"Question : {question}"}]
        cout, trace, outils, invente = {}, [], [], 0
        for _ in range(config["max_appels"]):
            # stop : on coupe dès que le modèle commence à écrire lui-même une observation
            msg, c = appeler(messages, config["tokens_etape"], stop=["Observation"])
            ajouter(cout, c)
            texte = msg.content.strip()
            # Le modèle écrit souvent lui-même la suite : « Observation : 65000 » (inventée),
            # puis une réponse finale fondée sur cette invention. On coupe le texte à la
            # première « Observation » : seul notre code a le droit d'en produire.
            coupe = OBSERVATION.search(texte)
            if coupe:
                invente += 1
                texte = texte[: coupe.start()].rstrip()
            trace.append(texte)
            messages.append({"role": "assistant", "content": texte})
            action = ACTION.search(texte)
            if FINALE.search(texte) and not action:
                return {"reponse": lire_nombre(texte), "trace": trace, "cout": cout,
                        "outils": outils, "observations_inventees": invente}
            if action is None:   # ni action ni réponse : on le rappelle au format
                observation = "Format non reconnu. Écris une ligne Action : ... ou Réponse finale : ..."
            else:
                nom, argument = action.group(1).lower(), action.group(2).strip()
                observation = OUTILS[nom](argument)
                outils.append(nom)
            trace.append(f"Observation : {observation}")
            messages.append({"role": "user", "content": f"Observation : {observation}"})
        return {"reponse": None, "trace": trace, "cout": cout, "outils": outils,
                "observations_inventees": invente}  # budget d'appels épuisé

    return CONSIGNE_REACT, react_texte


@app.cell
def _(ajouter, appeler, calculatrice, chercher, config, lire_nombre):
    # --- Stratégie 3 : appel de fonctions natif (function calling) ---
    # Les outils sont décrits par un schéma JSON (le format popularisé par l'API d'OpenAI
    # en 2023, repris par Ollama, Anthropic, MCP…). Le modèle a été affiné pour répondre,
    # quand il veut un outil, par un bloc structuré que le serveur sépare du texte :
    # plus de regex à écrire, et les arguments arrivent déjà en JSON.
    SCHEMAS = [
        {"type": "function", "function": {
            "name": "calculatrice",
            "description": "Calcule une expression arithmétique, par exemple (12 + 3) * 4 / 7.",
            "parameters": {"type": "object", "required": ["expression"], "properties": {
                "expression": {"type": "string", "description": "l'expression à calculer"}}}}},
        {"type": "function", "function": {
            "name": "chercher",
            "description": "Cherche dans la documentation de la coopérative de vélos Roulevent "
                           "et renvoie les passages les plus pertinents.",
            "parameters": {"type": "object", "required": ["requete"], "properties": {
                "requete": {"type": "string", "description": "mots-clés à chercher"}}}}},
    ]
    CONSIGNE_FONCTIONS = (
        "Réponds à la question en français. Utilise l'outil chercher pour tout fait sur la "
        "coopérative Roulevent et l'outil calculatrice pour tout calcul. Quand tu as la "
        "réponse, termine par une ligne « Réponse : <nombre> » (le nombre seul).")

    def executer(nom: str, arguments: dict) -> str:
        valeurs = list(arguments.values())   # tolère un nom d'argument inattendu
        if nom == "calculatrice" and valeurs:
            return calculatrice(str(valeurs[0]))
        if nom == "chercher" and valeurs:
            return chercher(str(valeurs[0]))
        return f"Erreur : outil {nom} inconnu ou sans argument"

    def appel_fonctions(question: str) -> dict:
        messages = [{"role": "system", "content": CONSIGNE_FONCTIONS},
                    {"role": "user", "content": question}]
        cout, trace, outils = {}, [], []
        for _ in range(config["max_appels"]):
            msg, c = appeler(messages, config["tokens_etape"], outils=SCHEMAS)
            ajouter(cout, c)
            messages.append(msg)
            if not msg.tool_calls:   # pas d'appel d'outil : c'est la réponse finale
                trace.append(msg.content)
                return {"reponse": lire_nombre(msg.content), "trace": trace, "cout": cout,
                        "outils": outils}
            for appel in msg.tool_calls:
                resultat = executer(appel.function.name, dict(appel.function.arguments))
                outils.append(appel.function.name)
                trace.append(f"→ {appel.function.name}({dict(appel.function.arguments)}) = {resultat}")
                messages.append({"role": "tool", "tool_name": appel.function.name,
                                 "content": resultat})
        return {"reponse": None, "trace": trace, "cout": cout, "outils": outils}

    return SCHEMAS, appel_fonctions


@app.cell
def _(mo):
    bouton = mo.ui.run_button(label="Lancer l'agent et le juge")
    bouton
    return (bouton,)


@app.cell
def _(
    appel_fonctions,
    appeler,
    bouton,
    est_juste,
    mo,
    react_texte,
    sans_outils,
    taches,
    time,
):
    # En script (make ex MODE=run|smoke), personne ne clique : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer l'agent et le juge » (≈ 6 min en mode complet).*"))

    t0 = time.perf_counter()
    appeler([{"role": "user", "content": "Bonjour"}], 1)   # charge le modèle (non compté)
    print(f"Chargement du modèle : {time.perf_counter() - t0:.1f} s")

    STRATEGIES = {"sans outils": sans_outils, "ReAct (texte)": react_texte,
                  "appel de fonctions": appel_fonctions}
    resultats_agent = []
    t_agent = time.perf_counter()
    for _t in taches:
        _ligne = {**_t}
        for _nom, _f in STRATEGIES.items():
            _t1 = time.perf_counter()
            _r = _f(_t["question"])
            _r["juste"] = est_juste(_r["reponse"], _t["attendu"])
            _r["duree"] = time.perf_counter() - _t1
            _ligne[_nom] = _r
        resultats_agent.append(_ligne)
        print(f"{_t['id']} attendu {_t['attendu']:g} | " + " | ".join(
            f"{_n} : {_ligne[_n]['reponse']} {'✓' if _ligne[_n]['juste'] else '✗'} "
            f"({_ligne[_n]['cout']['appels']} appel(s))" for _n in STRATEGIES))
    duree_agent = time.perf_counter() - t_agent
    print(f"Partie A : {duree_agent:.0f} s")
    return STRATEGIES, duree_agent, resultats_agent


@app.cell
def _(STRATEGIES, mo, np, resultats_agent):
    # --- Bilan de la partie A : précision par catégorie, coût moyen ---
    CATEGORIES = ["calcul", "recherche", "composée"]
    bilan_agent = {}
    for _nom in STRATEGIES:
        _b = {"precision": float(np.mean([r[_nom]["juste"] for r in resultats_agent]))}
        for _cat in CATEGORIES:
            _sel = [r[_nom]["juste"] for r in resultats_agent if r["categorie"] == _cat]
            _b[_cat] = float(np.mean(_sel)) if _sel else float("nan")
        for _k in ("appels", "tokens", "tokens_lus"):
            _b[_k] = float(np.mean([r[_nom]["cout"].get(_k, 0) for r in resultats_agent]))
        _b["duree"] = float(np.mean([r[_nom]["duree"] for r in resultats_agent]))
        _b["sans_reponse"] = int(sum(r[_nom]["reponse"] is None for r in resultats_agent))
        _b["outils"] = float(np.mean([len(r[_nom]["outils"]) for r in resultats_agent]))
        # étapes où le modèle a écrit lui-même une « Observation » (ReAct texte seulement)
        _b["observations_inventees"] = int(sum(r[_nom].get("observations_inventees", 0)
                                               for r in resultats_agent))
        bilan_agent[_nom] = _b

    _lignes = [("| stratégie | calcul | recherche | composée | total | appels / tâche | "
                "tokens générés / tâche | outils / tâche | secondes / tâche |"), "|---" * 9 + "|"]
    for _nom, _b in bilan_agent.items():
        _lignes.append(
            f"| {_nom} | " + " | ".join(f"{100 * _b[c]:.0f} %" for c in CATEGORIES)
            + f" | **{100 * _b['precision']:.0f} %** | {_b['appels']:.1f} | {_b['tokens']:.0f} "
            f"| {_b['outils']:.1f} | {_b['duree']:.1f} |")
    _lignes.append(f"\nReAct (texte) : {bilan_agent['ReAct (texte)']['observations_inventees']} "
                   "étape(s) où le modèle a inventé une observation (coupée par notre code).")
    print("\n".join(_lignes))
    mo.md("\n".join(_lignes))
    return CATEGORIES, bilan_agent


@app.cell
def _(SMOKE, mo, resultats_agent):
    # --- Une trajectoire complète, pour voir la boucle Pensée → Action → Observation ---
    _r = next((r for r in resultats_agent if r["categorie"] == "composée"), resultats_agent[0])
    _texte = "\n".join(_r["ReAct (texte)"]["trace"])
    _f = "\n".join(_r["appel de fonctions"]["trace"])
    if not SMOKE:
        print(f"Trajectoire ReAct pour {_r['id']} :\n{_texte}\n\nAppel de fonctions :\n{_f}")
    mo.md(f"**{_r['id']}. {_r['question']}** (attendu : {_r['attendu']:g})\n\n"
          f"ReAct (texte) :\n```text\n{_texte}\n```\nAppel de fonctions :\n```text\n{_f}\n```")


@app.cell
def _(JUGE, ajouter, appeler, config, re, taches_par_id, time):
    # --- Partie B : le LLM juge (Zheng et al., 2023), contre le jugement humain ---
    CONSIGNE_JUGE_REF = (
        "Tu es correcteur. On te donne une question, la réponse de référence et la réponse "
        "d'un candidat. La réponse du candidat est correcte si sa valeur finale est la même "
        "que celle de la référence ; la formulation, l'unité et les séparateurs de milliers "
        "ne comptent pas. Réponds par un seul mot : CORRECT ou INCORRECT.")
    CONSIGNE_JUGE_SANS_REF = (
        "Tu es correcteur. On te donne une question et la réponse d'un candidat. Dis si la "
        "réponse est juste. Réponds par un seul mot : CORRECT ou INCORRECT.")
    CONSIGNE_PAIRE = (
        "On te donne une question et deux réponses, A et B. Une seule est juste. Laquelle ? "
        "Réponds par une seule lettre : A ou B.")

    def verdict(texte: str) -> bool | None:
        t = texte.upper()
        if "INCORRECT" in t:          # à tester en premier : « CORRECT » est dans « INCORRECT »
            return False
        return True if "CORRECT" in t else None

    def juger(question, reponse, reference=None):
        if reference is None:
            contenu, consigne = f"Question : {question}\nRéponse du candidat : {reponse}", CONSIGNE_JUGE_SANS_REF
        else:
            contenu = (f"Question : {question}\nRéférence : {reference:g}\n"
                       f"Réponse du candidat : {reponse}")
            consigne = CONSIGNE_JUGE_REF
        msg, c = appeler([{"role": "system", "content": consigne},
                          {"role": "user", "content": contenu}], 8)
        return verdict(msg.content), c

    def comparer(question, rep_a, rep_b):
        msg, c = appeler([{"role": "system", "content": CONSIGNE_PAIRE},
                          {"role": "user", "content": f"Question : {question}\n\nRéponse A : {rep_a}"
                                                      f"\n\nRéponse B : {rep_b}"}], 4)
        lettre = re.search(r"\b([AB])\b", msg.content.upper())
        return (lettre.group(1) if lettre else None), c

    t_juge = time.perf_counter()
    cout_juge = {}
    reponses_jugees = []
    for _j in JUGE["reponses"][: config["n_reponses_jugees"]]:
        _t = taches_par_id[_j["tache"]]
        _avec, _c1 = juger(_t["question"], _j["reponse"], _t["attendu"])
        _sans, _c2 = juger(_t["question"], _j["reponse"])
        ajouter(cout_juge, _c1)
        ajouter(cout_juge, _c2)
        reponses_jugees.append({**_j, "juge_avec_reference": _avec, "juge_sans_reference": _sans})
        print(f"{_j['id']} humain {'✓' if _j['humain'] else '✗'} | juge avec référence "
              f"{_avec} | sans référence {_sans} | {_j['reponse'][:50]}")

    # Paires : la bonne réponse d'abord en A, puis en B. Un juge sans biais choisit la même.
    _reps = {j["id"]: j for j in JUGE["reponses"]}
    paires = []
    for _bonne, _mauvaise in JUGE["paires"][: config["n_paires"]]:
        _q = taches_par_id[_reps[_bonne]["tache"]]["question"]
        _o1, _c1 = comparer(_q, _reps[_bonne]["reponse"], _reps[_mauvaise]["reponse"])
        _o2, _c2 = comparer(_q, _reps[_mauvaise]["reponse"], _reps[_bonne]["reponse"])
        ajouter(cout_juge, _c1)
        ajouter(cout_juge, _c2)
        paires.append({"bonne": _bonne, "mauvaise": _mauvaise,
                       "bonne_en_A": _o1, "bonne_en_B": _o2,
                       "juste_ordre_1": _o1 == "A", "juste_ordre_2": _o2 == "B",
                       "coherent": (_o1 == "A") == (_o2 == "B")})
        print(f"paire {_bonne}/{_mauvaise} : bonne en A → {_o1}, bonne en B → {_o2}")
    duree_juge = time.perf_counter() - t_juge
    print(f"Partie B : {duree_juge:.0f} s, {cout_juge.get('appels', 0)} appels")
    return duree_juge, paires, reponses_jugees


@app.cell
def _(mo, np, paires, reponses_jugees):
    # --- Bilan de la partie B : accord, kappa de Cohen, matrice de confusion, biais de position ---
    def accord(cle):
        h = np.array([r["humain"] for r in reponses_jugees])
        j = np.array([bool(r[cle]) for r in reponses_jugees])      # None (illisible) = non
        po = float(np.mean(h == j))                                 # accord observé
        pe = float(np.mean(h) * np.mean(j) + (1 - np.mean(h)) * (1 - np.mean(j)))  # par hasard
        kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
        conf = {"vrai_ok_juge_ok": int(np.sum(h & j)), "vrai_ok_juge_faux": int(np.sum(h & ~j)),
                "vrai_faux_juge_ok": int(np.sum(~h & j)), "vrai_faux_juge_faux": int(np.sum(~h & ~j))}
        return {"accord": po, "kappa": float(kappa), "confusion": conf,
                "illisibles": int(sum(r[cle] is None for r in reponses_jugees))}

    bilan_juge = {
        "avec_reference": accord("juge_avec_reference"),
        "sans_reference": accord("juge_sans_reference"),
        "paires": {
            "n": len(paires),
            "juste_bonne_en_A": float(np.mean([p["juste_ordre_1"] for p in paires])),
            "juste_bonne_en_B": float(np.mean([p["juste_ordre_2"] for p in paires])),
            "coherence": float(np.mean([p["coherent"] for p in paires])),
            "choix_A": float(np.mean([[p["bonne_en_A"] == "A", p["bonne_en_B"] == "A"]
                                      for p in paires])),
        },
    }
    _a, _s, _p = bilan_juge["avec_reference"], bilan_juge["sans_reference"], bilan_juge["paires"]
    _md = (
        f"| juge | accord avec l'humain | kappa de Cohen | faux positifs (faux jugé juste) |"
        f" faux négatifs |\n|---|---|---|---|---|\n"
        f"| avec référence | {100 * _a['accord']:.0f} % | {_a['kappa']:.2f} | "
        f"{_a['confusion']['vrai_faux_juge_ok']} | {_a['confusion']['vrai_ok_juge_faux']} |\n"
        f"| sans référence | {100 * _s['accord']:.0f} % | {_s['kappa']:.2f} | "
        f"{_s['confusion']['vrai_faux_juge_ok']} | {_s['confusion']['vrai_ok_juge_faux']} |\n\n"
        f"Comparaisons par paires ({_p['n']} paires, chacune dans les deux ordres) : bonne "
        f"réponse choisie {100 * _p['juste_bonne_en_A']:.0f} % des fois quand elle est en A, "
        f"{100 * _p['juste_bonne_en_B']:.0f} % quand elle est en B ; verdict identique dans "
        f"les deux ordres pour {100 * _p['coherence']:.0f} % des paires ; « A » choisi "
        f"{100 * _p['choix_A']:.0f} % des fois.")
    print(_md)
    mo.md(_md)
    return (bilan_juge,)


@app.cell
def _(
    CATEGORIES,
    PALETTE,
    SMOKE,
    bilan_agent,
    bilan_juge,
    mo,
    plt,
    sauvegarder_figure,
):
    # --- Figure : précision par catégorie et par stratégie ; accord du juge ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 2]})
    _x = range(len(CATEGORIES) + 1)
    _larg = 0.26
    for _i, (_nom, _b) in enumerate(bilan_agent.items()):
        _v = [100 * _b[c] for c in CATEGORIES] + [100 * _b["precision"]]
        ax1.bar([x + (_i - 1) * _larg for x in _x], _v, _larg, label=_nom, color=PALETTE[_i])
    ax1.set_xticks(list(_x), CATEGORIES + ["total"])
    ax1.set(ylabel="réponses justes (%)", ylim=(0, 105), title="Partie A : l'agent")
    ax1.legend(fontsize=9)
    _noms = ["juge avec\nréférence", "juge sans\nréférence", "paires :\nmême verdict\ndans les 2 ordres"]
    _vals = [100 * bilan_juge["avec_reference"]["accord"], 100 * bilan_juge["sans_reference"]["accord"],
             100 * bilan_juge["paires"]["coherence"]]
    ax2.bar(_noms, _vals, color=[PALETTE[2], PALETTE[3], PALETTE[4]])
    ax2.axhline(50, color="#666666", ls="--", lw=1)
    ax2.set(ylabel="%", ylim=(0, 105), title="Partie B : le juge (accord avec l'humain)")
    fig.tight_layout()
    _chemins = sauvegarder_figure(fig, "agent-juge-smoke" if SMOKE else "agent-juge",
                                  mo.notebook_dir() / "figures")
    print(f"Figure : {_chemins[0]}")
    fig


@app.cell
def _(
    CONSIGNE_REACT,
    CONSIGNE_SANS_OUTILS,
    DOSSIER,
    SCHEMAS,
    SMOKE,
    bilan_agent,
    bilan_juge,
    config,
    duree_agent,
    duree_juge,
    json,
    paires,
    reponses_jugees,
    resultats_agent,
):
    # --- Sauvegarde de tous les résultats (lus par les figures du site) ---
    _sortie = DOSSIER / "figures" / ("resultats-smoke.json" if SMOKE else "resultats.json")
    _sortie.write_text(json.dumps({
        "config": config, "duree_agent_s": duree_agent, "duree_juge_s": duree_juge,
        "bilan_agent": bilan_agent, "bilan_juge": bilan_juge,
        "taches": resultats_agent, "reponses_jugees": reponses_jugees, "paires": paires,
        "consignes": {"sans_outils": CONSIGNE_SANS_OUTILS, "react": CONSIGNE_REACT},
        "schemas": SCHEMAS,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Résultats : {_sortie}")


@app.cell
def _(SMOKE, bilan_agent, bilan_juge, duree_agent, duree_juge, mo):
    # --- Bilan : seuils chiffrés du README. En smoke, on vérifie seulement que tout tourne.
    _sans = bilan_agent["sans outils"]["precision"]
    _fct = bilan_agent["appel de fonctions"]["precision"]
    _react = bilan_agent["ReAct (texte)"]["precision"]
    _juge = bilan_juge["avec_reference"]["accord"]
    _juge_sans = bilan_juge["sans_reference"]["accord"]
    SEUILS = {
        "appel de fonctions ≥ 75 %": _fct >= 0.75,
        "appel de fonctions ≥ ReAct (texte) + 20 points": _fct >= _react + 0.20,
        "ReAct (texte) ≥ sans outils + 10 points": _react >= _sans + 0.10,
        "sans outils ≤ 35 %": _sans <= 0.35,
        "juge avec référence : accord ≥ 70 %": _juge >= 0.70,
        "juge sans référence : accord ≤ juge avec référence − 15 points": _juge_sans <= _juge - 0.15,
    }
    _echecs = [k for k, ok in SEUILS.items() if not ok]
    if not SMOKE and _echecs:
        raise SystemExit(f"Seuils non atteints : {_echecs}")
    mo.md(f"**Bilan** : sans outils {100 * _sans:.0f} %, ReAct {100 * _react:.0f} %, appel de "
          f"fonctions {100 * _fct:.0f} % ; juge avec référence {100 * _juge:.0f} % d'accord ; "
          f"{duree_agent + duree_juge:.0f} s de générations.")


if __name__ == "__main__":
    app.run()
