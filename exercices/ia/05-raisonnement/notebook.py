# Exercice 05 : raisonner sans apprendre (règles, contraintes, planification).
#
#   make ex NUM=05              → éditeur marimo (interactif, bouton « Lancer les comparaisons »)
#   make ex NUM=05 MODE=run     → script complet (vérifie les seuils du README)
#   make ex NUM=05 MODE=smoke   → test rapide (< 30 s)
#
# Python pur (pas de GPU, pas de données téléchargées). Trois parties indépendantes :
#   1. un moteur de règles (chaînage avant, chaînage arrière façon MYCIN), vérifié par
#      un solveur SAT générique (sympy) ;
#   2. un problème de satisfaction de contraintes (sudoku) : retour arrière naïf, puis
#      heuristique MRV + vérification en avant, puis maintien de l'arc-consistance (AC-3) ;
#   3. un planificateur STRIPS du monde des blocs : recherche en largeur, recherche
#      gloutonne, anomalie de Sussman, explosion de l'espace d'états.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import heapq
    import itertools
    import time
    from collections import deque, namedtuple

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from ia_commun import PALETTE, is_smoke, sauvegarder_figure, seed_everything, style_commun

    return (
        PALETTE,
        deque,
        heapq,
        is_smoke,
        itertools,
        mo,
        namedtuple,
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
    # Exercice 05 : raisonner avec des règles, des contraintes et des plans

    De 1956 à la fin des années 1980, l'IA dominante ne **apprend** rien : elle **déduit**.
    Un humain écrit des faits et des règles ; un moteur générique en tire des conclusions.
    Vous allez programmer les trois moteurs de ce chapitre, en Python pur :

    1. **Système de règles** : une petite base de diagnostic de panne d'un site web,
       interrogée en *chaînage avant* (des faits vers les conclusions) puis en *chaînage
       arrière* (d'une hypothèse vers les faits à vérifier, comme MYCIN). Un solveur SAT
       générique (`sympy`) vérifie que les déductions sont exactes.
    2. **Contraintes (CSP)** : le sudoku, résolu par retour arrière naïf, puis avec
       l'heuristique MRV, puis en maintenant l'**arc-consistance** (AC-3).
    3. **Planification STRIPS** : le monde des blocs, l'anomalie de Sussman et
       l'explosion du nombre d'états.

    **Fonction de perte : aucune.** Rien n'est optimisé sur des données : on cherche une
    preuve, une affectation qui respecte toutes les contraintes ou une suite d'actions.
    Le seul « coût » est la longueur du plan (partie 3).
    """)


@app.cell
def _(is_smoke, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)  # rien d'aléatoire ici sauf le tirage des instances de blocs
    style_commun()
    config = dict(
        # nombre de blocs maximal pour l'expérience d'explosion (partie 3)
        n_blocs_max=5 if SMOKE else 8,
        # instances aléatoires tirées par taille de problème
        n_instances=2 if SMOKE else 5,
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SEED, SMOKE, config


@app.cell
def _(mo):
    mo.md(r"""
    ## Partie 1 : un système expert en miniature

    Une règle a la forme **SI** prémisse₁ **ET** prémisse₂ … **ALORS** conclusion. Ce sont
    des *clauses de Horn* : une conjonction de faits positifs implique un fait. Les faits
    observables (les symptômes) sont fournis par l'utilisateur ; les autres sont déduits.
    """)


@app.cell
def _():
    # --- Base de connaissances : diagnostic de panne d'un site web ---
    # Chaque règle : (nom, [prémisses], conclusion). Les conclusions « diagnostic: »
    # et « action: » sont ce que l'utilisateur veut obtenir.
    REGLES = [
        ("R1", ["statut_502", "serveur_web_repond"], "proxy_ok_appli_ko"),
        ("R2", ["proxy_ok_appli_ko", "deploiement_recent"], "diagnostic:appli_plante_au_demarrage"),
        ("R3", ["proxy_ok_appli_ko", "memoire_saturee"], "diagnostic:appli_tuee_par_manque_de_memoire"),
        ("R4", ["statut_504", "temps_reponse_eleve"], "appli_lente"),
        ("R5", ["appli_lente", "cpu_base_sature"], "diagnostic:base_de_donnees_surchargee"),
        ("R6", ["page_blanche", "erreur_console_js"], "erreur_front"),
        ("R7", ["erreur_front", "deploiement_recent"], "diagnostic:bundle_js_casse"),
        ("R8", ["alerte_securite_navigateur", "date_certificat_depassee"], "diagnostic:certificat_expire"),
        ("R9", ["nom_de_domaine_ne_resout_pas"], "diagnostic:probleme_dns"),
        ("R10", ["diagnostic:appli_plante_au_demarrage"], "action:revenir_a_la_version_precedente"),
        ("R11", ["diagnostic:bundle_js_casse"], "action:revenir_a_la_version_precedente"),
        ("R12", ["diagnostic:base_de_donnees_surchargee"], "action:ajouter_un_index_ou_un_cache"),
        ("R13", ["diagnostic:appli_tuee_par_manque_de_memoire"], "action:augmenter_la_memoire"),
        ("R14", ["diagnostic:certificat_expire"], "action:renouveler_le_certificat"),
        ("R15", ["diagnostic:probleme_dns"], "action:verifier_la_zone_dns"),
    ]
    conclusions = {c for _, _, c in REGLES}
    # Faits observables = ceux qu'aucune règle ne conclut : on les demande à l'utilisateur.
    OBSERVABLES = sorted({p for _, ps, _ in REGLES for p in ps} - conclusions)
    DIAGNOSTICS = sorted(c for c in conclusions if c.startswith("diagnostic:"))

    # Cas de test : symptômes observés -> diagnostics attendus. Le dernier cas sort du
    # domaine couvert par les règles : le système doit rester muet (fragilité).
    CAS = [
        ("déploiement raté", {"statut_502", "serveur_web_repond", "deploiement_recent"},
         {"diagnostic:appli_plante_au_demarrage"}),
        ("base saturée", {"statut_504", "temps_reponse_eleve", "cpu_base_sature"},
         {"diagnostic:base_de_donnees_surchargee"}),
        ("front cassé", {"page_blanche", "erreur_console_js", "deploiement_recent"},
         {"diagnostic:bundle_js_casse"}),
        ("deux pannes", {"alerte_securite_navigateur", "date_certificat_depassee",
                         "nom_de_domaine_ne_resout_pas"},
         {"diagnostic:certificat_expire", "diagnostic:probleme_dns"}),
        ("hors domaine", {"images_absentes", "page_lente_sur_mobile"}, set()),
    ]
    print(f"{len(REGLES)} règles, {len(OBSERVABLES)} faits observables, {len(DIAGNOSTICS)} diagnostics")
    return CAS, DIAGNOSTICS, OBSERVABLES, REGLES


@app.cell
def _():
    def chainage_avant(faits_initiaux, regles, trace=False):
        """Applique les règles jusqu'au point fixe (plus aucun fait nouveau).

        Renvoie l'ensemble des faits et le nombre de tests de règles effectués.
        """
        faits = set(faits_initiaux)
        declenchees = set()
        tests = 0
        nouveau = True
        while nouveau:  # un « cycle » = un passage sur toutes les règles
            nouveau = False
            for nom, premisses, conclusion in regles:
                if nom in declenchees:
                    continue
                tests += 1
                if all(p in faits for p in premisses):
                    declenchees.add(nom)
                    if conclusion not in faits:
                        faits.add(conclusion)
                        nouveau = True
                        if trace:
                            print(f"  {nom} : {' ET '.join(premisses)}  =>  {conclusion}")
        return faits, tests

    return (chainage_avant,)


@app.cell
def _(CAS, REGLES, chainage_avant):
    # Trace détaillée du premier cas, puis bilan de tous les cas
    print(f"Cas « {CAS[0][0]} », symptômes : {sorted(CAS[0][1])}")
    _faits, _tests = chainage_avant(CAS[0][1], REGLES, trace=True)
    print(f"  ({_tests} tests de règles)\n")

    resultats_avant = []
    for _nom, _symptomes, _attendus in CAS:
        _faits, _tests = chainage_avant(_symptomes, REGLES)
        _diags = {f for f in _faits if f.startswith("diagnostic:")}
        _actions = sorted(f for f in _faits if f.startswith("action:"))
        resultats_avant.append((_nom, _diags, _diags == _attendus))
        print(f"{_nom:16s} -> {sorted(_diags) or 'aucun diagnostic'} {_actions} "
              f"{'OK' if _diags == _attendus else 'ERREUR'} ({_tests} tests)")
    cas_corrects = sum(ok for *_, ok in resultats_avant)
    print(f"\nChaînage avant : {cas_corrects}/{len(CAS)} cas corrects")
    return cas_corrects, resultats_avant


@app.cell
def _(mo):
    mo.md(r"""
    ### Chaînage arrière : partir de l'hypothèse

    MYCIN raisonnait dans l'autre sens : pour établir une hypothèse, il cherche les règles
    qui la concluent, puis tente d'établir leurs prémisses, récursivement. Il ne **pose une
    question** que pour un fait observable dont il a besoin. Comptez les questions posées.
    """)


@app.cell
def _(DIAGNOSTICS, OBSERVABLES, REGLES):
    def chainage_arriere(but, regles, repondre, memo, questions):
        """Vrai si `but` est établi. `repondre(fait)` joue l'utilisateur ; `memo` garde
        les faits déjà établis ou réfutés ; `questions` enregistre les questions posées."""
        if but in memo:
            return memo[but]
        regles_utiles = [r for r in regles if r[2] == but]
        if not regles_utiles:  # fait observable : on le demande
            questions.append(but)
            memo[but] = repondre(but)
            return memo[but]
        memo[but] = False  # protège contre les boucles
        for _nom, premisses, _ in regles_utiles:
            if all(chainage_arriere(p, regles, repondre, memo, questions) for p in premisses):
                memo[but] = True
                break
        return memo[but]

    def diagnostiquer_en_arriere(symptomes):
        memo, questions = {}, []
        etablis = {d for d in DIAGNOSTICS
                   if chainage_arriere(d, REGLES, lambda f: f in symptomes, memo, questions)}
        return etablis, questions

    # Hypothèse unique : « le certificat a-t-il expiré ? »
    _memo, _questions = {}, []
    chainage_arriere("diagnostic:certificat_expire", REGLES,
                     lambda f: f in {"alerte_securite_navigateur", "date_certificat_depassee"},
                     _memo, _questions)
    print(f"Hypothèse « certificat expiré » : {len(_questions)} questions posées : {_questions}")
    print(f"(le chaînage avant, lui, attend les {len(OBSERVABLES)} observations d'emblée)")
    return (diagnostiquer_en_arriere,)


@app.cell
def _(CAS, OBSERVABLES, diagnostiquer_en_arriere):
    accord_arriere = 0
    for _nom, _symptomes, _attendus in CAS:
        _diags, _questions = diagnostiquer_en_arriere(_symptomes)
        accord_arriere += _diags == _attendus
        print(f"{_nom:16s} -> {sorted(_diags) or 'aucun'} ; "
              f"{len(_questions)}/{len(OBSERVABLES)} questions posées")
    print(f"Chaînage arrière : {accord_arriere}/{len(CAS)} cas corrects")
    return (accord_arriere,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Vérification par la logique : un solveur SAT générique

    Pour des clauses de Horn, le chaînage avant est **correct et complet** : il déduit
    exactement ce que la base implique logiquement. Vérifions-le avec `sympy`. Une base
    $B$ implique un fait $q$ ($B \models q$) si et seulement si $B \wedge \neg q$ n'a aucun
    modèle (aucune affectation vrai/faux ne la rend vraie).
    """)


@app.cell
def _(CAS, DIAGNOSTICS, REGLES, chainage_avant):
    import sympy
    from sympy.logic.boolalg import And, Implies, Not
    from sympy.logic.inference import satisfiable

    _symboles = {}

    def sym(nom):
        # sympy veut des noms de symboles simples : on remplace « : »
        if nom not in _symboles:
            _symboles[nom] = sympy.Symbol(nom.replace(":", "__"))
        return _symboles[nom]

    base_logique = And(*[Implies(And(*[sym(p) for p in ps]), sym(c)) for _, ps, c in REGLES])

    accord_sat = 0
    for _nom, _symptomes, _ in CAS:
        _B = And(base_logique, *[sym(s) for s in _symptomes])
        _impliques = {d for d in DIAGNOSTICS if not satisfiable(And(_B, Not(sym(d))))}
        _faits, _ = chainage_avant(_symptomes, REGLES)
        _avant = {f for f in _faits if f.startswith("diagnostic:")}
        accord_sat += _impliques == _avant
        print(f"{_nom:16s} SAT : {sorted(_impliques) or 'rien'} ; "
              f"identique au chaînage avant : {_impliques == _avant}")
    print(f"Accord SAT / chaînage avant : {accord_sat}/{len(CAS)}")
    return (accord_sat,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Partie 2 : satisfaction de contraintes (CSP)

    Un CSP, c'est des **variables**, un **domaine** de valeurs pour chacune et des
    **contraintes** entre elles. Un arc $(X_i, X_j)$ est *consistant* si chaque valeur de
    $X_i$ a au moins une valeur compatible dans $X_j$. **AC-3** (Mackworth, 1977) retire
    les valeurs sans support jusqu'à ce que tous les arcs soient consistants.

    Exemple à la main : $X < Y$, $Y < Z$, domaines $\{1, 2, 3\}$. AC-3 doit trouver
    $X = 1$, $Y = 2$, $Z = 3$ **sans aucune recherche**.
    """)


@app.cell
def _(deque):
    def ac3(domaines, voisins, compatible, file=None):
        """AC-3 générique. `domaines` : dict variable -> set (modifié sur place) ;
        `voisins[x]` : variables liées à x ; `compatible(xi, v, xj, w)` : la contrainte.
        Renvoie (consistant, nombre de révisions d'arcs)."""
        if file is None:
            file = deque((xi, xj) for xi in domaines for xj in voisins[xi])
        revisions = 0
        while file:
            xi, xj = file.popleft()
            revisions += 1
            # « réviser » l'arc : retirer de Di les valeurs sans support dans Dj
            retirees = {v for v in domaines[xi]
                        if not any(compatible(xi, v, xj, w) for w in domaines[xj])}
            if retirees:
                domaines[xi] -= retirees
                if not domaines[xi]:
                    return False, revisions  # domaine vide : aucune solution
                # Di a rétréci : les arcs qui pointent vers xi sont à revérifier
                file.extend((xk, xi) for xk in voisins[xi] if xk != xj)
        return True, revisions

    # Exemple jouet X < Y < Z
    _D = {"X": {1, 2, 3}, "Y": {1, 2, 3}, "Z": {1, 2, 3}}
    _voisins = {"X": ["Y"], "Y": ["X", "Z"], "Z": ["Y"]}
    _ordre = {"X": 0, "Y": 1, "Z": 2}

    def _inferieur(xi, v, xj, w):
        # la contrainte entre deux voisins : la variable « avant » est plus petite
        return v < w if _ordre[xi] < _ordre[xj] else v > w

    _ok, _rev = ac3(_D, _voisins, _inferieur)
    print(f"Exemple X<Y<Z : consistant={_ok}, domaines={_D}, {_rev} révisions d'arcs")
    return (ac3,)


@app.cell
def _(ac3, deque):
    # --- Le sudoku comme CSP : 81 variables, domaines 1..9, contrainte ≠ entre « pairs » ---
    CASES = [(r, c) for r in range(9) for c in range(9)]
    PAIRS = {}
    for _r, _c in CASES:
        _p = {(_r, _k) for _k in range(9)} | {(_k, _c) for _k in range(9)}
        _p |= {(3 * (_r // 3) + _i, 3 * (_c // 3) + _j) for _i in range(3) for _j in range(3)}
        _p.discard((_r, _c))
        PAIRS[(_r, _c)] = sorted(_p)  # 20 pairs par case : 8 en ligne, 8 en colonne, 4 dans le bloc

    def different(xi, v, xj, w):
        return v != w

    def lire_grille(texte):
        return {CASES[i]: (0 if ch == "." else int(ch)) for i, ch in enumerate(texte)}

    def domaines_initiaux(grille):
        return {x: ({grille[x]} if grille[x] else set(range(1, 10))) for x in CASES}

    def est_solution(sol, grille):
        return (sol is not None
                and all(1 <= sol[x] <= 9 for x in CASES)
                and all(sol[x] == grille[x] for x in CASES if grille[x])
                and all(sol[x] != sol[p] for x in CASES for p in PAIRS[x]))

    # Méthode 1 : retour arrière naïf, cases dans l'ordre, valeurs 1 à 9.
    def resoudre_naif(grille):
        g = dict(grille)
        vides = [x for x in CASES if g[x] == 0]
        stats = {"essais": 0}

        def rec(i):
            if i == len(vides):
                return True
            x = vides[i]
            for v in range(1, 10):
                stats["essais"] += 1
                if all(g[p] != v for p in PAIRS[x]):
                    g[x] = v
                    if rec(i + 1):
                        return True
                    g[x] = 0  # retour arrière
            return False

        return (g if rec(0) else None), stats

    # Méthode 2 : MRV (« minimum remaining values », la case la plus contrainte d'abord)
    # + vérification en avant (retirer la valeur choisie des domaines des pairs).
    def resoudre_mrv(grille):
        stats = {"essais": 0}

        def rec(D, assignees):
            libres = [x for x in CASES if x not in assignees]
            if not libres:
                return D
            # MRV : la case non affectée qui a le moins de candidats (1 = coup forcé)
            x = min(libres, key=lambda c: len(D[c]))
            for v in sorted(D[x]):
                stats["essais"] += 1
                D2 = {k: set(d) for k, d in D.items()}
                D2[x] = {v}
                # vérification en avant : v disparaît des domaines des pairs non affectés
                ok = True
                for p in PAIRS[x]:
                    if p not in assignees:
                        D2[p].discard(v)
                        if not D2[p]:
                            ok = False  # un pair n'a plus de candidat : inutile d'aller plus loin
                            break
                if ok:
                    res = rec(D2, assignees | {x})
                    if res:
                        return res
            return None

        D0 = domaines_initiaux(grille)
        donnees = {x for x in CASES if grille[x]}
        # les chiffres donnés sont retirés des domaines de leurs pairs
        for x in donnees:
            for p in PAIRS[x]:
                if p not in donnees:
                    D0[p].discard(grille[x])
        res = rec(D0, donnees)
        return ({k: next(iter(d)) for k, d in res.items()} if res else None), stats

    # Méthode 3 : MAC (« maintaining arc consistency ») = MRV + AC-3 après chaque choix.
    def resoudre_mac(grille):
        stats = {"essais": 0, "revisions": 0}
        D0 = domaines_initiaux(grille)
        ok, rev = ac3(D0, PAIRS, different)
        stats["revisions"] += rev
        if not ok:
            return None, stats

        def rec(D):
            libres = [x for x in CASES if len(D[x]) > 1]
            if not libres:
                return D
            x = min(libres, key=lambda c: len(D[c]))
            for v in sorted(D[x]):
                stats["essais"] += 1
                D2 = {k: set(d) for k, d in D.items()}
                D2[x] = {v}
                # seuls les arcs (pair -> x) sont à revérifier après ce choix
                ok, rev = ac3(D2, PAIRS, different, file=deque((p, x) for p in PAIRS[x]))
                stats["revisions"] += rev
                if ok:
                    res = rec(D2)
                    if res:
                        return res
            return None

        res = rec(D0)
        return ({k: next(iter(d)) for k, d in res.items()} if res else None), stats

    return (
        CASES,
        PAIRS,
        different,
        domaines_initiaux,
        est_solution,
        lire_grille,
        resoudre_mac,
        resoudre_mrv,
        resoudre_naif,
    )


@app.cell
def _(CASES, PAIRS, ac3, different, domaines_initiaux, lire_grille):
    # Deux grilles à solution unique (vérifié par énumération) : une facile, une difficile.
    GRILLES = {
        "facile": "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79",
        "difficile": "8..........36......7..9.2...5...7.......457.....1...3...1....68..85...1..9....4..",
    }
    # AC-3 seul, sans aucune recherche : combien de cases sont résolues ?
    domaines_ac3 = {}
    for _nom, _texte in GRILLES.items():
        _g = lire_grille(_texte)
        _D = domaines_initiaux(_g)
        _avant = sum(len(_D[x]) for x in CASES if not _g[x])
        _, _rev = ac3(_D, PAIRS, different)
        _apres = sum(len(_D[x]) for x in CASES if not _g[x])
        _resolues = sum(len(_D[x]) == 1 for x in CASES if not _g[x])
        _vides = sum(1 for x in CASES if not _g[x])
        domaines_ac3[_nom] = (_g, _D)
        print(f"{_nom:9s} : {_vides} cases vides ; AC-3 seul en résout {_resolues} ; "
              f"candidats {_avant} -> {_apres} ({_rev} révisions d'arcs)")
    return GRILLES, domaines_ac3


@app.cell
def _(mo):
    # Le bouton évite de relancer les comparaisons à chaque modification d'une cellule amont.
    bouton = mo.ui.run_button(label="Lancer les comparaisons")
    bouton
    return (bouton,)


@app.cell
def _(
    GRILLES,
    bouton,
    est_solution,
    lire_grille,
    mo,
    resoudre_mac,
    resoudre_mrv,
    resoudre_naif,
    time,
):
    # En script (make ex MODE=run|smoke), personne ne clique : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer les comparaisons ».*"))

    resultats_sudoku = {}  # (grille, méthode) -> (essais, secondes, valide)
    for _nom, _texte in GRILLES.items():
        _g = lire_grille(_texte)
        for _meth, _f in [("naïf", resoudre_naif), ("MRV", resoudre_mrv), ("MAC (AC-3)", resoudre_mac)]:
            _t0 = time.perf_counter()
            _sol, _st = _f(_g)
            _dt = time.perf_counter() - _t0
            _ok = est_solution(_sol, _g)
            resultats_sudoku[(_nom, _meth)] = (_st["essais"], _dt, _ok)
            _extra = f", {_st['revisions']} révisions d'arcs" if "revisions" in _st else ""
            print(f"{_nom:9s} {_meth:11s} : {_st['essais']:>7d} affectations essayées, "
                  f"{_dt * 1000:7.1f} ms, solution valide : {_ok}{_extra}")
    gain_mac = (resultats_sudoku[("difficile", "naïf")][0]
                / max(1, resultats_sudoku[("difficile", "MAC (AC-3)")][0]))
    print(f"\nGrille difficile : MAC essaie {gain_mac:.0f} fois moins d'affectations que le naïf")
    return gain_mac, resultats_sudoku


@app.cell
def _(mo):
    mo.md(r"""
    ## Partie 3 : planification STRIPS (monde des blocs)

    STRIPS (Fikes et Nilsson, 1971) décrit un état par un ensemble de faits et chaque
    action par trois listes : **préconditions**, faits **ajoutés**, faits **retirés**.
    Tout fait non mentionné reste inchangé. Le planificateur cherche une suite d'actions
    qui mène de l'état initial à un état où le but est vrai.

    Quatre actions pour un bras robot : `prendre(x)` sur la table, `poser(x)` sur la
    table, `empiler(x, y)` et `depiler(x, y)`.
    """)


@app.cell
def _(deque, heapq, itertools, namedtuple):
    Action = namedtuple("Action", "nom pre ajout retrait")

    def actions_blocs(blocs):
        """Instancie les 4 schémas d'action pour chaque bloc (et chaque paire de blocs)."""
        A = []
        for x in blocs:
            A.append(Action(f"prendre({x})",
                            pre=frozenset({("libre", x), ("sur_table", x), ("main_vide",)}),
                            ajout=frozenset({("tient", x)}),
                            retrait=frozenset({("libre", x), ("sur_table", x), ("main_vide",)})))
            A.append(Action(f"poser({x})",
                            pre=frozenset({("tient", x)}),
                            ajout=frozenset({("sur_table", x), ("libre", x), ("main_vide",)}),
                            retrait=frozenset({("tient", x)})))
            for y in blocs:
                if x == y:
                    continue
                A.append(Action(f"empiler({x},{y})",
                                pre=frozenset({("tient", x), ("libre", y)}),
                                ajout=frozenset({("sur", x, y), ("libre", x), ("main_vide",)}),
                                retrait=frozenset({("tient", x), ("libre", y)})))
                A.append(Action(f"depiler({x},{y})",
                                pre=frozenset({("sur", x, y), ("libre", x), ("main_vide",)}),
                                ajout=frozenset({("tient", x), ("libre", y)}),
                                retrait=frozenset({("sur", x, y), ("libre", x), ("main_vide",)})))
        return A

    def etat_depuis_piles(piles):
        """[['A', 'C'], ['B']] : A sur la table, C sur A ; B seul sur la table."""
        s = {("main_vide",)}
        for pile in piles:
            s.add(("sur_table", pile[0]))
            s |= {("sur", haut, bas) for bas, haut in itertools.pairwise(pile)}
            s.add(("libre", pile[-1]))
        return frozenset(s)

    def appliquer(etat, a):
        # Le cœur de STRIPS : (état − retraits) ∪ ajouts
        return (etat - a.retrait) | a.ajout

    def reconstruire(parent, etat):
        plan = []
        while parent[etat] is not None:
            etat, a = parent[etat]
            plan.append(a.nom)
        return plan[::-1]

    def planifier_largeur(init, but, actions):
        """Recherche en largeur dans l'espace d'états : plan le plus court garanti."""
        if but <= init:
            return [], 0
        parent, file, developpes = {init: None}, deque([init]), 0
        while file:
            s = file.popleft()
            developpes += 1
            for a in actions:
                if a.pre <= s:
                    t = appliquer(s, a)
                    if t not in parent:
                        parent[t] = (s, a)
                        if but <= t:
                            return reconstruire(parent, t), developpes
                        file.append(t)
        return None, developpes

    def planifier_glouton(init, but, actions):
        """Meilleur d'abord, heuristique h = nombre de faits du but encore faux.
        h est admissible (chaque action ajoute au plus un fait du but), mais la recherche
        ignore le coût déjà payé g : rien ne garantit le plan le plus court."""
        compteur = itertools.count()
        h = lambda s: len(but - s)
        tas, parent, developpes = [(h(init), next(compteur), init)], {init: None}, 0
        while tas:
            _, _, s = heapq.heappop(tas)
            developpes += 1
            if but <= s:
                return reconstruire(parent, s), developpes
            for a in actions:
                if a.pre <= s:
                    t = appliquer(s, a)
                    if t not in parent:
                        parent[t] = (s, a)
                        heapq.heappush(tas, (h(t), next(compteur), t))
        return None, developpes

    def nombre_etats(init, actions):
        """Taille de l'espace d'états atteignable (énumération complète en largeur)."""
        vus, file = {init}, deque([init])
        while file:
            s = file.popleft()
            for a in actions:
                if a.pre <= s:
                    t = appliquer(s, a)
                    if t not in vus:
                        vus.add(t)
                        file.append(t)
        return len(vus)

    return (
        actions_blocs,
        appliquer,
        etat_depuis_piles,
        nombre_etats,
        planifier_glouton,
        planifier_largeur,
    )


@app.cell
def _(
    actions_blocs,
    appliquer,
    etat_depuis_piles,
    planifier_glouton,
    planifier_largeur,
):
    # --- L'anomalie de Sussman : C sur A, B sur la table ; but : A sur B ET B sur C ---
    A3 = actions_blocs("ABC")
    init_sussman = etat_depuis_piles([["A", "C"], ["B"]])
    BUT_AB, BUT_BC = ("sur", "A", "B"), ("sur", "B", "C")
    but_sussman = frozenset({BUT_AB, BUT_BC})

    plan_sussman, dev_largeur = planifier_largeur(init_sussman, but_sussman, A3)
    _plan_g, _dev_g = planifier_glouton(init_sussman, but_sussman, A3)
    print(f"Largeur : {len(plan_sussman)} actions, {dev_largeur} états développés")
    for _i, _a in enumerate(plan_sussman, 1):
        print(f"  {_i}. {_a}")
    print(f"Glouton : {len(_plan_g)} actions, {_dev_g} états développés\n")

    # Stratégie « un sous-but après l'autre » (ce que faisaient les premiers planificateurs
    # linéaires) : atteindre le 1er sous-but, puis le 2e à partir de là, sans protéger le 1er.
    sous_buts_echouent = True
    for _ordre in [(BUT_AB, BUT_BC), (BUT_BC, BUT_AB)]:
        _s, _plan = init_sussman, []
        for _sb in _ordre:
            _p, _ = planifier_largeur(_s, frozenset({_sb}), A3)
            for _nom in _p:
                _s = appliquer(_s, next(a for a in A3 if a.nom == _nom))
            _plan += _p
        _atteint = but_sussman <= _s
        sous_buts_echouent &= not _atteint
        print(f"Sous-buts dans l'ordre {_ordre[0][1:]} puis {_ordre[1][1:]} : {len(_plan)} actions, "
              f"A sur B : {BUT_AB in _s}, B sur C : {BUT_BC in _s} -> but atteint : {_atteint}")
    return dev_largeur, plan_sussman, sous_buts_echouent


@app.cell
def _(
    SEED,
    actions_blocs,
    bouton,
    config,
    etat_depuis_piles,
    itertools,
    mo,
    nombre_etats,
    np,
    planifier_glouton,
    planifier_largeur,
    time,
):
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer les comparaisons ».*"))

    def piles_aleatoires(blocs, rng):
        """Répartit les blocs (dans un ordre aléatoire) en piles aléatoires."""
        ordre = list(rng.permutation(blocs))
        piles, courante = [], []
        for b in ordre:
            if courante and rng.random() < 0.4:
                piles.append(courante)
                courante = []
            courante.append(b)
        piles.append(courante)
        return piles

    def but_complet(piles):
        """But = configuration entièrement spécifiée (chaque bloc sur un bloc ou la table)."""
        but = set()
        for pile in piles:
            but.add(("sur_table", pile[0]))
            but |= {("sur", h, b) for b, h in itertools.pairwise(pile)}
        return frozenset(but)

    rng = np.random.default_rng(SEED)
    explosion = []  # (n, états atteignables, dev. largeur moyen, dev. glouton moyen, ...)
    for _n in range(2, config["n_blocs_max"] + 1):
        _blocs = [chr(ord("A") + i) for i in range(_n)]
        _A = actions_blocs(_blocs)
        _etats = nombre_etats(etat_depuis_piles([_blocs]), _A)
        _dl, _dg, _ll, _lg, _tl = [], [], [], [], 0.0
        for _ in range(config["n_instances"]):
            _init = etat_depuis_piles(piles_aleatoires(_blocs, rng))
            _but = but_complet(piles_aleatoires(_blocs, rng))
            _t0 = time.perf_counter()
            _p, _d = planifier_largeur(_init, _but, _A)
            _tl += time.perf_counter() - _t0
            _pg, _dgl = planifier_glouton(_init, _but, _A)
            _dl.append(_d)
            _dg.append(_dgl)
            _ll.append(len(_p))
            _lg.append(len(_pg))
        explosion.append((_n, len(_A), _etats, np.mean(_dl), np.mean(_dg), np.mean(_ll), np.mean(_lg), _tl))
        print(f"{_n} blocs : {len(_A):3d} actions, {_etats:6d} états atteignables ; "
              f"largeur : {np.mean(_dl):8.0f} états développés, plan {np.mean(_ll):4.1f} ; "
              f"glouton : {np.mean(_dg):6.0f} développés, plan {np.mean(_lg):4.1f} "
              f"({_tl:.2f} s en largeur)")
    return (explosion,)


@app.cell
def _(
    CASES,
    PALETTE,
    SMOKE,
    domaines_ac3,
    explosion,
    mo,
    np,
    plt,
    resultats_sudoku,
    sauvegarder_figure,
):
    # --- Figure : (a) affectations au sudoku, (b) candidats après AC-3, (c) explosion du monde des blocs ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 4.8), layout="constrained")

    _meths = ["naïf", "MRV", "MAC (AC-3)"]
    _x = np.arange(2)
    for _i, _m in enumerate(_meths):
        _vals = [max(1, resultats_sudoku[(g, _m)][0]) for g in ("facile", "difficile")]
        _barres = ax1.bar(_x + (_i - 1) * 0.27, _vals, 0.27, label=_m, color=PALETTE[_i])
        ax1.bar_label(_barres, labels=[f"{resultats_sudoku[(g, _m)][0]:,}".replace(",", " ")
                                       for g in ("facile", "difficile")], fontsize=8)
    ax1.set_yscale("log")
    ax1.set_xticks(_x, ["grille facile", "grille difficile"])
    ax1.set(ylabel="affectations essayées (log)", title="Sudoku : recherche contre propagation")
    ax1.legend(loc="upper left")
    ax1.set_ylim(0.7, 5e6)  # une barre à 0 essai est dessinée à 1 (échelle log)

    _g, _D = domaines_ac3["difficile"]
    _taille = np.array([[len(_D[(r, c)]) if not _g[(r, c)] else 0 for c in range(9)] for r in range(9)])
    _im = ax2.imshow(_taille, cmap="Blues", vmin=0, vmax=9)
    for (r, c) in CASES:
        if _g[(r, c)]:  # chiffre donné, en gras
            ax2.text(c, r, str(_g[(r, c)]), ha="center", va="center", fontsize=10, fontweight="bold")
        elif len(_D[(r, c)]) == 1:  # case déduite par AC-3
            ax2.text(c, r, str(next(iter(_D[(r, c)]))), ha="center", va="center", color=PALETTE[3])
        else:  # nombre de candidats restants
            ax2.text(c, r, str(len(_D[(r, c)])), ha="center", va="center", fontsize=7,
                     color="white" if len(_D[(r, c)]) >= 5 else "black")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.grid(False)
    for _k in (2.5, 5.5):
        ax2.axhline(_k, color="black", lw=1.5)
        ax2.axvline(_k, color="black", lw=1.5)
    plt.colorbar(_im, ax=ax2, fraction=0.046, label="candidats restants")
    ax2.set_title("Grille difficile : candidats par case\naprès AC-3 seul (gras : chiffres donnés)")

    _n = [e[0] for e in explosion]
    ax3.plot(_n, [e[2] for e in explosion], "o-", label="états atteignables")
    ax3.plot(_n, [e[3] for e in explosion], "s-", label="développés (largeur)")
    ax3.plot(_n, [e[4] for e in explosion], "^-", label="développés (glouton)")
    ax3.set_yscale("log")
    ax3.set_xticks(_n)
    ax3.set(xlabel="nombre de blocs", ylabel="nombre d'états (log)", title="Monde des blocs : explosion")
    ax3.legend()

    dossier = mo.notebook_dir() / "figures"
    chemins = sauvegarder_figure(fig, "raisonnement-smoke" if SMOKE else "raisonnement", dossier)
    print(f"Figure : {chemins[0]}")
    fig


@app.cell
def _(
    CAS,
    SMOKE,
    accord_arriere,
    accord_sat,
    cas_corrects,
    gain_mac,
    mo,
    plan_sussman,
    resultats_sudoku,
    sous_buts_echouent,
):
    # --- Bilan : seuils du README. En smoke, on vérifie seulement que tout tourne. ---
    _sudoku_ok = all(ok for (_, _, ok) in resultats_sudoku.values())
    controles = {
        f"chaînage avant : {cas_corrects}/{len(CAS)} cas corrects": cas_corrects == len(CAS),
        f"chaînage arrière : {accord_arriere}/{len(CAS)} cas corrects": accord_arriere == len(CAS),
        f"accord SAT : {accord_sat}/{len(CAS)}": accord_sat == len(CAS),
        "sudoku : 6 solutions valides": _sudoku_ok,
        f"MAC : {gain_mac:.0f} fois moins d'affectations que le naïf (seuil 100)": gain_mac >= 100,
        f"Sussman : plan de {len(plan_sussman)} actions (attendu 6)": len(plan_sussman) == 6,
        "Sussman : les deux ordres de sous-buts échouent": sous_buts_echouent,
    }
    for _k, _v in controles.items():
        print(f"[{'ok' if _v else 'ÉCHEC'}] {_k}")
    if not SMOKE and not all(controles.values()):
        raise SystemExit("Au moins un seuil du README n'est pas atteint")
    mo.md("**Bilan** : " + " ; ".join(f"{k} {'✓' if v else '✗'}" for k, v in controles.items()))


if __name__ == "__main__":
    app.run()
