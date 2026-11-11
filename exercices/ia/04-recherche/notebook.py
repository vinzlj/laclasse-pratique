# Exercice 04 : chercher dans un espace d'états (chapitre 04).
#
#   make ex NUM=04              → éditeur marimo (interactif)
#   make ex NUM=04 MODE=run     → script complet (vérifie les seuils du README)
#   make ex NUM=04 MODE=smoke   → test rapide (< 30 s)
#
# Python pur : aucun GPU, aucun apprentissage, aucune donnée téléchargée.
# Les algorithmes sont des fonctions « @app.function » : elles sont importables depuis
# ce fichier, ce qui permet aux scripts de figures du site
# (site/scripts/figures/ch04_*.py) de refaire EXACTEMENT les mêmes calculs.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import heapq
    import itertools
    import math
    import random
    import time
    from collections import deque

    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    from ia_commun import PALETTE, is_smoke, sauvegarder_figure, seed_everything, style_commun


@app.cell
def _():
    mo.md(r"""
    # Exercice 04 : chercher dans un espace d'états

    Avant l'apprentissage, l'IA « symbolique » résolvait un problème en **explorant** les
    états possibles : positions sur une carte, configurations d'un puzzle, plateaux d'un
    jeu. Rien n'est appris ; tout est dans la façon de choisir quel état regarder ensuite.

    1. **Grille** : largeur (BFS), profondeur (DFS), coût uniforme (Dijkstra), glouton et
       A\* sur la même carte, avec des cases de boue qui coûtent 5.
    2. **Taquin à 8 cases** : A\* avec trois heuristiques admissibles, puis une
       heuristique non admissible.
    3. **Morpion** : minimax complet, puis alpha-bêta, avec et sans bon ordre des coups.
    4. **Encadré « optimiser sans gradient »** : 8 reines par escalade, recuit simulé et
       algorithme génétique.

    **Fonction de perte : aucune.** On ne minimise pas une loss sur des données : on
    cherche un chemin de **coût minimal** (parties 1 et 2), la **valeur minimax** d'une
    position (partie 3) ou un état sans conflit (partie 4).
    """)
    return


@app.cell
def _():
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    # Écart assumé au gabarit : ni get_device() (Python pur, CPU seul, aucun tenseur) ni
    # mo.ui.run_button (le run complet dure une dizaine de secondes, sans entraînement).
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        n_taquins=3 if SMOKE else 10,        # instances aléatoires du taquin
        n_taquins_h0=1 if SMOKE else 10,     # h = 0 (coût uniforme) est le plus lent
        n_parties=20 if SMOKE else 200,      # parties de morpion contre un joueur aléatoire
        n_reines=20 if SMOKE else 200,       # départs aléatoires pour les 8 reines
        n_genetique=3 if SMOKE else 20,      # lancements de l'algorithme génétique
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SEED, SMOKE, config


@app.cell
def _():
    mo.md(r"""
    ## 1. Chercher un chemin sur une grille

    `D` est le départ, `B` le but, `#` un mur, `~` de la boue (coût 5 au lieu de 1).
    On se déplace en 4-connexité (haut, droite, bas, gauche). La carte contient un
    **piège en U** ouvert vers le départ et une **bande de boue** qu'on ne contourne que
    par la dernière ligne.

    Toutes les stratégies partagent le même squelette : une **frontière** (l'ensemble
    ouvert) d'états découverts mais pas encore développés, et un **ensemble fermé**
    d'états déjà développés. Seul change l'ordre de sortie de la frontière :

    | Stratégie | Structure | Priorité | Optimal ? |
    |---|---|---|---|
    | Largeur (BFS) | file FIFO | ordre d'arrivée | en nombre de pas seulement |
    | Profondeur (DFS) | pile LIFO | dernier arrivé | non |
    | Coût uniforme (Dijkstra) | tas | $g(n)$ | oui |
    | Glouton | tas | $h(n)$ | non |
    | A\* | tas | $f(n) = g(n) + h(n)$ | oui si $h$ est admissible |

    $g(n)$ est le coût déjà payé depuis le départ, $h(n)$ une estimation du coût restant.
    Ici $h$ est la distance de Manhattan : elle ne surestime jamais (chaque pas coûte au
    moins 1), elle est donc **admissible**.
    """)
    return


@app.function
def carte_u():
    """Carte principale (15 × 25) : piège en U et bande de boue. Identique dans la démo AStarGrid."""
    return [
        "....................~~...",
        "....................~~...",
        "..........#######...~~...",
        "................#...~~...",
        "................#...~~...",
        "................#...~~...",
        "................#...~~...",
        "..D.............#...~~.B.",
        "................#...~~...",
        "................#...~~...",
        "................#...~~...",
        "................#...~~...",
        "..........#######...~~...",
        "....................~~...",
        ".........................",
    ]


@app.function
def carte_ouverte():
    """Même taille, sans obstacle : le cas idéal pour une heuristique."""
    lignes = ["." * 25 for _ in range(15)]
    lignes[7] = "..D" + "." * 20 + "B."
    return lignes


@app.function
def lire_carte(carte):
    """Renvoie (départ, but) sous forme de coordonnées (ligne, colonne)."""
    for i, ligne in enumerate(carte):
        for j, c in enumerate(ligne):
            if c == "D":
                depart = (i, j)
            elif c == "B":
                but = (i, j)
    return depart, but


@app.function
def cout_case(c):
    """Coût pour ENTRER dans une case : 5 dans la boue, 1 ailleurs."""
    return 5 if c == "~" else 1


@app.function
def voisins(carte, p):
    """Voisins accessibles de p, dans un ordre FIXE (haut, droite, bas, gauche), avec leur coût."""
    i, j = p
    for di, dj in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        a, b = i + di, j + dj
        if 0 <= a < len(carte) and 0 <= b < len(carte[0]) and carte[a][b] != "#":
            yield (a, b), cout_case(carte[a][b])


@app.function
def manhattan(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


@app.function
def rechercher(carte, strategie, poids=1.0):
    """Recherche de chemin générique.

    strategie : "largeur", "profondeur", "cout_uniforme", "glouton" ou "a_etoile"
    (poids > 1 donne un A* pondéré, f = g + poids × h, plus rapide mais non optimal).
    Renvoie un dict : chemin, coût, nombre de pas, ordre de développement, frontière finale,
    taille maximale de la frontière.
    """
    depart, but = lire_carte(carte)
    h = lambda p: manhattan(p, but)  # noqa: E731
    parent = {depart: None}
    g = {depart: 0}
    developpes = []          # ordre dans lequel les états sont développés (fermés)
    ferme = set()
    taille_max = 1
    compteur = itertools.count()  # départage les égalités : premier arrivé, premier servi

    if strategie == "largeur":
        # BFS : un état est marqué dès qu'il entre dans la file (il ne sera jamais
        # « amélioré » : BFS ne regarde pas les coûts).
        file = deque([depart])
        while file:
            p = file.popleft()
            developpes.append(p)
            if p == but:
                break
            for q, c in voisins(carte, p):
                if q not in parent:
                    parent[q], g[q] = p, g[p] + c
                    file.append(q)
            taille_max = max(taille_max, len(file))
        frontiere = list(file)
    elif strategie == "profondeur":
        # DFS en version graphe : pile, état fermé quand on le dépile.
        pile = [depart]
        while pile:
            p = pile.pop()
            if p in ferme:
                continue
            ferme.add(p)
            developpes.append(p)
            if p == but:
                break
            # empilés à l'envers pour que « haut » soit essayé en premier
            for q, c in reversed(list(voisins(carte, p))):
                if q not in ferme:
                    parent[q], g[q] = p, g[p] + c
                    pile.append(q)
            taille_max = max(taille_max, len(pile))
        frontiere = [q for q in pile if q not in ferme]
    else:
        def priorite(gq, q):
            if strategie == "cout_uniforme":
                return (gq, next(compteur))
            if strategie == "glouton":
                return (h(q), next(compteur))
            # A* : à f égal, on préfère l'état le plus proche du but (h plus petit)
            return (gq + poids * h(q), h(q), next(compteur))

        tas = [(priorite(0, depart), depart)]
        while tas:
            _, p = heapq.heappop(tas)
            if p in ferme:
                continue  # entrée périmée : p a déjà été développé par un meilleur chemin
            ferme.add(p)
            developpes.append(p)
            if p == but:
                break
            for q, c in voisins(carte, p):
                ng = g[p] + c
                if q not in ferme and (q not in g or ng < g[q]):
                    g[q], parent[q] = ng, p
                    heapq.heappush(tas, (priorite(ng, q), q))
            taille_max = max(taille_max, len(tas))
        frontiere = list({q for _, q in tas if q not in ferme})

    chemin, p = [], but
    while p is not None:
        chemin.append(p)
        p = parent[p]
    chemin.reverse()
    cout = sum(cout_case(carte[i][j]) for i, j in chemin[1:])
    return dict(
        chemin=chemin, cout=cout, pas=len(chemin) - 1, developpes=developpes,
        n_developpes=len(developpes), frontiere=frontiere, taille_max=taille_max,
    )


@app.function
def dessiner_grille(ax, carte, res, titre):
    """Murs, boue, états développés, frontière finale et chemin trouvé."""
    n, m = len(carte), len(carte[0])
    # 0 libre, 1 boue, 2 mur, 3 développé, 4 développé dans la boue, 5 frontière
    code = [[{"#": 2, "~": 1}.get(c, 0) for c in ligne] for ligne in carte]
    for i, j in res["developpes"]:
        code[i][j] = 4 if carte[i][j] == "~" else 3
    for i, j in res["frontiere"]:
        code[i][j] = 5
    couleurs = ListedColormap(["#f4f4f4", "#c8b48a", "#3a3a3a", "#9ecae9", "#7f9fb8", "#f6c86b"])
    ax.imshow(code, cmap=couleurs, vmin=0, vmax=5, interpolation="nearest")
    ys, xs = zip(*res["chemin"])
    ax.plot(xs, ys, color=PALETTE[3], lw=2.2)
    (di, dj), (bi, bj) = lire_carte(carte)
    ax.plot(dj, di, "o", color=PALETTE[2], ms=7)
    ax.plot(bj, bi, "*", color=PALETTE[3], ms=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(titre, fontsize=10)


@app.cell
def _():
    STRATEGIES = [
        ("largeur", 1.0, "Largeur (BFS)"),
        ("profondeur", 1.0, "Profondeur (DFS)"),
        ("cout_uniforme", 1.0, "Coût uniforme"),
        ("glouton", 1.0, "Glouton (h seul)"),
        ("a_etoile", 1.0, "A* (g + h)"),
        ("a_etoile", 2.0, "A* pondéré (g + 2h)"),
    ]
    resultats_grille = {}
    print(f"{'stratégie':<22}{'développés':>11}{'coût':>6}{'pas':>5}{'frontière max':>15}")
    for _nom_carte, _carte in (("carte en U", carte_u()), ("carte ouverte", carte_ouverte())):
        print(f"--- {_nom_carte} (15 × 25, {sum(c != '#' for l in _carte for c in l)} cases libres)")
        for _s, _w, _lib in STRATEGIES:
            _r = rechercher(_carte, _s, _w)
            resultats_grille[(_nom_carte, _lib)] = _r
            print(f"{_lib:<22}{_r['n_developpes']:>11}{_r['cout']:>6}{_r['pas']:>5}{_r['taille_max']:>15}")
    return STRATEGIES, resultats_grille


@app.cell
def _(SMOKE, STRATEGIES, resultats_grille):
    _fig, _axes = plt.subplots(2, 3, figsize=(12, 6.6))
    for _ax, (_s, _w, _lib) in zip(_axes.flat, STRATEGIES):
        _r = resultats_grille[("carte en U", _lib)]
        dessiner_grille(_ax, carte_u(), _r, f"{_lib}\n{_r['n_developpes']} développés, coût {_r['cout']}")
    _fig.suptitle("Bleu : états développés ; orange : frontière restante ; rouge : chemin trouvé")
    _fig.tight_layout()
    sauvegarder_figure(_fig, "grille-smoke" if SMOKE else "grille", mo.notebook_dir() / "figures")
    _fig
    return


@app.cell
def _():
    mo.md(r"""
    **À lire dans le tableau.** BFS trouve le chemin le plus **court en pas** (33), qui
    traverse la boue (coût 41). Le coût uniforme et A\* trouvent le chemin le **moins
    cher** (35, par la dernière ligne). A\* y arrive en développant moins d'états, et sur
    la carte ouverte il ne développe presque que le chemin lui-même. Le glouton fonce vers
    le but, se fait piéger par le U et rend un chemin plus cher.

    ## 2. Taquin à 8 cases : la qualité de l'heuristique

    Un état = une permutation des tuiles 1 à 8 et du trou (0) ; 9!/2 = 181 440 états
    atteignables. Chaque déplacement coûte 1. Trois heuristiques **admissibles** :

    - $h_0 = 0$ : A\* devient le coût uniforme ;
    - $h_1$ = nombre de tuiles mal placées ;
    - $h_2$ = somme des distances de Manhattan des tuiles à leur place.

    $h_2 \ge h_1 \ge h_0$ partout : on dit que $h_2$ **domine** $h_1$. Une heuristique
    dominante, toujours admissible, développe moins d'états. On essaie enfin $2h_2$, qui
    surestime : plus rapide, mais sans garantie d'optimalité.
    """)
    return


@app.function
def taquin_but():
    return (1, 2, 3, 4, 5, 6, 7, 8, 0)


@app.function
def taquin_successeurs(etat):
    """Faire glisser une tuile dans le trou (0)."""
    z = etat.index(0)
    i, j = divmod(z, 3)
    for di, dj in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        a, b = i + di, j + dj
        if 0 <= a < 3 and 0 <= b < 3:
            k = a * 3 + b
            nouveau = list(etat)
            nouveau[z], nouveau[k] = nouveau[k], 0
            yield tuple(nouveau)


@app.function
def h_nulle(etat):
    return 0


@app.function
def h_mal_placees(etat):
    but = taquin_but()
    return sum(1 for i, v in enumerate(etat) if v and v != but[i])


@app.function
def h_manhattan(etat):
    # position cible de la tuile v : (v - 1) // 3, (v - 1) % 3
    return sum(abs(i // 3 - (v - 1) // 3) + abs(i % 3 - (v - 1) % 3) for i, v in enumerate(etat) if v)


@app.function
def taquin_soluble(etat):
    """Sur une grille 3 × 3, soluble ⇔ nombre pair d'inversions (trou ignoré)."""
    t = [v for v in etat if v]
    return sum(1 for a in range(8) for b in range(a + 1, 8) if t[a] > t[b]) % 2 == 0


@app.function
def a_etoile_taquin(depart, h, poids=1):
    """A* sur le taquin. Renvoie (longueur de la solution, états développés)."""
    but = taquin_but()
    compteur = itertools.count()
    g = {depart: 0}
    tas = [(poids * h(depart), next(compteur), depart)]
    ferme = set()
    while tas:
        _, _, e = heapq.heappop(tas)
        if e in ferme:
            continue
        ferme.add(e)
        if e == but:
            return g[e], len(ferme)
        for f in taquin_successeurs(e):
            ng = g[e] + 1
            if f not in ferme and ng < g.get(f, math.inf):
                g[f] = ng
                heapq.heappush(tas, (ng + poids * h(f), next(compteur), f))
    raise ValueError("instance insoluble")


@app.function
def instances_taquin(n, graine):
    rng = random.Random(graine)
    res = []
    while len(res) < n:
        t = list(range(9))
        rng.shuffle(t)
        if taquin_soluble(tuple(t)):
            res.append(tuple(t))
    return res


@app.cell
def _(SEED, config):
    taquins = instances_taquin(config["n_taquins"], SEED)
    HEURISTIQUES = [
        ("h0 = 0", h_nulle, 1, config["n_taquins_h0"]),
        ("h1 mal placées", h_mal_placees, 1, config["n_taquins"]),
        ("h2 Manhattan", h_manhattan, 1, config["n_taquins"]),
        ("2 × h2 (non admissible)", h_manhattan, 2, config["n_taquins"]),
    ]
    resultats_taquin = {}
    print(f"{'heuristique':<26}{'dév. moyens':>12}{'durée':>8}  longueurs")
    for _nom, _h, _w, _n in HEURISTIQUES:
        _t0 = time.perf_counter()
        _r = [a_etoile_taquin(e, _h, _w) for e in taquins[:_n]]
        _duree = time.perf_counter() - _t0
        resultats_taquin[_nom] = _r
        _moy = sum(x[1] for x in _r) / len(_r)
        print(f"{_nom:<26}{_moy:>12.0f}{_duree:>7.2f}s  {[x[0] for x in _r]}")
    return HEURISTIQUES, resultats_taquin


@app.cell
def _():
    mo.md(r"""
    ## 3. Morpion : minimax et élagage alpha-bêta

    MAX (les X) choisit le coup qui maximise la valeur, MIN (les O) celui qui la minimise ;
    valeur d'une feuille : +1 si X gagne, −1 si O gagne, 0 pour un nul. Alpha-bêta garde
    deux bornes : $\alpha$, ce que MAX est déjà sûr d'obtenir, $\beta$, ce que MIN est déjà
    sûr d'imposer. Dès que $\alpha \ge \beta$, les frères restants ne peuvent plus changer
    la décision : on les **élague**. Même valeur, beaucoup moins de nœuds, et d'autant
    moins que les meilleurs coups sont essayés en premier.

    D'abord un arbre à la main (profondeur 2, trois coups pour MAX, trois réponses pour
    MIN), puis le morpion complet.
    """)
    return


@app.function
def minimax_arbre(noeud, est_max, compte):
    """Minimax sur un arbre explicite (listes imbriquées, feuilles = nombres)."""
    if not isinstance(noeud, list):
        compte["feuilles"] += 1
        return noeud
    vals = [minimax_arbre(f, not est_max, compte) for f in noeud]
    return max(vals) if est_max else min(vals)


@app.function
def alphabeta_arbre(noeud, est_max, alpha, beta, compte):
    if not isinstance(noeud, list):
        compte["feuilles"] += 1
        compte["vues"].append(noeud)
        return noeud
    v = -math.inf if est_max else math.inf
    for f in noeud:
        x = alphabeta_arbre(f, not est_max, alpha, beta, compte)
        if est_max:
            v, alpha = max(v, x), max(alpha, x)
        else:
            v, beta = min(v, x), min(beta, x)
        if alpha >= beta:
            break  # élagage : les frères restants ne seront jamais choisis
    return v


@app.cell
def _():
    ARBRE = [[3, 12, 8], [2, 4, 6], [14, 5, 2]]
    _c1 = {"feuilles": 0}
    _v1 = minimax_arbre(ARBRE, True, _c1)
    _c2 = {"feuilles": 0, "vues": []}
    _v2 = alphabeta_arbre(ARBRE, True, -math.inf, math.inf, _c2)
    print(f"Arbre {ARBRE}")
    print(f"minimax    : valeur {_v1}, {_c1['feuilles']} feuilles évaluées")
    print(f"alpha-bêta : valeur {_v2}, {_c2['feuilles']} feuilles évaluées : {_c2['vues']}")
    arbre_res = dict(v_minimax=_v1, v_ab=_v2, f_minimax=_c1["feuilles"], f_ab=_c2["feuilles"])
    return (arbre_res,)


@app.function
def morpion_gagnant(p):
    for a, b, c in ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)):
        if p[a] != " " and p[a] == p[b] == p[c]:
            return p[a]
    return None


@app.function
def morpion_minimax(p, joueur, compte):
    """Minimax complet ; p est une liste de 9 cases (« X », « O » ou « »), compte[0] = nœuds visités."""
    compte[0] += 1
    g = morpion_gagnant(p)
    if g:
        return 1 if g == "X" else -1
    coups = [i for i in range(9) if p[i] == " "]
    if not coups:
        return 0
    autre = "O" if joueur == "X" else "X"
    vals = []
    for i in coups:
        p[i] = joueur
        vals.append(morpion_minimax(p, autre, compte))
        p[i] = " "
    return max(vals) if joueur == "X" else min(vals)


@app.function
def morpion_alphabeta(p, joueur, alpha, beta, ordre, compte):
    """Alpha-bêta ; « ordre » = ordre dans lequel on essaie les cases."""
    compte[0] += 1
    g = morpion_gagnant(p)
    if g:
        return 1 if g == "X" else -1
    coups = [i for i in ordre if p[i] == " "]
    if not coups:
        return 0
    autre = "O" if joueur == "X" else "X"
    v = -2 if joueur == "X" else 2
    for i in coups:
        p[i] = joueur
        x = morpion_alphabeta(p, autre, alpha, beta, ordre, compte)
        p[i] = " "
        if joueur == "X":
            v, alpha = max(v, x), max(alpha, x)
        else:
            v, beta = min(v, x), min(beta, x)
        if alpha >= beta:
            break
    return v


@app.cell
def _():
    ORDRE_NATUREL = tuple(range(9))
    ORDRE_BON = (4, 0, 2, 6, 8, 1, 3, 5, 7)  # centre, coins, bords
    resultats_morpion = {}
    for _nom, _f in (
        ("minimax", lambda c: morpion_minimax([" "] * 9, "X", c)),
        ("alpha-bêta", lambda c: morpion_alphabeta([" "] * 9, "X", -2, 2, ORDRE_NATUREL, c)),
        ("alpha-bêta, bon ordre", lambda c: morpion_alphabeta([" "] * 9, "X", -2, 2, ORDRE_BON, c)),
    ):
        _c = [0]
        _t0 = time.perf_counter()
        _v = _f(_c)
        resultats_morpion[_nom] = dict(valeur=_v, noeuds=_c[0], duree=time.perf_counter() - _t0)
    _ref = resultats_morpion["minimax"]["noeuds"]
    for _nom, _r in resultats_morpion.items():
        _n_txt = f"{_r['noeuds']:,}".replace(",", " ")
        print(f"{_nom:<24} valeur {_r['valeur']:+d}  {_n_txt:>8} nœuds ({_r['noeuds'] / _ref:6.1%})  {_r['duree']:.2f} s")
    return ORDRE_BON, resultats_morpion


@app.function
def meilleur_coup(p, joueur, ordre):
    """Coup choisi par alpha-bêta (premier coup de valeur optimale dans l'ordre donné)."""
    autre = "O" if joueur == "X" else "X"
    meilleur, best_v = None, None
    for i in ordre:
        if p[i] != " ":
            continue
        p[i] = joueur
        v = morpion_alphabeta(p, autre, -2, 2, ordre, [0])
        p[i] = " "
        if best_v is None or (v > best_v if joueur == "X" else v < best_v):
            meilleur, best_v = i, v
    return meilleur


@app.function
def partie_contre_hasard(ia, rng, ordre):
    """Une partie : l'IA joue « ia » (X ou O), l'autre joue au hasard. Renvoie le gagnant ou None."""
    p, joueur = [" "] * 9, "X"
    while morpion_gagnant(p) is None and " " in p:
        if joueur == ia:
            i = meilleur_coup(p, joueur, ordre)
        else:
            i = rng.choice([k for k in range(9) if p[k] == " "])
        p[i] = joueur
        joueur = "O" if joueur == "X" else "X"
    return morpion_gagnant(p)


@app.cell
def _(ORDRE_BON, SEED, config):
    _rng = random.Random(SEED)
    bilan_parties = {"victoires": 0, "nuls": 0, "defaites": 0}
    _t0 = time.perf_counter()
    for _k in range(config["n_parties"]):
        _ia = "X" if _k % 2 == 0 else "O"  # l'IA commence une partie sur deux
        _g = partie_contre_hasard(_ia, _rng, ORDRE_BON)
        bilan_parties["nuls" if _g is None else ("victoires" if _g == _ia else "defaites")] += 1
    print(f"{config['n_parties']} parties contre un joueur aléatoire : {bilan_parties} "
          f"({time.perf_counter() - _t0:.1f} s)")
    return (bilan_parties,)


@app.cell
def _():
    mo.md(r"""
    ## 4. Encadré : optimiser sans gradient (8 reines)

    Placer 8 reines sur un échiquier sans qu'aucune paire ne se menace. Un état = une
    reine par colonne (8 lignes à choisir, $8^8 \approx 16{,}8$ millions d'états) ; on
    minimise le **nombre de paires en conflit**. Pas de chemin cette fois : seul l'état
    final compte. Trois méthodes de **recherche locale** :

    - **escalade** (*hill climbing*) : prendre le meilleur voisin tant qu'il améliore ;
      s'arrête dans le premier minimum local ;
    - **recuit simulé** (*simulated annealing*) : voisin au hasard, accepté s'il améliore,
      sinon avec la probabilité $e^{-\Delta/T}$ ; la température $T$ baisse peu à peu ;
    - **algorithme génétique** : une population d'états, sélection des meilleurs,
      croisement de deux parents, mutation.
    """)
    return


@app.function
def conflits(etat):
    """Nombre de paires de reines qui se menacent (même ligne ou même diagonale)."""
    n = len(etat)
    return sum(
        1 for a in range(n) for b in range(a + 1, n)
        if etat[a] == etat[b] or abs(etat[a] - etat[b]) == b - a
    )


@app.function
def escalade(etat):
    """Escalade par plus forte pente ; renvoie l'état final (minimum local ou solution)."""
    etat = list(etat)
    while True:
        actuel = conflits(etat)
        meilleur, coup = actuel, None
        for col in range(8):
            for lig in range(8):
                if lig != etat[col]:
                    ancien, etat[col] = etat[col], lig
                    c = conflits(etat)
                    etat[col] = ancien
                    if c < meilleur:
                        meilleur, coup = c, (col, lig)
        if coup is None:
            return etat
        etat[coup[0]] = coup[1]


@app.function
def recuit(etat, rng, t0=2.0, refroidissement=0.995, n_pas=5000):
    """Recuit simulé ; s'arrête à la première solution ou après n_pas."""
    etat, T = list(etat), t0
    c = conflits(etat)
    for _ in range(n_pas):
        if c == 0:
            break
        col, lig = rng.randrange(8), rng.randrange(8)
        ancien, etat[col] = etat[col], lig
        nc = conflits(etat)
        delta = nc - c
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            c = nc
        else:
            etat[col] = ancien
        T = max(T * refroidissement, 1e-3)
    return etat


@app.function
def genetique(rng, taille=200, generations=300, p_mutation=0.5):
    """Algorithme génétique minimal. Renvoie (meilleur état, générations utilisées)."""
    pop = [[rng.randrange(8) for _ in range(8)] for _ in range(taille)]
    for gen in range(generations):
        pop.sort(key=conflits)
        if conflits(pop[0]) == 0:
            return pop[0], gen
        elite = pop[: taille // 2]  # sélection : la meilleure moitié
        enfants = []
        while len(enfants) < taille - len(elite):
            a, b = rng.sample(elite, 2)
            coupe = rng.randrange(1, 8)
            enfant = a[:coupe] + b[coupe:]  # croisement en un point
            if rng.random() < p_mutation:
                enfant[rng.randrange(8)] = rng.randrange(8)
            enfants.append(enfant)
        pop = elite + enfants
    pop.sort(key=conflits)
    return pop[0], generations


@app.cell
def _(SEED, config):
    _rng = random.Random(SEED)
    _departs = [[_rng.randrange(8) for _ in range(8)] for _ in range(config["n_reines"])]
    _t0 = time.perf_counter()
    _ok_esc = sum(conflits(escalade(e)) == 0 for e in _departs)
    _t1 = time.perf_counter()
    _rng_r = random.Random(SEED + 1)
    _ok_rec = sum(conflits(recuit(e, _rng_r)) == 0 for e in _departs)
    _t2 = time.perf_counter()
    _rng_g = random.Random(SEED + 2)
    _gen = [genetique(_rng_g) for _ in range(config["n_genetique"])]
    _ok_gen = sum(conflits(e) == 0 for e, _ in _gen)
    _t3 = time.perf_counter()
    _gen_ok = [g for e, g in _gen if conflits(e) == 0]
    reines = dict(
        escalade=_ok_esc / len(_departs), recuit=_ok_rec / len(_departs),
        genetique=_ok_gen / len(_gen), n=len(_departs), n_gen=len(_gen),
        gen_moy=(sum(_gen_ok) / len(_gen_ok)) if _gen_ok else float("nan"),
    )
    print(f"escalade  : {reines['escalade']:.0%} de solutions sur {len(_departs)} départs ({_t1 - _t0:.1f} s)")
    print(f"recuit    : {reines['recuit']:.0%} de solutions sur {len(_departs)} départs ({_t2 - _t1:.1f} s)")
    print(f"génétique : {reines['genetique']:.0%} de solutions sur {len(_gen)} lancements, "
          f"{reines['gen_moy']:.0f} générations en moyenne ({_t3 - _t2:.1f} s)")
    return (reines,)


@app.cell
def _(HEURISTIQUES, SMOKE, reines, resultats_morpion, resultats_taquin):
    # --- Figure de synthèse : effort de recherche selon la méthode ---
    _fig, (_a1, _a2, _a3) = plt.subplots(1, 3, figsize=(13, 3.8))
    _noms = [n for n, *_ in HEURISTIQUES]
    _moy = [sum(x[1] for x in resultats_taquin[n]) / len(resultats_taquin[n]) for n in _noms]
    _a1.bar(range(len(_noms)), _moy, color=PALETTE[:4])
    _a1.set_yscale("log")
    _a1.set_xticks(range(len(_noms)), ["h0", "h1", "h2", "2h2"])
    _a1.set(title="Taquin : états développés (moyenne)", ylabel="états (échelle log)")
    _nm = list(resultats_morpion)
    _a2.bar(range(3), [resultats_morpion[n]["noeuds"] for n in _nm], color=PALETTE[:3])
    _a2.set_yscale("log")
    _a2.set_xticks(range(3), ["minimax", "alpha-bêta", "α-β ordonné"])
    _a2.set(title="Morpion : nœuds visités", ylabel="nœuds (échelle log)")
    _a3.bar(range(3), [100 * reines[k] for k in ("escalade", "recuit", "genetique")], color=PALETTE[:3])
    _a3.set_xticks(range(3), ["escalade", "recuit", "génétique"])
    _a3.set(title="8 reines : taux de réussite", ylabel="% de solutions", ylim=(0, 105))
    _fig.tight_layout()
    sauvegarder_figure(_fig, "synthese-smoke" if SMOKE else "synthese", mo.notebook_dir() / "figures")
    _fig
    return


@app.cell
def _(arbre_res, bilan_parties, reines, resultats_grille, resultats_morpion, resultats_taquin):
    # --- Bilan : seuils du README. En smoke, on vérifie seulement que tout tourne.
    _u = lambda lib: resultats_grille[("carte en U", lib)]  # noqa: E731
    _o = lambda lib: resultats_grille[("carte ouverte", lib)]  # noqa: E731
    _h0 = [x[0] for x in resultats_taquin["h0 = 0"]]
    _n = len(_h0)
    _moy = {k: sum(x[1] for x in v[:_n]) / _n for k, v in resultats_taquin.items()}
    _mm = resultats_morpion
    verifs = {
        "A* trouve le coût optimal (= coût uniforme)": _u("A* (g + h)")["cout"] == _u("Coût uniforme")["cout"],
        "A* développe moins d'états que le coût uniforme": _u("A* (g + h)")["n_developpes"] < _u("Coût uniforme")["n_developpes"],
        "BFS rend un chemin plus cher que l'optimum (boue)": _u("Largeur (BFS)")["cout"] > _u("Coût uniforme")["cout"],
        "Glouton : chemin plus cher que l'optimum": _u("Glouton (h seul)")["cout"] > _u("A* (g + h)")["cout"],
        "Carte ouverte : A* développe au plus 2 × le nombre de pas": _o("A* (g + h)")["n_developpes"] <= 2 * _o("A* (g + h)")["pas"],
        "Taquin : h0, h1, h2 donnent les mêmes longueurs": all(
            [x[0] for x in resultats_taquin[k][:_n]] == _h0 for k in ("h1 mal placées", "h2 Manhattan")),
        "Taquin : h2 développe ≥ 10 × moins que h1 et h1 ≥ 5 × moins que h0": (
            _moy["h2 Manhattan"] * 10 <= _moy["h1 mal placées"] and _moy["h1 mal placées"] * 5 <= _moy["h0 = 0"]),
        "Taquin : 2h2 jamais plus court que l'optimum": all(
            a[0] >= b[0] for a, b in zip(resultats_taquin["2 × h2 (non admissible)"], resultats_taquin["h2 Manhattan"])),
        "Arbre à la main : même valeur, 7 feuilles sur 9": arbre_res["v_minimax"] == arbre_res["v_ab"] and arbre_res["f_ab"] == 7,
        "Morpion : même valeur (0, nul) partout": all(r["valeur"] == 0 for r in _mm.values()),
        "Morpion : alpha-bêta ≤ 5 % des nœuds de minimax": _mm["alpha-bêta"]["noeuds"] <= 0.05 * _mm["minimax"]["noeuds"],
        "Morpion : l'IA ne perd jamais contre le hasard": bilan_parties["defaites"] == 0,
        "8 reines : recuit ≥ 2 × escalade": reines["recuit"] >= 2 * reines["escalade"],
    }
    for _k, _ok in verifs.items():
        print(("OK    " if _ok else "ÉCHEC ") + _k)
    return (verifs,)


@app.cell
def _(SMOKE, verifs):
    _echecs = [k for k, ok in verifs.items() if not ok]
    if not SMOKE and _echecs:
        raise SystemExit("Seuils non atteints : " + " ; ".join(_echecs))
    mo.md(f"**Bilan** : {sum(verifs.values())} vérifications sur {len(verifs)} passent.")
    return


if __name__ == "__main__":
    app.run()
