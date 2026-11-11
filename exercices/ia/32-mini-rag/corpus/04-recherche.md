# 04. Chercher : espaces d’états, heuristiques, jeux

Repère historique (1950-1997) : Résoudre un problème en explorant un espace d’états, sans rien apprendre.

En 1950, deux textes posent le programme : Turing demande si une machine peut penser et propose un test pour en décider ; Shannon décrit comment un ordinateur pourrait jouer aux échecs en examinant les coups possibles. En 1955, la proposition de Dartmouth donne un nom au domaine. Pendant trente ans, l’« intelligence » d’un programme sera d’abord sa capacité à chercher : parcourir les situations possibles d’un problème jusqu’à trouver une solution, en évitant d’en examiner trop. A* (1968) trouve le plus court chemin en se laissant guider par une estimation ; l’élagage alpha-bêta rend la recherche praticable dans les jeux ; en 1997, Deep Blue bat le champion du monde d’échecs. Aucun de ces programmes n’apprend de son expérience.

## D’où l’on part : Turing, Shannon, Dartmouth

En octobre 1950, Alan Turing publie dans la revue de philosophie *Mind* « Computing Machinery and Intelligence ». Il écarte la question « Les machines peuvent-elles penser ? », trop floue, et la remplace par un jeu : un interrogateur converse par écrit avec deux interlocuteurs cachés et doit deviner lequel est la machine. C’est le test de Turing. Turing parie qu’« in about fifty years’ time », des machines dotées d’une mémoire d’environ $10^9$ joueront si bien qu’un interrogateur moyen n’aura pas plus de 70 % de chances de les démasquer après cinq minutes de questions.

La même année, Claude Shannon publie « Programming a computer for playing chess ». Il fait le calcul qui va hanter le domaine : environ $10^3$ possibilités par paire de coups (un coup blanc, un coup noir), une partie typique de 40 paires de coups, donc de l’ordre de $10^{120}$ variantes depuis la position initiale. À une variante par microseconde, il faudrait plus de $10^{90}$ années pour jouer le premier coup. Sa réponse : n’explorer que quelques coups d’avance et noter les positions atteintes avec une fonction d’évaluation. C’est exactement ce que fera Deep Blue, 47 ans plus tard.

Le 31 août 1955, John McCarthy (Dartmouth College), Marvin Minsky (Harvard), Nathaniel Rochester (IBM) et Shannon (Bell Labs) proposent « a 2 month, 10 man study of artificial intelligence » pour l’été 1956 à Dartmouth. L’expression « artificial intelligence » figure dans le titre. La conjecture de départ est ambitieuse : « every aspect of learning or any other feature of intelligence can in principle be so precisely described that a machine can be made to simulate it ». Le programme le plus remarqué de cette période vient pourtant d’ailleurs : le Logic Theorist de Newell, Shaw et Simon (RAND Corporation et Carnegie Tech, 1956), qui démontre 38 des 52 premiers théorèmes du chapitre 2 des *Principia Mathematica* (chiffres d’après des sources secondaires) en cherchant parmi les déductions possibles, guidé par des règles empiriques. Le mot clé est déjà là : heuristique.

## L’idée en une image

Un GPS qui calcule un itinéraire, un solveur de taquin, un programme d’échecs : trois problèmes qui se ramènent au même schéma. On a des situations (une position sur la carte, une disposition des tuiles, une position de l’échiquier), des actions qui font passer de l’une à l’autre (prendre une rue, glisser une tuile, jouer un coup), une situation de départ et une ou plusieurs situations « but ». L’ensemble forme un graphe, l’espace d’états. Résoudre le problème, c’est trouver un chemin dans ce graphe.

Le graphe est presque toujours trop grand pour être dessiné : on le découvre au fur et à mesure. L’algorithme tient une liste d’états découverts mais pas encore examinés, la frontière. À chaque tour, il en retire un, regarde si c’est le but, sinon le développe : il calcule ses voisins et ajoute les nouveaux à la frontière. Toute la différence entre les méthodes de ce chapitre tient à une seule question : quel état sortir de la frontière en premier ?

- Le plus ancien : on explore en cercles concentriques (recherche en largeur).
- Le plus récent : on fonce dans une direction et on revient en arrière quand on est bloqué (recherche en profondeur).
- Le moins cher à atteindre : on avance par « courbes de niveau » de coût (recherche à coût uniforme).
- Celui qui semble le plus proche du but : on se laisse guider par une estimation (recherche gloutonne).
- Celui dont le coût total estimé (déjà payé + restant) est le plus faible : c’est A*.

Pour les jeux, il y a un adversaire : on ne cherche plus un chemin mais un coup, en supposant que l’autre joue au mieux contre nous. C’est minimax, et son accélérateur, l’élagage alpha-bêta.

| | Avant : énumération aveugle | Après : recherche heuristique et élagage |
|---|---|---|
| Ce qu’on fait | examiner les états dans un ordre fixe, sans savoir où est le but | estimer la distance au but (A*) ou borner la valeur d’une branche (alpha-bêta) |
| Taquin, 10 positions (exercice 04) | 87 572 états développés en moyenne | 1 131 avec l’heuristique de Manhattan, même solution optimale |
| Grille ouverte 15 × 25 (exercice 04) | 297 états développés (coût uniforme) | 22 (A*), pour un chemin de 21 pas |
| Morpion, arbre complet | 549 946 nœuds (minimax) | 7 275 (alpha-bêta, bons coups d’abord), même valeur |
| Garantie | solution optimale avec le coût uniforme, si on a le temps | la même, tant que l’estimation ne surestime jamais (A*) ; alpha-bêta rend toujours la valeur exacte |
| Ce qui coince encore | la taille de l’espace explose avec la profondeur | il faut inventer l’heuristique ou la fonction d’évaluation à la main |

## Le mécanisme

### Formuler un problème de recherche

Un problème de recherche se décrit par cinq éléments :

1. un état initial (la position de départ) ;
2. pour chaque état, les actions possibles et l’état qu’elles produisent (les successeurs) ;
3. un coût positif par action : 1 par case sur une grille, 5 dans la boue ;
4. un test de but ;
5. et c’est tout : aucune donnée, aucun paramètre à régler.

La difficulté tient à la taille. Si chaque état a en moyenne $b$ successeurs (le facteur de branchement) et que la solution est à $d$ actions du départ, l’arbre de recherche compte de l’ordre de $b^d$ nœuds. Quelques ordres de grandeur :

| Problème | Facteur de branchement | Taille |
|---|---|---|
| Grille 15 × 25 de l’exercice | au plus 4 | 352 cases franchissables |
| Taquin à 8 cases (3 × 3) | 2 à 4 | $9!/2 = 181\,440$ positions atteignables |
| Morpion | 9, puis 8, 7… | 549 946 nœuds dans l’arbre de jeu complet |
| Échecs (estimation de Shannon) | environ 30 coups par joueur | environ $10^{120}$ variantes |

### Ce que la méthode optimise

Aucune fonction de perte : rien n’est appris, il n’y a ni données ni paramètres. Ce qu’on minimise, c’est un coût de chemin, la somme des coûts des actions :

Dans les jeux, on calcule une valeur (gagné, nul, perdu) ; dans l’encadré final, on minimise un nombre de conflits. Une fonction de perte, au sens du chapitre 01, mesure l’écart entre les prédictions d’un modèle et des données : il n’y a ici ni modèle ni données. La différence compte pour la suite du parcours : à partir du chapitre 06, c’est la perte qui guidera la machine, plus le programmeur.

### Un seul algorithme, cinq stratégies

Voici le squelette commun, en Python (version simplifiée de la fonction `rechercher` de l’exercice) :

```python
import heapq

def rechercher(depart, est_but, voisins, priorite):
    """Recherche générique. `priorite(g, etat)` décide de l'ordre de sortie de la frontière."""
    g = {depart: 0}              # meilleur coût connu depuis le départ
    parent = {depart: None}
    frontiere = [(priorite(0, depart), depart)]   # file de priorité (tas)
    developpes = set()
    while frontiere:
        _, etat = heapq.heappop(frontiere)        # l'état le plus prioritaire
        if etat in developpes:
            continue                              # déjà traité par un meilleur chemin
        if est_but(etat):
            return reconstruire(parent, etat), g[etat]   # remonte les parents jusqu'au départ
        developpes.add(etat)
        for suivant, cout in voisins(etat):
            ng = g[etat] + cout
            if suivant not in g or ng < g[suivant]:    # chemin nouveau ou meilleur
                g[suivant], parent[suivant] = ng, etat
                heapq.heappush(frontiere, (priorite(ng, suivant), suivant))
    return None, float("inf")    # aucun chemin
```

Seule la fonction `priorite` change :

| Stratégie | Priorité (on sort la plus petite) | Optimale ? |
|---|---|---|
| recherche à coût uniforme | $g(n)$, le coût déjà payé | oui |
| recherche gloutonne | $h(n)$, le coût restant estimé | non |
| A* | $g(n) + h(n)$ | oui si $h$ est admissible |
| A* pondéré | $g(n) + w\,h(n)$, avec $w$ supérieur à 1 | non, mais coût au plus $w$ fois l’optimum si $h$ est admissible |

La recherche en largeur et la recherche en profondeur se codent plus simplement avec une file (premier entré, premier sorti) ou une pile (dernier entré, premier sorti) ; dans l’exercice, elles ont leur propre boucle, mais le principe est le même.

### Recherche aveugle : largeur, profondeur, coût uniforme

On appelle ces trois stratégies aveugles : elles n’ont aucune idée de la direction du but.

Largeur. La frontière est une file : on développe tous les états à 1 action du départ, puis tous ceux à 2 actions, etc. Elle trouve le chemin qui a le moins d’actions, ce qui n’est le moins coûteux que si toutes les actions coûtent pareil. Son défaut : la mémoire. Elle garde toute la couche courante, soit de l’ordre de $b^d$ états.

Profondeur. La frontière est une pile : on suit toujours le dernier voisin découvert, et on revient en arrière en cas d’impasse. Dans sa version sans mémoire des états visités, elle ne garde qu’un chemin et les frères de ses nœuds : mémoire minimale, mais aucune garantie : sur la grille de l’exercice, elle trouve un chemin de 51 pas et de coût 59, quand l’optimum est 35.

Coût uniforme. La frontière est une file de priorité sur $g(n)$, le coût depuis le départ. C’est l’algorithme publié par Dijkstra en 1959 pour le plus court chemin dans un graphe, arrêté dès que le but sort de la frontière. Pourquoi est-ce optimal ? Quand un état sort de la file, tous les états restants ont un $g$ supérieur ou égal ; comme les coûts sont positifs, aucun détour par eux ne peut revenir moins cher.

Exemple à la main. Un petit graphe, départ S, but G :

Le meilleur chemin est S → A → B → G, de coût $1 + 2 + 1 = 4$. Voici ce que rendent les stratégies (calcul refait par script) :

| Stratégie | Ordre de développement | Chemin rendu | Coût |
|---|---|---|---|
| Largeur | S, A, B, G | S → A → G (2 actions) | 7 |
| Coût uniforme | S (g = 0), A (1), B (3), G (4) | S → A → B → G | 4 |
| Glouton (h seul) | S (h = 3), B (1), G (0) | S → B → G | 5 |
| A* (g + h) | S (f = 3), A (3), B (4), G (4) | S → A → B → G | 4 |

La largeur découvre G dès qu’elle développe A, par l’arête de coût 6, et ne revient jamais dessus : elle compte les actions, pas les coûts. Le glouton file vers B, qui « semble » à 1 du but, sans voir que l’arête S–B coûte 4. Sur un graphe aussi petit, A* ne gagne rien sur le coût uniforme ; le gain apparaît quand l’espace est grand et l’heuristique informative.

### Heuristiques et A*

Une heuristique $h(n)$ est une estimation du coût restant entre $n$ et le but, rapide à calculer. Sur une grille où l’on se déplace en haut, en bas, à gauche ou à droite, la distance de Manhattan $|x_n - x_B| + |y_n - y_B|$ (B désigne le but) compte les cases à parcourir s’il n’y avait aucun obstacle.

A* trie la frontière par

Avec $h = 0$, on retrouve le coût uniforme ; sans $g$, on retrouve le glouton. A* fait la synthèse : Nilsson voulait au départ guider le robot Shakey avec la seule heuristique (méthode du « Graph Traverser »), Raphael a proposé d’ajouter $g$, et Hart a dégagé les conditions qui garantissent l’optimalité (répartition des rôles d’après des sources secondaires).

La condition clé : ne jamais surestimer. Une heuristique est une heuristique admissible si

La distance de Manhattan est admissible sur la grille : un mur ou de la boue ne peuvent que rallonger le trajet, jamais le raccourcir.

Pourquoi A* est alors optimal (esquisse). Soit $C^*$ le coût du meilleur chemin. Tant que le but n’est pas sorti, un état $n$ de ce chemin optimal attend dans la frontière, avec $f(n) = g(n) + h(n) \le g(n) + h^*(n) = C^*$. Si A* sortait le but par un chemin trop cher, de coût $C$ supérieur à $C^*$, ce but aurait $f = C + 0$, plus grand que $f(n)$ : $n$ serait sorti avant lui. En déroulant l’argument, tout le chemin optimal sort avant tout mauvais chemin. Conséquence pratique : A* ne développe jamais un état dont $f$ dépasse $C^*$. Dans l’exercice, sur la carte en U, aucun état développé n’a $g + h$ au-dessus de 35.

En simplifiant : l’argument suppose aussi qu’un état déjà développé n’a pas besoin d’être rouvert. C’est garanti quand l’heuristique est une heuristique cohérente : $h(n) \le c(n, n') + h(n')$ pour chaque voisin $n'$ (l’estimation ne baisse pas plus vite qu’on n’avance). Manhattan est cohérente sur la grille. Avec une heuristique admissible mais pas cohérente, il faut autoriser A* à rouvrir des états pour garder l’optimalité.

Ce que montre la grille de l’exercice. Une carte 15 × 25 avec un piège en U ouvert vers le départ et une bande de boue (coût 5 par case) devant le but, qu’on ne contourne que par la dernière ligne :

| Stratégie | États développés | Pas | Coût |
|---|---|---|---|
| Largeur | 350 | 33 | 41 (traverse la boue) |
| Profondeur | 52 | 51 | 59 |
| Coût uniforme | 327 | 35 | 35 (optimal) |
| Glouton | 123 | 33 | 41 |
| A* | 244 | 35 | 35 (optimal) |
| A* pondéré (w = 2) | 163 | 33 | 41 |

Sur la carte sans obstacle, A* et le glouton développent 22 états, exactement les cases du chemin (21 pas plus le départ), contre 297 pour la largeur et le coût uniforme. Sur la carte en U, Manhattan aide beaucoup moins (244 contre 327) : elle ignore le mur et la boue, et A* doit explorer tout ce qui paraît prometteur quand on ignore les obstacles. Une heuristique vaut ce qu’elle sait du problème.

Rejouez ces stratégies pas à pas, déplacez le départ et le but, dessinez des murs ou de la boue. Le curseur de poids $w$ montre ce que l’on perd quand l’heuristique compte plus que le coût déjà payé ; survolez une case pour lire ses valeurs $g$, $h$ et $f$.

Le taquin : comparer deux heuristiques. Le taquin à 8 cases se joue sur une grille 3 × 3 : on fait glisser les tuiles dans la case vide jusqu’à l’ordre 1 à 8. Deux heuristiques admissibles classiques :

- $h_1$, le nombre de tuiles mal placées : chaque tuile mal placée doit bouger au moins une fois ;
- $h_2$, la somme des distances de Manhattan de chaque tuile à sa place : chaque coup ne déplace qu’une tuile d’une case.

Exemple, avec le but `1 2 3 / 4 5 6 / 7 8 _` :

```text
7 2 4      tuiles mal placées : 7, 4, 5, 8, 3, 1          → h1 = 6
5 _ 6      distances : 7→2, 2→0, 4→3, 5→1, 6→0, 8→1,
8 3 1                  3→3, 1→4                           → h2 = 14
```

La solution optimale de cette position fait 20 coups (calculé par A*) : les deux estimations sont bien inférieures ou égales à 20. Comme $h_2(n) \ge h_1(n)$ pour tout état, on dit que $h_2$ domine $h_1$ : elle est plus proche de la vérité, et A* développe moins d’états. Sur cette position, A* développe 57 593 états avec $h = 0$, 3 667 avec $h_1$ et 283 avec $h_2$. Sur les dix positions de l’exercice (solutions de 18 à 26 coups), la moyenne passe de 87 572 états ($h = 0$) à 11 924 ($h_1$) puis 1 131 ($h_2$), avec exactement les mêmes longueurs. Une heuristique qui surestime, comme $2h_2$, va plus vite (516 états) mais rend des solutions trop longues dans 6 cas sur 10, de 2 à 4 coups.

### Jeux à deux joueurs : minimax

Au morpion ou aux échecs, on ne choisit pas toute la suite des actions : un adversaire joue un coup sur deux. On note MAX le joueur qui cherche à maximiser le score (+1 s’il gagne, 0 pour un nul, −1 s’il perd) et MIN son adversaire. La valeur minimax d’une position se définit récursivement :

MAX joue le coup qui mène à la position de plus grande valeur, en supposant que MIN répondra toujours par le pire coup pour MAX. Sur l’arbre ci-dessous, MAX a trois coups ; MIN répond par l’une de trois réponses, dont les valeurs sont `[[3, 12, 8], [2, 4, 6], [14, 5, 2]]`. Les nœuds MIN valent $\min(3, 12, 8) = 3$, $\min(2, 4, 6) = 2$ et $\min(14, 5, 2) = 2$ ; la racine vaut $\max(3, 2, 2) = 3$. MAX joue le premier coup.

Minimax examine tout l’arbre : 549 946 nœuds pour le morpion depuis le plateau vide, pour conclure que la valeur est 0 (un nul avec un jeu parfait des deux côtés). Contre un joueur qui joue au hasard, ce programme ne perd jamais : 188 victoires et 12 nuls sur 200 parties dans l’exercice.

### L’élagage alpha-bêta

Revenons à l’arbre. Après la première branche, MAX est sûr d’obtenir 3. Dans la deuxième branche, la première feuille vaut 2 : MIN pourra donc imposer au plus 2 dans cette branche. Quelles que soient les deux feuilles suivantes (4 et 6), MAX ne choisira jamais cette branche : inutile de les regarder. C’est l’élagage alpha-bêta. L’algorithme transporte deux bornes :

- $\alpha$, la meilleure valeur que MAX est déjà sûr d’obtenir sur le chemin courant ;
- $\beta$, la meilleure valeur (la plus basse) que MIN est déjà sûr d’imposer.

Dès que $\alpha \ge \beta$, la position courante ne sera jamais atteinte avec un jeu correct des deux côtés : on arrête d’examiner ses autres coups.

```python
def alphabeta(etat, alpha, beta, max_joue):
    if fin_de_partie(etat):
        return score(etat)
    v = float("-inf") if max_joue else float("inf")
    for coup in coups(etat):
        x = alphabeta(jouer(etat, coup), alpha, beta, not max_joue)
        if max_joue:
            v, alpha = max(v, x), max(alpha, x)
        else:
            v, beta = min(v, x), min(beta, x)
        if alpha >= beta:
            break          # élagage : les coups restants ne changeront pas la décision
    return v

# appel initial : alphabeta(position, -inf, +inf, True)
```

Trace sur l’arbre : branche 1, MIN voit 3, 12, 8 et vaut 3, donc $\alpha = 3$ à la racine. Branche 2, MIN voit 2 : $\beta = 2 \le \alpha = 3$, coupure, les feuilles 4 et 6 sont sautées. Branche 3, MIN voit 14 ($\beta = 14$), 5 ($\beta = 5$), puis 2 : il faut tout regarder, car les deux premières feuilles ne descendent pas sous 3. Résultat : valeur 3, 7 feuilles évaluées sur 9.

Alpha-bêta rend toujours la même valeur que minimax : elle ne fait qu’éviter du travail inutile. Le gain dépend de l’ordre des coups. Dans le pire cas (les mauvais coups d’abord), on n’élague rien. Dans le meilleur cas (le meilleur coup toujours essayé en premier), sur un arbre de facteur $b$ et de profondeur $d$, on n’évalue que $b^{\lceil d/2 \rceil} + b^{\lfloor d/2 \rfloor} - 1$ feuilles au lieu de $b^d$ (⌈ ⌉ et ⌊ ⌋ désignent l’arrondi supérieur et l’arrondi inférieur ; résultat classique, que l’on retrouve dans l’analyse de Knuth et Moore). Avec $b = 30$ et $d = 8$, cela fait environ 1,6 million de feuilles au lieu de $6{,}6 \times 10^{11}$ : à temps égal, on peut chercher deux fois plus profond.

Au morpion, l’exercice mesure : 549 946 nœuds pour minimax, 18 297 pour alpha-bêta (3,3 %), et 7 275 (1,3 %) en essayant d’abord le centre, puis les coins, puis les bords. Même valeur, 0, dans les trois cas.

L’idée est née plusieurs fois dans les années 1950 et 1960 : McCarthy en parle dès 1956, Newell, Shaw et Simon en utilisent une approximation en 1958, Brudno la décrit en 1963 (historique d’après chessprogramming.org, source secondaire). L’article de Knuth et Moore (1975) en est l’analyse de référence.

### Deep Blue : chercher vite et évaluer à la main

Aux échecs, même avec alpha-bêta, on n’atteint pas la fin de la partie. On fait comme Shannon l’avait proposé : chercher jusqu’à une profondeur limitée, puis remplacer la valeur exacte des feuilles par une fonction d’évaluation, écrite par des humains (matériel, sécurité du roi, mobilité…). Deep Blue, d’IBM Research, combinait cette recherche alpha-bêta avec du matériel spécialisé pour examiner de 100 à 200 millions de positions par seconde selon la position (126 millions en moyenne pendant le match de 1997, d’après Campbell, Hoane et Hsu). Après avoir perdu 4-2 contre Garry Kasparov en février 1996, il gagne la revanche 3½-2½ en mai 1997 à New York.

Deep Blue n’apprend rien de ses parties : l’essentiel de sa « connaissance » des échecs est dans une fonction d’évaluation réglée en grande partie à la main et dans une bibliothèque d’ouvertures préparée par des grands maîtres. Seuls quelques poids de l’évaluation ont été ajustés par un outil automatique avant 1997, et une « bibliothèque étendue » résumait automatiquement une base de parties de grands maîtres. Pour le go, où le facteur de branchement est bien plus grand et où l’on ne sait pas écrire une bonne fonction d’évaluation, il faudra deux idées de plus : la recherche arborescente Monte-Carlo, qui estime une position en jouant des parties au hasard (chapitre 11), puis des réseaux de neurones qui apprennent l’évaluation (chapitre 17).

### Encadré : optimiser sans gradient

Recherche locale : quand seul l’état final compte :
Dans certains problèmes, le chemin n’a aucune importance : on veut seulement un bon état. Exemple de l’exercice : placer 8 reines sur un échiquier sans qu’aucune paire ne se menace. Un état place une reine par colonne ($8^8 \approx 16{,}8$ millions d’états), et on minimise le nombre de paires en conflit. La recherche locale ne garde qu’un état (ou une population) et l’améliore par petites modifications.

- escalade : passer au meilleur voisin tant qu’il améliore. Rapide, mais bloquée au premier minimum local : 16 % de solutions sur 200 départs aléatoires.
- recuit simulé (Kirkpatrick, Gelatt et Vecchi, 1983) : tirer un voisin au hasard ; l’accepter s’il améliore, sinon avec la probabilité $e^{-\Delta/T}$, où $\Delta$ est la dégradation et $T$ une « température » qui baisse peu à peu. Avec $T = 2$ et $\Delta = 1$, un coup qui ajoute un conflit passe 61 % du temps ($e^{-0{,}5} \approx 0{,}61$) ; avec $T = 0{,}1$, presque jamais ($e^{-10} \approx 4{,}5 \times 10^{-5}$). Résultat : 98 % de solutions.
- algorithme génétique (Holland, 1975) : une population de 200 états ; on garde la meilleure moitié, on croise deux parents (début de l’un, fin de l’autre), on mute au hasard. 95 % de solutions (19 lancements sur 20), en 21 générations en moyenne.

C’est de l’optimisation, comme au chapitre 02, mais sans gradient : on ne sait que comparer deux états. Quand l’objectif est dérivable, la descente de gradient est bien plus efficace ; ces méthodes restent utiles pour les problèmes combinatoires et, plus tard, pour chercher des hyperparamètres ou des architectures.

## Ce qui coinçait

- L’explosion combinatoire. Toutes ces méthodes restent exponentielles dans le pire cas. Une bonne heuristique repousse le mur, elle ne le supprime pas. Les échecs n’ont été maîtrisés qu’avec du matériel dédié et une recherche tronquée.
- L’heuristique et l’évaluation s’écrivent à la main. Manhattan pour le taquin, le matériel pour les échecs : chaque problème demande un expert qui code sa connaissance. Ces programmes ne s’améliorent pas en jouant (le programme de dames de Samuel, dès 1959, faisait exception : chapitre 11). C’est précisément ce que l’apprentissage va changer : estimer une valeur à partir de l’expérience (renforcement, chapitre 11), puis avec des réseaux (chapitre 17).
- Il faut un monde parfaitement décrit. États, actions, coûts et but doivent être donnés exactement. Une image, une phrase, une situation incertaine ne se laissent pas réduire à un graphe propre.
- Chercher n’est pas savoir. Un solveur de taquin ne sait rien du monde. Pour qu’un programme raisonne sur un domaine (médecine, planification d’actions), il faut d’abord représenter des connaissances : règles, faits, contraintes. C’est l’objet du chapitre suivant.

La recherche n’a pas disparu pour autant : des variantes d’A* et de Dijkstra tournent dans les GPS et les jeux vidéo, alpha-bêta et Monte-Carlo dans les programmes de jeu, et l’idée de dépenser du calcul à l’inférence pour chercher une meilleure réponse (best-of-N, vérificateurs) reviendra avec les modèles de langage au chapitre 31.

- Recherche en largeur, en profondeur : Explorer sans boussole, en cercles ou en ligne droite.
- Coût uniforme (Dijkstra, 1959) : Toujours prolonger le chemin le moins cher payé jusqu’ici.
- Glouton : Toujours aller vers ce qui semble le plus proche du but, quitte à payer cher.
- A* (Hart, Nilsson et Raphael, 1968) : Coût payé + estimation optimiste du reste : optimal et bien plus rapide.
- Minimax appliqué aux échecs (Shannon, 1950) : Supposer que l’adversaire joue au mieux contre nous.
- Alpha-bêta (analyse : Knuth et Moore, 1975) : Abandonner une branche dès qu’elle ne peut plus changer la décision.
- Recuit simulé (1983), algorithmes génétiques (1975) : Optimiser sans gradient, en acceptant parfois de reculer.
- Logique, règles, contraintes : Ne plus seulement chercher, mais représenter ce que l’on sait.

Et ensuite ? Chercher ne suffit pas : encore faut-il décrire le problème, et la plupart des problèmes réels se décrivent par des connaissances (« si la fièvre persiste et que… alors… ») plutôt que par un graphe d’états. Le chapitre 05 passe à la logique, aux contraintes, à la planification et aux systèmes experts : la seconde moitié de l’IA symbolique, et sa crise.

## Exercice

En Python pur, sans GPU ni apprentissage, vous écrivez une recherche générique et vous comparez six stratégies sur la même grille, trois heuristiques sur le taquin, minimax et alpha-bêta au morpion, puis trois méthodes de recherche locale sur les 8 reines. Le run complet dure une dizaine de secondes.

Ce qu’il faut observer :

- le code est le même pour toutes les stratégies de la grille : seul change l’ordre de sortie de la frontière. La largeur minimise le nombre de pas (33), pas le coût (41) ;
- passer de « tuiles mal placées » à Manhattan divise l’effort par 10 sur le taquin, sans changer une seule longueur de solution ;
- alpha-bêta rend la même valeur que minimax avec 30 fois moins de nœuds, 75 fois moins quand les bons coups sont essayés d’abord.

Questions de réflexion : pourquoi Manhattan aide-t-elle si peu sur la carte en U, et quelle heuristique admissible ferait mieux ? Que devient le recuit simulé si la température reste nulle ? Et si elle ne baisse jamais ?

## Auto-évaluation

Question : Sur la grille de l’exercice, la recherche en largeur trouve un chemin de 33 pas et de coût 41, le coût uniforme un chemin de 35 pas et de coût 35. Pourquoi la largeur se trompe-t-elle ?

Réponse : La largeur développe les états par nombre d’actions depuis le départ : elle trouve le chemin qui a le moins de pas. Elle ignore que les cases de boue coûtent 5. Le coût uniforme trie par $g(n)$, le coût payé : il préfère faire deux pas de plus pour contourner la boue par la dernière ligne. Les deux coïncident seulement quand toutes les actions ont le même coût.

Question : Pourquoi la distance de Manhattan est-elle une heuristique admissible sur la grille, et que perd-on avec une heuristique qui surestime, comme 2 × Manhattan ?

Réponse : Manhattan compte les cases à parcourir sans aucun obstacle, avec un coût de 1 par case : murs et boue ne peuvent que rallonger le vrai trajet, donc $h(n) \le h^*(n)$. Avec une heuristique qui surestime, A* peut sortir le but par un chemin trop cher avant d’avoir examiné le bon : il va plus vite mais perd la garantie d’optimalité. Sur le taquin, $2h_2$ développe 516 états au lieu de 1 131, mais 6 solutions sur 10 sont plus longues que l’optimum.

Question : Sur la position 7 2 4 / 5 _ 6 / 8 3 1, h1 = 6 et h2 = 14 ; la solution optimale fait 20 coups. Laquelle des deux heuristiques préférer, et pourquoi ?

Réponse : $h_2$. Les deux sont admissibles (6 et 14 sont inférieurs ou égaux à 20), mais $h_2 \ge h_1$ partout : elle domine $h_1$, elle est plus proche du vrai coût, donc A* écarte plus tôt les états inutiles. Sur cette position : 283 états développés avec $h_2$, 3 667 avec $h_1$, pour la même solution de 20 coups.

Question : Sur l’arbre [[3, 12, 8], [2, 4, 6], [14, 5, 2]] (MAX à la racine, MIN en dessous), pourquoi alpha-bêta n’évalue-t-il pas les feuilles 4 et 6, et pourquoi doit-il évaluer les trois feuilles de la dernière branche ?

Réponse : Après la première branche, MAX est sûr d’avoir 3 ($\alpha = 3$). Dans la deuxième, la feuille 2 garantit que MIN imposera au plus 2 ($\beta = 2 \le \alpha$) : MAX ne choisira jamais cette branche, quelles que soient les autres feuilles. Dans la troisième, 14 puis 5 laissent $\beta$ au-dessus de 3 : la branche pourrait encore valoir plus que 3, il faut donc regarder la feuille suivante (2). Valeur 3, 7 feuilles sur 9.

Question : Deep Blue a battu Kasparov en 1997. Qu’est-ce qui, dans ce programme, venait de l’apprentissage ?

Réponse : Presque rien : Deep Blue cherchait par alpha-bêta jusqu’à une profondeur limitée, avec du matériel dédié (126 millions de positions par seconde en moyenne en 1997), et notait les positions avec une fonction d’évaluation réglée en grande partie à la main et une bibliothèque d’ouvertures préparée par des grands maîtres. Seuls quelques poids de l’évaluation avaient été ajustés automatiquement, et sa « bibliothèque étendue » résumait une base de parties de grands maîtres : rien n’était appris de ses propres parties. C’est le schéma proposé par Shannon en 1950. L’apprentissage de la fonction d’évaluation viendra avec le renforcement (chapitre 11, chapitre 17).

## Pour aller plus loin

- Articles originaux (fiches ci-dessus) : Turing, « Computing Machinery and Intelligence » (1950) ; Shannon, « Programming a computer for playing chess » (1950) ; McCarthy, Minsky, Rochester et Shannon, proposition de Dartmouth (1955, reproduite en 2006) ; Hart, Nilsson et Raphael, « A Formal Basis for the Heuristic Determination of Minimum Cost Paths » (1968) ; Knuth et Moore, « An analysis of alpha-beta pruning » (1975) ; Campbell, Hoane et Hsu, « Deep Blue » (2002).
- Le manuel de référence : S. Russell et P. Norvig, *Artificial Intelligence: A Modern Approach*, 4e édition. Les chapitres sur la recherche, les jeux et la recherche locale couvrent ce chapitre en détail.
- L’histoire racontée par un acteur : N. J. Nilsson, *The Quest for Artificial Intelligence* (Cambridge University Press, 2009 ; version web gratuite de l’auteur), coauteur d’A*.
- Pour l’intuition visuelle : Red Blob Games, « Introduction to the A* Algorithm », explications interactives de la largeur, de Dijkstra, du glouton et d’A* sur des grilles ; Computerphile, « A* (A Star) Search Algorithm » et « Dijkstra’s Algorithm » (vidéos) ; Sebastian Lague, « Algorithms Explained – minimax and alpha-beta pruning » (vidéo).
- Cours : Harvard, CS50’s Introduction to Artificial Intelligence with Python (recherche et jeux en Python) ; MIT OpenCourseWare, 6.034 Artificial Intelligence (P. Winston, 2010) ; Berkeley, CS188 Introduction to Artificial Intelligence.
- Alpha-bêta en pratique : la page « Alpha-Beta » du wiki chessprogramming.org (historique, variantes, ordre des coups).
