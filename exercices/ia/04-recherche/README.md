# Exercice 04 : chercher dans un espace d'états (A*, minimax, alpha-bêta)

| | |
|---|---|
| **Chapitre lié** | [04. Chercher : espaces d'états, heuristiques, jeux](https://laclasse.poneyponey.fr/ia/p01-symbolique/04-recherche/) |
| **Exécution** | locale, CPU, Python pur (aucun GPU, aucun apprentissage) |
| **Commande** | `make ex NUM=04` (éditeur), `MODE=run` (complet), `MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 10 s au total (dont 2,2 s pour A* sans heuristique sur le taquin et 3,2 s pour 200 parties de morpion) ; smoke : 5 s (M1 Pro, CPU, 14/10/2026) |
| **Travail estimé** | ~1 h |
| **Seed** | 42 (`seed_everything(42)` ; `random.Random(42)` pour les taquins et le morpion, 43 et 44 pour le recuit et l'algorithme génétique) |
| **Données** | aucune : cartes écrites dans le notebook, positions tirées au hasard |

## Objectif

Résoudre des problèmes **sans rien apprendre**, en explorant un espace d'états : comparer les stratégies de recherche aveugle et informée (BFS, DFS, coût uniforme, glouton, A*), mesurer l'effet d'une heuristique, puis jouer au morpion par minimax et constater ce que l'élagage alpha-bêta économise.

Fonction de perte : **aucune**. On minimise un coût de chemin (parties 1 et 2), on calcule une valeur de jeu (partie 3), on minimise un nombre de conflits (partie 4).

## Ce que vous allez faire

1. **Grille (15 × 25)** avec un piège en U et une bande de boue (coût 5) : lancer les six stratégies (largeur, profondeur, coût uniforme, glouton, A*, A* pondéré `g + 2h`) sur la même carte, puis sur une carte sans obstacle. Toutes partagent le même code (`rechercher`) : seul change l'ordre de sortie de la frontière.
2. **Taquin à 8 cases** : 10 positions aléatoires solubles, A* avec `h0 = 0`, `h1` (tuiles mal placées), `h2` (Manhattan), puis `2·h2` (non admissible).
3. **Morpion** : un petit arbre à la main (`[[3, 12, 8], [2, 4, 6], [14, 5, 2]]`), puis l'arbre complet depuis le plateau vide par minimax, alpha-bêta, et alpha-bêta avec un bon ordre des coups (centre, coins, bords). Enfin 200 parties de l'IA contre un joueur aléatoire.
4. **Encadré « optimiser sans gradient »** : 8 reines par escalade, recuit simulé et algorithme génétique.

## Résultat attendu

Valeurs obtenues au run complet (seed 42) ; le notebook s'arrête avec une erreur si l'une des 13 vérifications échoue.

**Grille, carte en U** (352 cases franchissables) :

| Stratégie | États développés | Pas | Coût |
|---|---|---|---|
| Largeur (BFS) | 350 | 33 | 41 (traverse la boue) |
| Profondeur (DFS) | 52 | 51 | 59 |
| Coût uniforme | 327 | 35 | **35** (optimal) |
| Glouton | 123 | 33 | 41 |
| A* | 244 | 35 | **35** (optimal) |
| A* pondéré (w = 2) | 163 | 33 | 41 |

Sur la **carte ouverte** : largeur et coût uniforme développent 297 états, glouton et A* seulement **22** pour un chemin de 21 pas.

**Taquin** (moyenne sur 10 positions, longueurs optimales de 18 à 26 coups) : `h0` 87 572 états développés, `h1` 11 924, `h2` **1 131**, mêmes longueurs ; `2·h2` 516 états mais 6 solutions sur 10 plus longues que l'optimum (de +2 à +4 coups).

**Morpion** : valeur 0 (nul) dans les trois cas ; minimax **549 946** nœuds, alpha-bêta **18 297** (3,3 %), alpha-bêta bien ordonné **7 275** (1,3 %). Arbre à la main : valeur 3, 7 feuilles évaluées sur 9 (4 et 6 élaguées). Contre le hasard : 188 victoires, 12 nuls, **0 défaite** sur 200 parties.

**8 reines** (200 départs) : escalade **16 %** de solutions, recuit simulé **98 %** ; algorithme génétique 95 % (19 lancements sur 20, 21 générations en moyenne).

Seuils vérifiés : A* au coût optimal et avec moins d'états que le coût uniforme ; BFS et glouton plus chers que l'optimum ; A* ≤ 2 × le nombre de pas sur la carte ouverte ; `h2` ≥ 10 × moins d'états que `h1`, `h1` ≥ 5 × moins que `h0`, longueurs identiques ; `2·h2` jamais plus court que l'optimum ; alpha-bêta ≤ 5 % des nœuds de minimax, même valeur ; aucune défaite contre le hasard ; recuit ≥ 2 × escalade.

Figures produites : `figures/grille.png` (six stratégies sur la carte en U) et `figures/synthese.png` (taquin, morpion, 8 reines).

## Ce qu'il faut observer

- **Le même squelette, six comportements.** BFS s'étend en losange et minimise le nombre de pas, pas le coût : il traverse la boue. Le coût uniforme s'étend en « courbes de niveau » de coût et contourne la boue par la dernière ligne. A* ne développe jamais un état dont `g + h` dépasse le coût optimal (35) : les coins éloignés du départ et toute la zone derrière la boue restent intacts.
- **L'heuristique fait l'essentiel du travail.** Sur la carte ouverte, A* passe de 297 à 22 états. Sur le taquin, passer de « tuiles mal placées » à « Manhattan » divise l'effort par 10, sans changer une seule longueur de solution : Manhattan domine l'autre heuristique.
- **Admissible ou rapide.** Le glouton et A* pondéré vont plus vite mais rendent un chemin plus cher : une heuristique qui surestime (ou qu'on suit seule) perd la garantie d'optimalité.
- **Alpha-bêta ne change pas la réponse, seulement le travail.** Même valeur (0), 30 fois moins de nœuds, 75 fois moins avec un bon ordre des coups. Sur l'arbre à la main, voir pourquoi 4 et 6 ne sont jamais regardés : après la feuille 2, MIN peut déjà imposer au plus 2, et MAX a déjà 3 ailleurs.
- **La recherche locale ne garde pas de chemin.** L'escalade reste coincée dans un minimum local 84 fois sur 100 ; accepter parfois un coup moins bon (recuit) suffit à s'en sortir.

## Questions de réflexion

1. Sur la carte en U, A* développe encore 244 états sur 352. Pourquoi l'heuristique de Manhattan aide-t-elle si peu ici, alors qu'elle fait merveille sur la carte ouverte ? Quelle heuristique (toujours admissible) aiderait davantage ?
2. Si l'on multiplie le coût de la boue par 10, quelles stratégies changent de chemin ? Lesquelles changent seulement de nombre d'états développés ? (Modifiez `cout_case` dans le notebook : la démo du chapitre garde un coût fixe de 5.)
3. Le morpion a 549 946 nœuds dans son arbre mais seulement quelques milliers de positions distinctes. Comment exploiter ce fait pour aller encore plus vite ? Pourquoi Deep Blue avait-il besoin, en plus, d'une **fonction d'évaluation** ?
4. Le recuit simulé avec une température toujours nulle, que devient-il ? Et avec une température qui ne baisse jamais ?

## Pour aller plus loin

- Remplacer l'ordre « centre, coins, bords » par un ordre aléatoire et mesurer le nombre de nœuds d'alpha-bêta : dans le meilleur cas, sur un arbre uniforme, alpha-bêta n'examine qu'environ la racine carrée du nombre de feuilles (analyse de Knuth et Moore, 1975).
- Ajouter une table de transposition (dictionnaire position → valeur) à minimax et compter les positions distinctes.
