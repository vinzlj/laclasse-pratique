# Exercice 05 : règles, contraintes et plans, trois façons de raisonner sans apprendre

| | |
|---|---|
| **Chapitre lié** | [05. Raisonner : logique, CSP, planification, systèmes experts](https://laclasse.poneyponey.fr/ia/p01-symbolique/05-raisonnement/) |
| **Exécution** | locale, CPU, Python pur (aucun GPU) |
| **Commande** | `make ex NUM=05` (éditeur), `make ex NUM=05 MODE=run` (complet), `make ex NUM=05 MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 20 s, l'essentiel pour le monde à 8 blocs (695 417 états énumérés) ; smoke : 4 s (M1 Pro, CPU, 14/10/2026) |
| **Travail estimé** | 1 h 30 |
| **Seed** | 42 (`seed_everything(42)`) ; seul le tirage des instances du monde des blocs est aléatoire |
| **Données** | aucune : règles, grilles et instances sont dans le notebook ; rien n'est téléchargé |

## Objectif

Programmer les trois moteurs génériques de l'IA symbolique (chaînage de règles, propagation de contraintes, planification STRIPS) et mesurer ce qui les rend efficaces ou les fait exploser. Aucune fonction de perte : rien n'est appris, tout est déduit ou cherché.

## Ce que vous allez faire

1. **Système expert miniature.** Une base de 15 règles diagnostique la panne d'un site web (502, 504, page blanche, certificat expiré, DNS…).
   - Chaînage avant : appliquer les règles jusqu'au point fixe, sur 5 cas (dont un hors domaine).
   - Chaînage arrière, comme MYCIN : partir de chaque hypothèse et ne poser que les questions utiles ; compter les questions.
   - Vérifier avec un solveur SAT générique (`sympy.logic.inference.satisfiable`) que les conclusions sont exactement celles que la base implique : $B \models q$ si et seulement si $B \wedge \neg q$ est insatisfiable.
2. **Contraintes (CSP).** AC-3 générique, testé à la main sur $X < Y < Z$, puis le sudoku (81 variables, contrainte ≠ entre les 20 « pairs » de chaque case) :
   - AC-3 seul, sans recherche, sur une grille facile et une grille difficile ;
   - retour arrière naïf, puis MRV (la case la plus contrainte d'abord) + vérification en avant, puis MAC (AC-3 maintenu après chaque choix) ; compter les affectations essayées.
3. **Planification STRIPS.** Le monde des blocs avec 4 actions (`prendre`, `poser`, `empiler`, `depiler`), chacune décrite par préconditions, ajouts et retraits :
   - recherche en largeur (plan le plus court) et recherche gloutonne (heuristique : nombre de faits du but encore faux) sur l'anomalie de Sussman ;
   - stratégie naïve « un sous-but après l'autre », dans les deux ordres ;
   - explosion : taille de l'espace d'états et états développés de 2 à 8 blocs (5 instances aléatoires par taille).

## Résultat attendu

Le run complet s'arrête avec une erreur si l'un de ces seuils n'est pas atteint (valeurs mesurées entre parenthèses) :

| Contrôle | Seuil | Mesuré |
|---|---|---|
| Chaînage avant, 5 cas | 5/5 corrects | 5/5 |
| Chaînage arrière, 5 cas | 5/5 corrects | 5/5 |
| Accord solveur SAT / chaînage avant | 5/5 | 5/5 |
| Sudoku, 3 méthodes × 2 grilles | 6 solutions valides | 6 |
| Grille difficile : affectations naïf / MAC | **≥ 100** | 445 778 / 1 850 = **241** |
| Anomalie de Sussman, recherche en largeur | plan de **6 actions** | 6 |
| Sous-buts traités l'un après l'autre | échec dans les deux ordres | échec (8 actions, un sous-but défait) |

Autres valeurs mesurées (déterministes, sauf les durées, indicatives sur M1 Pro, qui varient de quelques pour cent d'un run à l'autre) :

| Sudoku | Facile (51 cases vides) | Difficile (60 cases vides) |
|---|---|---|
| AC-3 seul | résout les 51 cases, 459 → 51 candidats | ne résout **aucune** case, 540 → 254 candidats |
| Naïf | 37 652 affectations, 24 ms | 445 778 affectations, 285 ms |
| MRV + vérification en avant | 51 affectations, 1 ms | 10 101 affectations, 170 ms |
| MAC (AC-3) | 0 affectation, 8 ms | 1 850 affectations, 440 ms (516 257 révisions d'arcs) |

| Blocs | Actions instanciées | États atteignables | Développés, largeur (moy.) | Développés, glouton (moy.) | Plan largeur / glouton (moy.) |
|---|---|---|---|---|---|
| 3 | 18 | 22 | 5 | 5 | 3,6 / 3,6 |
| 5 | 50 | 866 | 285 | 23 | 8,0 / 9,2 |
| 7 | 98 | 65 990 | 9 270 | 72 | 10,8 / 13,6 |
| 8 | 128 | 695 417 | 66 977 | 76 | 13,2 / 15,2 |

Figure produite : `figures/raisonnement.png` (affectations au sudoku, candidats après AC-3, explosion du monde des blocs).

## Ce qu'il faut observer

- **Chaînage avant contre arrière.** Le chaînage avant veut toutes les observations d'emblée (12 faits observables) ; le chaînage arrière n'en demande que 5 à 8 pour tester tous les diagnostics, et 2 seulement pour vérifier une hypothèse unique (« certificat expiré ? »). C'est pourquoi un système de consultation comme MYCIN raisonnait à rebours : chaque question coûte du temps au médecin.
- **La fragilité.** Le cas « hors domaine » (images absentes, lenteur sur mobile) ne produit **aucun** diagnostic, sans message d'erreur : le système ne sait pas qu'il ne sait pas. Chaque panne nouvelle exige une règle écrite par un humain (goulot de l'acquisition des connaissances).
- **La propagation remplace la recherche.** Sur la grille facile, AC-3 seul trouve la solution sans aucun choix. Sur la grille difficile, il ne fixe aucune case mais divise par deux le nombre de candidats ; combiné à la recherche (MAC), il essaie 241 fois moins d'affectations que le naïf. Mais chaque affectation coûte plus cher : en temps, MAC (≈ 440 ms) est plus lent que MRV (≈ 170 ms) sur cette grille. Moins de nœuds n'est pas toujours moins de secondes.
- **L'anomalie de Sussman.** Viser « A sur B » puis « B sur C » défait le premier sous-but, et l'ordre inverse aussi : les sous-buts interagissent. La recherche dans l'espace des états complets trouve le plan de 6 actions.
- **L'explosion combinatoire.** L'espace d'états est multiplié par 4 à 11 à chaque bloc ajouté (de 5 états à 2 blocs à 695 417 à 8). La recherche en largeur suit cette croissance ; la recherche gloutonne développe quelques dizaines d'états seulement, mais rend des plans plus longs (15,2 actions contre 13,2 à 8 blocs). Son heuristique est pourtant admissible : chaque action ajoute au plus un fait du but, donc h ne surestime jamais le nombre d'actions restantes. Le défaut vient de la recherche gloutonne elle-même, qui ne regarde que h et ignore le coût déjà payé g : elle ne garantit jamais le plan le plus court. A\* (f = g + h, chapitre 04) le garantirait.

## Questions de réflexion

1. Ajoutez la règle « SI statut_502 ALORS diagnostic:appli_plante_au_demarrage ». Que devient le cas « déploiement raté » ? Et un 502 dû à un manque de mémoire ? Que dit cette expérience de la maintenance d'une base de 500 règles ?
2. Le chaînage avant et le solveur SAT donnent les mêmes conclusions. Pourquoi cette garantie disparaît-elle si une règle a une conclusion de la forme « A OU B » (qui n'est plus une clause de Horn) ?
3. MAC essaie moins d'affectations mais prend plus de temps que MRV sur la grille difficile. Quelle partie du code de `ac3` faudrait-il spécialiser pour la contrainte ≠ afin de réduire le coût d'une révision d'arc ?
4. Écrivez l'heuristique « nombre de blocs mal placés » pour le monde des blocs. Est-elle admissible ? Utilisez-la dans un A\* (chapitre 04) : le plan redevient-il optimal, et combien d'états sont développés à 8 blocs ?

## Pour aller plus loin

- Ajouter des facteurs de certitude à la MYCIN (une valeur entre −1 et 1 par règle) et comparer les diagnostics quand deux règles concluent la même chose.
- Résoudre le sudoku avec le solveur SAT de `sympy` (729 variables booléennes) et comparer à MAC.
- Écrire le domaine du monde des blocs en PDDL et le donner à un planificateur existant.
