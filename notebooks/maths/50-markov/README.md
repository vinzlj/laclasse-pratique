# 50 · Chaînes de Markov, PageRank et équation de Bellman

| | |
|---|---|
| **Chapitre lié** | [50. Chaînes de Markov et équation de Bellman](https://laclasse.poneyponey.fr/maths/p9-discret/50-markov-bellman/), sections 3 (loi après n pas), 4 (distribution invariante et convergence), 5 (PageRank), 7 (équation de Bellman) et 8 (fréquences de visite) |
| **Commande** | `make nb NUM=50` (éditeur), `make nb NUM=50 MODE=run` (complet), `make nb NUM=50 MODE=smoke` (moins de 30 s) |
| **Durée mesurée** | run complet : 2 s, dont 0,2 s de calcul ; smoke : 2 s (machine de travail, 27/10/2026) |
| **Travail estimé** | 30 min |
| **Graine** | 42 (`seed_everything(42)`, variable `SEED` du Makefile) |
| **Données** | aucune : les chaînes, le graphe aléatoire et le court texte de l’expérience 5 sont écrits ou générés par le notebook, rien n’est téléchargé |

## Objectif

Manipuler une matrice de transition en numpy : calculer la loi $\pi_0 P^n$ avec `np.linalg.matrix_power`, trouver la distribution invariante par `np.linalg.eig` sur $P^\top$ et par un système linéaire, simuler une marche et comparer ses fréquences de visite à l’invariante, calculer un PageRank par itération, évaluer une politique par `np.linalg.solve(I − γP, R)` et par l’itération de Bellman, et générer du texte avec une chaîne de lettres.

## Ce que vous allez faire

1. Lire la cellule de configuration : elle seule change entre mode complet et mode smoke (longueur de la marche, taille du graphe aléatoire, nombre de lettres générées).
2. Lire les deux calculs de l’invariante : le vecteur propre de `P.T` pour la valeur propre 1, divisé par sa somme ; puis le système $(P^\top - I)\pi^\top = 0$ où une équation est remplacée par « somme = 1 ».
3. Cliquer sur « Lancer le calcul » dans l’éditeur marimo, puis lire les résultats imprimés et la figure.

## Résultat attendu

- Chaîne du site : $\pi_2 = (0{,}68 \,;\, 0{,}18 \,;\, 0{,}14)$ exactement, $\pi_{10} \approx (0{,}4206 \,;\, 0{,}3833 \,;\, 0{,}1961)$, invariante $(0{,}4 \,;\, 0{,}4 \,;\, 0{,}2)$ par les deux méthodes à **10⁻¹² près**, $\pi_{30}$ à moins de **10⁻³** de l’invariante ; le plus grand module des autres valeurs propres vaut $\frac{1}{2} + \frac{\sqrt{5}}{10} \approx 0{,}7236$.
- Marche simulée : fréquences de visite à moins de **0,01** de l’invariante après 200 000 pas (mesuré : 0,0007) ; **0,03** en smoke, 20 000 pas (mesuré : 0,0017).
- PageRank des quatre pages du cours ($d = 0{,}85$) : $(0{,}3725 \,;\, 0{,}1958 \,;\, 0{,}3941 \,;\, 0{,}0375)$, égal à la valeur exacte de sympy à **10⁻⁹ près** ; sur un graphe aléatoire de 500 pages (quelques-unes sans lien sortant), l’itération rejoint la résolution directe à 10⁻⁹ près (mesuré : 41 itérations, écart 2 × 10⁻¹⁴).
- Grille $2 \times 3$ du cours, $\gamma = 0{,}9$ : $V = (0{,}81 \,;\, 0{,}9 \,;\, 1 \,;\, 0{,}9 \,;\, 1 \,;\, 0)$, atteint en **3 balayages au plus**. Glacier de la section 7 : $V = (7{,}1875 \,;\, 5{,}625)$ ; à chaque balayage, le plus grand écart à $V$ est multiplié par **0,9 au plus** (237 balayages pour descendre sous 10⁻¹⁰).
- Chaîne de lettres : la lettre la plus probable après « q » est « u ».
- sympy refait exactement les nombres du cours : invariante $(2/5, 2/5, 1/5)$, $\pi_2$, PageRank pour $d = 17/20$, valeurs du glacier $(115/16, 45/8)$.

Le notebook s’arrête par `SystemExit` si un seuil n’est pas atteint. Figure : `figures/markov.png`.

## Ce qu’il faut observer

- Les trois probabilités oscillent autour de l’invariante avant de s’y caler : les valeurs propres $-0{,}276$ et $-0{,}724$ sont négatives.
- Les fréquences de visite convergent beaucoup plus lentement que la loi $\pi_n$ : il faut des milliers de pas pour deux chiffres justes, contre une vingtaine de clics pour la loi.
- L’écart de l’itération de Bellman décroît exactement comme $0{,}9^k$ : une droite sur l’échelle logarithmique.
- Le texte généré « ressemble » à du français par ses paires de lettres, sans aucun mot : une chaîne d’ordre 1 ne voit qu’une lettre en arrière.

## Questions de réflexion

1. Que devient le PageRank des quatre pages quand $d$ tend vers 1 ? Pourquoi la page 4 tombe-t-elle à 0 ?
2. Combien de balayages faudrait-il pour $\gamma = 0{,}99$ au lieu de $0{,}9$, avec la même précision ?
3. Comment modifier la chaîne de lettres pour qu’elle tienne compte des deux lettres précédentes ? Combien d’états faut-il alors ?

## Pour aller plus loin

- Ajouter du glissement à la grille (l’action réussit avec la probabilité 0,8, sinon l’agent reste sur place) et comparer les valeurs.
- Remplacer la chaîne de lettres par une chaîne de mots sur un texte plus long du domaine public.
