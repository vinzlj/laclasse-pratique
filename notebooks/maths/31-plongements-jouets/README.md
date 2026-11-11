# 31 · Plongements jouets : distance ou cosinus ?

| | |
|---|---|
| **Chapitre lié** | [31. Vecteurs de ℝⁿ, normes, similarité cosinus, projections](https://laclasse.poneyponey.fr/maths/p5-algebre-lineaire/31-vecteurs-rn/), sections 5 (similarité cosinus) et 6 (distances, plus proches voisins et k-means) |
| **Commande** | `make nb NUM=31` (éditeur), `make nb NUM=31 MODE=run` (complet), `make nb NUM=31 MODE=smoke` (moins de 30 s) |
| **Durée mesurée** | run complet : 2 s, dont 0,3 s de calcul ; smoke : 1 s (machine de travail, 20/10/2026) |
| **Travail estimé** | 25 min |
| **Graine** | 42 (`seed_everything(42)`, variable `SEED` du Makefile) |
| **Données** | aucune : les « textes » sont des vecteurs de comptages tirés au hasard, rien n’est téléchargé |

## Objectif

Voir, sur des plongements jouets en numpy, pourquoi on compare des textes par la similarité cosinus plutôt que par la distance, pourquoi les deux reviennent au même pour des vecteurs normalisés, pourquoi le codage one-hot ne dit rien du sens, et comment k-means fait baisser son inertie.

## Ce que vous allez faire

1. Lire la cellule de configuration : elle seule change entre mode complet et mode smoke (nombre de requêtes, taille du vocabulaire one-hot, nombre de points de k-means).
2. Lire l’exemple du blog (section « Pourquoi ça compte » du cours) : $d(A, B) = \sqrt{80}$ mais $d(A, C) = \sqrt{22}$, alors que $\cos(A, B) = 1$ et $\cos(A, C) = 0$.
3. Cliquer sur « Lancer le calcul » dans l’éditeur marimo : plus proche voisin, vecteurs normalisés, one-hot, k-means.

## Résultat attendu

- Exemple du blog : $d(A, B) = \sqrt{80} \approx 8{,}94$, $d(A, C) = \sqrt{22} \approx 4{,}69$, $\cos(A, B) = 1$, $\cos(A, C) = 0$ (à $10^{-12}$ près).
- Plus proche voisin parmi 12 textes de référence : bon thème dans **au moins 97 %** des cas par le cosinus, et au moins **10 points de moins** par la distance (mesuré en mode complet, 30 000 requêtes : 99,8 % contre 82,4 % ; en smoke, 600 requêtes : 99,8 % contre 79,7 %).
- Vecteurs normalisés : $\lVert \mathbf{u} - \mathbf{v} \rVert_2^2 = 2 - 2\cos(\mathbf{u}, \mathbf{v})$ à $10^{-12}$ près, et même voisin pour les deux mesures.
- One-hot : toutes les distances entre deux mots différents valent $\sqrt 2$, tous les cosinus 0 (2 000 mots en complet, 50 en smoke).
- k-means sur les textes normalisés : inertie jamais croissante d’une étape à la suivante, pureté des groupes **≥ 95 %** (mesuré : 100 %).
- sympy vérifie $\lVert \mathbf{u} - \mathbf{v} \rVert^2 = \lVert \mathbf{u} \rVert^2 - 2\langle \mathbf{u}, \mathbf{v} \rangle + \lVert \mathbf{v} \rVert^2$, l’identité de Lagrange $\lVert \mathbf{u} \rVert^2 \lVert \mathbf{v} \rVert^2 - \langle \mathbf{u}, \mathbf{v} \rangle^2 = \sum_{i < j} (u_i v_j - u_j v_i)^2$ (d’où Cauchy-Schwarz en dimension 3) et l’orthogonalité de $\mathbf{u} - p(\mathbf{u})$ à $\mathbf{v}$.

Le notebook s’arrête par `SystemExit` si un seuil n’est pas atteint. Figure : `figures/plongements-jouets.png`.

## Ce qu’il faut observer

- La distance se trompe surtout sur les textes très courts : leurs vecteurs sont tous près de l’origine, donc près les uns des autres, quel que soit leur thème. Le cosinus ne regarde que la direction.
- Après normalisation, la distance retrouve exactement le classement du cosinus : c’est pourquoi les bases vectorielles stockent des vecteurs de norme 1.
- k-means se stabilise dès la deuxième étape sur ces données bien séparées ; sur des données réelles, il lui en faut davantage, et son résultat dépend des centres de départ.

## Questions de réflexion

1. Pourquoi le vecteur nul pose-t-il un problème pour le cosinus ? Comment le code l’évite-t-il ?
2. Avec un facteur de longueur tiré entre 1 et 2 seulement (au lieu de 0,2 et 10), la distance ferait-elle mieux ? Essayez.
3. Le one-hot donne la même distance $\sqrt 2$ entre tous les mots : qu’est-ce qu’un modèle peut en apprendre sur le sens des mots ?

## Pour aller plus loin

- Remplacer la distance euclidienne par la distance L1 (`np.abs(...).sum()`) dans l’expérience 1 : le défaut de longueur change-t-il ?
- Lancer k-means avec plusieurs graines de départ et comparer les inerties finales.
