# 51 · La convolution, à la main et avec numpy

| | |
|---|---|
| **Chapitre lié** | [51. Convolution discrète 1D et 2D](https://laclasse.poneyponey.fr/maths/p9-discret/51-convolution/), sections 1 (corrélation et convolution 1D), 2 et 3 (convolution 2D, taille de sortie), 4 (filtres), 5 (matrice), 6 (équivariance) et 7 (deux dés) |
| **Commande** | `make nb NUM=51` (éditeur), `make nb NUM=51 MODE=run` (complet), `make nb NUM=51 MODE=smoke` (moins de 30 s) |
| **Durée mesurée** | run complet : 2 s, dont 0,1 s de calcul ; smoke : 2 s (machine de travail, 27/10/2026) |
| **Travail estimé** | 30 min |
| **Graine** | 42 (`seed_everything(42)`, variable `SEED` du Makefile) |
| **Données** | aucune : signaux pseudo-aléatoires et image synthétique, rien n’est téléchargé |

## Objectif

Écrire la corrélation croisée $y_i = \sum_j w_j\, x_{i+j}$ avec des tranches de tableau et la comparer à `np.correlate`, puis à `np.convolve`, qui retourne le noyau ; lisser un signal bruité par une moyenne glissante ; faire glisser les filtres du cours sur une image avec `sliding_window_view` ; retrouver la corrélation comme un produit $T x$ ; vérifier l’équivariance par translation et sa perte avec un pas 2 ; obtenir la loi de la somme de deux dés par `np.convolve`.

## Ce que vous allez faire

1. Lire la cellule de configuration : elle seule change entre mode complet et mode smoke (nombre de tirages, longueur du signal, taille de l’image).
2. Lire la fonction `correlation` : $k$ tranches décalées de $x$, multipliées par les poids et additionnées, sans boucle sur $i$.
3. Cliquer sur « Lancer le calcul » dans l’éditeur marimo, puis lire les résultats imprimés et la figure.

## Résultat attendu

- Corrélation à la main contre `np.correlate`, et contre `np.convolve` avec le noyau retourné : écart maximal **≤ 10⁻¹²** sur 5 000 couples (signal, noyau) tirés au hasard (mesuré : 3,6 × 10⁻¹⁵).
- Exemple du cours, signal $(0, 0, 1, 3, 2, 0, 0, 0, 0)$ et noyau $(1, 0, -1)$ : corrélation $(-1, -3, -1, 3, 2, 0, 0)$, convolution $(1, 3, 1, -3, -2, 0, 0)$ ; ventes $(120, 90, 150, 60, 180)$ : moyennes $(120, 100, 130)$.
- Moyenne glissante sur 21 points d’une sinusoïde bruitée : erreur quadratique divisée par au moins 3 (mesuré : 0,513 → 0,111).
- Image de 64 pixels, noyau 3 : sorties de **62, 64 et 32** pixels ($p = 0$ ; $p = 1$ ; $p = 1$ et $s = 2$). Bord vertical $6 \times 6$ et noyau de Prewitt : carte de colonnes $0, 3, 3, 0$. Sobel x sur le carré (0,2 et 0,9) : maximum $4 \times 0{,}7 = 2{,}8$. Sur une zone uniforme de 0,2 : 0,2 pour les flous et la netteté, 0 pour Sobel et le laplacien.
- $T x = (5, 3, 5)$ pour $w = (\frac{1}{4}, \frac{1}{2}, \frac{1}{4})$ et $x = (4, 8, 0, 4, 12)$.
- Équivariance (pas 1, décalage de 7) : écart exactement nul loin des bords. Pas 2, noyau $(1, 1)$, impulsion décalée de 0, 1, 2 : sorties $(0, 1, 0, 0)$, $(0, 1, 0, 0)$, $(0, 0, 1, 0)$.
- Deux dés : $36\,P(X + Y = t) = 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1$, vérifié en fractions exactes par sympy.

Le notebook s’arrête par `SystemExit` si un seuil n’est pas atteint. Figure : `figures/convolution.png`.

## Ce qu’il faut observer

- `np.convolve` et `np.correlate` ne donnent pas la même chose pour un noyau qui n’est pas symétrique : l’un retourne le noyau, l’autre non. Pour la moyenne glissante, ils coïncident.
- Avec un remplissage de zéros, Sobel x répond aussi sur le bord de l’image (le fond de 0,2 est comparé aux zéros ajoutés) : c’est l’effet de bord du remplissage.
- Avec un pas 2, décaler l’impulsion d’une case ne décale pas la sortie (elle reste la même), alors qu’un décalage de deux cases la décale d’une case.

## Questions de réflexion

1. Pourquoi la moyenne glissante sur 21 points retarde-t-elle un peu le signal si on place chaque moyenne au début de sa fenêtre plutôt qu’en son milieu ?
2. Combien de cases non nulles contient la matrice $T$ pour $n = 1\,000$ et $k = 3$, et quelle part de la matrice cela représente-t-il ?
3. Remplacez le dé par un dé truqué ($P(6) = \frac{1}{2}$, les autres faces à $\frac{1}{10}$) : quelle somme devient la plus probable ?

## Pour aller plus loin

- Remplacer l’image synthétique par une photo en niveaux de gris (tableau numpy) et comparer les filtres.
- Écrire la convolution 2D par la matrice $T$ d’une image aplatie et la comparer à `sliding_window_view`.
