# 32 · Tenseurs en numpy : formes, @, *, transposée, broadcasting

| | |
|---|---|
| **Chapitre lié** | [32. Matrices, produit matriciel, tenseurs et broadcasting](https://laclasse.poneyponey.fr/maths/p5-algebre-lineaire/32-matrices/), sections 3 (produit matriciel), 4 (transposée), 6 (produit terme à terme), 7 (tenseurs et broadcasting) et 8 (paramètres d’une couche) |
| **Commande** | `make nb NUM=32` (éditeur), `make nb NUM=32 MODE=run` (complet), `make nb NUM=32 MODE=smoke` (moins de 30 s) |
| **Durée mesurée** | run complet : 2 s, dont 0,05 s de calcul ; smoke : 2 s (machine de travail, 20/10/2026) |
| **Travail estimé** | 25 min |
| **Graine** | 42 (`seed_everything(42)`, variable `SEED` du Makefile) |
| **Données** | aucune : matrices tirées au hasard, rien n’est téléchargé |

## Objectif

Prévoir la forme du résultat d’une opération numpy avant de la lancer, et voir les quatre réflexes du chapitre à l’œuvre : la règle des dimensions du produit `@`, le choix de l’ordre des produits (associativité et coût), la différence entre `@` et `*`, et les règles du broadcasting, y compris son piège silencieux.

## Ce que vous allez faire

1. Lire la cellule de configuration : elle seule change entre mode complet et mode smoke (taille des matrices, taille du lot, taille de la matrice orthogonale).
2. Lire l’exemple de la boutique (section « Pourquoi ça compte » du cours) : `commandes @ tarifs` donne un tableau (3, 2), et `tarifs @ commandes` échoue.
3. Cliquer sur « Lancer le calcul » dans l’éditeur marimo : ordre des produits, `@` contre `*`, transposée, broadcasting, couche dense sur un lot, matrice orthogonale.

## Résultat attendu

- Boutique : 38 € et 0,8 kg, 24 € et 1,2 kg, 31 € et 1 kg ; `tarifs @ commandes` lève une `ValueError` (« mismatch in its core dimension »).
- $(AB)\mathbf{x} = A(B\mathbf{x})$ à $10^{-10}$ près (écart relatif mesuré : $10^{-15}$), mais $n^3 + n^2$ multiplications contre $2n^2$ : **au moins 100 fois plus** (mesuré : 500 fois pour $n = 1\,000$, 150 fois pour $n = 300$ en smoke).
- `M @ N` donne $\begin{pmatrix} 19 & 22 \\ 43 & 50 \end{pmatrix}$, `M * N` donne $\begin{pmatrix} 5 & 12 \\ 21 & 32 \end{pmatrix}$ ; $(PR)^\top = R^\top P^\top$ exactement, et $P^\top R^\top$ n’existe pas pour $P$ de forme (2, 3) et $R$ de forme (3, 4).
- Broadcasting : $(3, 1) + (1, 4)$ donne la table $(3, 4)$ de 1 à 24 ; $(3, 4) + (3,)$ lève « operands could not be broadcast together » ; $(40,) - (40, 1)$ donne un tableau $(40, 40)$ et une erreur quadratique **au moins 10 fois trop grande** (mesuré : 1,18 au lieu de 0,0083).
- Couche dense $784 \to 128$ sur un lot : $Y = XW^\top + \mathbf{b}$ de forme $(256, 128)$ (32 en smoke), identique à $W\mathbf{x} + \mathbf{b}$ exemple par exemple à $10^{-10}$ près ; $100\,480$ paramètres, et $101\,770$ pour le réseau $784 \to 128 \to 10$.
- Matrice orthogonale (décomposition QR) : $\lvert Q^\top Q - I \rvert$ et l’écart des normes inférieurs à $10^{-10}$.
- sympy vérifie $(AB)^\top = B^\top A^\top$ et $(AB)C = A(BC)$ sur des matrices symboliques, $(\mathbf{u}\mathbf{v}^\top)_{ij} = u_i v_j$, l’exemple $AB \ne BA$ du cours et $Q^\top Q = I$ pour la matrice de coefficients $3/5$ et $4/5$.

Le notebook s’arrête par `SystemExit` si un seuil n’est pas atteint. Figure : `figures/tenseurs.png`.

## Ce qu’il faut observer

- Le temps mesuré pour $(AB)\mathbf{x}$ et $A(B\mathbf{x})$ suit le nombre de multiplications, mais pas exactement : numpy calcule un produit de matrices très vite (bibliothèque BLAS, plusieurs cœurs). Le nombre de multiplications, lui, ne dépend pas de la machine.
- Le piège $(40,) - (40, 1)$ ne produit aucun message : seule la forme $(40, 40)$ trahit l’erreur. Un `assert` sur la forme l’aurait arrêté.
- La couche calculée sur le lot en un seul produit donne exactement les mêmes nombres que la boucle sur les exemples : le lot ne change que la vitesse.

## Questions de réflexion

1. Pour $A$ de forme $(n, n)$ et $\mathbf{x}$ de forme $(n,)$, pourquoi `A @ x` marche-t-il alors que $\mathbf{x}$ n’est ni une ligne ni une colonne ? Quelle est la forme du résultat ?
2. Dans le piège, comment corriger le calcul de deux façons différentes (`reshape`, ou `[:, 0]`) ?
3. Une couche $784 \to 128$ coûte combien de multiplications pour un lot de 256 exemples ?

## Pour aller plus loin

- Mesurer le temps de `(A @ B) @ C` et `A @ (B @ C)` avec $A$ de forme $(1\,000, 10)$, $B$ de forme $(10, 1\,000)$ et $C$ de forme $(1\,000, 10)$ : compter d’abord les multiplications de chaque ordre.
- Remplacer `X @ W.T + b` par `np.einsum("bi,oi->bo", X, W) + b` et vérifier l’égalité.
