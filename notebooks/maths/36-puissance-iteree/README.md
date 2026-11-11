# 36 · La puissance itérée, et le pas critique d’une descente

| | |
|---|---|
| **Chapitre lié** | [36. Valeurs propres, diagonalisation, matrices symétriques et formes quadratiques](https://laclasse.poneyponey.fr/maths/p5-algebre-lineaire/36-valeurs-propres/), sections 4 (rayon spectral, pas critique), 5 (matrices symétriques) et 7 (puissance itérée) |
| **Commande** | `make nb NUM=36` (éditeur), `make nb NUM=36 MODE=run` (complet), `make nb NUM=36 MODE=smoke` (moins de 30 s) |
| **Durée mesurée** | run complet : 2 s, dont 0,1 s de calcul ; smoke : 1 s (machine de travail, 27/10/2026) |
| **Travail estimé** | 30 min |
| **Graine** | 42 (`seed_everything(42)`, variable `SEED` du Makefile) |
| **Données** | aucune : matrices pseudo-aléatoires, rien n’est téléchargé |

## Objectif

Programmer la puissance itérée $\mathbf{x}_{k+1} = A\mathbf{x}_k / \lVert A\mathbf{x}_k \rVert$ en numpy, sur la matrice $2 \times 2$ du cours puis sur des matrices symétriques $50 \times 50$ et $200 \times 200$, et comparer le résultat à `np.linalg.eigh` ; mesurer la vitesse de convergence selon le rapport $|\lambda_2/\lambda_1|$ ; vérifier le pas critique $2/\lambda_{\max}$ d’une descente de gradient sur $\frac{1}{2}\mathbf{x}^\top A\mathbf{x}$.

## Ce que vous allez faire

1. Lire la cellule de configuration : elle seule change entre mode complet et mode smoke (tailles des matrices, nombre de pas).
2. Lire la fonction `puissance_iteree` : un produit matrice-vecteur, une division par la norme, le quotient de Rayleigh $\mathbf{x}^\top A\mathbf{x}$ (ici $\lVert\mathbf{x}\rVert = 1$).
3. Cliquer sur « Lancer le calcul » dans l’éditeur marimo, puis lire les résultats imprimés et la figure.

## Résultat attendu

- Matrice $\begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$, départ $(1 \,;\, 0)$ : vecteurs non normalisés $(1 \,;\, 0)$, $(2 \,;\, 1)$, $(5 \,;\, 4)$, $(14 \,;\, 13)$, $(41 \,;\, 40)$ ; quotients de Rayleigh $2$ ; $\frac{14}{5}$ ; $\frac{122}{41}$ ; $\frac{1\,094}{365}$ ; $\frac{9\,842}{3\,281}$ (vérifiés exactement par sympy), et **3 à 10⁻⁹ près** après 12 pas.
- Matrices symétriques aléatoires ($50 \times 50$, et $200 \times 200$ en mode complet) : la valeur propre de plus grand module est retrouvée avec un écart relatif **≤ 10⁻⁸** à `np.linalg.eigh` (mesuré : 7,1 × 10⁻¹³ et 1,9 × 10⁻¹²). Mais il faut 2 154 et 5 306 pas : dans ces matrices, $|\lambda_2/\lambda_1|$ vaut 0,993 et 0,997.
- Vitesse : pour $|\lambda_2/\lambda_1| = 0{,}5$ ; $0{,}9$ ; $0{,}99$, le sinus de l’angle avec la direction propre passe sous 10⁻⁶ en 21, 133 et 1 387 pas, à **20 % près** de la prévision $\log(10^{-6}/C)/\log|\lambda_2/\lambda_1|$ (20, 129 et 1 350).
- Descente sur $\frac{1}{2}\mathbf{x}^\top A\mathbf{x}$ ($A$ symétrique définie positive $20 \times 20$) : avec $\eta = 0{,}99 \times 2/\lambda_{\max}$, la norme de $\mathbf{x}$ est divisée par plus de 100 ; avec $\eta = 1{,}01 \times 2/\lambda_{\max}$, elle est multipliée par plus de 10 (en mode complet, 2 000 pas : 10⁻¹⁸ contre 10¹⁶). Exemple du cours $\frac{1}{2}(x^2 + 10y^2)$ : facteurs $(0{,}85 \,;\, -0{,}5)$, $(0{,}8 \,;\, -1)$ et $(0{,}75 \,;\, -1{,}5)$ pour $\eta = 0{,}15$ ; $0{,}2$ ; $0{,}25$.
- sympy vérifie aussi $A^{10}$ pour $A = \begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix}$ et les valeurs propres $1$ et $0{,}7$ du tableau des abonnements.

Le notebook s’arrête par `SystemExit` si un seuil n’est pas atteint. Figure : `figures/puissance-iteree.png`.

## Ce qu’il faut observer

- Sur une échelle logarithmique, l’écart à la direction propre descend en ligne droite : il est multiplié par le même nombre, $|\lambda_2/\lambda_1|$, à chaque pas.
- Une matrice aléatoire a souvent des valeurs propres de modules voisins : la puissance itérée y est lente, et les bibliothèques utilisent d’autres méthodes (`np.linalg.eigh`).
- Juste au-dessus du pas critique, la descente ne diverge pas tout de suite : les directions douces convergent d’abord, puis la direction la plus raide prend le dessus et explose.

## Questions de réflexion

1. Que se passe-t-il si le vecteur de départ est exactement orthogonal à la direction propre cherchée ? Et en pratique, avec les arrondis des flottants ?
2. La matrice $200 \times 200$ a donné une valeur propre négative. Pourquoi la puissance itérée trouve-t-elle la plus grande en **module** et pas la plus grande tout court ?
3. Comment adapter la méthode pour trouver la **plus petite** valeur propre d’une matrice inversible ?

## Pour aller plus loin

- Remplacer la matrice aléatoire par une matrice de covariance de données simulées (chapitre 37) : le premier vecteur propre donne l’axe principal du nuage.
- Ajouter un décalage : la puissance itérée sur $A - \mu I$ trouve une autre valeur propre.
