# 43 · Gaussiennes à plusieurs dimensions, covariance et règle de la chaîne

| | |
|---|---|
| **Chapitre lié** | [43. Plusieurs variables aléatoires](https://laclasse.poneyponey.fr/maths/p7-probas-ia/43-vecteurs-aleatoires/), sections 3 (règle de la chaîne), 6 (covariance), 7 (normale multivariée et ellipses), 8 (somme de gaussiennes) et 9 (changement de variables) |
| **Commande** | `make nb NUM=43` (éditeur), `make nb NUM=43 MODE=run` (complet), `make nb NUM=43 MODE=smoke` (moins de 30 s) |
| **Durée mesurée** | run complet : 2 s, dont 0,1 s de calcul ; smoke : 1 à 2 s (machine de travail, 27/10/2026) |
| **Travail estimé** | 20 min |
| **Graine** | 42 (`seed_everything(42)`, variable `SEED` du Makefile) |
| **Données** | aucune : tirages pseudo-aléatoires, rien n’est téléchargé |

## Objectif

Tirer une loi normale à deux dimensions par $\mathbf{x} = \boldsymbol{\mu} + L\mathbf{z}$ (factorisation de Cholesky $LL^\top = \Sigma$), comparer la covariance du nuage à $\Sigma$, tracer les ellipses et les axes propres (`np.linalg.eigh`), puis vérifier par simulation quatre résultats du cours : la somme de gaussiennes indépendantes, une covariance nulle sans indépendance ($Y = X^2$), la règle de la chaîne sur un modèle de bigrammes, et la densité d’un flot affine $x = 2z + 1$.

## Ce que vous allez faire

1. Lire la cellule de configuration : elle seule change entre mode complet et mode smoke (nombre de tirages).
2. Lire la vérification sympy : $\det \Sigma = 16/25$, valeurs propres $8/5$ et $2/5$, forme quadratique $\frac{x_1^2 - 1{,}2\,x_1x_2 + x_2^2}{0{,}64}$, proportion $1 - e^{-c/2}$ des points dans l’ellipse de niveau $c$, et $f_Y(3) = \frac{e^{-1/2}}{2\sqrt{2\pi}}$.
3. Cliquer sur « Lancer le calcul » dans l’éditeur marimo, puis lire les deux graphiques.

## Résultat attendu

Run complet, 400 000 tirages par expérience (seuils du smoke entre parenthèses, 40 000 tirages) :

- Covariance du nuage pour $\Sigma = \begin{pmatrix} 1 & 0{,}6 \\ 0{,}6 & 1 \end{pmatrix}$ : écart maximal **0,003** (seuil 0,01 ; 0,03 en smoke). Valeurs propres 0,4 et 1,6.
- Points dans l’ellipse $c = 1$ : **0,394** pour $1 - e^{-1/2} \approx 0{,}3935$ ; dans l’ellipse $c = 4$ : **0,865** pour $0{,}8647$ (écart toléré 0,005 ; 0,015 en smoke).
- Somme de $\mathcal{N}(1 \,;\, 1)$ et $\mathcal{N}(2 \,;\, 1{,}5^2)$ indépendantes : moyenne **3,00**, variance **3,24** pour 3,25 (et non $2{,}5^2 = 6{,}25$), 0,683 des valeurs à moins d’un écart-type de la moyenne.
- $Y = X^2$ : covariance **−0,0002**, alors que $P(X = 0, Y = 1) = 0$ et $P(X = 0)\,P(Y = 1) \approx 0{,}222$.
- « le chat dort » : probabilité par la règle de la chaîne $1 \times 0{,}6 \times 0{,}7 = 0{,}42$, fréquence observée **0,421**.
- Flot affine : densité observée **0,121** autour de 3, pour $f_X(3) \approx 0{,}1210$ ; $\ln f_X(3) \approx -2{,}112$.

Le notebook s’arrête par `SystemExit` si un seuil n’est pas atteint. Figure : `figures/gaussiennes.png`.

## Ce qu’il faut observer

- Le nuage remplit les ellipses : un peu moins de 40 % des points dans la première, environ 86 % dans la seconde, et les axes propres suivent les diagonales $y = x$ et $y = -x$.
- La variance de la somme est la somme des variances : l’écart-type vaut environ 1,80, pas 2,5.
- $X$ et $X^2$ ont une covariance nulle et sont pourtant liées au point qu’une case du tableau est vide.

## Questions de réflexion

1. Remplacez $\rho = 0{,}6$ par $0{,}95$ : que deviennent les valeurs propres, la forme des ellipses et la factorisation de Cholesky ? Que se passe-t-il pour $\rho = 1$ ?
2. Pourquoi la proportion de points dans l’ellipse de niveau $c$ ne dépend-elle pas de $\Sigma$ ?
3. Dans le modèle de bigrammes, calculez $p(\text{le chien mange})$ par la règle de la chaîne, puis ajoutez-la au calcul.
