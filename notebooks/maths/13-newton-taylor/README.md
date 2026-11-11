# 13 · Newton et Taylor

| | |
|---|---|
| **Chapitre lié** | [13. Étude de fonctions, optimisation, convexité, Taylor](https://laclasse.poneyponey.fr/maths/p2-analyse/13-etude-fonctions/), sections 6 (polynôme de Taylor à l’ordre 1 et 2) et 7 (méthode de Newton) |
| **Commande** | `make nb NUM=13` (éditeur), `make nb NUM=13 MODE=run` (complet), `make nb NUM=13 MODE=smoke` (moins de 30 s) |
| **Durée mesurée** | run complet : 2 s, dont 0,1 s de calcul ; smoke : 2 s (M1 Pro, 18/10/2026) |
| **Travail estimé** | 20 min |
| **Graine** | 42 (`seed_everything(42)`, variable `SEED` du Makefile) |
| **Données** | aucune : départs tirés au hasard, rien n’est téléchargé |

## Objectif

Voir pourquoi la méthode de Newton est si rapide près d’une racine (le nombre de décimales justes double à chaque pas), mesurer l’erreur des polynômes de Taylor d’ordre 1 et 2 de √(1 + h) (en $h^2$ et en $h^3$), et constater que Newton peut aussi échouer selon le point de départ.

## Ce que vous allez faire

1. Lire la cellule de configuration : elle seule change entre mode complet et mode smoke.
2. Cliquer sur « Lancer le calcul » dans l’éditeur marimo.
3. Lire la figure : à gauche, les décimales justes de √2 au fil des pas de Newton, comparées à la dichotomie ; à droite, les erreurs de Taylor selon $h$ (échelles logarithmiques).

## Résultat attendu

- **Newton pour √2**, depuis $x_0 = 2$, en fractions exactes : au moins **40 décimales justes en 6 pas** (mesuré : 48,5) ; le rapport $e_{n+1}/e_n^2$ tend vers $\frac{1}{2\sqrt{2}} \approx 0{,}3536$ (**± 0,001**, mesuré : 0,3536). La dichotomie sur $[1 ; 2]$ demande au moins **100 pas** pour la même précision (mesuré : 162).
- **Taylor pour √(1 + h)**, $h$ entre $10^{-3}$ et $10^{-0,5}$ : pentes log-log de l’erreur **2 ± 0,05** (ordre 1) et **3 ± 0,05** (ordre 2) en mode complet, ± 0,15 en smoke (mesuré : 1,98 et 2,98) ; l’erreur d’ordre 2 reste sous $h^3/16$ (majoration démontrée dans l’exercice papier du chapitre).
- **x³ − 2x + 2** : le départ 0 donne exactement 0, 1, 0, 1, 0 ; la part des 50 000 départs tirés dans $[-3 ; 3]$ qui atteignent la racine $\approx -1{,}769$ en 60 pas est comprise **entre 0,5 et 1** (1 exclu ; mesuré : 77,8 %).

Le notebook s’arrête par `SystemExit` si un seuil n’est pas atteint. sympy vérifie le développement $\sqrt{1 + h} = 1 + \frac{h}{2} - \frac{h^2}{8} + \frac{h^3}{16} + \cdots$ et l’identité exacte $x_{n+1} - \sqrt{2} = \frac{(x_n - \sqrt{2})^2}{2x_n}$. Figure : `figures/newton-taylor.png`.

## Ce qu’il faut observer

- L’identité $x_{n+1} - \sqrt{2} = \frac{(x_n - \sqrt{2})^2}{2x_n}$ explique tout : l’erreur suivante est le **carré** de l’erreur actuelle, divisé par environ $2\sqrt{2}$. Une erreur de $10^{-6}$ devient environ $3,5 \times 10^{-13}$.
- Sur la figure de droite, les deux droites ont des pentes 2 et 3 : diviser $h$ par 10 divise l’erreur par 100 ou par 1 000. La majoration $h^3/16$ colle presque à l’erreur d’ordre 2, parce que c’est exactement le terme suivant du développement.
- Pour $x^3 - 2x + 2$, environ un départ sur cinq ne trouve pas la racine en 60 pas, et la plupart de ceux-là (environ 90 %, mesuré à part) finissent dans le cycle 0, 1, 0, 1 : ce cycle attire les départs voisins. Les départs proches d’un point où la tangente est horizontale ($x = \pm\sqrt{2/3}$) sont projetés très loin au premier pas.

## Questions de réflexion

1. Pourquoi la dichotomie gagne-t-elle toujours un bit (environ 0,3 décimale) par pas, quelle que soit la fonction ?
2. Que devient la vitesse de Newton pour $f(x) = x^2$ (racine double en 0) ? Essayez : l’erreur est-elle encore élevée au carré à chaque pas ?
3. Pour minimiser $g(x) = x + 4/x$ (section 6 du cours), écrivez l’itération de Newton sur $g'$ et comparez-la à une descente de gradient de pas $0{,}5$.

## Pour aller plus loin

- Remplacer √2 par √5 (fonction $x^2 - 5$) et vérifier que le rapport tend vers $\frac{1}{2\sqrt{5}}$.
- Tracer, pour chaque départ de $[-3 ; 3]$, le nombre de pas nécessaires pour converger : les zones de non-convergence apparaissent.
