# 18 · Intégrale numérique

| | |
|---|---|
| **Chapitre lié** | [18. Intégration](https://laclasse.poneyponey.fr/maths/p2-analyse/18-integration/), sections 2 (méthode des rectangles) et 8 (intégrale le long d’un segment) |
| **Commande** | `make nb NUM=18` (éditeur), `make nb NUM=18 MODE=run` (complet), `make nb NUM=18 MODE=smoke` (moins de 30 s) |
| **Durée mesurée** | run complet : 2 s, dont 0,002 s de calcul ; smoke : 2 s (20/10/2026) |
| **Travail estimé** | 20 min |
| **Graine** | 42 (`seed_everything(42)`, variable `SEED` du Makefile) ; aucun tirage au hasard |
| **Données** | aucune : les fonctions sont écrites dans le notebook, rien n’est téléchargé |

## Objectif

Approcher une intégrale par des sommes de rectangles (à gauche, à droite, au milieu) et de trapèzes, mesurer comment l’erreur diminue quand le nombre de bandes $n$ grandit, puis calculer des gradients intégrés en une dimension et vérifier leur complétude : $\mathrm{IG} = f(x) - f(x')$.

## Ce que vous allez faire

1. Lire la cellule de configuration : elle seule change entre mode complet et mode smoke.
2. Cliquer sur « Lancer le calcul » dans l’éditeur marimo.
3. Lire la figure : à gauche, l’erreur commise sur $\int_0^1 \mathrm{e}^{x}\,\mathrm{d}x$ selon $n$ (échelles logarithmiques) ; à droite, le rapport entre le gradient intégré approché et $f(x) - f(x')$ selon le nombre de pas $m$.

## Résultat attendu

En mode complet ($n = 2, 4, \ldots, 2^{14}$) : pente log-log de l’erreur **−1 ± 0,1** pour les rectangles à gauche et à droite, **−2 ± 0,1** pour le milieu et les trapèzes, sur $x^2$, $\mathrm{e}^{x}$ et $1/x$ (mesuré : −0,97 à −1,03 et −1,99 à −2,00). Valeurs du cours retrouvées exactement : 1,75 ; 3,75 ; 2,625 (gauche, droite, milieu, $x^2$ sur ${[0 \,;\, 2]}$, $n = 4$) et 2,662668 ($n = 1\,000$, à gauche). Gradients intégrés par rectangles à droite, avec 300 pas : écart à $f(x) - f(x')$ **≤ 0,04** (mesuré : 9,03 au lieu de 9 pour $x^2$ de 0 à 3 ; 0,993 au lieu de 1 pour $1 - \mathrm{ReLU}(1 - x)$ de 0 à 2). Le notebook s’arrête par `SystemExit` si un seuil n’est pas atteint. Figure : `figures/integrale-numerique.png`.

## Ce qu’il faut observer

- Diviser la largeur des bandes par 2 divise l’erreur par 2 pour les rectangles à gauche ou à droite, par 4 pour le milieu et les trapèzes : à précision égale, ces deux méthodes demandent bien moins de calculs.
- Les courbes « gauche » et « droite » sont presque confondues : leurs erreurs sont de même taille et de signes opposés, d’où les trapèzes, qui en font la moyenne.
- Pour $1 - \mathrm{ReLU}(1 - x)$, la pente vaut 0 en $x = 2$, et pourtant le gradient intégré retrouve que l’entrée a fait passer la sortie de 0 à 1 : il additionne les pentes sur tout le chemin.

## Questions de réflexion

1. Pourquoi la méthode du milieu donne-t-elle exactement 9 pour $x^2$ de 0 à 3, quel que soit le nombre de pas ?
2. Avec 4 pas, le gradient intégré de $1 - \mathrm{ReLU}(1 - x)$ vaut 0,5. Quels points du chemin ont été pris, et lequel tombe exactement sur l’angle ?
3. Les auteurs des gradients intégrés conseillent de 20 à 300 pas : à quelle erreur cela correspond-il ici ?

## Pour aller plus loin

- Ajouter la méthode de Simpson, $\dfrac{1}{3}\,\text{trapèzes} + \dfrac{2}{3}\,\text{milieu}$, et mesurer la pente de son erreur.
- Remplacer les rectangles à droite des gradients intégrés par les trapèzes ou le milieu.
