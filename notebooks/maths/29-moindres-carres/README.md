# 29 · Moindres carrés, corrélation, point aberrant

| | |
|---|---|
| **Chapitre lié** | [29. Statistique à deux variables : écart-type, centrer-réduire, moindres carrés, corrélation](https://laclasse.poneyponey.fr/maths/p4-probas/29-statistique-deux-variables/), sections 5 (droite des moindres carrés), 6 (covariance et corrélation), 7 (coefficient de détermination) et 8 (ajustement par changement de variable) |
| **Commande** | `make nb NUM=29` (éditeur), `make nb NUM=29 MODE=run` (complet), `make nb NUM=29 MODE=smoke` (moins de 30 s) |
| **Durée mesurée** | run complet : 2 s, dont moins de 0,1 s de calcul ; smoke : 2 s (machine de travail, 26/10/2026) |
| **Travail estimé** | 20 min |
| **Graine** | 42 (`seed_everything(42)`, variable `SEED` du Makefile) |
| **Données** | les cinq pages et le nuage exponentiel du cours (écrits dans le notebook), un nuage simulé ; rien n’est téléchargé |

## Objectif

Voir que les formules du cours, $a = \frac{\mathrm{cov}(x, y)}{V(x)}$ et $b = \bar{y} - a\,\bar{x}$, donnent la même droite que `np.polyfit`, que les résidus ont une somme nulle et que $R^2 = r^2$, qu’un seul point mal placé peut faire tomber $r$ près de 0, et qu’un nuage exponentiel s’ajuste par une droite sur $\ln y$.

## Ce que vous allez faire

1. Lire la vérification sympy : sur les cinq pages du cours, $G(3 \,;\, 2{,}5)$, $\mathrm{cov}(x, y) = \frac{34}{25}$, $V(x) = 2$, donc $a = \frac{17}{25} = 0{,}68$ et $b = \frac{23}{50} = 0{,}46$, somme des carrés des résidus $\frac{9}{250}$, et $R^2 = r^2 = \frac{1156}{1165}$ exactement.
2. Lire la cellule de configuration : elle seule change entre mode complet et mode smoke (taille du nuage simulé, pas de la grille du point aberrant).
3. Cliquer sur « Lancer le calcul » dans l’éditeur marimo, puis lire les trois graphiques.

## Résultat attendu

- Cinq pages : $a = 0{,}68$, $b = 0{,}46$ à $10^{-12}$ près, par les formules et par `np.polyfit` ; $r \approx 0{,}9961$, $R^2 \approx 0{,}9923$.
- Nuage simulé autour de $y = 0{,}7x + 0{,}5$ (bruit d’écart-type 0,4) : formules et `np.polyfit` d’accord à $10^{-9}$ près ; pente **0,7 ± 0,02** en complet, 2 000 points (mesuré : 0,7012, $r \approx 0{,}981$), **0,7 ± 0,1** en smoke, 50 points (mesuré : 0,7048) ; somme des résidus nulle et $|R^2 - r^2| \le 10^{-12}$.
- Point aberrant : le nuage de six points de la démo a $r \approx 0{,}989$ ; avec $M(0 \,;\, 9)$, $r \approx 0{,}030$. Sur la grille $[0 \,;\, 10]^2$ au pas de 0,5, 17 positions de $M$ donnent $|r| \le 0{,}1$ (6 au pas de 1 en smoke).
- Nuage exponentiel : $\ln y \approx 0{,}6993\,x + 4{,}7950$, soit $y \approx 120{,}9 \times 2{,}012^x$, avec $R^2 \approx 0{,}9995$ sur $\ln y$ ; la droite ajustée directement sur $y$ a $R^2 \approx 0{,}80$ et vaut environ $-504$ au mois 0.

Le notebook s’arrête par `SystemExit` si un seuil n’est pas atteint. Figure : `figures/moindres-carres.png`.

## Ce qu’il faut observer

- Les segments pointillés du premier graphique sont les résidus : certains au-dessus de la droite, d’autres en dessous, de somme nulle.
- Sur la carte de $|r|$, la zone sombre (r proche de 0) est loin du nuage, dans les coins haut gauche et bas droit : un point qui contredit la tendance pèse d’autant plus qu’il est loin du point moyen.
- Sur le dernier graphique, la droite ajustée sur $y$ passe sous zéro au mois 0 et sous-estime la fin : un $R^2$ de 0,80 ne dit pas que le modèle est le bon.

## Questions de réflexion

1. Pourquoi la droite des moindres carrés passe-t-elle toujours par le point moyen ? Vérifiez-le sur le nuage simulé.
2. Où faut-il placer $M$ pour que $r$ devienne négatif ? Pour qu’il reste au-dessus de 0,99 ?
3. La droite ajustée sur $\ln y$ minimise des écarts sur $\ln y$, pas sur $y$. Qu’est-ce que cela change pour les grandes valeurs ?

## Pour aller plus loin

- Remplacer le bruit gaussien par un bruit dont l’écart-type grandit avec $x$, et regarder les résidus.
- Ajuster une loi puissance $y = c\,x^k$ par une droite sur $(\ln x \,;\, \ln y)$.
