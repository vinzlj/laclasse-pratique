# 28 · La loi des grands nombres, simulée

| | |
|---|---|
| **Chapitre lié** | [28. Sommes de variables aléatoires, concentration, loi des grands nombres](https://laclasse.poneyponey.fr/maths/p4-probas/28-loi-grands-nombres/), sections 3 (moyenne d’un échantillon i.i.d.), 4 (Bienaymé-Tchebychev), 5 (loi faible des grands nombres) et 7 (Hoeffding) |
| **Commande** | `make nb NUM=28` (éditeur), `make nb NUM=28 MODE=run` (complet), `make nb NUM=28 MODE=smoke` (moins de 30 s) |
| **Durée mesurée** | run complet : 2 s, dont 0,2 s de calcul ; smoke : 2 s (machine de travail, 26/10/2026) |
| **Travail estimé** | 20 min |
| **Graine** | 42 (`seed_everything(42)`, variable `SEED` du Makefile) |
| **Données** | aucune : lancers de dés et de pièces pseudo-aléatoires, rien n’est téléchargé |

## Objectif

Voir que la moyenne de $n$ lancers de dé a un écart-type $\sigma/\sqrt{n}$, qu’une moyenne observée entre dans la bande $3{,}5 \pm 2\sigma/\sqrt{n}$ et s’y maintient, et que la fréquence des grands écarts d’une fréquence de Bernoulli reste sous les majorants de Bienaymé-Tchebychev et de Hoeffding.

## Ce que vous allez faire

1. Lire la cellule de configuration : elle seule change entre mode complet et mode smoke (nombre de répétitions et tailles).
2. Lire la vérification sympy : $\mu = \frac{7}{2}$, $\sigma^2 = \frac{35}{12}$ pour un dé, $\frac{35}{6}$ pour la somme de deux dés.
3. Cliquer sur « Lancer le calcul » dans l’éditeur marimo, puis lire les trois graphiques.

## Résultat attendu

- Écart-type mesuré des moyennes de $n$ dés, pour $n$ de 10 à 10 000 (4 000 répétitions par taille) : rapport à $\sigma/\sqrt{n}$ dans **1 ± 0,05** (mesuré : entre 0,980 et 1,014) ; pente log-log **−0,5 ± 0,03** (mesuré : −0,501). En smoke (400 répétitions, $n$ jusqu’à 3 000) : 1 ± 0,15 et −0,5 ± 0,06 (mesuré : 0,953 à 1,059 et −0,511).
- Trajectoire de 20 000 lancers : moyenne finale à **3,5 ± 0,05** (mesuré : 3,4853 en complet, 3,4809 en smoke) ; dans la bande $\pm 2\sigma/\sqrt{n}$ pour **au moins 75 %** des $n \ge 100$ (Bienaymé-Tchebychev garantit 3 sur 4 pour un $n$ donné ; mesuré : 100 % et 93,9 %).
- Pièce de paramètre $p = 0{,}3$, écart $\varepsilon = 0{,}05$, $n$ de 50 à 800 : la fréquence observée de $|F_n - p| \ge 0{,}05$ ne dépasse aucun des deux majorants (mesuré pour $n = 800$ : 0,0024, contre 0,105 pour Bienaymé-Tchebychev et 0,037 pour Hoeffding).

Le notebook s’arrête par `SystemExit` si un seuil n’est pas atteint. Figure : `figures/grands-nombres.png`.

## Ce qu’il faut observer

- Les points de l’écart-type mesuré s’alignent sur la droite $\sigma/\sqrt{n}$, de pente −1/2 en log-log : 100 fois plus de lancers pour une erreur 10 fois plus petite.
- Les deux majorants sont justes mais larges : la fréquence observée est souvent bien plus petite (40 fois pour $n = 800$). Bienaymé-Tchebychev, qui utilise la vraie variance $p(1 - p) = 0{,}21$, est meilleur pour $n$ petit ; Hoeffding, qui décroît exponentiellement, le dépasse vers $n \approx 500$.
- La trajectoire saute beaucoup au début, puis se colle à 3,5 : la loi des grands nombres porte sur la moyenne, les écarts passés sont dilués.

## Questions de réflexion

1. Combien de lancers faudrait-il pour que l’écart-type de la moyenne passe sous 0,001 ?
2. Pourquoi Hoeffding finit-il toujours par battre Bienaymé-Tchebychev quand $n$ grandit, et pourquoi pas pour $n$ petit ?
3. Remplacez le dé par la somme de deux dés : par combien l’écart-type de la moyenne est-il multiplié ?

## Pour aller plus loin

- Remplacer $p = 0{,}3$ par $p = 0{,}5$ : le majorant de Bienaymé-Tchebychev devient $\frac{1}{4n\varepsilon^2}$, le cas le plus défavorable.
