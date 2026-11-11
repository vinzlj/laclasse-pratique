# 19 · Méthode d’Euler

| | |
|---|---|
| **Chapitre lié** | [19. Équations différentielles et méthode d’Euler](https://laclasse.poneyponey.fr/maths/p2-analyse/19-equations-differentielles/), sections 4 (méthode d’Euler, pas et erreur) et 5 (modèle logistique) |
| **Commande** | `make nb NUM=19` (éditeur), `make nb NUM=19 MODE=run` (complet), `make nb NUM=19 MODE=smoke` (moins de 30 s) |
| **Durée mesurée** | run complet : 4 s, dont 0,4 s de calcul ; smoke : 2 s (20/10/2026) |
| **Travail estimé** | 20 min |
| **Graine** | 42 (`seed_everything(42)`, variable `SEED` du Makefile) ; aucun tirage au hasard |
| **Données** | aucune : les équations sont écrites dans le notebook, rien n’est téléchargé |

## Objectif

Simuler une équation différentielle par la méthode d’Euler, $y_{n+1} = y_n + h\,f(t_n, y_n)$, sur le café du cours, $T' = -0{,}1\,(T - 20)$, et sur le modèle logistique $y' = 0{,}8\,y\,(1 - y/8)$ ; mesurer l’erreur finale en fonction du pas $h$ ; montrer ce que devient le calcul quand $h$ dépasse $\dfrac{2}{|a|}$.

## Ce que vous allez faire

1. Lire la cellule de configuration : elle seule change entre mode complet et mode smoke.
2. Cliquer sur « Lancer le calcul » dans l’éditeur marimo.
3. Lire la figure : à gauche, l’erreur finale selon le pas $h$ (échelles logarithmiques) pour les deux équations ; à droite, les trajectoires d’Euler du café pour $h = 5$, $15$, $20$ et $25$ minutes, comparées à la solution exacte.

## Résultat attendu

En mode complet (de 12 à 98 304 pas pour le café, de 20 à 163 840 pour la logistique) : pente log-log de l’erreur **1 ± 0,1** pour les deux équations (mesuré : 0,998 et 1,000). Valeurs du cours retrouvées exactement : 21,09375 (café, $h = 5$, $t = 30$) ; 44,4075 (café, $h = 1$, $t = 10$) ; $1{,}1^{10} \approx 2{,}5937$ et $1{,}01^{100} \approx 2{,}7048$ ($y' = y$) ; 0,6875 (un pas de 0,5 sur la logistique). Stabilité du café : avec $h = 15$, l’oscillation s’amortit vers 20 °C ; avec $h = 20$, le calcul alterne 90 ; −50 sans fin ; avec $h = 25$, $|T|$ dépasse 1 000 en 12 pas (mesuré : 9 102). Le notebook s’arrête par `SystemExit` si un seuil n’est pas atteint. Figure : `figures/euler.png`.

## Ce qu’il faut observer

- Diviser le pas par 2 divise l’erreur finale par 2 : les points s’alignent sur une droite de pente 1. Pour gagner un chiffre de précision, il faut 10 fois plus de pas.
- Pour le café, le polygone d’Euler reste sous la vraie courbe : il garde sur tout un pas la vitesse du début, trop forte.
- La frontière $h = 20 = \dfrac{2}{0{,}1}$ ne dépend pas de la précision voulue : au-delà, chaque pas multiplie l’écart à 20 °C par un nombre plus grand que 1 en valeur absolue.

## Questions de réflexion

1. Pour $h = 10$, un seul pas amène le café exactement à 20 °C. Pourquoi ? Est-ce une bonne approximation de la vraie courbe ?
2. La courbe de la logistique est sous celle du café sur la figure de gauche : qu’est-ce qui rend l’erreur plus petite (indice : $|y''|$) ?
3. Quel pas faut-il pour que l’erreur du café en $t = 30$ passe sous 0,01 ?

## Pour aller plus loin

- Ajouter la méthode de Heun (moyenne des pentes au début et à la fin du pas) et mesurer la pente de son erreur : on doit trouver 2.
- Ajouter un bruit $\sigma\sqrt{h}\,Z_n$ à chaque pas (section 6) et tracer plusieurs trajectoires avec la graine 42.
