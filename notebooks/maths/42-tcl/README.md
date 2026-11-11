# 42 · Le théorème central limite, mesuré

| | |
|---|---|
| **Chapitre lié** | [42. Lois continues, loi normale, théorème central limite](https://laclasse.poneyponey.fr/maths/p7-probas-ia/42-loi-normale-tcl/), sections 5 (loi normale, Φ et ses quantiles) et 6 (théorème central limite, retour sur l’intervalle à 1,96) |
| **Commande** | `make nb NUM=42` (éditeur), `make nb NUM=42 MODE=run` (complet), `make nb NUM=42 MODE=smoke` (moins de 30 s) |
| **Durée mesurée** | run complet : 2 s, dont 0,1 s de calcul ; smoke : 1 à 2 s (machine de travail, 27/10/2026) |
| **Travail estimé** | 20 min |
| **Graine** | 42 (`seed_everything(42)`, variable `SEED` du Makefile) |
| **Données** | aucune : tirages pseudo-aléatoires et probabilités binomiales exactes, rien n’est téléchargé |

## Objectif

Voir la cloche apparaître : la moyenne centrée réduite de $n$ tirages d’une loi très dissymétrique (exponentielle, dé, Bernoulli de paramètre 0,05) a une fonction de répartition de plus en plus proche de $\Phi$ quand $n$ passe de 1 à 100 ; mesurer la couverture réelle de $\hat{p} \pm 1{,}96\,\text{SE}$ selon $n$ et $p$ ; calculer $\Phi$ et ses quantiles.

## Ce que vous allez faire

1. Lire la cellule de configuration : elle seule change entre mode complet et mode smoke (nombre de moyennes tirées).
2. Lire la vérification sympy : $\varphi$ a une aire 1, une espérance 0 et une variance 1 ; l’exponentielle de paramètre $\lambda$ a pour espérance $1/\lambda$ et pour variance $1/\lambda^2$.
3. Cliquer sur « Lancer le calcul » dans l’éditeur marimo, puis lire les cinq graphiques.

## Résultat attendu

- Écart maximal $D_n$ entre la fonction de répartition observée de $Z_n$ et $\Phi$ (sur $[-3 \,;\, 3]$, 40 000 moyennes par cas) : pour l’exponentielle, de **0,159** ($n = 1$) à **0,017** ($n = 100$) ; pour le dé, de 0,145 à 0,013. Seuils : $D_{100} \le 0{,}03$ (0,05 en smoke, 4 000 moyennes) et $D_{100} \le D_1/3$. Pour la Bernoulli rare, $D_{100}$ vaut encore **0,116** : la loi est discrète et $np = 5$ seulement (seuil : au moins 0,08, et division par 3).
- Couverture exacte de $\hat{p} \pm 1{,}96\,\text{SE}$ : **0,946** pour $p = 0{,}5$ et $n = 1\,000$ (seuil : 0,95 ± 0,01), mais **0,332** pour $p = 0{,}02$ et $n = 20$ (seuil : moins de 0,8) ; 0,950 pour $p = 0{,}02$ et $n = 1\,000$.
- Règle 68-95-99,7 : 0,6827 ; 0,9545 ; 0,9973 ; quantiles d’ordre 0,95 ; 0,975 ; 0,995 : 1,6449 ; 1,9600 ; 2,5758.

Le notebook s’arrête par `SystemExit` si un seuil n’est pas atteint. Figure : `figures/tcl.png`.

## Ce qu’il faut observer

- Pour $n = 1$, l’histogramme de l’exponentielle est collé à gauche avec une longue traîne ; dès $n = 30$, il épouse la cloche, alors que rien de normal n’a été tiré.
- La couverture de l’intervalle à 1,96 n’est « environ 95 % » que si $n$ est assez grand **et** $p$ pas trop proche de 0 : pour $p = 0{,}02$ et $n = 20$, on observe le plus souvent 0 succès, et l’intervalle se réduit au point 0.
- Le TCL ne dit pas à partir de quel $n$ la cloche est là : cela dépend de la loi de départ.

## Questions de réflexion

1. Pourquoi l’intervalle $\hat{p} \pm 1{,}96\,\text{SE}$ ne contient-il jamais $p$ quand on observe 0 succès ?
2. Pour la Bernoulli rare, à partir de quel $n$ environ peut-on espérer $np$ et $n(1 - p)$ supérieurs à 10 ?
3. Remplacez l’exponentielle par la loi de densité $1/x^2$ sur $[1 \,;\, +\infty[$ (section 3) : que devrait-il se passer, et pourquoi le TCL ne s’applique-t-il pas ?

## Pour aller plus loin

- Ajouter une loi uniforme : sa moyenne de 5 tirages est déjà très proche de la cloche (démo TCL du cours).
