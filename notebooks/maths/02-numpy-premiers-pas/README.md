# 02 · Premiers pas avec numpy

| | |
|---|---|
| **Chapitre lié** | [02. numpy pour les maths](https://laclasse.poneyponey.fr/maths/p0-outils/02-numpy/), sections 1, 3, 4, 5 et 7 |
| **Commande** | `make nb NUM=02` (éditeur), `make nb NUM=02 MODE=run` (complet), `MODE=smoke` (moins de 30 s) |
| **Durée mesurée** | run complet : 2 s ; smoke : 1 s (M1 Pro, 20/10/2026) |
| **Travail estimé** | 30 min |
| **Graine** | 42 (`np.random.default_rng(42)`, variable `SEED` du Makefile) |
| **Données** | aucune : tout est calculé ou tiré par le notebook, rien n’est téléchargé |

## Objectif

Pratiquer les quatre réflexes du chapitre : vectoriser un calcul au lieu d’écrire une boucle, fixer une graine pour rejouer le hasard, tracer des courbes avec matplotlib, et faire vérifier un résultat exact par sympy.

## Ce que vous allez faire

1. Lire la cellule de configuration : elle seule change entre mode complet et mode smoke.
2. Cliquer sur « Lancer le calcul » dans l’éditeur marimo.
3. Comparer les deux temps de la somme des carrés, puis les trois suites de tirages de dé.
4. Lire la figure : les mêmes coûts en échelle linéaire, puis logarithmique.

## Résultat attendu

En mode complet ($N = 2\,000\,000$) : la boucle et numpy donnent la même somme, 2 666 664 666 667 000 000, égale à la formule $\frac{(N-1)N(2N-1)}{6}$ vérifiée par sympy ; numpy est **au moins 5 fois** plus rapide (mesuré : 66 fois, 72 ms contre 1,1 ms). Deux générateurs de graine 42 tirent la même suite (1, 5, 4, 3, 3, 6, 1, 5, 2, 1…), la graine 43 une autre (4, 4, 3, 1, 4…). Le plus petit $n$ tel que $2^n > 1\,000\,n^2$ vaut **19**, par numpy et en entiers exacts. Le notebook s’arrête par `SystemExit` si l’une de ces conditions n’est pas remplie. Figure : `figures/couts.png`.

## Ce qu’il faut observer

- Le rapport des temps dépend de la machine, pas le résultat : la boucle fait un tour de Python par élément, numpy une seule instruction exécutée en code compilé.
- En échelle linéaire, $n$ et $n^2$ sont écrasés contre l’axe ; en échelle logarithmique, $2^n$ devient une droite et finit par croiser $1\,000\,n^2$ entre 18 et 19.
- Changer la graine change les tirages ; garder la graine les rejoue à l’identique, sur n’importe quelle machine qui a la même version de numpy.

## Questions de réflexion

1. Si l’on double $N$, par combien les deux temps sont-ils multipliés à peu près ? Pourquoi ?
2. Que devient le seuil si l’on remplace $1\,000\,n^2$ par $1\,000\,000\,n^2$ ? Vérifiez-le en changeant une ligne (on trouve 30) : mille fois plus de marge ne recule le seuil que de 11 crans.
3. Pourquoi `np.random.default_rng(42)` donne-t-il les mêmes tirages à chaque exécution, alors que l’on parle de hasard ?

## Pour aller plus loin

- Remplacer la somme des carrés par $\sum i^3$ et vérifier avec sympy que le résultat vaut $\left(\frac{N(N-1)}{2}\right)^2$.
