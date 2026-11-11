# Notebook _gabarit : approcher une dérivée par un taux d’accroissement

> Gabarit à copier (`cp -r notebooks/maths/_gabarit notebooks/maths/NN-slug`), puis à remplir rubrique par rubrique. Toutes les rubriques sont obligatoires ; conventions dans `notebooks/README.md`.

| | |
|---|---|
| **Chapitre lié** | [12. Dérivation](https://laclasse.poneyponey.fr/maths/p2-analyse/12-derivation/) |
| **Commande** | `make nb NUM=_gabarit` (éditeur), `MODE=run` (complet), `MODE=smoke` (moins de 30 s) |
| **Durée mesurée** | run complet et smoke : 2 à 4 s, dont moins de 0,1 s de calcul (M1 Pro, 18/10/2026) |
| **Travail estimé** | 15 min |
| **Graine** | 42 (`seed_everything(42)`, variable `SEED` du Makefile) |
| **Données** | aucune : points tirés au hasard, rien n’est téléchargé |

## Objectif

Voir que le taux d’accroissement approche le nombre dérivé, que la différence centrée est bien plus précise que la différence avant, et que les flottants limitent la précision quand $h$ devient trop petit.

## Ce que vous allez faire

1. Lire la cellule de configuration : elle seule change entre mode complet et mode smoke.
2. Cliquer sur « Lancer le calcul » dans l’éditeur marimo.
3. Lire la figure : erreur relative en fonction du pas $h$, en échelle logarithmique.

## Résultat attendu

En mode complet (2 000 points, 60 pas) : pente log-log de l’erreur **1 ± 0,1** pour la différence avant, **2 ± 0,1** pour la différence centrée (mesuré : 1,01 et 2,00), meilleure erreur centrée **≤ 10⁻⁹** (mesuré : 2,4 × 10⁻¹¹ pour $h \approx 8 \times 10^{-6}$). Le notebook s’arrête par `SystemExit` si un seuil n’est pas atteint. Figure : `figures/erreur-pas.png`.

## Ce qu’il faut observer

- Pour $h$ grand, les deux droites ont des pentes 1 et 2 : diviser $h$ par 10 divise l’erreur par 10 (avant) ou par 100 (centrée). sympy le confirme : erreurs $\frac{h}{2}e^x$ et $\frac{h^2}{6}e^x$.
- Pour $h$ très petit, l’erreur remonte : la soustraction $f(x + h) - f(x)$ perd des chiffres significatifs (chapitre 48).

## Questions de réflexion

1. Pourquoi la meilleure valeur de $h$ est-elle plus grande pour la différence centrée que pour la différence avant ?
2. Que deviendraient les pentes pour $f(x) = x^2$ ? Et pour $f(x) = x$ ?

## Pour aller plus loin

- Remplacer `exp` par `sin` et vérifier que les pentes ne changent pas.
