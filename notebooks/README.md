# Notebooks marimo du parcours de maths

Les exercices interactifs du site restent le cœur de la pratique. Les notebooks servent seulement là où le code éclaire la notion : vérifier une identité par sympy, simuler une loi, approcher une dérivée ou une intégrale, voir une transformation linéaire, faire une descente de gradient à la main. Il y en a 26 au plus, un par chapitre au plus, écrits avec leur chapitre.

## Organisation

Un dossier par notebook, numéroté comme son chapitre :

```
notebooks/maths/NN-slug/
  notebook.py     notebook marimo (fichier .py : diffs lisibles, exécutable comme un script)
  README.md       objectif, chapitre lié, commande, durée mesurée, résultat attendu chiffré,
                  ce qu’il faut observer, questions de réflexion, graine (42)
  figures/        figures produites (celles du mode smoke portent le suffixe -smoke)
```

## Commandes

```sh
make nb NUM=12              # ouvre le notebook 12 dans l’éditeur marimo
make nb NUM=12 MODE=run     # exécution complète comme un script
make nb NUM=12 MODE=smoke   # test rapide, moins de 30 s, sous le verrou des notebooks
make smoke-nb               # tous les notebooks en mode smoke, en série
```

Les modes `run` et `smoke` passent par `scripts/run_locked.sh` (verrou `/tmp/laclasse-nb.lock`, un notebook à la fois) pour que les durées mesurées ne soient pas faussées par un autre notebook.

## Conventions

- Graine 42 (variable `SEED` du Makefile), mode smoke détecté par l’option `--smoke`.
- Un bloc `config` est le seul endroit où le mode smoke réduit le travail.
- Les calculs longs attendent un bouton dans l’éditeur (`mo.stop(mo.running_in_notebook() and not bouton.value)`).
- Le bilan final lève `SystemExit` si un seuil chiffré n’est pas atteint.
- Pas de GPU, pas de cloud, pas de téléchargement : tout tourne en quelques secondes à quelques minutes sur un portable.

## Gabarit

Point de départ : `cp -r notebooks/maths/_gabarit notebooks/maths/NN-slug`, puis remplacer l’exemple (dérivée numérique de exp) et remplir le README. Le gabarit se lance comme les autres : `make nb NUM=_gabarit MODE=smoke` (il n’est pas compté par `make smoke-nb`, qui ne prend que les dossiers numérotés).

Structure imposée, dans cet ordre : imports ; énoncé (`mo.md`) ; configuration (`SMOKE = is_smoke()`, `seed_everything(42)`, dictionnaire `config`, seul endroit où le mode smoke réduit le travail) ; bouton puis calcul derrière `mo.stop(...)` ; vérification exacte par sympy ; figure sauvegardée dans `figures/` (suffixe `-smoke` en mode smoke) ; bilan qui affiche une ligne « Bilan : OK (…) » ou lève `SystemExit("Bilan : ÉCHEC : …")`.

Le gabarit importe `math_commun` (`is_smoke`, `seed_everything`, `style_commun`, `sauvegarder_figure`) et garde un repli local aux mêmes signatures tant que ce paquet n’existe pas. Hors du Makefile, `MATH_SMOKE=1` force le mode smoke (utile dans l’éditeur, qui ne transmet pas d’arguments). Vérifier la syntaxe : `uv run marimo check notebooks/maths/NN-slug/notebook.py`.
