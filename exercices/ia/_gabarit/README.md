# Exercice _gabarit : classer deux demi-lunes avec un petit MLP

> Gabarit à copier (`cp -r exercices/ia/_gabarit exercices/ia/NN-nom-court`), puis à remplir rubrique par rubrique. Toutes les rubriques sont obligatoires.

| | |
|---|---|
| **Chapitre lié** | [NN. Titre du chapitre](https://laclasse.poneyponey.fr/ia/pXX-partie/NN-slug/) |
| **Exécution** | locale (MPS) |
| **Commande** | `make ex NUM=_gabarit` (éditeur), `MODE=run` (complet), `MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 5 s dont 0,9 s d'entraînement ; smoke : 4 à 8 s (M1 Pro, MPS, 14/10/2026) |
| **Travail estimé** | 10 min |
| **Seed** | 42 (`seed_everything(42)`) |
| **Données** | générées (`sklearn.datasets.make_moons`), rien n'est téléchargé |

## Objectif

Montrer la structure attendue d'un exercice : configuration, données, modèle, entraînement derrière un bouton, figure, bilan chiffré.

## Ce que vous allez faire

1. Lire la cellule de configuration : elle seule change entre mode complet et mode smoke.
2. Cliquer sur « Entraîner » dans l'éditeur marimo.
3. Observer la courbe de loss et la frontière de décision.

## Résultat attendu

Précision sur l'ensemble d'entraînement **≥ 95 %** en mode complet. Le notebook s'arrête avec une erreur si ce seuil n'est pas atteint. Figure produite : `figures/frontiere.png`.

## Ce qu'il faut observer

- La loss chute vite pendant les premières époques, puis plafonne.
- La frontière de décision est courbe : un modèle linéaire n'y arriverait pas.

## Questions de réflexion

1. Que se passe-t-il si vous retirez la couche cachée (modèle linéaire) ?
2. Pourquoi la précision mesurée sur les données d'entraînement surestime-t-elle la performance réelle ?

## Pour aller plus loin

- Ajouter un ensemble de validation et tracer les deux courbes.
