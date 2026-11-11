# 16 · Softmax et température

| | |
|---|---|
| **Chapitre lié** | [16. Les fonctions de l’IA](https://laclasse.poneyponey.fr/maths/p2-analyse/16-fonctions-ia/), section 5 (softmax, invariance par translation, température) |
| **Commande** | `make nb NUM=16` (éditeur), `make nb NUM=16 MODE=run` (complet), `make nb NUM=16 MODE=smoke` (moins de 30 s) |
| **Durée mesurée** | run complet : 2 s, dont 0,03 s de calcul ; smoke : 2 s (M1 Pro, 18/10/2026) |
| **Travail estimé** | 20 min |
| **Graine** | 42 (`seed_everything(42)`, variable `SEED` du Makefile) |
| **Données** | aucune : les scores sont fixés, les tirages sont pseudo-aléatoires, rien n’est téléchargé |

## Objectif

Voir pourquoi on calcule la softmax en retranchant d’abord le plus grand score, comment un modèle de langage tire le mot suivant selon la température $T$, et comment l’étalement de la loi (son entropie) grandit avec $T$.

## Ce que vous allez faire

1. Lire la cellule de configuration : elle seule change entre mode complet et mode smoke (nombre de tirages et de points).
2. Lire la première expérience : la formule directe appliquée aux scores $(1000 ; 999 ; 998)$ donne `nan`, la version stable donne les probabilités de $(2 ; 1 ; 0)$.
3. Cliquer sur « Lancer le calcul » dans l’éditeur marimo : tirages pour $T = 0{,}25$ ; $0{,}5$ ; $1$ ; $2$ ; $4$, puis courbe de l’entropie.

## Résultat attendu

- Softmax naïve de $(1000 ; 999 ; 998)$ : `[nan nan nan]` ; stable : $(0{,}6652 ; 0{,}2447 ; 0{,}0900)$, identique (à $10^{-12}$ près) à la softmax de $(2 ; 1 ; 0)$.
- Probabilité de « canapé » pour $T = 0{,}5$ : $0{,}8668$ (à $10^{-4}$ près), comme dans le cours.
- Écart maximal entre fréquences observées et probabilités : **≤ 0,005** en mode complet, 400 000 tirages par température (mesuré : 0,0011) ; **≤ 0,02** en smoke, 20 000 tirages (mesuré : 0,0053).
- Entropie croissante en $T$, de moins de 0,01 bit pour $T = 0{,}05$ à $\log_2 3 \approx 1{,}585$ bit à 0,01 près pour $T = 50$ (mesuré : 0,0000 et 1,5848).
- sympy vérifie $\operatorname{softmax}(z + c) = \operatorname{softmax}(z)$, les limites de $p_0$ quand $T \to 0$ et $T \to +\infty$, et $\operatorname{softmax}(z_1, z_2)_1 = \sigma(z_1 - z_2)$.

Le notebook s’arrête par `SystemExit` si un seuil n’est pas atteint. Figure : `figures/softmax-temperature.png`.

## Ce qu’il faut observer

- `nan` ne vient pas d’une erreur de formule : $\mathrm{e}^{1000}$ dépasse le plus grand flottant, devient `inf`, et `inf / inf` n’a pas de valeur (chapitre 48). Retrancher le maximum ne change pas le résultat mathématique (section 5 du cours).
- Avec $T = 0{,}25$, « canapé » sort 98 fois sur 100 : presque toujours le même mot. Avec $T = 4$, « toit » sort une fois sur quatre.
- L’entropie passe de 0 à son maximum en S quand on lit $T$ en échelle logarithmique : l’essentiel du changement a lieu entre $T = 0{,}2$ et $T = 2$.

## Questions de réflexion

1. Pourquoi la version stable retranche-t-elle le **plus grand** score, et pas le plus petit ?
2. Avec $T = 0{,}25$, combien de fois sur 400 000 tirages s’attend-on à voir « toit » ? Comparez avec la fréquence affichée.
3. Une API de chat accepte `temperature: 0`. Que peut-elle faire, puisque diviser par 0 est impossible ?

## Pour aller plus loin

- Ajouter un quatrième mot de score $-3$ : que devient l’entropie maximale ?
- Ajouter un filtrage top-k ($k = 2$) avant le tirage et renormaliser : retrouver $\sigma(1) \approx 0{,}731$ pour « canapé » à $T = 1$.
