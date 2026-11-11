# 48 · Flottants en numpy : formats, absorption, softmax stable, quantification, gradient

| | |
|---|---|
| **Chapitre lié** | [48. Calcul numérique : flottants, quantification, stabilité, différences finies](https://laclasse.poneyponey.fr/maths/p8-numerique/48-calcul-numerique/), sections 1 à 7 |
| **Commande** | `make nb NUM=48` (éditeur), `make nb NUM=48 MODE=run` (complet), `make nb NUM=48 MODE=smoke` (moins de 30 s) |
| **Durée mesurée** | run complet : 2 s, dont 0,13 s de calcul ; smoke : 2 s (machine de travail, 27/10/2026) |
| **Travail estimé** | 30 min |
| **Graine** | 42 (`seed_everything(42)`, variable `SEED` du Makefile) ; elle fixe la matrice de poids et les données de la perte logistique |
| **Données** | aucune téléchargée : poids et exemples tirés au hasard dans la cellule de calcul |

## Objectif

Voir de ses yeux, dans numpy, ce que le cours calcule à la main : les constantes des formats float16, bfloat16 et float32, l’absorption, la softmax qui renvoie NaN et sa version stable, l’erreur d’une quantification int8 par groupes, et la vérification d’un gradient par différence centrée. Le bfloat16 est simulé en arrondissant les 16 bits de poids faible d’un float32, sans bibliothèque supplémentaire.

## Ce que vous allez faire

1. Lire la cellule de configuration : nombre de termes de la longue somme ($10^5$ en smoke, $10^7$ en complet), taille de la matrice de poids (64 ou 1 024 lignes de 256), taille des groupes (32), nombre d’exemples et de paramètres de la perte, pas $h = 10^{-5}$.
2. Cliquer sur « Lancer le calcul ».
3. Lire, dans l’ordre : le tableau des constantes, le compteur bloqué, la longue somme, la softmax, la quantification et la vérification de gradient.

## Résultat attendu

- Constantes : float16 plus grand $65\,504$, $\varepsilon = 2^{-10}$ ; float32 $\varepsilon = 2^{-23}$ ; $0{,}1$ devient $0{,}0999755859375$ en float16 et $0{,}10009765625$ en bfloat16.
- Un compteur float16 qui ajoute 1 reste **bloqué à 2 048** ; $257$ devient $256$ en bfloat16.
- La somme de $10^7$ termes $0{,}1$ accumulée en float32 dérive de plus de $10^{-4}$ en valeur relative (mesuré : $8{,}8 \times 10^{-2}$ en complet, $1{,}4 \times 10^{-4}$ en smoke) ; en float64, moins de $10^{-9}$.
- La softmax naïve de $(1\,000 \,;\, 999 \,;\, 998)$ renvoie NaN ; la version stable donne $(0{,}6652 \,;\, 0{,}2447 \,;\, 0{,}0900)$ et $\mathrm{LSE} \approx 1\,000{,}4076$.
- Quantification int8 : l’erreur ne dépasse jamais $\frac{s}{2}$ (rapport mesuré au plus 1) ; avec un poids isolé 25 fois plus grand, une seule échelle pour toute la matrice annule environ 8 % des poids, une échelle par groupe de 32 moins de 1 %.
- Vérification de gradient en float64 : erreur relative sous $10^{-7}$ pour le gradient juste (mesuré : $6{,}9 \times 10^{-9}$), au-dessus de $10^{-2}$ pour le gradient faux d’un facteur 2 (mesuré : $0{,}5$) ; en float32, même le gradient juste dépasse $10^{-6}$ (mesuré : $8{,}8 \times 10^{-2}$).
- sympy vérifie les erreurs exactes des différences avant et centrée pour $x^3$, l’invariance $\mathrm{LSE}(\mathbf{z} - c) = \mathrm{LSE}(\mathbf{z}) - c$ et les plus grands finis $65\,504$, $57\,344$ et $448$.

Le notebook s’arrête par `SystemExit` si un seuil n’est pas atteint. Figure : `figures/flottants.png`.

## Ce qu’il faut observer

- Le compteur float16 ne plante pas : il continue de tourner, faux, sans aucun message.
- La dérive de la longue somme en float32 grandit avec le nombre de termes : chaque ajout perd une fraction de $0{,}1$ quand la somme devient grande devant lui.
- Un seul grand poids suffit à écraser les petits quand toute la matrice partage une échelle.
- En float32, la vérification de gradient « échoue » sur un code juste : il faut la faire en float64.

## Questions de réflexion

1. À partir de quelle valeur de la somme, en float32, l’ajout de $0{,}1$ commence-t-il à être absorbé complètement ? (Cherchez quand l’écart entre voisins dépasse $0{,}2$.)
2. Pourquoi la version stable de la softmax donne-t-elle exactement la même loi pour $(1\,000 \,;\, 999 \,;\, 998)$ et pour $(2 \,;\, 1 \,;\, 0)$ ?
3. Que devient la vérification de gradient en float64 avec $h = 10^{-12}$ ? Essayez, et expliquez avec la section 7 du cours.

## Pour aller plus loin

- Accumuler la longue somme par paquets (somme de sommes partielles) en float32 : la dérive diminue-t-elle ?
- Quantifier sur 4 bits au lieu de 8 et comparer l’erreur maximale à $\frac{s}{2}$, avec $q = 7$.
