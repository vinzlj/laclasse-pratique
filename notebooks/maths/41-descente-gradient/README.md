# 41 · Descente de gradient : le pas, le bruit, l’élan

| | |
|---|---|
| **Chapitre lié** | [41. Optimisation : descente de gradient, SGD, conditionnement, flots, contraintes](https://laclasse.poneyponey.fr/maths/p6-multivariee/41-optimisation/), sections 2 (un pas), 3 (pas critique), 4 (conditionnement), 5 (SGD) et 6 (momentum) |
| **Commande** | `make nb NUM=41` (éditeur), `make nb NUM=41 MODE=run` (complet), `make nb NUM=41 MODE=smoke` (moins de 30 s) |
| **Durée mesurée** | run complet : 9 s, dont 0,5 s de calcul ; smoke : 6 s (machine de travail, 27/10/2026) |
| **Travail estimé** | 20 min |
| **Graine** | 42 (`seed_everything(42)`, variable `SEED` du Makefile) |
| **Données** | aucune : la perte de l’ouverture est écrite dans le code, les 2 000 points du mini-lot sont tirés au hasard ; rien n’est téléchargé |

## Objectif

Retrouver par le calcul trois résultats du chapitre 41 : sur la perte de l’ouverture, le nombre de pas dépend du pas η, et tout bascule au pas critique $2/\lambda_{\max} \approx 0{,}1803$ ; l’écart-type du gradient de mini-lot vaut $\sigma/\sqrt{B}$ ; le momentum traverse une vallée allongée en beaucoup moins de pas que la descente simple.

## Ce que vous allez faire

1. Lire la cellule de configuration : elle seule change entre mode complet et mode smoke (nombre de mini-lots tirés pour mesurer le bruit).
2. Lire la vérification sympy : le minimum $(3/2 \,;\, 1/3)$, la perte minimale $1/18$, la hessienne $\begin{pmatrix} 28/3 & 4 \\ 4 & 2 \end{pmatrix}$ et $\lambda_{\max} = \frac{17 + \sqrt{265}}{3}$.
3. Cliquer sur « Lancer le calcul » dans l’éditeur marimo, puis lire les deux graphiques.

## Résultat attendu

Seuils vérifiés par le bilan, en run complet comme en smoke :

- Perte de l’ouverture, départ $(0 \,;\, 0)$, arrivée à 0,001 du minimum : **99** pas pour η = 0,05, **49** pour 0,1, **41** pour 0,17, **1 441** pour 0,18 ; pour 0,19, la perte dépasse 1 000 au **22ᵉ** pas. Pas critique **0,1803**.
- Bruit du mini-lot, au point $(0 \,;\, 0)$, pour $B = 1, 4, 16, 64$ : l’écart-type mesuré divisé par $\sigma/\sqrt{B}$ vaut 1 à 0,02 près (0,08 en smoke) ; quadrupler $B$ divise l’écart-type par 2,00 ; la moyenne des gradients de mini-lot est le gradient complet (biais relatif sous la même tolérance).
- Vallée $\frac{1}{2}(x^2 + 16y^2)$, départ $(-4 \,;\, 1)$, distance au minimum divisée par 1 000 : **56** pas avec le meilleur pas fixe $2/17$ (facteur $15/17 \approx 0{,}882$), **19** pas avec le momentum $\eta = 0{,}16$, $\mu = 0{,}36$ (facteur asymptotique 0,6 ; la règle $\lceil \ln 1000 / \ln(1/0{,}6) \rceil$ en annonce 14).

Le notebook s’arrête par `SystemExit` si un seuil n’est pas atteint. Figure : `figures/descente-gradient.png`.

## Ce qu’il faut observer

- À η = 0,18, la perte baisse vite au début puis presque plus : la direction raide rebondit avec le facteur −0,997 ; à 0,19, elle repart vers le haut.
- Sur le graphique log-log, les points du bruit mesuré tombent sur une droite de pente −1/2.
- Le momentum fait mieux que le meilleur pas fixe, mais pas autant que sa règle du facteur ne l’annonce : le facteur 0,6 est double (racine double du polynôme du cours), ce qui ajoute un facteur proportionnel à k au début.

## Questions de réflexion

1. Centrez les poids (modèle $t \approx a(p - 2) + b$) : que devient la hessienne, et combien de pas faut-il alors avec le meilleur pas fixe ?
2. Recommencez la mesure du bruit au minimum de la perte des 2 000 points : le gradient complet y est nul ; le bruit du mini-lot l’est-il aussi ?
3. Avec μ = 0,9 et η = 0,01 sur la vallée, la descente converge-t-elle plus vite qu’avec μ = 0,36 et η = 0,16 ? Pourquoi ?
