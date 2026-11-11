# 39 · Une différentiation automatique maison

| | |
|---|---|
| **Chapitre lié** | [39. Jacobienne, règle de la chaîne multivariée, graphe de calcul, différentiation automatique](https://laclasse.poneyponey.fr/maths/p6-multivariee/39-jacobienne-autodiff/), sections 3, 4, 6, 7 et 9 |
| **Commande** | `make nb NUM=39` (éditeur), `make nb NUM=39 MODE=run` (complet), `make nb NUM=39 MODE=smoke` (moins de 30 s) |
| **Durée mesurée** | run complet : 3 s, dont 0,8 s de calcul ; smoke : 2 s (machine de travail, 27/10/2026) |
| **Travail estimé** | 30 min |
| **Graine** | 42 (`seed_everything(42)`, variable `SEED` du Makefile) : réseaux tirés pour la vérification et poids de départ de l’entraînement |
| **Données** | aucune téléchargée : les quatre points du « ou exclusif », écrits dans le notebook |

## Objectif

Écrire en quelques dizaines de lignes une différentiation automatique en mode inverse, comme `loss.backward()` de PyTorch : une classe `Valeur` qui enregistre le graphe de calcul pendant la passe avant (addition, soustraction, multiplication, puissance, sigmoïde, $-\ln$), et une méthode `backward()` qui le parcourt à rebours après un tri topologique, en additionnant les contributions des enfants. La vérifier sur les exemples du cours et par différences centrées, puis s’en servir pour entraîner le réseau 2-2-1 du cours.

## Ce que vous allez faire

1. Lire la classe `Valeur` : chaque opération retient ses parents et ses dérivées locales (la somme recopie, le produit envoie le facteur échangé, la sigmoïde $s(1 - s)$, $-\ln a$ donne $-1/a$).
2. Comparer ses résultats aux calculs à la main du cours : ouverture, variable qui sert deux fois, neurone, réseau 2-2-1.
3. Cliquer sur « Lancer le calcul » : vérification par différences centrées sur des réseaux 2-2-1 tirés au hasard (5 en smoke, 50 en complet), puis entraînement du réseau sur le « ou exclusif » (1 500 pas de descente en smoke, 6 000 en complet, taux 2).

## Résultat attendu

- Exemples du cours, exacts : $L = 9$, $\bar{w} = -18$, $\bar{b} = -6$ (ouverture) ; $L = 18$, $\bar{w} = -22$ (pénalité) ; neurone : $L \approx 0{,}3133$, $\bar{w} \approx -0{,}5379$, $\bar{b} \approx -0{,}2689$.
- Réseau 2-2-1 du cours : $L \approx 0{,}5782$, $\partial L / \partial W \approx \begin{pmatrix} -0{,}1032 & 0 \\ 0{,}1032 & 0 \end{pmatrix}$, $\partial L / \partial \mathbf{v} \approx (-0{,}2733 \,;\, -0{,}1658)$, identiques (à $10^{-12}$ près) aux formules de la section 6, $\boldsymbol{\delta}_1 \mathbf{x}^\top$ et $\delta_2 \mathbf{h}$.
- Différences centrées ($h = 10^{-5}$) : erreur relative maximale **au plus $10^{-6}$** (mesuré : $1{,}9 \times 10^{-8}$ en smoke, $2{,}1 \times 10^{-7}$ en complet).
- « Ou exclusif » : les 4 points bien classés, perte moyenne finale **sous 0,05** en complet (mesuré : 0,0015) et sous 0,2 en smoke (mesuré : 0,004).
- sympy vérifie $\sigma' = \sigma(1 - \sigma)$, $\frac{dL}{dz} = \sigma(z) - y$ pour la log-loss et $\partial(-\ln p_c)/\partial \mathbf{z} = \mathbf{p} - \mathbf{y}$ pour trois classes.

Le notebook s’arrête par `SystemExit` si un seuil n’est pas atteint. Figure : `figures/perte-xor.png` (perte moyenne au fil des pas, échelle logarithmique).

## Ce qu’il faut observer

- `backward()` ne connaît aucune formule globale : il ne fait que multiplier des dérivées locales et additionner. C’est pourtant assez pour obtenir les gradients exacts de n’importe quelle expression écrite avec ces opérations.
- Le `+=` de `p.grad += v.grad * d` est essentiel : un nœud qui sert deux fois (le poids $w$ de l’exemple avec pénalité) reçoit deux contributions. Remplacez-le par `=` : l’exemple avec pénalité ne garde plus qu’une des deux contributions, $-24$ (par $u$) ou $+2$ (par la pénalité), au lieu de leur somme $-22$.
- La perte du « ou exclusif » baisse d’abord lentement, entre 0,7 et 0,4, puis chute brutalement vers le 250ᵉ pas (run complet) : les deux neurones cachés doivent se spécialiser avant que la sortie puisse séparer les points.

## Questions de réflexion

1. Pourquoi faut-il un tri topologique avant de remonter ? Que se passerait-il si l’on traitait un nœud avant tous ses enfants ?
2. Combien de passes avant faudrait-il pour obtenir les 9 dérivées par différences centrées ? Et par `backward()` ?
3. Ajoutez à `Valeur` une méthode `relu()`. Quelle dérivée locale lui donner en 0 ?

## Pour aller plus loin

- Ajouter `exp()` et `tanh()`, puis entraîner le même réseau avec des cachés tanh.
- Comparer avec la vidéo d’A. Karpathy sur micrograd, citée dans « Pour aller plus loin » du chapitre.
