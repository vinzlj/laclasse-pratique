# 12 · Dérivée numérique

| | |
|---|---|
| **Chapitre lié** | [12. Dérivation](https://laclasse.poneyponey.fr/maths/p2-analyse/12-derivation/), sections 2 (le nombre dérivé, une limite) et 8 (point anguleux, valeur absolue et ReLU) |
| **Commande** | `make nb NUM=12` (éditeur), `make nb NUM=12 MODE=run` (complet), `make nb NUM=12 MODE=smoke` (moins de 30 s) |
| **Durée mesurée** | run complet : 2 s, dont 0,01 s de calcul ; smoke : 1 à 2 s (M1 Pro, 18/10/2026) |
| **Travail estimé** | 20 min |
| **Graine** | 42 (`seed_everything(42)`, variable `SEED` du Makefile) |
| **Données** | aucune : abscisses tirées au hasard, rien n’est téléchargé |

## Objectif

Approcher un nombre dérivé par un taux de variation : voir que la différence centrée est bien plus précise que la différence avant, que les flottants limitent la précision quand $h$ devient trop petit, et qu’en un point anguleux (|x| et ReLU en 0) les taux à gauche et à droite ne se rejoignent jamais.

## Ce que vous allez faire

1. Lire la cellule de configuration : elle seule change entre mode complet et mode smoke.
2. Cliquer sur « Lancer le calcul » dans l’éditeur marimo.
3. Lire la figure : à gauche, l’erreur commise sur la dérivée de sin selon le pas $h$ (échelles logarithmiques) ; à droite, les taux de |x| en 0.

## Résultat attendu

En mode complet (5 000 abscisses, 80 pas entre 10⁻¹³ et 10⁻¹) : pente log-log de l’erreur **1 ± 0,1** pour la différence avant, **2 ± 0,1** pour la différence centrée (mesuré : 1,00 et 2,00) ; meilleure erreur centrée **≤ 10⁻⁹** (mesuré : 1,9 × 10⁻¹¹ pour $h \approx 8 \times 10^{-6}$). En 0, pour tous les pas : |x| a des taux 1 à droite, −1 à gauche, 0 centré ; ReLU a 1 à droite, 0 à gauche, 0,5 centré. Le notebook s’arrête par `SystemExit` si un seuil n’est pas atteint. Figure : `figures/erreur-pas.png`.

## Ce qu’il faut observer

- Pour $h$ assez grand, diviser $h$ par 10 divise l’erreur par 10 (avant) ou par 100 (centrée). sympy le confirme : erreurs $-\frac{h}{2}\sin x$ et $-\frac{h^2}{6}\cos x$.
- Pour $h$ très petit, l’erreur remonte : la soustraction $f(a + h) - f(a)$ de deux nombres presque égaux perd des chiffres significatifs (chapitre 48).
- En 0, la différence centrée de |x| vaut 0 : elle « invente » une pente moyenne, alors que la fonction n’est pas dérivable. C’est un sous-gradient possible, pas une dérivée.

## Questions de réflexion

1. Pourquoi la meilleure valeur de $h$ est-elle plus grande pour la différence centrée que pour la différence avant ?
2. Pour $f(x) = x^2$, que vaut exactement l’erreur de la différence avant ? Et celle de la différence centrée ?
3. Un réseau qui utilise ReLU reçoit l’entrée 0 exactement : quelle « dérivée » devrait-il utiliser ? Que fait le tableau des activations du chapitre IA07 ?

## Pour aller plus loin

- Remplacer `sin` par `exp` et vérifier que les pentes ne changent pas.
- Ajouter la fonction $x \mapsto x\,|x|$ : ses taux en 0 tendent-ils vers une même valeur ?
