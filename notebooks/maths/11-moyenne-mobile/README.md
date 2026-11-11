# 11 · Moyenne mobile exponentielle

| | |
|---|---|
| **Chapitre lié** | [11. Suites](https://laclasse.poneyponey.fr/maths/p2-analyse/11-suites/), section 6 (suites arithmético-géométriques, moyenne mobile exponentielle) |
| **Commande** | `make nb NUM=11` (éditeur), `make nb NUM=11 MODE=run` (complet), `make nb NUM=11 MODE=smoke` (moins de 30 s) |
| **Durée mesurée** | run complet : 2 s, dont 0,2 s de calcul ; smoke : 2 s (20/10/2026) |
| **Travail estimé** | 20 min |
| **Graine** | 42 (`seed_everything(42)`, variable `SEED` du Makefile) |
| **Données** | aucune : mesures bruitées tirées au hasard autour de 200 ms, rien n’est téléchargé |

## Objectif

Lisser un signal bruité par la moyenne mobile exponentielle $m_t = \beta m_{t-1} + (1 - \beta) x_t$, partie de $m_0 = 0$, pour $\beta = 0{,}5$, $0{,}9$ et $0{,}99$ ; vérifier que, sur un signal constant $c$, la boucle donne exactement $c(1 - \beta^t)$ et que la correction de biais $m_t / (1 - \beta^t)$ redonne $c$ ; mesurer la demi-vie (premier $t$ tel que $\beta^t \le 0{,}5$) dans un tableau.

## Ce que vous allez faire

1. Lire la cellule de configuration : elle seule change entre mode complet et mode smoke.
2. Cliquer sur « Lancer le calcul » dans l’éditeur marimo.
3. Lire la figure : les mesures (points gris) et leurs trois moyennes mobiles.

## Résultat attendu

- Signal constant : écart entre la boucle et la formule fermée $c(1 - \beta^t)$ **≤ 10⁻⁹** (mesuré : 2 × 10⁻¹³), et moyenne corrigée égale au niveau à 10⁻⁹ près. sympy le vérifie exactement pour $t = 1$ à $7$.
- Signal bruité ($\beta = 0{,}9$, 10 premiers pas, 2 000 tirages en mode complet) : l’erreur moyenne avec correction doit être **au moins trois fois plus petite** que sans (mesuré : 117 ms sans, 16 ms avec).
- Demi-vies : **1** pas pour $\beta = 0{,}5$, **7** pour $0{,}9$, **69** pour $0{,}99$, **173** pour $0{,}996$ (le coefficient du réseau cible du chapitre IA26).

Le notebook s’arrête par `SystemExit` si un seuil n’est pas atteint. Figure : `figures/moyenne-mobile.png`.

## Ce qu’il faut observer

- Avec $\beta = 0{,}99$, la courbe est la plus lisse, mais elle met plus de 200 pas à rejoindre 200 ms : c’est le défaut de départ, en $1 - 0{,}99^t$.
- Avec $\beta = 0{,}5$, la courbe suit presque chaque mesure : elle lisse à peine.
- La correction de biais ne sert qu’au début : quand $t$ grandit, $\beta^t$ tend vers 0 et $1 - \beta^t$ vers 1.

## Questions de réflexion

1. Pourquoi la demi-vie de $\beta = 0{,}99$ est-elle environ dix fois celle de $\beta = 0{,}9$ ?
2. Adam utilise $\beta_2 = 0{,}999$ pour la moyenne des carrés des gradients : que vaudrait la moyenne non corrigée au premier pas, comparée au bon niveau ?
3. Le signal saute brusquement de 200 à 300 ms : combien de pas faut-il, avec $\beta = 0{,}9$, pour que la moyenne dépasse 250 ms ?

## Pour aller plus loin

- Remplacer le signal constant par un échelon (200 ms puis 300 ms à mi-parcours) et mesurer le retard de chaque courbe.
- Programmer la moyenne mobile d’Adam sur les carrés des mesures ($\beta_2 = 0{,}999$) et tracer sa version corrigée.
