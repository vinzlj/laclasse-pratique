# 20 · Signaux

| | |
|---|---|
| **Chapitre lié** | [20. Trigonométrie : cercle, radians, fonctions sinus et cosinus](https://laclasse.poneyponey.fr/maths/p2-analyse/20-trigonometrie-cercle/), sections 3 (périodicité), 5 (fréquence, période, phase), 6 (formules d’addition), 7 (encodage de position), 8 (petits angles) et 9 (créneau) |
| **Commande** | `make nb NUM=20` (éditeur), `make nb NUM=20 MODE=run` (complet), `make nb NUM=20 MODE=smoke` (moins de 30 s) |
| **Durée mesurée** | run complet : 3 s, dont 1 s de calcul ; smoke : 3 s (20/10/2026) |
| **Travail estimé** | 25 min |
| **Graine** | 42 (`seed_everything(42)`, variable `SEED` du Makefile) : elle fixe le bruit du signal de l’expérience 2 |
| **Données** | aucune : les signaux sont calculés dans le notebook, rien n’est téléchargé |

## Objectif

Manipuler des signaux sinusoïdaux en numpy : période d’une somme de sinusoïdes, amplitude et phase retrouvées grâce à la formule d’addition, encodage de position sinusoïdal d’un transformer (carte $p \times i$, périodes en progression géométrique, décalage égal à une rotation), erreur de $\sin \theta \approx \theta$, et approximation d’un créneau par des sommes de sinusoïdes.

## Ce que vous allez faire

1. Lire la cellule de configuration : elle seule change entre mode complet et mode smoke (finesse de la grille, dimension $d$ de l’encodage, nombre de positions et de termes).
2. Cliquer sur « Lancer le calcul » dans l’éditeur marimo.
3. Lire la figure : la somme $\sin 2t + \sin 3t$ et sa période ; le signal bruité et son ajustement ; la carte de l’encodage de position (en abscisse la position, en ordonnée la composante) ; l’écart entre les sommes de sinusoïdes et le créneau.

## Résultat attendu

- Période de $\sin 2t + \sin 3t$ trouvée à **$2\pi$ près de $0{,}002$** ; pour $T = \pi$, l’écart maximal $|s(t + \pi) - s(t)|$ vaut 2 (mesuré : 6,281 et 2,000).
- Signal $1{,}5 \sin(4\pi t + \pi/3)$ avec un bruit d’écart-type $0{,}1$ : amplitude et phase retrouvées **à $0{,}05$ près** (mesuré : $A = 1{,}497$, $\varphi = 1{,}045$ pour $1{,}047$).
- Encodage de position : périodes en progression géométrique de premier terme $2\pi$ et de raison $10\,000^{2/d}$ (mesuré pour $d = 512$ : raison $1{,}036\,63$, dernière période **60 611** positions) ; décaler de $k = 7$ positions revient à tourner chaque paire, à **$10^{-9}$ près** (mesuré : $2 \times 10^{-13}$).
- Erreur relative de $\sin \theta \approx \theta$ : **0,167 %** pour 0,1 rad, **0,51 %** pour 10°, **4,72 %** pour 30°.
- Créneau : l’écart maximal sur le palier diminue avec le nombre de termes (**≤ 0,02** avec 50 termes, mesuré 0,016) ; près du saut, le maximum reste vers **0,926** au lieu de $\pi/4 \approx 0{,}785$.

Le notebook s’arrête par `SystemExit` si un seuil n’est pas atteint. Figure : `figures/signaux.png`.

## Ce qu’il faut observer

- $\sin 2t$ revient au bout de $\pi$, mais pas $\sin 3t$ : la somme n’a pour période que le plus petit multiple commun des deux, $2\pi$.
- Sur la carte de l’encodage, les premières composantes oscillent très vite d’une position à l’autre, les dernières presque pas : ce sont les aiguilles rapides et lentes de l’horloge.
- Ajouter des termes au créneau améliore les paliers, mais pas le dépassement près des sauts (le phénomène de Gibbs) : il se resserre sans diminuer.

## Questions de réflexion

1. Pourquoi l’ajustement par moindres carrés sur $\sin(\omega t)$ et $\cos(\omega t)$ suffit-il à retrouver une phase, alors que $\varphi$ apparaît à l’intérieur du sinus ?
2. Avec $d = 64$ (mode smoke), la raison de la progression vaut $1{,}333\,5$ au lieu de $1{,}036\,6$, et la dernière période $47\,117$ positions au lieu de $60\,611$. Pourquoi la première période vaut-elle $2\pi$ dans les deux cas, et pourquoi la dernière reste-t-elle sous $2\pi \times 10\,000$ ?
3. Que devient la période de $\sin 2t + \sin(\sqrt{2}\,t)$ ? Essayez de la chercher avec la même méthode.

## Pour aller plus loin

- Remplacer l’encodage additif par RoPE : faire tourner deux vecteurs aux positions $m$ et $n$, et vérifier que leur produit $x_1 y_1 + x_2 y_2$ ne dépend que de $n - m$.
- Approcher un signal en dents de scie par $\sin t - \dfrac{\sin 2t}{2} + \dfrac{\sin 3t}{3} - \cdots$.
