# 33 · Transformations du plan en numpy : colonnes, composition, déterminant

| | |
|---|---|
| **Chapitre lié** | [33. Matrices comme transformations du plan, déterminant](https://laclasse.poneyponey.fr/maths/p5-algebre-lineaire/33-transformations-determinant/), sections 1 (la grille), 2 (colonnes), 3 (transformations usuelles), 4 (composition), 5 (déterminant) et 7 (RoPE) |
| **Commande** | `make nb NUM=33` (éditeur), `make nb NUM=33 MODE=run` (complet), `make nb NUM=33 MODE=smoke` (moins de 30 s) |
| **Durée mesurée** | run complet : 2 s, dont 0,2 s de calcul ; smoke : 2 s (machine de travail, 26/10/2026) |
| **Travail estimé** | 25 min |
| **Graine** | 42 (`seed_everything(42)`, variable `SEED` du Makefile) |
| **Données** | aucune : matrices, polygones et vecteurs tirés au hasard, rien n’est téléchargé |

## Objectif

Voir en numpy les quatre idées du chapitre : les colonnes d’une matrice sont les images des vecteurs de base, composer deux transformations revient à multiplier leurs matrices (et l’ordre compte), la valeur absolue du déterminant est le facteur d’aire, et les rotations par blocs de RoPE donnent un score qui ne dépend que de l’écart entre les positions.

## Ce que vous allez faire

1. Lire la cellule de configuration : elle seule change entre mode complet et mode smoke (nombre d’essais, dimension des vecteurs de RoPE, nombre de couples de positions).
2. Lire l’expérience 1 : `A @ e1` et `A @ e2` sont les colonnes de A ; une maison d’aire 1,3 et ses images par une rotation, un cisaillement et une symétrie.
3. Cliquer sur « Lancer le calcul » dans l’éditeur marimo : ordre de composition, aire de polygones tirés au hasard, déterminant nul, RoPE.

## Résultat attendu

- Avec $A$ de lignes $(2, -1)$ et $(1, 1)$ : $A\mathbf{e}_1 = (2 \,;\, 1)$, $A\mathbf{e}_2 = (-1 \,;\, 1)$, et $A(3 \,;\, 2) = (4 \,;\, 5)$, comme dans le cours.
- Aire orientée de la maison : 1,3 ; image par la rotation de $\pi/6$ et par le cisaillement : 1,3 ; par la symétrie d’axe $(Ox)$ : $-1{,}3$ (orientation renversée).
- Rotation d’un quart de tour $A$ puis symétrie d’axe $(Ox)$ $B$ : $\mathbf{e}_1$ va en $(0 \,;\, -1)$ ; $BA$ a pour lignes $(0, -1)$ et $(-1, 0)$, $AB$ a pour lignes $(0, 1)$ et $(1, 0)$.
- Aire de l’image d’un polygone (formule du lacet) égale à $\det A$ fois l’aire de départ, à $10^{-9}$ près en relatif (mesuré : $2 \times 10^{-15}$ sur 5 000 essais, 200 en smoke).
- Matrice de lignes $(1, 2)$ et $(2, 4)$ : déterminant 0, toutes les images sur la droite $y = 2x$ ; $(2 \,;\, 0)$ et $(0 \,;\, 1)$ ont la même image $(2 \,;\, 4)$.
- RoPE avec $\theta_i = 10\,000^{-2(i-1)/d}$ : décaler les deux positions d’autant change le score de moins de $10^{-8}$ (mesuré : $2 \times 10^{-12}$ pour $d = 64$) ; la norme est conservée ; pour $d = 128$, la paire la plus lente a une période d’environ 54 410 positions.
- sympy vérifie $\det(AB) = \det A \det B$ en $2 \times 2$, $R_a R_b = R_{a+b}$, $R_\theta^\top = R_{-\theta}$, $\det R_\theta = 1$, la projection sur la droite dirigée par $(2 \,;\, 1)$ et le déterminant triangulaire du cours (3).

Le notebook s’arrête par `SystemExit` si un seuil n’est pas atteint. Figure : `figures/transformations.png` (grille et maison transformées, colonnes en flèches, déterminant en titre).

## Ce qu’il faut observer

- Dans numpy, on range souvent les points **en lignes** ; l’image de tous les points s’écrit alors `P @ A.T`, et non `A @ P`.
- Le calcul flottant de $\cos(\pi/2)$ donne $6{,}1 \times 10^{-17}$ au lieu de 0 : c’est le même nombre que le navigateur affiche parfois dans `matrix(…)`.
- Le signe de l’aire orientée suit celui du déterminant : la symétrie retourne la maison.

## Questions de réflexion

1. Pourquoi `P @ A.T` donne-t-il les images des points rangés en lignes dans `P` ? (Pensez à $(A\mathbf{p})^\top = \mathbf{p}^\top A^\top$.)
2. Le cisaillement change toutes les distances sauf sur l’axe $(Ox)$ ; pourquoi l’aire ne change-t-elle pas ?
3. Dans l’expérience 5, que se passe-t-il si l’on fait tourner la requête de $m\theta$ et la clé de $-n\theta$ ?

## Pour aller plus loin

- Écrire la matrice de la symétrie d’axe la droite d’angle $\alpha$ et vérifier numériquement que deux symétries d’axes d’angles $\alpha$ et $\beta$ donnent la rotation d’angle $2(\beta - \alpha)$.
- Tirer des matrices $3 \times 3$ et comparer `np.linalg.det` au volume du parallélépipède construit sur leurs colonnes.
