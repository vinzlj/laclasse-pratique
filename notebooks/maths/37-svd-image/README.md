# 37 · Compresser une image par la SVD, et l’ACP par la SVD

| | |
|---|---|
| **Chapitre lié** | [37. SVD, approximation de rang faible, pseudo-inverse, ACP](https://laclasse.poneyponey.fr/maths/p5-algebre-lineaire/37-svd-acp/), sections 3 (approximation de rang faible, Eckart et Young) et 6 (l’ACP par la SVD des données centrées) |
| **Commande** | `make nb NUM=37` (éditeur), `make nb NUM=37 MODE=run` (complet), `make nb NUM=37 MODE=smoke` (moins de 30 s) |
| **Durée mesurée** | run complet : 2 s, dont 0,1 s de calcul ; smoke : 1 à 2 s (machine de travail, 27/10/2026) |
| **Travail estimé** | 30 min |
| **Graine** | 42 (`seed_everything(42)`, variable `SEED` du Makefile) |
| **Données** | aucune : l’image et les « chiffres » $8 \times 8$ sont générés par le notebook, rien n’est téléchargé (pas de scikit-learn) |

## Objectif

Calculer la SVD d’une image en niveaux de gris avec `np.linalg.svd`, la reconstruire au rang $k$, mesurer la part de l’énergie gardée et l’erreur, et vérifier le théorème d’Eckart et Young ($\lVert A - A_k \rVert_F = \sqrt{\sigma_{k+1}^2 + \dots}$, $\lVert A - A_k \rVert_2 = \sigma_{k+1}$, et aucune autre approximation de rang $k$ ne fait mieux) ; puis faire l’ACP d’un jeu de chiffres $8 \times 8$ par la SVD des données centrées, et vérifier qu’elle donne les mêmes axes et les mêmes variances $\lambda_j = \sigma_j^2 / n$ que la matrice de covariance $C = \frac{1}{n}X_c^\top X_c$.

## Ce que vous allez faire

1. Lire la cellule de configuration : elle seule change entre mode complet et mode smoke (taille de l’image, nombre de tirages au hasard, nombre de chiffres).
2. Lire la reconstruction `(U[:, :k] * S[:k]) @ Vh[:k]` : `S` est le vecteur des valeurs singulières, `Vh` est $V^\top$, et multiplier les colonnes de `U` par `S` revient à calculer $U_k\Sigma_k$.
3. Cliquer sur « Lancer le calcul » dans l’éditeur marimo, puis lire les résultats imprimés et la figure.

## Résultat attendu

- Image $256 \times 256$ (mode complet) : au rang 20, $10\,260$ nombres stockés ($15{,}7\,\%$ de l’image) gardent **au moins 99 %** de l’énergie $\sum_{i \le k} \sigma_i^2 / \sum \sigma_i^2$ (mesuré : $99{,}44\,\%$, erreur relative de Frobenius $7{,}5\,\%$). L’énergie croît avec $k$.
- Eckart et Young : les erreurs mesurées en norme de Frobenius et en norme spectrale égalent $\sqrt{\sigma_{k+1}^2 + \dots}$ et $\sigma_{k+1}$ à **10⁻⁸ près** (mesuré : 2 × 10⁻¹⁴).
- Au rang 10, aucune des 200 approximations obtenues en projetant sur 10 directions tirées au hasard ne bat la SVD, ni aucune des 200 obtenues à partir de directions **proches** de $U_{10}$ ; les directions au hasard font au moins **2 fois** pire (mesuré : 9,6 fois).
- ACP de 2 000 chiffres $8 \times 8$ générés (64 variables) : $\sigma_j^2 / n$ et les valeurs propres de $C$ coïncident à **10⁻¹⁰ près** en écart relatif, les cinq premiers axes aussi (au signe près) ; les parts de variance sont les mêmes avec $\frac{1}{n}$ ou $\frac{1}{n - 1}$ ; deux axes gardent moins de 90 % de la variance (mesuré : $45{,}6\,\%$) et il en faut entre 3 et 60 pour 90 % (mesuré : 19).
- sympy refait exactement la SVD du cours : $A = \begin{pmatrix} 2 & 2 \\ -1 & 2 \end{pmatrix}$, $A^\top A = \begin{pmatrix} 5 & 2 \\ 2 & 8 \end{pmatrix}$, valeurs propres 9 et 4, $A = U\Sigma V^\top$ avec $\sigma = 3$ et $2$, $\lVert A - A_1 \rVert_F = 2$, aucune valeur propre réelle (discriminant $-8$).

Le notebook s’arrête par `SystemExit` si un seuil n’est pas atteint. Figure : `figures/svd-image.png`.

## Ce qu’il faut observer

- Au rang 1, l’image est un produit « colonne fois ligne » : des bandes horizontales et verticales, rien d’oblique. Les bandes obliques demandent beaucoup de calques, le disque un peu moins.
- Le premier calque porte déjà plus de 90 % de l’énergie : c’est surtout la luminosité moyenne de l’image. L’erreur relative, elle, baisse lentement ; c’est elle qui se voit.
- Les valeurs singulières chutent vite, puis forment un long plateau : c’est le bruit, que les petits rangs laissent de côté.

## Questions de réflexion

1. Pour une image $256 \times 256$, à partir de quel rang la SVD tronquée occupe-t-elle plus de place que l’image ?
2. Pourquoi des directions proches de $U_{10}$, mais pas exactement égales, donnent-elles déjà une erreur bien plus grande que la SVD ?
3. Que deviendraient les axes de l’ACP si l’on oubliait de centrer les données ?

## Pour aller plus loin

- Remplacer l’image générée par une photo en niveaux de gris du domaine public, chargée avec `matplotlib.pyplot.imread`, et comparer la vitesse de décroissance des valeurs singulières.
- Retirer la moyenne de l’image avant la SVD, et comparer l’énergie gardée au rang 1.
