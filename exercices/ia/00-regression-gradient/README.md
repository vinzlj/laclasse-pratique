# Exercice 00 : régression linéaire, descente de gradient, Ridge et validation croisée

| | |
|---|---|
| **Chapitre lié** | [00. Apprendre pour une machine : les fondations](https://laclasse.poneyponey.fr/ia/p00-outils/00-fondations/) |
| **Exécution** | locale, CPU (NumPy et scikit-learn ; aucun GPU nécessaire) |
| **Commande** | `make ex NUM=00` (éditeur marimo), `make ex NUM=00 MODE=run` (complet), `make ex NUM=00 MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 5,6 à 8 s au total (quatre mesures), dont 3,6 à 3,8 s de calcul (le reste : démarrage de Python, import de torch par `ia_commun`) ; smoke : 5 à 7 s (M1 Pro, 14 et 15/10/2026) |
| **Travail estimé** | 45 min à 1 h |
| **Seed** | 42 (`seed_everything(42)` et `np.random.default_rng(42)`) |
| **Données** | générées par le notebook (droite bruitée, sinusoïde bruitée) ; rien n'est téléchargé |

## Objectif

Voir sur un exemple minuscule que « apprendre » consiste à minimiser une perte (ici la MSE) : la résoudre exactement, la minimiser par descente de gradient, constater le surapprentissage, le corriger par une pénalité (Ridge, Lasso, arrêt précoce) et régler cette pénalité par validation croisée.

## Ce que vous allez faire

**Partie A : la droite** (40 points : poids d'une page web en Mo → temps de chargement en s, vraie relation y = 0,8 + 1,5 x, bruit d'écart-type 0,3 s).

1. Écrire la MSE et son gradient $\frac{2}{N}X^\top(X\theta - y)$ ; vérifier le gradient par différences finies.
2. Résoudre les équations normales $(X^\top X)\theta = X^\top y$ (formule fermée).
3. Calculer la hessienne $H = \frac{2}{N}X^\top X$ et le pas critique $\eta_c = 2/\lambda_{\max}(H)$.
4. Lancer la descente de gradient depuis $(b, w) = (2, 0)$ avec $\eta$ = 0,05, 0,3, 0,42 et 0,45 ; comparer à la formule fermée.
5. Vérifier avec `sklearn.linear_model.LinearRegression`.

**Partie B : la courbe** (20 points d'entraînement, 20 de validation, 1 000 de test, tirés de $y = \sin(2\pi x)$ + bruit d'écart-type 0,3).

6. Ajuster par moindres carrés (maximum de vraisemblance) des polynômes de degré 0 à 15 ; comparer erreurs d'entraînement et de test.
7. Au degré 12, comparer le maximum de vraisemblance (λ = 0) à Ridge (formule fermée, $(A^\top A + \lambda P)\theta = A^\top y$) sur une grille de λ de 10⁻⁸ à 10² ; relier λ à l'a priori gaussien du MAP ($\lambda = \sigma^2/\tau^2$).
8. Lasso (scikit-learn) : compter les coefficients exactement nuls.
9. Arrêt précoce : descente de gradient SANS pénalité au degré 12, en surveillant l'erreur de validation (mesurée à 300 instants, espacés logarithmiquement).

**Partie C : validation croisée.**

10. Programmer une validation croisée à 5 plis pour choisir λ ; refaire le même calcul avec `cross_val_score` sur les mêmes plis.
11. Réentraîner avec le λ choisi et mesurer une seule fois la MSE et le R² sur le jeu de test.

Figures produites : `figures/descente.png` (trajectoires et courbes de perte), `figures/regularisation.png` (quatre panneaux de la partie B et C).

## Résultat attendu

Le notebook vérifie ces seuils en mode complet et s'arrête avec une erreur si l'un manque. En mode smoke (moins d'itérations), un seuil manqué s'affiche « smoke : non vérifié » et n'est pas une erreur.

| Contrôle | Seuil | Obtenu (14/10/2026) |
|---|---|---|
| Gradient analytique contre différences finies | écart < 10⁻⁶ | 2,7 × 10⁻¹⁰ |
| Formule fermée | — | b = 0,8980, w = 1,4119, MSE = 0,0442 |
| Pas critique | — | λ(H) = 0,281 et 4,650 (conditionnement 16,6) ; η_c = 0,430 |
| η = 0,05 et 0,3 : écart à la formule fermée | < 10⁻⁶ | atteint en 1 000 et 161 itérations (η = 0,42 : 263 itérations) |
| η = 0,45 (> η_c) | divergence | MSE > 10⁸ après 110 itérations |
| scikit-learn contre formule fermée | écart < 10⁻⁸ | 1,1 × 10⁻¹⁵ ; R² = 0,936 |
| Ridge, λ choisi par validation croisée | MSE test ≤ 0,16 | λ = 5,6 × 10⁻³, MSE test 0,141 |
| Maximum de vraisemblance au degré 12 | MSE test ≥ 10 × celle de Ridge | MSE test 39,0 (277 fois plus) |
| R² sur le test | ≥ 0,70 | 0,765 |
| Validation croisée à la main contre scikit-learn | écart relatif < 10⁻⁶ | 6,9 × 10⁻⁹ |
| Arrêt précoce | MSE test au meilleur moment < MSE test finale | 0,136 (itération 4 182, meilleure validation parmi 300 instants de mesure espacés logarithmiquement) contre 0,216 (itération 500 000) |

Autres valeurs affichées : erreur de test minimale au degré 3 (0,131) parmi les polynômes sans pénalité ; Lasso α = 10⁻² : 9 coefficients nuls sur 12 (MSE test 0,108). La MSE minimale atteignable est la variance du bruit, 0,09.

## Ce qu'il faut observer

- **Partie A.** Les trois pas inférieurs à η_c retrouvent la formule fermée à 10⁻⁶ près ; η = 0,45, à peine 5 % au-dessus de η_c, diverge. η = 0,42 est plus lent que 0,3 bien qu'il soit plus grand : il oscille d'un bord à l'autre de la vallée (facteur |1 − η λ_max| = 0,95 par pas dans la direction raide). η = 0,05 est lent dans l'autre direction, la moins courbée (facteur 1 − η λ_min = 0,986).
- **Partie B.** L'erreur d'entraînement baisse toujours avec le degré ; l'erreur de test, elle, remonte (0,13 au degré 3, 39 au degré 12, plus de 10⁵ au degré 15). Les coefficients du polynôme de degré 12 non pénalisé atteignent plusieurs milliers : c'est la signature du surapprentissage. Ridge les ramène à quelques unités.
- **Arrêt précoce.** Sans aucune pénalité, arrêter la descente tôt donne un modèle presque aussi bon que Ridge : la descente apprend d'abord les directions bien déterminées par les données, et les oscillations parasites en dernier.
- **Partie C.** Les deux implémentations de la validation croisée donnent les mêmes nombres. La courbe de validation croisée a un minimum net ; le jeu de test n'a servi qu'une fois, à la fin.

## Questions de réflexion

1. Pourquoi η = 0,42 converge-t-il moins vite que η = 0,3 alors qu'il fait de plus grands pas ? Que se passerait-il si on centrait les données x (moyenne nulle) avant d'apprendre ? (Indice : regardez le conditionnement de H.)
2. La MSE d'entraînement du polynôme de degré 12 sans pénalité est plus faible que celle de Ridge. Pourquoi est-ce un mauvais critère pour choisir entre les deux ?
3. Avec un bruit d'écart-type σ = 0,3 et un a priori $w_j \sim \mathcal{N}(0, \tau^2)$, quel τ correspond au λ choisi par validation croisée ? Est-ce un a priori « raisonnable » pour des coefficients qui valent quelques unités ?
4. Si l'on choisissait λ en regardant directement l'erreur de test, que vaudrait encore ce chiffre comme estimation de la performance future ?

## Pour aller plus loin

- Remplacer le départ (2, 0) par d'autres points dans la démo `GradientDescent` du chapitre : le pas critique ne dépend pas du départ.
- Refaire la partie B avec 200 points d'entraînement au lieu de 20 : le surapprentissage du degré 12 disparaît-il ?
- Notes du cours CS229 (Stanford), partie sur la régression linéaire et la régularisation : https://cs229.stanford.edu/main_notes.pdf
