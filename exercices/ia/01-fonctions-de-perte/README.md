# Exercice 01 : même données, autre loss, autre modèle

| | |
|---|---|
| **Chapitre lié** | [01. Fonctions de perte](https://laclasse.poneyponey.fr/ia/p00-outils/01-fonctions-de-perte/) |
| **Exécution** | locale, CPU (numpy, scikit-learn ; PyTorch seulement pour comparer des valeurs) |
| **Commande** | `make ex NUM=01` (éditeur), `make ex NUM=01 MODE=run` (complet), `MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 3,4 s au total (`time make ex NUM=01 MODE=run`), dont 1,6 s de calcul ; smoke : 4 s (M1 Pro, CPU, 14/10/2026, attente du verrou exclue) |
| **Travail estimé** | ~40 min |
| **Seed** | 42 (`seed_everything(42)`, `np.random.default_rng(42)` pour chaque jeu de données) |
| **Données** | générées dans le notebook (salaires fictifs, droite bruitée, deux nuages gaussiens) ; rien n'est téléchargé |

## Objectif

Constater qu'avec les mêmes données et le même modèle, changer la fonction de perte change le modèle appris, et l'expliquer par la **dérivée** de chaque perte : c'est elle qui fixe le poids de chaque point dans le gradient.

## Ce que vous allez faire

1. **Modèle constant.** Chercher la constante qui résume au mieux 11 salaires (dont un de 25 000 €) au sens de la MSE, de la MAE et de Huber, par simple balayage.
2. **Régression avec points aberrants.** 100 points `y = 2x + 1 + bruit` dont les 10 de plus grand `x` sont faussés de −25. Ajuster une droite par descente de gradient écrite en numpy avec trois pertes (MSE, MAE, Huber δ = 1), puis refaire avec scikit-learn (`LinearRegression`, `QuantileRegressor`, `HuberRegressor`). Comparer la loss d'entraînement et une métrique sur des données propres.
3. **Classification.** Deux nuages gaussiens, puis les mêmes avec 20 points de classe +1 placés très loin, du bon côté. Entraîner un classifieur linéaire avec trois pertes : moindres carrés sur les étiquettes ±1, log-loss (régression logistique), hinge (SVM linéaire). Comparer avec scikit-learn (`LogisticRegression`, `SVC(kernel="linear")`).
4. **Softmax, température, entropie croisée, KL.** Calculer la softmax d'un vecteur de logits à trois températures, vérifier par différences finies que le gradient de l'entropie croisée vaut `p − y`, constater que la KL n'est pas symétrique, et retrouver les valeurs de PyTorch (`mse_loss`, `l1_loss`, `huber_loss`, `binary_cross_entropy_with_logits`, `cross_entropy`, `kl_div`).

## Résultat attendu

En mode complet, le notebook vérifie dix seuils et s'arrête avec une erreur si l'un d'eux n'est pas atteint. En mode smoke (données réduites), les seuils ne sont pas vérifiés : une ligne « [non vérifié en smoke] » n'est pas une erreur. Valeurs obtenues le 14/10/2026 (seed 42) :

| Vérification | Seuil | Obtenu |
|---|---|---|
| Constante MSE = moyenne | à 1 € près | 4 627 € (moyenne 4 627 €) |
| Constante MAE = médiane | exacte | 2 600 € (médiane 2 600 €) ; Huber (δ = 1 000 €) : 2 690 € |
| Pente MSE faussée | écart à 2 **≥ 0,5** | 0,589 (ordonnée 5,35) |
| Pente MAE juste | écart à 2 **≤ 0,15** | 1,923 (ordonnée 1,11) |
| Pente Huber juste | écart à 2 **≤ 0,15** | 1,918 (ordonnée 1,17) |
| numpy ≈ scikit-learn (MSE, MAE) | pente à 0,02 près | 0,589 / 0,589 ; 1,923 / 1,923 (Huber : 1,918 / 1,914) |
| Moindres carrés et points lointains | perte d'exactitude **≥ 5 points** | 98,0 % → 88,8 % |
| Log-loss et hinge et points lointains | perte **≤ 1 point** | 98,0 % → 98,0 % ; 98,0 % → 97,9 % |
| Gradient de l'entropie croisée = `p − y` | écart ≤ 1e-6 | 8,6e-11 |
| numpy = PyTorch (6 pertes) | écart ≤ 1e-6 | 2,2e-16 |

Autres valeurs affichées : erreur absolue moyenne sur 1 000 points de test **propres** : 4,13 (modèle MSE) contre 0,84 (MAE et Huber) ; force moyenne des 20 points lointains, `−ℓ′(m)` avec son signe : −1,55 pour les moindres carrés (négative : elle les ramène vers la frontière, donc incline celle-ci) contre +1,5e-12 (log-loss) et 0 (hinge) ; 18 points sur 220 ont une marge inférieure à 1 avec la hinge (18 vecteurs de support pour `SVC`). Softmax des logits (2 ; 1 ; 0,1 ; −1) : entropie 0,468 nat à T = 0,5, 0,961 à T = 1, 1,249 à T = 2 ; entropie croisée pour la classe 0 : 0,449 ; KL(P‖uniforme) = 0,426 et KL(uniforme‖P) = 0,538.

Figures produites : `figures/regression-aberrants.png`, `figures/classification-pertes.png`.

## Ce qu'il faut observer

- **La moyenne contre la médiane.** Un seul salaire extrême fait passer la meilleure constante MSE de ~2 600 € à 4 627 €. La MAE n'est pas touchée : sa dérivée vaut ±1 quel que soit l'écart, chaque point « vote » avec le même poids.
- **Dix points sur cent suffisent à coucher la droite MSE** (pente 0,59 au lieu de 2). Sa dérivée `2r` donne à chaque point un poids proportionnel à son résidu : autour de la vraie droite, un aberrant (résidu ≈ 25) pèse environ vingt-cinq fois plus qu'un point normal (résidu ≈ 1). Huber, quadratique près de zéro mais à dérivée bornée par δ, se comporte comme la MAE face aux aberrants.
- **Loss contre métrique.** Le modèle MSE a la plus petite MSE d'entraînement (44,1 contre 62,4) : il a fait exactement ce qu'on lui demandait. Sur la métrique qui compte ici (l'erreur sur des données propres), il est cinq fois moins bon.
- **Être « trop bien classé » est puni par les moindres carrés.** Avec `(f − y)²`, un point +1 pour lequel `f(x) = 12` coûte autant qu'une erreur grossière ; la frontière bascule pour réduire ces sorties. La log-loss donne à ces points un poids σ(−m) presque nul et la hinge un poids exactement nul (marge supérieure à 1) : leurs frontières ne bougent pas.
- **La hinge ne regarde que quelques points** (18 sur 220) : les autres pourraient être déplacés sans rien changer, tant qu'ils restent au-delà de la marge (m > 1).
- **Température** : diviser les logits par T > 1 aplatit la distribution (entropie en hausse), T < 1 la pique.

## Questions de réflexion

1. En reprenant la dérivée de chaque perte, expliquez pourquoi la MSE mène à la moyenne et la MAE à la médiane. Quelle constante obtiendriez-vous avec Huber si δ devenait très grand ? très petit ?
2. Le modèle MSE de la partie 2 a la meilleure loss d'entraînement mais la pire erreur sur des données propres. Dans quelle situation réelle préférer malgré tout la MSE ? (Indice : que suppose-t-elle sur le bruit ?)
3. Les points lointains de la partie 3 sont tous **correctement** classés. Pourquoi est-ce quand même un problème pour les moindres carrés, et en quoi cela justifie-t-il des pertes faites pour la classification ?
4. Pourquoi le gradient de l'entropie croisée par rapport aux logits, `p − y`, est-il borné entre −1 et 1 ? Qu'est-ce que cela implique pour un exemple très mal classé, comparé à la MSE appliquée aux probabilités ?

## Pour aller plus loin

- Faire varier `delta_huber` (0,1 ; 1 ; 10) et la part d'aberrants (0 %, 10 %, 30 %) dans la cellule de configuration.
- Remplacer les aberrants « tous du même côté » par des aberrants symétriques (±25) : la pente MSE est-elle encore faussée ? Et son ordonnée ?
- Tracer la loss d'entraînement de chaque méthode au fil des itérations et comparer la vitesse de convergence de la MAE (sous-gradient, pas décroissant) à celle de Huber.
