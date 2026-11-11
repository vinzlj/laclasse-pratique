# Exercice 06 : neurone formel, perceptron et XOR

| | |
|---|---|
| **Chapitre lié** | [06. Neurone formel, perceptron, XOR et premier hiver](https://laclasse.poneyponey.fr/ia/p02-premiers-neurones/06-perceptron/) |
| **Exécution** | locale, CPU (numpy ; scikit-learn pour comparer) |
| **Commande** | `make ex NUM=06` (éditeur), `make ex NUM=06 MODE=run` (complet), `MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 5 s au total (`time make ex NUM=06 MODE=run`), dont 2,3 s de calcul ; smoke : 4 s (M1 Pro, CPU, 14/10/2026, attente du verrou exclue) |
| **Travail estimé** | ~45 min |
| **Seed** | 42 (`seed_everything(42)` ; `np.random.default_rng(42)` pour les données, graines 42 à 51 pour l'étude de la marge) |
| **Données** | générées dans le notebook (deux nuages gaussiens 2D, table de XOR) ; rien n'est téléchargé |

## Objectif

Programmer en numpy les premiers neurones artificiels (neurone formel de McCulloch et Pitts, règle de Hebb, perceptron de Rosenblatt), vérifier que le perceptron converge sur des données linéairement séparables et qu'il échoue sur XOR, puis résoudre XOR avec deux couches de neurones dont les poids sont fixés à la main.

## Ce que vous allez faire

1. **Neurone formel.** Écrire `neurone_mp(excitatrices, inhibitrices, seuil)` (entrées 0/1, une entrée inhibitrice active bloque la sortie) et câbler ET, OU, NON, « a ET NON b ». Aucun apprentissage : on choisit le seuil.
2. **Règle de Hebb.** Appliquer Δw = η·y·x̃ à chaque exemple (entrées et sorties en ±1, x̃ = (x₁, x₂, 1)). Elle trouve ET, mais échoue sur deux nuages pourtant séparables : elle renforce sans jamais corriger.
3. **Perceptron.** 200 points (100 par classe) étiquetés par la droite x₁ + 2·x₂ = 6, bande vide de demi-largeur 0,1 autour d'elle. Règle : si y·w·x̃ ≤ 0, alors w ← w + η·y·x̃. Puis :
   - réécrire la même boucle comme une **descente de sous-gradient** sur le critère du perceptron max(0, −y·w·x̃) et constater que les poids sont identiques ;
   - refaire avec `sklearn.linear_model.Perceptron` (même ordre des exemples) : mêmes poids ;
   - comparer le nombre de corrections à la borne (R/γ)² du théorème de convergence, R étant la plus grande norme d'un x̃ et γ la marge du séparateur de marge maximale (SVM linéaire, `C` très grand) ;
   - répéter sur 50 jeux de données (5 largeurs de bande × 10 graines) pour voir le lien entre marge et nombre de corrections.
4. **XOR.** Lancer le perceptron 100 époques sur les quatre points de XOR, chercher un cycle dans les poids, essayer 200 000 droites tirées au hasard, puis scikit-learn.
5. **XOR à deux couches.** Câbler h₁ = OU, h₂ = ET, sortie = h₁ ET NON h₂, en neurones formels puis en neurones à seuil (poids et biais). Entraîner un perceptron **sur la représentation cachée (h₁, h₂)**.

## Résultat attendu

En mode complet, le notebook vérifie treize seuils et s'arrête avec une erreur si l'un d'eux n'est pas atteint. Valeurs obtenues le 14/10/2026 (seed 42) :

| Vérification | Seuil | Obtenu |
|---|---|---|
| Neurone formel : ET, OU, NON, a ET NON b | tables exactes | exactes |
| Hebb sur ET bipolaire | 100 % | 100 % (w = (2 ; 2 ; −2)) |
| Hebb sur les nuages séparables | **sous 80 %** | 54,0 % (w = (217,6 ; 169,5 ; 0)) |
| Perceptron sur les nuages séparables | **100 %** | 100 %, 95 corrections en 7 époques |
| Corrections ≤ (R/γ)² | oui | 95 ≤ 22 195 (R = 6,63, γ = 0,0445) |
| Corrections ≤ borne, 50 jeux | oui sur les 50 | oui |
| Perceptron = SGD sur max(0, −y·w·x̃) | écart ≤ 1e-9 | 0 (identiques) |
| numpy = scikit-learn | écart ≤ 1e-9 | 0 ; w = (3,260 ; 7,388 ; −21) |
| XOR : meilleur poids visité par le perceptron | **≤ 75 %** | 50 % |
| XOR : 200 000 droites aléatoires | **≤ 75 %** | 75 % (jamais 4 sur 4) |
| XOR : scikit-learn | ≤ 75 % | 50 % |
| XOR à deux couches (neurones formels et à seuil) | exact | (0, 1, 1, 0) |
| Perceptron sur l'espace caché (h₁, h₂) | 100 % | 100 %, 11 corrections en 7 époques, w = (2 ; −3 ; −1) |

Autres valeurs affichées :

- erreurs par époque sur les nuages : 42, 17, 11, 8, 16, 1, 0 ; frontière apprise 3,26·x₁ + 7,39·x₂ − 21 = 0 (la vraie : x₁ + 2·x₂ − 6 = 0, soit 3·x₁ + 6·x₂ − 18 = 0 à un facteur près) ;
- point le plus proche de la frontière du perceptron : à 0,0096 ; marge du SVM sur les mêmes données : 0,1286 (13 fois plus) ;
- critère du perceptron : 0 en w = 0, 2,575 après la première correction, au plus 5,732, 0 à la fin ;
- marge contre corrections (moyenne sur 10 jeux, maximum entre parenthèses) :

  | Bande | γ moyen | Corrections | Borne (R/γ)² moyenne |
  |---|---|---|---|
  | 0,05 | 0,0289 | 112,6 (191) | 61 023 |
  | 0,1 | 0,0468 | 65,1 (95) | 20 056 |
  | 0,25 | 0,1003 | 31,2 (58) | 4 349 |
  | 0,5 | 0,1827 | 29,4 (67) | 1 329 |
  | 1 | 0,3561 | 13,6 (19) | 353 |

- XOR : 4 erreurs à **chaque** époque ; après 4 corrections, les poids reviennent à (0 ; 0 ; 0) : cycle d'une époque, répété 100 fois ;
- représentation cachée : (0,0) → (0,0), (0,1) → (1,0), (1,0) → (1,0), (1,1) → (1,1).

Figures produites : `figures/perceptron-separable.png`, `figures/xor.png`, `figures/marge.png`.

## Ce qu'il faut observer

- **Hebb renforce, le perceptron corrige.** Hebb ajoute y·x̃ pour chaque exemple : le résultat est (à un facteur près) la différence des moyennes des deux classes, et le biais vaut Σy = 0 quand les classes sont équilibrées. La droite passe donc par l'origine, loin des données (54 %). Le perceptron, lui, ne bouge que sur une erreur : c'est l'**erreur** qui pilote l'apprentissage.
- **La frontière se stabilise d'un coup.** Sur la figure `perceptron-separable.png`, les droites successives balaient le plan, puis plus rien : dès qu'une époque passe sans erreur, aucune mise à jour n'a lieu. Le nombre d'erreurs par époque n'est pas monotone (8 puis 16) : corriger un point peut en dérégler d'autres.
- **La droite trouvée n'est pas la « meilleure ».** Elle sépare les 200 points, mais n'importe quelle droite qui y parvient arrête l'apprentissage : celle-ci passe à 0,0096 du point le plus proche, quand la droite du SVM laisse 0,1286 de chaque côté. Chercher la droite la plus éloignée des deux classes, c'est l'idée de la marge (SVM, chapitre 08).
- **Perceptron = descente de gradient sur une perte.** La boucle « si erreur, corriger » est exactement un pas de SGD (η = 1) sur max(0, −y·w·x̃) : les poids sont identiques au bit près. C'est une perte charnière **sans marge** ; avec la marge 1, on retrouve la hinge du chapitre 01. Piège : en w = 0, cette perte vaut déjà 0 (minimum trivial) ; c'est la condition « ≤ 0 » qui fait démarrer le perceptron.
- **La borne de convergence est très large, mais elle dit la bonne tendance.** Moins de marge, plus de corrections ; la borne (R/γ)² dépasse le nombre observé de 26 fois (bande 1) à 540 fois (bande 0,05) en moyenne : elle doit valoir pour n'importe quel ordre de présentation et n'importe quels points de même R et γ, donc pour le pire cas.
- **XOR : le perceptron tourne en rond.** Les poids repassent exactement par (0 ; 0 ; 0) à la fin de chaque époque. Aucune droite ne classe les quatre points : 200 000 essais au hasard plafonnent à 3 sur 4.
- **Une couche cachée change l'espace.** Après la couche cachée, (0,1) et (1,0) tombent sur le même point (1,0) ; les quatre points deviennent séparables et un perceptron ordinaire les sépare en 11 corrections. La difficulté qui reste : ici, la couche cachée est **câblée à la main**. L'apprendre est l'objet du chapitre 07.

## Questions de réflexion

1. Pourquoi la règle de Hebb réussit-elle ET en codage ±1 mais échouerait-elle avec le codage 0/1 (essayez : `X_et` en 0/1, `y_et` en 0/1) ? Qu'est-ce que cela dit d'une règle qui ne regarde pas l'erreur ?
2. Le perceptron s'arrête à la première droite qui sépare les données. Proposez une modification de la règle de mise à jour qui l'empêcherait de s'arrêter sur une droite qui frôle des points. Quelle perte optimise-t-elle alors ?
3. Le pas η n'a aucun effet sur la suite des frontières quand w part de 0 : pourquoi ? (Indice : que deviennent les poids si on les multiplie tous par η ?)
4. Sur XOR, les poids reviennent à (0 ; 0 ; 0) à chaque époque. Que se passerait-il sur des données « presque » séparables (quelques points du mauvais côté) ? Comment choisir alors les poids à garder ?

## Pour aller plus loin

- Changer `bande` dans la configuration (0,01 ; 0,5 ; 2) et regarder le nombre de corrections.
- Mélanger les exemples à chaque époque (`rng.permutation`) : le nombre de corrections change-t-il beaucoup ? La borne, elle, ne dépend pas de l'ordre.
- Remplacer la condition `yi * (w @ xi) <= 0` par `yi * (w @ xi) <= 1` (perceptron avec marge) et comparer la frontière obtenue à celle du SVM.
- Ajouter une troisième entrée x₁·x₂ aux points de XOR : le perceptron converge-t-il ? Pourquoi ?
