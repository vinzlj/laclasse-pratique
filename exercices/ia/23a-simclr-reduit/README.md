# Exercice 23a : apprentissage contrastif, d'InfoNCE en NumPy à SimCLR réduit

| | |
|---|---|
| **Chapitre lié** | [23. Apprentissage contrastif](https://laclasse.poneyponey.fr/ia/p07-transformer/23-contrastif/) |
| **Exécution** | locale, MPS (partie A en NumPy ; partie B en PyTorch, possible sur CPU mais plusieurs fois plus lente) |
| **Commande** | `make ex NUM=23a` (éditeur), `make ex NUM=23a MODE=run` (complet), `MODE=smoke` (< 30 s) ; `IA_EPOQUES=…` et `IA_TAU=…` changent le nombre d'époques et la température |
| **Durée mesurée** | run complet : **≈ 6 min 30 s** (6 min 18 s du premier entraînement au bilan), dont 5 min 8 s d'entraînements auto-supervisés (SimCLR 2 min 28 s, Barlow Twins 2 min 13 s, alignement seul 26 s), 21 s de sondes, 34 s pour le scénario « peu d'étiquettes », 12 s de t-SNE ; smoke : 21 s (M1 Pro, MPS, 15/10/2026, attente du verrou exclue) |
| **Travail estimé** | ~2 h |
| **Seed** | 42 (`seed_everything(42)` ; même graine avant chaque modèle, sous-ensembles et ordre des lots fixés par `numpy.random.default_rng(42)` et `torch.Generator().manual_seed(42)`) |
| **Données** | CIFAR-10 via `ia_commun.donnees.CIFAR10` (copie Hugging Face `uoft-cs/cifar10`), cache `data/cifar10/` (176 Mo, ≈ 20 s au premier lancement) ; partie A : vecteurs écrits à la main ou tirés au hasard |

## Objectif

Écrire InfoNCE (la perte de SimCLR, NT-Xent) à la main et voir ce que règlent la température et les négatifs, puis entraîner un petit CNN **sans étiquette** sur 25 000 images de CIFAR-10 et mesurer ce que vaut sa représentation : sonde linéaire, kNN, peu d'étiquettes, t-SNE. Comparer avec Barlow Twins (aucun négatif) et montrer qu'un simple alignement des deux vues s'effondre.

## Ce que vous allez faire

1. **A. Les pertes à la main (NumPy).**
   - **A1.** Trois images, deux vues chacune, représentations placées sur le cercle unité (chat à 0° et 20°, avion à 120° et 150°, camion à 240° et 215°) : matrice des 6 × 6 cosinus, softmax de chaque vue sur ses 5 candidates, perte NT-Xent pour τ = 1 ; 0,5 ; 0,1. Même calcul en PyTorch (entropie croisée dont la « classe » d'une vue est l'indice de sa jumelle), et perte sigmoïde de SigLIP (t = 10, b = −10, initialisation de l'article).
   - **A2.** Température : une vue, sa jumelle à 0,8, six négatives de 0,7 à −0,5. Comme ∂ℓ/∂s_k = (p_k − 1[k = positive]) / τ, chaque négative est repoussée en proportion de sa probabilité : on mesure la part de la plus difficile et de la plus facile.
   - **A3.** Ce que chaque perte « voit » de trois situations (256 images, 128 dimensions) : bonnes représentations, vues indépendantes, tout au même point (effondrement). Alignement seul, InfoNCE, Barlow Twins.
2. **B. SimCLR réduit sur CIFAR-10 (PyTorch, MPS).**
   - 25 000 images tirées au hasard, **sans étiquette** ; deux vues par l'augmentation de SimCLR (recadrage redimensionné 20-100 %, retournement, distorsion de couleur p = 0,8, gris p = 0,2), **tirées image par image** dans 6 processus de chargement (`ia_commun.donnees.PairesVues` + `ia_commun.augmentations.AugmentationSimCLR`).
   - Encodeur : le petit CNN de l'exercice 13a sans sa couche de classification (6 convolutions, 288 992 paramètres), sortie h de 128 nombres ; tête de projection 128 → 512 → 128 (132 736 paramètres).
   - Trois entraînements, même optimiseur (AdamW, taux « one cycle » de maximum 2·10⁻³, lots de 512 images donc 1 023 candidates par vue), seule la perte change : **SimCLR** (NT-Xent, τ = 0,5) 15 époques ; **Barlow Twins** (λ = 5·10⁻³) 15 époques ; **alignement seul** (‖z_i − z_j‖² sur vecteurs normalisés) 3 époques **en partant de l'encodeur SimCLR entraîné**.
   - Évaluations sur l'encodeur gelé : sonde linéaire (régression logistique sur h, 50 000 images étiquetées, 10 000 de test), kNN (k = 20, cosinus), indicateur d'effondrement (écart-type de z/‖z‖ par dimension × √d, repère de SimSiam), nombre de dimensions de h qui portent 99 % de la variance, retrouver la jumelle parmi 1 023 vues de test.
   - **Peu d'étiquettes** : sonde sur 100 puis 500 images étiquetées (10 et 50 par classe), contre le même CNN entraîné de zéro sur ces seules images (augmentation classique du ch. 13, 500 pas de gradient).
   - t-SNE des h de 2 000 images de test avant et après SimCLR ; matrice 16 × 16 des similarités de 8 images × 2 vues.

**Perte optimisée** : InfoNCE (NT-Xent) pour SimCLR ; Barlow Twins et alignement seul en comparaison ; entropie croisée pour les sondes et le CNN supervisé.

## Résultat attendu

En mode complet, le notebook vérifie neuf seuils et s'arrête avec une erreur si l'un n'est pas atteint (sauf en variante `IA_TAU` ou `IA_EPOQUES`, voir la question 2). Valeurs obtenues le 15/10/2026 (M1 Pro, MPS, seed 42) :

**A. Les pertes à la main**

| Vérification | Seuil | Obtenu |
|---|---|---|
| NT-Xent et SigLIP : NumPy = PyTorch | écart ≤ 10⁻⁹ | 5,6·10⁻¹⁷ |

Autres valeurs : exemple à 3 images, perte NT-Xent 0,7503 (τ = 1), 0,3206 (τ = 0,5), 0,0034 (τ = 0,1) ; SigLIP 1,2932. Température (jumelle à 0,8) : perte 1,447 / 1,117 / 0,644 / 0,351 / 0,129 et part de la négative la plus difficile 28 / 40 / 67 / 87 / 98 % pour τ = 1 / 0,5 / 0,2 / 0,1 / 0,05. Effondrement (N = 256, d = 128) :

| Situation | Alignement | InfoNCE (τ = 0,5) | Barlow Twins |
|---|---|---|---|
| bonnes représentations | 0,164 | 4,426 | 1,19 |
| vues indépendantes | 1,991 | 6,243 | 127,64 |
| tout au même point | **0,000** | 6,236 = log(511) | 128,00 = d |

**B. CIFAR-10 (encodeur gelé, 10 000 images de test)**

| Encodeur | Sonde linéaire (seuil) | kNN | Écart-type × √d (seuil) | Dimensions pour 99 % de la variance de h | Jumelle retrouvée parmi 1 023 vues |
|---|---|---|---|---|---|
| CNN au hasard | 45,5 % | 40,6 % | 0,008 | 31 | 2,6 % |
| SimCLR, 15 époques | **62,6 %** (≥ 60 %) | 59,6 % | 0,979 (≥ 0,5) | 64 | 47,2 % |
| Barlow Twins, 15 époques | **59,3 %** (≥ 55 %) | 55,0 % | 0,912 (≥ 0,5) | 57 | 28,1 % |
| SimCLR puis alignement seul, 3 époques | 56,5 % | 50,6 % | **0,023** (≤ 0,1 : effondré) | 15 | 19,5 % |

Seuil supplémentaire : SimCLR gagne au moins 10 points de sonde sur le CNN au hasard (obtenu : + 17,1). Le hasard donne 10 %. Pertes : SimCLR 6,22 → 5,50 ; Barlow Twins 36,1 → 9,20 ; alignement seul 0,17 → 0,0006. Sur les 8 images de la figure des similarités, cosinus moyen entre jumelles 0,80 et entre images différentes −0,05 après SimCLR (1,00 et 1,00 pour le CNN au hasard : ses représentations h, issues de ReLU, sont toutes positives et partagent une forte composante commune, que la tête au hasard conserve, si bien que tous les z pointent presque dans la même direction).

**Peu d'étiquettes (exactitude en test)**

| Étiquettes | CNN au hasard + sonde | SimCLR + sonde | Barlow Twins + sonde | CNN supervisé de zéro | Seuil |
|---|---|---|---|---|---|
| 100 (10 par classe) | 22,7 % | **41,3 %** | 39,1 % | 32,6 % | SimCLR − supervisé ≥ 5 points (+ 8,8) |
| 500 (50 par classe) | 29,1 % | **49,2 %** | 46,6 % | 47,6 % | SimCLR − supervisé ≥ 0 (+ 1,6) |

## Ce qu'il faut observer

- **A1 : InfoNCE est une entropie croisée.** À τ = 0,5, la vue « chat 1 » met 85,5 % de sa probabilité sur sa jumelle (cosinus 0,94) ; à τ = 0,1, 100 %. La perte passe de 0,750 (τ = 1) à 0,321 (τ = 0,5) et 0,003 (τ = 0,1) sans qu'aucun vecteur ne bouge : la température change l'échelle, pas la géométrie.
- **A2 : la température choisit les négatives qui comptent.** À τ = 1, la négative la plus ressemblante reçoit 28 % de la poussée et la plus éloignée 8,4 % ; à τ = 0,1, 87 % contre 0,0 % ; à τ = 0,05, 98 %. Basse température = attention aux négatives difficiles.
- **A3 : sans négatifs, la solution triviale est parfaite.** Tout au même point donne un alignement de 0,000 (meilleur que les « bonnes » représentations, 0,164), mais InfoNCE vaut alors log(511) = 6,236, aussi mauvais que des vues indépendantes (6,243), et Barlow Twins vaut 128 (= d : la corrélation d'une dimension constante est indéfinie, 0/0, mais le petit ε du code, qui évite la division par zéro, rend la matrice C nulle, diagonale comprise). Ce sont les négatifs, ou la contrainte sur les statistiques des sorties, qui interdisent l'effondrement.
- **B : la représentation apprise sans étiquette sépare les classes.** La sonde linéaire passe de 45,5 % (même CNN au hasard, qui n'est pas nul : des convolutions aléatoires gardent des couleurs et des textures) à 62,6 % avec SimCLR. Repère supervisé (exercice 13a, même CNN, 50 000 étiquettes) : 84,6 %. L'écart se réduit avec plus d'images, d'époques et un encodeur plus profond (exercice 23b).
- **B : l'effondrement est immédiat sans négatifs.** En repartant de l'encodeur SimCLR, l'alignement seul fait tomber l'indicateur de 0,98 à 0,05 en une époque, la perte à 0,0006, et h ne garde plus que 15 directions utiles (contre 64) ; la sonde retombe à 56,5 % (h, en amont de la tête, s'effondre moins vite que z).
- **B : peu d'étiquettes, c'est là que l'auto-supervisé paie.** Avec 100 étiquettes, la sonde sur SimCLR bat le CNN entraîné de zéro de 8,8 points ; avec 500, l'écart n'est plus que de 1,6 point (le CNN supervisé avec augmentation rattrape vite) : à cette petite échelle, l'avantage se concentre sur les tout petits jeux étiquetés.
- **InfoNCE et information mutuelle** : perte finale 5,503 pour 1 023 candidates, donc I(vue 1 ; vue 2) ≥ log(1 023) − 5,503 = 1,43 nat ; la borne ne peut jamais dépasser log(1 023) = 6,93 nats, quelle que soit la qualité de l'encodeur.

Figures produites dans `figures/` : `courbes.png` (pertes relatives et indicateur d'effondrement), `sondes.png`, `tsne.png`, `similarites.png`, et `resultats.npz` (résultats bruts relus par les scripts `site/scripts/figures/ch23_*.py` pour les figures du chapitre).

## Questions de réflexion

1. La perte de SimCLR ne descend que de 6,22 à 5,50 en 15 époques, alors que la sonde gagne 17 points. Pourquoi la valeur d'InfoNCE est-elle un mauvais indicateur de la qualité de la représentation ? (Indice : la borne et log(2N − 1).)
2. Relancez avec `IA_TAU=0.1` puis `IA_TAU=1`. Que deviennent la sonde et l'indicateur d'effondrement ? Reliez à la partie A2. Les variantes (`IA_TAU` différent de 0,5 ou `IA_EPOQUES` posé) écrivent `figures/*-tau….png` et `resultats-tau….npz`, sans écraser les fichiers du run de référence relus par le site, et ne vérifient pas les seuils.
3. Barlow Twins n'a aucun négatif et ne s'effondre pas ; l'alignement seul s'effondre. Qu'est-ce qui, dans la formule de Barlow Twins, joue le rôle des négatifs ? (Regardez la partie A3 : que vaut la matrice C quand tout est au même point ?)
4. On évalue h (avant la tête de projection), pas z. D'après SimCLR, pourquoi la représentation avant la tête garde-t-elle plus d'information utile ? Vérifiez-le en lançant la sonde sur z.

## Pour aller plus loin

- Ajouter une file de négatifs et un encodeur à moment (MoCo) pour découpler le nombre de négatifs de la taille du lot.
- Exercice 23b (cloud) : la même expérience avec un ResNet-18, 50 000 images et 100 époques.
