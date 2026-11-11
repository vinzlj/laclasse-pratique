# Exercice 26 : reconstruire les pixels (MAE) ou prédire la représentation (JEPA) ?

| | |
|---|---|
| **Chapitre lié** | [26. Le point LeCun : masquer, prédire, contraster, prédire en espace latent (JEPA)](https://laclasse.poneyponey.fr/ia/p07-transformer/26-point-lecun-jepa/) |
| **Exécution** | locale (MPS ; tourne aussi sur CPU, plus lentement) |
| **Commande** | `make ex NUM=26` (éditeur), `MODE=run` (complet), `MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 3 min 57 s (MAE 29 s, JEPA avec EMA 46 s, JEPA sans EMA 64 s, JEPA + VICReg 65 s, évaluation 26 s, le reste en chargement) ; smoke : 12 s (M1 Pro, MPS, 15/10/2026, attente du verrou GPU exclue) |
| **Travail estimé** | ~2 h |
| **Seed** | 42 (`seed_everything(42)`, mêmes graines pour chaque modèle) |
| **Données** | MNIST (`ia_commun.donnees.mnist_tenseurs`, cache `data/mnist/`, téléchargé au premier lancement) : 20 000 images d’entraînement tirées au hasard, les 10 000 images de test |

## Objectif

Comparer deux façons d’apprendre sans étiquette à partir d’une image dont on cache une partie : **reconstruire les pixels** cachés (MAE) ou **prédire leur représentation** dans un espace latent (JEPA), puis provoquer et empêcher l’**effondrement**, le piège propre aux méthodes qui comparent des représentations.

## Ce que vous allez faire

1. **Partie A (NumPy, sans apprentissage).** Calculer à la main les pertes Barlow Twins et VICReg sur trois lots de 512 paires de vues en 8 dimensions : sain, effondré (toutes les images ont le même vecteur), redondant (8 dimensions qui recopient 2 facteurs). Constater que l’**alignement seul** (la MSE entre les deux vues) préfère le lot effondré.
2. **Partie B.** Découper chaque image en 4 × 4 patchs de 7 × 7 pixels et construire un petit Transformer (4 blocs, 128 dimensions, 537 000 paramètres) qui sert d’encodeur aux deux méthodes.
3. **Partie C (MAE).** Cacher 75 % des patchs au hasard (12 sur 16), encoder les 4 visibles, reconstruire les pixels des 12 autres avec un petit décodeur. Perte : MSE sur les pixels cachés.
4. **Partie D (JEPA).** Cacher trois blocs rectangulaires, encoder le reste (le contexte), prédire avec un petit prédicteur la représentation que l’encodeur **cible** donne des patchs cachés (normalisée par LayerNorm). Perte : MSE entre vecteurs latents. Trois variantes, mêmes réseaux et mêmes masques :
   - **JEPA (EMA)** : l’encodeur cible est une moyenne mobile exponentielle de l’encodeur de contexte (moment de 0,996 à 1, comme I-JEPA) et le gradient ne le traverse pas ;
   - **JEPA sans EMA** : la cible est calculée par le même encodeur et le gradient la traverse ;
   - **JEPA + VICReg** : comme la précédente, plus les termes de variance et de covariance de VICReg sur la représentation moyenne de chaque image.
5. **Partie E.** Geler chaque encodeur et mesurer sa représentation (moyenne des 16 vecteurs de patchs, protocole de MAE et d’I-JEPA) : sonde linéaire avec 20 000 étiquettes, avec 100 par classe, k plus proches voisins (k = 20), dispersion d’une image à l’autre, rang effectif. Repères : pixels bruts et même encodeur au hasard.
6. **Partie F.** Regarder ce que produit chaque modèle sur trois chiffres de test : des pixels pour le MAE, des vecteurs (rendus en couleurs par une ACP) pour la JEPA. Ces données alimentent la démo `MaskingCompare` du chapitre.

Pour expérimenter sans modifier le fichier : `IA26_CONFIG='{"epoques": 30}' make ex NUM=26 MODE=run` (toute clé de `config`).

## Résultat attendu

Run complet (le notebook s’arrête avec une erreur si un seuil n’est pas atteint). Valeurs obtenues le 15/10/2026 :

| Représentation | Sonde linéaire | Sonde, 100/classe | kNN (k = 20) | Dispersion | Sonde sur les 16 vecteurs bout à bout |
|---|---|---|---|---|---|
| Pixels bruts (784) | 91,4 % | 86,7 % | 95,8 % | (0,184, non comparable) | 91,4 % |
| Encodeur au hasard | 73,3 % | 63,8 % | 62,3 % | 0,205 | 95,1 % |
| MAE | **78,7 %** | 69,2 % | **73,9 %** | 0,475 | 95,9 % |
| JEPA (EMA) | 73,7 % | 64,4 % | 61,9 % | 0,450 | 95,3 % |
| JEPA sans EMA | 77,4 % | 65,4 % | 67,5 % | **0,005** | 93,0 % |
| JEPA + VICReg | 67,1 % | 61,4 % | 63,7 % | 0,832 | 93,2 % |

Dispersion : écart-type d’une image à l’autre, moyenné sur les 128 dimensions de sortie de l’encodeur (normalisées par sa LayerNorm finale) ; 0 = toutes les images ont la même représentation. Les pixels bruts ne passent pas par une LayerNorm : leur valeur n’est pas comparable aux autres. Le kNN utilise la similarité cosinus.

Seuils vérifiés :

- **A** : alignement seul 0,000 pour le lot effondré contre 0,021 pour le lot sain ; Barlow Twins 7,80 contre 0,00 et VICReg 24,75 contre 0,74 (poids de l’article, 25, 25 et 1, mais variance et covariance moyennées sur les deux vues : la formule de l’article les additionne, ce qui doublerait ces deux termes ; seuil : effondré au moins 5 fois plus pénalisé) ; lot redondant : Barlow hors diagonale 0,118 contre 0,000, covariance VICReg 3,06 contre 0,009.
- **MAE** : sonde linéaire **≥ encodeur au hasard + 3 points** (78,7 % contre 73,3 %) et kNN **≥ hasard + 5 points** (73,9 % contre 62,3 %). Perte finale 0,061 (prédire la moyenne : 0,094).
- **JEPA (EMA)** : **pas d’effondrement** (dispersion 0,450 ≥ 0,205 du hasard) et prédictions meilleures que la représentation moyenne de chaque position (**cosinus + 0,1** au moins : 0,861 / 0,918 / 0,857 contre 0,766 / 0,741 / 0,756 sur les trois chiffres de la partie F).
- **JEPA sans EMA** : **effondrée** (dispersion ≤ 10 % du hasard : 0,005) avec une perte de prédiction ≤ 0,01 (0,0004).
- **JEPA + VICReg** : pas d’effondrement (dispersion 0,832 ≥ 0,205).

Figures produites : `figures/mae-jepa.png` (courbes, dispersion, sondes, et ce que prédit chaque modèle), `figures/resultats.npz` (lu par les figures du site et la démo).

## Ce qu’il faut observer

- **La perte la plus basse est la pire.** La JEPA sans EMA passe sous une perte de 0,01 en moins de 100 pas et finit à 0,0004 : elle a trouvé la solution triviale (tout envoyer sur la même représentation, que le prédicteur recopie). La dispersion tombe à 0,02 dès la première époque, puis à 0,005. Une perte qui compare deux représentations calculées par le réseau lui-même ne dit rien de la qualité si rien n’interdit l’effondrement.
- **Deux remèdes différents marchent.** L’asymétrie (encodeur cible en moyenne mobile, sans gradient, plus un prédicteur), qui est celle de BYOL et d’I-JEPA, et la régularisation explicite (VICReg : variance minimale par dimension), qui est l’esprit de VICReg puis de LeJEPA. Les deux gardent une dispersion supérieure à celle du réseau au hasard. Aucune des deux n’est contrastive (pas de négatif) : l’une repose sur une asymétrie, l’autre est une méthode régularisée au sens des modèles à énergie (chapitre 26, section 1).
- **Pixels contre représentations.** Le MAE dessine des chiffres flous (partie F) : sous l’incertitude, la MSE en pixels récompense la moyenne des possibles. La JEPA ne dessine rien ; ses vecteurs prédits sont proches des cibles (cosinus 0,86 à 0,92) et bien meilleurs que la représentation moyenne de la position.
- **À cette échelle, la JEPA ne gagne pas.** Sa représentation moyenne ne classe pas mieux les chiffres que le réseau au hasard (73,7 % contre 73,3 %), alors que le MAE gagne 5 points. Les gains d’I-JEPA sur MAE sont mesurés sur ImageNet, avec des ViT de centaines de millions de paramètres et des centaines d’époques. Trois minutes sur MNIST ne les reproduisent pas ; l’exercice montre le mécanisme, pas le résultat de l’article.
- **Le protocole d’évaluation compte.** Mis bout à bout, les 16 vecteurs d’un encodeur au hasard donnent 95,1 % : plus que les pixels bruts (91,4 %). Une sonde trop riche mesure la capacité de la sonde, pas ce que l’encodeur a appris ; d’où la moyenne des patchs.
- La JEPA sans EMA garde 77,4 % de sonde et 67,5 % de kNN malgré l’effondrement : il est presque total en amplitude (dispersion 0,005), pas en information. Les infimes écarts restants suffisent à classer, et ni la sonde (qui standardise chaque dimension) ni le kNN cosinus (qui normalise chaque vecteur) ne regardent l’échelle. Seule la dispersion révèle l’effondrement ; une telle représentation est inutilisable pour toute tâche sensible à l’échelle ou au bruit numérique.

## Questions de réflexion

1. Dans la partie A, pourquoi la diagonale de la matrice de Barlow Twins suffit-elle à pénaliser l’effondrement complet, mais pas l’effondrement « dimensionnel » du lot redondant ?
2. La JEPA sans EMA s’effondre. Que se passerait-il si l’on gardait l’arrêt du gradient sur la cible mais sans moyenne mobile (cible = encodeur courant, détaché) ? Essayez en modifiant `entrainer_jepa` (indice : c’est la recette de SimSiam, Chen et He, 2020).
3. Le MAE reconstruit des chiffres flous. Pourquoi une MSE en pixels produit-elle du flou quand plusieurs complétions sont plausibles, et en quoi prédire une représentation contourne-t-il le problème ?
4. Avec `IA26_CONFIG='{"epoques": 40}'`, la sonde de la JEPA (EMA) dépasse-t-elle celle de l’encodeur au hasard ? Qu’en conclure sur la comparaison MAE / JEPA à petite échelle ?

## Pour aller plus loin

- Remplacer la MSE de la JEPA par une perte L1 (comme V-JEPA) et comparer.
- Prédire chaque bloc cible séparément (comme I-JEPA) au lieu de tous ensemble, et tirer un masque par image plutôt qu’un par lot.

## Écarts assumés par rapport aux articles

- Grille de 4 × 4 patchs de 7 × 7 pixels (I-JEPA : 14 × 14 patchs ou plus) : lors de nos premiers essais (7 × 7 patchs de 4 × 4 pixels, entrées brutes), on voyait la JEPA s’effondrer sur la seule position (la cible ne codait presque que le plongement de position).
- Entrées centrées-réduites et plongement de position multiplié par 0,5 ; même raison.
- Trois blocs cibles de 12 à 25 % de l’image (I-JEPA : quatre de 15 à 20 %), contexte = tout le reste (I-JEPA : un bloc de 85 à 100 % privé des cibles), un masque par lot, tous les blocs prédits en une passe.
