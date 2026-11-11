# Exercice 13a : réseaux convolutifs, de la convolution en NumPy à CIFAR-10

| | |
|---|---|
| **Chapitre lié** | [13. Réseaux convolutifs (CNN)](https://laclasse.poneyponey.fr/ia/p06-deep-learning/13-cnn/) |
| **Exécution** | locale, MPS (partie A en NumPy ; parties B et C en PyTorch, possibles sur CPU mais environ 6 fois plus lentes) |
| **Commande** | `make ex NUM=13a` (éditeur), `make ex NUM=13a MODE=run` (complet), `MODE=smoke` (< 30 s) ; option ResNet-18 : `IA_RESNET18=1 make ex NUM=13a MODE=run` |
| **Durée mesurée** | run complet : 5 min 56 s, dont 5 min 45 s d'entraînement (MLP 26 s, CNN 2 min 36 s, CNN + augmentation 2 min 44 s) ; smoke : 11 à 13 s (M1 Pro, MPS, 14 et 15/10/2026, attente du verrou exclue). Option ResNet-18 non mesurée ici (≈ 1 min 30 s par époque d'après les mesures de la phase 1, soit ≈ 8 min de plus pour 5 époques) |
| **Travail estimé** | ~2 h |
| **Seed** | 42 (`seed_everything(42)` ; même graine avant chaque modèle, sous-ensembles et ordre des lots fixés par `torch.Generator().manual_seed(42)`) |
| **Données** | CIFAR-10 via `ia_commun.donnees.CIFAR10` (copie Hugging Face `uoft-cs/cifar10`), cache `data/cifar10/` (176 Mo, ≈ 20 s au premier lancement) ; partie A : image synthétique 64×64 générée |

## Objectif

Comprendre ce que calcule une couche de convolution en l'écrivant à la main, puis mesurer sur CIFAR-10 ce qu'elle apporte face à un MLP (exactitude, nombre de paramètres, robustesse aux décalages), ce qu'ajoute l'augmentation de données, et où le réseau « regarde » (Grad-CAM).

## Ce que vous allez faire

1. **A. La convolution à la main (NumPy).**
   - Écrire la corrélation croisée 2D avec quatre boucles, puis en vectorisé (`sliding_window_view` + `einsum`, le principe d'im2col), et vérifier qu'elle donne exactement `torch.nn.functional.conv2d` et `scipy.signal.correlate2d` ; constater que `scipy.signal.convolve2d` (noyau retourné) inverse le signe de Sobel.
   - Appliquer huit noyaux classiques (identité, flous, netteté, Sobel x et y, Laplacien, relief) à une image synthétique 64×64 (carré, disque, rayures), la même que dans la démo du chapitre ; détailler le calcul en un pixel.
   - Vérifier la formule de taille de sortie ⌊(n + 2p − k) / s⌋ + 1, l'**équivariance par translation** de la convolution, et le fait que le max pooling 2×2 n'est invariant qu'aux décalages compris dans son bloc.
   - Compter les paramètres d'une couche dense et d'une convolution qui produisent la même sortie ; calculer le champ récepteur du CNN de la partie B.
2. **B. MLP contre CNN sur CIFAR-10 (PyTorch).** Même budget (15 époques, AdamW, taux « one cycle » de maximum 2·10⁻³, lots de 128) pour trois modèles :
   - MLP 3 072 → 512 → 256 → 10 (1 707 274 paramètres) ;
   - petit CNN : 6 convolutions (32, 32, 64, 64, 128, 128 canaux ; BatchNorm + ReLU), 2 max pooling, moyenne globale, couche linéaire (290 282 paramètres) ;
   - le même CNN avec **augmentation de données** : recadrage aléatoire (marge de 4 pixels par réflexion, découpe 32×32) et retournement horizontal une fois sur deux, **tirés image par image** dans `__getitem__` (`ia_commun.augmentations.AugmentationClassique`), dans 6 processus de chargement.
   Puis évaluer les trois sur les 10 000 images de test d'origine, retournées horizontalement, et décalées de 4 pixels vers la droite.
3. **C. Ce que le CNN a appris.** Afficher les 32 filtres 5×5×3 de la première couche ; écrire **Grad-CAM** (Selvaraju et al., 2017) en une dizaine de lignes et l'appliquer à 8 images de test bien classées, une par classe.
4. Option : ResNet-18 adapté aux images 32×32 (première convolution 3×3, pas de max pooling initial), 5 époques avec augmentation.

**Perte optimisée** (B et C) : entropie croisée multiclasse. La partie A n'optimise rien.

## Résultat attendu

En mode complet, le notebook vérifie huit seuils et s'arrête avec une erreur si l'un d'eux n'est pas atteint. Valeurs obtenues le 14/10/2026 (M1 Pro, MPS, seed 42) :

**A. Convolution à la main**

| Vérification | Seuil | Obtenu |
|---|---|---|
| NumPy (vectorisé) = `torch.conv2d` = boucles = `scipy.correlate2d` | écart < 10⁻¹² | 4,4·10⁻¹⁶ |
| Équivariance : conv(décalée) = décalée(conv), loin des bords | écart < 10⁻¹² | 0 |

Autres valeurs : boucles 10 ms, vectorisé 0,20 ms (× 50) ; `convolve2d` avec Sobel x : écart 6,80 (signe inversé), avec le flou gaussien symétrique : 3,3·10⁻¹⁶. Au pixel (ligne 15, colonne 6), bord gauche du carré : fenêtre de trois colonnes 0,2238 / 0,9 / 0,9 ; Sobel x = 4 × (0,9 − 0,2238) = **+2,7048**, Sobel y = 0, flou (moyenne) = 0,6746, Laplacien = −0,6762. Décaler la carte de Sobel d'un pixel change 23 % des sorties du max pooling 2×2, mais pas le maximum global. Couche dense 3 072 → 32 768 : **100 696 064** paramètres ; convolution 5×5 de 3 vers 32 canaux (même sortie 32×32×32), avec biais : **2 432** (41 405 fois moins) ; sans biais, comme dans le CNN de la partie B où la BatchNorm en tient lieu : 2 400. Champ récepteur du CNN de la partie B : 5, 7, 8, 12, 16, 18, 26, puis **34×34** pixels après la dernière convolution (plus que l'image entière).

**B. CIFAR-10 (10 000 images de test, 15 époques)**

| Modèle | Paramètres | Entraînement | **Test** (seuil) | Images retournées | Décalées de 4 px |
|---|---|---|---|---|---|
| MLP | 1 707 274 | 87,7 % | **56,3 %** (≥ 50 %) | 54,8 % | 45,9 % (− 10,4 points ; seuil : perte ≥ 7) |
| CNN | 290 282 | 99,1 % | **84,6 %** (≥ 80 %) | 84,6 % | 81,4 % (− 3,2) |
| CNN + augmentation | 290 282 | 89,9 % | **87,4 %** (≥ 86 %) | 87,6 % (seuil : perte < 2 points) | 86,7 % (− 0,7) |

Seuil supplémentaire : l'augmentation gagne au moins 2 points en test (obtenu : + 2,8). Le hasard donne 10 %. Durées d'entraînement : 1,7 s par époque pour le MLP, 10,4 s pour le CNN, 10,9 s pour le CNN avec augmentation.

**C.** Grad-CAM sur 8 images de test (chat 95 %, bateau 100 %, avion 88 %, grenouille 100 %, automobile 66 %, camion 100 %, chien 98 %, cheval 100 %).

Figures produites dans `figures/` : `filtres-classiques.png`, `courbes.png`, `filtres-appris.png`, `gradcam.png`, et `resultats.npz` (résultats bruts, relus par les scripts `site/scripts/figures/ch13_courbes_cifar.py` pour la figure du chapitre).

## Ce qu'il faut observer

- **Moins de paramètres, bien meilleur résultat.** Le CNN a six fois moins de paramètres que le MLP et le dépasse de 28 points. Le partage des poids et la localité ne sont pas des contraintes qui coûtent : ce sont des hypothèses justes sur les images.
- **Le MLP est fragile aux décalages.** Décaler l'image de 4 pixels lui coûte 10 points, contre 3 au CNN : chaque poids du MLP est lié à une position précise, alors qu'une convolution applique le même filtre partout (équivariance, vérifiée à 0 près dans la partie A). Le CNN n'est pas pour autant parfaitement invariant : le pooling et les bords laissent passer un peu de l'effet du décalage.
- **Le CNN sans augmentation surapprend.** Il finit à 99,1 % en entraînement pour 84,6 % en test (écart de 14,5 points), avec une perte d'entraînement de 0,055. Avec l'augmentation, l'écart tombe à 2,5 points et le test gagne 2,8 points : le réseau ne voit jamais deux fois la même image. L'augmentation rend aussi le modèle presque insensible aux décalages (− 0,7 point), puisqu'elle lui en a montré.
- **Les filtres appris ne ressemblent pas à ceux d'AlexNet.** Les 32 filtres 5×5 de la première couche paraissent presque aléatoires (corrélation entre pixels voisins ≈ 0,19, contre ≈ 0 à l'initialisation). Les beaux détecteurs de bords orientés d'AlexNet viennent de filtres 11×11 appris sur des images 224×224 ; sur des images 32×32 et des filtres 5×5 suivis de BatchNorm, la structure est diffuse. Ne pas attendre de jolies images dans tous les cas.
- **Grad-CAM n'est qu'une carte grossière.** Elle vient de cartes 8×8, agrandies : elle couvre l'objet (cheval, chien, voiture) mais déborde, et parfois s'étale sur le contexte (le chat : la chaleur couvre aussi la couverture sur laquelle il est couché).

## Questions de réflexion

1. Le CNN a 290 282 paramètres et le MLP 1 707 274. Où sont passés les paramètres économisés, et qu'est-ce que le MLP pourrait apprendre que le CNN ne peut pas représenter ?
2. Pourquoi le CNN sans augmentation est-il insensible au retournement horizontal (84,6 % dans les deux cas) alors qu'il n'en a jamais vu ? Indice : regardez CIFAR-10 et la moyenne globale finale.
3. Le champ récepteur du dernier neurone fait 34×34 pixels. Pourquoi un champ récepteur plus grand que l'image ne suffit-il pas à garantir que le réseau « voit » toute l'image avec la même attention ?
4. L'augmentation choisie ici (recadrage, retournement horizontal) convient aux photos d'objets. Laquelle serait une erreur pour des chiffres manuscrits ? Pour des radiographies ?

## Pour aller plus loin

- Lancer l'option ResNet-18 (`IA_RESNET18=1`) : environ 11 millions de paramètres, blocs résiduels ; comparer l'exactitude après 5 époques à celle du petit CNN après 15.
- Remplacer le max pooling par des convolutions de pas 2 et refaire le test des images décalées.
- Exercice 13b (cloud) : apprentissage par transfert avec ResNet-50 pré-entraîné sur ImageNet.
- Démo `Convolution` du chapitre 13 : mêmes noyaux, même image, calcul détaillé pixel par pixel.
