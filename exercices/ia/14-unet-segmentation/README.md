# Exercice 14 : boîtes, IoU, NMS et segmentation avec un U-Net réduit

| | |
|---|---|
| **Chapitre lié** | [14. Détection et segmentation](https://laclasse.poneyponey.fr/ia/p06-deep-learning/14-detection-segmentation/) |
| **Exécution** | locale, MPS (partie A en NumPy ; parties C et D en PyTorch, possibles sur CPU mais environ 10 fois plus lentes) |
| **Commande** | `make ex NUM=14` (éditeur), `make ex NUM=14 MODE=run` (complet), `MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 4 min 44 s, 4 min 52 s et 5 min 36 s (trois mesures), dont 4 min 36 s à 5 min 29 s d'entraînement (quatre variantes de 58 à 76 s chacune) ; smoke : 15 s (M1 Pro, MPS, 14/10/2026, attente du verrou exclue) |
| **Travail estimé** | ~1 h 30 |
| **Seed** | 42 (`seed_everything(42)` ; même initialisation `torch.manual_seed(42)` avant chaque variante, ordre des lots fixé par `torch.Generator().manual_seed(42)` ; données : graines 42 (entraînement) et 4242 (validation)). Trois runs complets ont donné exactement les mêmes chiffres |
| **Données** | générées par le notebook (formes sur fond bruité), rien n'est téléchargé |

## Objectif

Passer de « nommer l'image » à « localiser et délimiter les objets » : calculer à la main l'IoU, la suppression des non-maxima (NMS), le Dice et la perte focale, puis entraîner un U-Net réduit en segmentation sémantique et mesurer ce qu'apportent ses ponts (connexions de saut) et le choix de la perte (entropie croisée, focale, entropie croisée + Dice).

## Ce que vous allez faire

1. **A. Boîtes et masques en NumPy, vérifiés contre torchvision.**
   - `iou_boites` (vectorisée, N × M boîtes) sur un exemple vérifiable à la main : A = (0, 0, 10, 10), B = (5, 5, 15, 15) → intersection 25, union 175, IoU = 0,143 ; décalage de 2 px → 0,667 ; moitié de A → 0,5.
   - `nms` : garder la boîte la mieux notée, supprimer celles qui la recouvrent au-delà du seuil, recommencer. Comparaison sur 200 boîtes aléatoires avec `torchvision.ops.box_iou` et `torchvision.ops.nms`.
   - IoU et Dice de deux masques (deux disques de rayon 10 décalés de 8 px : IoU 0,346, Dice 0,514 = 2·IoU/(1 + IoU)).
   - Perte focale $-(1-p_t)^\gamma \log p_t$ pour γ = 0, 1, 2, 5 et trois pixels (p_t = 0,9 facile, 0,5, 0,1 difficile).
2. **B. Données.** 3 000 images 64×64 d'entraînement, 500 de validation (1 471 objets). Chaque image : fond en dégradé de deux couleurs aléatoires, bruit gaussien σ = 0,08, 3 traits « leurres » (qui font partie du fond), 1 à 5 objets qui ne se touchent pas : disques, carrés tournés, triangles équilatéraux tournés, de couleur aléatoire. La classe dépend **de la forme seule**. Le fond occupe 90,4 % des pixels.
3. **C. Segmentation (PyTorch).** Un U-Net à trois étages (16, 32, 64 canaux, fond du U à 128, 0,48 M paramètres ; blocs de deux convolutions 3×3 + BatchNorm + ReLU ; max pooling ; convolutions transposées). Quatre variantes, chacune 6 époques, Adam (OneCycle, lr max 2·10⁻³), lots de 32 :
   - U-Net, entropie croisée ;
   - le même réseau **sans ponts** (0,43 M paramètres), entropie croisée ;
   - U-Net, perte focale (γ = 2, sans pondération α) ;
   - U-Net, entropie croisée + Dice douce (moyenne sur les 3 classes d'objets).
4. **D. Des masques aux boîtes.** Chaque tache connexe d'une classe dans le masque prédit (`scipy.ndimage.label`, taches de moins de 6 px ignorées) devient une boîte notée par la probabilité moyenne de sa classe ; appariement glouton à IoU ≥ 0,5 ; précision, rappel, AP par classe et mAP@0,5. Puis NMS sur des boîtes candidates en double simulées autour des vraies boîtes.

## Résultat attendu

Le run complet s'arrête avec une erreur si l'un de ces seuils n'est pas atteint :

| Vérification | Seuil | Mesuré (14/10/2026) |
|---|---|---|
| IoU NumPy contre `torchvision.ops.box_iou` | écart < 10⁻⁹ | 0 |
| NMS NumPy contre `torchvision.ops.nms` | même liste | identique (141 boîtes gardées sur 200) |
| U-Net, entropie croisée : mIoU des 3 classes d'objets | ≥ 80 % | **85,0 %** |
| U-Net, entropie croisée : mAP@0,5 (boîtes tirées des masques) | ≥ 85 % | **90,1 %** |
| Gain des ponts en mIoU (U-Net − sans ponts) | ≥ 4 points | **+7,1** (85,0 contre 77,9 %) |
| Gain des ponts sur les pixels de bord | ≥ 5 points | **+10,2** (86,1 contre 75,9 %) |
| U-Net, entropie croisée + Dice : mIoU objets | ≥ 85 % | **89,2 %** |

Tous les chiffres (500 images de validation) :

| Variante | Pixels justes | IoU cercle / carré / triangle | mIoU objets | Pixels de bord justes | mAP@0,5 | Durée (run de `resultats.json`) |
|---|---|---|---|---|---|---|
| « Toujours fond » (référence) | 90,5 % | 0 / 0 / 0 | 0 % | — | 0 % | — |
| U-Net, entropie croisée | 98,89 % | 87,4 / 83,2 / 84,3 | 85,0 % | 86,1 % | 90,1 % | 64 s |
| Sans ponts, entropie croisée | 98,21 % | 85,4 / 76,8 / 71,5 | 77,9 % | 75,9 % | 83,8 % | 59 s |
| U-Net, perte focale (γ = 2) | 98,18 % | 79,1 / 70,7 / 74,8 | 74,9 % | 80,6 % | 81,4 % | 62 s |
| U-Net, entropie croisée + Dice | 99,16 % | 91,9 / 88,4 / 87,4 | 89,2 % | 88,4 % | 93,6 % | 70 s |

Autres chiffres : pixels de bord (à moins d'un pixel d'un changement de classe) = 5,6 % de l'image ; NMS (seuil 0,3) : 108 boîtes candidates → 23 gardées pour 18 objets.

Figures produites dans `figures/` : `exemples.png` (images, vérité, quatre prédictions avec leurs boîtes), `courbes.png` (mIoU de validation par époque), et les données `resultats.json` et `exemples.npz` lues par les figures du site (`site/scripts/figures/ch14_segmentation.py`, `ch14_variantes.py`).

## Ce qu'il faut observer

- **La précision par pixel ment.** Répondre « fond » partout donne déjà 90,5 % de pixels justes et une IoU nulle sur chaque objet. Entre le pire et le meilleur réseau, la précision par pixel ne bouge que d'un point (98,2 → 99,2 %) alors que la mIoU objets passe de 75 à 89 %. D'où l'IoU (et le Dice) comme mesures de référence en segmentation.
- **Les ponts servent surtout aux bords.** Loin des contours, U-Net et réseau sans ponts se valent (99,65 contre 99,52 % de pixels justes) ; sur les pixels de bord, l'écart est de 10 points (86,1 contre 75,9 %). Le réseau sans ponts doit reconstruire un contour net à partir d'une carte 8×8 : il arrondit les coins des triangles et des carrés (figure `exemples.png`, contours orange des pixels faux).
- **Le Dice accélère et améliore, la focale non (ici).** Avec entropie croisée + Dice, la mIoU atteint 72 % dès la 3ᵉ époque (45 % pour l'entropie croisée seule, 28 % pour la focale). La perte focale, conçue pour un détecteur à une étape qui évalue ≈ 100 000 positions candidates presque toutes faciles, termine ici sous l'entropie croisée (74,9 contre 85,0 %) : avec 90 % de fond seulement, sans la pondération α ni l'initialisation du biais de l'article, réduire le poids des pixels faciles ralentit surtout l'apprentissage en 6 époques. Ce n'est pas une mesure générale de la focale, seulement de ce réglage sur ce problème.
- **Un masque donne des boîtes.** Les boîtes tirées des taches du masque atteignent une mAP@0,5 de 90 % sans aucune tête de régression ; les erreurs viennent des taches coupées en deux ou des petites formes mal classées (un petit carré pris pour un cercle).
- **La NMS ne règle pas tout.** Avec des doublons décalés de ±3 px, le seuil 0,3 laisse 23 boîtes pour 18 objets : sur un petit objet, deux doublons peuvent avoir une IoU inférieure à 0,3 (voir la figure de l'IoU du chapitre : 2 px de décalage coûtent beaucoup plus à un objet de 10 px qu'à un objet de 40 px).

## Questions de réflexion

1. Pourquoi le réseau sans ponts se trompe-t-il surtout sur les bords et presque pas à l'intérieur des objets ? Que contient la carte 8×8 du fond du U, et que lui manque-t-il ?
2. La perte focale a été inventée pour la détection en une étape. Quelle différence d'échelle de déséquilibre y a-t-il entre ce problème (90 % de fond) et un détecteur qui évalue ≈ 100 000 boîtes candidates par image ? Essayez γ = 0,5 ou 1, ou ajoutez une pondération α par classe : l'écart avec l'entropie croisée change-t-il ?
3. Le Dice se calcule sur des ensembles de pixels, pas pixel par pixel : pourquoi est-il insensible au nombre de pixels de fond ? Que se passerait-il avec un lot qui ne contient aucun pixel d'une classe (d'où le terme `eps`) ?
4. Ici, deux objets ne se touchent jamais. Que deviendraient les boîtes de la partie D si deux disques se touchaient ? Quelle tâche (segmentation d'instances) faudrait-il alors résoudre ?

## Pour aller plus loin

- Ronneberger, Fischer & Brox, « U-Net: Convolutional Networks for Biomedical Image Segmentation », arXiv:1505.04597 (2015) : l'architecture d'origine (23 couches de convolution, quatre sous-échantillonnages, 64 à 1 024 canaux ; entrée 572×572, sortie 388×388 car les convolutions sont sans remplissage ; cartes de l'encodeur recadrées avant concaténation). Celle de l'exercice a trois étages, 16 à 128 canaux, et des convolutions avec remplissage (sortie de même taille que l'entrée).
- Lin, Goyal, Girshick, He & Dollár, « Focal Loss for Dense Object Detection », arXiv:1708.02002 (2017) : la perte focale et le détecteur RetinaNet (déséquilibre « par exemple 1:1000 », perte appliquée aux ≈ 100 000 ancres de chaque image, γ = 2 avec α = 0,25, biais de la dernière couche initialisé pour prédire une probabilité π = 0,01 d'objet).
- Remplacer le décodeur par un simple agrandissement bilinéaire (×8) de la carte du fond, puis comparer la précision sur les bords avec les deux variantes de l'exercice.
