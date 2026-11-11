# Exercice 24a : un mini-CLIP de zéro, avec un ViT ou un CNN comme encodeur d'image

| | |
|---|---|
| **Chapitre lié** | [24. Vision moderne et multimodalité](https://laclasse.poneyponey.fr/ia/p07-transformer/24-vision-multimodalite/) |
| **Exécution** | locale (MPS) |
| **Commande** | `make ex NUM=24a` (éditeur), `MODE=run` (complet), `MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 4 min 9 s, dont 1 min 47 s d'entraînement du ViT et 2 min 15 s du CNN ; smoke : 12 s (M1 Pro, MPS, 15/10/2026, attente du verrou GPU exclue) |
| **Travail estimé** | ~2 h |
| **Seed** | 42 (`seed_everything(42)` ; images d'entraînement tirées avec la graine 42, de test avec 43) |
| **Données** | générées (images synthétiques 32×32 et légendes en français), rien n'est téléchargé |

## Objectif

Faire tourner à petite échelle les deux idées du chapitre : un **Vision Transformer** lit une image comme une suite de patchs, et **CLIP** aligne un encodeur d'image et un encodeur de texte dans un même espace par une perte contrastive, ce qui permet de classer des images **sans exemple étiqueté** (zero-shot) en les comparant à des phrases. Puis comparer, à budget égal, un ViT et un CNN comme encodeur d'image.

## Ce que vous allez faire

**Partie A (NumPy, quelques millisecondes)**

1. Découper une image 3×32×32 en 64 patchs de 4×4 (vecteurs de 48 nombres), les projeter par une matrice $E$, et vérifier que c'est exactement une convolution de noyau 4×4 et de pas 4 (`F.conv2d`).
2. Compter les tokens et la taille de la matrice d'attention pour le mini-ViT, le ViT-B/32 de l'exercice 24b, le ViT-B/16 à 224 et à 384 pixels.
3. Calculer à la main la perte contrastive symétrique de CLIP sur 4 paires image-légende (vecteurs de dimension 3, échelle 10) et la comparer à `F.cross_entropy`.

**Partie B (PyTorch, MPS)**

4. Générer 20 000 images d'entraînement : une forme (carré, cercle, triangle, croix), une couleur (rouge, vert, bleu, jaune), une taille (petit, grand), une position (quatre coins), avec du bruit de fond, et leur légende française accordée (« une petite croix verte en bas à droite »). La moitié des légendes sont partielles (« un objet rouge », « un grand cercle »). Quatre couples forme + couleur sont **exclus de l'entraînement** : triangle bleu, cercle jaune, croix rouge, carré vert.
5. Entraîner deux mini-CLIP qui ne diffèrent que par l'encodeur d'image, 1 500 pas de 256 paires chacun (AdamW, taux 10⁻³ en « one cycle ») :
   - **ViT** : patchs 4×4, dimension 64, 3 blocs, 4 têtes, token `[CLS]`, positions apprises (161 600 paramètres) ;
   - **CNN** : trois convolutions 3×3 (32, 64, 96 canaux) avec pooling, puis moyenne globale (81 376 paramètres).

   L'encodeur de texte est un petit Transformer (2 blocs, dimension 64) sur 25 mots ; espace commun de dimension 64 ; échelle des logits apprise, partie de 1/0,07 ≈ 14,3.
6. Évaluer en zero-shot sur 2 000 images de test (dont 534 de couples jamais vus) : retrouver la légende complète parmi les 128 possibles, puis chaque attribut seul avec un prompt (« un cercle », « un objet rouge »…) ou avec un **ensemble** (moyenne des plongements des 32 légendes complètes qui contiennent la valeur).
7. Regarder la matrice de similarité image × légende et les plongements de position appris par le ViT.

Fichiers produits dans `figures/` : `exemples.png`, `bilan.png` (courbes de perte et scores), `similarites.png`, `positions.png`, et `resultats.npz` (lu par les figures du site, `site/scripts/figures/ch24_*.py`).

## Résultat attendu

Le notebook s'arrête avec une erreur si l'un de ces seuils n'est pas atteint en mode complet :

- partie A : patchs NumPy = convolution (écart < 10⁻¹², mesuré 6,7 × 10⁻¹⁶) ; perte de CLIP NumPy = PyTorch (0,1352 des deux côtés ; le hasard donnerait log 4 ≈ 1,386) ;
- **ViT : légende complète retrouvée parmi 128 pour ≥ 80 % des images de couples vus** (mesuré : 84,8 %) ;
- **CNN : ≥ 95 %** (mesuré : 100,0 %) ;
- couleur en zero-shot ≥ 95 % pour les deux (mesuré : 100 % et 100 %) ;
- forme en zero-shot (ensemble) du ViT inférieure ou égale à celle du CNN (mesuré : 58,7 % contre 86,0 %) ;
- couples jamais vus au moins 40 points sous les couples vus (mesuré : ViT 0,0 %, CNN 37,6 %).

Mesures du run complet (15/10/2026), hasard entre parenthèses :

| Zero-shot en test | ViT | CNN |
|---|---|---|
| Légende complète parmi 128, couples vus (0,8 %) | 84,8 % | 100,0 % |
| Légende complète parmi 128, couples jamais vus (0,8 %) | 0,0 % | 37,6 % |
| Forme, prompt seul « un cercle » (25 %) | 62,1 % | 78,3 % |
| Forme, ensemble de légendes (25 %) | 58,7 % | 86,0 % |
| Couleur « un objet rouge » (25 %) | 100,0 % | 100,0 % |
| Taille « un grand objet » (50 %) | 99,9 % | 99,9 % |
| Position « un objet en haut à gauche » (25 %) | 100,0 % | 100,0 % |
| Perte contrastive finale (hasard : log 256 ≈ 5,55) | ≈ 2,4 | ≈ 2,2 |
| Échelle des logits apprise (départ 14,3) | 19,5 | 19,8 |

Sur les images de couples jamais vus, la légende choisie a toujours la bonne couleur (100 % pour les deux encodeurs) ; la forme est juste dans 0 % des cas pour le ViT et 38 % pour le CNN.

## Ce qu'il faut observer

- **La projection des patchs est une convolution** de noyau et de pas égaux à la taille du patch : un ViT n'a pas d'autre opération « locale » que celle-là ; tout le reste (qui regarde qui) est appris par l'attention.
- **Le coût de l'attention** : 65 tokens pour le mini-ViT, 197 pour un ViT-B/16 en 224×224, 577 en 384×384, soit une matrice d'attention 8,6 fois plus grande quand on passe de 224 à 384 pixels.
- **La courbe de perte du ViT a deux temps** (`bilan.png`) : un palier vers 3,8 (il a appris couleur, taille et position, faciles à lire dans un seul patch), puis une seconde descente vers le pas 400 quand il commence à lire la forme, qui s'étale sur plusieurs patchs. Le CNN descend d'un coup : ses convolutions voient d'emblée des voisinages de pixels.
- **À budget égal, le CNN gagne** sur ces 20 000 petites images, avec deux fois moins de paramètres : c'est le constat de l'article du ViT (Dosovitskiy et al., 2020) : entraîné sur ImageNet seul, il reste « quelques points » derrière des ResNet de taille comparable, et il ne les rattrape ou ne les dépasse qu'avec un pré-entraînement sur 14 à 300 millions d'images. Le biais inductif de la convolution (localité, équivariance par translation) vaut beaucoup de données.
- **Le zero-shot marche pour ce qui a été vu** : « un objet rouge » trouve 100 % des images rouges alors qu'aucun classifieur de couleur n'a été entraîné. Mais il **ne compose pas bien** : pour un triangle bleu jamais vu, le ViT répond toujours une autre forme bleue (0 %), le CNN trouve la bonne légende dans 38 % des cas. Les vrais modèles vision-langage ont une faiblesse voisine : ils traitent souvent une phrase comme un sac de mots et distinguent mal « the horse is eating the grass » de « the grass is eating the horse » (Yuksekgonul et al., « When and why vision-language models behave like bags-of-words, and what to do about it? », ICLR 2023, arXiv:2210.01936).
- **La perte ne descend pas à zéro** (≈ 2,2) : avec 128 légendes complètes et des légendes partielles, un lot de 256 contient souvent plusieurs fois la même légende ; pour la perte, ce sont des « négatifs » alors que ce sont de bonnes réponses (faux négatifs).
- **La matrice de similarité** (`similarites.png`) : la diagonale ressort sur chaque ligne (cosinus 0,67 à 0,73 contre au plus 0,32 ailleurs) ; c'est exactement ce que la perte pousse.
- **Les positions apprises** (`positions.png`) : chaque case est surtout semblable à ses voisines (cosinus moyen 0,38 entre voisines d'une même rangée, −0,10 et −0,16 entre coins opposés). Personne n'a dit au ViT que l'image est une grille 2D : il le redécouvre en partie.

## Questions de réflexion

1. Pourquoi l'ensemble de légendes aide-t-il le CNN pour la forme (78 % → 86 %) mais pas le ViT (62 % → 59 %) ? Que fait la moyenne des 32 plongements quand l'encodeur d'image ne sépare pas bien les formes ?
2. Un classifieur supervisé à 128 sorties (une par légende complète), entraîné sur les mêmes données, que répondrait-il pour un triangle bleu ? Et quatre classifieurs séparés (forme, couleur, taille, position) ?
3. Avec des patchs de 8×8 au lieu de 4×4, combien de tokens ? Qu'attendez-vous pour la vitesse et pour la reconnaissance des petites formes (rayon 4 pixels) ?
4. Comment réduire les faux négatifs dans la perte contrastive sans changer les données ?

## Pour aller plus loin

- Doubler le nombre de pas (`config["pas"] = 3000`) : lors de la mise au point, le ViT est alors monté à 98,9 % sur les couples vus, mais pas au-dessus de 2 % sur les couples jamais vus.
- Passer `p_partiel` à 0 (légendes toujours complètes) et regarder ce que deviennent les prompts « un cercle » ou « un objet rouge ».
- Exercice 24b : le même zero-shot avec un vrai CLIP (ViT-B/32, LAION-2B) sur des images satellite, puis le réglage fin.
