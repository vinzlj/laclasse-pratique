# 13. Réseaux convolutifs (CNN)

Repère historique (1989 et 1998 (précurseurs : 1962, 1980 ; percée : 2012)) : Reconnaître des images en exploitant deux propriétés des images, la localité (un motif tient dans une petite zone) et l’invariance par translation (un chiffre décalé reste le même chiffre), avec beaucoup moins de paramètres qu’un réseau entièrement connecté.

En 1962, les neurophysiologistes David Hubel et Torsten Wiesel (Harvard) décrivent dans le cortex visuel du chat des cellules « simples », qui répondent à un bord orienté dans une petite zone de la rétine, et des cellules « complexes », qui répondent au même bord un peu n'importe où dans une zone plus large. En 1980, Kunihiko Fukushima (laboratoires de recherche de la NHK, à Tokyo) en tire le Neocognitron : une pile alternée de filtres locaux et de couches de regroupement, qui reconnaît des formes décalées, mais qui n'apprend pas par gradient. En 1989, aux Bell Labs, Yann LeCun et ses collègues entraînent une architecture de ce type par rétropropagation pour lire des codes postaux manuscrits. En 1998, l'article de LeCun, Bottou, Bengio et Haffner décrit LeNet-5 et un système de lecture de chèques, bâti sur ces réseaux, qui en lit plusieurs millions par jour. Il faudra pourtant attendre 2012 et AlexNet, entraîné sur deux GPU avec 1,2 million d'images d'ImageNet, pour que les réseaux convolutifs s'imposent dans toute la vision par ordinateur.

## L'idée en une image

Prenez une petite photo couleur de 32×32 pixels, comme celles de CIFAR-10 : 32 × 32 × 3 = 3 072 nombres. Un perceptron multicouche (chapitre 07) les aligne en un long vecteur et relie chaque pixel à chaque neurone de la première couche. Deux défauts en découlent :

1. Il oublie la géométrie. Pour lui, le pixel (10, 10) et son voisin (10, 11) sont deux entrées sans rapport, exactement comme le pixel (10, 10) et le pixel (31, 0). Mélangez les pixels de toutes les images dans le même ordre : le MLP apprend aussi bien qu'avant. Toute la structure spatiale est à réapprendre à partir des exemples.
2. Il apprend chaque chose à chaque endroit. Un détecteur d'œil appris en haut à gauche de l'image ne sert à rien en bas à droite : ce sont d'autres poids. Décalez l'image de quelques pixels et la réponse change.

Le réseau convolutif part de deux hypothèses sur les images, qu'on lui impose au lieu de les lui faire apprendre :

- Localité. Un motif utile (un bord, un coin, une texture) tient dans une petite fenêtre : un neurone n'a besoin de regarder que 3×3 ou 5×5 pixels voisins.
- Stationnarité. Un motif utile à un endroit l'est partout : on applique le même petit filtre à toutes les positions. C'est le partage des poids.

L'image mentale : une loupe de 3×3 pixels, toujours la même, qu'on promène sur toute l'image ligne par ligne, et qui note à chaque position « à quel point je vois mon motif ici ». La feuille de notes obtenue est une nouvelle image, une carte de caractéristiques. Une couche a plusieurs loupes (32, 64…), donc plusieurs cartes. La couche suivante promène ses loupes sur ces cartes et repère des motifs de motifs : des bords on passe aux coins, des coins aux formes, des formes aux objets.

Le tableau compare les deux sur CIFAR-10 (exercice 13a), après 15 époques (15 passages complets sur les images d'entraînement). L'exactitude est la part d'images bien classées ; la dernière variante ajoute l'augmentation de données de la section 7.

| | Avant : MLP sur les pixels | Après : réseau convolutif (exercice 13a) |
|---|---|---|
| Ce que voit un neurone de la 1<sup>re</sup> couche | toute l'image (3 072 entrées) | une fenêtre de 5×5 pixels × 3 couleurs (75 entrées) |
| Poids | propres à chaque position | partagés par toutes les positions |
| Paramètres (CIFAR-10, exercice) | 1 707 274 | 290 282 (six fois moins) |
| Exactitude en test, 15 époques | 56,3 % | 84,6 % (87,4 % avec augmentation) |
| Image décalée de 4 pixels | − 10,4 points | − 3,2 points (− 0,7 avec augmentation) |
| Ce qui coince | ignore la géométrie, énorme dès que l'image grandit | ne voit que localement ; profondeur difficile à entraîner (jusqu'à ResNet) |

Le CNN a six fois moins de paramètres et fait 28 points de mieux. Les contraintes qu'on lui impose ne lui coûtent rien parce qu'elles sont vraies pour des images : c'est un biais inductif bien choisi (chapitre 10), c'est-à-dire une hypothèse faite avant d'avoir vu les données.

## Le mécanisme

Le flux de données du petit CNN de l'exercice 13a, de l'image aux probabilités :

Chaque « conv » est suivie d'une normalisation par lot (chapitre 12) et d'une ReLU. La largeur et la hauteur diminuent (32, 16, 8) pendant que le nombre de canaux augmente (3, 32, 64, 128) : moins de positions, mais des descriptions plus riches de chaque position. On détaille chaque brique.

### 1. La convolution : un filtre qu'on fait glisser

Sur une image en niveaux de gris. La convolution fait glisser sur l'image un noyau, un tableau de 3×3 poids. Pour calculer le pixel de sortie $(i, j)$, on pose le noyau sur la fenêtre 3×3 centrée en $(i, j)$, on multiplie terme à terme et on additionne.

Exemple pris dans l'exercice et dans la démo ci-dessous : le pixel (ligne 15, colonne 6) de l'image synthétique est sur le bord gauche d'un carré clair. La fenêtre a trois lignes identiques, avec un fond sombre à gauche (0,2238) et le carré à droite (0,9). Le noyau de Sobel x détecte les bords verticaux :

```text
fenêtre 3×3                 noyau Sobel x            produits terme à terme
0,2238  0,9  0,9            −1   0   +1              −0,2238   0   +0,9
0,2238  0,9  0,9      ×     −2   0   +2       →      −0,4476   0   +1,8
0,2238  0,9  0,9            −1   0   +1              −0,2238   0   +0,9

somme = (0,9 − 0,2238) × (1 + 2 + 1) = 4 × 0,6762 = 2,7048
```

Sur une zone uniforme, la colonne de gauche et celle de droite sont égales : le résultat vaut 0. Sur un bord vertical, il est grand, positif si la droite est plus claire, négatif sinon. Le même noyau, posé partout, produit une carte qui « s'allume » sur tous les bords verticaux de l'image et seulement là. Un autre noyau, un autre motif : la moyenne des 9 pixels floute (ici $(0{,}2238 + 0{,}9 + 0{,}9)/3 = 0{,}6746$) ; le Laplacien $\begin{smallmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{smallmatrix}$ réagit à tous les contours ($0{,}9 + 0{,}9 + 0{,}2238 + 0{,}9 - 4 \times 0{,}9 = -0{,}6762$).

À vous : modifiez les 9 poids, choisissez un préréglage, ajoutez une marge de zéros, changez le pas, et cliquez (ou touchez) un pixel de la sortie pour voir son calcul terme à terme. Les flèches du clavier déplacent le pixel inspecté. Vous pouvez aussi charger votre propre image : elle reste dans votre navigateur.

Cherchez : un noyau qui détecte les bords horizontaux ; ce que donne un noyau dont les poids ont une somme nulle sur une zone uniforme ; la taille de la sortie avec un pas de 2.

Le cas général : plusieurs canaux en entrée et en sortie. Une photo a trois canaux (rouge, vert, bleu) ; après une couche de 32 noyaux, l'entrée de la couche suivante en a 32. Un noyau a alors une tranche par canal d'entrée, et la couche a un noyau par canal de sortie :

(Formule : Couche de convolution (telle que la calculent les bibliothèques).)

En simplifiant : en mathématiques, une convolution retourne le noyau avant de le faire glisser ($u$ et $v$ apparaissent avec un signe moins). Ce que calculent `torch.nn.Conv2d` et toutes les bibliothèques d'apprentissage profond est une corrélation croisée, sans retournement. Comme les poids sont appris, cela ne change rien à ce que le réseau peut représenter : il apprendrait simplement le noyau retourné. L'exercice le vérifie : `scipy.signal.correlate2d` donne exactement le résultat de PyTorch, alors que `scipy.signal.convolve2d`, qui retourne le noyau, inverse le signe de Sobel x (écart maximal de 6,80, soit 2 × 3,40 : la carte de Sobel va de −3,40 à +3,40 et son signe est inversé partout), mais pas celui du flou gaussien, symétrique (écart de $3{,}3 \times 10^{-16}$).

Taille de la sortie. Avec une entrée de $n \times n$ pixels, un noyau $k \times k$, une marge $p$ et un pas $s$ :

$$
n_{\text{sortie}} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1.
$$

Sur l'image 64×64 de la démo avec un noyau 3×3 : sans marge, $(64 - 3)/1 + 1 = 62$ (on perd un pixel de chaque côté) ; avec une marge de 1, 64 (même taille) ; avec une marge de 1 et un pas de 2, $\lfloor 63/2 \rfloor + 1 = 32$.

Nombre de paramètres. Une couche de $C_{\text{out}}$ noyaux $k \times k$ sur $C_{\text{in}}$ canaux a $k^2\, C_{\text{in}}\, C_{\text{out}} + C_{\text{out}}$ paramètres, quelle que soit la taille de l'image. Une couche de 32 noyaux 5×5 sur 3 canaux, avec biais : $5 \times 5 \times 3 \times 32 + 32 = 2\,432$. (Dans le CNN de l'exercice, les convolutions n'ont pas de biais, la normalisation par lot qui suit en tient lieu : sa première couche a 2 400 poids.) Une couche dense qui produirait la même sortie (32 cartes de 32×32, soit 32 768 nombres) à partir des 3 072 entrées en aurait $3\,072 \times 32\,768 + 32\,768 = 100\,696\,064$ : 41 405 fois plus. C'est le gain du partage des poids et de la localité réunis.

Le code. L'exercice écrit d'abord la corrélation avec quatre boucles, puis en vectorisé : on découpe toutes les fenêtres d'un coup, sans copie, et on fait le produit avec le noyau. C'est l'idée d'im2col qu'utilisent les bibliothèques pour ramener la convolution à un produit de matrices, l'opération pour laquelle les GPU sont faits.

```python
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def correlation(x, k, pad=0, pas=1):
    """Corrélation croisée 2D d'une image x (H, W) par un noyau k (kh, kw)."""
    x = np.pad(x, pad)                                          # marge de zéros
    fenetres = sliding_window_view(x, k.shape)[::pas, ::pas]   # (h, w, kh, kw), sans copie
    return np.einsum("hwab,ab->hw", fenetres, k)                # Σ fenêtre × noyau, à chaque (h, w)

def taille_sortie(n, k, p, s):
    return (n + 2 * p - k) // s + 1
```

Sur l'image 64×64, cette version prend 0,20 ms contre 10 ms pour les boucles (50 fois plus rapide), et coïncide avec `torch.nn.functional.conv2d` à $4{,}4 \times 10^{-16}$ près. Dans un vrai réseau, on écrit simplement `nn.Conv2d(3, 32, kernel_size=5, padding=2)`.

Équivariance par translation. Décaler l'image puis la convoluer donne la même chose que la convoluer puis décaler le résultat :

$$
\operatorname{conv}\bigl(\tau_d\, x\bigr) = \tau_d\, \operatorname{conv}(x),
$$

où $\tau_d$ décale l'image de $d$ pixels (avec un pas de 1 ; avec un pas $s$, l'égalité ne vaut que pour des décalages multiples de $s$). Un chat déplacé de 4 pixels produit les mêmes cartes, déplacées de 4 pixels. C'est l'équivariance par translation ; l'exercice la vérifie avec un écart exactement nul (loin des bords, où la marge de zéros casse la symétrie). Attention : équivariant n'est pas invariant. Les cartes bougent avec l'objet ; or un classifieur veut une réponse qui ne bouge pas (invariance par translation). C'est le rôle du pooling et de la moyenne finale.

Le champ récepteur. Un neurone de la première couche voit $k \times k$ pixels. Un neurone de la deuxième voit $k \times k$ neurones de la première, qui voient chacun $k \times k$ pixels, avec recouvrement : deux couches 3×3 voient 5×5 pixels, trois en voient 7×7. Le champ récepteur grandit à chaque couche, et plus vite après chaque réduction de taille :

(Formule : Champ récepteur, couche après couche.)

Pour le CNN de l'exercice : 5 (conv 5×5), 7 (conv 3×3), 8 (pooling 2×2, l'écart passe à 2), 12, 16, 18 (pooling, écart 4), 26, puis 34×34 après la dernière convolution, plus que l'image entière. Cela ne veut pas dire que le dernier neurone regarde toute l'image avec la même attention : les pixels du centre de son champ récepteur passent par beaucoup plus de chemins que ceux du bord.

### 2. Le pooling : résumer et rétrécir

Après une ou deux convolutions, on réduit la taille des cartes. Le max pooling 2×2 de pas 2 découpe chaque carte en blocs de 4 pixels et ne garde que le maximum de chaque bloc :

```text
carte 4×4                 max pooling 2×2           carte 2×2
 1  3 | 2  1
 4  2 | 0  1                max(1,3,4,2) = 4         4  2
------+------     →         max(2,1,0,1) = 2    →
 5  0 | 1  1                max(5,0,7,2) = 7         7  3
 7  2 | 3  0                max(1,1,3,0) = 3
```

Trois effets, pour zéro paramètre :

- Quatre fois moins de valeurs à traiter par la couche suivante.
- Le champ récepteur double plus vite : après le pooling, un noyau 3×3 couvre 6×6 pixels de la carte d'avant.
- Une tolérance aux petits décalages : si le maximum bouge d'un pixel à l'intérieur de son bloc, la sortie ne change pas. Mais ce n'est qu'une tolérance : dans l'exercice, décaler la carte de Sobel d'un pixel change 23 % des sorties du max pooling (les maxima qui passent d'un bloc au voisin), même si le maximum global, lui, ne bouge pas.

Le pooling par la moyenne existe aussi. Les réseaux récents se terminent par une moyenne globale : chaque carte, quelle que soit sa taille, est résumée par la moyenne de tous ses pixels. Il reste un nombre par canal (128 dans l'exercice), indépendant de la position de l'objet : c'est là que l'équivariance des convolutions devient, approximativement, une invariance. Le CNN de l'exercice n'a ainsi qu'une petite couche linéaire 128 → 10 à la fin, au lieu des grosses couches denses de LeNet ou d'AlexNet.

### 3. LeNet : la convolution apprise par rétropropagation

Le Neocognitron de 1980 avait déjà l'architecture (filtres locaux partagés, regroupement, hiérarchie), mais ses filtres s'organisaient par une règle non supervisée. La contribution de LeCun et de ses collègues aux Bell Labs est d'entraîner toute la pile, filtres compris, par rétropropagation d'une perte de classification (chapitre 07). Le partage des poids ne pose aucun problème au gradient : un poids partagé reçoit la somme des gradients de toutes les positions où il a servi.

En 1989, le réseau lit les chiffres manuscrits de codes postaux fournis par le service postal américain. En 1998, l'article de synthèse décrit LeNet-5 : une image de 32×32, deux convolutions 5×5 (6 puis 16 cartes) alternées avec des sous-échantillonnages 2×2, une troisième convolution (C5, 120 cartes, qui sur des cartes de 5×5 revient à une couche dense), puis une couche entièrement connectée de 84 unités et les 10 sorties.

Les chiffres de l'article : 340 908 connexions, mais seulement 60 000 paramètres grâce au partage des poids ; 0,95 % d'erreur en test sur MNIST, et 0,8 % quand on entraîne sur des versions déformées des chiffres (petites translations, changements d'échelle, cisaillements), une forme précoce d'augmentation de données. Le système de lecture de chèques construit autour de ces réseaux lit, selon l'article, plusieurs millions de chèques par jour.

Pourquoi les CNN n'ont-ils pas conquis la vision dès 1998 ? Sur des images naturelles de taille réelle, il aurait fallu beaucoup plus de données étiquetées et de calcul que ce dont on disposait. Pendant les années 2000, la vision par ordinateur s'appuie surtout sur des caractéristiques conçues à la main (SIFT, HOG) suivies d'un classifieur comme une SVM (chapitre 08).

### 4. ImageNet et AlexNet : le déclic de 2012

Deux ingrédients manquaient : les données et le calcul.

Les données. En 2009, Jia Deng, Fei-Fei Li et leurs collègues (Princeton) présentent ImageNet : des images du web rangées selon la hiérarchie des noms de WordNet et vérifiées par des annotateurs payés à la tâche (Amazon Mechanical Turk). L'article décrit déjà 3,2 millions d'images dans 5 247 catégories. À partir de 2010, le concours ILSVRC en extrait un sous-ensemble devenu la référence : environ 1,2 million d'images d'entraînement, 1 000 classes. La mesure principale est l'erreur top-5 : la réponse compte comme juste si la bonne classe est parmi les cinq classes jugées les plus probables.

Le calcul. Les GPU, conçus pour le jeu vidéo, font très vite des produits de matrices, donc des convolutions. En 2012, Dan Cireşan, Ueli Meier et Jürgen Schmidhuber (IDSIA, à Manno, près de Lugano) montrent avec des CNN entraînés sur GPU des records sur MNIST, des caractères manuscrits et des panneaux routiers.

AlexNet. Fin 2012, Alex Krizhevsky, Ilya Sutskever et Geoffrey Hinton (Université de Toronto) gagnent ILSVRC 2012 avec 15,3 % d'erreur top-5, contre 26,2 % pour l'équipe classée deuxième. Le réseau, c'est LeNet en beaucoup plus grand, avec quatre changements qui comptent :

| | LeNet-5 (1998) | AlexNet (2012) |
|---|---|---|
| Entrée | 32×32, niveaux de gris | 224×224×3, couleur |
| Couches | 3 convolutions (dont C5, qui revient à une couche dense) + 2 entièrement connectées | 5 convolutions + 3 denses |
| Paramètres | 60 000 | 60 millions |
| Données | 60 000 chiffres | 1,2 million d'images, 1 000 classes |
| Activation | tangente hyperbolique mise à l'échelle | ReLU (chapitre 07) |
| Régularisation | partage des poids | partage des poids, dropout, augmentation de données |
| Calcul | processeur | deux GPU GTX 580 de 3 Go, cinq à six jours d'entraînement |

L'article l'écrit lui-même : la taille du réseau est limitée par la mémoire des GPU et par le temps d'entraînement qu'on accepte, et les résultats s'amélioreraient avec des GPU plus rapides et plus de données. La suite lui donnera raison. Les ReLU accélèrent l'entraînement ; le dropout (chapitre 12) et l'augmentation (section 7) limitent le surapprentissage, sérieux même avec 1,2 million d'images.

Ce que voient les couches :
En 2013, Matthew Zeiler et Rob Fergus (NYU) projettent dans l'espace des pixels ce qui active chaque couche d'un réseau de type AlexNet : la première couche réagit à des bords orientés et à des taches de couleur, les suivantes à des textures, puis à des parties d'objets (roues, visages, pattes). C'est la hiérarchie que Hubel et Wiesel avaient décrite chez le chat et que Fukushima avait voulu reproduire. Attention à ne pas généraliser : avec des images de 32×32 et des filtres 5×5 suivis d'une normalisation, les 32 filtres de première couche de l'exercice 13a restent presque informes (corrélation entre pixels voisins d'un filtre ≈ 0,19, contre ≈ 0 à l'initialisation).

### 5. Plus profond : VGG et Inception (2014)

Après 2012, la course se joue sur la profondeur, avec deux styles.

VGG (Karen Simonyan et Andrew Zisserman, Oxford) : la simplicité. Uniquement des convolutions 3×3, empilées par deux ou trois entre deux poolings, 16 ou 19 couches à poids. L'argument tient dans le champ récepteur : trois convolutions 3×3 voient 7×7 pixels, comme une seule convolution 7×7, mais avec $3 \times 3^2\, C^2 = 27\,C^2$ poids au lieu de $7^2\, C^2 = 49\,C^2$ (pour $C$ canaux en entrée et en sortie), et trois non-linéarités au lieu d'une. VGG finit deuxième en classification à ILSVRC 2014 (7,3 % d'erreur top-5 avec un ensemble de 7 réseaux), avec 138 à 144 millions de paramètres, surtout dans ses couches denses finales.

GoogLeNet (Christian Szegedy et ses collègues, Google), qui gagne ILSVRC 2014 avec 6,67 % : l'économie. Il a 22 couches, mais 12 fois moins de paramètres qu'AlexNet. Ses modules Inception appliquent en parallèle des convolutions 1×1, 3×3 et 5×5 et un pooling, puis concatènent les résultats : le réseau choisit lui-même l'échelle utile. Pour que cela reste abordable, des convolutions 1×1 réduisent d'abord le nombre de canaux : elles ne regardent qu'un pixel, mais mélangent ses canaux, comme une petite couche dense appliquée à chaque position. GoogLeNet remplace aussi les grosses couches denses finales par une moyenne globale.

Une idée a aussi été introduite en 2015 pour entraîner ces réseaux : la normalisation par lot, qui renormalise les activations de chaque couche sur le lot. Elle est traitée au chapitre 12 ; le CNN de l'exercice l'utilise après chaque convolution.

### 6. Highway Networks, puis ResNet : la connexion résiduelle

Le paradoxe de la profondeur. On s'attend à ce qu'un réseau plus profond fasse au moins aussi bien qu'un moins profond : il suffirait que les couches en trop recopient leur entrée. En pratique, en 2015, ce n'est pas le cas. Kaiming He et ses collègues (Microsoft Research) montrent qu'un réseau « simple » de 56 couches a une erreur d'entraînement plus élevée qu'un réseau de 20 couches sur CIFAR-10. Ce n'est donc pas du surapprentissage : c'est l'optimisation qui échoue. Ils appellent cela le problème de dégradation. Apprendre l'identité avec une pile de couches non linéaires n'a rien d'évident pour la descente de gradient.

La solution. Au lieu de demander à un bloc de couches de calculer la fonction voulue $H(x)$, on lui fait calculer la correction $F(x) = H(x) - x$, et on rajoute l'entrée par un raccourci :

(Formule : Bloc résiduel (He et al., 2015).)

Deux lectures de la même formule :

- Pour la passe avant, recopier l'entrée devient trivial : il suffit que $F$ soit nulle, c'est-à-dire des poids proches de zéro. Une couche inutile ne nuit plus.
- Pour la passe arrière, le terme $I$ de la dérivée fait passer le gradient directement d'un bloc au précédent, ce qui facilite l'optimisation des réseaux très profonds. En simplifiant : à l'intérieur d'un même étage (même taille de cartes), l'entrée traverse les blocs par un chemin sans multiplication par des poids ; dans la version de 2015, une ReLU suit encore chaque addition, et la variante « pré-activation » de 2016 (He et al., *Identity Mappings in Deep Residual Networks*) la retire pour que ce chemin soit exactement l'identité.

Attention à une idée reçue : la dégradation ne vient pas d'une disparition du gradient, le phénomène décrit au chapitre 12. Les auteurs le vérifient : leurs réseaux simples, normalisés par lot, gardent des gradients de norme saine. C'est l'optimisation elle-même qui peine à trouver une bonne solution, et le raccourci la lui rend accessible.

Quand $x$ et $F(x)$ n'ont pas la même forme (après une réduction de taille, ou un changement du nombre de canaux), le raccourci passe par une convolution 1×1 apprise (de pas 2 quand la taille diminue), ou, variante testée par l'article, complète $x$ par des zéros sur les canaux ajoutés. ResNet atteint 152 couches, huit fois plus que VGG pour un calcul moindre, et gagne ILSVRC 2015 avec 3,57 % d'erreur top-5 (ensemble de réseaux). Le bilan du concours estime l'erreur humaine à 5,1 %, mais pour un annotateur entraîné et sur un échantillon de 1 500 images : la comparaison est indicative, pas une preuve que la machine « voit mieux ».

Highway Networks : le précurseur. Sept mois plus tôt, en mai 2015, Rupesh Srivastava, Klaus Greff et Jürgen Schmidhuber (IDSIA) proposent les Highway Networks, qui entraînent des réseaux d'une centaine de couches (et jusqu'à 900 dans des essais préliminaires) avec une porte apprise :

$$
y = H(x)\cdot T(x) + x \cdot \bigl(1 - T(x)\bigr),
$$

où $T(x)$, entre 0 et 1, décide quelle part de l'entrée passe telle quelle. La connexion résiduelle $y = F(x) + x$ revient à supprimer la porte, et à laisser toujours passer l'entrée entière. La paternité de l'idée est discutée. L'article de ResNet cite les Highway Networks comme un travail concurrent et souligne la différence : ses raccourcis n'ont pas de paramètres et ne sont « jamais fermés ». Schmidhuber, en 2025, présente au contraire ResNet comme « un Highway Net aux portes ouvertes » et fait remonter l'idée du raccourci de poids 1 au mémoire de Sepp Hochreiter (1991) sur les réseaux récurrents (chapitre 16). Les faits ne sont pas contestés, leur poids l'est ; c'est en tout cas ResNet, plus simple, que tout le monde a repris.

Pourquoi ce chapitre compte pour la suite. La connexion résiduelle est sortie des CNN : chaque bloc de chaque Transformer (chapitre 20) l'utilise, et c'est elle qui permet d'empiler les dizaines de couches des grands modèles de langage. Elle est encore un sujet de recherche : fin 2025, les mHC de DeepSeek (*Manifold-Constrained Hyper-Connections*) élargissent le flux résiduel en plusieurs flux mélangés d'une couche à l'autre, comme le proposaient les Hyper-Connections de ByteDance (Zhu et al., 2024, ICLR 2025), mais contraignent la matrice de mélange à être doublement stochastique (coefficients positifs ou nuls, lignes et colonnes de somme 1) pour retrouver la stabilité du raccourci identité, que le mélange libre compromettait.

Les CNN après les Transformers. À partir de 2020, les Vision Transformers (chapitre 24) dépassent les CNN sur ImageNet quand on les pré-entraîne sur assez de données. En 2022, Zhuang Liu et ses collègues (FAIR et Berkeley) reprennent un ResNet et lui appliquent un à un les choix de conception des Transformers (noyaux plus grands, moins de normalisations et d'activations, autre recette d'entraînement). Le résultat, ConvNeXt, fait jeu égal ou mieux que les Transformers hiérarchiques (Swin) de taille comparable ; son plus grand modèle, pré-entraîné sur ImageNet-22K (environ 14 millions d'images), atteint 87,8 % d'exactitude top-1 sur ImageNet. Une part de l'avantage attribué à l'attention venait donc de la recette autour.

### 7. L'augmentation de données

Ce chapitre est le propriétaire de l'augmentation de données dans le parcours. L'idée : pendant l'entraînement, on présente au réseau des variantes de chaque image qui gardent la même étiquette. Un chat recadré un peu à gauche, ou vu dans un miroir, reste un chat. À chaque époque, le réseau voit d'autres variantes : il ne peut plus apprendre les images par cœur, et il apprend à ignorer les transformations qu'on lui montre.

Les transformations classiques en vision :

| Transformation | Ce qu'elle enseigne | À éviter quand… |
|---|---|---|
| Recadrage aléatoire (avec une marge) | l'objet peut être n'importe où | l'objet touche les bords et serait coupé |
| Retournement horizontal | gauche et droite sont équivalentes | le sens compte : texte, chiffres, organes (cœur à gauche) |
| Retournement vertical | haut et bas sont équivalents | presque toujours pour des photos ; utile pour des images satellite ou de microscope |
| Couleur, luminosité, contraste | l'éclairage ne change pas la classe | la couleur est l'information (fruits mûrs, feux) |
| Rotation, échelle, cisaillement | l'angle de vue ne change pas la classe | « 6 » et « 9 » ; des radiographies orientées par convention |

L'exercice 13a, sur CIFAR-10 : recadrage aléatoire (marge de 4 pixels par réflexion, puis découpe d'un carré 32×32) et retournement horizontal une fois sur deux. Le même CNN, mêmes 15 époques :

| | Entraînement | Test | Écart | Test, images décalées de 4 px |
|---|---|---|---|---|
| CNN sans augmentation | 99,1 % | 84,6 % | 14,5 points | 81,4 % |
| CNN avec augmentation | 89,9 % | 87,4 % | 2,5 points | 86,7 % |

Sans augmentation, le réseau finit à 99,1 % sur ses images d'entraînement (perte de 0,055) : il les connaît presque par cœur. Avec, il fait moins bien sur l'entraînement et mieux en test (+ 2,8 points) : la tâche d'entraînement est plus dure, et ce qu'il apprend se généralise mieux. Il est aussi devenu presque insensible aux décalages (− 0,7 point), puisqu'on lui en a montré des milliers.

Un piège pratique, vérifié pendant la préparation de l'exercice : la transformation doit être tirée image par image. Appliquées à un lot entier, les transformations de `torchvision` tirent un seul jeu de paramètres pour toutes les images du lot (le même recadrage pour les 128), ce qui affaiblit beaucoup l'effet. L'exercice tire les paramètres dans la méthode `__getitem__` du jeu de données, une image à la fois.

L'augmentation ne date pas d'AlexNet : LeNet-5 s'entraînait déjà sur des chiffres déformés (0,8 % d'erreur au lieu de 0,95 %). AlexNet en fait un ingrédient central (recadrages 224×224 dans des images 256×256, retournements, variations de couleur) et note que sans elle le réseau surapprend nettement. D'autres régularisations qui modifient les exemples ou les cibles, comme mixup, sont au chapitre 12 ; l'apprentissage contrastif (chapitre 23) fera de l'augmentation le signal d'apprentissage lui-même : deux variantes d'une même image doivent avoir des représentations proches.

### 8. L'apprentissage par transfert

Un ResNet-50 entraîné sur ImageNet a appris, dans ses couches successives, des détecteurs de bords, de textures, de parties d'objets. Ces détecteurs ne sont pas propres aux 1 000 classes d'ImageNet : ils servent pour presque toute photo. L'apprentissage par transfert en profite : on garde le réseau, on remplace sa dernière couche (1 000 sorties) par une nouvelle couche adaptée à sa tâche, et on entraîne sur ses propres images, même peu nombreuses.

Deux façons de faire, de la plus économe à la plus complète :

- Tronc gelé (extraction de caractéristiques) : seuls les 75 813 paramètres de la nouvelle couche apprennent ($2\,048 \times 37 + 37$). Rapide, robuste avec très peu de données.
- réglage fin : tout le réseau continue d'apprendre, avec un taux d'apprentissage petit pour le tronc (pour ne pas effacer ce qu'il sait) et plus grand pour la nouvelle tête.

En 2014, Jason Yosinski et ses collègues mesurent couche par couche ce qui se transfère : les premières couches sont générales, les dernières de plus en plus spécifiques à la tâche d'origine, et le transfert des couches hautes se dégrade quand les deux tâches s'éloignent (ils le mesurent en séparant les classes d'ImageNet en objets naturels et objets fabriqués). On peut en déduire, sans que l'article le teste, que pour des radiographies ou des images satellite le tronc gelé suffira moins et qu'il faudra ajuster plus de couches.

L'exercice 13b (dans le cloud) compare sur les 37 races d'Oxford-IIIT Pet un ResNet-50 entraîné de zéro, tronc gelé et réglé finement. Ses seuils (moins de 50 % de zéro, au moins 85 % tronc gelé, au moins 88 % en réglage fin) sont des estimations prudentes, pas encore mesurées sur le GPU T4 de Colab. Ce schéma (pré-entraîner une fois sur beaucoup de données, adapter ensuite à peu de frais) est celui de toute la suite du parcours, jusqu'aux grands modèles de langage (chapitre 30).

### 9. Grad-CAM : où le réseau regarde-t-il ?

Un CNN répond « chat à 95 % ». Sur quelle partie de l'image s'appuie-t-il ? Grad-CAM (Ramprasaath Selvaraju et ses collègues, Virginia Tech, 2016-2017) donne une réponse approchée avec un seul calcul de gradient. On prend les cartes $A^k$ de la dernière couche de convolution (128 cartes de 8×8 dans l'exercice) : elles gardent encore une position, et décrivent déjà des motifs de haut niveau. On mesure combien chaque carte compte pour la classe $c$, puis on fait leur somme pondérée :

(Formule : Grad-CAM.)

Le code de l'exercice, sans l'affichage :

```python
import torch.nn.functional as F

def grad_cam(modele, x):
    A = modele.extraire(x)                         # cartes de la dernière convolution (B, 128, 8, 8)
    A.retain_grad()                                # garder ∂y/∂A après backward
    logits = modele.classer(A)                     # moyenne globale + couche linéaire
    c = logits.argmax(1)                           # classe prédite
    logits.gather(1, c[:, None]).sum().backward()  # gradient du score de cette classe
    alpha = A.grad.mean(dim=(2, 3), keepdim=True)  # α : un poids par carte
    cam = F.relu((alpha * A).sum(1, keepdim=True)) # somme pondérée, puis ReLU
    return F.interpolate(cam, size=32, mode="bilinear")  # 8×8 → 32×32
```

Sur les images synthétiques de la figure, la carte tombe bien sur l'objet : 65 % de la chaleur est dans une boîte qui ne couvre que 19 % de l'image. Sur les photos de CIFAR-10, l'exercice montre les limites : la carte vient d'une grille 8×8, elle est grossière ; elle couvre l'objet mais déborde, et parfois s'étale sur le contexte (pour le chat, elle couvre aussi la couverture sur laquelle il est couché). Grad-CAM montre ce qui a fait monter le score, pas ce qui le cause au sens du monde réel : si toutes les photos de bateaux de l'entraînement contiennent de l'eau, la carte d'un bateau peut tomber sur l'eau. C'est justement à cela qu'elle sert : détecter ce genre de raccourci. Les méthodes d'explication et leurs limites sont le sujet du chapitre 38.

### Ce que la méthode optimise

Un CNN de classification est entraîné exactement comme le MLP du chapitre 07 : sa dernière couche donne un score par classe, la softmax les transforme en probabilités, et on minimise l'entropie croisée multiclasse, moyennée sur des mini-lots, par descente de gradient. La convolution ne change que l'architecture, pas la perte. Pour un exemple de classe $y$ parmi $K$ classes, la perte vaut $-\ln p_y$ : 2,30 ($= \ln 10$) pour un réseau qui répond au hasard sur les 10 classes de CIFAR-10, 0,16 pour un réseau qui donne 85 % à la bonne classe. Fiche complète et comparaison avec les autres pertes au chapitre 01.

L'exercice 13a utilise l'optimiseur AdamW avec un taux d'apprentissage « one cycle » (montée puis descente, maximum $2 \times 10^{-3}$) et des lots de 128 (chapitre 02). La partie A (convolution à la main) n'optimise rien. Grad-CAM non plus : il réutilise le gradient du score, sans rien entraîner.

## MLP contre CNN sur CIFAR-10, en chiffres

La figure qui résume l'exercice 13a. CIFAR-10 : 60 000 photos de 32×32 pixels en couleur (50 000 pour l'entraînement, 10 000 pour le test), 10 classes (avion, automobile, oiseau, chat, cerf, chien, grenouille, cheval, bateau, camion), constituées par Alex Krizhevsky, Vinod Nair et Geoffrey Hinton et décrites dans le rapport technique de Krizhevsky (2009). Mêmes 15 époques, même optimiseur pour les trois modèles, seed 42, sur le GPU d'un Mac M1 Pro :

| Modèle | Param. | Entraîn. | Test | Miroir | Décalé 4 px |
|---|---|---|---|---|---|
| MLP | 1 707 274 | 87,7 % | 56,3 % | 54,8 % | 45,9 % |
| CNN | 290 282 | 99,1 % | 84,6 % | 84,6 % | 81,4 % |
| CNN + augm. | 290 282 | 89,9 % | 87,4 % | 87,6 % | 86,7 % |

(MLP : 3 072 → 512 → 256 → 10 ; CNN : 6 convolutions. « Miroir » : images de test retournées horizontalement. Durée par époque : 1,7 s pour le MLP, 10,4 s pour le CNN, 10,9 s avec augmentation.)

Ce qu'il faut y lire :

- Moins de paramètres, bien meilleur résultat : le partage des poids n'est pas une économie qui coûte, c'est une hypothèse juste sur les images.
- Le MLP surapprend aussi (87,7 % contre 56,3 %) : avec ses 1,7 million de poids liés chacun à un pixel précis, il apprend des positions plutôt que des formes.
- Le décalage de 4 pixels coûte 10,4 points au MLP, 3,2 au CNN. Le CNN n'est pas parfaitement invariant : les bords (marges de zéros) et le pooling laissent passer une partie du décalage.
- Le retournement horizontal ne gêne presque aucun modèle, même sans augmentation (− 1,5 point pour le MLP, 0 pour le CNN) : les objets de CIFAR-10 sont photographiés tournés d'un côté comme de l'autre, le miroir d'une image d'entraînement ressemble donc à d'autres images d'entraînement.
- Le CNN prend six fois plus de temps par époque que le MLP malgré ses six fois moins de paramètres : chaque poids d'un noyau sert à toutes les positions. Peu de paramètres ne veut pas dire peu de calcul.

## Ce qui coinçait

Les CNN ont dominé la vision de 2012 à 2020, et restent très utilisés. Leurs limites :

- Une vue locale. Un neurone ne relie deux régions éloignées de l'image qu'après de nombreuses couches, et même un champ récepteur plus grand que l'image ne garantit pas que l'information du bord atteigne le centre. Les Vision Transformers (chapitre 24) relient toutes les régions dès la première couche, au prix d'un biais inductif plus faible et donc d'un besoin de données plus grand.
- Une invariance approximative. Équivariants par translation, oui ; invariants aux rotations, aux changements d'échelle ou de point de vue, non : cela s'apprend par l'augmentation et par les données, imparfaitement. Un petit décalage peut encore changer une prédiction (− 3,2 points dans l'exercice sans augmentation).
- Des étiquettes en masse. ImageNet, c'est plus d'un million d'images annotées à la main. L'apprentissage par transfert repousse le problème vers une seule grande base étiquetée ; l'apprentissage auto-supervisé (chapitres 23 et 26) cherchera à s'en passer.
- Des raccourcis. Un CNN s'appuie volontiers sur la texture ou le fond quand ils prédisent la classe ; Grad-CAM aide à le voir, il ne l'empêche pas (chapitre 38).
- Une seule réponse par image. Un classifieur dit « il y a un chien », pas où est le chien, ni s'il y en a trois.

- Hubel et Wiesel (1962) : Dans le cortex visuel, des cellules réagissent à un bord orienté dans une petite zone, d’autres au même bord un peu partout.
- Neocognitron (Fukushima, 1980) : Empiler filtres locaux et regroupements donne une reconnaissance tolérante aux décalages, mais sans gradient.
- MLP sur les pixels (années 1980) : Chaque pixel relié à chaque neurone : aucune notion de voisinage, des poids pour chaque position.
- LeNet (LeCun et al., 1989-1998) : Un même petit filtre glissé partout, appris par rétropropagation : peu de paramètres, équivariance par translation.
- AlexNet (2012) : Le même principe, avec 1,2 million d’images, deux GPU, ReLU, dropout et augmentation de données.
- VGG, Inception (2014) : Plus profond : des piles de 3×3, des modules multi-échelles et des convolutions 1×1 qui mélangent les canaux.
- Highway, ResNet (2015) : y = x + F(x) : les couches n’apprennent que la correction, le gradient passe par le raccourci ; 152 couches.
- Transfert et Grad-CAM (2014-2017) : Réutiliser un réseau pré-entraîné pour une autre tâche ; regarder où il s’appuie.
- Détection et segmentation (2014-2015) : Ne plus seulement nommer l’image : localiser et délimiter chaque objet.

Et ensuite ? Classer ne suffit pas : une voiture autonome doit savoir où sont les piétons, un logiciel d'imagerie médicale doit délimiter une tumeur pixel par pixel. Les mêmes troncs convolutifs vont servir de base à des réseaux qui prédisent des boîtes autour des objets (R-CNN, YOLO) ou une classe par pixel (U-Net). C'est l'objet du chapitre 14 : Détection et segmentation.

## Exercice

Ce qu'il faut observer :

- La convolution à la main : boucles, version vectorisée, PyTorch et SciPy donnent le même résultat à $4{,}4 \times 10^{-16}$ près ; le pixel (15, 6) vaut 2,7048 avec Sobel x, comme dans la démo ; l'équivariance est exacte (écart 0) ; 2 432 paramètres pour une convolution 5×5 de 3 vers 32 canaux (avec biais) contre 100 696 064 pour la couche dense équivalente.
- MLP contre CNN : 56,3 % contre 84,6 % en test avec six fois moins de paramètres ; − 10,4 points pour le MLP sur les images décalées, − 3,2 pour le CNN.
- L'augmentation : l'écart entre entraînement et test passe de 14,5 à 2,5 points, le test gagne 2,8 points.
- Les filtres et Grad-CAM : des filtres de première couche presque informes (pas de beaux détecteurs de bords comme ceux d'AlexNet) ; des cartes Grad-CAM grossières, parfois posées sur le contexte.

Questions de réflexion : où sont passés les 1,4 million de paramètres économisés par le CNN, et que pourrait représenter le MLP que le CNN ne peut pas ? Pourquoi le CNN sans augmentation est-il insensible au retournement horizontal ? Quelle augmentation serait une erreur pour des chiffres manuscrits ?

Ouvrir le notebook dans Colab : récupérez `exercices/13b-cloud-transfer/notebook.ipynb` dans le dépôt du parcours, puis, dans Colab, Fichier › Importer un notebook (détails). Activez le GPU T4 (Exécution › Modifier le type d'exécution). Ce qu'on s'attend à observer (pas encore mesuré sur T4) : un écart net, dès la première époque, entre le réseau entraîné de zéro et les deux versions pré-entraînées ; 300 fois moins de paramètres appris avec le tronc gelé (ce point-là est un simple calcul) ; des cartes Grad-CAM plutôt posées sur la tête des animaux. La durée (25 à 35 min) et les seuils sont estimés : notez vos propres résultats.

## Auto-évaluation

Question : Une image 32×32×3 passe dans une couche de 16 noyaux 3×3, avec une marge de 1 et un pas de 2. Quelle est la taille de la sortie, et combien la couche a-t-elle de paramètres ?

Réponse : Taille : $\lfloor (32 + 2 \times 1 - 3)/2 \rfloor + 1 = \lfloor 15{,}5 \rfloor + 1 = 16$, donc 16 cartes de 16×16 (sortie 16×16×16). Paramètres : chaque noyau a une tranche 3×3 par canal d'entrée, soit $3 \times 3 \times 3 = 27$ poids, plus un biais ; $16 \times (27 + 1) = 448$ paramètres. Le nombre ne dépend pas de la taille de l'image : la même couche sur une photo de 1 000×1 000 en aurait toujours 448.

Question : Pourquoi décaler une image de 4 pixels fait-il perdre 10 points au MLP et seulement 3 au CNN de l’exercice ? Le CNN est-il pour autant invariant par translation ?

Réponse : Chaque poids du MLP est attaché à un pixel précis : décaler l'image envoie les mêmes motifs sur d'autres poids, que le réseau n'a pas entraînés à les reconnaître. Une convolution applique le même noyau partout : elle est équivariante, les cartes de caractéristiques se décalent avec l'image. La moyenne globale finale rend le résultat presque indépendant de la position. Mais pas tout à fait : les marges de zéros au bord et le max pooling (dont 23 % des sorties changent pour un décalage d'un pixel dans l'exercice) laissent passer une partie du décalage, d'où les 3,2 points perdus. Avec l'augmentation par recadrages aléatoires, la perte tombe à 0,7 point.

Question : VGG empile trois convolutions 3×3 au lieu d’une seule 7×7. Quel champ récepteur obtient-on, et que gagne-t-on, pour C canaux en entrée et en sortie ?

Réponse : Le champ récepteur grandit de $k - 1 = 2$ pixels par couche (pas de 1) : $3 \to 5 \to 7$, donc 7×7, comme une convolution 7×7. Mais il faut $3 \times (3 \times 3 \times C^2) = 27\,C^2$ poids au lieu de $49\,C^2$ (45 % de moins, sans compter les biais), et la pile a trois non-linéarités au lieu d'une : elle peut représenter des fonctions plus riches avec moins de paramètres.

Question : Pourquoi un réseau « simple » de 56 couches pouvait-il avoir une erreur d’entraînement plus élevée qu’un réseau de 20 couches, et comment le bloc résiduel règle-t-il le problème ?

Réponse : En théorie, les 36 couches en trop pourraient recopier leur entrée, mais une pile de convolutions suivies de ReLU n'apprend pas facilement l'identité : l'optimisation échoue (ce n'est pas du surapprentissage, puisque c'est l'erreur d'entraînement qui monte). Ce n'est pas non plus une disparition du gradient : les auteurs vérifient qu'avec la normalisation par lot, les gradients gardent une norme saine. Avec $y = F(x) + x$, recopier l'entrée revient à avoir $F = 0$, des poids proches de zéro, ce qui est facile ; et la dérivée $\partial y / \partial x = \partial F / \partial x + I$ contient un terme identité qui transmet le gradient d'un bloc au précédent, ce qui facilite l'optimisation.

Question : Un hôpital veut classer des radiographies pulmonaires et n’a que 2 000 images étiquetées. Que lui conseillez-vous en matière d’augmentation et de transfert, et à quoi Grad-CAM peut-il servir ?

Réponse : Partir d'un réseau pré-entraîné (ImageNet ou, mieux, un modèle pré-entraîné sur des images médicales) plutôt que de zéro ; comme les radiographies sont loin d'ImageNet, ajuster plus que la dernière couche (réglage fin avec un petit taux pour le tronc), et vérifier sur un jeu de test séparé. Augmentation : petits recadrages, contraste, légères rotations ; pas de retournement horizontal (le cœur est à gauche, et les clichés ont une orientation conventionnelle), pas de retournement vertical. Grad-CAM permet de vérifier que le réseau s'appuie sur les poumons et non sur un raccourci : une étiquette imprimée sur le cliché, un marqueur propre à un appareil ou à un service.

## Pour aller plus loin

- Articles originaux : Hubel et Wiesel (1962), Fukushima (1980), LeCun et al. (1989 et 1998 ; ce dernier, long, est très lisible dans ses sections II et III), Krizhevsky, Sutskever et Hinton (2012, neuf pages), Simonyan et Zisserman (2014), Szegedy et al. (2014), Srivastava et al. (2015), He et al. (2015), Selvaraju et al. (2016), Liu et al. (2022) : liens dans les fiches ci-dessus. Le bilan des cinq premières années d'ILSVRC : Russakovsky et al., « ImageNet Large Scale Visual Recognition Challenge » (2015).
- Stanford CS231n, *Deep Learning for Computer Vision* : le site du cours et les notes « Convolutional Neural Networks » (arithmétique des couches, architectures, avec une démo animée de la convolution multi-canaux) et « Transfer Learning ».
- 3Blue1Brown, « But what is a convolution? » : la convolution en général (probabilités, flou d'image, polynômes), avant le cas des réseaux.
- V. Dumoulin et F. Visin, « A guide to convolution arithmetic for deep learning », avec ses animations : marges, pas, et toutes les tailles de sortie.
- CNN Explainer (Georgia Tech) : un petit CNN qu'on inspecte couche par couche dans le navigateur ; Image Kernels explained visually (Setosa) : des noyaux 3×3 appliqués en direct, comme la démo de ce chapitre.
- Dive into Deep Learning : chapitre 7, « Convolutional Neural Networks » (dont LeNet), « Residual Networks (ResNet) and ResNeXt » et « Fine-Tuning », avec du code.
- C. Olah, « Conv Nets: A Modular Perspective » (2014), et Olah, Mordvintsev et Schubert, « Feature Visualization » (Distill, 2017) : ce que détectent les neurones des différentes couches.
- A. Karpathy, « What I learned from competing against a ConvNet on ImageNet » (2014) : l'expérience à l'origine de l'estimation de l'erreur humaine sur ImageNet.
- Documentation PyTorch, « Transfer Learning for Computer Vision Tutorial » : tronc gelé et réglage fin, pas à pas.
- Y. LeCun au Collège de France : chaire annuelle Informatique et sciences numériques (2015-2016), dont la leçon inaugurale « L'apprentissage profond : une révolution en intelligence artificielle », en français ; ses démos MNIST de LeNet, des années 1990.
