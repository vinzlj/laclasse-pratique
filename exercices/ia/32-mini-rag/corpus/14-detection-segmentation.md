# 14. Détection et segmentation

Repère historique (2013-2015 (précurseurs : 2001, 2005 ; suites : 2017, 2020, 2023)) : Localiser et délimiter les objets, pas seulement nommer l’image : dire où se trouve chaque objet (une boîte), combien il y en a, et quels pixels lui appartiennent (un masque).

Bien avant les réseaux profonds, on savait chercher un objet précis dans une image en promenant un classifieur sur toutes les positions et toutes les tailles : en 2001, Paul Viola et Michael Jones détectent ainsi des visages avec une cascade de classifieurs boostés (chapitre 08) ; en 2005, Navneet Dalal et Bill Triggs détectent des piétons avec des histogrammes de gradients orientés (HOG). Depuis 2005, le concours PASCAL VOC compare les détecteurs, sur 20 classes d'objets courants à partir de 2007. Vers 2010-2012, les progrès y stagnent. En novembre 2013, Ross Girshick et ses collègues de Berkeley branchent sur ce problème le réseau convolutif d'AlexNet (chapitre 13) : leur méthode, R-CNN, améliore de plus de 30 % en relatif le meilleur score publié. En 2015, la détection devient plus rapide (Fast puis Faster R-CNN), puis tient en un seul passage du réseau (YOLO, à 45 images par seconde). La même année, à Fribourg, Olaf Ronneberger, Philipp Fischer et Thomas Brox publient U-Net, un réseau qui donne une classe à chaque pixel d'une image de microscopie après un entraînement sur seulement 30 images annotées.

## L'idée en une image

Un classifieur d'images (chapitre 13) répond à une seule question : « qu'y a-t-il sur cette photo ? ». Il rend une étiquette pour toute l'image. Sur la photo ci-dessous, il dirait peut-être « ordinateur portable ». Mais il y a aussi deux bouteilles, une tasse, une souris, une chaise, et la réponse ne dit pas où ils sont.

Trois tâches répondent à trois questions de plus en plus fines :

- La détection d’objets donne la liste des objets présents. Pour chacun, elle fournit une boîte englobante, une classe et un score de confiance, et le nombre d'objets n'est pas fixé à l'avance. Comme une légende, mais avec un cadre autour de chaque objet.
- La segmentation sémantique donne une classe à chaque pixel : route, trottoir, piéton, ciel. Comme un coloriage par numéros. Deux piétons qui se touchent forment une seule tache « piéton ».
- La segmentation d’instances fait les deux : un masque par objet, même si deux objets de la même classe se chevauchent.

| | Avant : classification (ch. 13) | Après : détection | Après : segmentation sémantique |
|---|---|---|---|
| Question | qu'y a-t-il ? | quoi, où, combien ? | quelle classe pour chaque pixel ? |
| Sortie | 1 classe pour l'image | une liste de longueur variable de (boîte, classe, score) | une image de classes, de la taille de l'entrée |
| Décisions pour une image 64×64 (exercice) | 1 | autant que d'objets (1 à 5) | 4 096, une par pixel |
| Annotation d'un exemple | un mot | un rectangle par objet | un contour précis par objet |
| Mesure de référence | exactitude | précision moyenne (AP) à un seuil d'IoU | IoU par classe (mIoU) |
| Ce qui coince | ne localise rien | des boîtes grossières, des doublons à éliminer | ne sépare pas deux objets qui se touchent |

(L'IoU, l'AP et la mIoU sont définies dans la suite du chapitre : sections 1, 2 et 7 du mécanisme.)

Le point commun : un réseau convolutif sert toujours de « tronc » qui extrait les caractéristiques de l'image. Ce qui change, c'est la tête posée dessus et la forme de la sortie. Pour la détection, il faut prédire des coordonnées et gérer un nombre variable d'objets. Pour la segmentation, il faut rendre une image aussi fine que l'entrée, alors que le tronc a justement réduit la résolution (32×32 → 8×8 dans le CNN de l'exercice 13a).

## Le mécanisme

### 1. Une boîte, et une mesure de recouvrement : l'IoU

Une boîte est décrite par quatre nombres. Les bibliothèques utilisent deux conventions : deux coins $(x_0, y_0, x_1, y_1)$, ou un centre, une largeur et une hauteur $(x, y, w, h)$. Pour dire si une boîte prédite est « juste », il faut mesurer à quel point elle recouvre la vraie. La mesure standard est l'intersection sur union :

(Formule : Intersection sur union.)

Exemple vérifiable à la main (celui de l'exercice). La vraie boîte est $A = (0, 0, 10, 10)$, un carré de 10 pixels de côté, d'aire 100.

- Prédiction décalée de 5 pixels en diagonale, $B = (5, 5, 15, 15)$ : l'intersection est le carré $(5, 5, 10, 10)$, d'aire 25 ; l'union vaut $100 + 100 - 25 = 175$ ; IoU $= 25/175 = 0{,}14$.
- Décalée de 2 pixels vers la droite, $(2, 0, 12, 10)$ : intersection $8 \times 10 = 80$, union $200 - 80 = 120$, IoU $= 0{,}67$.
- La moitié gauche de $A$, $(0, 0, 5, 10)$ : intersection 50, union 100, IoU $= 0{,}5$.

Pourquoi un rapport d'aires plutôt qu'une distance entre les coins ? Parce qu'il ne dépend pas de la taille de l'objet. Pour une boîte carrée de côté $a$ décalée de $d$ pixels, l'intersection vaut $a(a - d)$ et l'union $a(a + d)$, donc IoU $= (a - d)/(a + d)$. Une erreur de 2 pixels coûte beaucoup sur un objet de 10 pixels (0,67) et peu sur un objet de 40 pixels (0,90). C'est voulu : 2 pixels d'écart, c'est grave pour un petit objet et négligeable pour un grand.

Le code de l'exercice calcule toutes les IoU entre $N$ boîtes et $M$ boîtes d'un coup, par diffusion NumPy :

```python
def iou_boites(a, b):
    """IoU de chaque boîte de a (N, 4) avec chaque boîte de b (M, 4) → (N, M)."""
    # Rectangle d'intersection : le plus grand des coins hauts gauches,
    # le plus petit des coins bas droits ((N, 1) contre (1, M))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)  # 0 si disjointes
    aire_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    aire_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aire_a[:, None] + aire_b[None, :] - inter)
```

Sur 200 boîtes aléatoires, l'écart avec `torchvision.ops.box_iou` est exactement nul.

### 2. Juger un détecteur : précision, rappel et AP

Un détecteur rend une liste de boîtes notées. Pour l'évaluer, on trie ses détections par score décroissant et on les apparie une à une aux vrais objets :

- une détection est un vrai positif si elle recouvre un vrai objet de sa classe, pas encore apparié, avec une IoU supérieure à 0,5 ;
- sinon c'est un faux positif. Cela inclut un doublon : une deuxième boîte sur un objet déjà trouvé est comptée fausse.

C'est la règle du concours PASCAL VOC. Son article de référence précise que le seuil de 0,5 a été choisi « délibérément bas », parce que les boîtes annotées à la main sont elles-mêmes imprécises : où s'arrête la boîte d'une personne bras et jambes écartés ?

En descendant la liste triée, on suit la précision (part des détections justes parmi celles retenues) et le rappel (part des vrais objets trouvés), deux mesures définies au chapitre 00. La précision moyenne résume la courbe précision-rappel par l'aire sous cette courbe.

Exemple. Trois vrais chats dans le jeu de test, cinq détections triées par score : juste, juste, fausse, juste, fausse.

```text
rang  verdict   précision      rappel
 1    juste     1/1 = 1,00     1/3
 2    juste     2/2 = 1,00     2/3
 3    fausse    2/3 = 0,67     2/3
 4    juste     3/4 = 0,75     3/3
 5    fausse    3/5 = 0,60     3/3
```

À chaque niveau de rappel, on retient la meilleure précision atteinte à ce rappel ou au-delà (la courbe « interpolée ») : 1 jusqu'à un rappel de 2/3, puis 0,75 jusqu'à 1. L'aire vaut $\tfrac13 \times 1 + \tfrac13 \times 1 + \tfrac13 \times 0{,}75 = 0{,}917$. On fait la moyenne sur les classes pour obtenir la mAP (*mean average precision*). On écrit « mAP@0,5 » pour préciser le seuil d'IoU. En simplifiant : la façon d'interpoler varie selon les outils (l'article de PASCAL VOC fait la moyenne sur 11 niveaux de rappel, 0 ; 0,1 ; … ; 1 ; l'exercice utilise tous les points). COCO fait la moyenne de l'AP sur dix seuils d'IoU, de 0,5 à 0,95 par pas de 0,05, ce qui récompense les boîtes précises.

### 3. La détection en deux étapes : la famille R-CNN

Avant : la fenêtre glissante. On entraîne un classifieur « objet ou pas » sur des imagettes, puis on le promène partout. Une seule taille de fenêtre de 64×64 pixels, déplacée de 8 en 8 sur une image de 640×480, donne déjà $73 \times 53 = 3\,869$ positions. Il faut multiplier par le nombre de tailles et de proportions essayées. C'est faisable avec un classifieur très rapide (les visages de Viola et Jones, les piétons de HOG), pas avec un gros CNN appliqué à chaque fenêtre.

R-CNN (2013) : ne regarder que les régions prometteuses. L'idée est de séparer le problème en deux étapes, d'où le nom de détecteur à deux étapes. D'abord, un algorithme sans apprentissage, Selective Search (Uijlings et al., 2013), regroupe les pixels voisins semblables et produit environ 2 000 propositions de régions par image : des rectangles qui contiennent « peut-être un objet ». Ensuite, chaque région est redimensionnée en 227×227 et passe dans le CNN d'AlexNet.

Deux idées font son succès :

- Le transfert. PASCAL VOC compte trop peu d'images pour entraîner un gros CNN de zéro. On part d'AlexNet pré-entraîné sur ImageNet, puis on le règle finement sur les régions de PASCAL : c'est l'apprentissage par transfert du chapitre 13. D'après l'article, le réglage fin rapporte à lui seul 8 points de mAP.
- La correction des boîtes. Une région proposée tombe rarement pile sur l'objet. Une régression de boîtes apprend à la décaler et à l'étirer.

Résultat : 53,3 % de mAP sur VOC 2012, plus de 30 % de mieux en relatif que le meilleur système précédent. Le prix est la lenteur. Calculer les propositions et les caractéristiques des 2 000 régions prend 13 secondes par image sur GPU (53 s sur processeur).

Ce que prédit la régression de boîtes. Le réseau ne prédit pas directement les coordonnées de la vraie boîte $G$. Il prédit des corrections relatives à la boîte proposée $P$ :

(Formule : Cibles de la régression de boîtes (R-CNN, puis Fast et Faster R-CNN).)

Exemple : une région proposée de centre (50, 40), de 40 pixels de large et 20 de haut ; le vrai objet a pour centre (54, 38) et mesure 50 × 20. Les cibles valent $t_x = 4/40 = 0{,}1$, $t_y = -2/20 = -0{,}1$, $t_w = \ln(50/40) = 0{,}223$ et $t_h = \ln 1 = 0$. Diviser par la taille rend la cible indépendante de l'échelle : un décalage de 0,1 largeur veut dire la même chose pour un objet de 40 ou de 400 pixels. Le logarithme rend « deux fois plus grand » ($+0{,}693$) et « deux fois plus petit » ($-0{,}693$) symétriques.

Fast R-CNN (2015) : un seul passage du CNN. Les 2 000 régions d'une image se recouvrent largement, et R-CNN recalcule 2 000 fois des convolutions presque identiques. Ross Girshick, passé chez Microsoft Research, inverse l'ordre des opérations. Le CNN traite l'image entière une fois. Chaque région est ensuite découpée dans la carte de caractéristiques et ramenée à une grille fixe de 7×7 par un max pooling, le « RoI pooling » (*region of interest*). Une même tête dense donne à la fois la classe et la correction de boîte, entraînées ensemble. Avec VGG16, l'entraînement est 9 fois plus rapide que celui de R-CNN, le test 146 fois plus rapide (213 fois avec une décomposition en valeurs singulières tronquée des couches denses), et la mAP sur VOC 2012 passe de 62 à 66 %.

Faster R-CNN (2015) : le réseau propose lui-même les régions. Il reste un goulot : Selective Search prend environ 2 secondes par image sur processeur. Shaoqing Ren, Kaiming He, Ross Girshick et Jian Sun le remplacent par un petit réseau posé sur la même carte de caractéristiques, le RPN (*Region Proposal Network*). À chaque position de la carte, il considère 9 ancres, des boîtes de référence de 3 tailles et 3 proportions. Pour chacune, il prédit un score « objet ou fond » et une correction, avec les mêmes cibles $t$ que ci-dessus. Une carte d'environ 2 400 positions donne ainsi $2\,400 \times 9 = 21\,600$ ancres. Après tri et élimination des doublons, on en garde 300 comme propositions. Le système tourne à 5 images par seconde avec VGG16, toutes étapes comprises. L'ancre est l'idée qui va rester : RetinaNet (section 5) et bien d'autres détecteurs la reprendront.

Éliminer les doublons : la NMS. Un détecteur produit presque toujours plusieurs boîtes très proches autour d'un même objet. Les règles d'évaluation comptent toutes ces boîtes sauf une comme des fautes. La suppression des non-maxima fait le ménage, classe par classe :

1. trier les boîtes par score décroissant ;
2. garder la première ;
3. supprimer toutes les boîtes restantes qui la recouvrent avec une IoU supérieure au seuil (souvent 0,5) ;
4. recommencer avec la mieux notée de celles qui restent.

Exemple : A = (0, 0, 10, 10) notée 0,9 ; B = (1, 1, 11, 11) notée 0,8 ; C = (20, 20, 30, 30) notée 0,7. On garde A. L'IoU de A et B vaut $81/119 = 0{,}68$, au-dessus de 0,5 : B est supprimée. C ne recouvre pas A : elle est gardée. Il reste A et C.

```python
def nms(boites, scores, seuil=0.5):
    ordre = np.argsort(-scores)              # indices, du meilleur score au pire
    gardees = []
    while ordre.size > 0:
        i = ordre[0]
        gardees.append(i)
        recouvrement = iou_boites(boites[i:i + 1], boites[ordre[1:]])[0]
        ordre = ordre[1:][recouvrement <= seuil]   # ne garder que les boîtes assez éloignées
    return gardees
```

Sur les 200 boîtes aléatoires de l'exercice, cette fonction garde les mêmes 141 boîtes que `torchvision.ops.nms`. La NMS est une règle écrite à la main, pas apprise. Son seuil est un compromis : trop bas, il supprime l'une de deux personnes qui se chevauchent vraiment ; trop haut, il laisse passer des doublons. Dans l'exercice, 108 boîtes candidates autour de 18 objets donnent 23 boîtes après une NMS à 0,3 : sur un petit objet, deux doublons dont chaque coordonnée est décalée de 3 pixels au plus (donc jusqu'à 6 pixels l'un de l'autre) peuvent avoir une IoU inférieure à 0,3.

### 4. La détection en une étape : YOLO

En juin 2015, Joseph Redmon (Université de Washington) et ses coauteurs, dont Ross Girshick, posent la question autrement. Pourquoi classer des régions une par une ? On peut traiter la détection comme une seule régression de l'image entière vers un tableau de nombres. C'est le détecteur à une étape. Le nom le résume : *You Only Look Once*, on ne regarde l'image qu'une fois.

L'image est découpée en une grille de $S \times S = 7 \times 7$ cases. Chaque case est responsable des objets dont le centre tombe chez elle. Elle prédit $B = 2$ boîtes, chacune décrite par 5 nombres : $x, y, w, h$ et une confiance. Elle prédit aussi $C = 20$ probabilités de classe (les 20 classes de PASCAL VOC). La sortie du réseau est donc un tenseur de

$$
S \times S \times (5B + C) = 7 \times 7 \times 30 = 1\,470 \text{ nombres},
$$

soit 98 boîtes par image ($49 \times 2$), produites en un seul passage d'un CNN de 24 couches de convolution suivies de 2 couches denses. La confiance d'une boîte est entraînée à valoir $\Pr(\text{objet}) \times \mathrm{IoU}$ avec la vraie boîte : 0 s'il n'y a pas d'objet, l'IoU obtenue sinon. Une NMS finale retire les doublons.

La perte de YOLO. Tout est appris par une somme d'erreurs quadratiques, que l'article choisit parce qu'elle est « facile à optimiser », tout en reconnaissant qu'elle ne correspond pas exactement à la précision moyenne visée. Deux pondérations corrigent ses défauts :

(Formule : Perte de YOLO (2015), écrite en simplifiant les notations de l’article.)

La racine carrée de $w$ et $h$ joue le même rôle que l'IoU : 2 pixels d'erreur sur une boîte de 10 pixels comptent plus que sur une boîte de 100. Même avec $\lambda_{\text{noobj}}$, on voit déjà ici le problème du fond, qui reviendra : la plupart des boîtes ne contiennent rien.

Le compromis. Sur VOC 2007, YOLO atteint 63,4 % de mAP à 45 images par seconde. Faster R-CNN (VGG-16) fait 73,2 %, mais à 7 images par seconde (mesure de l'article YOLO, sur un autre GPU que les 5 images/s annoncées par l'article Faster R-CNN). L'analyse des erreurs dans l'article est nette : YOLO localise moins bien que Fast R-CNN, mais confond beaucoup moins souvent le fond avec un objet, parce qu'il voit toute l'image d'un coup. Sa grille impose aussi une limite : 2 boîtes et une seule classe par case. Des petits objets serrés, comme une volée d'oiseaux, lui échappent.

| | Deux étapes (Faster R-CNN, 2015) | Une étape (YOLO, 2015) |
|---|---|---|
| Principe | proposer des régions, puis classer et corriger chacune | prédire boîtes et classes partout en un passage |
| VOC 2007 (table de l'article YOLO) | 73,2 % de mAP, 7 images/s | 63,4 % de mAP, 45 images/s |
| Point fort | précision, petits objets | vitesse, peu de fausses détections sur le fond |
| Point faible | lent, pipeline complexe | localisation moins précise, objets groupés |
| Suite | Mask R-CNN (2017) | SSD, RetinaNet (2017), YOLO v2 et suivants |

### 5. Le fond écrase tout : la perte focale

Un détecteur à une étape évalue toutes les positions de l'image, et presque toutes ne contiennent que du fond. En 2017, Tsung-Yi Lin et ses collègues de Facebook AI Research (FAIR) chiffrent ce déséquilibre des classes : environ 100 000 positions candidates par image, et un rapport fond/objet de l'ordre de 1 000 pour 1. Ils y voient « la cause centrale » du retard des détecteurs à une étape sur les détecteurs à deux étapes. Dans ces derniers, l'étape de proposition ne garde que 1 000 à 2 000 régions et élimine ainsi la plupart du fond avant la classification.

Pourquoi c'est grave, en chiffres. Prenons 1 000 positions de fond faciles, que le réseau classe déjà bien (probabilité de la bonne classe $p_t = 0{,}9$), et un objet difficile ($p_t = 0{,}1$). Avec l'entropie croisée du chapitre 01, chaque position coûte $-\ln p_t$ :

```text
                          entropie croisée          focale (γ = 2)
1 000 fonds faciles       1 000 × 0,1054 = 105,4    1 000 × 0,00105 = 1,05
1 objet difficile                   2,303                     1,865
part de l'objet                       2 %                      64 %
```

Avec l'entropie croisée, l'objet pèse 2 % de la perte totale. Le gradient sert surtout à rendre le fond encore un peu plus « fond ». En simplifiant (c'est le gradient qui compte, pas la valeur de la perte, mais les deux suivent ici la même tendance) : le signal utile est noyé.

La perte focale multiplie l'entropie croisée par un facteur qui s'éteint quand l'exemple est déjà bien classé :

(Formule : Perte focale (Lin et al., 2017).)

Avec $\gamma = 2$, un exemple à $p_t = 0{,}9$ est multiplié par $(0{,}1)^2 = 0{,}01$ : sa perte est 100 fois plus petite qu'avec l'entropie croisée. Un exemple à $p_t = 0{,}5$ n'est divisé que par 4. L'exercice calcule ces valeurs :

| $p_t$ | 0,9 (facile) | 0,5 | 0,1 (difficile) | rapport facile / difficile |
|---|---|---|---|---|
| $\gamma = 0$ (entropie croisée) | 0,105 | 0,693 | 2,303 | 0,046 |
| $\gamma = 1$ | 0,0105 | 0,347 | 2,072 | 0,005 |
| $\gamma = 2$ | 0,00105 | 0,173 | 1,865 | 0,0006 |

Un deuxième détail de l'article compte autant. Au début de l'entraînement, un réseau initialisé au hasard prédit $p \approx 0{,}5$ partout, soit une perte de $\ln 2 = 0{,}693$ par position. Sur 100 000 positions de fond, cela fait environ 69 000 de perte au premier pas, assez pour déstabiliser l'entraînement. Les auteurs initialisent donc le biais de la dernière couche pour que le réseau prédise $p = 0{,}01$ (« objet ») partout dès le départ. La perte d'une position de fond tombe à $-\ln 0{,}99 = 0{,}01$.

Leur détecteur, RetinaNet, est à une étape avec des ancres. Il atteint 39,1 d'AP sur COCO à 5 images par seconde (ResNet-101) et dépasse les meilleurs détecteurs à deux étapes publiés jusque-là.

Ce que l’exercice montre, et ne montre pas :
Dans l'exercice, la perte focale (γ = 2, sans α ni initialisation du biais) fait moins bien que l'entropie croisée en segmentation : 74,9 % de mIoU contre 85,0 %, avec un apprentissage plus lent (28 % à la 3ᵉ époque contre 45 %). Le problème n'est pas celui de l'article. Le fond y occupe 90 % des pixels (un rapport de 9 pour 1, pas 1 000 pour 1), et il n'y a que 6 époques. Éteindre les pixels faciles ralentit surtout l'apprentissage. C'est un constat sur ce réglage et ce problème, pas une mesure générale de la perte focale.

### 6. Une classe par pixel : FCN puis U-Net

Du classifieur à la carte de classes. Fin 2014, Jonathan Long, Evan Shelhamer et Trevor Darrell (Berkeley) remarquent qu'une couche dense n'est qu'une convolution dont le noyau couvre toute la carte. En réécrivant les couches denses d'un classifieur ImageNet comme des convolutions, on obtient un réseau entièrement convolutif. Il accepte une image de n'importe quelle taille et rend, au lieu d'un vecteur de scores, une carte de scores : une prédiction par zone de l'image. Elle est grossière, puisque le tronc a divisé la résolution par 32. On la ramène à la taille de l'image avec des couches de suréchantillonnage : les intermédiaires sont apprises (initialisées en interpolation bilinéaire), la dernière reste une interpolation bilinéaire fixe. Le réseau est entraîné avec une entropie croisée par pixel. Pour affiner les contours, on combine la carte grossière du fond du réseau avec celles de couches moins profondes, plus fines : ce sont des connexions de saut. Résultat : 62,2 % d'IoU moyenne sur VOC 2012 (+ 20 % en relatif), en moins d'un cinquième de seconde par image.

Le dilemme. Pour savoir ce qu'est une forme, il faut du contexte : un grand champ récepteur (chapitre 13), donc beaucoup de réductions de taille. Pour savoir où passe exactement son bord, il faut la résolution d'origine, que ces réductions ont détruite. Un max pooling 2×2 garde le maximum d'un bloc de 4 pixels mais oublie lequel des quatre c'était.

U-Net (mai 2015) : un encodeur, un décodeur, et des ponts. Ronneberger, Fischer et Brox reprennent l'idée de Long et al. et la rendent symétrique :

- un encodeur (chemin descendant) : à chaque étage, deux convolutions 3×3 puis un max pooling. La résolution est divisée par 2, le nombre de canaux multiplié par 2 ;
- un décodeur (chemin montant) : à chaque étage, une convolution transposée double la résolution, puis deux convolutions 3×3 ;
- des ponts : à chaque étage, les cartes de l'encodeur sont recopiées et concaténées à celles du décodeur de même taille. Le décodeur reçoit à la fois le « quoi » qui remonte du fond du U et le « où » précis de l'encodeur ;
- une convolution 1×1 finale donne un score par classe et par pixel.

C'est le U-Net de l'exercice, en plus petit (0,48 M de paramètres). Les tailles, étage par étage, pour une image de 64×64 :

```text
encodeur                                  décodeur
64×64×3   → bloc → 64×64×16  ─────── pont ─────→ [64×64×16 | 64×64×16] → bloc → 64×64×16 → conv 1×1 → 64×64×4
  pool ↓                                                ↑ conv. transposée
32×32×16  → bloc → 32×32×32  ─────── pont ─────→ [32×32×32 | 32×32×32] → bloc → 32×32×32
  pool ↓                                                ↑
16×16×32  → bloc → 16×16×64  ─────── pont ─────→ [16×16×64 | 16×16×64] → bloc → 16×16×64
  pool ↓                                                ↑
 8×8×64   → bloc →  8×8×128  (fond du U) ───────────────┘
```

La convolution transposée. C'est l'opération inverse, pour les tailles, d'une convolution à pas. Chaque valeur d'entrée est multipliée par le noyau entier, et le résultat est déposé dans la sortie à la position correspondante, en avançant du pas. Là où deux dépôts se chevauchent, on additionne. En 1D, avec l'entrée [1, 2], le noyau [1 ; 0,5] et un pas de 2 :

```text
entrée      1          2
dépôt de 1  1   0,5
dépôt de 2             2   1
sortie      1   0,5    2   1        (taille (2 − 1) × 2 + 2 = 4)
```

La taille de sortie vaut $(n - 1)\,s - 2p + k$. Dans l'exercice, $n = 8$, $s = 2$, $p = 0$, $k = 2$ : $7 \times 2 + 2 = 16$. Les poids du noyau sont appris comme ceux d'une convolution ordinaire. Le nom « déconvolution », qu'on rencontre aussi (y compris dans l'article FCN), est trompeur : l'opération n'inverse pas une convolution. Quand le noyau n'est pas un multiple du pas (noyau 3, pas 2), les dépôts se chevauchent inégalement et peuvent dessiner un damier dans l'image : c'est l'objet d'un article de Distill cité en fin de chapitre.

Le code du réseau de l'exercice, sans les détails :

```python
class UNet(nn.Module):
    def __init__(self, n_classes=4, c=16, ponts=True):
        super().__init__()
        self.ponts = ponts
        self.enc1, self.enc2, self.enc3 = Bloc(3, c), Bloc(c, 2*c), Bloc(2*c, 4*c)
        self.fond = Bloc(4*c, 8*c)                              # 8×8, le bas du U
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(8*c, 4*c, 2, stride=2)    # 8 → 16
        self.up2 = nn.ConvTranspose2d(4*c, 2*c, 2, stride=2)    # 16 → 32
        self.up1 = nn.ConvTranspose2d(2*c, c, 2, stride=2)      # 32 → 64
        k = 2 if ponts else 1                                   # la concaténation double les canaux
        self.dec3, self.dec2, self.dec1 = Bloc(k*4*c, 4*c), Bloc(k*2*c, 2*c), Bloc(k*c, c)
        self.tete = nn.Conv2d(c, n_classes, 1)                  # un score par classe et par pixel

    def monter(self, x, saut):
        return torch.cat([x, saut], dim=1) if self.ponts else x  # le pont : copie + concaténation

    def forward(self, x):
        e1 = self.enc1(x)                                       # 64×64×c
        e2 = self.enc2(self.pool(e1))                           # 32×32×2c
        e3 = self.enc3(self.pool(e2))                           # 16×16×4c
        f = self.fond(self.pool(e3))                            # 8×8×8c
        d3 = self.dec3(self.monter(self.up3(f), e3))
        d2 = self.dec2(self.monter(self.up2(d3), e2))
        d1 = self.dec1(self.monter(self.up1(d2), e1))
        return self.tete(d1)                                    # (lot, n_classes, 64, 64)
```

(`Bloc` : deux convolutions 3×3 avec marge, chacune suivie d'une normalisation par lot du chapitre 12 et d'une ReLU, chapitre 07.)

Ce que mesure l'exercice. Le même réseau sans ses ponts (`ponts=False`, 0,43 M de paramètres) est un encodeur-décodeur simple. Il doit reconstruire un contour net à partir de la seule carte 8×8 du fond du U. Après 6 époques :

| | U-Net | Sans ponts | Écart |
|---|---|---|---|
| mIoU objets | 85,0 % | 77,9 % | + 7,1 points |
| Pixels de bord justes (5,6 % de l'image) | 86,1 % | 75,9 % | + 10,2 points |
| Pixels loin des bords justes | 99,65 % | 99,52 % | + 0,1 point |
| IoU triangle | 84,3 % | 71,5 % | + 12,8 points |

Les ponts ne servent presque à rien à l'intérieur des objets. Ils servent aux contours, et surtout aux angles : sans eux, le réseau arrondit les coins des carrés et des triangles.

L'article d'origine est taillé pour la biologie, où les images annotées sont rares et chères. Quelques choix à connaître, parce que beaucoup de réimplémentations les changent :

- 23 couches de convolution, 64 à 1 024 canaux, et des convolutions sans marge : chaque convolution 3×3 rogne un pixel de chaque côté. Une tuile de 572×572 en entrée donne une carte de 388×388 en sortie, et les cartes de l'encodeur sont recadrées avant concaténation. Une grande image est traitée tuile par tuile. L'exercice, comme la plupart des versions actuelles, met une marge et garde la taille ;
- l'entraînement sur 30 images de microscopie électronique (512×512) s'appuie sur une forte augmentation de données (chapitre 13), surtout des déformations élastiques aléatoires, plausibles pour des tissus ;
- pour séparer des cellules qui se touchent, une carte de poids donne beaucoup plus d'importance dans l'entropie croisée aux fins liserés de fond entre deux cellules ($w_0 = 10$, $\sigma \approx 5$ pixels).

Le réseau gagne le concours de suivi de cellules ISBI 2015 et segmente une image de 512×512 en moins d'une seconde sur GPU.

### 7. Mesurer une segmentation : la précision par pixel ment

Dans l'exercice, le fond occupe 90,4 % des pixels. Un « réseau » qui répond fond partout obtient 90,5 % de pixels justes et une IoU nulle sur chaque objet. Entre la pire et la meilleure variante entraînée, la précision par pixel ne bouge que d'un point (98,2 → 99,2 %), alors que l'IoU moyenne sur les objets passe de 75 à 89 %. On mesure donc l'IoU par classe, pixels prédits contre pixels vrais, puis on fait la moyenne sur les classes : la mIoU.

L'autre mesure courante est le coefficient de Dice, venu de l'écologie (Lee Dice, 1945, pour mesurer à quel point deux espèces fréquentent les mêmes lieux) :

(Formule : Coefficient de Dice, et son lien avec l’IoU.)

Deux disques de rayon 10 décalés de 8 pixels : IoU 0,346, Dice $2 \times 0{,}346 / 1{,}346 = 0{,}514$. Le Dice est toujours au moins égal à l'IoU, et les deux classent les prédictions dans le même ordre. Il pardonne simplement plus.

La perte Dice. En 2016, Fausto Milletari, Nassir Navab et Seyed-Ahmad Ahmadi (Munich) entraînent V-Net, un encodeur-décodeur 3D proche de U-Net (blocs résiduels, convolutions à pas au lieu du max pooling), sur des IRM de la prostate, en maximisant directement le Dice, rendu dérivable en remplaçant le masque prédit par les probabilités. Le Dice ne compte que les pixels de l'objet, prédits ou vrais. Les milliers de pixels de fond bien classés n'apparaissent pas au numérateur, et au dénominateur seulement par leur faible probabilité résiduelle d'être un objet. Cela rend le Dice peu sensible au déséquilibre fond/objet. L'exercice ajoute à l'entropie croisée la perte $1 - \text{Dice}$, moyennée sur les trois classes d'objets :

```python
def perte_dice(logits, cible, eps=1.0):
    p = logits.softmax(dim=1)[:, 1:]                       # probabilités des 3 classes d'objets
    y = F.one_hot(cible, 4).permute(0, 3, 1, 2)[:, 1:]     # masques vrais (0 ou 1), fond exclu
    inter = (p * y).sum(dim=(0, 2, 3))                     # « intersection » douce, par classe
    somme = p.sum(dim=(0, 2, 3)) + y.sum(dim=(0, 2, 3))
    return 1 - ((2 * inter + eps) / (somme + eps)).mean()  # eps : pas de 0/0 si une classe manque

perte = F.cross_entropy(logits, cible) + perte_dice(logits, cible)
```

C'est la meilleure variante de l'exercice : 89,2 % de mIoU (contre 85,0 % pour l'entropie croisée seule), et 72 % dès la 3ᵉ époque (contre 45 %).

### 8. Séparer les objets : segmentation d'instances, et des masques aux boîtes

La segmentation sémantique ne sait pas que deux disques qui se touchent sont deux objets : elle colorie une seule tache « cercle ». En 2017, Kaiming He, Georgia Gkioxari, Piotr Dollár et Ross Girshick (FAIR) ajoutent à Faster R-CNN une troisième branche, à côté de la classe et de la correction de boîte. Pour chaque région détectée, elle prédit un petit masque, par exemple de 28×28. Ce masque est ensuite agrandi à la taille de la boîte. C'est Mask R-CNN, à 5 images par seconde. Un détail compte pour la précision des masques : le RoI pooling de Fast R-CNN arrondit les coordonnées des régions à la case entière. Le RoIAlign de Mask R-CNN les garde exactes et interpole.

L'exercice fait le chemin inverse, plus modeste. Dans ses images, deux objets ne se touchent jamais. Chaque tache connexe du masque prédit est donc un objet, et sa boîte englobante se lit directement sur le masque. Ces boîtes, notées par la probabilité moyenne de leur classe, atteignent une mAP@0,5 de 90,1 % sans aucune tête de régression (93,6 % avec la perte Dice). Les erreurs restantes viennent de taches coupées en deux et de petites formes mal classées, par exemple un petit carré pris pour un cercle.

### 9. Après 2017 : les Transformers arrivent, et U-Net reviendra

DETR (2020). Nicolas Carion, Francisco Massa et leurs collègues de Facebook AI posent la détection comme la prédiction d'un ensemble. Un Transformer (chapitre 20) produit directement un nombre fixe de prédictions (100 dans l'article), chacune « un objet » ou « rien ». L'entraînement apparie chaque vrai objet à une seule prédiction, par un appariement optimal (algorithme hongrois). Il n'y a plus ni ancres ni NMS : les doublons sont pénalisés par la perte elle-même. Sur COCO, DETR atteint 42,0 d'AP, autant qu'un Faster R-CNN bien réglé de taille comparable.

Segment Anything (2023). Alexander Kirillov et l'équipe de Meta AI entraînent un modèle de segmentation sur un jeu de plus d'un milliard de masques répartis sur 11 millions d'images, dont 99,1 % ont été produits automatiquement par le modèle lui-même. On lui désigne un objet par un point ou une boîte, et il en rend le masque, y compris sur des images d'un domaine qu'il n'a jamais vu. Le modèle est présenté au chapitre 24.

Pourquoi U-Net reviendra. Un U-Net prend une image et rend une image de même taille, en combinant le contexte global (le fond du U) et le détail local (les ponts). C'est exactement ce dont a besoin un modèle de diffusion (chapitre 25) : à chaque étape, prédire le bruit à retirer de chaque pixel d'une image bruitée. L'article DDPM de 2020 (Ho, Jain et Abbeel) utilise un U-Net comme débruiteur. Le U-Net est expliqué en détail ici ; le chapitre 25 n'en fera qu'un rappel.

### Ce que la méthode optimise

Il n'y a pas une perte de la détection. Chaque famille en combine plusieurs, une par sortie du réseau :

| Méthode | Classe (ou classe par pixel) | Position | Ce qui est réglé à la main |
|---|---|---|---|
| R-CNN (2013) | une SVM par classe, perte charnière (ch. 08) | régression ridge des cibles $t$ | NMS, seuils |
| Fast / Faster R-CNN (2015) | entropie croisée sur $K + 1$ classes (fond compris) | « smooth L1 » sur les cibles $t$ | NMS, ancres |
| YOLO (2015) | erreur quadratique | erreur quadratique, $\sqrt{w}$ et $\sqrt{h}$ | $\lambda_{\text{coord}}$, $\lambda_{\text{noobj}}$, NMS |
| RetinaNet (2017) | perte focale | smooth L1 | γ, α, initialisation du biais |
| U-Net (2015) | entropie croisée par pixel, pondérée par une carte de poids | — | carte de poids |
| V-Net (2016) | perte Dice | — | — |
| Exercice 14 | entropie croisée, focale, ou entropie croisée + Dice | aucune (boîtes lues sur les masques) | seuil de NMS |

La perte multitâche de Fast R-CNN additionne les deux termes. Pour une région de vraie classe $u$ (0 pour le fond) :

$$
\Loss = -\log p_u + \lambda\,[u \ge 1] \sum_{i \in \{x, y, w, h\}} \operatorname{smooth}_{L_1}\!\bigl(t_i^u - v_i\bigr),
\qquad
\operatorname{smooth}_{L_1}(z) = \begin{cases} 0{,}5\,z^2 & \text{si } |z| < 1 \\ |z| - 0{,}5 & \text{sinon} \end{cases}
$$

Ici $p_u$ est la probabilité donnée à la vraie classe, $t^u$ la correction prédite pour cette classe et $v$ la cible ; $[u \ge 1]$ vaut 1 pour un objet et 0 pour le fond, qui n'a pas de boîte à corriger ; l'article prend $\lambda = 1$. Le « smooth L1 » est la perte de Huber du chapitre 01 avec un seuil de 1 : quadratique près de zéro, linéaire au-delà, donc moins sensible qu'une erreur quadratique aux régions très mal placées. Fiches complètes de l'entropie croisée, de la perte de Huber et des autres pertes au chapitre 01. La perte focale et la perte Dice sont présentées en détail plus haut.

## Quatre variantes d'un même U-Net, en chiffres

L'exercice entraîne quatre variantes avec la même initialisation (seed 42), 6 époques chacune, sur 3 000 images 64×64 générées : 1 à 5 disques, carrés et triangles tournés, de couleur aléatoire, sur un fond en dégradé bruité avec 3 traits « leurres » qui font partie du fond. La classe ne dépend que de la forme. La validation porte sur 500 images (1 471 objets).

| Variante | Pixels justes | mIoU objets | Bords justes | mAP@0,5 |
|---|---|---|---|---|
| « Toujours fond » | 90,5 % | 0 % | — | 0 % |
| U-Net, entropie croisée | 98,9 % | 85,0 % | 86,1 % | 90,1 % |
| Sans ponts, entropie croisée | 98,2 % | 77,9 % | 75,9 % | 83,8 % |
| U-Net, focale (γ = 2, sans α) | 98,2 % | 74,9 % | 80,6 % | 81,4 % |
| U-Net, entropie croisée + Dice | 99,2 % | 89,2 % | 88,4 % | 93,6 % |

IoU par classe (cercle / carré / triangle), en % :

| Variante | Cercle | Carré | Triangle |
|---|---|---|---|
| U-Net, entropie croisée | 87,4 | 83,2 | 84,3 |
| Sans ponts, entropie croisée | 85,4 | 76,8 | 71,5 |
| U-Net, focale | 79,1 | 70,7 | 74,8 |
| U-Net, entropie croisée + Dice | 91,9 | 88,4 | 87,4 |

(U-Net de 0,48 M de paramètres, Adam avec un taux d'apprentissage « one cycle » (il monte puis redescend au cours de l'entraînement, comme l'échauffement suivi d'une décroissance du chapitre 02), de maximum $2 \times 10^{-3}$, lots de 32 ; 58 à 76 s par variante sur le GPU d'un Mac M1 Pro.)

Ce qu'il faut y lire :

- La précision par pixel ne départage rien : un point d'écart entre la pire et la meilleure variante, quatorze points de mIoU.
- Les triangles sont la classe la plus dure sans ponts (71,5 %). Ce sont les formes aux angles les plus aigus, ceux que le décodeur sans ponts arrondit.
- Le Dice aide surtout au début : 72 % de mIoU à la 3ᵉ époque, quand l'entropie croisée seule en est à 45 %. Il pousse directement vers ce qui est mesuré, le recouvrement de chaque classe.
- Les boîtes suivent les masques : un meilleur masque donne une meilleure mAP, dans le même ordre pour les quatre variantes.

## Ce qui coinçait

- Un vocabulaire fermé. Un détecteur entraîné sur les 80 catégories de COCO ne trouvera jamais une agrafeuse. Ajouter une classe, c'est annoter et réentraîner. Les modèles qui relient images et texte (chapitre 24) ouvriront ce vocabulaire.
- Des annotations chères. Une étiquette d'image se pose vite ; une boîte prend plus de temps ; un masque précis, bien plus encore. COCO compte 2,5 millions d'instances délimitées à la main. U-Net s'en sort avec 30 images grâce à l'augmentation, mais sur un problème étroit. L'apprentissage auto-supervisé (chapitres 23 et 26) et les modèles comme Segment Anything cherchent à réduire ce coût.
- Des règles écrites à la main. Tailles et proportions des ancres, seuils d'IoU pour apparier, seuil de NMS : autant d'hyperparamètres qui ne s'apprennent pas, et qui échouent sur des objets serrés ou superposés. DETR montrera qu'on peut s'en passer.
- Les petits objets. Une erreur de 2 pixels fait perdre un tiers de l'IoU à un objet de 10 pixels (figure de l'IoU), et le tronc a réduit la résolution d'un facteur 16 ou 32 avant que la tête ne voie quoi que ce soit.
- Des mesures qui trompent. La précision par pixel récompense le fond ; la mAP dépend du seuil d'IoU choisi. Comparer deux méthodes demande de vérifier qu'elles sont mesurées de la même façon.

- Fenêtre glissante (Viola-Jones 2001, HOG 2005) : Promener un classifieur rapide sur toutes les positions et toutes les tailles.
- Classification par CNN (2012) : Un CNN nomme l’image entière, sans dire où est l’objet.
- R-CNN (2013) : 2 000 régions candidates, chacune classée par un CNN pré-entraîné ; corriger la boîte par régression.
- Fast et Faster R-CNN (2015) : Un seul passage du CNN par image ; des ancres pour que le réseau propose lui-même les régions.
- YOLO (2015) : Une seule régression de l’image vers une grille de boîtes : temps réel, localisation moins fine.
- FCN, U-Net (2014-2015) : Une classe par pixel : encoder pour comprendre, décoder pour localiser, des ponts pour retrouver les bords.
- Perte focale, Dice (2016-2017) : Ne plus laisser le fond, facile et omniprésent, dominer la perte.
- Mask R-CNN (2017) : Un masque par objet détecté : séparer deux objets de la même classe.
- DETR, Segment Anything (2020-2023) : Des Transformers prédisent l’ensemble des objets sans règles à la main, et segmentent sur commande.

Et ensuite ? Les réseaux convolutifs savent maintenant nommer, localiser et délimiter ce qu'ils voient. Une image se prête bien aux convolutions : des pixels voisins se ressemblent, et un motif garde son sens où qu'il soit. Le langage n'a pas cette structure. Un mot n'est pas une grandeur qu'on additionne, et « chat » n'est pas plus proche de « chien » que de « chaise » si on le code par sa position dans un dictionnaire. Avant de traiter des phrases, il faut donc donner aux mots une représentation où « proche » veut dire « semblable » : c'est l'objet du chapitre 15 : plongements de mots.

## Exercice

Ce qu'il faut observer :

- Les calculs à la main : IoU de 0,143, 0,667 et 0,5 sur les boîtes de la section 1, écart nul avec `torchvision.ops.box_iou`, même NMS que `torchvision.ops.nms` (141 boîtes gardées sur 200) ; Dice = 2·IoU/(1 + IoU) sur deux disques (0,346 → 0,514) ; la perte focale qui divise par 100 le poids d'un pixel facile.
- La précision par pixel ment : 90,5 % pour « toujours fond », et un seul point d'écart entre les variantes entraînées, contre 14 points de mIoU.
- Les ponts servent aux bords : + 7,1 points de mIoU, + 10,2 sur les pixels de bord, rien à l'intérieur des objets.
- La perte change la vitesse d'apprentissage : avec Dice, 72 % de mIoU dès la 3ᵉ époque ; avec la focale, 28 %, et 74,9 % à la fin, sous l'entropie croisée seule (85,0 %).
- Un masque donne des boîtes : mAP@0,5 de 90 % sans tête de régression.

Questions de réflexion : que contient la carte 8×8 du fond du U, et que lui manque-t-il pour tracer un contour ? Quelle différence d'échelle de déséquilibre sépare les 90 % de fond de l'exercice des ≈ 100 000 ancres par image de RetinaNet, et l'écart avec l'entropie croisée change-t-il avec γ = 0,5 ou une pondération α ? Que deviendraient les boîtes de la partie D si deux disques se touchaient ?

## Auto-évaluation

Question : Un vrai objet occupe la boîte (0, 0, 10, 10). Une prédiction est décalée de 5 pixels vers la droite : (5, 0, 15, 10). Est-elle comptée juste au seuil de PASCAL VOC ? Et pour un objet de 40×40 décalé de la même façon ?

Réponse : Intersection : $5 \times 10 = 50$ ; union : $100 + 100 - 50 = 150$ ; IoU $= 50/150 = 0{,}33$, en dessous de 0,5 : c'est un faux positif (et l'objet reste non détecté). Pour le carré de 40, la formule $(a - d)/(a + d)$ donne $35/45 = 0{,}78$ : la même erreur de 5 pixels est acceptée. L'IoU juge l'erreur relativement à la taille de l'objet.

Question : Un réseau de segmentation annonce 97 % de pixels justes sur des images où le fond occupe 95 % des pixels. Est-ce un bon résultat ? Que faudrait-il mesurer ?

Réponse : On ne peut pas le savoir : répondre « fond » partout donnerait déjà 95 %. Le réseau peut avoir raté la moitié des objets. Il faut l'IoU (ou le Dice) de chaque classe d'objet, pixels prédits contre pixels vrais, puis leur moyenne (mIoU). Dans l'exercice, la précision par pixel varie d'un point entre la pire et la meilleure variante, la mIoU de 14 points.

Question : Avec la perte focale et γ = 2, de combien la perte d’un exemple à p_t = 0,9 est-elle réduite par rapport à l’entropie croisée ? Et celle d’un exemple à p_t = 0,5 ? Pourquoi est-ce utile pour un détecteur à une étape ?

Réponse : Le facteur vaut $(1 - p_t)^2$ : $0{,}1^2 = 0{,}01$, soit une perte 100 fois plus petite (0,00105 au lieu de 0,105) ; pour $p_t = 0{,}5$, $0{,}5^2 = 0{,}25$, 4 fois plus petite seulement. Un détecteur à une étape évalue environ 100 000 positions par image, presque toutes du fond facile. Avec l'entropie croisée, leur somme domine la perte (dans l'exemple du chapitre, 1 000 fonds faciles pèsent 98 % de la perte contre 2 % pour un objet difficile) ; la perte focale les éteint et rend le gradient aux exemples difficiles. Sur un problème peu déséquilibré, comme les 90 % de fond de l'exercice, ce n'est pas forcément un gain.

Question : Le même U-Net, privé de ses ponts, perd 7 points de mIoU mais presque rien à l’intérieur des objets. Pourquoi ?

Réponse : Sans ponts, toute l'information doit passer par la carte 8×8 du fond du U. Cette carte sait quelle forme se trouve dans une zone (elle a un grand champ récepteur), mais chacune de ses cases couvre 8×8 pixels de l'image. Le décodeur doit inventer la position exacte du bord à l'intérieur de ce bloc ; il produit des contours flous et arrondit les angles. L'intérieur d'un objet, lui, se devine sans finesse. Les ponts recopient les cartes de l'encodeur à pleine résolution (64×64, 32×32, 16×16), où la position exacte des bords est encore présente.

Question : Deux personnes se tiennent l’une derrière l’autre ; leurs vraies boîtes ont une IoU de 0,6. Le détecteur trouve les deux, avec des scores 0,95 et 0,80. Que fait une NMS au seuil 0,5, et quelles solutions existent ?

Réponse : La NMS garde la boîte à 0,95, puis supprime celle à 0,80 parce qu'elle la recouvre au-delà de 0,5 : une personne disparaît, alors que le réseau l'avait trouvée. Monter le seuil (0,7) la sauverait, mais laisserait passer des doublons ailleurs. C'est la limite d'une règle écrite à la main. Des variantes comme Soft-NMS (Bodla et al., 2017) baissent le score des boîtes qui se recouvrent au lieu de les supprimer ; DETR (2020) supprime la NMS en entraînant le réseau à ne prédire chaque objet qu'une fois.

## Pour aller plus loin

- Articles originaux : R-CNN (2013), Fast R-CNN et Faster R-CNN (2015), YOLO (2015), FCN (2014), U-Net (2015, huit pages très lisibles), V-Net (2016), Mask R-CNN et la perte focale (2017), DETR (2020) et Segment Anything (2023) : liens dans les fiches ci-dessus. Les précurseurs cités dans le repère historique : Viola et Jones, « Rapid object detection using a boosted cascade of simple features » (CVPR 2001) ; Dalal et Triggs, « Histograms of Oriented Gradients for Human Detection » (CVPR 2005).
- Stanford CS231n, *Deep Learning for Computer Vision* : le programme 2026 consacre sa leçon 9 à la détection et à la segmentation (détecteurs à une et deux étapes, segmentation sémantique, d'instances et panoptique), avec les articles FCN, R-CNN, Fast et Faster R-CNN, YOLO et DETR en lecture ; la leçon « Detection and Segmentation » de 2017 est en vidéo.
- Dive into Deep Learning, chapitre 14 (vision par ordinateur), avec du code : boîtes englobantes, ancres, R-CNN, segmentation sémantique, convolution transposée et réseaux entièrement convolutifs.
- A. Odena, V. Dumoulin et C. Olah, « Deconvolution and Checkerboard Artifacts » (Distill, 2016) : pourquoi une convolution transposée mal réglée dessine un damier, en images interactives. Les animations de V. Dumoulin et F. Visin, « A guide to convolution arithmetic for deep learning », montrent aussi la convolution transposée.
- Documentation PyTorch : « TorchVision Object Detection Finetuning Tutorial » (régler finement un Mask R-CNN pré-entraîné sur ses propres images) et les opérateurs de torchvision (`box_iou`, `nms`, `sigmoid_focal_loss`, `roi_align`), contre lesquels l'exercice vérifie ses fonctions.
- La page du projet U-Net (Université de Fribourg), la page de YOLO de Joseph Redmon, et les sites des jeux de données PASCAL VOC et COCO (images annotées et règles d'évaluation).
