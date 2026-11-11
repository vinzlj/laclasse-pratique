# 12. Rendre les réseaux profonds entraînables

Repère historique (Juillet 2006 (pré-entraînement) à février 2015 (BatchNorm)) : Les réseaux entièrement connectés de plus de deux ou trois couches s'entraînaient mal : le gradient s'éteignait ou explosait en traversant les couches, et les grands réseaux apprenaient leurs exemples par cœur. Il fallait les rendre entraînables, puis les empêcher de surapprendre.

La rétropropagation date de 1986 (chapitre 07), mais pendant vingt ans les réseaux de neurones restent peu profonds, capricieux et battus par les SVM et le boosting (chapitre 08). En juillet 2006, Geoffrey Hinton, Simon Osindero et Yee-Whye Teh (Université de Toronto et Université nationale de Singapour) publient dans *Neural Computation* une façon détournée d'entraîner un réseau à trois couches cachées : l'apprendre une couche à la fois, presque sans étiquettes, avant de l'ajuster. Il fait 1,25 % d'erreurs sur MNIST, mieux que les meilleurs réseaux entraînés par rétropropagation seule sans connaissance a priori de la géométrie des images (1,5 %) et que les SVM (1,4 %). C'est le signal de départ. Les neuf années suivantes remplacent cette béquille par des outils plus simples : une initialisation qui respecte la profondeur (Glorot et Bengio, 2010 ; He et al., 2015), le dropout (Hinton et al., 2012), la normalisation par lot (Ioffe et Szegedy, 2015). En 2015, on entraîne directement des réseaux de trente couches.

## L'idée en une image

Un réseau profond est un téléphone arabe de trente joueurs. Chaque joueur (une couche) reçoit un message, le transforme et le transmet. Si chacun parle un peu trop bas, disons à 90 % du volume reçu, le dernier entend $0{,}9^{30} \approx 4\,\%$ du message. S'il parle un peu trop fort, à 110 %, le dernier reçoit $1{,}1^{30} \approx 17$ fois le volume de départ. À trente étapes, l'à-peu-près ne suffit plus : seul un facteur très proche de 1 tient.

La rétropropagation fait passer deux messages par ce téléphone : les activations, de l'entrée vers la sortie, puis le gradient, de la sortie vers l'entrée. Le gradient qui arrive à la première couche est un produit de trente facteurs. S'ils sont un peu trop petits, c'est la disparition du gradient : les premières couches ne bougent plus. Trop grands, c'est l'explosion du gradient : les pas de descente deviennent absurdes. Le premier remède consiste à régler le volume de chaque joueur dès le départ (l'initialisation), le second à le régler en continu pendant l'entraînement (la normalisation par lot).

Un second problème attend le réseau qui arrive enfin à apprendre : le surapprentissage. Un perceptron multicouche (MLP) de 270 000 poids entraîné sur 5 000 images les apprend toutes par cœur (100 % de réussite) et se trompe sur 6 % des nouvelles. Les remèdes de ce chapitre sont des formes de régularisation propres aux réseaux : éteindre des neurones au hasard (le dropout), adoucir les cibles (le lissage des étiquettes), mélanger les exemples (mixup).

| | Avant : MLP des années 1990 | Après : réseau profond de 2015 |
|---|---|---|
| Profondeur typique | 1 ou 2 couches cachées | 20 à 30 couches (et plus avec les connexions résiduelles, chapitre 13) |
| Activation | sigmoïde ou tanh | ReLU (chapitre 07) |
| Initialisation | petits poids « au jugé », ou pré-entraînement (2006-2011) | échelle calculée : initialisation de Glorot pour tanh, initialisation de He pour ReLU |
| Stabiliser l'entraînement | taux d'apprentissage prudent | normalisation par lot à chaque couche |
| Lutter contre le surapprentissage | pénalité L2, arrêt précoce | les mêmes, plus dropout, label smoothing, mixup, augmentation de données |
| Mesuré dans l'exercice (MLP de 20 couches, init. par défaut de PyTorch, 5 époques) | sans normalisation : 11,3 %, le réseau répond toujours « 1 » | avec BatchNorm : 87,5 % |

## Le mécanisme

Un bloc de réseau profond moderne (version MLP) enchaîne quatre opérations. Les deux dernières se comportent différemment à l'entraînement et à l'inférence, source classique de bugs :

Le chapitre part de la béquille de 2006 (section 1), pose le diagnostic (section 2), puis présente les outils qui l'ont remplacée (sections 3 à 8), un encadré sur le curriculum (section 9) et un rappel sur les optimiseurs (section 10).

### 1. 2006 : le pré-entraînement couche par couche

En 2006, un réseau profond entièrement connecté, entraîné par rétropropagation depuis des poids aléatoires, reste bloqué ou fait moins bien qu'un réseau à une ou deux couches cachées. (Les réseaux convolutifs de LeCun, dont les poids sont partagés, font exception depuis 1989 : chapitre 13.) L'idée de Hinton, Osindero et Teh est de ne pas commencer par la rétropropagation. On apprend d'abord chaque couche seule à modéliser les données qu'elle reçoit, sans étiquettes pour les deux premières (la dernière RBM reçoit aussi les 10 étiquettes) ; on empile ; et l'on n'ajuste l'ensemble qu'à la fin. C'est le pré-entraînement couche par couche.

La brique est la machine de Boltzmann restreinte, une version simplifiée de la machine de Boltzmann du chapitre 07 : une couche visible $v$ (les pixels), une couche cachée $h$, des connexions uniquement entre les deux couches. Elle attribue une énergie à chaque configuration :

(Formule : Énergie d'une RBM à unités binaires.)

Les configurations de basse énergie sont les plus probables ; apprendre, c'est baisser l'énergie des images réelles et monter celle des images que le modèle invente. La restriction (pas de lien à l'intérieur d'une couche) rend chaque étape de tirage facile, et Hinton avait proposé en 2002 une approximation rapide du gradient, la divergence contrastive (*contrastive divergence*) : un seul aller-retour visible → caché → visible au lieu des longues simulations de la machine de Boltzmann complète.

L'empilement, pour le réseau MNIST de l'article (trois couches cachées, environ 1,7 million de poids) :

```python
# Pré-entraînement glouton, en pseudo-code (Hinton, Osindero et Teh, 2006)
donnees = images                        # les deux premières RBM : sans étiquettes
couches = []
for taille in [500, 500, 2000]:
    if taille == 2000:                  # la RBM du sommet voit aussi l'étiquette
        donnees = concat(donnees, one_hot(etiquettes))
    rbm = RBM(entrees=donnees.shape[1], cachees=taille)
    rbm.entrainer(donnees, methode="divergence contrastive")   # couche seule, figée ensuite
    couches.append(rbm)
    donnees = rbm.probabilites_cachees(donnees)   # la couche suivante apprend sur cette représentation
# Puis : ajustement de l'ensemble (algorithme « up-down » dans l'article de 2006,
# rétropropagation ordinaire dans l'article de Science et les travaux suivants).
```

Le réseau de croyance profond ainsi obtenu est un modèle génératif : on peut le faire « rêver » des chiffres. La même année, Hinton et Salakhutdinov utilisent ces poids pour initialiser un autoencodeur profond qui compresse des images en 30 nombres mieux que l'ACP (chapitre 08), et notent que, sans pré-entraînement, leur autoencodeur le plus profond reconstruit toujours la moyenne des données, même après un long ajustement.

En simplifiant : on a longtemps expliqué le succès du pré-entraînement par un bon point de départ, proche d'une « bonne vallée » de la perte, et par un effet de régularisation des données non étiquetées. Ce qui est sûr, c'est qu'il a été abandonné en quelques années pour l'apprentissage supervisé : Glorot, Bordes et Bengio montrent en 2011 que des réseaux à ReLU s'entraînent aussi bien sans lui (chapitre 07), et AlexNet (2012) s'en passe (chapitre 13). L'idée de fond, apprendre d'abord une représentation sur des données sans étiquettes puis l'ajuster, reviendra à très grande échelle avec le pré-entraînement des modèles de langage (chapitre 28). L'exercice ne reproduit pas les RBM : il montre directement ce qui les a rendues inutiles.

### 2. Le gradient est un produit de facteurs

Reprenons la formule de la passe arrière du chapitre 07 : $\delta_{\ell-1} = (\delta_\ell\, W_\ell^\top) \odot f'(z_{\ell-1})$. Chaque couche multiplie l'erreur par ses poids et par la dérivée de son activation. Après $L$ couches :

(Formule : Le gradient de la première couche.)

La deuxième écriture est une approximation (les facteurs varient d'une couche et d'un exemple à l'autre), mais elle dit l'essentiel : le gradient dépend exponentiellement de la profondeur. Quelques valeurs :

| Facteur par couche $c$ | 10 couches | 20 couches | 30 couches |
|---|---|---|---|
| 0,5 | $9{,}8 \times 10^{-4}$ | $9{,}5 \times 10^{-7}$ | $9{,}3 \times 10^{-10}$ |
| 0,9 | 0,35 | 0,12 | 0,042 |
| 1 | 1 | 1 | 1 |
| 1,1 | 2,6 | 6,7 | 17 |
| 2 | 1 024 | $10^6$ | $10^9$ |

Ce qui était tolérable à 2 couches (un facteur 0,5 divise le gradient par 4) devient fatal à 20. Deux leviers fixent $c$ : la dérivée de l'activation, traitée au chapitre 07 (la sigmoïde plafonne à 0,25, la unité linéaire rectifiée vaut 1 sur les entrées positives), et l'échelle des poids, qui est l'objet de la section suivante.

L'exercice le mesure sans entraînement, sur 30 couches de 256 neurones sans biais : on fait traverser le réseau à 512 vecteurs gaussiens, on mesure l'écart type des activations après chaque couche, puis on rétropropage un gradient aléatoire.

| Variante | σ des activations, couche 30 | ‖∂ℒ/∂z‖, couche 30 | ‖∂ℒ/∂z‖, couche 1 | Diagnostic |
|---|---|---|---|---|
| tangente hyperbolique, poids σ = 0,01 | $1{,}4 \times 10^{-24}$ | 0,71 | $5{,}9 \times 10^{-24}$ | tout s'éteint |
| tanh, poids σ = 1 | 0,975 | 0,128 | $5{,}8 \times 10^{12}$ | saturé, le gradient explose |
| tanh, Glorot | 0,133 | 0,697 | 0,112 | stable |
| ReLU, Glorot | $3{,}7 \times 10^{-5}$ | 0,497 | $2{,}5 \times 10^{-5}$ | s'éteint lentement |
| ReLU, He | 1,22 | 0,497 | 0,578 | stable |

La variante « tanh, σ = 1 » montre que l'explosion n'est pas l'inverse exact de la disparition : les activations ne grandissent pas (tanh les borne à ±1), elles saturent ; c'est le gradient qui, multiplié par des poids trop grands, monte de 0,13 à $6 \times 10^{12}$ en redescendant. L'écrêtage du gradient du chapitre 02 plafonne ce genre de valeur, mais il soigne le symptôme, pas la cause.

### 3. Initialiser à la bonne échelle : Glorot (2010) et He (2015)

Le calcul. Prenons un neurone de la couche $\ell$ : $z = \sum_{i=1}^{n} w_i\, a_i$, avec $n$ entrées. Si les poids $w_i$ sont tirés indépendamment, de moyenne nulle et de variance $\sigma_w^2$, et indépendants des $a_i$ :

$$
\operatorname{Var}(z) = n\, \sigma_w^2\, \E[a^2].
$$

Pour que le signal ne grossisse ni ne rétrécisse, il faut $\operatorname{Var}(z_\ell) = \operatorname{Var}(z_{\ell-1})$. Tout se joue sur $\E[a^2]$, c'est-à-dire sur l'activation :

- tanh, près de zéro, se comporte comme l'identité : $\E[a^2] \approx \operatorname{Var}(z_{\ell-1})$, d'où $n\,\sigma_w^2 = 1$, soit $\sigma_w^2 = 1/n$. C'est déjà la recommandation de LeCun et ses coauteurs en 1998 (écart type $1/\sqrt{n}$).
- ReLU met à zéro la moitié négative d'une entrée symétrique : $\E[a^2] = \operatorname{Var}(z_{\ell-1}) / 2$. Il faut doubler : $\sigma_w^2 = 2/n$.

(Formule : Initialisations de Glorot (Xavier) et de He (Kaiming).)

Glorot et Bengio veulent aussi garder constante la variance du gradient en remontant, qui dépend du nombre de sorties ; ne pouvant satisfaire les deux conditions à la fois, ils prennent un compromis : l'inverse de la moyenne de $n_{\text{entrées}}$ et $n_{\text{sorties}}$, soit $2/(n_{\text{entrées}} + n_{\text{sorties}})$. Pour une couche carrée ($256 \to 256$), on retrouve $1/n$. Leur article de 2010 montre aussi pourquoi la sigmoïde convient mal aux réseaux profonds : sa moyenne de 0,5 pousse la dernière couche cachée vers la saturation. He, Zhang, Ren et Sun font en 2015 le calcul pour ReLU et entraînent ainsi, depuis des poids aléatoires, un réseau convolutif de 30 couches qu'une initialisation de Glorot ne fait pas converger.

L'exemple chiffré. Avec $n = 256$ :

| Initialisation | Écart type $\sigma_w$ | Facteur par couche sur l'écart type | Après 30 couches |
|---|---|---|---|
| σ = 0,01 (tanh) | 0,01 | $\sigma_w \sqrt{n} = 0{,}16$ | $0{,}16^{30} \approx 1{,}3 \times 10^{-24}$ (mesuré : $1{,}4 \times 10^{-24}$) |
| Glorot, avec ReLU | $\sqrt{1/256} = 0{,}0625$ | $\sqrt{1/2} = 0{,}707$ | $2^{-15} \approx 3{,}1 \times 10^{-5}$ (mesuré : $3{,}7 \times 10^{-5}$) |
| He, avec ReLU | $\sqrt{2/256} = 0{,}088$ | 1 | 1 (mesuré : 1,22) |
| Défaut de PyTorch, avec ReLU | $1/\sqrt{3 \times 256} = 0{,}036$ | $0{,}577 / \sqrt 2 = 0{,}41$ | sur 20 couches : $0{,}41^{20} \approx 1{,}6 \times 10^{-8}$ |

La dernière ligne est un piège réel. `nn.Linear` tire par défaut ses poids uniformément dans $[-1/\sqrt{n}, 1/\sqrt{n}]$, soit une variance $1/(3n)$ : convenable pour un réseau peu profond, désastreux pour 20 couches de ReLU. Dans l'exercice, le gradient de la première couche d'un tel réseau vaut $1{,}8 \times 10^{-8}$ au premier pas, environ un million de fois moins que celui de la vingtième ($1{,}2 \times 10^{-2}$). De la couche 20 à la couche 5, il perd un facteur proche de 0,4 à chaque couche, l'ordre de grandeur du calcul ci-dessus ; plus bas, il se stabilise vers $10^{-8}$ (figure de la section « Ce que montrent les expériences »).

```python
import torch.nn as nn

couche = nn.Linear(256, 256)
nn.init.xavier_normal_(couche.weight)                        # Glorot : pour tanh
nn.init.kaiming_normal_(couche.weight, nonlinearity="relu")  # He : pour ReLU, Var = 2 / n_entrées
nn.init.zeros_(couche.bias)
```

### 4. Le dropout (2012) : entraîner un ensemble de sous-réseaux

Une fois le réseau profond entraînable, il surapprend. En juillet 2012, Hinton et quatre coauteurs de l'Université de Toronto (Srivastava, Krizhevsky, Sutskever, Salakhutdinov) proposent un remède d'une simplicité déroutante, le dropout : à chaque exemple, éteindre au hasard la moitié des neurones cachés.

(Formule : Dropout « inversé » (la version de PyTorch).)

Avec $h = (0{,}8 ;\ 1{,}2 ;\ 0{,}5 ;\ 2{,}0)$, $p = 0{,}5$ et le masque tiré $m = (1, 0, 0, 1)$, la couche transmet $(1{,}6 ;\ 0 ;\ 0 ;\ 4{,}0)$. Au pas suivant, un autre masque. En moyenne sur les masques, chaque composante vaut bien $h_j$ : $0{,}5 \times 2 h_j + 0{,}5 \times 0$. L'article d'origine faisait l'inverse (pas de mise à l'échelle à l'entraînement, poids multipliés par $1 - p$ au test) ; le résultat est le même, la version inversée laisse simplement l'inférence intacte.

Pourquoi ça marche : deux lectures.

1. Contre la co-adaptation. Un neurone ne peut pas compter sur la présence d'un voisin précis pour corriger ses erreurs : il doit apprendre une caractéristique utile seule, dans des contextes très variés. C'est l'explication du titre de 2012 (« preventing co-adaptation of feature detectors »).
2. Un ensemble gratuit. Le MLP de l'exercice a 512 neurones cachés, donc $2^{512} \approx 10^{154}$ sous-réseaux possibles, qui partagent leurs poids. Chaque pas d'entraînement en entraîne un, tiré au sort. À l'inférence, garder tous les neurones revient (approximativement) à faire la moyenne de leurs prédictions : un vote de modèles, comme le bagging du chapitre 08, pour le prix d'un seul réseau.

L'exercice vérifie la seconde lecture sur le modèle MNIST entraîné avec dropout :

| Prédire avec… | Précision (validation) |
|---|---|
| le réseau complet, `model.eval()` | 94,15 % |
| un seul sous-réseau tiré au hasard (`model.train()`) | 92,43 %, et deux passes successives se contredisent sur 6,2 % des images |
| la moyenne des probabilités de 100 sous-réseaux | 94,12 %, même réponse que `eval()` sur 99,5 % des images |

La moyenne de cent sous-réseaux et le réseau complet donnent presque toujours la même réponse : la règle de mise à l'échelle est une bonne approximation de l'ensemble. La ligne du milieu est le bug classique : oublier `model.eval()` avant de prédire, c'est répondre avec un sous-réseau tiré au sort, moins précis et qui change d'avis d'un appel à l'autre.

```python
import numpy as np

def dropout(h, p, rng, entrainement=True):
    """Dropout inversé : identité à l'inférence."""
    if not entrainement or p == 0:
        return h
    masque = rng.random(h.shape) >= p          # True avec la probabilité 1 − p
    return h * masque / (1 - p)
```

Mesuré dans l'exercice : 49,98 % de zéros pour $p = 0{,}5$, survivants multipliés par exactement 2,000 dans PyTorch, espérance conservée à 0,03 % près. L'article de référence (Srivastava et al., *JMLR* 2014) montre des gains en vision, en reconnaissance de la parole, en classification de documents et en biologie computationnelle ; AlexNet l'utilise dans ses couches denses (chapitre 13).

### 5. Pénalité L2 et arrêt précoce : les régularisations classiques, appliquées aux réseaux

Le chapitre 00 a présenté la pénalité L2 (Ridge) et l'arrêt précoce sur un polynôme ; elles s'appliquent telles quelles aux réseaux. Rappel en une ligne chacune :

- Pénalité L2 : on ajoute $\lambda \sum w^2$ à la perte (sur les matrices de poids, pas sur les biais ni sur les paramètres de BatchNorm), ce qui tire tous les poids vers zéro. Avec SGD, c'est exactement la décroissance des poids ; avec Adam, ce n'est plus vrai, d'où AdamW (chapitre 02). L'exercice utilise Adam avec une pénalité dans la perte, $\lambda = 5 \times 10^{-4}$.
- Arrêt précoce : on garde les poids de l'époque où la perte sur le jeu de validation était la plus basse.

L'exercice réserve une surprise sur ce second point. Sans régularisation, la perte de validation moyenne atteint son minimum (0,248) vers l'époque 11, puis remonte jusqu'à 0,308 à l'époque 30 : c'est la signature du surapprentissage, alors que la précision d'entraînement affiche 100 % sans rien signaler. Pourtant, le modèle de la meilleure époque (9 à 11 selon la graine) fait moins bien au test (93,57 %) que celui de l'époque 30 (94,13 %). Le détail (graine 42, validation) explique le paradoxe :

| | Époque de la meilleure perte de validation | Dernière époque |
|---|---|---|
| Erreurs (sur 10 000 images) | 666 | 625 |
| Confiance moyenne sur ces erreurs | 72 % | 82 % |
| Perte moyenne par erreur | 3,05 | 4,82 |
| Part de la perte due aux erreurs | 83 % | 95 % |

Le modèle continue de s'améliorer (41 erreurs de moins) mais devient trop sûr de lui quand il se trompe. Une erreur coûte $-\ln p$, où $p$ est la probabilité laissée à la bonne classe : une perte de 3,05 correspond à $p \approx 4{,}7\,\%$, une perte de 4,82 à $p \approx 0{,}8\,\%$. Une erreur commise avec aplomb coûte bien plus cher qu'une erreur hésitante. La perte remonte à cause d'une poignée d'images (6 % des images font 95 % de la perte). Leçon : la perte et la précision ne mesurent pas la même chose, et l'arrêt précoce sur la perte n'est pas toujours le bon critère quand c'est la précision qui compte. Ce même excès de confiance est ce que vise le label smoothing (section 7).

### 6. La normalisation par lot (2015) : régler le volume en continu

Une bonne initialisation règle l'échelle au départ. Mais les poids bougent pendant l'entraînement, et l'échelle de chaque couche dérive avec eux. En février 2015, Sergey Ioffe et Christian Szegedy (Google) proposent de renormaliser chaque neurone à chaque pas, avec les statistiques du mini-lot courant : c'est la normalisation par lot.

(Formule : BatchNorm, pour un neurone et un mini-lot de m exemples.)

Exemple chiffré, un neurone, un lot de 4 : $z = (1, 2, 3, 6)$. Moyenne $\mu_B = 3$, variance $\sigma_B^2 = (4 + 1 + 0 + 9)/4 = 3{,}5$, écart type 1,871. Valeurs normalisées : $\hat z = (-1{,}069 ;\ -0{,}535 ;\ 0 ;\ 1{,}604)$, de moyenne nulle. Avec $\gamma = 2$ et $\beta = 0{,}5$ appris, la sortie vaut $(-1{,}64 ;\ -0{,}57 ;\ 0{,}5 ;\ 3{,}71)$.

À l'inférence, on prédit souvent une image à la fois : pas de lot sur lequel calculer une moyenne, et la prédiction d'une image ne doit pas dépendre de ses voisines. La couche utilise donc des moyennes glissantes accumulées à l'entraînement, $\mu \leftarrow 0{,}9\,\mu + 0{,}1\,\mu_B$ (et de même pour la variance), puis les fige. Un détail que l'exercice met en évidence : PyTorch accumule la variance non biaisée ($\frac{1}{m-1}\sum$, soit $14/3 = 4{,}67$ dans l'exemple) mais normalise le lot avec la variance biaisée (3,5), comme le prescrit l'article. La réimplémentation NumPy de l'exercice retrouve `nn.BatchNorm1d` à $1{,}4 \times 10^{-6}$ près à l'entraînement et $1{,}2 \times 10^{-6}$ près à l'inférence.

```python
import numpy as np

def batchnorm(z, gamma, beta, etat, entrainement, momentum=0.1, eps=1e-5):
    """z : (taille du lot, neurones). etat : moyennes glissantes, mises à jour sur place."""
    if entrainement:
        mu, var = z.mean(axis=0), z.var(axis=0)                    # variance biaisée (1/m)
        m = z.shape[0]
        etat["moyenne"] = (1 - momentum) * etat["moyenne"] + momentum * mu
        etat["variance"] = (1 - momentum) * etat["variance"] + momentum * var * m / (m - 1)
    else:
        mu, var = etat["moyenne"], etat["variance"]                # figées
    return gamma * (z - mu) / np.sqrt(var + eps) + beta
```

Ce que BatchNorm change, mesuré. Partie D de l'exercice : un MLP de 20 couches cachées, initialisation par défaut de PyTorch, 5 époques de SGD avec momentum :

| Variante | ‖gradient‖ de la couche 1 au premier pas | Précision de validation |
|---|---|---|
| ReLU, défaut de PyTorch | $1{,}8 \times 10^{-8}$ | 11,29 % |
| ReLU, Glorot | $4{,}7 \times 10^{-4}$ | 11,29 % |
| tanh, σ = 0,01 | $6{,}7 \times 10^{-16}$ | 11,29 % |
| tanh, Glorot | — | 81,26 % |
| ReLU, He | — | 71,17 % (pic de perte vers le 30ᵉ pas) |
| ReLU, défaut de PyTorch + BatchNorm | 68,7 | 87,50 % |

Les trois premiers réseaux n'apprennent rien : leur perte reste à $\ln 10 \approx 2{,}303$, celle d'une réponse uniforme sur 10 classes, et ils répondent toujours la même classe. 11,29 % est exactement la part des « 1 » dans la validation (1 129 images sur 10 000). Avec BatchNorm, la même mauvaise initialisation devient sans importance : chaque couche renormalise ce qu'elle reçoit, quelle que soit l'échelle des poids. Le gradient de 68,7 sur des poids minuscules a la même cause : si l'on multiplie les poids d'une couche suivie de BatchNorm par $c$, la sortie ne change pas (la normalisation divise par $c$) et le gradient est divisé par $c$. Des poids 100 fois trop petits reçoivent donc un gradient 100 fois plus grand, qui les ramène vers une échelle raisonnable.

Pourquoi ça marche ? La question reste ouverte. Ioffe et Szegedy l'expliquent par le décalage de covariable interne : chaque couche doit sans cesse se réadapter aux changements de distribution de ses entrées, et la normalisation supprimerait ce problème. En 2018, Santurkar, Tsipras, Ilyas et Madry (MIT) montrent qu'on peut réinjecter un décalage de distribution après BatchNorm sans perdre son bénéfice ; selon eux, l'effet principal est de rendre la surface de perte plus lisse, ce qui autorise des pas plus grands. Le débat n'est pas clos : retenez ce qu'on mesure (entraînement plus rapide, moins sensible à l'initialisation et au taux d'apprentissage), plus que l'explication d'origine.

Ses défauts, qui ont motivé les normalisations suivantes : elle dépend de la taille du lot (avec 2 exemples, la moyenne est très bruitée), elle se comporte différemment en entraînement et en inférence, et elle se prête mal aux séquences de longueur variable. La normalisation de couche (LayerNorm, 2016), qui normalise sur les neurones d'un même exemple plutôt que sur le lot, et sa variante simplifiée RMSNorm sont les normalisations des Transformers ; elles sont expliquées au chapitre 20.

### 7. Le label smoothing (2015) : interdire la certitude

Avec l'entropie croisée ordinaire, la cible est « 100 % sur la bonne classe ». Pour l'atteindre, la softmax exige que l'écart entre le logit de la bonne classe et les autres tende vers l'infini : le modèle est récompensé à chaque pas pour devenir plus sûr de lui, même sur les exemples d'entraînement déjà bien classés. C'est la dérive mesurée en section 5. En décembre 2015, dans l'article d'Inception-v3, Szegedy et ses coauteurs proposent d'adoucir la cible, le lissage des étiquettes :

(Formule : Cible lissée.)

Avec $\varepsilon = 0{,}1$ et $K = 10$ : la cible vaut 0,91 sur la bonne classe et 0,01 sur chacune des neuf autres. La meilleure réponse possible est $p = q$, ce qui demande un écart de logits fini : $\ln(0{,}91 / 0{,}01) = \ln 91 \approx 4{,}5$. Et la perte ne peut plus descendre à 0 : son minimum est l'entropie de la cible,

$$
-\bigl(0{,}91 \ln 0{,}91 + 9 \times 0{,}01 \ln 0{,}01\bigr) = 0{,}086 + 0{,}414 = 0{,}500.
$$

C'est la valeur que l'exercice retrouve, et sa réimplémentation coïncide avec `F.cross_entropy(..., label_smoothing=0.1)` à $4{,}8 \times 10^{-7}$ près. Conséquence pratique : on ne compare pas la perte d'un modèle entraîné avec label smoothing à celle des autres. Dans l'exercice, c'est la meilleure configuration au test (95,77 %), avec une perte de validation (0,259) plus haute que celle de toutes les autres régularisations.

En 2019, Müller, Kornblith et Hinton montrent que le label smoothing améliore aussi la calibration (une confiance de 80 % correspond plus souvent à 80 % de réussite, chapitre 39), mais qu'un modèle ainsi entraîné fait un moins bon professeur en distillation : en resserrant les représentations de chaque classe, il efface l'information sur les ressemblances entre classes (« ce 4 ressemble un peu à un 9 ») dont l'élève se nourrit (chapitre 28). Le Transformer de 2017 l'utilise aussi, avec $\varepsilon = 0{,}1$.

### 8. Mixup (2017) : apprendre entre les exemples

Le mixup fabrique de nouveaux exemples d'entraînement en mélangeant deux exemples et leurs cibles :

(Formule : Mixup.)

Exemple : avec $\lambda = 0{,}7$, un « 3 » et un « 8 » superposés donnent une image fantôme dont la cible vaut 0,7 sur la classe 3 et 0,3 sur la classe 8. Le modèle apprend à répondre linéairement entre les exemples, ce qui l'empêche de tracer des frontières brutales tout contre les points d'entraînement. La perte reste l'entropie croisée, calculée contre la cible mélangée $\tilde y$ (ce qui revient à $\lambda\,\Loss(y_i) + (1-\lambda)\,\Loss(y_j)$).

Avec $\alpha = 0{,}2$ (la valeur de l'exercice ; sur ImageNet, l'article trouve $\alpha \in [0{,}1 ;\ 0{,}4]$ utile et un $\alpha$ grand nuisible), $\lambda$ tombe sous 0,1 ou au-dessus de 0,9 dans 67 % des tirages : la plupart des exemples mélangés restent proches d'une seule image, avec une légère trace de l'autre. Résultat sur MNIST : 95,09 % au test, contre 94,13 % sans régularisation.

Mixup est une forme d'augmentation de données. L'augmentation classique (recadrer, retourner, décaler une image sans changer son étiquette) est le premier rempart contre le surapprentissage en vision ; elle est expliquée au chapitre 13, qui en est le propriétaire.

### 9. Le curriculum : l'ordre des exemples

Encadré : l'apprentissage par curriculum (2009) :
On apprend aux enfants l'addition avant les équations. Bengio, Louradour, Collobert et Weston (Université de Montréal et NEC Labs) formalisent en 2009 l'apprentissage par curriculum : présenter d'abord des exemples faciles, puis de plus en plus difficiles, plutôt que dans un ordre aléatoire. Sur la reconnaissance de formes géométriques et sur un modèle de langue, ils obtiennent une meilleure généralisation, et l'interprètent comme une méthode de continuation : optimiser d'abord une version lisse du problème, puis la déformer progressivement vers le vrai problème, pour éviter les mauvais minima locaux. Difficulté pratique : il faut savoir ce qu'est un exemple « facile ». L'idée n'est pas devenue un outil standard comme BatchNorm ou le dropout, mais elle survit sous d'autres formes, par exemple dans le choix et l'ordre des données de pré-entraînement des LLM (chapitre 28).

### 10. Rappel daté : les optimiseurs adaptatifs (2011-2017)

Pendant la même période, l'optimiseur change aussi. AdaGrad (2011), RMSProp (2012), Adam (2014) et AdamW (2017) donnent à chaque poids son propre pas, adapté à la taille de ses gradients ; Adam devient l'optimiseur par défaut des réseaux profonds. Ils sont expliqués au chapitre 02, dont les sections 4 et 5 (Adam, AdamW, échauffement, écrêtage) se relisent avec profit maintenant. L'exercice utilise Adam pour les MLP de deux couches (partie C) et SGD avec momentum pour les réseaux de 20 couches (partie D), où l'effet de l'initialisation est plus visible.

### Ce que la méthode optimise

Aucune des techniques de ce chapitre ne change fondamentalement la perte : on minimise toujours l'entropie croisée entre la softmax des logits et la cible, par descente de gradient. Ce qui change :

- la pénalité L2 ajoute $\lambda \sum w^2$ à la perte ;
- le label smoothing et mixup remplacent la cible one-hot par une cible adoucie ou mélangée (même formule, autre $y$) ;
- le dropout et BatchNorm ne touchent pas à la perte mais au réseau : la perte minimisée est celle d'un sous-réseau tiré au sort, ou d'un réseau normalisé par lot ;
- l'initialisation et l'arrêt précoce choisissent le point de départ et le point d'arrivée de la descente.

Fiche complète de l'entropie croisée et comparaison avec les autres pertes au chapitre 01 :

La variante lissée, dont ce chapitre est le propriétaire :

Le tout, dans la boucle d'entraînement de l'exercice (PyTorch, sans les détails de mesure) :

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def mlp(dropout=0.0, batchnorm=False, cachee=256):
    """784-256-256-10. Ordre par couche : Linéaire → BatchNorm → ReLU → Dropout."""
    couches, entree = [nn.Flatten()], 784
    for _ in range(2):
        couches.append(nn.Linear(entree, cachee))
        if batchnorm:
            couches.append(nn.BatchNorm1d(cachee))
        couches.append(nn.ReLU())
        if dropout:
            couches.append(nn.Dropout(dropout))
        entree = cachee
    couches.append(nn.Linear(entree, 10))             # logits : ni ReLU ni dropout
    return nn.Sequential(*couches)

def une_epoque(modele, lots, optimiseur, l2=0.0, ls=0.0, mixup=0.0, rng=None):
    modele.train()                                    # dropout actif, BatchNorm sur le lot
    for x, y in lots:
        if mixup:
            lam = float(rng.beta(mixup, mixup))
            perm = torch.randperm(len(x), device=x.device)
            sortie = modele(lam * x + (1 - lam) * x[perm])
            perte = lam * F.cross_entropy(sortie, y) + (1 - lam) * F.cross_entropy(sortie, y[perm])
        else:
            perte = F.cross_entropy(modele(x), y, label_smoothing=ls)
        if l2:                                        # matrices de poids seulement
            perte = perte + l2 * sum((p  2).sum() for p in modele.parameters() if p.dim() > 1)
        optimiseur.zero_grad()
        perte.backward()
        optimiseur.step()

# Avant toute évaluation : modele.eval()  (dropout coupé, BatchNorm sur ses moyennes glissantes)
```

## Ce que montrent les expériences

La régularisation sur MNIST. Le même MLP 784-256-256-10, entraîné sur 5 000 images seulement (pour qu'il surapprenne vite), 30 époques, Adam, sept configurations, trois graines chacune :

| Configuration | Précision d'entraînement | Perte de validation finale | Test (moyenne de 3 graines ± écart type) |
|---|---|---|---|
| sans régularisation | 100,00 % | 0,308 (minimum 0,248 vers l'époque 11) | 94,13 % ± 0,13 |
| pénalité L2 | 99,97 % | 0,217 | 94,40 % ± 0,03 |
| dropout | 99,83 % | 0,226 | 94,83 % ± 0,03 |
| dropout + BatchNorm | 99,92 % | 0,200 | 95,01 % ± 0,06 |
| mixup | 99,95 % | 0,209 | 95,09 % ± 0,24 |
| BatchNorm | 100,00 % | 0,204 | 95,16 % ± 0,03 |
| label smoothing | 100,00 % | 0,259 (cible différente : non comparable) | 95,77 % ± 0,12 |

Trois choses à lire :

1. La précision d'entraînement ne dit rien : elle est à 100 % ou presque partout. Le surapprentissage se voit sur la perte de validation.
2. Les gains sont réels mais modestes (+ 0,3 à + 1,6 point), et plus grands que les écarts entre graines (± 0,03 à ± 0,24 point) : d'où l'intérêt de répéter avec plusieurs graines avant de conclure.
3. Ils ne s'additionnent pas : dropout + BatchNorm (95,01 %) ne bat pas BatchNorm seul (95,16 %). Chaque technique s'attaque en partie au même excès de capacité.

La profondeur. La figure suivante reprend la partie D (section 6) : trois réseaux de 20 couches restent plats à $\ln 10$, parce que le gradient qui atteint leurs premières couches est de l'ordre de $10^{-15}$ à $10^{-4}$ ; BatchNorm donne à la fois le plus grand gradient et la meilleure précision.

## Ce qui coinçait

Ces outils ont rendu entraînables des réseaux de quelques dizaines de couches. Ce qui restait :

- La profondeur, encore. Même bien initialisé et normalisé, un réseau « en chaîne » se dégrade au-delà de quelques dizaines de couches (He et al. le constatent en 2015, chapitre 13). Dans l'exercice, le réseau de 20 couches non normalisé « ReLU + He » fait un pic de perte et plafonne à 71 % en 5 époques ; avec BatchNorm, il atteint 87,5 %. La solution sera de laisser le signal contourner les couches, par des connexions résiduelles (ResNet, 2015, chapitre 13).
- BatchNorm et le lot. Statistiques bruitées sur de petits lots, comportement différent en entraînement et en inférence, mal adaptée aux séquences : les Transformers la remplacent par LayerNorm et RMSNorm (chapitre 20).
- Des recettes plus que des théories. Le décalage de covariable interne n'explique sans doute pas, à lui seul, l'effet de BatchNorm ; le dropout est une approximation d'ensemble ; les régularisations ne s'additionnent pas et se règlent à la main ($p$, $\lambda$, $\varepsilon$, $\alpha$). Et une question de fond reste entière : pourquoi des réseaux qui ont bien plus de paramètres que d'exemples généralisent-ils quand même, contre les bornes du chapitre 10 ? C'est le sujet du chapitre 27.
- Le MLP ignore la structure des données. Notre MLP traite une image comme 784 nombres sans ordre : il ne sait pas que deux pixels voisins sont liés, ni qu'un « 3 » décalé de deux pixels reste un « 3 ». Il doit tout réapprendre à chaque position, avec beaucoup de poids, d'où le surapprentissage. C'est le défaut que lèvent les réseaux convolutifs.

- Rétropropagation et ReLU (1986, 2010-2011) : Le gradient d’une couche est un produit de facteurs, un par couche au-dessus ; la dérivée de ReLU vaut 1.
- Pré-entraînement couche par couche (2006) : Apprendre chaque couche seule, presque sans étiquettes, puis ajuster le tout : une béquille pour démarrer dans une bonne région.
- Initialisation de Glorot (2010) et de He (2015) : Choisir l’échelle des poids pour que chaque couche multiplie le signal par 1 : Var(W) = 1/n pour tanh, 2/n pour ReLU.
- Dropout (2012) : Éteindre la moitié des neurones à chaque pas : un ensemble de sous-réseaux qui partagent leurs poids, moyenné à l’inférence.
- BatchNorm (2015) : Recentrer et réduire chaque neurone sur le lot, à chaque pas : l’échelle est réglée en continu, l’initialisation devient presque indifférente.
- Label smoothing (2015), mixup (2017) : Adoucir ou mélanger les cibles : le modèle n’a plus intérêt à être infiniment sûr de lui.
- Connexions résiduelles (2015) : Laisser le signal sauter des couches : la profondeur passe de 30 à plus de 100 couches.
- LayerNorm, RMSNorm (2016-2019) : Normaliser chaque exemple sur ses propres neurones, sans dépendre du lot : la norme des Transformers.

Et ensuite ? On sait désormais entraîner des réseaux profonds et limiter leur surapprentissage. Mais pour les images, le MLP gaspille ses poids : chaque neurone de la première couche regarde les 784 pixels à la fois et doit réapprendre le même motif à chaque position. Les réseaux convolutifs imposent deux idées simples, la localité (un neurone ne regarde qu'un petit voisinage) et le partage des poids (le même filtre glisse sur toute l'image). Avec les outils de ce chapitre, ReLU, dropout et bientôt BatchNorm, ils vont gagner ImageNet en 2012 et lancer la vague du deep learning : c'est le chapitre 13 : Réseaux convolutifs (CNN).

## Exercice

Ce qu'il faut observer :

- Le produit de facteurs (partie A) : avec des poids d'écart type 0,01, le signal tombe à $10^{-24}$ en 30 couches ; avec σ = 1, tanh sature et le gradient monte à $6 \times 10^{12}$ ; ReLU perd un facteur $\sqrt 2$ par couche avec Glorot et reste stable avec He.
- Les mécanismes recodés (partie B) : BatchNorm et label smoothing en NumPy coïncident avec PyTorch à $10^{-6}$ près, le dropout a les mêmes statistiques ; la perte lissée ne descend pas sous 0,500.
- Le surapprentissage (partie C) : la perte de validation remonte alors que la précision d'entraînement est à 100 % ; l'arrêt précoce sur la perte fait ici moins bien que d'aller au bout ; un seul sous-réseau de dropout perd 1,7 point.
- La profondeur (partie D) : trois réseaux de 20 couches répondent toujours « 1 » (11,29 %) ; BatchNorm sauve même l'initialisation par défaut de PyTorch.

Questions de réflexion : pourquoi un facteur 0,7 par couche est-il anodin à 3 couches et fatal à 30 ? Montrez que multiplier par $c$ les poids d'une couche suivie de BatchNorm ne change pas sa sortie et divise son gradient par $c$ : qu'est-ce que cela implique pour une pénalité L2 sur ces poids ? Quelle mesure vérifierait que le label smoothing rend le modèle moins trop sûr de lui ?

## Auto-évaluation

Question : Un réseau de 40 couches transmet en moyenne 80 % du gradient d'une couche à la précédente. Quelle fraction du gradient de la dernière couche atteint la première, et que faire ?

Réponse : Le gradient est un produit d'un facteur par couche : $0{,}8^{40} \approx 1{,}3 \times 10^{-4}$ (ou $0{,}8^{39}$ si l'on compte les passages entre couches, même ordre de grandeur). Les premières couches apprennent environ dix mille fois plus lentement que les dernières : c'est la disparition du gradient. Remèdes vus ici : une activation dont la dérivée vaut 1 (ReLU), une initialisation qui ramène le facteur à 1 (He pour ReLU, Glorot pour tanh), une normalisation par lot qui le maintient pendant l'entraînement. Au-delà de quelques dizaines de couches, il faudra des connexions résiduelles (chapitre 13).

Question : Une couche ReLU a 512 entrées. Quel écart type donner à ses poids avec l'initialisation de He, et pourquoi le double de la variance qui convient à tanh ?

Réponse : He : $\operatorname{Var}(W) = 2/n = 2/512$, soit un écart type $\sqrt{2/512} = 0{,}0625$. Pour tanh (quasi linéaire près de 0), la variance des sorties d'une couche vaut $n\,\operatorname{Var}(W)$ fois celle de ses entrées, d'où $\operatorname{Var}(W) = 1/n$ (écart type 0,044). ReLU met à zéro la moitié négative de ses entrées, ce qui divise par deux $\E[a^2]$ : il faut doubler la variance des poids pour compenser. Avec $1/n$, ReLU perd un facteur $\sqrt 2$ par couche, soit $3 \times 10^{-5}$ après 30 couches.

Question : Un réseau entraîné avec un dropout p = 0,2 donne des prédictions qui changent d'un appel à l'autre sur la même image. Quel est le bug, et par combien les survivants sont-ils multipliés à l'entraînement ?

Réponse : Le modèle est resté en mode entraînement : il faut appeler `model.eval()` avant de prédire. Sinon chaque appel tire un nouveau masque et répond avec un seul sous-réseau, moins précis (dans l'exercice, 92,4 % contre 94,2 %) et instable (6 % de désaccords entre deux passes). À l'entraînement, chaque neurone est conservé avec la probabilité 0,8 et les survivants sont multipliés par $1/(1 - 0{,}2) = 1{,}25$, pour que l'espérance de chaque sortie ne change pas ; à l'inférence, rien n'est éteint ni multiplié.

Question : Pourquoi BatchNorm utilise-t-il des moyennes glissantes à l'inférence au lieu des statistiques du lot ? Que deviendrait la prédiction d'une image seule sinon ?

Réponse : À l'inférence, on prédit souvent une image à la fois, et la prédiction d'une image ne doit pas dépendre des autres images du lot. Avec un lot d'une seule image, la variance du lot vaut 0 et la valeur centrée vaut 0 pour tous les neurones : la couche effacerait l'information (seul $\beta$ resterait). La couche accumule donc pendant l'entraînement une moyenne et une variance glissantes ($\mu \leftarrow 0{,}9\,\mu + 0{,}1\,\mu_B$), puis les fige ; c'est ce que fait `model.eval()`.

Question : Avec label smoothing (ε = 0,2, K = 5 classes), quelle est la cible, et pourquoi un modèle ainsi entraîné peut-il avoir à la fois la meilleure précision et une perte de validation plus haute ?

Réponse : Cible : $1 - 0{,}2 + 0{,}2/5 = 0{,}84$ sur la bonne classe et $0{,}2/5 = 0{,}04$ sur chacune des quatre autres. La perte est minimale quand le modèle prédit exactement cette cible, et ce minimum n'est pas nul : c'est l'entropie de la cible, $-(0{,}84 \ln 0{,}84 + 4 \times 0{,}04 \ln 0{,}04) \approx 0{,}66$. Le modèle n'est donc jamais récompensé pour être sûr à 100 %, ce qui améliore souvent la précision et la calibration, mais sa perte reste au-dessus de ce plancher : la comparer à la perte d'un modèle entraîné sur des cibles one-hot n'a pas de sens.

## Pour aller plus loin

- Articles originaux : Hinton, Osindero et Teh (2006), Glorot et Bengio (2010, court et très lisible, avec les courbes d'activations couche par couche), Srivastava et al. (2014, l'article complet sur le dropout), He et al. (2015), Ioffe et Szegedy (2015), Santurkar et al. (2018), Müller et al. (2019), Zhang et al. (2017) : liens dans les fiches ci-dessus. Pour la vue d'ensemble de la période, la synthèse de LeCun, Bengio et Hinton, « Deep learning » (*Nature*, 2015).
- A. Karpathy, « Building makemore Part 3: Activations & Gradients, BatchNorm », dans la série Neural Networks: Zero to Hero : les statistiques des activations et des gradients couche par couche, l'initialisation et BatchNorm, recodés devant vous. Du même auteur, « A Recipe for Training Neural Networks » (2019), les réflexes pratiques pour déboguer un entraînement.
- Stanford CS231n, notes « Neural Networks Part 2: Setting up the Data and the Loss » (prétraitement, initialisation, BatchNorm, régularisation dont le dropout) et « Part 3: Learning and Evaluation » (surveiller un entraînement, contrôler les gradients).
- deeplearning.ai, « Initializing neural networks » : une page interactive qui montre l'explosion et la disparition selon l'échelle des poids.
- Dive into Deep Learning : « Numerical Stability and Initialization », « Dropout », « Batch Normalization », avec le code.
- I. Goodfellow, Y. Bengio et A. Courville, *Deep Learning* (2016), en ligne : chapitre 7, « Regularization for Deep Learning », chapitre 8, « Optimization for Training Deep Models » (initialisation, normalisation par lot) et chapitre 15, « Representation Learning » (pré-entraînement couche par couche).
- Y. LeCun, cours au Collège de France, chaire « Informatique et sciences numériques » 2015-2016, en français.
