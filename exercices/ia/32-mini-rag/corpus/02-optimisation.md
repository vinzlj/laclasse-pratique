# 02. Optimisation

Repère historique (1951-2024) : Minimiser une loss non convexe (plusieurs creux et cols, pas une simple cuvette) en très grande dimension, avec des gradients bruités.

En 1951, Robbins et Monro montrent qu’on peut trouver la racine d’une fonction qu’on ne mesure qu’avec du bruit : c’est l’ancêtre de la descente de gradient stochastique. En 1964, Polyak ajoute une mémoire des pas passés (le momentum) ; en 1983, Nesterov en donne une version accélérée. Ces outils viennent de l’optimisation numérique. Quand les réseaux profonds arrivent, avec des millions puis des milliards de paramètres et des gradients estimés sur quelques exemples, une deuxième vague naît en dix ans : AdaGrad (2010), RMSProp (2012), Adam (2014), AdamW (2017). En 2024, Muon rouvre la question pour les très grands modèles.

Ordre de lecture :
Ce chapitre est une boîte à outils. Les sections 1 à 3 (lots, surfaces, momentum) se lisent dès maintenant. Les sections 4 à 6 (Adam et AdamW, plannings du pas, second ordre) et la section 7 (Muon) parlent de réseaux que vous n’avez pas encore vus : survolez-les, puis relisez-les après le chapitre 12.

## L’idée en une image

Le chapitre 00 a posé la descente de gradient : on calcule la pente de la perte, on fait un petit pas vers le bas, on recommence. La taille du pas est le taux d’apprentissage, noté η. Sur une cuvette bien ronde, c’est tout ce qu’il faut.

Le vrai paysage d’un réseau n’est pas une cuvette. Imaginez une randonneuse dans le brouillard qui ne voit que la pente sous ses pieds, et ce sur un milliard de dimensions :

- Les vallées sont étroites. La pente est très raide en travers et presque nulle en long. Un pas assez petit pour ne pas rebondir d’un flanc à l’autre est ridicule dans le sens de la vallée : on avance d’un pas de fourmi.
- Il y a des cols. Au sommet d’un col, la pente est nulle dans toutes les directions : on croit être arrivé, on est juste immobile.
- Le sol tremble. On ne mesure pas la pente sur toutes les données, trop cher, mais sur un petit échantillon : chaque mesure est un peu fausse, dans une direction au hasard (et d’autant plus qu’on approche du but, où la vraie pente devient faible).

Chaque optimiseur de ce chapitre corrige l’un de ces défauts. Le momentum donne de l’élan : on garde la vitesse acquise, qui s’accumule dans le sens de la vallée et s’annule dans les zigzags. Les méthodes adaptatives (AdaGrad, RMSProp, Adam) donnent un pas propre à chaque direction : grand là où la pente est faible, petit là où elle est forte. Les plannings du pas font de grands pas au début et de petits à la fin, pour se poser malgré le tremblement. Les méthodes du second ordre mesurent la courbure et sautent directement au fond, mais c’est hors de prix en grande dimension.

| | Avant : descente de gradient « nue » | Après : optimiseur moderne (AdamW + planning du pas) |
|---|---|---|
| Ce qu’on calcule à chaque pas | le gradient de la perte sur toutes les données | le gradient sur un mini-lot de 32 à quelques milliers d’exemples |
| Taille du pas | un seul η pour tous les paramètres, fixe | un pas par paramètre (normalisé par l’historique de ses gradients), qui monte puis redescend |
| Mémoire | aucune | moyenne glissante des gradients et de leurs carrés (2 nombres par paramètre) |
| Vallée étroite (exercice 02) | 205 pas pour atteindre le seuil | 73 pas avec le momentum seul, 90 avec Adam |
| Col (point selle) | 202 pas pour en sortir | 35 pas avec Adam |
| Ce qui coince encore | lent et instable dès que la courbure varie | 2 à 3 fois plus de mémoire que les poids ; réglages (η, planning) toujours à chercher |

La lignée, en une frise :

## Le mécanisme

Un pas d’entraînement, quel que soit l’optimiseur, suit toujours le même circuit. Seule la boîte « optimiseur » change d’un algorithme à l’autre :

### Ce que la méthode optimise

Aucune fonction de perte propre à ce chapitre : un optimiseur minimise la perte qu’on lui donne. Dans l’exercice, ce sont d’abord des fonctions test en 2D, puis l’erreur quadratique moyenne d’une régression linéaire et l’entropie croisée d’un classifieur MNIST. Ces pertes sont expliquées et comparées au chapitre 01.

### 1. Lot complet, SGD, mini-lots, taux d’apprentissage

La perte d’entraînement est une moyenne sur les N exemples. Son gradient aussi :

Calculer le vrai gradient coûte un passage sur toutes les données pour un seul pas. La descente de gradient stochastique (SGD) le remplace par $\hat g$, calculé sur un mini-lot : en moyenne, c’est le bon gradient, mais chaque estimation est bruitée. En simplifiant (exemples tirés indépendamment), la variance de $\hat g$ décroît comme $1/B$ : quadrupler le lot divise le bruit (l’écart type) par deux, pour quatre fois plus de calcul. Un passage complet sur les données s’appelle une époque ; avec N = 4 000 et B = 32, une époque compte 125 pas au lieu d’un seul.

Le gain est énorme parce que le début de l’entraînement n’a pas besoin d’un gradient précis, seulement d’une bonne direction. Mesures de l’exercice, sur une régression linéaire (4 000 exemples, 20 variables), en « excès de perte » $\Loss(w) - \Loss(w^*)$ :

| Configuration | Après 1 époque | Après 20 époques |
|---|---|---|
| Plein lot, η = 0,4 | 3,65 | 0,016 |
| Lots de 32, η = 0,1 constant | 0,016 | 0,0065 |
| Lots de 32, η = 0,4 constant | | 0,052 |
| Lots de 32, échauffement + cosinus, η max = 0,4 | | 0,000028 |

À calcul égal (une époque), les mini-lots sont 200 fois plus bas que le plein lot. Mais à pas constant, ils stagnent : le bruit les fait danser autour du minimum, d’autant plus fort que η est grand (0,052 contre 0,0065). La dernière ligne règle ce problème ; on y revient en section 5.

Le pas : trop petit, trop grand. Prenons la « vallée étroite » de l’exercice :

Le gradient vaut $(x,\ 50y)$. Un pas de descente donne $x \leftarrow (1-\eta)\,x$ et $y \leftarrow (1 - 50\eta)\,y$. Calcul à la main, avec η = 0,025 et le départ (−8 ; 1) :

- en $y$, le facteur vaut $1 - 50 \times 0{,}025 = -0{,}25$ : $y$ change de signe et est divisé par 4 à chaque pas. Réglé en quelques pas ;
- en $x$, le facteur vaut $0{,}975$ : $x$ ne perd que 2,5 % par pas. Après 60 pas, $x = -8 \times 0{,}975^{60} \approx -1{,}75$. Pour passer sous $f = 10^{-3}$, il faut $|x| \leq 0{,}045$, soit $0{,}975^n \leq 0{,}0056$, donc $n \geq 205$ pas. C’est exactement ce que mesure l’exercice ;
- pour aller plus vite en $x$, il faudrait augmenter η, mais en $y$ le facteur $|1 - 50\eta|$ dépasse 1 dès que $\eta > 2/50 = 0{,}04$, et $y$ explose. C’est la règle $\eta < 2/\lambda_{\max}$ du chapitre 00, vue ici sur un exemple ($\lambda_{\max}$ = 50, la plus forte courbure). L’exercice le vérifie : à η = 0,039, $f$ finit à $1{,}4 \cdot 10^{-9}$ ; à η = 0,041, $f$ atteint $2{,}7 \cdot 10^{13}$.

Le pas maximal est fixé par la direction la plus raide, la vitesse par la direction la plus plate. Le rapport des deux courbures (ici 50) mesure le conditionnement : plus il est grand, plus la descente de gradient est lente.

### 2. Les surfaces difficiles : vallée, col, banane, minima locaux

L’exercice et la démo utilisent trois surfaces, choisies pour isoler chacune une difficulté :

| Surface | Formule | Minimum | Difficulté |
|---|---|---|---|
| Vallée étroite | $\tfrac12(x^2 + 50y^2)$ | (0 ; 0) | mauvais conditionnement : il faut un petit pas pour $y$ et un grand pour $x$ |
| Point selle | $x^2 + \tfrac14 y^4 - \tfrac12 y^2$ | (0 ; ±1), f = −0,25 | un col en (0 ; 0) : le gradient y est nul sans que ce soit un minimum |
| Rosenbrock (1960) | $(1-x)^2 + 100\,(y - x^2)^2$ | (1 ; 1) | une vallée courbe en forme de banane : on y tombe vite, la suivre est long |

Le point selle mérite qu’on s’y arrête. Partant de (1,5 ; 0,0001), la descente glisse vers le col en (0 ; 0), où la pente en $y$ vaut $y^3 - y \approx -y$ : pour $y = 10^{-4}$, elle est minuscule. Chaque pas multiplie $y$ par $1 + \eta = 1{,}05$ ; il en faut environ $\ln(10^4)/\ln(1{,}05) \approx 190$ pour que $y$ atteigne l’ordre de 1. L’exercice mesure 202 pas pour SGD.

En grande dimension, ces cols sont la règle : en un point où le gradient s’annule, il suffit qu’une direction sur des millions descende encore pour que ce ne soit pas un minimum. Dauphin et al. (2014) ont défendu que la prolifération des points selles, plus que les minima locaux, freine l’optimisation des réseaux. Leur argument (physique statistique, matrices aléatoires, expériences) : les points critiques à forte erreur sont surtout des cols entourés de plateaux, qui donnent « l’impression illusoire » d’un minimum local. Les minima locaux existent aussi ; aucun théorème ne garantit qu’on évite les mauvais, c’est un constat empirique sur les grands réseaux. On y revient avec la généralisation au chapitre 10.

La fonction de Rosenbrock vient d’un article de 1960 sur la recherche automatique du minimum d’une fonction ; elle est restée la surface de test classique des optimiseurs.

### 3. Momentum et Nesterov

Le momentum (Polyak, 1964, méthode de la « boule pesante ») remplace le gradient par une vitesse qui accumule les gradients passés. On l’écrit ici comme PyTorch :

Deux effets, que l’on voit sur la vallée :

- dans le sens de la vallée, le gradient garde le même signe : la vitesse s’accumule jusqu’à $g/(1-\mu) = 10\,g$. Avec μ = 0,9, le pas effectif est jusqu’à 10 fois plus grand ;
- en travers, le gradient change de signe à chaque pas : les contributions s’annulent en partie, le zigzag est amorti.

Mesuré dans l’exercice, au même η = 0,025 : 73 pas au lieu de 205. Le gain réel (×2,8) est loin du ×10 théorique, parce que la vitesse fait aussi déborder la trajectoire en travers de la vallée : le momentum zigzague à son tour, plus lentement.

Le gradient accéléré de Nesterov (1983) corrige le momentum en regardant où la vitesse va nous mener avant de corriger. Dans la forme de PyTorch :

Sur les fonctions convexes (en forme de cuvette, sans autre creux que le minimum) et lisses, Nesterov a démontré une erreur en $O(1/k^2)$ après $k$ pas, contre $O(1/k)$ pour la descente de gradient : doubler le nombre de pas divise l’erreur garantie par 4 au lieu de 2. Dans l’exercice : 53 pas sur la vallée (contre 73 pour le momentum) et 557 sur Rosenbrock (contre 666). Mais il n’est pas « momentum en mieux » à tous les coups : à η = 0,028, le momentum converge encore sur la vallée alors que Nesterov diverge vers le pas 157 (question de réflexion 2 de l’exercice).

En 2013, Sutskever, Martens, Dahl et Hinton ont montré qu’avec une bonne initialisation et un μ qui augmente progressivement, SGD avec momentum (classique ou de Nesterov) entraîne des réseaux profonds et récurrents qu’on jugeait jusque-là presque impossibles à entraîner ainsi. SGD + momentum est longtemps resté la référence en classification d’images : en 2017, l’article d’AdamW note qu’Adam y était en général battu par lui.

### 4. De AdaGrad à Adam, puis AdamW

Le momentum garde un seul η pour tous les paramètres. Or sur la vallée, le pas idéal en $x$ est 50 fois plus grand qu’en $y$. Les méthodes adaptatives divisent le pas de chaque paramètre par la taille typique de ses gradients.

AdaGrad (Duchi, Hazan et Singer, 2010 ; version de revue JMLR en 2011) divise par la racine de la somme des gradients au carré :

Un paramètre aux gradients forts reçoit un petit pas, un paramètre aux gradients rares (un mot rare, une variable creuse) un grand pas : c’était la motivation de l’article. Le défaut est dans la somme, qui ne fait que croître : le pas effectif rétrécit sans cesse. Dans l’exercice, AdaGrad ne passe sous le seuil sur aucune des trois surfaces ; au col, il cale à f = −0,169 au lieu de −0,25.

RMSProp (2012) remplace la somme par une moyenne glissante, qui oublie les vieux gradients :

RMSProp n’a jamais été publié en article : il apparaît dans la leçon 6e du cours en ligne de Geoffrey Hinton (2012, date donnée par l’article d’Adam), qui l’attribue à Tijmen Tieleman, « unpublished ».

Diviser par la taille du gradient a un effet spectaculaire au col : un gradient de $10^{-4}$ divisé par sa taille typique donne un pas de l’ordre de η, comme si la pente était normale. RMSProp va même plus loin au premier pas : sa moyenne $s$ part de 0, donc $\sqrt{s_1} = \sqrt{0{,}01\,g^2} = 0{,}1\,|g|$ et le premier pas vaut 10 η (c’est le défaut que la correction de biais d’Adam supprime). Mesuré : avec η = 0,05, $y$ passe de $10^{-4}$ à 0,4996 dès le premier pas. RMSProp sort du col en 9 pas, contre 202 pour SGD.

Adam (Kingma et Ba, 2014) réunit les deux idées : une moyenne glissante des gradients (le momentum, pour la direction) et une moyenne glissante de leurs carrés (RMSProp, pour l’échelle) :

Les valeurs par défaut (η = 0,001, β₁ = 0,9, β₂ = 0,999, ε = 10⁻⁸) sont celles de l’article (version publiée à ICLR 2015 ; la première version arXiv de décembre 2014 proposait η = 0,0002), et elles marchent souvent sans réglage : c’est une grande partie du succès d’Adam.

Pourquoi la correction de biais ? Au premier pas, $m_1 = 0{,}1\,g$ et $v_1 = 0{,}001\,g^2$. Sans correction, le pas vaudrait $\eta \times 0{,}1|g| / \sqrt{0{,}001\,g^2} \approx 3{,}16\,\eta$ : trois fois trop grand, parce que $v$ est sous-estimé encore plus que $m$. Avec la correction, $\hat m_1 = g$ et $\hat v_1 = g^2$, et le premier pas vaut η dans chaque coordonnée, au signe près (et à ε près).

C’est aussi ce qui fait sortir Adam du col : avec $g = 10^{-4}$ et η = 0,05, SGD fait un pas de $5 \cdot 10^{-6}$, Adam un pas d’environ 0,05, 10 000 fois plus grand. Mesuré : 35 pas contre 202.

Le code complet tient en quelques lignes (extrait de l’exercice, mêmes calculs que `torch.optim.Adam` à $3 \cdot 10^{-15}$ près) :

```python
class Adam:
    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m, self.v, self.t = 0.0, 0.0, 0

    def pas(self, p, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g        # direction (momentum)
        self.v = self.b2 * self.v + (1 - self.b2) * g * g    # échelle (RMSProp)
        m_hat = self.m / (1 - self.b1  self.t)             # corrections de biais
        v_hat = self.v / (1 - self.b2  self.t)
        return p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

Une limite à connaître : Adam dépend des axes. La normalisation se fait coordonnée par coordonnée. Elle répare la vallée étroite parce que celle-ci est alignée sur les axes $x$ et $y$. Tournez la même vallée de 45° : la courbure forte est maintenant répartie sur $x$ et $y$, et la normalisation ne la voit plus. Pour comparer à problème égal, il faut tourner aussi le point de départ : on part de (−4,95 ; −6,36) sur la vallée alignée et de (−8 ; 1) sur la vallée tournée, qui est le même point vu par la vallée. Calcul de la question de réflexion 1, mêmes η que dans l’exercice :

| Optimiseur | Vallée alignée | Vallée tournée de 45° |
|---|---|---|
| SGD | 186 pas | 186 pas |
| momentum | 96 pas | 96 pas |
| Nesterov | 33 pas | 33 pas |
| RMSProp | 76 pas | 472 pas |
| Adam | 95 pas | 309 pas |

SGD, momentum et Nesterov ne voient pas la rotation : leur pas est une somme de gradients, qui tourne avec la vallée. Adam et RMSProp, eux, divisent chaque coordonnée par sa propre échelle et deviennent 3 à 6 fois plus lents (309 pas contre 95 pour Adam, 472 contre 76 pour RMSProp).

AdamW (Loshchilov et Hutter, 2017) corrige un défaut discret, mais important pour les grands modèles. Le chapitre 00 a présenté la pénalité L2, qui ajoute $\tfrac{\lambda}{2}\|\theta\|^2$ à la perte et donc $\lambda\theta$ au gradient. Avec SGD, cela revient à rétrécir chaque poids d’une fraction $\eta\lambda$ à chaque pas : c’est la décroissance des poids. Avec Adam, ce n’est plus vrai : le terme $\lambda\theta$ passe dans $m$ et dans $v$, puis il est divisé par sa propre taille. Pour un poids qui ne reçoit pas d’autre gradient, $\hat m / \sqrt{\hat v} \approx \lambda\theta / |\lambda\theta| = \pm 1$ : λ disparaît.

AdamW sort la décroissance du gradient et l’applique directement aux poids. Dans la forme de PyTorch (λ multiplié par le pas η), juste avant le pas d’Adam :

L’article écrit la même idée un peu autrement : il multiplie λ par le seul facteur du planning du pas (pas par le pas de base) et applique la décroissance en même temps que le pas d’Adam, $\theta_t \leftarrow \theta_{t-1} - \eta_t\,\bigl(\alpha\,\hat m_t / (\sqrt{\hat v_t} + \varepsilon) + \lambda\,\theta_{t-1}\bigr)$. Les valeurs de λ ne se transposent donc pas telles quelles d’une convention à l’autre.

L’exercice le montre sur un poids $w_2$ qui part de 1 et ne reçoit aucun gradient de la perte. Avec Adam + L2, $w_2$ vaut 0,2244 au pas 100, pour λ = 0,01 comme pour λ = 0,1 : multiplier λ par 10 ne change rien. Avec AdamW, $w_2 = (1 - \eta\lambda)^t$ : au pas 500, $(1 - 10^{-4})^{500} \approx 0{,}951$ pour λ = 0,01, et $(1 - 10^{-3})^{500} \approx 0{,}606$ pour λ = 0,1. λ retrouve son sens.

AdamW est devenu l’optimiseur de référence des grands modèles de langage : en 2025, l’article de Moonshot AI sur Muon le présente encore comme l’optimiseur « de facto » qu’il s’agit de remplacer. Petite histoire : la première version de l’article (novembre 2017) s’intitulait « Fixing Weight Decay Regularization in Adam ».

### 5. Échauffement, cosinus, écrêtage, taille de lot

Le taux d’apprentissage n’a pas à rester constant. Un planning du taux d’apprentissage courant aujourd’hui :

- L’échauffement évite les gros pas au tout début, quand le réseau change vite et que les moyennes d’Adam reposent sur trop peu de gradients. Goyal et al. (2017) l’ont popularisé pour entraîner ResNet-50 sur ImageNet (un réseau de vision, chapitre 13, sur environ 1,28 million d’images) en une heure, avec des lots de 8 192 images et une règle simple : multiplier la taille du lot par $k$, c’est multiplier η par $k$ (« linear scaling rule »). Ils notent que cette règle ne tient pas pendant les premières époques, d’où l’échauffement.
- La décroissance en cosinus (Loshchilov et Hutter, SGDR, 2016) fait de grands pas au début, quand il faut avancer vite, et de tout petits à la fin, quand il faut se poser malgré le bruit des mini-lots.

Dans l’exercice, ce planning garde le meilleur des deux pas constants : il avance aussi vite que η = 0,4 au début, puis descend sous le plancher de bruit de η = 0,1. Excès de perte final : $2{,}8 \cdot 10^{-5}$, contre $6{,}5 \cdot 10^{-3}$ à pas constant, 230 fois mieux.

L’écrêtage du gradient (Pascanu, Mikolov et Bengio, 2013, contre les gradients explosifs des réseaux récurrents) plafonne la norme du gradient avant le pas :

La direction ne change pas, seule la longueur est bornée. Sur Rosenbrock avec η = 3·10⁻³, SGD diverge en 5 pas : le premier gradient, énorme sur le flanc de la banane, projette le point au loin. Avec la norme plafonnée à 10, il reste stable et finit en (0,92 ; 0,82).

Taille de lot. Un lot plus grand donne un gradient moins bruité et se parallélise mieux sur GPU, mais chaque pas coûte plus. On ne peut pas l’agrandir indéfiniment : dans les expériences de Goyal et al. sur ImageNet, la précision reste stable jusqu’à environ 8 000 images par lot, puis se dégrade rapidement. Cette limite dépend du problème.

### 6. Newton et L-BFGS : utiliser la courbure

Tous les optimiseurs précédents sont du premier ordre : ils n’utilisent que le gradient. La méthode de Newton utilise aussi la courbure, résumée par la hessienne $H$ (les dérivées secondes) :

Sur la vallée étroite, qui est exactement quadratique, $H = \mathrm{diag}(1, 50)$ et Newton arrive en un pas : il divise le gradient en $y$ par 50 et celui en $x$ par 1, ce que la descente de gradient ne sait pas faire. Sur Rosenbrock, Newton atteint (1 ; 1) à $10^{-8}$ près en 7 itérations, là où Adam met 1 564 pas pour passer sous $f = 10^{-3}$.

Le coût est rédhibitoire en grande dimension. Pour un modèle de $10^9$ paramètres, $H$ a $10^{18}$ coefficients, soit 4 exaoctets en float32, et résoudre le système coûte de l’ordre de $n^3$ opérations. Il y a pire : sur une perte non convexe, Newton cherche un point de gradient nul, pas un minimum, et Dauphin et al. (2014) montrent que les points selles deviennent alors attractifs. Et sur des gradients de mini-lots, le bruit s’ajoute aux estimations de courbure.

L-BFGS (Liu et Nocedal, 1989) est le compromis classique. Il garde les $m$ derniers déplacements $s_k = \theta_{k+1} - \theta_k$ et changements de gradient $y_k = \nabla\Loss_{k+1} - \nabla\Loss_k$, et en déduit l’effet de $H^{-1}$ sur un vecteur sans jamais former la matrice. L’article teste $m$ de 3 à 40 : passer de 3 à 7 accélère nettement la méthode sur 70 % des problèmes, et au-delà de 15 le gain devient faible. En pratique, on garde quelques vecteurs (scipy en garde 10 par défaut). Sur Rosenbrock, L-BFGS (scipy) converge en 44 itérations (54 évaluations de la fonction), erreur $1{,}1 \cdot 10^{-11}$. Il est excellent sur les problèmes déterministes de taille moyenne (régression logistique, ajustement de quelques milliers de paramètres) ; en pratique, il s’accommode mal du bruit des mini-lots, ce qui l’a écarté de l’entraînement des réseaux profonds.

### 7. Les optimiseurs récents : Muon et les autres

Muon a été présenté par Keller Jordan dans un billet du 8 décembre 2024. Il s’applique seulement aux matrices de poids des couches cachées ; les plongements (tables qui transforment chaque mot en vecteur, chapitre 15), la couche de sortie et les biais restent sous AdamW. Son idée : la mise à jour du momentum, vue comme une matrice $M$, a souvent un très mauvais conditionnement. Quelques directions dominent, les autres sont écrasées. Muon remplace $M$ par la matrice « orthogonalisée » la plus proche, $UV^\top$, où $M = U \Sigma V^\top$ est la décomposition en valeurs singulières : on garde les directions, on met toutes les amplitudes à 1.

Calculer une décomposition en valeurs singulières à chaque pas serait trop lent sur GPU. Muon l’approche par 5 itérations de Newton-Schulz, de simples produits de matrices :

Sur chaque valeur singulière σ, une itération applique le polynôme $a\sigma + b\sigma^3 + c\sigma^5$ : les petites valeurs sont multipliées par environ 3,4 à chaque tour, les grandes sont ramenées vers 1. Les coefficients sont choisis pour aller vite, pas pour converger exactement vers 1 : le résultat est approché. Mesuré dans l’exercice sur une mise à jour 64 × 32 : avant, les valeurs singulières vont de 0,018 à 8,4 (rapport 462) ; après, de 0,56 à 1,19.

En simplifiant (on omet la variante Nesterov et le facteur d’échelle selon la forme de la matrice), un pas de Muon s’écrit :

```python
def pas_muon(W, g, M, lr, mu=0.95):
    M = mu * M + g                      # momentum, comme SGD
                                        # (PyTorch : moyenne glissante M = μM + (1 − μ)g)
    O = newton_schulz(M, n_iter=5)      # ≈ U Vᵀ : toutes les directions à amplitude ~1
    return W - lr * O, M
```

Ce qu’on sait de son efficacité. Dans son billet, Jordan donne comme principale preuve le record de vitesse « NanoGPT speedrunning », battu en passant d’AdamW à Muon en octobre 2024. Liu et al. (Moonshot AI, 2025) ont ajouté une décroissance des poids et un réglage de l’échelle de la mise à jour, puis entraîné avec Muon Moonlight, un modèle à mélange d’experts (seule une partie des paramètres sert pour chaque token) de 16 milliards de paramètres (3 milliards actifs) ; d’après leurs lois d’échelle, Muon est environ deux fois plus efficace en calcul qu’AdamW. Kimi K2 (2025) a été pré-entraîné avec MuonClip, une variante de Muon, et la fiche du modèle DeepSeek-V4 sur Hugging Face (avril 2026) annonce l’emploi de l’optimiseur Muon. PyTorch 2.14 fournit `torch.optim.Muon`.

Dans l’exercice, Muon ne gagne pas : sur un jouet quadratique, Adam reste devant ($7{,}2 \cdot 10^{-7}$ contre $1{,}3 \cdot 10^{-5}$), et sur un petit perceptron multicouche (réseau à couches, chapitre 07) entraîné sur MNIST (2 époques), momentum, Adam, AdamW et Muon finissent à 0,15 point d’écart. L’intérêt de Muon se mesure à grande échelle, que l’exercice ne peut pas reproduire.

D’autres optimiseurs récents sont souvent cités. Leur adoption dans des entraînements de référence n’a pas été vérifiée pour ce cours :

| Optimiseur | Année | Idée (d’après le résumé de l’article) | Adoption |
|---|---|---|---|
| Lion (Chen et al.) | 2023 | trouvé par recherche automatique de programmes ; ne garde que le momentum, et la mise à jour prend le signe de chaque coordonnée | non vérifiée |
| Sophia (Liu et al.) | 2023 | second ordre allégé : estimation de la diagonale de la hessienne comme préconditionneur (matrice qui remet les directions à la même échelle), puis écrêtage élément par élément | non vérifiée |
| Schedule-Free (Defazio et al.) | 2024 | se passe de planning du pas (et donc de connaître à l’avance le nombre total de pas), grâce à une moyenne des itérés | non vérifiée |
| SOAP (Vyas et al.) | 2024 | fait tourner Adam dans la base propre du préconditionneur de Shampoo (Shampoo lui-même équivaut à Adafactor, une version économe d’Adam, dans cette base) | non vérifiée |

## Les optimiseurs à l’œuvre

La figure reprend les réglages de l’exercice : un même η pour SGD, momentum et Nesterov sur une surface donnée (pour isoler l’effet de la mémoire), un η propre aux méthodes adaptatives (leur pas se mesure en unités de paramètre, pas en unités de gradient).

Nombre de pas pour passer sous le seuil (vallée et Rosenbrock : $f$ inférieur à $10^{-3}$ ; col : $f$ inférieur à −0,24), mesuré dans l’exercice :

| Optimiseur | Vallée | Col | Rosenbrock (3 000 pas) |
|---|---|---|---|
| SGD | 205 | 202 | jamais |
| momentum | 73 | 54 | 666 |
| Nesterov | 53 | 49 | 557 |
| AdaGrad | jamais | jamais | jamais |
| RMSProp | 83 | 9 | jamais |
| Adam | 90 | 35 | 1 564 |

À vous de jouer. Choisissez une surface, cochez des optimiseurs, multipliez le pas par 0,25 à 4, et cliquez (ou touchez) la carte pour changer le point de départ. Le tableau sous la carte donne, pour chaque optimiseur, le pas où le seuil est franchi : avec les réglages par défaut, ce sont les chiffres de l’exercice.

À essayer : sur la vallée, poussez le multiplicateur à ×2 et regardez SGD exploser (η = 0,05, au-delà de la limite 0,04). Sur le col, comparez SGD et Adam, puis placez le départ exactement sur l’axe $y = 0$ : plus aucun gradient ne pousse hors du col.

## Ce qui coinçait

- Le réglage reste du bricolage. Adam a rendu les valeurs par défaut utilisables, mais η, le planning et λ se cherchent toujours par essais. Les réglages trouvés pour un petit modèle ne se transposent pas directement à un grand.
- La mémoire coûte. Adam et AdamW stockent deux nombres par paramètre ($m$ et $v$) : pour un modèle de 7 milliards de paramètres en float32, 56 Go rien que pour l’état de l’optimiseur. D’où les versions 8 bits et les optimiseurs plus économes (Lion ne garde que $m$).
- Les méthodes adaptatives dépendent des axes, comme le montre la vallée tournée. Muon et Shampoo (un préconditionneur matriciel de 2018) cherchent à corriger cela en traitant chaque matrice comme un tout.
- Trouver un minimum ne suffit pas. Un bon optimiseur fait baisser la perte d’entraînement ; savoir si le modèle se comporte bien sur de nouvelles données est une autre question, celle de la généralisation (chapitre 10). Les écarts de généralisation entre SGD et Adam ont d’ailleurs été une motivation d’AdamW.
- Le gradient doit exister. Tout ce chapitre suppose qu’on sait calculer $\nabla\Loss$ pour des millions de paramètres. C’est la rétropropagation, au chapitre 07, qui le rend possible.

- Descente de gradient (Cauchy, 1847) : Un pas vers le bas, de taille fixe, sur toutes les données.
- SGD (Robbins et Monro, 1951) : Une pente mesurée sur un échantillon suffit, si on accepte le bruit.
- Momentum (Polyak, 1964 ; Nesterov, 1983) : Garder l’élan : accélérer dans les directions constantes, amortir les zigzags.
- AdaGrad (2010), RMSProp (2012) : Un pas par paramètre, divisé par la taille de ses gradients.
- Adam (2014) : Momentum pour la direction, RMSProp pour l’échelle, correction du démarrage.
- AdamW (2017) : Rétrécir les poids à part, pour que λ garde son sens.
- Échauffement + cosinus : Grands pas pour avancer, petits pas pour se poser.
- Newton, L-BFGS (1989) : Utiliser la courbure : très peu de pas, mais chacun coûte cher.
- Muon (2024) : Donner la même amplitude à toutes les directions d’une mise à jour matricielle.

Et ensuite ? La boîte à outils est complète : des données, une perte (chapitre 01), un optimiseur (ce chapitre). Le chapitre 03 fait le tour des façons d’obtenir un signal d’apprentissage (supervisé, non supervisé, par renforcement). Puis le parcours historique commence par un retour en 1950 : avant d’apprendre à partir de données, l’IA a d’abord cherché à raisonner et à explorer des espaces de solutions, au chapitre 04.

## Exercice

Vous écrivez vous-même SGD, momentum, Nesterov, AdaGrad, RMSProp, Adam et AdamW en numpy (quelques lignes chacun), vous vérifiez qu’ils font exactement les mêmes calculs que `torch.optim` (écart maximal $3{,}3 \cdot 10^{-15}$ sur 500 pas), puis vous les comparez sur trois surfaces, une régression en mini-lots et un MLP sur MNIST. Le bonus implémente Newton-Schulz et Muon.

Ce qu’il faut observer :

- sur la vallée, SGD rampe dans la direction plate, le momentum va trois fois plus vite ;
- au col, les méthodes adaptatives font un pas de l’ordre de η même quand le gradient est minuscule (10 η au premier pas pour RMSProp, faute de correction de biais) ; AdaGrad finit par caler ;
- les mini-lots à pas constant butent sur un plancher de bruit, que la décroissance en cosinus fait disparaître ;
- sur le MLP MNIST (2 époques), SGD sans momentum atteint 95,04 % en test ; SGD + Nesterov, Adam, AdamW et Muon + AdamW finissent entre 97,03 % et 97,17 %.

Questions de réflexion : pourquoi Adam et RMSProp ralentissent-ils quand on tourne la vallée de 45°, alors que le momentum ne bouge pas ? Que garde le planning en cosinus du grand pas et du petit pas ?

## Auto-évaluation

Question : Sur f(x, y) = ½(x² + 50y²), pourquoi la descente de gradient diverge-t-elle pour η > 0,04, et pourquoi est-elle lente pour η = 0,025 ?

Réponse : Un pas multiplie $y$ par $1 - 50\eta$. Si $\eta > 2/50 = 0{,}04$, ce facteur est inférieur à $-1$ : $y$ change de signe et grandit à chaque pas, la descente diverge. À η = 0,025, le facteur en $x$ vaut $1 - \eta = 0{,}975$ : $x$ ne perd que 2,5 % par pas, il faut 205 pas pour passer sous $f = 10^{-3}$. La direction raide fixe le pas maximal, la direction plate fixe la vitesse.

Question : Avec μ = 0,9 et un gradient constant g, vers quelle valeur tend la vitesse du momentum ? Qu’est-ce que cela change sur une vallée étroite ?

Réponse : $b = 0{,}9\,b + g$ tend vers $g / (1 - 0{,}9) = 10\,g$ : le pas effectif est jusqu’à 10 fois plus grand dans les directions où le gradient garde son signe (le long de la vallée), alors que dans la direction où il alterne (en travers), les contributions s’annulent en partie. Dans l’exercice, le gain réel est ×2,8 (73 pas contre 205), car le momentum déborde aussi en travers.

Question : Pourquoi Adam sort-il d’un point selle bien plus vite que SGD, avec le même η = 0,05 ?

Réponse : Près du col, le gradient vaut environ $10^{-4}$. SGD fait un pas de $\eta g = 5 \cdot 10^{-6}$. Adam divise le gradient par sa taille typique $\sqrt{\hat v} \approx 10^{-4}$ : son pas vaut environ η = 0,05, quelle que soit la petitesse du gradient. Mesuré : 35 pas contre 202.

Question : Avec Adam, une pénalité L2 de force λ = 0,1 agit-elle plus fort que λ = 0,01 ? Et avec AdamW ?

Réponse : Avec Adam + L2, non : le terme $\lambda\theta$ entre dans $m$ et dans $v$, puis il est divisé par sa propre taille, et λ disparaît (dans l’exercice, $w_2 = 0{,}2244$ au pas 100 pour les deux valeurs). AdamW applique $\theta \leftarrow \theta(1 - \eta\lambda)$ hors du gradient : la décroissance est proportionnelle à λ (0,951 contre 0,606 au pas 500).

Question : La méthode de Newton converge en 7 itérations sur Rosenbrock, Adam en 1 564 pas. Pourquoi n’entraîne-t-on pas les réseaux avec Newton ?

Réponse : Il faut la hessienne : $n^2$ coefficients, soit $10^{18}$ pour $10^9$ paramètres, et la résolution d’un système en $O(n^3)$. Sur une perte non convexe, Newton est en plus attiré par les points selles, et le bruit des mini-lots fausse ses estimations de courbure. L-BFGS évite de former la hessienne, mais supporte mal ce bruit : il reste réservé aux problèmes déterministes de taille moyenne.

## Pour aller plus loin

- Articles originaux (fiches ci-dessus) : Robbins et Monro, « A Stochastic Approximation Method » (1951) ; Polyak, « Some methods of speeding up the convergence of iteration methods » (1964) ; Kingma et Ba, « Adam: A Method for Stochastic Optimization » (2014) ; Loshchilov et Hutter, « Decoupled Weight Decay Regularization » (2017) ; Liu et Nocedal, « On the limited memory BFGS method for large scale optimization » (1989) ; Nesterov, article de 1983 (en russe) ; Duchi, Hazan et Singer, « Adaptive Subgradient Methods for Online Learning and Stochastic Optimization » (2011) ; Sutskever et al., « On the importance of initialization and momentum in deep learning » (2013) ; Pascanu et al., « On the difficulty of training recurrent neural networks » (2013) ; Dauphin et al., « Identifying and attacking the saddle point problem in high-dimensional non-convex optimization » (2014) ; Goyal et al., « Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour » (2017).
- RMSProp à la source : G. Hinton, *Neural Networks for Machine Learning*, diapositives de la leçon 6 (leçon 6e pour RMSProp).
- Muon : K. Jordan, « Muon: An optimizer for hidden layers in neural networks » (billet du 8 décembre 2024) et son code ; J. Bernstein, « Deriving Muon », pour la justification théorique ; Liu et al., « Muon is Scalable for LLM Training » (2025).
- Pour l’intuition visuelle : G. Goh, « Why Momentum Really Works » (Distill, 2017), article interactif sur le momentum ; 3Blue1Brown, « Gradient descent, how neural networks learn ».
- Cours et synthèses : Stanford CS231n, notes « Neural Networks Part 3 » (mises à jour des paramètres, plannings du pas) ; S. Ruder, « An overview of gradient descent optimization algorithms » (2016) ; *Dive into Deep Learning*, chapitre 12 « Optimization Algorithms » ; Goodfellow, Bengio et Courville, *Deep Learning*, chapitre 8 « Optimization for Training Deep Models ».
- Documentation : `torch.optim` (PyTorch), qui documente les formules exactes de chaque optimiseur.
