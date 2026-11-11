# 07. Rétropropagation, MLP et fonctions d'activation

Repère historique (9 octobre 1986 (précurseurs de 1960 à 1982)) : Calculer efficacement le gradient de la perte par rapport à tous les poids d'un réseau multicouche, y compris ceux des neurones cachés dont personne ne connaît la sortie attendue.

En 1969, *Perceptrons* de Minsky et Papert a montré ce qu'un perceptron à une couche ne sait pas faire, et le connexionnisme a reflué (chapitre 06). Tout le monde savait qu'une couche cachée résout XOR ; personne ne savait la régler automatiquement. L'outil mathématique existait pourtant : le gradient d'une suite de calculs s'obtient en la parcourant à l'envers (Kelley en commande optimale, 1960 ; Linnainmaa pour les erreurs d'arrondi, 1970 ; Werbos, qui l'applique aux réseaux de neurones en 1982). Au début des années 1980, les réseaux reviennent par la physique : Hopfield (1982), puis la machine de Boltzmann (1985). Le 9 octobre 1986, David Rumelhart et Ronald Williams (Université de Californie à San Diego) et Geoffrey Hinton (Carnegie-Mellon) publient dans *Nature* une lettre de quatre pages : la rétropropagation des erreurs, démontrée sur des tâches où les neurones cachés inventent eux-mêmes une représentation utile. La lettre cite elle-même deux découvertes indépendantes, celles de David Parker et de Yann Le Cun (conférence Cognitiva 85, à Paris).

## L'idée en une image

Un réseau à plusieurs couches se trompe : il prédit 0,56 là où la réponse est 1. Il a des milliers de poids. Lequel faut-il bouger, dans quel sens, et de combien ? Pour les poids de la dernière couche, c'est facile : on voit directement leur effet sur la sortie. Pour les poids d'une couche cachée, rien ne dit ce que ses neurones « auraient dû » produire. C'est le problème de l'attribution du blâme, qu'on appelle plus souvent problème d’attribution du mérite ; la règle du perceptron butait dessus.

La rétropropagation répond par une idée de comptable : faire remonter l'erreur, de la sortie vers l'entrée, en la répartissant à chaque étage selon la part que chacun a prise au résultat. Chaque neurone reçoit de l'étage du dessus un message : « si ta sortie montait d'un cran, la perte varierait de tant ». Il multiplie ce message par sa propre sensibilité et le transmet à l'étage du dessous. Arrivé à un poids, le blâme vaut :

> gradient d'un poids = activation en amont × erreur qui arrive en aval
>
> (ce qui est entré dans le poids, multiplié par le message rétropropagé)

Le point décisif est le coût. Pour estimer le gradient « à la main », on peut bouger chaque poids un peu et regarder la perte : deux calculs du réseau par poids. Prenons le réseau de l'exercice : un perceptron multicouche, ou MLP, de 269 322 poids. Cela fait 538 644 passes. La rétropropagation obtient tous les gradients pour le prix d'une passe avant et d'une passe arrière, soit environ trois passes. C'est ce rapport de cinq ordres de grandeur qui rend l'apprentissage des réseaux de neurones possible.

| | Avant : perceptron (Rosenblatt, 1958) | Après : MLP et rétropropagation (1986) |
|---|---|---|
| Architecture apprise | une seule couche de poids | plusieurs couches, dont des couches cachées |
| Neurone | seuil (échelon) : dérivée nulle partout | fonction lisse (sigmoïde, puis ReLU) : dérivée utile |
| Règle d'apprentissage | corriger sur erreur, sortie par sortie | descente de gradient sur une perte, gradient calculé par la règle de la chaîne |
| Ce qu'on sait résoudre | classes linéairement séparables ; XOR impossible | XOR et toute frontière continue (approximation universelle) |
| Représentations | fixées à la main (les entrées) | apprises : les neurones cachés inventent leurs propres caractéristiques |
| Ce qui coince | la droite | la profondeur (le gradient s'éteint), le calcul, le surapprentissage |

## Le mécanisme

### 1. Le renouveau par la physique : Hopfield (1982) et Boltzmann (1985)

Avant la rétropropagation, deux modèles ont redonné une légitimité aux réseaux de neurones, en les reliant à la physique statistique. Ni l'un ni l'autre n'est un MLP, et ils ne s'entraînent pas par rétropropagation, mais ils ont apporté une idée qui reviendra souvent dans le parcours : une énergie qui baisse.

Le réseau de Hopfield est un réseau récurrent : $n$ neurones binaires $s_i \in \{-1, +1\}$, tous reliés entre eux par des poids symétriques ($w_{ij} = w_{ji}$, $w_{ii} = 0$). On met à jour un neurone à la fois, selon le signe de la somme pondérée qu'il reçoit. Hopfield montre qu'à chaque mise à jour une quantité ne peut que baisser ou rester constante :

(Formule : Énergie d'un réseau de Hopfield.)

En simplifiant : l'article de 1982 utilise des états 0/1 et des seuils ; la forme ±1 ci-dessus est la plus courante aujourd'hui. Pour mémoriser un motif $p$, on pose $w_{ij} = p_i\, p_j$ (une règle de Hebb, chapitre 06). Le motif devient un minimum de l'énergie, et le réseau y retombe à partir d'une version abîmée. Exemple avec trois neurones et le motif $p = (+1, -1, +1)$ :

```text
poids        : w12 = p1·p2 = −1    w13 = p1·p3 = +1    w23 = p2·p3 = −1
départ abîmé : s = (+1, +1, +1)                      E = −(w12 + w13 + w23) = −(−1 + 1 − 1) = +1
neurone 2    : w21·s1 + w23·s3 = (−1)(+1) + (−1)(+1) = −2   →   s2 = −1
arrivée      : s = (+1, −1, +1) = p                  E = −(1 + 1 + 1) = −3
```

C'est une mémoire adressable par le contenu : on la consulte avec un fragment, pas avec une adresse.

La machine de Boltzmann (Ackley, Hinton et Sejnowski, 1985) rend les neurones aléatoires (chacun s'allume avec une probabilité qui dépend de sa somme pondérée et d'une « température ») et ajoute des unités cachées. Elle apprend avec une règle locale élégante : $\Delta w_{ij} = \varepsilon\,(p_{ij} - p'_{ij})$, où $p_{ij}$ est la fréquence à laquelle les unités $i$ et $j$ sont allumées ensemble quand les données sont imposées, et $p'_{ij}$ la même fréquence quand le réseau tourne librement. Mesurer ces fréquences exige de longues simulations : l'apprentissage est très lent. La rétropropagation, un an plus tard, sera déterministe et bien plus rapide. Les machines de Boltzmann reviendront, sous une forme restreinte, dans le pré-entraînement des réseaux profonds en 2006 (chapitre 12), et les modèles à énergie au chapitre 26. En 2024, John Hopfield et Geoffrey Hinton ont reçu le prix Nobel de physique pour ces travaux fondateurs.

### 2. Le graphe de calcul et la règle de la chaîne

Toute la rétropropagation repose sur une règle de dérivation du lycée : la règle de la chaîne. Si $y = f(u)$ et $u = g(x)$, alors

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}.
$$

Un réseau de neurones est une longue composition de fonctions simples. On la dessine comme un graphe de calcul : chaque nœud est une opération élémentaire (produit par un poids, somme, sigmoïde, perte), chaque flèche une valeur intermédiaire. Pour un neurone unique, $\hat y = \sigma(w\,x + b)$ suivi d'une perte $\Loss$ :

Les flèches pleines sont la passe avant : on calcule et on garde chaque valeur intermédiaire. Les pointillés sont la passe arrière : chaque nœud reçoit la dérivée de la perte par rapport à sa sortie, la multiplie par sa dérivée locale (qu'il sait calculer seul, avec les valeurs gardées), et transmet le résultat à ses entrées. Aucun nœud n'a besoin de connaître le reste du réseau. Quand une valeur alimente plusieurs nœuds, les dérivées qui reviennent de chacun s'additionnent.

Pourquoi parcourir le graphe à l'envers plutôt qu'à l'endroit ? Parce qu'on a une seule sortie (la perte, un nombre) et beaucoup d'entrées (les poids). En partant de la perte, un seul parcours donne la dérivée par rapport à tout ce qui est en amont. En partant d'un poids, un parcours donne l'effet de ce poids sur tout ce qui est en aval : il en faudrait un par poids. C'est le mode inverse de la différentiation automatique, décrit par Seppo Linnainmaa en 1970 pour un tout autre usage (mesurer la propagation des erreurs d'arrondi). Linnainmaa n'est pas le seul ni même le premier : Andreas Griewank, qui a retracé cette histoire, montre que l'idée a été trouvée et retrouvée de nombreuses fois, avant et après 1986. Gerardi Ostrowski l'utilisait déjà quelques années plus tôt en génie chimique, Hachtel et ses collègues l'appliquaient aux circuits électroniques dans les années 1960, puis elle est réapparue en compilation et en prévision météorologique.

### 3. La rétropropagation pas à pas : un réseau 2-2-1

Tout le calcul, à la main, sur le plus petit réseau qui ait une couche cachée : 2 entrées, 2 neurones cachés à sigmoïde, 1 sortie à sigmoïde, perte d'entropie croisée binaire (chapitre 01). Ce sont les valeurs de l'exercice, vérifiables avec une calculatrice.

Chaque flèche porte son poids : $x_1$ envoie 0,5 vers $h_1$ et −0,5 vers $h_2$, $x_2$ envoie 0,3 vers $h_1$ et 0,8 vers $h_2$. Les poids sont rangés en matrices : $W_1 = \begin{pmatrix} 0{,}5 & -0{,}5 \\ 0{,}3 & 0{,}8 \end{pmatrix}$ (ligne $i$ = entrée $x_i$, colonne $j$ = neurone caché $h_j$), $W_2 = (1 ;\ -1)$, biais nuls. L'exemple est $x = (1, 0)$, de classe $y = 1$.

Passe avant. On calcule de gauche à droite et on garde tout :

| Étape | Calcul | Valeur |
|---|---|---|
| Entrée de la couche cachée | $z_1 = x\,W_1 = (1 \cdot 0{,}5 + 0 \cdot 0{,}3 ;\ 1 \cdot (-0{,}5) + 0 \cdot 0{,}8)$ | $(0{,}5 ;\ -0{,}5)$ |
| Sortie de la couche cachée | $h = \sigma(z_1)$ | $(0{,}6225 ;\ 0{,}3775)$ |
| Entrée de la sortie | $z_2 = h\,W_2 = 0{,}6225 - 0{,}3775$ | $0{,}2449$ |
| Prédiction | $\hat y = \sigma(z_2)$ | $0{,}5609$ |
| Perte | $\Loss = -\ln \hat y$ | $0{,}5782$ |

Passe arrière. On repart de la perte. Au chapitre 01, on a vu que pour une sigmoïde suivie de l'entropie croisée binaire, la dérivée par rapport au score se simplifie en « sortie moins cible ». On note $\delta$ (« erreur ») la dérivée de la perte par rapport à l'entrée $z$ d'un neurone :

| Étape | Formule | Valeur |
|---|---|---|
| Erreur en sortie | $\delta_2 = \partial\Loss/\partial z_2 = \hat y - y$ | $-0{,}4391$ |
| Gradient de $W_2$ | $\partial\Loss/\partial W_2 = h \cdot \delta_2$ | $(-0{,}2733 ;\ -0{,}1658)$ |
| Erreur qui redescend vers $h$ | $\partial\Loss/\partial h = \delta_2\, W_2 = (-0{,}4391 \times 1 ;\ -0{,}4391 \times (-1))$ | $(-0{,}4391 ;\ +0{,}4391)$ |
| Dérivée locale de la sigmoïde | $\sigma'(z_1) = h\,(1 - h)$ | $(0{,}2350 ;\ 0{,}2350)$ |
| Erreur des neurones cachés | $\delta_1 = \partial\Loss/\partial h \odot \sigma'(z_1)$ | $(-0{,}1032 ;\ 0{,}1032)$ |
| Gradient de $W_1$ | $\partial\Loss/\partial W_1 = x^\top \delta_1$ (ligne $i$ = $x_i \cdot \delta_1$) | $\begin{pmatrix} -0{,}1032 & 0{,}1032 \\ 0 & 0 \end{pmatrix}$ |

(Les gradients des biais valent les $\delta$ eux-mêmes : un biais est un poids dont l'entrée vaut toujours 1.)

Trois choses à lire dans ce tableau :

1. Gradient d'un poids = activation en amont × erreur en aval. La deuxième ligne de $\partial\Loss/\partial W_1$ est nulle parce que $x_2 = 0$ : un poids qui n'a rien transmis n'est pour rien dans l'erreur, quel que soit le reste du réseau.
2. Le signe guide la correction. $\delta_2$ est négatif (la sortie est trop basse) ; le neurone $h_1$, relié à la sortie par un poids positif, doit monter ($\delta_1$ négatif : la descente augmentera $z_1$) ; $h_2$, relié par un poids négatif, doit baisser.
3. La sigmoïde rabote l'erreur. En traversant $\sigma'$, l'erreur est multipliée par 0,235 : $|\delta_1| = 0{,}1032$ contre $|\delta_2| = 0{,}4391$. Retenez ce facteur, il reviendra en section 7.

Un pas de descente de gradient de taux d’apprentissage 1 ($W \leftarrow W - \partial\Loss/\partial W$) donne $W_1 = \begin{pmatrix} 0{,}6032 & -0{,}6032 \\ 0{,}3 & 0{,}8 \end{pmatrix}$, $W_2 = (1{,}2733 ;\ -0{,}8342)$. La nouvelle prédiction vaut 0,6290 et la perte passe de 0,5782 à 0,4636. L'exercice refait ce calcul avec PyTorch : les gradients coïncident à $5{,}6 \times 10^{-17}$ près, c'est-à-dire à l'arrondi de la machine.

### 4. Le cas général : n'importe quel nombre de couches

Un perceptron multicouche (MLP) enchaîne $L$ couches. La couche $\ell$ reçoit le vecteur $a_{\ell-1}$ de la couche précédente ($a_0 = x$), calcule une combinaison linéaire puis applique une fonction d’activation $f$ :

$$
z_\ell = a_{\ell-1}\, W_\ell + b_\ell, \qquad a_\ell = f(z_\ell),
$$

pour $\ell = 1, \dots, L$.

La dernière couche ne prend pas d'activation : ses sorties $z_L$ sont des logits, transformés en probabilités par la softmax de la perte. La rétropropagation tient alors en trois lignes, appliquées de $\ell = L$ jusqu'à $\ell = 1$ :

(Formule : Rétropropagation (convention de l'exercice : vecteurs en ligne).)

Pour un poids isolé, $w_{ij}$ allant du neurone $i$ d'une couche au neurone $j$ de la suivante, la deuxième formule s'écrit $\partial \Loss / \partial w_{ij} = a_i\, \delta_j$ : la phrase de l'intuition, mot pour mot. Sur un lot de $N$ exemples, $a_{\ell-1}$ et $\delta_\ell$ ont $N$ lignes et le produit matriciel $a_{\ell-1}^\top \delta_\ell$ fait la somme sur les exemples ; on divise par $N$ pour la moyenne.

Le code de l'exercice, sans les détails de stabilité numérique :

```python
import numpy as np

ACTIVATIONS = {  # (f, f') ; f' reçoit z et a = f(z) déjà calculé
    "sigmoïde": (lambda z: 1 / (1 + np.exp(-z)), lambda z, a: a * (1 - a)),
    "tanh":     (np.tanh,                         lambda z, a: 1 - a * a),
    "ReLU":     (lambda z: np.maximum(z, 0),      lambda z, a: (z > 0).astype(z.dtype)),
}

def propager(params, X, act):
    """Passe avant : renvoie les logits et le cache des (z, a) de chaque couche."""
    f = ACTIVATIONS[act][0]
    cache, a = [(None, X)], X
    for i, (W, b) in enumerate(params):
        z = a @ W + b
        a = z if i == len(params) - 1 else f(z)      # pas d'activation sur les logits
        cache.append((z, a))
    return a, cache

def retropropager(params, cache, y, act):
    """Passe arrière : gradients de l'entropie croisée moyenne pour tous les poids."""
    df = ACTIVATIONS[act][1]
    logits = cache[-1][0]
    delta = np.exp(logits - logits.max(axis=1, keepdims=True))
    delta /= delta.sum(axis=1, keepdims=True)        # softmax
    delta[np.arange(len(y)), y] -= 1                 # softmax − cible
    delta /= len(y)                                  # moyenne sur le lot
    grads = [None] * len(params)
    for i in range(len(params) - 1, -1, -1):         # de la dernière couche à la première
        a_prec = cache[i][1]
        grads[i] = [a_prec.T @ delta, delta.sum(axis=0)]   # ∂ℒ/∂W, ∂ℒ/∂b
        if i > 0:
            z, a = cache[i]
            delta = (delta @ params[i][0].T) * df(z, a)    # δ de la couche précédente
    return grads
```

Coût. La passe arrière fait les mêmes produits matriciels que la passe avant (avec les matrices transposées), plus un pour les gradients des poids : en pratique, deux à trois fois le coût d'une passe avant, quel que soit le nombre de poids. En contrepartie, il faut garder en mémoire toutes les activations de la passe avant : c'est la mémoire, plus que le calcul, qui limite la taille des lots sur un GPU.

Vérifier une passe arrière : le contrôle du gradient. Une passe arrière fausse ne plante pas : le réseau apprend mal, sans message d'erreur. Le premier test à écrire est le contrôle du gradient, qui compare chaque dérivée à une différence finie centrée :

$$
\frac{\partial \Loss}{\partial w} \approx \frac{\Loss(w + \varepsilon) - \Loss(w - \varepsilon)}{2\varepsilon},
$$

puis on mesure l'écart global entre le gradient rétropropagé $g_{\text{rétro}}$ et le gradient numérique $g_{\text{num}}$ :

$$
\text{erreur relative} = \frac{\lVert g_{\text{rétro}} - g_{\text{num}} \rVert}{\lVert g_{\text{rétro}} \rVert + \lVert g_{\text{num}} \rVert}.
$$

Sur un réseau 3-5-4-3 (59 paramètres, $\varepsilon = 10^{-6}$, en float64), l'exercice obtient :

| Activation | Passe arrière correcte | Passe arrière qui oublie $f'(z)$ |
|---|---|---|
| sigmoïde | $1{,}1 \times 10^{-9}$ | 0,41 |
| tanh | $7{,}9 \times 10^{-10}$ | 0,19 |
| ReLU | $8{,}3 \times 10^{-10}$ | 0,42 |

Un écart de l'ordre de $10^{-9}$ correspond aux arrondis ; un écart de 0,2 trahit une erreur. Le contrôle a coûté 118 passes avant pour 59 paramètres : on le fait sur un petit réseau, une fois, jamais pendant l'entraînement.

### Ce que la méthode optimise

La rétropropagation n'est pas un algorithme d'apprentissage : c'est une façon de calculer le gradient d'une perte. L'apprentissage, c'est la descente de gradient qui utilise ce gradient (chapitre 02). Elle marche pour n'importe quelle perte dérivable. Dans l'article de 1986, c'était la somme des carrés des écarts entre sorties et cibles, $E = \frac12 \sum_c \sum_j (\hat y_{j,c} - y_{j,c})^2$, sur les exemples $c$ et les sorties $j$, avec des sorties sigmoïdes (l'article note $y$ la sortie et $d$ la cible). C'est une somme d'erreurs quadratiques, sans moyenne. Aujourd'hui, en classification, on utilise presque toujours l'entropie croisée après une softmax : c'est celle de l'exercice, et elle donne l'erreur de sortie la plus simple possible, $\delta_L = \operatorname{softmax}(z_L) - y$. Fiche complète et comparaison avec les autres pertes au chapitre 01.

L'entropie croisée a un avantage concret sur la MSE avec une sortie sigmoïde : avec la MSE, l'erreur de sortie vaut $(\hat y - y)\,\sigma'(z)$, et $\sigma'$ est presque nulle quand la sortie est saturée, même si elle est fausse ($\hat y \approx 0$ pour $y = 1$). Le neurone le plus coupable apprend alors le plus lentement. Avec l'entropie croisée, le facteur $\sigma'$ disparaît et il reste $\hat y - y$.

L'article de 1986 entraîne avec le gradient cumulé sur tous les exemples et un terme d'inertie (le momentum, avec $\alpha = 0{,}9$). L'exercice utilise la descente de gradient stochastique par mini-lots de 64, la pratique actuelle.

### 5. Le MLP et l'approximation universelle

Pourquoi des couches cachées ? Parce qu'elles changent la représentation des données. Le chapitre 06 a construit à la main deux neurones cachés qui rendent le ou exclusif linéairement séparable ; la rétropropagation les trouve seule. Dans l'exercice, un XOR bruité de 400 points en quatre nuages :

| Modèle | Perte finale | exactitude en test (2 000 points) |
|---|---|---|
| Sans couche cachée (régression logistique) | 0,692 ≈ ln 2 | 45,8 % (le hasard) |
| MLP 2-16-2 (tanh, 3 000 pas, taux 0,5) | 0,014 | 99,5 % |

La perte du modèle sans couche cachée reste bloquée à $\ln 2 \approx 0{,}693$ : c'est la perte d'un modèle qui répond « 50/50 » partout. Aucune droite ne fait mieux sur un XOR. Avec 16 neurones cachés, la frontière contourne les nuages.

Jusqu'où va ce pouvoir ? En 1989, deux articles indépendants donnent une réponse surprenante. George Cybenko montre que les sommes de la forme

(Formule : Théorème d'approximation universelle (Cybenko, 1989).)

approchent d'aussi près qu'on veut toute fonction continue sur le cube $[0, 1]^n$, pourvu que $N$ soit assez grand. Hornik, Stinchcombe et White obtiennent la même année un résultat analogue pour une famille plus large de fonctions d'activation « écrasantes ». C'est le théorème d’approximation universelle : une seule couche cachée suffit, en principe.

Ce théorème est souvent mal lu. Il dit qu'un réseau existe ; il ne dit pas :

- combien de neurones il faut (pour certaines fonctions, le nombre nécessaire peut être énorme) ;
- comment trouver les poids : la descente de gradient peut ne pas les atteindre ;
- si le réseau généralisera à des exemples qu'il n'a pas vus (chapitre 10, chapitre 27).

Un polynôme de degré assez grand approche aussi toute fonction continue (théorème de Weierstrass) sans qu'on en fasse un modèle d'apprentissage universel. En pratique, on empile plusieurs couches plutôt que d'élargir une seule couche. Encore faut-il que le gradient traverse ces couches : c'est le rôle des fonctions d'activation.

### 6. Les fonctions d'activation en détail

Sans fonction d'activation, empiler des couches ne sert à rien : $(x W_1) W_2 = x\,(W_1 W_2)$, une seule couche linéaire. L'activation apporte la non-linéarité. La rétropropagation lui impose en plus une condition : avoir une dérivée utile, car $f'(z)$ multiplie l'erreur à chaque couche. Le chapitre 07 est le propriétaire de ces fonctions dans le parcours ; la figure de la rubrique Les activations côte à côte les trace toutes, avec leurs dérivées.

Les formules et leurs dérivées :

| Activation | $f(z)$ | $f'(z)$ |
|---|---|---|
| Échelon | 1 si $z > 0$, sinon 0 | 0 (non définie en 0) |
| Sigmoïde | $\sigma(z) = \frac{1}{1 + e^{-z}}$ | $\sigma(z)\,(1 - \sigma(z)) \le 0{,}25$ |
| Tanh | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - \tanh^2 z \le 1$ |
| ReLU | $\max(0, z)$ | 1 si $z > 0$, sinon 0 |
| Leaky ReLU | $z$ si $z > 0$, sinon $\alpha z$ | 1 ou $\alpha$ |
| ELU | $z$ si $z > 0$, sinon $\alpha(e^z - 1)$ | 1 ou $\alpha e^z$ |
| GELU | $z\,\Phi(z)$ | $\Phi(z) + z\,\varphi(z)$ |
| SiLU / Swish | $z\,\sigma(z)$ | $\sigma(z)\,\bigl(1 + z(1 - \sigma(z))\bigr)$ |
| Softmax | $\frac{e^{z_k}}{\sum_j e^{z_j}}$ | voir chapitre 01 |

Leurs valeurs, leur origine et leur usage :

| Activation | Valeurs | Origine | Usage typique |
|---|---|---|---|
| Échelon | {0, 1} | McCulloch et Pitts 1943 ; perceptron | aucun en apprentissage par gradient |
| Sigmoïde | ]0, 1[ | réseaux des années 1980 | sortie binaire, portes (LSTM) |
| Tanh | ]−1, 1[ | idem | RNN ; couches cachées avant 2010 |
| ReLU | [0, +∞[ | Nair et Hinton 2010 ; Glorot et al. 2011 | défaut des réseaux profonds |
| Leaky ReLU | ℝ | Maas, Hannun et Ng 2013 (α = 0,01) | éviter les neurones morts |
| ELU | ]−α, +∞[ | Clevert et al. 2015 | variante lisse de ReLU |
| GELU | ≈ [−0,17, +∞[ | Hendrycks et Gimpel 2016 | GPT, BERT, ViT |
| SiLU / Swish | ≈ [−0,28, +∞[ | Elfwing et al. 2017 ; Ramachandran et al. 2017 | dans SwiGLU |
| Softmax | probabilités | Bridle 1989 | dernière couche, classification |

($\Phi$ et $\varphi$ sont la fonction de répartition et la densité de la loi normale centrée réduite.)

#### Échelon : la raison pour laquelle le perceptron ne pouvait pas avoir de couches cachées

Le neurone formel et le perceptron utilisent un seuil. Sa dérivée est nulle partout (et non définie en 0) : dans la passe arrière, $\delta_{\ell-1} = (\delta_\ell W_\ell^\top) \odot 0 = 0$. Aucune information ne remonte. Remplacer le seuil par une fonction lisse qui lui ressemble, la sigmoïde, est ce qui rend la règle de la chaîne utilisable. Rumelhart, Hinton et Williams le notent : toute fonction de sortie à dérivée bornée convient.

#### Sigmoïde et tanh : lisses, mais saturantes

La sigmoïde $\sigma$ a deux défauts, visibles dans le calcul du 2-2-1 :

1. Sa dérivée ne dépasse pas 0,25 (atteint en $z = 0$, où $\sigma = 0{,}5$ et $\sigma' = 0{,}5 \times 0{,}5$). À chaque couche traversée, l'erreur est au moins divisée par 4.
2. Elle sature. Pour $z = 5$, $\sigma(5) = 0{,}9933$ et $\sigma'(5) = 0{,}0066$ : un neurone saturé ne transmet presque plus rien, qu'il ait raison ou tort.

S'y ajoute un défaut plus discret : ses sorties sont toujours positives (moyenne autour de 0,5), ce qui biaise les gradients de la couche suivante dans une même direction. La tangente hyperbolique (tanh) corrige ce dernier point : c'est une sigmoïde recentrée et étirée, $\tanh(z) = 2\,\sigma(2z) - 1$, à valeurs dans ]−1, 1[, de dérivée 1 en zéro. Elle sature aussi aux extrémités.

#### ReLU : la dérivée 1

La ReLU (unité linéaire rectifiée), $\max(0, z)$, a une dérivée qui vaut exactement 1 pour toute entrée positive : l'erreur traverse un neurone actif sans être réduite, quelle que soit la profondeur. Elle est aussi très peu coûteuse (une comparaison) et produit des représentations creuses : en moyenne une bonne partie des neurones sortent exactement 0. Nair et Hinton l'utilisent en 2010 dans des machines de Boltzmann restreintes ; Glorot, Bordes et Bengio montrent en 2011 que des réseaux profonds de ReLU s'entraînent bien par simple descente de gradient, sans pré-entraînement non supervisé, sur des tâches supervisées avec beaucoup de données étiquetées. AlexNet l'adopte en 2012 (chapitre 13), et elle devient l'activation par défaut des réseaux profonds.

Son défaut est le revers de sa qualité : pour $z$ négatif, la dérivée vaut 0. Un neurone dont l'entrée est négative pour tous les exemples (par exemple après un grand pas de gradient qui a trop baissé son biais) ne transmet plus aucun gradient : ses poids ne bougent plus jamais. C'est un neurone ReLU mort.

#### Leaky ReLU et ELU : ne jamais couper complètement

La ReLU à fuite garde une petite pente $\alpha$ pour $z$ négatif ($\alpha = 0{,}01$ chez Maas, Hannun et Ng ; la figure utilise 0,1 pour que la pente se voie). Le gradient n'est jamais nul, un neurone peut toujours revenir. L'unité linéaire exponentielle remplace la partie négative par une exponentielle qui tend vers $-\alpha$ : la fonction est lisse, et ses sorties négatives rapprochent de zéro la moyenne des activations, ce que la ReLU ne fait pas.

#### GELU et SiLU : des ReLU lisses

La GELU vaut $z\,\Phi(z)$ : l'entrée multipliée par la probabilité qu'une variable normale soit inférieure à $z$. Pour $z$ très positif, $\Phi(z) \approx 1$ et $f(z) \approx z$ ; pour $z$ très négatif, $f(z) \approx 0$ ; entre les deux, la courbe est lisse et descend légèrement sous zéro (minimum −0,17 vers $z \approx -0{,}75$). Dès sa deuxième version (juillet 2016), l'article signale une approximation par tanh, plus rapide à calculer. GPT (2018) et BERT (2018) l'ont adoptée.

La SiLU (Swish) vaut $z\,\sigma(z)$ : la même idée, avec la sigmoïde à la place de $\Phi$ (minimum −0,28 vers $z \approx -1{,}28$). Elfwing, Uchibe et Doya la proposent en février 2017 pour l'apprentissage par renforcement. En octobre 2017, Ramachandran, Zoph et Le cherchent de nouvelles activations par une recherche automatique (exhaustive dans les petits espaces de fonctions candidates, guidée par apprentissage par renforcement dans les grands, jusqu'à environ $10^{12}$ combinaisons). La meilleure, qu'ils nomment Swish, est définie comme $z\,\sigma(\beta z)$, avec $\beta$ constant ou appris ; avec $\beta = 1$, c'est la SiLU. C'est pourquoi on emploie souvent les deux noms pour la même fonction.

#### GLU et SwiGLU : l'activation devient une porte

Dernière étape, la plus récente. L'unité linéaire à porte (Dauphin et al., prépublication de 2016, publiée à ICML 2017) ne transforme plus une valeur : elle calcule deux projections de la même entrée et se sert de l'une comme d'une porte pour l'autre, $\operatorname{GLU}(x) = (x W) \odot \sigma(x V)$. En 2020, Noam Shazeer compare des variantes de cette porte dans la couche dense des Transformers et trouve que la version avec Swish, SwiGLU, améliore la qualité :

(Formule : Couche dense à SwiGLU (Shazeer, 2020).)

PaLM et LLaMA ont repris cette couche. Elle sera replacée dans le Transformer au chapitre 20.

En avance sur la chronologie :
Ce chapitre est daté de 1986, mais il est le propriétaire de toutes les fonctions d'activation du parcours, y compris ReLU (2010), GELU (2016), SiLU (2017) et SwiGLU (2020). Les chapitres suivants les rappellent en une ligne et renvoient ici. En 1986 et pendant les vingt années suivantes, on utilisait surtout la sigmoïde et la tanh.

### 7. Le gradient qui s'éteint en profondeur

Ce qui suit est le problème central de la période suivante ; on n'en montre ici que la cause liée aux activations. Reprenons la formule $\delta_{\ell-1} = (\delta_\ell W_\ell^\top) \odot f'(z_{\ell-1})$. En traversant $k$ couches, l'erreur est multipliée $k$ fois par une dérivée d'activation (et par des poids). Avec la sigmoïde, chaque facteur $f'$ vaut au plus 0,25 : après 8 couches, au mieux $0{,}25^8 \approx 1{,}5 \times 10^{-5}$. Les premières couches ne reçoivent presque rien. C'est la disparition du gradient.

L'exercice le mesure sur un réseau de 8 couches cachées de 32 neurones, avec la même initialisation pour les trois activations (poids tirés selon $\mathcal N(0, 1/n)$, $n$ = nombre d'entrées du neurone), 2 000 pas de descente de gradient de taux 0,05 sur le XOR bruité :

| Activation | Gradient couche 1 / couche 9 (couche de sortie), à l'initialisation | Perte finale | Pas pour passer sous 0,1 |
|---|---|---|---|
| sigmoïde | $7{,}8 \times 10^{-7}$ | 0,693 (bloquée à ln 2) | jamais |
| tanh | 0,22 | 0,012 | 86 |
| ReLU | 0,87 | 0,010 | 106 |

Avec la sigmoïde, la première couche reçoit un gradient environ un million de fois plus petit que la dernière, soit un facteur moyen d'environ 0,17 par couche ($0{,}172^8 \approx 7{,}7 \times 10^{-7}$), sous le plafond de 0,25. En 2 000 pas, le réseau n'apprend rien. Tanh (dérivée 1 en zéro) et ReLU (dérivée 1 pour $z$ positif) gardent un gradient du même ordre d'une couche à l'autre et apprennent en une centaine de pas.

Ce tableau dépend aussi de l'initialisation, volontairement identique ici : choisir l'échelle des poids selon l'activation (initialisations de Glorot et de He), le cas inverse de l'explosion du gradient et les autres remèdes (normalisation, connexions résiduelles) sont traités au chapitre 12 et au chapitre 13. La même disparition, dans le temps cette fois, frappe les réseaux récurrents (chapitre 16).

### 8. Autograd : la rétropropagation devenue automatique

Écrire la passe arrière à la main, comme en section 4, est instructif mais fragile : chaque nouvelle couche demande sa dérivée, et une erreur ne se voit pas. Les bibliothèques actuelles appliquent la différentiation automatique en mode inverse : pendant la passe avant, elles enregistrent le graphe de calcul ; à l'appel de `.backward()`, elles le parcourent à l'envers, chaque opération connaissant sa propre dérivée locale. C'est l'autograd de PyTorch :

```python
import torch

x  = torch.tensor([1.0, 0.0], dtype=torch.float64)
W1 = torch.tensor([[0.5, -0.5], [0.3, 0.8]], dtype=torch.float64, requires_grad=True)
W2 = torch.tensor([1.0, -1.0], dtype=torch.float64, requires_grad=True)

y_chap = torch.sigmoid(torch.sigmoid(x @ W1) @ W2)            # passe avant (graphe enregistré)
perte  = torch.nn.functional.binary_cross_entropy(y_chap, torch.tensor(1.0, dtype=torch.float64))
perte.backward()                                              # passe arrière

print(perte.item())   # 0.5782
print(W1.grad)        # [[-0.1032, 0.1032], [0., 0.]]
print(W2.grad)        # [-0.2733, -0.1658]
```

Le graphe enregistré se lit depuis `perte.grad_fn`. Pour ce réseau, l'exercice affiche, de la perte vers l'entrée :

```text
BinaryCrossEntropyBackward0 ← SigmoidBackward0 ← DotBackward0 ← SigmoidBackward0 ← SqueezeBackward4 ← MmBackward0
```

Chaque nom est un nœud du graphe de la section 2, avec sa dérivée locale. Sur le MLP MNIST de l'exercice (784-256-256-10, ReLU, 269 322 poids), les gradients d'autograd et ceux de notre `retropropager` en numpy diffèrent au plus de $8{,}7 \times 10^{-18}$ (en float64) : c'est le même calcul. Entraînés dix époques avec les mêmes réglages (lots de 64, taux 0,1), les deux atteignent 98,16 % (numpy) et 97,83 % (PyTorch) d'exactitude en test ; l'écart vient de l'initialisation et de l'ordre des mini-lots, pas du gradient.

Détail instructif : sur ce petit réseau, numpy sur le processeur (0,35 à 0,6 s par époque) va plus vite que PyTorch sur le GPU du Mac (0,62 à 0,73 s par époque). Avec des lots de 64, chaque opération est minuscule, et le coût fixe de chaque lancement de calcul sur le GPU domine. Le GPU ne gagne que quand les matrices sont grandes.

## Les activations côte à côte

La figure qui résume la section 6 : chaque activation en trait plein, sa dérivée en tirets. L'échelon n'a pas de dérivée utile ; celle de la sigmoïde plafonne à 0,25 ; celles de tanh, ReLU et de leurs successeurs valent 1 (ou presque) sur une large zone, ce qui laisse passer le gradient.

À vous de manipuler : cochez les activations, affichez leurs dérivées, déplacez le curseur (ou touchez le graphique) pour lire les valeurs. Cherchez le point où la dérivée de la sigmoïde est maximale, et sa valeur à $z = 4$. (La démo trace la GELU par son approximation tanh, la figure par sa forme exacte : les deux courbes se superposent à l'œil.)

## Ce qui coinçait

La rétropropagation a résolu le problème de 1969 : on sait entraîner des couches cachées. Elle n'a pas pour autant lancé le deep learning, qui attendra 2006-2012. Ce qui coinçait :

- La profondeur. Au-delà de quelques couches, avec des sigmoïdes, le gradient s'éteint (section 7). Remèdes : ReLU, initialisations adaptées, normalisation, connexions résiduelles (chapitres 12 et 13).
- Les minima locaux, réels ou supposés. La perte d'un MLP n'est pas convexe : rien ne garantit que la descente de gradient trouve le meilleur réseau. L'article de 1986 le signale lui-même, en notant que c'est rare en pratique et que des connexions supplémentaires aident. La crainte a pesé dans les années 1990 ; en grande dimension, on pense aujourd'hui que les points selles gênent davantage (chapitre 02).
- Le surapprentissage. Beaucoup de poids, peu de données (les tâches de l'article de 1986 comptent 64 et 100 exemples) : un réseau trop grand apprend par cœur (chapitre 00 ; régularisation des réseaux au chapitre 12).
- Le calcul. Sans GPU ni bibliothèque d'autograd, la puissance des ordinateurs de l'époque limite la taille des réseaux et des données ; chaque passe arrière s'écrit à la main.
- Le réglage. Taux d'apprentissage, nombre de neurones, initialisation, nombre d'époques : tout se règle à la main, sans théorie pour guider. L'exercice en donne un exemple : avec un taux de 0,1 au lieu de 0,05, la perte du réseau profond à ReLU fait un bond brutal vers 10,7 au pas 1 150 environ.
- La plausibilité biologique. La passe arrière transporte une erreur précise le long des connexions, à l'envers : rien ne montre que le cerveau fasse cela. Les auteurs de 1986 l'écrivent eux-mêmes : la procédure, « sous sa forme actuelle, n'est pas un modèle plausible de l'apprentissage dans le cerveau » (« in its current form, is not a plausible model of learning in brains », traduit).

- Perceptron (Rosenblatt, 1958) : Une droite corrigée sur erreur ; dérivée du seuil nulle, donc aucune couche cachée apprenable.
- Mode inverse (Kelley 1960, Linnainmaa 1970, Werbos 1982) : Le gradient d’une longue suite de calculs s’obtient en un seul parcours à l’envers.
- Hopfield (1982), Boltzmann (1985) : Une énergie qui baisse : mémoire associative, puis unités cachées apprises, mais lentement.
- Rétropropagation (Rumelhart, Hinton, Williams, 1986) : Gradient d’un poids = activation en amont × erreur en aval ; tous les gradients pour environ trois passes.
- Approximation universelle (1989) : Une couche cachée assez large approche toute fonction continue, sans dire comment la trouver.
- ReLU (2010-2011) : Une dérivée de 1 : le gradient traverse les couches sans s’éteindre.
- GELU, SiLU, SwiGLU (2016-2020) : Des ReLU lisses, puis l’activation qui devient une porte : la couche dense des LLM.
- Réseaux profonds entraînables (2006-2015) : Initialisation, normalisation, dropout, connexions résiduelles : faire passer le gradient sur des dizaines de couches.

Et ensuite ? La rétropropagation marche, mais dans les années 1990 les réseaux restent capricieux : peu profonds, longs à régler, sans garanties, gourmands en données qu'on n'a pas. Pendant ce temps, une autre école propose des méthodes efficaces sur des données de taille moyenne, avec des optimisations convexes et des garanties théoriques : arbres de décision, k plus proches voisins, machines à vecteurs de support, boosting. Elles domineront l'apprentissage automatique jusqu'au début des années 2010. C'est l'objet du chapitre 08 : Apprentissage automatique « classique ».

## Exercice

Ce qu'il faut observer :

- Le 2-2-1 à la main : chaque valeur de la section 3 (perte 0,5782 → 0,4636 après un pas), les gradients nuls des poids issus de $x_2 = 0$, et le même résultat par autograd à $5{,}6 \times 10^{-17}$ près.
- Le contrôle du gradient : $10^{-9}$ pour une passe arrière juste, 0,2 à 0,4 quand on oublie $f'(z)$.
- XOR : 45,8 % sans couche cachée (perte bloquée à ln 2), 99,5 % avec 16 neurones cachés.
- La profondeur : rapport de gradient couche 1 / couche 9 de $7{,}8 \times 10^{-7}$ avec la sigmoïde, contre 0,22 (tanh) et 0,87 (ReLU) ; la sigmoïde n'apprend rien en 2 000 pas.
- MNIST : 98,16 % en numpy, 97,83 % en PyTorch, gradients identiques à $10^{-17}$ près ; numpy sur CPU plus rapide que le GPU sur ce petit réseau.

Questions de réflexion : dans le 2-2-1, pourquoi les deux composantes de $\delta_1$ sont-elles opposées, et que deviendrait $\partial\Loss/\partial W_1$ avec $x = (1 ;\ 0{,}5)$ ? Combien de passes avant demanderait le gradient par différences finies pour un million de poids ? Que se passe-t-il pour un neurone ReLU qui reçoit une entrée négative pour tous les exemples ?

## Auto-évaluation

Question : Un neurone de sortie à sigmoïde, entraîné avec l'entropie croisée binaire, prédit ŷ = 0,8 pour un exemple de classe y = 0. Le neurone caché h₃ (activation 0,5) lui est relié par le poids w = 2. Que valent le gradient de ce poids et l'erreur transmise à h₃ ?

Réponse : Erreur de sortie : $\delta = \hat y - y = 0{,}8$. Gradient du poids : activation en amont × erreur en aval, $0{,}5 \times 0{,}8 = 0{,}4$ (la descente fera baisser $w$). Erreur qui redescend vers $h_3$ : $\partial\Loss/\partial h_3 = \delta \cdot w = 1{,}6$ ; si $h_3$ est une sigmoïde, on la multiplie ensuite par $\sigma'(z_3) = 0{,}5 \times 0{,}5 = 0{,}25$, soit $\delta_3 = 0{,}4$.

Question : Pourquoi la rétropropagation ne coûte-t-elle que l'équivalent de deux à trois passes avant, quel que soit le nombre de poids, alors que les différences finies coûtent deux passes par poids ?

Réponse : Parce qu'il y a une seule sortie (la perte, un nombre) et beaucoup d'entrées (les poids). En partant de la perte et en remontant le graphe, chaque nœud reçoit la dérivée de la perte par rapport à sa sortie et la transmet à ses entrées après une multiplication locale : un seul parcours fournit les dérivées par rapport à tout ce qui est en amont. Les différences finies mesurent l'effet d'un poids à la fois. C'est le mode inverse de la différentiation automatique. Le prix : garder en mémoire toutes les valeurs intermédiaires de la passe avant.

Question : Pourquoi un perceptron à seuils ne peut-il pas être entraîné par rétropropagation, et qu'ont changé les auteurs de 1986 ?

Réponse : La dérivée de l'échelon est nulle partout (et non définie en 0). Dans la passe arrière, l'erreur est multipliée par $f'(z) = 0$ à chaque couche : rien ne remonte vers les couches cachées. Rumelhart, Hinton et Williams utilisent des neurones à sortie sigmoïde, une version lisse du seuil dont la dérivée est non nulle ; ils notent que toute fonction à dérivée bornée convient.

Question : Dans un réseau de 10 couches cachées à sigmoïdes, quel est au mieux le facteur par lequel les dérivées d'activation réduisent l'erreur entre la sortie et la première couche ? Pourquoi ReLU change-t-elle la situation ?

Réponse : Chaque traversée multiplie l'erreur par $\sigma'(z) \le 0{,}25$ : au mieux $0{,}25^{10} \approx 9{,}5 \times 10^{-7}$, soit un million de fois moins (sans compter l'effet des poids). C'est la disparition du gradient : les premières couches n'apprennent presque pas. La dérivée de ReLU vaut 1 pour toute entrée positive : un neurone actif laisse passer l'erreur intacte. Contrepartie : elle vaut 0 pour une entrée négative, d'où le risque de neurones morts (corrigé par Leaky ReLU ou ELU).

Question : Le théorème d'approximation universelle garantit qu'une seule couche cachée suffit. Pourquoi utilise-t-on quand même des réseaux profonds, et que ne dit pas le théorème ?

Réponse : Le théorème dit qu'un réseau à une couche cachée existe pour approcher toute fonction continue, si la couche est assez large. Il ne dit ni combien de neurones il faut (parfois énormément), ni comment la descente de gradient pourrait trouver les poids, ni si le réseau généralisera à de nouveaux exemples. En pratique, empiler des couches donne de bons modèles avec beaucoup moins de paramètres, à condition que le gradient traverse les couches.

## Pour aller plus loin

- Articles originaux : Rumelhart, Hinton et Williams (1986, quatre pages très lisibles, avec les exemples de la symétrie et des arbres généalogiques), Hopfield (1982), Ackley, Hinton et Sejnowski (1985), Cybenko (1989), Nair et Hinton (2010), Glorot et al. (2011), Hendrycks et Gimpel (2016), Shazeer (2020) : liens dans les fiches ci-dessus.
- A. Griewank, « Who invented the reverse mode of differentiation? » (*Documenta Mathematica*, 2012) : l'histoire des redécouvertes du mode inverse, de Linnainmaa à la météorologie.
- 3Blue1Brown, série Neural networks : en particulier « Backpropagation, intuitively » et « Backpropagation calculus » (chapitres 3 et 4), l'intuition du blâme puis la règle de la chaîne, en animations.
- A. Karpathy, « The spelled-out intro to neural networks and backpropagation: building micrograd », première vidéo de la série Neural Networks: Zero to Hero : un autograd scalaire écrit de zéro, ligne par ligne (code de micrograd). Pour s'entraîner ensuite à la passe arrière matricielle : « Building makemore Part 4: Becoming a Backprop Ninja ».
- Stanford CS231n, « Backpropagation, Intuitions » : graphes de calcul, motifs du flux arrière (l'addition distribue, la multiplication échange, le max aiguille) et gradients vectorisés.
- C. Olah, « Calculus on Computational Graphs: Backpropagation » (2015) : pourquoi le mode inverse est plus rapide que le mode direct, avec des schémas.
- M. Nielsen, *Neural Networks and Deep Learning* (livre en ligne) : chapitre 2, « How the backpropagation algorithm works », et chapitre 4, « A visual proof that neural nets can compute any function », une preuve visuelle de l'approximation universelle.
- Dive into Deep Learning, « Forward Propagation, Backward Propagation, and Computational Graphs » ; documentation PyTorch, « Automatic Differentiation with torch.autograd ».
- A. G. Baydin et al., « Automatic differentiation in machine learning: a survey » : modes direct et inverse, et leur place en apprentissage automatique.
- Y. LeCun, « L'apprentissage profond : une révolution en intelligence artificielle », leçon inaugurale au Collège de France (2016), en français.
