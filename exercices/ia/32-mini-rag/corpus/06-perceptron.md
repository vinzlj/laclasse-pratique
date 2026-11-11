# 06. Neurone formel, perceptron, XOR et premier hiver

Repère historique (1943 (McCulloch et Pitts) à 1969 (Minsky et Papert)) : Apprendre une règle de décision à partir d'exemples, avec une mise à jour locale : chaque erreur corrige un peu les poids, sans que personne n'écrive la règle à la main.

En 1943, le neurophysiologiste Warren McCulloch et le logicien Walter Pitts, à Chicago, montrent qu'un neurone idéalisé, qui s'allume ou non selon un seuil, se comporte comme une porte logique. Leur neurone calcule, mais il n'apprend pas : on règle ses seuils à la main. En 1949, le psychologue Donald Hebb, à McGill, propose un mécanisme d'apprentissage biologique, le renforcement des connexions qui servent ensemble, sans en donner de formule. En janvier 1957, Frank Rosenblatt, psychologue au Cornell Aeronautical Laboratory de Buffalo, décrit le perceptron : un réseau d'unités à seuil dont les connexions se renforcent avec l'expérience. Il le publie en 1958 dans *Psychological Review*, et la marine américaine le présente à la presse en juillet de la même année. La règle qui ne modifie les poids qu'en cas de mauvaise réponse, celle qu'on appelle aujourd'hui « règle du perceptron », est formalisée dans ses travaux suivants : la machine Mark I apprend ainsi en 1960, et *Principles of Neurodynamics* (1962) l'expose avec son théorème de convergence. En 1969, *Perceptrons*, de Marvin Minsky et Seymour Papert (MIT), analyse ce qu'un perceptron à une couche ne peut pas calculer. Le livre accompagne le reflux du courant neuronal pendant les années 1970.

## L'idée en une image

Un perceptron est un jury qui vote avec des voix pondérées. Chaque caractéristique de l'entrée (la taille d'un animal, son degré de domestication…) apporte sa voix, multipliée par un poids : positif si elle plaide pour la classe +1, négatif si elle plaide contre. On additionne, on ajoute un biais (une préférence de départ), et on tranche selon le signe du total. Géométriquement, cette règle coupe le plan en deux par une droite : d'un côté la classe +1, de l'autre la classe −1.

L'apprentissage tient en une phrase : quand le jury se trompe sur un exemple, on déplace la droite vers cet exemple ; quand il a raison, on ne touche à rien. Pas de calcul global, pas de formule fermée : chaque exemple, l'un après l'autre, corrige un peu les poids. C'est cette mise à jour « locale », pilotée par l'erreur, qui fait la nouveauté du perceptron.

La limite tient aussi en une phrase : une droite ne peut séparer que ce qui est séparable par une droite. Le cas d'école est le ou exclusif : « l'une ou l'autre entrée, mais pas les deux ». Aucune droite ne met (0, 1) et (1, 0) d'un côté, (0, 0) et (1, 1) de l'autre. Il faut deux couches de neurones, et en 1969 personne ne sait les entraîner.

| | McCulloch et Pitts (1943) | Hebb (1949) | Rosenblatt (1957-1962) |
|---|---|---|---|
| Ce que fait le neurone | compare une somme d'entrées binaires à un seuil | somme pondérée, seuil | somme pondérée plus biais, signe |
| D'où viennent les poids | câblés à la main | renforcés quand entrée et sortie sont actives ensemble | corrigés à chaque erreur de classement |
| Ce qu'on sait faire | toute fonction logique, avec assez de neurones câblés | retrouver ET ; la droite des nuages séparables de l'exercice ne classe que 54 % des points | séparer sans erreur des données séparables (100 % dans l'exercice, en 95 corrections) |
| Ce qui coince | rien n'est appris | la règle renforce sans jamais regarder l'erreur | une seule droite : XOR est hors de portée |

## Le mécanisme

Le flux de données d'un perceptron, de l'entrée à la mise à jour des poids :

### 1. Le neurone formel de McCulloch et Pitts (1943)

Le neurone formel idéalise le neurone biologique tel qu'on le voit en 1943 : il est « tout ou rien », il décharge ou non. Ses entrées valent 0 ou 1 ; certaines sont excitatrices, d'autres inhibitrices. Il s'allume si aucune entrée inhibitrice n'est active et si le nombre d'entrées excitatrices actives atteint son seuil $\theta$ :

Avec deux entrées $a$ et $b$, il suffit de choisir le seuil : $\theta = 2$ donne ET, $\theta = 1$ donne OU. NON $a$ s'obtient avec $a$ en entrée inhibitrice et un seuil nul (le neurone s'allume tant que $a$ ne l'en empêche pas) ; « $a$ ET NON $b$ » avec $a$ excitatrice, $b$ inhibitrice et $\theta = 1$. L'exercice vérifie les quatre tables. Comme ET, OU et NON suffisent à écrire n'importe quelle fonction logique, un réseau de neurones formels assez grand peut calculer n'importe quelle fonction de ses entrées binaires. C'est la thèse de l'article de 1943 : le système nerveux peut être vu comme une machine logique.

La figure ci-dessus montre la forme qui s'est imposée ensuite : chaque entrée reçoit un poids réel au lieu d'être simplement excitatrice ou inhibitrice, et le seuil devient un nombre réel. C'est ce neurone-là que le perceptron va rendre apprenant. Le neurone formel, lui, ne l'est pas : quelqu'un doit choisir les seuils et le câblage.

### 2. La règle de Hebb (1949) : renforcer ce qui sert ensemble

En 1949, dans *The Organization of Behavior*, Donald Hebb énonce un postulat (p. 62). En substance : quand un neurone A participe de façon répétée à faire décharger un neurone B, un changement se produit dans l'un ou les deux, qui rend A plus efficace pour activer B. Hebb ne donne aucune formule. La règle de Hebb en est une traduction ultérieure, en notant $\tilde x = (x_1, x_2, 1)$ l'entrée augmentée d'un 1 (on y reviendra) et $y = \pm 1$ la sortie voulue :

Le pas d'apprentissage $\eta$ règle la taille de chaque modification. Sur ET en codage ±1, avec $\eta = 1$, on additionne $y\,\tilde x$ pour les quatre exemples :

| Entrée $(x_1, x_2)$ | $y$ | $y\,\tilde x$ |
|---|---|---|
| (−1, −1) | −1 | (1 ; 1 ; −1) |
| (−1, +1) | −1 | (1 ; −1 ; −1) |
| (+1, −1) | −1 | (−1 ; 1 ; −1) |
| (+1, +1) | +1 | (1 ; 1 ; 1) |
| Somme | | (2 ; 2 ; −2) |

Le neurone obtenu calcule $2x_1 + 2x_2 - 2$ : positif seulement pour (+1, +1). ET est appris, comme dans l'exercice. Mais la règle a un défaut de conception : elle renforce à chaque exemple, qu'il soit bien ou mal classé. Elle ne regarde jamais l'erreur. Sur les deux nuages de 200 points de l'exercice, pourtant séparables par une droite, elle donne $\mathbf{w} = (217{,}6\,;\ 169{,}5\,;\ 0)$ et seulement 54,0 % de bonnes réponses. Le biais vaut la somme des $y$, nulle quand les classes sont équilibrées : la droite passe par l'origine, loin des données. Les poids, eux, pointent dans la direction qui va de la moyenne d'une classe à celle de l'autre, sans aucun moyen de se corriger.

### 3. Le perceptron de Rosenblatt (1957-1962) : corriger sur erreur

Le perceptron garde le neurone à poids et change la règle : on ne modifie les poids que sur un exemple mal classé. Le calcul du neurone :

(Formule : Décision du perceptron.)

La frontière de décision est l'ensemble des points où le score s'annule, $\mathbf{w}\cdot\mathbf{x} + b = 0$ : une droite en dimension 2, un plan en dimension 3, un hyperplan au-delà. Le vecteur $\mathbf{w}$ est perpendiculaire à cette droite et pointe vers le côté +1.

(Formule : Règle d'apprentissage du perceptron.)

Pourquoi cette correction va dans le bon sens : après la mise à jour, le score de l'exemple corrigé devient

$$
y\,(\tilde{\mathbf w} + \eta\,y\,\tilde{\mathbf x})\cdot\tilde{\mathbf x} = y\,(\tilde{\mathbf w}\cdot\tilde{\mathbf x}) + \eta\,\lVert\tilde{\mathbf x}\rVert^2,
$$

puisque $y^2 = 1$. La quantité $y\,(\tilde{\mathbf w}\cdot\tilde{\mathbf x})$ augmente de $\eta\,\lVert\tilde{\mathbf x}\rVert^2 > 0$ : la droite pivote et se décale vers l'exemple. Elle ne passe pas forcément de l'autre côté en une fois, et elle peut au passage dérégler d'autres exemples. C'est pour cela qu'on fait plusieurs passages sur les données, des époques, jusqu'à une époque sans erreur.

Exemple chiffré : OU. Entrées 0/1, étiquettes $y = -1$ pour (0, 0) et $+1$ sinon, $\tilde{\mathbf w}$ part de $(0\,;\ 0\,;\ 0)$, $\eta = 1$. Première époque :

| Exemple $\tilde x$ | $y$ | score $\tilde w\cdot\tilde x$ | $y \times$ score | Action | $\tilde w$ après |
|---|---|---|---|---|---|
| (0 ; 0 ; 1) | −1 | 0 | 0 | erreur : $\tilde w - \tilde x$ | (0 ; 0 ; −1) |
| (0 ; 1 ; 1) | +1 | −1 | −1 | erreur : $\tilde w + \tilde x$ | (0 ; 1 ; 0) |
| (1 ; 0 ; 1) | +1 | 0 | 0 | erreur : $\tilde w + \tilde x$ | (1 ; 1 ; 1) |
| (1 ; 1 ; 1) | +1 | 3 | 3 | bien classé | (1 ; 1 ; 1) |

À la fin de l'époque, (0, 0) a un score de +1 et reste mal classé. En continuant, le perceptron s'arrête après 9 corrections en 6 époques avec $\tilde{\mathbf w} = (2\,;\ 2\,;\ -1)$, soit la frontière $2x_1 + 2x_2 = 1$ : seul (0, 0) est du côté négatif. On le vérifie à la main en une minute.

Le code complet tient en dix lignes ; c'est celui de l'exercice :

```python
import numpy as np

def perceptron(X, y, eta=1.0, epoques_max=1000):
    """X : (n, d) entrées réelles ; y : (n,) étiquettes ±1."""
    Xb = np.c_[X, np.ones(len(X))]          # entrée augmentée : le biais devient un poids
    w = np.zeros(Xb.shape[1])
    for _ in range(epoques_max):
        erreurs = 0
        for xi, yi in zip(Xb, y):           # un exemple à la fois, dans l'ordre
            if yi * (w @ xi) <= 0:          # mal classé (ou sur la frontière)
                w = w + eta * yi * xi       # on tire la frontière vers le bon côté
                erreurs += 1
        if erreurs == 0:                    # une époque sans erreur : terminé
            break
    return w
```

Sur les 200 points de l'exercice, étiquetés par la vraie frontière $x_1 + 2x_2 = 6$ avec une bande vide de ±0,1 autour d'elle, ce code fait 95 corrections en 7 époques. Les erreurs par époque valent 42, 17, 11, 8, 16, 1, puis 0 : elles ne décroissent pas toujours (8 puis 16), car corriger un point en dérègle d'autres. Les poids finaux $(3{,}26\,;\ 7{,}39\,;\ -21)$ donnent la droite $3{,}26\,x_1 + 7{,}39\,x_2 = 21$, proche de la vraie (multipliée par 3,5, elle s'écrit $3{,}5\,x_1 + 7\,x_2 = 21$). `sklearn.linear_model.Perceptron`, avec les mêmes réglages et le même ordre des exemples, donne exactement les mêmes poids.

En simplifiant : le perceptron de Rosenblatt était plus qu'un neurone :
Le perceptron de 1958 est un système à trois étages, pensé comme un modèle du cerveau : une rétine d'unités sensorielles (S), reliées, en partie au hasard et de façon fixe, à des unités d'association (A) (dans la figure ci-dessous, les connexions vers l'aire de projection sont topographiques ; seules celles vers les unités A sont aléatoires), elles-mêmes reliées aux unités de réponse (R). Seules les connexions A → R s'apprennent. Ce qu'on appelle aujourd'hui « le perceptron » est une unité R avec sa règle de correction, appliquée directement aux entrées. La règle vectorielle $\mathbf w \leftarrow \mathbf w + \eta\,y\,\mathbf x$ est elle aussi une écriture moderne. Le rapport de 1957 décrit un renforcement des unités A actives quand la réponse voulue l'emporte, sans condition d'erreur ; la « procédure de correction d'erreur » et sa preuve de convergence viennent avec *Principles of Neurodynamics* (1962, théorème 4), parmi plusieurs variantes d'apprentissage.

### Ce que la méthode optimise

Le perceptron a l'air d'une recette (« si erreur, corriger »), mais c'est exactement une descente de (sous-)gradient exemple par exemple, « en ligne », sur une fonction de perte, le critère du perceptron. Quand l'ordre des exemples est tiré au hasard, cette forme s'appelle descente de gradient stochastique (chapitre 02) ; ici l'ordre est fixe :

(Formule : Critère du perceptron.)

La perte a un coin en 0 : à cet endroit, elle n'a pas de dérivée, et on prend l'une des pentes possibles, ici celle de la branche non nulle. C'est ce qu'on appelle un sous-gradient. Un pas de descente $\tilde{\mathbf w} \leftarrow \tilde{\mathbf w} - \eta\,\partial\ell$ donne $\tilde{\mathbf w} + \eta\,y\,\tilde{\mathbf x}$ sur une erreur, rien sinon : c'est la règle de Rosenblatt, mot pour mot. L'exercice réécrit la boucle sous cette forme et obtient des poids identiques au bit près. Avec la marge $m = y\,f(x)$ du chapitre 01, ce critère s'écrit $\max(0, -m)$ : c'est la perte charnière $\max(0, 1 - m)$ sans la marge de 1. Le perceptron se contente d'être du bon côté ; la perte charnière exige d'y être avec de la marge, ce qui mènera aux machines à vecteurs de support (chapitre 08).

Piège : le minimum trivial :
En $\tilde{\mathbf w} = 0$, tous les scores sont nuls et le critère du perceptron vaut déjà 0, son minimum. Une descente de gradient qui ne regarderait que $y\,\tilde{\mathbf w}\cdot\tilde{\mathbf x} < 0$ (inégalité stricte) ne bougerait jamais. C'est la condition « $\le 0$ », qui compte un point posé sur la frontière comme une erreur, qui fait démarrer l'apprentissage. Dans l'exercice, le critère moyen vaut 0 en $\tilde{\mathbf w} = 0$, 2,575 après la première correction, au plus 5,732, puis 0 à la fin : il n'a rien d'une courbe qui descend gentiment. Ce n'est pas un défaut de calcul : ce critère, nul pour toute droite qui sépare les données, ne préfère aucune d'entre elles.

### 4. Le théorème de convergence (Rosenblatt, Block, Novikoff, 1962)

Sur des données séparables, le perceptron finit-il toujours ? Oui, et on sait borner le nombre de corrections. Rosenblatt et le mathématicien H. D. Block, de Cornell, le démontrent chacun en 1962 ; la preuve courte en $(R/\gamma)^2$, reprise ici, est celle d'Albert Novikoff, la même année. On suppose qu'il existe un vecteur unitaire $\mathbf u$ ($\lVert \mathbf u \rVert = 1$) qui classe tous les exemples avec une marge $\gamma > 0$, que toutes les entrées augmentées sont dans une boule de rayon $R$, et que le perceptron part de $\tilde{\mathbf w}_0 = 0$ :

(Formule : Théorème de convergence du perceptron.)

La preuve tient en trois lignes, avec $\tilde{\mathbf w}_0 = 0$ et $\eta = 1$ :

1. Le produit avec $\mathbf u$ grandit vite. Chaque correction ajoute $y\,\tilde{\mathbf x}$, donc ajoute $y\,\mathbf u\cdot\tilde{\mathbf x} \ge \gamma$ à $\mathbf u\cdot\tilde{\mathbf w}$. Après $k$ corrections : $\mathbf u\cdot\tilde{\mathbf w}_k \ge k\gamma$.
2. La norme grandit lentement. $\lVert\tilde{\mathbf w} + y\tilde{\mathbf x}\rVert^2 = \lVert\tilde{\mathbf w}\rVert^2 + 2\,y\,\tilde{\mathbf w}\cdot\tilde{\mathbf x} + \lVert\tilde{\mathbf x}\rVert^2$, et le terme du milieu est négatif ou nul puisqu'on ne corrige que sur une erreur. Donc $\lVert\tilde{\mathbf w}_k\rVert^2 \le k R^2$.
3. On compare. Un produit scalaire avec un vecteur unitaire ne dépasse pas la norme : $k\gamma \le \mathbf u\cdot\tilde{\mathbf w}_k \le \lVert\tilde{\mathbf w}_k\rVert \le \sqrt{k}\,R$, d'où $k \le (R/\gamma)^2$.

La borne ne dépend ni du nombre d'exemples, ni de leur ordre, ni de la dimension : seulement du rapport entre la taille des données et la marge. Dans l'exercice, $R = 6{,}63$ et $\gamma = 0{,}0445$ donnent une borne de $(6{,}63 / 0{,}0445)^2 \approx$ 22 195, pour 95 corrections réelles. Ici $\gamma$ est la marge de la frontière du SVM (la droite de marge maximale dans le plan), mesurée dans l'espace augmenté, où le biais compte dans la norme : elle est plus petite que la marge géométrique dans le plan (0,1286), parce que la frontière passe à environ 2,7 unités de l'origine ($0{,}1286 / \sqrt{1 + 2{,}71^2} \approx 0{,}0445$). N'importe quel séparateur $\mathbf u$ donne une borne valable ; celle-ci n'est pas forcément la plus serrée.

Sur 50 jeux de données (5 largeurs de bande, 10 graines), la borne est toujours respectée, et le nombre moyen de corrections passe de 13,6 à 112,6 quand la marge moyenne diminue de 0,356 à 0,029. La borne dépasse les corrections observées de 26 à 540 fois en moyenne : elle doit valoir pour le pire ordre de présentation et les pires points de mêmes $R$ et $\gamma$. Elle est donc très pessimiste, mais elle donne la bonne tendance : moins de marge, plus de corrections.

Deux limites découlent de la preuve. Sans séparabilité, il n'existe aucun $\mathbf u$ et rien ne garantit l'arrêt. Et le perceptron s'arrête sur la première droite qui sépare les données, pas sur la meilleure : dans l'exercice, sa droite passe à 0,0096 du point le plus proche, quand la droite de marge maximale laisse 0,1286 de chaque côté, 13 fois plus. Chercher cette droite-là est l'idée des SVM (chapitre 08).

### 5. Séparabilité linéaire et XOR

Un jeu de données à deux classes est en séparabilité linéaire si une droite (un hyperplan) le sépare sans erreur. Le théorème précédent ne s'applique qu'à ce cas. Le contre-exemple minimal est XOR, sur les quatre coins du carré :

| $x_1$ | $x_2$ | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Preuve qu'aucune droite ne convient. Supposons que $w_1 x_1 + w_2 x_2 + b$ soit positif exactement sur les deux points de classe 1. Alors (0, 0) impose $b \le 0$ ; (1, 0) et (0, 1) imposent $w_1 + b \gt 0$ et $w_2 + b \gt 0$. En additionnant ces deux dernières, $w_1 + w_2 + 2b \gt 0$, donc $w_1 + w_2 + b \gt -b \ge 0$. Or (1, 1) exige $w_1 + w_2 + b \lt 0$ : contradiction. La meilleure droite classe 3 points sur 4, et 200 000 droites tirées au hasard dans l'exercice ne font jamais mieux que 75 %.

Ce que fait le perceptron. Il tourne en rond, et on peut le suivre à la main (étiquettes $y = -1$ pour la classe 0, $\tilde{\mathbf w}$ part de 0). Les trois premiers exemples sont les mêmes que pour OU, donc les trois premières corrections aussi : $\tilde{\mathbf w}$ passe par $(0\,;\ 0\,;\ -1)$, $(0\,;\ 1\,;\ 0)$, $(1\,;\ 1\,;\ 1)$. Puis (1, 1), de classe −1, a un score de 3 : erreur, et $\tilde{\mathbf w} - (1\,;\ 1\,;\ 1) = (0\,;\ 0\,;\ 0)$. Retour au point de départ après 4 erreurs : l'époque suivante est identique, et ainsi de suite. L'exercice détecte ce cycle ; les meilleurs poids visités ne classent que 50 % des points, et scikit-learn ne fait pas mieux.

Ce que font deux couches. Le problème disparaît si l'on ajoute une couche cachée de deux neurones, câblés à la main :

- $h_1 = \text{OU}(x_1, x_2)$ : s'allume si $x_1 + x_2 - 0{,}5 \gt 0$ ;
- $h_2 = \text{ET}(x_1, x_2)$ : s'allume si $x_1 + x_2 - 1{,}5 \gt 0$ ;
- sortie $= h_1$ ET NON $h_2$ : s'allume si $h_1 - h_2 - 0{,}5 \gt 0$.

| $(x_1, x_2)$ | $h_1$ (OU) | $h_2$ (ET) | $h_1 - h_2 - 0{,}5$ | sortie |
|---|---|---|---|---|
| (0, 0) | 0 | 0 | −0,5 | 0 |
| (0, 1) | 1 | 0 | 0,5 | 1 |
| (1, 0) | 1 | 0 | 0,5 | 1 |
| (1, 1) | 1 | 1 | −0,5 | 0 |

Dans l'espace caché $(h_1, h_2)$, (0, 1) et (1, 0) tombent au même point (1, 0), et les quatre exemples deviennent séparables par une droite. Un perceptron ordinaire, entraîné sur $(h_1, h_2)$, les sépare en 11 corrections avec $\tilde{\mathbf w} = (2\,;\ -3\,;\ -1)$. Le même circuit s'écrit aussi en neurones formels de 1943, avec $h_2$ comme entrée inhibitrice de la sortie.

La leçon est double. Une couche cachée change la représentation des données jusqu'à rendre le problème linéaire. Mais ici, cette couche a été conçue par quelqu'un qui connaissait la réponse. La règle du perceptron ne dit rien sur la façon de régler $h_1$ et $h_2$ : quand la sortie se trompe, quel neurone caché faut-il blâmer, et de combien ? C'est le problème que résoudra la rétropropagation (chapitre 07). Le même argument du carré XOR sert à mesurer la capacité des classifieurs linéaires (dimension VC, chapitre 10).

### 6. Le Mark I et la démonstration de 1958

En juillet 1958, l'Office of Naval Research présente le perceptron à la presse. Ce n'est pas encore une machine dédiée, mais une simulation sur un IBM 704, un ordinateur de cinq tonnes : nourri de cartes perforées, il apprend en 50 essais à distinguer les cartes marquées à gauche de celles marquées à droite. Le *New York Times* titre « New Navy Device Learns by Doing », sur l'« embryon » d'un ordinateur qui saurait un jour lire et devenir plus savant. Rosenblatt est alors psychologue de recherche et ingénieur de projet au Cornell Aeronautical Laboratory (Cornell Chronicle, 2019).

La machine vient ensuite. Le Mark I Perceptron, financé par la marine, compte 400 cellules photoélectriques disposées en grille 20 × 20 (les unités S), 512 unités d'association et 8 unités de réponse ; les poids réglables sont des potentiomètres entraînés par des moteurs. Elle reconnaît des lettres présentées devant son « œil ». Une photographie diffusée par l'US Navy le 24 juin 1960 la montre face à une lettre C (notice sur Wikimedia Commons). Le communiqué qui l'accompagne décrit l'apprentissage : quand la machine se trompe, l'opérateur la force à donner la bonne réponse par une commande électrique. Elle est aujourd'hui dans les collections de la Smithsonian Institution.

Dates du Mark I : ce qui est attesté :
Le repère le plus sûr est le communiqué de l'US Navy qui accompagne cette photographie : il annonce la machine et porte la mention « Photograph released June 24, 1960 » (notice de la photo sur Wikimedia Commons). D'autres textes datent le Mark I de 1957 ou de 1958, en le confondant avec la simulation sur IBM 704 ; la date d'assemblage, souvent donnée comme 1959, ne vient que de sources secondaires et reste sous réserve. L'architecture (400, 512 et 8 unités) est celle qu'indique George Nagy, ancien doctorant de Rosenblatt, dans une présentation de 2011 consacrée à son directeur de thèse.

### 7. *Perceptrons* (1969) et le premier hiver

En 1969, Marvin Minsky et Seymour Papert, du groupe d'intelligence artificielle du MIT, alors rattaché au Project MAC (il devient en janvier 1971 un laboratoire à part, qu'ils codirigent), publient *Perceptrons: An Introduction to Computational Geometry*. Le livre ne se résume pas à XOR, que tout spécialiste savait hors de portée d'un neurone seul. Il étudie des perceptrons dont chaque unité d'association ne regarde qu'une partie de la rétine, et montre que certaines propriétés globales d'une image en demandent beaucoup trop : la parité (le nombre de pixels allumés est-il pair ?) exige des unités qui regardent toute la rétine ; la connexité (la figure est-elle d'un seul tenant ?) exige des unités dont la taille croît avec celle de l'image. XOR est la parité de deux pixels. Sur les réseaux à plusieurs couches, les auteurs n'ont pas de théorème, seulement une conjecture : ils jugent « intuitivement » que l'extension est « stérile » (sterile, p. 231-232) et en font un problème de recherche.

Dans les années qui suivent, la recherche sur les réseaux de neurones reflue et les financements vont à l'IA symbolique (chapitres 04 et 05). Rosenblatt meurt en 1971, noyé lors d'une sortie en voilier sur la baie de Chesapeake, le jour de ses 43 ans. Le livre est souvent présenté comme la cause de ce reflux. Le sociologue Mikel Olazaran (1996) nuance : l'interprétation « officielle » des preuves de Minsky et Papert, selon laquelle tout progrès des réseaux de neurones était impossible, tient selon lui à l'installation de l'IA symbolique, de ses circuits de financement et de ses autorités, plus qu'aux théorèmes eux-mêmes ; et l'histoire du débat a été réécrite lors du retour des réseaux de neurones à la fin des années 1980.

Le reflux du connexionnisme n'est pas, à lui seul, le « premier hiver de l’IA ». Deux événements extérieurs aux perceptrons pèsent sur toute l'IA :

- Aux États-Unis, l'amendement Mansfield, voté fin 1969, limite les crédits de recherche du ministère de la Défense aux projets ayant une relation « directe et apparente » avec une fonction militaire précise (GAO, 1970). L'agence ARPA, principal financeur de l'IA, exige ensuite des justifications plus appliquées.
- Au Royaume-Uni, le rapport de James Lighthill (texte de juillet 1972, publié en 1973 par le Science Research Council) juge sévèrement les ambitions de l'IA généraliste. D'après l'historien Thomas Haigh, il est commandé pour justifier l'arrêt du financement du laboratoire de Donald Michie à Édimbourg, le principal centre britannique.

Haigh conteste même qu'il y ait eu un hiver général dans les années 1970 : le groupe d'intérêt SIGART de l'ACM passe de 1 241 membres en 1973 à environ 3 500 à la mi-1978. Selon lui, le ralentissement a surtout frappé quelques grands laboratoires financés par la DARPA, très visibles, et le vrai « hiver de l'IA » est celui de la fin des années 1980, après la bulle des systèmes experts. Le plus prudent est de retenir ceci : *Perceptrons* explique en partie le reflux des réseaux de neurones, pas à lui seul un hiver de toute l'IA.

## Le perceptron à l'œuvre

La figure qui résume l'exercice : en haut, les 95 frontières successives balaient le plan, puis se figent dès qu'une époque passe sans erreur ; la frontière finale est presque confondue avec la vraie, alors que la droite de Hebb passe loin des données. En bas, les erreurs par époque.

À vous de jouer. La démo charge les 200 points de l'exercice et rejoue la même règle, exemple par exemple : avancez d'un exemple, d'une correction ou d'une époque, ou lancez l'animation. Le calcul $y\,\tilde{\mathbf w}\cdot\tilde{\mathbf x}$ s'affiche à chaque pas ; la frontière d'avant la dernière correction reste en tirets, et la flèche montre $(w_1, w_2)$, perpendiculaire à la frontière. Touchez ou cliquez le graphique pour ajouter un point de la classe choisie par les boutons « classe +1 » et « classe −1 » (Maj + clic : l'autre classe ; toucher un point le retire), et voyez combien un seul point mal placé coûte de corrections.

Les boutons du haut changent de données. OU rejoue l'exemple chiffré de la section 3 (9 corrections en 6 époques, poids (2 ; 2 ; −1)) ; ET (entrées 0/1) demande 18 corrections en 9 époques ; XOR tourne en rond, et la démo signale le cycle dès la fin de la deuxième époque, quand les poids reviennent à (0 ; 0 ; 0).

## Ce qui coinçait

- Une seule droite. Un perceptron ne sépare que des classes linéairement séparables : XOR, la parité, la connexité lui échappent. Des couches cachées lèvent cette limite (chapitre 07), à condition de savoir les entraîner.
- Pas de règle pour les couches cachées. La règle du perceptron corrige une unité dont on connaît la sortie attendue. Pour un neurone caché, on ne la connaît pas : il faut répartir le blâme de l'erreur finale. Ce sera la rétropropagation du gradient, qui suppose de remplacer le seuil, dont la dérivée est nulle presque partout, par une fonction dérivable (sigmoïde ; plus tard ReLU, dérivable presque partout).
- Aucun arrêt garanti sans séparabilité. Sur des données bruitées, le perceptron oscille sans fin ; il faut garder les meilleurs poids rencontrés ou faire la moyenne des poids visités, des variantes plus tardives.
- Une frontière quelconque. Le critère du perceptron vaut zéro pour toute droite qui sépare les données : il ne préfère pas la plus sûre. La marge maximale des SVM (chapitre 08) répond à ce défaut, avec la perte charnière.
- Pas de probabilité. Le perceptron répond +1 ou −1, sans degré de confiance ; la régression logistique (chapitre 01) donne une probabilité avec le même modèle linéaire.
- Des promesses trop grandes. La présentation de 1958 annonçait une machine qui lirait et apprendrait seule ; l'écart avec ce qu'un perceptron à une couche sait faire a nourri la réaction de 1969.

- Neurone formel (McCulloch et Pitts, 1943) : Un seuil sur des entrées binaires : une porte logique ; câblé à la main, il n’apprend rien.
- Règle de Hebb (1949) : Renforcer ce qui s’active ensemble ; ne regarde jamais l’erreur.
- Perceptron (Rosenblatt, 1957-1962) : Corriger les poids seulement sur erreur : w ← w + η·y·x ; une descente de gradient sur max(0, −y·w·x).
- Convergence (Rosenblatt, Block, Novikoff, 1962) : Données séparables avec une marge γ : au plus (R/γ)² corrections.
- Perceptrons (Minsky et Papert, 1969) : Une couche ne calcule ni XOR, ni la parité, ni la connexité ; plusieurs couches le peuvent, mais comment les entraîner ?
- Rétropropagation (1986) : Un seuil lisse et la règle de la chaîne répartissent le blâme jusqu’aux couches cachées.
- SVM (Boser, Guyon et Vapnik, 1992 ; Cortes et Vapnik, 1995) : Parmi les droites qui séparent, choisir celle de marge maximale : la perte charnière.

Et ensuite ? Deux couches résolvent XOR, l'exercice le montre, mais leurs poids ont été fixés à la main. Pour qu'un réseau à couches cachées apprenne, il faut savoir de combien modifier chaque poids, y compris ceux des neurones dont on ne connaît pas la sortie attendue. Il faut pour cela un neurone dérivable et un calcul efficace du gradient de la perte par rapport à tous les poids : c'est l'objet du chapitre 07 : Rétropropagation, MLP et fonctions d'activation.

## Exercice

Ce qu'il faut observer :

- Hebb renforce, le perceptron corrige : la règle de Hebb apprend ET (poids (2 ; 2 ; −2)) mais plafonne à 54,0 % sur deux nuages séparables ; le perceptron les sépare à 100 % en 95 corrections et 7 époques.
- La règle du perceptron est une descente de gradient sur $\max(0, -y\,\tilde{\mathbf w}\cdot\tilde{\mathbf x})$ : les deux boucles, et scikit-learn, donnent exactement les mêmes poids, (3,26 ; 7,39 ; −21).
- La borne de convergence est large mais juste : 95 corrections pour une borne de 22 195 ; sur 50 jeux, plus la marge est petite, plus il faut de corrections.
- XOR tourne en rond, les poids revenant à (0 ; 0 ; 0) à chaque époque, puis deux couches câblées à la main le résolvent, et un perceptron sépare l'espace caché en 11 corrections.

Questions de réflexion : pourquoi le pas $\eta$ ne change-t-il aucune décision quand les poids partent de zéro ? Quelle modification de la condition « $\le 0$ » empêcherait le perceptron de s'arrêter sur une droite qui frôle des points, et quelle perte optimiserait-il alors ?

## Auto-évaluation

Question : Poids augmentés w̃ = (1 ; −1 ; 0), exemple x = (2, 1) de classe y = −1, η = 1. Le perceptron le classe-t-il bien ? Que deviennent les poids ?

Réponse : Score : $1 \times 2 - 1 \times 1 + 0 = 1$, donc $y \times \text{score} = -1 \le 0$ : erreur (le perceptron prédit +1). Correction : $\tilde w \leftarrow \tilde w + y\,\tilde x = (1\,;\ -1\,;\ 0) - (2\,;\ 1\,;\ 1) = (-1\,;\ -2\,;\ -1)$. Nouveau score : $-2 - 2 - 1 = -5$, bien classé. La quantité $y \times \text{score}$ est passée de −1 à +5, soit une hausse de $\lVert\tilde x\rVert^2 = 4 + 1 + 1 = 6$, comme le prévoit la formule.

Question : La règle de Hebb et celle du perceptron ajoutent toutes deux η·y·x̃ aux poids. Qu'est-ce qui les sépare, et pourquoi Hebb échoue-t-elle sur des données pourtant séparables ?

Réponse : Hebb ajoute $\eta\,y\,\tilde x$ pour chaque exemple, bien ou mal classé ; le perceptron seulement sur une erreur. Hebb ne voit donc jamais si la frontière est bonne : le résultat est fixé d'avance (en gros, la différence des moyennes des classes, et un biais égal à $\sum y$, nul si les classes sont équilibrées). Dans l'exercice, la droite passe par l'origine, loin des nuages : 54,0 %. Le perceptron, piloté par l'erreur, continue tant qu'il reste un point mal classé.

Question : Montrez qu'aucun perceptron à une couche ne calcule XOR, puis donnez un réseau à deux couches qui le calcule.

Réponse : Si $w_1 x_1 + w_2 x_2 + b$ était positif exactement sur (0, 1) et (1, 0) : (0, 0) donne $b \le 0$, (1, 0) et (0, 1) donnent $w_1 + b \gt 0$ et $w_2 + b \gt 0$, d'où $w_1 + w_2 + b \gt -b \ge 0$ ; mais (1, 1) exige $w_1 + w_2 + b \lt 0$. Contradiction. Deux couches : $h_1 = $ OU (seuil 0,5 sur $x_1 + x_2$), $h_2 = $ ET (seuil 1,5), sortie $= h_1$ ET NON $h_2$ (seuil 0,5 sur $h_1 - h_2$). Dans l'espace $(h_1, h_2)$, (0, 1) et (1, 0) se confondent et le problème devient séparable.

Question : Que garantit le théorème de convergence du perceptron, et pourquoi la borne (R/γ)² est-elle si loin du nombre de corrections observé (22 195 contre 95) ?

Réponse : Si une frontière sépare les données avec une marge $\gamma$ et que les entrées sont de norme au plus $R$, le perceptron fait au plus $(R/\gamma)^2$ corrections, quels que soient le nombre d'exemples, leur ordre et la dimension. La borne vaut pour le pire cas (pire ordre, pires points de mêmes $R$ et $\gamma$) ; sur des données ordinaires, le perceptron fait beaucoup mieux. Elle ne dit rien si les données ne sont pas séparables, et elle ne promet pas une bonne frontière : seulement une frontière qui sépare.

Question : Le livre « Perceptrons » (1969) a-t-il provoqué le premier hiver de l'IA ?

Réponse : Pas à lui seul. Il a démontré les limites des perceptrons à une couche à unités locales (parité, connexité) et exprimé un jugement « intuitif » pessimiste sur les réseaux à plusieurs couches ; il a accompagné le reflux des réseaux de neurones dans les années 1970. Mais les financements de toute l'IA ont surtout été touchés par d'autres causes : l'amendement Mansfield (1969) aux États-Unis, le rapport Lighthill (1973) au Royaume-Uni. Et l'historien Thomas Haigh (2023) conteste qu'il y ait eu un hiver général : la communauté de l'IA a continué de croître pendant les années 1970.

## Pour aller plus loin

- Articles originaux : McCulloch et Pitts (1943), Rosenblatt (1958), Novikoff (1962), Minsky et Papert (1969) : liens dans les fiches ci-dessus. Le rapport de Rosenblatt de janvier 1957 (Cornell Aeronautical Laboratory, n° 85-460-1) est consultable en scan.
- K. Weinberger, « Lecture 3: The Perceptron », cours CS4780 de Cornell (2018) : l'algorithme, sa géométrie et la preuve de convergence avec la marge, en une page.
- M. Nielsen, *Neural Networks and Deep Learning*, chapitre 1, « Using neural nets to recognize handwritten digits » (livre en ligne) : perceptrons, poids et biais, un perceptron qui calcule NAND, puis le passage aux neurones sigmoïdes.
- 3Blue1Brown, « But what is a neural network? » : poids, biais et couches en images ; il prépare le chapitre 07.
- Stanford CS231n, « Linear classification » : un classifieur linéaire vu comme un ensemble d'hyperplans, et ses limites.
- M. Lefkowitz, « Professor's perceptron paved the way for AI – 60 years too soon » (Cornell Chronicle, 2019) : Rosenblatt, la démonstration de 1958 et le Mark I.
- J. Lighthill, « Artificial Intelligence: A General Survey » (1973), avec les réponses de Sutherland, Needham, Longuet-Higgins et Michie sur le même site.
- Pour l'histoire du « premier hiver » : T. Haigh, « There Was No 'First AI Winter' » (*Communications of the ACM*, 2023), et M. Olazaran (1996) sur la controverse des perceptrons : liens dans les fiches ci-dessus.
