# 08. Apprentissage automatique « classique »

Repère historique (1951 à 2016 (cœur de la période : 1984-2001)) : Apprendre à partir d'exemples des modèles efficaces, peu coûteux à entraîner et dotés de garanties, sur des données de taille moyenne (des centaines à des centaines de milliers d'exemples, des dizaines de variables).

En 1986, la rétropropagation a rendu les réseaux de neurones entraînables (chapitre 07). Mais dans les années 1990, ils restent capricieux : peu profonds, lents à régler, sans garantie de trouver une bonne solution, et gourmands en données qu'on n'a pas. Une autre école, issue de la statistique et de la théorie de l'apprentissage, prend le dessus. Ses outils ont souvent des racines anciennes (l'analyse en composantes principales date de 1901, les plus proches voisins de 1951, les k-moyennes de 1957), mais c'est entre 1984 et 2001 qu'ils deviennent la boîte à outils standard : arbres de décision (CART, 1984 ; ID3, 1986), machines à vecteurs de support (Bell Labs, 1992-1995), boosting (1990-1997), forêts aléatoires (2001), gradient boosting (2001). Leur atout commun : une optimisation bien posée (souvent convexe, ou une recherche exhaustive), peu de réglages, et des résultats solides sur des tableaux de quelques milliers de lignes. Ils domineront l'apprentissage automatique jusqu'au début des années 2010, et dominent toujours les données tabulaires.

## L'idée en une image

Vous avez un tableau : une ligne par client, une colonne par variable (âge, ancienneté, montant du panier…), et une colonne « a résilié son abonnement : oui ou non ». Vous voulez prédire cette dernière colonne pour de nouveaux clients. C'est de l'apprentissage supervisé (chapitre 03). Ce chapitre présente les grandes familles de modèles qui répondent à cette question sans réseau de neurones. Chacune part d'une idée simple, qu'on peut énoncer en une phrase :

| Famille | L'idée | Ce qu'elle fabrique |
|---|---|---|
| Plus proches voisins | « Dis-moi qui te ressemble, je te dirai qui tu es. » | rien : elle garde les exemples et les consulte |
| Bayes naïf | « Quelle classe aurait le plus probablement produit cet exemple ? » | une loi de probabilité par classe |
| Arbre de décision | « Pose les bonnes questions, dans le bon ordre. » | une suite de questions « x ≤ seuil ? » |
| Forêt aléatoire | « Demande à 300 arbres différents et fais voter. » | beaucoup d'arbres décorrélés |
| Boosting | « Chaque nouvel élève corrige les erreurs des précédents. » | une somme de petits modèles |
| SVM | « Trace la frontière la plus loin possible des deux camps. » | une frontière à marge maximale, éventuellement courbe |

Quand il n'y a pas de colonne à prédire (apprentissage non supervisé), deux autres questions restent : quels groupes forment les clients (partitionnement : k-means, partitionnement hiérarchique, DBSCAN) et comment résumer 64 variables en 2 pour les voir (réduction de dimension : ACP, t-SNE, UMAP).

Face au réseau de neurones de 1986, le changement est surtout pratique :

| | Avant : MLP et rétropropagation (1986) | Après : apprentissage « classique » (1984-2001) |
|---|---|---|
| Optimisation | descente de gradient sur une perte non convexe : minima locaux, résultat qui dépend de l'initialisation | souvent convexe (SVM, régression logistique) ou recherche exhaustive (coupes d'un arbre) : même résultat à chaque fois |
| Réglages | taux d'apprentissage, nombre de neurones, d'époques, initialisation | un ou deux hyperparamètres (k, profondeur, C et γ), choisis par validation croisée |
| Données nécessaires | beaucoup | des centaines à des milliers d'exemples suffisent |
| Garanties | aucune | bornes d'erreur (plus proche voisin, marge des SVM ; chapitre 10) |
| Représentation des données | apprise par les couches cachées | fournie à la main : les variables du tableau, éventuellement transformées |
| Ce qui coince | la profondeur, le réglage | les données brutes (images, son, texte), où il faut inventer les bonnes variables |

La dernière ligne est la clé de tout le parcours : ces méthodes excellent quand les variables ont déjà un sens (un âge, un montant), et peinent quand il faut d'abord construire ces variables à partir de pixels ou de mots. C'est ce que reprendra le deep learning (chapitre 12).

## Le mécanisme

Chaque section ci-dessous présente une famille : l'idée, la formule, un calcul à la main, le code minimal, puis ce que l'exercice mesure. Les chiffres de l'exercice portent presque tous sur le même jeu jouet, deux lunes : 200 points d'entraînement et 1 000 de test en deux dimensions, deux classes entrelacées en forme de croissants, avec du bruit (`make_moons`, bruit 0,3). Aucune droite ne les sépare ; le meilleur modèle possible se trompe encore sur quelques points à cause du bruit.

### 1. Génératif ou discriminatif : deux façons de classer

Il y a deux manières de construire un classifieur probabiliste.

- Un classifieur discriminatif modélise directement $p(y \mid x)$ : « sachant ces variables, quelle est la probabilité de chaque classe ? ». La régression logistique en est l'exemple type (chapitre 01).
- Un classifieur génératif modélise comment chaque classe produit ses exemples, $p(x \mid y)$, et la fréquence de chaque classe, $p(y)$. Pour classer, il retourne la question avec la formule de Bayes :

(Formule : Classifieur génératif : la formule de Bayes.)

Le génératif doit modéliser toute la distribution des $x$, ce qui est plus difficile mais permet aussi de générer des exemples, de traiter des variables manquantes ou de détecter un exemple improbable. En 2001, Andrew Ng et Michael Jordan comparent les deux sur la paire Bayes naïf contre régression logistique : le discriminatif atteint une erreur plus basse quand les données sont nombreuses, mais le génératif s'approche bien plus vite de sa propre erreur limite (plus haute). Avec peu d'exemples, le génératif peut donc gagner.

Le Bayes naïf. Modéliser $p(x \mid y)$ quand $x$ a 30 variables est difficile : il faudrait estimer comment elles varient ensemble. Le Bayes naïf fait une hypothèse brutale : une fois la classe connue, les variables sont indépendantes. La vraisemblance devient un produit de lois à une variable, chacune facile à estimer (une moyenne et une variance pour une variable continue, une fréquence pour un mot) :

(Formule : Bayes naïf.)

Exemple chiffré, un filtre à spams. Sur l'historique : 40 % des messages sont des spams ; « gratuit » apparaît dans 30 % des spams et 2 % des messages normaux ; « réunion » dans 1 % des spams et 10 % des messages normaux. Un message contient « gratuit » et pas « réunion ». En simplifiant (on ne regarde que ces deux mots) :

```text
score(spam)   = 0,4 × 0,30 × (1 − 0,01) = 0,1188
score(normal) = 0,6 × 0,02 × (1 − 0,10) = 0,0108
p(spam | message) = 0,1188 / (0,1188 + 0,0108) ≈ 0,92
```

L'hypothèse d'indépendance est presque toujours fausse (« gratuit » et « offre » vont ensemble), donc les probabilités obtenues sont mal calibrées, souvent trop proches de 0 ou de 1. Mais la classe prédite est souvent bonne, et le modèle s'entraîne en un seul passage sur les données. Dans l'exercice, le Bayes naïf gaussien obtient 93,9 % d'exactitude sur le jeu médical *breast cancer* (30 mesures de cellules, 569 patientes), contre 97,4 % pour la régression logistique : ses 30 variables sont fortement corrélées entre elles, ce qui viole l'hypothèse.

Les modèles génératifs plus riches (mélanges de gaussiennes, modèles à variables cachées) sont le sujet du chapitre 09, puis des chapitres 18 et 25.

### 2. Les k plus proches voisins

La méthode la plus simple qui soit : pour classer un nouveau point, chercher les $k$ exemples d'entraînement les plus proches, et leur faire voter la classe. Les k plus proches voisins (k-NN) n'apprennent aucun paramètre : « entraîner » consiste à ranger les exemples. Tout le travail se fait au moment de prédire.

(Formule : Règle des k plus proches voisins (classification).)

Le code de l'exercice tient en quelques lignes de numpy :

```python
import numpy as np

def voisins_tries(X_train, X):
    """Pour chaque ligne de X, indices des exemples d'entraînement du plus proche au plus lointain."""
    d2 = ((X[:, None, :] - X_train[None, :, :])  2).sum(axis=2)   # distances au carré, (m, n)
    return np.argsort(d2, axis=1, kind="stable")                    # à égalité : le plus petit indice d'abord

def knn_predire(ordre, y_train, k):
    """Vote majoritaire des k premiers voisins (deux classes 0/1)."""
    votes = y_train[ordre[:, :k]].mean(axis=1)                       # part de voisins de classe 1
    return (votes > 0.5).astype(int)
```

Comparé à `KNeighborsClassifier` de scikit-learn sur les 1 000 points de test, pour k = 1, 5, 11, 25 et 99 : zéro prédiction différente.

Le seul réglage, k, est un bouton biais-variance. Avec $k = 1$, chaque point d'entraînement est son propre plus proche voisin : l'erreur d'entraînement vaut 0 %, mais la frontière contourne chaque point bruité et l'erreur de test monte à 13,9 %. Avec $k = 199$ (presque tout le jeu), chaque vote consulte presque tout le monde : la frontière devient une droite raide, 29,5 % d'erreur. C'est le compromis biais-variance du chapitre 00, avec $1/k$ dans le rôle de la complexité : petit $k$, forte variance (surapprentissage) ; grand $k$, fort biais (sous-apprentissage).

| k | Erreur d'entraînement | Erreur de test |
|---|---|---|
| 1 | 0 % | 13,9 % |
| 5 | 7,0 % | 10,4 % |
| 11 | 7,5 % | 11,3 % |
| 25 | 8,0 % | 9,1 % |
| 35 | 9,0 % | 8,9 % (meilleur sur le test) |
| 51 | 12,0 % | 9,6 % |
| 99 | 18,0 % | 16,8 % |
| 199 | 29,0 % | 29,5 % |

Comment choisir $k$ sans regarder le test ? Par validation croisée à 5 plis sur les 200 points d'entraînement (méthode expliquée au chapitre 00). Elle choisit $k = 11$ : 7,0 % d'erreur en validation, 11,3 % au test, alors que $k = 35$ aurait fait 8,9 %. Ce n'est pas un échec : avec 200 points, l'estimation de validation est bruitée, et choisir $k$ en regardant le jeu de test rendrait le chiffre final trop optimiste.

À vous : faites varier $k$, affichez les points de test, ajoutez des points d'entraînement (clic ou toucher ; au clavier, flèches puis Entrée) et regardez la frontière se déformer. Cherchez la valeur de $k$ à partir de laquelle un point ajouté isolé ne crée plus d'îlot.

Ce que garantit la théorie. En 1967, Thomas Cover et Peter Hart montrent que, quand le nombre d'exemples tend vers l'infini, l'erreur $R$ du plus proche voisin ($k = 1$) est encadrée par l'erreur minimale possible $R^*$ (l'erreur de Bayes, celle du meilleur classifieur imaginable, qui connaîtrait la vraie distribution) : $R^* \le R \le 2R^*$ (plus précisément $R \le R^*(2 - M R^*/(M-1))$ pour $M$ classes). Autrement dit, la moitié de l'information utile pour classer est contenue dans le seul plus proche voisin. La règle elle-même avait été proposée en 1951 par Evelyn Fix et Joseph Hodges, dans un rapport pour l'US Air Force.

Deux défauts pratiques.

1. La prédiction coûte cher. Chaque prédiction calcule la distance aux $n$ exemples : $n \times d$ opérations. Pour des millions d'exemples, on utilise des structures d'index et des recherches approchées de voisins ; c'est exactement ce que font les bases de données vectorielles du RAG (chapitre 32).
2. Les unités comptent. La distance additionne des écarts de variables différentes : un montant en euros (écarts de l'ordre de 1 000) écrase un âge (écarts de l'ordre de 10). Il faut donc appliquer une standardisation : pour chaque variable, retrancher sa moyenne et diviser par son écart-type (calculés sur l'entraînement seulement). Dans l'exercice, le 11-NN fait 11,3 % sur les variables brutes et 10,0 % sur les variables standardisées. Un arbre de décision, lui, n'y est pas sensible : multiplier une variable par 1 000 multiplie le seuil de sa coupe par 1 000, sans rien changer d'autre.

### 3. La malédiction de la dimension

Les plus proches voisins reposent sur une hypothèse : des points proches ont la même classe. En deux dimensions, elle marche. En cent, elle s'effondre, pour deux raisons que résume l'expression malédiction de la dimension (née en programmation dynamique, voir le chapitre 11).

L'espace se vide. Supposons les données uniformes dans un cube $[0, 1]^d$. Pour qu'un voisinage cubique contienne 1 % des points, son côté doit valoir $0{,}01^{1/d}$ :

| Dimension $d$ | Côté du cube qui contient 1 % des données | Part du volume à moins de 0,05 du bord |
|---|---|---|
| 1 | 0,01 | 10 % |
| 10 | $0{,}01^{1/10} = 0{,}63$ | $1 - 0{,}9^{10} = 65$ % |
| 100 | $0{,}01^{1/100} = 0{,}955$ | $1 - 0{,}9^{100} \approx 100$ % |

En dimension 100, le « voisinage local » qui capture 1 % des points s'étend sur 95 % de chaque axe : il n'a plus rien de local. Et presque tout le volume est près du bord.

Les distances se ressemblent toutes. C'est la concentration des distances. L'exercice tire 1 000 points au hasard dans le cube et mesure, depuis un point, le rapport entre la distance au plus lointain et au plus proche :

| $d$ | 1 | 2 | 10 | 100 | 1 000 |
|---|---|---|---|---|---|
| plus lointain / plus proche | 611 | 101 | 3,22 | 1,35 | 1,11 |

En dimension 1 000, le point le plus lointain n'est que 11 % plus loin que le plus proche : « le plus proche voisin » ne veut presque plus rien dire. En 1999, Kevin Beyer et ses coauteurs prouvent que ce resserrement se produit sous des conditions larges, et l'observent dès 10 à 15 dimensions sur des données réelles.

Conséquence pratique : ajouter des variables inutiles noie les variables utiles. L'exercice ajoute aux deux lunes 2, 10 puis 50 variables de pur bruit :

| Variables de bruit ajoutées | 0 | 2 | 10 | 50 |
|---|---|---|---|---|
| Erreur de test du 11-NN | 11,3 % | 15,9 % | 26,5 % | 40,4 % |
| Erreur de test de la forêt aléatoire | 11,6 % | 11,6 % | 12,4 % | 12,1 % |

Le k-NN ne sait pas quelles variables comptent : toutes pèsent autant dans la distance. Les arbres, qui choisissent leurs variables coupe par coupe, résistent bien mieux : la forêt aléatoire de la section 5 ne perd que 0,5 point avec 50 variables de bruit ; les réseaux de neurones apprennent une représentation où la distance a un sens (les plongements du chapitre 15). Les vraies données échappent en partie à la malédiction parce qu'elles ne remplissent pas l'espace : elles vivent près de surfaces de bien plus petite dimension (l'hypothèse de variété du chapitre 03).

### 4. Les arbres de décision

Un arbre de décision pose une suite de questions binaires sur une variable à la fois : « $x_2 \le 0{,}38$ ? », puis, selon la réponse, une autre question, jusqu'à une feuille qui donne la prédiction (la classe majoritaire des exemples d'entraînement arrivés là). Il découpe le plan en rectangles. C'est le modèle le plus lisible de ce chapitre : on peut l'imprimer et le suivre à la main.

Comment choisir une question ? On veut que chaque question sépare les classes : que les deux groupes obtenus soient chacun aussi « purs » que possible. On mesure le mélange d'un nœud par son impureté de Gini :

(Formule : Impureté de Gini et choix d'une coupe.)

Exemple chiffré. Un nœud contient 10 exemples, 6 bleus et 4 orange : $G = 1 - 0{,}6^2 - 0{,}4^2 = 0{,}48$. Une coupe candidate envoie 5 bleus et 1 orange à gauche, 1 bleu et 3 orange à droite :

```text
gauche : G = 1 − (5/6)² − (1/6)² = 10/36 = 0,278
droite : G = 1 − (1/4)² − (3/4)² =  6/16 = 0,375
moyenne pondérée : 0,6 × 0,278 + 0,4 × 0,375 = 0,317     gain : 0,48 − 0,317 = 0,163
```

On répète ce calcul pour toutes les variables et tous les seuils possibles (entre deux valeurs consécutives triées) et on garde la meilleure coupe. Puis on recommence dans chaque enfant. C'est une méthode gloutonne : chaque coupe est la meilleure sur le moment, sans regarder plus loin. Trouver l'arbre optimal est un problème combinatoire hors de portée ; la gloutonnerie marche bien en pratique.

```python
def meilleure_coupe(X, y):
    """Souche de Gini (deux classes 0/1) : la coupe x_j <= s qui minimise l'impureté pondérée."""
    def gini(y):
        p = y.mean() if len(y) else 0.0
        return 1 - p2 - (1 - p)2
    meilleure = (None, None, gini(y))                    # (variable, seuil, impureté)
    for j in range(X.shape[1]):
        valeurs = np.unique(X[:, j])
        for s in (valeurs[:-1] + valeurs[1:]) / 2:       # seuils entre deux valeurs consécutives
            g = X[:, j] <= s
            imp = g.mean() * gini(y[g]) + (1 - g.mean()) * gini(y[~g])
            if imp < meilleure[2]:
                meilleure = (j, s, imp)
    return meilleure
```

Sur les deux lunes, la souche numpy de l'exercice choisit $x_2 \le 0{,}3810$, qui fait passer l'impureté de 0,50 à 0,2939 : exactement la coupe de `DecisionTreeClassifier(max_depth=1)`.

Ce critère est celui de CART (Breiman, Friedman, Olshen et Stone, 1984). En 1986, J. R. Quinlan publie ID3, qui choisit la coupe par le gain d’information : la baisse de l'entropie ($-\sum_k p_k \log_2 p_k$, chapitre 00) des étiquettes. Pour le nœud de l'exemple, l'entropie vaut 0,971 bit ; les deux critères ont presque la même forme et choisissent presque toujours la même coupe (courbes comparées dans la fiche de l'impureté de Gini ci-dessous).

Un arbre laissé libre apprend par cœur. Sans limite, l'arbre coupe jusqu'à ce que chaque feuille soit pure : 0 % d'erreur d'entraînement. Sur les lunes, l'arbre complet a 12 niveaux et 24 feuilles, et fait 14,3 % au test ; limité à 2 niveaux, 10,7 % ; à 1 niveau, 18,7 %. On le bride par la profondeur maximale, le nombre minimal d'exemples par feuille, ou par élagage (faire pousser l'arbre puis couper les branches qui n'améliorent pas la validation, la méthode de CART). Le défaut de fond reste : un arbre est instable. Changez quelques exemples, la première coupe change, et tout l'arbre en dessous avec. Forte variance : c'est ce que corrigent les méthodes d'ensemble.

### 5. Bagging et forêts aléatoires : moyenner pour réduire la variance

Une méthode d’ensemble combine plusieurs modèles. L'idée du bagging (Breiman, 1996, pour *bootstrap aggregating*) : si un modèle varie beaucoup d'un jeu d'entraînement à l'autre, entraînons-en beaucoup sur des jeux différents et faisons la moyenne. On n'a qu'un jeu, mais on peut en fabriquer d'autres par bootstrap : tirer $n$ exemples avec remise parmi les $n$ disponibles. Chaque tirage contient en moyenne $1 - (1 - 1/n)^n \approx 1 - 1/e \approx 63{,}2$ % d'exemples distincts ; les autres sont des doublons.

Combien gagne-t-on ? Si $B$ modèles ont chacun une variance $\sigma^2$ et sont corrélés deux à deux avec un coefficient $\rho$, la variance de leur moyenne vaut

(Formule : Variance d'une moyenne de modèles corrélés.)

Le second terme disparaît quand $B$ grandit ; le premier reste. Avec $\sigma^2 = 1$, $\rho = 0{,}3$ et $B = 300$ : $0{,}3 + 0{,}7/300 \approx 0{,}302$. Des arbres entraînés sur des tirages bootstrap se ressemblent beaucoup (ils choisissent tous la même première coupe, la meilleure variable), donc $\rho$ est élevé et le bagging plafonne.

D'où l'idée de la forêt aléatoire (Breiman, 2001) : à chaque nœud de chaque arbre, la meilleure coupe n'est cherchée que parmi un petit nombre de variables tirées au hasard (par défaut dans les implémentations courantes, la racine carrée du nombre de variables en classification). Les arbres sont un peu moins bons individuellement, mais beaucoup moins corrélés, et la moyenne y gagne. Bonus : les exemples absents d'un tirage bootstrap (environ 37 %) servent à estimer l'erreur de chaque arbre sans jeu de validation séparé (l'erreur « hors sac »).

Dans l'exercice, une forêt de 300 arbres complets descend à 11,6 % au test, contre 14,3 % pour l'arbre complet seul. Pris un par un, ses arbres sont pourtant moins bons que lui : 16,0 % au test en moyenne (de 10,9 % à 23,7 %). Chacun est parfait sur son propre tirage bootstrap, mais n'a vu qu'environ 63 % des exemples et ne regarde qu'une partie des variables à chaque nœud. Le biais est celui d'un arbre profond, faible ; c'est la variance qui a fondu dans le vote. Sur ce jeu jouet, la forêt reste moins bonne que l'arbre de profondeur 2 (10,7 %, profondeur choisie en regardant le test) : avec deux variables et une frontière simple, un arbre court a déjà peu de biais et peu de variance. La forêt prend l'avantage quand les variables sont nombreuses ou inutiles (le tableau du bruit de la section 3). Une forêt se règle peu (le nombre d'arbres ne fait jamais surapprendre : il ne fait que stabiliser la moyenne), d'où sa réputation de premier modèle à essayer.

### 6. Boosting : AdaBoost, gradient boosting, XGBoost

Le bagging moyenne des modèles en parallèle, indépendamment. Le boosting les construit en série, chacun corrigeant les erreurs des précédents. Il part d'une question théorique : un apprenant faible, à peine meilleur que le hasard (une souche d'arbre à une seule coupe), peut-il être « dopé » en un apprenant aussi précis qu'on veut ? Michael Kearns et Leslie Valiant posent la question en 1988-1989 ; Robert Schapire répond oui en 1990.

AdaBoost (Yoav Freund et Robert Schapire, AT&T Labs, 1997) rend l'idée pratique. Les classes valent $y \in \{-1, +1\}$. À chaque tour $t$ :

(Formule : AdaBoost (forme usuelle).)

Exemple chiffré : une souche se trompe sur 30 % du poids, $\varepsilon = 0{,}3$. Alors $\alpha = \frac12 \ln(0{,}7/0{,}3) = 0{,}424$. Les poids des exemples mal classés sont multipliés par $e^{0{,}424} = 1{,}53$, ceux des bien classés par $e^{-0{,}424} = 0{,}65$. Avant normalisation, les mal classés pèsent $0{,}3 \times 1{,}53 = 0{,}458$ et les bien classés $0{,}7 \times 0{,}65 = 0{,}458$ : exactement la moitié chacun. Le tour suivant voit donc un problème où le dernier apprenant ferait 50 % d'erreur : il est forcé de chercher autre chose. (L'article de 1997 écrit la mise à jour avec $\beta = \varepsilon/(1-\varepsilon)$ ; la forme ci-dessus est la forme usuelle, équivalente.)

En 2000, Jerome Friedman, Trevor Hastie et Robert Tibshirani montrent ce qu'AdaBoost optimise sans le dire : la perte charnière des SVM et la log-loss de la régression logistique ont une cousine, la perte exponentielle $e^{-y F(x)}$, et AdaBoost la minimise un apprenant à la fois. Elle explique son point faible : un exemple très mal classé (marge $-3$) coûte $e^3 \approx 20$, et AdaBoost s'acharne sur lui, même si c'est une étiquette fausse.

Le gradient boosting (Friedman, 2001 ; idée développée en même temps par Llew Mason, Jonathan Baxter, Peter Bartlett et Marcus Frean, NIPS 1999) généralise cette lecture à n'importe quelle perte dérivable. On construit $F$ par petits pas, comme une descente de gradient, mais dans l'espace des fonctions : à chaque tour, on calcule pour chaque exemple la direction dans laquelle sa prédiction devrait bouger pour faire baisser la perte (le gradient négatif), et on entraîne un petit arbre à prédire ces directions.

(Formule : Gradient boosting.)

Exemple chiffré, en régression avec la MSE : trois cibles $y = (3, 5, 10)$. On part de la moyenne, $F_0 = 6$ partout ; les résidus valent $(-3, -1, 4)$ ; erreur quadratique moyenne $(9 + 1 + 16)/3 = 8{,}67$. Une souche sépare $\{3, 5\}$ de $\{10\}$ et prédit $-2$ et $4$ (moyennes des résidus). Avec $\eta = 0{,}1$, $F_1 = (5{,}8 ;\ 5{,}8 ;\ 6{,}4)$, nouveaux résidus $(-2{,}8 ;\ -0{,}8 ;\ 3{,}6)$, erreur $7{,}15$. Des centaines de tours plus tard, les résidus sont épuisés. Pour la classification avec la log-loss, le résidu est $y - p$, l'écart entre l'étiquette et la probabilité prédite : la même quantité que dans la régression logistique du chapitre 01.

XGBoost (Tianqi Chen et Carlos Guestrin, Université de Washington, 2016) industrialise le gradient boosting : il utilise aussi la dérivée seconde de la perte, pénalise la complexité de chaque arbre (nombre de feuilles $T$ et taille des valeurs des feuilles $w$, terme $\gamma T + \frac12 \lambda \lVert w \rVert^2$), gère les valeurs manquantes et se parallélise. La valeur optimale d'une feuille devient $w^* = -G/(H + \lambda)$, où $G$ et $H$ sont les sommes des dérivées premières et secondes de ses exemples. Sur la souche de l'exemple (MSE, dérivée première $F - y$, seconde 1) : la feuille $\{3, 5\}$ a $G = 3 + 1 = 4$, $H = 2$, et avec $\lambda = 1$, $w^* = -4/3 \approx -1{,}33$ au lieu de $-2$ : la régularisation tire chaque feuille vers zéro. Les auteurs rapportent que parmi les 29 solutions gagnantes de compétitions Kaggle publiées en 2015, 17 utilisaient XGBoost.

L'exercice utilise `HistGradientBoostingClassifier`, l'implémentation de scikit-learn qui, comme XGBoost et LightGBM, regroupe les valeurs de chaque variable en intervalles (histogrammes) pour accélérer la recherche des coupes. Sur *breast cancer* : 96,7 %, comme AdaBoost. Le gradient boosting reste en 2022 la référence sur les données tabulaires de taille moyenne : sur 45 jeux de données ramenés à 10 000 exemples d'entraînement au plus, Léo Grinsztajn, Edouard Oyallon et Gaël Varoquaux trouvent que les modèles à base d'arbres (XGBoost, forêts) battent encore les réseaux de neurones profonds conçus pour ces données.

Bagging ou boosting ? :
Le bagging part de modèles à faible biais et forte variance (arbres profonds) et réduit la variance en moyennant. Le boosting part de modèles à fort biais et faible variance (souches, arbres de 3 à 6 niveaux) et réduit le biais en additionnant. Le boosting surapprend si on enchaîne trop de tours : on règle le nombre de tours par validation, ou par arrêt précoce (chapitre 00).

### 7. Les machines à vecteurs de support

Le perceptron du chapitre 06 s'arrête sur la première droite qui sépare les données. Mais quand plusieurs droites séparent parfaitement l'entraînement, laquelle généralisera le mieux ? Intuitivement, celle qui passe le plus loin possible des deux camps : un point de test un peu décalé restera du bon côté. La machine à vecteurs de support (SVM) cherche la droite (l'hyperplan, en dimension quelconque) de marge maximale.

La marge dure. La frontière est $w \cdot x + b = 0$. On fixe l'échelle de $w$ en demandant $y_i\,(w \cdot x_i + b) \ge 1$ pour tous les exemples : les plus proches sont alors exactement sur les droites $w \cdot x + b = \pm 1$, à une distance $1/\lVert w \rVert$ de la frontière. La largeur de la bande vaut $2/\lVert w \rVert$ ; la maximiser revient à minimiser $\lVert w \rVert$ :

(Formule : SVM à marge dure (données séparables).)

C'est un problème convexe (une fonction quadratique sous des contraintes linéaires) : il a un seul minimum, que des algorithmes standards trouvent sans dépendre d'une initialisation. Sa solution a une propriété remarquable : $w$ est une combinaison des seuls exemples qui touchent la marge, les vecteurs de support. Tous les autres pourraient être déplacés (sans franchir la marge) ou supprimés sans rien changer. Dans l'exercice, sur 80 points séparables, 3 vecteurs de support fixent la frontière ; la marge vaut $2/\lVert w \rVert = 1{,}123$, donc $\lVert w \rVert \approx 1{,}78$, et le plus petit $y_i f(x_i)$ vaut 0,9997, soit 1 aux arrondis de l'optimiseur près.

La marge souple. Sur des données réelles, aucune droite ne sépare tout. Corinna Cortes et Vladimir Vapnik (AT&T Bell Labs, 1995) autorisent des violations $\xi_i \ge 0$, payées au prix $C$ :

(Formule : SVM à marge souple (forme moderne).)

L'article de 1995 écrit le problème avec les variables d'écart $\xi_i$ et une pénalité $C \cdot F\bigl(\sum_i \xi_i\bigr)$, étudiée surtout pour $F(u) = u^2$ ; la forme ci-dessus, où $F(u) = u$, est devenue la forme standard. C'est une régularisation L2 plus la perte charnière, déjà rencontrée au chapitre 01 : seuls les exemples de marge strictement inférieure à 1 ont une perte non nulle ; les vecteurs de support sont les exemples de marge inférieure ou égale à 1 : ceux qui touchent la marge (perte nulle, comme les 3 de l'exercice) et ceux qui la franchissent (perte positive). Le même article baptise la méthode *support-vector networks*.

L'astuce du noyau. Une droite ne sépare pas les deux lunes. Idée classique : transformer les données par une fonction $\varphi$ vers un espace de plus grande dimension où elles deviennent séparables (par exemple ajouter $x_1^2$, $x_1 x_2$, $x_2^2$ aux variables). Le problème : $\varphi(x)$ peut avoir des millions de coordonnées. Or, écrit sous sa forme « duale », le SVM n'utilise les données qu'à travers des produits scalaires $x_i \cdot x_j$. L'astuce du noyau remplace chacun d'eux par une fonction $k(x_i, x_j)$ qui vaut $\varphi(x_i) \cdot \varphi(x_j)$ sans jamais calculer $\varphi$. Exemple en 2D avec $k(x, z) = (x \cdot z)^2$ et $\varphi(x) = (x_1^2,\ \sqrt2\, x_1 x_2,\ x_2^2)$, pour $x = (1, 2)$ et $z = (3, 1)$ :

```text
noyau      : (x·z)² = (1×3 + 2×1)² = 5² = 25
explicite  : φ(x) = (1, 2√2, 4), φ(z) = (9, 3√2, 1)
             φ(x)·φ(z) = 1×9 + (2√2)(3√2) + 4×1 = 9 + 12 + 4 = 25
```

Même nombre, sans passer par l'espace transformé. Le noyau le plus utilisé est le noyau gaussien (RBF), $k(x, z) = e^{-\gamma \lVert x - z \rVert^2}$, qui correspond à un espace de dimension infinie : deux points confondus ont une similarité 1 ; à une distance au carré de 4 avec $\gamma = 0{,}5$, $e^{-2} = 0{,}135$. La fonction de décision devient

(Formule : SVM à noyau : fonction de décision.)

Un SVM à noyau RBF ressemble donc à un k-NN pondéré et appris : chaque vecteur de support « vote » pour sa classe avec un poids qui décroît avec la distance. γ règle la portée de chaque vote (grand γ : votes très locaux, frontière tortueuse), C la tolérance aux erreurs. Sur les lunes, il atteint 10,0 % d'erreur avec 77 vecteurs de support sur 200 points ; sur *breast cancer*, 97,7 % ± 1,6, le meilleur score de l'exercice. L'association de la marge maximale et des noyaux date de 1992 (Bernhard Boser, Isabelle Guyon et Vladimir Vapnik) ; elle ne vaut pas que pour les SVM : toute méthode qui ne dépend que de produits scalaires (ACP, régression ridge) a sa version à noyau.

Pourquoi les SVM ont-ils séduit ? Optimisation convexe (pas de minimum local, contrairement au MLP), solution creuse (seuls les vecteurs de support comptent) et une justification théorique : la largeur de la marge contrôle la capacité du classifieur, indépendamment de la dimension de l'espace des caractéristiques (les bornes sont au chapitre 10). Leur limite : l'entraînement manipule une matrice $n \times n$ de noyaux, et son coût croît en pratique entre $n^2$ et $n^3$. Au-delà de quelques dizaines de milliers d'exemples, c'est trop lent.

### 8. Le partitionnement : k-means, hiérarchique, DBSCAN

Sans étiquettes, on cherche des groupes de points proches. L'algorithme des k-moyennes (k-means) représente chaque groupe par un centre et minimise l'inertie, la somme des carrés des distances de chaque point à son centre :

(Formule : Inertie minimisée par k-means.)

L'algorithme de Lloyd alterne deux étapes, dont chacune fait baisser $J$ (ou le laisse égal) :

Exemple en une dimension, points $1, 2, 3, 10, 11, 12$, $k = 2$, centres de départ 1 et 2 (mal choisis) :

```text
itération 1 : affectation  {1} | {2, 3, 10, 11, 12}   centres → 1 et 7,6
itération 2 : affectation  {1, 2, 3} | {10, 11, 12}   centres → 2 et 11
itération 3 : plus rien ne bouge.                     inertie = (1 + 0 + 1) + (1 + 0 + 1) = 4
```

L'étape d'affectation ne peut pas augmenter $J$ (chaque point choisit le centre le plus proche) ; l'étape de recentrage non plus (le barycentre est le point qui minimise la somme des carrés des distances). Comme il n'existe qu'un nombre fini d'affectations, l'algorithme s'arrête. Stuart Lloyd l'a décrit aux Bell Labs en 1957, pour choisir les niveaux de quantification d'un signal numérisé (modulation par impulsions codées, PCM) ; son texte n'a été publié qu'en 1982. Le nom « k-means » vient de James MacQueen (1967).

```python
def kmeans(X, centres, n_iter=100):
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - centres[None, :, :])  2).sum(axis=2)
        groupe = d2.argmin(axis=1)                                          # affectation
        nouveaux = np.array([X[groupe == j].mean(axis=0) if np.any(groupe == j)    # recentrage
                             else centres[j] for j in range(len(centres))])        # groupe vide : centre inchangé
        if np.allclose(nouveaux, centres):
            break
        centres = nouveaux
    return centres, groupe, d2.min(axis=1).sum()                            # inertie
```

Dans l'exercice (600 points en 5 groupes), la même fonction (qui garde en plus l'inertie de chaque itération) donne les mêmes centres que `KMeans` de scikit-learn à $3{,}6 \times 10^{-15}$ près ; l'inertie passe de 23 519 à 2 362 en 9 itérations.

Trois limites, toutes visibles dans l'exercice.

1. Minima locaux. L'inertie n'est pas convexe : sur 50 initialisations au hasard, 40 % s'arrêtent à plus de 1 % de la meilleure inertie (la pire à 5 118, plus du double), avec deux vrais groupes fusionnés et un autre coupé en trois. Le remède standard est k-means++ (Arthur et Vassilvitskii, 2007) : choisir chaque nouveau centre de départ au hasard avec une probabilité proportionnelle au carré de sa distance au centre déjà choisi le plus proche, ce qui écarte les centres les uns des autres. Les échecs tombent à 20 % ; on relance de toute façon plusieurs fois et on garde la meilleure inertie.
2. Il faut choisir k. On trace l'inertie en fonction de $k$ (elle baisse toujours) et on cherche le « coude », ou on utilise un critère comme la silhouette. Aucun n'est infaillible.
3. Groupes ronds seulement. Chaque point va au centre le plus proche : les frontières sont des droites (des cellules de Voronoï) et les groupes sont convexes. Sur deux lunes peu bruitées, k-means coupe par le milieu ; l'accord avec les vraies lunes, mesuré par l'indice de Rand ajusté (1 = identique, 0 = hasard), n'est que de 0,248.

DBSCAN (Martin Ester et ses coauteurs, Université de Munich, 1996) part d'une autre définition du groupe : une région dense. Un point est un « cœur » si son voisinage de rayon $\varepsilon$ contient au moins `min_samples` points (lui compris, dans scikit-learn) ; deux cœurs voisins sont dans le même groupe, de proche en proche ; les points non denses rattachés à un cœur sont en bordure, les autres sont déclarés bruit. DBSCAN n'a pas besoin de $k$, trouve des groupes de forme quelconque et isole les points aberrants. Sur les lunes : indice de Rand ajusté de 1,000, contre 0,248 pour k-means. Son point faible : un seul rayon $\varepsilon$ pour tout l'espace, ce qui marche mal quand les groupes ont des densités différentes.

Le partitionnement hiérarchique construit tous les niveaux de regroupement à la fois. Version ascendante : chaque point est d'abord son propre groupe, puis on fusionne à chaque étape les deux groupes les plus proches, jusqu'à n'en avoir qu'un. Le résultat est un arbre (un dendrogramme) qu'on coupe à la hauteur voulue. Tout dépend de la définition de « proches » : distance minimale entre leurs points (groupes allongés, effet de chaîne), maximale, moyenne, ou critère de Ward (1963) : fusionner la paire qui fait le moins augmenter l'inertie, la quantité même de k-means. Le partitionnement hiérarchique coûte au moins $n^2$ distances : réservé à quelques milliers de points.

Le partitionnement « doux », où chaque point appartient à chaque groupe avec une probabilité (mélanges de gaussiennes et algorithme EM, dont k-means est un cas limite), est au chapitre 09.

### 9. La réduction de dimension : ACP, t-SNE, UMAP

L'ACP. Les 1 797 images du jeu *digits* sont des chiffres manuscrits de 8 × 8 pixels : 64 variables. Pour les voir, il faut les ramener à 2. L'analyse en composantes principales (ACP) cherche la direction $w$ le long de laquelle les données varient le plus, puis la suivante orthogonale à la première, et ainsi de suite. En 1901, Karl Pearson posait la même question sous une autre forme : la droite (le plan) qui passe au plus près du nuage de points, au sens des distances perpendiculaires. Les deux formulations donnent la même réponse.

(Formule : Analyse en composantes principales.)

La solution est le vecteur propre de $\Sigma$ de plus grande valeur propre. En pratique, on passe par la décomposition en valeurs singulières $X_c = U S V^\top$ : les lignes de $V^\top$ sont les composantes, et les variances valent $s_j^2/(n-1)$. La part de variance expliquée d'une composante est sa variance divisée par la variance totale.

```python
def acp(X, n_composantes):
    Xc = X - X.mean(axis=0)                              # centrer
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)    # Xc = U · diag(S) · Vt
    variance = S2 / (len(X) - 1)
    return Xc @ Vt[:n_composantes].T, variance / variance.sum()   # coordonnées, part de variance
```

Sur *digits*, cette fonction donne les mêmes parts de variance que `PCA` à $5{,}6 \times 10^{-17}$ près. Les deux premières composantes gardent 28,5 % de la variance (14,9 % + 13,6 %) ; il en faut 21 sur 64 pour atteindre 90 %. En 2D, les chiffres se chevauchent : un 5-NN dans le plan de l'ACP ne retrouve le bon chiffre que dans 60,3 % des cas. L'ACP est linéaire : elle projette, elle ne peut pas « déplier » une structure courbe.

t-SNE. Laurens van der Maaten et Geoffrey Hinton (2008) reprennent une idée de Hinton et Sam Roweis (SNE, 2002) : au lieu de préserver la variance, préserver les voisinages. On convertit les distances en probabilités d'être voisins, dans l'espace d'origine et dans la carte 2D, et on déplace les points de la carte pour que les deux distributions se ressemblent :

(Formule : t-SNE.)

La perplexité (t-SNE) fixe le nombre effectif de voisins de chaque point (l'article indique des valeurs typiques de 5 à 50). La loi de Student sur la carte est la trouvaille de 2008 : sa queue lourde laisse des points modérément éloignés s'écarter franchement, ce qui forme des groupes nets au lieu d'une masse écrasée au centre (le « problème d'encombrement » de la version gaussienne). Sur *digits*, t-SNE forme dix îlots : 97,6 % de 5-NN dans la carte (perplexité 30), contre 60,3 % pour l'ACP ; 97,4 % et 96,5 % pour des perplexités de 5 et 100.

Ce qu'une carte t-SNE ne dit pas :
t-SNE préserve qui est voisin de qui, pas les distances ni les tailles. L'exercice fabrique deux groupes en dimension 10, l'un 9,9 fois plus étalé que l'autre : sur la carte, ils ont presque la même taille (rapport 1,02), car chaque point s'adapte à la densité locale par son $\sigma_i$. Les distances entre groupes ne se lisent pas non plus, et des groupes peuvent apparaître dans du pur bruit à faible perplexité. Une carte t-SNE sert à explorer, jamais à mesurer. Le résultat dépend aussi de la graine et de la perplexité : en comparer plusieurs.

UMAP (Leland McInnes et ses coauteurs, 2018) poursuit la même idée avec un autre appareil mathématique : construire un graphe des plus proches voisins pondéré, puis chercher une disposition en basse dimension dont le graphe ressemble le plus possible au premier. UMAP est nettement plus rapide que t-SNE sur de gros jeux. On lui prête une meilleure conservation de la structure globale, mais elle tient surtout à son initialisation : Dmitry Kobak et George Linderman (2021) montrent qu'avec la même initialisation, t-SNE fait aussi bien. Ses cartes appellent les mêmes précautions de lecture. Il n'est pas dans l'environnement de l'exercice (paquet `umap-learn`), d'où son absence des figures.

### 10. Choisir les hyperparamètres

$k$ pour le k-NN, la profondeur d'un arbre, $C$ et $\gamma$ pour un SVM, le taux et le nombre de tours d'un boosting : ce ne sont pas des paramètres appris par l'entraînement, ce sont des réglages. On les choisit en comparant des scores de validation croisée. Trois stratégies :

- La recherche sur grille essaie toutes les combinaisons d'une liste de valeurs par hyperparamètre.
- La recherche aléatoire tire au hasard le même nombre de combinaisons, typiquement selon une loi log-uniforme pour les réglages qui varient sur plusieurs ordres de grandeur ($C$ entre 0,01 et 1 000).
- L'optimisation bayésienne ajuste aux essais déjà faits un modèle probabiliste du score (un processus gaussien chez Snoek, Larochelle et Adams, 2012 ; voir le chapitre 09) et choisit l'essai suivant là où le gain espéré est le plus grand (l'*expected improvement*). Hyperband (Li et al., 2016-2018) prend une autre voie : lancer beaucoup d'essais avec un petit budget (peu de tours, peu de données) et n'en poursuivre qu'une fraction, les meilleurs, à chaque étape.

L'argument de James Bergstra et Yoshua Bengio (2012) en faveur du hasard tient en un dessin. Avec un budget de 30 essais sur deux hyperparamètres, une grille 5 × 6 n'essaie que 5 valeurs de $C$ et 6 de $\gamma$ ; 30 tirages au hasard en essaient 30 de chacun. Si un seul des deux compte vraiment (le cas fréquent), le hasard l'explore cinq à six fois plus finement.

L'exercice compare les deux pour un SVM RBF sur *digits*, même budget, même validation croisée à 5 plis : les deux trouvent 99,03 % en validation croisée et 99,11 % sur le test mis de côté, en 9 s chacun. La zone des bons réglages ($\gamma$ autour de $10^{-3}$, $C$ assez grand) est large, et la grille y tombe aussi. L'avantage du hasard apparaît quand l'espace a plus de dimensions ou qu'un réglage est inutile.

Le piège de la fuite :
Le score de validation croisée du meilleur réglage est optimiste : on a choisi ce réglage parce qu'il avait bien marché sur ces plis-là. D'où le jeu de test mis de côté avant la recherche, consulté une fois à la fin. Même règle pour tout prétraitement (standardisation, ACP) : il doit être ajusté sur les plis d'entraînement seulement, à l'intérieur de la validation croisée (`Pipeline` de scikit-learn), sinon l'information du pli de validation fuit dans le modèle.

### 11. En une ligne

- GLM (modèles linéaires généralisés, Nelder et Wedderburn, 1972) : le cadre commun de la régression linéaire, de la régression logistique et de la régression de Poisson ; une combinaison linéaire des variables, passée par une fonction de lien, et une loi de la famille exponentielle pour le bruit.
- ICA (analyse en composantes indépendantes, Comon, 1994) : comme l'ACP, une transformation linéaire, mais qui cherche des composantes statistiquement indépendantes plutôt que décorrélées ; l'exemple type est de séparer plusieurs voix enregistrées par plusieurs micros.
- RVM (machine à vecteurs de pertinence, Tipping, 2001) : la même forme de fonction de décision qu'un SVM, $\sum_i w_i\, k(x_i, x)$, apprise dans un cadre bayésien qui annule la plupart des poids ; elle donne des probabilités et utilise souvent moins d'exemples de support.

### Ce que chaque méthode optimise

Il n'y a pas une perte dans ce chapitre mais une par famille, et certaines méthodes n'optimisent rien par gradient. Fiches complètes et comparaisons au chapitre 01.

| Méthode | Ce qui est optimisé | Comment |
|---|---|---|
| k plus proches voisins | aucune perte : pas d'entraînement ; seul $k$ est choisi, par validation croisée | — |
| Bayes naïf | vraisemblance des données (maximum de vraisemblance, chapitre 00) | formule fermée : fréquences, moyennes, variances |
| Arbres (CART, ID3) | impureté de Gini ou entropie, coupe par coupe | recherche exhaustive gloutonne, non dérivable |
| Forêts aléatoires | rien de global : moyenne d'arbres optimisés chacun pour Gini | — |
| AdaBoost | perte exponentielle $e^{-yF(x)}$ | ajout d'un apprenant à la fois |
| Gradient boosting, XGBoost | toute perte dérivable : MSE, log-loss | descente de gradient dans l'espace des fonctions (pas de Newton pour XGBoost) |
| SVM | perte charnière + $\frac12\lVert w\rVert^2$ | optimisation quadratique convexe |
| k-means | inertie | alternance affectation / recentrage (minimum local) |
| Partitionnement hiérarchique (Ward) | hausse d'inertie à chaque fusion, la plus faible possible | fusions gloutonnes, sans retour en arrière |
| DBSCAN | aucune perte : un critère de densité ($\varepsilon$, `min_samples`) | parcours des voisinages de proche en proche |
| ACP | variance projetée (ou erreur de reconstruction) | formule fermée : décomposition en valeurs singulières |
| t-SNE | divergence KL entre voisinages | descente de gradient |
| UMAP | entropie croisée « floue » entre le graphe d'origine et celui de la carte | descente de gradient stochastique |

## Neuf frontières côte à côte

La figure qui résume le chapitre : les neuf modèles de l'exercice sur les mêmes deux lunes, avec leur erreur de test. Chaque famille a sa forme de frontière : droites pour les modèles linéaires (régression logistique, SVM linéaire ; le Bayes naïf gaussien trace en principe une courbe du second degré, ici presque droite), escaliers alignés sur les axes pour les arbres et leurs ensembles, courbes lisses pour le k-NN et le noyau RBF. Sur ces données, ce sont ces derniers qui gagnent.

Sur des données réelles, l'écart se resserre. Exactitude en validation croisée à 5 plis sur *breast cancer* (569 patientes, 30 variables) :

| Modèle | Lunes (erreur de test) | *Breast cancer* (exactitude) |
|---|---|---|
| Bayes naïf gaussien | 13,9 % | 93,9 % |
| Régression logistique | 14,1 % | 97,4 % |
| k plus proches voisins | 10,0 % (k = 11, variables standardisées) | 96,3 % (k = 5) |
| Arbre (profondeur 4 sur les lunes) | 15,4 % | 92,3 % |
| Forêt aléatoire | 11,6 % | 95,3 % |
| AdaBoost | 13,4 % | 96,7 % |
| Gradient boosting (HGB) | 12,0 % | 96,7 % |
| SVM linéaire | 13,8 % | 97,4 % |
| SVM à noyau RBF | 10,0 % | 97,7 % (± 1,6) |

Presque tout le monde est entre 94 et 98 %, et l'écart-type d'un pli à l'autre (1 à 3 points) est du même ordre que les écarts entre modèles : sur 569 exemples, on ne peut pas départager sérieusement les cinq premiers. C'est une leçon générale : le choix du modèle compte souvent moins que la qualité des variables et que l'honnêteté de l'évaluation.

## Ce qui coinçait

Ces méthodes ont dominé de 1995 à 2012 environ, et dominent encore les tableaux de données. Ce qui coinçait :

- Les données brutes. Sur des pixels, des sons, du texte, aucune de ces méthodes ne trouve seule les bonnes variables. Les années 2000 ont été celles de l'ingénierie des caractéristiques : des descripteurs conçus à la main (histogrammes de gradients orientés pour les images, sacs de mots pour le texte), puis un SVM par-dessus. Les réseaux convolutifs apprendront ces représentations eux-mêmes (chapitre 13).
- Le passage à l'échelle. Un SVM à noyau coûte entre $n^2$ et $n^3$ ; un k-NN garde tout le jeu et calcule $n$ distances par prédiction. Les arbres boostés passent à l'échelle, mais pas sur des entrées de millions de pixels.
- La grande dimension. Les distances perdent leur sens (section 3) ; les méthodes fondées sur elles (k-NN, noyau RBF, k-means) en souffrent les premières.
- L'incertitude et les causes cachées. Ces modèles prédisent une classe ou une valeur, souvent sans probabilité fiable (Bayes naïf mal calibré, SVM sans probabilité native : il faut la calibrer après coup, par exemple par la méthode de Platt), et ne représentent pas les variables qu'on n'observe pas : un état caché qui évolue dans le temps, un sujet latent derrière les mots d'un document. Le partitionnement de k-means est « dur » : un point appartient à un groupe, sans nuance.
- Les garanties, avec un astérisque. Les bornes de la théorie (marge, dimension VC) expliquent pourquoi ces modèles généralisent ; elles deviendront incapables d'expliquer pourquoi des réseaux avec bien plus de paramètres que d'exemples généralisent aussi (chapitre 10, puis chapitre 27).

- ACP (Pearson, 1901) : Résumer un nuage par ses directions de plus grande variance.
- Plus proches voisins (1951 ; Cover et Hart, 1967) : Copier l’étiquette des voisins ; k règle le biais et la variance ; ruiné par la grande dimension.
- k-means (Lloyd, 1957) : Affecter au centre le plus proche, recentrer, recommencer ; minimum local, groupes ronds.
- Arbres (CART 1984, ID3 1986) : Des questions « x ≤ s ? » choisies gloutonnement pour purifier les nœuds ; lisible mais instable.
- SVM (1992-1995) : La frontière la plus loin des deux camps, fixée par quelques vecteurs de support ; courbe grâce au noyau.
- Bagging et forêts (1996, 2001) : Moyenner des arbres décorrélés pour tuer la variance.
- Boosting (1997, 2001, 2016) : Ajouter des petits arbres qui corrigent les erreurs des précédents : une descente de gradient sur des fonctions.
- t-SNE (2008), UMAP (2018) : Préserver les voisinages pour voir la structure, au prix des distances et des tailles.
- Modèles probabilistes et variables latentes : Modéliser l’incertitude et les causes cachées : EM, HMM, MCMC.

Et ensuite ? Les méthodes de ce chapitre prédisent, mais raisonnent mal sous incertitude et ignorent ce qu'on ne voit pas. Un médecin, un système de reconnaissance de la parole, un filtre de suivi ont besoin de manipuler des probabilités en chaîne et des variables cachées : la maladie derrière les symptômes, le mot derrière le son, la position derrière une mesure bruitée. Ce sont les réseaux bayésiens, l'algorithme EM (dont k-means est un cas limite), les modèles de Markov cachés et l'échantillonnage MCMC, l'objet du chapitre 09 : Modèles probabilistes et variables latentes.

## Exercice

Le notebook écrit en numpy le k-NN, une coupe d'arbre de Gini, k-means et l'ACP, vérifie chacun contre scikit-learn, puis compare neuf modèles. Ce qu'il faut observer :

- Le U du compromis biais-variance en fonction de k : 0 % d'entraînement mais 13,9 % de test pour k = 1, 29,5 % pour k = 199, minimum vers k = 35 ; la validation croisée choisit k = 11 (11,3 %), et c'est normal avec 200 points.
- La malédiction de la dimension : rapport plus lointain / plus proche de 611 en dimension 1 à 1,11 en dimension 1 000 ; 50 variables de bruit font passer le k-NN de 11 % à 40 % d'erreur.
- L'arbre glouton et la forêt : la souche numpy coupe en $x_2 \le 0{,}3810$ comme scikit-learn ; l'arbre complet (0 % / 14,3 %) surapprend, la forêt corrige sa variance (11,6 %, alors que ses arbres font 16,0 % en moyenne) et résiste aux variables de bruit.
- Le SVM ne dépend que de quelques points : 3 vecteurs de support sur 80.
- k-means : l'inertie baisse toujours (23 519 → 2 362), mais 40 % des initialisations au hasard finissent dans un minimum local ; k-means coupe les lunes (ARI 0,25) quand DBSCAN les retrouve (1,00).
- t-SNE sépare les dix chiffres (97,6 % contre 60,3 % pour l'ACP) mais rend de même taille deux groupes d'étalements très différents.

Questions de réflexion : le 1-NN a 0 % d'erreur d'entraînement ; quelle mesure, calculée sur les seules données d'entraînement, aurait prévenu du problème ? Pourquoi standardiser avant un k-NN ou un SVM, mais pas avant un arbre ? Une forêt de 300 arbres complets a 0 % d'erreur d'entraînement, comme l'arbre complet seul, mais généralise mieux ; qu'est-ce qui a changé ? Face à un tableau de 50 000 lignes et 40 colonnes hétérogènes, quel modèle essayeriez-vous en premier, et lequel jamais ?

## Auto-évaluation

Question : Un nœud d'arbre contient 8 exemples : 4 d'une classe, 4 de l'autre. Une coupe envoie 4 exemples (3 de la classe A, 1 de B) à gauche et 4 (1 de A, 3 de B) à droite. Que vaut le gain d'impureté de Gini ?

Réponse : Parent : $G = 1 - 0{,}5^2 - 0{,}5^2 = 0{,}5$. Chaque enfant : $1 - 0{,}75^2 - 0{,}25^2 = 1 - 0{,}5625 - 0{,}0625 = 0{,}375$. Moyenne pondérée : $0{,}5 \times 0{,}375 + 0{,}5 \times 0{,}375 = 0{,}375$. Gain : $0{,}5 - 0{,}375 = 0{,}125$. L'arbre comparerait ce gain à celui de toutes les autres coupes possibles et garderait la meilleure.

Question : Pourquoi le 1-NN a-t-il toujours 0 % d'erreur d'entraînement, et pourquoi ce chiffre ne dit-il rien ? Comment choisir k honnêtement ?

Réponse : Chaque exemple d'entraînement est son propre plus proche voisin (distance nulle) : il se prédit lui-même. Ce chiffre mesure la mémoire, pas la généralisation ; au test, le 1-NN fait 13,9 % dans l'exercice, car sa frontière contourne chaque point bruité (forte variance). On choisit $k$ par validation croisée sur l'entraînement, en laissant le jeu de test intact jusqu'à la mesure finale ; une mesure de type « laisser un exemple de côté » (chaque point classé par ses voisins autres que lui-même) aurait aussi révélé le problème.

Question : Une forêt aléatoire et AdaBoost combinent tous deux des arbres. Qu'est-ce qui les distingue, et quelle erreur (biais ou variance) chacun réduit-il ?

Réponse : La forêt entraîne des arbres profonds, indépendamment et en parallèle, sur des tirages bootstrap et avec des variables tirées au hasard à chaque nœud, puis les fait voter : elle réduit la variance de modèles déjà peu biaisés. AdaBoost (et le gradient boosting) entraîne des arbres petits, en série, chacun concentré sur les erreurs des précédents (poids augmentés, ou résidus), et les additionne : il réduit le biais de modèles faibles. Conséquence pratique : ajouter des arbres à une forêt ne fait pas surapprendre ; ajouter des tours de boosting, si.

Question : Un SVM à noyau RBF surapprend : erreur d'entraînement nulle, erreur de validation élevée. Quels réglages modifier, et dans quel sens ?

Réponse : Baisser $\gamma$ : chaque vecteur de support influence une zone plus large, la frontière devient plus lisse. Baisser $C$ : les erreurs d'entraînement coûtent moins cher, la marge s'élargit. Les deux se choisissent ensemble par validation croisée, idéalement par recherche aléatoire log-uniforme ou sur grille logarithmique. Vérifier aussi que les variables sont standardisées : le noyau RBF repose sur des distances.

Question : Sur une carte t-SNE, un groupe apparaît deux fois plus grand qu'un autre, et très loin d'un troisième. Que pouvez-vous en conclure ?

Réponse : Presque rien sur les tailles et les distances. t-SNE préserve les voisinages : des points proches dans l'espace d'origine restent proches sur la carte. Mais chaque point adapte la largeur de son voisinage à la densité locale, ce qui égalise les tailles apparentes (dans l'exercice, un groupe 9,9 fois plus étalé ressort à 1,02 fois), et les distances entre groupes éloignés ne sont pas conservées. On peut conclure que les points d'un même îlot se ressemblent ; pour mesurer des tailles ou des distances, revenir aux données d'origine (ou à l'ACP, qui est linéaire).

## Pour aller plus loin

- Articles originaux : Cover et Hart (1967), Quinlan (1986), Cortes et Vapnik (1995), Breiman (1996, 2001), Freund et Schapire (1997), Friedman (2001), Chen et Guestrin (2016), van der Maaten et Hinton (2008), Bergstra et Bengio (2012) : liens dans les fiches ci-dessus. Les deux articles de Breiman et celui de Friedman sont particulièrement lisibles.
- Stanford CS229, notes de cours : chapitres 4 (algorithmes génératifs, Bayes naïf), 5 et 6 (noyaux, SVM), 10 (k-means), 12 (ACP), avec les dérivations complètes.
- T. Hastie, R. Tibshirani et J. Friedman, *The Elements of Statistical Learning* (2e éd., PDF gratuit sur le site) : la référence pour les arbres, le boosting (chapitre 10) et les forêts (chapitre 15), écrite par trois des auteurs de ces méthodes. Version plus accessible, avec code R et Python : *An Introduction to Statistical Learning*.
- M. Wattenberg, F. Viégas et I. Johnson, « How to Use t-SNE Effectively » (*Distill*, 2016) : cartes interactives qui montrent comment la perplexité déforme tailles, distances et formes. Pour UMAP : « Understanding UMAP » (équipe PAIR de Google) et la page « How UMAP Works » de sa documentation.
- Stanford CS231n, « Image Classification » : le k-NN appliqué aux pixels, et pourquoi la distance entre images brutes ne mesure pas leur ressemblance, la transition vers les réseaux convolutifs.
- Documentation de scikit-learn : « Classifier comparison » et « Comparing different clustering algorithms on toy datasets » (la version longue des figures de ce chapitre), puis les guides ensembles et SVM.
- Documentation de XGBoost, « Introduction to Boosted Trees » : la dérivation de la valeur optimale d'une feuille et du gain d'une coupe, pas à pas.
- F. Bach, *Learning Theory from First Principles* (MIT Press, PDF en accès libre) : les méthodes à noyaux, les plus proches voisins et la malédiction de la dimension sous l'angle théorique, par un chercheur de l'Inria et de l'ENS.
