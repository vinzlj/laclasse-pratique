# 03. Panorama des paradigmes

Repère historique (1965-2020) : Savoir quel signal d'apprentissage utiliser selon les données dont on dispose.

Ce chapitre n’a pas de date fondatrice : les paradigmes sont nés séparément et chacun a son histoire dans les chapitres qui suivent. Le vocabulaire, lui, s’est fixé par étapes. Dès 1965, un article de Scudder est cité comme l’une des premières traces de l’auto-apprentissage : une machine qui s’entraîne sur ses propres décisions. En 1989, le neurophysiologiste Horace Barlow publie une synthèse intitulée « Unsupervised Learning », qui demande ce qu’un cerveau peut tirer du flot d’informations sensorielles qui lui arrive sans récompense ni punition. En 2006, le livre dirigé par Chapelle, Schölkopf et Zien fait la synthèse du semi-supervisé, actif depuis les années 1970, et formalise ses hypothèses. En 2018, Zhou propose une typologie de la supervision faible, terme déjà en usage : incomplète, grossière, erronée. Entre 2018 et 2020, l’auto-supervisé, qui fabrique ses cibles à partir des données, devient la façon standard de pré-entraîner les grands modèles.

Comment lire ce chapitre :
C’est une page d’orientation. Elle explique en détail ce qu’elle est seule à couvrir : le semi-supervisé et la supervision faible. Pour le reste (renforcement, auto-supervisé, partitionnement), elle donne l’idée, une formule et le chapitre où la méthode est vraiment expliquée. Revenez-y quand vous vous demanderez « mais d’où vient le signal, ici ? ».

## L’idée en une image

Les chapitres 00 à 02 ont posé la boucle : des données, un modèle, une perte, un optimiseur. Ils supposaient tous que chaque exemple arrive avec sa bonne réponse. Ce chapitre pose la question qui vient avant : quelle information avez-vous vraiment sous la main ? C’est elle qui décide du paradigme, bien plus que le modèle.

Prenons un site e-commerce qui veut ranger automatiquement les photos de produits par catégorie. Selon ce que l’équipe possède, ce n’est pas le même problème :

| Ce que vous avez | Le signal | Paradigme | Chapitres |
|---|---|---|---|
| 50 000 photos classées à la main par les marchands | la bonne catégorie de chaque photo | apprentissage supervisé | 08, 13 |
| 2 millions de photos, aucune catégorie | seulement les photos : leurs ressemblances | apprentissage non supervisé | 08, 09, 18 |
| 500 photos classées, 2 millions non classées | quelques réponses et beaucoup de structure | apprentissage semi-supervisé | ce chapitre |
| des catégories tirées des flux marchands, fausses dans 10 à 20 % des cas, ou une seule catégorie pour tout un lot | des réponses imparfaites | apprentissage faiblement supervisé | ce chapitre |
| aucune réponse, mais des clics et des achats après avoir montré un produit | une récompense, qui dépend de ce qu’on a fait | apprentissage par renforcement | 11, 17 |
| les fiches produits elles-mêmes : on cache un mot de la description et on le fait deviner | une cible fabriquée à partir des données | apprentissage auto-supervisé | 20, 23, 26 |

La ressource rare, presque toujours, c’est l’étiquette : la réponse attendue, écrite par un humain. Une photo coûte quelques millisecondes à stocker ; la classer correctement coûte quelques secondes à quelques minutes d’attention humaine, beaucoup plus pour une radiographie ou un contrat. Les paradigmes se lisent comme une échelle : à quel point peut-on se passer d’étiquettes, et qu’est-ce que cela coûte en qualité ?

La figure suivante montre l’essentiel sur un jeu jouet : 300 points d’apprentissage en forme de deux croissants imbriqués, 300 points de test. Seul le signal change d’un panneau à l’autre.

Ce qu’on y lit (seed 42, précision sur les 300 points de test) :

1. Supervisé, 300 étiquettes justes, SVM à noyau gaussien (chapitre 08) : 98,7 % (296 points sur 300).
2. Faiblement supervisé, chaque étiquette inversée avec une probabilité de 25 %, soit 85 étiquettes fausses sur 300 : 99,3 %. La frontière ne bouge presque pas. L’écart avec le panneau 1 tient à 2 points de test sur 300 : ce n’est pas une amélioration, c’est du hasard. On verra pourquoi un bruit symétrique gêne si peu.
3. Semi-supervisé, 3 étiquettes par classe et 294 points sans étiquette, propagation d’étiquettes : 98,7 %. Le même SVM, entraîné sur les 6 points étiquetés seuls, tombe à 94,0 % (frontière en tirets). Ce tirage-là est favorable : sur 20 autres tirages des 6 points (graines 42 à 61, imprimés par le script de la figure), la moyenne est de 87,1 % contre 80,4 %, et la propagation va de 69,7 % à 99,0 % selon le tirage. Le gain est réel, son ampleur varie.
4. Non supervisé, k-moyennes à deux groupes, aucune étiquette : 75,0 % d’accord avec les vraies classes. L’algorithme trouve bien deux groupes, mais pas ceux qu’on voulait : il coupe le nuage par une droite.

Le passage « supervisé seul » → « supervisé + données non étiquetées » est l’avant/après de ce chapitre :

| | Avant : on n’utilise que les exemples étiquetés | Après : on exploite aussi les exemples sans étiquette |
|---|---|---|
| Données utilisées | les $l$ exemples $(x_i, y_i)$ | les $l$ exemples étiquetés et $u$ exemples $x_j$ seuls, avec souvent $u \gg l$ |
| Ce que le modèle sait de la forme des données | ce que montrent $l$ points | la densité des données : où elles sont regroupées, où il y a des vides |
| Croissants, 6 étiquettes (seed 42) | 94,0 % | 98,7 % |
| Chiffres manuscrits, 20 étiquettes (moyenne de 5 tirages) | 75,6 % | 86,1 % |
| Chiffres manuscrits, 1 000 étiquettes | 98,7 % | 98,1 % |
| Ce qui coince | il faut beaucoup d’étiquettes | ne marche que si la structure des données suit les classes ; le gain disparaît quand les étiquettes abondent |

La carte ci-dessous situe chaque paradigme dans le parcours.

Les grands modèles combinent ces cases : pré-entraînement (*pre-training*) auto-supervisé, affinage (*fine-tuning*) supervisé, puis renforcement (chapitres 28, 30 et 31). L’apprentissage par imitation, lui, est du supervisé sur des actions (chapitre 17).

## Le mécanisme

Tous les paradigmes partagent la boucle des chapitres 00 à 02. Ce qui change, c’est ce que contient le jeu de données et donc ce que la perte peut comparer. Le plus simple est de l’écrire comme du code : la boucle est identique, seule la fonction qui fournit la cible change.

```python
# Une seule boucle d'entraînement ; le paradigme = d'où vient la cible.
for lot in donnees:
    x = lot.entree
    if paradigme == "supervisé":
        cible = lot.etiquette                      # écrite par un humain
    elif paradigme == "semi-supervisé":
        cible = lot.etiquette if lot.a_une_etiquette else pseudo_etiquette(modele, x)
    elif paradigme == "faiblement supervisé":
        cible = lot.etiquette_bruitee              # en partie fausse, ou trop grossière
    elif paradigme == "auto-supervisé":
        x, cible = masquer_une_partie(x)           # la cible est un morceau de l'entrée
    perte = fonction_de_perte(modele(x), cible)
    perte.backward(); optimiseur.step()

# Le non supervisé n'a pas de cible : la perte ne dépend que de x (distance aux centres,
# vraisemblance des données…). Le renforcement n'a pas de jeu de données fixe :
# c'est l'agent qui le produit en agissant, et la « cible » est une récompense.
```

### Ce que la méthode optimise

Aucune fonction de perte propre à ce chapitre. Chaque paradigme réutilise les pertes du chapitre 01, appliquées à des cibles différentes : entropie croisée pour une classe (vraie, bruitée ou pseudo-étiquette, mot masqué), erreur quadratique pour une valeur. Le non supervisé et le renforcement ont leurs critères propres, rappelés plus bas et détaillés aux chapitres 08, 09 et 11.

### Supervisé : une réponse par exemple

Le jeu de données est une liste de couples $(x_i, y_i)$ : une entrée et la réponse attendue. On cherche le modèle qui se trompe le moins sur ces couples :

C’est le cadre des chapitres 00 et 01, et celui de la majorité des applications industrielles : détection de spam, prédiction de prix, reconnaissance d’images. Sa force : le signal est direct, chaque erreur est mesurée. Sa limite : il faut $N$ étiquettes, et la qualité du modèle plafonne à celle des étiquettes. Sur les croissants, 300 étiquettes donnent 98,7 % ; sur les chiffres manuscrits de scikit-learn (1 257 images d’apprentissage), le plus proche voisin passe de 67,1 % avec 10 étiquettes à 98,7 % avec les 1 257.

### Non supervisé : seulement des entrées

Le jeu de données est une liste de $x_i$ sans réponse. Il n’y a plus rien à « prédire » : on cherche une structure. Trois familles de tâches :

- le partitionnement : répartir les exemples en groupes (k-moyennes, chapitre 08 ; mélanges de gaussiennes et EM, chapitre 09) ;
- la réduction de dimension : résumer chaque exemple par quelques nombres. Au chapitre 08, l’analyse en composantes principales et t-SNE ; au chapitre 18, les autoencodeurs ;
- l’estimation de densité et la génération : apprendre la distribution des données pour en tirer de nouvelles (chapitres 09, 18 et 25).

Faute de cible, c’est le critère choisi qui définit ce qu’est une « bonne » structure. Les k-moyennes, par exemple, minimisent l’inertie, la somme des carrés des distances de chaque point au centre de son groupe (rappel en une ligne, explication au chapitre 08) :

Ce critère favorise des groupes ronds, de taille comparable. Un point est affecté au centre le plus proche, donc deux groupes sont toujours séparés par une droite (la médiatrice des deux centres). Les croissants ne sont pas séparables par une droite : d’où les 75,0 % du panneau 4. Le résultat n’est pas « faux », il répond à une autre question que la vôtre.

Le non supervisé reste précieux quand on n’a aucune étiquette. Sur les chiffres manuscrits, on lance les k-moyennes avec 10 groupes, puis on ne fait étiqueter qu’une image par groupe (celle qui est la plus proche de son centre) et on donne sa classe à tout le groupe. Avec ces 10 étiquettes seulement : 78,1 %, mieux que le supervisé (67,1 %) et que la propagation d’étiquettes (74,0 %) à 10 étiquettes tirées au hasard, une par classe (moyenne de 5 tirages). Choisir quelles images faire étiqueter compte autant que leur nombre : c’est l’idée de l’apprentissage actif.

### Semi-supervisé : peu d’étiquettes, beaucoup de données

Le jeu de données contient $l$ exemples étiquetés et $u$ exemples non étiquetés, en général $u \gg l$. La question de fond : en quoi une donnée sans réponse peut-elle aider à prédire la réponse ? Elle ne dit rien sur $y$. Elle dit en revanche où se trouvent les données, c’est-à-dire la forme de $p(x)$. Cela n’aide que si cette forme est liée aux classes. Le livre de Chapelle, Schölkopf et Zien (2006) formule trois hypothèses qui rendent ce lien plausible :

- continuité : deux points proches dans une région dense ont probablement la même étiquette ;
- hypothèse de regroupement : les points d’un même amas ont probablement la même classe ; autrement dit, la frontière passe dans les zones vides ;
- hypothèse de variété : les données de grande dimension se trouvent près d’une surface de petite dimension. Une image de chiffre a 64 pixels, mais les images de « 3 » ne remplissent pas l’espace des 64 dimensions : elles forment une surface étroite, qu’on peut cartographier sans étiquette.

Sur les croissants, les trois hypothèses tiennent : chaque croissant est dense, et un couloir vide les sépare. Avec 6 points étiquetés seulement, un SVM trace une frontière raisonnable autour de ces 6 points mais ne sait pas que les croissants existent ; les 294 points gris le lui montrent.

La plupart des méthodes semi-supervisées minimisent, sous une forme ou une autre, une perte en deux termes :

(Formule : Objectif semi-supervisé (forme générale).)

Trois façons de remplir $R$ ont marqué l’histoire.

#### 1. L’auto-apprentissage et les pseudo-étiquettes

L’idée la plus ancienne (Scudder, 1965, d’après Chapelle et al.) est l’auto-apprentissage : on entraîne un modèle sur les exemples étiquetés, on lui fait prédire les exemples non étiquetés, on garde ses prédictions les plus sûres comme pseudo-étiquettes, on réentraîne sur le tout, et on recommence.

```python
# Auto-apprentissage, version minimale (scikit-learn propose SelfTrainingClassifier)
X_lab, y_lab = X[etiquetes], y[etiquetes]
X_sans = X[~etiquetes]
for tour in range(10):
    modele.fit(X_lab, y_lab)
    proba = modele.predict_proba(X_sans)
    sur = proba.max(axis=1) >= 0.95                 # seulement les prédictions confiantes
    if not sur.any():
        break
    X_lab = np.vstack([X_lab, X_sans[sur]])         # les pseudo-étiquettes deviennent
    y_lab = np.concatenate([y_lab, modele.classes_[proba[sur].argmax(axis=1)]])  # des étiquettes
    X_sans = X_sans[~sur]
```

Le risque est évident : si le modèle se trompe avec assurance, il apprend sa propre erreur et la renforce. Le seuil de confiance limite les dégâts sans les supprimer. L’idée a pourtant passé à l’échelle : en 2019-2020, Noisy Student (Xie et al.) entraîne un réseau sur ImageNet, lui fait produire des pseudo-étiquettes pour 300 millions d’images non étiquetées, puis entraîne un réseau plus grand sur le tout, avec du bruit (dropout, augmentation de données) pour que l’élève dépasse le maître ; 88,4 % de précision top-1 sur ImageNet dans la version publiée à CVPR 2020, 2 points de plus que le meilleur modèle précédent, qui utilisait 3,5 milliards d’images Instagram faiblement étiquetées. En 2020, FixMatch (Sohn et al.) combine pseudo-étiquette et cohérence :

(Formule : Terme non étiqueté de FixMatch.)

En clair : « si tu es sûr de toi sur l’image à peine retouchée, alors donne la même réponse sur l’image très déformée ». Avec 4 images étiquetées par classe (40 en tout) et le reste de CIFAR-10 sans étiquette, FixMatch atteint 88,61 % de précision, en moyenne sur 5 tirages des 40 images étiquetées (± 3,35 points d’un tirage à l’autre).

#### 2. La propagation d’étiquettes sur un graphe

La deuxième famille prend l’hypothèse de regroupement au pied de la lettre. On construit un graphe dont les nœuds sont tous les exemples, étiquetés ou non, et dont les arêtes relient les exemples proches, avec un poids d’autant plus fort qu’ils sont proches. Puis on laisse les étiquettes couler le long des arêtes : c’est la propagation d’étiquettes de Zhu et Ghahramani (2002). Dans sa forme la plus simple, dite harmonique (Zhu, Ghahramani et Lafferty, 2003), chaque nœud non étiqueté prend la moyenne pondérée des étiquettes de ses voisins ; les nœuds étiquetés sont remis à leur vraie valeur à chaque tour.

(Formule : Propagation d’étiquettes, forme harmonique (une itération).)

Exemple à la main. Cinq points sur une ligne : A est étiqueté bleu, E est étiqueté orange, B, C et D ne le sont pas. Notons $F$ la probabilité d’être bleu : $F_A = 1$, $F_E = 0$, et on part de $F = 0{,}5$ pour les autres. Les arêtes A–B, B–C et D–E ont un poids 1 (points proches) ; l’arête C–D a un poids 0,1 : il y a un vide entre C et D.

```text
 A ━━━━ B ━━━━ C ┄┄┄┄┄┄ D ━━━━ E
bleu    ?      ?  vide  ?   orange
F=1                          F=0
   w=1    w=1   w=0,1    w=1
```

Les équations d’équilibre sont $F_B = \frac{F_A + F_C}{2}$, $F_C = \frac{F_B + 0{,}1\,F_D}{1{,}1}$ et $F_D = \frac{0{,}1\,F_C + F_E}{1{,}1}$. Premier tour : $F_B = (1 + 0{,}5)/2 = 0{,}75$, $F_C = (0{,}5 + 0{,}05)/1{,}1 = 0{,}5$, $F_D = 0{,}05/1{,}1 \approx 0{,}045$. En itérant, on converge vers :

$$
F_B = \tfrac{12}{13} \approx 0{,}92, \qquad F_C = \tfrac{11}{13} \approx 0{,}85, \qquad F_D = \tfrac{1}{13} \approx 0{,}08.
$$

C est à deux arêtes de A comme de E : un simple « plus proche étiqueté en nombre de sauts » hésiterait. La propagation le classe nettement bleu, parce que l’étiquette passe facilement dans la zone dense et presque pas à travers le vide. La frontière se place dans le vide, exactement ce que dit l’hypothèse de regroupement. Sans le vide (quatre points A–B–C–D à poids égaux, A bleu, D orange), on trouverait $F_B = 2/3$ et $F_C = 1/3$ : une frontière au milieu.

(Le rapport de 2002 relie tous les points, boucles comprises, et normalise la matrice des poids par colonnes puis par lignes. Appliqué à ce graphe, son point fixe est proche mais pas identique : 0,89 / 0,76 / 0,06 sans boucles, 0,90 / 0,80 / 0,07 avec. La frontière reste dans le vide. La version exacte « moyenne des voisins » est la fonction harmonique de Zhu, Ghahramani et Lafferty, en 2003.)

```python
import numpy as np

# Graphe de l'exemple : A, B, C, D, E ; le poids 0,1 entre C et D est le « vide »
W = np.zeros((5, 5))
for i, j, w in [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 0.1), (3, 4, 1.0)]:
    W[i, j] = W[j, i] = w

F = np.array([1.0, 0.5, 0.5, 0.5, 0.0])   # probabilité d'être bleu
for _ in range(200):
    F = W @ F / W.sum(axis=1)              # moyenne pondérée des voisins
    F[0], F[4] = 1.0, 0.0                  # on remet les étiquettes connues
print(F.round(4))                          # [1. 0.9231 0.8462 0.0769 0.]
```

En 2003, Zhou, Bousquet, Lal, Weston et Schölkopf proposent une variante (*label spreading*) : les points étiquetés ne sont plus remis de force à leur valeur, ils sont seulement tirés vers elle. Son itération s’écrit $F \leftarrow \alpha S F + (1 - \alpha) Y$, où $S$ est la matrice des poids normalisée, $Y$ les étiquettes initiales et $\alpha \in (0, 1)$ dose l’équilibre entre « écouter ses voisins » et « garder son étiquette ». Une étiquette fausse peut donc être corrigée par ses voisins. C’est cette variante (`LabelSpreading` de scikit-learn, $\alpha = 0{,}2$ par défaut) qui produit les figures du chapitre ; leur article l’illustrait déjà sur deux demi-lunes.

Sur les chiffres manuscrits, la propagation se fait sur le graphe des 7 plus proches voisins des 1 257 images d’apprentissage. On la compare au plus proche voisin entraîné sur les seules images étiquetées. Les deux reposent sur la même idée, « les voisins se ressemblent » : l’écart vient donc des images non étiquetées, pas d’un modèle plus puissant.

| Images étiquetées | Supervisé (plus proche voisin) | Semi-supervisé (propagation) | Gain |
|---|---|---|---|
| 10 | 67,1 % | 74,0 % | +6,9 points |
| 20 | 75,6 % | 86,1 % | +10,5 |
| 50 | 84,6 % | 92,3 % | +7,7 |
| 100 | 91,5 % | 95,1 % | +3,6 |
| 500 | 97,7 % | 97,9 % | +0,2 |
| 1 000 | 98,7 % | 98,1 % | −0,6 |

Moyenne de 5 tirages, 540 images de test ; référence avec les 1 257 étiquettes : 98,7 %. Avec 20 étiquettes, la propagation fait aussi bien que le supervisé avec 50. À 1 000 étiquettes, elle fait même un peu moins bien : quand les étiquettes abondent, le graphe n’apporte plus d’information et son lissage peut gommer des détails. Les données non étiquetées valent beaucoup quand les étiquettes sont rares, et presque rien quand elles abondent.

#### 3. Pré-entraîner sans étiquette, puis affiner

La troisième voie, devenue dominante, découple les deux temps : on apprend d’abord une représentation sur toutes les données, sans étiquette (souvent par auto-supervision, plus bas), puis on l’adapte avec les quelques exemples étiquetés. Le terme $R$ disparaît de la perte : les données non étiquetées servent avant, pas pendant. C’est la recette des grands modèles actuels (chapitre 28, chapitre 30).

Quand le semi-supervisé fait pire :
Si la structure des données ne suit pas les classes, les données non étiquetées trompent le modèle. Exemple : classer des photos de produits par marque, alors que les photos se regroupent surtout par fond (studio blanc, photo amateur). La propagation suivra les fonds. Chapelle et al. le rappellent : le semi-supervisé n’aide que sous ses hypothèses, et rien ne garantit d’avance qu’il fera mieux que le supervisé seul. Gardez toujours le modèle supervisé comme point de comparaison, mesuré sur un jeu de validation étiqueté.

### Faiblement supervisé : des étiquettes imparfaites

En 2018, Zhou classe en trois types les situations de supervision faible (le terme était déjà en usage), où les étiquettes existent mais ne sont pas celles qu’on voudrait :

| Type | Ce qui manque | Exemple | Réponses typiques |
|---|---|---|---|
| Incomplète (*incomplete*) | une partie des exemples n’a pas d’étiquette | 500 photos classées sur 2 millions | semi-supervisé (ci-dessus), apprentissage actif |
| Grossière (*inexact*) | l’étiquette porte sur un groupe, pas sur l’exemple | « cette commande contient un article défectueux », sans dire lequel ; « cette molécule a au moins une forme active » | apprentissage multi-instances |
| Erronée (*inaccurate*) | une partie des étiquettes est fausse | catégories tirées des flux marchands, annotation par la foule, règles automatiques | modèles du bruit, pertes robustes, nettoyage des étiquettes |

Le cas le plus fréquent en pratique est le bruit d’étiquetage. Pourquoi les 85 étiquettes fausses sur 300 du panneau 2 n’ont-elles presque rien changé ? Supposons que chaque étiquette soit inversée avec une probabilité $\rho$, quelle que soit sa classe (bruit symétrique). Pour un point $x$, la probabilité de voir l’étiquette « orange » devient :

Avec $\rho = 0{,}25$ : $\tilde P = 0{,}25 + 0{,}5\,P$. Un point orange à 90 % est vu orange à $0{,}25 + 0{,}45 = 70\,\%$ ; un point à 50 % reste à 50 %. La transformation est croissante et laisse 0,5 en place : le seuil de décision ne bouge pas. Un modèle entraîné à l’entropie croisée sur les étiquettes bruitées estime $\tilde P$, moins tranché que $P$, mais du même côté de 0,5 partout. La frontière est la même, les probabilités sont seulement moins confiantes. (En simplifiant : il faut assez de données et un modèle assez souple pour bien estimer $\tilde P$ ; avec peu de points, chaque étiquette fausse déforme localement la frontière.)

Deux cas cassent ce raisonnement :

- $\rho \geq 0{,}5$ : le facteur $1 - 2\rho$ devient nul ou négatif, l’information disparaît ou s’inverse ;
- un bruit qui dépend de la classe (asymétrique). Si 40 % des orange sont étiquetés bleus et aucun bleu n’est étiqueté orange, $\tilde P(\text{orange} \mid x) = 0{,}6\, P(\text{orange} \mid x)$. Pour que l’étiquette bruitée penche vers orange, il faut $0{,}6\,P > 0{,}5$, soit $P > 0{,}83$ : la frontière recule nettement dans la zone orange, et tous les points orange un peu ambigus seront classés bleus. C’est le cas réaliste (un annotateur pressé confond toujours les mêmes catégories), et il faut alors estimer le bruit pour le corriger.

La supervision faible est aussi une méthode de travail. Avec Snorkel (Ratner et al., 2017), des experts n’étiquettent rien à la main : ils écrivent des *fonctions d’étiquetage*, de petites règles imparfaites (« si la description contient “taille XL”, c’est un vêtement »). Le système estime la fiabilité de chaque règle à partir de leurs accords et désaccords, sans vérité terrain, puis produit des étiquettes probabilistes pour entraîner un modèle. Dans leur étude, des experts ont construit des modèles 2,8 fois plus vite qu’en étiquetant à la main, avec une performance prédictive en moyenne 45,5 % meilleure qu’après sept heures d’étiquetage manuel. Pour un développeur, c’est une transposition naturelle : écrire des règles est plus rapide qu’annoter.

### Auto-supervisé : la cible est dans les données

L’apprentissage auto-supervisé n’a pas d’étiquette humaine, comme le non supervisé, mais il a une cible, comme le supervisé : on la fabrique en cachant une partie de l’entrée et en demandant au modèle de la retrouver. On résout ainsi une tâche prétexte, qui n’intéresse personne en elle-même, parce qu’elle force le modèle à comprendre les données.

(Formule : Forme générale.)

Le gain est quantitatif : la phrase « le chat dort sur le canapé » fournit, sans annotateur, 5 exemples de prédiction du mot suivant (« le » → « chat », « le chat » → « dort », etc.). Un texte de $n$ mots en donne $n - 1$. Le web entier devient un jeu d’entraînement étiqueté gratuitement. Les variantes se distinguent par ce qu’on cache et par l’espace où l’on compare :

| Tâche prétexte | Exemple | Chapitre |
|---|---|---|
| prédire le mot suivant | GPT et les LLM | 20, 28 |
| prédire les mots voisins | word2vec | 15 |
| retrouver des mots masqués | BERT | 20, 26 |
| reconnaître deux vues d’un même exemple parmi d’autres | SimCLR, CLIP | 23 |
| reconstruire des morceaux d’image, ou les prédire dans un espace de représentation | MAE, JEPA | 26 |

Supervisé ou non supervisé ? Les deux étiquettes se défendent : techniquement, la perte est supervisée ; mais aucune étiquette humaine n’intervient. Liu et al. (2020) rapportent la définition donnée par Yann LeCun en 2020 : la machine prédit n’importe quelle partie de son entrée à partir de n’importe quelle partie observée. Le chapitre 26 revient sur ce débat et sur la question de savoir quoi prédire.

### Renforcement : une récompense, après coup

En renforcement, il n’y a pas de jeu de données fixé à l’avance. Un agent agit dans un environnement et reçoit une récompense chiffrée ; c’est lui qui produit ses propres données, et elles dépendent de ses choix.

Ce que l’agent cherche à maximiser, c’est le retour $G_t$ : la somme de ses récompenses futures, pondérées par un facteur d’actualisation $\gamma$ (rappel, explication complète au chapitre 11) :

Trois différences avec le supervisé, qui expliquent pourquoi le renforcement a ses propres chapitres :

- la récompense ne dit pas quelle était la bonne action, seulement si le résultat était bon. Un « 0 » ne dit pas ce qu’il aurait fallu faire ;
- elle est retardée. Aux échecs, on ne sait qu’au 40e coup si la partie est gagnée ; lequel des 40 coups a fait la différence ? Avec $\gamma = 0{,}9$, une récompense de 1 reçue 10 pas plus tard ne vaut que $0{,}9^{10} \approx 0{,}35$ au moment de la décision ;
- l’agent choisit ses données : s’il ne tente jamais une action, il ne saura jamais ce qu’elle rapporte. Il doit arbitrer entre exploiter ce qu’il sait et explorer.

Le renforcement se combine avec les autres paradigmes : l’apprentissage par imitation (copier les actions d’un expert, chapitre 17) est du supervisé sur des actions, et l’alignement des LLM (chapitre 30) enchaîne pré-entraînement auto-supervisé, affinage supervisé et renforcement à partir de préférences humaines.

## Ce qui coinçait

Ce chapitre n’a pas de méthode qui « remplace » une autre : il compare des façons d’obtenir un signal. Chacune a sa limite, et chaque limite a poussé vers une autre case de la carte.

- Le supervisé coûte des étiquettes. C’est le plafond de la plupart des projets réels, et ce qui a motivé tout le reste du tableau.
- Le non supervisé répond à sa propre question. Sans cible, le critère (distance, vraisemblance) décide de ce qu’est un groupe : 75,0 % d’accord seulement sur les croissants. Les modèles probabilistes du chapitre 09 rendent ce critère explicite.
- Le semi-supervisé dépend de ses hypothèses. Si les amas ne suivent pas les classes, il trompe ; quand les étiquettes abondent, il n’apporte plus rien (98,1 % contre 98,7 % à 1 000 étiquettes). Les graphes de voisins reposent aussi sur une distance pertinente entre exemples, ce qui est rarement vrai pour des pixels bruts d’images naturelles : il faudra apprendre les représentations (chapitres 13, 23, 26).
- La supervision faible déplace le problème vers l’estimation du bruit : symétrique, il est presque inoffensif ; dépendant de la classe, il déplace la frontière.
- Le renforcement est lent et instable : récompenses rares et retardées, données produites par l’agent lui-même. Ce sera le sujet des chapitres 11 et 17.
- L’auto-supervisé demande beaucoup de calcul et un bon choix de tâche prétexte : prédire chaque pixel d’une vidéo gaspille la capacité du modèle sur des détails imprévisibles (chapitre 26).
- Les frontières entre paradigmes sont floues. Un LLM moderne passe par trois d’entre eux. Retenez la question plutôt que les cases : d’où vient le signal, et que coûte-t-il ?

- Supervisé : Un professeur donne la bonne réponse pour chaque exemple.
- Non supervisé : Pas de réponse : on cherche des groupes ou une forme, et le critère décide de ce qu’on trouve.
- Auto-apprentissage (1965), pseudo-étiquettes : Le modèle étiquette lui-même ce dont il est sûr, puis s’entraîne dessus.
- Propagation d’étiquettes (2002) : Les étiquettes coulent entre voisins ; la frontière s’installe dans les vides.
- Supervision faible (typologie de 2018) : Des étiquettes fausses ou grossières, mais nombreuses ; un bruit symétrique ne déplace pas la frontière.
- Renforcement : Pas de bonne réponse, seulement une note, souvent tardive, sur ce qu’on a fait.
- Auto-supervisé : Cacher une partie des données et la faire deviner : des étiquettes gratuites à l’infini.

Et ensuite ? La boîte à outils est complète : on sait ce qu’apprendre veut dire (chapitre 00), ce qu’on mesure (01), comment on minimise (02) et d’où peut venir le signal (03). Le parcours historique commence maintenant, en 1950, par une IA qui n’apprend rien : elle cherche. Avant les données et les pertes, il y a eu les espaces d’états et les heuristiques, au chapitre 04.

## Exercice

Ce chapitre d’orientation n’a pas d’exercice propre. Chaque paradigme se pratique dans son chapitre : le supervisé dans les exercices des chapitres 00 et 01, les k-moyennes au chapitre 08, le renforcement au chapitre 11, l’auto-supervisé aux chapitres 23 et 26.

Pour voir de vos yeux ce que vaut une donnée non étiquetée, ce script d’une vingtaine de lignes suffit (scikit-learn, quelques secondes sur CPU, à lancer avec `uv run python` depuis la racine du dépôt) :

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelSpreading

X, y = load_digits(return_X_y=True)
X = X / 16.0                                          # pixels 0..16 → 0..1
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

rng = np.random.default_rng(42)
idx = np.concatenate([rng.choice(np.where(ytr == c)[0], 2, replace=False) for c in range(10)])

# Supervisé : le plus proche voisin, sur les 20 images étiquetées seulement
sup = KNeighborsClassifier(n_neighbors=1).fit(Xtr[idx], ytr[idx])

# Semi-supervisé : les 1 257 images, dont 1 237 marquées -1 (« sans étiquette »)
y_partiel = np.full_like(ytr, -1)
y_partiel[idx] = ytr[idx]
semi = LabelSpreading(kernel="knn", n_neighbors=7, max_iter=100).fit(Xtr, y_partiel)

print(f"supervisé      : {sup.score(Xte, yte):.1%}")    # affiche 76.7%
print(f"semi-supervisé : {semi.score(Xte, yte):.1%}")    # affiche 85.4%
```

Un avertissement `RuntimeWarning: invalid value encountered in divide` peut s’afficher ; il est sans effet sur le résultat.

Ce qu’il faut observer :

- avec 2 étiquettes par classe, la propagation gagne environ 9 points sur le plus proche voisin ;
- en passant à 50 étiquettes par classe (remplacez le `2`), l’écart disparaît : 98,1 % des deux côtés avec la graine 42 ;
- avec `random_state` ou la graine de `rng` changés, les deux scores bougent de plusieurs points : à 20 étiquettes, lesquelles on a compte autant que combien.

Questions de réflexion : quelle hypothèse du semi-supervisé les chiffres manuscrits vérifient-ils, et laquelle serait violée si l’on classait les images selon l’épaisseur du trait plutôt que selon le chiffre ? Pourquoi le plus proche voisin et la propagation sont-ils une comparaison plus honnête qu’un réseau profond contre la propagation ?

## Auto-évaluation

Question : Une boutique a 2 millions de photos de produits, dont 3 000 classées par catégorie. Quel paradigme, et à quelle condition fonctionnera-t-il ?

Réponse : Le semi-supervisé (supervision incomplète). Il ne fonctionnera que si la structure des photos suit les catégories : des photos proches dans l’espace de représentation utilisé doivent en général appartenir à la même catégorie (hypothèses de continuité et de regroupement). Si les photos se regroupent par fond ou par éclairage plutôt que par catégorie, les données non étiquetées induiront le modèle en erreur. Comparez toujours au modèle entraîné sur les 3 000 étiquettes seules, sur un jeu de validation étiqueté.

Question : Dans l’exemple à cinq points (A bleu, E orange, poids 0,1 entre C et D), pourquoi C est-il classé bleu alors qu’il est à deux arêtes de A comme de E ?

Réponse : La propagation fait la moyenne des voisins pondérée par les poids. C est relié à B avec un poids 1 et à D avec un poids 0,1 : son étiquette vient presque entièrement de B, lui-même tiré vers A. À l’équilibre, $F_C = 11/13 \approx 0{,}85$. L’étiquette circule bien dans les zones denses et mal à travers les vides : la frontière se place dans le vide entre C et D.

Question : 25 % des étiquettes d’un jeu binaire sont inversées au hasard, indépendamment de la classe. Pourquoi la frontière apprise change-t-elle peu ? Et si seules les étiquettes « orange » étaient inversées, à 40 % ?

Réponse : Avec un bruit symétrique de taux ρ, la probabilité d’observer « orange » vaut $\rho + (1 - 2\rho)P$, soit $0{,}25 + 0{,}5\,P$ : une fonction croissante qui vaut 0,5 exactement quand $P = 0{,}5$. Le seuil de décision est inchangé, seules les probabilités sont moins tranchées (avec assez de données). Si seuls les orange sont inversés à 40 %, la probabilité observée vaut $0{,}6\,P$ ; elle ne dépasse 0,5 que pour $P > 0{,}83$ : la frontière recule dans la zone orange.

Question : Prédire le mot suivant d’un texte : est-ce du supervisé ou du non supervisé ?

Réponse : C’est de l’auto-supervisé : la perte est une entropie croisée ordinaire entre la prédiction et le vrai mot suivant, donc supervisée sur le plan technique, mais la cible est extraite du texte lui-même, sans annotateur. Un texte de n mots fournit n − 1 exemples gratuitement. C’est ce qui permet d’entraîner les LLM sur des quantités de texte impossibles à étiqueter à la main.

Question : Pourquoi un agent de renforcement ne peut-il pas simplement être entraîné comme un classifieur supervisé, avec la récompense comme étiquette ?

Réponse : Parce que la récompense n’indique pas quelle action aurait été la bonne, seulement si le résultat était bon ; parce qu’elle arrive souvent bien après les actions qui l’ont causée (il faut répartir le mérite entre elles) ; et parce que les données dépendent des choix de l’agent : une action jamais tentée ne sera jamais évaluée. D’où l’arbitrage exploration/exploitation et les méthodes propres du chapitre 11.

## Pour aller plus loin

- Articles et livres cités (fiches ci-dessus) : O. Chapelle, B. Schölkopf et A. Zien (dir.), *Semi-Supervised Learning* (MIT Press, 2006), dont le chapitre 1 présente les hypothèses et l’histoire du domaine ; Z.-H. Zhou, « A brief introduction to weakly supervised learning » (*National Science Review*, 2018, en accès libre) ; X. Zhu et Z. Ghahramani, « Learning from Labeled and Unlabeled Data with Label Propagation » (rapport CMU, 2002) ; X. Zhu, Z. Ghahramani et J. Lafferty, « Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions » (ICML 2003) ; D. Zhou et al., « Learning with Local and Global Consistency » (NIPS 2003) ; H. B. Barlow, « Unsupervised Learning » (*Neural Computation*, 1989).
- Semi-supervisé moderne : Q. Xie et al., « Self-training with Noisy Student improves ImageNet classification » (CVPR 2020) ; K. Sohn et al., « FixMatch » (NeurIPS 2020) ; la synthèse de J. E. van Engelen et H. H. Hoos, « A survey on semi-supervised learning » (*Machine Learning*, 2020, en accès libre).
- Supervision faible en pratique : A. Ratner et al., « Snorkel: Rapid Training Data Creation with Weak Supervision » (*Proceedings of the VLDB Endowment*, 2017).
- Auto-supervisé : X. Liu et al., « Self-supervised Learning: Generative or Contrastive » (prépublication 2020, *IEEE TKDE* 2023), source de la figure ci-dessus.
- Synthèses de blog, claires et sourcées : L. Weng, « Learning with not Enough Data Part 1: Semi-Supervised Learning » (2021) et « Self-Supervised Representation Learning » (2019).
- Documentation : scikit-learn, « Semi-supervised learning » (auto-apprentissage, propagation d’étiquettes, avec exemples).
- Pour le renforcement, avant le chapitre 11 : R. S. Sutton et A. G. Barto, *Reinforcement Learning: An Introduction* (2e édition, 2018, en ligne), chapitre 1.
