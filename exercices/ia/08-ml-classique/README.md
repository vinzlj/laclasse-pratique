# Exercice 08 : k-NN, arbres, SVM, k-means, ACP et t-SNE

| | |
|---|---|
| **Chapitre lié** | [08. Apprentissage automatique « classique »](https://laclasse.poneyponey.fr/ia/p04-statistique/08-ml-classique/) |
| **Exécution** | locale, CPU (numpy, puis scikit-learn ; aucun GPU) |
| **Commande** | `make ex NUM=08` (éditeur), `make ex NUM=08 MODE=run` (complet), `MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 43 à 47 s (processus entier, attente du verrou exclue), dont 40 à 44 s de calcul ; smoke : 10 à 14 s (M1 Pro, CPU, 14 et 15/10/2026 ; dernier run complet le 15/10 : 44 s, dont 42 s de calcul) |
| **Travail estimé** | ~2 h |
| **Seed** | 42 (`seed_everything(42)` ; `random_state=42` pour scikit-learn, 43 pour le jeu de test des lunes ; graines 0 à 49 pour l'étude des minima locaux de k-means) |
| **Données** | générées (`make_moons`, `make_blobs`) ou livrées avec scikit-learn (`load_breast_cancer`, `load_digits`) ; rien n'est téléchargé |

## Objectif

Programmer en numpy les briques du *machine learning* « classique » (k plus proches voisins, coupe d'arbre par l'impureté de Gini, k-means, ACP), vérifier qu'elles donnent exactement les résultats de scikit-learn, puis comparer neuf modèles sur les mêmes données, choisir des hyperparamètres par validation croisée et voir ce que t-SNE montre et ne montre pas.

## Ce que vous allez faire

1. **k-NN en numpy.** Deux lunes bruitées : 200 points d'entraînement, 1 000 de test (arrondis à 4 décimales : ce sont les points de la démo `KnnBoundary` du site). Écrire `voisins_tries` (distances au carré, tri stable) et `knn_predire` (vote majoritaire), tracer les erreurs d'entraînement et de test pour k = 1, 3, …, 199. Choisir k **sans regarder le test**, par validation croisée à 5 plis écrite à la main. Vérifier que `KNeighborsClassifier` donne les mêmes prédictions.
2. **Malédiction de la dimension.** Mesurer, pour 1 000 points uniformes dans [0, 1]^d, le rapport « distance au plus lointain / distance au plus proche » ; ajouter 2, 10, 50 variables de bruit aux lunes et regarder l'erreur du k-NN.
3. **Souche d'arbre de Gini en numpy.** Essayer toutes les coupes « x_j ≤ s » et garder celle qui minimise l'impureté de Gini pondérée des deux enfants ; comparer à `DecisionTreeClassifier(max_depth=1)`. Puis faire varier la profondeur de l'arbre et comparer à une forêt aléatoire.
4. **Neuf modèles scikit-learn.** Bayes naïf gaussien, régression logistique, k-NN, arbre de profondeur 4, forêt aléatoire, AdaBoost, *gradient boosting* (`HistGradientBoostingClassifier`), SVM linéaire et à noyau RBF : frontières sur les lunes, puis exactitude en validation croisée (5 plis stratifiés) sur le jeu médical *breast cancer* (569 patientes, 30 variables). Sur deux nuages séparables, mesurer la marge d'un SVM linéaire et compter ses vecteurs de support.
5. **k-means (algorithme de Lloyd) en numpy.** 600 points en 5 groupes ; suivre l'inertie à chaque itération, comparer à `KMeans` avec la même initialisation, relancer avec 50 initialisations au hasard (et k-means++). Puis k-means contre DBSCAN sur des lunes peu bruitées.
6. **Recherche d'hyperparamètres.** SVM RBF sur `digits` (1 797 images 8 × 8) : grille 5 × 6 contre 30 tirages log-uniformes de (C, γ), même budget, même validation croisée ; score sur un test mis de côté.
7. **ACP et t-SNE.** ACP par SVD en numpy contre `PCA` ; projection 2D de `digits` par ACP et par t-SNE (perplexités 5, 30, 100) ; mesurer la séparation par l'exactitude d'un 5-NN dans la carte 2D. Enfin, un piège : deux groupes dont l'un est 10 fois plus étalé.

## Résultat attendu

En mode complet, le notebook vérifie dix-neuf seuils et s'arrête avec une erreur si l'un d'eux n'est pas atteint. Valeurs obtenues le 14/10/2026 (seed 42) :

| Vérification | Seuil | Obtenu |
|---|---|---|
| k-NN numpy = scikit-learn | 0 prédiction différente | 0 sur 1 000, pour k = 1, 5, 11, 25, 99 |
| 1-NN : erreur d'entraînement | 0 % | 0 % |
| 1-NN : erreur de test | ≥ meilleur k + 3 points | 13,9 % (meilleur : 8,9 % à k = 35) |
| k = 199 : erreur de test | ≥ meilleur k + 10 points | 29,5 % |
| k choisi par validation croisée | erreur de test ≤ 12 % | k = 11, 11,3 % (validation : 7,0 %) |
| Dimension 1 000 : plus lointain / plus proche | ≤ 1,5 | 1,11 |
| Lunes + 50 variables de bruit | erreur ≥ +10 points | 11,3 % → 40,4 % |
| Souche de Gini numpy = scikit-learn | même variable, même seuil | x₂ ≤ 0,3810, Gini pondéré 0,2939 (gain 0,2061) |
| Arbre complet | 0 % en entraînement, test pire qu'à la meilleure profondeur | 0 % / 14,3 % (profondeur 2 : 10,7 %) |
| Forêt aléatoire (300 arbres) | test ≤ arbre complet | 11,6 % |
| *Breast cancer* : SVM RBF | ≥ 96 % en validation croisée | 97,7 % ± 1,6 % |
| *Breast cancer* : forêt et HGB | ≥ 94 % | 95,3 % et 96,7 % |
| k-means numpy = scikit-learn | centres identiques à 1e-6 | écart 3,6e-15 |
| k-means : inertie | décroissante | 23 519 → 2 362 en 9 itérations |
| Lunes peu bruitées | ARI DBSCAN ≥ 0,95, k-means ≤ 0,6 | 1,000 et 0,248 |
| *Digits* : meilleur SVM | ≥ 97 % sur le test | 99,1 % |
| ACP numpy = scikit-learn | variance expliquée à 1e-10 | écart 5,6e-17 |
| t-SNE contre ACP | 5-NN dans la carte ≥ ACP + 25 points | 97,6 % contre 60,3 % |
| t-SNE : groupe 10 fois plus étalé | rapport d'étalement < 3 dans la carte | 9,9 → 1,02 |

Autres valeurs affichées :

- erreurs du k-NN (entraînement / test) : k = 1 : 0 % / 13,9 % ; k = 5 : 7,0 % / 10,4 % ; k = 25 : 8,0 % / 9,1 % ; k = 51 : 12,0 % / 9,6 % ; k = 99 : 18,0 % / 16,8 % ; k = 199 : 29,0 % / 29,5 % ;
- dimension : rapport plus lointain / plus proche 611 (d = 1), 101 (d = 2), 3,22 (d = 10), 1,35 (d = 100), 1,11 (d = 1 000) ; erreur du 11-NN avec 0, 2, 10, 50 variables de bruit : 11,3 %, 15,9 %, 26,5 %, 40,4 % ; celle d'une forêt aléatoire de 300 arbres : 11,6 %, 11,6 %, 12,4 %, 12,1 % ;
- arbre : profondeur 1 : 18,7 % au test ; arbre complet : 12 niveaux, 24 feuilles ; arbres de la forêt pris un par un : 5,5 % d'erreur sur tout l'entraînement et 16,0 % au test en moyenne (de 10,9 % à 23,7 %) ;
- erreurs de test sur les lunes : Bayes naïf 13,9 %, régression logistique 14,1 %, 11-NN 10,0 % (variables standardisées, d'où l'écart avec les 11,3 % du k-NN numpy sur les variables brutes), arbre de profondeur 4 15,4 %, forêt 11,6 %, AdaBoost 13,4 %, HGB 12,0 %, SVM linéaire 13,8 %, SVM RBF 10,0 % ;
- *breast cancer* (5 plis) : Bayes naïf 93,9 %, logistique 97,4 %, 5-NN 96,3 %, arbre 92,3 %, forêt 95,3 %, AdaBoost 96,7 %, HGB 96,7 %, SVM linéaire 97,4 %, SVM RBF 97,7 % ;
- SVM linéaire à marge dure sur 80 points séparables : **3 vecteurs de support**, marge 2/‖w‖ = 1,123, min y·f(x) = 0,9997 ; SVM RBF sur les lunes : 77 vecteurs de support sur 200 ;
- k-means : sur 50 initialisations au hasard, 40 % finissent à plus de 1 % de la meilleure inertie (pire : 5 118) ; 20 % avec k-means++ ; ARI de la solution numpy : 0,860 ;
- recherche : grille (6 valeurs de γ essayées) et hasard (30 valeurs) atteignent tous deux 99,03 % en validation croisée et 99,11 % au test, environ 9 s chacune ;
- ACP : 2 composantes gardent 28,5 % de la variance (14,9 % + 13,6 %) ; il en faut 21 sur 64 pour 90 % ;
- t-SNE : 5-NN dans la carte 97,4 % (perplexité 5), 97,6 % (30), 96,5 % (100), 4 à 6 s chacune.

Figures produites : `figures/knn.png`, `figures/frontieres.png`, `figures/kmeans.png`, `figures/recherche.png`, `figures/acp-tsne.png`.

## Ce qu'il faut observer

- **Le k-NN n'apprend rien, il mémorise.** Avec k = 1, chaque point d'entraînement est son propre voisin : 0 % d'erreur d'entraînement, mais 13,9 % au test, car la frontière contourne chaque point bruité. Avec k = 199 (presque tout le jeu), elle devient une droite raide : 29,5 %. Entre les deux, la courbe de test dessine le U du compromis biais-variance du chapitre 00.
- **La validation croisée ne retrouve pas le meilleur k du test, et c'est normal.** Elle choisit k = 11 (11,3 % au test) alors que k = 35 ferait 8,9 %. Avec 200 points, l'estimation est bruitée ; choisir k en regardant le test donnerait un chiffre trop optimiste.
- **En grande dimension, « plus proche voisin » ne veut plus rien dire.** En dimension 1 000, le point le plus lointain n'est que 1,11 fois plus loin que le plus proche. Ajouter 50 variables de bruit fait passer le k-NN de 11 % à 40 % d'erreur : il ne sait pas quelles variables comptent. Une forêt aléatoire, qui choisit ses variables coupe par coupe, reste à 12 %.
- **Un arbre choisit ses coupes gloutonnement.** La première coupe (x₂ ≤ 0,38) fait passer l'impureté de Gini de 0,50 à 0,29 ; c'est exactement celle de scikit-learn. Laissé libre, l'arbre descend à 0 % d'erreur d'entraînement (24 feuilles) et remonte à 14,3 % au test ; moyenner 300 arbres décorrélés (forêt) ramène à 11,6 %, alors que ces arbres font 16,0 % en moyenne pris un par un.
- **Chaque famille a sa forme de frontière.** Droites pour les modèles linéaires (logistique, SVM linéaire) ; le Bayes naïf gaussien, qui estime une variance par classe, trace en général une courbe du second degré, ici presque droite ; escaliers pour les arbres et leurs ensembles, courbes lisses pour le k-NN et le noyau RBF. Sur les lunes, ce sont ces derniers qui gagnent ; sur *breast cancer*, presque tout le monde est entre 94 et 98 % et l'écart-type entre plis (1 à 3 points) est du même ordre que les écarts entre modèles.
- **Le SVM ne dépend que de quelques points.** Sur 80 points séparables, 3 vecteurs de support suffisent à fixer la frontière ; ils sont exactement sur la marge (y·f(x) = 1).
- **k-means descend toujours, mais pas toujours au bon endroit.** L'inertie baisse à chaque itération (23 519 → 2 362), mais 40 % des initialisations au hasard s'arrêtent dans un minimum local (deux groupes fusionnés, un groupe coupé en trois). Et k-means ne sait faire que des groupes convexes : il coupe les lunes en deux (ARI 0,25) quand DBSCAN, qui suit la densité, les retrouve (ARI 1,00).
- **Grille ou hasard ?** Avec le même budget, la grille n'essaie que 6 valeurs de γ, le hasard 30. Ici, les deux trouvent 99 % : la zone des bons réglages est large. L'avantage du hasard apparaît surtout quand certains hyperparamètres comptent peu.
- **t-SNE préserve les voisinages, pas les distances ni les tailles.** Il sépare les dix chiffres (97,6 % de 5-NN dans la carte, contre 60 % pour l'ACP), mais un groupe 10 fois plus étalé qu'un autre y apparaît de la même taille. Les distances entre groupes d'une carte t-SNE ne se lisent pas non plus.

## Questions de réflexion

1. Le 1-NN a 0 % d'erreur d'entraînement. Pourquoi ce chiffre ne dit-il rien de sa qualité ? Proposez une mesure sur les données d'entraînement seules qui aurait prévenu du problème.
2. Pourquoi faut-il standardiser les variables avant un k-NN ou un SVM, mais pas avant un arbre de décision ? (Indice : que fait une coupe « x_j ≤ s » si l'on multiplie x_j par 1 000 ?)
3. Une forêt aléatoire de 300 arbres complets a 0 % d'erreur d'entraînement, comme l'arbre complet seul, mais généralise mieux, alors que ses arbres pris un par un font moins bien au test que l'arbre complet. Qu'est-ce qui a changé ?
4. Vous devez classer des tableaux de 50 000 lignes et 40 colonnes hétérogènes (âges, montants, catégories). Quel modèle de ce notebook essayeriez-vous en premier, et lequel jamais ? Pourquoi ?

## Pour aller plus loin

- Remplacer la validation croisée maison par `GridSearchCV(KNeighborsClassifier(), {"n_neighbors": ks})` et comparer le k choisi.
- Pondérer les voisins par l'inverse de leur distance (`weights="distance"`) : que devient l'erreur d'entraînement pour tous les k ?
- Ajouter une troisième variable inutile (par exemple `degree` d'un noyau polynomial) à la recherche d'hyperparamètres : le tirage au hasard prend-il l'avantage sur la grille ?
- Essayer UMAP (paquet `umap-learn`, absent de l'environnement du parcours) sur `digits` et comparer à t-SNE.
