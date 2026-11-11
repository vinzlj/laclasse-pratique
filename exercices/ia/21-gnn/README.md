# Exercice 21 : un GCN en numpy sur le club de karaté de Zachary

| | |
|---|---|
| **Chapitre lié** | [21. Réseaux de neurones sur graphes (GNN)](https://laclasse.poneyponey.fr/ia/p07-transformer/21-gnn/) |
| **Exécution** | locale, CPU (numpy, puis PyTorch en float64 sur CPU ; 34 nœuds : le GPU n'apporterait rien) |
| **Commande** | `make ex NUM=21` (éditeur), `make ex NUM=21 MODE=run` (complet), `MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 24 s (processus entier, attente du verrou exclue), dont 21 s de calcul ; smoke : 7 s (M1 Pro, CPU, 15/10/2026) |
| **Travail estimé** | ~1 h 30 |
| **Seed** | 42 (`seed_everything(42)`, `np.random.default_rng(42)` pour le GCN numpy et les tirages ; graines 0 à 4 pour le GAT et l'étude de profondeur, 0 à 49 pour les GCN à poids aléatoires). Calcul en float64 sur CPU : résultats identiques d'un lancement à l'autre |
| **Données** | graphe du club de karaté de Zachary (1977), fourni par `networkx` (`nx.karate_club_graph()`, groupe uv `misc`) : rien n'est téléchargé |

## Objectif

Classer les 34 membres d'un club de karaté entre les deux camps de la scission en ne connaissant le camp que de **deux** personnes (le professeur, nœud 0, et le président, nœud 33), grâce au graphe de leurs amitiés. Écrire pour cela un GCN (Kipf et Welling) en numpy, gradient compris, puis en PyTorch ; le comparer à un réseau qui ignore le graphe, à une propagation sans apprentissage et à une couche d'attention sur graphe (GAT) ; vérifier que l'attention du Transformer est une attention sur le graphe complet ; voir pourquoi on n'empile pas 30 couches de GCN.

## Ce que vous allez faire

1. **Le graphe.** Matrice d'adjacence A (34 × 34, 78 arêtes, poids des arêtes ignorés), camps réels (17 et 17), aucune caractéristique de nœud : X = I. Construire Â = D̃^(−1/2) (A + I) D̃^(−1/2).
2. **Une étape de passage de messages à la main.** Signal +1 sur le nœud 0, −1 sur le nœud 33, 0 ailleurs ; calculer h' = Â h nombre par nombre pour les nœuds 11, 8 et 2 (ex. nœud 11, un seul ami, le nœud 0 : h'₁₁ = 1/√(2 × 17) = +0,1715 ; nœud 8, ami des deux : 1/√(6 × 17) − 1/√(6 × 18) = 0,0990 − 0,0962 = +0,0028).
3. **Propagation sans apprentissage** : signe de Â^k h pour k = 1 à 8.
4. **GCN à deux couches en numpy** : Z = Â ReLU(Â X W₁) W₂, softmax, entropie croisée sur les deux nœuds étiquetés, gradient écrit à la main, Adam (pas 0,01, L2 5·10⁻⁴, 16 neurones cachés, 300 itérations : les valeurs de l'article, qui n'applique toutefois la L2 qu'à la première couche, quand l'exercice l'applique aux deux). Contrôle du gradient par l'autograd de PyTorch. Même code avec Â remplacée par I : un MLP qui ignore le graphe.
5. **Le même GCN en PyTorch** : Â en matrice creuse (`torch.sparse.mm`), autograd, `torch.optim.Adam`, même initialisation : comparer les poids finaux à ceux de numpy.
6. **Robustesse** : 50 tirages d'une personne étiquetée par camp ; GCN, MLP et propagation (k = 4).
7. **GCN à poids aléatoires** (annexe A de Kipf et Welling) : trois couches tanh de dimensions 4, 4, 2, sans entraînement, 50 tirages ; mesurer si les deux camps sont linéairement séparables dans le plan obtenu, avec et sans Â.
8. **GAT** en PyTorch : 4 têtes de 8 dimensions puis une tête de sortie, softmax sur les seuls voisins ; 5 graines ; regarder les poids d'attention appris par le nœud 0.
9. **Transformer = attention sur le graphe complet** : une attention par produit scalaire masquée par un graphe ; avec le graphe complet, comparer à `F.scaled_dot_product_attention`.
10. **Profondeur** : cosinus moyen entre les lignes de Â^k X (k = 1 à 64) ; GCN de 1 à 32 couches, sans et avec connexions résiduelles (H' = ReLU(Â H W) + H, variante de l'annexe B de l'article), 5 graines.

Ce qui est optimisé : l'entropie croisée sur les nœuds étiquetés seulement (apprentissage semi-supervisé), plus une pénalité L2 sur les poids. La propagation des parties 2, 3 et le GCN à poids aléatoires n'optimisent rien.

## Résultat attendu

En mode complet, le notebook vérifie dix seuils et s'arrête avec une erreur si l'un d'eux n'est pas atteint. Valeurs obtenues le 15/10/2026 (seed 42) :

| Seuil | Valeur obtenue |
|---|---|
| gradient numpy = autograd (écart < 10⁻¹⁰) | 3,5 × 10⁻¹⁸ |
| GCN numpy ≥ 94 % (32/34) | **97,1 %** (33/34 ; seul le nœud 8 est mal classé) |
| MLP sans graphe ≤ 65 % | **47,1 %** (16/34) |
| PyTorch = numpy (poids à moins de 10⁻⁸ après 300 itérations) | 8,9 × 10⁻¹⁶ ; précision 97,1 % |
| 50 tirages : GCN moyen ≥ MLP moyen + 25 points | 89,6 % contre 50,7 % |
| poids aléatoires : séparabilité GCN ≥ sans graphe + 10 points | 87,2 % contre 59,0 % |
| GAT ≥ 80 % en moyenne | 94,1 % (91,2 / 97,1 / 94,1 / 94,1 / 94,1) |
| Transformer = attention sur graphe complet (écart < 10⁻¹²) | 3,6 × 10⁻¹⁶ |
| cosinus moyen de Â³² X ≥ 0,99 | 0,999 |
| GCN à 16 couches ≤ 60 % et résiduel à 16 couches ≥ 85 % | 50,0 % contre 96,5 % |

Détail :

| Méthode (2 nœuds étiquetés : 0 et 33) | Précision sur les 34 nœuds |
|---|---|
| MLP sans graphe | 47,1 % (perte d'entraînement finale 0,007) |
| propagation sans apprentissage, k = 1 / 2 / 3 et plus | 88,2 % / 94,1 % / 97,1 % |
| GCN 2 couches (numpy ou PyTorch) | 97,1 % (perte finale 0,018) |
| GAT 2 couches, moyenne de 5 graines | 94,1 % |

| 50 paires étiquetées tirées au hasard | moyenne | médiane | min | max |
|---|---|---|---|---|
| GCN | 89,6 % | 94,1 % | 64,7 % | 100 % |
| propagation (k = 4) | 90,9 % | 94,1 % | 55,9 % | 97,1 % |
| MLP sans graphe | 50,7 % | 50,0 % | 41,2 % | 67,6 % |

| Couches | 1 | 2 | 3 | 4 | 6 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|---|
| GCN | 94,7 % | 97,1 % | 96,5 % | 96,5 % | 77,6 % | 60,0 % | 50,0 % | 50,0 % |
| GCN résiduel | 94,7 % | 97,1 % | 97,1 % | 97,1 % | 95,3 % | 96,5 % | 96,5 % | 80,0 % |

Cosinus moyen entre deux nœuds après k propagations : 0,14 (k = 1), 0,34 (2), 0,61 (4), 0,83 (8), 0,97 (16), 0,999 (32), 1,000 (64).

Figures produites : `figures/gcn-karate.png` (prédictions MLP et GCN sur le graphe, pertes, profondeur), `figures/aleatoire.png` (GCN à poids aléatoires) ; résultats bruts pour le site : `figures/resultats.npz` (lu par `site/scripts/figures/ch21_*.py`).

## Ce qu'il faut observer

- **Le MLP apprend parfaitement ses deux exemples (perte 0,007, plus basse que celle du GCN) et ne généralise pas du tout (47 %).** Avec X = I, la ligne i de W₁ est le plongement du nœud i ; l'entropie croisée n'envoie de gradient qu'aux lignes des deux nœuds étiquetés (2 lignes sur 34). Les 32 autres ne reçoivent que le gradient de la L2, qu'Adam transforme en pas de taille presque fixe : leur norme médiane passe de 0,79 à 1,1 × 10⁻⁷ en 300 itérations, et le camp prédit pour ces membres tient au bruit numérique. Avec Â, le gradient atteint les 34 lignes : le voisinage à deux sauts de 0 et 33 couvre tout le club.
- **Le GCN ne bat pas la propagation sans apprentissage sur ce graphe** (97,1 % pour les deux, et 89,6 % contre 90,9 % en moyenne sur 50 tirages). Sans caractéristiques de nœud, toute l'information est dans la structure, et Â^k h l'exploite déjà. L'intérêt du GCN apparaît quand les nœuds portent des caractéristiques (mots d'un article, atomes d'une molécule) que les poids W apprennent à combiner avec la structure.
- **Le nœud 8 est mal classé par le GCN** : il a 2 amis dans le camp du professeur et 3 dans celui du président (voisins 0, 2, 30, 32, 33), mais a suivi le professeur. Une moyenne des voisins ne peut pas le deviner ; une analyse plus fine de la structure (les « 2-clubs » de Laan, Marx et Mokken, 2016) y parvient.
- **Le GCN à poids aléatoires sépare déjà les camps** (87 % en moyenne contre 59 % sans graphe) : multiplier par Â rapproche les amis, quels que soient les poids.
- **L'attention du GAT n'est pas uniforme** : pour le nœud 0 (17 voisins en comptant lui-même), les poids appris par la première tête de la première couche (modèle de la graine 4) vont de 0,027 à 0,543 (uniforme : 0,059), alors que les poids fixes du GCN vont de 0,059 à 0,171 et ne dépendent que des degrés.
- **Le Transformer est une attention sur graphe complet** : 1 156 paires (i, j) contre 190 pour le graphe du club ; avec le masque complet, l'écart à `scaled_dot_product_attention` est de l'ordre de 10⁻¹⁶ (arrondi flottant).
- **Lissage excessif** : à force de moyenner avec ses voisins, chaque nœud ressemble à tous les autres (cosinus 0,999 à k = 32). Au-delà de 6 couches, le GCN simple n'arrive même plus à classer ses deux nœuds étiquetés (perte bloquée à ln 2 = 0,693 : les deux nœuds étiquetés restent à 50/50) ; les connexions résiduelles tiennent jusqu'à 16 couches. Kipf et Welling notaient déjà (annexe B) que les meilleurs résultats viennent de 2 ou 3 couches et qu'au-delà de 7 couches l'entraînement sans connexions résiduelles devient difficile.

## Questions de réflexion

1. Le MLP sans graphe atteint une perte d'entraînement plus basse que le GCN, et pourtant classe moitié moins bien. Que mesure la perte ici, et qu'est-ce qui manque au MLP pour généraliser ?
2. La propagation sans apprentissage fait aussi bien que le GCN sur ce graphe. Construisez un exemple (graphe et caractéristiques de nœuds) où le GCN ferait nettement mieux.
3. Pourquoi Â contient-elle A + I et pas seulement A ? Essayez sans la boucle sur chaque nœud et comparez.
4. Un Transformer de 12 couches traite un graphe complet : pourquoi ne souffre-t-il pas du lissage excessif comme un GCN de 12 couches ? (Pistes : connexions résiduelles, poids d'attention qui dépendent du contenu.)

## Pour aller plus loin

- Reprendre l'expérience avec les 4 communautés utilisées par Kipf et Welling (classement par modularité, `networkx.community.greedy_modularity_communities`) et un nœud étiqueté par communauté.
- Passer à un vrai jeu de données de classification de nœuds avec caractéristiques (Cora, 2 708 articles) avec PyTorch Geometric ; la bibliothèque n'est pas dans `pyproject.toml`.
