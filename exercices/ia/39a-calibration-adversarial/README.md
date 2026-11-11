# Exercice 39a : quand se fier à une prédiction ? Calibration, incertitude, hors distribution, exemples adversariaux

| | |
|---|---|
| **Chapitre lié** | [39. Fiabilité et sécurité](https://laclasse.poneyponey.fr/ia/p10-fiabilite/39-fiabilite-securite/) |
| **Exécution** | locale, MPS (partie A en NumPy ; parties B à F en PyTorch, possibles sur CPU mais environ 6 fois plus lentes) |
| **Commande** | `make ex NUM=39a` (éditeur), `make ex NUM=39a MODE=run` (complet), `MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 4 min 01 s à 5 min 55 s sur trois runs (dont 3 min 40 s à 5 min 23 s pour entraîner les trois CNN ; attaques 7 à 8 s) ; smoke : 11 à 13 s (M1 Pro, MPS, 16/10/2026, attente du verrou exclue ; l'écart entre les runs vient de la charge de la machine, les résultats sont identiques) |
| **Travail estimé** | ~2 h |
| **Seed** | 42 (`seed_everything(42)` ; les trois CNN de l'ensemble ont les graines 42, 43 et 44 ; découpage entraînement/calibration, bruit et départ de PGD tirés avec la graine 42) |
| **Données** | CIFAR-10 via `ia_commun.donnees.CIFAR10` (cache `data/cifar10/`, 176 Mo, ≈ 20 s au premier lancement) ; MNIST via `ia_commun.donnees.mnist_tenseurs` (cache `data/mnist/`) ; bruit généré |

## Objectif

Mesurer si les probabilités d'un classifieur d'images disent vraiment à quel point se fier à lui, les corriger (température, ensemble, prédiction conforme), voir où ces corrections cessent de marcher (décalage de distribution, entrées hors distribution), puis casser le modèle avec des perturbations invisibles (FGSM, PGD).

## Ce que vous allez faire

1. **A. Les outils en NumPy**, chacun vérifié contre une bibliothèque :
   - l'erreur de calibration attendue (ECE) et les intervalles du diagramme de fiabilité, d'abord sur dix prédictions calculables à la main ;
   - la NLL (égale à `F.cross_entropy`) ; l'ajustement de la température par recherche du nombre d'or, comparé à L-BFGS de PyTorch (sur des logits jouets dont la « vraie » température est 1,5) ;
   - l'AUROC par les rangs (Mann-Whitney), comparée à `sklearn.metrics.roc_auc_score` ;
   - la prédiction conforme « split » (seuil sur $1 - \hat p_y$, quantile corrigé $\lceil (n+1)(1-\alpha) \rceil / n$) ;
   - l'argument linéaire de Goodfellow et al. : une perturbation $\varepsilon\,\mathrm{signe}(w)$ déplace un score linéaire de $\varepsilon \sum_i |w_i|$, qui grandit avec la dimension.
2. **B. Trois CNN sur CIFAR-10** : le petit CNN de l'exercice 13a (290 282 paramètres), 8 époques sans augmentation, AdamW et taux « one cycle » (maximum 2·10⁻³), sur 45 000 images ; 5 000 images d'entraînement mises de côté forment le **jeu de calibration**. La normalisation est la première couche du modèle : les images restent en pixels dans [0, 1], ce qui rend $\varepsilon$ lisible (en niveaux de gris sur 255).
3. **C. Calibration** du modèle seul, après température, et de l'ensemble des trois modèles (moyenne des probabilités), sur le test et sur le test bruité (bruit gaussien σ = 0,1).
4. **D. Prédiction conforme** à α = 0,1 sur les probabilités après température. En toute rigueur, la fonction de score ne doit pas dépendre du jeu de calibration ; l'exercice y ajuste aussi T (un seul paramètre, effet négligeable). Pour une garantie exacte, couper ce jeu en deux (une moitié pour T, l'autre pour le seuil).
5. **E. Hors distribution** : probabilité maximale (MSP) sur 2 000 chiffres MNIST (28×28 bordés à 32×32, gris recopié sur 3 canaux) et 2 000 images de bruit uniforme ; AUROC contre le test CIFAR-10.
6. **F. Exemples adversariaux** sur 1 000 images de test : FGSM (un pas) et PGD (départ aléatoire, 10 pas de taille ε/4, projection), ε ∈ {1, 2, 4, 8}/255 ; les mêmes images testées sur les deux autres modèles (transfert).

**Pertes** : l'entraînement minimise l'entropie croisée (= NLL) ; la température minimise la même NLL sur le jeu de calibration, avec un seul paramètre ; l'ECE est une **métrique**, pas une perte ; FGSM et PGD **maximisent** l'entropie croisée par rapport à l'image, les poids restant fixes.

## Résultat attendu

En mode complet, le notebook vérifie dix-sept seuils et s'arrête avec une erreur si l'un d'eux n'est pas atteint. Valeurs obtenues le 16/10/2026 (M1 Pro, MPS, seed 42 ; trois runs identiques) :

**A. Outils** : ECE de l'exemple à la main = 6/10 × |0,667 − 0,888| + 4/10 × |0,500 − 0,625| = **0,183** ; NLL NumPy − `F.cross_entropy` : 2,2·10⁻¹⁶ ; température jouet retrouvée : 1,5669 par le nombre d'or comme par L-BFGS (seuil : écart < 10⁻³) ; AUROC NumPy − scikit-learn : 0. Argument linéaire, ε = 2/255 et tous les poids à 0,04 (déplacement ε × n × 0,04, calculable à la main) : le score bouge de 0,003 pour 10 valeurs, 0,031 pour 100, **0,96 pour 3 072** (une image CIFAR-10) et **47** pour 150 528 (une image 224×224 d'ImageNet).

**B et C. Calibration** (10 000 images de test ; exactitude d'entraînement en fin de run : 91,5 à 92,1 %) :

| Jeu | Méthode | Exactitude | Confiance moyenne | **ECE** (15 intervalles) | NLL |
|---|---|---|---|---|---|
| test | modèle seul | 83,9 % (seuil ≥ 80 %) | 85,7 % | 1,89 % | 0,465 |
| test | température T = 1,095 | 83,9 % | 84,2 % | **0,80 %** (seuil ≤ 1,5 % et ≤ moitié du modèle seul) | 0,462 |
| test | ensemble de 3 | **86,0 %** (seuil ≥ seul + 1,5 point) | 83,8 % | 2,47 % | 0,415 |
| test bruité | modèle seul | 42,4 % | 72,5 % | **30,2 %** (seuil ≥ 20 %) | 2,172 |
| test bruité | température | 42,4 % | 70,2 % | 27,8 % (seuil ≥ 15 %) | 2,052 |
| test bruité | ensemble de 3 | 45,5 % | 67,4 % | 21,9 % (seuil : < modèle seul) | 1,835 |

**D. Prédiction conforme** (α = 0,1, 5 000 images de calibration) : seuil $\hat q$ = 0,765, soit garder toute classe de probabilité ≥ 0,235. Couverture sur le test **89,5 %** (seuil : entre 89 et 91 %) ; taille moyenne de l'ensemble 1,17 classe (1,10 quand l'argmax est juste, 1,53 quand il est faux) ; répartition : 9 ensembles vides, 8 372 d'une classe, 1 555 de deux, 64 de trois. Sur le test bruité, avec le même seuil : couverture **50,8 %**.

**E. Hors distribution** (modèle seul après température / ensemble) :

| Jeu | MSP moyenne | Confiance ≥ 90 % | AUROC contre CIFAR-10 | Classe la plus prédite |
|---|---|---|---|---|
| CIFAR-10 (test) | 0,842 / 0,838 | 57,8 % / 56,9 % | — | — |
| MNIST | 0,697 / 0,699 | 23,3 % / 22,8 % | **0,73** (seuil ≥ 0,65) / 0,72 | oiseau (65 % / 72 % des chiffres) |
| bruit uniforme | **0,863** / 0,780 | **49,0 %** (seuil ≥ 30 %) / 12,1 % | **0,58** / 0,68 | grenouille (98,8 % / 99,8 %) |

**F. Exemples adversariaux** (1 000 images de test, exactitude du modèle attaqué ; entre parenthèses, les mêmes images sur un autre modèle de l'ensemble) :

| ε (sur 255) | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| FGSM | 84,0 % | 46,0 % (73,6 %) | 23,7 % (62,0 %) | 6,4 % (43,1 %) | **2,8 %** (23,9 %) (seuil ≤ 25 %) |
| PGD, 10 pas | 84,0 % | 38,6 % (73,2 %) | 10,2 % (57,6 %) | 0,2 % (31,5 %) | **0,0 %** (6,9 %) (seuil ≤ 2 %) |
| perte (PGD) | 0,45 | 2,22 | 5,51 | 12,33 | 21,29 |
| confiance moyenne (PGD) | 85,7 % | 80,5 % | 89,6 % | 99,0 % | 99,9 % |

Seuil supplémentaire : PGD fait au moins aussi bien que FGSM à chaque ε. Exemple (image de test n° 0) : « chat » à 98,6 %, puis, après PGD à ε = 4/255, « bateau » à 99,99 %. Durée des attaques : 7 à 8 s.

Figures produites dans `figures/` : `fiabilite.png`, `hors-distribution.png`, `exemple-adversarial.png`, `attaques.png`, et `resultats.npz` (résultats bruts, relus par les scripts `site/scripts/figures/ch39_*.py` pour les figures du chapitre).

## Ce qu'il faut observer

- **La température ne change aucune décision.** L'exactitude reste 83,9 % ; seule la confiance baisse (85,7 → 84,2 %) et l'ECE passe de 1,9 à 0,8 %. Un seul nombre, appris sur 5 000 images mises de côté.
- **Ce CNN n'est que légèrement trop sûr de lui** (T = 1,095, ECE 1,9 %). Guo et al. (2017, table 1, 15 intervalles) mesurent sur CIFAR-10 une ECE de 4,6 % pour un ResNet-110 (0,83 % après température) et de 3,0 % pour LeNet-5 ; sur CIFAR-100, 16,5 % pour le ResNet-110. Ils relient ce défaut à la capacité des réseaux et au surapprentissage de la NLL. Ici, 8 époques laissent l'exactitude d'entraînement à 92 % : le modèle n'a pas eu le temps de surapprendre, et le défaut reste petit.
- **La calibration ne survit pas au décalage de distribution.** Un bruit à peine visible (σ = 0,1) divise l'exactitude par deux (42 %) alors que la confiance reste à 72 % : ECE 30 %. La température apprise sur des images propres ne répare presque rien (28 %) ; l'ensemble aide (22 %) sans résoudre. La garantie de la prédiction conforme tombe aussi (50,8 % de couverture au lieu de 90 %) : elle suppose des données échangeables.
- **L'ensemble gagne 2 points d'exactitude** et la meilleure NLL, mais sur le test propre il est un peu **sous-confiant** (confiance 83,8 % pour 86,0 % de réussite) : moyenner des probabilités les aplatit.
- **Les ensembles conformes sont petits quand le modèle a raison** (1,10 classe) et plus grands quand il se trompe (1,53) : la taille de l'ensemble est un signal d'incertitude utilisable.
- **Le modèle n'a pas de case « je ne sais pas ».** Du bruit pur reçoit une confiance moyenne de 86 %, presque toujours pour « grenouille » ; la MSP sépare à peine le bruit des vraies images (AUROC 0,58). Les chiffres MNIST sont mieux reconnus comme étrangers (0,73) mais 23 % reçoivent quand même plus de 90 %.
- **Deux niveaux de gris suffisent.** À ε = 2/255, PGD fait tomber l'exactitude de 84 à 10 % ; à 4/255, à 0,2 %, avec une confiance moyenne de 99 % **dans la mauvaise réponse**. La perte, que l'attaque maximise, passe de 0,45 à 21. PGD (10 pas) est toujours plus fort que FGSM (1 pas).
- **Les attaques se transfèrent.** Fabriquées sur le modèle 1, les images PGD à 8/255 font tomber les modèles 2 et 3, entraînés avec d'autres graines, à 7 %. C'est ce qui rend possibles des attaques « en boîte noire », sans accès aux poids.

## Questions de réflexion

1. La température ne change pas l'argmax. Pourquoi ? Et pourquoi la choisir sur un jeu de calibration plutôt que sur le jeu d'entraînement ?
2. L'ECE de l'ensemble (2,5 %) est moins bonne que celle du modèle seul (1,9 %), mais sa NLL est meilleure (0,415 contre 0,465). Comment deux mesures de « qualité des probabilités » peuvent-elles se contredire ? Laquelle est une règle de score propre ?
3. La prédiction conforme garantit 90 % de couverture « en moyenne ». Sur le test, on obtient 89,5 % : est-ce une violation ? Que garantit-elle exactement, et sur quoi ne dit-elle rien (par exemple, la couverture classe par classe) ?
4. D'après l'argument linéaire, pourquoi les images 224×224 d'ImageNet sont-elles encore plus vulnérables que celles de CIFAR-10 ? Pourquoi un réseau avec des ReLU se comporte-t-il « presque linéairement » vis-à-vis de ces perturbations ?

## Pour aller plus loin

- Entraîner plus longtemps (30 époques, sans weight decay) et regarder l'ECE monter : c'est le régime étudié par Guo et al.
- Ajouter du *dropout* avant la dernière couche et comparer le **dropout Monte-Carlo** (MC dropout, Gal et Ghahramani, ICML 2016, prépublication 2015 : plusieurs passes avec le dropout actif) à l'ensemble.
- Entraînement adversarial (Madry et al.) : remplacer chaque lot par sa version PGD à ε = 4/255. L'exactitude sous attaque remonte, l'exactitude propre baisse, et l'entraînement coûte environ 10 fois plus cher.
- Ensembles conformes adaptatifs (APS) et couverture par classe : Angelopoulos et Bates, *A Gentle Introduction to Conformal Prediction* (arXiv:2107.07511).
- Calibration sous décalage de distribution : Ovadia et al., *Can You Trust Your Model's Uncertainty?* (arXiv:1906.02530).
