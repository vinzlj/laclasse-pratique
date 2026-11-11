# Exercice 27 : étiquettes aléatoires, double descente, régularisation implicite et grokking

| | |
|---|---|
| **Chapitre lié** | [27. Pourquoi le deep learning généralise : la partie empirique](https://laclasse.poneyponey.fr/ia/p08-llm/27-generalisation-empirique/) |
| **Exécution** | locale (MPS pour les parties A et D ; NumPy sur CPU pour B et C) |
| **Commande** | `make ex NUM=27` (éditeur), `make ex NUM=27 MODE=run` (complet), `make ex NUM=27 MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 6 min (5 min 59 s ; A 51 s, B 23 s, C 40 s, D 4 min 1 s soit 16,1 ms par pas, le reste en chargement ; trois runs entre 5 min 31 s et 5 min 59 s) ; smoke : 11 à 14 s (M1 Pro, MPS, 16/10/2026, attente du verrou GPU exclue) |
| **Travail estimé** | ~1 h 30 |
| **Seed** | 42 (`seed_everything(42)`, `np.random.default_rng(42)`, `torch.Generator().manual_seed(42)` pour les tirages de données) |
| **Données** | MNIST (`ia_commun.donnees.mnist_tenseurs`, cache `data/mnist/`, téléchargé au premier lancement) ; l'addition modulaire de la partie D est générée par le notebook |

## Objectif

Le chapitre 10 dit qu'un modèle trop riche pour ses données devrait surapprendre. Mesurer quatre faits qui contredisent cette intuition : un réseau mémorise des étiquettes tirées au hasard (donc sa capacité n'explique pas pourquoi il généralise sur les vraies) ; l'erreur de test d'un modèle à features aléatoires culmine au seuil d'interpolation puis redescend sous le meilleur modèle « classique » (double descente) ; la descente de gradient partie de zéro choisit, parmi une infinité de solutions parfaites, celle de norme minimale (régularisation implicite) ; un petit Transformer généralise l'addition modulaire des milliers de pas après l'avoir apprise par cœur (grokking).

Pertes : **entropie croisée** (A, et D avec un weight decay de 1 via AdamW) ; **erreur quadratique (MSE)** sur les étiquettes codées en one-hot (B : minimisée exactement par moindres carrés ; C : par descente de gradient).

## Ce que vous allez faire

**Partie A : mémoriser du bruit** (PyTorch, MPS ; expérience de Zhang et al., refaite en petit sur MNIST).

1. Entraîner un MLP 784-512-512-10 (669 706 paramètres) pendant 100 époques sur 5 000 images MNIST, avec Adam (lr 10⁻³, lots de 128), trois fois : vraies étiquettes ; étiquettes tirées uniformément au hasard ; images remplacées par du bruit gaussien (même moyenne et même écart-type que MNIST) avec les vraies étiquettes.
2. Remplacer 0, 20, 40, 60, 80 puis 100 % des étiquettes par du hasard ; 60 époques chaque fois ; mesurer l'exactitude sur les 10 000 images de test (vraies étiquettes).

**Partie B : double descente** (NumPy, « from scratch »).

3. Tirer une matrice gaussienne $W$ de taille 784 × 20 000 (moyenne 0, variance 1/784). Une image centrée $x$ devient $\phi(x) = \mathrm{ReLU}(W_{:,1..p}^\top x)/\sqrt p$ : un réseau à une couche cachée dont seule la dernière couche est apprise ($10p$ paramètres).
4. Pour $n$ = 1 000 images d'entraînement et $p$ de 50 à 20 000 (33 valeurs, serrées autour de 1 000) : résoudre les moindres carrés par `np.linalg.lstsq`, qui renvoie la solution de **norme minimale** $B = \Phi^+ Y$ quand $p \ge n$ ; mesurer la MSE d'entraînement et de test, l'erreur de classification de test (sur 10 000 images) et la norme $\lVert B\rVert$. Comparer avec ridge, $\lambda = 10^{-3}$ et $10^{-1}$ (on minimise $\lVert \Phi B - Y\rVert^2 + \lambda \lVert B\rVert^2$, sans facteur $1/n$ ; forme duale $\Phi^\top(\Phi\Phi^\top + \lambda I)^{-1}Y$ quand $p > n$).
5. Inverser les rôles : $p$ = 1 000 fixé, $n$ de 200 à 4 000.

**Partie C : régularisation implicite** (NumPy).

6. Avec $p$ = 4 000 et $n$ = 1 000, faire 1 500 pas de descente de gradient sur la MSE ($\eta = n/\sigma_{\max}^2$) depuis quatre points de départ : $B_0 = 0$, et $B_0$ gaussien d'écart-type $1/\sqrt p$ (échelle usuelle d'une couche linéaire), $100/\sqrt p$, $300/\sqrt p$.
7. Vérifier par la SVD de $\Phi$ que la composante de $B$ dans le **noyau** de $\Phi$ (les directions que les données ne voient pas) ne bouge pas, et calculer exactement la limite de la descente, $\Phi^+ Y + P_{\text{noyau}} B_0$ : sa norme et son erreur de test.

**Partie D : grokking** (PyTorch, MPS ; réglages de Nanda et al., 2023).

8. Construire les 113² = 12 769 séquences « $a$ $b$ = » de cible $(a + b) \bmod 113$ ; 30 % (3 830) pour l'entraînement, le reste (8 939) pour le test.
9. Entraîner un Transformer d'une couche (plongements de dimension 128, 4 têtes d'attention, MLP ReLU de 512, pas de LayerNorm, 226 688 paramètres) sur le lot complet, AdamW (lr 10⁻³, β = (0,9 ; 0,98), weight decay 1, 10 pas d'échauffement), 15 000 pas. Évaluer tous les 20 pas jusqu'au pas 1 000, puis tous les 100 pas : exactitudes, pertes, norme des poids, et exactitude de test séparée selon que la paire « miroir » $(b, a)$ est dans l'entraînement ou non.

Sorties : `figures/generalisation.png` (les quatre expériences) et `figures/resultats.npz` (toutes les courbes ; lu par les scripts `site/scripts/figures/ch27_*.py` qui produisent les figures du chapitre).

## Résultat attendu

Le notebook vérifie ces seuils en mode complet et s'arrête avec une erreur si l'un manque :

| Contrôle | Seuil | Obtenu (16/10/2026) |
|---|---|---|
| A. Vraies étiquettes : exactitude d'entraînement | 100 % | 100 % (atteint à l'époque 14) |
| A. Étiquettes aléatoires : exactitude d'entraînement | ≥ 99 % | 100 % (époque 56) |
| A. Pixels aléatoires : exactitude d'entraînement | ≥ 99 % | 100 % (époque 6) |
| A. Exactitude de test, vraies étiquettes | ≥ 90 % | 94,8 % |
| A. Exactitude de test, étiquettes aléatoires | ≤ 15 % | 9,3 % (hasard : 10 %) |
| B. Pic au seuil $p = n$ | MSE de test ≥ 10 × le minimum pour $p \le n/2$ | 117,0 contre 0,463 (× 253) |
| B. Interpolation | MSE d'entraînement ≤ 10⁻⁶ pour tout $p \ge n$ | ≈ 10⁻²⁷ |
| B. Le plus grand modèle bat le meilleur modèle classique | erreur de test à $p$ = 20 000 inférieure au minimum pour $p \le 500$ | 7,8 % contre 15,8 % |
| B. Ridge $\lambda = 10^{-3}$ efface le pic | erreur à $p = n$ ≤ 20 % | 11,2 % |
| C. Composante « noyau » de $B$ | déplacée de ≤ 10⁻⁸ | ≤ 3 × 10⁻¹² |
| C. Départ à l'échelle usuelle ($1/\sqrt p$) | même limite que $B_0 = 0$ (à 0,5 point près) | 9,0 % contre 9,1 % |
| C. Grand départ ($300/\sqrt p$) : norme | ≥ 2 × la norme minimale | 868,5 contre 268,1 |
| C. Grand départ : erreur de test | ≥ celle de $B_0 = 0$ + 5 points | 22,2 % contre 9,1 % |
| D. Mémorisation (entraînement ≥ 99 %) | avant le pas 2 000 | pas 260 |
| D. Généralisation (test ≥ 95 %) | atteinte en 15 000 pas | pas 12 000 |
| D. Plateau (au pas 5 000) | ≥ 50 % sur les paires « miroir », ≤ 5 % sur les autres | 94,7 % et 0,1 % |
| D. Retard du grokking | ≥ 5 000 pas entre mémorisation et généralisation | 11 740 pas |

Autres valeurs affichées :

- **Partie A** : exactitude de test selon la part d'étiquettes aléatoires : 94,7 % (0 %), 85,7 % (20 %), 73,3 % (40 %), 50,8 % (60 %), 30,1 % (80 %), 10,7 % (100 %), alors que l'exactitude d'entraînement reste entre 99,8 et 100 % ; perte d'entraînement finale : 3 × 10⁻⁵ (vraies étiquettes), 7 × 10⁻⁴ (étiquettes aléatoires, courbe plus lente et plus heurtée), 6 × 10⁻⁶ (pixels aléatoires).
- **Partie B**, $n$ = 1 000 : erreur de test 31,4 % ($p$ = 50), 15,8 % ($p$ = 300, minimum du régime classique), 31,0 % ($p$ = 800), 85,1 % ($p$ = 1 000), 42,1 % ($p$ = 1 100), 11,4 % ($p$ = 2 000), 9,1 % ($p$ = 4 000), 7,8 % ($p$ = 20 000). Norme de la solution : 26 ($p$ = 50), 111 ($p$ = 300), 461 ($p$ = 800), 5 394 ($p$ = 1 000), 238 ($p$ = 20 000). Ridge $\lambda = 10^{-3}$ : de 31,4 % à 7,7 % sans pic ; $\lambda = 10^{-1}$ : plafonne à 11,9 % (trop régularisé pour les grands $p$).
- **Partie B2**, $p$ = 1 000 : erreur de test 22,4 % ($n$ = 200), 17,6 % ($n$ = 400), 29,9 % ($n$ = 800), 85,1 % ($n$ = 1 000), 18,1 % ($n$ = 1 500), 7,6 % ($n$ = 4 000).
- **Partie C** : après 1 500 pas, la descente n'a pas convergé (distance relative à $\Phi^+Y$ : 0,69 depuis zéro), car $\Phi$ a des valeurs singulières de 3,3 à 0,011 ; arrêtée là, elle fait même mieux que sa limite (8,1 % contre 9,1 % d'erreur de test : l'arrêt précoce est une autre régularisation implicite). Départ $100/\sqrt p$ : limite de norme 384, erreur 10,3 %.
- **Partie D** : 29,4 % des paires de test ont leur miroir dans l'entraînement ; l'exactitude de test reste entre 25 et 31 % du pas 300 au pas 10 000 (entre 85 et 99 % sur les paires « miroir ») ; la norme des poids passe de 123 à environ 60 en 2 000 pas, oscille entre 50 et 64 pendant le plateau, puis tombe à 30 quand le test passe à 100 %. La perte d'entraînement oscille entre 10⁻⁷ et 0,4 (pics réguliers, avec de brèves chutes de l'exactitude d'entraînement, jusqu'à 97,9 % au pas 6 300).

Sur MPS, deux runs avec la même graine donnent les mêmes courbes à quelques décimales près, pas au bit près : le pas de généralisation de la partie D a varié de 11 900 à 12 400 selon les runs (12 000 pour le run rapporté ici). Les parties B et C (NumPy) sont exactement reproductibles.

## Ce qu'il faut observer

- **A.** Le même réseau, avec le même optimiseur, atteint 100 % sur les vraies étiquettes et sur du pur hasard. Une borne qui ne dépend que de la classe de modèles (dimension VC, chapitre 10) est la même dans les deux cas : elle ne peut pas expliquer pourquoi le premier généralise (94,8 %) et pas le second (9,3 %). La différence vient des données et de l'algorithme. Le hasard coûte quand même plus cher : 56 époques au lieu de 14, avec une courbe heurtée.
- **A, bruit progressif.** L'exactitude de test baisse de plus en plus vite avec la part d'étiquettes fausses (−9, −12, −22, −21, −19 points), alors que l'entraînement reste parfait : le réseau apprend la structure là où il y en a et apprend le reste par cœur. Ce par-cœur coûte : à 20 % d'étiquettes fausses, la bonne classe reste majoritaire pour chaque image, mais le test tombe de 94,7 à 85,7 %.
- **B.** Tant que $p < n$, on retrouve la courbe en U du chapitre 00. À $p = n$, il existe une seule solution qui passe par les 1 000 points, et elle n'y arrive qu'avec des coefficients énormes (norme × 49 par rapport à $p$ = 300) : c'est elle qui explose en test. Au-delà, il y a une infinité de solutions parfaites ; la pseudo-inverse prend la plus « lisse » (norme minimale), et plus $p$ grandit, plus cette solution de norme minimale est petite et bonne. Le plus grand modèle n'a jamais été aussi bon.
- **B, ridge.** Le pic n'est pas une fatalité : une petite pénalité $\lambda = 10^{-3}$ le supprime. La double descente est surtout visible quand rien ne contrôle la norme au seuil.
- **B2.** Près du seuil, ajouter des données fait **monter** l'erreur (17,6 % à 400 images, 85 % à 1 000) : la difficulté n'est pas « trop de paramètres » mais « juste assez pour tout interpoler ».
- **C.** La descente de gradient ne touche jamais la partie de $B$ que les données ne voient pas (déplacement de 10⁻¹² : l'arrondi de la machine). Elle ne choisit donc pas « une » solution au hasard : partie de zéro, elle va vers celle de norme minimale, sans pénalité explicite. Partie d'un point trop grand, elle garde ce bruit (norme 868, erreur 22 %). L'initialisation usuelle ($1/\sqrt p$) est assez petite pour que cela ne change rien.
- **D.** Trois temps : mémorisation en 260 pas ; long plateau où l'exactitude de test reste vers 28 % ; généralisation brutale vers le pas 11 000 à 12 000. Les 28 % du plateau ne sont pas de l'addition : le modèle a appris $a + b = b + a$ et réussit 95 % des paires dont le miroir est connu, 0 % des autres. Pendant le plateau, le weight decay comprime les poids ; la généralisation coïncide avec une chute de la norme de 60 à 30 : une solution plus simple que la table apprise par cœur.

## Questions de réflexion

1. En partie A, les étiquettes aléatoires demandent environ 4 fois plus d'époques que les vraies. Que dit cette différence de ce que le réseau apprend d'abord ? Pourquoi les pixels aléatoires sont-ils, eux, mémorisés le plus vite ?
2. En partie B, pourquoi le pic tombe-t-il exactement à $p = n$ ? Que se passe-t-il pour la matrice $\Phi^\top\Phi$ (ou $\Phi\Phi^\top$) à ce moment-là, et pourquoi un $\lambda$ minuscule suffit-il à tout arranger ?
3. La borne VC du chapitre 10 grandit avec le nombre de paramètres. Quelle grandeur, mesurée ici, suit mieux l'erreur de test que le nombre de paramètres au seuil et au-delà ? (Comparez $\lVert B \rVert$ et l'erreur de test en partie B ; pourquoi cela ne marche-t-il pas à gauche du seuil, où la norme grandit avec $p$ alors que l'erreur fait un U ?)
4. En partie D, que prévoyez-vous si l'on met `d_wd` à 0 ? Et avec 50 % des paires au lieu de 30 % ? (Essayez : `IA27_CONFIG='{"d_wd": 0}' make ex NUM=27 MODE=run` ; le run sera déclaré en échec si le seuil de généralisation n'est pas atteint, c'est attendu.)

## Pour aller plus loin

- Ajouter 20 % d'étiquettes bruitées en partie B (`b_n` inchangé) : le pic grandit-il ? La remontée vers $p = n$ commence-t-elle plus tôt ?
- Partie D sans weight decay ou avec une autre fraction (`IA27_CONFIG='{"d_fraction": 0.5}'`) : mesurer le pas de généralisation.
- Articles : Zhang et al., « Understanding deep learning requires rethinking generalization », arXiv:1611.03530 ; Belkin et al., « Reconciling modern machine learning practice and the bias-variance trade-off », arXiv:1812.11118 ; Nakkiran et al., « Deep Double Descent: Where Bigger Models and More Data Hurt », arXiv:1912.02292 ; Power et al., « Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets », arXiv:2201.02177 ; Nanda et al., « Progress measures for grokking via mechanistic interpretability », arXiv:2301.05217 (ils montrent que le modèle finit par calculer l'addition avec des rotations, par transformée de Fourier discrète).
