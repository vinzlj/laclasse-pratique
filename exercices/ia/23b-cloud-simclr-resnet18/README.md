# Exercice 23b : SimCLR avec ResNet-18 sur CIFAR-10 complet (cloud)

| | |
|---|---|
| **Chapitre lié** | [23. Apprentissage contrastif](https://laclasse.poneyponey.fr/ia/p07-transformer/23-contrastif/) |
| **Exécution** | cloud : Google Colab gratuit, GPU T4 (secours : Kaggle, « GPU T4 x2 ») |
| **Commande** | ouvrir `notebook.ipynb` dans Colab, puis Exécution › Tout exécuter (pas de `make ex` : notebook cloud) |
| **Durée mesurée** | **non mesurée sur T4**. Estimation : 1 h à 1 h 30 pour 100 époques (la cellule 7 affiche la durée prévue dès la première époque ; `EPOQUES = 50` divise le temps par deux). Seule la version de test (`IA_TEST_LOCAL=1`, 2 048 images, 2 époques) a tourné, sur M1 Pro (MPS) : voir « Vérification locale » |
| **Travail estimé** | ~1 h (plus le temps de calcul) |
| **Seed** | 42 (`seed_everything()` : `random`, NumPy, PyTorch, CUDA ; ordre des lots tiré par `torch.Generator().manual_seed(42)`) |
| **Données** | CIFAR-10 (Krizhevsky, 2009), copie Hugging Face [`uoft-cs/cifar10`](https://huggingface.co/datasets/uoft-cs/cifar10) (deux fichiers parquet, ~120 Mo, jeu non restreint, téléchargé dans la session) ; secours : `torchvision.datasets.CIFAR10` (serveur de l'Université de Toronto, ~170 Mo, parfois très lent) |
| **Coût** | 0 € (aucune carte bancaire demandée) |

## Objectif

Refaire l'exercice 23a à l'échelle où SimCLR commence à donner sa mesure : un ResNet-18 (11,2 millions de paramètres) entraîné sans étiquette sur les 50 000 images de CIFAR-10 pendant 100 époques, puis évalué par une sonde linéaire, un kNN et une sonde à 500 étiquettes, et comparé au même ResNet-18 aux poids aléatoires.

## Ce que vous allez faire

1. Vérifier le GPU (cellule 1), charger CIFAR-10 **directement sur le GPU** en octets (150 Mo).
2. Regarder l'augmentation de SimCLR recalculée sur le GPU (cellule 4) : recadrage redimensionné, retournement, distorsion de couleur, niveaux de gris, avec des **paramètres tirés indépendamment pour chaque image**. Pourquoi sur le GPU : Colab gratuit n'a que 2 cœurs de CPU, et 100 000 vues augmentées par époque y feraient un goulot.
3. Mesurer la référence : sonde linéaire et kNN sur un ResNet-18 aux poids aléatoires (cellule 6).
4. Entraîner SimCLR 100 époques (cellule 7) : lots de 512 images, donc 1 024 vues et 1 023 candidates par vue, perte NT-Xent (τ = 0,5), AdamW, taux « one cycle » de maximum 10⁻³, float16. Un kNN de suivi tourne toutes les 10 époques ; un point de reprise (`simclr_23b.pt`) est écrit à chaque époque.
5. Évaluer (cellule 8) : sonde linéaire sur les 50 000 étiquettes, kNN (k = 20), sonde avec 500 étiquettes seulement.
6. Variante : `CONFIG["PERTE"] = "barlow"` (Barlow Twins, sans négatifs, λ = 5·10⁻³), tout le reste identique.

**Perte optimisée** : InfoNCE dans sa version SimCLR (NT-Xent), ou Barlow Twins dans la variante. Les étiquettes ne servent qu'aux évaluations.

Fichiers produits (dans la session, puis téléchargés par la dernière cellule) : `resultats_23b.json`, `courbes_23b.png`, `simclr_23b.pt` (poids et point de reprise, ~90 Mo).

## Procédure pas à pas (Colab)

1. **Compte** : un compte Google suffit (Colab gratuit). Plan B Kaggle : compte Kaggle, vérification par téléphone probable pour activer GPU et Internet.
2. **Ouvrir le notebook** : dans Colab, Fichier › Importer un notebook, puis choisir `notebook.ipynb` de ce dossier.
3. **Activer le GPU** : Exécution › Modifier le type d'exécution › GPU T4 › Enregistrer. La première cellule affiche `nvidia-smi` et « CUDA disponible : True » ; sans GPU, elle s'arrête avec un message clair.
4. **Token Hugging Face** : inutile (le jeu `uoft-cs/cifar10` est public et non restreint ; aucun `push_to_hub`).
5. **Exécution › Tout exécuter**. Rien à installer : torch, torchvision, numpy, matplotlib, pyarrow et huggingface_hub sont déjà dans l'image Colab (torch n'est volontairement pas épinglé). Durée : à mesurer ; la cellule 7 affiche la durée totale prévue après la première époque. Si elle dépasse ce que vous voulez y consacrer, arrêtez, mettez `EPOQUES=50` dans la cellule 2 et relancez.
6. **Sauvegarder avant la fin de session** : la dernière cellule télécharge les trois fichiers. Une session gratuite dure 12 h au plus et s'arrête plus tôt en cas d'inactivité. En cas de coupure, relancer la cellule 7 reprend à la dernière époque **si** `simclr_23b.pt` est encore là (copiez-le sur Google Drive pour qu'il survive à une déconnexion).
7. **Plan B Kaggle** : Create › New Notebook › File › Import Notebook (`notebook.ipynb`) ; Settings › Accelerator › **GPU T4 x2** (jamais P100) ; Settings › Internet › On ; Run All. Le notebook n'utilise qu'un des deux GPU. Les fichiers restent dans `/kaggle/working` (onglet Output).
8. **Coût : 0 €.**

## Résultat attendu

Seuils vérifiés par la cellule « 9. Bilan » (`assert`) en version complète. **Ils n'ont pas été mesurés sur T4** : ce sont des seuils prudents, à confirmer au premier lancement (et à reporter ici avec la durée).

| Mesure | Seuil |
|---|---|
| Sonde linéaire SimCLR, 10 000 images de test | ≥ 75 % |
| Gain sur le ResNet-18 au hasard | ≥ 25 points |
| Écart-type de z/‖z‖ × √d (pas d'effondrement) | ≥ 0,5 |

Repères : l'exercice 23a (petit CNN, 25 000 images, 15 époques) donne la valeur locale ; l'article SimCLR rapporte 94,0 % en sonde linéaire sur CIFAR-10 avec un ResNet-50 (adapté de la même façon), des lots de 1 024 et un entraînement beaucoup plus long (Chen et al., 2020, annexe B.9). Le hasard donne 10 %.

## Vérification locale

- Validation `nbformat.validate` et compilation de chaque cellule (lignes `!` ignorées) : OK, 11 cellules.
- Dépendances : aucune installation (torch, torchvision, numpy, matplotlib, pyarrow, huggingface_hub, Pillow préinstallés dans Colab). Jeu `uoft-cs/cifar10` vérifié par l'API du Hub le 15/10/2026 : `gated: false`, fichiers `plain_text/train-00000-of-00001.parquet` et `plain_text/test-00000-of-00001.parquet` (même source que `ia_commun` pour l'exercice 23a). Le secours torchvision répond (redirection vers `cave.cs.toronto.edu`, 170 498 071 octets), mais n'a débité que ~68 ko/s lors d'un essai le 15/10/2026 (≈ 40 min) : c'est pourquoi le Hub passe en premier.
- Exécution du code de toutes les cellules en version de test (`IA_TEST_LOCAL=1` : 2 048 images d'entraînement, 1 000 de test, lots de 256, 2 époques, MPS, sans float16) : OK le 15/10/2026, ≈ 35 s hors attente du verrou GPU (jeu déjà en cache). Sorties : ResNet-18 au hasard, sonde 29,0 % ; après 2 époques, perte 5,90 → 5,50, écart-type × √d 0,86, sonde 34,8 %, sonde à 500 étiquettes 30,3 % (sur 1 000 images de test). Ces chiffres ne vérifient que le code, pas les seuils ; ils ont été obtenus avant que la graine de la sonde linéaire passe de 0 à 42 (convention du projet), et peuvent donc différer légèrement d'un nouveau lancement. Chargement du jeu depuis le Hub, augmentation sur le GPU, entraînement avec point de reprise, kNN, sondes, courbes et écriture des trois fichiers fonctionnent. La variante `PERTE = "barlow"` utilise la même fonction que l'exercice 23a (testée là), pas relancée ici.
- Contrôle visuel de l'augmentation sur le GPU (même code, exécuté sur CPU) : recadrages, retournements, distorsions de couleur et passages en gris variés d'une image à l'autre et d'un tirage à l'autre, comme ceux de torchvision.

## Ce qu'il faut observer

- **Le suivi kNN monte sans qu'aucune étiquette ne soit utilisée** pendant l'entraînement : la tâche « retrouver la jumelle parmi 1 023 vues » suffit à regrouper les classes.
- **L'écart avec la version locale** (exercice 23a) : plus de données, plus d'époques, un encodeur plus profond. Lequel des trois compte le plus ? (voir les questions).
- **L'indicateur d'effondrement reste loin de 0** avec InfoNCE comme avec Barlow Twins : les négatifs (InfoNCE) ou la décorrélation des dimensions (Barlow Twins) suffisent à l'empêcher.
- **Avec 500 étiquettes**, la sonde sur l'encodeur SimCLR doit rester proche de la sonde à 50 000 étiquettes : la représentation a fait l'essentiel du travail.

## Questions de réflexion

1. La perte de SimCLR baisse lentement après les premières époques alors que le kNN continue de monter. Pourquoi la valeur de la perte est-elle un mauvais indicateur de la qualité de la représentation ?
2. Avec des lots de 512, la borne d'InfoNCE sur l'information mutuelle plafonne à log(1 023) ≈ 6,9 nats. Que changerait un lot de 4 096 ? Pourquoi MoCo a-t-il introduit une file de négatifs ?
3. Ici l'augmentation est recalculée sur le GPU avec trois simplifications (voir la cellule 4). Laquelle pourrait le plus changer le résultat, et comment le vérifier ?
4. Barlow Twins n'a pas de négatifs. Que deviennent ses performances si vous réduisez le lot à 128 ? Et celles de SimCLR ?

## Pour aller plus loin

- Remplacer la sonde linéaire par un réglage fin complet avec 1 % des étiquettes et comparer à un ResNet-18 supervisé sur les mêmes 500 images.
- Bibliothèque pour aller beaucoup plus loin : [solo-learn](https://github.com/vturrisi/solo-learn) (licence MIT ; SimCLR, MoCo, BYOL, Barlow Twins, VICReg… sur CIFAR et ImageNet, PyTorch Lightning).
