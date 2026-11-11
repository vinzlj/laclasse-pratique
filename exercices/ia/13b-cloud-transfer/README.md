# Exercice 13b : apprentissage par transfert (transfer learning) avec ResNet-50 (cloud)

| | |
|---|---|
| **Chapitre lié** | [13. Réseaux convolutifs (CNN)](https://laclasse.poneyponey.fr/ia/p06-deep-learning/13-cnn/) |
| **Exécution** | cloud : Google Colab gratuit, GPU T4 (secours : Kaggle, « GPU T4 x2 ») |
| **Commande** | ouvrir `notebook.ipynb` dans Colab, puis Exécution › Tout exécuter (pas de `make ex` : notebook cloud) |
| **Durée mesurée** | **non mesurée sur T4** (estimation : 25 à 35 min). Seule la version de test (64 images, 1 époque par expérience, `IA_TEST_LOCAL=1`) a tourné, sur M1 Pro (MPS), le 14/10/2026 : voir « Vérification locale » |
| **Travail estimé** | ~1 h |
| **Seed** | 42 (`seed_everything()` : `random`, NumPy, PyTorch, CUDA) |
| **Données** | Oxford-IIIT Pet (Parkhi et al., 2012), copie Hugging Face [`timm/oxford-iiit-pet`](https://huggingface.co/datasets/timm/oxford-iiit-pet) (CC BY-SA 4.0, non restreinte, ~790 Mo, téléchargée dans la session Colab) ; poids ImageNet de torchvision `ResNet50_Weights.IMAGENET1K_V2` (~98 Mo) |
| **Coût** | 0 € (aucune carte bancaire demandée) |

## Objectif

Mesurer ce que rapporte un réseau pré-entraîné sur ImageNet quand on n'a que 3 680 images étiquetées : classer les 37 races de chats et de chiens d'Oxford-IIIT Pet avec le même ResNet-50 entraîné de zéro, puis réutilisé tronc gelé (extraction de caractéristiques), puis en réglage fin (fine-tuning) de tout le réseau.

## Ce que vous allez faire

1. **A. De zéro** : ResNet-50 aux poids aléatoires, 23,6 M paramètres entraînés, 10 époques (AdamW, taux 10⁻³, cosinus).
2. **B. Tronc gelé** : poids ImageNet, tout gelé (BatchNorm comprises), nouvelle dernière couche 2 048 → 37 (75 813 paramètres), 5 époques.
3. **C. Réglage fin (fine-tuning)** : poids ImageNet, nouvelle tête, tout le réseau entraîné 5 époques, taux 10⁻⁴ pour le tronc et 10⁻³ pour la tête.
4. Comparer les exactitudes en test (3 669 images) époque par époque, puis regarder les cartes **Grad-CAM** du modèle réglé finement.

Augmentation (entraînement seulement), tirée image par image : `RandomResizedCrop(224, scale=(0,35 ; 1))` et retournement horizontal. Calcul en float16 (`torch.autocast` + `GradScaler`) : le T4 ne fait pas de bfloat16.

Fichiers produits (dans la session Colab, puis téléchargés par la dernière cellule) : `resultats_13b.json`, `courbes_13b.png`, `gradcam_13b.png`.

## Procédure pas à pas (Colab)

1. **Compte** : un compte Google suffit (Colab gratuit). Plan B Kaggle : compte Kaggle, vérification par téléphone probable pour activer GPU et Internet.
2. **Ouvrir le notebook** :
   - badge : [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinzlj/laclasse-pratique/blob/main/exercices/ia/13b-cloud-transfer/notebook.ipynb) : le notebook s'ouvre directement depuis le dépôt public ;
   - ou import manuel : Fichier › Importer un notebook, puis choisir `notebook.ipynb`.
3. **Activer le GPU** : Exécution › Modifier le type d'exécution › GPU T4 › Enregistrer. La première cellule affiche `nvidia-smi` et « CUDA disponible : True » ; sinon, elle l'écrit en clair.
4. **Token Hugging Face** : inutile (jeu de données et poids publics, aucun `push_to_hub`).
5. **Exécution › Tout exécuter**. La cellule `pip install datasets==5.1.0` peut demander de redémarrer la session (mise à jour de `pyarrow`) : Exécution › Redémarrer la session, puis Tout exécuter à nouveau. Durée : à mesurer (estimation 25 à 35 min, dont la moitié pour l'expérience A).
6. **Sauvegarder avant la fin de session** : la dernière cellule télécharge les trois fichiers. Une session gratuite dure 12 h au plus et s'arrête plus tôt en cas d'inactivité.
7. **Plan B Kaggle** : Create › New Notebook › File › Import Notebook (`notebook.ipynb`) ; Settings › Accelerator › **GPU T4 x2** (jamais P100) ; Settings › Internet › On ; Run All. Les fichiers restent dans `/kaggle/working` (onglet Output).
8. **Coût : 0 €.**

## Résultat attendu

Seuils vérifiés par la cellule « 9. Bilan et courbes » (`assert`) en version complète. **Ils n'ont pas été mesurés sur T4** : ce sont des seuils prudents, à confirmer au premier lancement (et à reporter ici avec la durée).

| Expérience | Seuil (exactitude en test, 37 classes) |
|---|---|
| A. De zéro, 10 époques | ≤ 50 % |
| B. Tronc gelé, 5 époques | ≥ 85 % |
| C. Réglage fin, 5 époques | ≥ 88 % |

Le hasard donne 1/37 ≈ 2,7 %.

## Vérification locale

- Validation `nbformat.validate` et compilation de chaque cellule (lignes `!` ignorées) : OK, 18 cellules.
- `datasets==5.1.0` vérifié sur PyPI : `requires_python >= 3.10` (Colab : Python 3.12) ; torch et torchvision non épinglés (déjà dans l'image Colab).
- Jeu `timm/oxford-iiit-pet` vérifié par l'API du Hub : `gated: false`, licence `cc-by-sa-4.0`, 3 680 + 3 669 images, colonnes `image`, `label` (37 noms), `image_id`, `label_cat_dog`.
- Exécution du code des cellules en version de test (`IA_TEST_LOCAL=1` : 64 images, 1 époque par expérience, MPS, sans float16) : OK le 14/10/2026, ≈ 42 s hors attente du verrou GPU (dont le téléchargement des poids ResNet-50, 98 Mo). Sorties : A 0,0 %, B 4,7 %, C 6,2 % sur 64 images de test après une époque sur 64 images : ces chiffres ne vérifient que le code, pas les seuils. Le chargement du jeu (3 680 + 3 669 images), les trois expériences, les courbes, Grad-CAM et l'écriture des trois fichiers fonctionnent.

## Ce qu'il faut observer

- **L'écart entre A et B/C** (attendu, à confirmer au premier lancement) : avec 3 680 images, un ResNet-50 parti de zéro reste loin derrière ; le même réseau avec les poids ImageNet dépasse largement ce score **dès la première époque**, même quand on n'entraîne que sa dernière couche.
- **Le tronc gelé apprend 300 fois moins de paramètres** (75 813 contre 23,6 M) et va plus vite par époque : les caractéristiques d'ImageNet (bords, textures, parties d'animaux) suffisent presque pour séparer des races qu'ImageNet contient en partie.
- **Le réglage fin gagne encore quelques points** (attendu, à confirmer au premier lancement) en ajustant tout le réseau, à condition d'un taux faible pour le tronc : avec le taux de la tête partout, on risque d'effacer ce qu'ImageNet a appris dans les premières itérations.
- **Grad-CAM** (attendu, à confirmer au premier lancement) : sur les races, les zones chaudes devraient tomber en général sur la tête de l'animal (museau, oreilles, motif du pelage), rarement sur le fond.

## Questions de réflexion

1. Pourquoi geler aussi les BatchNorm du tronc dans l'expérience B (`geler_bn=True`) ? Que se passerait-il si leurs statistiques étaient recalculées sur des lots de 64 photos d'animaux ?
2. Oxford-IIIT Pet recoupe en partie les classes d'ImageNet (plusieurs races de chiens et de chats y figurent). Le gain de l'apprentissage par transfert serait-il aussi grand sur des radiographies ou des images satellite ? Que faudrait-il changer ?
3. L'expérience A a 10 époques, B et C n'en ont que 5. Est-ce injuste pour A ? Que montrent les courbes ?
4. Combien d'images par classe faudrait-il, à votre avis, pour que l'entraînement de zéro rattrape le réglage fin ?

## Pour aller plus loin

- Remplacer ResNet-50 par `models.convnext_tiny(weights="IMAGENET1K_V1")` (ConvNeXt, 2022, le CNN conçu après les Vision Transformers) et comparer.
- Ne dégeler que `layer4` et `fc` : un compromis entre B et C.
- Exercice 13a : le même Grad-CAM écrit à la main, sur un petit CNN entraîné en local.
