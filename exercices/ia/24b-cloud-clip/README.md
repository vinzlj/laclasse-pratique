# Exercice 24b : CLIP pré-entraîné, zero-shot puis réglage fin sur images satellite (cloud)

| | |
|---|---|
| **Chapitre lié** | [24. Vision moderne et multimodalité](https://laclasse.poneyponey.fr/ia/p07-transformer/24-vision-multimodalite/) |
| **Exécution** | cloud : Google Colab gratuit, GPU T4 (secours : Kaggle, « GPU T4 x2 ») |
| **Commande** | ouvrir `notebook.ipynb` dans Colab, puis Exécution › Tout exécuter (pas de `make ex` : notebook cloud) |
| **Durée mesurée** | **non mesurée sur T4** (estimation : 15 à 25 min, dont ≈ 5 min de téléchargement et d'encodage). Mesures locales sur M1 Pro (MPS) le 15/10/2026 : voir « Vérification locale » |
| **Travail estimé** | ~1 h |
| **Seed** | 42 (`seed_everything()` : `random`, NumPy, PyTorch, CUDA ; sous-ensembles et tirages few-shot à graine fixe) |
| **Données** | EuroSAT RGB (Helber et al., 2019), copie Hugging Face [`timm/eurosat-rgb`](https://huggingface.co/datasets/timm/eurosat-rgb) (licence MIT, non restreinte, ~92 Mo) ; 2 000 images de test de CIFAR-10 ([`uoft-cs/cifar10`](https://huggingface.co/datasets/uoft-cs/cifar10), ~24 Mo) ; poids CLIP [`laion/CLIP-ViT-B-32-laion2B-s34B-b79K`](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K) (licence MIT, non restreint, ~600 Mo). Tout est téléchargé dans la session Colab |
| **Coût** | 0 € (aucune carte bancaire demandée) |

## Objectif

Mesurer, sur un vrai CLIP (ViT-B/32 entraîné sur LAION-2B, 2 milliards de paires image-texte en anglais) et sur un domaine sans doute peu présent dans les légendes du web (des images satellite), ce que donne le **zero-shot** avec des phrases, combien d'images étiquetées il faut à une **sonde linéaire** pour faire mieux, ce qu'apporte le **réglage fin** (fine-tuning) de l'encodeur d'image, et ce qu'il coûte sur un autre jeu (CIFAR-10).

## Ce que vous allez faire

1. **Charger CLIP** avec `open_clip` : un ViT-B/32 (87,8 M paramètres, image 224×224 en 7×7 = 49 patchs de 32×32) et un Transformer de texte (63,4 M paramètres avec le reste), espace commun de dimension 512.
2. **A. Zero-shot** sur les 5 400 images de test d'EuroSAT (10 classes), avec quatre jeux de phrases : nom brut de la classe (`AnnualCrop`), un gabarit (`a centered satellite photo of annual crop land.`), les trois gabarits du dépôt officiel de CLIP pour EuroSAT (moyenne des plongements), et une phrase en français (`une photo satellite de cultures annuelles.`).
3. **B. Sonde linéaire** (régression logistique, chapitre 01) sur les plongements d'image figés, avec 1, 4 et 16 images étiquetées par classe (moyenne de 3 tirages), puis les 16 200 images d'entraînement.
4. **C. Réglage fin** de tout l'encodeur d'image, 2 époques, avec une tête 512 → 10 **initialisée par les phrases du zero-shot** (le modèle part exactement du score de A). AdamW, taux 10⁻⁵, échauffement puis cosinus, float16 (`torch.autocast` + `GradScaler`) : le T4 ne fait pas de bfloat16.
5. **D. Oubli** : zero-shot de CLIP sur 2 000 images de test de CIFAR-10 (`a photo of a cat.`…) avant et après le réglage fin sur EuroSAT.

Fichiers produits (dans la session Colab, puis téléchargés par la dernière cellule) : `resultats_24b.json`, `clip_24b.png`.

## Procédure pas à pas (Colab)

1. **Compte** : un compte Google suffit (Colab gratuit). Plan B Kaggle : compte Kaggle, vérification par téléphone probable pour activer GPU et Internet.
2. **Ouvrir le notebook** : télécharger le fichier `notebook.ipynb` de ce dossier, puis dans Colab, Fichier › Importer un notebook › Importer, et le choisir (pas de badge ni de lien direct : le dépôt du cours est privé).
3. **Activer le GPU** : Exécution › Modifier le type d'exécution › GPU T4 › Enregistrer. La première cellule affiche `nvidia-smi` et « CUDA disponible : True » ; sinon, elle l'écrit en clair.
4. **Token Hugging Face** : inutile (jeux de données et poids publics, aucun `push_to_hub`). L'avertissement « You are sending unauthenticated requests to the HF Hub » est sans conséquence.
5. **Exécution › Tout exécuter**. La cellule 2 installe `open_clip_torch==3.3.0` et `timm==1.0.30` (quelques secondes). Durée : à mesurer (estimation 15 à 25 min).
6. **Sauvegarder avant la fin de session** : la dernière cellule télécharge les deux fichiers. Une session gratuite dure 12 h au plus et s'arrête plus tôt en cas d'inactivité.
7. **Plan B Kaggle** : Create › New Notebook › File › Import Notebook (`notebook.ipynb`) ; Settings › Accelerator › **GPU T4 x2** (jamais P100) ; Settings › Internet › On ; Run All. Les fichiers restent dans `/kaggle/working` (onglet Output).
8. **Coût : 0 €.**

## Résultat attendu

Seuils vérifiés par la cellule « # 11. Bilan, figure et seuils » (`assert`) en version complète. **Ils n'ont pas été mesurés sur T4** ; ils sont fixés avec de la marge sous la mesure locale complète sur M1 Pro (47,3 %, 96,6 % et 98,0 % après une seule époque ; voir « Vérification locale »).

| Étape | Seuil (exactitude en test EuroSAT, 10 classes) |
|---|---|
| A. Zero-shot, meilleur des quatre jeux de phrases | ≥ 35 % |
| B. Sonde linéaire, 16 200 images | ≥ 90 % |
| C. Réglage fin, 2 époques | ≥ 95 % |

Le hasard donne 10 %.

## Vérification locale

- Validation `nbformat.validate` et compilation de chaque cellule (lignes `!` ignorées) : OK, 19 cellules.
- Dépendances vérifiées sur PyPI le 15/10/2026 : `open_clip_torch` 3.3.0 (`requires_python >= 3.9`, licence MIT), `timm` 1.0.30 (`>= 3.8`) ; Colab a Python 3.12. torch et torchvision non épinglés (déjà dans l'image Colab).
- Modèle et jeux vérifiés par l'API du Hub : `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` (`gated: false`, licence `mit`, bibliothèque `open_clip`) ; `timm/eurosat-rgb` (`gated: false`, licence `mit`, 16 200 / 5 400 / 5 400 images, colonnes `image`, `label`, `image_id`) ; `uoft-cs/cifar10` (`gated: false`, licence non renseignée sur le Hub).
- Exécution du code des cellules en version de test (`IA_TEST_LOCAL=1` : 320 images d'entraînement, 200 de test, 1 époque de réglage fin, MPS, sans float16) : OK le 15/10/2026, 29 s une fois les poids en cache.
- **Mesure complète en local** (M1 Pro, MPS, float32, 15/10/2026), même code avec trois écarts : **1 époque** de réglage fin au lieu de 2, lots de 64 au lieu de 128, DataLoader sans processus auxiliaires. Durée totale 8 min 12 s (encodage de 7 400 images en 37 s, des 16 200 images d'entraînement en 80 s, une époque de réglage fin en 305 s). Résultats sur les 5 400 images de test :

| Étape | Exactitude (test EuroSAT) |
|---|---|
| A. Zero-shot, nom brut (`AnnualCrop`) | 36,3 % |
| A. Zero-shot, un gabarit anglais | 46,0 % |
| A. Zero-shot, trois gabarits moyennés | 47,3 % |
| A. Zero-shot, une phrase en français | 29,6 % |
| B. Sonde linéaire, 1 image par classe (3 tirages : 57,5 / 65,7 / 49,4 %) | 57,5 % en moyenne |
| B. Sonde linéaire, 4 images par classe | 78,0 % |
| B. Sonde linéaire, 16 images par classe | 87,7 % |
| B. Sonde linéaire, 16 200 images | 96,6 % |
| C. Réglage fin, avant le premier pas (doit égaler A, 3 gabarits) | 47,3 % |
| C. Réglage fin, après 1 époque (253 pas de 64 images) | 98,0 % |
| D. CIFAR-10 zero-shot (2 000 images), avant → après C | 93,5 % → 82,1 % |

Ces chiffres sont enregistrés dans `figures/resultats-mps.json`, lu par la figure du site `site/scripts/figures/ch24_clip_eurosat.py`.

Sur T4, avec 2 époques en float16, attendez-vous à des chiffres proches (réglage fin un peu au-dessus, oubli sur CIFAR-10 peut-être un peu plus marqué) : **à confirmer au premier lancement**, puis reporter ici la durée.

## Ce qu'il faut observer

- **Le zero-shot dépend de la phrase** : le nom de classe brut (`AnnualCrop`, `SeaLake`) fait nettement moins bien qu'une phrase qui ressemble aux légendes du web (« a centered satellite photo of… »). Le français fait moins bien que l'anglais : le modèle a été entraîné sur LAION-2B, le sous-ensemble anglais de LAION-5B, et sa fiche le dit réservé à l’anglais.
- **Le zero-shot reste faible sur ce domaine** (47,3 % en mesure locale, quand le même CLIP fait 93,5 % sur CIFAR-10 avec la même méthode) : les images satellite et les légendes du type « annual crop land » sont sans doute rares dans ses données d’entraînement.
- **Quelques étiquettes suffisent à dépasser le zero-shot** : avec 1 image par classe, la sonde linéaire fait déjà 57,5 % en moyenne (mais de 49,4 à 65,7 % selon l'image tirée), 87,7 % avec 16, 96,6 % avec tout le jeu. Les plongements contiennent l'information ; ce sont les phrases qui la désignent mal.
- **Le réglage fin gagne encore**, mais l'encodeur d'image s'éloigne de l'espace commun : le zero-shot sur CIFAR-10 est passé de 93,5 % à 82,1 % après une seule époque sur EuroSAT (oubli), pendant qu'EuroSAT montait de 47,3 % à 98,0 %. Le réglage fin achète la spécialisation au prix de la généralité.

## Questions de réflexion

1. Pourquoi initialiser la tête du réglage fin avec les plongements des phrases plutôt qu'au hasard ? Que se passerait-il pendant les premiers pas avec une tête aléatoire et un encodeur pré-entraîné ?
2. Le taux d'apprentissage est 10⁻⁵, cent fois moins que pour un réseau entraîné de zéro. Que risquerait-on avec 10⁻³ ?
3. La sonde linéaire à 1 image par classe varie beaucoup d'un tirage à l'autre. Pourquoi ? Qu'est-ce que cela dit des évaluations « few-shot » publiées sur un seul tirage ?
4. Comment limiter l'oubli mesuré à l'étape D ? (Pistes : geler une partie du réseau, moyenner les poids avant et après réglage, mélanger des données génériques.)

## Pour aller plus loin

- Remplacer `"ViT-B-32", pretrained="laion2b_s34b_b79k"` par un CLIP plus grand (`open_clip.list_pretrained()`), en vérifiant la licence de ses poids sur le Hub.
- Exercice 24a : le même principe entraîné de zéro, à toute petite échelle, en local.
- Exercice 13b : l'apprentissage par transfert avec un CNN pré-entraîné sur ImageNet, sans texte.
