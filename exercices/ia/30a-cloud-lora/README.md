# Exercice 30a : d'un complèteur de texte à un assistant, par réglage fin complet, LoRA et QLoRA (cloud)

| | |
|---|---|
| **Chapitre lié** | [30. Adaptation et alignement](https://laclasse.poneyponey.fr/ia/p08-llm/30-adaptation-alignement/) |
| **Exécution** | cloud : Google Colab gratuit, GPU T4 (secours : Kaggle, « GPU T4 x2 ») |
| **Commande** | ouvrir `notebook.ipynb` dans Colab, puis Exécution › Tout exécuter (pas de `make ex` : notebook cloud) |
| **Durée mesurée** | **non mesurée sur T4** (la cellule 8 affiche la durée de chaque variante). Estimation non vérifiée : 20 à 30 min en tout, dont ≈ 1 min de téléchargement (modèle ≈ 0,7 Go, données ≈ 13 Mo). Seule une version miniature a tourné localement (voir « Vérification locale ») |
| **Travail estimé** | ~1 h 30 (plus l'attente) |
| **Seed** | 42 (`random`, NumPy, PyTorch, `SFTConfig(seed=42)`, mélange des données) |
| **Données** | `databricks/databricks-dolly-15k` (Hugging Face, licence CC BY-SA 3.0) : 15 011 paires instruction/réponse écrites par des employés de Databricks. On garde les questions sans texte d'appui (9 296), puis 2 000 pour l'entraînement et 200 pour l'évaluation. Téléchargées dans la session |
| **Modèle** | `HuggingFaceTB/SmolLM2-360M`, le modèle **de base** (pré-entraîné seulement), licence Apache-2.0, non restreint : aucun token, aucune demande d'accès (vérifié via l'API du Hub le 15/10/2026, `gated: false`) |
| **Coût** | 0 € (aucune carte bancaire demandée) |

## Objectif

Voir ce que le réglage par instructions (SFT) apprend à un modèle pré-entraîné (répondre, puis s'arrêter), et ce que coûte cet apprentissage selon la méthode : réglage fin complet, LoRA (poids gelés en float16, matrices de rang 16 entraînées) ou QLoRA (poids gelés sur 4 bits). Avant cela, écrire une couche LoRA à la main pour comprendre ce que fait la bibliothèque PEFT.

## Ce que vous allez faire

1. Vérifier le GPU (cellule 1), installer les dépendances épinglées (cellule 2), lire la configuration (cellule 3). `COURT = True` limite chaque entraînement à 30 pas.
2. **LoRA à la main** (cellule 4, CPU) : une couche $Wx + \frac{\alpha}{r}BAx$ avec $B = 0$ au départ (sortie identique à la couche d'origine), puis l'apprentissage d'une correction cachée de rang 4 avec des rangs r = 1, 2, 4 et 16.
3. **Données** (cellule 5) : chaque exemple devient un couple *prompt* (`### Instruction:` … `### Response:`) / *completion* (la réponse).
4. **Avant** (cellule 6) : le modèle de base face à trois questions (dont une en français).
5. **SFT, trois variantes** (cellules 7 et 8), avec TRL (`SFTTrainer`), 250 pas de lots de 8 exemples, perte calculée sur la réponse seulement (`completion_only_loss=True`), calcul en float16, *gradient checkpointing* :
   - **complet** : les 362 M paramètres, AdamW, taux 2 × 10⁻⁵ ;
   - **LoRA** : poids gelés en float16, matrices de rang 16 (α = 32) sur toutes les projections linéaires, taux 2 × 10⁻⁴ ;
   - **QLoRA** : idem, mais poids gelés quantifiés en NF4 sur 4 bits (bitsandbytes), double quantification.
   Pour chacune : paramètres entraînés, pic de mémoire GPU, durée, perte d'évaluation avant et après, et réponses aux trois questions.
6. **Bilan** (cellule 9) : tableau, courbes de perte d'évaluation ; **sauvegarde** (cellule 10) : résultats JSON, figure, adaptateur LoRA.

**Perte optimisée** : l'entropie croisée du token suivant, sur les tokens de la réponse seulement : $\mathcal L = -\frac{1}{|y|}\sum_t \log p_\theta(y_t \mid x, y_{<t})$ ($x$ : l'instruction mise en forme, $y$ : la réponse, suivie du token de fin de texte).

## Procédure pas à pas (Colab)

1. **Compte** : un compte Google suffit (Colab gratuit). Plan B Kaggle : compte Kaggle, vérification par téléphone probable pour activer GPU et Internet.
2. **Ouvrir le notebook** : récupérer le fichier `notebook.ipynb` de ce dossier, puis, dans Colab, Fichier › Importer un notebook › Importer, et choisir ce fichier (sur Kaggle : voir le plan B ci-dessous). Il n'y a pas de lien direct vers Colab.
3. **Activer le GPU** : Exécution › Modifier le type d'exécution › GPU T4 › Enregistrer. La première cellule affiche `nvidia-smi` ; sans GPU, elle s'arrête avec un message clair.
4. **Token Hugging Face** : inutile. Le modèle et les données ne sont pas restreints ; rien n'est publié sur le Hub.
5. **Exécution › Tout exécuter.** La cellule 2 installe transformers, trl, peft, accelerate, datasets et bitsandbytes aux versions épinglées ; torch n'est volontairement pas épinglé (celui de l'image Colab convient). Si Colab propose de redémarrer la session après l'installation, accepter puis relancer « Tout exécuter ».
6. **Sauvegarder avant la fin de session** : la cellule 10 télécharge `30a-resultats.json`, `30a-pertes.png` et `adaptateur-lora.zip` (quelques dizaines de Mo). Une session gratuite dure 12 h au plus et s'arrête plus tôt en cas d'inactivité.
7. **Plan B Kaggle** : Create › New Notebook › File › Import Notebook ; Settings › Accelerator › **GPU T4 x2** (jamais P100) ; Settings › Internet › On ; Run All. Les fichiers restent dans `/kaggle/working` (onglet Output). TRL refuse de tourner quand deux GPU sont visibles sans lanceur distribué : rien à faire, la cellule 1 détecte Kaggle et n'en garde qu'un (`CUDA_VISIBLE_DEVICES=0`, avant `import torch`).
8. **Coût : 0 €.**

## Résultat attendu

**Non mesuré sur T4.** Les valeurs ci-dessous viennent de la vérification locale (M1 Pro, MPS, float32, 15/10/2026, graine 42 ; détails plus bas) ; celles du T4 seront du même ordre pour la perte, pas pour la durée.

- Cellule 4 (mesuré) : « Sortie identique au départ : True » ; LoRA de rang 16 sur une couche 960 × 960 : 30 720 paramètres entraînés pour 921 600 gelés (3,3 %). Pour une correction cachée de rang 4, erreur résiduelle après 300 pas : **69,5 %** (r = 1), **47,4 %** (r = 2), **0,0 %** (r = 4 et r = 16).
- Paramètres entraînés (mesuré) : complet 361,8 M (100 %) ; LoRA **8,7 M (2,40 %)** ; QLoRA idem (mêmes matrices).
- Avant SFT (mesuré, sorties gloutonnes de 120 tokens) : le modèle de base ne s'arrête sur aucune des trois questions. Sur la question Python, il répond une phrase, puis invente la suite d'un questionnaire (`### Question:` / `### Response:`) qu'il répète ; sur la question en français, il enchaîne `### Explanation:` puis `### Activity:`.
- Après SFT LoRA (mesuré en local, **80 pas** de lots de 8, soit 640 exemples, contexte 384) : les trois réponses **s'arrêtent seules** (token de fin de texte) ; ex. « 1. Set a schedule for yourself. 2. Take breaks. 3. Stay hydrated. ». La perte d'évaluation (100 exemples) baisse peu : **2,186 → 2,098** (2,122 au pas 16, puis 2,107 ; 2,100 ; 2,099). La réponse en français reste pauvre et répétitive : 640 exemples en anglais n'enseignent pas le français.
- Sur T4, avec 250 pas : la perte d'évaluation des trois variantes devrait passer un peu sous celle de la mesure locale ; LoRA et QLoRA proches l'une de l'autre ; le réglage complet a le pic de mémoire le plus élevé (poids maîtres, gradients et deux états d'Adam en float32 : environ 16 octets par paramètre, ≈ 5,8 Go avant activations), QLoRA le plus bas pour les poids (4 bits au lieu de 16), au prix d'un pas plus lent (décompression).

## Vérification locale

- `nbformat.validate` et compilation de chaque cellule (lignes `!` ignorées) : OK, 11 cellules, le 15/10/2026.
- Dépendances épinglées vérifiées sur PyPI le 15/10/2026, toutes compatibles Python 3.12 : transformers 5.19.0, trl 1.15.0 (`requires_python` ≥ 3.11), peft 0.21.2, accelerate 1.15.0, datasets 5.1.0, bitsandbytes 0.50.2.
- Code des cellules 3 à 10 exécuté localement le 15/10/2026 sur M1 Pro (MPS, float32), variable `IA_30A_TEST=1` : miniature (48 exemples, 4 pas, variantes complet et LoRA) puis mesure réduite de la variante LoRA (**80 pas** de lots de 8, soit 640 exemples, contexte 384, 100 exemples d'évaluation). Vitesse : 4,55 s/pas en LoRA (6,1 min d'entraînement) ; 31 s/pas pour le réglage complet dans la miniature.
- **Non exécutés** : QLoRA (bitsandbytes 4 bits demande CUDA), le calcul en float16 et la durée sur T4. Sur macOS, TRL 1.15 n'a pas de repli PyTorch quand Triton manque (calcul « fused » des log-probabilités) ; le test local fournissait ce repli. Sur Colab (Linux, Triton présent), rien à faire.

## Ce qu'il faut observer

- **Ce que le SFT apprend, c'est surtout un format.** 640 exemples ont suffi en local pour que le modèle réponde puis s'arrête, alors que la perte n'a baissé que de 4 %. Ses connaissances, elles, viennent du pré-entraînement : la réponse sur Python est la même phrase (fausse : la différence est la mutabilité, pas le type des éléments) avant et après.
- **LoRA entraîne quarante fois moins de paramètres pour une perte proche.** L'hypothèse de Hu et al. (2021) : la mise à jour utile des poids a un rang faible. La cellule 4 le montre en petit : une correction de rang 4 est retrouvée dès r = 4.
- **La mémoire du réglage complet vient surtout de l'optimiseur.** Adam garde deux nombres par paramètre ; avec les gradients et les poids maîtres en float32, on compte environ 16 octets par paramètre. LoRA supprime presque tout cela ; QLoRA comprime en plus les poids gelés.
- **Le fichier de l'adaptateur pèse quelques dizaines de Mo** contre ≈ 0,7 Go pour le modèle en float16 : on peut garder un adaptateur par tâche et un seul modèle de base.

## Questions de réflexion

1. Pour une matrice 960 × 2 560 (une projection du MLP de SmolLM2), combien de paramètres pour un LoRA de rang 16 ? Quelle fraction de la matrice ?
2. Pourquoi initialise-t-on $B$ à zéro et pas $A$ ? Que se passerait-il si les deux partaient de zéro ?
3. La perte ne porte que sur la réponse. Qu'apprendrait le modèle si elle portait aussi sur l'instruction ?
4. Le modèle de base `Qwen/Qwen3-0.6B-Base` répond déjà correctement au format `### Instruction:` sans aucun SFT (constaté le 15/10/2026). Qu'est-ce que cela suggère sur les données de pré-entraînement des modèles récents ?

## Pour aller plus loin

- Comparer les rangs 4, 16 et 64 (`rang`, garder `alpha = 2 × rang`) : perte, durée, taille de l'adaptateur.
- Remplacer les données par un jeu d'instructions en français et refaire la question en français.
- Fusionner l'adaptateur dans les poids (`merge_and_unload()` de PEFT) et vérifier que les réponses ne changent pas.
- Articles : Hu et al., LoRA, [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) ; Dettmers et al., QLoRA, [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) ; Wei et al., FLAN, [arXiv:2109.01652](https://arxiv.org/abs/2109.01652).
