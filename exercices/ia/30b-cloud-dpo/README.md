# Exercice 30b : aligner SmolLM2-360M-Instruct sur des préférences avec DPO (cloud)

| | |
|---|---|
| **Chapitre lié** | [30. Adaptation et alignement](https://laclasse.poneyponey.fr/ia/p08-llm/30-adaptation-alignement/) |
| **Exécution** | cloud : Google Colab gratuit, GPU T4 (secours : Kaggle, « GPU T4 x2 ») |
| **Commande** | ouvrir `notebook.ipynb` dans Colab, puis Exécution › Tout exécuter (pas de `make ex` : notebook cloud) |
| **Durée mesurée** | **non mesurée sur T4** (les cellules 6 et 7 affichent leurs durées). Estimation non vérifiée : 20 à 30 min en tout, dont ≈ 1 min de téléchargement (modèle ≈ 0,7 Go, données ≈ 250 Mo). Seule une version miniature a tourné localement (voir « Vérification locale ») |
| **Travail estimé** | ~1 h 30 (plus l'attente) |
| **Seed** | 42 (`random`, NumPy, PyTorch, `DPOConfig(seed=42)`, mélange des données) |
| **Données** | `HuggingFaceH4/ultrafeedback_binarized` (Hugging Face, licence MIT) : pour chaque prompt d'UltraFeedback, la réponse la mieux notée par GPT-4 (« chosen ») et une autre tirée au hasard parmi les moins bien notées (« rejected »). On garde les prompts de 600 caractères au plus et les réponses de 1 500 au plus, sans égalité de note, puis 2 000 paires de `train_prefs` et 200 de `test_prefs`. Téléchargées dans la session |
| **Modèle** | `HuggingFaceTB/SmolLM2-360M-Instruct` (licence Apache-2.0, non restreint : aucun token, aucune demande d'accès ; vérifié via l'API du Hub le 15/10/2026, `gated: false`) |
| **Coût** | 0 € (aucune carte bancaire demandée) |

## Objectif

Aligner un petit assistant déjà réglé par instructions sur des préférences (paires réponse choisie / réponse rejetée) avec DPO, en n'entraînant qu'un LoRA, et mesurer ce que l'entraînement change : récompenses implicites $\beta \log \pi_\theta / \pi_{\text{ref}}$, préférence du modèle sur des paires qu'il n'a pas vues, longueur des réponses. L'exercice 30c montre sur un exemple jouet pourquoi DPO vise la même politique que le RLHF ; ici, c'est la version « bibliothèque » (TRL) sur un vrai LLM.

## Ce que vous allez faire

1. Vérifier le GPU (cellule 1), installer les dépendances épinglées (cellule 2), lire la configuration (cellule 3). `COURT = True` limite l'entraînement à 30 pas.
2. **Données** (cellule 4) : une paire d'exemple ; longueur moyenne des réponses choisies et rejetées, et part des paires où la choisie est la plus longue.
3. **La perte DPO à la main** (cellule 5) : $\log \pi(y \mid x)$ = somme des log-probabilités des tokens de la réponse ; au départ ($\pi_\theta = \pi_{\text{ref}}$) la perte vaut $\log 2 = 0{,}693$ ; si l'entraînement montait la réponse choisie de 5 nats et descendait la rejetée de 5, la marge vaudrait $0{,}1 \times 10 = 1$ et la perte $-\log\sigma(1) = 0{,}313$.
4. **Avant** (cellule 6) : sur les 200 paires de test, part où le modèle donne déjà plus de probabilité à la réponse choisie (somme, puis moyenne par token) ; 40 réponses générées et leur longueur.
5. **DPO avec TRL** (cellule 7) : `DPOTrainer`, β = 0,1, LoRA de rang 16 sur toutes les projections linéaires, taux 5 × 10⁻⁵, 250 pas de lots de 8 paires, float16. Avec un modèle PEFT, TRL n'a pas besoin d'une copie de référence : $\pi_{\text{ref}}$ est le même réseau, adaptateur désactivé.
6. **Après** (cellule 8) : récompense implicite recalculée à la main sur une paire (adaptateur activé puis désactivé), préférences sur les paires de test, longueur des réponses, deux réponses avant/après.
7. **Courbes et sauvegarde** (cellules 9 et 10).

**Perte optimisée** : $\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}\big[\log \sigma\big(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\big)\big]$, avec β = 0,1 ; seuls les paramètres du LoRA reçoivent un gradient.

## Procédure pas à pas (Colab)

1. **Compte** : un compte Google suffit (Colab gratuit). Plan B Kaggle : compte Kaggle, vérification par téléphone probable pour activer GPU et Internet.
2. **Ouvrir le notebook** : récupérer le fichier `notebook.ipynb` de ce dossier, puis, dans Colab, Fichier › Importer un notebook › Importer, et choisir ce fichier (sur Kaggle : voir le plan B ci-dessous). Il n'y a pas de lien direct vers Colab.
3. **Activer le GPU** : Exécution › Modifier le type d'exécution › GPU T4 › Enregistrer. La première cellule affiche `nvidia-smi` ; sans GPU, elle s'arrête avec un message clair.
4. **Token Hugging Face** : inutile. Le modèle et les données ne sont pas restreints ; rien n'est publié sur le Hub.
5. **Exécution › Tout exécuter.** La cellule 2 installe transformers, trl, peft, accelerate et datasets aux versions épinglées ; torch n'est volontairement pas épinglé. Si Colab propose de redémarrer la session après l'installation, accepter puis relancer « Tout exécuter ».
6. **Sauvegarder avant la fin de session** : la cellule 10 télécharge `30b-resultats.json`, `30b-dpo.png` et `adaptateur-dpo.zip`. Une session gratuite dure 12 h au plus et s'arrête plus tôt en cas d'inactivité.
7. **Plan B Kaggle** : Create › New Notebook › File › Import Notebook ; Settings › Accelerator › **GPU T4 x2** (jamais P100) ; Settings › Internet › On ; Run All. TRL refuse de tourner quand deux GPU sont visibles sans lanceur distribué : rien à faire, la cellule 1 détecte Kaggle et n'en garde qu'un (`CUDA_VISIBLE_DEVICES=0`, avant `import torch`). Les fichiers restent dans `/kaggle/working` (onglet Output).
8. **Coût : 0 €.**

## Résultat attendu

**Non mesuré sur T4** : critères à confirmer au premier lancement.

- Cellule 5 : perte à la main au départ exactement $\log 2 = 0{,}6931$ ; exemple +5/−5 nats : marge 1,0, perte 0,3133.
- Cellule 7 : avant l'entraînement, `eval_rewards/margins` = 0 et `eval_loss` = 0,693 (politique = référence). Après : `eval_loss` sous 0,693, récompenses implicites des réponses choisies au-dessus de celles des rejetées (marge positive), `eval_rewards/accuracies` au-dessus de 50 %.
- Cellule 8 : la récompense implicite recalculée à la main sur la paire n° 0 est du même ordre que celles affichées par TRL (même formule ; de petits écarts viennent de la mise en forme des tokens) ; la part des paires de test où la réponse choisie est la plus probable augmente.
- Cellule 4 (mesuré) : sur les 2 000 paires d'entraînement filtrées, réponses choisies de 516 caractères en moyenne contre 498 pour les rejetées ; la choisie est la plus longue dans **51 %** des paires seulement. Le filtre de longueur (1 500 caractères au plus) retire l'essentiel du biais de longueur ; surveillez quand même la longueur des réponses générées.
- Cellule 6 (mesuré localement sur les 50 premières paires de test) : avant DPO, le modèle donne déjà plus de probabilité à la réponse choisie dans **58 %** des paires (somme comme moyenne par token) : le SFT d'origine de SmolLM2 a déjà appris une partie de ces préférences.
- Mesure locale réduite (40 pas de 4 paires, soit 160 paires vues, β = 0,1, taux 5 × 10⁻⁵) : `eval_loss` 0,6931 → 0,6889 ; marge 0 → 0,0105 ; récompenses implicites choisies −0,0035 et rejetées −0,0140 (les deux baissent, la rejetée plus vite) ; `eval_rewards/accuracies` 0 → 54 % ; préférences sur les paires de test inchangées (58 %). C'est trop peu d'entraînement pour conclure : sur T4, les 250 pas de 8 paires (2 000 paires) doivent donner une marge nettement positive. Attention, TRL compte `rewards/accuracies` avec une inégalité stricte : il vaut 0 (et non 50 %) avant l'entraînement, quand toutes les récompenses sont nulles.

## Vérification locale

- `nbformat.validate` et compilation de chaque cellule (lignes `!` ignorées) : OK, 11 cellules, le 15/10/2026.
- Dépendances épinglées vérifiées sur PyPI le 15/10/2026, toutes compatibles Python 3.12 : transformers 5.19.0, trl 1.15.0 (`requires_python` ≥ 3.11), peft 0.21.2, accelerate 1.15.0, datasets 5.1.0.
- Code des cellules 3 à 10 exécuté localement le 15/10/2026 sur M1 Pro (MPS, float32), variable `IA_30B_TEST=1` : miniature (32 paires, 4 pas) puis mesure réduite (2 000 paires chargées, 40 pas de 2 × 2 paires, contexte 512, 50 paires de test, 8 générations de 200 tokens au plus) : OK du chargement à la sauvegarde de l'adaptateur ; 16,5 s/pas sur MPS (11,0 min d'entraînement). Le float16 n'a pas été testé. Sur macOS, TRL 1.15 n'a pas de repli PyTorch quand Triton manque (calcul « fused » des log-probabilités) ; le test local fournissait ce repli. Sur Colab (Linux, Triton présent), rien à faire.

## Ce qu'il faut observer

- **La perte part de log 2 et les récompenses de 0.** Politique et référence sont identiques au départ : DPO ne peut que les écarter, et β mesure combien cet écart « coûte ».
- **Les deux récompenses implicites baissent souvent ensemble.** DPO ne demande qu'une marge positive : il peut l'obtenir en baissant la réponse rejetée plus vite que la choisie. C'est un comportement connu de DPO, pas un bogue.
- **Une seule copie des poids suffit.** Le RLHF par PPO garde en mémoire politique, référence, modèle de récompense et critique ; DPO avec LoRA garde un seul réseau, que l'on évalue avec et sans adaptateur.
- **La longueur est un piège.** Dans UltraFeedback sans filtre (61 135 paires de `train_prefs`), la réponse choisie fait 1 295 caractères en moyenne contre 1 121 pour la rejetée, et elle est la plus longue dans 55 % des paires (mesuré le 15/10/2026) ; un modèle peut alors gagner des préférences en allongeant ses réponses sans les améliorer : c'est la sur-optimisation de l'exercice 30c, en vrai. Ici, le filtre de longueur réduit ce biais (51 %) ; comparez les longueurs avant et après.

## Questions de réflexion

1. Avec β = 0,1, de combien faut-il augmenter l'écart des log-ratios (en nats) pour faire passer la perte d'une paire de 0,693 à 0,313 ? Et avec β = 0,5 ?
2. Pourquoi comparer des probabilités par rapport à $\pi_{\text{ref}}$ plutôt que des probabilités brutes ? Pensez à une réponse longue, dont la log-probabilité totale est forcément très négative.
3. Les paires d'UltraFeedback ont été produites par d'autres modèles que SmolLM2. En quoi est-ce différent du RLHF, où la politique génère elle-même les réponses que le modèle de récompense note ?
4. Les préférences ont été données par GPT-4, pas par des humains. Quel nom porte cette approche dans le chapitre (Constitutional AI, RLAIF) et quels risques ajoute-t-elle ?

## Pour aller plus loin

- Refaire l'entraînement avec β = 0,5 puis β = 0,02 et comparer marge, préférences et longueur.
- Remplacer la perte par IPO (`loss_type=["ipo"]` dans `DPOConfig`) et comparer.
- Articles : Rafailov et al., DPO, [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) ; Ouyang et al., InstructGPT, [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) ; Bai et al., Constitutional AI, [arXiv:2212.08073](https://arxiv.org/abs/2212.08073).
