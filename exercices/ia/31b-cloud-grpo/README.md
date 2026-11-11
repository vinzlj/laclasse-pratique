# Exercice 31b : apprendre à raisonner par renforcement avec GRPO (cloud)

| | |
|---|---|
| **Chapitre lié** | [31. Raisonnement et calcul à l'inférence](https://laclasse.poneyponey.fr/ia/p08-llm/31-raisonnement-inference/) |
| **Exécution** | cloud : Google Colab gratuit, GPU T4 (secours : Kaggle, « GPU T4 x2 ») |
| **Commande** | ouvrir `notebook.ipynb` dans Colab, puis Exécution › Tout exécuter (pas de `make ex` : notebook cloud) |
| **Durée mesurée** | **non mesurée sur T4** (estimation : 45 à 60 min, dont ~35 min d'entraînement pour 150 pas et ~2 × 3 min d'évaluation ; la cellule 6 affiche la durée réelle, à reporter ici). Seul le code a été exécuté localement, en miniature (voir « Vérification locale ») |
| **Travail estimé** | ~1 h (plus l'attente) |
| **Seed** | 42 (`random`, NumPy, PyTorch, `GRPOConfig(seed=42)`, mélange de GSM8K) |
| **Données** | GSM8K (`openai/gsm8k`, licence MIT, révision figée dans le notebook) et le modèle `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0, non restreint), téléchargés dans la session depuis le Hub Hugging Face (~1 Go) |
| **Coût** | 0 € (aucune carte bancaire demandée) |

## Objectif

Entraîner un petit LLM à mieux résoudre des problèmes d'arithmétique en plusieurs étapes, **sans exemples de raisonnement** : seulement une récompense vérifiable (1 si la réponse finale est exacte, 0 sinon) et l'algorithme GRPO (TRL 1.15), puis mesurer la précision avant et après sur 200 problèmes de test.

## Ce que vous allez faire

1. Vérifier le GPU (cellule 1), installer les dépendances épinglées (cellule 2 : `trl==1.15.0`, `transformers==5.19.0`, `datasets==5.1.0`, `accelerate==1.15.0`, `peft==0.21.2` ; torch n'est pas épinglé).
2. Lire la configuration (cellule 3) : 150 pas, groupes de G = 8 réponses, 2 problèmes par pas, réponses de 256 tokens au plus, `beta = 0.04`, `epsilon = 0.2`, LoRA de rang 16 sur les projections de l'attention. `COURT = True` : 10 pas et 50 problèmes d'évaluation, pour tester (~5 min).
3. Charger GSM8K et écrire la **fonction de récompense** (cellule 4) : même extraction de la réponse que l'exercice 31a (« Answer: n », sinon dernier nombre).
4. Évaluer le modèle de départ en décodage glouton (cellule 5).
5. Entraîner avec `GRPOTrainer` (cellule 6) : pour chaque problème, 8 réponses échantillonnées à température 1 ; avantage = (récompense − moyenne du groupe) / écart-type du groupe (TRL prend l'écart-type corrigé, en n − 1 : pour [0, 1, 0, 0], +1,5 et −0,5 au lieu de +1,73 et −0,58 au chapitre) ; objectif écrêté de PPO plus pénalité KL envers le modèle de départ (l'adaptateur LoRA désactivé). Calcul en float16.
6. Réévaluer (cellule 7), tracer les courbes (cellule 8), lire une réponse et sauvegarder (cellule 9).

## Procédure pas à pas (Colab)

1. **Compte** : un compte Google suffit (Colab gratuit). Plan B Kaggle : compte Kaggle, vérification par téléphone probable pour activer GPU et Internet.
2. **Ouvrir le notebook** : télécharger `notebook.ipynb` depuis le dossier de l'exercice, puis dans Colab, Fichier › Importer un notebook, et choisir ce fichier.
3. **Activer le GPU** : Exécution › Modifier le type d'exécution › GPU T4 › Enregistrer. La première cellule affiche `nvidia-smi` ; sans GPU, elle s'arrête avec un message clair.
4. **Token Hugging Face** : inutile (modèle et données non restreints, rien n'est publié sur le Hub).
5. **Exécution › Tout exécuter.** Si Colab propose de redémarrer la session après l'installation de la cellule 2, accepter puis relancer « Tout exécuter ». Noter la durée affichée par la cellule 6.
6. **Sauvegarder avant la fin de session** : la cellule 9 télécharge `resultats-31b.zip` (adaptateur LoRA, ~9 Mo), `bilan-31b.json` (précisions, historique de l'entraînement) et `courbes-31b.png`. Une session gratuite dure 12 h au plus et s'arrête plus tôt en cas d'inactivité.
7. **Plan B Kaggle** : Create › New Notebook › File › Import Notebook ; Settings › Accelerator › **GPU T4 x2** (jamais P100) ; Settings › Internet › On ; « Save Version › Save & Run All » pour un run sans surveillance. Les fichiers restent dans `/kaggle/working` (onglet Output). La cellule 1 ne laisse voir qu'un GPU (`CUDA_VISIBLE_DEVICES=0`) : GRPOTrainer refuse `nn.DataParallel`, que `Trainer` emploierait avec deux GPU visibles.
8. **Coût : 0 €.**

## Résultat attendu

**Non mesuré sur T4** : seuils prudents, à confirmer au premier lancement et à reporter ici.

- Récompense moyenne pendant l'entraînement (courbe de gauche, lissée) : en hausse entre les 20 premiers et les 20 derniers pas.
- Précision en test après GRPO **supérieure ou égale** à la précision avant (attendu : quelques points de gain sur 200 problèmes ; un écart de moins de 3 points n'est pas significatif à cette taille d'échantillon).
- Risque non vérifié : avec `lr=2e-5` en LoRA sur 150 pas, le gain peut être nul. Si c'est le cas au premier lancement, relancer avec `lr=5e-5` (cellule 3) et reporter les deux résultats ici.

## Vérification locale

- `nbformat.validate` et compilation de chaque cellule (lignes `!` ignorées) : OK, 10 cellules, le 15/10/2026.
- Dépendances épinglées vérifiées sur PyPI le 15/10/2026 (Python 3.12 accepté par chacune). Licences vérifiées via l'API du Hub : `Qwen/Qwen2.5-0.5B-Instruct` apache-2.0, non restreint ; `openai/gsm8k` mit, non restreint.
- Cellules 3 à 9 exécutées en miniature le 15/10/2026 sur le M1 Pro (MPS, `fp16=True`), avec les versions épinglées et Python 3.12 : modèle de test `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5` (2,4 M de paramètres, poids aléatoires), 3 pas, groupes de 2, réponses de 16 tokens, 6 problèmes d'évaluation. Tout s'exécute (données, récompense, évaluation, `GRPOTrainer` avec LoRA, courbes, sauvegarde) ; précision 0 % avant et après, attendu pour un modèle aléatoire. Ces chiffres ne vérifient que le code, pas le seuil ni la durée.
- Particularité de TRL 1.15 : le calcul des log-probabilités de GRPO passe par un noyau Triton, qui n'existe que sous Linux. Pour la validation sur macOS seulement, ce noyau a été remplacé par un calcul PyTorch équivalent ; sur Colab et Kaggle (Linux), le notebook tourne sans modification. Conséquence : **cet exercice ne peut pas tourner tel quel sur un Mac.**

## Ce qu'il faut observer

- **La part des groupes sans signal** (courbe du milieu, `frac_reward_zero_std`) : quand les 8 réponses d'un groupe sont toutes justes ou toutes fausses, l'avantage est nul pour toutes et le pas n'apprend rien de ce problème. C'est le même calcul que la fin de l'exercice 31a, à l'échelle d'un entraînement ; c'est aussi ce que l'échantillonnage dynamique de DAPO filtre.
- **La longueur des réponses** (courbe de droite) : avec `loss_type="grpo"`, chaque réponse est moyennée sur sa propre longueur, ce qui pèse moins par token sur les longues réponses fausses ; Dr. GRPO a montré que ce biais peut allonger les réponses. Comparer avant et après.
- **Le gain en test** : modeste en 150 pas, et dans les limites d'un petit modèle. Le débat du chapitre (le RL élargit-il le raisonnement, ou rend-il plus probables des réponses que le modèle savait déjà produire ?) se lit ici : la précision gloutonne monte, mais le modèle de départ trouvait souvent déjà la bonne réponse parmi ses 8 essais.

## Questions de réflexion

1. Les valeurs par défaut de TRL 1.15 sont `loss_type="dapo"` et `beta=0.0`. Qu'est-ce que chacune change par rapport à la formulation de DeepSeekMath utilisée ici, et pourquoi DAPO (arXiv:2503.14476) se passe-t-il de la pénalité KL ?
2. Avec des groupes de G = 8 et une précision de départ p par problème, quelle est la probabilité qu'un groupe n'ait aucun signal ? Calculez-la pour p = 0,1, 0,5 et 0,9 (réponses supposées indépendantes).
3. Pourquoi la récompense vérifiable de cet exercice est-elle difficile à « pirater » (chapitre 30), et que pourrait faire le modèle si l'on récompensait seulement la présence de « Answer: » ?
4. Pourquoi n'a-t-on pas besoin d'un réseau de valeur (critique), contrairement à PPO (chapitre 17) ?

## Pour aller plus loin

- Passer `taille_groupe` à 4 puis 16 (en gardant 16 complétions par pas, donc 4 puis 1 problème) et comparer la part de groupes sans signal.
- Essayer `loss_type="dr_grpo"` et comparer la longueur des réponses.
- Remplacer le modèle par `Qwen/Qwen3-0.6B` (Apache-2.0), en désactivant son mode de réflexion dans le gabarit de conversation.
