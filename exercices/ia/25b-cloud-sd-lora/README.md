# Exercice 25b : Stable Diffusion 1.5, échantillonneurs, guidage et LoRA (cloud)

| | |
|---|---|
| **Chapitre lié** | [25. Modèles génératifs II : diffusion, flow matching, vidéo](https://laclasse.poneyponey.fr/ia/p07-transformer/25-generatifs-2-diffusion/) |
| **Exécution** | cloud : Google Colab gratuit, GPU T4 (secours : Kaggle, « GPU T4 x2 ») |
| **Commande** | ouvrir `notebook.ipynb` dans Colab, puis Exécution › Tout exécuter (pas de `make ex` : notebook cloud) |
| **Durée mesurée** | **non mesurée sur T4** (à mesurer au premier lancement : la cellule 10 affiche la durée de l'entraînement, la cellule 5 celle de chaque image). Estimation non vérifiée : 30 à 45 min en tout, dont le téléchargement du modèle (≈ 2,7 Go : fichiers float16 du U-Net, du VAE, de l'encodeur de texte et du filtre de sécurité) et des données (≈ 240 Mo). Seule une version miniature a tourné localement (voir « Vérification locale ») |
| **Travail estimé** | ~1 h (plus l'attente) |
| **Seed** | 42 (`random`, NumPy, PyTorch ; tirage des lots par un `torch.Generator` de graine 42) ; images générées avec des graines fixes (42 pour les parties 1a et 1b, 0, 1 et 2 pour l'avant/après) |
| **Données** | `huggan/smithsonian_butterflies_subset` (Hugging Face) : 1 000 photos de papillons de collection, sous-ensemble recadré de `ceyda/smithsonian_butterflies`, lui-même composé des seules images marquées CC0 des collections du Smithsonian. Téléchargées dans la session |
| **Modèle** | `stable-diffusion-v1-5/stable-diffusion-v1-5` (licence CreativeML OpenRAIL-M, non restreint : aucun token, aucune demande d'accès ; vérifié via l'API du Hub le 15/10/2026, `gated: false`) |
| **Coût** | 0 € (aucune carte bancaire demandée) |

## Objectif

Passer du petit modèle de diffusion de l'exercice 25a (U-Net de 0,3 million de paramètres, chiffres 28 × 28) à un vrai modèle de **diffusion latente** guidé par du texte, voir sur de vraies images l'effet du nombre de pas de DDIM et du poids du guidage sans classifieur, puis l'adapter à un nouveau domaine en n'entraînant qu'un **LoRA** de rang 4 avec la même perte que dans l'exercice 25a.

## Ce que vous allez faire

1. Vérifier le GPU (cellule 1) et installer les dépendances épinglées (cellule 2).
2. Lire la configuration (cellule 3). `COURT = True` limite l'entraînement à 100 pas pour vérifier que tout tourne.
3. Charger Stable Diffusion 1.5 en float16 avec l'échantillonneur DDIM (cellule 4).
4. **Partie 1a** (cellule 5) : la même image (même prompt, même graine) en 50, 20, 10 et 5 pas de DDIM ; noter la durée de chacune.
5. **Partie 1b** (cellule 6) : même image avec un guidage de 1 ; 3 ; 7,5 (valeur par défaut) et 15.
6. **Partie 3, avant** (cellule 7) : demander « a photo of a smsn butterfly specimen » au modèle d'origine. « smsn » ne veut rien dire pour lui.
7. **Données** (cellule 8) : 1 000 photos, recadrées en 512 × 512, retournées horizontalement au hasard, une seule légende pour toutes (« a photo of a smsn butterfly specimen »).
8. **Partie 2, LoRA** (cellules 9 et 10) : geler le U-Net, le VAE et l'encodeur de texte ; ajouter des matrices de rang 4 aux projections q, k, v et de sortie de chaque attention du U-Net (recette du script officiel `train_text_to_image_lora.py` de diffusers) ; les entraîner 800 pas, lots de 2, AdamW 8 bits (bitsandbytes), taux 10⁻⁴, calcul en float16 (`autocast` + `GradScaler` : le T4 ne fait pas de bfloat16), gradient écrêté à 1, *gradient checkpointing* pour tenir dans les 15 Go du T4. À chaque pas : image → latent 64 × 64 × 4 par le VAE, bruit à un instant t tiré au hasard, prédiction du bruit par le U-Net conditionné par le texte.
9. Tracer la perte, sauvegarder les poids du LoRA (cellule 11), puis **partie 3, après** (cellule 12) : mêmes prompts et mêmes graines, LoRA désactivé puis activé ; et un prompt sans rapport (un phare) pour voir ce que le LoRA change ailleurs.
10. Télécharger les résultats (cellule 13).

**Perte optimisée** : l'erreur quadratique entre le bruit ajouté au latent et le bruit prédit par le U-Net (ε-prédiction), exactement la perte de l'exercice 25a, mais sur des latents et avec le texte en condition. Seuls les paramètres du LoRA reçoivent un gradient.

## Procédure pas à pas (Colab)

1. **Compte** : un compte Google suffit (Colab gratuit). Plan B Kaggle : compte Kaggle, vérification par téléphone probable pour activer GPU et Internet.
2. **Ouvrir le notebook** : importer le fichier à la main. Dans Colab (colab.research.google.com), Fichier › Importer un notebook › onglet « Importer », puis choisir `notebook.ipynb` (copié depuis `exercices/ia/25b-cloud-sd-lora/` sur votre machine). Le notebook est enregistré dans votre Google Drive (dossier « Colab Notebooks »).
3. **Activer le GPU** : Exécution › Modifier le type d'exécution › GPU T4 › Enregistrer. La première cellule affiche `nvidia-smi` ; sans GPU, elle s'arrête avec un message clair.
4. **Token Hugging Face** : inutile. Le modèle et les données ne sont pas restreints ; rien n'est publié sur le Hub.
5. **Exécution › Tout exécuter.** La cellule 2 installe diffusers, peft, bitsandbytes, accelerate, transformers et datasets aux versions épinglées ; torch n'est volontairement pas épinglé (celui de l'image Colab, PyTorch 2.11.0 en 2026.07, convient). Si Colab propose de redémarrer la session après l'installation, accepter puis relancer « Tout exécuter ».
6. **Sauvegarder avant la fin de session** : la cellule 13 télécharge `resultats-25b.zip` (poids du LoRA, quelques Mo) et les planches PNG. Une session gratuite dure 12 h au plus et s'arrête plus tôt en cas d'inactivité.
7. **Plan B Kaggle** : Create › New Notebook › File › Import Notebook ; Settings › Accelerator › **GPU T4 x2** (jamais P100) ; Settings › Internet › On ; Run All. Les fichiers restent dans `/kaggle/working` (onglet Output). Le notebook n'utilise qu'un des deux GPU.
8. **Coût : 0 €.**

Le LoRA sauvegardé se recharge sur n'importe quel pipeline Stable Diffusion 1.5 avec `pipe.load_lora_weights("lora-papillons")`.

## Résultat attendu

**Non mesuré sur T4** : critères qualitatifs, à confirmer au premier lancement.

- Partie 1a : à 50 et 20 pas, deux images presque identiques ; à 10 pas, des détails changent ; à 5 pas, une image plus pauvre ou floue. La durée par image est à peu près proportionnelle au nombre de pas.
- Partie 1b : avec un guidage de 1 (prédiction conditionnelle seule, sans extrapolation), une image terne, souvent peu fidèle au prompt ; 7,5 donne l'image attendue ; 15, des couleurs saturées et un contraste exagéré.
- Partie 2 : perte moyenne de l'ordre de 0,1, bruitée d'un pas à l'autre (chaque pas tire un autre instant t) et presque plate ; ce n'est pas elle qui montre l'effet du LoRA, ce sont les images.
- Partie 3 : sans LoRA, « smsn butterfly specimen » donne des papillons quelconques (photo dans la nature, dessin…) ; avec le LoRA, mêmes graines, des papillons de collection vus de dessus, ailes étalées, sur fond uni, comme les photos du Smithsonian. Poids sauvegardés : quelques Mo, contre environ 1,7 Go pour le U-Net complet en float16.

## Vérification locale

- `nbformat.validate` et compilation de chaque cellule (lignes `!` ignorées) : OK, 14 cellules, le 15/10/2026.
- Dépendances épinglées vérifiées sur PyPI le 15/10/2026, toutes compatibles Python 3.12 (`requires_python` ≥ 3.10) : diffusers 0.41.0, peft 0.21.2, bitsandbytes 0.50.2, accelerate 1.15.0, transformers 5.19.0, datasets 5.1.0.
- Code des cellules 3 à 13 exécuté sur CPU en miniature (`IA_25B_TEST=1` : pipeline de test `hf-internal-testing/tiny-stable-diffusion-torch`, images 64 × 64, 3 pas d'inférence, 3 pas d'entraînement, 8 images, AdamW de PyTorch à la place d'AdamW 8 bits, float32), le 15/10/2026 : OK, du chargement à la sauvegarde du LoRA. Ces essais ne vérifient que le code : ni la qualité des images, ni la mémoire du T4, ni l'optimiseur 8 bits.

## Ce qu'il faut observer

- **Même réseau, moins de pas.** DDIM réutilise le U-Net entraîné pour 1 000 pas sans le réentraîner ; 20 à 50 pas suffisent ici, comme dans l'exercice 25a. Le coût d'une image est proportionnel au nombre de pas : c'est la raison d'être des échantillonneurs rapides et du flow matching.
- **Le guidage échange diversité contre fidélité.** Le modèle a été entraîné en retirant le texte dans 10 % des cas (fiche du modèle) : il sait prédire le bruit avec et sans légende, et le guidage extrapole de l'un vers l'autre. Même compromis que la partie D de l'exercice 25a : plus de fidélité à la condition, moins de variété, des images saturées à poids fort.
- **Un LoRA entraîne moins de 1 % des paramètres** et suffit à associer un mot nouveau (« smsn ») à un style d'image. La perte est la même qu'en 25a ; seule la partie entraînée change.
- **Le LoRA déborde.** Le phare généré avec le LoRA peut changer lui aussi (fond plus uni, couleurs) : les matrices ajoutées modifient toutes les attentions du U-Net, pas seulement la réponse au mot « smsn ».

## Questions de réflexion

1. Une image 512 × 512 × 3 devient un latent 64 × 64 × 4. Par combien le nombre de valeurs à débruiter est-il divisé ? Pourquoi la diffusion latente rend-elle Stable Diffusion utilisable sur un GPU grand public ?
2. Avec un guidage w, chaque pas d'échantillonnage appelle le U-Net deux fois (avec et sans texte). Combien d'appels pour une image en 50 pas ? Que propose la distillation (chapitre 28) pour réduire ce coût ?
3. Pourquoi garde-t-on les matrices LoRA en float32 alors que le U-Net est en float16 ? Que se passerait-il pour des mises à jour de l'ordre de 10⁻⁴ × gradient en float16 ?
4. Le LoRA a appris un style à partir d'une seule légende répétée. Que faudrait-il changer dans les données pour qu'il apprenne aussi à distinguer les espèces (champ `name` du jeu de données) ?

## Pour aller plus loin

- Comparer les rangs 1, 4 et 16 (`rang`) : taille du fichier, temps par pas, et effet sur le phare.
- Remplacer DDIM par `DPMSolverMultistepScheduler` (diffusers) et refaire la partie 1a.
- Articles : Rombach et al., diffusion latente, [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) ; Ho et Salimans, guidage sans classifieur, [arXiv:2207.12598](https://arxiv.org/abs/2207.12598) ; Song, Meng et Ermon, DDIM, [arXiv:2010.02502](https://arxiv.org/abs/2010.02502) ; Hu et al., LoRA, [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).
