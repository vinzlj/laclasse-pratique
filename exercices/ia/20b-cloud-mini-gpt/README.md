# Exercice 20b : le mini-GPT en taille nanoGPT (cloud, facultatif)

| | |
|---|---|
| **Chapitre lié** | [20. Le Transformer (2017)](https://laclasse.poneyponey.fr/ia/p07-transformer/20-transformer/) |
| **Exécution** | cloud : Google Colab gratuit, GPU T4 (secours : Kaggle, « GPU T4 x2 ») |
| **Commande** | ouvrir `notebook.ipynb` dans Colab, puis Exécution › Tout exécuter (pas de `make ex` : notebook cloud) |
| **Durée mesurée** | **non mesurée sur T4** (à mesurer au premier lancement : la cellule 5 affiche la durée). Repères : environ 3 min sur un A100 d'après le dépôt nanoGPT, environ 56 min sur le M1 Pro (MPS) d'après la phase 1 du cours. Seul un essai du code en miniature (2 couches de 64 dimensions, 20 pas, CPU) a tourné localement, le 15/10/2026 |
| **Travail estimé** | ~30 min (plus l'attente) |
| **Seed** | 42 (`random`, NumPy, PyTorch ; tirage des lots par un `torch.Generator` de graine 42) |
| **Données** | TinyShakespeare (1,1 Mo), téléchargé dans la session depuis le dépôt `karpathy/char-rnn` |
| **Coût** | 0 € (aucune carte bancaire demandée) |

## Objectif

Entraîner le même Transformer décodeur seul que l'exercice 20a, mais dans la configuration `train_shakespeare_char` de nanoGPT (6 couches, 6 têtes, 384 dimensions, contexte 256 caractères, dropout 0,2, 5 000 pas, ≈ 10,7 millions de paramètres), et mesurer ce que trois fois plus de paramètres, un contexte deux fois plus long et trois fois plus de pas apportent par rapport au modèle local (val = 1,513).

## Ce que vous allez faire

1. Vérifier le GPU (cellule 1).
2. Lire la configuration (cellule 2) : ce sont les valeurs de `config/train_shakespeare_char.py` de nanoGPT, sauf l'évaluation (50 lots au lieu de 200, pour aller plus vite). `COURT = True` limite l'entraînement à 500 pas pour tester.
3. Télécharger le texte, construire le tokeniseur caractère par caractère, séparer 90 % / 10 % (cellule 3), exactement comme l'exercice 20a.
4. Construire le modèle (cellule 4) : copie de `src/ia_commun/mini_gpt.py`, mêmes noms de poids.
5. Entraîner (cellule 5) : AdamW (β₂ = 0,99), taux qui monte pendant 100 pas puis décroît en cosinus de 10⁻³ à 10⁻⁴, gradient écrêté à 1, calcul en **float16** (`torch.autocast` + `GradScaler` : le T4 ne fait pas de bfloat16). Le meilleur modèle (val la plus basse) est gardé.
6. Tracer les courbes, générer 500 caractères, sauvegarder (cellules 6 à 8).

## Procédure pas à pas (Colab)

1. **Compte** : un compte Google suffit (Colab gratuit). Plan B Kaggle : compte Kaggle, vérification par téléphone probable pour activer GPU et Internet.
2. **Ouvrir le notebook** : sur https://colab.research.google.com, Fichier › Importer un notebook, onglet « Importer », puis choisir le fichier `notebook.ipynb` de ce dossier (copié au préalable sur votre ordinateur). Pas de badge ni de lien vers le dépôt : il est privé et le site du cours est public.
3. **Activer le GPU** : Exécution › Modifier le type d'exécution › GPU T4 › Enregistrer. La première cellule affiche `nvidia-smi` ; sans GPU, elle s'arrête avec un message clair.
4. **Token Hugging Face** : inutile (rien n'est téléchargé depuis le Hub).
5. **Exécution › Tout exécuter.** Aucune installation : torch, numpy et matplotlib sont déjà dans l'image Colab (torch n'est volontairement pas épinglé). Notez la durée affichée par la cellule 5.
6. **Sauvegarder avant la fin de session** : la cellule 8 télécharge `mini-gpt-shakespeare-grand.pt` (≈ 43 Mo) et `courbes-20b.png`. Une session gratuite dure 12 h au plus et s'arrête plus tôt en cas d'inactivité.
7. **Plan B Kaggle** : Create › New Notebook › File › Import Notebook ; Settings › Accelerator › **GPU T4 x2** (jamais P100) ; Settings › Internet › On ; Run All. Les fichiers restent dans `/kaggle/working` (onglet Output).
8. **Coût : 0 €.**

Le checkpoint se recharge en local, dans un notebook du cours, avec
`MiniGPT.charger("data/checkpoints/mini-gpt-shakespeare-grand.pt")` (`ia_commun.mini_gpt`) : il a le même format que celui de l'exercice 20a.

## Résultat attendu

Meilleure perte de validation **≤ 1,55** (la cellule 5 affiche un avertissement au-dessus). Référence publiée : 1,4697 dans le README de nanoGPT (A100, 200 lots d'évaluation, `torch.compile`). **Non mesuré sur T4** : seuil prudent, à confirmer au premier lancement et à reporter ici avec la durée.

## Vérification locale

- `nbformat.validate` et compilation de chaque cellule (lignes `!` ignorées) : OK, 9 cellules, le 15/10/2026.
- Aucune dépendance à installer ou à épingler : torch, numpy, matplotlib (image Colab).
- Code des cellules 2 à 8 exécuté sur CPU en miniature (2 couches, 2 têtes, 64 dimensions, 20 pas, CPU : autocast float16 inactif) : OK ; le checkpoint écrit se recharge avec `ia_commun.mini_gpt.MiniGPT.charger`. Ces chiffres ne vérifient que le code, pas le seuil.

## Ce qu'il faut observer

- **Le surapprentissage** : avec 1 million de caractères d'entraînement et 10,7 millions de paramètres, la perte d'entraînement continue de baisser alors que la validation plafonne puis remonte ; c'est pour cela que nanoGPT, et ce notebook, gardent le meilleur point et non le dernier (dropout 0,2 au lieu de 0,1).
- **Le gain par rapport au modèle local** : de val = 1,513 (exercice 20a, 3,2 M de paramètres, 1 500 pas) à moins de 1,50 attendu. Le texte généré garde mieux la forme d'une pièce (noms de personnages, longueur des répliques), mais reste dépourvu de sens.
- **La perplexité** passe d'environ 4,5 à environ 4,4 caractères équiprobables : un petit gain chiffré, très visible à la lecture.

## Questions de réflexion

1. Le contexte passe de 128 à 256 caractères : de combien augmente la taille de la matrice des scores d'attention de chaque tête ?
2. Pourquoi la perte de validation peut-elle remonter alors que la perte d'entraînement baisse encore ? Quelle solution, autre que garder le meilleur point, suggère le chapitre 12 ?
3. Pourquoi le calcul de la perte (dernière ligne de `MiniGPT.forward`) est-il converti en float32 même sous `autocast` ?

## Pour aller plus loin

- Passer `taille_bloc` à 512 et mesurer la durée par pas.
- Comparer à l'exercice 28 (inférence, échantillonnage) avec ce plus grand modèle.
