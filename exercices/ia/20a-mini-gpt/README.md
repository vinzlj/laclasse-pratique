# Exercice 20a : le Transformer, de l'attention en NumPy au mini-GPT

| | |
|---|---|
| **Chapitre lié** | [20. Le Transformer (2017)](https://laclasse.poneyponey.fr/ia/p07-transformer/20-transformer/) |
| **Exécution** | locale, MPS (GPU Apple) ; partie A en NumPy sur CPU |
| **Commande** | `make ex NUM=20a` (éditeur), `make ex NUM=20a MODE=run` (complet), `MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 4 min 36 s (processus entier, attente du verrou exclue), dont 2 min 50 s d'entraînement du mini-GPT (1 500 pas, 113 ms par pas), 39 s pour l'ablation sans masque et 60 s pour l'ablation sans position ; smoke : 12 s (M1 Pro, MPS, 15/10/2026) |
| **Travail estimé** | ~3 h |
| **Seed** | 42 (`seed_everything(42)`, `torch.manual_seed(42)` avant chaque modèle, lots tirés par un `torch.Generator` de graine 42, lots d'évaluation fixes). Sur MPS : même graine = mêmes courbes à quelques millièmes près, pas au bit près |
| **Données** | TinyShakespeare (1,1 Mo), téléchargé une fois dans `data/input.txt` par `ia_commun.donnees.charger_tinyshakespeare()` |
| **Produit** | `data/checkpoints/mini-gpt-shakespeare.pt` (≈ 13 Mo, ignoré par git), rechargé par les exercices des chapitres 28 et 38 |

## Objectif

Écrire l'attention softmax(QKᵀ/√d_k)·V, le masque causal, LayerNorm, RMSNorm et un bloc Transformer complet en NumPy en les vérifiant contre PyTorch, puis entraîner un mini-GPT caractère par caractère sur TinyShakespeare, le comparer à des modèles de comptage, lire ses cartes d'attention et mesurer ce que coûte le retrait du masque causal ou des positions.

## Ce que vous allez faire

**A. L'attention à la main (NumPy, CPU)**

1. **Exemple de quatre tokens**, « le chat noir dort », avec Q, K, V écrits à la main (d_k = 4, dimensions *nom, adjectif, déterminant, verbe*) : scores, poids, sortie, avec et sans masque causal ; comparaison avec `F.scaled_dot_product_attention`. Mêmes nombres que la démo `Attention` du chapitre.
2. **Le rôle de √d_k** : écart-type de q·k et poids de la clé la plus forte pour d_k = 4 à 1 024, avec et sans division.
3. **LayerNorm et RMSNorm** : exemple à la main sur (2, 4, 6, 8), puis comparaison avec `torch.nn.LayerNorm` et `torch.nn.RMSNorm`.
4. **Un bloc Transformer pré-normalisé complet** (attention causale 4 têtes, MLP 256 → 1 024 → 256 avec GELU, connexions résiduelles) réécrit en NumPy, avec les poids d'un `ia_commun.mini_gpt.Bloc` initialisé au hasard.
5. **Équivariance par permutation** : sans masque ni position, mélanger les tokens mélange la sortie de la même façon ; le masque causal casse cette symétrie.
6. **Encodage sinusoïdal** de 2017 : le produit scalaire PE(p)·PE(p + k) ne dépend que de k, et PE(p + k) est une rotation de PE(p) (l'idée de RoPE).

**B. Le mini-GPT (MPS)**

7. Tokeniseur caractère par caractère (65 symboles), 90 % / 10 % entraînement / validation ; pertes des modèles uniforme, unigramme et bigramme sur la validation.
8. Entraînement de `ia_commun.mini_gpt.MiniGPT` (4 couches, 4 têtes, 256 dimensions, contexte 128, dropout 0,1 ; 3 164 672 paramètres hors positions) : AdamW (β₂ = 0,99, décroissance des poids 0,1), lots de 64 × 128 caractères, taux qui monte en 100 pas jusqu'à 2 × 10⁻³ puis décroît en cosinus jusqu'à 2 × 10⁻⁴, 1 500 pas, gradient écrêté à 1 ; évaluation tous les 100 pas sur 20 lots fixes.
9. Sauvegarde du checkpoint, génération de 500 caractères (température 0,8).
10. Cartes d'attention des 16 têtes sur une réplique de validation (« BAPTISTA: / Mistake me not; I speak but as I find. », 48 caractères) ; profil de chaque tête.

**C. Ablations et coût**

11. Sans masque causal (300 pas) : perte d'entraînement contre perte de la seule dernière position.
12. Sans vecteurs de position (500 pas) : comparaison avec le modèle normal au même pas.
13. Temps d'une attention causale (lot 8, 4 têtes de 64 dimensions) pour T = 128 à 2 048.

Ce qui est optimisé (parties B et C) : l'**entropie croisée du token suivant**, ℒ = −(1/T) Σₜ log p_θ(x_{t+1} | x_{≤t}), moyennée sur les 8 192 positions d'un lot ; la perplexité vaut exp(ℒ). La partie A n'optimise rien.

## Résultat attendu

En mode complet, le notebook vérifie quatorze seuils et s'arrête avec une erreur si l'un d'eux n'est pas atteint. Valeurs obtenues le 15/10/2026 (seed 42, M1 Pro, MPS) :

| Seuil | Valeur obtenue |
|---|---|
| attention NumPy = PyTorch (écart < 10⁻¹²) | 3,3 × 10⁻¹⁶ |
| LayerNorm et RMSNorm NumPy = PyTorch (< 10⁻¹⁰) | 1,8 × 10⁻¹⁵ |
| bloc Transformer NumPy = PyTorch (< 10⁻¹⁰) | 2,5 × 10⁻¹⁴ |
| sans masque ni position : équivariance (< 10⁻¹²) | 8,9 × 10⁻¹⁶ |
| avec masque causal : plus d'équivariance (> 0,01) | 7,41 |
| encodage sinusoïdal : rotation exacte (< 10⁻¹²) | 1,3 × 10⁻¹⁵ |
| scores capturés cohérents avec le passage avant (< 10⁻³) | 3,6 × 10⁻⁶ |
| perte initiale ≈ ln 65 = 4,17 (à 0,15 près) | 4,230 |
| **perte de validation finale ≤ 1,60** | **1,513** (perplexité 4,54) |
| validation < bigramme − 0,8 | 1,513 contre 2,482 |
| sans masque : perte d'entraînement < 0,5 | 0,023 |
| sans masque : perte de la dernière position > 2,0 | 2,708 |
| modèle normal : perte de la dernière position ≤ 1,70 | 1,529 |
| sans position : validation plus haute qu'avec, au pas 500 | 2,115 contre 1,766 |

Autres valeurs mesurées :

| Mesure | Valeur |
|---|---|
| Exemple jouet, sans masque | chaque token met 0,711 sur sa cible (e²/(e² + 3)) et 0,096 sur les trois autres ; sortie de « dort » : (animal 0,711 ; noir 0,096) |
| Exemple jouet, avec masque | « le » : 1 sur lui-même ; « chat » : 0,5 / 0,5 entre « le » et lui-même ; « noir » : 0,787 sur « chat » |
| Écart-type de q·k (d_k = 4, 16, 64, 256, 1 024) | 2,02 ; 4,02 ; 7,95 ; 15,90 ; 32,19 (≈ √d_k) ; après division : 0,99 à 1,01 |
| Poids moyen de la clé la plus forte parmi 16 | sans division : 0,43 ; 0,68 ; 0,85 ; 0,92 ; 0,96 ; avec : 0,24 à 0,25 |
| LayerNorm(2, 4, 6, 8) ; RMSNorm(2, 4, 6, 8) | (−1,342 ; −0,447 ; 0,447 ; 1,342) ; (0,365 ; 0,730 ; 1,095 ; 1,461) |
| Paramètres d'un bloc (256 dimensions) | 786 944 |
| PE(0)·PE(10) = PE(50)·PE(60) (d = 64) | 21,0516 |
| Pertes de référence (validation) | uniforme 4,174 (perplexité 65,0) ; unigramme 3,347 (28,4) ; bigramme 2,482 (12,0) |
| Courbe de validation du mini-GPT | 4,23 (pas 0) ; 2,46 (100) ; 1,77 (500) ; 1,57 (1 000) ; 1,51 (1 500) ; entraînement 1,28 au pas 1 500 |
| Sans position (val aux pas 100, 300, 500) | 2,47 ; 2,37 ; 2,12 (avec positions : 2,46 ; 1,94 ; 1,77) |
| Sans masque (pas 100, 200, 300) | entraînement 0,107 ; 0,024 ; 0,023 ; dernière position 3,62 ; 2,84 ; 2,71 |
| Têtes de la couche 1 (sur la réplique de validation) | tête 1 : 90 % du poids sur le caractère précédent ; tête 2 : 89 % deux caractères avant ; tête 3 : 56 % précédent + 22 % deux avant ; tête 4 : 30 % deux avant + 41 % trois avant |
| Couches 2 à 4 | attention plus diffuse (entropie moyenne 1,6 à 2,7 nats contre 0,1 à 1,6 en couche 1) ; couche 2, tête 1 : 45 % du poids sur les espaces et retours à la ligne |
| Attention causale, lot 8 × 4 têtes × 64 dim. (MPS) | T = 128 : 0,12 ms ; 256 : 0,22 ms ; 512 : 0,66 ms ; 1 024 : 2,18 ms ; 2 048 : 9,48 ms. Matrice des scores si elle était stockée en float32 : 2, 8, 32, 128, 512 Mio |

Texte généré (température 0,8, graine 42, début) :

```text
KING LEWIS XI:
No, my lord, I'll know not to the Volsces of France:
The dream a watering from the church,
That I ever her men, my lord love flesh by them and for the
presence for the both. What art dost?
```

Figures produites : `figures/courbes.png` ; résultats bruts pour le site : `figures/resultats.npz` (lu par `site/scripts/figures/ch20_*.py`, qui écrit aussi les données de la démo `Attention`).

Recharger le modèle ailleurs :

```python
from ia_commun import checkpoints_dir
from ia_commun.mini_gpt import MiniGPT
modele, infos = MiniGPT.charger(checkpoints_dir() / "mini-gpt-shakespeare.pt", device="mps")
# infos["caracteres"] : les 65 caractères, dans l'ordre des identifiants
# infos["val_loss"], infos["pas"], infos["historique"], infos["part_train"] (0,9)
```

## Ce qu'il faut observer

- **L'attention n'est qu'une moyenne pondérée.** Dans l'exemple jouet, tout se calcule à la main : un score q·k de 4, divisé par √4, donne 2 ; e²/(e² + 3) = 0,711. Le masque ne change rien à la formule, il remplace des scores par −∞.
- **Sans √d_k, le softmax sature.** L'écart-type de q·k vaut √d_k : à d_k = 64 (une tête du Transformer de 2017), la clé gagnante prend 85 % du poids au hasard, avant tout apprentissage. Diviser par √d_k ramène la clé la plus forte à environ 24 % sur 16 clés, quelle que soit la dimension.
- **Un bloc Transformer tient en quinze lignes de NumPy** et reproduit PyTorch à 10⁻¹⁴ près : LayerNorm, trois projections, attention par tête, concaténation, projection, résidu, MLP, résidu.
- **Le Transformer bat le comptage dès 100 pas** (2,46 contre 2,48 pour le bigramme) et finit à 1,51 : il hésite en moyenne entre 4,5 caractères, contre 12 pour le bigramme. L'écart entre entraînement (1,28) et validation (1,51) montre que le modèle commence à apprendre le corpus par cœur.
- **Toutes les positions sont prédites en un seul passage** : 8 192 prédictions par pas, en 113 ms. C'est le masque causal qui rend cela honnête.
- **Sans masque, le modèle triche** : la perte d'entraînement tombe à 0,02 en 200 pas (il recopie le caractère suivant, visible dans l'entrée), mais sur la dernière position, sans réponse visible, elle reste à 2,71, pire que le bigramme. Un modèle de langage ne voit jamais le futur au moment de générer.
- **Sans vecteurs de position, le modèle apprend quand même, plus lentement** (2,12 contre 1,77 au pas 500, mais mieux que le bigramme) : avec le masque causal, la position i fait une moyenne sur i éléments, ce qui laisse filtrer une information d'ordre. Sans masque ni position, l'attention ne verrait qu'un sac de caractères (partie A5).
- **Les têtes se spécialisent seules** : en couche 1, l'une regarde le caractère précédent, une autre deux caractères avant, une autre trois ; le modèle reconstruit un contexte de type n-gramme. Les couches suivantes ont des motifs plus diffus et plus difficiles à nommer.
- **Le coût est quadratique** : de T = 1 024 à 2 048, le temps passe de 2,2 à 9,5 ms (× 4,3) et la matrice des scores de 128 à 512 Mio.

## Questions de réflexion

1. Dans l'exemple jouet, que vaut le poids de « chat » vu par « dort » si l'on ne divise pas par √d_k ? Et si l'on multiplie toutes les requêtes par 10 ? Que devient alors le gradient vers les autres tokens ?
2. LayerNorm normalise sur les 256 composantes d'un token, la normalisation par lot (chapitre 12) sur les exemples d'un lot. Pourquoi la seconde est-elle mal adaptée à la génération, où l'on traite un seul texte, un token à la fois ?
3. Le modèle sans masque a une perte d'entraînement de 0,02. Pourquoi ce chiffre ne dit-il rien de sa capacité à générer du texte ? Quelle mesure le dit ?
4. Les têtes de la couche 1 regardent 1, 2 ou 3 caractères en arrière. Pourquoi ce motif est-il utile pour un modèle caractère par caractère ? Qu'attendriez-vous d'un modèle qui travaille sur des sous-mots (chapitre 19) ?

## Pour aller plus loin

- Remplacer `nn.LayerNorm` par `nn.RMSNorm` dans un `Bloc` (copie locale du module) et comparer la courbe de validation.
- Remplacer les vecteurs de position appris par l'encodage sinusoïdal de la partie A6 (fixe, non appris) : la perte finale change-t-elle ?
- Passer `n_pas` à 3 000 : la validation baisse-t-elle encore, ou l'écart avec l'entraînement se creuse-t-il ?
- Exercice 20b (Colab, facultatif) : la configuration nanoGPT complète (6 couches, 384 dimensions, contexte 256, 5 000 pas).
