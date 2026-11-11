# Exercice 28b : servir un modèle de langage à moindre coût

| | |
|---|---|
| **Chapitre lié** | [28. LLM modernes : pré-entraînement, échelle, efficacité](https://laclasse.poneyponey.fr/ia/p08-llm/28-llm-modernes/) |
| **Exécution** | locale, MPS (GPU Apple) ; partie A en NumPy, partie B aussi sur CPU |
| **Commande** | `make ex NUM=28b` (éditeur), `make ex NUM=28b MODE=run` (complet), `MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 2 min 27 s (processus entier, attente du verrou exclue), dont 24 s de réentraînement GQA/MQA et 1 min 24 s pour les trois élèves de la distillation ; smoke : 10 s (M1 Pro, MPS, 16/10/2026). Si le checkpoint de l'exercice 20a manque, ajouter ≈ 3 min pour le réentraîner |
| **Travail estimé** | ~2 h 30 |
| **Seed** | 42 (`seed_everything(42)`, `torch.manual_seed(42)` avant chaque modèle, générateurs de graine 42 pour les lots et les tirages, 200 fenêtres de validation fixes). Sur MPS : même graine = mêmes valeurs à quelques millièmes près, pas au bit près ; les durées varient d'un lancement à l'autre |
| **Données** | TinyShakespeare (`data/input.txt`) et le checkpoint `data/checkpoints/mini-gpt-shakespeare.pt` de l'exercice 20a (réentraîné avec la même recette s'il manque) |

## Objectif

Appliquer, sur le mini-GPT de l'exercice 20a (3,2 millions de paramètres), les techniques qui rendent l'inférence des grands modèles abordable : attention par blocs, cache KV, GQA/MQA, réglages d'échantillonnage, quantification, distillation et décodage spéculatif, en mesurant à chaque fois ce que l'on gagne et ce que l'on perd.

## Ce que vous allez faire

1. **A. Attention par blocs (NumPy).** Softmax « en ligne » de FlashAttention : maximum courant, somme courante, rééchelonnement par e^(m_ancien − m_nouveau). Comparaison avec l'attention standard, T = 1 024, d = 64, blocs de 128.
2. **B. Cache KV écrit à la main** (`passe_cache`) avec les poids du mini-GPT : vérification contre le passage complet (pas à pas et préremplissage), puis génération gloutonne de 127 caractères avec et sans cache, sur MPS et sur CPU (meilleur de 3 essais). Calcul de la taille du cache de Qwen3-0.6B (configuration publiée : 28 couches, 8 têtes K/V de 128 dimensions, bfloat16) en GQA, en MHA et en MQA.
3. **C. GQA et MQA par « uptraining ».** Moyenne des projections K et V des têtes d'un même groupe (recette de l'article GQA), puis 75 pas de 64 fenêtres (5 % des 1 500 pas de l'exercice 20a) en maintenant l'égalité des têtes d'un groupe ; comparaison avec le modèle d'origine réentraîné de la même façon.
4. **D. Échantillonnage.** Température, top-k et top-p écrits à la main (`filtrer_logits`) ; distribution du caractère suivant après « KING RICHARD III:⏎My lord, I » ; 32 débuts de répliques de validation, 96 caractères générés par réglage (9 réglages), avec le cache KV.
5. **E. Quantification simulée** de tous les poids 2D (couches linéaires, plongements) : float16, bfloat16, int8 et int4 avec une échelle par tenseur ou par ligne, int4/int3/int2 par groupes de 32, ternaire de BitNet b1.58 (échelle = moyenne des |w|) ; perte sur les mêmes 200 fenêtres.
6. **F. Distillation.** Un élève de 2 couches × 64 dimensions (98 624 paramètres hors plongements, 32 fois moins que le maître), 1 500 pas de 32 fenêtres, trois fois : sans maître ; KL pure vers le maître (τ = 1, λ = 1) ; recette de Hinton à τ = 2 (λ = 0,9 de KL × τ², 0,1 d'entropie croisée).
7. **G. Décodage spéculatif** (Leviathan et al. ; Chen et al.) : l'élève propose γ = 2, 4 ou 6 caractères, le maître les vérifie en un passage, acceptation-rejet min(1, p/q), tirage dans max(0, p − q) au premier refus. Exactitude vérifiée sur 400 000 tirages ; 8 débuts de validation, 100 caractères chacun, avec l'élève distillé puis l'élève sans maître comme brouillon.

Ce qui est optimisé : parties C et F, l'**entropie croisée du token suivant** (chapitre 20) ; partie F, en plus, la **perte de distillation** λ·τ²·KL(softmax(z_maître/τ) ‖ softmax(z_élève/τ)) + (1 − λ)·CE. Les parties A, B, D, E et G n'entraînent rien.

## Résultat attendu

En mode complet, le notebook vérifie quatorze seuils et s'arrête avec une erreur si l'un d'eux n'est pas atteint. Valeurs obtenues le 16/10/2026 (seed 42, M1 Pro, MPS) :

| Seuil | Valeur obtenue |
|---|---|
| attention par blocs = standard (écart < 10⁻¹²) | 6,7 × 10⁻¹⁶ |
| cache KV = passage complet (écart < 10⁻³) | 9,5 × 10⁻⁶ (pas à pas), 3,8 × 10⁻⁶ (préremplissage) |
| maître : perte de validation ≤ 1,60 | 1,510 (perplexité 4,53) |
| cache KV : génération ≥ 1,5 fois plus rapide sur CPU | × 4,00 (422 ms → 106 ms) |
| GQA 2 têtes K/V après uptraining : au plus 0,05 au-dessus du MHA réentraîné | 1,526 contre 1,509 |
| MQA : la perte baisse pendant l'uptraining | 2,621 → 1,611 |
| glouton : diversité plus faible que top-p 0,9 | 89,6 % contre 100 % de 3-grammes distincts |
| τ = 2 : moins de mots existants que top-p 0,9 | 42,3 % contre 92,9 % |
| int8 par ligne : au plus 0,01 au-dessus du float32 | 1,510 (écart < 0,001) |
| int4 par groupes de 32 meilleur que int4 par tenseur | 1,512 contre 1,542 |
| ternaire après coup : au moins 0,5 au-dessus du float32 | 2,894 |
| **distillation (τ = 1) : élève au moins 0,02 meilleur que sans maître** | **1,803 contre 1,856** |
| spéculatif : distribution exacte (variation totale < 0,01) | 0,0023 (brouillon seul : 0,148) |
| spéculatif γ = 4, brouillon distillé : ≥ 2 tokens par appel au maître | 3,16 |

Autres valeurs mesurées :

| Mesure | Valeur |
|---|---|
| Mémoire des scores (T = 1 024, float64) | 8 Mio en standard, 128 Kio au plus par blocs (÷ 64) |
| Génération de 127 caractères | MPS : 179 ms sans cache, 133 ms avec (× 1,35) ; CPU : 422 ms sans, 106 ms avec (× 4,0). Mêmes caractères avec et sans cache |
| Cache KV du mini-GPT, 128 tokens, float32 | 1 024 Kio |
| Cache KV de Qwen3-0.6B (bfloat16) | MHA (16 têtes K/V) : 224 Kio par token, 7,0 Gio pour 32 768 tokens ; GQA (8, la vraie configuration) : 112 Kio, 3,5 Gio ; MQA (1) : 14 Kio, 0,44 Gio. Lot de 16 séquences de 32 768 tokens : 112, 56 et 7 Gio |
| GQA/MQA juste après la moyenne → après 75 pas | 4 têtes K/V : 1,510 → 1,509 ; 2 : 1,964 → 1,526 ; 1 : 2,621 → 1,611 |
| Distribution après « My lord, I » | entropie 2,27 / 2,66 / 2,95 nats à τ = 0,5 / 1 / 1,5 ; caractère le plus probable (« w ») 0,20 / 0,13 / 0,10 ; noyau top-p 0,9 : 8 / 12 / 15 caractères |
| Échantillonnage : mots existants, 3-grammes distincts, perte du modèle sur ses choix | glouton 97,8 %, 89,6 %, 0,68 ; τ = 0,5 97,3 %, 99,6 %, 0,86 ; τ = 0,8 94,3 %, 100 %, 1,02 ; τ = 1 88,5 %, 100 %, 1,19 ; τ = 1,5 68,2 %, 100 %, 1,96 ; τ = 2 42,3 %, 100 %, 3,00 ; top-k 5 95,6 %, 100 %, 0,97 ; top-p 0,9 92,9 %, 100 %, 1,06 ; τ = 1,5 + top-p 0,9 79,8 %, 100 %, 1,47 |
| Quantification (perte, taille) | float32 1,510, 12,8 Mo ; float16 et bfloat16 1,510, 6,4 Mo ; int8 1,510, 3,2 Mo ; int4 par tenseur 1,542, 1,6 Mo ; int4 par ligne 1,520, 1,6 Mo ; int4 groupes de 32 1,512, 1,8 Mo ; int3 groupes de 32 1,544, 1,4 Mo ; int2 groupes de 32 2,414, 1,0 Mo ; ternaire 2,894, 0,63 Mo |
| Distillation, perte de validation aux pas 300 / 750 / 1 500 | sans maître 2,363 / 2,062 / 1,856 ; τ = 1 2,322 / 1,965 / 1,803 ; τ = 2, λ = 0,9 2,415 / 2,095 / 1,865 |
| Spéculatif, brouillon distillé (γ = 2 / 4 / 6) | acceptation 66 / 55 / 43 % ; 2,29 / 3,16 / 3,46 tokens par appel au maître |
| Spéculatif, brouillon sans maître (γ = 2 / 4 / 6) | acceptation 59 / 46 / 38 % ; 2,17 / 2,79 / 3,19 tokens par appel au maître |
| Spéculatif, temps réel | 2,5 s pour 800 caractères sans brouillon ; 2,9 à 4,0 s avec (× 0,63 à 0,89) : ici plus lent, voir ci-dessous |
| Taux d'acceptation théorique Σ min(p, q), contexte « My lord, I » | 0,852 |

Exemples (début de « PETRUCHIO:⏎Nay, hear you, Kate: ») :

```text
glouton   : the state of York,⏎And the state of the season of the country's death.
top-p 0,9 : if you have hereaftern'd their counsels⏎A day, gentlemen; and he had faired
τ = 2     : voll! O;' byFAMURLE::⏎Spek his contrnhupon, i, ge! turje;daudies I will
```

Figures produites : `figures/resultats.png` ; résultats bruts pour le site : `figures/resultats.npz` (lu par `site/scripts/figures/ch28_cache_kv.py`, `ch28_echantillonnage.py`, `ch28_quantification.py` et `ch28_distillation.py`).

## Ce qu'il faut observer

- **FlashAttention ne fait aucune approximation.** Le résultat par blocs égale l'attention standard à 10⁻¹⁵ près ; seule la mémoire change (128 Kio de scores au lieu de 8 Mio). Sur GPU, c'est ce qui permet de tout garder dans la mémoire rapide de chaque cœur.
- **Le cache KV donne exactement les mêmes logits** et rend chaque pas indépendant de la longueur du contexte. Sur CPU, sans cache, le pas passe de 2 à plus de 4 ms à mesure que le contexte grandit ; avec, il reste sous 1 ms. Sur le GPU, un modèle si petit est limité par le coût fixe de chaque appel (×1,35 seulement) : le gain grandit avec la taille du modèle et du contexte.
- **Le cache KV coûte cher en mémoire, d'où GQA.** Pour Qwen3-0.6B, 32 768 tokens occupent 3,5 Gio de cache avec 8 têtes K/V, 7 Gio s'il en avait 16 : plus que les poids du modèle (≈ 1,2 Go en bfloat16). Moyenner les têtes K/V détruit d'abord le modèle (1,96 et 2,62), mais 5 % de réentraînement suffisent à revenir à 1,526 avec 2 têtes K/V. MQA (1 tête) reste plus loin (1,611) : c'est le compromis qui a fait adopter GQA.
- **La température règle un curseur qualité / diversité.** Le glouton produit des mots existants mais tourne en rond (« the state of… the state of… »). À τ = 2, plus de la moitié des « mots » n'existent pas. Top-k et top-p coupent la traîne de la distribution, qui contient les choix absurdes, et gardent 93 à 96 % de mots existants sans boucle. Top-p s'adapte : 8 candidats quand le modèle est confiant, 15 quand il hésite.
- **Ce qui compte en quantification, c'est le partage des échelles.** En 8 bits, rien ne bouge. En 4 bits, une échelle pour tout un tenseur coûte 0,032 nat ; une par groupe de 32 poids, 0,002, pour 0,5 bit de plus par poids. En dessous de 3 bits, le modèle s'effondre si on quantifie après coup ; BitNet b1.58 n'obtient ses 1,58 bit qu'en entraînant le modèle ainsi dès le départ.
- **Un élève apprend mieux des probabilités du maître que des seules réponses** (1,803 contre 1,856 au même nombre de pas) : la cible douce dit quels caractères étaient plausibles et dans quelle proportion. La température 2 de Hinton n'aide pas ici (1,865) : avec 65 caractères et un maître déjà peu confiant, adoucir encore donne trop de poids à la traîne.
- **Le décodage spéculatif est exact.** Sur 400 000 tirages, la distribution produite est à 0,002 de celle du maître, alors que le brouillon seul en est à 0,15. Chaque appel au maître produit 3,2 caractères au lieu d'un avec γ = 4 et le brouillon distillé, mieux qu'avec le brouillon non distillé. Mais en temps réel, c'est ici **plus lent** : le brouillon (sans cache KV) coûte presque autant qu'un appel au maître, car à cette taille c'est le coût fixe d'un appel qui domine. Le gain réel suppose un maître beaucoup plus coûteux que le brouillon, comme entre un modèle de 70 et de 1 milliard de paramètres.

## Questions de réflexion

1. Le cache KV fait passer le coût d'un pas de O(T) passages de token à un seul. Pourquoi le **préremplissage** d'un long prompt reste-t-il, lui, quadratique en sa longueur ? Que reste-t-il à faire à chaque pas une fois le cache en place ?
2. Calculez le cache KV de Qwen3-0.6B pour un lot de 16 conversations de 32 768 tokens. Pourquoi les serveurs d'inférence (vLLM) découpent-ils ce cache en pages plutôt que de réserver d'avance la longueur maximale ?
3. Pourquoi une échelle par groupe de 32 poids protège-t-elle mieux contre les valeurs extrêmes qu'une échelle par tenseur ? Combien de bits par poids coûte une échelle en float16 partagée par 32 poids ? par 128 ?
4. Le décodage spéculatif accepte un token avec la probabilité min(1, p/q). Montrez que, si le brouillon est identique au maître, tout est accepté et chaque appel produit γ + 1 tokens. Que se passe-t-il si le brouillon est uniforme ?

## Pour aller plus loin

- Ajouter un cache KV au brouillon dans `generer_speculatif` et remesurer le temps réel.
- Quantifier aussi les activations (8 bits par token) et observer l'effet sur la perte.
- Leviathan et al., « Fast Inference from Transformers via Speculative Decoding » (2022), https://arxiv.org/abs/2211.17192 ; Ainslie et al., « GQA » (2023), https://arxiv.org/abs/2305.13245 ; Hinton, Vinyals et Dean, « Distilling the Knowledge in a Neural Network » (2015), https://arxiv.org/abs/1503.02531.
