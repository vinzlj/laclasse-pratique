# Exercice 29 : rappel associatif, attention contre mémoire de taille fixe

| | |
|---|---|
| **Chapitre lié** | [29. Au-delà de l’attention quadratique et de l’autorégressif](https://laclasse.poneyponey.fr/ia/p08-llm/29-au-dela-attention/) |
| **Exécution** | locale (MPS ; tourne aussi sur CPU, plus lentement) |
| **Commande** | `make ex NUM=29` (éditeur), `MODE=run` (complet), `MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 4 min 13 s (partie C, quatre modèles : 249 s, de 58 à 66 s chacun ; parties A et B en NumPy et chargement : quelques secondes) ; smoke : 8 à 11 s (M1 Pro, MPS, 16/10/2026, attente du verrou GPU exclue) |
| **Travail estimé** | ~2 h |
| **Seed** | 42 (`seed_everything(42)` ; même graine pour l’initialisation, `torch.manual_seed(42)` avant chaque modèle, et mêmes lots pour les quatre modèles) |
| **Données** | générées (séquences aléatoires de paires clé-valeur, vecteurs aléatoires), rien n’est téléchargé |

## Objectif

Mesurer ce que l’on gagne et ce que l’on perd en remplaçant l’attention (qui garde tout le passé dans son cache KV) par un **état de taille fixe** mis à jour à chaque token, comme le font l’attention linéaire, les modèles à espace d’états (SSM, Mamba) ou DeltaNet : un coût constant par token, contre une mémoire qui finit par déborder.

## Ce que vous allez faire

1. **Partie A1 (NumPy).** Calculer l’attention linéaire de deux façons : en **forme parallèle** (une matrice $n \times n$ masquée, comme l’attention) et en **forme récurrente** (un état $S_t = S_{t-1} + \phi(k_t) v_t^\top$ de $d \times d$ nombres). Même chose pour un SSM sélectif à oubli scalaire $S_t = a_t S_{t-1} + k_t v_t^\top$, dont la forme parallèle est une attention sans softmax multipliée par un **masque de décroissance** $L[t, s] = a_{s+1} \cdots a_t$ (la « dualité » de Mamba-2). Vérifier que les nombres sont identiques. (Convention : le notebook range l’état comme $k\,v^\top$ et lit $y_t = q_t^\top S_t$ ; le chapitre écrit la transposée, $v\,k^\top$ et $y_t = S_t q_t$. C’est le même calcul.)
2. **Partie A2 (NumPy).** Pour une tête de dimension 64, de $n = 256$ à $8\,192$ tokens : mesurer (avec `tracemalloc`) la mémoire de l’attention calculée sur toute la séquence, la comparer au cache KV et à l’état récurrent, puis chronométrer la production d’un token à la position $n$.
3. **Partie B1 (NumPy).** Ranger $N$ paires (clé, valeur) de vecteurs unitaires aléatoires dans une mémoire de $64 \times 64$ nombres, puis relire chaque clé, pour $N$ de 4 à 256. Quatre mémoires : attention softmax (garde tout), attention linéaire $S = \sum_i v_i k_i^\top$, la même avec $\phi = \mathrm{elu} + 1$, règle delta $S \leftarrow S + (v_i - S k_i) k_i^\top$.
4. **Partie B2 (NumPy).** Ranger 16 paires, puis donner une **nouvelle valeur** à 8 des clés : la mémoire renvoie-t-elle la nouvelle valeur, l’ancienne, ou un mélange ?
5. **Partie C (PyTorch).** Entraîner quatre petits modèles (2 blocs, 64 dimensions, environ 125 000 paramètres) sur une tâche de **rappel associatif à requêtes multiples** : la séquence `k1 v1 k2 v2 … kN vN` suivie de 16 clés déjà vues ; à chaque requête, il faut redonner la valeur associée (64 valeurs possibles, hasard 1,6 %). Seule la couche qui mélange les positions change :
   - **attention causale** (dimension des requêtes et clés : 16) ;
   - **attention linéaire** (Katharopoulos et al., 2020), état $16 \times 64$ ;
   - **SSM diagonal invariant** façon S4D (une convolution par un noyau fixe, 16 états par canal), état $64 \times 16$ ;
   - **SSM sélectif** façon Mamba-2 : $B_t$, $C_t$ et l’oubli $a_t$ calculés à partir du token, état $16 \times 64$.

   Les trois modèles récurrents ont donc le **même état** : 1 024 nombres par couche. Entraînement : 2 000 pas, lots de 64 séquences, $N$ tiré entre 4 et 128 à chaque pas, AdamW. Évaluation sur 512 séquences neuves par valeur de $N$.

Pour expérimenter sans modifier le fichier : `IA29_CONFIG='{"d_etat": 64}' make ex NUM=29 MODE=run` (toute clé de `config` ; `"modeles": ["SSM sélectif (Mamba-2)"]` n’entraîne que ce modèle, les seuils de la partie C sont alors sautés).

## Résultat attendu

Run complet (le notebook s’arrête avec une erreur si un seuil n’est pas atteint). Valeurs obtenues le 16/10/2026 :

**A1.** Écart maximal entre forme parallèle et forme récurrente : 4,4 × 10⁻¹⁶ (attention linéaire) et 1,8 × 10⁻¹⁵ (SSM sélectif), c’est-à-dire l’arrondi des flottants. Seuil : < 10⁻¹⁰.

**A2.** Une tête, $d = 64$, float32 :

| $n$ | 256 | 512 | 1 024 | 2 048 | 4 096 | 8 192 |
|---|---|---|---|---|---|---|
| attention sur toute la séquence (pic mesuré) | 0,4 Mio | 1,5 Mio | 6,0 Mio | 24 Mio | 96 Mio | 384 Mio |
| forme récurrente (pic mesuré) | 67 Kio | 66 Kio | 66 Kio | 66 Kio | 66 Kio | 66 Kio |
| cache KV en génération ($2 \times n \times 64$ float32) | 128 Kio | 256 Kio | 512 Kio | 1 Mio | 2 Mio | 4 Mio |
| état récurrent ($64 \times 64$ float32) | 16 Kio | 16 Kio | 16 Kio | 16 Kio | 16 Kio | 16 Kio |
| un token, attention (relire $n$ clés et valeurs) | 12,8 µs | 22,5 µs | 39,9 µs | 80,7 µs | 308 µs | 651 µs |
| un token, récurrence (mettre à jour l’état) | 4,1 µs | 4,0 µs | 4,0 µs | 4,0 µs | 4,0 µs | 3,9 µs |

Le pic de l’attention complète vaut 1,5 fois la matrice $n \times n$ de float32 : la matrice elle-même ($8\,192^2 \times 4$ octets = 256 Mio) plus le masque causal (deux tableaux booléens de 64 Mio). Le pic de la forme récurrente (66 Kio) comprend l’état et les petits tableaux temporaires d’un pas. Les temps (meilleur de 7 séries de 50 pas, NumPy sur CPU) varient d’une machine et d’un lancement à l’autre ; seul l’ordre de grandeur compte.

Seuils : la mémoire de l’attention complète croît comme $n^2$ (× 1 024 quand $n$ est multiplié par 32), celle de la forme récurrente reste constante, et à 8 192 tokens un pas d’attention coûte au moins 3 fois un pas récurrent.

**B1.** Mémoire de $64 \times 64$ nombres, 20 tirages par valeur de $N$ ; cosinus moyen entre valeur lue et bonne valeur (précision entre parenthèses : la bonne valeur est-elle la plus proche parmi les $N$ rangées ?) :

| $N$ | 4 | 16 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| attention softmax | 1,00 (100 %) | 1,00 (100 %) | 1,00 (100 %) | 1,00 (100 %) | 1,00 (100 %) |
| attention linéaire | 0,97 (100 %) | 0,90 (100 %) | 0,71 (100 %) | 0,58 (98,2 %) | 0,45 (76,4 %) |
| linéaire, $\phi = \mathrm{elu} + 1$ | 0,65 (91,2 %) | 0,35 (21,2 %) | 0,18 (4,1 %) | 0,13 (2,0 %) | 0,09 (0,9 %) |
| règle delta | 0,99 (100 %) | 0,93 (100 %) | 0,68 (90,5 %) | 0,45 (53,6 %) | 0,25 (26,7 %) |

Seuils : la softmax retrouve tout ; à 256 paires, la mémoire linéaire a un cosinus inférieur à 0,6 (0,45) ; avec $\phi = \mathrm{elu} + 1$, la précision tombe sous 50 % dès 64 paires (4,1 %).

**B2.** Après réécriture de 8 clés : la règle delta renvoie la nouvelle valeur (cosinus 0,95 avec la nouvelle, 0,03 avec l’ancienne, 100 % de nouvelles valeurs retrouvées) ; l’attention linéaire renvoie un mélange à parts égales (cosinus 0,67 avec la nouvelle, 0,65 avec l’ancienne). Seuil : cosinus avec la nouvelle valeur > 0,9 pour la règle delta, < 0,9 pour la somme.

**C.** Précision sur les requêtes, selon le nombre de paires $N$ :

| $N$ (longueur de séquence $2N + 16$) | 4 | 8 | 16 | 32 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|
| attention causale | 99,9 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % |
| attention linéaire | 91,3 % | 81,6 % | 66,6 % | 46,0 % | 26,7 % | 20,2 % | 15,9 % |
| SSM diagonal invariant (S4D) | 26,9 % | 16,1 % | 10,1 % | 6,7 % | 5,2 % | 4,2 % | 3,7 % |
| SSM sélectif (Mamba-2) | 86,1 % | 94,7 % | 92,4 % | 80,7 % | 55,4 % | 41,2 % | 32,6 % |

Perte finale (moyenne des 50 derniers pas, entropie croisée, hasard : ln 64 ≈ 4,16) : 0,002 (attention), 1,89 (linéaire), 3,87 (SSM invariant), 1,44 (SSM sélectif). Paramètres : 124 865 (attention et attention linéaire), 127 105 (SSM invariant), 125 251 (SSM sélectif). Sur MPS, deux lancements ont donné les mêmes précisions à 2 points près. Les nombres de cette page et `figures/resultats.npz` viennent du run du 16/10/2026 à 12 h 53 ; `figures/ssm-vs-attention.png` a été régénérée par un second run complet le même jour (graduations des axes corrigées), dont seul le SSM sélectif diffère, d’au plus 0,3 point (86,4 / 94,4 / 92,2 / 80,5 / 55,7 / 41,1 / 32,8 %).

Seuils : attention ≥ 99 % pour tout $N$ ; SSM invariant ≤ 40 % partout ; SSM sélectif ≥ 85 % pour au moins un $N \le 16$ ; attention linéaire et SSM sélectif perdent chacun au moins 20 points entre leur meilleur $N$ et $N = 128$.

Figures produites : `figures/ssm-vs-attention.png` (les trois parties), `figures/resultats.npz` (lu par les figures du site, `site/scripts/figures/ch29_*.py`).

## Ce qu’il faut observer

- **Mêmes nombres, deux coûts.** Les formes parallèle et récurrente donnent exactement la même sortie (A1). La forme parallèle sert à l’entraînement (tout est connu d’avance, le GPU calcule toutes les positions d’un coup) ; la forme récurrente sert à la génération (un token à la fois, état de taille fixe). C’est ce qui rend ces modèles intéressants : entraînement parallèle comme un Transformer, génération comme un RNN.
- **Le coût de l’attention.** La matrice $n \times n$ prend 256 Mio (384 Mio de pic avec le masque) à 8 192 tokens pour une seule tête et une seule couche (× 1 024 entre 256 et 8 192 tokens, comme $n^2$) ; le cache KV croît linéairement (4 Mio à 8 192 tokens) ; l’état récurrent reste à 16 Kio. Produire un token coûte 651 µs avec l’attention à la position 8 192, contre 3,9 µs avec la récurrence, quelle que soit la position.
- **Le prix de l’état fixe : le rappel.** En partie C, l’attention retrouve 100 % des valeurs jusqu’à 128 paires. L’attention linéaire et le SSM sélectif, avec 1 024 nombres d’état par couche, réussissent avec peu de paires puis s’effondrent à mesure que $N$ grandit : leur mémoire déborde. C’est le compromis « rappel contre débit » mis en évidence par Arora et al. (Zoology, 2023 ; Based, 2024).
- **Sans sélectivité, pas de rappel.** Le SSM diagonal invariant (façon S4/S4D) applique le même noyau de convolution quelle que soit l’entrée : il ne peut pas « décider » d’écrire une paire et d’en ignorer une autre. Il reste près du hasard (27 % à 4 paires, 4 % à 128), malgré autant de paramètres. Rendre $B$, $C$ et l’oubli dépendants du token (Mamba) change tout : 95 % à 8 paires.
- **Les clés doivent être presque orthogonales.** En partie B, une somme de produits extérieurs relit chaque valeur avec le bruit de toutes les autres, proportionnel au produit scalaire des clés : la qualité baisse doucement avec $N$ (cosinus 0,90 à 16 paires, 0,45 à 256). Avec $\phi = \mathrm{elu} + 1$, toutes les caractéristiques sont positives, donc toutes les clés se ressemblent : la lecture est déjà imprécise à 4 paires (cosinus 0,65) et brouillée à 16 (0,35). Le choix de $\phi$ compte.
- **La règle delta écrase au lieu d’empiler.** Elle n’écrit que l’erreur $v - S k$ : réécrire une clé remplace l’ancienne valeur (B2), là où la somme mélange les deux. Au-delà de $d = 64$ paires, elle oublie surtout les paires anciennes (c’est un effet de récence) : elle n’augmente pas la capacité, elle la gère mieux. Gated DeltaNet (2024) ajoute un oubli $a_t$ à cette règle.

## Questions de réflexion

1. En partie A2, le cache KV croît linéairement et non quadratiquement. Pourquoi la génération d’un Transformer n’a-t-elle pas besoin de la matrice $n \times n$ complète, et qu’est-ce qui reste quadratique sur l’ensemble d’une génération de $n$ tokens ?
2. Relancez la partie C avec `IA29_CONFIG='{"d_etat": 64}'` (état 4 fois plus grand). Comment évoluent les courbes de l’attention linéaire et du SSM sélectif ? Et celle de l’attention ?
3. Pourquoi le SSM diagonal invariant échoue-t-il, alors qu’un noyau de convolution de longueur 272 « voit » toute la séquence ? (Indice : il calcule la même combinaison pondérée des positions passées pour toutes les séquences.)
4. Les modèles hybrides (Jamba, Nemotron-H…) gardent quelques couches d’attention parmi beaucoup de couches récurrentes. Au vu de la partie C, à quoi servent ces quelques couches ?

## Pour aller plus loin

- Ajouter une cinquième couche de mélange qui applique la règle delta (DeltaNet) en forme récurrente, et la comparer à l’attention linéaire en partie C.
- Remplacer $\phi = \mathrm{elu} + 1$ par l’identité (avec une normalisation des clés) dans `AttentionLineaire` et observer l’effet sur la partie C.
- Articles : Katharopoulos et al., « Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention » (arXiv:2006.16236, 2020) ; Schlag, Irie & Schmidhuber, « Linear Transformers Are Secretly Fast Weight Programmers » (arXiv:2102.11174, 2021, règle delta) ; Gu, Gupta, Goel & Ré, « On the Parameterization and Initialization of Diagonal State Space Models » (arXiv:2206.11893, 2022, S4D) ; Gu & Dao, « Mamba: Linear-Time Sequence Modeling with Selective State Spaces » (arXiv:2312.00752, 2023) ; Dao & Gu, « Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality » (arXiv:2405.21060, 2024, Mamba-2) ; Arora et al., « Zoology: Measuring and Improving Recall in Efficient Language Models » (arXiv:2312.04927, 2023, tâche MQAR) ; Arora et al., « Simple linear attention language models balance the recall-throughput tradeoff » (arXiv:2402.18668, 2024) ; Yang, Kautz & Hatamizadeh, « Gated Delta Networks: Improving Mamba2 with Delta Rule » (arXiv:2412.06464, 2024).

## Écarts assumés par rapport aux articles

- **Décalage de token.** Chaque position reçoit aussi le plongement du token précédent (comme le « token shift » de RWKV ou les courtes convolutions placées avant le SSM dans Mamba). Sans lui, il faudrait une couche de plus pour associer chaque valeur à sa clé ; avec lui, la tâche se joue entièrement dans la couche de mélange, ce qui isole ce que l’on veut comparer.
- **SSM diagonal** : valeurs propres réelles, initialisation « S4D-Real » ($A_n = -(n+1)$), pas de discrétisation bilinéaire ni de partie imaginaire ; noyau appliqué par une matrice de Toeplitz $T \times T$ (S4 utilise une FFT).
- **SSM sélectif** : un oubli **scalaire** $a_t$ par couche (comme Mamba-2, mais une seule tête), pas de convolution ni de porte multiplicative ; calcul en forme duale quadratique (Mamba et Mamba-2 utilisent un balayage parallèle ou des blocs pour rester en $O(n)$). Biais initial de $\Delta$ à −7 ($a \approx 0{,}999$) et normalisation (LayerNorm) avant la projection de sortie, comme Mamba-2 : dans un premier essai (16/10/2026) avec un oubli plus fort au départ ($a \approx 0{,}98$, la mémoire s’efface alors en une cinquantaine de tokens) et sans cette normalisation, le modèle n’apprenait pas la tâche en 2 000 pas (28 % à 4 paires, comme le SSM invariant).
- **Attention linéaire** : une seule tête, $\phi = \mathrm{elu} + 1$ comme Katharopoulos et al. ; normalisation par $\phi(q)^\top z_t$.
- **Partie B** : la règle delta est appliquée avec un pas $\beta = 1$ et des clés unitaires (DeltaNet apprend $\beta_t$).
