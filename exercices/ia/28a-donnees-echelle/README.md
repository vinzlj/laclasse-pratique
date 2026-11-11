# Exercice 28a : les données et l'échelle, en miniature

| | |
|---|---|
| **Chapitre lié** | [28. LLM modernes : pré-entraînement, échelle, efficacité](https://laclasse.poneyponey.fr/ia/p08-llm/28-llm-modernes/) |
| **Exécution** | locale, MPS (GPU Apple) ; partie A en NumPy sur CPU |
| **Commande** | `make ex NUM=28a` (éditeur), `make ex NUM=28a MODE=run` (complet), `MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 4 min 49 s (processus entier, attente du verrou exclue), dont 2 min 32 s pour les cinq modèles de la loi d'échelle (13 à 52 s chacun) et 1 min 58 s pour les trois modèles de la partie MoE ; smoke : 10 s (M1 Pro, MPS, 16/10/2026) |
| **Travail estimé** | ~2 h |
| **Seed** | 42 (`seed_everything(42)`, `torch.manual_seed(42)` avant chaque modèle, lots tirés par un `torch.Generator` de graine 42, lots de validation fixes, `numpy.random.default_rng(42)` pour les variantes et la fuite). Sur MPS : même graine = mêmes courbes à quelques millièmes près, pas au bit près |
| **Données** | TinyShakespeare (1,1 Mo), téléchargé une fois dans `data/input.txt` par `ia_commun.donnees.charger_tinyshakespeare()` |

## Objectif

Reproduire en miniature quatre gestes du pré-entraînement d'un LLM : dédupliquer un corpus (MinHash et LSH), détecter une contamination de l'ensemble de test (13-grammes), mesurer une loi d'échelle sur cinq modèles et s'en servir pour prédire le plus gros, puis entraîner un mélange d'experts avec et sans perte d'équilibrage.

## Ce que vous allez faire

**A. Les données (NumPy, CPU)**

1. **Déduplication exacte puis approximative.** Les 1 318 répliques d'entraînement de plus de 40 mots, plus 400 variantes (copies où une part tirée entre 0 et 30 % des mots est remplacée au hasard). Hachage exact du texte, puis MinHash écrit à la main avec les réglages de FineWeb : 5-grammes de mots, 112 fonctions de hachage h(x) = (a·x + b) mod (2³¹ − 1), 14 bandes de 8 valeurs (hachage sensible à la localité). Comparaison avec la vraie similarité de Jaccard et avec la courbe théorique 1 − (1 − J⁸)¹⁴.
2. **Contamination.** 100 répliques de validation (≥ 30 mots) sont recopiées à la fin des données d'entraînement ; un détecteur à 13-grammes de mots (méthode de GPT-3) doit les retrouver, et elles seules.

**B. Loi d'échelle (MPS)**

3. Cinq mini-GPT (`ia_commun.mini_gpt`, sans dropout) : 1 × 32, 2 × 48, 2 × 80, 3 × 96, 4 × 128 (couches × dimensions), soit N = 12 384 à 787 584 paramètres hors plongements. Même recette pour tous : 2 000 pas de 32 fenêtres de 128 caractères (8,2 millions de tokens), AdamW (β = 0,9 ; 0,95, décroissance des poids 0,1), taux qui monte en 100 pas jusqu'à 3 × 10⁻³ puis décroît en cosinus jusqu'à 3 × 10⁻⁴. Données d'entraînement **contaminées** par l'étape 2 ; validation propre (sans les répliques qui ont fui).
4. Ajustement de L(N) = E + A·N^−α sur les cinq pertes finales, puis sur les quatre plus petites seulement pour **prédire** la cinquième.
5. Courbes de perte en fonction du calcul C ≈ 6·N·D et « frontière » : à chaque budget, le meilleur des cinq modèles.
6. Perte de chaque modèle sur les 100 répliques qui ont fui et sur 100 répliques témoins.

**C. Mélange d'experts (MPS)**

7. Un mini-GPT de 2 couches × 64 dimensions où le MLP de chaque bloc devient 4 experts (même forme que le MLP) et un routeur linéaire top-1. Trois entraînements de 1 500 pas : dense, MoE sans perte d'équilibrage, MoE avec la perte du Switch Transformer (α = 10⁻²). Mesure de la part des tokens reçue par chaque expert.

Ce qui est optimisé : parties B et C, l'**entropie croisée du token suivant** (chapitres 01 et 20) ; partie C, en plus, la **perte d'équilibrage** α·N_e·Σ f_i·P_i (f_i : part des tokens envoyés à l'expert i, P_i : probabilité moyenne que le routeur lui donne). La partie A n'optimise rien : ce sont des algorithmes de comptage et de hachage.

## Résultat attendu

En mode complet, le notebook vérifie dix seuils et s'arrête avec une erreur si l'un d'eux n'est pas atteint. Valeurs obtenues le 16/10/2026 (seed 42, M1 Pro, MPS) :

| Seuil | Valeur obtenue |
|---|---|
| MinHash : erreur moyenne d'estimation de J ≤ 0,05 | 0,028 |
| LSH : ≥ 95 % des variantes de J ≥ 0,8 détectées | 100 % (25 sur 25) |
| LSH : ≤ 10 % des variantes de J < 0,5 détectées | 0 % (278 variantes) |
| contamination : rappel 100 %, aucune fausse alerte | 100 sur 100, 0 sur 86 |
| loi d'échelle : la perte baisse à chaque taille | 2,147 ; 1,891 ; 1,700 ; 1,638 ; 1,528 |
| **plus gros modèle : perte de validation ≤ 1,75** | **1,528** |
| **prédiction du plus gros modèle à 0,05 près** | **1,530 prédit, 1,528 observé** |
| fuite : écart témoins − fuite plus grand pour le plus gros modèle que pour le plus petit | 0,025 contre 0,115 |
| MoE avec équilibrage : aucun expert au-dessus de 40 % des tokens | 27 % au plus |
| MoE avec équilibrage meilleur que dense | 1,753 contre 1,850 |

Autres valeurs mesurées :

| Mesure | Valeur |
|---|---|
| Hachage exact | 23 doublons trouvés sur 1 718 documents : les 23 variantes où aucun mot n'a été tiré ; les 377 autres lui échappent |
| MinHash + LSH | 0,1 s pour 1 718 documents ; 72 paires candidates sur 1 474 903 possibles, toutes de vraies variantes |
| Détection par tranche de J (observé / théorie) | [0,5 ; 0,65[ : 14 % / 16 % ; [0,65 ; 0,8[ : 60 % / 65 % ; ≥ 0,8 : 100 % / 99 % |
| Contamination naturelle | aucune réplique de validation ne partage de 13-gramme avec l'entraînement avant la fuite |
| Ajustement sur les 5 modèles | L(N) = 0,887 + 5,87·N^−0,163 |
| Ajustement sur les 4 plus petits | E = 0,911, A = 5,98, α = 0,167 |
| Temps par pas (lot de 4 096 tokens) | 6,4 ; 10,6 ; 13,7 ; 19,4 ; 26,0 ms |
| Meilleur modèle à calcul fixé | C = 10¹¹ : N = 12 384 (D/N ≈ 109) ; C = 10¹² : N = 55 536 (D/N ≈ 54) ; C = 10¹³ : N = 332 448 (D/N ≈ 15) |
| Perte sur la fuite / sur les témoins | 2,148 / 2,172 ; 1,868 / 1,911 ; 1,688 / 1,759 ; 1,605 / 1,687 ; 1,476 / 1,591 |
| MoE : paramètres | dense 110 976 ; MoE 308 096 (× 2,8), même calcul par token |
| MoE : perte de validation finale | dense 1,850 ; sans équilibrage 1,799 ; avec 1,753 |
| MoE sans équilibrage : charge des 4 experts | couche 1 : 0,79 ; 0,19 ; 0,02 ; 0,01 ; couche 2 : 0,19 ; 0,30 ; 0,24 ; 0,27 |
| MoE avec équilibrage : charge des 4 experts | couche 1 : 0,22 ; 0,27 ; 0,24 ; 0,27 ; couche 2 : 0,26 ; 0,25 ; 0,24 ; 0,26 |

Figures produites : `figures/resultats.png` ; résultats bruts pour le site : `figures/resultats.npz` (lu par `site/scripts/figures/ch28_loi_echelle.py`, `ch28_minhash.py` et `ch28_moe.py`).

## Ce qu'il faut observer

- **Un doublon approximatif échappe au hachage exact.** Un seul mot changé suffit à changer toute l'empreinte : 377 variantes sur 400 passent. MinHash, lui, estime la similarité à 0,03 près avec 112 nombres par document, et LSH ne compare que les documents qui partagent une bande : 72 paires examinées au lieu de 1,5 million.
- **Le seuil de LSH est réglable et prévisible.** Avec 14 bandes de 8 valeurs, la probabilité de détection passe de presque 0 à J = 0,5 à presque 1 à J = 0,85 ; les points mesurés tombent sur la courbe 1 − (1 − J⁸)¹⁴. Plus de bandes abaissent le seuil, des bandes plus longues le rendent plus raide.
- **Une fuite se détecte, et elle fausse la mesure.** Le détecteur à 13-grammes retrouve les 100 répliques glissées dans l'entraînement sans fausse alerte. Sur ces répliques, le plus gros modèle a une perte de 1,476 contre 1,591 sur les témoins : un score mesuré sur des données contaminées surestime le modèle, et **d'autant plus que le modèle est gros** (écart de 0,025 pour le plus petit, 0,115 pour le plus gros), parce qu'un modèle plus grand mémorise davantage.
- **La perte suit une loi de puissance en N.** Ajustée sur les quatre plus petits modèles, elle prédit le cinquième à 0,002 près. C'est l'usage pratique des lois d'échelle : choisir la taille et le budget d'un grand entraînement à partir de petits.
- **À calcul fixé, le meilleur modèle n'est pas le plus gros.** Sur la figure de gauche, les courbes se croisent : à petit budget, un petit modèle qui voit plus de données gagne ; à gros budget, un plus gros modèle. Le rapport D/N des meilleurs modèles diminue quand le budget augmente (109, puis 54, puis 15) : chez nous, les données, répétées plus de 8 fois, ne sont pas illimitées, et chaque point intermédiaire est pris en cours de décroissance du taux (approximation déjà relevée par Hoffmann et al.).
- **Un MoE a plus de paramètres pour le même calcul par token**, et en profite (1,753 contre 1,850). Sans contrainte, le routeur s'effondre en partie : dans la première couche, un expert reçoit 79 % des tokens et deux experts ne servent presque à rien. La perte d'équilibrage ramène chaque expert entre 22 et 27 % et donne la meilleure perte.

## Questions de réflexion

1. Avec 14 bandes de 8 valeurs, quelle est la probabilité de détecter une paire de similarité 0,6 ? Comment changer les bandes (en gardant 112 valeurs) pour détecter les paires à partir de 0,5 ? Que coûte ce changement ?
2. Pourquoi le détecteur à 13-grammes ne produit-il aucune fausse alerte ici, alors qu'il en produirait beaucoup sur du code ou sur des textes juridiques ? Que se passerait-il avec des 5-grammes ?
3. L'exposant α ≈ 0,16 dit de combien baisse la part réductible de la perte (L − E, au-dessus de la constante E) quand on multiplie N par 10. Calculez-le à partir de la loi ajustée. Combien de paramètres faudrait-il, selon elle, pour atteindre 1,2 ? Pourquoi ne faut-il pas trop croire cette extrapolation ?
4. Dans la partie MoE, chaque expert calcule ici la sortie de tous les tokens (version pédagogique, plus rapide sur MPS). Où se ferait l'économie de calcul dans un vrai MoE, et quel nouveau problème pose alors la répartition des tokens entre plusieurs GPU ?

## Pour aller plus loin

- Augmenter `n_pas` pour les petits modèles seulement et refaire la frontière : la règle « D ≈ 20 N » de Chinchilla réapparaît-elle quand chaque modèle a son propre planning de taux d'apprentissage ?
- Remplacer le routage top-1 par un top-2 (Mixtral) : la perte et l'équilibre changent-ils ?
- Lee et al., « Deduplicating Training Data Makes Language Models Better » (2022, ACL), https://arxiv.org/abs/2107.06499 ; Penedo et al., « The FineWeb Datasets » (2024), https://arxiv.org/abs/2406.17557.
