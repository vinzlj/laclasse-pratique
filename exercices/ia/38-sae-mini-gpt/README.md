# Exercice 38 : regarder dans la boîte noire

| | |
|---|---|
| **Chapitre lié** | [38. Interprétabilité](https://laclasse.poneyponey.fr/ia/p10-fiabilite/38-interpretabilite/) |
| **Exécution** | locale, MPS (GPU Apple) ; parties A et E sur CPU |
| **Commande** | `make ex NUM=38` (éditeur), `make ex NUM=38 MODE=run` (complet), `MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 2 min 46 s à 2 min 50 s (deux mesures, processus entier, attente du verrou exclue), dont 85 s pour le SAE et 54 s pour les deux Transformers « attention seule » ; smoke : 11 à 14 s (M1 Pro, MPS, 16/10/2026). Si le checkpoint de l'exercice 20a manque, ajouter ≈ 3 min pour le réentraîner |
| **Travail estimé** | ~2 h |
| **Seed** | 42 (`seed_everything(42)`, `torch.manual_seed(42)` avant chaque modèle, générateurs de graine 42 pour les lots ; fenêtres d'évaluation fixes). Sur MPS : même graine = mêmes valeurs à quelques millièmes près, pas au bit près |
| **Données** | TinyShakespeare (`data/input.txt`) et le checkpoint `data/checkpoints/mini-gpt-shakespeare.pt` de l'exercice 20a (réentraîné avec la même recette s'il manque) ; jeu `diabetes` fourni avec scikit-learn (442 patients, rien à télécharger) |

## Objectif

Essayer, en miniature et en mesurant, les principales façons d'ouvrir un réseau : la superposition dans un modèle jouet, les sondes linéaires, un autoencodeur parcimonieux (SAE) sur les neurones du mini-GPT de l'exercice 20a, la détection et l'apparition d'une tête d'induction, puis les attributions sur un modèle tabulaire (valeurs de Shapley exactes contre la bibliothèque `shap`, raccourci démasqué, gradients intégrés, LIME).

## Ce que vous allez faire

1. **A. Superposition** (Elhage et al., 2022). Modèle jouet x' = ReLU(WᵀWx + b), 5 caractéristiques, 2 dimensions, importance 0,9^i (i = 0 à 4, soit 0,9^(i−1) pour la caractéristique i = 1 à 5 du chapitre) ; chaque caractéristique vaut 0 avec la probabilité S, sinon un uniforme sur [0, 1]. Quatre entraînements (S = 0 ; 0,7 ; 0,9 ; 0,97), 4 000 pas de 1 024 exemples. Une caractéristique est « représentée » si ‖W_i‖ > 0,5.
2. **Passage avant instrumenté** (`passe_detaillee`) : même calcul que `MiniGPT.forward`, qui garde le flux résiduel après chaque couche, les poids d'attention de chaque tête et les 1 024 neurones de chaque MLP, et permet de remplacer les neurones d'une couche. Vérifié contre `forward` (écart 6 × 10⁻⁶).
3. **B. Sondes linéaires** (Alain et Bengio, 2016) : régression logistique sur le flux résiduel, couche par couche (0 = plongements), 400 fenêtres de validation × 24 positions, 75 % des fenêtres pour entraîner la sonde. Deux propriétés : « le caractère précédent est une voyelle » ; « on est dans la première ligne d'une réplique » (la ligne qui suit « ROMEO: »), sur les minuscules seulement. Témoin : le même réseau aux poids aléatoires.
4. **C. SAE** (Bricken et al., 2023 ; Cunningham et al., 2023) sur les 1 024 neurones du MLP de la deuxième couche : 2 048 caractéristiques, f = ReLU(W_e(x − b_d) + b_e), x̂ = W_d f + b_d, colonnes de W_d renormalisées à 1, λ = 3 (monté en 5 % des pas), Adam 10⁻³, 2 500 pas de 16 fenêtres de 128 caractères, activations calculées à la volée. Mesures sur 200 fenêtres de validation : variance expliquée, L0, caractéristiques mortes, perte récupérée, pureté des neurones et des caractéristiques, contextes des plus fortes activations de 8 caractéristiques tirées au hasard.
5. **D. Têtes d'induction** (Olsson et al., 2022). Sur le mini-GPT : séquences répétées (extraits de validation, minuscules aléatoires), perte sur la 1re et la 2de copie, score d'induction de chacune des 16 têtes. Puis un Transformer « attention seule » (64 dimensions, 4 têtes, ni MLP ni normalisation), 2 couches puis 1 couche, 4 000 pas de 64 séquences de 64 tokens aléatoires (vocabulaire 64) dont un segment de longueur L (6 à 24) est recopié juste après lui-même ; L et sa position changent à chaque lot.
6. **E. Attributions** sur le jeu `diabetes` (10 variables : âge, sexe, IMC `bmi`, tension `bp`, six mesures sanguines `s1` à `s6` ; cible : progression de la maladie à un an), gradient boosting de 200 arbres :
   - valeurs de Shapley **exactes** en NumPy (`shapley_exact` : les 1 024 coalitions, variables absentes tirées de 100 individus de fond) sur 40 patients de test, comparées à `shap.explainers.Exact` avec le masqueur `Independent` (même définition), efficacité (Σφ = f(x) − E[f]) vérifiée ;
   - un **raccourci** : une colonne « fuite » = cible standardisée + bruit (σ = 0,3), présente à l'entraînement et au test, sans lien avec la cible « en production » ;
   - **gradients intégrés** (Sundararajan et al., 2017) sur un petit MLP (10-32-32-1, tanh), référence = patient moyen, 64 pas (trapèzes), comparés à « gradient × entrée » ;
   - **LIME** écrit à la main (Ribeiro et al., 2016) : régression ridge pondérée par un noyau exponentiel sur des voisins gaussiens, 10 graines, avec 30 puis 500 voisins.

Ce qui est optimisé : A, erreur quadratique pondérée par l'importance ; B, entropie croisée (régression logistique) ; C, **MSE + λ·L1** du SAE ; D, entropie croisée du token suivant ; E, erreur quadratique (gradient boosting, MLP), ridge pondérée (LIME). Le mini-GPT n'est jamais modifié.

## Résultat attendu

En mode complet, le notebook vérifie seize seuils et s'arrête avec une erreur si l'un d'eux n'est pas atteint. Valeurs obtenues le 16/10/2026 (seed 42, M1 Pro, MPS) :

| Seuil | Valeur obtenue |
|---|---|
| `passe_detaillee` = `forward` (écart < 10⁻⁴) | 6,2 × 10⁻⁶ |
| A. S = 0 : au plus 2 caractéristiques représentées | 2 |
| A. S = 0,97 : les 5 représentées | 5 |
| B. voyelle précédente : ≥ 95 % à la meilleure couche, ≤ 75 % à la couche 0 | 99,5 % (couche 1) ; 69,1 % |
| B. première ligne de réplique : ≥ 80 % et ≥ réseau aléatoire + 5 points | 85,5 % contre 76,0 % |
| C. SAE : variance expliquée ≥ 80 % | 86,2 % |
| C. SAE : L0 ≤ 40 | 11,9 caractéristiques actives par token (sur 2 048) |
| C. SAE : perte récupérée ≥ 80 % | 89,9 % |
| C. pureté médiane : caractéristiques ≥ neurones + 20 points | 73 % contre 50 % |
| D. mini-GPT : aucun score d'induction ≥ 0,1 | 0,007 au mieux |
| D. 2 couches : perte 2de copie ≤ 1,5 et score d'induction ≥ 0,3 | 1,12 ; 0,76 |
| D. 1 couche : perte 2de copie ≥ 3 | 3,84 |
| E. Shapley exact = `shap` (écart < 10⁻⁶) | 3,8 × 10⁻¹³ |
| E. efficacité Σφ + E[f] = f(x) (écart < 10⁻⁶) | 1,1 × 10⁻¹³ |
| E. fuite : ≥ 50 % de l'importance SHAP et R² en chute d'au moins 0,3 | 81 % ; R² 0,926 → −0,580 |
| E. complétude : gradients intégrés ≤ 1 %, gradient × entrée ≥ 5 % | 0,01 % ; 98 % |

Autres valeurs mesurées :

| Mesure | Valeur |
|---|---|
| A. caractéristiques représentées selon S (0 / 0,7 / 0,9 / 0,97) | 2 / 4 / 5 / 5 ; perte 0,182 / 0,109 / 0,024 / 0,0025 ; biais des caractéristiques représentées ≈ 0 à S ≤ 0,7, ≈ −0,21 à S = 0,9, ≈ −0,26 à S = 0,97 |
| B. sonde « voyelle précédente », couches 0 à 4 | entraîné 69,1 / 99,5 / 98,9 / 97,4 / 96,1 % ; aléatoire 68,5 / 68,5 / 68,5 / 69,1 / 67,5 % |
| B. sonde « première ligne de réplique », couches 0 à 4 | entraîné 49,5 / 55,2 / 77,4 / 84,9 / 85,5 % ; aléatoire 49,1 / 76,0 / 75,9 / 75,8 / 76,0 % (29,7 % de positions positives) |
| C. neurones de la couche 2 | positifs sur 10,3 % des tokens ; pureté ≥ 90 % pour 8 % des neurones (34 % des caractéristiques) |
| C. entropie croisée (200 fenêtres de validation) | intact 1,518 ; neurones remplacés par le SAE 1,529 ; neurones à zéro 1,629 |
| C. caractéristiques mortes | 0 sur 2 048 (23 actives moins de 30 fois, exclues de la pureté) |
| C. trajectoire du SAE (pas 250 / 1 000 / 2 500) | L0 66 / 19 / 11 ; part de variance non expliquée 0,28 / 0,16 / 0,12 |
| D. mini-GPT, séquences répétées | extraits : perte 1,61 (1re copie) → 1,68 (2de) ; minuscules aléatoires : 7,73 → 7,70 |
| D. 2 couches, perte 2de copie aux pas 2 400 / 2 600 / 2 800 / 4 000 | 3,94 / 3,50 / 1,66 / 1,12 ; score d'induction 0,06 / 0,21 / 0,67 / 0,76 ; têtes de la couche 1 : 0,01 à 0,03 |
| E. importance moyenne \|φ\| (40 patients) | bmi 24,7 ; s5 22,1 ; bp 14,6 ; s3 8,8 ; s1 7,7 ; s6 6,9 ; sex 6,8 ; age 5,5 ; s2 3,8 ; s4 2,6. R² test du modèle : 0,434 |
| E. patient 0 | f(x) = 148,4, E[f] = 155,6 ; Shapley : s1 −31,3, s5 +17,6, sex +8,3, bp −8,2 |
| E. LIME, patient 0 | 30 voisins : 8 trios de tête différents sur 10 graines (le plus fréquent 3 fois) ; 500 voisins : le même trio (bmi, s5, bp) 10 fois sur 10 |
| E. durée | Shapley exact NumPy 4,1 s, `shap.explainers.Exact` 4,2 s (40 patients × 1 024 coalitions × 100 individus) |

Exemples de caractéristiques du SAE (tirées au hasard, 5 plus fortes activations ; le caractère entre crochets est celui où la caractéristique s'allume) :

```text
182  : … a little pot a[n]   …ixt your duke a[n]   …with sighs⏎In a[n]          → « an » après « a »
897  : … that both of u[s]   …ise⏎By all of u[s]   … liberty make u[s]          → « us »
1337 : …Hortensio.⏎⏎B[I]     …supper-time.⏎⏎B[I]   …, Vincentio?⏎⏎B[I]          → BIANCA en début de réplique
1428 : …ces from those [e]   …e water,⏎Whose [e]   … thy wandering [e]          → « e » en début de mot
1756 : …pity move my fa[t]   …ee my loving fa[t]   …thy master's fa[t]          → « fat(her) »
```

Figures produites : `figures/resultats.png` ; résultats bruts pour le site : `figures/resultats.npz` (lu par `site/scripts/figures/ch38_superposition.py`, `ch38_sondes.py`, `ch38_sae.py`, `ch38_induction.py` et `ch38_shap.py`).

## Ce qu'il faut observer

- **La superposition est une affaire de rareté.** Avec des caractéristiques toujours présentes, 2 dimensions n'en gardent que 2, comme une ACP. Dès qu'elles sont rares, le réseau en range 4 puis 5, en pentagone : deux caractéristiques actives en même temps se gêneraient, mais cela arrive rarement, et le biais négatif (−0,26) avec le ReLU efface les petites interférences. C'est l'explication proposée des neurones polysémantiques : un réseau a plus de choses à représenter que de neurones.
- **Une sonde mesure ce qui est lisible, pas ce qui est utilisé.** « Le caractère précédent était une voyelle » est illisible dans les plongements (69 %, autant que ce que permet le caractère courant) et lisible à 99,5 % juste après la première attention : une tête est allée le chercher. Mais pour « première ligne de réplique », le réseau **aléatoire** atteint déjà 76 % ; explication probable, non testée : une attention aléatoire fait une moyenne du contexte, qui contient des indices (majuscules, « : »). Sans ce témoin, on attribuerait au modèle un savoir que la sonde construit seule.
- **Le SAE transforme 1 024 neurones flous en 2 048 caractéristiques plus nettes.** 12 caractéristiques actives par token suffisent à reconstruire 86 % de la variance et 90 % de l'apport de la couche à la prédiction (1,529 contre 1,518, au lieu de 1,629 sans la couche). La pureté médiane passe de 50 % à 73 %, et un tiers des caractéristiques sont presque monosémantiques (« an » après « a », « us » dans « of us », le nom BIANCA en début de réplique). La perte récupérée n'est pas de 100 % : ce qui manque à la reconstruction est en partie ce qui compte, c'est ce qu'Engels et al. (2024, https://arxiv.org/abs/2410.14670) appellent la « matière noire » des SAE.
- **Notre mini-GPT n'a pas de tête d'induction** : il ne prédit pas mieux la seconde copie d'un passage que la première. Hypothèse (non testée ici) : 1 500 pas de Shakespeare caractère par caractère offrent trop peu de répétitions longues pour que ce circuit rapporte. Un Transformer à 2 couches entraîné sur des répétitions apprend le circuit, et d'un coup : la perte reste au niveau du hasard pendant 2 500 pas, puis tombe de 3,9 à 1,7 en 400 pas, en même temps que le score d'induction d'une tête de la couche 2 passe de 0,06 à 0,7. Avec une seule couche, rien : il faut composer deux têtes (une qui note le token précédent, une qui s'en sert pour chercher).
- **Un raccourci appris dès qu'on le laisse traîner.** Dans une première version de la partie D, la longueur répétée était fixe (L = 32) : le modèle à **une** couche résolvait aussi la tâche, en apprenant « regarde 31 positions en arrière », sans aucune induction. C'est pourquoi L varie maintenant. De même, en partie E, la colonne « fuite » fait passer le R² de 0,43 à 0,93 sur un test tiré comme l'entraînement, puis à −0,58 quand elle ne porte plus l'information ; SHAP la désigne au premier coup d'œil (81 % de l'importance).
- **Shapley exact = shap, à 10⁻¹³ près** : la bibliothèque calcule exactement la formule de la page, et l'efficacité (Σφ = f(x) − E[f], la « précision locale » de SHAP) est vérifiée. Mais le coût double avec chaque variable (2^d coalitions) : au-delà de 15 à 20 variables, on échantillonne (KernelSHAP) ou on exploite la structure du modèle (TreeSHAP pour les arbres, Lundberg, Erion et Lee, 2018, https://arxiv.org/abs/1802.03888).
- **Les attributions ne répondent pas toutes à la même question.** Pour le patient 0, Shapley met s1 en tête (−31) : par rapport au patient moyen, sa valeur de s1 fait baisser la prédiction. LIME, avec 500 voisins, met bmi, s5 et bp en tête : c'est la pente locale du modèle autour de ce patient, pas une comparaison avec la moyenne. Avec 30 voisins, LIME donne 8 trios différents sur 10 graines. Les gradients intégrés respectent la complétude (Σ attributions = f(x) − f(référence) à 0,01 % près) ; « gradient × entrée » la viole de 98 % en médiane.

## Questions de réflexion

1. Dans le modèle jouet, que se passerait-il avec la sortie linéaire x' = WᵀWx + b (sans ReLU) ? Pourquoi la superposition y serait-elle toujours perdante ?
2. La sonde « première ligne de réplique » réussit à 76 % sur un réseau aléatoire. Proposez une **tâche témoin** (au sens de Hewitt et Liang, 2019, https://arxiv.org/abs/1909.03368 : étiquettes aléatoires attachées à chaque caractère) et expliquez ce qu'elle permettrait de conclure.
3. Augmenter λ dans le SAE rend les caractéristiques plus parcimonieuses (L0 plus petit). Que devient alors la perte récupérée ? Quel compromis choisiriez-vous pour interpréter un modèle, et pourquoi la seule variance expliquée ne suffit-elle pas ?
4. La valeur de Shapley dépend de l'ensemble de fond. Que deviennent les attributions du patient 0 si le fond ne contient que des patients d'IMC élevé ? Laquelle des questions « pourquoi cette prédiction ? » et « pourquoi cette prédiction plutôt que la moyenne ? » Shapley traite-t-il ?

## Pour aller plus loin

- Faire varier λ (1, 3, 10) et tracer la frontière L0 / perte récupérée ; entraîner le SAE sur une autre couche.
- Remplacer `shap.explainers.Exact` par `shap.TreeExplainer` : comparer les durées et les valeurs (attention, les arbres de scikit-learn comparent en float32).
- Elhage et al., « Toy Models of Superposition » (2022), https://arxiv.org/abs/2209.10652 ; Olsson et al., « In-context Learning and Induction Heads » (2022), https://arxiv.org/abs/2209.11895 ; Cunningham et al., « Sparse Autoencoders Find Highly Interpretable Features in Language Models » (2023), https://arxiv.org/abs/2309.08600 ; Lundberg et Lee, « A Unified Approach to Interpreting Model Predictions » (2017), https://arxiv.org/abs/1705.07874 ; Sundararajan, Taly et Yan, « Axiomatic Attribution for Deep Networks » (2017), https://arxiv.org/abs/1703.01365 ; Alain et Bengio, « Understanding intermediate layers using linear classifier probes » (2016), https://arxiv.org/abs/1610.01644.
