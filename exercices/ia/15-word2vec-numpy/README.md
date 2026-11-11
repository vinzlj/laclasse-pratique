# Exercice 15 : du one-hot à word2vec, en numpy

| | |
|---|---|
| **Chapitre lié** | [15. Plongements avant les Transformers et préhistoire contrastive](https://laclasse.poneyponey.fr/ia/p06-deep-learning/15-plongements/) |
| **Exécution** | locale, CPU (numpy, puis scikit-learn ; PyTorch seulement pour vérifier un gradient ; aucun GPU) |
| **Commande** | `make ex NUM=15` (éditeur), `make ex NUM=15 MODE=run` (complet), `MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 45 s (processus entier, attente du verrou exclue), dont 27 s d'entraînement word2vec et 7 s de LSA ; smoke : 6 s (M1 Pro, CPU, 14/10/2026) |
| **Travail estimé** | ~2 h |
| **Seed** | 42 (`seed_everything(42)`, `np.random.default_rng(42)` ; `random_state=42` pour `TruncatedSVD`, `PCA`, `TSNE`). Calcul numpy sur CPU : résultats identiques d'un lancement à l'autre |
| **Données** | TinyShakespeare (1,1 Mo), téléchargé une fois dans `data/input.txt` par `ia_commun.donnees.charger_tinyshakespeare()` |

## Objectif

Construire sur le même petit corpus quatre représentations des mots (one-hot, TF-IDF, LSA, word2vec skip-gram avec échantillonnage négatif), les programmer en numpy et les vérifier contre scikit-learn et l'autograd de PyTorch, puis mesurer ce que chacune capture : voisins, groupes de mots, analogies, et ce que les cartes 2D déforment.

## Ce que vous allez faire

1. **Corpus et découpage naïf.** Les 7 222 répliques de TinyShakespeare deviennent des documents ; une expression régulière (`[a-z]+(?:'[a-z]+)?`) les coupe en 194 017 mots, dont 12 350 distincts. On garde les 3 251 mots vus au moins 5 fois : 7,7 % des occurrences tombent hors vocabulaire.
2. **One-hot.** Vérifier que le cosinus entre deux mots distincts vaut toujours 0.
3. **Sac de mots et TF-IDF en numpy** (IDF de Sparck Jones, ln(N/df)) ; comparer à `TfidfVectorizer` (qui ajoute 1 à l'IDF) ; vérifier que « the dog bit the man » et « the man bit the dog » ont le même sac de mots ; écrire un moteur de recherche de répliques au cosinus.
4. **LSA** : vecteurs propres de X Xᵀ (X = matrice TF-IDF mots × répliques), 100 dimensions ; comparer les valeurs singulières à `TruncatedSVD`.
5. **Skip-gram avec échantillonnage négatif en numpy** : paires (centre, voisin) avec une fenêtre de 1 à 4 mots tirée au hasard, sous-échantillonnage des mots fréquents (t = 10⁻³), 5 négatifs tirés selon la fréquence puissance 3/4, gradient écrit à la main et vérifié par `torch.autograd` sur un lot. 10 époques, 50 dimensions, pas de 0,025 qui décroît linéairement. Comme dans le code C d'origine de word2vec : 5 négatifs, `min_count = 5`, `sample = 1e-3`, `alpha = 0.025` (valeur du skip-gram ; 0,05 en CBOW), puissance 0,75, même initialisation. Le code C diffère par la fenêtre (5), la dimension (100) et le nombre d'époques (5).
6. **Comparer TF-IDF, LSA et word2vec** : voisins de huit mots ; test de groupes (50 mots en 5 groupes : famille, nombres, corps, titres, temps ; part des mots dont le plus proche voisin parmi ces 50 est du même groupe ; hasard ≈ 20 %) ; 16 analogies « a : b :: c : ? » par 3CosAdd.
7. **Projeter en 2D** les 1 004 mots de la démo `EmbeddingExplorer` par ACP et t-SNE, et mesurer la part des 10 plus proches voisins (à 50 dimensions) qui le restent sur la carte.

Ce qui est optimisé : TF-IDF, rien (on compte) ; LSA, l'erreur de reconstruction quadratique de la matrice TF-IDF par une matrice de rang 100 ; word2vec, l'entropie croisée binaire « vrai voisin contre mot tiré au hasard » :
ℒ = −log σ(v_w·u_c) − Σₖ log σ(−v_w·u_{nₖ}).

## Résultat attendu

En mode complet, le notebook vérifie neuf seuils et s'arrête avec une erreur si l'un d'eux n'est pas atteint. Valeurs obtenues le 14/10/2026 (seed 42) :

| Seuil | Valeur obtenue |
|---|---|
| TF-IDF numpy = `TfidfVectorizer` (écart < 10⁻⁴) | 0 (identique) |
| LSA : valeurs singulières = `TruncatedSVD` (écart relatif < 10⁻⁶) | 1,5 × 10⁻⁷ |
| gradient skip-gram = autograd (écart < 10⁻¹⁰) | 5,6 × 10⁻¹⁷ |
| perte initiale ≈ 6 ln 2 = 4,159 (à 0,01 près) | 4,159 |
| perte moyenne de la dernière époque ≤ 2,60 | 2,535 (3,00 à la première époque) |
| word2vec : test de groupes ≥ 75 % | 80 % |
| word2vec : au moins 6 analogies sur 16 en top 1 | 8 (9 en top 5) |
| word2vec au moins aussi bon que TF-IDF aux analogies | 8 contre 6 |
| t-SNE conserve plus de voisins que l'ACP | 35,0 % contre 7,4 % |

Comparaison complète :

| Représentation | Dimension | Test de groupes | Analogies top 1 (top 5) |
|---|---|---|---|
| one-hot | 3 251 | — (cosinus nul partout) | 0 |
| TF-IDF (lignes) | 7 222 | 78 % | 6/16 (7) |
| LSA | 100 | 82 % | 0/16 (2) |
| word2vec skip-gram | 50 | 80 % | 8/16 (9) |

Voisins de « king » : TF-IDF *and, the, of, to, henry, his* ; LSA *regal, henry's, seated, henry, lewis, london* ; word2vec *henry, edward, richard, lewis, warwick, vi*. Analogies réussies par word2vec : he:she::his:her, he:she::him:her, lord:lady::king:queen, i:my::he:his, i:me::he:him, we:our::they:their, eye:eyes::hand:hands, man:men::lord:lords. Ratées : man:woman::king:? → *england*, father:mother::son:? → *wife*, king:queen::prince:? → *mother*…

Figures produites : `figures/perte.png`, `figures/projections.png` ; résultats bruts pour le site : `figures/resultats.npz` (lu par `site/scripts/figures/ch15_*.py`, qui écrit aussi les données de la démo `EmbeddingExplorer`).

## Ce qu'il faut observer

- **One-hot ne sait rien** : tous les cosinus valent 0. Dès qu'on décrit un mot par ses contextes (lignes de TF-IDF), « three » ressemble à « four » : c'est l'hypothèse distributionnelle, avant tout apprentissage.
- **Sur un petit corpus, compter est compétitif.** Au test de groupes, TF-IDF (78 %), LSA (82 %) et word2vec (80 %) sont au coude à coude. word2vec gagne en compacité (50 dimensions denses contre 7 222 creuses) et aux analogies grammaticales ; son avantage historique vient surtout de sa capacité à passer à des milliards de mots.
- **Les voisins TF-IDF sont pollués par les mots outils** (« king » → *and, the, of*) : une réplique longue contient de tout. La fenêtre locale de word2vec et le sous-échantillonnage de « the » règlent ce problème.
- **Les analogies célèbres ratent** : man:woman::king:? donne *england*. Celles qui marchent portent sur des mots très fréquents (pronoms, pluriels). Il faut des corpus des milliers de fois plus grands pour que king − man + woman ≈ queen.
- **La perte part de 6 ln 2 ≈ 4,16** (vecteurs de contexte nuls, σ = 1/2 partout), tombe vite sous 2,7 puis décroît lentement : la valeur de la perte dit peu de la qualité des vecteurs, il faut des tests extérieurs.
- **Une carte 2D ment** : l'ACP ne garde que 21 % de la variance et 7 % des voisins ; t-SNE garde 35 % des voisins et forme de beaux amas, mais les distances entre amas n'ont pas de sens.

## Questions de réflexion

1. Pourquoi l'IDF d'un mot présent dans toutes les répliques vaut-il 0 ? Est-ce souhaitable pour un moteur de recherche ? pour des vecteurs de mots ?
2. word2vec donne deux vecteurs par mot (v au centre, u en contexte). Que se passerait-il si on forçait v = u ? (Indice : quel serait le score d'un mot avec lui-même ?)
3. Le test de groupes ne départage pas TF-IDF, LSA et word2vec. Proposez un test qui les départagerait, ou expliquez pourquoi ce corpus ne le permet pas.
4. Un mot vu 3 fois n'a pas de vecteur. Comment fastText (vecteurs de n-grammes de caractères) contourne-t-il ce problème, et que gagnerait-on sur « king's » ?

## Pour aller plus loin

- Passer `dim` à 100, `epoques` à 20 ou `fenetre` à 2 dans la cellule de configuration : à l'essai (même graine), 20 époques donnent 88 % au test de groupes mais 6 analogies, une fenêtre de 2 donne 8 analogies en top 1 et 11 en top 5. Les scores bougent de plusieurs points d'un réglage à l'autre : sur un corpus aussi petit, ne pas surinterpréter un écart de 2 %.
- Remplacer la LSA sur la matrice mots × répliques par une SVD de la matrice de cooccurrences (fenêtre de 4 mots) pondérée en PMI positive, et comparer à word2vec.
- Écrire la variante CBOW (moyenne des voisins → mot central) en modifiant `perte_et_gradients`.
