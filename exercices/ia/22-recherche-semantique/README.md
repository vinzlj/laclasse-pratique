# Exercice 22 : embeddings de phrases et recherche vectorielle

| | |
|---|---|
| **Chapitre lié** | [22. Embeddings contextuels et recherche vectorielle](https://laclasse.poneyponey.fr/ia/p07-transformer/22-embeddings-contextuels/) |
| **Exécution** | locale, MPS (inférence seulement : aucun entraînement) ; faiss sur CPU, un seul thread |
| **Commande** | `make ex NUM=22` (éditeur), `make ex NUM=22 MODE=run` (complet), `MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 3 min 30 s (processus entier, attente du verrou exclue, données en cache) ; premier lancement : ajouter le téléchargement (~1,4 Go de modèles, ~8 Mo de textes) ; smoke : 21 s (M1 Pro, MPS, 15/10/2026) |
| **Travail estimé** | ~1 h 30 |
| **Seed** | 42 (`seed_everything(42)`, `np.random.default_rng(42)` pour les tirages et les k-moyennes). Aucun entraînement : les scores ne dépendent que des modèles figés ; les latences dépendent de la machine |
| **Données** | téléchargées une fois : modèles Hugging Face dans `data/hf/` (révisions figées dans le notebook) ; STS-B traduit en français (`PhilipMay/stsb_multi_mt`, partie test) et PIAF (`AgentPublic/piaf`) dans `data/hf/` ; 12 volumes du projet Gutenberg dans `data/gutenberg/` |

## Objectif

Voir pourquoi le vecteur [CLS] d'un Transformer pré-entraîné n'est pas un embedding de phrase, comparer des méthodes de recherche sémantique sur une vraie tâche en français, puis écrire et mesurer les index de recherche approchée (IVF, quantification de produit, HNSW) qui permettent de chercher parmi des millions de vecteurs en quelques millisecondes.

## Ce que vous allez faire

1. **Trois modèles de même taille** (12 couches, 384 dimensions, vocabulaire de XLM-R) :
   - *brut* : `microsoft/Multilingual-MiniLM-L12-H384` (MIT), modèle de langue pré-entraîné, jamais entraîné à comparer des phrases ;
   - *SBERT* : `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0), même architecture, entraîné à rapprocher les paraphrases (version multilingue d'un modèle anglais, par imitation de ce modèle sur des phrases traduites) ; embedding = moyenne des vecteurs de tokens ;
   - *e5* : `intfloat/multilingual-e5-small` (MIT), entraîné de façon contrastive pour la recherche ; sa carte demande de préfixer chaque texte par `query: ` ou `passage: ` (et `query: ` des deux côtés pour une tâche symétrique).
2. **[CLS] brut contre embedding entraîné** sur STS-B traduit en français (1 379 paires notées de 0 à 5) : corrélation de Spearman entre cosinus et notes humaines ; cosinus moyen entre deux phrases tirées au hasard (anisotropie).
3. **Recherche questions → passages** sur PIAF (3 835 questions, 761 paragraphes de Wikipédia) : TF-IDF, modèle brut, SBERT, e5 avec et sans préfixes, et la moyenne des scores TF-IDF et e5. La recherche est écrite en numpy : produit matriciel, puis `argpartition` pour les k meilleurs.
4. **Coût d'un cross-encodeur**, mesuré : passer 761 textes « question + passage » dans un réseau de même taille, contre un seul encodage de la question pour le bi-encodeur.
5. **Base de 52 780 phrases** de romans français (Hugo, *Les Misérables* ; Verne, trois romans ; Dumas, *Le Comte de Monte-Cristo*), encodées par SBERT ; 1 000 phrases mises de côté comme requêtes ; quatre requêtes écrites à la main, comparées entre TF-IDF et SBERT.
6. **Recherche exacte** en numpy (la vérité : 10 plus proches voisins par requête) et sa latence requête par requête.
7. **IVF en numpy** : k-moyennes sphériques (256 cellules), recherche dans les `nprobe` cellules les plus proches.
8. **Quantification de produit en numpy** : 48 sous-vecteurs de 8 dimensions, 256 centroïdes chacun, un octet par sous-vecteur ; scores approchés par table de correspondance ; reclassement des 100 premiers candidats avec les vrais vecteurs.
9. **faiss** : `IndexFlatIP` (exacte), `IndexHNSWFlat` (M = 32), `IndexIVFFlat` et `IndexIVFPQ` (256 cellules) ; rappel@10, latence, octets par vecteur ; latence en fonction de la taille de la base (2 000 à 52 780 vecteurs).

Ce qui est optimisé : **rien**. Aucun réseau n'est entraîné. On mesure des cosinus, des corrélations, des rappels et des temps.

## Résultat attendu

En mode complet, le notebook vérifie six seuils et s'arrête avec une erreur si l'un d'eux n'est pas atteint. Valeurs obtenues le 15/10/2026 (seed 42, M1 Pro) :

| Seuil | Valeur obtenue |
|---|---|
| Spearman SBERT ≥ Spearman [CLS] brut + 30 points | **79,9** contre 48,4 (+31,5) |
| cosinus moyen entre phrases au hasard : [CLS] brut ≥ 0,8 et SBERT ≤ 0,3 | **0,9997** et 0,036 |
| PIAF : e5 avec préfixes, rappel@5 ≥ 85 % et meilleur que sans préfixe | **87,2 %** contre 86,0 % |
| faiss `IndexFlatIP` = recherche exacte numpy (rappel ≥ 99,9 %) | 100 % |
| HNSW : rappel@10 ≥ 95 % au moins 10 fois plus vite que la recherche exacte | 98,2 % en 0,19 ms contre 2,41 ms (13 fois) ; 94,9 % en 0,11 ms |
| PQ : codes 32 fois plus petits, rappel@10 après reclassement ≥ 90 % | 1 536 → 48 octets ; 99,2 % |

**STS-B fr (test, 1 379 paires)** :

| Méthode | Spearman × 100 | cosinus moyen, paires au hasard |
|---|---|---|
| TF-IDF | 66,5 | 0,026 |
| MiniLM brut, [CLS] | 48,4 | 0,9997 |
| MiniLM brut, moyenne des tokens | 51,3 | 0,901 |
| SBERT (même architecture, entraînée) | **79,9** | 0,036 |
| e5 (`query: `) | 79,2 | 0,814 |

**PIAF (3 835 questions, 761 passages)** :

| Méthode | r@1 | r@5 | r@10 | MRR@10 |
|---|---|---|---|---|
| TF-IDF | 64,0 % | 82,5 % | 86,9 % | 0,719 |
| MiniLM brut, moyenne | 1,5 % | 4,4 % | 8,2 % | 0,030 |
| SBERT (paraphrases) | 44,8 % | 67,8 % | 74,8 % | 0,546 |
| e5, `query: ` / `passage: ` | 66,8 % | 87,2 % | 90,9 % | 0,757 |
| e5 sans préfixe | 65,7 % | 86,0 % | 89,8 % | 0,744 |
| moyenne des scores TF-IDF et e5 | **70,6 %** | **89,7 %** | **93,0 %** | **0,787** |

Coût mesuré : cross-encodeur ≈ 1,9 s par question (761 paires dans le réseau ; borne basse, textes tronqués à 128 tokens par SBERT, un vrai cross-encodeur en lirait jusqu'à 512) ; bi-encodeur ≈ 21 ms (un encodage de la question, puis 761 produits scalaires), soit environ 90 fois moins. Ce rapport varie d'un lancement à l'autre (de 77 à 89 sur trois runs) : l'encodage d'une seule question est court et sensible à la charge de la machine.

**Recherche approchée (52 780 vecteurs, 1 000 requêtes, 10 voisins)** :

| Méthode | réglage | rappel@10 | latence par requête | octets par vecteur |
|---|---|---|---|---|
| exacte numpy (BLAS multithread) | — | 100 % | 2,41 ms | 1 536 |
| exacte faiss `Flat` (1 thread) | — | 100 % | 1,74 ms | 1 536 |
| HNSW (M = 32) | efSearch 8 / 32 / 64 / 256 | 78,0 / 94,9 / 98,2 / 99,7 % | 0,04 / 0,11 / 0,19 / 0,64 ms | 1 808 |
| IVF (256 cellules) | nprobe 1 / 8 / 32 / 64 | 42,8 / 84,9 / 96,4 / 98,8 % | 0,02 / 0,08 / 0,28 / 0,53 ms | 1 551 |
| IVF-PQ (48 octets) | nprobe 1 / 8 / 64 | 38,7 / 62,3 / 64,6 % | 0,06 / 0,08 / 0,25 ms | 71 |
| IVF numpy | nprobe 1 / 8 / 32 | 42,7 / 84,0 / 96,5 % | (0,5 / 3,5 / 13,9 % de la base comparée) | — |
| PQ numpy seul / avec reclassement des 100 premiers | — | 62,4 % / 99,2 % | — | 48 |

| Taille de la base | 2 000 | 8 000 | 32 000 | 52 780 |
|---|---|---|---|---|
| exacte numpy | 0,04 ms | 0,31 ms | 1,55 ms | 2,56 ms |
| HNSW, efSearch 64 | 0,05 ms | 0,08 ms | 0,14 ms | 0,18 ms |
| rappel HNSW | 99,5 % | 99,0 % | 98,3 % | 98,2 % |

Encodage des 53 780 phrases par SBERT : ~50 s sur MPS (~1 000 phrases/s). Construction de l'index HNSW : 8 s. Valeurs du 3e run complet (15/10/2026) ; les rappels et corrélations sont identiques sur les trois runs, les latences varient de ±15 %.

Les latences dépendent de la machine et de sa charge ; les rappels et les corrélations, non.

Figures produites : `figures/sts.png` (cosinus contre note humaine), `figures/piaf.png`, `figures/recherche-approchee.png` ; résultats bruts pour le site : `figures/resultats.npz` (lu par `site/scripts/figures/ch22_*.py`).

## Ce qu'il faut observer

- **Le [CLS] brut met toutes les phrases au même endroit** : cosinus moyen de 0,9997 entre deux phrases sans rapport. Son Spearman (48) n'est pas nul, car l'ordre des cosinus garde un peu d'information, mais un seuil sur le cosinus n'a aucun sens. La moyenne des tokens fait à peine mieux (51, cosinus au hasard 0,90). **Le modèle de même architecture** entraîné sur des paires de phrases (SBERT) monte à 80 et ses cosinus s'étalent de −0,2 à 1. Ce n'est pas l'architecture qui fait l'embedding de phrase, c'est l'objectif d'entraînement.
- **TF-IDF est une référence redoutable** : 66,5 de Spearman sur STS-B, et 64 % de bonne réponse en première position sur PIAF, où les questions reprennent souvent les mots du passage. Le modèle « paraphrases » (SBERT) fait **moins bien** que TF-IDF sur PIAF : il a été entraîné à comparer deux phrases de même nature, pas une question courte à un paragraphe long. e5, entraîné pour la recherche, dépasse TF-IDF ; c'est l'argument de MTEB : choisir un modèle sur la tâche visée, pas sur un classement global.
- **Les préfixes de e5 aident peu ici** (+1,1 point de r@1) mais toujours dans le même sens ; ne pas les mettre est une erreur silencieuse.
- **Lexical et sémantique se complètent** : la simple moyenne des scores TF-IDF et e5 gagne 4 points de r@1 sur le meilleur des deux. C'est l'idée de la recherche hybride (BM25 + vecteurs, fusion RRF) du chapitre 32.
- **Requêtes écrites à la main** (quatre) : pour l'évasion de Dantès, TF-IDF trouve la bonne phrase (une longue phrase de *Monte-Cristo* qui contient « prit sa place dans le sac où on l'avait cousu ») grâce aux mots rares partagés ; SBERT renvoie des phrases courtes sur « un prisonnier » en général : dans une phrase longue, le détail décisif est noyé dans la moyenne. Pour le vol de l'argenterie de l'évêque, SBERT renvoie en tête une phrase du lendemain du vol (l'évêque devant la plante écrasée par le panier d'argenterie jeté dans le jardin), sans reprendre aucun mot important de la requête ; TF-IDF tombe aussi sur cette scène, en 2e position, grâce au mot « argenterie ». Pour le gentleman anglais qui parie de faire le tour de la planète, SBERT trouve en 3e position la phrase du *Tour du monde en quatre-vingts jours* sur le pari en faveur de Phileas Fogg ; pour le volcan d'Islande, les deux trouvent *Voyage au centre de la Terre*. Les embeddings de phrases captent le thème, pas toujours le détail qui compte.
- **Le cross-encodeur coûte ~90 fois plus par question** sur 761 passages, et ce rapport croît avec la taille du corpus : sur un million de passages, un cross-encodeur ne peut que reclasser les quelques dizaines de candidats d'un bi-encodeur.
- **HNSW** : 98 % des vrais voisins en 0,19 ms au lieu de 2,4 ms, en ne calculant qu'une petite partie des distances ; et quand la base passe de 2 000 à 52 780 vecteurs (×26), la latence exacte est multipliée par ~60 (plus que ×26 : à 2 000 vecteurs, la base, 3 Mo, tient sans doute dans le cache du processeur ; à 52 780, 81 Mo, non), celle de HNSW par ~4. Le prix : la mémoire (1 808 octets par vecteur au lieu de 1 536, à cause des listes de voisins) et 8 s de construction.
- **IVF** : visiter 32 cellules sur 256 (14 % de la base) retrouve 96 % des voisins. Avec 1 seule cellule, plus de la moitié des vrais voisins sont dans une cellule voisine et sont perdus.
- **PQ** : 32 fois moins de mémoire, mais 62 % de rappel seulement ; IVF-PQ plafonne à 65 % même en visitant 64 cellules, car l'erreur vient de la compression, pas de la partition. Reclasser les 100 premiers candidats avec les vrais vecteurs remonte à 99 % : c'est la combinaison employée en pratique (codes compacts en mémoire, vecteurs complets sur disque).

## Questions de réflexion

1. Le [CLS] brut a un Spearman de 48 alors que tous ses cosinus valent 0,9997 à quelques dix-millièmes près. Pourquoi le Spearman peut-il rester correct quand les valeurs sont si concentrées ? Pourquoi est-ce quand même inutilisable pour une recherche avec un seuil de similarité ?
2. Sur PIAF, TF-IDF bat le modèle SBERT. Imaginez une collection de questions où ce serait l'inverse (indice : synonymes, reformulations, autre langue que celle des passages).
3. Pourquoi l'IVF-PQ plafonne-t-il à 65 % de rappel même quand `nprobe` augmente, alors que l'IVF simple atteint 99 % ? Que faut-il ajouter pour dépasser ce plafond ?
4. Une base de 100 millions de vecteurs de 384 dimensions : combien de mémoire en float32, avec HNSW, avec des codes PQ de 48 octets ? Laquelle tient dans la RAM d'un serveur de 64 Go ?

## Pour aller plus loin

- Remplacer TF-IDF par BM25 (`bm25s`, déjà dans le groupe `nlp`) et la moyenne des scores par une fusion par rangs (RRF) : c'est le début de l'exercice 32.
- Encoder la base de phrases avec e5 (`passage: `) au lieu de SBERT et refaire les quatre requêtes écrites à la main.
- Faire varier `M` et `efConstruction` de HNSW : mémoire, temps de construction, rappel.
- Articles : Reimers et Gurevych, *Sentence-BERT* (arXiv:1908.10084) ; Malkov et Yashunin, HNSW (arXiv:1603.09320) ; Jégou, Douze et Schmid, quantification de produit (doi:10.1109/TPAMI.2010.57) ; Johnson, Douze et Jégou, faiss (arXiv:1702.08734) ; Muennighoff et al., MTEB (arXiv:2210.07316) ; Wang et al., *Multilingual E5 Text Embeddings* (arXiv:2402.05672) ; Keraron et al., *Project PIAF* (LREC 2020, https://aclanthology.org/2020.lrec-1.673/).

## Licences des données

- STS-B traduit (`PhilipMay/stsb_multi_mt`, traduction automatique DeepL de STSbenchmark) : notes sous CC BY-SA 4.0, textes sous les licences de leurs sources d'origine (voir https://github.com/PhilipMay/stsb-multi-mt/blob/main/LICENSE) ; usage local de recherche et d'enseignement, rien n'est redistribué.
- PIAF (`AgentPublic/piaf`) : licence MIT indiquée sur le Hub ; paragraphes tirés de Wikipédia.
- Textes Gutenberg : œuvres du domaine public (Hugo, Verne, Dumas et Maquet, tous morts avant 1906) ; téléchargés, jamais versionnés.
