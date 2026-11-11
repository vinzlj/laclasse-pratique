# Exercice 32 : un mini-RAG sur le texte du cours

| | |
|---|---|
| **Chapitre lié** | [32. RAG : génération augmentée par récupération](https://laclasse.poneyponey.fr/ia/p08-llm/32-rag/) |
| **Exécution** | locale : embeddings et reclassement sur MPS, faiss sur CPU (un thread), LLM local par Ollama (`qwen3.5:2b`) |
| **Commande** | `make ex NUM=32` (éditeur), `make ex NUM=32 MODE=run` (complet), `MODE=smoke` (≈ 20 à 30 s selon que le LLM est déjà chargé ou non) |
| **Durée mesurée** | run complet : 8 min 58 s (processus entier, aucune attente du verrou, modèles en cache), dont 4 min 9 s pour les 132 générations et 2 min 13 s pour l’expérience « perdu au milieu » ; premier lancement : ajouter le téléchargement (~2,4 Go de modèles Hugging Face) ; smoke : 18 à 28 s (M1 Pro, MPS, 16/10/2026 ; 28 s quand Ollama doit charger le modèle ; la condition « sans la réponse » est sautée en smoke) |
| **Travail estimé** | ~3 h |
| **Seed** | 42 (`seed_everything(42)` ; Ollama : `temperature: 0`, `seed: 42`). Aucun entraînement : la recherche est déterministe ; les réponses du LLM sont stables d’un run à l’autre à quelques questions près (même graine, mais calcul GPU) |
| **Données** | **versionnées** : `corpus/` (texte des chapitres 00 à 15 du cours, ≈ 656 000 caractères, extrait des pages du site) et `questions.json` (44 questions écrites à la main). **Téléchargées une fois** dans `data/hf/` : `intfloat/multilingual-e5-small` (MIT), `cross-encoder/ms-marco-MiniLM-L6-v2` (Apache-2.0), `BAAI/bge-reranker-v2-m3` (Apache-2.0, 2,3 Go), révisions figées dans le notebook. **Ollama** : `ollama pull qwen3.5:2b` (≈ 3 Go) |

## Objectif

Construire une chaîne RAG complète sur un corpus que vous connaissez (le cours lui-même) et mesurer chaque étage séparément : découpage, recherche lexicale (BM25), recherche dense (embeddings + faiss), fusion hybride (RRF), reclassement par cross-encodeur, puis génération par un petit LLM local qui doit citer ses sources.

## Ce que vous allez faire

1. **Corpus et questions.** Le corpus est le texte des chapitres 00 à 15, converti de MDX en Markdown simple et figé dans `corpus/` (le notebook ne dépend donc pas de l’état du site). `questions.json` contient 44 questions, chacune avec le **passage exact** du corpus qui contient la réponse et les fragments attendus dans une bonne réponse. 23 questions reprennent des mots rares du passage (noms, nombres : « lexicales »), 21 disent la même chose avec d’autres mots (« reformulées »).
2. **Découpage** (*chunking*). Découpage principal : on coupe aux titres, puis on remplit des morceaux d’environ 1 000 caractères phrase par phrase, avec 150 caractères de recouvrement ; chaque morceau commence par le chemin de ses titres (« 07. Rétropropagation… › Le mécanisme › … »). Comparaison avec des fenêtres fixes de 1 000 caractères coupées n’importe où, et avec des sections de 300 et 3 000 caractères.
3. **BM25 en numpy** (variante « Lucene » de `bm25s`, sans le facteur constant $(k_1 + 1)$ ; $k_1 = 1{,}5$, valeur par défaut de `bm25s`, Lucene utilisant $k_1 = 1{,}2$ ; $b = 0{,}75$) : une matrice creuse morceaux × vocabulaire précalculée ; noter une question revient à additionner quelques colonnes. Vérification contre la bibliothèque `bm25s` avec les mêmes tokens.
4. **Recherche dense** : `multilingual-e5-small` avec ses préfixes `query: ` / `passage: `, vecteurs normalisés, `faiss.IndexFlatIP` (recherche exacte ; inutile d’approcher sur 862 vecteurs).
5. **Hybride** : moyenne des scores normalisés, puis fusion par les rangs (RRF, $k = 60$).
6. **Reclassement** des 10 premiers candidats de RRF par deux cross-encodeurs : MiniLM entraîné sur MS MARCO (anglais seulement, 22 M de paramètres) et bge-reranker-v2-m3 (multilingue, 568 M).
7. **Génération** par `qwen3.5:2b` (Ollama, mode « réflexion » désactivé), trois conditions par question : sans contexte ; avec les 3 meilleurs morceaux, numérotés et à citer ; avec les 3 meilleurs morceaux **qui ne contiennent pas** la réponse (le bon comportement est alors « Je ne sais pas »).
8. **« Perdu au milieu »** : 10 questions, 8 passages (le bon et les 7 meilleurs distracteurs), le bon passage en position 1, 4 ou 8.

Ce qui est optimisé : **rien**. Aucun réseau n’est entraîné ; on mesure des rappels, des taux de bonnes réponses et des temps.

## Résultat attendu

En mode complet, le notebook vérifie cinq seuils et s’arrête avec une erreur si l’un d’eux n’est pas atteint. Valeurs obtenues le 15/10/2026 (seed 42, M1 Pro) :

| Seuil | Valeur obtenue |
|---|---|
| BM25 numpy = `bm25s` (écart < 10⁻³, mêmes 10 premiers) | écart 1,9 × 10⁻⁶, accord 100 % |
| RRF : rappel@5 ≥ 95 % et ≥ celui de BM25 et de e5 | **100 %** (BM25 95,5 %, e5 88,6 %) |
| bge-m3 : rappel@1 ≥ 88 % et ≥ RRF + 5 points | **95,5 %** (RRF 81,8 %) |
| RAG : réponses justes ≥ 70 % et ≥ sans contexte + 50 points | **84,1 %** contre 2,3 % |
| RAG sans la réponse : « je ne sais pas » ≥ 35 % | 40,9 % |

**Recherche (44 questions, 862 morceaux)** : « rappel@k » = part des questions dont un morceau contenant le passage de référence est parmi les k premiers.

| Méthode | r@1 | r@3 | r@5 | r@10 | MRR@10 | r@5 lexicales / reformulées |
|---|---|---|---|---|---|---|
| BM25 | 79,5 % | 90,9 % | 95,5 % | 95,5 % | 0,860 | 100 % / 90 % |
| e5 (dense) | 61,4 % | 86,4 % | 88,6 % | 93,2 % | 0,722 | 96 % / 81 % |
| moyenne des scores normalisés | 79,5 % | 93,2 % | 97,7 % | 100 % | 0,875 | 100 % / 95 % |
| hybride RRF | 81,8 % | 93,2 % | **100 %** | 100 % | 0,882 | 100 % / 100 % |
| RRF + MiniLM (anglais) | 81,8 % | 93,2 % | 97,7 % | 100 % | 0,882 | 100 % / 95 % |
| RRF + bge-m3 (multilingue) | **95,5 %** | **100 %** | 100 % | 100 % | **0,977** | 100 % / 100 % |

Coûts mesurés : index BM25 en 0,15 s ; encodage des 862 morceaux par e5 en 7,6 s ; recherche faiss 0,13 ms par question ; reclassement de 10 candidats : 63 ms par question (MiniLM), 2,1 s (bge-m3).

**Découpage** (même chaîne BM25 + e5 + RRF) :

| Découpage | morceaux | longueur médiane | r@5 BM25 | r@5 e5 | r@5 RRF | r@1 RRF |
|---|---|---|---|---|---|---|
| fenêtres fixes 1 000 caractères | 665 | 1 000 | 84,1 % | 75,0 % | 90,9 % | 61,4 % |
| sections ≈ 300 caractères | 2 787 | 331 | 86,4 % | 79,5 % | 88,6 % | 68,2 % |
| sections ≈ 1 000 caractères | 862 | 980 | 95,5 % | 88,6 % | 100 % | 81,8 % |
| sections ≈ 3 000 caractères | 363 | 1 858 | 95,5 % | 88,6 % | 97,7 % | 84,1 % |

**Génération (`qwen3.5:2b`, 44 questions)** :

| Condition | justes | « je ne sais pas » | fausses | tokens de prompt | temps par réponse |
|---|---|---|---|---|---|
| sans contexte | 2,3 % (1 question) | 4,5 % | 93,2 % | — | 1,1 s |
| RAG, 3 meilleurs morceaux | **84,1 %** | 2,3 % | 13,6 % | 974 | 2,5 s |
| RAG, 3 morceaux sans la réponse | 25,0 % | 40,9 % | 34,1 % | ≈ 1 000 | ≈ 2,5 s |

Avec RAG, 100 % des réponses portent au moins une citation [n], et pour 89 % des réponses justes, une citation désigne bien un morceau qui contient la réponse.

**« Perdu au milieu » (10 questions, 8 passages, ≈ 2 260 tokens)** : bon passage en position 1 : 70 % de réponses justes ; position 4 : 80 % ; position 8 : 80 %.

Figure produite : `figures/mini-rag.png` ; résultats bruts pour le site : `figures/resultats.json` (lu par `site/scripts/figures/ch32_*.py`), qui contient aussi toutes les réponses du LLM.

## Ce qu’il faut observer

- **BM25 est une base très solide sur ce corpus** : 95,5 % de rappel@5, et 100 % sur les questions qui reprennent des mots rares (« Mansfield », « 269 322 », « SIGART »). Les nombres et les noms propres sont exactement ce qu’un embedding de 384 dimensions écrase.
- **La recherche dense ne gagne pas là où on l’attend** : e5 fait moins bien que BM25 même sur les questions reformulées (81 % contre 90 % de r@5). Mais elle ne se trompe pas sur les mêmes questions ; la fusion RRF obtient 100 % de rappel@5, mieux que chacune. C’est l’argument de la recherche hybride, et il ne demande aucun réglage : RRF n’utilise que les rangs.
- **Le reclassement est l’étage qui fait gagner la première place** : bge-m3 passe le rappel@1 de 81,8 % à 95,5 %. Le cross-encodeur MiniLM, entraîné sur des requêtes anglaises, n’apporte rien sur un corpus français (81,8 % → 81,8 %, et un rappel@5 qui baisse) : un reclasseur n’est utile que dans la langue et le domaine de son entraînement. Le prix : 2,1 s par question pour 10 paires sur le GPU d’un portable, contre 0,13 ms pour la recherche faiss.
- **Le découpage compte autant que le modèle** : des fenêtres fixes coupées n’importe où perdent 9 points de rappel@5 (et 20 points de rappel@1) face au découpage aux titres de même taille. Les morceaux de 300 caractères perdent leur contexte ; ceux de 3 000 restent bons pour la recherche mais ont une longueur médiane presque double (1 858 caractères contre 980), que le LLM doit lire.
- **Sans contexte, le petit LLM ne sait presque rien de ce cours** (1 réponse juste sur 44, celle du dropout qui « éteint la moitié des neurones », connaissance générale) et il **invente** au lieu de dire qu’il ne sait pas (93 % de réponses fausses, 4,5 % de « je ne sais pas ») : par exemple « La constante obtenue est de 1,25 € par heure » pour la perte de Huber. Avec les 3 bons morceaux, il répond juste 84 % du temps et cite.
- **Le contexte ne garantit pas la réponse** : les 7 échecs de RAG ont tous le bon morceau dans le prompt (rappel@3 de 100 %). Le modèle lit mal un tableau (q02 : « itération 30 » au lieu de 4 182), donne la mauvaise ligne d’un passage (q21 : 3 500 membres en 1978 au lieu de 1 241 en 1973), ou répond à côté (q24). La recherche a fait son travail ; c’est la lecture qui échoue, et un modèle de 2 milliards de paramètres lit moins bien qu’un grand.
- **Une citation n’est pas une preuve** : 11 % des réponses justes citent un morceau qui ne contient pas la réponse. Vérifier les citations (ce que mesure la « fidélité » de RAGAS) est un contrôle distinct de la justesse.
- **Sans la bonne information, le modèle ne dit « je ne sais pas » que 41 % du temps** ; 34 % des réponses sont fausses, souvent avec une citation qui leur donne une apparence de sérieux. Les 25 % de réponses justes viennent de morceaux voisins qui répètent l’information (questions d’auto-évaluation, tableaux récapitulatifs, recouvrement) ou de la mémoire du modèle (le nombre de Shannon, 10¹²⁰).
- **« Perdu au milieu » ne se voit pas ici** : 70 / 80 / 80 % pour les positions 1, 4 et 8. Sur 10 questions, une question vaut 10 points : la différence n’est pas significative, et un prompt de 2 260 tokens est court par rapport aux expériences de Liu et al. (2023), qui allaient jusqu’à 30 documents. Ne pas conclure que l’effet n’existe pas ; conclure qu’il faut le mesurer pour son propre modèle et sa longueur de contexte.

## Questions de réflexion

1. BM25 a 100 % de rappel@5 sur les questions lexicales, e5 96 %. Écrivez une question sur le cours que BM25 raterait à coup sûr et que e5 devrait trouver (indice : aucun mot en commun avec le passage, synonymes, autre langue).
2. RRF avec $k = 60$ : un morceau classé 1er par BM25 et absent des 100 premiers de e5 (comptez 0 pour la liste où il n’apparaît pas) contre un morceau classé 5e dans les deux listes. Lequel passe devant ? Que change $k = 1$ ?
3. Le reclassement bge-m3 coûte 2,1 s par question pour 10 candidats. Combien coûterait-il sur tout le corpus (862 morceaux) ? Pourquoi ne fait-on jamais cela, et que perd-on en ne reclassant que les 10 premiers ?
4. Avec les 3 morceaux sans la réponse, le modèle invente une réponse fausse une fois sur trois. Proposez deux modifications (du prompt, du seuil de recherche ou de la chaîne) pour qu’il dise plus souvent « je ne sais pas », et dites ce qu’elles coûteraient sur les questions auxquelles il répond juste.

## Pour aller plus loin

- Remplacer `qwen3.5:2b` par un modèle plus grand (si la mémoire le permet) et comparer les 7 échecs de RAG : la recherche ne change pas, seule la lecture change.
- Ajouter un juge : demander au LLM si chaque phrase de la réponse est appuyée par le passage cité (la « fidélité » de RAGAS) et comparer à la vérification par fragments de ce notebook.
- Articles : Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (arXiv:2005.11401) ; Robertson et Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009, doi:10.1561/1500000019) ; Cormack, Clarke et Büttcher, *Reciprocal rank fusion outperforms Condorcet and individual rank learning methods* (SIGIR 2009, doi:10.1145/1571941.1572114) ; Nogueira et Cho, *Passage Re-ranking with BERT* (arXiv:1901.04085) ; Chen et al., *M3-Embedding* (arXiv:2402.03216) ; Liu et al., *Lost in the Middle: How Language Models Use Long Contexts* (arXiv:2307.03172) ; Es et al., *Ragas: Automated Evaluation of Retrieval Augmented Generation* (arXiv:2309.15217).

## Licences des données et des modèles

- `corpus/` : texte de ce cours (chapitres 00 à 15), sous la même licence que le site.
- `intfloat/multilingual-e5-small` : MIT ; `cross-encoder/ms-marco-MiniLM-L6-v2` et `BAAI/bge-reranker-v2-m3` : Apache-2.0 (champ `license` de l’API du Hub, vérifié le 15/10/2026, aucun n’est restreint).
- `qwen3.5:2b` (Ollama) : la page Ollama n’indique pas de licence ; la carte Hugging Face de `Qwen/Qwen3.5-2B` indique Apache-2.0.
