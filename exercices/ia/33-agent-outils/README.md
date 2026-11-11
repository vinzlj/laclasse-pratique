# Exercice 33 : un agent avec des outils, et un LLM qui juge

| | |
|---|---|
| **Chapitre lié** | [33. Utiliser les LLM : prompting, outils, agents, évaluation](https://laclasse.poneyponey.fr/ia/p08-llm/33-agents-evaluation/) |
| **Exécution** | locale : LLM `qwen3.5:2b` servi par Ollama (GPU du Mac via Metal) ; aucun entraînement |
| **Commande** | `make ex NUM=33` (éditeur), `make ex NUM=33 MODE=run` (complet), `MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 5 min 35 s (processus entier ; un premier run avait pris 6 min 53 s, la durée dépend de la longueur des trajectoires) ; smoke : 13 à 24 s, dont 4 à 8 s de chargement du modèle s'il n'est pas déjà en mémoire (M1 Pro 16 Go, Ollama 0.40.4, mesures des 15 et 16/10/2026, attente du verrou exclue) |
| **Travail estimé** | ~2 h 30 |
| **Seed** | 42 (graine passée à Ollama, température 0 : décodage glouton, donc runs reproductibles à quelques réponses près) |
| **Données** | rien n'est téléchargé par le notebook : la documentation fictive est dans `corpus/` (10 fiches), les tâches dans `taches.json`, les réponses à juger dans `juge.json`. Prérequis : Ollama lancé et `ollama pull qwen3.5:2b` (≈ 3 Go, une fois) |

## Objectif

Voir ce qu'un petit LLM gagne quand on lui donne des outils (une calculatrice, une recherche dans une documentation) et une boucle pour s'en servir, comparer un agent ReAct écrit à la main à l'appel de fonctions natif, puis mesurer à quel point on peut faire confiance à un LLM pour noter des réponses à la place d'un humain.

## Ce que vous allez faire

**Partie A : l'agent** (20 tâches de `taches.json`, réponse = un nombre, vérifiée par le programme).

1. Lire la documentation `corpus/` : dix fiches sur **Roulevent**, une coopérative de vélos **inventée** dans une ville inventée (Corvenne). Le modèle ne peut donc pas connaître ces faits : sans outil, il ne peut que deviner.
2. Écrire les deux outils :
   - `chercher[mots-clés]` : recherche par mots-clés (mots réduits à leurs 4 premières lettres, pondérés par leur rareté, façon TF-IDF), qui renvoie les 3 fiches les mieux classées. Volontairement rudimentaire : la vraie recherche (BM25, embeddings, reranking) est l'objet du chapitre 32 ;
   - `calculatrice[expression]` : évalue + − × ÷, puissances, modulo et les fonctions `round`, `abs`, `min`, `max` en parcourant l'arbre syntaxique Python (jamais `eval` sur un texte écrit par un LLM).
3. Comparer trois stratégies sur les 20 tâches (6 de calcul, 7 de recherche, 7 composées : chercher deux chiffres puis calculer) :
   - **sans outils** : le modèle raisonne puis écrit « Réponse : n » ;
   - **ReAct écrit à la main** : la consigne décrit les outils et le format « Pensée / Action / Observation », avec un exemple complet (few-shot) ; notre code lit l'action par expression régulière, exécute l'outil et renvoie « Observation : … » ; au plus 6 appels au modèle ;
   - **appel de fonctions natif** : les mêmes outils décrits par un schéma JSON (`tools=` d'Ollama) ; le modèle renvoie un appel structuré, sans regex à écrire ; au plus 6 appels.
4. Lire une trajectoire complète de chaque agent (affichée par le notebook).

**Partie B : le juge** (`juge.json`, 20 réponses écrites à la main, 10 justes et 10 fausses, avec le verdict de l'auteur de l'exercice).

5. Faire noter chaque réponse par le même modèle, « CORRECT » ou « INCORRECT », **avec** puis **sans** la réponse de référence ; mesurer l'accord avec le verdict humain et le kappa de Cohen (accord corrigé du hasard).
6. Présenter 10 paires (une réponse juste, une fausse, pour la même question) dans les deux ordres, A/B puis B/A : un juge sans biais de position choisit la même réponse dans les deux cas.

Ce qui est optimisé : **rien**. Le modèle est figé (pré-entraîné à prédire le token suivant, chapitre 20, puis affiné par ses auteurs pour suivre des consignes et appeler des outils). On mesure la part de réponses justes, le coût (appels au modèle, tokens générés) et l'accord juge/humain.

## Résultat attendu

En mode complet, le notebook vérifie six seuils et s'arrête avec une erreur si l'un d'eux n'est pas atteint. Valeurs obtenues le 15/10/2026 (seed 42, température 0, M1 Pro) :

| Seuil | Run validé | Premier run |
|---|---|---|
| appel de fonctions ≥ 75 % | **95 %** (19/20) | 85 % |
| appel de fonctions ≥ ReAct (texte) + 20 points | 95 contre 45 | 85 contre 40 |
| ReAct (texte) ≥ sans outils + 10 points | 45 contre 15 | 40 contre 20 |
| sans outils ≤ 35 % | **15 %** | 20 % |
| juge avec référence : accord ≥ 70 % | **80 %** (kappa 0,60) | 80 % |
| juge sans référence ≤ juge avec référence − 15 points | **55 %** (kappa 0,10) | 55 % |

Détail du run validé :

| stratégie | calcul (6) | recherche (7) | composée (7) | total | appels / tâche | tokens générés / tâche | durée / tâche |
|---|---|---|---|---|---|---|---|
| sans outils | 50 % | 0 % | 0 % | **15 %** | 1,0 | 101 | 1,9 s |
| ReAct (texte) | 67 % | 43 % | 29 % | **45 %** | 4,6 | 453 | 10,1 s |
| appel de fonctions | 100 % | 100 % | 86 % | **95 %** | 2,8 | 153 | 3,7 s |

ReAct (texte) : 11 tâches sur 20 finissent sans réponse (budget de 6 appels épuisé) et le modèle a écrit lui-même une fausse « Observation » à 28 étapes.

Juge (20 réponses, 10 justes, 10 fausses) : avec la référence, 10 justes acceptées, 6 fausses rejetées, **4 fausses acceptées** ; sans la référence, 9 fausses acceptées sur 10. Paires (10, dans les deux ordres) : la bonne réponse est choisie 30 % des fois quand elle est en A, 80 % quand elle est en B ; même choix dans les deux ordres pour 3 paires sur 10 ; « A » répondu 25 % des fois.

**Reproductibilité** : même à température 0 et graine fixée, Ollama ne rend pas les générations identiques d'un run à l'autre (les deux runs ci-dessus diffèrent sur quelques tâches de la partie A ; la partie B est identique). Les seuils portent sur des écarts larges pour rester stables.

Figure produite : `figures/agent-juge.png` ; tous les résultats et toutes les trajectoires : `figures/resultats.json` (lu par les figures du site).

## Ce qu'il faut observer

- **Sans outil, la recherche est à 0 %** : le modèle invente une date de fondation, un tarif, un nombre de stations, avec aplomb. Il ne dit presque jamais « je ne sais pas ». Les calculs à plusieurs chiffres sont justes une fois sur deux.
- **L'appel de fonctions natif fait le plus gros saut**, avec moins d'appels et trois fois moins de tokens que ReAct écrit à la main (153 contre 453 par tâche). Le modèle a été affiné pour ce format : l'appel arrive séparé du texte, avec des arguments déjà structurés.
- **ReAct écrit à la main échoue pour des raisons de format, pas de connaissances.** Deux pannes dominent dans les trajectoires : le modèle **écrit lui-même la ligne « Observation : … »** (il invente le résultat de l'outil ; le notebook coupe le texte à cet endroit et compte ces étapes), et il **tourne en rond** : il a la bonne information (« l'abonnement annuel coûte 39 euros ») mais veut « vérifier », relance la même recherche jusqu'au bout du budget et n'écrit jamais « Réponse finale ». Point technique constaté avec Ollama 0.40.4 et ce modèle : l'option `stop: ["Observation"]` n'arrête pas la génération ; la coupure doit être faite côté code.
- **Les erreurs restantes de l'agent sont des erreurs de lecture** : pour remplacer les batteries de toute la flotte, l'agent lit « 60 vélos mécaniques » dans la fiche historique (la flotte de 2018) au lieu des 145 vélos électriques de 2024, et calcule 60 × 410 € juste… sur le mauvais chiffre. Pour une location de 25 minutes, ReAct calcule 25 × 0,12 = 3 € et oublie l'euro de déblocage écrit dans la même phrase. Les outils ne corrigent pas un raisonnement faux.
- **Sans outil, le modèle part parfois complètement à côté** : à « combien de stations compte le réseau Roulevent ? », il décrit un « réseau routier » de 13 000 km en région parisienne.
- **Le juge est indulgent.** Avec la référence, il ne rejette jamais une réponse juste, mais il accepte plusieurs réponses fausses (« 24 € » quand la référence est 39). Sans référence, il déclare presque tout « CORRECT » : un kappa proche de 0, c'est un accord à peine meilleur que le hasard.
- **Biais de position** : dans les comparaisons par paires, le juge préfère nettement la **seconde** réponse (B), quelle qu'elle soit. Changer l'ordre change son verdict pour la majorité des paires.

## Questions de réflexion

1. Dans les trajectoires ReAct qui tournent en rond, quel garde-fou ajouteriez-vous dans la boucle (détection d'une action répétée, message « tu as déjà cette information, conclus », budget plus court) ? Essayez-le : de combien monte la précision, et combien d'appels coûte-t-il ?
2. Pourquoi l'appel de fonctions natif est-il plus fiable que ReAct écrit à la main avec le **même** modèle et les **mêmes** outils ? Qu'est-ce qui a été appris pendant l'affinage du modèle, et qu'est-ce que l'API fait à notre place ?
3. Le juge avec référence a un accord de 80 % : est-ce suffisant pour classer deux modèles dont les scores diffèrent de 5 points ? Que changent ses faux positifs (jamais de faux négatifs) au classement ?
4. Comment corriger le biais de position dans une évaluation par paires ? (Indice : présenter chaque paire dans les deux ordres et ne compter que les verdicts cohérents, comme le recommandent Zheng et al., 2023.)

## Pour aller plus loin

- Remplacer `chercher` par le moteur BM25 ou par les embeddings de l'exercice 32 et mesurer l'effet sur les tâches de recherche.
- Ajouter un troisième outil (une date du jour, une conversion d'unités) et une tâche qui en a besoin : le modèle choisit-il le bon outil ?
- Écrire vos propres réponses dans `juge.json`, en particulier des réponses fausses longues et sûres d'elles (biais de verbosité).
