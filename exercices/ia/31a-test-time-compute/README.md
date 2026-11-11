# Exercice 31a : dépenser du calcul à l'inférence (chaîne de pensée, vote, vérificateur)

| | |
|---|---|
| **Chapitre lié** | [31. Raisonnement et calcul à l'inférence](https://laclasse.poneyponey.fr/ia/p08-llm/31-raisonnement-inference/) |
| **Exécution** | locale : LLM `qwen3.5:2b` servi par Ollama (GPU du Mac via Ollama) ; analyse en numpy. Aucun entraînement |
| **Commande** | `make ex NUM=31a` (éditeur), `make ex NUM=31a MODE=run` (complet), `MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 7 min 58 s (processus entier, attente du verrou exclue, modèle et données en cache ; 474 s de génération, 23 758 tokens, 53,6 tokens/s) ; smoke : 9 à 16 s (M1 Pro, Ollama 0.40.4, 15/10/2026) |
| **Travail estimé** | ~1 h 30 |
| **Seed** | 42 : tirage des 30 problèmes (`np.random.default_rng(42)`) ; graines Ollama 42 à 45 pour les 4 chaînes échantillonnées. Même à graine fixe, deux runs d'Ollama ne donnent pas exactement les mêmes textes (voir « Résultat attendu ») |
| **Données** | GSM8K (`openai/gsm8k`, licence MIT, partie test, révision figée), ≈ 0,4 Mo dans `data/hf/` ; modèle `qwen3.5:2b` (≈ 3 Go, Apache-2.0 d'après la carte `Qwen/Qwen3.5-2B` du Hub) dans le dossier d'Ollama |

## Prérequis

- Ollama ≥ 0.19 installé et lancé (application, ou `ollama serve`) ; testé avec 0.40.4.
- Le modèle, une fois : `ollama pull qwen3.5:2b`.
- Le notebook vérifie les deux au démarrage et s'arrête avec un message clair s'il manque quelque chose. Il décharge le modèle à la fin (`keep_alive=0`) pour libérer la mémoire partagée.

## Objectif

Mesurer ce que rapporte, en précision, chaque token dépensé **après** l'entraînement, sur des problèmes d'arithmétique en plusieurs étapes, et comparer deux façons de dépenser plus : une chaîne de raisonnement plus longue (en série) ou plusieurs chaînes et un vote (en parallèle). Puis calculer, sur les mêmes réponses, l'avantage de groupe qu'utilise GRPO.

## Ce que vous allez faire

1. Tirer 30 problèmes du test de GSM8K (graine 42) ; la réponse attendue est le nombre après `####` dans la solution de référence.
2. Vérifier qu'Ollama répond, puis interroger `qwen3.5:2b` avec `think=False` (le mode « réflexion » de Qwen 3.5, qu'Ollama active par défaut pour ce modèle, ajouterait des centaines de tokens cachés) et `presence_penalty=0` (Ollama applique 1,5 par défaut à ce modèle, ce qui pénalise chaque token déjà écrit, donc les nombres qu'un calcul recopie).
3. Pour chaque problème, quatre stratégies (consignes en anglais, comme les problèmes) :
   - **réponse directe** : « le nombre final seulement », température 0, 24 tokens au plus ;
   - **chaîne courte** : raisonnement en 6 lignes au plus puis `Answer: n`, **4 échantillons** à température 0,7 (graines 42 à 45), 384 tokens au plus ;
   - **chaîne libre** : « raisonne étape par étape » sans limite de lignes, 1 échantillon, 768 tokens au plus ;
   - **vote majoritaire** (*self-consistency*) sur les k premières chaînes courtes, k = 1 à 4 ; à égalité, la réponse rencontrée en premier.
4. Lire la réponse finale (`Answer: n`, sinon dernier nombre du texte ; un texte coupé par le budget sans ligne `Answer:` compte comme sans réponse) et la comparer à la référence.
5. Calculer en numpy la précision et le nombre moyen de tokens générés par problème pour chaque stratégie, et la précision d'un **vérificateur parfait** (juste si au moins une des 4 chaînes est juste, pass@4).
6. Calculer l'**avantage de groupe de GRPO** : pour chaque problème, récompenses r = 1 (juste) ou 0 sur les 4 chaînes, puis A = (r − moyenne) / écart-type ; compter les groupes sans signal.
7. Tracer la figure `figures/precision-cout.png` et écrire `figures/resultats.json` (relu par la figure du site).

## Résultat attendu

Run du 15/10/2026 (30 problèmes) :

| Stratégie | Précision | Tokens générés par problème |
|---|---|---|
| Réponse directe | 0 % (0/30) | 3 |
| Une chaîne courte (moyenne des 4 échantillons) | 42 % | 95 (1 % coupées) |
| Une chaîne libre | 63 % (19/30) | 410 (10 % coupées à 768) |
| Vote sur 1, 2, 3, 4 chaînes courtes | 33 %, 33 %, 47 %, 53 % | 88, 180, 278, 379 |
| Vérificateur parfait (pass@4) | 67 % | 379 |

Groupes GRPO sans signal : **14 sur 30 (47 %)** (10 problèmes à 0/4, 4 à 4/4). Exemple de groupe utile : récompenses [0, 1, 0, 0] → avantages [−0,58 ; 1,73 ; −0,58 ; −0,58].

Seuils vérifiés en mode complet (le notebook s'arrête sinon) : gain de la chaîne courte sur la réponse directe **≥ 25 points**, et vote sur 4 **≥ chaîne courte + 5 points**.

Variabilité : Ollama n'est pas reproductible au token près, même à graine fixe (constaté, cause non étudiée ici). Deux runs de la même configuration ont donné 42 % et 42 % pour une chaîne courte, mais 60 % puis 53 % pour le vote sur 4. Avec 30 problèmes, un problème vaut 3,3 points : ne pas lire d'écart de moins de ~10 points comme un résultat.

## Ce qu'il faut observer

- **Répondre directement ne marche pas** (0 %, contre 42 % avec une chaîne courte) : le modèle connaît les opérations, mais ne peut pas faire plusieurs étapes « de tête » en un seul token. Écrire les résultats intermédiaires lui sert de mémoire de travail. C'est l'effet de Wei et al. (2022), ici sans exemple dans l'invite.
- **Deux façons de dépenser plus** : à budget presque égal (~400 tokens), une chaîne libre (63 %) fait mieux que le vote sur 4 chaînes courtes (53 %) sur ces problèmes. Snell et al. (2024) trouvent que la meilleure répartition entre série et parallèle dépend de la difficulté du problème ; ici, nos chaînes courtes sont trop contraintes (6 lignes) pour certains problèmes, ce que le vote ne rattrape pas.
- **Le vote aide, mais pas au début** : vote sur 1 chaîne = 33 % (la première chaîne seule, moins bonne que la moyenne des 4 par hasard), sur 2 chaînes toujours 33 % (à égalité, on garde la première ; seule une première chaîne sans réponse peut être rattrapée), puis 47 % et 53 %. Il faut au moins 3 chaînes pour qu'une majorité existe.
- **L'écart vote/vérificateur parfait** (53 % contre 67 %) est ce qu'un bon vérificateur (modèle de récompense, PRM) pourrait encore gagner : la bonne réponse est souvent parmi les 4, mais minoritaire.
- **Les groupes sans signal** (47 %) : un entraînement GRPO sur ces 30 problèmes n'apprendrait rien de 14 d'entre eux, trop durs (0/4) ou déjà maîtrisés (4/4). C'est ce que filtre l'échantillonnage dynamique de DAPO, et ce que mesure la courbe `frac_reward_zero_std` de l'exercice 31b.

## Questions de réflexion

1. La chaîne libre est 4 fois plus chère qu'une chaîne courte. Si vous aviez 1 600 tokens par problème, comment les dépenseriez-vous, et que faudrait-il mesurer pour décider ?
2. Avec la règle d'égalité utilisée ici, dans quel seul cas le vote sur 2 chaînes peut-il faire mieux que la première chaîne seule ?
3. Un groupe a les récompenses [0, 1, 0, 0]. Calculez à la main les avantages (écart-type de population, sans correction). Pourquoi la réponse juste reçoit-elle un avantage plus grand en valeur absolue que chacune des fausses ?
4. Le vérificateur parfait suppose qu'on connaît la bonne réponse. Quels vérificateurs réels peut-on utiliser pour des problèmes de maths, et pour du code ?

## Pour aller plus loin

- Passer `n_echantillons` à 8 (≈ 4 min de plus) et tracer la courbe du vote jusqu'à 8.
- Activer le mode réflexion (`think=True`, en lisant `r.message.thinking`) sur 5 problèmes, avec `num_predict` ≈ 2 000 : combien de tokens, et quelle précision ?
- Remplacer le vote par un « juge » : redemander au même modèle laquelle des 4 réponses est juste, et comparer au vote.
