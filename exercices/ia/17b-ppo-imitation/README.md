# Exercice 17b : PPO sur CartPole, puis imiter l'expert (BC et DAgger)

| | |
|---|---|
| **Chapitre lié** | [17. Renforcement profond et imitation](https://laclasse.poneyponey.fr/ia/p06-deep-learning/17-renforcement-profond/) |
| **Exécution** | locale, CPU (PyTorch sur un seul fil, gymnasium, Stable-Baselines3 ; aucun GPU : réseaux de quelques milliers de paramètres) |
| **Commande** | `make ex NUM=17b` (éditeur marimo), `make ex NUM=17b MODE=run` (complet), `make ex NUM=17b MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : environ 2 min 55 s (PPO maison : 6 entraînements de 14 à 15 s, 1 min 30 s en tout ; SB3 30 s ; imitation, avec le témoin, 48 s) ; smoke : 9 s (M1 Pro, CPU, 16/10/2026, attente du verrou exclue) |
| **Travail estimé** | 2 h |
| **Seed** | 42 (`seed_everything(42)`) ; graines 42, 43 et 44 pour les trois entraînements de chaque variante |
| **Données** | aucune : environnement simulé `CartPole-v1` de gymnasium ≥ 1.4 (groupe uv `rl`, installé par défaut) |

## Objectif

Écrire PPO à la main (avantages par GAE, ratio écrêté, plusieurs époques sur le même lot), mesurer ce que l'écrêtage empêche, retrouver le résultat avec Stable-Baselines3, puis se servir de la politique obtenue comme **expert** pour comparer deux façons d'imiter : le clonage comportemental et DAgger.

Ce qui est optimisé : PPO minimise $-\hat{\mathbb{E}}_t[\min(r_t \hat A_t, \operatorname{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\hat A_t)] + 0{,}5\,\hat{\mathbb{E}}_t[(V_\phi(s_t) - \hat R_t)^2] - 0{,}01\,\hat{\mathbb{E}}_t[\mathcal{H}(\pi_\theta(\cdot \mid s_t))]$ (objectif écrêté, erreur du critique, bonus d'entropie), avec $\varepsilon = 0{,}2$ ; l'imitation minimise l'**entropie croisée** entre l'action de l'expert et la politique de l'élève.

## Ce que vous allez faire

1. Calculer à la main l'objectif écrêté pour $r = 1{,}3$ et $r = 0{,}5$ (fonction `objectif_clip`, la même que dans la démo du chapitre).
2. Lire `entrainer_ppo` : collecte de 4 × 128 pas, avantages GAE(λ = 0,95), 10 époques de 4 mini-lots, pas d'apprentissage $10^{-3}$ décroissant jusqu'à 0. Trois graines, 100 000 pas chacune.
3. Ablation : la même chose **sans écrêtage** (objectif $r_t \hat A_t$). Après chaque mise à jour, on mesure la divergence KL entre ancienne et nouvelle politique, estimée par la moyenne de $(r - 1) - \log r$, et le plus grand ratio $r$ du lot.
4. `PPO("MlpPolicy", env)` de Stable-Baselines3 avec ses réglages par défaut, même budget.
5. Imitation : l'expert est la politique PPO de la graine 42. Pour 25 à 800 états étiquetés par l'expert, entraîner un élève 4 → 32 → 2 par **clonage comportemental** (les N premiers états de démonstrations enregistrées par l'expert), par **clonage comportemental sur N états tirés au hasard** dans ces démonstrations (un témoin, qui isole l'effet de la diversité des états) et par **DAgger** (N/4 états de démonstration, puis trois tours de N/4 états *visités par l'élève*, étiquetés par l'expert). L'élève est évalué avec 5 % d'actions remplacées au hasard. Cinq répétitions (démonstrations différentes).

## Résultat attendu

Le run complet vérifie 10 seuils et s'arrête avec une erreur si l'un manque. Valeurs mesurées le 15/10/2026, confirmées à l'identique le 16/10/2026 (graines 42, 43, 44 ; CPU, résultat identique d'un lancement à l'autre) :

| Partie | Seuil vérifié | Mesuré |
|---|---|---|
| Objectif écrêté | r = 1,3, Â = +1 → 1,2 ; r = 0,5, Â = −1 → −0,8, les deux écrêtés (gradient nul) | identique ; r = 1,3, Â = −1 → −1,3 (terme brut, gradient actif) |
| PPO | politique gloutonne finale **≥ 475** pour chaque graine (20 épisodes) | 500 ± 0 pour les trois graines ; 14 à 16 s par entraînement |
| PPO | après avoir atteint 500, pire évaluation ≥ 400 pour au moins 2 graines sur 3 | 205,5 (graine 42, une seule rechute, à 15 360 pas) ; 500 et 500 |
| Sans écrêtage | KL médiane par mise à jour **≥ 3 ×** celle de PPO | 0,083 contre 0,0035 (× 24) ; KL max 4,6 contre 0,023 |
| Sans écrêtage | ratio r maximal **≥ 10 ×** celui de PPO | 192 contre 3,8 |
| Sans écrêtage | au moins 2 graines sur 3 retombent **sous 200** après avoir atteint 500 | 24,4 ; 9,2 ; 106,1 (scores finaux 500 ; 304,9 ± 239 ; 500) |
| SB3 PPO (défauts) | score ≥ 475 | 500 ± 0 en 30 s |
| Expert | ≥ 475 avec 5 % de bruit d'action | 500 |
| Imitation | à chaque budget où BC reste sous 400, DAgger fait au moins 100 de plus | voir le tableau ci-dessous |
| Imitation | au plus grand budget, DAgger ≥ 90 % de l'expert | 500 |

Retour moyen de l'élève (5 répétitions × 20 épisodes, 5 % de bruit d'action ; expert : 500) :

| États étiquetés par l'expert | 25 | 50 | 100 | 200 | 400 | 800 |
|---|---|---|---|---|---|---|
| Clonage comportemental (N premiers états) | 105,3 | 187,4 | 375,9 | 460,8 | 436,2 | 477,8 |
| Clonage comportemental (N états tirés au hasard, témoin) | 281,6 | 458,6 | 500,0 | 500,0 | 500,0 | 500,0 |
| DAgger | 500,0 | 441,3 | 484,9 | 500,0 | 500,0 | 500,0 |

Le témoin n'est soumis à aucun seuil : c'est un constat (ajouté le 16/10/2026 ; les deux autres lignes n'ont pas changé).

Figures produites dans `figures/` : `ppo.png` (évaluations et KL des deux variantes), `imitation.png` ; résultats chiffrés dans `figures/resultats.json` (lus par les figures du site).

## Ce qu'il faut observer

- **L'écrêtage ne bloque que les excès « dans le bon sens ».** Avec ε = 0,2, une action à avantage positif dont la probabilité a déjà été multipliée par 1,3 ne rapporte plus rien à renforcer davantage (gradient nul) ; mais une action à avantage négatif rendue *plus* probable garde un gradient pour corriger l'erreur.
- **La KL dit de combien la politique bouge.** Avec écrêtage, chaque mise à jour déplace la politique d'environ 0,0035 nat (médiane) ; sans écrêtage, d'environ 0,08 nat, avec des pointes au-delà de 4 nat et des ratios r de 100 à 190 : sur certaines transitions, l'action est devenue cent fois plus probable qu'au moment de la collecte. Les 10 époques sur le même lot n'ont plus aucun garde-fou.
- **Sans écrêtage, on atteint 500… puis on retombe.** Les trois graines sans écrêtage touchent 500 plusieurs fois, mais s'effondrent ensuite à 24, 9 et 106 : une seule mise à jour trop grande détruit une bonne politique. PPO n'a qu'une rechute, au tout début (graine 42 : 500 à 10 240 pas, 206 à 15 360 pas), et ses trois graines restent à 500 à chaque évaluation à partir de 25 600 pas.
- **Stable-Baselines3 fait la même chose**, avec d'autres réglages (lots de 2 048 pas, 10 époques, mini-lots de 64) : 500 en 30 s. La bibliothèque cache les 150 lignes de `entrainer_ppo` ; les détails (normalisation des avantages, correction des épisodes coupés à 500 pas, initialisation orthogonale) sont ceux qui font marcher l'algorithme.
- **Clonage comportemental : il faut beaucoup de démonstrations.** Avec 25 ou 50 états d'une démonstration, l'élève n'a vu que le début d'un épisode bien joué : au premier écart (et le bruit d'action en crée), il arrive dans des états inconnus et le bâton tombe (105 et 187 en moyenne). Il lui faut quelques centaines d'états pour approcher l'expert, et il reste irrégulier (436 à 400 états).
- **Une bonne part de l'écart tient à la diversité.** Le témoin reçoit les mêmes démonstrations, mais des états tirés au hasard dans plusieurs épisodes : il atteint 500 dès 100 états. Les N premiers états d'une démonstration se suivent et décrivent une seule situation, le début d'un épisode.
- **DAgger corrige là où l'élève se trompe.** À nombre d'étiquettes égal, les états étiquetés par l'expert sont ceux *où l'élève est allé*, y compris ceux où il était en train de perdre. Son apport propre, au-delà de la diversité, se voit avec 25 étiquettes : 500, contre 282 pour le témoin. Avec si peu d'étiquettes, la variance est forte : DAgger fait 441 à 50 états, parce qu'une répétition sur cinq tombe à 211. C'est la distinction centrale du chapitre : imiter des trajectoires contre apprendre à se rattraper.

## Questions de réflexion

1. Dans l'objectif écrêté, pourquoi prend-on le **minimum** des deux termes plutôt que le seul terme écrêté ? Que se passerait-il pour une action à avantage négatif rendue plus probable ($\hat A < 0$, $r = 1{,}3$) ?
2. Sans écrêtage, la politique finit souvent à 500 quand même. Pourquoi est-ce un mauvais critère pour juger un algorithme de RL, et que montre la courbe des évaluations que le score final cache ?
3. DAgger exige un expert qu'on peut **interroger** sur n'importe quel état. Dans quelles situations réelles est-ce possible (simulateur, humain devant un écran) et dans lesquelles ne l'est-ce pas (conduite réelle) ?
4. L'expert d'ici a été obtenu par renforcement. Pourquoi, dans les LLM, commence-t-on par une phase d'imitation (réglage supervisé sur des réponses humaines) avant le renforcement (chapitre 30) ?

## Pour aller plus loin

- Tracer la fraction écrêtée (`frac_ecretee`) au fil de l'entraînement, et la comparer à la KL.
- Remplacer l'écrêtage par une pénalité $\beta \cdot \mathrm{KL}$ envers l'ancienne politique (variante de la section 4 de l'article de PPO ; à ne pas confondre avec la pénalité KL du RLHF, envers le modèle de départ, chapitre 30) et chercher un β qui donne la même stabilité.
- Rendre l'élève plus petit (8 neurones cachés) : l'écart entre BC et DAgger grandit-il ?
- L'article de PPO : Schulman et al., *Proximal Policy Optimization Algorithms*, 2017, https://arxiv.org/abs/1707.06347 ; DAgger : Ross, Gordon et Bagnell, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*, AISTATS 2011, https://arxiv.org/abs/1011.0686.
- *The 37 Implementation Details of Proximal Policy Optimization* (ICLR Blog Track, 2022), https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ : tous les détails que `entrainer_ppo` reprend de CleanRL.
