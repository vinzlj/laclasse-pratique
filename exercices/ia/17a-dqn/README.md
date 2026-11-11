# Exercice 17a : DQN sur CartPole, à la main puis avec Stable-Baselines3

| | |
|---|---|
| **Chapitre lié** | [17. Renforcement profond et imitation](https://laclasse.poneyponey.fr/ia/p06-deep-learning/17-renforcement-profond/) |
| **Exécution** | locale, CPU (PyTorch sur un seul fil, gymnasium, Stable-Baselines3 ; aucun GPU : le réseau a 17 410 paramètres et l'environnement avance un pas à la fois) |
| **Commande** | `make ex NUM=17a` (éditeur marimo), `make ex NUM=17a MODE=run` (complet), `make ex NUM=17a MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 7 min 05 s, dont 12 entraînements maison de 26 à 37 s (6 min 06 s) et 55 s pour Stable-Baselines3 ; smoke : 6 s (M1 Pro, CPU, 15/10/2026, attente du verrou exclue) |
| **Travail estimé** | 1 h 30 |
| **Seed** | 42 (`seed_everything(42)`) ; graines 42, 43 et 44 pour les trois entraînements de chaque variante |
| **Données** | aucune : environnement simulé `CartPole-v1` de gymnasium ≥ 1.4 (groupe uv `rl`, installé par défaut) |

## Objectif

Passer du Q-learning tabulaire du chapitre 11 à **DQN** : remplacer la table par un réseau, puis vérifier par l'expérience à quoi servent ses deux garde-fous, la **mémoire de rejeu** et le **réseau cible**, et constater que même avec eux l'apprentissage reste capricieux d'une graine à l'autre.

Ce qui est optimisé : l'erreur TD passée dans la perte de Huber, $\operatorname{Huber}\big(r + \gamma\,(1 - \text{fini})\max_{a'} Q_{\bar\theta}(s', a') - Q_\theta(s, a)\big)$, moyennée sur un lot tiré de la mémoire ; la cible (calculée avec le réseau cible $Q_{\bar\theta}$) ne reçoit pas de gradient.

## Ce que vous allez faire

1. Lire `entrainer_dqn` (une centaine de lignes, dans l'esprit de CleanRL) : ε décroissant, mémoire circulaire, cible, perte de Huber, copie du réseau cible tous les 500 pas.
2. Entraîner DQN sur **trois graines** (80 000 pas chacune ; réseau 4 → 128 → 128 → 2, Adam $10^{-3}$, $\gamma = 0{,}98$, lots de 128, une mise à jour tous les 2 pas, ε de 1 à 0,05 sur 5 000 pas). La politique gloutonne est évaluée sur 10 épisodes tous les 2 500 pas, puis sur 20 épisodes à la fin.
3. Refaire la même chose **sans réseau cible** (la cible est calculée avec le réseau qu'on corrige), puis **sans mémoire de rejeu** (on apprend sur les 128 dernières transitions, consécutives), puis avec **Double DQN**.
4. Entraîner `DQN("MlpPolicy", env)` de Stable-Baselines3, d'abord avec ses réglages par défaut, puis avec les nôtres, au même budget.

## Résultat attendu

Le run complet vérifie 6 seuils et s'arrête avec une erreur si l'un manque. Valeurs mesurées le 15/10/2026 (graines 42, 43, 44 ; CPU, résultat identique d'un lancement à l'autre) :

| Variante | Seuil vérifié | Score final (20 épisodes, max. 500) | Évaluations à 500 (sur 32, par graine) | Q prédit en fin d'entraînement |
|---|---|---|---|---|
| DQN complet | chaque graine atteint 500 au moins une fois ; score final moyen **≥ 400** | 500 ; 500 ; 500 | 9 ; 3 ; 4 | 42 à 45 |
| Sans réseau cible | moyenne des évaluations < moitié de celle de DQN ; Q final > 1 000 | 9,4 ; 9,4 ; 9,3 | 0 ; 0 ; 0 | 1,5 à 1,8 × 10⁹ |
| Sans mémoire de rejeu | score final moyen ≤ 60 % de celui de DQN ; moins d'évaluations à 500 | 348,9 ; 117,8 ; 213,2 (moyenne 226,6) | 0 ; 1 ; 0 | 40 à 44 |
| Double DQN | chaque graine atteint 500 au moins une fois | 91,0 ; 163,8 ; 147,2 | 3 ; 4 ; 2 | 44 à 47 |

Moyenne de toutes les évaluations : 225 (DQN), 9 (sans cible), 179 (sans rejeu), 228 (Double DQN). Valeur maximale possible d'un état avec $\gamma = 0{,}98$ et 500 pas : $\sum_{k<500} 0{,}98^k \approx 50$.

Stable-Baselines3, 80 000 pas, graine 42 (non vérifié par un seuil, à lire comme un constat) : réglages par défaut 228,3 ± 41,4 en 17 s ; nos hyperparamètres 172,8 ± 6,1 en 38 s.

Figures produites dans `figures/` : `dqn-graines.png` (trois graines et valeurs prédites), `ablations.png` ; résultats chiffrés dans `figures/resultats.json` (lus par les figures du site).

## Ce qu'il faut observer

- **Même recette, trois histoires.** Les trois graines de DQN complet passent par de longues phases entre 100 et 250, avec des pics isolés à 500 suivis de rechutes (graine 42 : 500 à 30 000 pas, 115 à 32 500, de nouveau 500 à 40 000, puis 102 à 42 500). En fin d'entraînement, les graines 42 et 44 tiennent 500 à chaque évaluation à partir de 65 000 et 72 500 pas ; la graine 43 oscille encore sur ses trois dernières évaluations (500 à 75 000 pas, 347 à 77 500, 500 à 80 000). En renforcement profond, une seule courbe ne prouve rien : il faut plusieurs graines.
- **Sans réseau cible, la valeur diverge.** La cible est calculée avec les poids qu'on modifie : chaque correction déplace aussi la cible, dans le même sens. Q passe de quelques milliers (dès 2 500 pas) à plus d'un milliard alors qu'aucun retour ne peut dépasser 50, et l'agent ne tient jamais plus de 10 pas. C'est la triade mortelle du chapitre 11 (approximation, *bootstrapping*, hors politique) sous sa forme la plus brutale.
- **Sans mémoire de rejeu, l'agent progresse moins et oublie.** Les 128 dernières transitions se suivent : elles décrivent une seule situation, et chaque mise à jour fait oublier les autres. Moyenne de toutes les évaluations : 179 contre 225 pour DQN ; score final moyen 227 contre 500 ; une seule évaluation sur 96 atteint 500.
- **Double DQN fait comme DQN sur l'ensemble de l'entraînement.** Moyenne de toutes les évaluations : 228 contre 225. Ses scores finaux plus bas (91, 164, 147) tiennent au moment où l'on s'arrête : ses trois graines atteignent 500 puis retombent, comme celles de DQN plus tôt. Il corrige la surestimation due au $\max$, mais avec deux actions la surestimation est faible (les Q prédits sont les mêmes, 42 à 47) : l'instabilité vient d'ailleurs. L'article de van Hasselt et al. (prépublication 2015, AAAI 2016) mesure son effet sur les jeux Atari, où les actions sont nombreuses.
- **La bibliothèque n'est pas magique.** Avec les mêmes hyperparamètres, Stable-Baselines3 finit à 173 là où notre code finit à 500 : les deux implémentations ne font pas exactement les mêmes opérations dans le même ordre, et DQN est sensible à ces détails comme à la graine (une seule graine ici pour SB3) ; dans l'exercice 17b, le PPO de Stable-Baselines3 résout CartPole avec ses réglages par défaut (500).

## Questions de réflexion

1. Pourquoi enregistre-t-on `termine` et non `tronque` dans la mémoire ? Que se passerait-il si l'agent croyait que l'épisode coupé à 500 pas était une chute ?
2. Le réseau cible est recopié tous les 500 pas. Que prévoyez-vous si on le recopie à chaque pas ? Tous les 10 000 pas ?
3. Le score final de DQN complet est 500 pour les trois graines, mais il aurait été entre 113 et 123 selon la graine si l'entraînement s'était arrêté à 37 500 pas. Comment choisir le réseau à garder en pratique ?
4. DQN apprend la politique gloutonne en jouant ε-glouton, sur des transitions anciennes : en quoi est-ce « hors politique », et pourquoi est-ce justement ce qui permet la mémoire de rejeu ?

## Pour aller plus loin

- Ajouter une **mise à jour douce** du réseau cible ($\bar\theta \leftarrow 0{,}995\,\bar\theta + 0{,}005\,\theta$ à chaque pas) et comparer aux copies tous les 500 pas.
- Remplacer la perte de Huber par la MSE : la divergence sans réseau cible est-elle plus rapide ?
- L'article fondateur : Mnih et al., *Playing Atari with Deep Reinforcement Learning*, 2013, https://arxiv.org/abs/1312.5602 ; la version *Nature* de 2015 ajoute un réseau cible distinct, recopié tous les C pas (doi:10.1038/nature14236). Double DQN : van Hasselt, Guez et Silver, https://arxiv.org/abs/1509.06461.
- CleanRL (Huang et al., *JMLR* 23, 2022), https://jmlr.org/papers/v23/21-1342.html : des implémentations en un fichier, comme celle-ci ; Stable-Baselines3 (Raffin et al., *JMLR* 22, 2021), https://jmlr.org/papers/v22/20-1364.html.
