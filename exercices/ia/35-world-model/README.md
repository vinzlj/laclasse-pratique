# Exercice 35 : un modèle du monde pour CartPole

| | |
|---|---|
| **Chapitre lié** | [35. Modèles du monde](https://laclasse.poneyponey.fr/ia/p09-absorption/35-modeles-du-monde/) |
| **Exécution** | locale, CPU (numpy pour la physique et la planification, PyTorch sur 4 fils pour entraîner les MLP, gymnasium, Stable-Baselines3 pour PPO) |
| **Commande** | `make ex NUM=35` (éditeur marimo), `make ex NUM=35 MODE=run` (complet), `make ex NUM=35 MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 2 min 28 s (attente du verrou exclue), dont grille MPC 111 s, PPO 13 s, entraînement des 4 MLP 8,5 s, exploitation 11 s ; smoke : 8 s (M1 Pro, CPU, 16/10/2026) |
| **Travail estimé** | 1 h 30 |
| **Seed** | 42 (`seed_everything(42)`) ; mêmes graines d'environnement (100 à 104) pour toutes les évaluations |
| **Données** | aucune téléchargée : les transitions sont jouées au hasard dans `CartPole-v1` au début du run (gymnasium ≥ 1.4 et Stable-Baselines3, groupe uv `rl`, installé par défaut) |

## Objectif

Voir ce qu'apporte un **modèle du monde** : apprendre, sur quelques centaines de transitions réelles jouées au hasard, un modèle qui prédit l'état suivant, puis **planifier** dans ce modèle (commande prédictive par tirages aléatoires) au lieu d'apprendre une politique par essais et erreurs. Mesurer le gain en interactions réelles face à PPO, puis les deux faiblesses du chapitre : l'erreur qui **s'accumule** quand on déroule le modèle, et le planificateur qui **exploite** les erreurs du modèle.

Ce qui est optimisé : seul le modèle du monde apprend, par **erreur quadratique moyenne** sur la variation d'état, $\mathcal{L}(\theta) = \frac{1}{N}\sum_i \lVert \hat f_\theta(s_i, a_i) - (s_i' - s_i) \rVert^2$ (moindres carrés exacts pour le modèle linéaire, Adam sur les variations normalisées pour le MLP). La planification n'apprend rien : elle cherche, à chaque pas, la meilleure suite d'actions dans le modèle. La fonction de récompense est supposée **connue** (hypothèse de Nagabandi et al., 2018, et de PETS) ; Dreamer et MuZero l'apprennent aussi.

## Ce que vous allez faire

1. **En numpy** : réécrire la physique de CartPole pour un tableau d'états (`physique`), et vérifier qu'elle donne exactement les états de gymnasium.
2. **Planifier par tirages aléatoires** (`planifier`) : à chaque pas réel, tirer N = 200 suites de H = 10 actions, les dérouler dans le modèle, noter chaque futur par la récompense adoucie $\tilde r = 1 - \tfrac12(\theta/12°)^2 - \tfrac12(x/2{,}4)^2$ tant que la perche tient, jouer la première action du meilleur, recommencer. D'abord avec le modèle parfait (la vraie physique).
3. **Un premier modèle appris** : régression linéaire de $s' - s$ sur $[s, \pm 1, 1]$ par `np.linalg.lstsq`, sur 100 transitions jouées au hasard (5 épisodes).
4. **Un modèle générique** (`ModeleMonde`) : MLP 5 → 128 → 128 → 4 (SiLU), entraîné 3 000 pas (lots de 256, Adam $10^{-3}$) sur 100, 300, 1 000 et 3 000 transitions au hasard. Pour planifier vite, ses poids sont recopiés dans numpy (calcul en float32).
5. **Erreur accumulée** (`erreur_deroulee`) : sur 5 trajectoires réelles de 100 pas, rejouer les mêmes actions dans le modèle sans jamais le recaler, et mesurer l'écart sur l'angle après h = 1 à 80 pas.
6. **Horizon de planification** : MPC avec le MLP (100, 300 et 3 000 transitions), H = 5, 10, 20, 40, 5 épisodes chacun.
7. **Exploitation des erreurs** (`mesurer_exploitation`) : depuis 100 états tirés plus largement que les données ($|x|, |\dot x|, |\dot\theta| \le 1$, $|\theta| \le 8°$), tirer N = 10, 100, 1 000 plans de 40 pas, garder celui que le modèle note le mieux, et noter les mêmes plans dans la vraie physique.
8. **PPO** (Stable-Baselines3, réglages par défaut) arrêté après 2 048 à 32 768 pas, évalué sur les mêmes épisodes.

## Résultat attendu

Le run complet vérifie 10 seuils et s'arrête avec une erreur si l'un manque. Valeurs mesurées le 16/10/2026, identiques sur deux lancements (CPU, graine 42) :

| Partie | Seuil vérifié | Mesuré |
|---|---|---|
| Physique numpy | écart à gymnasium **≤ 1e-9** | **0** (200 pas) |
| Modèle parfait, H = 10 | 500 à chaque épisode | **500** × 5 |
| Modèle linéaire, 100 transitions | retour moyen **≥ 475** | **500** × 5 |
| MLP, 300 transitions, H = 10 | retour moyen **≥ 450** | **500** × 5 |
| Accumulation (MLP, 1 000 transitions) | écart à 40 pas **≥ 10 ×** écart à 5 pas | 0,135° contre 0,0014° (× 96) |
| MLP, 100 transitions, H = 40 | retour moyen **≤ 300** | **290,4** (181 à 388) |
| MLP, 3 000 transitions, H = 40 | retour moyen **≥ 450** | **466,6** (382 à 500) |
| Exploitation (MLP, 100 transitions, N = 1 000) | plan retenu surestimé d'au moins 5 points de plus que la moyenne | **+11,9** contre −1,2 |
| Exploitation | surestimation et regret croissants de N = 10 à N = 1 000 | +4,2 → +11,9 ; regret 7,3 → 12,4 |
| PPO, 4 096 pas | retour moyen **≤ 200** | **109,0** |

Écart médian sur l'angle (degrés) après h pas en boucle ouverte :

| Modèle | h = 1 | 2 | 5 | 10 | 20 | 40 | 80 |
|---|---|---|---|---|---|---|---|
| MLP, 100 transitions | 0,006 | 0,014 | 0,046 | 0,120 | 0,410 | 2,434 | 66,1 |
| MLP, 300 transitions | 0,001 | 0,001 | 0,003 | 0,014 | 0,058 | 0,369 | 7,77 |
| MLP, 1 000 transitions | < 0,001 | < 0,001 | 0,001 | 0,005 | 0,026 | 0,135 | 3,18 |
| MLP, 3 000 transitions | 0,001 | 0,001 | 0,004 | 0,009 | 0,024 | 0,126 | 3,13 |
| linéaire, 100 transitions | < 0,001 | 0,001 | 0,007 | 0,028 | 0,130 | 0,823 | 17,2 |

Retour moyen du MPC avec le MLP (5 épisodes, entre parenthèses min et max) :

| Transitions | H = 5 | H = 10 | H = 20 | H = 40 |
|---|---|---|---|---|
| 100 | 500 | 500 | 491,4 (457-500) | 290,4 (181-388) |
| 300 | 500 | 500 | 499,4 (497-500) | 411,8 (257-500) |
| 3 000 | 500 | 500 | 482,8 (424-500) | 466,6 (382-500) |

Exploitation (100 états, plans de 40 pas, note maximale 40) :

| Modèle | N | écart moyen (tous les candidats) | écart du plan retenu | regret |
|---|---|---|---|---|
| MLP, 100 transitions | 10 | −1,34 | +4,25 | 7,28 |
| | 100 | −1,19 | +10,19 | 11,86 |
| | 1 000 | −1,15 | +11,89 | 12,37 |
| MLP, 3 000 transitions | 10 | −0,71 | +3,48 | 4,80 |
| | 100 | −0,45 | +6,58 | 7,30 |
| | 1 000 | −0,44 | +7,57 | 7,92 |
| parfait | tous | 0 | 0 | 0 |

PPO (action la plus probable, 5 épisodes) : 98,6 après 2 048 pas ; 109,0 après 4 096 ; 360,0 après 8 192 ; 414,6 après 16 384 ; 500 après 32 768.

Figure produite dans `figures/` : `modele-monde.png` (erreur selon l'horizon, retour selon H, comparaison avec PPO) ; résultats chiffrés dans `figures/resultats.json` (lus par les figures du site, `site/scripts/figures/ch35_*.py`).

## Ce qu'il faut observer

- **Cent essais au lieu de trente mille.** Avec un modèle appris sur 100 transitions jouées au hasard (5 épisodes ratés), le planificateur tient 500 pas à chaque épisode ; PPO, sans modèle, n'en est qu'à 109 après 4 096 pas et n'atteint 500 qu'après 32 768. Le prix est ailleurs : à chaque pas réel, le planificateur imagine 200 × 10 pas, et la récompense lui est donnée. On échange des essais réels (chers pour un robot) contre du calcul.
- **Le linéaire suffit sur CartPole, et c'est une particularité.** Près de la verticale, $\sin\theta \approx \theta$ : la physique est presque linéaire, et le modèle linéaire prédit $x$ et $\theta$ à $10^{-17}$ près (ce sont de simples intégrations d'Euler). Le MLP, qui ne sait rien de la physique, fait aussi bien avec 300 transitions : c'est lui qui se généralise à une image, une main ou un jeu vidéo.
- **L'erreur s'accumule, et de plus en plus vite.** Une erreur de 0,006° à un pas devient 2,4° à 40 pas et 66° à 80 pas (la perche réelle n'a jamais dépassé quelques degrés). Chaque prédiction devient l'entrée de la suivante, et le système est instable. Plus de données retardent la dérive sans la supprimer : de 1 000 à 3 000 transitions, l'écart à 80 pas ne bouge presque plus (3,2° puis 3,1°), à nombre de pas d'optimisation fixé (3 000) ; le modèle à 3 000 transitions est même un peu moins précis à court horizon.
- **Voir loin dans un modèle faux ne paie pas.** Avec H = 5 ou 10, tous les modèles tiennent 500 pas ; avec H = 40, le modèle à 100 transitions tombe à 290 en moyenne. D'où les horizons courts des méthodes récentes (3 pas de planification pour TD-MPC2, une quinzaine de pas imaginés pour Dreamer et DreamerV3 (H = 15 dans Dreamer, T = 16 dans le texte de DreamerV3), qui complètent tous deux par une fonction de valeur au bout de l'horizon), et le replanning à chaque pas.
- **Le planificateur trouve les erreurs du modèle.** En moyenne, le MLP pauvre est légèrement *pessimiste* (−1,2 point sur 40). Pourtant, le plan qu'il préfère est surestimé de 4 points avec 10 candidats et de 12 avec 1 000, et le regret suit. Chercher plus fort dans un modèle imparfait, c'est surtout trouver les endroits où il se trompe en votre faveur. C'est le problème que MBPO traite en ne déroulant le modèle que sur de courts morceaux, et PETS en planifiant avec un ensemble de modèles.

## Questions de réflexion

1. Pourquoi prédire la **variation** $s' - s$ plutôt que l'état suivant $s'$ ? Que vaudrait le modèle « ne rien changer » ($s' = s$) dans chacun des deux cas ?
2. Avec la vraie récompense (1 par pas debout) au lieu de la récompense adoucie, le planificateur au modèle parfait et H = 10 ne tient plus 500 pas. Pourquoi ? Quel lien avec le facteur d'actualisation et la fonction de valeur du chapitre 11 (que Dreamer et TD-MPC ajoutent au bout de l'horizon) ?
3. Le regret du MLP pauvre passe de 7,3 à 12,4 quand N va de 10 à 1 000. Que se passerait-il avec le modèle parfait et N = 1 000 000 ? Proposez une façon de planifier qui pénalise les futurs où le modèle est incertain (piste : entraîner cinq MLP sur des tirages différents des données et regarder leur désaccord).
4. Les transitions jouées au hasard restent près de la verticale. Que devrait faire l'agent pour améliorer son modèle là où le planificateur l'emmène ? Comparez avec Dyna (chapitre 11) et avec la boucle « collecter, réapprendre, replanifier » de Dreamer.

## Pour aller plus loin

- Remplacer les tirages aléatoires par la **méthode de l'entropie croisée** (CEM) : garder les 10 % meilleures suites, réestimer la probabilité de chaque action à chaque pas, recommencer trois fois.
- Planifier avec un **ensemble** de 5 MLP (moyenne ou pire des notes) et refaire la mesure d'exploitation.
- Ajouter une boucle d'apprentissage : jouer quelques épisodes avec le MPC, ajouter leurs transitions au jeu, réentraîner (c'est l'algorithme de Nagabandi et al.).
- Articles : Nagabandi, Kahn, Fearing et Levine, *Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning*, ICRA 2018, https://arxiv.org/abs/1708.02596 ; Chua, Calandra, McAllister et Levine, *Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models* (PETS), NeurIPS 2018, https://arxiv.org/abs/1805.12114 ; Janner, Fu, Zhang et Levine, *When to Trust Your Model: Model-Based Policy Optimization* (MBPO), NeurIPS 2019, https://arxiv.org/abs/1906.08253.
