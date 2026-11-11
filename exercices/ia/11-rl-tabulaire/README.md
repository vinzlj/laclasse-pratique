# Exercice 11 : renforcement classique, version tabulaire

| | |
|---|---|
| **Chapitre lié** | [11. Apprentissage par renforcement classique](https://laclasse.poneyponey.fr/ia/p05-renforcement/11-renforcement-classique/) |
| **Exécution** | locale, CPU (NumPy, Python pur, gymnasium pour FrozenLake ; aucun GPU ni réseau de neurones) |
| **Commande** | `make ex NUM=11` (éditeur marimo), `make ex NUM=11 MODE=run` (complet), `make ex NUM=11 MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 35 s au total, dont 15,5 s pour REINFORCE et 6,4 s pour les 800 parties de morpion ; smoke : 5 s (M1 Pro, CPU, 14/10/2026, attente du verrou exclue) |
| **Travail estimé** | 2 h |
| **Seed** | 42 (`seed_everything(42)` ; chaque partie dérive ses générateurs `np.random.default_rng` de cette graine) |
| **Données** | aucune : environnements simulés (FrozenLake de gymnasium ≥ 1.4, groupe uv `rl`, installé par défaut ; les autres sont codés dans le notebook) |

## Objectif

Implémenter en NumPy les grandes méthodes du renforcement « à table » (une case par état ou par couple état-action) et mesurer ce que chacune apporte : explorer intelligemment, planifier quand on connaît le modèle, apprendre sans le connaître, apprendre la politique directement, planifier avec un modèle appris, chercher dans un arbre, et voir pourquoi tout cela devient instable dès qu'on remplace la table par une approximation.

Ce qui est optimisé : les méthodes de valeur (parties 3 à 5 et 7) réduisent l'**erreur TD** $\delta = r + \gamma \max_{a'} Q(s',a') - Q(s,a)$ par $Q(s,a) \leftarrow Q(s,a) + \alpha\,\delta$ (un semi-gradient : on ne dérive pas à travers la cible) ; REINFORCE (partie 6) **monte** le gradient de $J(\theta) = \mathbb{E}_{\pi_\theta}[G_0]$. Les parties 1, 2 et 8 n'optimisent aucune perte (moyennes empiriques, équation de Bellman, statistiques de visites).

## Ce que vous allez faire

1. **Bandits à 10 bras** (2 000 problèmes × 1 000 tirages, vectorisés) : glouton, ε-greedy (ε = 0,01 et 0,1) et UCB (c = 2).
2. **Programmation dynamique** sur FrozenLake 4 × 4 glissant (modèle lu dans `env.unwrapped.P`, γ = 0,99) : itération sur les valeurs et itération sur les politiques ; carte de valeur sur 4 × 4 et 8 × 8.
3. **Monte-Carlo contre TD(0)** sur la marche aléatoire à 5 états (100 répétitions, trois pas α chacun).
4. **SARSA contre Q-learning** sur la falaise 4 × 12 (ε = 0,1, α = 0,5, 500 épisodes, 20 répétitions).
5. **Q-learning** sur FrozenLake sans lire le modèle (`reset`/`step` seulement), 10 000 épisodes, comparé à l'optimum de l'étape 2.
6. **REINFORCE tabulaire** (politique softmax sur une table de préférences), avec et sans baseline, 15 000 épisodes.
7. **Dyna-Q** sur le labyrinthe 6 × 9 de Sutton et Barto, avec n = 0, 5 et 50 mises à jour imaginaires par pas réel.
8. **MCTS/UCT** au morpion, 10 à 1 000 itérations par coup, 100 parties contre un joueur au hasard et 100 contre un joueur parfait (minimax).
9. **Triade mortelle** : TD semi-gradient en espérance sur le contre-exemple de Baird (1995), dans la version légèrement modifiée de Sutton et Barto (2018, §11.2 : 7 états, 8 poids), puis en retirant un ingrédient à la fois.

## Résultat attendu

Le run complet vérifie 14 seuils et s'arrête avec une erreur si l'un manque. Valeurs mesurées le 14/10/2026 (graine 42) :

| Partie | Seuil vérifié | Mesuré |
|---|---|---|
| 1. Bandits (100 derniers pas) | UCB > ε = 0,1 > glouton en récompense ; ε = 0,1 choisit le meilleur bras **≥ 75 %**, le glouton **≤ 45 %** | récompense 1,504 / 1,379 / 1,036 (ε = 0,01 : 1,325) ; meilleur bras 86,1 % (UCB), 81,0 % (ε = 0,1), 60,2 % (ε = 0,01), 36,0 % (glouton) |
| 2. Programmation dynamique | mêmes valeurs par les deux méthodes (écart < 10⁻⁶) ; itération sur les politiques en ≤ 10 améliorations, sur les valeurs en ≥ 100 balayages | 7 améliorations contre 571 balayages, écart 2,8·10⁻⁹ ; V*(départ) = 0,542 |
| 2. FrozenLake | politique optimale **≥ 70 %** de réussite, politiques au hasard **≤ 5 %** | 74,2 % contre 2,2 % ; 8 × 8 : V*(départ) = 0,415, 62,5 % |
| 3. Marche aléatoire | après 100 épisodes, le meilleur TD(0) bat le meilleur Monte-Carlo | erreur RMS 0,035 (TD, α = 0,05) contre 0,082 (MC, α = 0,02) |
| 4. Falaise | SARSA gagne plus que Q-learning en explorant ; chemin glouton de Q-learning = 13 pas, SARSA plus long | −26,1 contre −49,9 par épisode ; 13 pas (Q-learning) contre 17 pas (SARSA) |
| 5. Q-learning FrozenLake | politique gloutonne à moins de **3 points** de l'optimum | 73,7 % (optimum 74,2 %) ; max Q(départ) = 0,537 pour V* = 0,542 ; même action que l'optimum dans 91 % des états |
| 6. REINFORCE | avec baseline, politique gloutonne **≥ 65 %** | 73,7 % avec comme sans baseline ; en jouant (1 000 derniers épisodes) 71,4 % avec, 70,6 % sans |
| 7. Dyna-Q | pas réels cumulés sur 50 épisodes : n = 50 < n = 5 < n = 0 | 1 689 < 2 442 < 5 521 ; 2ᵉ épisode en 39, 175 et 1 024 pas ; plus court chemin 14 pas |
| 8. UCT | 1 000 itérations : aucune défaite contre le hasard, **≥ 90 %** de nuls contre le parfait ; 10 itérations : perd la majorité contre le parfait | 1 000 it. : 93 V / 7 N / 0 D (hasard), 0 V / 96 N / 4 D (parfait) ; 10 it. : 70 défaites sur 100 contre le parfait |
| 9. Baird | la triade multiplie ‖w‖ par plus de 100 ; chaque variante privée d'un ingrédient reste sous 3 × la norme de départ | 10,4 → 5 280 après 2 000 balayages ; sans approximation 24,5, sans hors-politique 9,06, sans bootstrapping 8,63 (départ ≈ 10,3) |

Figures produites dans `figures/` : `bandits.png`, `valeurs.png`, `td.png` (marche aléatoire et falaise), `apprentissage.png` (Q-learning et REINFORCE), `planification.png` (Dyna-Q et UCT), `triade.png`.

## Ce qu'il faut observer

- **Bandits** : le glouton se fige sur le premier levier qui a payé ; explorer au hasard le sauve, explorer là où l'incertitude est grande (UCB) le sauve plus vite. Explorer coûte au début et rapporte ensuite.
- **Carte de valeur** : près des trous, la politique optimale vise souvent un mur (la case de départ indique « gauche ») : sur la glace on ne glisse jamais dans la direction opposée à l'action choisie, donc pousser contre le mur évite le trou. Même optimale, la politique échoue une fois sur quatre : c'est la glace, pas l'algorithme.
- **Itération sur les politiques** : 7 améliorations suffisent là où l'itération sur les valeurs fait 571 balayages ; chaque amélioration coûte plus cher (un système linéaire), mais les deux arrivent aux mêmes valeurs.
- **TD contre Monte-Carlo** : en s'appuyant sur sa propre estimation de l'état suivant (*bootstrapping*), TD(0) apprend plus vite que Monte-Carlo, qui attend la fin de chaque épisode, pour tous les α essayés.
- **Falaise** : Q-learning apprend le chemin optimal le long du vide mais y tombe en explorant ; SARSA évalue la politique qu'il joue réellement (avec ses 10 % de coups au hasard) et préfère un détour sûr. Hors politique contre sur politique, en une image.
- **Q-learning sans modèle** retrouve l'optimum de la programmation dynamique (73,7 % contre 74,2 %) sans jamais lire `P`, mais sa courbe de réussite *en jouant* plafonne vers 53 % : c'est le prix de l'exploration (ε ≥ 0,05) sur une glace qui punit chaque faux pas.
- **REINFORCE** atteint aussi la politique optimale. Le baseline accélère l'apprentissage (vers l'épisode 4 000, environ 55 % de réussite en moyenne glissante contre 30 % sans), mais sur un problème aussi petit les deux finissent au même niveau.
- **Dyna-Q** : chaque pas réel réutilisé 50 fois dans le modèle appris divise par plus de trois le nombre de pas réels nécessaires.
- **UCT** : sans aucune connaissance du jeu, seulement des simulations, il passe de 70 défaites sur 100 à 4 contre un joueur parfait en augmentant le budget de recherche.
- **Triade mortelle** : la vraie valeur (0 partout) est représentable, et pourtant les poids explosent. Il suffit de retirer un seul ingrédient (approximation, bootstrapping ou hors politique) pour que tout reste borné (sans approximation, la norme monte d'abord jusqu'à 24,5, le temps que les six premiers états rejoignent γ·V(s₇) ≈ 9,9, puis décroît très lentement vers 0). C'est la racine de l'instabilité des premiers DQN.

## Questions de réflexion

1. Dans FrozenLake, pourquoi la case de départ a-t-elle pour action optimale « gauche » alors que le but est en bas à droite ?
2. Sur la falaise, que deviendraient les chemins de SARSA et de Q-learning si ε décroissait jusqu'à 0 ? Lequel préféreriez-vous pour un robot réel qui apprend en marchant ?
3. TD(0) est plus rapide que Monte-Carlo sur la marche aléatoire. Dans quelle situation l'appui sur sa propre estimation devient-il un défaut (indice : partie 9) ?
4. UCT utilise la même formule que UCB pour les bandits. Qu'est-ce qui joue le rôle du « bras » à chaque nœud de l'arbre, et pourquoi ce choix vaut-il mieux qu'une exploration uniforme ?

## Pour aller plus loin

- Remplacer ε-greedy par UCB dans Q-learning sur FrozenLake et comparer la réussite *en jouant*.
- Ajouter une variante n-pas de TD sur la marche aléatoire (n = 2, 4, 8) et chercher le meilleur compromis entre TD(0) et Monte-Carlo.
- Rendre le labyrinthe de Dyna-Q changeant (un mur qui se déplace à l'épisode 25) : Dyna-Q s'adapte-t-il ? (c'est l'origine de Dyna-Q+, Sutton et Barto 2018, §8.3).
