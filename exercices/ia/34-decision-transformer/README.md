# Exercice 34 : le Decision Transformer, et les deux pièges du conditionnement par le retour

| | |
|---|---|
| **Chapitre lié** | [34. Le RL comme modélisation de séquences](https://laclasse.poneyponey.fr/ia/p09-absorption/34-rl-sequences/) |
| **Exécution** | locale, CPU (PyTorch sur 4 fils, gymnasium, Stable-Baselines3 ; le Transformer a 183 000 paramètres) |
| **Commande** | `make ex NUM=34` (éditeur marimo), `make ex NUM=34 MODE=run` (complet), `make ex NUM=34 MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 2 min 25 s à 2 min 40 s (deux lancements, attente du verrou exclue), dont jeu hors ligne 13 à 14 s, entraînement du Decision Transformer 116 à 126 s, évaluation 8 s, clonage comportemental 5 s ; smoke : 19 s (M1 Pro, CPU, 16/10/2026) |
| **Travail estimé** | 2 h |
| **Seed** | 42 (`seed_everything(42)`) ; mêmes graines d'environnement (50 000 à 50 009) pour toutes les évaluations |
| **Données** | aucune téléchargée : le jeu hors ligne est produit au début du run par des politiques PPO (Stable-Baselines3) sur `CartPole-v1` (gymnasium ≥ 1.4, groupe uv `rl`, installé par défaut) |

## Objectif

Comprendre ce que veut dire « faire du RL comme on modélise du texte » : apprendre, par simple apprentissage supervisé, à prédire l'action à partir du **retour qu'il reste à obtenir**, de l'état et du passé, puis **demander** un retour au moment d'agir. Voir où cela marche (CartPole, où l'agent obtient presque exactement ce qu'on lui demande) et où cela casse (environnement aléatoire, trajectoires à recoller).

Ce qui est optimisé : l'**entropie croisée** entre l'action du jeu de données et l'action prédite, $\mathcal{L}(\theta) = -\frac{1}{N}\sum_t \log \pi_\theta(a_t \mid \hat R_{\le t}, s_{\le t}, a_{<t})$, c'est-à-dire du clonage comportemental conditionné par le retour. Ni critique, ni cible bootstrappée, ni facteur d'actualisation.

## Ce que vous allez faire

1. **En numpy, sans réseau** : la politique conditionnée par le retour « en comptant » (`table_rcsl`, `agir_rcsl`). Pour un état et un retour demandé, on joue l'action la plus fréquente du jeu de données parmi les passages à ce retour restant (le plus proche s'il n'a jamais été vu).
   - **1.a Le pari** : « sûr » rapporte 1 ; « pari » rapporte 3 avec une probabilité de 0,25, sinon 0. Jeu de 2 000 épisodes joués à pile ou face. On demande 3, puis 1, puis on applique l'idée d'ESPER : conditionner sur le retour moyen du paquet (ici, de l'action) au lieu du retour de la trajectoire.
   - **1.b Le recollage** : un petit monde à quatre états (départ, carrefour, impasse, fin) et trois sortes de trajectoires (tableau dans le notebook). Le meilleur chemin (10) n'est contenu dans aucune ; on compare le conditionnement par le retour et l'itération sur les valeurs (chapitre 11) sur les mêmes transitions.
2. **Le jeu hors ligne CartPole** : PPO (Stable-Baselines3, réglages par défaut) entraîné jusqu'à 30 720 pas ; on garde sa politique à 4 096, 8 192, 12 288, 16 384 et 30 720 pas, plus une politique au hasard. Elles jouent 100, 100, 80, 50, 30 et 5 épisodes en tirant leurs actions : 365 épisodes, 28 485 pas.
3. **Le Decision Transformer** (`DecisionTransformer`) : tokens $\hat R_t, s_t, a_t$ entrelacés, plongement du pas de temps, 3 blocs Transformer causaux du mini-GPT (`ia_commun.mini_gpt.Bloc`, 64 dimensions, 1 tête), action lue au token d'état. Fenêtres de $K = 20$ pas, lots de 64, AdamW $10^{-3}$, 2 500 pas.
4. **Évaluation** : on demande 20, 50, 100, 150, 200, 250, 300, 400, 500 et 1 000 au départ, 10 épisodes chacun, en retirant la récompense obtenue à chaque pas ($\hat R_{t+1} = \hat R_t - r_t$).
5. **Témoins** : clonage comportemental (MLP 4 → 64 → 64 → 2) sur tout le jeu, évalué en tirant l'action ou en prenant la plus probable, et « 10 % BC » (clonage des meilleurs épisodes jusqu'à 10 % des pas, le témoin %BC de l'article du Decision Transformer).

## Résultat attendu

Le run complet vérifie 8 seuils et s'arrête avec une erreur si l'un manque. Valeurs mesurées le 16/10/2026, identiques sur deux lancements (CPU, graine 42) :

| Partie | Seuil vérifié | Mesuré |
|---|---|---|
| 1.a Pari | demander 3 fait choisir le pari, retour moyen obtenu **≤ 0,8** | pari, **0,757** (10 000 épisodes simulés) ; demander 1 : sûr, 1,000 |
| 1.a ESPER | avec le retour moyen du paquet, on choisit « sûr » et l'on obtient 1 | étiquettes : sûr 1,00, pari 0,758 → sûr, **1,000** |
| 1.b Recollage | conditionné par le retour (10 demandé) → 1 ; itération sur les valeurs → 10 | **1** (droite puis sortir) contre **10** (gauche puis foncer) ; au départ, seuls les retours 0 et 1 ont été vus |
| DT | pour chaque demande ≤ 400, retour moyen à **± 15 %** de la demande | tableau ci-dessous : écart de 0,3 à 5 %, sauf 20 → 17,3 (−13,5 %) |
| DT | retour moyen croissant avec la demande jusqu'à 500 | oui |
| DT | demande 500 → retour moyen **≥ 450** | **483,9** (min 369) |
| BC, action tirée | retour moyen **≤ 250** (il imite le mélange) | **51,9** (min 13, max 169) |
| 10 % BC | retour moyen **≥ 450** | **500** (6 épisodes gardés, tous à 500, 3 000 pas) |

Retour obtenu par le Decision Transformer (moyenne de 10 épisodes, entre parenthèses min et max) :

| Retour demandé | 20 | 50 | 100 | 150 | 200 | 250 | 300 | 400 | 500 | 1 000 |
|---|---|---|---|---|---|---|---|---|---|---|
| Retour obtenu | 17,3 (16-19) | 52,4 (50-54) | 105,0 (95-114) | 153,7 (146-162) | 206,6 (189-217) | 261,6 (256-267) | 312,3 (284-330) | 401,1 (330-424) | 483,9 (369-500) | 500 (500-500) |

Constat sans seuil : le clonage comportemental sur tout le jeu, en jouant l'action **la plus probable**, obtient **461,4** (huit épisodes à 500, deux à 282 et 332).

Jeu hors ligne (retour moyen par politique) : au hasard 22,7 ; PPO à 4 096 pas 28,9 ; 8 192 pas 58,6 ; 12 288 pas 156,7 ; 16 384 pas 276,6 ; 30 720 pas 500. Ensemble : moyenne 78,0, médiane 34, 7 épisodes à 500 sur 365. Perte du Decision Transformer : 0,83 au départ, 0,375 en moyenne sur les 100 derniers pas.

Figures produites dans `figures/` : `tableau.png` (partie 1), `cartpole.png` (jeu, perte, retour obtenu selon le retour demandé) ; résultats chiffrés dans `figures/resultats.json` (lus par les figures du site, `site/scripts/figures/ch34_*.py`).

## Ce qu'il faut observer

- **On obtient ce qu'on demande.** Sur CartPole, le retour restant dit au modèle combien de pas il doit encore tenir. Demandé 100, il tient 105 pas en moyenne ; demandé 300, 312. Il a appris *plusieurs* comportements dans le même réseau (des politiques faibles et bonnes) et le retour demandé choisit lequel imiter : c'est le « prompting de performance ». Pour échouer au bon moment, il suffit d'imiter les politiques faibles, qui échouent tôt : sur CartPole, viser bas est facile.
- **Pas au-delà du jeu de données.** Demander 1 000 donne 500, le maximum de l'environnement ; demander 500 donne 484, avec un épisode à 369. Le modèle ne fait pas mieux que les meilleures trajectoires qu'il a vues : il les imite. La bande min–max s'élargit au-delà de 300, là où le jeu ne contient que quelques dizaines d'épisodes.
- **Imiter le mélange ne suffit pas.** Le clonage comportemental qui tire ses actions comme les données obtient 52 : il reproduit fidèlement la moyenne d'un jeu dominé par des politiques faibles. Le Decision Transformer, entraîné sur exactement les mêmes données et avec la même perte (entropie croisée sur les actions), fait 484 parce qu'on lui dit quel retour viser.
- **Mais le témoin trivial est fort.** Trier les épisodes par retour et cloner les meilleurs (10 % BC) donne 500, avec un MLP minuscule. Et même le clonage de tout le jeu, en jouant l'action la plus probable, fait 461 : sur CartPole, dans chaque état, la bonne action reste majoritaire dans les données, et prendre l'action la plus probable retire une bonne partie du bruit des politiques faibles. C'est le constat de RvS (Emmons et al., ICLR 2022) : sur beaucoup de tâches, ce qui compte est l'information sur laquelle on conditionne ou filtre, pas le Transformer.
- **La chance ne se commande pas.** Dans le pari, les trajectoires à 3 sont toutes des paris *gagnés* : conditionner sur le retour de la trajectoire confond ce que l'agent a fait et ce que le hasard lui a donné. Demander 3 rapporte 0,76, moins que demander 1. En conditionnant sur le retour moyen d'un paquet de trajectoires qui ne dépend pas du hasard (ESPER), le pari vaut 0,76 et le choix redevient le bon.
- **Pas de recollage.** Au départ, le conditionnement par le retour ne voit que les retours des trajectoires *entières* qui y passent (0 et 1) ; il ne sait pas que le carrefour, atteint par A, mène à 10 dans C. L'itération sur les valeurs propage cette valeur d'état en état (bootstrapping) et recolle le début de A et la fin de C. C'est précisément ce que le Decision Transformer a retiré pour gagner en stabilité.

## Questions de réflexion

1. Pourquoi le retour **restant** $\hat R_t$, et non le retour déjà obtenu, est-il le bon conditionnement ? Que se passerait-il si l'on ne retirait pas la récompense obtenue à chaque pas pendant l'évaluation ?
2. Le Decision Transformer obtient 17 quand on lui demande 20 et 484 quand on lui demande 500 : d'où vient chacun des deux écarts ? Comment évolueraient-ils si l'on retirait du jeu les épisodes de l'experte ?
3. Dans le pari, le jeu de données contient 255 paris gagnés et 754 paris perdus. Qu'obtiendrait-on en demandant 0 ? Pourquoi un environnement déterministe comme CartPole ne tombe-t-il pas dans ce piège ?
4. Construisez un jeu de données CartPole sans aucun épisode complet à 500, mais dont des morceaux, mis bout à bout, tiendraient 500 pas. Quelle méthode du chapitre 17 (CQL, IQL) pourrait faire mieux que le Decision Transformer, et pourquoi ?

## Pour aller plus loin

- Mettre $K = 1$ (le modèle ne voit que $\hat R_t$ et $s_t$) : l'article compare cette ablation sur Atari (tableau 5) ; sur CartPole, la courbe retour demandé / obtenu change-t-elle ?
- Remplacer le Transformer par un MLP qui reçoit $(s_t, \hat R_t)$ : c'est RvS-R. Fait-il aussi bien ?
- Évaluer le Decision Transformer en **tirant** ses actions au lieu de prendre la plus probable.
- Articles : Chen et al., *Decision Transformer: Reinforcement Learning via Sequence Modeling*, 2021, https://arxiv.org/abs/2106.01345 (le témoin %BC est en section 5.1) ; Emmons et al., *RvS: What is Essential for Offline RL via Supervised Learning?*, ICLR 2022, https://arxiv.org/abs/2112.10751 ; Paster, McIlraith et Ba, *You Can't Count on Luck: Why Decision Transformers and RvS Fail in Stochastic Environments*, 2022, https://arxiv.org/abs/2205.15967 (ESPER) ; Brandfonbrener et al., *When does return-conditioned supervised learning work for offline reinforcement learning?*, 2022, https://arxiv.org/abs/2206.01079 (conditions formelles, exemples où le recollage échoue).
