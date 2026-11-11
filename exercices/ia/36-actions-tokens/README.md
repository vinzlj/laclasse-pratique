# Exercice 36 : des actions en tokens

| | |
|---|---|
| **Chapitre lié** | [36. Agents généralistes et robotique](https://laclasse.poneyponey.fr/ia/p09-absorption/36-agents-robotique/) |
| **Exécution** | locale, CPU (numpy, gymnasium pour la vérification du pendule, PyTorch sur 4 fils ; deux Transformers d'environ 120 000 et 138 000 paramètres) |
| **Commande** | `make ex NUM=36` (éditeur marimo), `make ex NUM=36 MODE=run` (complet), `make ex NUM=36 MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 1 min 55 s à 2 min 05 s (deux lancements, attente du verrou exclue), dont quatre entraînements de 25 à 32 s chacun (têtes « token » et « nombre », avec et sans bruit dans les démonstrations), partie 1 et évaluation quelques secondes ; smoke : 6 à 7 s (M1 Pro, CPU, 16/10/2026) |
| **Travail estimé** | 1 h 30 |
| **Seed** | 42 (`seed_everything(42)`) ; départs d'évaluation tirés avec la graine 50 000, identiques pour l'expert et tous les modèles |
| **Données** | aucune téléchargée : les démonstrations sont produites par un expert calculé (pas appris) sur le pendule `Pendulum-v1` de gymnasium (groupe uv `rl`, installé par défaut), réécrit en numpy |

## Objectif

Voir ce que veut dire « une action est un token » (RT-1, RT-2) : découper une action continue en 256 intervalles, faire prédire le numéro de l'intervalle par un petit Transformer entraîné par clonage comportemental, et comparer avec la régression directe de l'action. Mesurer ce que coûte l'arrondi, et ce que la régression perd quand l'expert hésite entre deux actions opposées.

Ce qui est optimisé : l'**entropie croisée** sur le token d'action, $\mathcal{L}(\theta) = -\frac{1}{N}\sum_t \log p_\theta\big(\mathrm{bin}(a_t)\mid o_{t-K+1..t}\big)$ (tête « token ») ; pour comparaison, l'**erreur quadratique** $\frac{1}{N}\sum_t (\hat a_\theta - a_t)^2$ (tête « nombre »). Dans les deux cas, c'est du clonage comportemental (chapitre 17) : imiter l'expert, sans récompense.

## Ce que vous allez faire

1. **Le pendule en numpy** (`pas_pendule`) : la dynamique de `Pendulum-v1` en trois lignes, vérifiée contre gymnasium sur 200 pas au hasard (écart mesuré : 0).
2. **Un expert sans apprentissage** (`expert`) : loin du haut, il pompe de l'énergie en poussant à fond ($\pm 2$) dans le sens du mouvement ; immobile en bas, il choisit un côté **à pile ou face** ; près du haut ($|\theta| < 0{,}5$, $|\dot\theta| < 3$), un régulateur **LQR** $u = -(19{,}69\,\theta + 5{,}26\,\dot\theta)$, dont le gain est calculé une fois pour toutes en itérant l'équation de Riccati sur le modèle linéarisé (`gain_lqr`).
3. **Les bins** (`vers_bins`, `depuis_bins`) : couple $u \in [-2, 2]$ → numéro de bin $\lfloor (u + 2)/4 \cdot n \rfloor$ → centre du bin. On fait jouer l'expert en arrondissant ses couples à 2, 3, 5, 9, 17, 33 puis 256 bins, sur les mêmes 80 départs (50 au hasard comme gymnasium, 30 au repos en bas).
4. **Les démonstrations** (`generer_demos`) : 300 épisodes de 200 pas (60 000 pas), moitié départs au hasard, moitié au repos. Le couple **exécuté** est celui de l'expert plus un bruit gaussien d'écart type 0,5 ; l'**étiquette** reste le couple de l'expert sans bruit (idée de DART, voir plus bas).
5. **Le modèle** (`PolitiqueTransformer`) : les $K = 4$ dernières observations $(\cos\theta, \sin\theta, \dot\theta/8)$ passent chacune par un petit MLP d'encodage, plus un plongement de position, puis deux blocs Transformer causaux du mini-GPT (`ia_commun.mini_gpt.Bloc`, 64 dimensions, 2 têtes) ; début d'épisode complété à gauche par un vecteur de remplissage appris. Deux têtes, entraînées séparément sur les mêmes lots : **token** (256 logits) et **nombre** (un réel). AdamW $10^{-3}$, 3 000 pas, lots de 256 fenêtres.
6. **Évaluation** : les 80 départs, joués avec la tête token décodée de trois façons (bin le plus probable = **mode**, bin **tiré** au hasard, **moyenne** $\sum_b p_b u_b$) et avec la tête nombre. **Témoin** : les deux mêmes têtes entraînées sur des démonstrations **sans bruit** (`temoin_sans_bruit`).
7. **La sonde** : pendule en bas ($\theta = \pi$), vitesse de $-0{,}3$ à $+0{,}3$ rad/s (61 valeurs), premier pas. L'expert y joue $-2$ ou $+2$, jamais entre les deux.

## Résultat attendu

Le run complet vérifie 9 seuils et s'arrête avec une erreur si l'un manque. Valeurs mesurées le 16/10/2026 (CPU, graine 42), identiques d’un lancement à l’autre (trois lancements) :

| Vérification | Seuil | Mesuré |
|---|---|---|
| Expert continu redressé (dans 0,2 rad de la verticale sur les 20 derniers pas) | 100 % des départs | **100 %** (retour moyen −153,7 au hasard, −372,0 au repos) |
| Expert arrondi à 256 bins | retour à 1 % du continu | **−153,7** (au repos −371,9) |
| Expert arrondi à 2 bins | retour moins bon d'au moins 10 % | **−244,1** (au repos −634,3) |
| Tête token, en bas immobile | ≥ 80 % de la masse sur $u \approx \pm 2$, ≥ 20 % sur chacun | **0,32** sur −2, **0,67** sur +2, 0,00 ailleurs |
| Sonde, tête token décodée par le mode | 0 % de couples « entre les deux » ($\lvert u\rvert < 1{,}5$) | **0 %** |
| Sonde, tête nombre | ≥ 10 % de couples « entre les deux » | **70 %** (et 49 % pour la tête token décodée par la moyenne) |
| Tête token (mode), en boucle fermée | redressé ≥ 90 % au hasard et au repos | **100 %** et **100 %** (retours −153,6 et −372,9) |
| Tête nombre, en boucle fermée | redressé ≥ 90 % au hasard et au repos | **100 %** et **100 %** (retours −154,6 et −380,1) |
| Témoin sans bruit (les deux têtes) | redressé ≤ 50 % des départs au hasard | **0 %** et **0 %** (retours −613,3 et −666,3) |

Arrondir le couple de l'expert (retour moyen ; 0 serait parfait) :

| Bins | 2 | 3 | 5 | 9 | 17 | 33 | 256 | continu |
|---|---|---|---|---|---|---|---|---|
| Largeur d'un bin | 2 | 1,33 | 0,8 | 0,44 | 0,24 | 0,12 | 0,016 | — |
| Départs au hasard | −244,1 | −196,2 | −174,8 | −162,7 | −158,1 | −155,8 | −153,7 | −153,7 |
| Départs au repos | −634,3 | −502,4 | −374,6 | −369,9 | −371,9 | −371,7 | −371,9 | −372,0 |

Les quatre politiques apprises (retour moyen, part des départs redressés ; au repos, temps avant de passer au-dessus de l'horizontale) :

| Politique | Départs au hasard | Départs au repos | Au-dessus de l'horizontale après |
|---|---|---|---|
| Expert | −153,7, 100 % | −372,0, 100 % | 1,60 s |
| Token, mode | −153,6, 100 % | −372,9, 100 % | 1,60 s |
| Token, tirage | −163,6, 98 % | −373,1, 100 % | 1,61 s |
| Token, moyenne | −160,6, 100 % | −379,0, 100 % | 1,65 s |
| Nombre (régression) | −154,6, 100 % | −380,1, 100 % | 1,65 s |
| Token, mode, **sans bruit** | −613,3, 0 % | −707,5, 7 % | 1,60 s |
| Nombre, **sans bruit** | −666,3, 0 % | −777,7, 0 % | 1,65 s |

Perte finale (moyenne des 100 derniers pas) : entropie croisée 2,40 (sur 256 classes ; le hasard donne $\ln 256 \approx 5{,}5$), erreur quadratique 0,14 ; sans bruit : 0,66 et 0,12. Les pertes sans bruit sont **plus basses** (données plus faciles à imiter) et les politiques bien pires. 21 % des pas de démonstration sont à couple maximal (bins 0 et 255).

Figures produites dans `figures/` : `bins.png` (partie 1), `tokens-contre-nombre.png` (pertes, distribution au repos, sonde, départs au repos) ; résultats chiffrés dans `figures/resultats.json` (lus par les figures du site, `site/scripts/figures/ch36_*.py`).

## Ce qu'il faut observer

- **256 bins ne coûtent rien.** Arrondi à 256 intervalles, l'erreur sur le couple est au plus de 0,008 (la moitié d'une largeur de 4/256) : le retour de l'expert ne bouge pas (−153,7 dans les deux cas). Avec 2 bins (couples ±1 seulement), il perd 59 % au hasard et ne remonte du bas qu'à grand-peine (−634 contre −372). Entre les deux, la courbe sature vite : sur ce problème, une trentaine de bins suffit déjà presque. RT-1 et RT-2 prennent 256 par dimension pour la même raison : c'est assez fin pour un bras, et cela tient dans un vocabulaire.
- **Le token garde les deux options, la régression les moyenne.** En bas, immobile, l'expert pousse à gauche ou à droite à pile ou face. La tête token le dit : 32 % sur −2, 67 % sur +2, rien entre les deux. La tête nombre doit sortir un seul réel : elle donne −0,97, un couple que l'expert n'a jamais joué. Sur la sonde, la régression trace une rampe douce là où l'expert fait une marche d'escalier : 70 % de ses couples sont « entre les deux ».
- **Ce n'est pas l'information qui manque, c'est le décodage.** La même tête token, décodée par la moyenne $\sum_b p_b u_b$, retombe sur une rampe (49 % entre les deux) : la moyenne d'une distribution à deux bosses est entre les bosses. Décodée par le mode ou par tirage, elle choisit une bosse. C'est la raison d'être des têtes d'action qui modélisent une **distribution** : tokens (RT-1, RT-2, OpenVLA), diffusion (Diffusion Policy), flow matching (π0).
- **Ici, la moyenne ne coûte presque rien en boucle fermée.** Les quatre politiques redressent le pendule (tous les départs au repos, 98 à 100 % des départs au hasard) ; au repos, la régression met 1,65 s au lieu de 1,60 s pour passer au-dessus de l'horizontale (retour −380 contre −373). Le pendule pardonne : un couple de −0,97 le fait bouger, et dès qu'il bouge le sens du mouvement tranche. Le prix monte quand les deux options sont éloignées et que la moyenne est dangereuse : contourner un obstacle par la gauche ou par la droite, la moyenne fonce dedans.
- **Sans bruit dans les démonstrations, tout échoue.** Entraînées sur des démonstrations sans bruit, les deux mêmes têtes atteignent une perte plus basse, mais ne redressent le pendule dans aucun des 50 départs au hasard (et 2 des 30 départs au repos pour la tête token) : la moindre erreur les emmène dans des états que l'expert n'a jamais visités, où elles ne savent plus quoi faire. C'est l'accumulation d'erreurs du clonage comportemental (chapitre 17). Exécuter un couple bruité tout en gardant l'étiquette de l'expert (DART, Laskey et al., 2017) montre au modèle comment revenir sur la trajectoire.

## Questions de réflexion

1. Avec 2 bins, les seuls couples possibles sont −1 et +1 ; avec 5 bins, −1,6, −0,8, 0, 0,8 et 1,6 : jamais ±2. Pourquoi les départs au repos souffrent-ils tant avec 2 ou 3 bins, puis plus du tout à partir de 5, alors que les départs au hasard continuent de s'améliorer jusqu'à 256 ? (Indice : quelle partie de l'expert, pompage ou LQR, demande de la finesse ?)
2. Un bras de RT-1 a 7 dimensions d'action pour le bras, 3 pour la base et une pour le mode, chacune en 256 bins. Combien de « mots » faut-il produire par pas de commande ? Si l'on discrétisait plutôt l'action **entière** (toutes les dimensions ensemble) en 256 niveaux par dimension, combien de classes faudrait-il ?
3. La tête token décodée par tirage redresse le pendule dans 98 % des départs au hasard, contre 100 % par le mode. D'où vient la différence ? Dans quel cas le tirage serait-il préférable au mode ?
4. Le modèle voit 4 pas d'observations. Pour ce pendule, dont l'observation contient déjà la vitesse, l'historique sert-il ? Pourquoi RT-1, qui voit des images, en a-t-il besoin ?

## Pour aller plus loin

- Mettre `action_precedente=True` dans la configuration : chaque pas reçoit aussi l'action précédente sous forme de token. L'imitateur peut alors « tricher » en recopiant l'action d'avant, très corrélée à la suivante (le problème du *copycat*, Wen et al., NeurIPS 2020, en est la forme subtile : leur imitateur ne reçoit qu'un historique d'observations et en déduit l'action précédente ; ici, c'est le cas extrême) ; comparez les politiques obtenues.
- Remplacer la tête « nombre » par un mélange de deux gaussiennes : retrouve-t-on la marche d'escalier sur la sonde ?
- Discrétiser aussi les **observations** en tokens (un plongement par bin et par dimension), comme Gato le fait pour toutes les modalités.
- Articles : Brohan et al., *RT-1: Robotics Transformer for Real-World Control at Scale*, 2022, https://arxiv.org/abs/2212.06817 (« each action dimension in RT-1 is discretized into 256 bins ») ; Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, 2023, https://arxiv.org/abs/2307.15818 (256 bins, l'action écrite comme une suite d'entiers, par exemple « 1 128 91 241 5 101 127 ») ; Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023, https://arxiv.org/abs/2303.04137 (distributions d'actions multimodales) ; Laskey et al., *DART: Noise Injection for Robust Imitation Learning*, 2017, https://arxiv.org/abs/1703.09327 ; Wen et al., *Fighting Copycat Agents in Behavioral Cloning from Observation Histories*, NeurIPS 2020, https://arxiv.org/abs/2010.14876.
