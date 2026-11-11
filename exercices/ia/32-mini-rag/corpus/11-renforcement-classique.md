# 11. Apprentissage par renforcement classique

Repère historique (1952-2006) : Apprendre à agir par essais et erreurs quand la récompense est retardée.

En 1952, Herbert Robbins formule, après Thompson (1933), le problème du joueur face à deux machines à sous : laquelle essayer, sachant qu’on apprend en jouant ? En 1957, Richard Bellman, à la RAND, publie *Dynamic Programming* : si l’on connaît parfaitement les règles d’un système, on sait calculer la meilleure suite de décisions. La même décennie, Arthur Samuel fait apprendre les dames à un ordinateur IBM (le 701, puis le 704) en le faisant jouer contre lui-même. Il faudra pourtant attendre la fin des années 1980 pour réunir les deux fils : Sutton (1988) et Watkins (1989) montrent comment apprendre la solution de Bellman sans connaître les règles, à partir de la seule expérience. Suivent REINFORCE (1992), TD-Gammon (1992), puis en 2006 la recherche arborescente Monte-Carlo, qui ouvrira le go aux machines.

## L’idée en une image

Le chapitre 03 a posé le décor de l’apprentissage par renforcement : pas de bonne réponse fournie, seulement une note chiffrée, souvent tardive, sur ce que l’on a fait. Ce chapitre ouvre la boîte : comment un programme transforme-t-il ces notes en une façon d’agir ?

Pensez à un livreur qui découvre une ville. Il ne reçoit pas d’itinéraire, seulement un pourboire à l’arrivée, plus ou moins gros selon le temps mis. Au début, il essaie des rues au hasard. Peu à peu, il retient une idée simple : « depuis ce carrefour, en général, j’arrive vite ». Cette estimation attachée à chaque lieu, c’est une valeur. Une fois les valeurs connues, agir devient facile : prendre la rue qui mène au carrefour le mieux noté. Tout le chapitre tourne autour de trois questions :

1. Explorer ou exploiter ? Reprendre la rue qui a marché, ou en tester une nouvelle qui est peut-être meilleure ?
2. À qui attribuer le mérite ? Le pourboire arrive à la fin : laquelle des vingt rues prises était la bonne idée ? Minsky a nommé ce casse-tête en 1961 le problème d’attribution du mérite.
3. Faut-il connaître la carte ? Avec le plan de la ville et les temps de trajet, on peut tout calculer sans sortir (programmation dynamique). Sans le plan, il faut apprendre en roulant (Q-learning), ou dessiner son propre plan en route et s’en servir (Dyna).

L’agent observe un état, choisit une action selon sa politique, et l’environnement lui renvoie un nouvel état et une récompense. La boucle recommence, jusqu’à la fin de l’épisode (une partie, une livraison) ou indéfiniment.

| | Avant : recherche et règles écrites à la main (ch. 04) | Après : renforcement |
|---|---|---|
| Ce qu’on donne à la machine | les règles exactes et une fonction d’évaluation écrite par un expert | des récompenses, et la possibilité d’essayer |
| D’où vient la « connaissance » | du programmeur | de l’expérience de l’agent |
| Hasard du monde | absent (graphe déterministe) | central : la glace glisse, le dé roule |
| Lac gelé 4 × 4 de l’exercice | pas de chemin « sûr » : A* ne sait pas quoi viser | Q-learning, sans lire les règles, réussit 73,7 % des traversées ; l’optimum calculé est 74,2 % |
| Morpion contre un joueur parfait | minimax exact (549 946 nœuds, 7 275 avec alpha-bêta), possible parce que l’arbre est minuscule et la fin de partie facile à noter | UCT ne connaît que les règles et simule des parties : 96 nuls sur 100 avec 1 000 simulations par coup |
| Ce qui coince encore | il faut tout écrire | il faut beaucoup d’essais, et les tables ne passent pas à l’échelle |

## Le mécanisme

Les méthodes sont présentées dans l’ordre où elles lèvent les difficultés : d’abord un seul état (bandits), puis un monde dont on connaît les règles (programmation dynamique), puis sans les règles (Monte-Carlo, TD, Q-learning), puis en apprenant directement la politique, enfin en planifiant (Dyna, MCTS) et en généralisant (approximation). La carte d’ensemble :

### 1. Bandits : explorer ou exploiter

Le cas le plus simple n’a qu’un seul état. Devant vous, 10 machines à sous : c’est le problème du bandit manchot à plusieurs bras. Chaque levier $a$ paie une récompense aléatoire de moyenne $q_*(a)$, que vous ne connaissez pas. Vous avez 1 000 tirages. L’estimation naturelle est la moyenne de ce qu’a rapporté chaque levier :

La même moyenne se calcule de façon incrémentale, $Q \leftarrow Q + \frac{1}{N}(r - Q)$, où $r$ est la récompense que l’on vient de recevoir et $N$ le nombre de tirages de ce levier, compte tenu de celui-ci. Cette forme cette forme reviendra partout : nouvelle estimation = ancienne + pas × (cible − ancienne). On n’a pas besoin de stocker l’historique.

Le joueur glouton tire toujours le levier de meilleure moyenne. Son défaut : si le premier levier essayé paie bien par chance, il ne regarde plus jamais les autres. C’est le dilemme exploration-exploitation. Deux remèdes :

- ε-glouton : jouer le meilleur levier, sauf avec une probabilité $\varepsilon$ où l’on en tire un au hasard. Simple, mais l’exploration est aveugle : elle gaspille des essais sur des leviers déjà connus comme mauvais.
- borne supérieure de confiance : ajouter à chaque estimation un bonus d’incertitude, et jouer le maximum. On explore là où l’on sait peu.

Exemple chiffré. Au tirage $t = 20$, le levier A a une moyenne de 1,0 sur 10 essais, le levier B de 0,8 sur 2 essais. Le glouton joue A. UCB calcule $1{,}0 + 2\sqrt{\ln 20 / 10} \approx 1{,}0 + 1{,}09 = 2{,}09$ pour A et $0{,}8 + 2\sqrt{\ln 20 / 2} \approx 0{,}8 + 2{,}45 = 3{,}25$ pour B : il rejoue B, parce que deux essais ne prouvent rien. C’est l’« optimisme face à l’incertitude ».

Auer, Cesa-Bianchi et Fischer ont montré en 2002 que la version UCB1 (avec $\sqrt{2 \ln t / N}$) a un regret qui ne croît que comme le logarithme du nombre de tirages, et ce à tout instant (pas seulement à la limite) : la part des mauvais choix tend vers zéro.

Dans l’exercice (2 000 problèmes de 10 leviers, 1 000 tirages chacun) :

| Stratégie | Récompense moyenne (100 derniers tirages) | Meilleur levier choisi |
|---|---|---|
| glouton | 1,036 | 36,0 % |
| ε = 0,01 | 1,325 | 60,2 % |
| ε = 0,1 | 1,379 | 81,0 % |
| UCB (c = 2) | 1,504 | 86,1 % |

Explorer coûte au début (on joue des leviers moyens) et rapporte ensuite. Ce dilemme ne disparaît jamais en renforcement ; il se cache dans chacune des méthodes suivantes.

### 2. Le cadre : processus de décision markovien

Dès qu’une action change la situation, il faut un cadre plus riche : le processus de décision markovien, formalisé par Bellman en 1957. Il comprend :

- des états $s$ (la case du lac gelé, la position du jeu) ;
- des actions $a$ possibles dans chaque état ;
- des probabilités de transition $p(s' \mid s, a)$ : où l’on arrive en jouant $a$ dans $s$ ;
- des récompenses $r$ ;
- un facteur d’actualisation $\gamma$ entre 0 et 1.

L’hypothèse « markovienne » : la suite ne dépend que de l’état présent et de l’action, pas du chemin suivi pour arriver là. Une position d’échecs la vérifie ; une seule image d’un jeu vidéo, souvent non (on n’y voit pas la vitesse de la balle).

Exemple : FrozenLake. Un lac gelé de 4 × 4 cases (environnement de la bibliothèque gymnasium). Départ en haut à gauche, but en bas à droite (récompense 1), quatre trous (fin de l’épisode, récompense 0). La glace glisse : l’action voulue n’est exécutée qu’une fois sur trois ; sinon l’agent part sur l’un des deux côtés perpendiculaires.

```text
S F F F     S : départ        F : glace
F H F H     H : trou (fin, récompense 0)
F F F H     G : but (fin, récompense 1)
H F F G     action « bas » : bas 1/3, gauche 1/3, droite 1/3
```

L’agent cherche à maximiser le retour, somme actualisée des récompenses futures (déjà croisé au chapitre 03 ; on dit aussi « gain ») :

La forme récursive de droite est la clé de tout ce qui suit : le retour d’aujourd’hui, c’est la récompense immédiate plus le retour de demain, un peu déprécié. Avec $\gamma = 0{,}9$, une récompense reçue 10 pas plus tard ne vaut que $0{,}9^{10} \approx 0{,}35$ maintenant.

On définit deux fonctions, qui sont ce que l’agent va estimer :

La fonction de valeur dit « où il fait bon être » ; la fonction de valeur d’action dit « quoi faire ». Si l’on connaît la meilleure, $Q^*$, la politique optimale s’en déduit sans calcul : dans chaque état, jouer $\argmax_a Q^*(s, a)$.

En remplaçant $G_t$ par sa forme récursive, on obtient l’équation de Bellman d’optimalité :

C’est une équation sur toutes les valeurs à la fois : chaque $V^*(s)$ dépend de celles de ses voisins. Elle a une solution unique dès que $\gamma \lt 1$.

### 3. Programmation dynamique : quand on connaît les règles

Si l’on connaît $p(s' \mid s, a)$ et les récompenses (le modèle), on peut résoudre l’équation de Bellman sans jamais jouer. C’est la programmation dynamique.

Itération sur les valeurs. Partir de $V = 0$ partout, puis appliquer l’équation comme une affectation, pour tous les états, jusqu’à ce que plus rien ne bouge.

À la main, sur un couloir déterministe A → B → but, récompense 1 en atteignant le but, $\gamma = 0{,}9$ :

| Balayage | $V(\text{A})$ | $V(\text{B})$ |
|---|---|---|
| 0 | 0 | 0 |
| 1 | $0 + 0{,}9 \times 0 = 0$ | $1 + 0{,}9 \times 0 = 1$ |
| 2 | $0 + 0{,}9 \times 1 = 0{,}9$ | 1 |
| 3 | 0,9 | 1 (stable) |

La valeur remonte du but vers le départ, d’un pas par balayage. Sur le lac gelé glissant, avec $\gamma = 0{,}99$ et une tolérance de $10^{-10}$, il faut 571 balayages.

```python
import numpy as np

def iteration_valeurs(T, R, gamma, tol=1e-10):
    """T[s, a, s'] : probabilités de transition ; R[s, a] : récompense moyenne immédiate."""
    V = np.zeros(T.shape[0])
    while True:
        Q = R + gamma * T @ V          # Q[s, a] = r + γ Σ_s' p(s'|s,a) V(s')
        V_nouveau = Q.max(axis=1)      # équation de Bellman d'optimalité
        if np.abs(V_nouveau - V).max() < tol:
            return V_nouveau, Q.argmax(axis=1)   # valeurs et politique gloutonne
        V = V_nouveau
```

Itération sur les politiques (Howard, 1960, d’après Sutton et Barto). Alterner deux étapes : évaluer exactement la politique courante (résoudre le système linéaire $V = R_\pi + \gamma T_\pi V$, où $R_\pi$ est la récompense moyenne et $T_\pi$ la matrice des probabilités de transition quand on suit $\pi$) puis l’améliorer en choisissant dans chaque état l’action la meilleure selon cette évaluation. Chaque tour coûte plus cher, mais il en faut très peu : 7 améliorations contre 571 balayages dans l’exercice, pour les mêmes valeurs (écart $2{,}8 \times 10^{-9}$).

Ce que donne la carte. $V^*(\text{départ}) = 0{,}542$ : même en jouant parfaitement, on n’arrive au but qu’un peu plus d’une fois sur deux en moyenne actualisée ; en simulation, la politique optimale réussit 74,2 % des traversées, contre 2,2 % pour des politiques tirées au hasard. Sur la grille 8 × 8, $V^*(\text{départ}) = 0{,}415$ et 62,5 % de réussite.

Un détail contre-intuitif : depuis la case de départ, l’action optimale est gauche, contre le mur, alors que le but est en bas à droite. Avec « gauche », on glisse soit à gauche (le mur : on reste sur place), soit en haut (le mur encore), soit en bas : on ne part jamais vers la droite, vers la case qui surplombe le premier trou. Pousser contre le mur, c’est choisir ses glissades. Aucun humain n’écrirait cette règle ; elle sort du calcul.

Bellman parlait de malédiction de la dimension : le nombre d’états explose avec le nombre de variables, et chaque balayage les parcourt tous. Surtout, la programmation dynamique exige le modèle, que l’on n’a presque jamais : personne ne donne au robot les probabilités de glissade de son sol.

### 4. Monte-Carlo : apprendre en attendant la fin

Sans modèle, il faut jouer. L’idée la plus directe : jouer des épisodes complets, puis rapprocher la valeur de chaque état visité du retour réellement obtenu. C’est la méthode de Monte-Carlo (le nom vient du casino : on estime une moyenne par tirages au hasard).

Avantage : la cible $G_t$ est vraie, sans biais. Défauts : il faut attendre la fin de l’épisode (impossible si la tâche ne finit jamais), et $G_t$ est très bruité, car il additionne le hasard de tous les pas suivants.

### 5. Différences temporelles : corriger à chaque pas

L’idée de Sutton (1988) : ne pas attendre la fin, mais corriger une estimation par l’estimation suivante. Après un seul pas, on connaît $r_{t+1}$ et l’état $s_{t+1}$ ; la forme récursive du retour suggère de viser $r_{t+1} + \gamma V(s_{t+1})$ :

C’est la méthode des différences temporelles, TD(0). On met à jour une estimation à partir d’une autre estimation : c’est le bootstrapping. Exemple : $V(s) = 0{,}5$, on reçoit $r = 0$ et on arrive dans $s'$ avec $V(s') = 0{,}7$ ($\gamma = 1$, $\alpha = 0{,}1$). L’erreur TD vaut $0 + 0{,}7 - 0{,}5 = 0{,}2$ et $V(s)$ passe à $0{,}52$, tout de suite, sans attendre la fin. Monte-Carlo aurait attendu le retour final, par exemple $G = 1$, pour faire passer $V(s)$ à $0{,}5 + 0{,}1 \times (1 - 0{,}5) = 0{,}55$.

Le prix : la cible TD est biaisée (elle contient une estimation peut-être fausse), mais beaucoup moins bruitée (un seul pas de hasard). Sur la marche aléatoire à 5 états de l’exercice, après 100 épisodes, le meilleur TD(0) a une erreur RMS de 0,035 ($\alpha = 0{,}05$), le meilleur Monte-Carlo de 0,082 ($\alpha = 0{,}02$).

TD(λ), en une ligne. Entre les deux extrêmes, TD(λ) mélange les cibles à 1, 2, 3… pas avec des poids qui décroissent en $\lambda^n$ : $\lambda = 0$ donne TD(0), $\lambda = 1$ redonne Monte-Carlo. C’est la version que Sutton analyse en 1988 et que TD-Gammon utilisera.

### SARSA et Q-learning : apprendre à agir

Évaluer une politique ne suffit pas : on veut la meilleure. On applique donc TD aux valeurs d’action $Q(s, a)$, et l’on joue de façon ε-gloutonne par rapport à $Q$. Deux cibles possibles :

| Méthode | Cible | Ce qu’elle évalue |
|---|---|---|
| SARSA (Rummery et Niranjan, 1994 ; nom donné par Sutton en 1996) | $r + \gamma\, Q(s', a')$, avec $a'$ l’action réellement jouée ensuite | la politique qui explore : apprentissage apprentissage sur politique |
| Q-learning (Watkins, 1989) | $r + \gamma \max_{a'} Q(s', a')$ | la politique gloutonne, quelle que soit l’action jouée : apprentissage apprentissage hors politique |

(Formule : Q-learning.)

Sur politique ou hors politique : la différence se voit sur la falaise (grille 4 × 12, chaque pas coûte −1, tomber coûte −100 et renvoie au départ). En explorant avec $\varepsilon = 0{,}1$ :

- Q-learning apprend le chemin le plus court, le long du vide (13 pas). C’est optimal… pour un agent qui ne se tromperait jamais. Mais l’agent explore, et un pas au hasard sur dix le jette dans la falaise : −49,9 par épisode en moyenne ;
- SARSA évalue la politique qu’il joue vraiment, maladresses comprises. Il apprend un détour par le haut (17 pas) et gagne −26,1 par épisode.

Exemple de cible : l’agent est au bord, $Q(s', \text{longer}) = -5$, $Q(s', \text{vers le vide}) = -10$, et l’exploration tire « vers le vide ». Avec $r = -1$ et $\gamma = 1$, SARSA vise $-1 + (-10) = -11$ : il apprend que ce bord est dangereux pour lui. Q-learning vise $-1 + \max(-5, -10) = -6$ : il ignore le risque qu’il fait courir à son propre apprentissage.

Pour un robot qui apprend en marchant, SARSA est plus prudent ; pour apprendre la politique optimale à partir de données quelconques (un autre agent, un humain, un ancien enregistrement), il faut le Q-learning. Watkins et Dayan ont prouvé en 1992 que le Q-learning tabulaire converge vers $Q^*$ avec probabilité 1, à condition que chaque couple état-action soit essayé indéfiniment et que le pas $\alpha$ décroisse convenablement.

Le Q-learning en entier, dans sa version tabulaire (simplifiée de l’exercice) :

```python
Q = np.zeros((n_etats, n_actions))
N = np.zeros((n_etats, n_actions))                 # visites, pour faire décroître α
for episode in range(10_000):
    eps = max(0.05, 1.0 - episode / 5_000)          # on explore beaucoup au début
    s, _ = env.reset()
    fini = False
    while not fini:
        a = epsilon_greedy(Q[s], eps, rng)          # agir
        s2, r, termine, tronque, _ = env.step(a)    # observer
        N[s, a] += 1
        alpha = 1.0 / N[s, a]  0.6
        cible = r + (0.0 if termine else gamma * Q[s2].max())
        Q[s, a] += alpha * (cible - Q[s, a])        # corriger d'une fraction de l'erreur TD
        s, fini = s2, termine or tronque
politique = Q.argmax(axis=1)
```

Sur le lac gelé, sans jamais lire les probabilités de glissade, ce code retrouve la solution de la programmation dynamique : la politique gloutonne réussit 73,7 % (optimum 74,2 %), $\max_a Q(\text{départ}, a) = 0{,}537$ pour $V^* = 0{,}542$, et choisit la même action que l’optimum dans 91 % des états. Pendant l’apprentissage, la réussite en jouant plafonne pourtant vers 53 % : c’est le prix de l’exploration (au moins 5 % de coups au hasard) sur une glace qui punit chaque faux pas.

### Ce qu’optimisent les méthodes de valeur : l’erreur TD

Les méthodes de valeur (TD, SARSA, Q-learning, et Dyna plus bas) réduisent l’erreur TD. On peut l’écrire comme une erreur quadratique entre la prédiction $Q(s, a)$ et la cible, à une nuance près, essentielle : on traite la cible comme une constante. La mise à jour $Q \leftarrow Q + \alpha\,\delta$ est le pas de gradient de $\tfrac12 \delta^2$ par rapport à la prédiction seulement. Ce n’est donc pas tout à fait une fonction de perte au sens du chapitre 01 : la « vérité » bouge à chaque mise à jour, et c’est ce qui pourra mal tourner à la section 9.

### 6. Gradient de politique : apprendre la politique elle-même

Autre voie : oublier les valeurs et paramétrer directement la politique. Dans l’exercice, une table de préférences $\theta(s, a)$ et une softmax :

$$
\pi_\theta(a \mid s) = \frac{e^{\theta(s, a)}}{\sum_b e^{\theta(s, b)}}
$$

On veut maximiser $J(\theta) = \E_{\pi_\theta}[G_0]$, le retour moyen. Difficulté : $J$ dépend de $\theta$ à travers les actions tirées et les glissades du monde, que l’on ne sait pas dériver. Le résultat de Williams (1992), qui donne l’algorithme REINFORCE, est que l’on peut quand même estimer le gradient à partir des seuls épisodes joués :

En mots : rendre plus probables les actions suivies d’un retour meilleur que prévu, moins probables les autres. Les dérivées ne traversent jamais l’environnement, seulement la politique. C’est un gradient de politique.

Exemple chiffré. Quatre actions, préférences nulles, donc $\pi = (0{,}25\ ;\ 0{,}25\ ;\ 0{,}25\ ;\ 0{,}25)$. L’agent joue l’action 3, obtient $G = 1$ alors que la ligne de base vaut $b = 0{,}4$ : l’avantage $G - b$ vaut $0{,}6$. Avec un pas $\eta = 0{,}1$, $\Delta\theta = 0{,}1 \times 0{,}6 \times (-0{,}25\ ;\ -0{,}25\ ;\ 0{,}75\ ;\ -0{,}25) = (-0{,}015\ ;\ -0{,}015\ ;\ 0{,}045\ ;\ -0{,}015)$. La probabilité de l’action 3 passe de 0,25 à 0,261. Si $G$ avait été 0, l’avantage serait $-0{,}4$ et l’action 3 serait devenue moins probable.

Pourquoi la ligne de base ? Sans elle, sur le lac gelé où $G \geq 0$ toujours, toutes les actions jouées sont renforcées, seulement plus ou moins : le signal utile est noyé dans le bruit. Soustraire $b(s)$ ne change pas la moyenne du gradient (car $\E[\nabla \log \pi] = 0$) mais réduit fortement sa variance. Dans l’exercice, avec ligne de base, la réussite en moyenne glissante atteint environ 55 % vers l’épisode 4 000, contre 30 % sans. Au bout de 15 000 épisodes, les deux politiques gloutonnes réussissent 73,7 %, l’optimum de la glace ; en jouant (1 000 derniers épisodes), 71,4 % avec ligne de base, 70,6 % sans.

En simplifiant comme la plupart des implémentations, l’exercice omet le facteur $\gamma^t$ que contient la formule exacte du gradient pour un problème actualisé.

```python
G = 0.0                                        # retour après le dernier pas
for s, a, r in reversed(trajectoire):         # après l'épisode, du dernier pas au premier
    G = r + gamma * G                          # retour depuis ce pas
    grad_log = -politique(s)
    grad_log[a] += 1.0                         # ∇ log π(a|s) = 1_a − π(·|s)
    theta[s] += eta * (G - b[s]) * grad_log    # montée de gradient
    b[s] += eta_b * (G - b[s])                 # la ligne de base suit les retours
```

### Ce qu’optimise REINFORCE

REINFORCE monte le gradient du retour espéré. Écrit comme une perte à minimiser, c’est une entropie croisée sur l’action jouée (rappel du chapitre 01), pondérée par l’avantage $G_t - b(s_t)$ : du supervisé, où l’« étiquette » est l’action qu’on a soi-même tirée, et où chaque exemple compte selon ce qu’il a rapporté.

Le gradient de politique a deux atouts que les tables de $Q$ n’ont pas : il s’applique à des actions continues (un angle de volant), et il peut apprendre une politique aléatoire quand c’est la meilleure (pierre-feuille-ciseaux). Son défaut : la variance, qui exige beaucoup d’épisodes. Le remède moderne (un critique appris comme ligne de base, puis PPO qui borne chaque pas) est l’objet du chapitre 17.

### 7. Dyna : apprendre un modèle et rêver avec

Q-learning jette chaque expérience après une seule mise à jour. Sutton (1990) propose l’architecture Dyna : garder aussi un modèle de l’environnement appris (« dans tel état, telle action m’a mené là avec telle récompense ») et s’en servir pour planifier, c’est-à-dire rejouer des expériences imaginaires.

À chaque pas réel : une mise à jour de Q-learning, puis $n$ mises à jour supplémentaires sur des couples $(s, a)$ déjà vus, tirés au hasard dans le modèle. Le calcul est bon marché ; ce qui coûte, dans le monde réel, ce sont les pas.

Sur le labyrinthe 6 × 9 de Sutton et Barto (plus court chemin : 14 pas), en 50 épisodes :

| Planification | Pas réels sur 50 épisodes | Durée du 2ᵉ épisode |
|---|---|---|
| $n = 0$ (Q-learning seul) | 5 521 | 1 024 pas |
| $n = 5$ | 2 442 | 175 pas |
| $n = 50$ | 1 689 | 39 pas |

Pourquoi le 2ᵉ épisode ? Au 1ᵉʳ, aucun des trois agents ne sait rien : la récompense n’apparaît qu’à l’arrivée. Q-learning seul n’en tire qu’une mise à jour, sur la dernière case. Dyna, lui, rejoue des transitions du modèle des dizaines de fois : la valeur du but remonte le long du chemin pendant que l’agent « rêve », comme les balayages de la programmation dynamique, mais sur un modèle appris.

Limite : un modèle faux fait planifier faux. Si un mur se déplace, Dyna-Q continue de croire l’ancien chemin ; Sutton et Barto (2018, §8.3) décrivent Dyna-Q+, qui ajoute un bonus aux actions pas essayées depuis longtemps.

### 8. MCTS et UCT : planifier par simulations

Au chapitre 04, le minimax explorait l’arbre de jeu et notait les feuilles avec une fonction d’évaluation écrite par des experts. Au go, personne ne savait écrire une bonne fonction d’évaluation. La recherche arborescente Monte-Carlo (en anglais *Monte Carlo tree search*, MCTS) la remplace par une idée de casino : pour estimer une position, jouer des parties au hasard depuis elle et compter les victoires, puis concentrer ces simulations sur les coups prometteurs. L’expression figure dans le titre de l’article de Rémi Coulom (2006), dont le programme Crazy Stone a gagné le 10ᵉ tournoi de go sur ordinateur KGS en 9 × 9.

Une itération a quatre temps, répétés des centaines de fois avant chaque coup :

Comment choisir, à chaque nœud, quel enfant descendre ? Kocsis et Szepesvári (2006) répondent : c’est un bandit. Chaque coup possible est un levier ; sa récompense est le résultat des simulations. On applique UCB à chaque nœud : c’est l’algorithme UCT.

Exemple. Un nœud a reçu 10 simulations, réparties en trois coups : A (3 gains sur 5), B (2 sur 3), C (1 sur 2). Avec $c = \sqrt 2$ : A $= 0{,}60 + 0{,}96 = 1{,}56$, B $= 0{,}67 + 1{,}24 = 1{,}91$, C $= 0{,}50 + 1{,}52 = 2{,}02$. UCT descend dans C, le moins bien noté, parce qu’il est le moins connu. Au fil des simulations, les bons coups accumulent les visites ; à la fin, on joue le coup le plus visité.

Au morpion, sans aucune connaissance du jeu à part ses règles :

| Simulations par coup | Contre le hasard (V / N / D) | Contre un joueur parfait (V / N / D) |
|---|---|---|
| 10 | 71 / 12 / 17 | 0 / 30 / 70 |
| 50 | 90 / 5 / 5 | 0 / 64 / 36 |
| 200 | 92 / 8 / 0 | 0 / 81 / 19 |
| 1 000 | 93 / 7 / 0 | 0 / 96 / 4 |

Le morpion bien joué est nul : 96 nuls sur 100 contre un adversaire parfait, c’est presque le jeu parfait. Au morpion, l’alpha-bêta du chapitre 04 fait mieux pour moins cher ; l’intérêt d’UCT est qu’il n’a besoin d’aucune fonction d’évaluation, ce qui manquait au go, où l’arbre est hors de portée. MCTS est anytime : on l’arrête quand on veut et il rend le meilleur coup trouvé jusque-là. Ce n’est pas de l’apprentissage au sens des sections précédentes (rien n’est retenu d’un coup à l’autre), mais de la planification avec un modèle exact : les règles du jeu.

Il manquait deux choses pour le go à 19 × 19 : des simulations moins bêtes que le hasard, et une évaluation des positions. AlphaGo (2016) fournira les deux avec des réseaux de neurones appris sur des parties humaines puis par renforcement (chapitre 17).

### 9. Quand la table ne suffit plus : approximation et triade mortelle

Tout ce qui précède range une valeur par état (ou par couple état-action) dans une table : 16 cases pour le lac gelé, 5 478 positions légales au morpion. Le backgammon en compte plus de $10^{20}$ d’après Tesauro ; une image de jeu vidéo, bien davantage. Il faut une approximation de fonction : une fonction paramétrée qui généralise d’un état aux états semblables. Le cas le plus simple est linéaire :

C’est un semi-gradient : le pas de gradient ne « voit » que la prédiction, pas la cible. Avec une table, c’était sans danger (chaque mise à jour ne touche qu’une case). Avec des poids partagés, corriger $\hat v(s)$ déplace aussi $\hat v(s')$, donc la cible elle-même.

Sutton et Barto (2018, §11.3) appellent triade mortelle la réunion de trois ingrédients :

1. approximation de fonction : les états partagent des poids ;
2. bootstrapping : la cible contient l’estimation elle-même (TD, programmation dynamique) ;
3. apprentissage hors politique : les états sont visités selon une autre distribution que celle de la politique évaluée.

Avec deux ingrédients sur trois, l’instabilité peut être évitée ; avec les trois, les poids peuvent diverger. Le contre-exemple de Baird (1995), dans la version légèrement modifiée de Sutton et Barto (2018, §11.2), le montre sur 7 états et 8 poids, avec des récompenses toutes nulles : la vraie valeur, 0 partout, est exactement représentable (tous les poids à 0). Les caractéristiques valent $\hat v(s_i) = 2w_i + w_8$ pour les six premiers états et $\hat v(s_7) = w_7 + 2w_8$. La politique évaluée va toujours vers $s_7$, mais les états sont visités uniformément. La cible de tous les états est $\gamma\,\hat v(s_7)$. Les mises à jour sur $s_1$ à $s_6$ font grandir $w_8$ (partagé), donc $\hat v(s_7)$, donc toutes les cibles ; $s_7$, visité une fois sur sept seulement, n’est jamais assez corrigé pour compenser. En appliquant la mise à jour TD en moyenne (sans bruit d’échantillonnage, $\gamma = 0{,}99$, $\alpha = 0{,}01$) :

| Variante | Norme des poids $\lVert \mathbf w \rVert$ après 2 000 balayages (départ ≈ 10,3) |
|---|---|
| les trois ingrédients | 5 280 |
| sans approximation (une case par état) | 24,5 |
| sans hors-politique (états visités selon la politique évaluée) | 9,06 |
| sans bootstrapping (cible = vrai retour, 0) | 8,63 |

Sans approximation, la norme monte d’abord (les six premiers états se rapprochent de $\gamma\,V(s_7) \approx 9{,}9$), puis décroît très lentement vers 0 : rien ne diverge.

Pourquoi c’est important pour la suite : le Q-learning est hors politique par nature (il apprend la politique gloutonne en jouant une politique qui explore) et fait du bootstrapping. Lui donner un réseau de neurones à la place de la table réunit les trois ingrédients. C’est la racine de l’instabilité des premiers essais de Q-learning profond ; les astuces de DQN, publiées par DeepMind en 2015 (mémoire de rejeu, réseau cible gelé), servent à contenir cette triade. Elles sont expliquées au chapitre 17.

### 10. Deux jeux, deux jalons : Samuel et TD-Gammon

Samuel et les dames (1952-1959). Arthur Samuel, chez IBM, écrit un premier programme de dames pour l’IBM 701 en 1952 ; sa première version qui apprend est achevée en 1955 (dates d’après Sutton et Barto, §16.2). Le programme cherche plusieurs coups à l’avance, comme minimax, et note les positions avec un polynôme linéaire de caractéristiques. Sa méthode « par généralisation » ajuste les coefficients pour que la note d’une position se rapproche de la note obtenue par la recherche un coup plus tard, en jouant contre une autre version de lui-même : c’est déjà, sans le nom, l’idée de TD. Il atteint un bon niveau, sans battre les maîtres.

TD-Gammon (1992-1995). Gerald Tesauro, chez IBM, combine TD(λ) et un perceptron multicouche entraîné par rétropropagation (chapitre 07) : le réseau prédit la probabilité de gagner depuis une position, et apprend en jouant contre lui-même. La première version, sans aucune connaissance du backgammon dans son codage d’entrée, atteint après 200 000 parties un « niveau intermédiaire fort », proche de Neurogammon, le précédent programme de Tesauro (Tesauro, 1995 ; Sutton et Barto, §16.1, donnent 300 000 parties et la classent à égalité avec les meilleurs programmes de l’époque). Les versions suivantes, avec des caractéristiques expertes et 1,5 million de parties, approchent le meilleur niveau humain : la version 2.1 perd d’un seul point contre l’ancien champion du monde Bill Robertie sur 40 parties (Tesauro, 1995).

TD-Gammon a deux chances qui expliquent pourquoi l’exploit ne s’est pas répété tout de suite aux échecs ou au go : les dés forcent l’exploration (l’agent visite naturellement des positions variées), et l’apprentissage se fait sur politique (le réseau évalue les positions qu’il rencontre en jouant). Deux des trois ingrédients de la triade seulement.

## Ce qui coinçait

- Les tables ne passent pas à l’échelle. Une valeur par état, c’est 16 cases pour le lac gelé, plus de $10^{20}$ pour le backgammon, et bien davantage dès que l’état est une image. Il faut une approximation qui généralise. Les réseaux de neurones sont le candidat naturel, mais en 2006 on ne sait pas encore entraîner des réseaux profonds de façon fiable (chapitre 12) ni traiter des images efficacement (chapitre 13).
- L’approximation déstabilise le bootstrapping. La triade mortelle fait diverger Q-learning avec approximation ; il faudra des garde-fous spécifiques (DQN, chapitre 17).
- L’exploration reste rudimentaire. ε-greedy explore au hasard : sur la glace, la réussite en jouant plafonne à 53 % quand la politique apprise en vaut 74 %. Dans un grand espace avec une récompense rare, un agent qui explore au hasard ne la trouve jamais.
- Le gradient de politique est bruité. REINFORCE a besoin de milliers d’épisodes sur 16 états ; la ligne de base aide, mais un pas trop grand peut détruire une bonne politique (PPO, chapitre 17).
- Il faut une récompense. Toutes ces méthodes supposent qu’un nombre bien choisi décrit le but. Quand on ne sait pas l’écrire (« répondre utilement »), il faudra l’apprendre à partir de préférences humaines (chapitre 30).
- Un modèle appris peut tromper. Dyna planifie dans un modèle qui peut être faux ; MCTS suppose un simulateur exact. Apprendre de bons modèles du monde sera le sujet du chapitre 35.

- Bandits, ε-greedy, UCB (1952-2002) : Essayer ce qu’on connaît mal, juste assez pour ne pas rater le meilleur levier.
- Programmation dynamique (Bellman, 1957) : Avec les règles du monde, la valeur remonte du but vers le départ par simple calcul.
- Monte-Carlo : Attendre la fin et moyenner les retours réels : juste mais lent et bruité.
- TD (Sutton, 1988) : Corriger chaque prédiction par la suivante, sans attendre la fin.
- SARSA, Q-learning (1989-1994) : TD sur les actions : évaluer la politique qu’on joue, ou celle qu’on voudrait jouer.
- REINFORCE (Williams, 1992) : Rendre plus probables les actions suivies d’un retour meilleur que prévu.
- Dyna (Sutton, 1990) : Retenir un modèle du monde et rêver avec pour économiser l’expérience réelle.
- MCTS, UCT (2006) : Évaluer une position en jouant des parties au hasard, et traiter chaque nœud comme un bandit.
- Triade mortelle (Baird, 1995) : Approximation + bootstrapping + hors politique : les valeurs peuvent exploser.
- Réseaux profonds entraînables : Avant de remplacer les tables par des réseaux, il faut savoir entraîner des réseaux profonds.
- DQN, PPO, AlphaGo : Le renforcement avec des réseaux profonds, et ses garde-fous.

Et ensuite ? Les tables ne passent pas à l’échelle des pixels : pour jouer à partir d’une image, il faut un réseau de neurones profond qui transforme l’image en valeurs ou en politique. Mais au milieu des années 2000, les réseaux profonds eux-mêmes ne s’entraînent pas encore bien : gradients qui s’éteignent, surapprentissage. Le chapitre 12 ouvre la révolution du deep learning par là : rendre les réseaux profonds entraînables. Le renforcement reviendra, équipé de ces réseaux, au chapitre 17.

## Exercice

En NumPy, sur CPU et en 35 s, vous implémentez les méthodes du chapitre sur de petits problèmes : bandits, programmation dynamique et Q-learning sur le lac gelé, TD contre Monte-Carlo, SARSA contre Q-learning sur la falaise, REINFORCE, Dyna-Q, UCT au morpion, puis vous faites diverger TD sur le contre-exemple de Baird. Le run complet vérifie 14 seuils chiffrés.

Ce qu’il faut observer :

- explorer coûte au début puis rapporte : UCB trouve le meilleur levier 86 % du temps, le glouton 36 % ;
- Q-learning retrouve, sans lire les règles du lac, la politique que la programmation dynamique calcule avec elles (73,7 % contre 74,2 %), y compris l’étrange « gauche » de la case de départ ;
- planifier économise l’expérience réelle : Dyna-Q avec 50 rêves par pas divise par trois le nombre de pas réels ; UCT passe de 70 défaites à 4 contre un joueur parfait en augmentant seulement son budget de simulations ;
- il suffit de retirer un seul ingrédient de la triade pour que les poids restent bornés.

Questions de réflexion : sur la falaise, que deviendraient les chemins de SARSA et de Q-learning si ε décroissait jusqu’à 0 ? UCT utilise la même formule que UCB : qu’est-ce qui joue le rôle du « levier » à chaque nœud ?

## Auto-évaluation

Question : Au tirage 20, le levier A a une moyenne de 1,0 sur 10 essais et le levier B de 0,8 sur 2 essais. Que jouent le glouton et UCB (c = 2), et pourquoi ?

Réponse : Le glouton joue A, meilleure moyenne. UCB ajoute un bonus $2\sqrt{\ln 20 / N}$ : environ 1,09 pour A (10 essais) et 2,45 pour B (2 essais). Scores 2,09 contre 3,25 : UCB joue B, parce que deux essais ne suffisent pas à le juger. Le bonus fond à mesure qu’un levier est essayé, donc UCB finit par se concentrer sur le meilleur.

Question : Sur la falaise, Q-learning apprend le chemin le plus court (13 pas) mais gagne moins par épisode (−49,9) que SARSA (−26,1), qui fait un détour. Comment est-ce possible ?

Réponse : Q-learning évalue la politique gloutonne (cible $r + \gamma \max_{a'} Q$) : pour elle, longer le vide est optimal. Mais l’agent joue ε-glouton et fait un pas au hasard une fois sur dix, ce qui le jette souvent dans la falaise (−100). SARSA évalue la politique réellement jouée, maladresses comprises (cible $r + \gamma Q(s', a')$) : il « sait » qu’il trébuche parfois et s’éloigne du bord. Hors politique contre sur politique. Si ε tendait vers 0, les deux convergeraient vers le chemin le long du vide.

Question : $V(s) = 0{,}5$. L’agent reçoit $r = 0$ et arrive dans $s'$ avec $V(s') = 0{,}7$ ; $\gamma = 1$, $\alpha = 0{,}1$. Que fait TD(0) ? Que ferait Monte-Carlo si l’épisode finissait avec un retour de 1 ?

Réponse : Erreur TD : $\delta = 0 + 0{,}7 - 0{,}5 = 0{,}2$, donc $V(s) \leftarrow 0{,}5 + 0{,}1 \times 0{,}2 = 0{,}52$, immédiatement. Monte-Carlo attend la fin de l’épisode et vise le retour observé : $V(s) \leftarrow 0{,}5 + 0{,}1 \times (1 - 0{,}5) = 0{,}55$. La cible TD est biaisée (elle s’appuie sur $V(s')$, peut-être faux) mais beaucoup moins bruitée ; sur la marche aléatoire de l’exercice, TD atteint une erreur de 0,035 contre 0,082.

Question : Dans REINFORCE, pourquoi soustraire une ligne de base b(s) au retour ? Change-t-elle la direction moyenne du gradient ?

Réponse : Non : comme $\E_{a \sim \pi}[\nabla \log \pi(a \mid s)] = 0$, soustraire une quantité qui ne dépend pas de l’action ne change pas l’espérance du gradient. Elle réduit sa variance : sans elle, sur le lac gelé où les retours sont toujours positifs ou nuls, toute action jouée est renforcée, et seul l’écart entre « un peu » et « beaucoup » porte l’information. Avec $b(s)$, les actions moins bonnes que prévu sont affaiblies. Dans l’exercice : environ 55 % de réussite vers l’épisode 4 000 avec ligne de base, 30 % sans ; même niveau final.

Question : Quels sont les trois ingrédients de la triade mortelle, et pourquoi le Q-learning avec un réseau de neurones les réunit-il tous ?

Réponse : Approximation de fonction (des poids partagés entre états), bootstrapping (la cible contient l’estimation elle-même) et apprentissage hors politique (les données ne suivent pas la politique évaluée). Le Q-learning fait du bootstrapping ($r + \gamma \max Q(s', \cdot)$) et il est hors politique par construction (il évalue la politique gloutonne en jouant une politique qui explore) ; remplacer la table par un réseau ajoute l’approximation. Sur le contre-exemple de Baird, la norme des poids passe de 10 à 5 280 avec les trois, et reste sous 25 dès qu’on en retire un.

## Pour aller plus loin

- Le manuel de référence, en accès libre : R. S. Sutton et A. G. Barto, *Reinforcement Learning: An Introduction*, 2ᵉ édition (MIT Press, 2018). Ce chapitre suit sa première partie (chapitres 2 à 8) ; la triade mortelle est au §11.3, REINFORCE au chapitre 13, TD-Gammon et Samuel aux §16.1-16.2.
- Articles originaux (fiches ci-dessus) : Bellman, « A Markovian Decision Process » (1957) ; Samuel, « Some Studies in Machine Learning Using the Game of Checkers » (1959) ; Sutton, « Learning to predict by the methods of temporal differences » (1988) ; Watkins, thèse « Learning from Delayed Rewards » (1989) ; Watkins et Dayan, « Q-learning » (1992) ; Williams, REINFORCE (1992) ; Tesauro, « Temporal Difference Learning and TD-Gammon » (1995, version HTML autorisée par l’ACM) ; Kocsis et Szepesvári, « Bandit Based Monte-Carlo Planning » (2006) ; Coulom, « Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search » (2006).
- Panorama de MCTS : C. Browne et al., « A Survey of Monte Carlo Tree Search Methods » (2012).
- Cours : D. Silver, cours de renforcement de l’UCL (diapositives et vidéos ; leçons 1 à 8 pour ce chapitre, dont la 8 sur Dyna, puis la 9 sur l’exploration et la 10 sur les jeux) ; Stanford, CS234 Reinforcement Learning.
- En pratique : Hugging Face, Deep RL Course, unité 2 « Introduction to Q-Learning », qui code un Q-learning sur FrozenLake ; la documentation de FrozenLake (gymnasium).
- Pour l’intuition : L. Weng, « A (Long) Peek into Reinforcement Learning » (vue d’ensemble avec toutes les équations) ; A. Karpathy, « Deep Reinforcement Learning: Pong from Pixels » (le gradient de politique expliqué simplement, en avant-goût du chapitre 17) ; OpenAI Spinning Up, « Part 3: Intro to Policy Optimization » (la dérivation du gradient de politique pas à pas).
