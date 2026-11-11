# Exercice 11 : renforcement classique, version tabulaire (chapitre 11).
#
#   make ex NUM=11              → éditeur marimo (interactif)
#   make ex NUM=11 MODE=run     → script complet (vérifie les seuils du README)
#   make ex NUM=11 MODE=smoke   → test rapide (< 30 s)
#
# CPU seulement : NumPy, Python pur et gymnasium (groupe uv « rl »). Aucun réseau de
# neurones : chaque valeur est une case d'un tableau.
# Les algorithmes sont des fonctions « @app.function » : elles sont importables depuis
# ce fichier, ce qui permet aux scripts de figures du site (site/scripts/figures/ch11_*.py)
# de refaire EXACTEMENT les mêmes calculs.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import functools
    import itertools
    import math
    import random
    import time
    
    import gymnasium as gym
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from ia_commun import PALETTE, is_smoke, sauvegarder_figure, seed_everything, style_commun


@app.cell
def _():
    mo.md(r"""
    # Exercice 11 : apprendre à agir par essais et erreurs

    Un **agent** observe un **état** $s$, choisit une **action** $a$, reçoit une
    **récompense** $r$ et un nouvel état $s'$. Il cherche la **politique** $\pi(a \mid s)$
    qui maximise le **retour** $G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots$
    (le facteur d'actualisation $\gamma \in [0, 1]$ fait compter moins le futur lointain).

    Tout ici est **tabulaire** : une case de tableau par état (ou par couple état-action).

    1. **Bandits** : un seul état, dix leviers. ε-greedy contre UCB.
    2. **Programmation dynamique** : on connaît le modèle de FrozenLake ; itération sur
       les valeurs et sur les politiques.
    3. **Monte-Carlo contre TD(0)** sur une marche aléatoire : attendre la fin de
       l'épisode ou s'appuyer sur sa propre estimation (*bootstrapping*).
    4. **SARSA contre Q-learning** au bord d'une falaise.
    5. **Q-learning** sur FrozenLake sans connaître le modèle, comparé à l'optimum de la partie 2.
    6. **REINFORCE** : apprendre directement la politique par gradient.
    7. **Dyna-Q** : apprendre un modèle et s'en servir pour planifier.
    8. **MCTS/UCT** au morpion.
    9. **La triade mortelle** : le contre-exemple de Baird.

    **Ce qui est optimisé.** Il n'y a pas de jeu de données étiqueté. Les méthodes de
    valeur (parties 3 à 5, 7) réduisent l'**erreur TD**
    $\delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a)$ par la mise à jour
    $Q(s,a) \leftarrow Q(s,a) + \alpha\,\delta$ (cible fixe, on ne dérive pas à travers
    elle : c'est un « semi-gradient » de $\tfrac12\delta^2$). REINFORCE (partie 6)
    **monte** le long du gradient de l'objectif $J(\theta) = \mathbb{E}_{\pi_\theta}[G_0]$.
    Les parties 1, 2 et 8 n'optimisent aucune perte : moyennes empiriques, équation de
    Bellman, statistiques de visites.
    """)
    return


@app.cell
def _():
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        bandits_problemes=200 if SMOKE else 2000,   # problèmes à 10 bras tirés au hasard
        bandits_pas=300 if SMOKE else 1000,         # tirages par problème
        eval_episodes=300 if SMOKE else 2000,       # épisodes pour mesurer une politique
        marche_repetitions=20 if SMOKE else 100,    # répétitions de la marche aléatoire
        falaise_repetitions=3 if SMOKE else 20,     # répétitions SARSA / Q-learning
        q_episodes=2000 if SMOKE else 10000,        # Q-learning sur FrozenLake
        reinforce_episodes=2000 if SMOKE else 15000,
        dyna_repetitions=2 if SMOKE else 10,
        uct_iterations=[10, 50, 200] if SMOKE else [10, 50, 200, 1000],
        uct_parties=10 if SMOKE else 100,           # parties par adversaire et par budget
        baird_balayages=300 if SMOKE else 2000,
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SEED, SMOKE, config


@app.function
def argmax_aleatoire(q, rng):
    """Indice du maximum ; en cas d'égalité, tirage au hasard (sinon l'action 0 est favorisée)."""
    meilleurs = np.flatnonzero(q == q.max())
    return int(meilleurs[rng.integers(len(meilleurs))])


@app.function
def epsilon_greedy(q, eps, rng):
    """Avec probabilité ε une action au hasard (exploration), sinon la meilleure (exploitation)."""
    if rng.random() < eps:
        return int(rng.integers(len(q)))
    return argmax_aleatoire(q, rng)


@app.cell
def _():
    mo.md(r"""
    ## 1. Bandits manchots : explorer ou exploiter ?

    Dix leviers ; le levier $a$ rapporte en moyenne $q_*(a)$, tiré une fois pour toutes
    selon $\mathcal{N}(0, 1)$, plus un bruit $\mathcal{N}(0, 1)$ à chaque tirage (le banc
    d'essai du chapitre 2 de Sutton et Barto). L'agent ne connaît pas les $q_*$ : il
    tient une moyenne $Q(a)$ de ce qu'il a obtenu.

    - **ε-greedy** : la meilleure moyenne, sauf une fois sur $1/\varepsilon$ un levier au hasard.
    - **UCB** (*upper confidence bound*) : le levier qui maximise
      $Q(a) + c\sqrt{\ln t / N(a)}$. Le bonus est grand pour un levier peu essayé : on
      explore là où l'incertitude est grande, pas au hasard.

    Tout est vectorisé : les 2 000 problèmes avancent ensemble, un pas à la fois.
    """)
    return


@app.function
def bandits(strategie, param, n_problemes, n_pas, graine):
    """Banc d'essai à 10 bras. Renvoie (récompense moyenne, fraction d'action optimale) par pas.

    strategie : "epsilon" (param = ε) ou "ucb" (param = c).
    """
    rng = np.random.default_rng(graine)
    k = 10
    q_vrai = rng.normal(0.0, 1.0, (n_problemes, k))
    optimal = q_vrai.argmax(axis=1)
    Q = np.zeros((n_problemes, k))       # moyennes empiriques
    N = np.zeros((n_problemes, k))       # nombre de tirages de chaque bras
    lignes = np.arange(n_problemes)
    recompense, action_opt = np.zeros(n_pas), np.zeros(n_pas)
    for t in range(n_pas):
        bruit = 1e-9 * rng.random(Q.shape)          # départage les égalités au hasard
        if strategie == "epsilon":
            a = (Q + bruit).argmax(axis=1)
            explore = rng.random(n_problemes) < param
            a = np.where(explore, rng.integers(0, k, n_problemes), a)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                score = Q + param * np.sqrt(np.log(t + 1) / N)
            score[N == 0] = np.inf                  # chaque bras est essayé une fois d'abord
            a = (score + bruit).argmax(axis=1)
        r = rng.normal(q_vrai[lignes, a], 1.0)
        N[lignes, a] += 1
        Q[lignes, a] += (r - Q[lignes, a]) / N[lignes, a]   # moyenne incrémentale
        recompense[t] = r.mean()
        action_opt[t] = (a == optimal).mean()
    return recompense, action_opt


@app.function
def strategies_bandits():
    """Les quatre réglages comparés (nom affiché, stratégie, paramètre)."""
    return [
        ("glouton (ε = 0)", "epsilon", 0.0),
        ("ε = 0,01", "epsilon", 0.01),
        ("ε = 0,1", "epsilon", 0.1),
        ("UCB, c = 2", "ucb", 2.0),
    ]


@app.cell
def _(SEED, config):
    _t0 = time.perf_counter()
    res_bandits = {}
    for _nom, _strat, _p in strategies_bandits():
        res_bandits[_nom] = bandits(_strat, _p, config["bandits_problemes"], config["bandits_pas"], SEED)
    _fin = slice(-100, None)
    for _nom, (_r, _o) in res_bandits.items():
        print(f"{_nom:16s} récompense moyenne (100 derniers pas) {_r[_fin].mean():.3f}, "
              f"action optimale {_o[_fin].mean():.1%}")
    print(f"({time.perf_counter() - _t0:.1f} s)")
    return (res_bandits,)


@app.cell
def _(SMOKE, res_bandits):
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for _nom, (_r, _o) in res_bandits.items():
        _a1.plot(_r, label=_nom, lw=1)
        _a2.plot(100 * _o, label=_nom, lw=1)
    _a1.set(xlabel="pas", ylabel="récompense moyenne", title="Récompense obtenue")
    _a2.set(xlabel="pas", ylabel="% action optimale", title="Fréquence du meilleur levier", ylim=(0, 100))
    _a2.legend()
    _fig.tight_layout()
    sauvegarder_figure(_fig, "bandits-smoke" if SMOKE else "bandits", mo.notebook_dir() / "figures")
    _fig
    return


@app.cell
def _():
    mo.md(r"""
    ## 2. Programmation dynamique : quand on connaît le modèle

    **FrozenLake** (gymnasium) : un lac gelé de 4 × 4 cases, départ en haut à gauche,
    but en bas à droite (récompense 1), des trous (fin de l'épisode, récompense 0). La
    glace est **glissante** : l'action voulue n'est exécutée qu'une fois sur trois, sinon
    l'agent part sur l'un des deux côtés. C'est un **processus de décision markovien**
    (MDP) : la suite ne dépend que de l'état courant et de l'action.

    gymnasium expose le modèle complet dans `env.unwrapped.P[s][a]` : la liste des
    `(probabilité, s', r, terminé)`. Avec lui, l'**équation de Bellman d'optimalité**

    $$V^*(s) = \max_a \sum_{s'} p(s' \mid s, a)\,\big[r + \gamma V^*(s')\big]$$

    se résout par itération, sans jamais jouer une partie :

    - **itération sur les valeurs** : appliquer l'équation comme une affectation jusqu'à
      ce que plus rien ne bouge ;
    - **itération sur les politiques** : évaluer exactement une politique (un système
      linéaire), la rendre gloutonne par rapport à cette évaluation, recommencer.
    """)
    return


@app.function
def modele_tabulaire(P, n_etats, n_actions):
    """Convertit le dictionnaire P de gymnasium en tableaux NumPy.

    T[s, a, s'] : probabilité d'arriver en s' ET de continuer (transitions terminales exclues),
    R[s, a]     : récompense moyenne immédiate.
    """
    T = np.zeros((n_etats, n_actions, n_etats))
    R = np.zeros((n_etats, n_actions))
    for s in range(n_etats):
        for a in range(n_actions):
            for p, s2, r, termine in P[s][a]:
                R[s, a] += p * r
                if not termine:
                    T[s, a, s2] += p
    return T, R


@app.function
def iteration_valeurs(T, R, gamma, tol=1e-10):
    """Renvoie (V*, politique gloutonne, nombre de balayages)."""
    V = np.zeros(T.shape[0])
    n = 0
    while True:
        Q = R + gamma * T @ V                  # Q[s, a] = r + γ Σ p(s'|s,a) V(s')
        V_nouveau = Q.max(axis=1)
        n += 1
        if np.abs(V_nouveau - V).max() < tol:
            return V_nouveau, Q.argmax(axis=1), n
        V = V_nouveau


@app.function
def iteration_politiques(T, R, gamma):
    """Renvoie (V, politique, nombre d'améliorations). Évaluation exacte par résolution linéaire."""
    n_etats = T.shape[0]
    pi = np.zeros(n_etats, dtype=int)
    n = 0
    while True:
        T_pi = T[np.arange(n_etats), pi]          # (n_etats, n_etats)
        R_pi = R[np.arange(n_etats), pi]
        V = np.linalg.solve(np.eye(n_etats) - gamma * T_pi, R_pi)   # V = R_π + γ T_π V
        Q = R + gamma * T @ V
        nouvelle = Q.argmax(axis=1)
        n += 1
        # On ne change d'action que si c'est strictement mieux (évite d'osciller entre ex aequo)
        stable = np.all(Q[np.arange(n_etats), pi] >= Q.max(axis=1) - 1e-12)
        if stable:
            return V, pi, n
        pi = nouvelle


@app.function
def creer_frozenlake(carte="4x4"):
    return gym.make("FrozenLake-v1", map_name=carte, is_slippery=True)


@app.function
def evaluer(env, politique, n_episodes, graine):
    """Taux de réussite d'une politique déterministe (tableau état → action).

    Chaque épisode s'arrête à la limite de pas de gymnasium (100 en 4 × 4, 200 en 8 × 8)."""
    succes = 0.0
    for i in range(n_episodes):
        s, _ = env.reset(seed=graine + i)
        fini = False
        r = 0.0
        while not fini:
            s, r, termine, tronque, _ = env.step(int(politique[s]))
            fini = termine or tronque
        succes += r
    return succes / n_episodes


@app.function
def fleches():
    """Actions de FrozenLake : 0 gauche, 1 bas, 2 droite, 3 haut."""
    return {0: "←", 1: "↓", 2: "→", 3: "↑"}


@app.function
def dessiner_carte_valeurs(ax, desc, V, politique, titre, taille_texte=9):
    """Carte de valeur d'un FrozenLake : couleur = V(s), flèche = action gloutonne, trous en noir."""
    h, w = desc.shape
    grille = V.reshape(h, w).copy()
    terminaux = (desc == b"H") | (desc == b"G")
    grille_aff = np.ma.masked_where(terminaux, grille)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#222222")                     # trous en gris très foncé
    im = ax.imshow(grille_aff, cmap=cmap, vmin=0, vmax=max(V.max(), 1e-9))
    for i, j in zip(*np.nonzero(desc == b"G")):  # but en orange (terminal : V = 0 par convention)
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=PALETTE[1]))
    f = fleches()
    for i in range(h):
        for j in range(w):
            s = i * w + j
            c = desc[i, j]
            if c == b"H":
                ax.text(j, i, "trou", ha="center", va="center", color="#bbbbbb", fontsize=taille_texte - 2)
            elif c == b"G":
                ax.text(j, i, "but", ha="center", va="center", color="black", fontsize=taille_texte, weight="bold")
            else:
                couleur = "white" if V[s] < 0.6 * V.max() else "black"
                ax.text(j, i - 0.18, f[int(politique[s])], ha="center", va="center", color=couleur,
                        fontsize=taille_texte + 3)
                ax.text(j, i + 0.25, f"{V[s]:.2f}".replace(".", ","), ha="center", va="center",
                        color=couleur, fontsize=taille_texte - 2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(titre)
    return im


@app.cell
def _(SEED, config):
    GAMMA_LAC = 0.99
    lac = creer_frozenlake("4x4")
    T_lac, R_lac = modele_tabulaire(lac.unwrapped.P, 16, 4)
    _t0 = time.perf_counter()
    V_iv, pi_iv, n_iv = iteration_valeurs(T_lac, R_lac, GAMMA_LAC)
    _t1 = time.perf_counter()
    V_ip, _pi_ip, n_ip = iteration_politiques(T_lac, R_lac, GAMMA_LAC)
    _t2 = time.perf_counter()
    succes_iv = evaluer(lac, pi_iv, config["eval_episodes"], SEED + 10_000)
    succes_hasard = np.mean([
        evaluer(lac, np.random.default_rng(SEED + k).integers(0, 4, 16), config["eval_episodes"] // 10, SEED + 10_000)
        for k in range(10)
    ])
    print(f"itération sur les valeurs    : {n_iv} balayages ({1000 * (_t1 - _t0):.1f} ms), V*(départ) = {V_iv[0]:.3f}")
    print(f"itération sur les politiques : {n_ip} améliorations ({1000 * (_t2 - _t1):.1f} ms), "
          f"écart max des valeurs {np.abs(V_iv - V_ip).max():.1e}")
    print(f"politique optimale : {succes_iv:.1%} de réussite sur {config['eval_episodes']} épisodes")
    print(f"politiques déterministes tirées au hasard : {succes_hasard:.1%} en moyenne")
    print("politique (← ↓ → ↑) :")
    for _i in range(4):
        print("  " + " ".join(fleches()[int(a)] for a in pi_iv[4 * _i: 4 * _i + 4]))
    # Même chose sur la grande carte 8 × 8 (pour la figure)
    lac8 = creer_frozenlake("8x8")
    T8, R8 = modele_tabulaire(lac8.unwrapped.P, 64, 4)
    V8, pi8, n8 = iteration_valeurs(T8, R8, GAMMA_LAC)
    succes8 = evaluer(lac8, pi8, config["eval_episodes"], SEED + 10_000)
    print(f"8 × 8 : {n8} balayages, V*(départ) = {V8[0]:.3f}, réussite {succes8:.1%}")
    return (
        GAMMA_LAC, R_lac, T_lac, V8, V_ip, V_iv, lac, lac8, n_ip, n_iv, pi8, pi_iv,
        succes8, succes_hasard, succes_iv,
    )


@app.cell
def _(SMOKE, V8, V_iv, lac, lac8, pi8, pi_iv):
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1, 1.6]})
    dessiner_carte_valeurs(_a1, lac.unwrapped.desc, V_iv, pi_iv, "FrozenLake 4 × 4 : V* et politique")
    _im = dessiner_carte_valeurs(_a2, lac8.unwrapped.desc, V8, pi8, "FrozenLake 8 × 8", taille_texte=7)
    _fig.colorbar(_im, ax=_a2, fraction=0.04, label="V*(s)")
    _fig.tight_layout()
    sauvegarder_figure(_fig, "valeurs-smoke" if SMOKE else "valeurs", mo.notebook_dir() / "figures")
    _fig
    return


@app.cell
def _():
    mo.md(r"""
    ## 3. Monte-Carlo contre TD(0) : attendre la fin ou s'appuyer sur soi-même

    Sans modèle, on estime $V$ à partir d'épisodes joués. Marche aléatoire à cinq états
    A, B, C, D, E (départ en C, un pas à gauche ou à droite avec probabilité ½) ;
    sortir à droite rapporte 1, à gauche 0. Les vraies valeurs sont $1/6, 2/6, \dots, 5/6$.

    - **Monte-Carlo** : à la fin de l'épisode, $V(s) \leftarrow V(s) + \alpha\,[G - V(s)]$.
    - **TD(0)** : après chaque pas, $V(s) \leftarrow V(s) + \alpha\,[r + \gamma V(s') - V(s)]$.
      La cible contient une **estimation** $V(s')$ : c'est le *bootstrapping*.

    Ici $\gamma = 1$. On mesure l'erreur quadratique moyenne (RMS) sur les cinq états,
    moyennée sur 100 répétitions.
    """)
    return


@app.function
def marche_aleatoire(n_episodes, alpha, methode, rng):
    """Erreur RMS après chaque épisode, pour TD(0) ou Monte-Carlo (α constant, V initial 0,5)."""
    V = np.full(7, 0.5)
    V[0] = V[6] = 0.0                     # états terminaux
    vrai = np.arange(1, 6) / 6
    erreurs = np.zeros(n_episodes)
    for k in range(n_episodes):
        s, traj = 3, [3]
        while s not in (0, 6):
            s += 1 if rng.random() < 0.5 else -1
            traj.append(s)
        G = 1.0 if s == 6 else 0.0
        if methode == "td":
            for s0, s1 in itertools.pairwise(traj):
                r = 1.0 if s1 == 6 else 0.0
                V[s0] += alpha * (r + V[s1] - V[s0])
        else:
            for s0 in traj[:-1]:
                V[s0] += alpha * (G - V[s0])
        erreurs[k] = np.sqrt(((V[1:6] - vrai) ** 2).mean())
    return erreurs


@app.cell
def _(SEED, config):
    _rng = np.random.default_rng(SEED)
    REGLAGES_MARCHE = [("td", 0.05), ("td", 0.1), ("td", 0.15), ("mc", 0.01), ("mc", 0.02), ("mc", 0.04)]
    res_marche = {}
    for _m, _a in REGLAGES_MARCHE:
        res_marche[(_m, _a)] = np.mean(
            [marche_aleatoire(100, _a, _m, _rng) for _ in range(config["marche_repetitions"])], axis=0)
        print(f"{_m.upper():2s} α = {_a:<5} erreur RMS après 10 épisodes {res_marche[(_m, _a)][9]:.3f}, "
              f"après 100 {res_marche[(_m, _a)][-1]:.3f}")
    return (res_marche,)


@app.cell
def _():
    mo.md(r"""
    ## 4. SARSA contre Q-learning au bord de la falaise

    **CliffWalking** (gymnasium) : grille 4 × 12, départ en bas à gauche, but en bas à
    droite, une falaise entre les deux. Chaque pas coûte −1 ; tomber coûte −100 et renvoie
    au départ. Les deux méthodes explorent avec ε = 0,1 et diffèrent d'un seul terme :

    | | Cible de la mise à jour | Apprend la valeur de… |
    |---|---|---|
    | **SARSA** | $r + \gamma\,Q(s', a')$, avec $a'$ l'action **réellement jouée** ensuite | la politique qui explore (*on-policy*) |
    | **Q-learning** | $r + \gamma \max_{a'} Q(s', a')$ | la politique gloutonne (*off-policy*) |

    Le modèle de gymnasium est déterministe ici : on lit directement `P[s][a]` pour aller
    plus vite que `env.step` (même environnement, sans l'enveloppe).
    """)
    return


@app.function
def falaise(methode, n_episodes, graine, alpha=0.5, eps=0.1, gamma=1.0):
    """SARSA ou Q-learning sur CliffWalking. Renvoie (Q, somme des récompenses par épisode)."""
    P = gym.make("CliffWalking-v1").unwrapped.P
    rng = np.random.default_rng(graine)
    Q = np.zeros((48, 4))
    sommes = np.zeros(n_episodes)
    for k in range(n_episodes):
        s = 36                                   # départ : ligne 3, colonne 0
        a = epsilon_greedy(Q[s], eps, rng)
        total, fini = 0.0, False
        while not fini:
            _, s2, r, fini = P[s][a][0]          # transition déterministe
            total += r
            a2 = epsilon_greedy(Q[s2], eps, rng)
            if methode == "sarsa":
                cible = r + (0.0 if fini else gamma * Q[s2, a2])
            else:
                cible = r + (0.0 if fini else gamma * Q[s2].max())
            Q[s, a] += alpha * (cible - Q[s, a])  # α × erreur TD
            s, a = s2, a2
        sommes[k] = total
    return Q, sommes


@app.function
def chemin_glouton(Q, P, depart, but, max_pas=100):
    """Suit la politique gloutonne de Q dans un modèle déterministe ; renvoie la liste des états."""
    s, chemin = depart, [depart]
    while s != but and len(chemin) <= max_pas:
        s = P[s][int(Q[s].argmax())][0][1]
        chemin.append(int(s))
    return chemin


@app.cell
def _(SEED, config):
    _t0 = time.perf_counter()
    _P = gym.make("CliffWalking-v1").unwrapped.P
    res_falaise = {}
    for _m in ("sarsa", "q"):
        _runs = [falaise(_m, 500, SEED + i) for i in range(config["falaise_repetitions"])]
        _Q = _runs[0][0]
        _ch = chemin_glouton(_Q, _P, 36, 47)
        res_falaise[_m] = dict(sommes=np.mean([r for _, r in _runs], axis=0), chemin=_ch, Q=_Q)
        print(f"{'SARSA' if _m == 'sarsa' else 'Q-learning':10s}: récompense moyenne par épisode "
              f"(100 derniers) {res_falaise[_m]['sommes'][-100:].mean():.1f} ; chemin glouton de "
              f"{len(_ch) - 1} pas, lignes {sorted({int(s) // 12 for s in _ch})}")
    print(f"({time.perf_counter() - _t0:.1f} s)")
    return (res_falaise,)


@app.cell
def _(SMOKE, res_falaise, res_marche):
    _fig, (_a1, _a2, _a3) = plt.subplots(1, 3, figsize=(15, 3.8))
    for (_m, _a), _e in res_marche.items():
        _a1.plot(np.arange(1, 101), _e, ls="-" if _m == "td" else "--",
                 color=PALETTE[0] if _m == "td" else PALETTE[3],
                 alpha={0.05: 0.5, 0.1: 0.75, 0.15: 1.0, 0.01: 0.5, 0.02: 0.75, 0.04: 1.0}[_a],
                 label=f"{'TD(0)' if _m == 'td' else 'Monte-Carlo'}, α = {str(_a).replace('.', ',')}")
    _a1.set(xlabel="épisodes", ylabel="erreur RMS", title="Marche aléatoire : TD contre MC")
    _a1.legend(fontsize=8)
    _noyau = np.ones(10) / 10
    for _m, _lib, _c in (("sarsa", "SARSA", PALETTE[2]), ("q", "Q-learning", PALETTE[1])):
        _a2.plot(np.convolve(res_falaise[_m]["sommes"], _noyau, mode="valid"), color=_c, label=_lib)
    _a2.set(xlabel="épisode", ylabel="somme des récompenses", ylim=(-120, 0),
            title="Falaise : récompense en apprenant (ε = 0,1)")
    _a2.legend()
    _a3.add_patch(plt.Rectangle((0.5, 2.5), 10, 1, color=PALETTE[3], alpha=0.35))
    _a3.text(5.5, 3, "falaise", ha="center", va="center")
    for _m, _lib, _c in (("sarsa", "SARSA", PALETTE[2]), ("q", "Q-learning", PALETTE[1])):
        _ch = res_falaise[_m]["chemin"]
        _a3.plot([s % 12 for s in _ch], [s // 12 for s in _ch], "-o", color=_c, ms=3, label=_lib)
    _a3.set(xlim=(-0.5, 11.5), ylim=(3.5, -0.5), title="Chemins gloutons appris", aspect="equal")
    _a3.set_xticks([])
    _a3.set_yticks([])
    _a3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8)
    _fig.tight_layout()
    sauvegarder_figure(_fig, "td-smoke" if SMOKE else "td", mo.notebook_dir() / "figures")
    _fig
    return


@app.cell
def _():
    mo.md(r"""
    ## 5. Q-learning sur FrozenLake, sans le modèle

    Même lac qu'à la partie 2, mais l'agent ne lit plus `P` : il joue avec `env.reset()`
    et `env.step()` et met à jour une table $Q$ de 16 × 4 cases.
    Exploration ε-greedy (ε décroît de 1 à 0,05 sur la première moitié), pas
    $\alpha = 1/N(s,a)^{0{,}6}$ (il décroît avec le nombre de visites : la glace glissante
    rend les cibles bruitées, un $\alpha$ constant ferait osciller la table).
    """)
    return


@app.function
def q_learning(env, n_episodes, graine, gamma=0.99):
    """Q-learning tabulaire avec l'API gymnasium. Renvoie (Q, succès par épisode)."""
    rng = np.random.default_rng(graine)
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA))
    N = np.zeros((nS, nA))
    succes = np.zeros(n_episodes)
    for k in range(n_episodes):
        eps = max(0.05, 1.0 - k / (0.5 * n_episodes))
        s, _ = env.reset(seed=graine + k)
        fini, r = False, 0.0
        while not fini:
            a = epsilon_greedy(Q[s], eps, rng)
            s2, r, termine, tronque, _ = env.step(a)
            N[s, a] += 1
            alpha = 1.0 / N[s, a] ** 0.6
            cible = r + (0.0 if termine else gamma * Q[s2].max())   # pas de futur après un trou ou le but
            Q[s, a] += alpha * (cible - Q[s, a])
            s, fini = s2, termine or tronque
        succes[k] = r
    return Q, succes


@app.cell
def _(GAMMA_LAC, SEED, V_iv, config, lac, pi_iv, succes_iv):
    _t0 = time.perf_counter()
    Q_ql, succes_ql = q_learning(lac, config["q_episodes"], SEED, GAMMA_LAC)
    duree_ql = time.perf_counter() - _t0
    pi_ql = Q_ql.argmax(axis=1)
    succes_glouton_ql = evaluer(lac, pi_ql, config["eval_episodes"], SEED + 10_000)
    _non_term = np.array([c not in (b"H", b"G") for c in lac.unwrapped.desc.flatten()])
    accord_ql = (pi_ql == pi_iv)[_non_term].mean()
    print(f"Q-learning : {config['q_episodes']} épisodes en {duree_ql:.1f} s")
    print(f"  max Q(départ) = {Q_ql[0].max():.3f} (V* = {V_iv[0]:.3f})")
    print(f"  politique gloutonne : {succes_glouton_ql:.1%} de réussite (optimum {succes_iv:.1%})")
    print(f"  même action que l'optimum dans {accord_ql:.0%} des états non terminaux "
          "(les autres sont souvent des ex aequo à 10⁻³ près)")
    return Q_ql, accord_ql, succes_glouton_ql, succes_ql


@app.cell
def _():
    mo.md(r"""
    ## 6. REINFORCE : apprendre la politique elle-même

    Plus de table $Q$ : une table de **préférences** $\theta(s, a)$, et une politique
    softmax $\pi_\theta(a \mid s) = e^{\theta(s,a)} / \sum_b e^{\theta(s,b)}$. On joue un
    épisode entier, puis pour chaque pas $t$ :

    $$\theta \leftarrow \theta + \eta\,\big(G_t - b(s_t)\big)\,\nabla_\theta \log \pi_\theta(a_t \mid s_t),
    \qquad \nabla_{\theta(s_t,\cdot)} \log \pi = \mathbb{1}_{a_t} - \pi_\theta(\cdot \mid s_t).$$

    C'est une montée de gradient sur $J(\theta) = \mathbb{E}[G_0]$ (théorème du gradient de
    la politique, Williams 1992). Le **baseline** $b(s)$, ici une moyenne mobile des retours
    observés depuis $s$, ne change pas l'espérance du gradient mais réduit sa variance.
    En simplifiant comme la plupart des implémentations, on omet le facteur $\gamma^t$
    de la formule exacte.
    """)
    return


@app.function
def reinforce(env, n_episodes, graine, eta=0.1, gamma=0.99, baseline=True, eta_b=0.05):
    """REINFORCE tabulaire (politique softmax). Renvoie (θ, succès par épisode)."""
    rng = np.random.default_rng(graine)
    nS, nA = env.observation_space.n, env.action_space.n
    theta = np.zeros((nS, nA))
    b = np.zeros(nS)
    succes = np.zeros(n_episodes)

    def politique(s):
        p = np.exp(theta[s] - theta[s].max())
        return p / p.sum()

    for k in range(n_episodes):
        s, _ = env.reset(seed=graine + k)
        trajectoire, fini, r = [], False, 0.0
        while not fini:
            a = int(rng.choice(nA, p=politique(s)))
            s2, r, termine, tronque, _ = env.step(a)
            trajectoire.append((s, a, r))
            s, fini = s2, termine or tronque
        succes[k] = r                              # 1 si l'épisode finit au but
        G = 0.0
        for s, a, r in reversed(trajectoire):
            G = r + gamma * G                      # retour à partir de ce pas
            grad_log = -politique(s)
            grad_log[a] += 1.0                     # 1_a − π(·|s)
            avantage = G - (b[s] if baseline else 0.0)
            theta[s] += eta * avantage * grad_log
            b[s] += eta_b * (G - b[s])
    return theta, succes


@app.cell
def _(GAMMA_LAC, SEED, config, lac):
    _t0 = time.perf_counter()
    res_reinforce = {}
    for _bl in (False, True):
        _theta, _succ = reinforce(lac, config["reinforce_episodes"], SEED, gamma=GAMMA_LAC, baseline=_bl)
        res_reinforce[_bl] = dict(
            succes=_succ, glouton=evaluer(lac, _theta.argmax(axis=1), config["eval_episodes"], SEED + 10_000))
        print(f"REINFORCE {'avec' if _bl else 'sans'} baseline : réussite en jouant (1 000 derniers épisodes) "
              f"{_succ[-1000:].mean():.1%}, politique gloutonne {res_reinforce[_bl]['glouton']:.1%}")
    duree_reinforce = time.perf_counter() - _t0
    print(f"({duree_reinforce:.1f} s)")
    return (res_reinforce,)


@app.cell
def _(SMOKE, res_reinforce, succes_iv, succes_ql):
    _fig, _ax = plt.subplots(figsize=(8, 3.8))
    _f = 500
    _noyau = np.ones(_f) / _f
    _ax.plot(100 * np.convolve(succes_ql, _noyau, mode="valid"), label="Q-learning (en jouant ε-greedy, ε ≥ 0,05)", color=PALETTE[1])
    _ax.plot(100 * np.convolve(res_reinforce[False]["succes"], _noyau, mode="valid"),
             label="REINFORCE sans baseline", color=PALETTE[4])
    _ax.plot(100 * np.convolve(res_reinforce[True]["succes"], _noyau, mode="valid"),
             label="REINFORCE avec baseline", color=PALETTE[0])
    _ax.axhline(100 * succes_iv, color="gray", ls=":", label="politique optimale (partie 2)")
    _ax.set(xlabel="épisode", ylabel=f"% de réussite (moyenne sur {_f})",
            title="FrozenLake 4 × 4 glissant : apprendre sans le modèle", ylim=(0, 100))
    _ax.legend(fontsize=9)
    _fig.tight_layout()
    sauvegarder_figure(_fig, "apprentissage-smoke" if SMOKE else "apprentissage", mo.notebook_dir() / "figures")
    _fig
    return


@app.cell
def _():
    mo.md(r"""
    ## 7. Dyna-Q : apprendre un modèle pour planifier

    Q-learning n'utilise chaque transition qu'une fois. **Dyna-Q** (Sutton, 1990) la
    mémorise aussi dans un **modèle** (ici un dictionnaire `(s, a) → (s', r)`, le
    labyrinthe est déterministe), puis fait $n$ mises à jour **imaginaires** en rejouant
    des transitions tirées du modèle. Labyrinthe de 6 × 9 cases (celui de Sutton et
    Barto), récompense 1 au but, $\gamma = 0{,}95$, $\alpha = 0{,}1$, ε = 0,1.
    """)
    return


@app.function
def labyrinthe_dyna():
    """Labyrinthe 6 × 9 : murs, départ, but (coordonnées ligne, colonne)."""
    murs = {(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)}
    return murs, (2, 0), (0, 8), (6, 9)


@app.function
def dyna_q(n_planif, n_episodes, graine, alpha=0.1, gamma=0.95, eps=0.1):
    """Dyna-Q tabulaire ; renvoie le nombre de pas de chaque épisode."""
    murs, depart, but, (H, W) = labyrinthe_dyna()
    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def pas(s, a):
        n = (s[0] + dirs[a][0], s[1] + dirs[a][1])
        if not (0 <= n[0] < H and 0 <= n[1] < W) or n in murs:
            n = s                                    # bloqué : on reste sur place
        return n, (1.0 if n == but else 0.0), n == but

    rng = np.random.default_rng(graine)
    Q = np.zeros((H, W, 4))
    modele = {}
    pas_par_episode = np.zeros(n_episodes, dtype=int)
    for k in range(n_episodes):
        s, n, fini = depart, 0, False
        while not fini:
            a = epsilon_greedy(Q[s], eps, rng)
            s2, r, fini = pas(s, a)
            n += 1
            Q[s][a] += alpha * (r + (0.0 if fini else gamma * Q[s2].max()) - Q[s][a])   # expérience réelle
            modele[(s, a)] = (s2, r, fini)
            cles = list(modele)
            for _ in range(n_planif):                                                   # expérience simulée
                ps, pa = cles[rng.integers(len(cles))]
                ps2, pr, pf = modele[(ps, pa)]
                Q[ps][pa] += alpha * (pr + (0.0 if pf else gamma * Q[ps2].max()) - Q[ps][pa])
            s = s2
        pas_par_episode[k] = n
    return pas_par_episode


@app.cell
def _(SEED, config):
    _t0 = time.perf_counter()
    res_dyna = {}
    for _n in (0, 5, 50):
        res_dyna[_n] = np.mean([dyna_q(_n, 50, SEED + i) for i in range(config["dyna_repetitions"])], axis=0)
        print(f"n = {_n:2d} pas de planification : épisode 2 en {res_dyna[_n][1]:.0f} pas, "
              f"épisode 5 en {res_dyna[_n][4]:.0f}, total des 50 épisodes {res_dyna[_n].sum():.0f} pas réels")
    print(f"(chemin le plus court : 14 pas ; {time.perf_counter() - _t0:.1f} s)")
    return (res_dyna,)


@app.cell
def _():
    mo.md(r"""
    ## 8. MCTS/UCT au morpion

    Pas de table de valeurs apprise une fois pour toutes : à **chaque coup**, on construit
    un arbre de recherche par simulations. Une itération de **MCTS** (*Monte Carlo tree
    search*) a quatre temps :

    1. **sélection** : descendre dans l'arbre en choisissant l'enfant qui maximise le
       score UCT $\frac{W}{N} + c\sqrt{\frac{\ln N_\text{parent}}{N}}$ (l'UCB de la partie 1,
       appliqué à chaque nœud : c'est l'algorithme **UCT** de Kocsis et Szepesvári, 2006) ;
    2. **expansion** : ajouter un enfant non encore essayé ;
    3. **simulation** : finir la partie au hasard ;
    4. **rétropropagation** : remonter le résultat (1 victoire, ½ nul, 0 défaite) le long
       du chemin.

    On joue le coup le plus visité. Adversaires : un joueur au hasard, et un joueur
    **parfait** (minimax complet du chapitre 04, qui ne perd jamais).
    """)
    return


@app.function
def morpion_lignes():
    return [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


@app.function
def morpion_gagnant(p):
    """1 ou −1 si un joueur a aligné trois pions, 0 sinon."""
    for a, b, c in morpion_lignes():
        if p[a] != 0 and p[a] == p[b] == p[c]:
            return p[a]
    return 0


@app.function
def morpion_fini(p):
    return morpion_gagnant(p) != 0 or 0 not in p


@app.function
@functools.cache
def morpion_negamax(p, joueur):
    """Valeur exacte de la position pour `joueur` (qui doit jouer) : 1, 0 ou −1."""
    g = morpion_gagnant(p)
    if g:
        return g * joueur
    if 0 not in p:
        return 0
    return max(-morpion_negamax(p[:i] + (joueur,) + p[i + 1:], -joueur) for i in range(9) if p[i] == 0)


@app.function
def joueur_parfait(p, joueur, rng):
    """Un des meilleurs coups (minimax), tiré au hasard parmi les ex aequo."""
    valeurs = [(-morpion_negamax(p[:i] + (joueur,) + p[i + 1:], -joueur), i) for i in range(9) if p[i] == 0]
    m = max(v for v, _ in valeurs)
    return rng.choice([i for v, i in valeurs if v == m])


@app.function
def joueur_hasard(p, joueur, rng):
    return rng.choice([i for i in range(9) if p[i] == 0])


@app.class_definition
class NoeudMCTS:
    """Nœud de l'arbre : position, joueur qui doit jouer, statistiques W (gains) et N (visites)."""

    __slots__ = ("N", "W", "a_essayer", "coup", "enfants", "joueur", "p", "parent")

    def __init__(self, p, joueur, parent=None, coup=None):
        self.p, self.joueur, self.parent, self.coup = p, joueur, parent, coup
        self.enfants = []
        self.a_essayer = [] if morpion_fini(p) else [i for i in range(9) if p[i] == 0]
        self.N, self.W = 0, 0.0


@app.function
def uct(p, joueur, n_iterations, rng, c=None):
    """Choisit un coup par MCTS/UCT avec n_iterations simulations (c = √2 par défaut)."""
    c = math.sqrt(2) if c is None else c
    racine = NoeudMCTS(p, joueur)
    for _ in range(n_iterations):
        n = racine
        # 1. Sélection : tant que le nœud est complètement développé, suivre le meilleur score UCT
        while not n.a_essayer and n.enfants:
            n = max(n.enfants, key=lambda e: e.W / e.N + c * math.sqrt(math.log(n.N) / e.N))
        # 2. Expansion : un coup pas encore essayé
        if n.a_essayer:
            i = n.a_essayer.pop(rng.randrange(len(n.a_essayer)))
            enfant = NoeudMCTS(n.p[:i] + (n.joueur,) + n.p[i + 1:], -n.joueur, n, i)
            n.enfants.append(enfant)
            n = enfant
        # 3. Simulation : partie finie au hasard
        q, j = list(n.p), n.joueur
        while not morpion_fini(q):
            q[rng.choice([i for i in range(9) if q[i] == 0])] = j
            j = -j
        g = morpion_gagnant(q)
        # 4. Rétropropagation : W compte du point de vue du joueur qui a joué le coup menant au nœud
        while n is not None:
            n.N += 1
            n.W += 1.0 if g == -n.joueur else (0.5 if g == 0 else 0.0)
            n = n.parent
    return max(racine.enfants, key=lambda e: e.N).coup


@app.function
def partie_morpion(joueur_x, joueur_o):
    """Joue une partie ; renvoie le gagnant (1 = X, −1 = O, 0 = nul)."""
    p, j = (0,) * 9, 1
    while not morpion_fini(p):
        i = (joueur_x if j == 1 else joueur_o)(p, j)
        p = p[:i] + (j,) + p[i + 1:]
        j = -j
    return morpion_gagnant(p)


@app.function
def tournoi_uct(n_iterations, adversaire, n_parties, graine):
    """UCT contre `adversaire`, en alternant qui commence. Renvoie (victoires, nuls, défaites) d'UCT."""
    rng = random.Random(graine)

    def mcts(p, j):
        return uct(p, j, n_iterations, rng)

    def adv(p, j):
        return adversaire(p, j, rng)

    v = nul = d = 0
    for k in range(n_parties):
        g = partie_morpion(mcts, adv) if k % 2 == 0 else -partie_morpion(adv, mcts)
        v, nul, d = v + (g == 1), nul + (g == 0), d + (g == -1)
    return v, nul, d


@app.cell
def _(SEED, config):
    _t0 = time.perf_counter()
    res_uct = {}
    for _n in config["uct_iterations"]:
        for _nom, _adv in (("hasard", joueur_hasard), ("parfait", joueur_parfait)):
            res_uct[(_n, _nom)] = tournoi_uct(_n, _adv, config["uct_parties"], SEED)
        _h, _p = res_uct[(_n, "hasard")], res_uct[(_n, "parfait")]
        print(f"UCT {_n:4d} itérations : contre le hasard {_h[0]} V / {_h[1]} N / {_h[2]} D ; "
              f"contre le joueur parfait {_p[0]} V / {_p[1]} N / {_p[2]} D")
    print(f"({time.perf_counter() - _t0:.1f} s)")
    return (res_uct,)


@app.cell
def _(SMOKE, config, res_dyna, res_uct):
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(12, 3.8))
    for _n, _c in zip((0, 5, 50), PALETTE):
        _a1.plot(np.arange(2, 51), res_dyna[_n][1:], color=_c, label=f"n = {_n}")
    _a1.set(xlabel="épisode", ylabel="pas pour atteindre le but", yscale="log",
            title="Dyna-Q : n mises à jour imaginaires par pas réel")
    _a1.legend()
    _its = config["uct_iterations"]
    _x = np.arange(len(_its))
    _larg = 0.38
    for _k, (_nom, _dec) in enumerate((("hasard", -_larg / 2), ("parfait", _larg / 2))):
        _bas = np.zeros(len(_its))
        for _idx, (_lib, _c) in enumerate((("victoires", PALETTE[2]), ("nuls", "#999999"), ("défaites", PALETTE[3]))):
            _val = np.array([100 * res_uct[(n, _nom)][_idx] / config["uct_parties"] for n in _its])
            _a2.bar(_x + _dec, _val, _larg * 0.95, bottom=_bas, color=_c, label=_lib if _k == 0 else None,
                    hatch="" if _nom == "hasard" else "//", edgecolor="white", lw=0.5)
            _bas += _val
    _a2.set_xticks(_x, [f"{n}\nhasard | parfait" for n in _its], fontsize=8)
    _a2.set(ylabel="% des parties", title="UCT au morpion selon le nombre d'itérations")
    _a2.legend(fontsize=8, loc="lower right")
    _fig.tight_layout()
    sauvegarder_figure(_fig, "planification-smoke" if SMOKE else "planification", mo.notebook_dir() / "figures")
    _fig
    return


@app.cell
def _():
    mo.md(r"""
    ## 9. La triade mortelle : le contre-exemple de Baird (1995)

    Dès qu'il y a trop d'états pour une table, on approche $V(s) \approx \mathbf{w}^\top
    \mathbf{x}(s)$ avec des caractéristiques $\mathbf{x}(s)$, et on fait une descente de
    **semi-gradient** sur l'erreur TD. Sutton et Barto (2018, §11.3) appellent **triade
    mortelle** (*deadly triad*) la réunion de trois ingrédients :

    1. **approximation de fonction** (les états partagent des poids) ;
    2. **bootstrapping** (la cible contient l'estimation elle-même) ;
    3. **apprentissage hors politique** (*off-policy* : les états sont visités selon une
       autre politique que celle qu'on évalue).

    Avec les trois, les poids peuvent **diverger**. Contre-exemple de Baird, dans la
    version légèrement modifiée de Sutton et Barto (2018, §11.2) : 7 états, 8 poids, toutes les récompenses nulles (donc $V = 0$ est représentable exactement).
    La politique évaluée va toujours vers l'état 7 ; les états sont visités uniformément.
    On applique la mise à jour TD **en espérance** (sans bruit d'échantillonnage), puis on
    retire un ingrédient à la fois.
    """)
    return


@app.function
def baird(variante, n_balayages, alpha=0.01, gamma=0.99):
    """Semi-gradient TD(0) en espérance sur le contre-exemple de Baird. Renvoie ‖w‖ à chaque balayage.

    variante : "triade" (les trois ingrédients), "tabulaire" (sans approximation),
    "on-policy" (états visités selon la politique évaluée), "monte-carlo" (sans bootstrapping :
    la cible est le vrai retour, 0).
    """
    if variante == "tabulaire":
        X = np.eye(7)
        w = np.array([1, 1, 1, 1, 1, 1, 10.0])
    else:
        X = np.zeros((7, 8))
        for i in range(6):
            X[i, i], X[i, 7] = 2.0, 1.0          # v(s_i) = 2 w_i + w_8
        X[6, 6], X[6, 7] = 1.0, 2.0              # v(s_7) = w_7 + 2 w_8
        w = np.array([1, 1, 1, 1, 1, 1, 10, 1.0])
    # Distribution des états visités : uniforme (off-policy) ou concentrée sur l'état 7 (on-policy)
    mu = np.eye(7)[6] if variante == "on-policy" else np.full(7, 1 / 7)
    normes = np.zeros(n_balayages)
    for k in range(n_balayages):
        v = X @ w
        cible = np.zeros(7) if variante == "monte-carlo" else gamma * v[6] * np.ones(7)
        w = w + alpha * ((mu * (cible - v))[:, None] * X).sum(axis=0)   # Σ μ(s) δ(s) ∇v(s)
        normes[k] = np.linalg.norm(w)
    return normes


@app.function
def variantes_baird():
    return [
        ("triade", "les trois ingrédients"),
        ("tabulaire", "sans approximation (tableau)"),
        ("on-policy", "sans hors-politique"),
        ("monte-carlo", "sans bootstrapping (cible = retour)"),
    ]


@app.cell
def _(SMOKE, config):
    res_baird = {v: baird(v, config["baird_balayages"]) for v, _ in variantes_baird()}
    for _v, _lib in variantes_baird():
        _n = res_baird[_v]
        print(f"{_lib:38s}: ‖w‖ = {_n[0]:.1f} au départ, {_n[-1]:.3g} après {len(_n)} balayages")
    _fig, _ax = plt.subplots(figsize=(8, 3.8))
    for (_v, _lib), _c in zip(variantes_baird(), (PALETTE[3], PALETTE[0], PALETTE[2], PALETTE[1])):
        _ax.plot(res_baird[_v], color=_c, label=_lib, lw=2 if _v == "triade" else 1.2)
    _ax.set(yscale="log", xlabel="balayages (mises à jour en espérance)", ylabel="‖w‖ (échelle log)",
            title="Contre-exemple de Baird : semi-gradient TD(0), α = 0,01, γ = 0,99")
    _ax.legend(fontsize=9)
    _fig.tight_layout()
    sauvegarder_figure(_fig, "triade-smoke" if SMOKE else "triade", mo.notebook_dir() / "figures")
    _fig
    return (res_baird,)


@app.cell
def _(
    SMOKE, V_ip, V_iv, accord_ql, config, n_ip, n_iv, res_baird, res_bandits, res_dyna, res_falaise,
    res_marche, res_reinforce, res_uct, succes_glouton_ql, succes_hasard, succes_iv,
):
    # --- Bilan : seuils du README. En smoke, on vérifie seulement que tout tourne.
    _fin = slice(-100, None)
    _rb = {k: (r[_fin].mean(), o[_fin].mean()) for k, (r, o) in res_bandits.items()}
    _td = min(v[-1] for (m, _), v in res_marche.items() if m == "td")
    _mc = min(v[-1] for (m, _), v in res_marche.items() if m == "mc")
    _uct_max = max(config["uct_iterations"])
    verifs = {
        "Bandits : UCB > ε = 0,1 > glouton (récompense des 100 derniers pas)":
            _rb["UCB, c = 2"][0] > _rb["ε = 0,1"][0] > _rb["glouton (ε = 0)"][0],
        "Bandits : ε = 0,1 choisit le meilleur bras ≥ 75 % du temps, le glouton ≤ 45 %":
            _rb["ε = 0,1"][1] >= 0.75 and _rb["glouton (ε = 0)"][1] <= 0.45,
        "Programmation dynamique : mêmes valeurs par les deux méthodes (écart < 10⁻⁶)":
            np.abs(V_iv - V_ip).max() < 1e-6,
        "Programmation dynamique : itération sur les politiques en ≤ 10 améliorations, valeurs en ≥ 100 balayages":
            n_ip <= 10 and n_iv >= 100,
        "FrozenLake : politique optimale ≥ 70 % de réussite, politiques au hasard ≤ 5 %":
            succes_iv >= 0.70 and succes_hasard <= 0.05,
        "Marche aléatoire : après 100 épisodes, le meilleur TD(0) bat le meilleur Monte-Carlo":
            _td < _mc,
        "Falaise : SARSA gagne plus que Q-learning en explorant (100 derniers épisodes)":
            res_falaise["sarsa"]["sommes"][_fin].mean() > res_falaise["q"]["sommes"][_fin].mean(),
        "Falaise : chemin glouton de Q-learning = 13 pas (le long du bord), SARSA plus long":
            len(res_falaise["q"]["chemin"]) - 1 == 13 and len(res_falaise["sarsa"]["chemin"]) - 1 > 13,
        "Q-learning FrozenLake : politique gloutonne à moins de 3 points de l'optimum":
            succes_glouton_ql >= succes_iv - 0.03,
        "REINFORCE avec baseline : politique gloutonne ≥ 65 % de réussite":
            res_reinforce[True]["glouton"] >= 0.65,
        "Dyna-Q : n = 50 < n = 5 < n = 0 en pas réels cumulés":
            res_dyna[50].sum() < res_dyna[5].sum() < res_dyna[0].sum(),
        f"UCT {_uct_max} itérations : aucune défaite contre le hasard, ≥ 90 % de nuls contre le joueur parfait":
            res_uct[(_uct_max, "hasard")][2] == 0
            and res_uct[(_uct_max, "parfait")][1] >= 0.9 * config["uct_parties"],
        "UCT 10 itérations : perd la majorité des parties contre le joueur parfait":
            res_uct[(10, "parfait")][2] > config["uct_parties"] / 2,
        "Baird : la triade diverge (‖w‖ × 100), les trois variantes restent bornées (‖w‖ ≤ 3 × départ)":
            res_baird["triade"][-1] > 100 * res_baird["triade"][0]
            and all(res_baird[v][-1] <= 3 * res_baird[v][0] for v in ("tabulaire", "on-policy", "monte-carlo")),
    }
    for _k, _ok in verifs.items():
        if _ok:
            print("OK    " + _k)
        elif SMOKE:   # données réduites : un seuil manqué n'est pas une panne
            print("--    " + _k + " (non vérifié en smoke)")
        else:
            print("ÉCHEC " + _k)
    print(f"(accord de la politique Q-learning avec l'optimum : {accord_ql:.0%} des états)")
    return (verifs,)


@app.cell
def _(SMOKE, verifs):
    _echecs = [k for k, ok in verifs.items() if not ok]
    if not SMOKE and _echecs:
        raise SystemExit("Seuils non atteints : " + " ; ".join(_echecs))
    if SMOKE:
        _msg = (f"**Bilan (smoke)** : le code tourne de bout en bout ; seuils non vérifiés sur ces données "
                f"réduites ({sum(verifs.values())} sur {len(verifs)} déjà atteints).")
    else:
        _msg = f"**Bilan** : {sum(verifs.values())} vérifications sur {len(verifs)} passent."
    mo.md(_msg)
    return


if __name__ == "__main__":
    app.run()
