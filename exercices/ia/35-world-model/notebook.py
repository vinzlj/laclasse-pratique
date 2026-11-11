# Exercice 35 : modèles du monde (chapitre 35).
# Partie 1 en numpy : la physique de CartPole réécrite en tableaux, un planificateur par
# tirages aléatoires (MPC), puis un premier modèle du monde appris : une régression linéaire.
# Partie 2 en PyTorch : un modèle du monde générique (MLP) appris sur quelques centaines de
# transitions au hasard ; erreur accumulée selon l'horizon, horizon de planification,
# exploitation des erreurs du modèle, et comparaison avec PPO à budget d'interactions égal.
#
#   make ex NUM=35              → éditeur marimo (interactif)
#   make ex NUM=35 MODE=run     → script complet (vérifie les seuils du README)
#   make ex NUM=35 MODE=smoke   → test rapide (< 30 s)
#
# Les fonctions « @app.function » sont importables ; les figures du site
# (site/scripts/figures/ch35_*.py) lisent figures/resultats.json, écrit par le run complet.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import json
    import time

    import gymnasium as gym
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch import nn

    from ia_commun import PALETTE, is_smoke, sauvegarder_figure, seed_everything, style_commun

    # Constantes de CartPole-v1, recopiées de gymnasium (classic_control/cartpole.py)
    GRAVITE, M_CHARIOT, M_PERCHE, DEMI_LONGUEUR = 9.8, 1.0, 0.1, 0.5
    FORCE, DT = 10.0, 0.02                       # newtons ; secondes par pas
    M_TOTALE = M_CHARIOT + M_PERCHE
    THETA_MAX = 12 * 2 * np.pi / 360              # 12° : au-delà, l'épisode s'arrête
    X_MAX = 2.4                                   # le chariot sort de la piste


@app.cell
def _():
    mo.md(r"""
    # Exercice 35 : un modèle du monde pour CartPole

    Au chapitre 17, PPO apprenait CartPole **sans modèle** (*model-free*) : il essaie,
    il reçoit des récompenses, il corrige sa politique. Il lui faut environ 30 000 pas
    d'interaction pour tenir 500 pas (mesuré en section 8 de cet exercice : 32 768). Un agent **avec
    modèle** (*model-based*) fait autrement :

    1. il apprend un **modèle du monde** $\hat f_\theta$ qui prédit la variation d'état,
       $\hat s_{t+1} = s_t + \hat f_\theta(s_t, a_t)$, à partir de transitions réelles ;
    2. à chaque pas, il **imagine** des milliers de futurs dans ce modèle et joue la
       première action du meilleur (commande prédictive, *MPC*) ;
    3. il ne touche l'environnement réel que pour exécuter cette action.

    **Ce qui est optimisé.** Seul le modèle du monde apprend : par moindres carrés sur la
    variation d'état (régression linéaire en partie 1, MLP en partie 2),
    $$
    \mathcal{L}(\theta) = \frac{1}{N}\sum_{i}\big\lVert \hat f_\theta(s_i, a_i) - (s_i' - s_i) \big\rVert^2 ,
    $$
    une **erreur quadratique moyenne** de dynamique. La planification n'apprend rien :
    elle cherche, à chaque pas, la meilleure suite d'actions dans le modèle.

    **Hypothèse de l'exercice** (celle de Nagabandi et al., 2018, et de PETS) : la fonction
    de récompense est connue ; seule la dynamique est apprise. Dreamer et MuZero (chapitre)
    apprennent aussi la récompense.

    Plan :

    1. **numpy** : la physique de CartPole en tableaux (le « modèle parfait »), le
       planificateur par tirages aléatoires, puis un modèle **linéaire** appris par moindres
       carrés sur 100 transitions jouées au hasard.
    2. **PyTorch** : un MLP qui ne sait rien de la physique, appris sur 100 à 3 000
       transitions ; erreur qui s'accumule quand on le déroule ; horizon de planification ;
       la planification qui **exploite** les erreurs du modèle ; PPO avec le même budget.
    """)
    return


@app.cell
def _():
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    torch.set_num_threads(4)
    config = dict(
        # Planificateur par tirages aléatoires : N suites d'actions de H pas
        n_candidats=200,
        horizon=10,
        n_lin=100,                            # transitions au hasard pour le modèle linéaire
        # Modèle du monde MLP : tailles du jeu de transitions au hasard
        tailles=[100, 300] if SMOKE else [100, 300, 1_000, 3_000],
        tailles_mpc=[100, 300] if SMOKE else [100, 300, 3_000],   # évaluées en MPC (le plus long)
        horizons=[5, 40] if SMOKE else [5, 10, 20, 40],
        n_episodes=1 if SMOKE else 5,         # épisodes d'évaluation (graines 100, 101, …)
        pas_max=100 if SMOKE else 500,        # 500 = limite de CartPole-v1
        # Erreur accumulée : déroulés en boucle ouverte sur des trajectoires de référence
        horizons_erreur=[1, 2, 5, 10, 20, 40, 80],
        n_ref=2 if SMOKE else 5, long_ref=100,
        # Exploitation des erreurs : nombre de candidats comparés
        candidats_exploit=[10, 100, 1_000],
        n_etats_exploit=10 if SMOKE else 100,
        # MLP : couches cachées, pas d'optimisation (lots de 256)
        cachee=128, pas_mlp=300 if SMOKE else 3_000, lr=1e-3,
        # PPO (Stable-Baselines3, réglages par défaut) : budgets d'interactions comparés
        budgets_ppo=[2_048] if SMOKE else [2_048, 4_096, 8_192, 16_384, 32_768],
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}\nconfig {config}")
    return SEED, SMOKE, config


@app.cell
def _():
    mo.md(r"""
    ## 1. La physique de CartPole, en tableaux

    gymnasium simule CartPole par les équations du pendule inversé sur chariot (Barto,
    Sutton et Anderson, 1983), intégrées par la méthode d'Euler avec un pas de 0,02 s.
    On les réécrit pour qu'elles s'appliquent à **un tableau d'états** d'un coup : le
    planificateur pourra ainsi faire avancer 200 futurs imaginés en une seule opération.

    L'état est $s = (x, \dot x, \theta, \dot\theta)$ : position et vitesse du chariot,
    angle et vitesse angulaire de la perche. L'action pousse le chariot de 10 N à gauche
    (0) ou à droite (1). L'épisode s'arrête si $|\theta| > 12°$ ou $|x| > 2{,}4$, et au
    bout de 500 pas ; chaque pas « debout » rapporte 1.
    """)
    return


@app.function
def physique(S, a):
    """Un pas de CartPole-v1 pour un tableau d'états S (n, 4) et d'actions a (n,).
    Mêmes équations, même ordre d'intégration (Euler explicite) que gymnasium."""
    x, x_point, theta, theta_point = S[..., 0], S[..., 1], S[..., 2], S[..., 3]
    force = np.where(a == 1, FORCE, -FORCE)
    c, s = np.cos(theta), np.sin(theta)
    temp = (force + M_PERCHE * DEMI_LONGUEUR * theta_point**2 * s) / M_TOTALE
    theta_acc = (GRAVITE * s - c * temp) / (DEMI_LONGUEUR * (4 / 3 - M_PERCHE * c**2 / M_TOTALE))
    x_acc = temp - M_PERCHE * DEMI_LONGUEUR * theta_acc * c / M_TOTALE
    return np.stack([x + DT * x_point, x_point + DT * x_acc,
                     theta + DT * theta_point, theta_point + DT * theta_acc], axis=-1)


@app.function
def tombe(S):
    """Vrai pour chaque état où l'épisode s'arrête (perche trop penchée ou chariot sorti)."""
    return (np.abs(S[..., 0]) > X_MAX) | (np.abs(S[..., 2]) > THETA_MAX)


@app.cell
def _(SEED):
    # Vérification : 200 pas de gymnasium et de notre fonction, à partir du même état.
    _env = gym.make("CartPole-v1")
    _env.reset(seed=SEED)
    _rng = np.random.default_rng(SEED)
    ecart_physique = 0.0
    for _k in range(200):
        _a = int(_rng.integers(2))
        _prevu = physique(np.array(_env.unwrapped.state, dtype=np.float64), np.array(_a))
        _, _, _termine, _, _ = _env.step(_a)
        ecart_physique = max(ecart_physique, float(np.abs(_prevu - _env.unwrapped.state).max()))
        if _termine:
            _env.reset(seed=SEED + _k)
    print(f"Écart maximal entre notre physique et gymnasium sur 200 pas : {ecart_physique:.1e}")
    return (ecart_physique,)


@app.cell
def _():
    mo.md(r"""
    ## 2. Planifier par tirages aléatoires (*random shooting*)

    À chaque pas réel, depuis l'état courant $s$ :

    ```text
    tirer N suites d'actions au hasard        a(1)₁ … a(1)_H,  …,  a(N)₁ … a(N)_H
    les dérouler dans le modèle               s → ŝ₁ → ŝ₂ → … → ŝ_H     (N futurs en parallèle)
    noter chaque futur                        somme des récompenses imaginées
    jouer la PREMIÈRE action du meilleur      puis tout recommencer au pas suivant
    ```

    Rejouer le plan à chaque pas (« horizon fuyant ») corrige les écarts entre le modèle
    et le monde au fur et à mesure : c'est la **commande prédictive** (*model predictive
    control*, MPC).

    **La note d'un futur.** La vraie récompense (1 par pas debout) est myope : sur un
    horizon de 10 pas, presque tous les futurs restent debout et se valent, alors que le
    chariot dérive lentement vers le bord. On utilise donc la même récompense **adoucie**
    à chaque pas imaginé, $\tilde r = 1 - \tfrac12(\theta/12°)^2 - \tfrac12(x/2{,}4)^2$
    tant que la perche n'est pas tombée, 0 ensuite : on préfère les futurs où la perche
    reste droite et le chariot au centre. Cette récompense est **donnée**, pas apprise.
    L'environnement réel, lui, compte toujours 1 par pas.
    """)
    return


@app.function
def planifier(modele, s, rng, n_candidats, horizon, renvoyer_plans=False):
    """Commande prédictive par tirages aléatoires. `modele(S, a)` fait avancer un tableau
    d'états d'un pas. Renvoie la première action de la meilleure suite (et, sur demande,
    toutes les suites, leurs notes imaginées et l'indice de la meilleure)."""
    A = rng.integers(0, 2, size=(n_candidats, horizon))
    S = np.repeat(np.asarray(s, dtype=np.float64)[None], n_candidats, axis=0)
    debout = np.ones(n_candidats, dtype=bool)
    note = np.zeros(n_candidats)
    for h in range(horizon):
        S = modele(S, A[:, h])
        debout &= ~tombe(S)
        note += debout * (1 - 0.5 * (S[:, 2] / THETA_MAX) ** 2 - 0.5 * (S[:, 0] / X_MAX) ** 2)
    i = int(np.argmax(note))
    if renvoyer_plans:
        return int(A[i, 0]), A, note, i
    return int(A[i, 0])


@app.function
def jouer_mpc(modele, graine, rng, n_candidats, horizon, pas_max=500):
    """Un épisode réel de CartPole-v1 piloté par le planificateur : renvoie le retour."""
    env = gym.make("CartPole-v1", max_episode_steps=pas_max)
    env.reset(seed=graine)
    retour, fini = 0.0, False
    while not fini:
        a = planifier(modele, env.unwrapped.state, rng, n_candidats, horizon)
        _, r, termine, tronque, _ = env.step(a)
        retour += r
        fini = termine or tronque
    env.close()
    return retour


@app.function
def evaluer(modele, rng, n_episodes, n_candidats, horizon, pas_max):
    """Retours de `n_episodes` épisodes (graines 100, 101, …, les mêmes pour tous)."""
    return np.array([jouer_mpc(modele, 100 + k, rng, n_candidats, horizon, pas_max)
                     for k in range(n_episodes)])


@app.cell
def _(SEED, config):
    # Le modèle parfait : la planification seule suffit-elle ?
    _rng = np.random.default_rng(SEED)
    _t0 = time.perf_counter()
    retours_parfait = evaluer(physique, _rng, config["n_episodes"], config["n_candidats"],
                              config["horizon"], config["pas_max"])
    duree_parfait = time.perf_counter() - _t0
    print(f"Modèle parfait (N = {config['n_candidats']}, H = {config['horizon']}) : "
          f"retours {retours_parfait.tolist()} en {duree_parfait:.1f} s")
    return duree_parfait, retours_parfait


@app.cell
def _():
    mo.md(r"""
    ## 3. Un premier modèle appris : la régression linéaire

    On ne donne plus la physique à l'agent. Il joue **au hasard** quelques épisodes
    (environ 22 pas chacun) et enregistre les transitions $(s, a, s')$. Puis il ajuste,
    par moindres carrés, un modèle linéaire de la **variation** d'état :
    $$
    s' - s \approx W^\top \,[\,s,\ \pm 1,\ 1\,]
    $$
    ($\pm 1$ code l'action, le 1 final donne un terme constant). C'est la régression
    linéaire du chapitre 00, avec quatre sorties : `np.linalg.lstsq` la résout en une ligne.
    """)
    return


@app.function
def collecter_au_hasard(n, graine):
    """n transitions (s, a, s') jouées au hasard dans CartPole-v1 (épisodes enchaînés)."""
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(graine)
    S, A, S2 = [], [], []
    episode = 0
    while len(S) < n:
        env.reset(seed=graine * 1_000 + episode)
        episode += 1
        fini = False
        while not fini and len(S) < n:
            s = np.array(env.unwrapped.state, dtype=np.float64)
            a = int(rng.integers(2))
            _, _, termine, tronque, _ = env.step(a)
            S.append(s)
            A.append(a)
            S2.append(np.array(env.unwrapped.state, dtype=np.float64))
            fini = termine or tronque
    env.close()
    return np.array(S), np.array(A), np.array(S2), episode


@app.function
def ajuster_lineaire(S, A, S2):
    """Moindres carrés : W minimise ||Z W - (S2 - S)||², Z = [s, ±1, 1]. Renvoie le modèle."""
    Z = np.c_[S, 2 * A - 1, np.ones(len(S))]
    W, *_ = np.linalg.lstsq(Z, S2 - S, rcond=None)

    def modele(Sb, a):
        return Sb + np.c_[Sb, 2 * a - 1, np.ones(len(Sb))] @ W

    return modele


@app.cell
def _(SEED, config):
    _S, _A, _S2, _n_ep = collecter_au_hasard(config["n_lin"], SEED)
    modele_lineaire = ajuster_lineaire(_S, _A, _S2)
    # Erreur à un pas, sur d'autres transitions au hasard
    _St, _At, _S2t, _ = collecter_au_hasard(2_000, SEED + 1)
    erreur_lin_1pas = np.sqrt(((modele_lineaire(_St, _At) - _S2t) ** 2).mean(0))
    _rng = np.random.default_rng(SEED)
    retours_lineaire = evaluer(modele_lineaire, _rng, config["n_episodes"], config["n_candidats"],
                               config["horizon"], config["pas_max"])
    print(f"Modèle linéaire appris sur {config['n_lin']} transitions ({_n_ep} épisodes au hasard)")
    print("  erreur à un pas (RMS) sur x, ẋ, θ, θ̇ :", np.array2string(erreur_lin_1pas, precision=6))
    print(f"  MPC avec ce modèle : retours {retours_lineaire.tolist()}")
    return erreur_lin_1pas, modele_lineaire, retours_lineaire


@app.cell
def _():
    mo.md(r"""
    Le modèle linéaire suffit ici, et ce n'est pas un hasard : près de la verticale,
    $\sin\theta \approx \theta$ et $\cos\theta \approx 1$, les équations de CartPole sont
    presque linéaires (c'est pour cela que la commande linéaire classique, LQR, le
    stabilise). Sur un robot qui marche, une main qui saisit ou un jeu vidéo, aucune
    forme simple n'est connue d'avance : on prend un réseau de neurones générique. C'est
    la partie 2.

    ## 4. Un modèle du monde générique : un MLP

    Le MLP reçoit l'état **normalisé** et l'action ($\pm 1$), et prédit la variation
    d'état normalisée ; perte : erreur quadratique moyenne (MSE). Il ne sait rien des
    sinus, des masses ni de la gravité.
    """)
    return


@app.class_definition
class ModeleMonde(nn.Module):
    """MLP 5 → 128 → 128 → 4 : (état normalisé, action ±1) → variation d'état normalisée."""

    def __init__(self, cachee=128):
        super().__init__()
        self.reseau = nn.Sequential(nn.Linear(5, cachee), nn.SiLU(),
                                    nn.Linear(cachee, cachee), nn.SiLU(),
                                    nn.Linear(cachee, 4))

    def forward(self, x):
        return self.reseau(x)


@app.function
def entrainer_mlp(S, A, S2, n_pas, lr=1e-3, cachee=128, graine=42, taille_lot=256):
    """Entraîne le modèle du monde par MSE sur les variations normalisées.
    Renvoie (modèle numpy pour la planification, pertes)."""
    torch.manual_seed(graine)
    moy, ecart = S.mean(0), S.std(0) + 1e-8
    D = S2 - S
    d_moy, d_ecart = D.mean(0), D.std(0) + 1e-8
    X = torch.tensor(np.c_[(S - moy) / ecart, 2 * A - 1], dtype=torch.float32)
    Y = torch.tensor((D - d_moy) / d_ecart, dtype=torch.float32)
    reseau = ModeleMonde(cachee)
    optim = torch.optim.Adam(reseau.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(graine)
    pertes = []
    for _ in range(n_pas):
        idx = torch.randint(len(X), (min(taille_lot, len(X)),), generator=gen)
        perte = ((reseau(X[idx]) - Y[idx]) ** 2).mean()
        optim.zero_grad()
        perte.backward()
        optim.step()
        pertes.append(perte.item())
    # Pour planifier vite (des milliers d'appels sur de petits lots), on recopie les poids
    # dans des tableaux numpy : même calcul, sans le coût d'appel de PyTorch.
    # Calcul en float32 (deux fois plus rapide) ; l'état, lui, reste en float64.
    W = [p.detach().numpy() for p in reseau.parameters()]

    def silu(z):
        return z / (1 + np.exp(-np.clip(z, -60, 60)))

    def modele(Sb, a):
        h = np.c_[(Sb - moy) / ecart, 2 * a - 1].astype(np.float32)
        h = silu(h @ W[0].T + W[1])
        h = silu(h @ W[2].T + W[3])
        return Sb + (h @ W[4].T + W[5]) * d_ecart + d_moy

    return modele, pertes


@app.cell
def _(mo):
    bouton = mo.ui.run_button(label="Entraîner les modèles du monde et planifier (≈ 3 min)")
    bouton
    return (bouton,)


@app.cell
def _(SEED, bouton, config):
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Entraîner les modèles du monde et planifier ».*"))
    _t0 = time.perf_counter()
    modeles_mlp, modeles_lin, pertes_mlp, jeux = {}, {}, {}, {}
    for _n in config["tailles"]:
        _S, _A, _S2, _n_ep = collecter_au_hasard(_n, SEED + _n)
        jeux[_n] = _n_ep
        modeles_mlp[_n], pertes_mlp[_n] = entrainer_mlp(_S, _A, _S2, config["pas_mlp"], config["lr"],
                                                        config["cachee"], SEED)
        modeles_lin[_n] = ajuster_lineaire(_S, _A, _S2)
        print(f"n = {_n:5d} transitions ({_n_ep:3d} épisodes au hasard) : MSE finale "
              f"{np.mean(pertes_mlp[_n][-100:]):.1e} (variations normalisées)")
    duree_mlp = time.perf_counter() - _t0
    print(f"Entraînement des {len(config['tailles'])} MLP : {duree_mlp:.1f} s")
    return duree_mlp, jeux, modeles_lin, modeles_mlp, pertes_mlp


@app.cell
def _():
    mo.md(r"""
    ## 5. L'erreur s'accumule quand on déroule le modèle

    Pour planifier sur $H$ pas, on applique le modèle **à ses propres prédictions** :
    $\hat s_{t+2} = \hat f(\hat s_{t+1}, a_{t+1})$, etc. Une petite erreur à un pas devient
    l'entrée du pas suivant, et CartPole est **instable** (une perche qui penche un peu
    penche de plus en plus vite) : l'écart grandit à peu près exponentiellement.

    Mesure : des trajectoires réelles de référence (pilotées par le modèle parfait, 100 pas
    chacune). Depuis chaque état, on rejoue dans le modèle appris les mêmes actions,
    **sans jamais le recaler** sur l'état réel (boucle ouverte), et l'on compare l'angle
    prédit à l'angle réel au bout de $h$ pas.
    """)
    return


@app.function
def trajectoires_reference(n, longueur, graine):
    """n trajectoires réelles de `longueur` pas, pilotées par MPC avec le modèle parfait."""
    rng = np.random.default_rng(graine)
    refs = []
    for k in range(n):
        env = gym.make("CartPole-v1")
        env.reset(seed=7_000 + k)
        S, A = [np.array(env.unwrapped.state, dtype=np.float64)], []
        for _ in range(longueur):
            a = planifier(physique, S[-1], rng, 100, 10)
            env.step(a)
            S.append(np.array(env.unwrapped.state, dtype=np.float64))
            A.append(a)
        refs.append((np.array(S), np.array(A)))
        env.close()
    return refs


@app.function
def erreur_deroulee(modele, refs, horizons, pas_depart=5):
    """Écart absolu médian (en degrés) entre l'angle prédit en boucle ouverte et l'angle
    réel, après h pas, pour chaque h de `horizons`."""
    resultats = []
    for h in horizons:
        ecarts = []
        for S, A in refs:
            departs = np.arange(0, len(A) - h + 1, pas_depart)
            Sh = S[departs].copy()
            for k in range(h):
                Sh = modele(Sh, A[departs + k])
            ecarts.append(np.abs(Sh[:, 2] - S[departs + h, 2]))
        resultats.append(float(np.degrees(np.median(np.concatenate(ecarts)))))
    return resultats


@app.cell
def _(SEED, config, modeles_lin, modeles_mlp):
    _t0 = time.perf_counter()
    refs = trajectoires_reference(config["n_ref"], config["long_ref"], SEED)
    erreurs = {f"MLP, {n} transitions": erreur_deroulee(m, refs, config["horizons_erreur"])
               for n, m in modeles_mlp.items()}
    _n0 = config["tailles"][0]
    erreurs[f"linéaire, {_n0} transitions"] = erreur_deroulee(modeles_lin[_n0], refs, config["horizons_erreur"])
    print(f"Écart médian sur l'angle (degrés) après h pas, h = {config['horizons_erreur']}")
    for _k, _v in erreurs.items():
        print(f"  {_k:24s} " + "  ".join(f"{e:8.3f}" for e in _v))
    print(f"({time.perf_counter() - _t0:.1f} s)")
    return (erreurs,)


@app.cell
def _():
    mo.md(r"""
    ## 6. Planifier avec le modèle appris : quelle longueur d'horizon ?

    Horizon court : le planificateur voit peu loin. Horizon long : il voit loin… dans un
    modèle qui se trompe de plus en plus. On évalue le MPC (N = 200 candidats) pour chaque
    taille de jeu et chaque horizon, sur 5 épisodes réels.
    """)
    return


@app.cell
def _(SEED, config, modeles_mlp):
    _t0 = time.perf_counter()
    retours_mlp = {}
    for _n in config["tailles_mpc"]:
        _m = modeles_mlp[_n]
        for _H in config["horizons"]:
            _rng = np.random.default_rng(SEED)
            retours_mlp[(_n, _H)] = evaluer(_m, _rng, config["n_episodes"], config["n_candidats"],
                                            _H, config["pas_max"])
    duree_grille = time.perf_counter() - _t0
    print("Retour moyen (min-max) du MPC avec le MLP, sur", config["n_episodes"], "épisodes")
    print("transitions | " + " | ".join(f"H = {h:2d}        " for h in config["horizons"]))
    for _n in config["tailles_mpc"]:
        print(f"{_n:11d} | " + " | ".join(
            f"{retours_mlp[(_n, h)].mean():5.1f} ({retours_mlp[(_n, h)].min():3.0f}-{retours_mlp[(_n, h)].max():3.0f})"
            for h in config["horizons"]))
    print(f"({duree_grille:.1f} s)")
    return duree_grille, retours_mlp


@app.cell
def _():
    mo.md(r"""
    ## 7. Le planificateur exploite les erreurs du modèle

    Le planificateur garde le futur qui a la **meilleure note imaginée**. Si le modèle se
    trompe, les futurs dont il surestime le plus la note ont plus de chances de gagner :
    la note du plan retenu est biaisée vers le haut, même quand le modèle est en moyenne
    pessimiste, et ce biais grandit avec le nombre de candidats comparés (la
    « malédiction de l'optimiseur »).

    Mesure, sur 100 états de départ tirés **plus largement** que les données au hasard
    ($|x| \le 1$, $|\dot x| \le 1$, $|\theta| \le 8°$, $|\dot\theta| \le 1$ ; les épisodes
    au hasard restent surtout près de la verticale) : on tire N plans de 40 pas, on garde
    celui que le modèle note le mieux, puis on note **les mêmes plans** dans la vraie
    physique. Trois chiffres :

    - écart moyen (note imaginée − note réelle) sur tous les candidats : le biais
      « ordinaire » du modèle ;
    - le même écart pour le plan retenu ;
    - le **regret** : note réelle du meilleur candidat (connu grâce à la vraie physique)
      moins note réelle du plan retenu.

    Avec le modèle parfait, les trois sont nuls par construction.
    """)
    return


@app.function
def noter_plans(modele, s, A):
    """Note (récompense adoucie cumulée) de chaque suite d'actions A (N, H) depuis l'état s."""
    S = np.repeat(np.asarray(s, dtype=np.float64)[None], len(A), axis=0)
    debout, note = np.ones(len(A), dtype=bool), np.zeros(len(A))
    for h in range(A.shape[1]):
        S = modele(S, A[:, h])
        debout &= ~tombe(S)
        note += debout * (1 - 0.5 * (S[:, 2] / THETA_MAX) ** 2 - 0.5 * (S[:, 0] / X_MAX) ** 2)
    return note


@app.function
def mesurer_exploitation(modele, etats, n_candidats, horizon, rng):
    """Pour chaque état : N plans au hasard, choix par le modèle, notes réelles.
    Renvoie les moyennes (écart moyen, écart du plan retenu, regret)."""
    ecart_moyen, ecart_retenu, regret = [], [], []
    for s in etats:
        A = rng.integers(0, 2, size=(n_candidats, horizon))
        imaginee = noter_plans(modele, s, A)
        reelle = noter_plans(physique, s, A)
        i = int(np.argmax(imaginee))
        ecart_moyen.append((imaginee - reelle).mean())
        ecart_retenu.append(imaginee[i] - reelle[i])
        regret.append(reelle.max() - reelle[i])
    return dict(moyen=float(np.mean(ecart_moyen)), retenu=float(np.mean(ecart_retenu)),
                regret=float(np.mean(regret)))


@app.cell
def _(SEED, config, modeles_mlp):
    _t0 = time.perf_counter()
    _rng = np.random.default_rng(SEED)
    _k = config["n_etats_exploit"]
    etats_exploit = np.c_[_rng.uniform(-1, 1, _k), _rng.uniform(-1, 1, _k),
                          np.radians(_rng.uniform(-8, 8, _k)), _rng.uniform(-1, 1, _k)]
    exploitation = {}
    _modeles = [(f"MLP, {n} transitions", modeles_mlp[n]) for n in (config["tailles"][0], config["tailles"][-1])]
    for _nom, _m in _modeles + [("parfait", physique)]:
        for _N in config["candidats_exploit"]:
            exploitation[(_nom, _N)] = mesurer_exploitation(_m, etats_exploit, _N, 40,
                                                            np.random.default_rng(SEED + _N))
    print(f"Sur {_k} états, plans de 40 pas (note max 40) :")
    for (_nom, _N), _v in exploitation.items():
        print(f"  {_nom:22s} N = {_N:5d} : écart moyen {_v['moyen']:+6.2f}, plan retenu "
              f"{_v['retenu']:+6.2f}, regret {_v['regret']:5.2f}")
    duree_exploit = time.perf_counter() - _t0
    print(f"({duree_exploit:.1f} s)")
    return duree_exploit, exploitation


@app.cell
def _():
    mo.md(r"""
    ## 8. Combien d'interactions réelles ? Comparaison avec PPO

    Le modèle du monde n'a vu que 100 à 3 000 transitions réelles. PPO (chapitre 17,
    Stable-Baselines3, réglages par défaut) apprend sans modèle ; on l'arrête après 2 048 à
    32 768 pas d'interaction et l'on évalue sa politique (action la plus probable) sur les
    mêmes 5 épisodes.
    """)
    return


@app.function
def ppo_par_budget(budgets, graine, n_episodes):
    """Entraîne PPO jusqu'au plus grand budget, photographie la politique à chaque budget
    et renvoie {budget: retours}."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    env = gym.make("CartPole-v1")
    modele = PPO("MlpPolicy", env, seed=graine, verbose=0, device="cpu")
    copies = {}

    class Photographe(BaseCallback):
        def _on_step(self):
            if self.num_timesteps in budgets:
                copies[self.num_timesteps] = {k: v.clone() for k, v in self.model.policy.state_dict().items()}
            return True

    modele.learn(max(budgets), callback=Photographe())
    resultats = {}
    for b in budgets:
        modele.policy.load_state_dict(copies[b])
        retours = []
        for k in range(n_episodes):
            obs, _ = env.reset(seed=100 + k)
            retour, fini = 0.0, False
            while not fini:
                obs, r, termine, tronque, _ = env.step(int(modele.predict(obs, deterministic=True)[0]))
                retour += r
                fini = termine or tronque
            retours.append(retour)
        resultats[b] = np.array(retours)
    env.close()
    return resultats


@app.cell
def _(SEED, config):
    _t0 = time.perf_counter()
    retours_ppo = ppo_par_budget(config["budgets_ppo"], SEED, config["n_episodes"])
    duree_ppo = time.perf_counter() - _t0
    for _b, _r in retours_ppo.items():
        print(f"PPO après {_b:6d} pas : retours {_r.tolist()} (moyenne {_r.mean():.1f})")
    print(f"({duree_ppo:.1f} s)")
    return duree_ppo, retours_ppo


@app.cell
def _(SMOKE, config, erreurs, exploitation, pertes_mlp, retours_mlp, retours_ppo):
    # --- Figure de l'exercice : erreur selon l'horizon, horizon de planification, PPO ---
    _fig, (_a1, _a2, _a3) = plt.subplots(1, 3, figsize=(15, 4.2))
    for _k, _v in erreurs.items():
        _a1.plot(config["horizons_erreur"], _v, marker="o", label=_k, ls="--" if "linéaire" in _k else "-")
    _a1.set(xscale="log", yscale="log", xlabel="pas déroulés en boucle ouverte (h)",
            ylabel="écart médian sur l'angle (degrés)", title="L'erreur s'accumule")
    _a1.axhline(12, color="grey", ls=":", lw=1)
    _a1.legend(fontsize=8)
    for _i, _n in enumerate(config["tailles_mpc"]):
        _a2.plot(config["horizons"], [retours_mlp[(_n, h)].mean() for h in config["horizons"]], "o-",
                 color=PALETTE[_i], label=f"{_n} transitions")
    _a2.set(xlabel="horizon de planification H", ylabel="retour moyen (max 500)",
            title="MPC avec le MLP", ylim=(0, 520))
    _a2.legend(fontsize=8)
    _b = list(retours_ppo)
    _a3.plot(_b, [retours_ppo[b].mean() for b in _b], "s-", color=PALETTE[3], label="PPO (sans modèle)")
    _a3.plot(config["tailles_mpc"], [retours_mlp[(n, 10)].mean() if (n, 10) in retours_mlp
                                     else retours_mlp[(n, config["horizons"][0])].mean()
                                     for n in config["tailles_mpc"]],
             "o-", color=PALETTE[0], label="MLP + MPC (H = 10)")
    _a3.set(xscale="log", xlabel="interactions réelles utilisées pour apprendre",
            ylabel="retour moyen (max 500)", title="Efficacité en interactions", ylim=(0, 520))
    _a3.legend(fontsize=8)
    _fig.tight_layout()
    sauvegarder_figure(_fig, "modele-monde-smoke" if SMOKE else "modele-monde", mo.notebook_dir() / "figures")
    _fig
    return


@app.cell
def _(
    SMOKE, config, duree_exploit, duree_grille, duree_mlp, duree_parfait, duree_ppo,
    ecart_physique, erreur_lin_1pas, erreurs, exploitation, jeux, pertes_mlp, retours_lineaire,
    retours_mlp, retours_parfait, retours_ppo,
):
    resume = dict(
        config=config, ecart_physique=ecart_physique,
        parfait=retours_parfait.tolist(), lineaire=retours_lineaire.tolist(),
        erreur_lin_1pas=erreur_lin_1pas.tolist(),
        episodes_au_hasard={str(n): e for n, e in jeux.items()},
        mse_finale={str(n): float(np.mean(p[-100:])) for n, p in pertes_mlp.items()},
        erreurs=erreurs,
        mpc={f"{n}|{h}": v.tolist() for (n, h), v in retours_mlp.items()},
        exploitation={f"{nom}|{N}": v for (nom, N), v in exploitation.items()},
        ppo={str(b): v.tolist() for b, v in retours_ppo.items()},
        durees=dict(parfait=duree_parfait, mlp=duree_mlp, grille=duree_grille,
                    exploitation=duree_exploit, ppo=duree_ppo),
    )
    _chemin = mo.notebook_dir() / "figures" / ("resultats-smoke.json" if SMOKE else "resultats.json")
    _chemin.parent.mkdir(exist_ok=True)
    _chemin.write_text(json.dumps(resume, ensure_ascii=False, indent=1))
    print(f"Résultats : {_chemin}")
    return


@app.cell
def _(SMOKE, config, ecart_physique, erreurs, exploitation, retours_lineaire, retours_mlp,
      retours_parfait, retours_ppo):
    _tailles = config["tailles"]
    _pauvre = f"MLP, {_tailles[0]} transitions"
    _e = erreurs.get("MLP, 1000 transitions", erreurs[_pauvre])
    _h = config["horizons_erreur"]
    _N = config["candidats_exploit"]
    verifs = {
        "Physique numpy identique à gymnasium (écart ≤ 1e-9)": ecart_physique <= 1e-9,
        "Modèle parfait, H = 10 : 500 à chaque épisode": bool((retours_parfait == config["pas_max"]).all()),
        "Modèle linéaire (100 transitions au hasard) : retour moyen ≥ 475":
            retours_lineaire.mean() >= 475,
        "MLP, 300 transitions, H = 10 : retour moyen ≥ 450":
            (300, 10) in retours_mlp and retours_mlp[(300, 10)].mean() >= 450,
        "MLP, 1 000 transitions : écart à 40 pas ≥ 10 × écart à 5 pas (accumulation)":
            _e[_h.index(40)] >= 10 * _e[_h.index(5)],
        "MLP, 100 transitions, H = 40 : retour moyen ≤ 300 (horizon trop long pour le modèle)":
            retours_mlp[(_tailles[0], 40)].mean() <= 300,
        "MLP, 3 000 transitions, H = 40 : retour moyen ≥ 450":
            (3000, 40) in retours_mlp and retours_mlp[(3000, 40)].mean() >= 450,
        "Exploitation (MLP, 100 transitions, N = 1 000) : plan retenu surestimé d'au moins 5 points de plus que la moyenne":
            exploitation[(_pauvre, _N[-1])]["retenu"] >= exploitation[(_pauvre, _N[-1])]["moyen"] + 5,
        "Exploitation : surestimation du plan retenu et regret croissants avec N (10 → 1 000)":
            exploitation[(_pauvre, _N[-1])]["retenu"] > exploitation[(_pauvre, _N[0])]["retenu"]
            and exploitation[(_pauvre, _N[-1])]["regret"] > exploitation[(_pauvre, _N[0])]["regret"],
        "PPO après 4 096 pas : retour moyen ≤ 200 (sans modèle, 4 096 pas ne suffisent pas)":
            4096 in retours_ppo and retours_ppo[4096].mean() <= 200,
    }
    for _k, _ok in verifs.items():
        print(("OK    " if _ok else ("--    " if SMOKE else "ÉCHEC ")) + _k
              + ("" if _ok or not SMOKE else " (non vérifié en smoke)"))
    return (verifs,)


@app.cell
def _(SMOKE, verifs):
    _echecs = [k for k, ok in verifs.items() if not ok]
    if not SMOKE and _echecs:
        raise SystemExit("Seuils non atteints : " + " ; ".join(_echecs))
    if SMOKE:
        _msg = (f"**Bilan (smoke)** : le code tourne de bout en bout ; seuils non vérifiés "
                f"({sum(verifs.values())} sur {len(verifs)} déjà atteints).")
    else:
        _msg = f"**Bilan** : {sum(verifs.values())} vérifications sur {len(verifs)} passent."
    mo.md(_msg)
    return


if __name__ == "__main__":
    app.run()
