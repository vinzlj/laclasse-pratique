# Exercice 17a : DQN sur CartPole, écrit à la main puis avec Stable-Baselines3 (chapitre 17).
#
#   make ex NUM=17a              → éditeur marimo (interactif)
#   make ex NUM=17a MODE=run     → script complet (vérifie les seuils du README)
#   make ex NUM=17a MODE=smoke   → test rapide (< 30 s)
#
# CPU volontairement : le réseau a 17 410 paramètres et l'environnement
# avance un pas à la fois ; sur MPS, le coût de transfert de chaque petit lot dépasse le
# calcul. Les algorithmes sont des fonctions « @app.function » : les scripts de figures du
# site (site/scripts/figures/ch17_*.py) lisent les résultats enregistrés par le run complet
# (figures/resultats.json) et ne réentraînent rien.

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
    import torch.nn.functional as F
    from torch import nn

    from ia_commun import PALETTE, is_smoke, sauvegarder_figure, seed_everything, style_commun

    # Un seul fil de calcul : pour des réseaux aussi petits, le parallélisme coûte plus
    # qu'il ne rapporte, et le résultat devient reproductible au bit près sur CPU.
    torch.set_num_threads(1)


@app.cell
def _():
    mo.md(r"""
    # Exercice 17a : DQN, le Q-learning avec un réseau de neurones

    Au chapitre 11, $Q(s, a)$ était une **table** : une case par couple état-action.
    CartPole a un état continu (position et vitesse du chariot, angle et vitesse angulaire
    du bâton) : impossible d'en faire une table. **DQN** (Mnih et al., 2013 puis 2015)
    remplace la table par un réseau $Q_\theta(s, \cdot)$ qui sort une valeur par action,
    et ajoute deux mécanismes pour que l'apprentissage ne diverge pas :

    - une **mémoire de rejeu** (*replay buffer*) : on stocke les transitions
      $(s, a, r, s')$ et on apprend sur des lots tirés au hasard dans le passé, au lieu de
      la dernière transition (très corrélée à la précédente) ;
    - un **réseau cible** $Q_{\bar\theta}$, copie gelée de $Q_\theta$ recopiée tous les
      $C$ pas, qui fabrique la cible : elle ne bouge plus à chaque mise à jour.

    **Ce qui est optimisé.** L'erreur TD du chapitre 11, passée dans la perte de Huber
    (quadratique près de 0, linéaire au-delà de 1, pour ne pas exploser sur les grosses
    erreurs) :
    $$
    \mathcal{L}(\theta) = \operatorname{Huber}\Big(r + \gamma\,(1 - \text{fini})\,
    \max_{a'} Q_{\bar\theta}(s', a') - Q_\theta(s, a)\Big)
    $$
    moyennée sur le lot. La cible (entre parenthèses, sans le dernier terme) ne reçoit pas
    de gradient. **Double DQN** (van Hasselt et al., 2015) choisit $a'$ avec $Q_\theta$
    et l'évalue avec $Q_{\bar\theta}$, pour corriger la surestimation due au $\max$.

    Plan :

    1. DQN complet, écrit en PyTorch dans l'esprit de CleanRL (un fichier, tout visible),
       sur **trois graines** : la même recette ne donne pas la même courbe.
    2. Ablations : **sans réseau cible**, **sans mémoire de rejeu** ; puis **Double DQN**.
    3. Le même problème avec la bibliothèque **Stable-Baselines3**.
    """)
    return


@app.cell
def _():
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        n_pas=3_000 if SMOKE else 80_000,          # pas d'environnement par entraînement
        graines=[SEED] if SMOKE else [SEED, SEED + 1, SEED + 2],
        eval_tous=1_000 if SMOKE else 2_500,       # évaluation gloutonne tous les … pas
        eval_episodes=2 if SMOKE else 10,          # épisodes par évaluation
        eval_finale=3 if SMOKE else 20,            # épisodes pour le score final
        sb3_pas=2_000 if SMOKE else 80_000,
    )
    # Hyperparamètres communs à toutes les variantes (seuls changent cible / rejeu / double)
    hyper = dict(
        lr=1e-3, gamma=0.98, lot=128, memoire=50_000, debut=1_000, maj_cible=500,
        eps_debut=1.0, eps_fin=0.05, eps_duree=5_000, cachee=128, freq=2,
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}\nconfig {config}\nhyper {hyper}")
    return SEED, SMOKE, config, hyper


@app.function
def creer_reseau_q(n_obs, n_actions, cachee=128):
    """Réseau Q : un état en entrée, une valeur Q(s, a) par action en sortie."""
    return nn.Sequential(
        nn.Linear(n_obs, cachee), nn.ReLU(),
        nn.Linear(cachee, cachee), nn.ReLU(),
        nn.Linear(cachee, n_actions),
    )


@app.function
def evaluer(choisir_action, graine, n_episodes, env_id="CartPole-v1"):
    """Retour moyen et écart type de la politique `choisir_action(obs) -> action`, sans exploration.

    Graines d'évaluation séparées de celles de l'entraînement (graine + 10 000 + k).
    """
    env = gym.make(env_id)
    retours = []
    for k in range(n_episodes):
        obs, _ = env.reset(seed=graine + 10_000 + k)
        fini, total = False, 0.0
        while not fini:
            obs, r, termine, tronque, _ = env.step(choisir_action(obs))
            total += r
            fini = termine or tronque
        retours.append(total)
    env.close()
    return float(np.mean(retours)), float(np.std(retours))


@app.function
def politique_gloutonne(reseau):
    """argmax_a Q(s, a) : la politique que l'on évalue (ε = 0)."""
    def choisir(obs):
        with torch.no_grad():
            return int(reseau(torch.as_tensor(obs, dtype=torch.float32)).argmax())
    return choisir


@app.function
def entrainer_dqn(graine, n_pas, cible=True, rejeu=True, double=False, eval_tous=2_500,
                  eval_episodes=10, lr=5e-4, gamma=0.99, lot=64, memoire=50_000, debut=1_000,
                  maj_cible=500, eps_debut=1.0, eps_fin=0.05, eps_duree=10_000, cachee=128, freq=1,
                  env_id="CartPole-v1"):
    """DQN dans l'esprit de CleanRL : une boucle, une mise à jour par pas d'environnement.

    cible=False : la cible est calculée avec le réseau en cours d'apprentissage.
    rejeu=False : la « mémoire » ne garde que les `lot` dernières transitions (consécutives,
                  donc très corrélées) : c'est l'apprentissage en ligne, sans rejeu.
    double=True : Double DQN (action choisie par Q_θ, évaluée par Q_θ̄).
    freq : une mise à jour des poids tous les `freq` pas d'environnement.
    Renvoie un dictionnaire de courbes (listes Python, prêtes pour JSON).
    """
    torch.manual_seed(graine)
    rng = np.random.default_rng(graine)
    env = gym.make(env_id)
    n_obs, n_act = env.observation_space.shape[0], env.action_space.n
    q = creer_reseau_q(n_obs, n_act, cachee)
    q_cible = creer_reseau_q(n_obs, n_act, cachee)
    q_cible.load_state_dict(q.state_dict())
    optim = torch.optim.Adam(q.parameters(), lr=lr)

    # Mémoire de rejeu circulaire en tableaux NumPy
    capacite = memoire if rejeu else lot
    S = np.zeros((capacite, n_obs), np.float32)
    A = np.zeros(capacite, np.int64)
    R = np.zeros(capacite, np.float32)
    S2 = np.zeros((capacite, n_obs), np.float32)
    FIN = np.zeros(capacite, np.float32)
    n_stockes, tete = 0, 0

    res = dict(pas_eval=[], score_eval=[], ecart_eval=[], pas_episodes=[], retours=[],
               pas_q=[], q_moyen=[], pertes=[])
    q_bloc, perte_bloc = [], []
    obs, _ = env.reset(seed=graine)
    retour = 0.0
    t0 = time.perf_counter()
    for pas in range(1, n_pas + 1):
        # ε décroît linéairement de eps_debut à eps_fin sur eps_duree pas
        eps = max(eps_fin, eps_debut - (eps_debut - eps_fin) * pas / eps_duree)
        if rng.random() < eps:
            action = int(rng.integers(n_act))
        else:
            with torch.no_grad():
                action = int(q(torch.as_tensor(obs, dtype=torch.float32)).argmax())
        obs2, r, termine, tronque, _ = env.step(action)
        retour += r
        # On stocke « termine » et non « tronque » : à la coupure des 500 pas, l'épisode
        # n'est pas perdu, la valeur de s' compte encore dans la cible.
        S[tete], A[tete], R[tete], S2[tete], FIN[tete] = obs, action, r, obs2, float(termine)
        tete = (tete + 1) % capacite
        n_stockes = min(n_stockes + 1, capacite)
        obs = obs2
        if termine or tronque:
            res["pas_episodes"].append(pas)
            res["retours"].append(retour)
            obs, _ = env.reset()
            retour = 0.0

        if pas >= debut and n_stockes >= lot and pas % freq == 0:
            if rejeu:
                idx = rng.integers(0, n_stockes, lot)        # lot tiré au hasard dans le passé
            else:
                idx = np.arange(capacite)                    # les `lot` dernières transitions
            s, a = torch.from_numpy(S[idx]), torch.from_numpy(A[idx])
            r_, s2, fin = torch.from_numpy(R[idx]), torch.from_numpy(S2[idx]), torch.from_numpy(FIN[idx])
            reseau_cible = q_cible if cible else q
            with torch.no_grad():                            # la cible ne reçoit pas de gradient
                q_suivant = reseau_cible(s2)
                if double:
                    a_suivant = q(s2).argmax(dim=1, keepdim=True)
                    v_suivant = q_suivant.gather(1, a_suivant).squeeze(1)
                else:
                    v_suivant = q_suivant.max(dim=1).values
                y = r_ + gamma * (1.0 - fin) * v_suivant
            q_sa_tous = q(s)
            q_sa = q_sa_tous.gather(1, a.unsqueeze(1)).squeeze(1)
            perte = F.smooth_l1_loss(q_sa, y)                # perte de Huber
            optim.zero_grad()
            perte.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            optim.step()
            q_bloc.append(float(q_sa_tous.detach().max(dim=1).values.mean()))
            perte_bloc.append(perte.item())
            if cible and pas % maj_cible == 0:
                q_cible.load_state_dict(q.state_dict())     # copie « dure » tous les C pas

        if pas % eval_tous == 0:
            m, e = evaluer(politique_gloutonne(q), graine + pas, eval_episodes, env_id)
            res["pas_eval"].append(pas)
            res["score_eval"].append(m)
            res["ecart_eval"].append(e)
            if q_bloc:
                res["pas_q"].append(pas)
                res["q_moyen"].append(float(np.mean(q_bloc)))
                res["pertes"].append(float(np.mean(perte_bloc)))
            q_bloc, perte_bloc = [], []
    res["duree"] = time.perf_counter() - t0
    env.close()
    res["_reseau"] = q            # retiré avant l'export JSON
    return res


@app.cell
def _():
    mo.md(r"""
    ## 1. DQN complet sur trois graines

    Hyperparamètres (cellule de configuration) : réseau 4 → 128 → 128 → 2, Adam
    $\eta = 10^{-3}$, $\gamma = 0{,}98$, lots de 128, mémoire de 50 000 transitions,
    1 000 pas au hasard avant d'apprendre, une mise à jour tous les 2 pas, réseau cible
    recopié tous les 500 pas, ε de 1 à 0,05 sur les 5 000 premiers pas, 80 000 pas en
    tout. Ces valeurs ont été choisies après quelques essais : avec $\gamma = 0{,}99$ et
    des lots de 64, aucune des graines n'atteignait 500 en 50 000 pas. Un épisode de CartPole rapporte +1 par pas tant que
    le bâton tient, **500 au plus** (l'épisode est alors coupé).

    Toutes les 2 500 étapes, on mesure la politique **gloutonne** (ε = 0) sur 10 épisodes.
    """)
    return


@app.cell
def _(mo):
    bouton = mo.ui.run_button(label="Entraîner (DQN, ablations, SB3)")
    bouton
    return (bouton,)


@app.cell
def _(bouton, config, hyper):
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Entraîner » : une à deux minutes par variante en mode complet.*"))
    VARIANTES = {
        "dqn": dict(cible=True, rejeu=True, double=False),
        "sans-cible": dict(cible=False, rejeu=True, double=False),
        "sans-rejeu": dict(cible=True, rejeu=False, double=False),
        "double": dict(cible=True, rejeu=True, double=True),
    }
    resultats, reseaux = {}, {}
    _t0 = time.perf_counter()
    for _nom, _opts in VARIANTES.items():
        for _g in config["graines"]:
            _r = entrainer_dqn(_g, config["n_pas"], eval_tous=config["eval_tous"],
                               eval_episodes=config["eval_episodes"], **_opts, **hyper)
            reseaux[(_nom, _g)] = _r.pop("_reseau")
            # Score final : politique gloutonne à la fin, sur plus d'épisodes
            _r["score_final"], _r["ecart_final"] = evaluer(
                politique_gloutonne(reseaux[(_nom, _g)]), _g + 777_000, config["eval_finale"])
            resultats[(_nom, _g)] = _r
            print(f"{_nom:11s} graine {_g} : {_r['duree']:5.1f} s, score final "
                  f"{_r['score_final']:5.1f} ± {_r['ecart_final']:4.1f}, meilleure éval "
                  f"{max(_r['score_eval']):5.1f}, Q moyen final {(_r['q_moyen'] or [float('nan')])[-1]:.1f}")
    duree_maison = time.perf_counter() - _t0
    print(f"Total DQN maison : {duree_maison:.0f} s")
    return VARIANTES, duree_maison, resultats


@app.cell
def _(SMOKE, config, hyper, resultats):
    # --- Figure 1 : les trois graines de DQN complet ---
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for _g, _c in zip(config["graines"], PALETTE):
        _r = resultats[("dqn", _g)]
        _a1.plot(_r["pas_episodes"], _r["retours"], color=_c, lw=0.6, alpha=0.5)
        _a1.plot(_r["pas_eval"], _r["score_eval"], color=_c, lw=2, marker="o", ms=3,
                 label=f"graine {_g} (politique gloutonne)")
    _a1.axhline(500, color="#999", ls=":", lw=1)
    _a1.set(xlabel="pas d'environnement", ylabel="retour de l'épisode",
            title="DQN : retours en jouant (fin) et évaluations (points)")
    _a1.legend(fontsize=8, loc="upper left")
    for _g, _c in zip(config["graines"], PALETTE):
        _r = resultats[("dqn", _g)]
        _a2.plot(_r["pas_q"], _r["q_moyen"], color=_c, lw=1.5, label=f"graine {_g}")
    _vmax = (1 - hyper["gamma"] ** 500) / (1 - hyper["gamma"])   # Σ γ^k pour k < 500
    _a2.axhline(_vmax, color="#999", ls=":", lw=1)
    _a2.set(xlabel="pas d'environnement", ylabel="max_a Q(s, a) moyen sur les lots",
            title=f"Valeurs prédites (pointillé : {_vmax:.1f}, le bâton tient 500 pas)".replace(".", ","))
    _fig.tight_layout()
    sauvegarder_figure(_fig, "dqn-graines-smoke" if SMOKE else "dqn-graines", mo.notebook_dir() / "figures")
    _fig
    return


@app.cell
def _():
    mo.md(r"""
    ## 2. Ablations : à quoi servent le réseau cible et la mémoire de rejeu ?

    Même code, mêmes graines, un mécanisme retiré à la fois :

    - **sans réseau cible** : la cible $r + \gamma \max_{a'} Q_\theta(s', a')$ bouge à
      chaque mise à jour, avec les poids que l'on est en train de corriger ;
    - **sans mémoire de rejeu** : on apprend sur les 128 dernières transitions, qui se
      suivent et se ressemblent (un même épisode, une même situation) ;
    - **Double DQN** : on garde tout et on corrige seulement le choix de $a'$ dans la cible.

    C'est la « triade mortelle » du chapitre 11 en vrai : approximation par un réseau,
    *bootstrapping* (la cible utilise $Q$) et apprentissage hors politique (on apprend la
    politique gloutonne en jouant ε-glouton, sur des transitions anciennes).
    """)
    return


@app.cell
def _(SMOKE, VARIANTES, config, resultats):
    # --- Figure 2 : moyenne des évaluations par variante (bande = min et max des graines) ---
    LIBELLES = {"dqn": "DQN complet", "sans-cible": "sans réseau cible",
                "sans-rejeu": "sans mémoire de rejeu", "double": "Double DQN"}
    _fig, _ax = plt.subplots(figsize=(8, 4))
    for _nom, _c in zip(VARIANTES, (PALETTE[0], PALETTE[3], PALETTE[1], PALETTE[2])):
        _scores = np.array([resultats[(_nom, _g)]["score_eval"] for _g in config["graines"]])
        _pas = resultats[(_nom, config["graines"][0])]["pas_eval"]
        _ax.plot(_pas, _scores.mean(0), color=_c, lw=2, label=LIBELLES[_nom])
        _ax.fill_between(_pas, _scores.min(0), _scores.max(0), color=_c, alpha=0.12)
    _ax.set(xlabel="pas d'environnement", ylabel="retour de la politique gloutonne",
            title=f"Ablations ({len(config['graines'])} graines : moyenne, bande min–max)")
    _ax.legend(fontsize=9, loc="upper left")
    _fig.tight_layout()
    sauvegarder_figure(_fig, "ablations-smoke" if SMOKE else "ablations", mo.notebook_dir() / "figures")
    tableau = {
        LIBELLES[_n]: {
            "score final moyen": float(np.mean([resultats[(_n, _g)]["score_final"] for _g in config["graines"]])),
            "scores finaux": [round(resultats[(_n, _g)]["score_final"], 1) for _g in config["graines"]],
            "moyenne des 4 dernières évaluations": float(np.mean(
                [np.mean(resultats[(_n, _g)]["score_eval"][-4:]) for _g in config["graines"]])),
            "moyenne de toutes les évaluations": float(np.mean(
                [resultats[(_n, _g)]["score_eval"] for _g in config["graines"]])),
            "évaluations à 500 (par graine)": [resultats[(_n, _g)]["score_eval"].count(500.0)
                                               for _g in config["graines"]],
            "Q moyen final (par graine)": [float(f"{resultats[(_n, _g)]['q_moyen'][-1]:.4g}")
                                           if resultats[(_n, _g)]["q_moyen"] else None
                                           for _g in config["graines"]],
            "durée moyenne (s)": float(np.mean([resultats[(_n, _g)]["duree"] for _g in config["graines"]])),
        }
        for _n in VARIANTES
    }
    for _k, _v in tableau.items():
        print(f"{_k:22s} {_v}")
    _fig
    return LIBELLES, tableau


@app.cell
def _():
    mo.md(r"""
    ## 3. La même chose avec Stable-Baselines3

    Stable-Baselines3 (SB3) emballe tout ce qui précède : `DQN("MlpPolicy", env)`. On garde
    ses **hyperparamètres par défaut** (réglés pour Atari) en ne changeant que le budget,
    puis on lui donne nos hyperparamètres : la bibliothèque n'est pas magique, ce sont les
    réglages qui comptent.
    """)
    return


@app.cell
def _(SEED, config, hyper):
    from stable_baselines3 import DQN

    def _sb3(**kw):
        _env = gym.make("CartPole-v1")
        _t = time.perf_counter()
        _m = DQN("MlpPolicy", _env, seed=SEED, verbose=0, device="cpu", **kw)
        _m.learn(total_timesteps=config["sb3_pas"])
        _d = time.perf_counter() - _t
        _s = evaluer(lambda o: int(_m.predict(o, deterministic=True)[0]), SEED + 777_000,
                     config["eval_finale"])
        return _d, _s

    sb3 = {}
    sb3["défauts SB3"] = _sb3()
    sb3["nos hyperparamètres"] = _sb3(
        learning_rate=hyper["lr"], gamma=hyper["gamma"], batch_size=hyper["lot"],
        buffer_size=hyper["memoire"], learning_starts=hyper["debut"],
        target_update_interval=hyper["maj_cible"], train_freq=hyper["freq"], gradient_steps=1,
        exploration_initial_eps=hyper["eps_debut"], exploration_final_eps=hyper["eps_fin"],
        exploration_fraction=hyper["eps_duree"] / config["sb3_pas"],
        policy_kwargs=dict(net_arch=[hyper["cachee"], hyper["cachee"]]),
        max_grad_norm=10.0,
    )
    for _k, (_d, (_m, _e)) in sb3.items():
        print(f"SB3 DQN, {_k:20s} : {_d:5.1f} s, score {_m:.1f} ± {_e:.1f}")
    return (sb3,)


@app.cell
def _(SMOKE, VARIANTES, config, duree_maison, hyper, resultats, sb3, tableau):
    # --- Export des résultats pour les figures du site (le run complet seulement) ---
    _export = dict(
        config=config, hyper=hyper, duree_maison=duree_maison,
        runs={f"{_n}/{_g}": resultats[(_n, _g)] for _n in VARIANTES for _g in config["graines"]},
        tableau=tableau,
        sb3={_k: dict(duree=_d, score=_s[0], ecart=_s[1]) for _k, (_d, _s) in sb3.items()},
    )
    _chemin = mo.notebook_dir() / "figures" / ("resultats-smoke.json" if SMOKE else "resultats.json")
    _chemin.write_text(json.dumps(_export, ensure_ascii=False, separators=(",", ":")))
    print(f"Résultats : {_chemin}")
    return


@app.cell
def _(LIBELLES, SMOKE, config, resultats, sb3, tableau):
    # --- Bilan : seuils du README. En smoke, on vérifie seulement que tout tourne.
    _dqn = tableau[LIBELLES["dqn"]]
    _sc = tableau[LIBELLES["sans-cible"]]
    _sr = tableau[LIBELLES["sans-rejeu"]]
    _db = tableau[LIBELLES["double"]]
    _max_dqn = [max(resultats[("dqn", _g)]["score_eval"]) for _g in config["graines"]]
    verifs = {
        "DQN complet : chaque graine atteint 500 à au moins une évaluation":
            min(_max_dqn) >= 500,
        "DQN complet : score final moyen ≥ 400 (3 graines, 20 épisodes chacune)":
            _dqn["score final moyen"] >= 400,
        "Sans réseau cible : moyenne des évaluations < la moitié de celle de DQN, Q moyen final > 1 000":
            _sc["moyenne de toutes les évaluations"] < 0.5 * _dqn["moyenne de toutes les évaluations"]
            and min(_sc["Q moyen final (par graine)"]) > 1000,
        "Sans mémoire de rejeu : score final moyen ≤ 60 % de celui de DQN":
            _sr["score final moyen"] <= 0.6 * _dqn["score final moyen"],
        "Sans mémoire de rejeu : moins d'évaluations à 500 que DQN":
            sum(_sr["évaluations à 500 (par graine)"]) < sum(_dqn["évaluations à 500 (par graine)"]),
        "Double DQN : chaque graine atteint 500 à au moins une évaluation":
            min(max(resultats[("double", _g)]["score_eval"]) for _g in config["graines"]) >= 500,
    }
    for _k, _ok in verifs.items():
        if _ok:
            print("OK    " + _k)
        elif SMOKE:
            print("--    " + _k + " (non vérifié en smoke)")
        else:
            print("ÉCHEC " + _k)
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
