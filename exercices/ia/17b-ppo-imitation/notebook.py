# Exercice 17b : PPO sur CartPole, écrit à la main puis avec Stable-Baselines3, puis
# imitation de la politique obtenue (clonage comportemental et DAgger) (chapitre 17).
#
#   make ex NUM=17b              → éditeur marimo (interactif)
#   make ex NUM=17b MODE=run     → script complet (vérifie les seuils du README)
#   make ex NUM=17b MODE=smoke   → test rapide (< 30 s)
#
# CPU volontairement (réseaux de quelques milliers de paramètres, environnement pas à pas :
# voir l'exercice 17a).
# Les fonctions « @app.function » sont importables ; les figures du site
# (site/scripts/figures/ch17_*.py) lisent figures/resultats.json, écrit par le run complet.
# Le calcul de l'objectif écrêté (objectif_clip) est le même que dans la démo
# site/src/components/demos/PpoClip.tsx.

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
    from torch.distributions import Categorical

    from ia_commun import PALETTE, is_smoke, sauvegarder_figure, seed_everything, style_commun

    torch.set_num_threads(1)   # petits réseaux : un fil suffit, et le run est reproductible


@app.cell
def _():
    mo.md(r"""
    # Exercice 17b : PPO, puis imiter un expert

    **REINFORCE** (chapitre 11) rend plus probables les actions suivies d'un bon retour.
    Deux défauts : le signal est très bruité, et chaque lot d'expériences ne sert qu'une
    fois (après un pas de gradient, la politique a changé, les données ne viennent plus
    d'elle). **PPO** (Schulman et al., 2017) corrige les deux :

    - un **critique** $V_\phi(s)$ estime la valeur de l'état ; l'**avantage**
      $\hat A_t$ (« mieux ou moins bien que prévu ») remplace le retour brut, calculé par
      GAE (Schulman et al., 2015) ;
    - on réutilise le même lot pendant plusieurs **époques**, en corrigeant par le
      **ratio** $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_\text{old}}(a_t \mid s_t)$,
      et l'on **écrête** ce ratio dans $[1 - \varepsilon, 1 + \varepsilon]$ pour que la
      nouvelle politique ne s'éloigne pas trop de celle qui a produit les données.

    **Ce qui est optimisé** (on minimise l'opposé de l'objectif de PPO) :
    $$
    \mathcal{L}(\theta, \phi) = -\,\hat{\mathbb{E}}_t\Big[\min\big(r_t \hat A_t,\;
    \operatorname{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\,\hat A_t\big)\Big]
    + c_1\,\hat{\mathbb{E}}_t\big[(V_\phi(s_t) - \hat R_t)^2\big]
    - c_2\,\hat{\mathbb{E}}_t\big[\mathcal{H}(\pi_\theta(\cdot \mid s_t))\big]
    $$
    avec $\varepsilon = 0{,}2$, $c_1 = 0{,}5$ (erreur du critique), $c_2 = 0{,}01$
    (bonus d'entropie, qui retarde le moment où la politique devient déterministe).

    Plan :

    1. L'objectif écrêté à la main : l'exemple $r = 1{,}3$ et $r = 0{,}5$.
    2. PPO écrit en PyTorch dans l'esprit de CleanRL, trois graines.
    3. Ablation : **sans écrêtage** ; on mesure de combien la politique bouge à chaque
       mise à jour (divergence KL) et si elle reste bonne une fois le maximum atteint.
    4. Le même problème avec **Stable-Baselines3**.
    5. **Imitation** : on prend la politique PPO comme expert et on apprend à l'imiter,
       par **clonage comportemental** (BC, de l'apprentissage supervisé sur ses
       décisions) puis par **DAgger**, qui demande à l'expert d'étiqueter les états où
       l'élève se perd. Perte : l'entropie croisée entre l'action de l'expert et la
       politique de l'élève.
    """)
    return


@app.cell
def _():
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        n_pas=4_096 if SMOKE else 100_000,       # pas d'environnement par entraînement PPO
        graines=[SEED] if SMOKE else [SEED, SEED + 1, SEED + 2],
        eval_episodes=2 if SMOKE else 10,        # évaluation gloutonne après chaque tranche
        eval_tous=4 if SMOKE else 10,            # … toutes les N mises à jour
        eval_finale=3 if SMOKE else 20,
        sb3_pas=4_096 if SMOKE else 100_000,
        # Imitation : budgets d'étiquettes de l'expert, épisodes d'évaluation par élève
        budgets=[50, 200] if SMOKE else [25, 50, 100, 200, 400, 800],
        repetitions_imitation=2 if SMOKE else 5,
        eval_imitation=3 if SMOKE else 20,
    )
    # Hyperparamètres de PPO : ceux de CleanRL pour CartPole (ppo.py), sauf le pas
    # d'apprentissage (1e-3 au lieu de 2,5e-4) et le nombre d'époques par lot (10 au lieu
    # de 4) : réglage plus agressif, qui résout CartPole en 100 000 pas sur nos trois graines
    # (avec les valeurs de CleanRL, 2 graines sur 3 restaient sous 475 à ce budget).
    hyper = dict(
        n_envs=4, n_etapes=128, lr=1e-3, gamma=0.99, lam=0.95, n_minilots=4, epoques=10,
        eps=0.2, c_valeur=0.5, c_entropie=0.01, norme_max=0.5,
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}\nconfig {config}\nhyper {hyper}")
    return SEED, SMOKE, config, hyper


@app.function
def apres_500(scores):
    """Pire évaluation après la première à 500 (500 si la politique n'y retombe jamais, 0 si jamais atteint)."""
    if 500 not in scores:
        return 0.0
    return float(min(scores[scores.index(500):]))


@app.function
def objectif_clip(r, A, eps=0.2):
    """Un terme de l'objectif de PPO : min(r·A, clip(r, 1−ε, 1+ε)·A).

    Renvoie (valeur, écrêté) ; « écrêté » vaut True quand c'est le terme écrêté qui est
    retenu ET qu'il diffère de r·A : la dérivée par rapport à r est alors nulle.
    Mêmes calculs que la démo PpoClip.tsx.
    """
    r_clip = min(max(r, 1 - eps), 1 + eps)
    brut, ecrete = r * A, r_clip * A
    return min(brut, ecrete), ecrete < brut


@app.cell
def _():
    mo.md(r"""
    ## 1. L'objectif écrêté, à la main

    $\varepsilon = 0{,}2$. Une action avait la probabilité 0,5 sous l'ancienne politique.

    - **Avantage positif** ($\hat A = +1$), la nouvelle politique lui donne 0,65 :
      $r = 1{,}3$. Terme brut $1{,}3$ ; terme écrêté $1{,}2$ ; le minimum retient $1{,}2$,
      une constante : **gradient nul**, on a déjà assez renforcé cette action.
    - **Avantage négatif** ($\hat A = -1$), la nouvelle politique lui donne 0,25 :
      $r = 0{,}5$. Terme brut $-0{,}5$ ; écrêté $0{,}8 \times (-1) = -0{,}8$ ; le minimum
      retient $-0{,}8$ : gradient nul aussi, on a déjà assez puni cette action.
    - Mais si $\hat A = -1$ et $r = 1{,}3$ (l'action a été rendue *plus* probable alors
      qu'elle était mauvaise), le minimum retient le terme brut $-1{,}3$ : le gradient
      reste actif pour corriger l'erreur. L'écrêtage ne bloque que les excès dans le bon sens.
    """)
    return


@app.cell
def _():
    for _r, _A in [(1.3, 1.0), (0.5, -1.0), (1.3, -1.0), (0.5, 1.0), (1.1, 1.0)]:
        _v, _e = objectif_clip(_r, _A)
        print(f"r = {_r:.1f}, A = {_A:+.0f} : objectif {_v:+.2f}  ".replace(".", ",").replace("-", "−")
              + f"({'écrêté, gradient nul' if _e else 'terme brut r·A, gradient actif'})")
    return


@app.function
def couche(n_entree, n_sortie, gain=2 ** 0.5):
    """Couche linéaire initialisée orthogonalement (comme CleanRL)."""
    c = nn.Linear(n_entree, n_sortie)
    nn.init.orthogonal_(c.weight, gain)
    nn.init.zeros_(c.bias)
    return c


@app.function
def creer_agent(n_obs, n_actions, cachee=64):
    """Acteur (politique, logits) et critique (valeur), deux réseaux séparés 64-64 tanh."""
    acteur = nn.Sequential(couche(n_obs, cachee), nn.Tanh(), couche(cachee, cachee), nn.Tanh(),
                           couche(cachee, n_actions, gain=0.01))
    critique = nn.Sequential(couche(n_obs, cachee), nn.Tanh(), couche(cachee, cachee), nn.Tanh(),
                             couche(cachee, 1, gain=1.0))
    return nn.ModuleDict(dict(acteur=acteur, critique=critique))


@app.function
def evaluer(choisir_action, graine, n_episodes, env_id="CartPole-v1"):
    """Retour moyen et écart type de `choisir_action(obs) -> action` (graines graine + 10 000 + k)."""
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
def politique_gloutonne(acteur):
    """Action la plus probable (pas de tirage) : la politique que l'on évalue."""
    def choisir(obs):
        with torch.no_grad():
            return int(acteur(torch.as_tensor(obs, dtype=torch.float32)).argmax())
    return choisir


@app.function
def entrainer_ppo(graine, n_pas, ecretage=True, n_envs=4, n_etapes=128, lr=1e-3, gamma=0.99,
                  lam=0.95, n_minilots=4, epoques=10, eps=0.2, c_valeur=0.5, c_entropie=0.01,
                  norme_max=0.5, eval_tous=10, eval_episodes=10, env_id="CartPole-v1"):
    """PPO à objectif écrêté, dans l'esprit de CleanRL (ppo.py), sans bibliothèque de RL.

    ecretage=False : objectif brut r·A (« no clipping or penalty » du tableau 1 de l'article).
    Renvoie un dictionnaire de courbes, une valeur par mise à jour.
    """
    torch.manual_seed(graine)
    envs = [gym.make(env_id) for _ in range(n_envs)]
    obs = np.stack([e.reset(seed=graine + i)[0] for i, e in enumerate(envs)]).astype(np.float32)
    n_obs, n_act = envs[0].observation_space.shape[0], envs[0].action_space.n
    agent = creer_agent(n_obs, n_act)
    optim = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    taille_lot = n_envs * n_etapes
    taille_minilot = taille_lot // n_minilots
    n_maj = n_pas // taille_lot
    rng = np.random.default_rng(graine)

    res = dict(pas=[], kl=[], frac_ecretee=[], entropie=[], ratio_max=[], ratio_min=[],
               pas_eval=[], score_eval=[], pas_episodes=[], retours=[])
    retours_cours = np.zeros(n_envs)
    pas_total = 0
    t0 = time.perf_counter()
    for maj in range(1, n_maj + 1):
        # Pas d'apprentissage qui décroît linéairement jusqu'à 0 (comme CleanRL)
        for g in optim.param_groups:
            g["lr"] = lr * (1.0 - (maj - 1) / n_maj)

        # --- 1. Collecte : n_etapes pas dans chacun des n_envs environnements ---
        B_obs = np.zeros((n_etapes, n_envs, n_obs), np.float32)
        B_act = np.zeros((n_etapes, n_envs), np.int64)
        B_logp = np.zeros((n_etapes, n_envs), np.float32)
        B_r = np.zeros((n_etapes, n_envs), np.float32)
        B_fin = np.zeros((n_etapes, n_envs), np.float32)
        B_v = np.zeros((n_etapes, n_envs), np.float32)
        for t in range(n_etapes):
            with torch.no_grad():
                o = torch.from_numpy(obs)
                dist = Categorical(logits=agent["acteur"](o))
                a = dist.sample()
                B_logp[t] = dist.log_prob(a).numpy()
                B_v[t] = agent["critique"](o).squeeze(1).numpy()
            B_obs[t], B_act[t] = obs, a.numpy()
            for i, e in enumerate(envs):
                o2, r, termine, tronque, _ = e.step(int(a[i]))
                retours_cours[i] += r
                if tronque and not termine:
                    # Coupure à 500 pas : l'épisode n'est pas « perdu ». On ajoute
                    # γ·V(dernier état) à la récompense (correction utilisée par SB3).
                    with torch.no_grad():
                        r += gamma * float(agent["critique"](torch.as_tensor(o2, dtype=torch.float32)))
                B_r[t, i] = r
                if termine or tronque:
                    B_fin[t, i] = 1.0
                    res["pas_episodes"].append(pas_total + t * n_envs + i + 1)
                    res["retours"].append(float(retours_cours[i]))
                    retours_cours[i] = 0.0
                    o2, _ = e.reset()
                obs[i] = o2
        pas_total += taille_lot

        # --- 2. Avantages par GAE(λ) : somme pondérée des erreurs TD à venir ---
        with torch.no_grad():
            v_fin = agent["critique"](torch.from_numpy(obs)).squeeze(1).numpy()
        avantages = np.zeros_like(B_r)
        gae = np.zeros(n_envs, np.float32)
        for t in reversed(range(n_etapes)):
            v_suiv = v_fin if t == n_etapes - 1 else B_v[t + 1]
            non_fin = 1.0 - B_fin[t]
            delta = B_r[t] + gamma * v_suiv * non_fin - B_v[t]     # erreur TD (chapitre 11)
            gae = delta + gamma * lam * non_fin * gae
            avantages[t] = gae
        cibles_v = avantages + B_v                                  # retours estimés R̂_t

        # --- 3. Optimisation : plusieurs époques sur le même lot, par mini-lots ---
        b_obs = torch.from_numpy(B_obs.reshape(-1, n_obs))
        b_act = torch.from_numpy(B_act.reshape(-1))
        b_logp = torch.from_numpy(B_logp.reshape(-1))
        b_av = torch.from_numpy(avantages.reshape(-1))
        b_cv = torch.from_numpy(cibles_v.reshape(-1))
        kls, fracs, ents, rmax, rmin = [], [], [], [], []
        for _ep in range(epoques):
            ordre = rng.permutation(taille_lot)
            for d in range(0, taille_lot, taille_minilot):
                idx = torch.from_numpy(ordre[d:d + taille_minilot])
                dist = Categorical(logits=agent["acteur"](b_obs[idx]))
                logp = dist.log_prob(b_act[idx])
                log_ratio = logp - b_logp[idx]
                ratio = log_ratio.exp()
                av = b_av[idx]
                av = (av - av.mean()) / (av.std() + 1e-8)          # avantages normalisés
                if ecretage:
                    objectif = torch.min(ratio * av, torch.clamp(ratio, 1 - eps, 1 + eps) * av)
                else:
                    objectif = ratio * av
                perte_pol = -objectif.mean()
                perte_v = ((agent["critique"](b_obs[idx]).squeeze(1) - b_cv[idx]) ** 2).mean()
                entropie = dist.entropy().mean()
                perte = perte_pol + c_valeur * perte_v - c_entropie * entropie
                optim.zero_grad()
                perte.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), norme_max)
                optim.step()
                with torch.no_grad():
                    # KL(π_old ‖ π_θ) estimée sans biais par (r − 1) − log r (estimateur « k3 »)
                    kls.append(float(((ratio - 1) - log_ratio).mean()))
                    fracs.append(float(((ratio - 1).abs() > eps).float().mean()))
                    ents.append(float(entropie))
                    rmax.append(float(ratio.max()))
                    rmin.append(float(ratio.min()))
        res["pas"].append(pas_total)
        # KL et ratios mesurés pendant la dernière époque : l'écart maximal à l'ancienne politique
        n_par_epoque = n_minilots
        res["kl"].append(float(np.mean(kls[-n_par_epoque:])))
        res["frac_ecretee"].append(float(np.mean(fracs[-n_par_epoque:])))
        res["entropie"].append(float(np.mean(ents)))
        res["ratio_max"].append(max(rmax[-n_par_epoque:]))
        res["ratio_min"].append(min(rmin[-n_par_epoque:]))
        if maj % eval_tous == 0 or maj == n_maj:
            m, _e = evaluer(politique_gloutonne(agent["acteur"]), graine + maj, eval_episodes, env_id)
            res["pas_eval"].append(pas_total)
            res["score_eval"].append(m)
    res["duree"] = time.perf_counter() - t0
    for e in envs:
        e.close()
    res["_agent"] = agent            # retiré avant l'export JSON
    return res


@app.cell
def _():
    mo.md(r"""
    ## 2. PPO complet sur trois graines, et 3. sans écrêtage

    Quatre environnements en parallèle, 128 pas chacun : un lot de 512 transitions,
    réutilisé pendant **10 époques** en 4 mini-lots de 128. Le pas d'apprentissage
    ($10^{-3}$) décroît linéairement jusqu'à 0. Après chaque mise à jour on mesure :

    - la **divergence KL** entre l'ancienne et la nouvelle politique sur le lot, estimée
      par la moyenne de $(r - 1) - \log r$ (sans biais quand les actions viennent de
      l'ancienne politique) ;
    - la **fraction écrêtée** : la part des transitions dont $|r - 1| > \varepsilon$ ;
    - le ratio $r$ le plus grand et le plus petit du lot.

    Pour l'ablation, on garde tout et on retire l'écrêtage : l'objectif devient
    $r_t \hat A_t$ (la ligne « No clipping or penalty » du tableau 1 de l'article). Dix
    passes sur les mêmes données sans garde-fou : la politique peut s'éloigner beaucoup
    de celle qui les a produites.
    """)
    return


@app.cell
def _(mo):
    bouton = mo.ui.run_button(label="Entraîner (PPO, ablation, SB3, imitation)")
    bouton
    return (bouton,)


@app.cell
def _(bouton, config, hyper):
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Entraîner » : quelques minutes en mode complet.*"))
    VARIANTES = {"ppo": dict(ecretage=True), "sans-ecretage": dict(ecretage=False)}
    _h = hyper
    resultats, agents = {}, {}
    _t0 = time.perf_counter()
    for _nom, _opts in VARIANTES.items():
        for _g in config["graines"]:
            _r = entrainer_ppo(_g, config["n_pas"], eval_tous=config["eval_tous"],
                               eval_episodes=config["eval_episodes"], **_opts, **_h)
            agents[(_nom, _g)] = _r.pop("_agent")
            _r["score_final"], _r["ecart_final"] = evaluer(
                politique_gloutonne(agents[(_nom, _g)]["acteur"]), _g + 777_000, config["eval_finale"])
            resultats[(_nom, _g)] = _r
            print(f"{_nom:25s} graine {_g} : {_r['duree']:5.1f} s, score final "
                  f"{_r['score_final']:5.1f} ± {_r['ecart_final']:5.1f}, KL médiane "
                  f"{np.median(_r['kl']):.4f} (max {max(_r['kl']):.3f}), ratio max {max(_r['ratio_max']):.2f}")
    duree_maison = time.perf_counter() - _t0
    print(f"Total PPO maison : {duree_maison:.0f} s")
    return VARIANTES, agents, duree_maison, resultats


@app.cell
def _(SMOKE, VARIANTES, config, resultats):
    LIBELLES = {"ppo": "PPO (écrêtage ε = 0,2)", "sans-ecretage": "sans écrêtage"}
    COULEURS = {"ppo": PALETTE[0], "sans-ecretage": PALETTE[3]}
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for _nom in VARIANTES:
        _sc = np.array([resultats[(_nom, _g)]["score_eval"] for _g in config["graines"]])
        _pas = resultats[(_nom, config["graines"][0])]["pas_eval"]
        _a1.plot(_pas, _sc.mean(0), color=COULEURS[_nom], lw=2, label=LIBELLES[_nom])
        _a1.fill_between(_pas, _sc.min(0), _sc.max(0), color=COULEURS[_nom], alpha=0.12)
        _kl = np.array([resultats[(_nom, _g)]["kl"] for _g in config["graines"]])
        _a2.plot(resultats[(_nom, config["graines"][0])]["pas"], np.median(_kl, 0),
                 color=COULEURS[_nom], lw=1.5, label=LIBELLES[_nom])
    _a1.set(xlabel="pas d'environnement", ylabel="retour de la politique gloutonne",
            title="Évaluations (moyenne des graines, bande min–max)")
    _a1.legend(fontsize=8, loc="lower right")
    _a2.set(yscale="log", xlabel="pas d'environnement", ylabel="KL(ancienne ‖ nouvelle)",
            title="Combien la politique bouge à chaque mise à jour")
    _fig.tight_layout()
    sauvegarder_figure(_fig, "ppo-smoke" if SMOKE else "ppo", mo.notebook_dir() / "figures")
    tableau = {
        LIBELLES[_n]: {
            "score final moyen": float(np.mean([resultats[(_n, _g)]["score_final"] for _g in config["graines"]])),
            "scores finaux": [round(resultats[(_n, _g)]["score_final"], 1) for _g in config["graines"]],
            "KL médiane": float(np.median(np.concatenate([resultats[(_n, _g)]["kl"] for _g in config["graines"]]))),
            "KL max": float(max(max(resultats[(_n, _g)]["kl"]) for _g in config["graines"])),
            "ratio max": float(max(max(resultats[(_n, _g)]["ratio_max"]) for _g in config["graines"])),
            "fraction écrêtée médiane": float(np.median(np.concatenate(
                [resultats[(_n, _g)]["frac_ecretee"] for _g in config["graines"]]))),
            # Pire évaluation APRÈS la première à 500 : une politique stable n'y retombe pas loin
            "pire évaluation après un 500": [apres_500(resultats[(_n, _g)]["score_eval"]) for _g in config["graines"]],
            "durée moyenne (s)": float(np.mean([resultats[(_n, _g)]["duree"] for _g in config["graines"]])),
        }
        for _n in VARIANTES
    }
    for _k, _v in tableau.items():
        print(f"{_k:27s} {_v}")
    _fig
    return LIBELLES, tableau


@app.cell
def _():
    mo.md(r"""
    ## 4. La même chose avec Stable-Baselines3

    `PPO("MlpPolicy", env)` avec les hyperparamètres **par défaut** de SB3 (lots de 2 048
    pas, 10 époques, mini-lots de 64, $\varepsilon = 0{,}2$), même budget de pas.
    """)
    return


@app.cell
def _(SEED, config):
    from stable_baselines3 import PPO

    _env = gym.make("CartPole-v1")
    _t = time.perf_counter()
    _m = PPO("MlpPolicy", _env, seed=SEED, verbose=0, device="cpu")
    _m.learn(total_timesteps=config["sb3_pas"])
    _d = time.perf_counter() - _t
    _s = evaluer(lambda o: int(_m.predict(o, deterministic=True)[0]), SEED + 777_000, config["eval_finale"])
    sb3 = dict(duree=_d, score=_s[0], ecart=_s[1])
    print(f"SB3 PPO (défauts) : {_d:.1f} s, score {_s[0]:.1f} ± {_s[1]:.1f}")
    return (sb3,)


@app.cell
def _():
    mo.md(r"""
    ## 5. Imitation : clonage comportemental contre DAgger

    L'**expert** est la politique PPO de la graine 42 (action la plus probable). On ne lui
    demande plus de récompense, seulement « que ferais-tu ici ? ». L'élève est un petit
    réseau 4 → 32 → 2, entraîné par **entropie croisée** sur les actions de l'expert
    (une classification, chapitre 01).

    - **Clonage comportemental (BC)** : les $N$ premiers états de démonstrations
      enregistrées *par l'expert* (des épisodes complets, dans l'ordre), avec ses actions.
      Problème : l'élève fait de petites erreurs, arrive dans des états que l'expert
      ne visite jamais, et n'a rien appris pour en sortir (les erreurs
      s'accumulent).
    - **BC, états tirés au hasard** : les mêmes démonstrations, mais $N$ états tirés au
      hasard dans plusieurs épisodes de l'expert au lieu des $N$ premiers. Ce témoin
      sépare deux effets que la comparaison BC / DAgger mélange : la **diversité** des
      états (25 états consécutifs ne montrent que le début d'un épisode) et le fait
      d'étiqueter les états **où va l'élève**.
    - **DAgger** (Ross, Gordon et Bagnell, 2011) : on commence comme BC avec quelques
      états, puis on **fait jouer l'élève**, on demande à l'expert d'étiqueter les états
      **où l'élève est allé**, on ajoute ces exemples aux précédents et on réentraîne.
      Ici $\beta_i = \mathbb{1}(i = 1)$ : l'expert ne joue qu'au premier tour, comme dans
      la version « sans paramètre » de l'article.

    On compare à **nombre d'étiquettes égal** (l'expert coûte cher : un humain, un
    simulateur lent), et l'on rend la tâche plus dure pour l'élève en l'évaluant avec un
    **bruit d'action** de 5 % (une action sur vingt est remplacée au hasard), comme un
    robot dont les moteurs ne répondent pas toujours.
    """)
    return


@app.function
def entrainer_eleve(X, y, graine, epoques=300, cachee=32):
    """Classifieur état → action, entropie croisée, plein lot (les données sont petites)."""
    torch.manual_seed(graine)
    eleve = nn.Sequential(nn.Linear(X.shape[1], cachee), nn.Tanh(), nn.Linear(cachee, 2))
    optim = torch.optim.Adam(eleve.parameters(), lr=1e-2)
    Xt, yt = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.int64)
    for _ in range(epoques):
        perte = F.cross_entropy(eleve(Xt), yt)
        optim.zero_grad()
        perte.backward()
        optim.step()
    return eleve


@app.function
def jouer(choisir_action, graine, n_episodes, bruit=0.0, max_pas=500, env_id="CartPole-v1"):
    """Joue `n_episodes` ; renvoie (retours, états visités). Bruit : action remplacée au hasard."""
    rng = np.random.default_rng(graine)
    env = gym.make(env_id, max_episode_steps=max_pas)
    retours, etats = [], []
    for k in range(n_episodes):
        obs, _ = env.reset(seed=graine + k)
        fini, total = False, 0.0
        while not fini:
            etats.append(obs)
            a = choisir_action(obs)
            if rng.random() < bruit:
                a = int(rng.integers(2))
            obs, r, termine, tronque, _ = env.step(a)
            total += r
            fini = termine or tronque
        retours.append(total)
    env.close()
    return np.array(retours), np.array(etats, np.float32)


@app.function
def comparer_imitation(expert, budgets, repetitions, n_eval, bruit=0.05, graine=42):
    """BC contre DAgger à nombre d'étiquettes égal. Renvoie {méthode: tableau (répétitions × budgets)}."""
    def etiqueter(etats):
        with torch.no_grad():
            return expert(torch.from_numpy(etats)).argmax(1).numpy()

    # Démonstrations : des épisodes complets joués par l'expert, sans bruit, mis bout à bout.
    # Comme un humain qui enregistre des démonstrations, on garde les trajectoires dans
    # l'ordre : avec n étiquettes, l'élève voit les n premiers états d'une démonstration.
    demos = [jouer(politique_gloutonne(expert), graine + 50_000 + k, 1)[1] for k in range(repetitions + 2)]
    res = {m: np.zeros((repetitions, len(budgets))) for m in ("bc", "bc-hasard", "dagger")}
    for rep in range(repetitions):
        rng = np.random.default_rng(graine + rep)
        rng_hasard = np.random.default_rng(graine + 7_000 + rep)   # tirages du témoin, à part
        flux = np.concatenate(demos[rep:])          # démonstration n° rep, puis les suivantes
        for j, n in enumerate(budgets):
            # BC : les n premiers états des démonstrations, étiquetés par l'expert
            X = flux[:n]
            eleve = entrainer_eleve(X, etiqueter(X), graine + rep)
            res["bc"][rep, j] = jouer(politique_gloutonne(eleve), graine + 90_000 + 100 * rep,
                                      n_eval, bruit)[0].mean()
            # BC, états tirés au hasard dans les mêmes démonstrations (témoin de diversité)
            X = flux[rng_hasard.choice(len(flux), min(n, len(flux)), replace=False)]
            eleve = entrainer_eleve(X, etiqueter(X), graine + rep)
            res["bc-hasard"][rep, j] = jouer(politique_gloutonne(eleve), graine + 90_000 + 100 * rep,
                                             n_eval, bruit)[0].mean()
            # DAgger : n/4 états de démonstration, puis 3 tours de n/4 états visités par l'élève
            part = n // 4
            X = flux[:part]
            Y = etiqueter(X)
            eleve = entrainer_eleve(X, Y, graine + rep)
            for tour in range(3):
                _, visites = jouer(politique_gloutonne(eleve), graine + 60_000 + 1000 * rep + 10 * tour,
                                   20, bruit)
                nouveaux = visites[rng.choice(len(visites), min(part, len(visites)), replace=False)]
                X = np.concatenate([X, nouveaux])
                Y = np.concatenate([Y, etiqueter(nouveaux)])   # l'expert étiquette les états de l'élève
                eleve = entrainer_eleve(X, Y, graine + rep)
            res["dagger"][rep, j] = jouer(politique_gloutonne(eleve), graine + 90_000 + 100 * rep,
                                          n_eval, bruit)[0].mean()
    return res


@app.cell
def _(SEED, agents, config):
    _t0 = time.perf_counter()
    expert = agents[("ppo", SEED)]["acteur"]
    score_expert = jouer(politique_gloutonne(expert), SEED + 90_000, config["eval_imitation"], 0.05)[0].mean()
    imitation = comparer_imitation(expert, config["budgets"], config["repetitions_imitation"],
                                   config["eval_imitation"], graine=SEED)
    duree_imitation = time.perf_counter() - _t0
    print(f"Expert (PPO graine {SEED}), bruit d'action 5 % : {score_expert:.1f}")
    for _m in ("bc", "bc-hasard", "dagger"):
        print(f"{_m:9s}", "  ".join(f"{n} ét. : {v:5.1f}" for n, v in zip(config["budgets"], imitation[_m].mean(0))))
    print(f"Imitation : {duree_imitation:.1f} s")
    return duree_imitation, imitation, score_expert


@app.cell
def _(SMOKE, config, imitation, score_expert):
    _fig, _ax = plt.subplots(figsize=(7, 4))
    for _m, _lib, _c in (("bc", "BC, N premiers états", PALETTE[1]),
                         ("bc-hasard", "BC, N états au hasard", PALETTE[2]), ("dagger", "DAgger", PALETTE[0])):
        _moy = imitation[_m].mean(0)
        _ax.plot(config["budgets"], _moy, color=_c, lw=2, marker="o", label=_lib)
        _ax.fill_between(config["budgets"], imitation[_m].min(0), imitation[_m].max(0), color=_c, alpha=0.12)
    _ax.axhline(score_expert, color="#999", ls=":", lw=1, label="expert (PPO)")
    _ax.set(xscale="log", xlabel="nombre d'états étiquetés par l'expert", ylabel="retour moyen (bruit d'action 5 %)",
            title="Imiter un expert : BC contre DAgger")
    _ax.set_xticks(config["budgets"], [str(b) for b in config["budgets"]])
    _ax.legend(fontsize=9, loc="lower right")
    _fig.tight_layout()
    sauvegarder_figure(_fig, "imitation-smoke" if SMOKE else "imitation", mo.notebook_dir() / "figures")
    _fig
    return


@app.cell
def _(
    SMOKE, VARIANTES, config, duree_imitation, duree_maison, hyper, imitation, resultats, sb3,
    score_expert, tableau,
):
    _export = dict(
        config=config, hyper=hyper, duree_maison=duree_maison, duree_imitation=duree_imitation,
        runs={f"{_n}/{_g}": resultats[(_n, _g)] for _n in VARIANTES for _g in config["graines"]},
        tableau=tableau, sb3=sb3, score_expert=float(score_expert),
        imitation={k: v.tolist() for k, v in imitation.items()},
    )
    _chemin = mo.notebook_dir() / "figures" / ("resultats-smoke.json" if SMOKE else "resultats.json")
    _chemin.write_text(json.dumps(_export, ensure_ascii=False, separators=(",", ":")))
    print(f"Résultats : {_chemin}")
    return


@app.cell
def _(LIBELLES, SMOKE, config, imitation, resultats, sb3, score_expert, tableau):
    _p = tableau[LIBELLES["ppo"]]
    _s = tableau[LIBELLES["sans-ecretage"]]
    _bc, _dg = imitation["bc"].mean(0), imitation["dagger"].mean(0)
    verifs = {
        "Objectif écrêté : r = 1,3, A = +1 → 1,2 ; r = 0,5, A = −1 → −0,8 (gradient nul)":
            objectif_clip(1.3, 1.0) == (1.2, True) and abs(objectif_clip(0.5, -1.0)[0] + 0.8) < 1e-12,
        "PPO : chaque graine a une politique gloutonne finale ≥ 475":
            min(resultats[("ppo", _g)]["score_final"] for _g in config["graines"]) >= 475,
        "Sans écrêtage : KL médiane par mise à jour ≥ 3 × celle de PPO":
            _s["KL médiane"] >= 3 * _p["KL médiane"],
        "PPO : après avoir atteint 500, la pire évaluation reste ≥ 400 pour au moins 2 graines sur 3":
            sum(v >= 400 for v in _p["pire évaluation après un 500"]) >= 2,
        "Sans écrêtage : ratio r maximal ≥ 10 × celui de PPO":
            _s["ratio max"] >= 10 * _p["ratio max"],
        "Sans écrêtage : au moins 2 graines sur 3 retombent sous 200 après avoir atteint 500":
            sum(v < 200 for v in _s["pire évaluation après un 500"]) >= 2,
        "SB3 PPO (défauts) : score ≥ 475": sb3["score"] >= 475,
        "Expert : ≥ 475 même avec 5 % de bruit d'action": score_expert >= 475,
        "Imitation : à chaque budget où BC reste sous 400, DAgger fait au moins 100 de plus":
            all(_dg[j] >= _bc[j] + 100 for j in range(len(_bc)) if _bc[j] < 400)
            and any(_bc < 400),
        "Imitation : avec le plus grand budget, DAgger atteint ≥ 90 % du score de l'expert":
            _dg[-1] >= 0.9 * score_expert,
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
