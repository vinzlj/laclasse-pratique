# Exercice 34 : le RL comme modélisation de séquences (chapitre 34).
# Partie 1 en numpy : l'apprentissage supervisé conditionné par le retour, version tableau,
# et ses deux pièges (la chance, le « recollage » de trajectoires).
# Partie 2 en PyTorch : un petit Decision Transformer sur un jeu hors ligne CartPole,
# conditionné sur différents retours, comparé au clonage comportemental.
#
#   make ex NUM=34              → éditeur marimo (interactif)
#   make ex NUM=34 MODE=run     → script complet (vérifie les seuils du README)
#   make ex NUM=34 MODE=smoke   → test rapide (< 30 s)
#
# Les fonctions « @app.function » sont importables ; les figures du site
# (site/scripts/figures/ch34_*.py) lisent figures/resultats.json, écrit par le run complet.

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
    from ia_commun.mini_gpt import Bloc, ConfigGPT


@app.cell
def _():
    mo.md(r"""
    # Exercice 34 : le Decision Transformer

    Au chapitre 17, le RL hors ligne apprenait une fonction de valeur à partir d'un jeu de
    trajectoires figé (CQL, IQL) : un critique, du *bootstrapping*, un facteur
    d'actualisation $\gamma$, et une instabilité à surveiller. Le **Decision Transformer**
    (Chen et al., 2021) retire tout cela. Chaque trajectoire devient une suite de tokens

    $$
    \hat R_1,\ s_1,\ a_1,\ \hat R_2,\ s_2,\ a_2,\ \dots
    \qquad\text{avec}\qquad \hat R_t = \sum_{t'=t}^{T} r_{t'}
    $$

    où $\hat R_t$ est le **retour restant** (*return-to-go*) : la somme des récompenses
    *qu'il reste à obtenir* à partir de l'instant $t$. Un Transformer causal lit cette suite
    et prédit chaque action $a_t$ à partir de tout ce qui précède, $\hat R_t$ et $s_t$
    compris.

    **Ce qui est optimisé** : l'entropie croisée entre l'action du jeu de données et
    l'action prédite (actions discrètes ; ce serait une erreur quadratique pour des actions
    continues), c'est-à-dire du **clonage comportemental conditionné par le retour** :
    $$
    \mathcal{L}(\theta) = -\,\frac{1}{N}\sum_{t} \log \pi_\theta\big(a_t \mid \hat R_{\le t},\ s_{\le t},\ a_{<t}\big)
    $$

    À l'usage, on **choisit** $\hat R_1$ (« fais-moi un épisode à 300 »), on joue l'action
    prédite, puis on retire la récompense obtenue : $\hat R_{t+1} = \hat R_t - r_t$.

    Plan :

    1. **En numpy, version tableau.** Sans réseau, « conditionner par le retour » veut
       dire compter : parmi les fois où le jeu de données était dans l'état $s$ avec un
       retour restant $\hat R$, quelle action a été jouée ? Deux petits problèmes faits
       à la main montrent les deux limites du chapitre : **la chance** (environnement
       aléatoire) et **le recollage** de morceaux de trajectoires (*stitching*), que la
       programmation dynamique sait faire et le conditionnement par le retour non.
    2. **En PyTorch.** Un jeu hors ligne CartPole produit par des politiques de qualités
       variées (Stable-Baselines3 arrêté à différents moments), un petit Decision
       Transformer, puis la question du chapitre : obtient-on le retour qu'on demande ?
       Comparaison avec le clonage comportemental sur tout le jeu et sur les 10 %
       meilleurs pas (« %BC », introduit dans le même article).
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
        # Partie 1 (numpy)
        n_traj_pari=2_000,                    # épisodes du jeu de données « pari »
        n_eval_pari=10_000,                   # épisodes simulés pour évaluer chaque politique
        # Partie 2 : jeu hors ligne CartPole. Paliers d'entraînement de PPO (SB3) dont on
        # garde la politique (0 = actions au hasard), et nombre d'épisodes joués par chacune.
        paliers=[0, 4_096, 8_192, 12_288, 16_384, 30_720],
        episodes=[20, 20, 15, 10, 6, 2] if SMOKE else [100, 100, 80, 50, 30, 5],
        # Decision Transformer
        contexte=20,                          # K : nombre de pas (donc 3K tokens) vus à la fois
        n_embd=64, n_couches=3, n_tetes=1, dropout=0.1,
        n_pas=60 if SMOKE else 2_500, taille_lot=64, lr=1e-3, echauffement=200,
        echelle_retour=100.0,                 # le retour restant est divisé par 100 en entrée
        # Évaluation : retours demandés, épisodes par retour (mêmes graines pour tous)
        cibles=[50, 300, 500] if SMOKE else [20, 50, 100, 150, 200, 250, 300, 400, 500, 1000],
        n_eval=2 if SMOKE else 10,
        # Clonage comportemental (MLP) : époques sur les transitions
        bc_epoques=2 if SMOKE else 40, bc_pourcentage=10,
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}\nconfig {config}")
    return SEED, SMOKE, config


@app.cell
def _():
    mo.md(r"""
    ## 1. Conditionner par le retour, en comptant

    Sans réseau de neurones, la politique conditionnée par le retour se lit dans un
    tableau. Pour un état $s$ et un retour demandé $\hat R$, on regarde toutes les fois où
    le jeu de données était en $s$ avec un retour restant égal à $\hat R$, et on joue
    l'action la plus fréquente. Si ce retour n'a jamais été vu dans cet état, on prend
    le plus proche qui l'a été (c'est le mieux qu'on puisse faire sans généraliser).

    ### 1.a Le piège de la chance

    Un seul choix, deux actions :

    - **sûr** : rapporte toujours 1 ;
    - **pari** : rapporte 3 avec une probabilité de 0,25, sinon 0 (en moyenne 0,75).

    Le jeu de données vient d'une politique qui tire les deux actions à pile ou face.
    Demander le retour maximal observé, 3, fait choisir le pari : toutes les trajectoires
    à 3 sont des paris *gagnés*. Mais la chance ne se commande pas.

    La parade d'**ESPER** (Paster et al., 2022) : regrouper les trajectoires en paquets
    qui ne disent rien du hasard de l'environnement, et conditionner sur le **retour moyen
    du paquet** plutôt que sur le retour de la trajectoire. Ici, il n'y a qu'une décision :
    regrouper par action suffit (ESPER apprend ces paquets avec un réseau). Le pari gagné
    et le pari perdu reçoivent alors la même étiquette, 0,75.
    """)
    return


@app.function
def retours_restants(recompenses):
    """R̂_t = r_t + r_{t+1} + … + r_T (pas d'actualisation : γ = 1, comme dans le Decision Transformer)."""
    return np.cumsum(np.asarray(recompenses, dtype=float)[::-1])[::-1]


@app.function
def table_rcsl(trajectoires):
    """Politique conditionnée par le retour, version tableau.

    trajectoires : liste de listes de (état, action, récompense).
    Renvoie {état: {retour restant: {action: nombre de fois}}}.
    """
    table = {}
    for traj in trajectoires:
        rtg = retours_restants([r for _s, _a, r in traj])
        for (s, a, _r), R in zip(traj, rtg):
            compte = table.setdefault(s, {}).setdefault(round(float(R), 6), {})
            compte[a] = compte.get(a, 0) + 1
    return table


@app.function
def agir_rcsl(table, etat, retour_demande):
    """Action la plus fréquente dans `etat` parmi les passages au retour restant le plus
    proche de `retour_demande` (égalité : le plus grand). Renvoie (action, retour utilisé)."""
    vus = table[etat]
    R = min(vus, key=lambda v: (abs(v - retour_demande), -v))
    return max(vus[R], key=lambda a: (vus[R][a], a)), R


@app.cell
def _(SEED, config):
    # --- Jeu de données « pari » : la politique de collecte tire sûr ou pari à pile ou face ---
    _rng = np.random.default_rng(SEED)

    def jouer_pari(action, rng):
        if action == "sûr":
            return 1.0
        return 3.0 if rng.random() < 0.25 else 0.0

    _traj = []
    for _ in range(config["n_traj_pari"]):
        _a = "sûr" if _rng.random() < 0.5 else "pari"
        _traj.append([("départ", _a, jouer_pari(_a, _rng))])
    _table = table_rcsl(_traj)
    print("Passages au départ, par retour restant :", _table["départ"])

    def evaluer_pari(action):
        _r = np.random.default_rng(SEED + 1)
        return float(np.mean([jouer_pari(action, _r) for _ in range(config["n_eval_pari"])]))

    pari = {}
    for _demande in (3.0, 1.0):
        _a, _R = agir_rcsl(_table, "départ", _demande)
        pari[f"rcsl-{_demande:g}"] = dict(demande=_demande, action=_a, obtenu=evaluer_pari(_a))
    # ESPER, version tableau : étiquette = retour moyen du paquet (ici, de l'action)
    _moy = {a: np.mean([t[0][2] for t in _traj if t[0][1] == a]) for a in ("sûr", "pari")}
    _traj_esper = [[("départ", t[0][1], float(_moy[t[0][1]]))] for t in _traj]
    _table_e = table_rcsl(_traj_esper)
    _max_e = max(_table_e["départ"])
    _a, _R = agir_rcsl(_table_e, "départ", _max_e)
    pari["esper"] = dict(demande=float(_max_e), action=_a, obtenu=evaluer_pari(_a),
                         etiquettes={a: float(v) for a, v in _moy.items()})
    for _k, _v in pari.items():
        print(f"{_k:9s} : retour demandé {_v['demande']:.2f} → action « {_v['action']} », "
              f"retour moyen obtenu {_v['obtenu']:.3f}")
    return (pari,)


@app.cell
def _():
    mo.md(r"""
    ### 1.b Le piège du recollage (*stitching*)

    Quatre états : **départ**, **carrefour**, **impasse**, et la fin. Le jeu de données
    contient trois sortes de trajectoires, 50 de chaque :

    | Trajectoire | Chemin | Récompenses | Retour |
    |---|---|---|---|
    | A | départ —gauche→ carrefour —attendre→ fin | 0, 0 | 0 |
    | B | départ —droite→ impasse —sortir→ fin | 0, 1 | 1 |
    | C | *commence au carrefour* —foncer→ fin | 10 | 10 |

    Le meilleur épisode possible part à gauche puis fonce : 10. Aucune trajectoire du jeu
    ne le contient en entier ; il faut **recoller** le début de A et la fin de C.

    - La **programmation dynamique** (itération sur les valeurs, chapitre 11) sur les
      transitions du jeu de données propage la valeur 10 du carrefour vers le départ :
      $Q(\text{départ}, \text{gauche}) = 0 + V(\text{carrefour}) = 10$.
    - Le **conditionnement par le retour** ne voit, au départ, que des retours 0 (A) et
      1 (B) : demander 10 renvoie au plus proche, 1, donc à droite.
    """)
    return


@app.function
def iteration_valeurs_hors_ligne(trajectoires, gamma=1.0, n_iter=50):
    """Q-itération sur le modèle empirique (transitions déterministes vues dans le jeu)."""
    trans = {}
    for traj in trajectoires:
        for k, (s, a, r) in enumerate(traj):
            s2 = traj[k + 1][0] if k + 1 < len(traj) else None   # None = fin d'épisode
            trans[(s, a)] = (r, s2)
    Q = {sa: 0.0 for sa in trans}
    for _ in range(n_iter):
        V = {}
        for (s, a), q in Q.items():
            V[s] = max(V.get(s, -np.inf), q)
        Q = {(s, a): r + (gamma * V[s2] if s2 is not None else 0.0) for (s, a), (r, s2) in trans.items()}
    return Q


@app.cell
def _():
    # --- Le petit monde du recollage : transitions déterministes ---
    MONDE = {("départ", "gauche"): (0.0, "carrefour"), ("départ", "droite"): (0.0, "impasse"),
             ("carrefour", "attendre"): (0.0, None), ("carrefour", "foncer"): (10.0, None),
             ("impasse", "sortir"): (1.0, None)}
    _A = [("départ", "gauche", 0.0), ("carrefour", "attendre", 0.0)]
    _B = [("départ", "droite", 0.0), ("impasse", "sortir", 1.0)]
    _C = [("carrefour", "foncer", 10.0)]
    _traj = [_A] * 50 + [_B] * 50 + [_C] * 50

    def derouler(choisir, R_demande=0.0):
        """Joue un épisode depuis « départ » ; choisir(état, retour restant) -> action."""
        s, total, chemin = "départ", 0.0, []
        while s is not None:
            a = choisir(s, R_demande)
            r, s2 = MONDE[(s, a)]
            chemin.append(a)
            total += r
            R_demande -= r                     # R̂_{t+1} = R̂_t − r_t
            s = s2
        return total, chemin

    _table = table_rcsl(_traj)
    recollage = {"retours vus au départ": sorted(_table["départ"])}
    for _demande in (1.0, 10.0):
        _tot, _ch = derouler(lambda s, R: agir_rcsl(_table, s, R)[0], _demande)
        recollage[f"rcsl-{_demande:g}"] = dict(demande=_demande, obtenu=_tot, chemin=_ch)
    _Q = iteration_valeurs_hors_ligne(_traj)
    _tot, _ch = derouler(lambda s, R: max((a for (s2, a) in _Q if s2 == s), key=lambda a: _Q[(s, a)]))
    recollage["q-iteration"] = dict(obtenu=_tot, chemin=_ch,
                                    Q={f"{s}/{a}": v for (s, a), v in _Q.items()})
    for _k, _v in recollage.items():
        print(f"{_k:22s} : {_v}")
    return (recollage,)


@app.cell
def _(SMOKE, pari, recollage):
    # --- Figure de la partie 1 ---
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    _lib = ["demandé 3\n(retour de la trajectoire)", "demandé 1", "ESPER\n(retour moyen du paquet)"]
    _dem = [pari["rcsl-3"]["demande"], pari["rcsl-1"]["demande"], pari["esper"]["demande"]]
    _obt = [pari["rcsl-3"]["obtenu"], pari["rcsl-1"]["obtenu"], pari["esper"]["obtenu"]]
    _x = np.arange(3)
    _a1.bar(_x - 0.2, _dem, 0.4, color=PALETTE[5], label="retour demandé")
    _a1.bar(_x + 0.2, _obt, 0.4, color=PALETTE[0], label="retour moyen obtenu")
    _a1.set_xticks(_x, _lib, fontsize=8)
    _a1.set(title="1.a La chance ne se commande pas", ylabel="retour")
    _a1.legend(fontsize=8)
    _lib2 = ["conditionné\npar le retour (10)", "itération sur\nles valeurs"]
    _a2.bar([0, 1], [recollage["rcsl-10"]["obtenu"], recollage["q-iteration"]["obtenu"]],
            color=[PALETTE[3], PALETTE[2]])
    _a2.axhline(10, color="grey", ls=":", lw=1)
    _a2.set_xticks([0, 1], _lib2, fontsize=8)
    _a2.set(title="1.b Recoller deux trajectoires", ylabel="retour obtenu (max. 10)")
    _fig.tight_layout()
    sauvegarder_figure(_fig, "tableau-smoke" if SMOKE else "tableau", mo.notebook_dir() / "figures")
    _fig
    return


@app.cell
def _():
    mo.md(r"""
    ## 2. Un jeu hors ligne CartPole

    On entraîne PPO (Stable-Baselines3, réglages par défaut, chapitre 17) et l'on garde sa
    politique à plusieurs moments : au hasard (0 pas), puis après 4 096, 8 192, 12 288,
    16 384 et 30 720 pas. Chacune joue des épisodes en **tirant** ses actions (pas
    d'action gloutonne : on veut un comportement varié). Les politiques faibles jouent
    beaucoup d'épisodes, l'experte seulement 5 : comme souvent dans un vrai jeu hors
    ligne, la bonne conduite est rare.

    Une fois le jeu enregistré, on ne touche plus l'environnement pour apprendre : il ne
    sert qu'à **évaluer**.
    """)
    return


@app.function
def generer_jeu_hors_ligne(paliers, episodes, graine):
    """Entraîne PPO (SB3) jusqu'au dernier palier en gardant la politique à chaque palier,
    puis fait jouer `episodes[i]` épisodes à la politique du palier i (0 = au hasard).

    Renvoie une liste d'épisodes : dict(s, a, r, palier), états en float32.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    env = gym.make("CartPole-v1")
    env.action_space.seed(graine)
    modele = PPO("MlpPolicy", env, seed=graine, verbose=0, device="cpu")
    copies = {}

    class Photographe(BaseCallback):
        def _on_step(self):
            if self.num_timesteps in paliers:
                copies[self.num_timesteps] = {k: v.clone() for k, v in self.model.policy.state_dict().items()}
            return True

    modele.learn(max(paliers), callback=Photographe())
    jeu, graine_ep = [], graine * 1_000
    for palier, n in zip(paliers, episodes):
        if palier:
            modele.policy.load_state_dict(copies[palier])
        for _ in range(n):
            obs, _ = env.reset(seed=graine_ep)
            graine_ep += 1
            S, A, R, fini = [], [], [], False
            while not fini:
                a = int(env.action_space.sample()) if palier == 0 else int(modele.predict(obs, deterministic=False)[0])
                S.append(obs)
                A.append(a)
                obs, r, termine, tronque, _ = env.step(a)
                R.append(r)
                fini = termine or tronque
            jeu.append(dict(s=np.array(S, np.float32), a=np.array(A, np.int64),
                            r=np.array(R, np.float32), palier=palier))
    env.close()
    return jeu


@app.cell
def _(mo):
    bouton = mo.ui.run_button(label="Générer le jeu et entraîner (quelques minutes)")
    bouton
    return (bouton,)


@app.cell
def _(SEED, bouton, config):
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Générer le jeu et entraîner ».*"))
    _t0 = time.perf_counter()
    jeu = generer_jeu_hors_ligne(config["paliers"], config["episodes"], SEED)
    duree_jeu = time.perf_counter() - _t0
    retours_jeu = np.array([e["r"].sum() for e in jeu])
    n_pas_jeu = int(sum(len(e["a"]) for e in jeu))
    par_palier = {p: retours_jeu[[e["palier"] == p for e in jeu]] for p in config["paliers"]}
    print(f"Jeu hors ligne : {len(jeu)} épisodes, {n_pas_jeu} pas, en {duree_jeu:.1f} s")
    print(f"Retour moyen {retours_jeu.mean():.1f}, médian {np.median(retours_jeu):.0f}, max {retours_jeu.max():.0f}")
    for _p, _r in par_palier.items():
        print(f"  politique après {_p:6d} pas : {len(_r):3d} épisodes, retour moyen {_r.mean():6.1f} "
              f"(min {_r.min():.0f}, max {_r.max():.0f})")
    return duree_jeu, jeu, n_pas_jeu, par_palier, retours_jeu


@app.cell
def _():
    mo.md(r"""
    ## 3. Le Decision Transformer

    Trois plongements, un par sorte de token : une couche linéaire pour le retour restant
    (divisé par 100), une pour l'état (normalisé), une table pour l'action ; on ajoute à
    chacun un plongement du **pas de temps** $t$ (le même pour les trois tokens d'un pas).
    Puis trois blocs Transformer causaux (ceux du mini-GPT du chapitre 20, 64 dimensions,
    une tête) et une tête linéaire qui lit la sortie **au token de l'état** $s_t$ pour
    prédire $a_t$ : à cette position, le modèle a vu $\hat R_t$ et $s_t$, pas encore $a_t$.

    ```text
    tokens   R̂₁  s₁  a₁  R̂₂  s₂  a₂  …  R̂_K  s_K  a_K
    prédit        ↓            ↓              ↓
                  a₁           a₂             a_K
    ```

    Entraînement : des fenêtres de $K = 20$ pas tirées au hasard dans le jeu (un épisode
    long est tiré plus souvent), perte d'entropie croisée sur les actions. Pas de
    critique, pas de cible bootstrappée, pas de $\gamma$.
    """)
    return


@app.class_definition
class DecisionTransformer(nn.Module):
    """Decision Transformer minimal pour des actions discrètes (Chen et al., 2021)."""

    def __init__(self, dim_etat, n_actions, n_embd=64, n_couches=3, n_tetes=1, dropout=0.1,
                 t_max=500):
        super().__init__()
        cfg = ConfigGPT(n_embd=n_embd, n_tetes=n_tetes, n_couches=n_couches, dropout=dropout, biais=True)
        self.plong_retour = nn.Linear(1, n_embd)
        self.plong_etat = nn.Linear(dim_etat, n_embd)
        self.plong_action = nn.Embedding(n_actions, n_embd)
        self.plong_temps = nn.Embedding(t_max, n_embd)
        self.ln_entree = nn.LayerNorm(n_embd)
        self.blocs = nn.ModuleList([Bloc(cfg) for _ in range(n_couches)])   # attention causale
        self.ln_sortie = nn.LayerNorm(n_embd)
        self.tete = nn.Linear(n_embd, n_actions)

    def forward(self, R, S, A, T):
        """R (B, K) retours restants mis à l'échelle ; S (B, K, d) ; A (B, K) ; T (B, K) pas de temps.
        Renvoie les logits des actions (B, K, n_actions), lus aux tokens d'état."""
        B, K = A.shape
        tps = self.plong_temps(T)
        x = torch.stack([self.plong_retour(R.unsqueeze(-1)) + tps,
                         self.plong_etat(S) + tps,
                         self.plong_action(A) + tps], dim=2)       # (B, K, 3, d)
        x = self.ln_entree(x.reshape(B, 3 * K, -1))                 # R̂₁ s₁ a₁ R̂₂ s₂ a₂ …
        for bloc in self.blocs:
            x = bloc(x)
        return self.tete(self.ln_sortie(x)[:, 1::3])               # positions des tokens s_t


@app.function
def fenetres(jeu, moyenne, ecart, K, taille_lot, echelle, rng):
    """Lot de fenêtres de K pas (complétées à droite par des zéros, masque M).

    Le remplissage à droite est sans effet sur les vrais tokens : l'attention est causale.
    """
    longueurs = np.array([len(e["a"]) for e in jeu])
    idx = rng.choice(len(jeu), taille_lot, p=longueurs / longueurs.sum())
    d_s = jeu[0]["s"].shape[1]
    R = np.zeros((taille_lot, K), np.float32)
    S = np.zeros((taille_lot, K, d_s), np.float32)
    A = np.zeros((taille_lot, K), np.int64)
    T = np.zeros((taille_lot, K), np.int64)
    M = np.zeros((taille_lot, K), np.float32)
    for i, j in enumerate(idx):
        e = jeu[j]
        n = len(e["a"])
        d = int(rng.integers(0, n))
        f = min(d + K, n)
        m = f - d
        R[i, :m] = e["rtg"][d:f] / echelle
        S[i, :m] = (e["s"][d:f] - moyenne) / ecart
        A[i, :m] = e["a"][d:f]
        T[i, :m] = np.arange(d, f)
        M[i, :m] = 1.0
    return [torch.from_numpy(x) for x in (R, S, A, T, M)]


@app.cell
def _(SEED, config, jeu):
    # --- Entraînement : clonage comportemental conditionné par le retour ---
    for _e in jeu:
        _e["rtg"] = retours_restants(_e["r"]).astype(np.float32)
    _tous = np.concatenate([e["s"] for e in jeu])
    moyenne_etat, ecart_etat = _tous.mean(0), _tous.std(0) + 1e-6
    torch.manual_seed(SEED)
    _rng = np.random.default_rng(SEED)
    dt = DecisionTransformer(4, 2, config["n_embd"], config["n_couches"], config["n_tetes"], config["dropout"])
    print(f"Decision Transformer : {sum(p.numel() for p in dt.parameters()):,} paramètres".replace(",", " "))
    _optim = torch.optim.AdamW(dt.parameters(), lr=config["lr"], weight_decay=1e-4)
    pertes_dt = []
    _t0 = time.perf_counter()
    for _it in range(config["n_pas"]):
        for _g in _optim.param_groups:       # échauffement linéaire, puis pas constant
            _g["lr"] = config["lr"] * min(1.0, (_it + 1) / config["echauffement"])
        _R, _S, _A, _T, _M = fenetres(jeu, moyenne_etat, ecart_etat, config["contexte"],
                                      config["taille_lot"], config["echelle_retour"], _rng)
        _logits = dt(_R, _S, _A, _T)
        _ce = F.cross_entropy(_logits.reshape(-1, 2), _A.reshape(-1), reduction="none")
        _perte = (_ce * _M.reshape(-1)).sum() / _M.sum()           # entropie croisée, masque
        _optim.zero_grad()
        _perte.backward()
        nn.utils.clip_grad_norm_(dt.parameters(), 0.25)
        _optim.step()
        pertes_dt.append(_perte.item())
        if _it % 500 == 0:
            print(f"pas {_it:5d} : perte {_perte.item():.3f}  ({time.perf_counter() - _t0:.0f} s)")
    duree_dt = time.perf_counter() - _t0
    print(f"Entraînement : {duree_dt:.0f} s, perte moyenne des 100 derniers pas "
          f"{np.mean(pertes_dt[-100:]):.3f}")
    return duree_dt, dt, ecart_etat, moyenne_etat, pertes_dt


@app.function
@torch.no_grad()
def evaluer_dt(modele, cibles, n_episodes, moyenne, ecart, K, echelle, graine=50_000):
    """Joue n_episodes par retour demandé, tous en parallèle (un lot = un pas de temps).

    Mêmes graines d'environnement pour chaque retour demandé : seules les demandes changent.
    Renvoie un tableau (len(cibles), n_episodes) de retours obtenus.
    """
    modele.eval()
    envs, demandes, obs = [], [], []
    for c in cibles:
        for k in range(n_episodes):
            env = gym.make("CartPole-v1")
            obs.append(env.reset(seed=graine + k)[0])
            envs.append(env)
            demandes.append(c)
    N, T_max = len(envs), 500
    HS = np.zeros((N, T_max, 4), np.float32)
    HR = np.zeros((N, T_max), np.float32)
    HA = np.zeros((N, T_max), np.int64)
    actif = np.ones(N, bool)
    obtenu = np.zeros(N)
    restant = np.array(demandes, np.float32)            # R̂_t, mis à jour après chaque pas
    for t in range(T_max):
        HS[:, t] = (np.stack(obs) - moyenne) / ecart
        HR[:, t] = restant / echelle
        d = max(0, t + 1 - K)                           # on ne garde que les K derniers pas
        logits = modele(torch.from_numpy(HR[:, d:t + 1]), torch.from_numpy(HS[:, d:t + 1]),
                        torch.from_numpy(HA[:, d:t + 1]), torch.arange(d, t + 1).expand(N, -1))
        actions = logits[:, -1].argmax(-1).numpy()      # action la plus probable
        for i in np.flatnonzero(actif):
            HA[i, t] = actions[i]
            obs[i], r, termine, tronque, _ = envs[i].step(int(actions[i]))
            obtenu[i] += r
            restant[i] -= r                             # R̂_{t+1} = R̂_t − r_t
            if termine or tronque:
                actif[i] = False
        if not actif.any():
            break
    for env in envs:
        env.close()
    modele.train()
    return obtenu.reshape(len(cibles), n_episodes)


@app.cell
def _(config, dt, ecart_etat, moyenne_etat):
    _t0 = time.perf_counter()
    obtenus_dt = evaluer_dt(dt, config["cibles"], config["n_eval"], moyenne_etat, ecart_etat,
                            config["contexte"], config["echelle_retour"])
    duree_eval = time.perf_counter() - _t0
    print(f"Évaluation : {duree_eval:.0f} s")
    for _c, _o in zip(config["cibles"], obtenus_dt):
        print(f"retour demandé {_c:5d} → obtenu {_o.mean():6.1f} (min {_o.min():.0f}, max {_o.max():.0f})")
    return duree_eval, obtenus_dt


@app.cell
def _():
    mo.md(r"""
    ## 4. Témoins : clonage comportemental, sur tout le jeu et sur les meilleurs pas

    Même jeu, sans retour en entrée : un MLP 4 → 64 → 64 → 2 apprend $\pi(a \mid s)$ par
    entropie croisée (chapitre 17).

    - **BC sur tout le jeu** : il imite la moyenne des six politiques. On l'évalue de deux
      façons : en **tirant** l'action selon $\pi(a \mid s)$ (imitation fidèle du mélange) et
      en jouant l'action **la plus probable**, comme on le fait pour le Decision Transformer.
    - **10 % BC** : les épisodes sont rangés par retour décroissant et l'on garde les
      meilleurs jusqu'à couvrir 10 % des pas du jeu. C'est le témoin « %BC » de l'article
      du Decision Transformer : il suppose qu'on sait trier, ce qu'on sait faire ici
      puisque chaque épisode porte son retour.
    """)
    return


@app.function
def entrainer_bc(S, A, moyenne, ecart, epoques, graine, taille_lot=256, lr=1e-3):
    """Clonage comportemental : MLP 4 → 64 → 64 → 2, entropie croisée, Adam."""
    torch.manual_seed(graine)
    rng = np.random.default_rng(graine)
    X = torch.from_numpy((S - moyenne) / ecart)
    Y = torch.from_numpy(A)
    pol = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
    optim = torch.optim.Adam(pol.parameters(), lr=lr)
    for _ in range(epoques):
        ordre = rng.permutation(len(Y))
        for d in range(0, len(Y), taille_lot):
            i = torch.from_numpy(ordre[d:d + taille_lot])
            perte = F.cross_entropy(pol(X[i]), Y[i])
            optim.zero_grad()
            perte.backward()
            optim.step()
    return pol


@app.function
@torch.no_grad()
def evaluer_bc(pol, n_episodes, moyenne, ecart, graine=50_000, tirer=False):
    """Retours obtenus (mêmes graines d'environnement que evaluer_dt).

    tirer=False : action la plus probable (comme pour le DT) ;
    tirer=True : action tirée selon π(a | s), l'imitation fidèle de la politique moyenne du jeu.
    """
    gen = torch.Generator().manual_seed(graine)
    retours = []
    for k in range(n_episodes):
        env = gym.make("CartPole-v1")
        o, _ = env.reset(seed=graine + k)
        fini, total = False, 0.0
        while not fini:
            logits = pol(torch.from_numpy(((o - moyenne) / ecart).astype(np.float32)))
            if tirer:
                a = int(torch.multinomial(logits.softmax(-1), 1, generator=gen))
            else:
                a = int(logits.argmax())
            o, r, termine, tronque, _ = env.step(a)
            total += r
            fini = termine or tronque
        retours.append(total)
        env.close()
    return np.array(retours)


@app.cell
def _(SEED, config, ecart_etat, jeu, moyenne_etat, n_pas_jeu):
    _t0 = time.perf_counter()
    _S = np.concatenate([e["s"] for e in jeu])
    _A = np.concatenate([e["a"] for e in jeu])
    _pol = entrainer_bc(_S, _A, moyenne_etat, ecart_etat, config["bc_epoques"], SEED)
    bc = dict(tout=evaluer_bc(_pol, config["n_eval"], moyenne_etat, ecart_etat),
              tout_tire=evaluer_bc(_pol, config["n_eval"], moyenne_etat, ecart_etat, tirer=True))
    # %BC : épisodes triés par retour décroissant, jusqu'à couvrir X % des pas
    _ordre = sorted(range(len(jeu)), key=lambda i: -jeu[i]["r"].sum())
    _garde, _n = [], 0
    for _i in _ordre:
        if _n >= config["bc_pourcentage"] / 100 * n_pas_jeu:
            break
        _garde.append(_i)
        _n += len(jeu[_i]["a"])
    _S = np.concatenate([jeu[i]["s"] for i in _garde])
    _A = np.concatenate([jeu[i]["a"] for i in _garde])
    # Même nombre de pas de gradient qu'avec tout le jeu (le sous-ensemble est 10 fois plus petit)
    _ep = max(1, int(config["bc_epoques"] * n_pas_jeu / len(_A)))
    _pol10 = entrainer_bc(_S, _A, moyenne_etat, ecart_etat, _ep, SEED)
    bc["pourcent"] = evaluer_bc(_pol10, config["n_eval"], moyenne_etat, ecart_etat)
    bc_info = dict(episodes_gardes=len(_garde), pas_gardes=int(len(_A)),
                   retour_min_garde=float(jeu[_garde[-1]]["r"].sum()))
    duree_bc = time.perf_counter() - _t0
    print(f"BC sur tout le jeu, action la plus probable : {bc['tout'].mean():.1f} "
          f"(min {bc['tout'].min():.0f}, max {bc['tout'].max():.0f})")
    print(f"BC sur tout le jeu, action tirée : {bc['tout_tire'].mean():.1f} "
          f"(min {bc['tout_tire'].min():.0f}, max {bc['tout_tire'].max():.0f})")
    print(f"{config['bc_pourcentage']} % BC ({bc_info['episodes_gardes']} épisodes, {bc_info['pas_gardes']} pas, "
          f"retour ≥ {bc_info['retour_min_garde']:.0f}) : {bc['pourcent'].mean():.1f}")
    print(f"Durée des deux BC : {duree_bc:.0f} s")
    return bc, bc_info, duree_bc


@app.cell
def _(SMOKE, bc, config, obtenus_dt, par_palier, pertes_dt, retours_jeu):
    # --- Figures de la partie 2 ---
    _fig, (_a1, _a2, _a3) = plt.subplots(1, 3, figsize=(15, 4))
    _bins = np.linspace(0, 500, 26)
    _a1.hist([par_palier[p] for p in config["paliers"]], bins=_bins, stacked=True,
             color=PALETTE[:len(config["paliers"])],
             label=["au hasard" if p == 0 else f"PPO, {p} pas" for p in config["paliers"]])
    _a1.set(xlabel="retour de l'épisode", ylabel="épisodes", title=f"Jeu hors ligne ({len(retours_jeu)} épisodes)")
    _a1.legend(fontsize=7)
    _a2.plot(np.convolve(pertes_dt, np.ones(50) / 50, mode="valid"))
    _a2.set(xlabel="pas d'entraînement", ylabel="entropie croisée (moyenne glissante)",
            title="Perte du Decision Transformer")
    _c = np.array(config["cibles"])
    _m = obtenus_dt.mean(1)
    _a3.plot([0, 520], [0, 520], color="grey", ls=":", lw=1, label="obtenu = demandé")
    _a3.errorbar(np.minimum(_c, 520), _m, yerr=[_m - obtenus_dt.min(1), obtenus_dt.max(1) - _m],
                 fmt="o-", color=PALETTE[0], capsize=3, label="Decision Transformer")
    _a3.axhline(bc["tout"].mean(), color=PALETTE[3], ls="--", label="BC, tout le jeu (action la plus probable)")
    _a3.axhline(bc["tout_tire"].mean(), color=PALETTE[4], ls=(0, (1, 1)), label="BC, tout le jeu (action tirée)")
    _a3.axhline(bc["pourcent"].mean(), color=PALETTE[2], ls="-.", label=f"{config['bc_pourcentage']} % BC")
    _a3.axhline(retours_jeu.max(), color=PALETTE[1], ls=":", label="meilleur épisode du jeu")
    _a3.set(xlabel="retour demandé (1 000 placé à 520)", ylabel="retour obtenu",
            title="Obtient-on ce qu'on demande ?", xlim=(0, 540), ylim=(0, 540))
    _a3.legend(fontsize=7, loc="upper left")
    _fig.tight_layout()
    sauvegarder_figure(_fig, "cartpole-smoke" if SMOKE else "cartpole", mo.notebook_dir() / "figures")
    _fig
    return


@app.cell
def _(
    SMOKE, bc, bc_info, config, duree_bc, duree_dt, duree_eval, duree_jeu, n_pas_jeu, obtenus_dt,
    pari, par_palier, pertes_dt, recollage, retours_jeu,
):
    resume = dict(
        config=config, pari=pari, recollage=recollage,
        jeu=dict(n_episodes=len(retours_jeu), n_pas=n_pas_jeu, retours=retours_jeu.tolist(),
                 par_palier={str(p): v.tolist() for p, v in par_palier.items()}),
        dt=dict(cibles=config["cibles"], obtenus=obtenus_dt.tolist(),
                pertes=[float(np.mean(pertes_dt[i:i + 50])) for i in range(0, len(pertes_dt), 50)]),
        bc={k: v.tolist() for k, v in bc.items()}, bc_info=bc_info,
        durees=dict(jeu=duree_jeu, dt=duree_dt, evaluation=duree_eval, bc=duree_bc),
    )
    _chemin = mo.notebook_dir() / "figures" / ("resultats-smoke.json" if SMOKE else "resultats.json")
    _chemin.write_text(json.dumps(resume, ensure_ascii=False, separators=(",", ":")))
    print(f"Résultats : {_chemin}")
    return


@app.cell
def _(SMOKE, bc, config, obtenus_dt, pari, recollage):
    _c = np.array(config["cibles"])
    _m = obtenus_dt.mean(1)
    _ok_cibles = [abs(m - c) <= 0.15 * c for c, m in zip(_c, _m) if c <= 400]
    _i500 = config["cibles"].index(500) if 500 in config["cibles"] else -1
    verifs = {
        "Pari : demander 3 fait choisir le pari, retour obtenu ≤ 0,8 (la chance ne se commande pas)":
            pari["rcsl-3"]["action"] == "pari" and pari["rcsl-3"]["obtenu"] <= 0.8,
        "Pari : ESPER (retour moyen du paquet) choisit « sûr » et obtient 1":
            pari["esper"]["action"] == "sûr" and pari["esper"]["obtenu"] == 1.0,
        "Recollage : conditionné par le retour (10 demandé) → 1 ; itération sur les valeurs → 10":
            recollage["rcsl-10"]["obtenu"] == 1.0 and recollage["q-iteration"]["obtenu"] == 10.0,
        "DT : pour chaque retour demandé ≤ 400, retour moyen obtenu à ± 15 % de la demande":
            len(_ok_cibles) > 0 and all(_ok_cibles),
        "DT : retour moyen obtenu croissant avec le retour demandé (jusqu'à 500)":
            all(np.diff(_m[_c <= 500]) > 0),
        "DT : retour demandé 500 → retour moyen ≥ 450": _m[_i500] >= 450,
        "BC sur tout le jeu, action tirée (imitation fidèle du mélange) : retour moyen ≤ 250":
            bc["tout_tire"].mean() <= 250,
        "10 % BC : retour moyen ≥ 450": bc["pourcent"].mean() >= 450,
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
