# Exercice 36 : des actions en tokens (chapitre 36, agents généralistes et robotique).
# Partie 1 en numpy : le pendule (Pendulum-v1) réécrit en numpy, un « expert » sans
# apprentissage (pompage d'énergie puis LQR), et la discrétisation des actions en bins
# à la RT-1 / RT-2 : combien de bins faut-il pour ne rien perdre ?
# Partie 2 en PyTorch : le même petit Transformer causal, entraîné par clonage
# comportemental de deux façons : action en token (256 bins, entropie croisée) ou action
# continue (régression, erreur quadratique). Au départ « au repos en bas », l'expert
# pousse à gauche ou à droite au hasard : la régression moyenne les deux et reste figée.
#
#   make ex NUM=36              → éditeur marimo (interactif)
#   make ex NUM=36 MODE=run     → script complet (vérifie les seuils du README)
#   make ex NUM=36 MODE=smoke   → test rapide (< 30 s)
#
# Les fonctions « @app.function » sont importables ; les figures du site
# (site/scripts/figures/ch36_*.py) lisent figures/resultats.json, écrit par le run complet.

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

    # Constantes de Pendulum-v1 (gymnasium) : g = 10, m = l = 1, dt = 0,05 s
    DT, COUPLE_MAX, VITESSE_MAX = 0.05, 2.0, 8.0


@app.cell
def _():
    mo.md(r"""
    # Exercice 36 : des actions en tokens

    Un modèle de langage ne sait produire que des **tokens**. RT-1 (2022) puis RT-2 (2023)
    en ont tiré une recette simple pour piloter un robot : découper chaque dimension
    continue de l'action (déplacement de la pince, rotation, ouverture…) en **256 intervalles
    égaux** (*bins*), et faire prédire le numéro de l'intervalle comme on prédit un mot.
    L'action devient une courte phrase de nombres, et l'entraînement redevient un
    **clonage comportemental** (chapitre 17) avec la perte du modèle de langage :

    $$
    \mathcal{L}(\theta) = -\frac{1}{N}\sum_t \log p_\theta\big(\mathrm{bin}(a_t)\mid o_{\le t},\ a_{<t}\big)
    $$

    (entropie croisée sur le token d'action). L'autre choix naturel est de **régresser**
    l'action continue avec une erreur quadratique,
    $\mathcal{L}(\theta) = \frac{1}{N}\sum_t \big(\hat a_\theta(o_{\le t}, a_{<t}) - a_t\big)^2$.

    Ici, pas de robot : le pendule de gymnasium (`Pendulum-v1`), une tige qu'un moteur à
    couple limité ($|u| \le 2$) doit amener à la verticale puis y maintenir. Le couple
    maximal ne suffit pas à la lever d'un coup : il faut la balancer pour prendre de l'élan.

    Plan :

    1. **En numpy.** Le pendule réécrit en quelques lignes (et vérifié contre gymnasium), un
       expert **sans apprentissage** (pomper de l'énergie, puis un régulateur LQR près de la
       verticale), puis la question : si l'on arrondit ses couples à $n$ bins, à partir de
       combien de bins ne perd-on plus rien ?
    2. **En PyTorch.** Un petit Transformer causal lit les 4 dernières observations et
       prédit l'action courante. Même réseau, mêmes données, deux têtes : 256 tokens
       (entropie croisée) ou un nombre (erreur quadratique). Le piège : en bas, immobile,
       l'expert part à gauche ou à droite **au hasard**. La régression apprend un compromis
       entre $+2$ et $-2$, un couple que l'expert ne joue jamais. Et un témoin : les mêmes
       modèles entraînés sur des démonstrations sans bruit.
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
        # Partie 1 : nombres de bins essayés pour arrondir les couples de l'expert
        bins_testes=[2, 3, 5, 9, 17, 33, 256],
        n_bins=256,                        # comme RT-1 et RT-2
        # Données de démonstration : la moitié des départs au hasard (comme gymnasium),
        # l'autre moitié au repos en bas (où l'expert choisit un côté au hasard)
        n_demo_hasard=12 if SMOKE else 150,
        n_demo_repos=12 if SMOKE else 150,
        T=200,                             # longueur d'un épisode de Pendulum-v1
        bruit_demo=0.5,                    # écart type du bruit ajouté au couple exécuté (pas à l'étiquette)
        temoin_sans_bruit=True,            # entraîne aussi les deux têtes sur des démonstrations sans bruit
        # Transformer
        contexte=4,                        # K : nombre de pas vus à la fois
        n_embd=64, n_couches=2, n_tetes=2, dropout=0.0,
        action_precedente=False,           # True : chaque pas reçoit aussi l'action précédente (voir README)
        n_pas=40 if SMOKE else 3_000, taille_lot=256, lr=1e-3, echauffement=100,
        # Évaluation (mêmes départs pour tous les modèles)
        n_eval_hasard=6 if SMOKE else 50,
        n_eval_repos=6 if SMOKE else 30,
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}\nconfig {config}")
    return SEED, SMOKE, config


@app.cell
def _():
    mo.md(r"""
    ## 1. Le pendule, l'expert et les bins

    L'angle $\theta$ vaut 0 à la verticale (en haut) et $\pm\pi$ en bas. Pendulum-v1 avance
    par pas de $\Delta t = 0{,}05$ s :

    $$
    \dot\theta \leftarrow \operatorname{clip}\big(\dot\theta + (15\sin\theta + 3u)\,\Delta t,\ \pm 8\big),
    \qquad \theta \leftarrow \theta + \dot\theta\,\Delta t
    $$

    et la récompense de chaque pas vaut $-(\theta^2 + 0{,}1\,\dot\theta^2 + 0{,}001\,u^2)$
    ($\theta$ ramené dans $[-\pi, \pi]$) : 0 au mieux, environ $-16$ par pas tout en bas.

    **L'expert** n'apprend rien, il calcule :

    - **loin du haut**, il pompe de l'énergie : l'énergie $E = \tfrac12\dot\theta^2 + 15\cos\theta$
      vaut 15 à la verticale au repos ; on pousse à fond ($\pm 2$) dans le sens du mouvement
      tant que $E < 15$, à contre-sens sinon ($\dot E = 3u\dot\theta$) ;
    - **au repos en bas**, le sens du mouvement n'existe pas : il choisit un côté **à pile
      ou face** au début de l'épisode ;
    - **près du haut** ($|\theta| < 0{,}5$, $|\dot\theta| < 3$), un régulateur **LQR** :
      $u = -K x$ avec $x = (\theta, \dot\theta)$ et $K$ calculé une fois pour toutes sur le
      modèle linéarisé ($\sin\theta \approx \theta$) en itérant l'équation de Riccati. C'est
      la commande optimale « classique » des automaticiens, sans aucune donnée.
    """)
    return


@app.function
def normaliser(angle):
    """Ramène un angle dans [-π, π[."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


@app.function
def pas_pendule(th, thd, u):
    """Un pas de Pendulum-v1, vectorisé (th, thd, u : tableaux de même forme).

    Renvoie le nouvel état et la récompense, exactement comme gymnasium."""
    u = np.clip(u, -COUPLE_MAX, COUPLE_MAX)
    recompense = -(normaliser(th) ** 2 + 0.1 * thd**2 + 0.001 * u**2)
    thd = np.clip(thd + (15.0 * np.sin(th) + 3.0 * u) * DT, -VITESSE_MAX, VITESSE_MAX)
    return th + thd * DT, thd, recompense


@app.function
def gain_lqr(q=(1.0, 0.1), r=0.001, n_iter=2_000):
    """Gain LQR du pendule linéarisé autour du haut (sin θ ≈ θ), en temps discret.

    Mêmes poids que la récompense de Pendulum-v1 : Q = diag(1 ; 0,1), R = 0,001."""
    A = np.array([[1 + 15 * DT**2, DT], [15 * DT, 1.0]])   # (θ, θ̇) → (θ, θ̇) au pas suivant
    B = np.array([[3 * DT**2], [3 * DT]])
    Q, R = np.diag(q), np.array([[r]])
    P = Q.copy()
    for _ in range(n_iter):                                  # itération de Riccati
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P = Q + A.T @ P @ (A - B @ K)
    return K[0]


@app.function
def expert(th, thd, cote, K):
    """Couple de l'expert (vectorisé). `cote` (±1) : côté tiré à pile ou face par épisode."""
    th = normaliser(th)
    energie = 0.5 * thd**2 + 15.0 * np.cos(th)
    pompe = COUPLE_MAX * np.sign(thd * (15.0 - energie))      # pomper (ou freiner) l'énergie
    au_repos = (np.abs(thd) < 0.05) & (np.abs(th) > np.pi - 0.3)
    pompe = np.where(au_repos, COUPLE_MAX * cote, pompe)       # en bas, immobile : pile ou face
    lqr = -(K[0] * th + K[1] * thd)
    pres_du_haut = (np.abs(th) < 0.5) & (np.abs(thd) < 3.0)
    return np.clip(np.where(pres_du_haut, lqr, pompe), -COUPLE_MAX, COUPLE_MAX)


@app.function
def departs(n_hasard, n_repos, rng):
    """Départs : au hasard comme gymnasium (θ ~ U(−π, π), θ̇ ~ U(−1, 1)), ou au repos en bas
    (θ = π ± 0,01, θ̇ = 0). Renvoie th, thd et un booléen « au repos »."""
    th = np.concatenate([rng.uniform(-np.pi, np.pi, n_hasard), np.pi + rng.uniform(-0.01, 0.01, n_repos)])
    thd = np.concatenate([rng.uniform(-1, 1, n_hasard), np.zeros(n_repos)])
    return th, thd, np.r_[np.zeros(n_hasard, bool), np.ones(n_repos, bool)]


@app.function
def vers_bins(u, n_bins):
    """Couple continu dans [-2, 2] → numéro de bin dans {0, …, n_bins − 1} (bins égaux)."""
    x = (np.asarray(u) + COUPLE_MAX) / (2 * COUPLE_MAX) * n_bins
    return np.clip(np.floor(x), 0, n_bins - 1).astype(np.int64)


@app.function
def depuis_bins(b, n_bins):
    """Numéro de bin → couple au centre du bin."""
    return -COUPLE_MAX + (np.asarray(b) + 0.5) * (2 * COUPLE_MAX / n_bins)


@app.function
def derouler(politique, th, thd, T):
    """Joue T pas en parallèle. `politique(t, th, thd)` renvoie le couple.
    Renvoie (retours, angles (N, T+1), couples (N, T))."""
    N = len(th)
    angles, couples, retours = np.zeros((N, T + 1)), np.zeros((N, T)), np.zeros(N)
    angles[:, 0] = th
    for t in range(T):
        u = politique(t, th, thd)
        th, thd, r = pas_pendule(th, thd, u)
        retours += r
        angles[:, t + 1], couples[:, t] = th, u
    return retours, angles, couples


@app.function
def temps_de_montee(angles):
    """Temps (s) avant que le pendule ne passe pour la première fois au-dessus de l'horizontale
    (|θ| < π/2) ; T·Δt s'il n'y passe jamais."""
    dessus = np.abs(normaliser(angles)) < np.pi / 2
    return np.where(dessus.any(1), dessus.argmax(1), angles.shape[1] - 1) * DT


@app.function
def redresse(angles, n_fin=20, tol=0.2):
    """Vrai si le pendule est resté à moins de `tol` rad de la verticale sur les n_fin derniers pas."""
    return np.all(np.abs(normaliser(angles[:, -n_fin:])) < tol, axis=1)


@app.cell
def _(SEED, config):
    # --- Vérification : notre pendule numpy = Pendulum-v1 de gymnasium ---
    _env = gym.make("Pendulum-v1")
    _obs, _ = _env.reset(seed=SEED)
    _th, _thd = _env.unwrapped.state
    _rng = np.random.default_rng(SEED)
    _ecart = 0.0
    for _t in range(200):
        _u = _rng.uniform(-2, 2)
        _obs, _r, *_ = _env.step(np.array([_u], np.float32))
        _th, _thd, _r_np = pas_pendule(_th, _thd, np.float32(_u))
        _ecart = max(_ecart, abs(_r - _r_np), *np.abs(np.array(_env.unwrapped.state) - (_th, _thd)))
    _env.close()
    assert _ecart < 1e-4, f"le pendule numpy s'écarte de gymnasium ({_ecart:.2e})"
    print(f"Pendule numpy contre gymnasium, 200 pas au hasard : écart max {_ecart:.1e}")

    K_LQR = gain_lqr()
    print(f"Gain LQR : u = −({K_LQR[0]:.2f} θ + {K_LQR[1]:.2f} θ̇)")

    # Départs d'évaluation : mêmes pour l'expert, les versions arrondies et les modèles
    eval_th, eval_thd, eval_repos = departs(config["n_eval_hasard"], config["n_eval_repos"],
                                            np.random.default_rng(50_000))
    eval_cote = np.random.default_rng(50_001).choice([-1.0, 1.0], len(eval_th))
    return K_LQR, eval_cote, eval_repos, eval_th, eval_thd


@app.cell
def _(K_LQR, config, eval_cote, eval_repos, eval_th, eval_thd):
    # --- L'expert, puis l'expert dont on arrondit le couple à n bins ---
    def _bilan(retours, angles):
        return dict(retour_hasard=float(retours[~eval_repos].mean()),
                    retour_repos=float(retours[eval_repos].mean()),
                    redresse_hasard=float(redresse(angles)[~eval_repos].mean()),
                    redresse_repos=float(redresse(angles)[eval_repos].mean()),
                    montee_repos=float(temps_de_montee(angles)[eval_repos].mean()))

    _r, angles_expert, couples_expert = derouler(
        lambda t, th, thd: expert(th, thd, eval_cote, K_LQR), eval_th, eval_thd, config["T"])
    ref_expert = _bilan(_r, angles_expert)
    quantif = {}
    for _n in config["bins_testes"]:
        _r, _a, _ = derouler(lambda t, th, thd, n=_n: depuis_bins(vers_bins(expert(th, thd, eval_cote, K_LQR), n), n),
                             eval_th, eval_thd, config["T"])
        quantif[_n] = _bilan(_r, _a) | dict(largeur=4.0 / _n, erreur_max=2.0 / _n)
    print(f"Expert continu : retour moyen {ref_expert['retour_hasard']:.1f} (départs au hasard), "
          f"{ref_expert['retour_repos']:.1f} (au repos) ; redressé {ref_expert['redresse_hasard']:.0%} "
          f"et {ref_expert['redresse_repos']:.0%} ; au repos, au-dessus de l'horizontale après "
          f"{ref_expert['montee_repos']:.2f} s")
    print(" bins | largeur | erreur max | retour (hasard) | retour (repos) | redressé (hasard / repos)")
    for _n, _q in quantif.items():
        print(f" {_n:4d} | {_q['largeur']:7.4f} | {_q['erreur_max']:10.4f} | {_q['retour_hasard']:15.1f} | "
              f"{_q['retour_repos']:14.1f} | {_q['redresse_hasard']:.0%} / {_q['redresse_repos']:.0%}")
    # Un exemple de « phrase d'action » à la RT-2 : les numéros de bins, séparés par des espaces
    _u = couples_expert[0, :8]
    exemple_tokens = dict(couples=_u.round(3).tolist(), bins=vers_bins(_u, 256).tolist(),
                          decodes=depuis_bins(vers_bins(_u, 256), 256).round(3).tolist())
    print("Couples de l'expert :", " ".join(f"{x:+.3f}" for x in _u))
    print("En tokens (256 bins) :", " ".join(map(str, exemple_tokens["bins"])))
    return angles_expert, couples_expert, exemple_tokens, quantif, ref_expert


@app.cell
def _(SMOKE, angles_expert, config, couples_expert, eval_repos, quantif):
    # --- Figure de la partie 1 : un épisode de l'expert, ses couples arrondis, et l'effet des bins ---
    _fig, (_a1, _a2, _a3) = plt.subplots(1, 3, figsize=(15, 4))
    _i = int(np.flatnonzero(eval_repos)[0])
    _t = np.arange(config["T"]) * DT
    _a1.plot(np.arange(config["T"] + 1) * DT, normaliser(angles_expert[_i]), color=PALETTE[0])
    _a1.set(xlabel="temps (s)", ylabel="θ (rad), 0 = en haut", title="L'expert, départ au repos en bas")
    _a2.plot(_t, couples_expert[_i], color="grey", lw=2, label="continu")
    for _n, _c in ((5, PALETTE[3]), (256, PALETTE[2])):
        _a2.step(_t, depuis_bins(vers_bins(couples_expert[_i], _n), _n), where="post", color=_c, lw=1,
                 label=f"{_n} bins")
    _a2.set(xlabel="temps (s)", ylabel="couple u", title="Couples de l'expert, arrondis", xlim=(2.5, 6))
    _a2.legend(fontsize=8)
    _n = list(quantif)
    _a3.semilogx(_n, [quantif[n]["retour_hasard"] for n in _n], "o-", base=2, label="départs au hasard")
    _a3.semilogx(_n, [quantif[n]["retour_repos"] for n in _n], "s-", base=2, label="départs au repos")
    _a3.set(xlabel="nombre de bins", ylabel="retour moyen", title="Arrondir le couple de l'expert")
    _a3.legend(fontsize=8)
    _fig.tight_layout()
    sauvegarder_figure(_fig, "bins-smoke" if SMOKE else "bins", mo.notebook_dir() / "figures")
    _fig
    return


@app.cell
def _():
    mo.md(r"""
    ## 2. Cloner l'expert : un token ou un nombre ?

    **Les démonstrations.** L'expert joue des épisodes de 200 pas, la moitié depuis un
    départ au hasard, l'autre moitié depuis le repos en bas. On garde l'observation de
    gymnasium $o_t = (\cos\theta, \sin\theta, \dot\theta / 8)$ et le couple $a_t$ de
    l'expert. Le couple **exécuté**, lui, reçoit un bruit gaussien (écart type 0,5) : les
    démonstrations passent ainsi par des états un peu à côté de la trajectoire idéale et
    montrent comment en revenir (DART). Sans ce bruit, la moindre erreur du modèle l'emmène
    dans des états jamais vus : c'est le témoin de la fin.

    **Le modèle** (`PolitiqueTransformer`) : pour chacun des $K = 4$ derniers pas, un token
    $= \mathrm{MLP}(o_t) + P[\text{position}]$ ; deux blocs Transformer causaux du mini-GPT du
    chapitre 20 ; une tête lue à chaque position. Les fenêtres plus courtes que $K$ (début
    d'épisode) sont complétées à gauche par un token de remplissage appris.

    ```text
    pas        t-3        t-2        t-1        t
    entrée     o          o          o          o
    sortie                                      ↓
                       tête « token » : 256 logits → entropie croisée
                       tête « nombre » : 1 réel    → erreur quadratique
    ```

    À l'usage, la tête « token » donne une **distribution** sur 256 couples. On peut la
    décoder de trois façons : le bin le plus probable (**mode**), un bin tiré au hasard
    (**tirage**), ou l'espérance $\sum_b p_b\,u_b$ (**moyenne**), qui redonne exactement ce que
    vise la régression.
    """)
    return


@app.function
def generer_demos(n_hasard, n_repos, T, K_lqr, rng, bruit=0.0):
    """Démonstrations de l'expert : observations (N, T, 3), couples étiquettes (N, T), départ au repos (N,).

    Avec `bruit` > 0, le couple *exécuté* est celui de l'expert plus un bruit gaussien
    (l'étiquette reste le couple de l'expert, sans bruit) : les démonstrations visitent
    aussi des états un peu à côté de la trajectoire idéale, et montrent comment en revenir
    (idée de DART, Laskey et al., 2017)."""
    th, thd, repos = departs(n_hasard, n_repos, rng)
    cote = rng.choice([-1.0, 1.0], len(th))
    N = len(th)
    obs = np.zeros((N, T, 3), np.float32)
    etiquettes = np.zeros((N, T), np.float32)
    for t in range(T):
        obs[:, t] = np.stack([np.cos(th), np.sin(th), thd / VITESSE_MAX], -1)
        u = expert(th, thd, cote, K_lqr)
        etiquettes[:, t] = u
        th, thd, _ = pas_pendule(th, thd, u + bruit * rng.standard_normal(N))
    return obs, etiquettes, repos


@app.class_definition
class PolitiqueTransformer(nn.Module):
    """Petit Transformer causal sur les K derniers pas ; tête « token » (n_bins logits) ou « nombre »."""

    def __init__(self, n_bins=256, tete="token", K=10, n_embd=64, n_couches=2, n_tetes=2, dropout=0.0,
                 action_precedente=False):
        super().__init__()
        cfg = ConfigGPT(n_embd=n_embd, n_tetes=n_tetes, n_couches=n_couches, dropout=dropout, biais=True)
        self.n_bins, self.tete_type, self.action_precedente = n_bins, tete, action_precedente
        # Encodeur d'observation non linéaire : une simple couche linéaire suivie des LayerNorm
        # du Transformer perdrait l'échelle de l'entrée (LN(W·o) ne change pas si o double).
        self.plong_obs = nn.Sequential(nn.Linear(3, n_embd), nn.GELU(), nn.Linear(n_embd, n_embd))
        self.plong_action = nn.Embedding(n_bins + 1, n_embd)    # + 1 : « pas d'action précédente »
        self.plong_pos = nn.Embedding(K, n_embd)
        self.remplissage = nn.Parameter(torch.zeros(n_embd))
        self.blocs = nn.ModuleList([Bloc(cfg) for _ in range(n_couches)])
        self.ln = nn.LayerNorm(n_embd)
        self.tete = nn.Linear(n_embd, n_bins if tete == "token" else 1)

    def forward(self, O, A_prec, M):
        """O (B, K, 3) observations ; A_prec (B, K) bins de l'action précédente ; M (B, K) vrai si réel.
        Renvoie (B, K, n_bins) logits, ou (B, K) couples pour la tête « nombre »."""
        K = O.shape[1]
        x = self.plong_obs(O) + self.plong_pos(torch.arange(K))
        if self.action_precedente:                                  # désactivé par défaut (voir README)
            x = x + self.plong_action(A_prec)
        x = torch.where(M.unsqueeze(-1), x, self.remplissage)     # remplissage à gauche
        for bloc in self.blocs:
            x = bloc(x)
        y = self.tete(self.ln(x))
        return y if self.tete_type == "token" else y.squeeze(-1)


@app.function
def fenetres(obs, couples, n_bins, K, taille_lot, rng):
    """Lot de fenêtres de K pas finissant à un pas tiré au hasard (remplies à gauche si besoin)."""
    N, T, _ = obs.shape
    ep = rng.integers(0, N, taille_lot)
    fin = rng.integers(0, T, taille_lot)
    idx = fin[:, None] + np.arange(-K + 1, 1)[None, :]          # (B, K) indices de pas
    M = idx >= 0
    idx_c = np.maximum(idx, 0)
    O = obs[ep[:, None], idx_c] * M[..., None]
    U = couples[ep[:, None], idx_c]
    bins = vers_bins(couples, n_bins)
    prec = np.where(idx_c >= 1, bins[ep[:, None], np.maximum(idx_c - 1, 0)], n_bins)
    prec = np.where(M, prec, n_bins)
    return (torch.from_numpy(O.astype(np.float32)), torch.from_numpy(prec), torch.from_numpy(M),
            torch.from_numpy(U.astype(np.float32)), torch.from_numpy(vers_bins(U, n_bins)))


@app.function
def entrainer(modele, obs, couples, config, graine):
    """Clonage comportemental : entropie croisée (tête token) ou erreur quadratique (tête nombre)."""
    torch.manual_seed(graine)
    rng = np.random.default_rng(graine)
    optim = torch.optim.AdamW(modele.parameters(), lr=config["lr"], weight_decay=1e-4)
    pertes = []
    for it in range(config["n_pas"]):
        for g in optim.param_groups:              # échauffement linéaire, puis pas constant
            g["lr"] = config["lr"] * min(1.0, (it + 1) / config["echauffement"])
        O, P, M, U, B = fenetres(obs, couples, modele.n_bins, config["contexte"], config["taille_lot"], rng)
        y = modele(O, P, M)
        if modele.tete_type == "token":
            perte = F.cross_entropy(y[M], B[M])
        else:
            perte = F.mse_loss(y[M], U[M])
        optim.zero_grad()
        perte.backward()
        nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
        optim.step()
        pertes.append(perte.item())
    return pertes


@app.function
@torch.no_grad()
def jouer_modele(modele, th, thd, T, decodage="mode", graine=0):
    """Joue T pas en parallèle avec le modèle ; décodage « mode », « tirage » ou « moyenne »
    pour la tête token (ignoré pour la tête nombre). Renvoie (retours, angles, couples)."""
    modele.eval()
    K, n_bins = modele.plong_pos.num_embeddings, modele.n_bins
    centres = torch.from_numpy(depuis_bins(np.arange(n_bins), n_bins).astype(np.float32))
    gen = torch.Generator().manual_seed(graine)
    N = len(th)
    O_hist = np.zeros((N, T, 3), np.float32)
    B_hist = np.full((N, T), n_bins, np.int64)                    # bin de l'action jouée
    angles, couples, retours = np.zeros((N, T + 1)), np.zeros((N, T)), np.zeros(N)
    angles[:, 0] = th
    for t in range(T):
        O_hist[:, t] = np.stack([np.cos(th), np.sin(th), thd / VITESSE_MAX], -1)
        idx = np.arange(t - K + 1, t + 1)
        M = np.broadcast_to(idx >= 0, (N, K))
        O = O_hist[:, np.maximum(idx, 0)] * M[..., None]
        prec = np.where(idx >= 1, B_hist[:, np.maximum(idx - 1, 0)], n_bins)
        prec = np.where(M, prec, n_bins)
        y = modele(torch.from_numpy(O), torch.from_numpy(prec), torch.from_numpy(M.copy()))[:, -1]
        if modele.tete_type == "nombre":
            u = y.clamp(-COUPLE_MAX, COUPLE_MAX)
        else:
            p = y.softmax(-1)
            if decodage == "mode":
                u = centres[p.argmax(-1)]
            elif decodage == "tirage":
                u = centres[torch.multinomial(p, 1, generator=gen).squeeze(-1)]
            else:
                u = p @ centres
        u = u.numpy().astype(np.float64)
        B_hist[:, t] = vers_bins(u, n_bins)
        th, thd, r = pas_pendule(th, thd, u)
        retours += r
        angles[:, t + 1], couples[:, t] = th, u
    return retours, angles, couples


@app.cell
def _(mo):
    bouton = mo.ui.run_button(label="Générer les démonstrations et entraîner (≈ 2 min)")
    bouton
    return (bouton,)


@app.cell
def _(K_LQR, SEED, bouton, config):
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Générer les démonstrations et entraîner ».*"))
    _t0 = time.perf_counter()
    obs_demo, couples_demo, repos_demo = generer_demos(
        config["n_demo_hasard"], config["n_demo_repos"], config["T"], K_LQR, np.random.default_rng(SEED),
        config["bruit_demo"])
    _premier = couples_demo[repos_demo, 0]
    print(f"Démonstrations : {obs_demo.shape[0]} épisodes, {obs_demo.shape[0] * config['T']} pas "
          f"({time.perf_counter() - _t0:.2f} s)")
    print(f"Au repos en bas, premier couple de l'expert : +2 dans {np.mean(_premier > 0):.0%} des cas, "
          f"−2 dans {np.mean(_premier < 0):.0%} ; moyenne {_premier.mean():+.2f}")
    _b = vers_bins(couples_demo, 256)
    print(f"Bins 0 et 255 (couple à fond) : {np.mean((_b == 0) | (_b == 255)):.0%} des pas")

    modeles, pertes, durees_entr = {}, {}, {}
    for _tete in ("token", "nombre"):
        _m = PolitiqueTransformer(config["n_bins"], _tete, config["contexte"], config["n_embd"],
                                  config["n_couches"], config["n_tetes"], config["dropout"],
                                  config["action_precedente"])
        _t0 = time.perf_counter()
        pertes[_tete] = entrainer(_m, obs_demo, couples_demo, config, SEED)
        durees_entr[_tete] = time.perf_counter() - _t0
        modeles[_tete] = _m
        print(f"Tête « {_tete} » : {sum(p.numel() for p in _m.parameters()):,} paramètres, ".replace(",", " ")
              + f"{durees_entr[_tete]:.0f} s, perte finale (moyenne des 100 derniers pas) "
              f"{np.mean(pertes[_tete][-100:]):.4f}")

    # Témoin : mêmes modèles, démonstrations SANS bruit (l'expert exécute exactement son couple)
    if config["temoin_sans_bruit"]:
        _obs0, _cou0, _ = generer_demos(config["n_demo_hasard"], config["n_demo_repos"], config["T"], K_LQR,
                                        np.random.default_rng(SEED), 0.0)
        for _tete in ("token", "nombre"):
            _m = PolitiqueTransformer(config["n_bins"], _tete, config["contexte"], config["n_embd"],
                                      config["n_couches"], config["n_tetes"], config["dropout"],
                                      config["action_precedente"])
            _t0 = time.perf_counter()
            pertes[_tete + "-sans-bruit"] = entrainer(_m, _obs0, _cou0, config, SEED)
            durees_entr[_tete + "-sans-bruit"] = time.perf_counter() - _t0
            modeles[_tete + "-sans-bruit"] = _m
            print(f"Témoin sans bruit, tête « {_tete} » : {durees_entr[_tete + '-sans-bruit']:.0f} s, perte finale "
                  f"{np.mean(pertes[_tete + '-sans-bruit'][-100:]):.4f}")
    return couples_demo, durees_entr, modeles, pertes


@app.cell
def _(config, eval_repos, eval_th, eval_thd, modeles):
    # --- Évaluation : mêmes départs que l'expert ---
    _t0 = time.perf_counter()
    evals, trajets = {}, {}
    for _nom, _tete, _dec in (("token, mode", "token", "mode"), ("token, tirage", "token", "tirage"),
                              ("token, moyenne", "token", "moyenne"), ("nombre (régression)", "nombre", "mode"),
                              ("token, mode, sans bruit", "token-sans-bruit", "mode"),
                              ("nombre, sans bruit", "nombre-sans-bruit", "mode")):
        if _tete not in modeles:
            continue
        _r, _a, _c = jouer_modele(modeles[_tete], eval_th.copy(), eval_thd.copy(), config["T"], _dec, graine=7)
        _ok = redresse(_a)
        evals[_nom] = dict(retour_hasard=float(_r[~eval_repos].mean()), retour_repos=float(_r[eval_repos].mean()),
                           redresse_hasard=float(_ok[~eval_repos].mean()), redresse_repos=float(_ok[eval_repos].mean()),
                           montee_repos=float(temps_de_montee(_a)[eval_repos].mean()))
        if "sans bruit" not in _nom:
            trajets[_nom] = _a[eval_repos][:3]
        print(f"{_nom:20s} : départs au hasard {evals[_nom]['retour_hasard']:7.1f}, redressé "
              f"{evals[_nom]['redresse_hasard']:4.0%} | au repos {evals[_nom]['retour_repos']:7.1f}, redressé "
              f"{evals[_nom]['redresse_repos']:4.0%}, au-dessus de l'horizontale après {evals[_nom]['montee_repos']:.2f} s")
    duree_eval = time.perf_counter() - _t0
    print(f"Évaluation : {duree_eval:.0f} s")
    return duree_eval, evals, trajets


@app.cell
def _(config, modeles):
    # --- Sonde : le pendule en bas (θ = π), vitesse θ̇ de −0,3 à +0,3, premier pas (aucune histoire) ---
    # L'expert y pousse à fond dans le sens du mouvement (+2 si θ̇ > 0, −2 si θ̇ < 0) et, presque
    # immobile (|θ̇| < 0,05), d'un côté tiré à pile ou face : jamais entre les deux.
    sonde_vitesses = np.round(np.linspace(-0.3, 0.3, 61), 3)
    _n, _K, _nb = len(sonde_vitesses), config["contexte"], config["n_bins"]
    _O = torch.zeros(_n, _K, 3)
    _O[:, -1, 0] = -1.0                                            # cos π
    _O[:, -1, 2] = torch.from_numpy(sonde_vitesses.astype(np.float32)) / VITESSE_MAX
    _M = torch.zeros(_n, _K, dtype=torch.bool)
    _M[:, -1] = True
    _P = torch.full((_n, _K), _nb)
    _centres = depuis_bins(np.arange(_nb), _nb)
    with torch.no_grad():
        _p = modeles["token"].eval()(_O, _P, _M)[:, -1].softmax(-1).numpy()
        _u_nombre = modeles["nombre"].eval()(_O, _P, _M)[:, -1].clamp(-2, 2).numpy()
    sonde = dict(vitesses=sonde_vitesses.tolist(),
                 token_mode=_centres[_p.argmax(1)].tolist(), token_moyenne=(_p @ _centres).tolist(),
                 nombre=_u_nombre.astype(float).tolist())
    _i0 = int(np.argmin(np.abs(sonde_vitesses)))
    p_repos = _p[_i0]
    u_nombre_repos = float(_u_nombre[_i0])
    entre_deux = {k: float(np.mean(np.abs(np.array(sonde[k])) < 1.5)) for k in ("token_mode", "token_moyenne", "nombre")}
    print(f"Au repos (θ̇ = 0) — tête token : P(u ≈ −2) = {p_repos[0]:.2f}, P(u ≈ +2) = {p_repos[-1]:.2f}, "
          f"reste {1 - p_repos[0] - p_repos[-1]:.2f} ; espérance {p_repos @ _centres:+.3f} ; tête nombre : {u_nombre_repos:+.3f}")
    print("Sur la sonde (61 vitesses), part des couples « entre les deux » (|u| < 1,5), que l'expert ne joue jamais :")
    for _k, _v in entre_deux.items():
        print(f"  {_k:14s} : {_v:.0%}")
    return entre_deux, p_repos, sonde, u_nombre_repos


@app.cell
def _(SMOKE, config, p_repos, pertes, sonde, trajets, u_nombre_repos):
    # --- Figure de la partie 2 ---
    _fig, _ax = plt.subplots(2, 2, figsize=(12, 8))
    (_a1, _a2), (_a3, _a4) = _ax
    _a1.plot(np.convolve(pertes["token"], np.ones(20) / 20, "valid"), label="token : entropie croisée")
    _a1b = _a1.twinx()
    _a1b.plot(np.convolve(pertes["nombre"], np.ones(20) / 20, "valid"), color=PALETTE[3], label="nombre : MSE")
    _a1.set(xlabel="pas d'entraînement", ylabel="entropie croisée", title="Pertes (moyenne glissante)")
    _a1b.set_ylabel("erreur quadratique")
    _a1.legend(loc="upper right", fontsize=8)
    _a1b.legend(loc="center right", fontsize=8)
    _centres = depuis_bins(np.arange(config["n_bins"]), config["n_bins"])
    _a2.bar(_centres, p_repos, width=4 / config["n_bins"] * 3, color=PALETTE[0], label="tête token : p(bin)")
    _a2.axvline(u_nombre_repos, color=PALETTE[3], ls="--", label=f"tête nombre : {u_nombre_repos:+.2f}")
    _a2.set(xlabel="couple u", ylabel="probabilité", title="En bas, immobile, premier pas")
    _a2.legend(fontsize=8)
    _v = np.array(sonde["vitesses"])
    _a3.step(_v, np.where(_v > 0, 2, -2), where="mid", color="grey", lw=3, alpha=0.5, label="expert (hors pile ou face)")
    _a3.axvspan(-0.05, 0.05, color="grey", alpha=0.15, label="pile ou face")
    _a3.plot(_v, sonde["token_mode"], "o", ms=3, color=PALETTE[0], label="token, mode")
    _a3.plot(_v, sonde["token_moyenne"], color=PALETTE[2], label="token, moyenne")
    _a3.plot(_v, sonde["nombre"], color=PALETTE[3], ls="--", label="nombre (régression)")
    _a3.set(xlabel="vitesse θ̇ (pendule en bas)", ylabel="couple prédit", title="Sonde : en bas, selon la vitesse")
    _a3.legend(fontsize=7)
    _t = np.arange(config["T"] + 1) * DT
    for (_nom, _a), _c in zip(trajets.items(), PALETTE):
        for _j, _x in enumerate(_a):
            _a4.plot(_t, np.abs(normaliser(_x)), color=_c, lw=1, label=_nom if _j == 0 else None)
    _a4.set(xlabel="temps (s)", ylabel="|θ| (0 = en haut, π = en bas)", title="Départs au repos (3 épisodes)")
    _a4.legend(fontsize=8)
    _fig.tight_layout()
    sauvegarder_figure(_fig, "tokens-contre-nombre-smoke" if SMOKE else "tokens-contre-nombre",
                       mo.notebook_dir() / "figures")
    _fig
    return


@app.cell
def _(
    SMOKE, angles_expert, config, couples_demo, couples_expert, duree_eval, durees_entr, entre_deux, eval_repos,
    evals, exemple_tokens, p_repos, pertes, quantif, ref_expert, sonde, trajets, u_nombre_repos,
):
    _i = int(np.flatnonzero(eval_repos)[0])
    resume = dict(
        config=config, expert=ref_expert, quantification={str(k): v for k, v in quantif.items()},
        exemple_tokens=exemple_tokens,
        episode_expert=dict(angles=normaliser(angles_expert[_i]).round(4).tolist(),
                            couples=couples_expert[_i].round(4).tolist()),
        part_couple_a_fond=float(np.mean(np.abs(couples_demo) >= 1.999)),
        modeles=evals, p_repos=p_repos.round(5).tolist(), u_nombre_repos=u_nombre_repos,
        sonde=sonde, entre_deux=entre_deux,
        trajets={k: normaliser(v).round(4).tolist() for k, v in trajets.items()},
        pertes={k: [float(np.mean(v[i:i + 20])) for i in range(0, len(v), 20)] for k, v in pertes.items()},
        durees=dict(entrainement=durees_entr, evaluation=duree_eval),
    )
    _chemin = mo.notebook_dir() / "figures" / ("resultats-smoke.json" if SMOKE else "resultats.json")
    _chemin.write_text(json.dumps(resume, ensure_ascii=False, separators=(",", ":")))
    print(f"Résultats : {_chemin}")
    return


@app.cell
def _(SMOKE, entre_deux, evals, p_repos, quantif, ref_expert):
    _tm, _nb = evals["token, mode"], evals["nombre (régression)"]
    verifs = {
        "Expert : redressé dans 100 % des départs (au hasard et au repos)":
            ref_expert["redresse_hasard"] == 1.0 and ref_expert["redresse_repos"] == 1.0,
        "Arrondi à 256 bins : retour de l'expert à moins de 1 % du continu (départs au hasard)":
            abs(quantif[256]["retour_hasard"] - ref_expert["retour_hasard"]) <= 0.01 * abs(ref_expert["retour_hasard"]),
        "Arrondi à 2 bins : retour moins bon d'au moins 10 % qu'avec 256 bins (départs au hasard)":
            quantif[2]["retour_hasard"] < 1.1 * quantif[256]["retour_hasard"],
        "Au repos, la tête token met ≥ 80 % de la masse sur les deux couples extrêmes, ≥ 20 % sur chacun":
            p_repos[0] + p_repos[-1] >= 0.8 and min(p_repos[0], p_repos[-1]) >= 0.2,
        "Sonde : la tête token (mode) ne joue jamais « entre les deux » (|u| < 1,5)": entre_deux["token_mode"] == 0.0,
        "Sonde : la régression joue « entre les deux » sur ≥ 10 % des vitesses": entre_deux["nombre"] >= 0.10,
        "Token (mode) : redressé ≥ 90 % des départs au hasard et ≥ 90 % des départs au repos":
            _tm["redresse_hasard"] >= 0.9 and _tm["redresse_repos"] >= 0.9,
        "Nombre (régression) : redressé ≥ 90 % des départs au hasard et au repos":
            _nb["redresse_hasard"] >= 0.9 and _nb["redresse_repos"] >= 0.9,
        "Témoin sans bruit : les deux têtes redressent ≤ 50 % des départs au hasard (erreurs qui s'accumulent)":
            all(evals[k]["redresse_hasard"] <= 0.5 for k in ("token, mode, sans bruit", "nombre, sans bruit")
                if k in evals),
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
