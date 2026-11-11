# Exercice 29 : rappel associatif, attention contre mémoire de taille fixe
#
#   make ex NUM=29              → éditeur marimo (bouton « Entraîner »)
#   make ex NUM=29 MODE=run     → script complet, vérifie les seuils du README
#   make ex NUM=29 MODE=smoke   → test rapide (< 30 s)
#
# Parties A et B en NumPy, sans apprentissage : trois façons de calculer la même sortie
# (forme parallèle et forme récurrente), coût en mémoire et en temps selon la longueur, puis
# capacité d'une mémoire associative de taille fixe (somme de produits extérieurs, règle delta).
# Partie C en PyTorch (GPU Apple via MPS si disponible) : quatre petits modèles entraînés sur
# la même tâche de rappel associatif (attention causale, attention linéaire, SSM diagonal
# invariant, SSM sélectif façon Mamba-2), précision selon le nombre de paires à retenir.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import math
    import os
    import time
    import tracemalloc

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from ia_commun import get_device, is_smoke, sauvegarder_figure, seed_everything, style_commun

    return (
        F,
        get_device,
        is_smoke,
        json,
        math,
        mo,
        nn,
        np,
        os,
        plt,
        sauvegarder_figure,
        seed_everything,
        style_commun,
        time,
        torch,
        tracemalloc,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Exercice 29 : se souvenir de tout ou résumer dans un état de taille fixe ?

    L'attention d'un Transformer garde **toutes** les clés et valeurs passées (le cache KV) :
    pour produire le token $t$, elle relit les $t$ précédents. Mémoire et calcul croissent avec
    la longueur, et la matrice d'attention complète en $O(n^2)$. Les alternatives de ce chapitre
    (attention linéaire, modèles à espace d'états, RWKV, DeltaNet…) remplacent ce cache par un
    **état de taille fixe**, mis à jour à chaque token : coût constant par token, mais une
    mémoire bornée. La question de l'exercice : **que perd-on ?**

    - **A. NumPy.** Attention linéaire et SSM sélectif calculés de deux façons (parallèle sur
      toute la séquence, ou récurrente token par token) : mêmes nombres. Puis la mémoire et le
      temps par token mesurés de 256 à 8 192 tokens.
    - **B. NumPy.** Une mémoire associative de taille fixe ($64 \times 64$ nombres) : combien de
      paires clé-valeur peut-on y ranger avant que les souvenirs se brouillent ? Somme de
      produits extérieurs (attention linéaire), règle delta (DeltaNet), attention softmax.
    - **C. PyTorch.** La tâche de **rappel associatif** (associative recall) : on lit
      `k1 v1 k2 v2 … kN vN`, puis des clés déjà vues ; il faut redonner la valeur associée.
      Quatre modèles identiques sauf la couche qui mélange les positions : attention causale,
      attention linéaire, SSM diagonal invariant (façon S4D), SSM sélectif (façon Mamba-2).

    **Perte optimisée (partie C)** : entropie croisée sur la valeur à redonner, uniquement aux
    positions des requêtes (prédiction du token suivant restreinte à ces positions). Parties A
    et B : aucune perte, rien n'est appris.
    """)


@app.cell
def _(get_device, is_smoke, json, os, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    device = get_device()
    style_commun()
    config = dict(
        # Partie A : longueurs testées pour la mémoire et le temps par token
        longueurs=[256, 512, 1024] if SMOKE else [256, 512, 1024, 2048, 4096, 8192],
        d_tete=64,               # dimension d'une tête (clés, valeurs, requêtes)
        # Partie B : mémoire associative 64 × 64, nombres de paires rangées
        paires_b=[4, 8, 16, 32, 64, 128, 256],
        essais_b=4 if SMOKE else 20,
        # Partie C : la tâche et les modèles
        n_cles=128,              # vocabulaire des clés (toutes distinctes dans une séquence)
        n_valeurs=64,            # vocabulaire des valeurs
        n_requetes=16,           # requêtes posées après les paires
        paires_min=4,
        paires_max=32 if SMOKE else 128,
        d_modele=64,
        d_etat=16,               # dimension des clés/requêtes (attention) = taille d'état N (SSM)
        couches=2,
        pas=40 if SMOKE else 2000,
        lot=64,
        lr=3e-3,
        biais_oubli=-7.0,        # SSM sélectif : biais initial de Δ ; a = exp(−softplus(biais)) ≈ 0,999
        modeles=["attention causale", "attention linéaire", "SSM diagonal (S4D)", "SSM sélectif (Mamba-2)"],
        paires_eval=[4, 8, 16, 32] if SMOKE else [4, 8, 16, 32, 64, 96, 128],
        n_eval=64 if SMOKE else 512,
    )
    # IA29_CONFIG='{"pas": 4000}' make ex NUM=29 MODE=run : modifier une clé sans éditer le fichier
    config.update(json.loads(os.environ.get("IA29_CONFIG", "{}")))
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SMOKE, config, device


@app.cell
def _(mo):
    mo.md(r"""
    ## A. Deux façons de calculer la même chose, et ce que chacune coûte

    **Attention linéaire** (Katharopoulos et al., 2020). On remplace
    $\mathrm{softmax}(q_t \cdot k_s)$ par un produit $\phi(q_t)\cdot\phi(k_s)$ avec
    $\phi(x) = \mathrm{elu}(x) + 1$ (toujours positif). Alors
    $$y_t = \frac{\sum_{s \le t} (\phi(q_t)\cdot\phi(k_s))\, v_s}{\sum_{s \le t} \phi(q_t)\cdot\phi(k_s)}
         = \frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t},\quad
    S_t = S_{t-1} + \phi(k_t)\, v_t^\top,\quad z_t = z_{t-1} + \phi(k_t).$$
    À gauche, la **forme parallèle** (une matrice $n \times n$ masquée, comme l'attention : idéale
    pour l'entraînement sur GPU). À droite, la **forme récurrente** : un état $S_t$ de
    $d \times d$ nombres, mis à jour à chaque token (idéale pour générer).

    **SSM sélectif à décroissance scalaire** (l'idée de Mamba-2) : $S_t = a_t S_{t-1} + k_t v_t^\top$,
    $y_t = q_t^\top S_t$, avec un oubli $a_t \in (0, 1)$ qui **dépend de l'entrée**. En déroulant :
    $y_t = \sum_{s \le t} \big(\prod_{s < r \le t} a_r\big)(q_t \cdot k_s)\, v_s$, c'est-à-dire une
    attention sans softmax dont la matrice est multipliée par un masque de décroissance $L$. C'est
    la « dualité » SSM / attention de Dao et Gu (2024).
    """)


@app.cell
def _(np):
    # --- A1. Vérifier numériquement que forme parallèle = forme récurrente ---
    rng_a = np.random.default_rng(42)
    T_a, d_a = 12, 4
    X_a = rng_a.normal(size=(T_a, d_a))
    Q_a, K_a, V_a = (X_a @ rng_a.normal(size=(d_a, d_a)) for _ in range(3))

    def phi(z):
        # elu(z) + 1 : z + 1 si z > 0, exp(z) sinon (toujours strictement positif)
        return np.where(z > 0, z + 1, np.exp(np.minimum(z, 0)))

    masque_a = np.tril(np.ones((T_a, T_a)))

    # Attention linéaire, forme parallèle : matrice n × n masquée, comme l'attention
    A_lin = (phi(Q_a) @ phi(K_a).T) * masque_a
    y_par_lin = A_lin @ V_a / A_lin.sum(1, keepdims=True)
    # Forme récurrente : un état S (d × d) et un normaliseur z (d), mis à jour token par token
    S_a, z_a, y_rec_lin = np.zeros((d_a, d_a)), np.zeros(d_a), []
    for t_a in range(T_a):
        S_a = S_a + np.outer(phi(K_a[t_a]), V_a[t_a])
        z_a = z_a + phi(K_a[t_a])
        y_rec_lin.append(phi(Q_a[t_a]) @ S_a / (phi(Q_a[t_a]) @ z_a))
    ecart_lin = float(np.abs(y_par_lin - np.array(y_rec_lin)).max())

    # SSM sélectif : oubli a_t ∈ (0, 1) qui dépend du token, forme parallèle (« duale »)
    a_t = 1 / (1 + np.exp(-rng_a.normal(size=T_a)))
    cum = np.cumsum(np.log(a_t))
    L_dec = np.exp(cum[:, None] - cum[None, :]) * masque_a   # L[t, s] = a_{s+1} × … × a_t
    y_par_sel = ((Q_a @ K_a.T) * L_dec) @ V_a
    S_b, y_rec_sel = np.zeros((d_a, d_a)), []
    for t_b in range(T_a):
        S_b = a_t[t_b] * S_b + np.outer(K_a[t_b], V_a[t_b])
        y_rec_sel.append(Q_a[t_b] @ S_b)
    ecart_sel = float(np.abs(y_par_sel - np.array(y_rec_sel)).max())
    print(f"Attention linéaire : écart max parallèle/récurrente = {ecart_lin:.1e}")
    print(f"SSM sélectif       : écart max parallèle/récurrente = {ecart_sel:.1e}")
    print("Masque de décroissance L (4 premières lignes) :\n", np.round(L_dec[:4, :4], 3))
    return ecart_lin, ecart_sel, phi


@app.cell
def _(config, np, time, tracemalloc):
    # --- A2. Mémoire et temps selon la longueur n (une tête, d = 64, float32) ---
    def softmax(a, axe=-1):
        a = a - a.max(axe, keepdims=True)
        e = np.exp(a)
        return e / e.sum(axe, keepdims=True)

    rng_m = np.random.default_rng(42)
    d_m = config["d_tete"]
    mesures_a = []
    for n_m in config["longueurs"]:
        q_m, k_m, v_m = (rng_m.normal(size=(n_m, d_m)).astype(np.float32) for _ in range(3))
        # (1) attention causale sur toute la séquence (entraînement, ou lecture du prompt) :
        #     la matrice n × n est construite en entier. tracemalloc voit les tableaux NumPy.
        #     Calcul en place pour ne garder qu'une matrice n × n de float32 (+ le masque booléen).
        tracemalloc.start()
        scores = q_m @ k_m.T                                   # n × n nombres : O(n²)
        scores *= 1 / np.sqrt(d_m)
        futur = np.logical_not(np.tri(n_m, dtype=bool))         # masque causal (1 octet par case)
        np.copyto(scores, -np.inf, where=futur)
        scores -= scores.max(1, keepdims=True)
        np.exp(scores, out=scores)
        scores /= scores.sum(1, keepdims=True)
        _y = scores @ v_m
        del scores, futur, _y
        pic_attention = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        # (2) même calcul en forme récurrente (attention linéaire) : un état d × d suffit
        tracemalloc.start()
        S_m = np.zeros((d_m, d_m), np.float32)
        for t_m in range(n_m):
            S_m += np.outer(k_m[t_m], v_m[t_m])
            _o = q_m[t_m] @ S_m
        pic_recurrent = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        # (3) génération du token n+1 : l'attention relit n clés et n valeurs (cache KV),
        #     la récurrence met à jour un état d × d. On chronomètre un pas : meilleur de 7 séries de 50
        #     (le minimum écarte les interruptions par d'autres programmes).
        def chrono(f):
            temps = []
            for _ in range(7):
                t0 = time.perf_counter()
                for _ in range(50):
                    f()
                temps.append((time.perf_counter() - t0) / 50)
            return float(np.min(temps))

        q_n = q_m[-1]
        t_attention = chrono(lambda: softmax(k_m @ q_n / np.sqrt(d_m)) @ v_m)
        t_recurrent = chrono(lambda: q_n @ (S_m + np.outer(k_m[-1], v_m[-1])))
        mesures_a.append(dict(
            n=n_m,
            pic_attention_octets=int(pic_attention),
            pic_recurrent_octets=int(pic_recurrent),
            cache_kv_octets=2 * n_m * d_m * 4,
            etat_octets=d_m * d_m * 4,
            t_token_attention_us=t_attention * 1e6,
            t_token_recurrent_us=t_recurrent * 1e6,
        ))
        print(f"n = {n_m:5d} : attention complète {pic_attention / 2**20:7.1f} Mio | "
              f"récurrente {pic_recurrent / 2**10:5.0f} Kio | cache KV {2 * n_m * d_m * 4 / 2**10:5.0f} Kio "
              f"contre état {d_m * d_m * 4 / 2**10:.0f} Kio | 1 token : attention {t_attention * 1e6:5.1f} µs, "
              f"récurrence {t_recurrent * 1e6:4.1f} µs")
    return mesures_a, softmax


@app.cell
def _(mo):
    mo.md(r"""
    ## B. Une mémoire de taille fixe déborde

    On range $N$ paires (clé $k_i$, valeur $v_i$), vecteurs unitaires aléatoires de dimension 64,
    puis on interroge chaque clé. Quatre mémoires :

    - **attention softmax** : garde les $N$ paires (le cache grandit avec $N$), lit
      $\mathrm{softmax}(\beta\, K k)^\top V$ avec $\beta = 20$ ;
    - **attention linéaire** (« règle de Hebb ») : $S = \sum_i v_i k_i^\top$, lecture $S k$.
      Taille fixe $64 \times 64$. Chaque lecture renvoie la bonne valeur **plus** le bruit de toutes
      les autres, proportionnel à $k_j \cdot k$ : tant que les clés sont presque orthogonales,
      ce bruit reste petit ;
    - **attention linéaire avec $\phi = \mathrm{elu} + 1$** : même chose sur des caractéristiques
      toujours positives, donc jamais orthogonales ;
    - **règle delta** (DeltaNet) : $S \leftarrow S + (v_i - S k_i)\, k_i^\top$ : on n'écrit que
      l'écart entre ce que la mémoire répond déjà et ce qu'elle devrait répondre.

    Mesures : la **précision** (la valeur lue est-elle plus proche de la bonne valeur que de toutes
    les autres valeurs rangées ?) et le **cosinus** entre valeur lue et bonne valeur (1 = parfait).
    """)


@app.cell
def _(config, np, phi, softmax):
    # --- B1. Capacité d'une mémoire associative de taille fixe ---
    rng_b = np.random.default_rng(42)
    d_b = 64

    def unitaires(n):
        x = rng_b.normal(size=(n, d_b))
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def evaluer(lu, V):
        # précision : la bonne valeur est la plus proche (produit scalaire) parmi les N rangées
        precision = float(((lu @ V.T).argmax(1) == np.arange(len(V))).mean())
        cosinus = float(np.mean(np.sum(lu * V, 1) / (np.linalg.norm(lu, axis=1) + 1e-12)))
        return precision, cosinus

    memoires = ("attention softmax", "attention linéaire", "linéaire, φ = elu + 1", "règle delta")
    capacite = {m: {"precision": [], "cosinus": []} for m in memoires}
    for N_b in config["paires_b"]:
        res_b = {m: [] for m in memoires}
        for _ in range(config["essais_b"]):
            K_b, V_b = unitaires(N_b), unitaires(N_b)
            # attention softmax : on garde tout (N × 64 clés + N × 64 valeurs)
            res_b["attention softmax"].append(evaluer(softmax(20 * K_b @ K_b.T) @ V_b, V_b))
            # attention linéaire : S = somme des produits extérieurs v kᵀ (64 × 64, quel que soit N)
            S_hebb = V_b.T @ K_b
            res_b["attention linéaire"].append(evaluer(K_b @ S_hebb.T, V_b))
            # même chose après φ = elu + 1 (caractéristiques positives), avec normalisation
            F_b = phi(K_b * np.sqrt(d_b))
            lu_elu = (F_b @ (V_b.T @ F_b).T) / (F_b @ F_b.sum(0))[:, None]
            res_b["linéaire, φ = elu + 1"].append(evaluer(lu_elu, V_b))
            # règle delta : on corrige l'erreur de la mémoire au lieu d'ajouter à l'aveugle
            S_delta = np.zeros((d_b, d_b))
            for i_b in range(N_b):
                S_delta += np.outer(V_b[i_b] - S_delta @ K_b[i_b], K_b[i_b])
            res_b["règle delta"].append(evaluer(K_b @ S_delta.T, V_b))
        for m in memoires:
            capacite[m]["precision"].append(float(np.mean([r[0] for r in res_b[m]])))
            capacite[m]["cosinus"].append(float(np.mean([r[1] for r in res_b[m]])))
    print("N paires :", config["paires_b"])
    for m in memoires:
        print(f"{m:24s} précision {[round(p, 3) for p in capacite[m]['precision']]}")
        print(f"{'':24s} cosinus   {[round(c, 3) for c in capacite[m]['cosinus']]}")

    # --- B2. Réécrire une clé : 16 paires, puis 8 de ces clés reçoivent une nouvelle valeur ---
    K_r, V_ancien, V_nouveau = unitaires(16), unitaires(16), unitaires(8)
    toutes = np.vstack([V_ancien, V_nouveau])
    ecrasement = {}
    S_h = V_ancien.T @ K_r + V_nouveau.T @ K_r[:8]
    S_d = np.zeros((d_b, d_b))
    for i_r in range(16):
        S_d += np.outer(V_ancien[i_r] - S_d @ K_r[i_r], K_r[i_r])
    for i_r in range(8):
        S_d += np.outer(V_nouveau[i_r] - S_d @ K_r[i_r], K_r[i_r])
    for nom_r, S_r in (("attention linéaire", S_h), ("règle delta", S_d)):
        lu_r = K_r[:8] @ S_r.T
        norme = np.linalg.norm(lu_r, axis=1)
        ecrasement[nom_r] = dict(
            nouvelle_trouvee=float(((lu_r @ toutes.T).argmax(1) == np.arange(16, 24)).mean()),
            cos_nouvelle=float(np.mean(np.sum(lu_r * V_nouveau, 1) / norme)),
            cos_ancienne=float(np.mean(np.sum(lu_r * V_ancien[:8], 1) / norme)),
        )
        print(f"Réécriture, {nom_r:18s} : nouvelle valeur trouvée {ecrasement[nom_r]['nouvelle_trouvee']:.0%}, "
              f"cos(nouvelle) {ecrasement[nom_r]['cos_nouvelle']:.2f}, cos(ancienne) {ecrasement[nom_r]['cos_ancienne']:.2f}")
    return capacite, ecrasement, memoires


@app.cell
def _(mo):
    mo.md(r"""
    ## C. Rappel associatif : quatre modèles, une seule couche qui change

    Séquence : $N$ paires `clé valeur` (clés toutes distinctes), puis 16 clés déjà vues. À
    chaque requête, le modèle doit sortir la valeur associée. Hasard : 1/64.

    Chaque modèle : plongement du token **plus** plongement du token précédent (un « décalage
    de token », comme dans RWKV ou les courtes convolutions des SSM), puis 2 blocs
    `couche de mélange → MLP` avec connexions résiduelles. Seule la couche de mélange change :

    | Modèle | Mémoire en génération | Mélange des positions |
    |---|---|---|
    | attention causale | cache KV : $2 \times n \times 64$ nombres par couche, **croît avec $n$** | $\mathrm{softmax}(QK^\top/\sqrt{d})V$, masque causal |
    | attention linéaire | état $16 \times 64$ (+ 16) | $\phi(Q)\phi(K)^\top$ masqué, normalisé |
    | SSM diagonal invariant | état $64 \times 16$ | convolution par un noyau fixe $\bar K_\ell = \sum_n C_n \bar a_n^\ell B_n$ (S4D) : **le même pour toutes les entrées** |
    | SSM sélectif | état $16 \times 64$ | $(C B^\top \odot L)\,V$ : $B_t$, $C_t$ et l'oubli $a_t$ dépendent du token (Mamba-2) |

    Les trois modèles récurrents ont le **même** état de 1 024 nombres par couche. Entraînement :
    longueurs mélangées ($N$ de 4 à 128 paires), puis évaluation à $N$ fixé.
    """)


@app.cell
def _(F, config, math, nn, torch):
    # --- Les quatre couches de mélange (forme parallèle pour l'entraînement) ---
    def masque_causal(T, device):
        return torch.ones(T, T, device=device, dtype=torch.bool).tril()

    class AttentionCausale(nn.Module):
        def __init__(self, d, d_etat):
            super().__init__()
            self.q, self.k = nn.Linear(d, d_etat), nn.Linear(d, d_etat)
            self.v, self.o = nn.Linear(d, d), nn.Linear(d, d)
            self.echelle = 1 / math.sqrt(d_etat)

        def forward(self, x):
            s = self.q(x) @ self.k(x).transpose(1, 2) * self.echelle      # (B, T, T) : O(n²)
            s = s.masked_fill(~masque_causal(x.shape[1], x.device), float("-inf"))
            return self.o(s.softmax(-1) @ self.v(x))

    class AttentionLineaire(nn.Module):
        """Katharopoulos et al. (2020) : φ(x) = elu(x) + 1 à la place de l'exponentielle."""

        def __init__(self, d, d_etat):
            super().__init__()
            self.q, self.k = nn.Linear(d, d_etat), nn.Linear(d, d_etat)
            self.v, self.o = nn.Linear(d, d), nn.Linear(d, d)

        def forward(self, x):
            q, k = F.elu(self.q(x)) + 1, F.elu(self.k(x)) + 1
            a = (q @ k.transpose(1, 2)) * masque_causal(x.shape[1], x.device)
            return self.o((a @ self.v(x)) / (a.sum(-1, keepdim=True) + 1e-6))

    class SSMDiagonal(nn.Module):
        """SSM diagonal invariant dans le temps (façon S4D, valeurs propres réelles).

        Pour chaque canal c et chaque état n : h_t = ā h_{t-1} + B u_t, y_t = Σ_n C h_t,
        avec ā = exp(Δ·A), A = −(n+1) (initialisation « S4D-Real »). Comme ā, B et C ne
        dépendent pas de l'entrée, la sortie est une convolution par un noyau fixe K̄.
        """

        def __init__(self, d, d_etat):
            super().__init__()
            self.entree, self.o = nn.Linear(d, d), nn.Linear(d, d)
            self.log_dt = nn.Parameter(torch.empty(d, 1).uniform_(math.log(1e-3), math.log(1e-1)))
            self.log_A = nn.Parameter(torch.log(torch.arange(1, d_etat + 1).float()).repeat(d, 1))
            self.B = nn.Parameter(torch.randn(d, d_etat) / math.sqrt(d_etat))
            self.C = nn.Parameter(torch.randn(d, d_etat) / math.sqrt(d_etat))
            self.D = nn.Parameter(torch.ones(d))

        def forward(self, x):
            T = x.shape[1]
            u = self.entree(x)
            log_a = -torch.exp(self.log_dt) * torch.exp(self.log_A)       # (d, N), log ā < 0
            l = torch.arange(T, device=x.device, dtype=x.dtype)
            noyau = torch.einsum("cn,cnl->cl", self.B * self.C, torch.exp(log_a[:, :, None] * l))
            ecart = l[:, None] - l[None, :]                                  # t − s
            toeplitz = noyau[:, ecart.clamp(min=0).long()] * (ecart >= 0)   # (d, T, T)
            y = torch.einsum("cts,bsc->btc", toeplitz, u) + self.D * u
            return self.o(F.gelu(y))

    class SSMSelectif(nn.Module):
        """SSM sélectif à décroissance scalaire (esprit de Mamba-2), forme « duale ».

        S_t = a_t S_{t-1} + B_t v_tᵀ, y_t = C_tᵀ S_t, avec B_t, C_t et a_t = exp(−softplus(Δ_t))
        calculés à partir du token : le modèle choisit quoi écrire, quoi lire et quoi oublier.
        """

        def __init__(self, d, d_etat):
            super().__init__()
            self.C_proj, self.B_proj = nn.Linear(d, d_etat), nn.Linear(d, d_etat)
            self.v, self.o = nn.Linear(d, d), nn.Linear(d, d)
            self.norme = nn.LayerNorm(d)               # comme Mamba-2 : normaliser avant la sortie
            self.delta = nn.Linear(d, 1)
            # oubli très faible au départ (a ≈ 0,999) : sinon la mémoire s'efface en ~50 tokens
            nn.init.constant_(self.delta.bias, config["biais_oubli"])

        def forward(self, x):
            log_a = -F.softplus(self.delta(x)).squeeze(-1)            # (B, T), log a_t ≤ 0
            cum = torch.cumsum(log_a, 1)
            seg = cum[:, :, None] - cum[:, None, :]                    # Σ_{s<r≤t} log a_r
            L = torch.exp(seg.masked_fill(~masque_causal(x.shape[1], x.device), float("-inf")))
            a = (self.C_proj(x) @ self.B_proj(x).transpose(1, 2)) * L
            return self.o(self.norme(a @ self.v(x)))

    MELANGEURS = {
        "attention causale": AttentionCausale,
        "attention linéaire": AttentionLineaire,
        "SSM diagonal (S4D)": SSMDiagonal,
        "SSM sélectif (Mamba-2)": SSMSelectif,
    }
    return (MELANGEURS,)


@app.cell
def _(F, config, nn, torch):
    # --- La tâche et le modèle commun ---
    N_CLES, N_VAL = config["n_cles"], config["n_valeurs"]
    VIDE = N_CLES + N_VAL            # token « rien avant » pour le décalage de la 1re position
    VOCAB = VIDE + 1

    def lot_rappel(B, N, gen):
        """B séquences de N paires puis n_requetes clés ; cibles = -100 hors requêtes."""
        cles = torch.argsort(torch.rand(B, N_CLES, generator=gen), dim=1)[:, :N]
        valeurs = torch.randint(0, N_VAL, (B, N), generator=gen) + N_CLES
        contexte = torch.stack([cles, valeurs], 2).reshape(B, 2 * N)
        quelles = torch.randint(0, N, (B, config["n_requetes"]), generator=gen)
        x = torch.cat([contexte, torch.gather(cles, 1, quelles)], 1)
        y = torch.full_like(x, -100)
        y[:, 2 * N:] = torch.gather(valeurs, 1, quelles)
        return x, y

    class ModeleRappel(nn.Module):
        def __init__(self, melangeur):
            super().__init__()
            d, n = config["d_modele"], config["couches"]
            self.emb, self.emb_prec = nn.Embedding(VOCAB, d), nn.Embedding(VOCAB, d)
            self.norm_m = nn.ModuleList(nn.LayerNorm(d) for _ in range(n))
            self.melange = nn.ModuleList(melangeur(d, config["d_etat"]) for _ in range(n))
            self.mlp = nn.ModuleList(
                nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
                for _ in range(n))
            self.norm_f, self.tete = nn.LayerNorm(d), nn.Linear(d, VOCAB)

        def forward(self, x):
            precedent = F.pad(x, (1, 0), value=VIDE)[:, :-1]        # décalage d'un token
            h = self.emb(x) + self.emb_prec(precedent)
            for nm, m, f in zip(self.norm_m, self.melange, self.mlp):
                h = h + m(nm(h))
                h = h + f(h)
            return self.tete(self.norm_f(h))

    return ModeleRappel, VOCAB, lot_rappel


@app.cell
def _(mo):
    # Le bouton empêche l'entraînement de se relancer à chaque modification d'une cellule amont.
    bouton = mo.ui.run_button(label="Entraîner")
    bouton
    return (bouton,)


@app.cell
def _(
    F,
    MELANGEURS,
    ModeleRappel,
    VOCAB,
    bouton,
    config,
    device,
    lot_rappel,
    mo,
    time,
    torch,
):
    # En script (make ex MODE=run|smoke), personne ne clique : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Entraîner » pour lancer l'apprentissage des quatre modèles.*"))

    def entrainer(nom, melangeur):
        torch.manual_seed(42)                       # mêmes poids initiaux de départ pour tous
        modele = ModeleRappel(melangeur).to(device)
        optim = torch.optim.AdamW(modele.parameters(), lr=config["lr"], weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(optim, config["lr"], total_steps=config["pas"],
                                                    pct_start=0.1)
        gen = torch.Generator().manual_seed(42)     # mêmes lots pour tous les modèles
        pertes = []
        t0 = time.perf_counter()
        for _pas in range(config["pas"]):
            N = int(torch.randint(config["paires_min"], config["paires_max"] + 1, (1,), generator=gen))
            x, y = (t.to(device) for t in lot_rappel(config["lot"], N, gen))
            perte = F.cross_entropy(modele(x).reshape(-1, VOCAB), y.reshape(-1))   # ignore -100
            optim.zero_grad()
            perte.backward()
            optim.step()
            sched.step()
            pertes.append(perte.item())
        duree = time.perf_counter() - t0
        # Évaluation : séquences jamais vues (autre graine), N fixé
        modele.eval()
        gen_eval = torch.Generator().manual_seed(1234)
        precision = []
        with torch.no_grad():
            for N in config["paires_eval"]:
                x, y = (t.to(device) for t in lot_rappel(config["n_eval"], N, gen_eval))
                pred, m = modele(x).argmax(-1), y != -100
                precision.append((pred[m] == y[m]).float().mean().item())
        n_param = sum(p.numel() for p in modele.parameters())
        print(f"{nom:24s} {duree:5.1f} s, {n_param} paramètres, perte finale {sum(pertes[-50:]) / 50:.3f}, "
              f"précision {[round(p, 3) for p in precision]}")
        return dict(pertes=pertes, precision=precision, duree=duree, n_param=n_param)

    t_total = time.perf_counter()
    resultats_c = {nom: entrainer(nom, MELANGEURS[nom]) for nom in config["modeles"]}
    duree_c = time.perf_counter() - t_total
    print(f"Partie C : {duree_c:.0f} s pour les quatre modèles")
    return duree_c, resultats_c


@app.cell
def _(
    SMOKE,
    capacite,
    config,
    json,
    mesures_a,
    mo,
    np,
    plt,
    resultats_c,
    sauvegarder_figure,
):
    # --- Figure : coût selon n (A), capacité (B), rappel associatif (C) ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.3))
    n_ = [m["n"] for m in mesures_a]
    ax1.loglog(n_, [m["pic_attention_octets"] / 2**20 for m in mesures_a], "o-", label="attention, matrice n × n")
    ax1.loglog(n_, [m["cache_kv_octets"] / 2**20 for m in mesures_a], "s-", label="cache KV (génération)")
    ax1.loglog(n_, [m["etat_octets"] / 2**20 for m in mesures_a], "^-", label="état récurrent 64 × 64")
    ax1.set(xlabel="longueur n (tokens)", ylabel="mémoire (Mio)", title="A. Mémoire d'une tête (d = 64)")
    ax1.minorticks_off()
    ax1.set_xticks(n_, [str(v) for v in n_])
    ax1.legend(fontsize=8)
    for m_ in capacite:
        ax2.semilogx(config["paires_b"], capacite[m_]["cosinus"], "o-", label=m_)
    ax2.axvline(64, color="grey", ls=":", lw=1)
    ax2.minorticks_off()
    ax2.set_xticks(config["paires_b"], [str(v) for v in config["paires_b"]])
    ax2.set(xlabel="paires rangées N", ylabel="cosinus (valeur lue, bonne valeur)",
            title="B. Mémoire 64 × 64 : qualité de lecture", ylim=(0, 1.05))
    ax2.legend(fontsize=8)
    for nom_, r_ in resultats_c.items():
        ax3.plot(config["paires_eval"], [100 * p for p in r_["precision"]], "o-", label=nom_)
    ax3.axhline(100 / config["n_valeurs"], color="grey", ls=":", lw=1)
    ax3.set(xlabel="paires à retenir N", ylabel="précision (%)", title="C. Rappel associatif appris",
            ylim=(0, 105))
    ax3.legend(fontsize=8)
    fig.tight_layout()
    dossier = mo.notebook_dir() / "figures"
    suffixe = "-smoke" if SMOKE else ""
    print(f"Figure : {sauvegarder_figure(fig, 'ssm-vs-attention' + suffixe, dossier)[0]}")

    # Résultats bruts, relus par les figures du site (site/scripts/figures/ch29_*.py)
    bilan_json = dict(
        config=config,
        mesures_a=mesures_a,
        capacite=capacite,
        rappel={k: dict(precision=v["precision"], duree=v["duree"], n_param=v["n_param"],
                        pertes=v["pertes"][::10]) for k, v in resultats_c.items()},
    )
    np.savez(dossier / f"resultats{suffixe}.npz", bilan=json.dumps(bilan_json))
    fig
    return


@app.cell
def _(
    SMOKE,
    capacite,
    config,
    duree_c,
    ecart_lin,
    ecart_sel,
    ecrasement,
    mesures_a,
    mo,
    resultats_c,
):
    # --- Bilan : seuils chiffrés du README. En smoke, on vérifie seulement que tout tourne.
    echecs = []

    def verifier(condition, message):
        if not condition:
            echecs.append(message)

    verifier(max(ecart_lin, ecart_sel) < 1e-10, "A1 : formes parallèle et récurrente différentes")
    if not SMOKE:
        m0, m1 = mesures_a[0], mesures_a[-1]
        rapport_n = m1["n"] / m0["n"]
        verifier(m1["pic_attention_octets"] / m0["pic_attention_octets"] > 0.5 * rapport_n**2,
                 "A2 : la mémoire de l'attention complète devrait croître comme n²")
        verifier(m1["pic_recurrent_octets"] < 2 * m0["pic_recurrent_octets"],
                 "A2 : la mémoire de la forme récurrente devrait rester constante")
        verifier(m1["t_token_attention_us"] > 3 * m1["t_token_recurrent_us"],
                 "A2 : à 8 192 tokens, un pas d'attention devrait coûter au moins 3 fois un pas récurrent")
        i64, i256 = config["paires_b"].index(64), config["paires_b"].index(256)
        verifier(min(capacite["attention softmax"]["precision"]) == 1.0, "B : la softmax devrait tout retrouver")
        verifier(capacite["attention linéaire"]["cosinus"][i256] < 0.6,
                 "B : à 256 paires, la mémoire linéaire 64 × 64 devrait être brouillée (cosinus < 0,6)")
        verifier(capacite["linéaire, φ = elu + 1"]["precision"][i64] < 0.5,
                 "B : avec φ = elu + 1, la mémoire devrait échouer dès 64 paires")
        verifier(ecrasement["règle delta"]["cos_nouvelle"] > 0.9 > ecrasement["attention linéaire"]["cos_nouvelle"],
                 "B2 : la règle delta devrait remplacer l'ancienne valeur, la somme les mélanger")
        p = {k: v["precision"] for k, v in resultats_c.items()}
        if len(p) == 4:   # seuils de la partie C seulement quand les quatre modèles ont tourné
            verifier(min(p["attention causale"]) >= 0.99, "C : l'attention devrait réussir à toutes les longueurs")
            verifier(max(p["SSM diagonal (S4D)"]) <= 0.4, "C : le SSM invariant ne devrait pas savoir faire le rappel")
            verifier(max(p["SSM sélectif (Mamba-2)"][:3]) >= 0.85,
                     "C : le SSM sélectif devrait réussir avec peu de paires (≥ 85 % pour N ≤ 16)")
            for nom in ("attention linéaire", "SSM sélectif (Mamba-2)"):
                verifier(p[nom][-1] <= max(p[nom]) - 0.2,
                         f"C : {nom} devrait perdre au moins 20 points à 128 paires (état fixe)")
        if echecs:
            raise SystemExit("Seuils non atteints :\n- " + "\n- ".join(echecs))
    lignes = "\n".join(f"| {k} | {' | '.join(f'{100 * x:.0f} %' for x in v['precision'])} |"
                       for k, v in resultats_c.items())
    entete = " | ".join(f"N = {n}" for n in config["paires_eval"])
    mo.md(f"""
    **Bilan** ({'smoke, seuils non vérifiés' if SMOKE else 'tous les seuils sont atteints'}) ;
    partie C en {duree_c:.0f} s.

    | Modèle | {entete} |
    |---|{'---|' * len(config['paires_eval'])}
    {lignes}
    """)


if __name__ == "__main__":
    app.run()
