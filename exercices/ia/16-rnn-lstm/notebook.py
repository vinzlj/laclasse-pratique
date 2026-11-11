# Exercice 16 : réseaux récurrents, LSTM, recherche en faisceau et attention.
#
#   make ex NUM=16              → éditeur marimo (interactif, bouton « Entraîner »)
#   make ex NUM=16 MODE=run     → script complet (sert à mesurer la durée)
#   make ex NUM=16 MODE=smoke   → test rapide, < 30 s
#
# Quatre parties, de la plus petite à la plus grosse :
#   1. un RNN écrit en numpy : propagation, rétropropagation dans le temps (BPTT) écrite à la
#      main et vérifiée par l'autograd de PyTorch, norme du gradient qui s'effondre ;
#   2. une tâche de mémoire (retenir le premier symbole pendant T pas) : RNN contre LSTM ;
#   3. un modèle de langage par caractère sur TinyShakespeare (RNN, GRU, LSTM), puis
#      décodage glouton, recherche en faisceau et échantillonnage ;
#   4. un modèle séquence à séquence (inverser une suite de lettres), sans puis avec
#      attention : le goulot du vecteur unique et la carte d'alignement.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import math
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from ia_commun import get_device, is_smoke, sauvegarder_figure, seed_everything, style_commun
    from ia_commun.donnees import charger_tinyshakespeare

    return (
        F,
        charger_tinyshakespeare,
        get_device,
        is_smoke,
        json,
        math,
        mo,
        nn,
        np,
        plt,
        sauvegarder_figure,
        seed_everything,
        style_commun,
        time,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Exercice 16 : RNN, LSTM, recherche en faisceau et attention

    **Objectif.** Voir de ses yeux les trois problèmes que l'histoire des séquences a dû
    résoudre avant le Transformer :

    1. **le gradient s'éteint dans le temps** dans un réseau récurrent simple (RNN), et le
       LSTM le fait passer ;
    2. **décoder** un modèle de langage n'est pas trivial : le choix glouton, la recherche en
       faisceau et le tirage au hasard donnent des textes très différents ;
    3. **résumer une phrase entière dans un seul vecteur** est un goulot, que l'attention lève.

    **Ce qui est optimisé.** Partout, l'entropie croisée de l'élément à prédire :
    $\mathcal{L} = -\frac{1}{N}\sum_i \log p_\theta(y_i \mid \text{contexte}_i)$.
    Pour la tâche de mémoire, $y$ est le symbole à retenir ; pour le modèle de langage et le
    modèle séquence à séquence, c'est l'élément suivant de la séquence, le contexte étant les
    **vrais** éléments précédents (forçage par l'enseignant, *teacher forcing*).
    """)


@app.cell
def _(get_device, is_smoke, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    device = get_device()
    style_commun()
    config = dict(
        # Partie 1 : RNN numpy
        rnn_np_cachee=32,
        rnn_np_pas=300 if SMOKE else 1500,
        rnn_np_retards=[5, 10] if SMOKE else [5, 10, 20],
        rayons=[0.5, 0.9, 1.2, 2.0],      # rayon spectral de W_h pour la figure du gradient
        T_gradient=60,
        # Partie 2 : tâche de mémoire, RNN contre LSTM (PyTorch, CPU : réseaux minuscules)
        memoire_cachee=32,
        memoire_pas=200 if SMOKE else 2000,
        memoire_retards=[5, 20] if SMOKE else [5, 10, 20, 30, 50],
        biais_oubli=3.0,                  # sigma(3) = 0,95 : la cellule commence par retenir
        # Partie 3 : modèle de langage par caractère (MPS)
        lm_cachee=128 if SMOKE else 256,
        lm_couches=2,
        lm_pas=30 if SMOKE else 1000,
        lm_lot=64,
        lm_fenetre=128,
        lm_eval_tous=10 if SMOKE else 100,
        # Partie 3 bis : décodage
        dec_longueur=40 if SMOKE else 200,
        dec_largeurs=[1, 2, 4] if SMOKE else [1, 2, 4, 8, 16],
        # Partie 4 : séquence à séquence (inverser une suite de lettres)
        s2s_pas=100 if SMOKE else 800,
        s2s_lmax=12 if SMOKE else 30,
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SEED, SMOKE, config, device


@app.cell
def _(mo):
    mo.md(r"""
    ## Partie 1 : un RNN en numpy, et son gradient

    **La tâche de mémoire.** Une séquence de $T$ symboles tirés parmi 4 ; le premier porte un
    drapeau « à retenir ». Après le dernier, un drapeau « réponds » : le réseau doit donner
    le **premier** symbole. Les $T-1$ symboles du milieu sont du bruit. Au hasard, on a 25 %.

    **Le RNN.** $h_t = \tanh(x_t W_x + h_{t-1} W_h + b)$, puis à la fin
    $\hat y = \mathrm{softmax}(h_T W_y + c)$. La rétropropagation dans le temps (BPTT) remonte
    le gradient pas par pas :
    $\frac{\partial \mathcal{L}}{\partial h_{t-1}} = \Big(\frac{\partial \mathcal{L}}{\partial h_t} \odot (1 - h_t^2)\Big) W_h^\top$.
    À chaque pas, on multiplie par $W_h^\top$ et par une dérivée de tanh inférieure à 1 :
    sur 50 pas, le produit s'effondre (ou explose).
    """)


@app.cell
def _(np):
    # --- RNN numpy : données, initialisation, propagation, BPTT ---
    K_SYMB = 4            # symboles possibles
    D_ENTREE = K_SYMB + 2  # one-hot du symbole + drapeau « à retenir » + drapeau « réponds »

    def lot_memoire(rng, B, T):
        """B séquences de T symboles (+ un pas « réponds ») ; cible = premier symbole."""
        s = rng.integers(0, K_SYMB, (B, T))
        x = np.zeros((B, T + 1, D_ENTREE))
        x[np.arange(B)[:, None], np.arange(T)[None, :], s] = 1.0
        x[:, 0, K_SYMB] = 1.0       # drapeau « à retenir » sur le premier pas
        x[:, T, K_SYMB + 1] = 1.0   # drapeau « réponds » sur le dernier pas
        return x, s[:, 0]

    def init_rnn(rng, H, rayon=None):
        P = {
            "Wx": rng.normal(0, 1 / np.sqrt(D_ENTREE), (D_ENTREE, H)),
            "Wh": rng.normal(0, 1 / np.sqrt(H), (H, H)),
            "b": np.zeros(H),
            "Wy": rng.normal(0, 1 / np.sqrt(H), (H, K_SYMB)),
            "c": np.zeros(K_SYMB),
        }
        if rayon is not None:  # on impose le rayon spectral (plus grande valeur propre en module)
            P["Wh"] *= rayon / np.max(np.abs(np.linalg.eigvals(P["Wh"])))
        return P

    def avant(P, x):
        """Propagation : renvoie la liste des états h_0..h_T et les probabilités finales."""
        B = x.shape[0]
        h = np.zeros((B, P["Wh"].shape[0]))
        hs = [h]
        for t in range(x.shape[1]):
            h = np.tanh(x[:, t] @ P["Wx"] + h @ P["Wh"] + P["b"])
            hs.append(h)
        z = h @ P["Wy"] + P["c"]
        z = z - z.max(1, keepdims=True)  # stabilité numérique du softmax
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        return hs, p

    def entropie_croisee(p, y):
        return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))

    def bptt(P, x, hs, p, y):
        """Rétropropagation dans le temps, écrite à la main.

        Renvoie les gradients et, pour chaque pas t, la norme moyenne de dL/dh_t
        (par exemple du lot) : c'est elle qui s'effondre quand on remonte le temps.
        """
        B, T1, _ = x.shape
        G = {k: np.zeros_like(v) for k, v in P.items()}
        dz = p.copy()
        dz[np.arange(B), y] -= 1.0      # dérivée softmax + entropie croisée
        dz /= B
        G["Wy"] = hs[-1].T @ dz
        G["c"] = dz.sum(0)
        dh = dz @ P["Wy"].T             # dL/dh_T
        normes = np.zeros(T1)
        for t in reversed(range(T1)):
            normes[t] = np.linalg.norm(dh, axis=1).mean() * B
            da = dh * (1 - hs[t + 1] ** 2)   # à travers tanh
            G["Wx"] += x[:, t].T @ da
            G["Wh"] += hs[t].T @ da          # mêmes poids à chaque pas : on additionne
            G["b"] += da.sum(0)
            dh = da @ P["Wh"].T              # vers h_{t-1}
        return G, normes

    return D_ENTREE, K_SYMB, avant, bptt, entropie_croisee, init_rnn, lot_memoire


@app.cell
def _(avant, bptt, entropie_croisee, init_rnn, lot_memoire, np, torch):
    # --- Vérification : notre BPTT contre l'autograd de PyTorch (float64, CPU) ---
    _rng = np.random.default_rng(42)
    _P = init_rnn(_rng, 16)
    _x, _y = lot_memoire(_rng, 8, 12)
    _hs, _p = avant(_P, _x)
    _G, _ = bptt(_P, _x, _hs, _p, _y)

    _Pt = {k: torch.tensor(v, requires_grad=True) for k, v in _P.items()}
    _h = torch.zeros(8, 16, dtype=torch.float64)
    _xt = torch.tensor(_x)
    for _t in range(_x.shape[1]):
        _h = torch.tanh(_xt[:, _t] @ _Pt["Wx"] + _h @ _Pt["Wh"] + _Pt["b"])
    _L = torch.nn.functional.cross_entropy(_h @ _Pt["Wy"] + _Pt["c"], torch.tensor(_y))
    _L.backward()
    ecart_gradient = max(float(np.abs(_G[k] - _Pt[k].grad.numpy()).max()) for k in _P)
    ecart_perte = abs(entropie_croisee(_p, _y) - _L.item())
    print(f"BPTT numpy contre autograd : écart max sur les gradients {ecart_gradient:.1e}, "
          f"sur la perte {ecart_perte:.1e}")
    return (ecart_gradient,)


@app.cell
def _(avant, bptt, config, init_rnn, lot_memoire, np):
    # --- Norme de dL/dh_t en remontant le temps, selon le rayon spectral de W_h ---
    # Réseau non entraîné, T = 60 : on regarde seulement comment le signal d'erreur
    # voyage vers le passé.
    normes_gradient = {}
    for _rayon in config["rayons"]:
        _rng = np.random.default_rng(42)
        _P = init_rnn(_rng, config["rnn_np_cachee"], rayon=_rayon)
        _x, _y = lot_memoire(_rng, 256, config["T_gradient"])
        _hs, _p = avant(_P, _x)
        _, _n = bptt(_P, _x, _hs, _p, _y)
        _n = _n[::-1] / _n[-1]   # index k = k pas avant la fin, normalisé à 1 au dernier pas
        normes_gradient[_rayon] = _n
        print(f"rayon {_rayon}: |dL/dh| relatif à 10 pas {_n[10]:.1e}, à 30 pas {_n[30]:.1e}, "
              f"à {config['T_gradient']} pas {_n[config['T_gradient']]:.1e}")
    return (normes_gradient,)


@app.cell
def _(mo):
    bouton = mo.ui.run_button(label="Entraîner")
    bouton
    return (bouton,)


@app.cell
def _(avant, bouton, bptt, config, init_rnn, lot_memoire, mo, np, time):
    # --- Entraîner le RNN numpy (Adam écrit à la main) sur la tâche de mémoire ---
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Entraîner » pour lancer les quatre parties.*"))

    def entrainer_rnn_numpy(T, n_pas, H, lr=3e-3):
        rng = np.random.default_rng(42)
        P = init_rnn(rng, H)
        m = {k: np.zeros_like(v) for k, v in P.items()}
        v = {k: np.zeros_like(v_) for k, v_ in P.items()}
        for it in range(1, n_pas + 1):
            x, y = lot_memoire(rng, 128, T)
            hs, p = avant(P, x)
            G, _ = bptt(P, x, hs, p, y)
            # écrêtage de la norme globale du gradient à 1 (contre l'explosion)
            gn = np.sqrt(sum(float((g ** 2).sum()) for g in G.values()))
            s = min(1.0, 1.0 / (gn + 1e-12))
            for k in P:
                g = G[k] * s
                m[k] = 0.9 * m[k] + 0.1 * g
                v[k] = 0.999 * v[k] + 0.001 * g * g
                P[k] -= lr * (m[k] / (1 - 0.9 ** it)) / (np.sqrt(v[k] / (1 - 0.999 ** it)) + 1e-8)
        x, y = lot_memoire(np.random.default_rng(7), 2000, T)
        _, p = avant(P, x)
        return float((p.argmax(1) == y).mean())

    _t0 = time.perf_counter()
    precision_numpy = {}
    for _T in config["rnn_np_retards"]:
        precision_numpy[_T] = entrainer_rnn_numpy(_T, config["rnn_np_pas"], config["rnn_np_cachee"])
        print(f"RNN numpy, retard T = {_T:>3} : précision {precision_numpy[_T]:.1%}")
    duree_numpy = time.perf_counter() - _t0
    print(f"Partie 1 : {duree_numpy:.1f} s")
    return duree_numpy, precision_numpy


@app.cell
def _(mo):
    mo.md(r"""
    ## Partie 2 : la même tâche avec PyTorch, RNN contre LSTM

    `nn.RNN` fait exactement le calcul de la partie 1. `nn.LSTM` ajoute un **état de
    cellule** $c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t$ : quand la porte d'oubli
    $f_t$ reste proche de 1, le gradient traverse le temps presque sans perte. On initialise
    le biais de la porte d'oubli à 3 ($\sigma(3) \approx 0{,}95$) pour que la cellule
    commence par « retenir » ; avec 1, le LSTM résout encore $T = 20$ mais échoue à $T = 30$
    dans ce budget de 2 000 pas (25 %, le hasard ; à essayer : `biais_oubli` dans la configuration).
    """)


@app.cell
def _(D_ENTREE, K_SYMB, config, lot_memoire, nn, np, time, torch):
    # --- Tâche de mémoire : RNN contre LSTM pour des retards croissants (CPU) ---
    class ModeleMemoire(nn.Module):
        def __init__(self, type_rnn, H):
            super().__init__()
            self.rnn = getattr(nn, type_rnn)(D_ENTREE, H, batch_first=True)
            self.sortie = nn.Linear(H, K_SYMB)
            if type_rnn == "LSTM":
                with torch.no_grad():  # biais de la porte d'oubli (2e quart des biais)
                    self.rnn.bias_ih_l0[H:2 * H].fill_(config["biais_oubli"])
                    self.rnn.bias_hh_l0[H:2 * H].fill_(0.0)

        def forward(self, x):
            o, _ = self.rnn(x)
            return self.sortie(o[:, -1])

    def entrainer_memoire(type_rnn, T):
        torch.manual_seed(42)
        rng = np.random.default_rng(42)
        m = ModeleMemoire(type_rnn, config["memoire_cachee"])
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        resolu = None   # premier pas où la perte du lot passe sous 0,05
        recentes = []
        for pas in range(config["memoire_pas"]):
            x, y = lot_memoire(rng, 128, T)
            x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y)
            perte = nn.functional.cross_entropy(m(x), y)
            opt.zero_grad()
            perte.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if resolu is None and perte.item() < 0.05:
                resolu = pas
            recentes = (recentes + [perte.item()])[-50:]
            if len(recentes) == 50 and max(recentes) < 0.01:
                break   # résolu et stable depuis 50 pas : inutile de continuer
        x, y = lot_memoire(np.random.default_rng(7), 2000, T)
        with torch.no_grad():
            acc = (m(torch.tensor(x, dtype=torch.float32)).argmax(1) == torch.tensor(y)).float().mean()
        return float(acc), resolu

    _t0 = time.perf_counter()
    precision_memoire = {"RNN": {}, "LSTM": {}}
    pas_resolution = {"RNN": {}, "LSTM": {}}
    for _type in ["RNN", "LSTM"]:
        for _T in config["memoire_retards"]:
            _acc, _res = entrainer_memoire(_type, _T)
            precision_memoire[_type][_T] = _acc
            pas_resolution[_type][_T] = _res
            print(f"{_type:>4}, retard T = {_T:>3} : précision {_acc:.1%}"
                  + (f", résolu au pas {_res}" if _res is not None else ", jamais résolu"))
    duree_memoire = time.perf_counter() - _t0
    print(f"Partie 2 : {duree_memoire:.1f} s")
    return duree_memoire, pas_resolution, precision_memoire


@app.cell
def _(mo):
    mo.md(r"""
    ## Partie 3 : un modèle de langage par caractère sur TinyShakespeare

    Le réseau lit 128 caractères et, à chaque position, prédit le suivant (65 caractères
    possibles). Forçage par l'enseignant : l'entrée au pas $t$ est le **vrai** caractère
    $t$, jamais celui que le modèle aurait produit. Trois cellules de même taille (2 couches,
    256 unités) : RNN simple, GRU et LSTM, même nombre de pas et même graine.

    La perte est en **nats par caractère** ; $e^{\mathcal{L}}$ est la perplexité (nombre
    moyen de caractères « entre lesquels le modèle hésite »). Au hasard : $\ln 65 \approx 4{,}17$.
    """)


@app.cell
def _(charger_tinyshakespeare, torch):
    # --- Données : TinyShakespeare, découpé en caractères, 90 % entraînement / 10 % validation ---
    texte = charger_tinyshakespeare()
    caracteres = sorted(set(texte))
    V_CAR = len(caracteres)
    stoi = {c: i for i, c in enumerate(caracteres)}
    itos = {i: c for c, i in stoi.items()}
    _ids = torch.tensor([stoi[c] for c in texte], dtype=torch.long)
    _n = int(0.9 * len(_ids))
    ids_train, ids_val = _ids[:_n], _ids[_n:]
    print(f"{len(texte):,} caractères, vocabulaire de {V_CAR}".replace(",", " "))
    return V_CAR, ids_train, ids_val, itos, stoi


@app.cell
def _(V_CAR, config, device, ids_train, ids_val, math, nn, time, torch):
    # --- Modèles RNN / GRU / LSTM par caractère, entraînés à l'identique ---
    class ModeleCaracteres(nn.Module):
        def __init__(self, type_rnn, H, couches):
            super().__init__()
            self.emb = nn.Embedding(V_CAR, 64)
            self.rnn = getattr(nn, type_rnn)(64, H, num_layers=couches, batch_first=True)
            self.sortie = nn.Linear(H, V_CAR)

        def forward(self, x, etat=None):
            o, etat = self.rnn(self.emb(x), etat)
            return self.sortie(o), etat

    def lot_texte(donnees, g):
        L, B = config["lm_fenetre"], config["lm_lot"]
        debut = torch.randint(0, len(donnees) - L - 1, (B,), generator=g)
        x = torch.stack([donnees[i:i + L] for i in debut])
        y = torch.stack([donnees[i + 1:i + L + 1] for i in debut])   # décalé d'un caractère
        return x.to(device), y.to(device)

    @torch.no_grad()
    def perte_validation(m, n_lots=10):
        m.eval()
        g = torch.Generator().manual_seed(0)   # toujours les mêmes lots de validation
        total = 0.0
        for _ in range(n_lots):
            x, y = lot_texte(ids_val, g)
            lo, _ = m(x)
            total += nn.functional.cross_entropy(lo.reshape(-1, V_CAR), y.reshape(-1)).item()
        m.train()
        return total / n_lots

    def entrainer_lm(type_rnn):
        torch.manual_seed(42)
        g = torch.Generator().manual_seed(42)
        m = ModeleCaracteres(type_rnn, config["lm_cachee"], config["lm_couches"]).to(device)
        n_param = sum(p.numel() for p in m.parameters())
        opt = torch.optim.Adam(m.parameters(), lr=2e-3)
        courbe = []   # (pas, perte d'entraînement, perte de validation)
        t0 = time.perf_counter()
        for pas in range(1, config["lm_pas"] + 1):
            x, y = lot_texte(ids_train, g)
            lo, _ = m(x)
            perte = nn.functional.cross_entropy(lo.reshape(-1, V_CAR), y.reshape(-1))
            opt.zero_grad()
            perte.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if pas % config["lm_eval_tous"] == 0 or pas == 1:
                courbe.append((pas, perte.item(), perte_validation(m)))
        duree = time.perf_counter() - t0
        print(f"{type_rnn:>4} ({n_param / 1e6:.2f} M paramètres) : val {courbe[-1][2]:.3f} nats/car. "
              f"(perplexité {math.exp(courbe[-1][2]):.2f}) en {duree:.0f} s")
        return m, courbe, duree, n_param

    return entrainer_lm, perte_validation


@app.cell
def _(bouton, entrainer_lm, mo, time):
    mo.stop(mo.running_in_notebook() and not bouton.value)
    _t0 = time.perf_counter()
    modeles_lm, courbes_lm, durees_lm, params_lm = {}, {}, {}, {}
    for _type in ["RNN", "GRU", "LSTM"]:
        modeles_lm[_type], courbes_lm[_type], durees_lm[_type], params_lm[_type] = entrainer_lm(_type)
    duree_lm = time.perf_counter() - _t0
    print(f"Partie 3 : {duree_lm:.1f} s")
    return courbes_lm, duree_lm, modeles_lm, params_lm


@app.cell
def _(mo):
    mo.md(r"""
    ### Décoder : glouton, faisceau, tirage

    Le modèle donne une distribution sur le caractère suivant ; il reste à **choisir**.

    - **Glouton** : toujours le plus probable.
    - **Recherche en faisceau** de largeur $k$ : on garde à chaque pas les $k$ débuts de texte
      dont la log-probabilité totale est la plus haute ($k = 1$ redonne le glouton).
    - **Tirage** à température $\tau$ : on tire au hasard selon $\mathrm{softmax}(z/\tau)$.

    On mesure la log-probabilité moyenne par caractère du texte produit (plus haut = plus
    « probable » aux yeux du modèle) et la part de 4-grammes de caractères distincts (bas =
    le texte se répète).
    """)


@app.cell
def _(F, config, device, itos, stoi, torch):
    # --- Décodage glouton, recherche en faisceau, tirage ---
    AMORCE = "ROMEO:\n"

    @torch.no_grad()
    def amorcer(m, amorce):
        x = torch.tensor([[stoi[c] for c in amorce]], device=device)
        lo, etat = m(x)
        return F.log_softmax(lo[0, -1], -1), etat

    @torch.no_grad()
    def faisceau(m, amorce, longueur, k):
        """Recherche en faisceau : garde les k suites de log-probabilité totale maximale."""
        m.eval()
        lp, etat = amorcer(m, amorce)
        # Chaque hypothèse : (score cumulé, liste d'ids) ; les états sont empilés sur le lot.
        scores, idx = lp.topk(k)
        suites = [[int(i)] for i in idx]
        scores = scores.clone()
        etat = tuple(e.repeat(1, len(suites), 1) for e in etat) if isinstance(etat, tuple) \
            else etat.repeat(1, len(suites), 1)
        for _ in range(longueur - 1):
            x = torch.tensor([[s[-1]] for s in suites], device=device)
            lo, etat = m(x, etat)
            lp = F.log_softmax(lo[:, -1], -1)                     # (k, V)
            total = (scores[:, None] + lp).reshape(-1)             # toutes les extensions
            meilleurs, pos = total.topk(k)
            parent = (pos // lp.shape[1]).tolist()
            car = (pos % lp.shape[1]).tolist()
            suites = [suites[p] + [c] for p, c in zip(parent, car)]
            scores = meilleurs
            sel = torch.tensor(parent, device=device)
            etat = tuple(e[:, sel] for e in etat) if isinstance(etat, tuple) else etat[:, sel]
        meilleur = int(scores.argmax())
        return "".join(itos[i] for i in suites[meilleur]), float(scores[meilleur]) / longueur

    @torch.no_grad()
    def tirage(m, amorce, longueur, temperature, graine=42):
        m.eval()
        g = torch.Generator(device="cpu").manual_seed(graine)
        lp, etat = amorcer(m, amorce)
        ids, total = [], 0.0
        for _ in range(longueur):
            p = F.softmax(lp / temperature, -1).cpu()
            i = int(torch.multinomial(p, 1, generator=g))
            total += float(lp[i])           # log-probabilité sous le modèle (température 1)
            ids.append(i)
            lo, etat = m(torch.tensor([[i]], device=device), etat)
            lp = F.log_softmax(lo[0, -1], -1)
        return "".join(itos[i] for i in ids), total / longueur

    def part_4grammes_distincts(s):
        g = [s[i:i + 4] for i in range(len(s) - 3)]
        return len(set(g)) / max(1, len(g))

    return AMORCE, faisceau, part_4grammes_distincts, tirage


@app.cell
def _(
    AMORCE,
    config,
    faisceau,
    mo,
    bouton,
    modeles_lm,
    part_4grammes_distincts,
    time,
    tirage,
):
    mo.stop(mo.running_in_notebook() and not bouton.value)
    _t0 = time.perf_counter()
    _m = modeles_lm["LSTM"]
    _L = config["dec_longueur"]
    decodages = {}
    for _k in config["dec_largeurs"]:
        _txt, _lp = faisceau(_m, AMORCE, _L, _k)
        decodages[f"faisceau k={_k}" if _k > 1 else "glouton"] = (_txt, _lp, part_4grammes_distincts(_txt))
    for _tau in [1.0, 0.7]:
        _txt, _lp = tirage(_m, AMORCE, _L, _tau)
        decodages[f"tirage τ={_tau}"] = (_txt, _lp, part_4grammes_distincts(_txt))
    for _nom, (_txt, _lp, _d) in decodages.items():
        print(f"--- {_nom} : log-prob. moyenne {_lp:.3f} nats/car., 4-grammes distincts {_d:.0%}")
        print(AMORCE + _txt[:160])
    duree_decodage = time.perf_counter() - _t0
    print(f"Décodage : {duree_decodage:.1f} s")
    return decodages, duree_decodage


@app.cell
def _(mo):
    mo.md(r"""
    ## Partie 4 : séquence à séquence, avec et sans attention

    Tâche jouet qui oblige à « traduire » toute une phrase : **inverser** une suite de 4 à 30
    lettres (`abcd` → `dcba`). Un encodeur GRU bidirectionnel lit la source ; un décodeur
    GRU écrit la cible une lettre à la fois (forçage par l'enseignant à l'entraînement,
    glouton au test).

    - **Sans attention** (Sutskever et al., Cho et al., 2014) : le décodeur ne voit que le
      **dernier état** de l'encodeur, un vecteur de 128 nombres pour toute la phrase.
    - **Avec attention** (Bahdanau, Cho et Bengio, 2014) : à chaque pas $i$, le décodeur
      calcule un score $e_{ij} = v^\top \tanh(W s_{i-1} + U h_j)$ pour chaque position $j$
      de la source, les normalise par un softmax ($\alpha_{ij}$) et lit la moyenne pondérée
      $c_i = \sum_j \alpha_{ij} h_j$. La matrice des $\alpha_{ij}$ est la **carte
      d'alignement**.
    """)


@app.cell
def _(config, nn, torch):
    # --- Modèle séquence à séquence (encodeur GRU bidirectionnel, décodeur GRU) ---
    N_LETTRES = 20
    PAD, BOS, EOS = 0, 1, 2
    V_S2S = N_LETTRES + 3

    def lot_inversion(B, g, longueur=None):
        """Suites de lettres (ids 3..22) de longueur 4..lmax ; cible = suite inversée + EOS."""
        L = (torch.randint(4, config["s2s_lmax"] + 1, (B,), generator=g) if longueur is None
             else torch.full((B,), longueur))
        Lm = int(L.max())
        src = torch.full((B, Lm), PAD)
        tgt = torch.full((B, Lm + 1), PAD)
        for b in range(B):
            s = torch.randint(3, V_S2S, (int(L[b]),), generator=g)
            src[b, :len(s)] = s
            tgt[b, :len(s)] = s.flip(0)
            tgt[b, len(s)] = EOS
        return src, tgt

    class Seq2Seq(nn.Module):
        def __init__(self, attention, E=32, H=64):
            super().__init__()
            self.attention = attention
            self.emb_src = nn.Embedding(V_S2S, E, padding_idx=PAD)
            self.emb_tgt = nn.Embedding(V_S2S, E, padding_idx=PAD)
            self.encodeur = nn.GRU(E, H, batch_first=True, bidirectional=True)
            self.init = nn.Linear(2 * H, 2 * H)
            self.decodeur = nn.GRUCell(E + 2 * H, 2 * H)
            # attention additive de Bahdanau : e_ij = v . tanh(W s_{i-1} + U h_j)
            self.W = nn.Linear(2 * H, 2 * H, bias=False)
            self.U = nn.Linear(2 * H, 2 * H)
            self.v = nn.Linear(2 * H, 1, bias=False)
            self.sortie = nn.Linear(4 * H + E, V_S2S)

        def encoder(self, src):
            masque = src != PAD
            longueurs = masque.sum(1).cpu()
            paquet = nn.utils.rnn.pack_padded_sequence(self.emb_src(src), longueurs,
                                                       batch_first=True, enforce_sorted=False)
            o, h_fin = self.encodeur(paquet)
            o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=src.shape[1])
            resume = torch.cat([h_fin[0], h_fin[1]], 1)   # LE vecteur unique (avant + arrière)
            return o, resume, masque

        def pas(self, y_prec, s, o, resume, masque, Uo):
            e = self.emb_tgt(y_prec)
            if self.attention:
                score = self.v(torch.tanh(self.W(s)[:, None] + Uo)).squeeze(2)
                alpha = score.masked_fill(~masque, -1e9).softmax(1)
                contexte = (alpha[:, :, None] * o).sum(1)
            else:
                alpha, contexte = None, resume       # toujours le même vecteur
            s = self.decodeur(torch.cat([e, contexte], 1), s)
            return self.sortie(torch.cat([s, contexte, e], 1)), s, alpha

        def forward(self, src, tgt):
            """Forçage par l'enseignant : l'entrée au pas i est la VRAIE lettre i-1."""
            o, resume, masque = self.encoder(src)
            s, Uo = torch.tanh(self.init(resume)), self.U(o)
            y = torch.full((src.shape[0],), BOS, device=src.device)
            logits = []
            for i in range(tgt.shape[1]):
                lo, s, _ = self.pas(y, s, o, resume, masque, Uo)
                logits.append(lo)
                y = tgt[:, i]
            return torch.stack(logits, 1)

        @torch.no_grad()
        def glouton(self, src, n_pas):
            o, resume, masque = self.encoder(src)
            s, Uo = torch.tanh(self.init(resume)), self.U(o)
            y = torch.full((src.shape[0],), BOS, device=src.device)
            sorties, alphas = [], []
            for _ in range(n_pas):
                lo, s, a = self.pas(y, s, o, resume, masque, Uo)
                y = lo.argmax(1)        # au test, le modèle se nourrit de SES sorties
                sorties.append(y)
                alphas.append(a)
            return torch.stack(sorties, 1), alphas

    return BOS, EOS, PAD, Seq2Seq, lot_inversion


@app.cell
def _(PAD, Seq2Seq, bouton, config, lot_inversion, mo, nn, time, torch):
    mo.stop(mo.running_in_notebook() and not bouton.value)
    # CPU : petits réseaux, boucle Python pas à pas dans le décodeur (MPS n'y gagne rien)
    _t0 = time.perf_counter()
    modeles_s2s, courbes_s2s = {}, {}
    for _att in [False, True]:
        torch.manual_seed(42)
        _g = torch.Generator().manual_seed(42)
        _m = Seq2Seq(_att)
        _opt = torch.optim.Adam(_m.parameters(), lr=3e-3)
        _courbe = []
        for _pas in range(1, config["s2s_pas"] + 1):
            _src, _tgt = lot_inversion(64, _g)
            _lo = _m(_src, _tgt)
            _perte = nn.functional.cross_entropy(_lo.reshape(-1, _lo.shape[-1]), _tgt.reshape(-1),
                                                 ignore_index=PAD)
            _opt.zero_grad()
            _perte.backward()
            nn.utils.clip_grad_norm_(_m.parameters(), 1.0)
            _opt.step()
            _courbe.append(_perte.item())
        _nom = "avec attention" if _att else "sans attention"
        modeles_s2s[_nom], courbes_s2s[_nom] = _m.eval(), _courbe
        print(f"{_nom} : perte finale (moyenne des 100 derniers pas) "
              f"{sum(_courbe[-100:]) / len(_courbe[-100:]):.3f}")
    duree_s2s = time.perf_counter() - _t0
    print(f"Partie 4 (entraînement) : {duree_s2s:.1f} s")
    return courbes_s2s, duree_s2s, modeles_s2s


@app.cell
def _(config, lot_inversion, modeles_s2s, np, torch):
    # --- Précision (suite entière exacte) selon la longueur, et carte d'alignement ---
    longueurs_test = list(range(4, config["s2s_lmax"] + 1, 2))
    precision_s2s = {}
    for _nom, _m in modeles_s2s.items():
        _g = torch.Generator().manual_seed(7)
        precision_s2s[_nom] = []
        for _L in longueurs_test:
            _src, _tgt = lot_inversion(200, _g, _L)
            _pred, _ = _m.glouton(_src, _L + 1)
            precision_s2s[_nom].append(float((_pred == _tgt).all(1).float().mean()))
        print(_nom, " ".join(f"{L}:{a:.0%}" for L, a in zip(longueurs_test, precision_s2s[_nom])))

    # Carte d'alignement sur un exemple de longueur 12
    _g = torch.Generator().manual_seed(3)
    exemple_src, exemple_tgt = lot_inversion(1, _g, 12)
    _pred, _alphas = modeles_s2s["avec attention"].glouton(exemple_src, 13)
    carte_alignement = torch.stack([a[0] for a in _alphas]).numpy()   # (sorties, entrées)
    lettres = "abcdefghijklmnopqrst"
    exemple_entree = "".join(lettres[i - 3] for i in exemple_src[0].tolist())
    exemple_sortie = "".join(lettres[i - 3] if i >= 3 else "#" for i in _pred[0].tolist())
    # masse d'attention sur la « bonne » position (anti-diagonale)
    _n = len(exemple_entree)
    masse_diagonale = float(np.mean([carte_alignement[i, _n - 1 - i] for i in range(_n)]))
    print(f"Exemple : {exemple_entree} → {exemple_sortie} ; attention moyenne sur la lettre "
          f"à recopier : {masse_diagonale:.0%}")
    return (
        carte_alignement,
        exemple_entree,
        exemple_sortie,
        longueurs_test,
        masse_diagonale,
        precision_s2s,
    )


@app.cell
def _(
    SMOKE,
    carte_alignement,
    config,
    courbes_lm,
    exemple_entree,
    exemple_sortie,
    longueurs_test,
    mo,
    normes_gradient,
    np,
    plt,
    precision_memoire,
    precision_s2s,
    sauvegarder_figure,
):
    # --- Figures de l'exercice (le site en refait des versions SVG à partir de resultats.npz) ---
    _fig, _ax = plt.subplots(2, 2, figsize=(12, 8.5))
    _a = _ax[0, 0]
    for _r, _n in normes_gradient.items():
        _a.semilogy(_n, label=f"rayon spectral {_r}")
    _a.set(xlabel="pas avant la fin (k)", ylabel="|dL/dh| relatif", title="Gradient qui remonte le temps (RNN numpy)")
    _a.legend()
    _a = _ax[0, 1]
    for _type, _d in precision_memoire.items():
        _a.plot(list(_d), list(_d.values()), "o-", label=_type)
    _a.axhline(0.25, ls=":", c="gray")
    _a.set(xscale="log", xlabel="retard T", ylabel="précision", title="Tâche de mémoire")
    _a.legend()
    _a = _ax[1, 0]
    for _type, _c in courbes_lm.items():
        _c = np.array(_c)
        _a.plot(_c[:, 0], _c[:, 2], label=f"{_type} (val)")
    _a.set(xlabel="pas", ylabel="nats / caractère", title="Modèle de langage par caractère", ylim=(1.3, 3.5))
    _a.legend()
    _a = _ax[1, 1]
    for _nom, _p in precision_s2s.items():
        _a.plot(longueurs_test, _p, "o-", label=_nom)
    _a.set(xlabel="longueur de la suite", ylabel="suites exactes", title="Inverser une suite")
    _a.legend()
    _fig.tight_layout()
    _dossier = mo.notebook_dir() / "figures"
    sauvegarder_figure(_fig, "bilan-smoke" if SMOKE else "bilan", _dossier)

    _fig2, _a2 = plt.subplots(figsize=(5, 5))
    _a2.imshow(carte_alignement[:, :len(exemple_entree)], cmap="Greys", vmin=0, vmax=1)
    _a2.set_xticks(range(len(exemple_entree)), list(exemple_entree))
    _a2.set_yticks(range(len(exemple_sortie)), list(exemple_sortie))
    _a2.set(xlabel="entrée", ylabel="sortie", title="Carte d'alignement")
    _a2.grid(False)
    sauvegarder_figure(_fig2, "alignement-smoke" if SMOKE else "alignement", _dossier)
    plt.close("all")
    print(f"Figures : {_dossier}")
    return


@app.cell
def _(
    SEED,
    SMOKE,
    carte_alignement,
    config,
    courbes_lm,
    courbes_s2s,
    decodages,
    duree_decodage,
    duree_lm,
    duree_memoire,
    duree_numpy,
    duree_s2s,
    ecart_gradient,
    exemple_entree,
    exemple_sortie,
    json,
    longueurs_test,
    masse_diagonale,
    mo,
    normes_gradient,
    np,
    params_lm,
    pas_resolution,
    precision_memoire,
    precision_numpy,
    precision_s2s,
):
    # --- Résultats bruts pour les figures du site (site/scripts/figures/ch16_*.py) ---
    bilan = {
        "graine": SEED,
        "ecart_gradient": ecart_gradient,
        "precision_numpy": precision_numpy,
        "precision_memoire": precision_memoire,
        "pas_resolution": pas_resolution,
        "val_finale": {k: c[-1][2] for k, c in courbes_lm.items()},
        "params_lm": params_lm,
        "decodages": {k: {"texte": t, "logprob": lp, "distincts": d} for k, (t, lp, d) in decodages.items()},
        "precision_s2s": precision_s2s,
        "longueurs_test": longueurs_test,
        "exemple_entree": exemple_entree,
        "exemple_sortie": exemple_sortie,
        "masse_diagonale": masse_diagonale,
        "durees": {"numpy": duree_numpy, "memoire": duree_memoire, "lm": duree_lm,
                   "decodage": duree_decodage, "s2s": duree_s2s},
        "config": config,
    }
    _rayons = sorted(normes_gradient)
    np.savez_compressed(
        mo.notebook_dir() / "figures" / ("resultats-smoke.npz" if SMOKE else "resultats.npz"),
        bilan=json.dumps(bilan, ensure_ascii=False),
        rayons=np.array(_rayons),
        normes_gradient=np.stack([normes_gradient[r] for r in _rayons]),
        courbe_lm_rnn=np.array(courbes_lm["RNN"]),
        courbe_lm_gru=np.array(courbes_lm["GRU"]),
        courbe_lm_lstm=np.array(courbes_lm["LSTM"]),
        courbe_s2s_sans=np.array(courbes_s2s["sans attention"]),
        courbe_s2s_avec=np.array(courbes_s2s["avec attention"]),
        carte_alignement=carte_alignement,
    )
    return (bilan,)


@app.cell
def _(SMOKE, bilan, mo):
    # --- Bilan : seuils chiffrés du README. En smoke, on vérifie seulement que tout tourne.
    _pm = bilan["precision_memoire"]
    _vf = bilan["val_finale"]
    _s2s = bilan["precision_s2s"]
    seuils = {
        "BPTT numpy = autograd (écart < 1e-10)": bilan["ecart_gradient"] < 1e-10,
        "RNN numpy : précision ≥ 99 % à T = 10": bilan["precision_numpy"].get(10, 0) >= 0.99,
        "RNN torch : précision ≤ 35 % au plus grand retard": _pm["RNN"][max(_pm["RNN"])] <= 0.35,
        "LSTM : précision ≥ 99 % au plus grand retard": _pm["LSTM"][max(_pm["LSTM"])] >= 0.99,
        "LSTM : val ≤ 1,60 nats/car.": _vf["LSTM"] <= 1.60,
        "LSTM meilleur que RNN (val)": _vf["LSTM"] < _vf["RNN"],
        "faisceau k=8 plus probable que glouton": bilan["decodages"].get("faisceau k=8", {"logprob": -9})["logprob"]
        >= bilan["decodages"]["glouton"]["logprob"],
        "attention : ≥ 95 % de suites exactes à la longueur max": _s2s["avec attention"][-1] >= 0.95,
        "sans attention : ≤ 50 % à la longueur max": _s2s["sans attention"][-1] <= 0.50,
    }
    _lignes = "\n".join(f"- {'✅' if ok else '❌'} {nom}" for nom, ok in seuils.items())
    if SMOKE:   # réseaux à peine entraînés, configuration réduite : les ❌ sont attendus
        _lignes = "(smoke : seuils non vérifiés, donnés pour information)\n" + _lignes
    print(_lignes)
    if not SMOKE and not all(seuils.values()):
        raise SystemExit("Seuil non atteint :\n" + _lignes)
    mo.md(f"**Bilan**\n\n{_lignes}")


if __name__ == "__main__":
    app.run()
