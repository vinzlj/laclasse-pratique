# Exercice 21 : réseaux de neurones sur graphes (chapitre 21).
#
#   make ex NUM=21              → éditeur marimo (bouton « Entraîner »)
#   make ex NUM=21 MODE=run     → script complet, vérifie les seuils du README
#   make ex NUM=21 MODE=smoke   → test rapide (< 30 s)
#
# Progression : le club de karaté de Zachary (34 membres, 78 liens d'amitié) ; une étape de
# passage de messages calculée à la main ; propagation sans apprentissage ; un GCN à deux
# couches écrit en numpy avec son gradient à la main (vérifié par l'autograd de PyTorch),
# comparé à un MLP qui ignore le graphe ; le même GCN en PyTorch (matrice creuse,
# autograd) ; un GCN à poids aléatoires ; une couche d'attention sur graphe (GAT) ;
# l'attention du Transformer vue comme une attention sur le graphe complet ; enfin ce qui
# se passe quand on empile beaucoup de couches (lissage excessif).
#
# Tout tourne sur CPU : 34 nœuds, le GPU n'apporterait rien.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import torch
    import torch.nn.functional as F

    from ia_commun import PALETTE, is_smoke, sauvegarder_figure, seed_everything, style_commun

    return (
        F,
        PALETTE,
        is_smoke,
        json,
        mo,
        np,
        nx,
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
    # Exercice 21 : réseaux de neurones sur graphes (GNN)

    **Objectif.** Classer les membres d'un club de karaté entre deux camps en ne
    connaissant le camp que de **deux** personnes (le professeur et le président), grâce
    au graphe de leurs amitiés. On écrit un GCN (Kipf et Welling, 2016) en numpy, gradient
    compris, puis en PyTorch, et on le compare à un réseau qui ignore le graphe, à une
    couche d'attention sur graphe (GAT) et à l'attention du Transformer.

    **Ce qu'on minimise.** L'entropie croisée (chapitre 01) sur les **seuls nœuds
    étiquetés** $\mathcal{E}$ :
    $\mathcal{L} = -\frac{1}{|\mathcal{E}|}\sum_{i \in \mathcal{E}} \log \operatorname{softmax}(z_i)_{y_i}$.
    Les 32 autres nœuds n'entrent pas dans la perte : ils servent seulement de relais
    dans le graphe (apprentissage **semi-supervisé**).

    La couche GCN : $H' = \sigma\big(\hat A\, H\, W\big)$ avec
    $\hat A = \tilde D^{-1/2}(A + I)\,\tilde D^{-1/2}$, où $A$ est la matrice d'adjacence
    et $\tilde D$ la matrice diagonale des degrés de $A + I$.

    Plan : (1) le graphe et une étape de passage de messages à la main ; (2) propagation
    sans apprentissage ; (3) GCN numpy contre MLP sans graphe ; (4) le même GCN en
    PyTorch ; (5) robustesse au choix des deux nœuds étiquetés ; (6) GCN à poids
    aléatoires ; (7) GAT ; (8) le Transformer, une attention sur le graphe complet ;
    (9) profondeur et lissage excessif.
    """)


@app.cell
def _(is_smoke, seed_everything, style_commun, torch):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    # Calculs en float64 sur CPU : comparaison numpy / PyTorch au bit près ou presque.
    torch.set_default_dtype(torch.float64)
    config = dict(
        cachee=16,             # neurones de la couche cachée du GCN (valeur de Kipf et Welling)
        lr=0.01,               # Adam, comme dans l'article
        l2=5e-4,               # pénalité L2 : valeur de l'article, qui ne l'applique qu'à la 1re couche ; ici aux deux
        iterations=40 if SMOKE else 300,
        n_tirages=4 if SMOKE else 50,          # partie 5 : paires de nœuds étiquetés tirées au hasard
        n_aleatoires=3 if SMOKE else 50,       # partie 6 : GCN à poids aléatoires
        graines_gat=[0] if SMOKE else [0, 1, 2, 3, 4],
        tetes_gat=4,
        profondeurs=[1, 2, 4] if SMOKE else [1, 2, 3, 4, 6, 8, 16, 32],
        graines_profondeur=[0] if SMOKE else [0, 1, 2, 3, 4],
        iterations_profondeur=40 if SMOKE else 300,
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SEED, SMOKE, config


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Le graphe, et une étape de passage de messages à la main

    Wayne Zachary (1977) a étudié un club de karaté universitaire d'une soixantaine de
    membres qui a fini par se scinder en deux : le camp du professeur (« Mr. Hi », nœud 0)
    et celui du président (« Officer », nœud 33). Il n'a retenu que les 34 membres qui se
    fréquentaient en dehors des cours, et 78 liens entre eux (au moins une interaction hors
    des cours : on parlera d'« amitiés » pour faire court). `networkx` fournit ce graphe
    (34 nœuds, 78 arêtes) et le camp de chacun après la scission. On ignore ici le poids des arêtes : une amitié
    existe ou non.

    Les nœuds n'ont **aucune caractéristique** : on prend $X = I$ (chaque nœud est un
    vecteur one-hot, son identité et rien d'autre), comme Kipf et Welling dans l'annexe A
    de leur article. Toute l'information utile est donc dans le graphe.
    """)


@app.cell
def _(np, nx):
    # --- Données : le graphe du club de karaté ---
    G = nx.karate_club_graph()
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G, nodelist=range(n), weight=None)  # 0/1, symétrique
    y = np.array([0 if G.nodes[i]["club"] == "Mr. Hi" else 1 for i in range(n)])
    noms_camps = ["Mr. Hi (professeur)", "Officer (président)"]
    etiquetes = np.array([0, 33])  # les deux seuls nœuds dont on connaît le camp
    degres = A.sum(1).astype(int)

    # Â = D̃^(-1/2) (A + I) D̃^(-1/2) : on ajoute une boucle sur chaque nœud (il garde sa
    # propre information), puis on normalise par la racine des degrés des deux extrémités.
    A_tilde = A + np.eye(n)
    d_tilde = A_tilde.sum(1)
    A_hat = A_tilde / np.sqrt(np.outer(d_tilde, d_tilde))
    X = np.eye(n)

    print(f"{n} nœuds, {int(A.sum() / 2)} arêtes ; degré min {degres.min()}, max {degres.max()}, "
          f"moyen {degres.mean():.2f}")
    print(f"Camps : {np.bincount(y)[0]} pour Mr. Hi, {np.bincount(y)[1]} pour Officer")
    print(f"Degrés des nœuds étiquetés : nœud 0 → {degres[0]}, nœud 33 → {degres[33]}")
    return A, A_hat, A_tilde, G, X, d_tilde, degres, etiquetes, n, noms_camps, y


@app.cell
def _(A, A_hat, d_tilde, degres, np):
    # --- Une étape de passage de messages, nombre par nombre ---
    # Caractéristique scalaire : +1 sur le professeur (nœud 0), −1 sur le président
    # (nœud 33), 0 ailleurs. Une couche GCN sans poids calcule h'_i = Σ_j Â_ij h_j :
    # chaque nœud reçoit un message de chaque voisin (et de lui-même), pondéré par
    # 1/√((d_i+1)(d_j+1)).
    h0 = np.zeros(len(A))
    h0[0], h0[33] = 1.0, -1.0
    h1 = A_hat @ h0
    exemple = {}
    for _i in [11, 8, 2]:
        _voisins = np.flatnonzero(A[_i])
        print(f"Nœud {_i} (degré {degres[_i]}), voisins {_voisins.tolist()}")
        for _j in [0, 33]:
            if A[_i, _j]:
                print(f"   message de {_j} : Â[{_i},{_j}] × h[{_j}] = 1/√({d_tilde[_i]:.0f}×{d_tilde[_j]:.0f}) "
                      f"× {h0[_j]:+.0f} = {A_hat[_i, _j] * h0[_j]:+.4f}")
        print(f"   h'[{_i}] = {h1[_i]:+.4f}")
        exemple[_i] = float(h1[_i])
    print(f"Après une étape, {np.count_nonzero(h1)} nœuds sur {len(A)} ont reçu un message non nul")
    return exemple, h0


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Propagation sans apprentissage

    Avant tout réseau de neurones : on applique $k$ fois $\hat A$ au signal $\pm 1$ des deux
    nœuds étiquetés et on classe chaque nœud par le **signe** du résultat. Aucun paramètre,
    aucune perte. C'est la ligne de base « le graphe seul ».
    """)


@app.cell
def _(A_hat, h0, np, y):
    precision_propagation = {}
    _h = h0.copy()
    for _k in range(1, 9):
        _h = A_hat @ _h
        _atteints = np.count_nonzero(_h)
        _pred = (_h < 0).astype(int)  # négatif → camp du président
        precision_propagation[_k] = float(((_pred == y) & (_h != 0)).mean())
        print(f"k = {_k} : {_atteints:2d} nœuds atteints, précision {precision_propagation[_k]:.1%} "
              "(un nœud non atteint compte comme une erreur)")
    return (precision_propagation,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. GCN en numpy, contre un MLP qui ignore le graphe

    Deux couches : $Z = \hat A\,\mathrm{ReLU}(\hat A X W_1)\,W_2$, puis softmax.
    Le MLP est **le même code** avec $\hat A$ remplacée par l'identité : chaque nœud ne voit
    que lui-même. Même initialisation (Glorot), même optimiseur (Adam, pas 0,01, L2 5·10⁻⁴).

    Passe arrière (gradient de la perte, écrit à la main) :

    - $G_2 = \partial\mathcal{L}/\partial Z = (\operatorname{softmax}(Z) - Y) \odot M / |\mathcal{E}|$,
      où $M$ vaut 1 sur les lignes des nœuds étiquetés et 0 ailleurs ;
    - $\partial\mathcal{L}/\partial W_2 = (\hat A H_1)^\top G_2$ ;
    - $\partial\mathcal{L}/\partial H_1 = \hat A^\top G_2 W_2^\top$, puis on passe la ReLU :
      $G_1 = \partial\mathcal{L}/\partial H_1 \odot [\hat A X W_1 > 0]$ ;
    - $\partial\mathcal{L}/\partial W_1 = (\hat A X)^\top G_1$.

    Le seul ajout par rapport au MLP du chapitre 07 : la multiplication par $\hat A$, qui
    réapparaît (transposée) à la descente.
    """)


@app.cell
def _(A_hat, X, np, y):
    # --- GCN à deux couches en numpy : passe avant, perte, gradient à la main, Adam ---
    def glorot(rng, entree, sortie):
        borne = np.sqrt(6.0 / (entree + sortie))
        return rng.uniform(-borne, borne, (entree, sortie))

    def softmax(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def passe_avant(W1, W2, P):
        """P = Â pour le GCN, P = I pour le MLP sans graphe."""
        PX = P @ X
        Z1 = PX @ W1              # agrégation des voisins puis transformation linéaire
        H1 = np.maximum(Z1, 0.0)  # ReLU
        PH1 = P @ H1
        Z2 = PH1 @ W2             # logits, un vecteur de 2 nombres par nœud
        return dict(PX=PX, Z1=Z1, H1=H1, PH1=PH1, Z2=Z2, proba=softmax(Z2))

    def perte_et_gradient(W1, W2, P, lab, l2):
        c = passe_avant(W1, W2, P)
        proba = c["proba"]
        perte = -np.mean(np.log(proba[lab, y[lab]])) + 0.5 * l2 * ((W1 ** 2).sum() + (W2 ** 2).sum())
        G2 = np.zeros_like(proba)
        G2[lab] = proba[lab]
        G2[lab, y[lab]] -= 1.0
        G2 /= len(lab)
        gW2 = c["PH1"].T @ G2 + l2 * W2
        gH1 = P.T @ G2 @ W2.T
        G1 = gH1 * (c["Z1"] > 0)
        gW1 = c["PX"].T @ G1 + l2 * W1
        return perte, (gW1, gW2), c

    def entrainer_numpy(P, lab, iterations, lr, l2, cachee, graine, instantanes=()):
        rng = np.random.default_rng(graine)
        W = [glorot(rng, X.shape[1], cachee), glorot(rng, cachee, 2)]
        W_init = [w.copy() for w in W]
        m = [np.zeros_like(w) for w in W]
        v = [np.zeros_like(w) for w in W]
        b1, b2, eps = 0.9, 0.999, 1e-8   # Adam, valeurs par défaut
        histo = dict(perte=[], precision=[], logits={})
        for t in range(1, iterations + 1):
            perte, grads, c = perte_et_gradient(W[0], W[1], P, lab, l2)
            histo["perte"].append(perte)
            histo["precision"].append(float((c["proba"].argmax(1) == y).mean()))
            if t - 1 in instantanes:
                histo["logits"][t - 1] = c["Z2"].copy()
            for k in range(2):
                m[k] = b1 * m[k] + (1 - b1) * grads[k]
                v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
                W[k] -= lr * (m[k] / (1 - b1 ** t)) / (np.sqrt(v[k] / (1 - b2 ** t)) + eps)
        c = passe_avant(W[0], W[1], P)
        histo["logits"][iterations] = c["Z2"].copy()
        return W, W_init, c, histo

    return entrainer_numpy, glorot, passe_avant, perte_et_gradient


@app.cell
def _(A_hat, X, etiquetes, glorot, np, perte_et_gradient, torch, y):
    # --- Contrôle du gradient : le calcul à la main contre l'autograd de PyTorch ---
    _rng = np.random.default_rng(0)
    _W1, _W2 = glorot(_rng, 34, 16), glorot(_rng, 16, 2)
    _perte, (_g1, _g2), _ = perte_et_gradient(_W1, _W2, A_hat, etiquetes, 5e-4)
    _tW1 = torch.tensor(_W1, requires_grad=True)
    _tW2 = torch.tensor(_W2, requires_grad=True)
    _P, _X = torch.tensor(A_hat), torch.tensor(X)
    _Z = _P @ torch.relu(_P @ _X @ _tW1) @ _tW2
    _tperte = torch.nn.functional.cross_entropy(_Z[etiquetes], torch.tensor(y[etiquetes])) \
        + 0.5 * 5e-4 * ((_tW1 ** 2).sum() + (_tW2 ** 2).sum())
    _tperte.backward()
    ecart_gradient = max(np.abs(_g1 - _tW1.grad.numpy()).max(), np.abs(_g2 - _tW2.grad.numpy()).max())
    print(f"Perte : numpy {_perte:.10f}, PyTorch {_tperte.item():.10f}")
    print(f"Écart maximal entre gradients numpy et autograd : {ecart_gradient:.2e}")
    return (ecart_gradient,)


@app.cell
def _(mo):
    # Le bouton empêche les entraînements de se relancer à chaque modification d'une cellule amont.
    bouton = mo.ui.run_button(label="Entraîner")
    bouton
    return (bouton,)


@app.cell
def _(A_hat, bouton, config, entrainer_numpy, etiquetes, mo, n, np, perte_et_gradient, time, y):
    # En script (make ex MODE=run|smoke), il n'y a personne pour cliquer : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Entraîner » pour lancer les entraînements.*"))
    T0 = time.perf_counter()  # chronomètre de l'ensemble des calculs
    _args = dict(lab=etiquetes, iterations=config["iterations"], lr=config["lr"],
                 l2=config["l2"], cachee=config["cachee"], graine=42)
    instantanes = [0, 10, 25, 50, 100, 200]
    W_gcn, W_gcn_init, sortie_gcn, histo_gcn = entrainer_numpy(A_hat, instantanes=instantanes, **_args)
    W_mlp, W_mlp_init, sortie_mlp, histo_mlp = entrainer_numpy(np.eye(n), **_args)
    pred_gcn = sortie_gcn["proba"].argmax(1)
    pred_mlp = sortie_mlp["proba"].argmax(1)
    precision_gcn = float((pred_gcn == y).mean())
    precision_mlp = float((pred_mlp == y).mean())
    erreurs_gcn = np.flatnonzero(pred_gcn != y)
    print(f"GCN numpy : perte finale {histo_gcn['perte'][-1]:.4f}, précision {precision_gcn:.1%} "
          f"({(pred_gcn == y).sum()}/{n}) ; nœuds mal classés : {erreurs_gcn.tolist()}")
    print(f"MLP sans graphe : perte finale {histo_mlp['perte'][-1]:.4f}, précision {precision_mlp:.1%} "
          f"({(pred_mlp == y).sum()}/{n})")
    # Pourquoi le MLP échoue : avec X = I, la ligne i de W1 est le plongement du nœud i, et
    # l'entropie croisée ne lui envoie un gradient que si le nœud i est étiqueté. Les 32
    # autres lignes ne reçoivent que le gradient de la pénalité L2, qu'Adam transforme en
    # pas de taille presque fixe (≈ lr) : en 300 itérations, elles tombent pratiquement à
    # zéro. Avec Â, le gradient atteint toutes les lignes des voisins (à deux sauts) des
    # nœuds étiquetés.
    for _nom, _P in [("MLP", np.eye(n)), ("GCN", A_hat)]:
        _, (_g1, _), _ = perte_et_gradient(*W_gcn_init, _P, etiquetes, 0.0)
        _lignes = np.flatnonzero(np.abs(_g1).sum(1) > 0)
        print(f"{_nom} : {len(_lignes)} lignes de W1 sur {n} reçoivent un gradient de la perte")
    _autres = np.setdiff1d(np.arange(n), etiquetes)
    _norme = lambda W: np.linalg.norm(W[_autres], axis=1)
    print(f"MLP : norme médiane des 32 lignes non étiquetées de W1, {np.median(_norme(W_mlp_init[0])):.3f} "
          f"au départ, {np.median(_norme(W_mlp[0])):.1e} après {config['iterations']} itérations "
          f"(lignes 0 et 33 : {np.linalg.norm(W_mlp[0][etiquetes], axis=1).round(2).tolist()})")
    return (
        T0,
        W_gcn,
        W_gcn_init,
        erreurs_gcn,
        histo_gcn,
        histo_mlp,
        instantanes,
        pred_gcn,
        pred_mlp,
        precision_gcn,
        precision_mlp,
        sortie_gcn,
    )


@app.cell
def _(A, erreurs_gcn, np, y):
    # Le nœud mal classé : combien de ses amis dans chaque camp ?
    for _i in erreurs_gcn:
        _v = np.flatnonzero(A[_i])
        print(f"Nœud {_i} : camp réel {y[_i]}, {np.sum(y[_v] == 0)} amis chez Mr. Hi, "
              f"{np.sum(y[_v] == 1)} chez Officer (voisins {_v.tolist()})")


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Le même GCN en PyTorch

    Même modèle, même initialisation, mais : $\hat A$ stockée en **matrice creuse**
    (seules les 78 × 2 + 34 = 190 cases non nulles sur 34 × 34 = 1 156), gradient par
    autograd, optimiseur `torch.optim.Adam`. Les bibliothèques spécialisées (PyTorch
    Geometric, DGL) font la même chose à grande échelle.
    """)


@app.cell
def _(A_hat, F, W_gcn, W_gcn_init, X, config, etiquetes, np, torch, y):
    class GCN(torch.nn.Module):
        def __init__(self, W1, W2):
            super().__init__()
            self.W1 = torch.nn.Parameter(torch.tensor(W1))
            self.W2 = torch.nn.Parameter(torch.tensor(W2))

        def forward(self, A_creuse, x):
            h = torch.relu(torch.sparse.mm(A_creuse, x @ self.W1))
            return torch.sparse.mm(A_creuse, h @ self.W2)

    A_creuse = torch.tensor(A_hat).to_sparse()
    _modele = GCN(*W_gcn_init)
    # weight_decay de torch.optim.Adam = gradient de la pénalité 0,5·l2·‖W‖², comme en numpy
    _opt = torch.optim.Adam(_modele.parameters(), lr=config["lr"], weight_decay=config["l2"])
    _x, _y, _lab = torch.tensor(X), torch.tensor(y), torch.tensor(etiquetes)
    for _ in range(config["iterations"]):
        _perte = F.cross_entropy(_modele(A_creuse, _x)[_lab], _y[_lab])
        _opt.zero_grad()
        _perte.backward()
        _opt.step()
    with torch.no_grad():
        _pred_torch = _modele(A_creuse, _x).argmax(1).numpy()
    ecart_torch = max(np.abs(_modele.W1.detach().numpy() - W_gcn[0]).max(),
                      np.abs(_modele.W2.detach().numpy() - W_gcn[1]).max())
    precision_torch = float((_pred_torch == y).mean())
    print(f"Cases non nulles de Â : {A_creuse._nnz()} sur {A_hat.size}")
    print(f"PyTorch : précision {precision_torch:.1%} ; écart maximal des poids avec numpy après "
          f"{config['iterations']} itérations : {ecart_torch:.2e}")
    return GCN, ecart_torch, precision_torch


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Et si on étiquetait deux autres personnes ?

    Le professeur et le président sont les deux nœuds les plus connectés : le cas est
    favorable. On tire au hasard une personne par camp, 50 fois, et on compare le GCN, le
    MLP sans graphe et la propagation sans apprentissage ($k = 4$).
    """)


@app.cell
def _(A_hat, config, entrainer_numpy, n, np, y):
    _rng = np.random.default_rng(42)
    tirages = dict(gcn=[], mlp=[], propagation=[], paires=[])
    for _t in range(config["n_tirages"]):
        _lab = np.array([_rng.choice(np.flatnonzero(y == 0)), _rng.choice(np.flatnonzero(y == 1))])
        _args = dict(lab=_lab, iterations=config["iterations"], lr=config["lr"],
                     l2=config["l2"], cachee=config["cachee"], graine=42)
        tirages["gcn"].append(float((entrainer_numpy(A_hat, **_args)[2]["proba"].argmax(1) == y).mean()))
        tirages["mlp"].append(float((entrainer_numpy(np.eye(n), **_args)[2]["proba"].argmax(1) == y).mean()))
        _h = np.zeros(n)
        _h[_lab[0]], _h[_lab[1]] = 1.0, -1.0
        _h = np.linalg.matrix_power(A_hat, 4) @ _h
        tirages["propagation"].append(float(((_h < 0).astype(int) == y).mean()))
        tirages["paires"].append(_lab.tolist())
    for _k in ["gcn", "propagation", "mlp"]:
        _v = np.array(tirages[_k])
        print(f"{_k:12s} : précision moyenne {_v.mean():.1%}, médiane {np.median(_v):.1%}, "
              f"min {_v.min():.1%}, max {_v.max():.1%}")
    return (tirages,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Un GCN qui n'a rien appris

    L'annexe A de Kipf et Welling : trois couches GCN à poids **aléatoires**
    $Z = \tanh(\hat A \tanh(\hat A \tanh(\hat A X W_0) W_1) W_2)$, dimensions 4, 4, 2, sans
    aucun entraînement. On mesure si les deux camps sont **linéairement séparables** dans le
    plan $Z$ (régression logistique ajustée sur les 34 nœuds : c'est une mesure de la
    représentation, pas une évaluation de classifieur). Témoin : le même réseau sans
    $\hat A$.
    """)


@app.cell
def _(A_hat, X, config, glorot, n, np, y):
    from sklearn.linear_model import LogisticRegression

    def gcn_aleatoire(P, graine):
        rng = np.random.default_rng(graine)
        h = X
        for d in [4, 4, 2]:
            h = np.tanh(P @ h @ glorot(rng, h.shape[1], d))
        return h

    def separabilite(Z):
        clf = LogisticRegression(C=1e4, max_iter=5000).fit(Z, y)
        return float(clf.score(Z, y))

    aleatoire = dict(gcn=[], sans_graphe=[])
    for _g in range(config["n_aleatoires"]):
        aleatoire["gcn"].append(separabilite(gcn_aleatoire(A_hat, _g)))
        aleatoire["sans_graphe"].append(separabilite(gcn_aleatoire(np.eye(n), _g)))
    # Tirage « représentatif » pour la figure : celui dont la séparabilité du GCN est la
    # plus proche de la médiane des tirages (mêmes poids pour la version sans graphe).
    tirage_figure = int(np.argmin(np.abs(np.array(aleatoire["gcn"]) - np.median(aleatoire["gcn"]))))
    Z_aleatoire = gcn_aleatoire(A_hat, tirage_figure)
    Z_aleatoire_sans = gcn_aleatoire(np.eye(n), tirage_figure)
    print(f"Tirage montré en figure : n° {tirage_figure} (séparabilité GCN "
          f"{aleatoire['gcn'][tirage_figure]:.1%}, sans graphe {aleatoire['sans_graphe'][tirage_figure]:.1%})")
    for _k, _v in aleatoire.items():
        print(f"Poids aléatoires, {_k:11s} : séparabilité linéaire moyenne {np.mean(_v):.1%} "
              f"(min {np.min(_v):.1%}, max {np.max(_v):.1%}, {len(_v)} tirages)")
    return Z_aleatoire, Z_aleatoire_sans, aleatoire, tirage_figure


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Attention sur graphe (GAT)

    Le GCN donne à chaque voisin un poids **fixé par les degrés**. Le GAT (Veličković et
    al., 2017) **apprend** ce poids : pour chaque arête $i \leftarrow j$, un score
    $e_{ij} = \mathrm{LeakyReLU}\big(a^\top [W h_i \,\|\, W h_j]\big)$, puis un softmax sur
    les **seuls voisins** de $i$ (et $i$ lui-même) :
    $\alpha_{ij} = \exp(e_{ij}) / \sum_{k \in \mathcal{N}(i)} \exp(e_{ik})$, et
    $h'_i = \sum_j \alpha_{ij} W h_j$. Ici 4 têtes de 8 dimensions concaténées, ELU, puis
    une tête de sortie. Même perte, mêmes deux nœuds étiquetés.
    """)


@app.cell
def _(A_tilde, F, X, config, etiquetes, np, torch, y):
    class CoucheGAT(torch.nn.Module):
        def __init__(self, entree, sortie, tetes):
            super().__init__()
            self.W = torch.nn.Linear(entree, sortie * tetes, bias=False)
            # a = [a_dst ; a_src] : le score se découpe en une part du nœud qui reçoit
            # et une part du voisin qui envoie
            self.a_dst = torch.nn.Parameter(torch.empty(tetes, sortie))
            self.a_src = torch.nn.Parameter(torch.empty(tetes, sortie))
            torch.nn.init.xavier_uniform_(self.W.weight)
            torch.nn.init.xavier_uniform_(self.a_dst)
            torch.nn.init.xavier_uniform_(self.a_src)
            self.tetes, self.sortie = tetes, sortie

        def forward(self, x, masque):
            Wh = self.W(x).view(-1, self.tetes, self.sortie)            # (n, têtes, d)
            e = (Wh * self.a_dst).sum(-1)[:, None, :] + (Wh * self.a_src).sum(-1)[None, :, :]
            e = F.leaky_relu(e, 0.2)                                    # (n_i, n_j, têtes)
            e = e.masked_fill(~masque[:, :, None], float("-inf"))       # seulement les voisins
            alpha = torch.softmax(e, dim=1)
            return torch.einsum("ijh,jhd->ihd", alpha, Wh), alpha

    masque_graphe = torch.tensor(A_tilde > 0)
    _x, _y, _lab = torch.tensor(X), torch.tensor(y), torch.tensor(etiquetes)
    resultats_gat = []
    for _g in config["graines_gat"]:
        torch.manual_seed(_g)
        _c1 = CoucheGAT(34, 8, config["tetes_gat"])
        _c2 = CoucheGAT(8 * config["tetes_gat"], 2, 1)
        _opt = torch.optim.Adam([*_c1.parameters(), *_c2.parameters()], lr=0.01, weight_decay=5e-4)
        for _ in range(config["iterations"]):
            _h, _alpha1 = _c1(_x, masque_graphe)
            _sortie, _ = _c2(F.elu(_h.reshape(34, -1)), masque_graphe)
            _perte = F.cross_entropy(_sortie.reshape(34, 2)[_lab], _y[_lab])
            _opt.zero_grad()
            _perte.backward()
            _opt.step()
        resultats_gat.append(float((_sortie.reshape(34, 2).argmax(1).numpy() == y).mean()))
    # Poids d'attention appris par le dernier modèle (graine 4 en mode complet), première couche,
    # première tête, pour le nœud 0
    alpha_noeud0 = _alpha1[0, :, 0].detach().numpy()[A_tilde[0] > 0]
    _k = len(alpha_noeud0)
    print(f"GAT : précision par graine {[f'{p:.1%}' for p in resultats_gat]}, moyenne {np.mean(resultats_gat):.1%}")
    print(f"Nœud 0 ({_k} voisins, lui compris) : α de {alpha_noeud0.min():.3f} à {alpha_noeud0.max():.3f} "
          f"(uniforme : {1 / _k:.3f}) ; poids GCN fixes de {(1 / np.sqrt(17 * A_tilde.sum(1)[A_tilde[0] > 0])).min():.3f} "
          f"à {(1 / np.sqrt(17 * A_tilde.sum(1)[A_tilde[0] > 0])).max():.3f}")
    return alpha_noeud0, masque_graphe, resultats_gat


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Le Transformer : une attention sur le graphe complet

    L'attention du Transformer (chapitre 20) calcule
    $\operatorname{softmax}(QK^\top/\sqrt{d_k})\,V$ entre **toutes** les paires de tokens.
    Écrite comme une attention sur graphe dont le masque est le graphe **complet**, elle
    donne exactement le résultat de `F.scaled_dot_product_attention`. Avec le masque du
    club de karaté, on obtient une attention restreinte aux voisins (un « Transformer de
    graphe »). En simplifiant : le GAT score les paires avec un petit réseau
    (LeakyReLU de $a^\top[Wh_i\|Wh_j]$), le Transformer avec un produit scalaire ; le
    principe « softmax sur les voisins, somme pondérée des messages » est le même.
    """)


@app.cell
def _(F, masque_graphe, np, torch):
    def attention_sur_graphe(q, k, v, masque):
        scores = (q @ k.transpose(-1, -2)) / np.sqrt(q.shape[-1])
        scores = scores.masked_fill(~masque, float("-inf"))
        return torch.softmax(scores, dim=-1) @ v

    torch.manual_seed(0)
    _q, _k, _v = torch.randn(3, 1, 34, 8).unbind(0)
    _complet = torch.ones(34, 34, dtype=torch.bool)
    ecart_transformer = (attention_sur_graphe(_q, _k, _v, _complet)
                         - F.scaled_dot_product_attention(_q, _k, _v)).abs().max().item()
    ecart_masque = (attention_sur_graphe(_q, _k, _v, masque_graphe)
                    - F.scaled_dot_product_attention(_q, _k, _v, attn_mask=masque_graphe)).abs().max().item()
    _diff = (attention_sur_graphe(_q, _k, _v, masque_graphe)
             - F.scaled_dot_product_attention(_q, _k, _v)).abs().max().item()
    print(f"Graphe complet contre scaled_dot_product_attention : écart {ecart_transformer:.1e}")
    print(f"Graphe du club contre scaled_dot_product_attention(attn_mask=graphe) : écart {ecart_masque:.1e}")
    print(f"Graphe du club contre graphe complet : écart {_diff:.2f} (ce ne sont plus les mêmes voisins)")
    print(f"Paires (i, j) traitées : graphe complet {_complet.sum().item()}, graphe du club "
          f"{masque_graphe.sum().item()}")
    return ecart_masque, ecart_transformer


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Empiler les couches : le lissage excessif

    Chaque couche élargit d'un saut le voisinage vu par un nœud. Mais $\hat A$ est une
    moyenne pondérée : appliquée $k$ fois, elle rend les représentations de plus en plus
    semblables. On mesure (a) le cosinus moyen entre les lignes de $\hat A^k X$ pour
    toutes les paires de nœuds ; (b) la précision de GCN de 1 à 32 couches (16 neurones
    par couche cachée, 5 graines), sans et avec connexion résiduelle
    $H' = \mathrm{ReLU}(\hat A H W) + H$ (variante testée par Kipf et Welling en annexe B).
    """)


@app.cell
def _(A_hat, F, X, config, etiquetes, n, np, torch, y):
    # (a) cosinus moyen entre représentations après k propagations
    cosinus_lissage = {}
    for _k in [1, 2, 4, 8, 16, 32, 64]:
        _H = np.linalg.matrix_power(A_hat, _k) @ X
        _Hn = _H / np.linalg.norm(_H, axis=1, keepdims=True)
        cosinus_lissage[_k] = float((_Hn @ _Hn.T)[np.triu_indices(n, 1)].mean())
    print("Cosinus moyen entre nœuds après k propagations : "
          + ", ".join(f"k={k} : {c:.3f}" for k, c in cosinus_lissage.items()))

    # (b) GCN profonds
    _P, _x, _y, _lab = torch.tensor(A_hat), torch.tensor(X), torch.tensor(y), torch.tensor(etiquetes)

    def gcn_profond(couches, graine, residuel):
        torch.manual_seed(graine)
        dims = [34] + [config["cachee"]] * (couches - 1) + [2]
        lin = [torch.nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(couches)]
        opt = torch.optim.Adam([p for c in lin for p in c.parameters()], lr=config["lr"],
                               weight_decay=config["l2"])
        for _ in range(config["iterations_profondeur"]):
            h = _x
            for i, c in enumerate(lin):
                z = _P @ c(h)
                if i < couches - 1:
                    z = torch.relu(z)
                    if residuel and i > 0:   # résiduel entre couches cachées de même taille
                        z = z + h
                h = z
            perte = F.cross_entropy(h[_lab], _y[_lab])
            opt.zero_grad()
            perte.backward()
            opt.step()
        return float((h.argmax(1).numpy() == y).mean()), perte.item()

    profondeur = dict(couches=config["profondeurs"], simple=[], residuel=[], perte_simple=[])
    for _L in config["profondeurs"]:
        _s = [gcn_profond(_L, _g, False) for _g in config["graines_profondeur"]]
        _r = [gcn_profond(_L, _g, True) for _g in config["graines_profondeur"]]
        profondeur["simple"].append([a for a, _ in _s])
        profondeur["perte_simple"].append([p for _, p in _s])
        profondeur["residuel"].append([a for a, _ in _r])
        print(f"{_L:2d} couches : sans résiduel {np.mean([a for a, _ in _s]):.1%} "
              f"(perte d'entraînement {np.mean([p for _, p in _s]):.3f}), "
              f"avec résiduel {np.mean([a for a, _ in _r]):.1%}")
    return cosinus_lissage, profondeur


@app.cell
def _(
    A,
    G,
    PALETTE,
    SMOKE,
    Z_aleatoire,
    Z_aleatoire_sans,
    cosinus_lissage,
    etiquetes,
    histo_gcn,
    histo_mlp,
    mo,
    np,
    nx,
    plt,
    pred_gcn,
    pred_mlp,
    profondeur,
    sauvegarder_figure,
    y,
):
    # --- Figures de l'exercice (les figures du site sont refaites par site/scripts/figures/) ---
    _pos = nx.spring_layout(G, seed=42)
    _coul = np.array([PALETTE[0], PALETTE[3]])
    fig1, _axes = plt.subplots(2, 2, figsize=(11, 9))
    for _ax, _pred, _titre in [(_axes[0, 0], pred_mlp, "MLP sans graphe"), (_axes[0, 1], pred_gcn, "GCN")]:
        nx.draw_networkx_edges(G, _pos, ax=_ax, alpha=0.3)
        nx.draw_networkx_nodes(G, _pos, ax=_ax, node_color=_coul[_pred].tolist(), node_size=180,
                               edgecolors=["black" if p == t else "gold" for p, t in zip(_pred, y)],
                               linewidths=[1 if p == t else 3 for p, t in zip(_pred, y)])
        nx.draw_networkx_labels(G, _pos, ax=_ax, font_size=7, font_color="white")
        _ax.set_title(f"{_titre} : {(_pred == y).mean():.0%} (contour jaune = erreur)")
        _ax.axis("off")
    _ax = _axes[1, 0]
    _ax.plot(histo_gcn["perte"], label="GCN")
    _ax.plot(histo_mlp["perte"], label="MLP sans graphe")
    _ax.set(xlabel="itération", ylabel="perte (2 nœuds étiquetés)", yscale="log",
            title="Perte d'entraînement")
    _ax.legend()
    _ax = _axes[1, 1]
    _c = profondeur["couches"]
    _ax.plot(_c, np.mean(profondeur["simple"], 1), "o-", label="GCN")
    _ax.plot(_c, np.mean(profondeur["residuel"], 1), "s-", label="GCN résiduel")
    _ax.plot(list(cosinus_lissage), list(cosinus_lissage.values()), "^--", label="cosinus moyen de Â^k X")
    _ax.set(xscale="log", xlabel="nombre de couches (ou de propagations k)", ylabel="précision / cosinus",
            title="Profondeur et lissage excessif")
    _ax.legend()
    fig1.tight_layout()
    _dossier = mo.notebook_dir() / "figures"
    print("Figure :", sauvegarder_figure(fig1, "gcn-karate-smoke" if SMOKE else "gcn-karate", _dossier)[0])

    fig2, _axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for _ax, _Z, _titre in [(_axes[0], Z_aleatoire_sans, "sans graphe (Â = I)"), (_axes[1], Z_aleatoire, "GCN")]:
        for _i, _j in zip(*np.nonzero(np.triu(A))):
            _ax.plot(_Z[[_i, _j], 0], _Z[[_i, _j], 1], color="grey", alpha=0.2, lw=0.6)
        _ax.scatter(_Z[:, 0], _Z[:, 1], c=_coul[y].tolist(), s=40, edgecolors="black", linewidths=0.5)
        _ax.scatter(_Z[etiquetes, 0], _Z[etiquetes, 1], s=160, facecolors="none", edgecolors="black")
        _ax.set_title(f"Poids aléatoires, {_titre}")
    fig2.tight_layout()
    print("Figure :", sauvegarder_figure(fig2, "aleatoire-smoke" if SMOKE else "aleatoire", _dossier)[0])
    fig1


@app.cell
def _(
    SMOKE,
    T0,
    aleatoire,
    alpha_noeud0,
    config,
    cosinus_lissage,
    ecart_gradient,
    ecart_masque,
    ecart_torch,
    ecart_transformer,
    erreurs_gcn,
    etiquetes,
    exemple,
    histo_gcn,
    histo_mlp,
    instantanes,
    json,
    mo,
    np,
    precision_gcn,
    precision_mlp,
    precision_propagation,
    precision_torch,
    pred_gcn,
    pred_mlp,
    profondeur,
    resultats_gat,
    sortie_gcn,
    time,
    tirage_figure,
    tirages,
    Z_aleatoire,
    Z_aleatoire_sans,
):
    # --- Bilan : seuils du README. En smoke, on vérifie seulement que tout tourne. ---
    duree_totale = time.perf_counter() - T0
    bilan = dict(
        precision_gcn=precision_gcn, precision_mlp=precision_mlp, precision_torch=precision_torch,
        erreurs_gcn=erreurs_gcn.tolist(), ecart_gradient=ecart_gradient, ecart_torch=ecart_torch,
        precision_propagation=precision_propagation,
        tirages={k: (float(np.mean(v)), float(np.min(v)), float(np.max(v)))
                 for k, v in tirages.items() if k != "paires"},
        aleatoire={k: float(np.mean(v)) for k, v in aleatoire.items()},
        aleatoire_figure={k: float(v[tirage_figure]) for k, v in aleatoire.items()},
        gat=resultats_gat, alpha_noeud0=(float(alpha_noeud0.min()), float(alpha_noeud0.max())),
        ecart_transformer=ecart_transformer, ecart_masque=ecart_masque,
        cosinus_lissage=cosinus_lissage, exemple_message=exemple,
        profondeur={str(L): (float(np.mean(s)), float(np.mean(r)))
                    for L, s, r in zip(profondeur["couches"], profondeur["simple"], profondeur["residuel"])},
        duree=duree_totale,
    )
    print(json.dumps(bilan, ensure_ascii=False, indent=1))
    # Résultats bruts pour les figures du site (site/scripts/figures/ch21_*.py)
    np.savez_compressed(
        mo.notebook_dir() / "figures" / ("resultats-smoke.npz" if SMOKE else "resultats.npz"),
        pred_gcn=pred_gcn, pred_mlp=pred_mlp, proba_gcn=sortie_gcn["proba"], etiquetes=etiquetes,
        perte_gcn=np.array(histo_gcn["perte"]), perte_mlp=np.array(histo_mlp["perte"]),
        precision_gcn_histo=np.array(histo_gcn["precision"]),
        precision_mlp_histo=np.array(histo_mlp["precision"]),
        instantanes=np.array(instantanes + [config["iterations"]]),
        logits=np.stack([histo_gcn["logits"][t] for t in instantanes + [config["iterations"]]
                         if t in histo_gcn["logits"]]),
        Z_aleatoire=Z_aleatoire, Z_aleatoire_sans=Z_aleatoire_sans,
        couches=np.array(profondeur["couches"]), prof_simple=np.array(profondeur["simple"]),
        prof_residuel=np.array(profondeur["residuel"]),
        cos_k=np.array(list(cosinus_lissage)), cos_valeur=np.array(list(cosinus_lissage.values())),
        bilan=json.dumps(bilan, ensure_ascii=False),
    )
    _seuils = [
        ("gradient numpy = autograd (écart < 1e-10)", ecart_gradient < 1e-10),
        ("GCN numpy ≥ 94 % (32/34)", precision_gcn >= 0.94),
        ("MLP sans graphe ≤ 65 %", precision_mlp <= 0.65),
        ("PyTorch = numpy (poids à moins de 1e-8)", ecart_torch < 1e-8),
        ("tirages au hasard : GCN moyen ≥ MLP moyen + 25 points",
         np.mean(tirages["gcn"]) >= np.mean(tirages["mlp"]) + 0.25),
        ("poids aléatoires : séparabilité GCN ≥ sans graphe + 10 points",
         np.mean(aleatoire["gcn"]) >= np.mean(aleatoire["sans_graphe"]) + 0.10),
        ("GAT ≥ 80 % en moyenne", np.mean(resultats_gat) >= 0.80),
        ("Transformer = attention sur graphe complet (écart < 1e-12)", ecart_transformer < 1e-12),
        ("cosinus moyen de Â^32 X ≥ 0,99", cosinus_lissage[32] >= 0.99),
        ("GCN à 16 couches ≤ 60 % et résiduel à 16 couches ≥ 85 %",
         16 in profondeur["couches"]
         and np.mean(profondeur["simple"][profondeur["couches"].index(16)]) <= 0.60
         and np.mean(profondeur["residuel"][profondeur["couches"].index(16)]) >= 0.85),
    ]
    _echecs = [nom for nom, ok in _seuils if not ok]
    if not SMOKE and _echecs:
        raise SystemExit("Seuils non atteints : " + " ; ".join(_echecs))
    mo.md(f"**Bilan** ({duree_totale:.1f} s de calcul) : GCN {precision_gcn:.1%}, MLP sans graphe "
          f"{precision_mlp:.1%}, GAT {np.mean(resultats_gat):.1%} ; "
          + ("mode smoke, seuils non vérifiés." if SMOKE else f"{len(_seuils)} seuils atteints."))


if __name__ == "__main__":
    app.run()
