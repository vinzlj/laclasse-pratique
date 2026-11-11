# Exercice 20a : le Transformer, de l'attention en NumPy au mini-GPT sur TinyShakespeare.
#
#   make ex NUM=20a              → éditeur marimo (bouton « Entraîner »)
#   make ex NUM=20a MODE=run     → script complet, vérifie les seuils du README
#   make ex NUM=20a MODE=smoke   → test rapide (< 30 s)
#
# Partie A en NumPy (attention, masque causal, √d_k, LayerNorm/RMSNorm, bloc complet,
# encodage de position), vérifiée contre PyTorch. Partie B : mini-GPT (ia_commun.mini_gpt)
# entraîné sur TinyShakespeare, checkpoint sauvé dans data/checkpoints/ (réutilisé par les
# exercices des chapitres 28 et 38). Partie C : deux ablations (sans masque causal, sans
# position) et le coût quadratique de l'attention.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.special import erf

    from ia_commun import (
        checkpoints_dir,
        get_device,
        is_smoke,
        sauvegarder_figure,
        seed_everything,
        style_commun,
    )
    from ia_commun.donnees import charger_tinyshakespeare
    from ia_commun.mini_gpt import (
        ConfigGPT,
        MiniGPT,
        TokeniseurCaracteres,
        estimer_loss,
        lot_aleatoire,
    )

    return (
        ConfigGPT,
        F,
        MiniGPT,
        TokeniseurCaracteres,
        charger_tinyshakespeare,
        checkpoints_dir,
        erf,
        estimer_loss,
        get_device,
        is_smoke,
        lot_aleatoire,
        math,
        mo,
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
    # Exercice 20a : le Transformer, de l'attention au mini-GPT

    Un RNN lit une phrase mot après mot : pour relier le 50ᵉ mot au 1ᵉʳ, l'information doit
    traverser 49 étapes, et l'entraînement ne peut pas traiter les positions en parallèle.
    Le Transformer (Vaswani et al., 2017) relie **directement** chaque position à toutes les
    autres par l'**attention** :

    $$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

    Plan :

    - **A. L'attention à la main (NumPy).** Un exemple de 4 tokens calculable à la main,
      le masque causal, le rôle de $\sqrt{d_k}$, LayerNorm et RMSNorm, un bloc Transformer
      complet comparé à PyTorch, l'équivariance par permutation, l'encodage sinusoïdal.
    - **B. Un mini-GPT sur TinyShakespeare.** Modèle de langage caractère par caractère
      (4 couches, 4 têtes, 256 dimensions, contexte 128) ; comparaison avec des modèles
      unigramme et bigramme ; génération ; cartes d'attention des 16 têtes.
    - **C. Ablations.** Retirer le masque causal (le modèle triche), retirer les positions,
      mesurer le coût quadratique de l'attention.

    **Perte optimisée** (partie B) : l'entropie croisée du token suivant,
    $\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_{t+1}\mid x_{\le t})$
    (chapitre 01). La **perplexité** vaut $e^{\mathcal{L}}$ : le nombre de caractères
    « équiprobables » entre lesquels le modèle hésite en moyenne. La partie A n'optimise
    rien : les poids y sont fixés à la main ou tirés au hasard.
    """)


@app.cell
def _(get_device, is_smoke, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    device = get_device()
    style_commun()
    config = dict(
        # Mini-GPT : configuration par défaut de ia_commun.mini_gpt.ConfigGPT
        n_couches=4, n_tetes=4, n_embd=256, taille_bloc=128, dropout=0.1,
        taille_lot=64,
        n_pas=30 if SMOKE else 1500,           # ≈ 3 min sur M1 Pro (MPS)
        lr_max=2e-3, lr_min=2e-4, pas_chauffe=100,   # montée linéaire puis décroissance en cosinus
        eval_tous=10 if SMOKE else 100,
        n_lots_eval=2 if SMOKE else 20,        # 20 lots de 64 × 128 = 164 000 caractères
        part_train=0.9,
        # Ablations (partie C), même planning de taux d'apprentissage, tronqué
        n_pas_sans_masque=10 if SMOKE else 300,
        n_pas_sans_position=10 if SMOKE else 500,
        longueurs_cout=[128, 256, 512] if SMOKE else [128, 256, 512, 1024, 2048],
        n_generes=200 if SMOKE else 500,
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, périphérique {device}")
    print(config)
    return SEED, SMOKE, config, device


@app.cell
def _(mo):
    mo.md(r"""
    ## A. L'attention à la main

    ### A1. Un exemple de quatre tokens

    Phrase : « le chat noir dort ». Au lieu de les apprendre, on écrit $Q$ et $K$ à la main,
    avec $d_k = 4$ dimensions qui ont un sens : *nom*, *adjectif*, *déterminant*, *verbe*.

    - La **clé** $K_j$ dit ce que le token $j$ **est** : « chat » est un nom → $(2, 0, 0, 0)$.
    - La **requête** $Q_i$ dit ce que le token $i$ **cherche** : « dort » cherche son sujet,
      un nom → $(2, 0, 0, 0)$ ; « chat » cherche un adjectif → $(0, 2, 0, 0)$.
    - La **valeur** $V_j$ est ce que le token $j$ **transmet** s'il est choisi, ici deux
      traits : *animal* et *noir*.

    Score = $Q_i\cdot K_j/\sqrt{d_k}$ = $Q_i\cdot K_j/2$. Pour « dort » : 4/2 = 2 avec « chat »,
    0 avec les autres. Softmax de (0, 2, 0, 0) : $e^2/(e^2+3) = 7{,}389/10{,}389 = 0{,}711$
    pour « chat », $1/10{,}389 = 0{,}096$ pour chacun des trois autres.
    """)


@app.cell
def _(np):
    TOKENS_JOUET = ["le", "chat", "noir", "dort"]
    # Dimensions : nom, adjectif, déterminant, verbe (mêmes nombres que la démo Attention)
    Q_JOUET = np.array([[2, 0, 0, 0],    # le   cherche un nom
                        [0, 2, 0, 0],    # chat cherche un adjectif
                        [2, 0, 0, 0],    # noir cherche un nom
                        [2, 0, 0, 0]],   # dort cherche un nom (son sujet)
                       dtype=np.float64)
    K_JOUET = np.array([[0, 0, 2, 0],    # le   est un déterminant
                        [2, 0, 0, 0],    # chat est un nom
                        [0, 2, 0, 0],    # noir est un adjectif
                        [0, 0, 0, 2]],   # dort est un verbe
                       dtype=np.float64)
    V_JOUET = np.array([[0, 0],          # traits transmis : (animal, noir)
                        [1, 0],
                        [0, 1],
                        [0, 0]], dtype=np.float64)

    def softmax(S, axis=-1):
        """Softmax stable : on retire le maximum de chaque ligne avant l'exponentielle."""
        S = S - S.max(axis=axis, keepdims=True)
        E = np.exp(S)
        return E / E.sum(axis=axis, keepdims=True)

    def attention_numpy(Q, K, V, causal=False, diviser=True):
        """softmax(QKᵀ/√d_k)·V, terme à terme. Q, K : (..., T, d_k) ; V : (..., T, d_v).

        Renvoie (sortie, poids). Masque causal : la position i ne voit que j ≤ i
        (les scores des positions futures sont mis à −∞ avant le softmax).
        """
        d_k = Q.shape[-1]
        S = Q @ np.swapaxes(K, -1, -2)
        if diviser:
            S = S / np.sqrt(d_k)
        if causal:
            T = Q.shape[-2]
            futur = np.triu(np.ones((T, T), dtype=bool), k=1)   # j > i
            S = np.where(futur, -np.inf, S)
        A = softmax(S)
        return A @ V, A

    return K_JOUET, Q_JOUET, TOKENS_JOUET, V_JOUET, attention_numpy, softmax


@app.cell
def _(F, K_JOUET, Q_JOUET, TOKENS_JOUET, V_JOUET, attention_numpy, np, torch):
    np.set_printoptions(precision=3, suppress=True)
    _S = Q_JOUET @ K_JOUET.T / np.sqrt(4)
    sortie_jouet, poids_jouet = attention_numpy(Q_JOUET, K_JOUET, V_JOUET)
    sortie_jouet_c, poids_jouet_c = attention_numpy(Q_JOUET, K_JOUET, V_JOUET, causal=True)
    print("Scores QKᵀ/√d_k (ligne = requête, colonne = clé) :\n", _S)
    print("Poids d'attention, sans masque :\n", poids_jouet)
    print("Sortie (animal, noir), sans masque :\n", sortie_jouet)
    print("Poids d'attention, masque causal :\n", poids_jouet_c)
    print("Sortie, masque causal :\n", sortie_jouet_c)

    # Vérification contre l'implémentation fusionnée de PyTorch (float64, CPU)
    _q, _k, _v = (torch.from_numpy(a)[None] for a in (Q_JOUET, K_JOUET, V_JOUET))
    _ref = F.scaled_dot_product_attention(_q, _k, _v)[0].numpy()
    _ref_c = F.scaled_dot_product_attention(_q, _k, _v, is_causal=True)[0].numpy()
    ecart_jouet = max(np.abs(_ref - sortie_jouet).max(), np.abs(_ref_c - sortie_jouet_c).max())
    print(f"Écart NumPy / PyTorch : {ecart_jouet:.1e}")
    for _i, _t in enumerate(TOKENS_JOUET):
        print(f"  « {_t} » regarde surtout « {TOKENS_JOUET[poids_jouet[_i].argmax()]} »"
              f" ({poids_jouet[_i].max():.3f}) ; avec masque : « {TOKENS_JOUET[poids_jouet_c[_i].argmax()]} »"
              f" ({poids_jouet_c[_i].max():.3f})")
    return ecart_jouet, poids_jouet, poids_jouet_c


@app.cell
def _(mo):
    mo.md(r"""
    Avec le masque causal, « chat » ne peut plus voir « noir » (qui vient après) : il
    partage son attention entre « le » et lui-même (0,5 / 0,5). « le » ne voit que lui-même.
    C'est la règle d'un **modèle de langage** : pour prédire la suite, on n'a pas le droit de
    la regarder.

    ### A2. Pourquoi diviser par $\sqrt{d_k}$ ?

    Si les composantes de $q$ et $k$ sont indépendantes, de moyenne 0 et de variance 1,
    $q\cdot k$ est une somme de $d_k$ termes de variance 1 : sa variance vaut $d_k$. Avec
    $d_k = 64$ (une tête du Transformer de 2017), les scores ont un écart-type de 8 : le
    softmax devient presque un *argmax*, et son gradient s'annule. Diviser par $\sqrt{d_k}$
    ramène l'écart-type à 1, quelle que soit la dimension.
    """)


@app.cell
def _(np, softmax):
    _rng = np.random.default_rng(42)
    dims_racine = [4, 16, 64, 256, 1024]
    racine_dk = {"d_k": [], "ecart_type_brut": [], "ecart_type_divise": [],
                 "poids_max_brut": [], "poids_max_divise": []}
    for _d in dims_racine:
        # 2 000 requêtes face à 16 clés, composantes ~ N(0, 1)
        _q = _rng.standard_normal((2000, 1, _d))
        _k = _rng.standard_normal((2000, 16, _d))
        _s = (_q @ np.swapaxes(_k, 1, 2))[:, 0, :]        # (2000, 16)
        racine_dk["d_k"].append(_d)
        racine_dk["ecart_type_brut"].append(_s.std())
        racine_dk["ecart_type_divise"].append((_s / np.sqrt(_d)).std())
        racine_dk["poids_max_brut"].append(softmax(_s).max(axis=1).mean())
        racine_dk["poids_max_divise"].append(softmax(_s / np.sqrt(_d)).max(axis=1).mean())
    print(" d_k | écart-type q·k | idem /√d_k | poids max moyen (16 clés) brut | /√d_k")
    for _i, _d in enumerate(dims_racine):
        print(f"{_d:5d}| {racine_dk['ecart_type_brut'][_i]:14.2f} | {racine_dk['ecart_type_divise'][_i]:10.2f} |"
              f" {racine_dk['poids_max_brut'][_i]:30.3f} | {racine_dk['poids_max_divise'][_i]:.3f}")
    return (racine_dk,)


@app.cell
def _(mo):
    mo.md(r"""
    ### A3. LayerNorm et RMSNorm

    Dans chaque bloc, avant (pré-normalisation, GPT-2 et suivants) ou après (Transformer de
    2017) chaque sous-couche, on **normalise** le vecteur de chaque position :

    - **LayerNorm** (Ba et al., 2016) : $\mathrm{LN}(x) = \gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}}+\beta$,
      moyenne et variance prises sur les $d$ composantes **d'un même token** (pas sur le lot,
      contrairement à la BatchNorm du chapitre 12) ;
    - **RMSNorm** (Zhang et Sennrich, 2019) : $\mathrm{RMS}(x) = \gamma\odot\frac{x}{\sqrt{\frac1d\sum_i x_i^2+\varepsilon}}$,
      sans recentrage : un peu moins de calcul, même effet en pratique (Llama, Qwen…).

    Exemple à la main : $x = (2, 4, 6, 8)$. Moyenne 5, variance $(9+1+1+9)/4 = 5$ :
    LN $= (-3, -1, 1, 3)/\sqrt5 = (-1{,}342 ; -0{,}447 ; 0{,}447 ; 1{,}342)$.
    RMS $=\sqrt{(4+16+36+64)/4} = \sqrt{30} = 5{,}477$ : RMSNorm $= (0{,}365 ; 0{,}730 ; 1{,}095 ; 1{,}461)$.
    """)


@app.cell
def _(np, torch):
    def layernorm_numpy(x, gamma=None, beta=None, eps=1e-5):
        """Normalise chaque vecteur (dernière dimension) : moyenne 0, variance 1."""
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)          # variance « biaisée » (÷ d), comme PyTorch
        y = (x - mu) / np.sqrt(var + eps)
        if gamma is not None:
            y = y * gamma
        if beta is not None:
            y = y + beta
        return y

    def rmsnorm_numpy(x, gamma=None, eps=1e-6):
        """Divise chaque vecteur par sa moyenne quadratique, sans le recentrer."""
        y = x / np.sqrt((x**2).mean(axis=-1, keepdims=True) + eps)
        return y if gamma is None else y * gamma

    _x = np.array([[2.0, 4.0, 6.0, 8.0]])
    ln_exemple = layernorm_numpy(_x)[0]
    rms_exemple = rmsnorm_numpy(_x)[0]
    print("LayerNorm(2, 4, 6, 8) =", ln_exemple)
    print("RMSNorm(2, 4, 6, 8)   =", rms_exemple)

    # Contre PyTorch, sur 10 vecteurs aléatoires de dimension 256, gamma et beta aléatoires
    _rng = np.random.default_rng(42)
    _X = _rng.standard_normal((10, 256)) * 3 + 1
    _g, _b = _rng.standard_normal(256), _rng.standard_normal(256)
    _ln = torch.nn.LayerNorm(256, dtype=torch.float64)
    _rms = torch.nn.RMSNorm(256, eps=1e-6, dtype=torch.float64)
    with torch.no_grad():
        _ln.weight.copy_(torch.from_numpy(_g))
        _ln.bias.copy_(torch.from_numpy(_b))
        _rms.weight.copy_(torch.from_numpy(_g))
        _Xt = torch.from_numpy(_X)
        ecart_normes = max(
            np.abs(_ln(_Xt).numpy() - layernorm_numpy(_X, _g, _b)).max(),
            np.abs(_rms(_Xt).numpy() - rmsnorm_numpy(_X, _g)).max(),
        )
    print(f"Écart NumPy / torch.nn.LayerNorm et torch.nn.RMSNorm : {ecart_normes:.1e}")
    return ecart_normes, layernorm_numpy, ln_exemple, rms_exemple


@app.cell
def _(mo):
    mo.md(r"""
    ### A4. Un bloc Transformer complet, en NumPy

    On réécrit en NumPy un bloc du mini-GPT de `ia_commun` (celui qu'on entraîne en
    partie B), on lui copie les poids d'un bloc PyTorch initialisé au hasard, et on compare
    les sorties. Le bloc est **pré-normalisé** :

    $$x \leftarrow x + \mathrm{Attn}(\mathrm{LN}_1(x)), \qquad x \leftarrow x + \mathrm{MLP}(\mathrm{LN}_2(x))$$

    Attention multi-têtes : $Q, K, V$ = projections de $\mathrm{LN}_1(x)$ découpées en
    4 têtes de 64 dimensions ; chaque tête calcule son attention causale ; les 4 sorties
    sont recollées puis projetées. MLP : $256 \to 1024 \to 256$ avec l'activation GELU.
    """)


@app.cell
def _(ConfigGPT, attention_numpy, config, erf, layernorm_numpy, np, torch):
    from ia_commun.mini_gpt import Bloc

    def bloc_numpy(x, p, n_tetes):
        """Un bloc pré-normalisé du mini-GPT. x : (T, C) ; p : poids (tableaux NumPy)."""
        T, C = x.shape
        dh = C // n_tetes
        # 1) attention multi-têtes causale
        h = layernorm_numpy(x, p["ln1.weight"])
        qkv = h @ p["attn.qkv.weight"].T                 # nn.Linear : y = x Wᵀ
        q, k, v = np.split(qkv, 3, axis=1)                # (T, C) chacun
        q, k, v = (a.reshape(T, n_tetes, dh).transpose(1, 0, 2) for a in (q, k, v))  # (têtes, T, dh)
        y, _ = attention_numpy(q, k, v, causal=True)
        y = y.transpose(1, 0, 2).reshape(T, C)            # on recolle les têtes
        x = x + y @ p["attn.proj.weight"].T               # connexion résiduelle
        # 2) MLP avec GELU exacte : x Φ(x) = x (1 + erf(x/√2)) / 2
        h = layernorm_numpy(x, p["ln2.weight"]) @ p["mlp.fc.weight"].T
        h = 0.5 * h * (1 + erf(h / np.sqrt(2)))
        return x + h @ p["mlp.proj.weight"].T

    torch.manual_seed(42)
    _cfg = ConfigGPT(n_embd=config["n_embd"], n_tetes=config["n_tetes"], dropout=0.0)
    _bloc = Bloc(_cfg).double().eval()
    with torch.no_grad():
        for _nom, _param in _bloc.named_parameters():   # poids plus grands que l'init GPT-2 : test plus exigeant
            _param.normal_(0, 0.1) if "ln" not in _nom else _param.uniform_(0.5, 1.5)
    _poids = {n: t.detach().numpy() for n, t in _bloc.state_dict().items()}
    _x = np.random.default_rng(42).standard_normal((16, config["n_embd"]))
    with torch.no_grad():
        _ref = _bloc(torch.from_numpy(_x)[None])[0].numpy()
    ecart_bloc = np.abs(bloc_numpy(_x, _poids, config["n_tetes"]) - _ref).max()
    n_param_bloc = sum(p.numel() for p in _bloc.parameters())
    print(f"Bloc : {n_param_bloc:_} paramètres ; écart NumPy / PyTorch : {ecart_bloc:.1e}".replace("_", " "))
    return ecart_bloc, n_param_bloc


@app.cell
def _(mo):
    mo.md(r"""
    ### A5. Sans position, l'attention ne voit qu'un sac de tokens

    Mélangez l'ordre des tokens d'entrée : sans masque et sans information de position, la
    sortie de l'attention est **la même, mélangée de la même façon** (équivariance par
    permutation). « le chien mord l'homme » et « l'homme mord le chien » donneraient les
    mêmes vecteurs. D'où l'**encodage de position**, ajouté aux plongements des tokens.
    Le masque causal, lui, casse cette symétrie : la position 1 voit 1 token, la position
    10 en voit 10.
    """)


@app.cell
def _(attention_numpy, np):
    _rng = np.random.default_rng(42)
    _X = _rng.standard_normal((6, 8))
    _Wq, _Wk, _Wv = (_rng.standard_normal((8, 8)) for _ in range(3))
    _perm = _rng.permutation(6)

    def _att(X, causal):
        return attention_numpy(X @ _Wq, X @ _Wk, X @ _Wv, causal=causal)[0]

    ecart_permutation = np.abs(_att(_X[_perm], False) - _att(_X, False)[_perm]).max()
    ecart_permutation_causal = np.abs(_att(_X[_perm], True) - _att(_X, True)[_perm]).max()
    print(f"Permutation {_perm} : sans masque, écart {ecart_permutation:.1e} (équivariance) ;"
          f" avec masque causal, écart {ecart_permutation_causal:.2f}")
    return ecart_permutation, ecart_permutation_causal


@app.cell
def _(mo):
    mo.md(r"""
    ### A6. L'encodage sinusoïdal (2017)

    $PE(p, 2i) = \sin(p/10000^{2i/d})$, $PE(p, 2i+1) = \cos(p/10000^{2i/d})$ : chaque paire
    de dimensions tourne à sa propre fréquence. Deux propriétés, vérifiées ci-dessous :

    1. le produit scalaire $PE(p)\cdot PE(p+k)$ ne dépend que du décalage $k$, pas de $p$ ;
    2. $PE(p+k)$ s'obtient à partir de $PE(p)$ par une **rotation** qui ne dépend que de $k$.
       RoPE (2021) applique cette rotation directement à $Q$ et $K$.

    Le mini-GPT de la partie B, comme GPT-2, **apprend** ses vecteurs de position à la place.
    """)


@app.cell
def _(np):
    def encodage_sinusoidal(n_positions, d):
        """Matrice (n_positions, d) de Vaswani et al. (2017), section 3.5."""
        p = np.arange(n_positions)[:, None]
        i = np.arange(d // 2)[None, :]
        angles = p / 10000 ** (2 * i / d)
        pe = np.zeros((n_positions, d))
        pe[:, 0::2] = np.sin(angles)
        pe[:, 1::2] = np.cos(angles)
        return pe

    pe_sinus = encodage_sinusoidal(128, 64)
    # 1) produit scalaire en fonction du décalage, pour deux positions de départ
    _k = 10
    print(f"PE(0)·PE(10) = {pe_sinus[0] @ pe_sinus[_k]:.4f} ; PE(50)·PE(60) = {pe_sinus[50] @ pe_sinus[50 + _k]:.4f}")
    # 2) rotation : chaque paire (sin, cos) de fréquence ω tourne de l'angle ω k
    _omega = 1 / 10000 ** (2 * np.arange(32) / 64)
    _c, _s = np.cos(_omega * _k), np.sin(_omega * _k)
    _sin, _cos = pe_sinus[50, 0::2], pe_sinus[50, 1::2]
    _rot = np.empty(64)
    _rot[0::2] = _sin * _c + _cos * _s       # sin(a + b) = sin a cos b + cos a sin b
    _rot[1::2] = _cos * _c - _sin * _s       # cos(a + b) = cos a cos b − sin a sin b
    ecart_rotation = np.abs(_rot - pe_sinus[60]).max()
    print(f"Rotation de PE(50) d'un décalage 10 contre PE(60) : écart {ecart_rotation:.1e}")
    return ecart_rotation, pe_sinus


@app.cell
def _(mo):
    mo.md(r"""
    ## B. Un mini-GPT sur TinyShakespeare

    ### B1. Données et modèles de référence

    TinyShakespeare : 1,1 million de caractères, 65 caractères distincts. Chaque caractère
    est un **token** (la tokenisation en sous-mots est le sujet du chapitre 19). On garde
    les 90 % du début pour l'entraînement, les 10 % de la fin pour la validation.

    Avant d'entraîner un Transformer, trois références, sur l'ensemble de validation :

    - **uniforme** : chaque caractère a la probabilité 1/65 : $\mathcal{L} = \ln 65 = 4{,}17$ ;
    - **unigramme** : fréquence de chaque caractère dans l'ensemble d'entraînement ;
    - **bigramme** : probabilité du caractère suivant sachant le caractère courant
      (comptage, lissage +1) : une mémoire d'un seul caractère.
    """)


@app.cell
def _(TokeniseurCaracteres, charger_tinyshakespeare, config, np, torch):
    texte = charger_tinyshakespeare()
    tok = TokeniseurCaracteres(texte)
    _ids = np.array(tok.encoder(texte), dtype=np.int64)
    _n = int(config["part_train"] * len(_ids))
    ids_train, ids_val = torch.from_numpy(_ids[:_n]), torch.from_numpy(_ids[_n:])
    V = tok.taille_vocab
    _milliers = lambda n: f"{n:,}".replace(",", " ")   # 1 115 394
    print(f"{_milliers(len(texte))} caractères, vocabulaire {V} ; train {_milliers(len(ids_train))}, val {_milliers(len(ids_val))}")

    # Modèles de référence, perte mesurée sur TOUT l'ensemble de validation
    _tr, _va = _ids[:_n], _ids[_n:]
    _uni = np.bincount(_tr, minlength=V) + 1.0
    _uni /= _uni.sum()
    _bi = np.ones((V, V))
    np.add.at(_bi, (_tr[:-1], _tr[1:]), 1)
    _bi /= _bi.sum(axis=1, keepdims=True)
    references = {
        "uniforme": float(np.log(V)),
        "unigramme": float(-np.log(_uni[_va]).mean()),
        "bigramme": float(-np.log(_bi[_va[:-1], _va[1:]]).mean()),
    }
    for _nom, _l in references.items():
        print(f"  {_nom:10s} perte {_l:.3f}  perplexité {np.exp(_l):5.1f}")
    return V, ids_train, ids_val, references, texte, tok


@app.cell
def _(mo):
    mo.md(r"""
    ### B2. Le modèle et son entraînement

    `ia_commun.mini_gpt.MiniGPT` : plongements des caractères + plongements de position
    appris → 4 blocs pré-normalisés (attention causale 4 têtes, MLP GELU) → LayerNorm
    finale → projection vers 65 logits (poids partagés avec les plongements d'entrée).

    Entraînement : AdamW (chapitre 02), lots de 64 fenêtres de 128 caractères tirées au
    hasard, taux d'apprentissage qui monte linéairement pendant 100 pas puis décroît en
    cosinus, gradient écrêté à norme 1. **Toutes les positions d'une fenêtre sont prédites
    en un seul passage** : 64 × 128 = 8 192 prédictions par pas, en parallèle. Un RNN
    devrait dérouler les 128 positions l'une après l'autre.
    """)


@app.cell
def _(ConfigGPT, MiniGPT, V, config, device, math, torch):
    def creer_modele(sans_position=False):
        torch.manual_seed(42)
        m = MiniGPT(ConfigGPT(taille_vocab=V, taille_bloc=config["taille_bloc"],
                              n_couches=config["n_couches"], n_tetes=config["n_tetes"],
                              n_embd=config["n_embd"], dropout=config["dropout"])).to(device)
        if sans_position:   # ablation : vecteurs de position nuls et gelés
            with torch.no_grad():
                m.emb_positions.weight.zero_()
            m.emb_positions.weight.requires_grad_(False)
        return m

    def taux(pas):
        """Montée linéaire puis cosinus, définie sur les n_pas du modèle principal."""
        if pas < config["pas_chauffe"]:
            return config["lr_max"] * (pas + 1) / config["pas_chauffe"]
        r = (pas - config["pas_chauffe"]) / max(1, config["n_pas"] - config["pas_chauffe"])
        return config["lr_min"] + 0.5 * (config["lr_max"] - config["lr_min"]) * (1 + math.cos(math.pi * min(r, 1.0)))

    _m = creer_modele()
    n_parametres = _m.nb_parametres()
    print(f"Mini-GPT : {n_parametres:,} paramètres (hors positions)".replace(",", " "))
    return creer_modele, n_parametres, taux


@app.cell
def _(estimer_loss, lot_aleatoire, time, torch):
    def entrainer(modele, donnees_train, donnees_val, n_pas, taux, cfg, device,
                  perte_honnete=None, journal="", ):
        """Boucle AdamW avec évaluation régulière. Renvoie un dictionnaire d'historiques.

        `perte_honnete(modele)` : mesure supplémentaire facultative (ablation sans masque).
        """
        params = [p for p in modele.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(params, lr=cfg["lr_max"], betas=(0.9, 0.99), weight_decay=0.1)
        gen = torch.Generator().manual_seed(42)
        h = {"pas": [], "train": [], "val": [], "honnete": [], "perte_lots": []}
        modele.train()
        t0 = time.perf_counter()
        for pas in range(n_pas + 1):
            if pas % cfg["eval_tous"] == 0 or pas == n_pas:
                # mêmes lots d'évaluation à chaque fois (générateur réinitialisé)
                _kw = dict(taille_bloc=cfg["taille_bloc"], taille_lot=cfg["taille_lot"], device=device,
                           n_lots=cfg["n_lots_eval"])
                h["pas"].append(pas)
                h["train"].append(estimer_loss(modele, donnees_train, generateur=torch.Generator().manual_seed(1), **_kw))
                h["val"].append(estimer_loss(modele, donnees_val, generateur=torch.Generator().manual_seed(2), **_kw))
                if perte_honnete is not None:
                    h["honnete"].append(perte_honnete(modele))
                print(f"{journal}pas {pas:5d}  train {h['train'][-1]:.3f}  val {h['val'][-1]:.3f}"
                      + (f"  dernière position {h['honnete'][-1]:.3f}" if perte_honnete else "")
                      + f"  ({time.perf_counter() - t0:.0f} s)")
            if pas == n_pas:
                break
            for g in optim.param_groups:
                g["lr"] = taux(pas)
            x, y = lot_aleatoire(donnees_train, cfg["taille_bloc"], cfg["taille_lot"], device, gen)
            _, loss = modele(x, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            h["perte_lots"].append(loss.item())
        h["duree"] = time.perf_counter() - t0
        modele.eval()
        return h

    return (entrainer,)


@app.cell
def _(mo):
    # Le bouton empêche l'entraînement de se relancer à chaque modification d'une cellule amont.
    bouton = mo.ui.run_button(label="Entraîner (≈ 3 min sur M1 Pro, puis ≈ 1 min 30 d'ablations)")
    bouton
    return (bouton,)


@app.cell
def _(
    bouton,
    config,
    creer_modele,
    device,
    entrainer,
    ids_train,
    ids_val,
    mo,
    np,
    taux,
):
    # En script (make ex MODE=run|smoke), il n'y a personne pour cliquer : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Entraîner » pour lancer l'apprentissage.*"))
    modele = creer_modele()
    historique = entrainer(modele, ids_train, ids_val, config["n_pas"], taux, config, device)
    val_finale = historique["val"][-1]
    perte_initiale = historique["val"][0]
    print(f"Entraînement : {historique['duree']:.0f} s, {1000 * historique['duree'] / config['n_pas']:.0f} ms par pas ;"
          f" val {perte_initiale:.3f} → {val_finale:.3f} (perplexité {np.exp(val_finale):.2f})")
    return historique, modele, perte_initiale, val_finale


@app.cell
def _(
    SMOKE,
    checkpoints_dir,
    config,
    historique,
    modele,
    n_parametres,
    tok,
    val_finale,
):
    # Checkpoint partagé : rechargé par les exercices des chapitres 28 (inférence) et 38
    # (interprétabilité) avec MiniGPT.charger(chemin). En smoke, fichier séparé.
    chemin_checkpoint = checkpoints_dir() / ("mini-gpt-shakespeare-smoke.pt" if SMOKE else "mini-gpt-shakespeare.pt")
    modele.sauvegarder(chemin_checkpoint, caracteres=tok.caracteres, pas=config["n_pas"],
                       val_loss=val_finale, n_parametres=n_parametres, part_train=config["part_train"],
                       historique={k: historique[k] for k in ("pas", "train", "val")})
    print(f"Checkpoint : {chemin_checkpoint}")
    return (chemin_checkpoint,)


@app.cell
def _(config, device, modele, tok, torch):
    # Génération autorégressive : un caractère à la fois, réinjecté dans le contexte.
    _gen = torch.Generator().manual_seed(42)
    _depart = torch.tensor([tok.encoder("\n")], device=device)
    texte_genere = tok.decoder(modele.generer(_depart, config["n_generes"], temperature=0.8,
                                              generateur=_gen)[0].tolist())
    print("--- Texte généré (température 0,8) ---")
    print(texte_genere)
    return (texte_genere,)


@app.cell
def _(mo):
    mo.md(r"""
    ### B3. Cartes d'attention

    On fait passer une réplique de l'ensemble de validation (jamais vue à l'entraînement)
    et on récupère, pour chacune des 16 têtes (4 couches × 4 têtes), les produits
    scalaires bruts $q_i\cdot k_j$ puis les poids $\mathrm{softmax}(q_i\cdot k_j/\sqrt{64})$
    avec le masque causal. Ces scores alimentent aussi la démo `Attention` du chapitre.
    Pour chaque tête, on mesure la part du poids mise sur le caractère lui-même, sur le
    caractère précédent, et sur le premier caractère de la fenêtre.
    """)


@app.cell
def _(F, device, ids_val, math, modele, np, tok, torch):
    PHRASE = "BAPTISTA:\nMistake me not; I speak but as I find."
    _val_texte = tok.decoder(ids_val.tolist())
    assert PHRASE in _val_texte, "la phrase doit venir de l'ensemble de validation"

    @torch.no_grad()
    def capturer_attention(m, ids):
        """Rejoue le passage avant bloc par bloc et garde q·k de chaque tête. ids : (1, T)."""
        m.eval()
        T = ids.shape[1]
        x = m.emb_tokens(ids) + m.emb_positions(torch.arange(T, device=ids.device))
        scores = []
        for bloc in m.blocs:
            C, h = x.shape[-1], bloc.attn.n_tetes
            q, k, _ = bloc.attn.qkv(bloc.ln1(x)).split(C, dim=2)
            q, k = (t.view(1, T, h, C // h).transpose(1, 2) for t in (q, k))
            scores.append((q @ k.transpose(-2, -1))[0].float().cpu())   # (têtes, T, T), non divisés
            x = bloc(x)
        logits = m.tete(m.ln_final(x))
        return torch.stack(scores), logits      # (couches, têtes, T, T)

    _ids = torch.tensor([tok.encoder(PHRASE)], device=device)
    _scores, _logits = capturer_attention(modele, _ids)
    # Contrôle : le passage rejoué donne les mêmes logits que modele(ids)
    ecart_capture = (_logits - modele(_ids)[0]).abs().max().item()
    scores_phrase = _scores.numpy()
    _d_tete = modele.cfg.n_embd // modele.cfg.n_tetes
    _T = len(PHRASE)
    _masque = torch.ones(_T, _T, dtype=torch.bool).tril()
    poids_phrase = F.softmax((_scores / math.sqrt(_d_tete)).masked_fill(~_masque, float("-inf")), dim=-1).numpy()

    # Profil de chaque tête : poids moyen sur soi, sur le précédent, sur la position 0
    _i = np.arange(1, _T)
    profil_tetes = []
    for _c in range(poids_phrase.shape[0]):
        for _h in range(poids_phrase.shape[1]):
            _P = poids_phrase[_c, _h]
            profil_tetes.append(dict(couche=_c + 1, tete=_h + 1,
                                     soi=float(_P[_i, _i].mean()),
                                     precedent=float(_P[_i, _i - 1].mean()),
                                     debut=float(_P[_i, 0].mean()),
                                     entropie=float(-(_P[_i] * np.log(_P[_i] + 1e-12)).sum(1).mean())))
    print(f"Phrase : {PHRASE!r} ({_T} caractères) ; écart de contrôle {ecart_capture:.1e}")
    print("couche tête | soi   précédent  début | entropie (nats)")
    for _p in profil_tetes:
        print(f"  {_p['couche']}     {_p['tete']}   | {_p['soi']:.2f}  {_p['precedent']:.2f}       {_p['debut']:.2f}  | {_p['entropie']:.2f}")
    return PHRASE, ecart_capture, poids_phrase, profil_tetes, scores_phrase


@app.cell
def _(mo):
    mo.md(r"""
    ## C. Ablations et coût

    ### C1. Sans masque causal, le modèle triche

    On entraîne le même modèle en retirant le masque : chaque position voit **toute** la
    fenêtre, y compris le caractère qu'elle doit prédire (il est à la position suivante de
    l'entrée). La perte d'entraînement s'effondre. Mais la **dernière position** de la
    fenêtre n'a pas sa réponse dans l'entrée : sa perte, elle, reste honnête et montre ce
    que le modèle sait vraiment prédire. En génération, on n'a jamais le futur.
    """)


@app.cell
def _(F, config, creer_modele, device, entrainer, ids_train, ids_val, lot_aleatoire, taux, torch):
    class _AttentionSansMasque(torch.nn.Module):
        """Même calcul que AttentionCausale, sans masque (et sans dropout)."""

        def __init__(self, a):
            super().__init__()
            self.a = a

        def forward(self, x):
            B, T, C = x.shape
            q, k, v = self.a.qkv(x).split(C, dim=2)
            q, k, v = (t.view(B, T, self.a.n_tetes, C // self.a.n_tetes).transpose(1, 2) for t in (q, k, v))
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            return self.a.proj(y.transpose(1, 2).contiguous().view(B, T, C))

    modele_sans_masque = creer_modele()
    for _bloc in modele_sans_masque.blocs:
        _bloc.attn = _AttentionSansMasque(_bloc.attn)

    @torch.no_grad()
    def perte_derniere_position(m):
        """Perte sur la dernière position seulement (sa cible n'est pas dans l'entrée)."""
        etait = m.training
        m.eval()
        g = torch.Generator().manual_seed(2)
        pertes = []
        for _ in range(config["n_lots_eval"]):
            x, y = lot_aleatoire(ids_val, config["taille_bloc"], config["taille_lot"], device, g)
            logits, _ = m(x)
            pertes.append(F.cross_entropy(logits[:, -1], y[:, -1]).item())
        m.train(etait)
        return sum(pertes) / len(pertes)

    historique_sans_masque = entrainer(modele_sans_masque, ids_train, ids_val, config["n_pas_sans_masque"],
                                       taux, config, device, perte_honnete=perte_derniere_position,
                                       journal="[sans masque] ")
    return historique_sans_masque, perte_derniere_position


@app.cell
def _(historique, perte_derniere_position, modele):
    # Même mesure sur le modèle normal, pour comparer
    perte_derniere_normale = perte_derniere_position(modele)
    print(f"Modèle normal : perte de la dernière position {perte_derniere_normale:.3f}"
          f" (val, toutes positions : {historique['val'][-1]:.3f})")
    return (perte_derniere_normale,)


@app.cell
def _(mo):
    mo.md(r"""
    ### C2. Sans vecteurs de position

    Même modèle, plongements de position mis à zéro et gelés : la seule information d'ordre
    qui reste est celle que crée le masque causal (la position $i$ fait une moyenne sur
    $i$ éléments). Même planning de taux d'apprentissage, arrêté plus tôt ; on compare à la
    courbe du modèle normal au même pas.
    """)


@app.cell
def _(config, creer_modele, device, entrainer, historique, ids_train, ids_val, taux):
    modele_sans_position = creer_modele(sans_position=True)
    historique_sans_position = entrainer(modele_sans_position, ids_train, ids_val,
                                         config["n_pas_sans_position"], taux, config, device,
                                         journal="[sans position] ")
    _pas = config["n_pas_sans_position"]
    val_normal_meme_pas = historique["val"][historique["pas"].index(_pas)] if _pas in historique["pas"] else float("nan")
    print(f"Au pas {_pas} : val sans position {historique_sans_position['val'][-1]:.3f},"
          f" avec positions {val_normal_meme_pas:.3f}")
    return historique_sans_position, val_normal_meme_pas


@app.cell
def _(mo):
    mo.md(r"""
    ### C3. Le coût quadratique

    La matrice des scores a $T\times T$ cases par tête : doubler la longueur du contexte
    multiplie par 4 sa taille et le travail de l'attention. On chronomètre une attention
    (lot 8, 4 têtes de 64 dimensions, masque causal) pour plusieurs longueurs.
    """)


@app.cell
def _(F, config, device, time, torch):
    def _synchro():
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    cout = {"T": [], "ms": [], "mo_scores": []}
    for _T in config["longueurs_cout"]:
        _q = torch.randn(8, 4, _T, 64, device=device)
        for _ in range(2):    # échauffement
            F.scaled_dot_product_attention(_q, _q, _q, is_causal=True)
        _synchro()
        _t0 = time.perf_counter()
        for _ in range(5):
            F.scaled_dot_product_attention(_q, _q, _q, is_causal=True)
        _synchro()
        cout["T"].append(_T)
        cout["ms"].append(1000 * (time.perf_counter() - _t0) / 5)
        cout["mo_scores"].append(8 * 4 * _T * _T * 4 / 2**20)   # float32, si la matrice est stockée
    for _i in range(len(cout["T"])):
        print(f"T = {cout['T'][_i]:5d} : {cout['ms'][_i]:7.2f} ms, matrice des scores {cout['mo_scores'][_i]:7.1f} Mio")
    return (cout,)


@app.cell
def _(
    PHRASE,
    SMOKE,
    config,
    cout,
    historique,
    historique_sans_masque,
    historique_sans_position,
    mo,
    np,
    pe_sinus,
    plt,
    poids_phrase,
    racine_dk,
    references,
    sauvegarder_figure,
    scores_phrase,
    texte_genere,
):
    # --- Figure : courbes d'apprentissage et cartes d'attention de deux têtes ---
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.2))
    _ax = _axes[0]
    _ax.plot(historique["pas"], historique["train"], label="train")
    _ax.plot(historique["pas"], historique["val"], label="validation")
    _ax.plot(historique_sans_position["pas"], historique_sans_position["val"], "--", label="val, sans position")
    _ax.plot(historique_sans_masque["pas"], historique_sans_masque["train"], ":", label="train, sans masque")
    for _nom, _l in references.items():
        if _nom != "uniforme":
            _ax.axhline(_l, color="grey", lw=0.8, ls="-.")
            _ax.text(historique["pas"][-1], _l, _nom, ha="right", va="bottom", fontsize=9, color="grey")
    _ax.set(xlabel="pas", ylabel="perte (nats par caractère)", ylim=(0, 4.4), title="Mini-GPT, TinyShakespeare")
    _ax.legend(fontsize=9)
    _axes[1].imshow(poids_phrase[-1].mean(0), cmap="viridis")
    _axes[1].set(title="Attention moyenne des 4 têtes, dernière couche")
    _lab = [c if c != "\n" else "⏎" for c in PHRASE]
    _axes[1].set_xticks(range(len(PHRASE)), _lab, fontsize=6)
    _axes[1].set_yticks(range(len(PHRASE)), _lab, fontsize=6)
    _axes[1].grid(False)
    _fig.tight_layout()
    _dossier = mo.notebook_dir() / "figures"
    _suffixe = "-smoke" if SMOKE else ""
    sauvegarder_figure(_fig, "courbes" + _suffixe, _dossier)

    # Résultats bruts pour les figures du site (site/scripts/figures/ch20_*.py) et la démo
    np.savez_compressed(
        _dossier / f"resultats{_suffixe}.npz",
        pas=np.array(historique["pas"]), train=np.array(historique["train"]), val=np.array(historique["val"]),
        perte_lots=np.array(historique["perte_lots"]),
        sm_pas=np.array(historique_sans_masque["pas"]), sm_train=np.array(historique_sans_masque["train"]),
        sm_val=np.array(historique_sans_masque["val"]), sm_honnete=np.array(historique_sans_masque["honnete"]),
        sp_pas=np.array(historique_sans_position["pas"]), sp_val=np.array(historique_sans_position["val"]),
        sp_train=np.array(historique_sans_position["train"]),
        references=np.array([references["uniforme"], references["unigramme"], references["bigramme"]]),
        phrase=np.array(PHRASE), scores=scores_phrase.astype(np.float32), poids=poids_phrase.astype(np.float32),
        pe_sinus=pe_sinus, racine_dk_d=np.array(racine_dk["d_k"]),
        racine_dk_brut=np.array(racine_dk["poids_max_brut"]), racine_dk_divise=np.array(racine_dk["poids_max_divise"]),
        cout_T=np.array(cout["T"]), cout_ms=np.array(cout["ms"]),
        texte_genere=np.array(texte_genere), duree_entrainement=historique["duree"],
        n_pas=config["n_pas"],
    )
    _fig


@app.cell
def _(
    SMOKE,
    ecart_bloc,
    ecart_capture,
    ecart_jouet,
    ecart_normes,
    ecart_permutation,
    ecart_permutation_causal,
    ecart_rotation,
    historique,
    historique_sans_masque,
    historique_sans_position,
    mo,
    np,
    perte_derniere_normale,
    perte_initiale,
    references,
    val_finale,
    val_normal_meme_pas,
):
    # --- Bilan : seuils chiffrés du README. En smoke, on vérifie seulement que tout tourne.
    _sm_train, _sm_honnete = historique_sans_masque["train"][-1], historique_sans_masque["honnete"][-1]
    seuils = [
        ("attention NumPy = PyTorch (écart < 1e-12)", ecart_jouet < 1e-12, f"{ecart_jouet:.1e}"),
        ("LayerNorm et RMSNorm NumPy = PyTorch (< 1e-10)", ecart_normes < 1e-10, f"{ecart_normes:.1e}"),
        ("bloc Transformer NumPy = PyTorch (< 1e-10)", ecart_bloc < 1e-10, f"{ecart_bloc:.1e}"),
        ("sans masque ni position : équivariance (< 1e-12)", ecart_permutation < 1e-12, f"{ecart_permutation:.1e}"),
        ("avec masque causal : plus d'équivariance (> 0,01)", ecart_permutation_causal > 0.01, f"{ecart_permutation_causal:.2f}"),
        ("encodage sinusoïdal : rotation exacte (< 1e-12)", ecart_rotation < 1e-12, f"{ecart_rotation:.1e}"),
        ("capture des scores = passage avant (< 1e-3)", ecart_capture < 1e-3, f"{ecart_capture:.1e}"),
        ("perte initiale ≈ ln 65 = 4,17 (à 0,15 près)", abs(perte_initiale - np.log(65)) < 0.15, f"{perte_initiale:.3f}"),
        ("val finale ≤ 1,60", val_finale <= 1.60, f"{val_finale:.3f}"),
        ("val finale < bigramme − 0,8", val_finale < references["bigramme"] - 0.8,
         f"{val_finale:.3f} contre {references['bigramme']:.3f}"),
        ("sans masque : train < 0,5 (triche)", _sm_train < 0.5, f"{_sm_train:.3f}"),
        ("sans masque : dernière position > 2,0", _sm_honnete > 2.0, f"{_sm_honnete:.3f}"),
        ("modèle normal : dernière position ≤ 1,70", perte_derniere_normale <= 1.70, f"{perte_derniere_normale:.3f}"),
        ("sans position : val plus haute qu'avec, au même pas", historique_sans_position["val"][-1] > val_normal_meme_pas,
         f"{historique_sans_position['val'][-1]:.3f} contre {val_normal_meme_pas:.3f}"),
    ]
    for _nom, _ok, _v in seuils:
        print(f"[{'ok' if _ok else 'ÉCHEC'}] {_nom} : {_v}")
    _echecs = [n for n, ok, _ in seuils if not ok]
    if not SMOKE and _echecs:
        raise SystemExit("Seuils non atteints : " + " ; ".join(_echecs))
    mo.md(f"**Bilan** : val {val_finale:.3f} (perplexité {np.exp(val_finale):.2f}) en "
          f"{historique['duree']:.0f} s ; {len(seuils) - len(_echecs)}/{len(seuils)} seuils atteints"
          + (" (smoke : seuils non vérifiés)" if SMOKE else "") + ".")


if __name__ == "__main__":
    app.run()
