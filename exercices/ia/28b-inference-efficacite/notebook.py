# Exercice 28b : servir un modèle de langage à moindre coût, sur le mini-GPT de l'exercice 20a.
#
#   make ex NUM=28b              → éditeur marimo (bouton « Lancer »)
#   make ex NUM=28b MODE=run     → script complet, vérifie les seuils du README
#   make ex NUM=28b MODE=smoke   → test rapide (< 30 s)
#
# Recharge data/checkpoints/mini-gpt-shakespeare.pt (exercice 20a) ; s'il manque, le
# réentraîne avec la même recette (≈ 3 min). Puis, du plus simple au plus fin :
#   A. attention par blocs avec softmax « en ligne » (l'idée de FlashAttention), en NumPy ;
#   B. cache KV écrit à la main, vérifié contre le passage complet, accélération mesurée ;
#   C. MQA et GQA par moyenne des têtes K/V, puis « uptraining » de 5 % (recette de GQA) ;
#   D. température, top-k, top-p écrits à la main : qualité contre diversité ;
#   E. quantification des poids de 16 à 1,58 bit : taille contre perplexité ;
#   F. distillation vers un élève 30 fois plus petit (KL avec température) ;
#   G. décodage spéculatif : l'élève propose, le maître vérifie, la distribution est exacte.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import re
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    from ia_commun import checkpoints_dir, get_device, is_smoke, sauvegarder_figure, seed_everything, style_commun
    from ia_commun.donnees import charger_tinyshakespeare
    from ia_commun.mini_gpt import ConfigGPT, MiniGPT, lot_aleatoire

    return (
        ConfigGPT,
        F,
        MiniGPT,
        charger_tinyshakespeare,
        checkpoints_dir,
        get_device,
        is_smoke,
        lot_aleatoire,
        math,
        mo,
        np,
        plt,
        re,
        sauvegarder_figure,
        seed_everything,
        style_commun,
        time,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Exercice 28b : servir un modèle de langage à moindre coût

    Entraîner un LLM coûte cher une fois ; le **servir** coûte cher à chaque requête. On
    reprend le mini-GPT de l'exercice 20a (3,2 millions de paramètres, perte de validation
    1,51 sur TinyShakespeare) et on applique, en miniature, les techniques qui rendent
    l'inférence des grands modèles abordable :

    | Partie | Technique | Ce qu'on mesure |
    |---|---|---|
    | A | attention par blocs (idée de FlashAttention) | même résultat sans jamais stocker la matrice T × T |
    | B | cache KV | logits identiques, accélération de la génération |
    | C | MQA, GQA | cache divisé par 4 ou 2, perte après 5 % de réentraînement |
    | D | température, top-k, top-p | mots existants contre diversité |
    | E | quantification 8, 4, 3, 2, 1,58 bits | taille contre perplexité |
    | F | distillation | élève de 100 000 paramètres, avec et sans maître |
    | G | décodage spéculatif | tokens acceptés par appel au gros modèle, distribution exacte |

    **Pertes optimisées.** Parties C et F : entropie croisée du token suivant (chapitre 20).
    Partie F, en plus : perte de distillation de Hinton, Vinyals et Dean (2015),
    $\mathcal{L} = \lambda\,\tau^2\,\mathrm{KL}\big(\mathrm{softmax}(z_m/\tau)\,\|\,\mathrm{softmax}(z_e/\tau)\big)
    + (1-\lambda)\,\mathrm{CE}(y, \mathrm{softmax}(z_e))$, où $z_m$ et $z_e$ sont les logits du
    maître et de l'élève et $\tau$ la température. Les parties A, B, D, E et G n'entraînent
    rien : elles changent la façon de calculer ou d'échantillonner.
    """)


@app.cell
def _(get_device, is_smoke, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    device = get_device()
    style_commun()
    config = dict(
        part_train=0.9, taille_bloc=128,
        n_fenetres_eval=16 if SMOKE else 200,     # fenêtres de validation fixes (200 × 128 = 25 600 caractères)
        # A. attention par blocs (NumPy)
        flash_T=256 if SMOKE else 1024, flash_d=64, flash_bloc=128,
        # B. cache KV
        kv_n_generes=40 if SMOKE else 127, kv_repetitions=1 if SMOKE else 3,
        # C. GQA : 5 % des 1 500 pas × 64 de l'exercice 20a = 75 pas de 64 fenêtres
        gqa_groupes=[4, 2, 1], gqa_n_pas=4 if SMOKE else 75, gqa_lot=64, gqa_lr=3e-4,
        # D. échantillonnage
        ech_n_prompts=8 if SMOKE else 32, ech_long_prompt=32, ech_n_generes=40 if SMOKE else 96,
        # E. quantification
        quant_groupe=32,
        # F. distillation : élève 2 couches × 64 dimensions
        el_couches=2, el_dim=64, el_tetes=4, el_n_pas=10 if SMOKE else 1500, el_lot=32,
        el_lr_max=3e-3, el_lr_min=3e-4, el_chauffe=100,
        kd_tau=1.0, kd_lambda=1.0,           # élève distillé principal : KL pure vers le maître
        kd_tau_haut=2.0, kd_lambda_haut=0.9,  # variante « à la Hinton » : température 2, 10 % de cibles dures
        # G. décodage spéculatif
        spec_gammas=[2, 4] if SMOKE else [2, 4, 6], spec_n_prompts=2 if SMOKE else 8,
        spec_long_prompt=16, spec_n_generes=24 if SMOKE else 100, spec_n_tirages=20_000 if SMOKE else 400_000,
        # Réentraînement du maître si le checkpoint manque (recette de l'exercice 20a)
        maitre_n_pas=30 if SMOKE else 1500,
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, périphérique {device}")
    print(config)
    return SEED, SMOKE, config, device


@app.cell
def _(
    ConfigGPT,
    MiniGPT,
    SMOKE,
    charger_tinyshakespeare,
    checkpoints_dir,
    config,
    device,
    lot_aleatoire,
    math,
    time,
    torch,
):
    texte = charger_tinyshakespeare()
    caracteres = sorted(set(texte))
    vers_id = {c: i for i, c in enumerate(caracteres)}

    def encoder(s):
        return [vers_id[c] for c in s]

    def decoder(ids):
        return "".join(caracteres[int(i)] for i in ids)

    _n = int(config["part_train"] * len(texte))
    texte_train, texte_val = texte[:_n], texte[_n:]
    ids_train = torch.tensor(encoder(texte_train), dtype=torch.long)
    ids_val = torch.tensor(encoder(texte_val), dtype=torch.long)

    def reentrainer_maitre(chemin):
        """Recette de l'exercice 20a (4 couches, 256 dimensions, 1 500 pas de 64 × 128)."""
        print(f"Checkpoint absent : réentraînement du maître ({config['maitre_n_pas']} pas, ≈ 3 min)…")
        torch.manual_seed(42)
        m = MiniGPT(ConfigGPT(taille_vocab=len(caracteres))).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3, betas=(0.9, 0.99), weight_decay=0.1)
        gen = torch.Generator().manual_seed(42)
        n = config["maitre_n_pas"]
        m.train()
        for pas in range(n):
            lr = 2e-3 * (pas + 1) / 100 if pas < 100 else \
                2e-4 + 0.5 * 1.8e-3 * (1 + math.cos(math.pi * (pas - 100) / max(1, n - 100)))
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = lot_aleatoire(ids_train, 128, 64, device, gen)
            _, loss = m(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        m.sauvegarder(chemin, caracteres=caracteres, pas=n, part_train=config["part_train"])
        return m

    _chemin = checkpoints_dir() / "mini-gpt-shakespeare.pt"
    if SMOKE and not _chemin.exists():
        _chemin = checkpoints_dir() / "mini-gpt-shakespeare-smoke.pt"
    _t0 = time.perf_counter()
    if _chemin.exists():
        maitre, _infos = MiniGPT.charger(_chemin, device=device)
        assert _infos["caracteres"] == caracteres, "vocabulaire du checkpoint différent"
        print(f"Maître rechargé : {_chemin.name} (pas {_infos.get('pas')}, val {_infos.get('val_loss', float('nan')):.3f})")
    else:
        maitre = reentrainer_maitre(_chemin)
    maitre.eval()
    print(f"{maitre.nb_parametres():,} paramètres, {time.perf_counter() - _t0:.1f} s".replace(",", " "))
    return decoder, encoder, ids_train, ids_val, maitre, texte_train, texte_val


@app.cell
def _(F, config, device, ids_val, torch):
    # Fenêtres de validation FIXES : toutes les variantes du modèle sont comparées sur les mêmes caractères
    _g = torch.Generator().manual_seed(2)
    _debuts = torch.randint(len(ids_val) - config["taille_bloc"] - 1, (config["n_fenetres_eval"],), generator=_g)
    X_EVAL = torch.stack([ids_val[i:i + config["taille_bloc"]] for i in _debuts]).to(device)
    Y_EVAL = torch.stack([ids_val[i + 1:i + 1 + config["taille_bloc"]] for i in _debuts]).to(device)

    @torch.no_grad()
    def perte_val(modele, lot=50):
        """Entropie croisée moyenne sur les fenêtres fixes (modèle en mode évaluation)."""
        etait = modele.training
        modele.eval()
        total = 0.0
        for i in range(0, len(X_EVAL), lot):
            logits, _ = modele(X_EVAL[i:i + lot])
            total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), Y_EVAL[i:i + lot].reshape(-1),
                                     reduction="sum").item()
        modele.train(etait)
        return total / Y_EVAL.numel()

    return (perte_val,)


@app.cell
def _(maitre, np, perte_val):
    perte_maitre = perte_val(maitre)
    print(f"Maître : perte de validation {perte_maitre:.3f} (perplexité {np.exp(perte_maitre):.2f}) sur les fenêtres fixes")
    return (perte_maitre,)


@app.cell
def _(mo):
    mo.md(r"""
    ## A. L'attention par blocs (l'idée de FlashAttention)

    L'attention standard calcule la matrice $S = QK^\top/\sqrt{d}$ de taille $T\times T$,
    la stocke, applique la softmax, puis multiplie par $V$. À $T = 1\,024$ en float64,
    c'est 8 Mo par tête ; à $T = 100\,000$, 80 Go. FlashAttention (Dao et al., 2022) ne la
    stocke jamais : elle parcourt les clés **par blocs** et met à jour, pour chaque requête,
    un maximum courant $m$, une somme courante $\ell$ et une sortie partielle. Quand un
    nouveau bloc arrive avec un maximum plus grand, on **rééchelonne** ce qui a déjà été
    accumulé par $e^{m_{\text{ancien}} - m_{\text{nouveau}}}$. Le résultat est **exact**,
    pas approché. Le gain réel vient de la mémoire du GPU (tout tient dans la mémoire rapide
    de chaque cœur) : en NumPy on ne vérifie que l'exactitude et la mémoire.
    """)


@app.cell
def _(config, np):
    def attention_standard(Q, K, V, causal=True):
        S = Q @ K.T / np.sqrt(Q.shape[1])
        if causal:
            S = np.where(np.tril(np.ones_like(S, dtype=bool)), S, -np.inf)
        P = np.exp(S - S.max(axis=1, keepdims=True))
        return (P / P.sum(axis=1, keepdims=True)) @ V, S.nbytes

    def attention_par_blocs(Q, K, V, b, causal=True):
        """Softmax « en ligne » : jamais plus d'un bloc b × b de scores en mémoire."""
        T, d = Q.shape
        O = np.zeros_like(Q)
        pic = 0
        for i0 in range(0, T, b):
            q = Q[i0:i0 + b]
            m = np.full(len(q), -np.inf)          # maximum courant de chaque ligne
            l = np.zeros(len(q))                  # somme courante des exponentielles
            acc = np.zeros_like(q)                # somme courante des exp × V
            for j0 in range(0, (i0 + b) if causal else T, b):
                s = q @ K[j0:j0 + b].T / np.sqrt(d)
                if causal:
                    s = np.where((np.arange(i0, i0 + len(q))[:, None] >= np.arange(j0, j0 + s.shape[1])[None, :]), s, -np.inf)
                pic = max(pic, s.nbytes)
                m_nouveau = np.maximum(m, s.max(axis=1))
                p = np.exp(s - m_nouveau[:, None])
                facteur = np.exp(m - m_nouveau)    # rééchelonne ce qui a été accumulé
                l = l * facteur + p.sum(axis=1)
                acc = acc * facteur[:, None] + p @ V[j0:j0 + b]
                m = m_nouveau
            O[i0:i0 + b] = acc / l[:, None]
        return O, pic

    _rng = np.random.default_rng(42)
    _T, _d, _b = config["flash_T"], config["flash_d"], config["flash_bloc"]
    _Q, _K, _V = (_rng.standard_normal((_T, _d)) for _ in range(3))
    _O1, octets_standard = attention_standard(_Q, _K, _V)
    _O2, octets_blocs = attention_par_blocs(_Q, _K, _V, _b)
    ecart_flash = float(np.abs(_O1 - _O2).max())
    print(f"T = {_T}, d = {_d}, blocs de {_b} : écart maximal {ecart_flash:.1e}")
    print(f"Matrice des scores : {octets_standard / 2**20:.1f} Mio en standard, "
          f"{octets_blocs / 2**10:.0f} Kio au plus par blocs (× {octets_standard / octets_blocs:.0f})")
    return ecart_flash, octets_blocs, octets_standard


@app.cell
def _(mo):
    mo.md(r"""
    ## B. Le cache KV

    En génération, le modèle produit un token, l'ajoute au contexte et recommence. Sans
    précaution, chaque pas **recalcule** les clés $K$ et les valeurs $V$ de tout le contexte,
    alors qu'elles n'ont pas changé (le masque causal garantit qu'un token ne dépend pas de ce
    qui le suit). Le **cache KV** les garde en mémoire : à chaque pas, on ne calcule $Q$, $K$,
    $V$ que pour le nouveau token, et on l'attache aux clés et valeurs déjà connues. Le coût
    d'un pas passe de « tout le contexte » à « un token ». Prix : une mémoire de
    $2 \times \text{couches} \times \text{têtes KV} \times d_{\text{tête}} \times T$ nombres
    par séquence.
    """)


@app.cell
def _(F, torch):
    @torch.no_grad()
    def passe_cache(m, idx, cache=None):
        """Passage avant sur les tokens `idx` (B, t) seulement, en réutilisant le cache.

        cache : liste par couche de (K, V) de forme (B, têtes, positions passées, d_tête), ou None.
        Renvoie (logits (B, t, V), nouveau cache). Mêmes poids que MiniGPT.forward.
        """
        B, t = idx.shape
        p0 = 0 if cache is None else cache[0][0].shape[2]
        pos = torch.arange(p0, p0 + t, device=idx.device)
        x = m.emb_tokens(idx) + m.emb_positions(pos)
        nouveau = []
        for i, bloc in enumerate(m.blocs):
            a = bloc.attn
            C, h = x.shape[-1], a.n_tetes
            q, k, v = a.qkv(bloc.ln1(x)).split(C, dim=2)
            q, k, v = (z.view(B, t, h, C // h).transpose(1, 2) for z in (q, k, v))
            if cache is not None:
                k = torch.cat([cache[i][0], k], dim=2)
                v = torch.cat([cache[i][1], v], dim=2)
            nouveau.append((k, v))
            if t == 1:
                y = F.scaled_dot_product_attention(q, k, v)          # le nouveau token voit tout le passé
            else:
                # requêtes p0..p0+t-1 contre clés 0..p0+t-1 : masque causal décalé de p0
                masque = torch.ones(t, p0 + t, dtype=torch.bool, device=idx.device).tril(diagonal=p0)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=masque)
            x = x + a.proj(y.transpose(1, 2).reshape(B, t, C))
            x = x + bloc.mlp(bloc.ln2(x))
        return m.tete(m.ln_final(x)), nouveau

    def taille_cache(cache):
        return sum(k.numel() + v.numel() for k, v in cache) * cache[0][0].element_size()

    return passe_cache, taille_cache


@app.cell
def _(config, device, encoder, maitre, passe_cache, taille_cache, texte_val, time, torch):
    # 1. Vérification : générer token par token avec le cache donne les mêmes logits qu'un passage complet
    _ids = torch.tensor([encoder(texte_val[:config["taille_bloc"]])], device=device)
    with torch.no_grad():
        _complet, _ = maitre(_ids)
    _cache, _morceaux = None, []
    for _t in range(_ids.shape[1]):
        _lg, _cache = passe_cache(maitre, _ids[:, _t:_t + 1], _cache)
        _morceaux.append(_lg)
    ecart_kv = (torch.cat(_morceaux, dim=1) - _complet).abs().max().item()
    # Préremplissage en un coup (prompt de 100 tokens) puis un pas : même résultat
    _lg_pre, _c = passe_cache(maitre, _ids[:, :100])
    _lg_un, _ = passe_cache(maitre, _ids[:, 100:101], _c)
    ecart_prefill = max((_lg_pre - _complet[:, :100]).abs().max().item(), (_lg_un[:, 0] - _complet[:, 100]).abs().max().item())
    octets_cache = taille_cache(_cache)
    print(f"Écart cache KV / passage complet : {ecart_kv:.1e} (pas à pas), {ecart_prefill:.1e} (préremplissage)")
    print(f"Cache du mini-GPT pour 128 tokens : {octets_cache / 1024:.0f} Kio "
          f"(2 × 4 couches × 4 têtes × 64 × 128 × 4 octets)")

    def synchro(dev):
        if dev.type == "mps":
            torch.mps.synchronize()

    @torch.no_grad()
    def generer_glouton(m, n, avec_cache, dev):
        """Génération gloutonne de n tokens depuis « \\n » ; renvoie (ids, temps de chaque pas)."""
        ids = torch.tensor([encoder("\n")], device=dev)
        cache, temps = None, []
        for _ in range(n):
            t0 = time.perf_counter()
            if avec_cache:
                lg, cache = passe_cache(m, ids[:, -1:] if cache is not None else ids, cache)
            else:
                lg, _ = m(ids)
            suivant = lg[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, suivant], dim=1)
            synchro(dev)
            temps.append(time.perf_counter() - t0)
        return ids, temps

    # 2. Mesure : MPS puis CPU, avec et sans cache (meilleur de plusieurs répétitions)
    mesures_kv = {}
    for _nom_dev in ([device.type, "cpu"] if device.type != "cpu" else ["cpu"]):
        _dev = torch.device(_nom_dev)
        _m = maitre.to(_dev)
        for _avec in (False, True):
            generer_glouton(_m, 5, _avec, _dev)                       # échauffement
            _essais = [generer_glouton(_m, config["kv_n_generes"], _avec, _dev) for _ in range(config["kv_repetitions"])]
            _meilleur = min(_essais, key=lambda e: sum(e[1]))
            mesures_kv[(_nom_dev, _avec)] = dict(ids=_meilleur[0].cpu(), temps=_meilleur[1])
        _identiques = torch.equal(mesures_kv[(_nom_dev, False)]["ids"], mesures_kv[(_nom_dev, True)]["ids"])
        _sans = sum(mesures_kv[(_nom_dev, False)]["temps"])
        _avec_c = sum(mesures_kv[(_nom_dev, True)]["temps"])
        print(f"{_nom_dev.upper()} : {config['kv_n_generes']} tokens en {_sans * 1000:.0f} ms sans cache, "
              f"{_avec_c * 1000:.0f} ms avec (× {_sans / _avec_c:.2f}) ; mêmes tokens : {_identiques}")
    maitre.to(device)
    return ecart_kv, ecart_prefill, mesures_kv, octets_cache


@app.cell
def _(mo):
    mo.md(r"""
    ### Ordres de grandeur pour un vrai modèle

    Configuration publiée de **Qwen3-0.6B** (`config.json` sur Hugging Face) : 28 couches,
    16 têtes de requête, **8 têtes K/V** (GQA), têtes de 128 dimensions, poids en bfloat16
    (2 octets). Le calcul ci-dessous compare son cache à celui qu'aurait la même architecture
    en attention multi-têtes classique (16 têtes K/V) et en MQA (1 tête K/V).
    """)


@app.cell
def _():
    def cache_par_token(couches, tetes_kv, d_tete, octets=2):
        return 2 * couches * tetes_kv * d_tete * octets      # K et V

    cache_qwen = {}
    for _nom, _tkv in [("MHA (16 têtes K/V)", 16), ("GQA (8, la vraie config)", 8), ("MQA (1)", 1)]:
        _o = cache_par_token(28, _tkv, 128)
        cache_qwen[_nom] = _o
        print(f"Qwen3-0.6B {_nom:26s}: {_o / 1024:6.0f} Kio par token ; "
              f"32 768 tokens : {_o * 32768 / 2**30:5.2f} Gio ; lot de 16 × 32 768 : {16 * _o * 32768 / 2**30:5.1f} Gio")
    return (cache_qwen,)


@app.cell
def _(mo):
    mo.md(r"""
    ## C. MQA et GQA : partager les clés et les valeurs entre têtes

    **MQA** (Shazeer, 2019) garde une seule tête K/V pour toutes les têtes de requête ;
    **GQA** (Ainslie et al., 2023) en garde une par **groupe** de têtes. Le cache est divisé
    par le nombre de têtes par groupe. Pour convertir un modèle déjà entraîné, l'article de
    GQA fait la **moyenne** des projections K et V des têtes d'un même groupe, puis
    réentraîne le modèle pendant **5 %** des pas de son pré-entraînement (« uptraining »).
    On fait exactement cela : 75 pas de 64 fenêtres, 5 % des 1 500 pas de l'exercice 20a. Pour
    rester comparable, le modèle d'origine (4 têtes K/V) est réentraîné de la même façon.

    Astuce d'implémentation : au lieu d'écrire une nouvelle attention, on force les têtes
    d'un groupe à avoir **les mêmes poids** K et V (et on rétablit cette égalité après chaque
    pas d'optimisation) ; le calcul est alors identique à une vraie GQA, seule la mémoire du
    cache n'est pas économisée ici.
    """)


@app.cell
def _(MiniGPT, config, device, ids_train, lot_aleatoire, maitre, perte_val, time, torch):
    @torch.no_grad()
    def grouper_kv(m, n_kv):
        """Moyenne des projections K et V des têtes d'un même groupe (n_kv groupes)."""
        for bloc in m.blocs:
            W = bloc.attn.qkv.weight                     # (3C, C) : lignes Q, puis K, puis V
            C, h = W.shape[1], bloc.attn.n_tetes
            for debut in (C, 2 * C):                     # bloc K, puis bloc V
                Wkv = W[debut:debut + C].view(n_kv, h // n_kv, C // h, C)
                W[debut:debut + C] = Wkv.mean(dim=1, keepdim=True).expand_as(Wkv).reshape(C, C)

    def copie_maitre():
        m = MiniGPT(maitre.cfg).to(device)
        m.load_state_dict(maitre.state_dict())
        return m

    resultats_gqa = {}
    for _n_kv in config["gqa_groupes"]:
        _m = copie_maitre()
        grouper_kv(_m, _n_kv)
        _avant = perte_val(_m)
        _opt = torch.optim.AdamW(_m.parameters(), lr=config["gqa_lr"], betas=(0.9, 0.99), weight_decay=0.1)
        _g = torch.Generator().manual_seed(42)
        _t0 = time.perf_counter()
        _m.train()
        for _pas in range(config["gqa_n_pas"]):
            _x, _y = lot_aleatoire(ids_train, config["taille_bloc"], config["gqa_lot"], device, _g)
            _, _loss = _m(_x, _y)
            _opt.zero_grad(set_to_none=True)
            _loss.backward()
            torch.nn.utils.clip_grad_norm_(_m.parameters(), 1.0)
            _opt.step()
            grouper_kv(_m, _n_kv)                        # les têtes d'un groupe restent identiques
        _m.eval()
        _apres = perte_val(_m)
        resultats_gqa[_n_kv] = dict(avant=_avant, apres=_apres, duree=time.perf_counter() - _t0)
        print(f"{_n_kv} tête(s) K/V : perte {_avant:.3f} juste après la moyenne, {_apres:.3f} après "
              f"{config['gqa_n_pas']} pas ({resultats_gqa[_n_kv]['duree']:.0f} s) ; cache ÷ {4 // _n_kv}")
    return (resultats_gqa,)


@app.cell
def _(mo):
    mo.md(r"""
    ## D. Échantillonner : température, top-k, top-p

    Le modèle donne une distribution sur les 65 caractères ; reste à **choisir**. Trois
    réglages écrits à la main :

    - **température** $\tau$ : on divise les logits par $\tau$ avant la softmax ; $\tau \to 0$
      donne le choix glouton (toujours le plus probable), $\tau > 1$ aplatit la distribution ;
    - **top-k** : on ne garde que les $k$ caractères les plus probables ;
    - **top-p** (échantillonnage du noyau, Holtzman et al., 2019) : on garde le plus petit
      ensemble de caractères dont la probabilité cumulée atteint $p$. Sa taille s'adapte :
      un seul caractère quand le modèle est sûr, beaucoup quand il hésite.

    On génère 96 caractères à partir de 32 débuts de répliques de la validation et on mesure
    la **part de mots qui existent** dans le corpus d'entraînement (qualité), la **part de
    3-grammes de mots distincts** dans chaque texte (diversité ; une boucle répétitive la fait
    chuter) et la perte moyenne du modèle sur ses propres choix.
    """)


@app.cell
def _(F, torch):
    def filtrer_logits(logits, temperature=1.0, top_k=None, top_p=None):
        """Renvoie les log-probabilités après température, top-k et top-p (−inf = exclu)."""
        logits = logits / max(temperature, 1e-8)
        if top_k is not None:
            seuil = torch.topk(logits, top_k, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < seuil, float("-inf"))
        if top_p is not None:
            tri, ordre = torch.sort(logits, dim=-1, descending=True)
            cumul = F.softmax(tri, dim=-1).cumsum(dim=-1)
            # on retire un caractère si la masse AVANT lui atteint déjà p (le premier est toujours gardé)
            retirer = (cumul - F.softmax(tri, dim=-1)) >= top_p
            tri = tri.masked_fill(retirer, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(-1, ordre, tri)
        return F.log_softmax(logits, dim=-1)

    return (filtrer_logits,)


@app.cell
def _(
    F,
    config,
    decoder,
    device,
    encoder,
    filtrer_logits,
    maitre,
    np,
    passe_cache,
    re,
    texte_train,
    texte_val,
    torch,
):
    _vocab = set(re.findall(r"[a-z']+", texte_train.lower()))
    # Débuts de répliques de la validation (après une ligne vide), 32 caractères chacun
    _rng = np.random.default_rng(42)
    _debuts = [m.end() for m in re.finditer(r"\n\n", texte_val) if m.end() + config["ech_long_prompt"] < len(texte_val)]
    _choix = _rng.choice(len(_debuts), size=config["ech_n_prompts"], replace=False)
    prompts = [texte_val[_debuts[i]:_debuts[i] + config["ech_long_prompt"]] for i in _choix]

    @torch.no_grad()
    def generer(m, prompts, n, glouton=False, gen=None, **reglages):
        """Génère n caractères pour un lot de prompts, avec le cache KV. Renvoie (textes, perte moyenne)."""
        ids = torch.tensor([encoder(p) for p in prompts], device=device)
        lg, cache = passe_cache(m, ids)
        sorties, nll = [], []
        for _ in range(n):
            dernier = lg[:, -1]
            if glouton:
                suivant = dernier.argmax(-1, keepdim=True)
            else:
                lp = filtrer_logits(dernier, **reglages)
                suivant = torch.multinomial(lp.exp().cpu(), 1, generator=gen).to(device)
            nll.append(-F.log_softmax(dernier, -1).gather(-1, suivant))   # perte du modèle sur ce choix
            sorties.append(suivant)
            lg, cache = passe_cache(m, suivant, cache)
        textes = [decoder(r) for r in torch.cat(sorties, dim=1).tolist()]
        return textes, torch.cat(nll, dim=1).mean().item()

    def mesurer(textes):
        mots_connus, distincts = [], []
        for t in textes:
            m = re.findall(r"[a-z']+", t.lower())[:-1]       # le dernier mot peut être coupé
            if m:
                mots_connus.append(np.mean([w in _vocab for w in m]))
            tri = [tuple(m[i:i + 3]) for i in range(len(m) - 2)]
            if tri:
                distincts.append(len(set(tri)) / len(tri))
        return float(np.mean(mots_connus)), float(np.mean(distincts))

    reglages_ech = [
        ("glouton", dict(glouton=True)),
        ("τ = 0,5", dict(temperature=0.5)),
        ("τ = 0,8", dict(temperature=0.8)),
        ("τ = 1", dict(temperature=1.0)),
        ("τ = 1,5", dict(temperature=1.5)),
        ("τ = 2", dict(temperature=2.0)),
        ("top-k 5", dict(temperature=1.0, top_k=5)),
        ("top-p 0,9", dict(temperature=1.0, top_p=0.9)),
        ("τ = 1,5 + top-p 0,9", dict(temperature=1.5, top_p=0.9)),
    ]
    resultats_ech = {}
    exemples_ech = {}
    for _nom, _r in reglages_ech:
        _textes, _perte = generer(maitre, prompts, config["ech_n_generes"], gen=torch.Generator().manual_seed(42), **_r)
        _connus, _dist = mesurer(_textes)
        resultats_ech[_nom] = dict(mots_connus=_connus, trigrammes_distincts=_dist, perte=_perte)
        exemples_ech[_nom] = _textes[0]
        print(f"{_nom:20s} mots existants {_connus:5.1%}   3-grammes distincts {_dist:5.1%}   perte du modèle {_perte:.2f}")
    print(f"\nPrompt : {prompts[0]!r}")
    for _nom in ("glouton", "top-p 0,9", "τ = 2"):
        print(f"--- {_nom} ---\n{exemples_ech[_nom]}")
    return exemples_ech, generer, prompts, resultats_ech


@app.cell
def _(F, device, encoder, filtrer_logits, maitre, torch):
    # Distribution du caractère suivant pour un contexte précis, et taille du noyau top-p 0,9
    CONTEXTE_DISTRIB = "KING RICHARD III:\nMy lord, I "
    with torch.no_grad():
        _lg = maitre(torch.tensor([encoder(CONTEXTE_DISTRIB)], device=device))[0][0, -1].cpu()
    distributions = {}
    for _tau in (0.5, 1.0, 1.5):
        _p = F.softmax(_lg / _tau, -1)
        _noyau = int(torch.isfinite(filtrer_logits(_lg, temperature=_tau, top_p=0.9)).sum())
        distributions[_tau] = dict(p=_p.numpy(), noyau=_noyau, entropie=float(-(_p * _p.clamp_min(1e-12).log()).sum()))
        print(f"τ = {_tau} : entropie {distributions[_tau]['entropie']:.2f} nats, max {_p.max():.2f}, noyau top-p 0,9 = {_noyau} caractères")
    return CONTEXTE_DISTRIB, distributions


@app.cell
def _(mo):
    mo.md(r"""
    ## E. Quantifier les poids

    Un poids en float32 occupe 32 bits. **Quantifier**, c'est le remplacer par un entier
    sur $b$ bits et une échelle : $\hat w = s \cdot \mathrm{arrondi}(w / s)$, avec
    $s = \max|w| / (2^{b-1} - 1)$ (quantification symétrique « absmax »). Toute la question
    est **sur combien de poids on partage une échelle** : tout le tenseur, chaque ligne, ou
    chaque groupe de 32 poids (comme GPTQ et AWQ). Un seul poids extrême dans le groupe
    écrase la précision de tous les autres.

    On quantifie après coup tous les poids des couches linéaires et des plongements, on
    revient en float32 pour calculer (« quantification simulée ») et on mesure la perte. On
    essaie aussi la quantification **ternaire** $\{-1, 0, +1\}$ de BitNet b1.58
    (échelle = moyenne des $|w|$), qui, elle, est conçue pour être apprise dès le
    pré-entraînement.
    """)


@app.cell
def _(MiniGPT, config, device, maitre, perte_val, torch):
    def quantifier(W, bits, mode, groupe=32):
        """Quantification symétrique absmax simulée (renvoie des float32)."""
        qmax = 2 ** (bits - 1) - 1
        if mode == "tenseur":
            s = W.abs().max().clamp_min(1e-12) / qmax
            return (W / s).round().clamp(-qmax, qmax) * s
        if mode == "ligne":
            s = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
            return (W / s).round().clamp(-qmax, qmax) * s
        if mode == "groupe":
            Wg = W.reshape(W.shape[0], -1, groupe)
            s = Wg.abs().amax(dim=2, keepdim=True).clamp_min(1e-12) / qmax
            return ((Wg / s).round().clamp(-qmax, qmax) * s).reshape(W.shape)
        raise ValueError(mode)

    def ternaire(W):
        """BitNet b1.58 : échelle γ = moyenne des |w|, arrondi dans {−1, 0, +1}."""
        g = W.abs().mean().clamp_min(1e-12)
        return (W / g).round().clamp(-1, 1) * g

    def copie_quantifiee(fonction):
        m = MiniGPT(maitre.cfg).to(device)
        m.load_state_dict(maitre.state_dict())
        with torch.no_grad():
            for nom, p in m.named_parameters():
                if p.dim() == 2:                          # Linear, plongements (la tête est partagée)
                    p.copy_(fonction(p.data))
        return m

    _n_poids = sum(p.numel() for p in maitre.parameters() if p.dim() == 2)
    _g = config["quant_groupe"]
    _n_lignes = sum(p.shape[0] for p in maitre.parameters() if p.dim() == 2)
    variantes_quant = [
        ("float32", 32, lambda W: W),
        ("float16", 16, lambda W: W.half().float()),
        ("bfloat16", 16, lambda W: W.bfloat16().float()),
        ("int8, une échelle par tenseur", 8, lambda W: quantifier(W, 8, "tenseur")),
        ("int8, une échelle par ligne", 8 + 16 * _n_lignes / _n_poids, lambda W: quantifier(W, 8, "ligne")),
        ("int4, une échelle par tenseur", 4, lambda W: quantifier(W, 4, "tenseur")),
        ("int4, une échelle par ligne", 4 + 16 * _n_lignes / _n_poids, lambda W: quantifier(W, 4, "ligne")),
        (f"int4, groupes de {_g}", 4 + 16 / _g, lambda W: quantifier(W, 4, "groupe", _g)),
        (f"int3, groupes de {_g}", 3 + 16 / _g, lambda W: quantifier(W, 3, "groupe", _g)),
        (f"int2, groupes de {_g}", 2 + 16 / _g, lambda W: quantifier(W, 2, "groupe", _g)),
        ("ternaire (1,58 bit), après coup", 1.58, ternaire),
    ]
    resultats_quant = {}
    for _nom, _bits, _f in variantes_quant:
        _perte = perte_val(copie_quantifiee(_f))
        resultats_quant[_nom] = dict(bits=_bits, perte=_perte, mo=_n_poids * _bits / 8 / 1e6)
        print(f"{_nom:34s} {_bits:5.2f} bits/poids  {resultats_quant[_nom]['mo']:5.2f} Mo  perte {_perte:.3f}")
    return (resultats_quant,)


@app.cell
def _(mo):
    mo.md(r"""
    ## F. Distillation

    L'élève : 2 couches, 64 dimensions, environ 100 000 paramètres hors plongements, 32 fois
    moins que le maître. Trois entraînements identiques (1 500 pas de 32 fenêtres) :

    - **sans maître** : entropie croisée sur le caractère suivant réel (« cibles dures ») ;
    - **avec maître, τ = 1** : KL pure ($\lambda = 1$) vers les probabilités du maître
      (« cibles douces ») ;
    - **avec maître, τ = 2** : $\lambda = 0{,}9$ de KL à température 2, multipliée par $\tau^2$
      comme le recommandent Hinton et al., plus $0{,}1$ d'entropie croisée.

    Les cibles douces disent à l'élève non seulement quel caractère vient, mais **lesquels
    auraient été plausibles** et dans quelle proportion : beaucoup plus d'information par
    exemple.
    """)


@app.cell
def _(mo):
    bouton = mo.ui.run_button(label="Lancer la distillation et le décodage spéculatif (≈ 2 min sur M1 Pro)")
    bouton
    return (bouton,)


@app.cell
def _(
    ConfigGPT,
    F,
    MiniGPT,
    bouton,
    config,
    device,
    ids_train,
    lot_aleatoire,
    maitre,
    math,
    mo,
    perte_val,
    time,
    torch,
):
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer » pour entraîner les élèves.*"))

    def creer_eleve():
        torch.manual_seed(42)
        return MiniGPT(ConfigGPT(taille_vocab=maitre.cfg.taille_vocab, n_couches=config["el_couches"],
                                 n_embd=config["el_dim"], n_tetes=config["el_tetes"], dropout=0.0)).to(device)

    def perte_distillation(z_eleve, z_maitre, y, tau, lam):
        """λ·τ²·KL(maître_τ ‖ élève_τ) + (1 − λ)·CE(y, élève)."""
        V = z_eleve.size(-1)
        kl = F.kl_div(F.log_softmax(z_eleve.reshape(-1, V) / tau, -1), F.log_softmax(z_maitre.reshape(-1, V) / tau, -1),
                      log_target=True, reduction="batchmean")
        ce = F.cross_entropy(z_eleve.reshape(-1, V), y.reshape(-1))
        return lam * tau ** 2 * kl + (1 - lam) * ce

    def entrainer_eleve(avec_maitre, tau=1.0, lam=1.0):
        m = creer_eleve()
        opt = torch.optim.AdamW(m.parameters(), lr=config["el_lr_max"], betas=(0.9, 0.95), weight_decay=0.1)
        g = torch.Generator().manual_seed(42)
        n = config["el_n_pas"]
        courbe = []
        t0 = time.perf_counter()
        for pas in range(n):
            r = min(1.0, max(0, pas - config["el_chauffe"]) / max(1, n - config["el_chauffe"]))
            lr = config["el_lr_max"] * (pas + 1) / config["el_chauffe"] if pas < config["el_chauffe"] else \
                config["el_lr_min"] + 0.5 * (config["el_lr_max"] - config["el_lr_min"]) * (1 + math.cos(math.pi * r))
            for grp in opt.param_groups:
                grp["lr"] = lr
            x, y = lot_aleatoire(ids_train, config["taille_bloc"], config["el_lot"], device, g)
            z, ce = m(x, y)
            if avec_maitre:
                with torch.no_grad():
                    z_m, _ = maitre(x)
                loss = perte_distillation(z, z_m, y, tau, lam)
            else:
                loss = ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if (pas + 1) % max(1, n // 10) == 0 or pas == n - 1:
                courbe.append((pas + 1, perte_val(m)))
        m.eval()
        return m, courbe, time.perf_counter() - t0

    eleve_dur, courbe_dur, _d1 = entrainer_eleve(False)
    eleve_distille, courbe_distille, _d2 = entrainer_eleve(True, config["kd_tau"], config["kd_lambda"])
    _, courbe_distille_haut, _d3 = entrainer_eleve(True, config["kd_tau_haut"], config["kd_lambda_haut"])
    perte_eleve_distille_haut = courbe_distille_haut[-1][1]
    n_params_eleve = sum(p.numel() for n, p in eleve_dur.named_parameters() if n.startswith(("blocs.", "ln_final")))
    n_params_maitre = sum(p.numel() for n, p in maitre.named_parameters() if n.startswith(("blocs.", "ln_final")))
    perte_eleve_dur, perte_eleve_distille = courbe_dur[-1][1], courbe_distille[-1][1]
    print(f"Élève : {n_params_eleve:,} paramètres hors plongements (maître : {n_params_maitre:,}, × {n_params_maitre / n_params_eleve:.0f})".replace(",", " "))
    print(f"Sans maître : perte {perte_eleve_dur:.3f} ({_d1:.0f} s) ; distillé τ = {config['kd_tau']}, λ = {config['kd_lambda']} : "
          f"{perte_eleve_distille:.3f} ({_d2:.0f} s) ; τ = {config['kd_tau_haut']}, λ = {config['kd_lambda_haut']} : {perte_eleve_distille_haut:.3f} ({_d3:.0f} s)")
    print("Courbes (pas, perte) sans / avec :", [(p, round(l, 3)) for p, l in courbe_dur], [(p, round(l, 3)) for p, l in courbe_distille])
    return (
        courbe_distille,
        courbe_distille_haut,
        courbe_dur,
        eleve_distille,
        eleve_dur,
        n_params_eleve,
        n_params_maitre,
        perte_eleve_distille,
        perte_eleve_distille_haut,
        perte_eleve_dur,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## G. Décodage spéculatif

    Le gros modèle est lent surtout parce qu'il est appelé **une fois par token**. Idée de
    Leviathan et al. (2022) et de Chen et al. (2023) : un petit modèle « brouillon » propose
    $\gamma$ tokens ; le gros modèle les **vérifie tous en un seul passage** (il calcule ses
    probabilités $p$ aux $\gamma + 1$ positions d'un coup, comme à l'entraînement). On
    accepte le $i$-ème token proposé $x$ avec la probabilité $\min(1, p(x)/q(x))$ ; au premier
    refus, on tire un remplaçant dans $\max(0, p - q)$ renormalisé et on s'arrête. Si tout
    est accepté, on tire un token de plus dans $p$.

    Ce schéma d'acceptation-rejet garantit que chaque token suit **exactement** la
    distribution du gros modèle : on ne perd rien en qualité. Le gain dépend du taux
    d'acceptation, d'autant plus haut que le brouillon ressemble au maître ; d'où l'intérêt
    d'un brouillon **distillé**.
    """)


@app.cell
def _(F, np, torch):
    def verifier_exactitude(p, q, n, rng):
        """Tire n fois le premier token du schéma spéculatif ; renvoie la distribution empirique."""
        V = len(p)
        x = rng.choice(V, size=n, p=q)                               # proposition du brouillon
        accepte = rng.random(n) < np.minimum(1.0, p[x] / q[x])
        residu = np.maximum(p - q, 0)
        residu = residu / residu.sum()
        x_final = np.where(accepte, x, rng.choice(V, size=n, p=residu))
        return np.bincount(x_final, minlength=V) / n

    @torch.no_grad()
    def generer_speculatif(cible, brouillon, ids, n, gamma, gen):
        """Décodage spéculatif (température 1). ids : (1, t). Renvoie (ids, nb d'appels à la cible, nb acceptés, nb proposés)."""
        appels, acceptes, proposes = 0, 0, 0
        fin = ids.shape[1] + n
        while ids.shape[1] < fin:
            g = min(gamma, fin - ids.shape[1])
            # 1. le brouillon propose g tokens, un par un
            props, qs = ids, []
            for _ in range(g):
                q = F.softmax(brouillon(props)[0][0, -1], -1)
                x = torch.multinomial(q.cpu(), 1, generator=gen).to(ids.device)
                qs.append(q)
                props = torch.cat([props, x[None]], dim=1)
            # 2. la cible calcule p aux g + 1 positions en UN passage
            p_tout = F.softmax(cible(props)[0][0, -(g + 1):], -1)
            appels += 1
            proposes += g
            nouveaux = []
            for i in range(g):
                x = props[0, ids.shape[1] + i]
                p, q = p_tout[i], qs[i]
                if torch.rand(1, generator=gen).item() < min(1.0, (p[x] / q[x]).item()):
                    nouveaux.append(x.item())
                    acceptes += 1
                else:
                    residu = (p - q).clamp_min(0)
                    nouveaux.append(torch.multinomial((residu / residu.sum()).cpu(), 1, generator=gen).item())
                    break
            else:
                nouveaux.append(torch.multinomial(p_tout[g].cpu(), 1, generator=gen).item())   # bonus
            ids = torch.cat([ids, torch.tensor([nouveaux], device=ids.device)], dim=1)[:, :fin]
        return ids, appels, acceptes, proposes

    @torch.no_grad()
    def generer_simple(cible, ids, n, gen):
        for _ in range(n):
            p = F.softmax(cible(ids)[0][0, -1], -1)
            ids = torch.cat([ids, torch.multinomial(p.cpu(), 1, generator=gen).to(ids.device)[None]], dim=1)
        return ids

    return generer_simple, generer_speculatif, verifier_exactitude


@app.cell
def _(
    F,
    config,
    device,
    eleve_distille,
    eleve_dur,
    encoder,
    generer_simple,
    generer_speculatif,
    maitre,
    np,
    texte_val,
    time,
    torch,
    verifier_exactitude,
):
    # 1. Exactitude : distribution du premier token, schéma spéculatif contre maître
    _ctx = torch.tensor([encoder("KING RICHARD III:\nMy lord, I ")], device=device)
    with torch.no_grad():
        _p = F.softmax(maitre(_ctx)[0][0, -1].cpu().double(), -1).numpy()
        _q = F.softmax(eleve_distille(_ctx)[0][0, -1].cpu().double(), -1).numpy()
    _p, _q = _p / _p.sum(), _q / _q.sum()
    _emp = verifier_exactitude(_p, _q, config["spec_n_tirages"], np.random.default_rng(42))
    vt_speculatif = 0.5 * np.abs(_emp - _p).sum()
    vt_brouillon = 0.5 * np.abs(_q - _p).sum()
    print(f"Distance en variation totale au maître : schéma spéculatif {vt_speculatif:.4f} "
          f"({config['spec_n_tirages']:,} tirages), brouillon seul {vt_brouillon:.3f}".replace(",", " "))
    print(f"Taux d'acceptation théorique sur ce contexte : Σ min(p, q) = {np.minimum(_p, _q).sum():.3f}")

    # 2. Génération : 8 débuts de validation, 100 caractères chacun
    _rng = np.random.default_rng(7)
    _debuts = _rng.choice(len(texte_val) - 200, size=config["spec_n_prompts"], replace=False)
    _prompts = [torch.tensor([encoder(texte_val[i:i + config["spec_long_prompt"]])], device=device) for i in _debuts]

    def _chrono(f):
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        r = f()
        if device.type == "mps":
            torch.mps.synchronize()
        return r, time.perf_counter() - t0

    _, temps_simple = _chrono(lambda: [generer_simple(maitre, p, config["spec_n_generes"], torch.Generator().manual_seed(42))
                                       for p in _prompts])
    resultats_spec = {}
    for _nom, _br in [("élève distillé", eleve_distille), ("élève sans maître", eleve_dur)]:
        for _gamma in config["spec_gammas"]:
            _res, _t = _chrono(lambda: [generer_speculatif(maitre, _br, p, config["spec_n_generes"], _gamma,
                                                            torch.Generator().manual_seed(42)) for p in _prompts])
            _appels = sum(r[1] for r in _res)
            _acc, _prop = sum(r[2] for r in _res), sum(r[3] for r in _res)
            _n_tok = config["spec_n_prompts"] * config["spec_n_generes"]
            resultats_spec[(_nom, _gamma)] = dict(acceptation=_acc / _prop, tokens_par_appel=_n_tok / _appels,
                                                   temps=_t, acceleration=temps_simple / _t)
            print(f"{_nom:18s} γ = {_gamma} : acceptation {_acc / _prop:5.1%}, {_n_tok / _appels:.2f} tokens par appel au maître, "
                  f"{_t:.1f} s contre {temps_simple:.1f} s (× {temps_simple / _t:.2f})")
    return resultats_spec, temps_simple, vt_brouillon, vt_speculatif


@app.cell
def _(
    SMOKE,
    courbe_distille,
    courbe_distille_haut,
    courbe_dur,
    distributions,
    mesures_kv,
    mo,
    np,
    perte_maitre,
    plt,
    resultats_ech,
    resultats_quant,
    resultats_spec,
    sauvegarder_figure,
):
    # --- Figure : six panneaux ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    # B : temps par pas
    _ax = axes[0, 0]
    for (_dev, _avec), _r in mesures_kv.items():
        _ax.plot(np.arange(1, len(_r["temps"]) + 1), 1000 * np.array(_r["temps"]),
                 label=f"{_dev.upper()}, {'avec' if _avec else 'sans'} cache", ls="-" if _avec else "--")
    _ax.set(xlabel="longueur du contexte (tokens)", ylabel="temps du pas (ms)", title="B. Cache KV", yscale="log")
    _ax.legend(fontsize=8)
    # D : distribution du caractère suivant
    _ax = axes[0, 1]
    _ordre = np.argsort(-distributions[1.0]["p"])[:12]
    for _k, _tau in enumerate((0.5, 1.0, 1.5)):
        _ax.bar(np.arange(12) + (_k - 1) * 0.27, distributions[_tau]["p"][_ordre], width=0.27,
                label=f"τ = {_tau} (noyau top-p 0,9 : {distributions[_tau]['noyau']})")
    _ax.set(xlabel="12 caractères les plus probables (rang)", ylabel="probabilité", title="D. Température")
    _ax.legend(fontsize=8)
    # D : qualité contre diversité
    _ax = axes[0, 2]
    for _nom, _r in resultats_ech.items():
        _ax.plot(_r["trigrammes_distincts"], _r["mots_connus"], "o")
        _ax.annotate(_nom, (_r["trigrammes_distincts"], _r["mots_connus"]), fontsize=8, xytext=(4, 2), textcoords="offset points")
    _ax.set(xlabel="3-grammes de mots distincts (diversité)", ylabel="mots existants (qualité)", title="D. Échantillonnage")
    # E : quantification
    _ax = axes[1, 0]
    for _nom, _r in resultats_quant.items():
        _ax.plot(_r["bits"], _r["perte"], "o")
        _ax.annotate(_nom, (_r["bits"], _r["perte"]), fontsize=7, xytext=(4, 2), textcoords="offset points")
    _ax.axhline(perte_maitre, color="0.6", ls="--", lw=1)
    _ax.set(xscale="log", xlabel="bits par poids", ylabel="perte de validation", title="E. Quantification",
            ylim=(perte_maitre - 0.05, min(5.0, max(r["perte"] for r in resultats_quant.values()) + 0.1)))
    # F : distillation
    _ax = axes[1, 1]
    _ax.plot(*zip(*courbe_dur), "o-", label="élève sans maître")
    _ax.plot(*zip(*courbe_distille), "s-", label="élève distillé, τ = 1")
    _ax.plot(*zip(*courbe_distille_haut), "^-", label="élève distillé, τ = 2, λ = 0,9")
    _ax.axhline(perte_maitre, color="0.6", ls="--", lw=1, label=f"maître ({perte_maitre:.2f})")
    _ax.set(xlabel="pas", ylabel="perte de validation", title="F. Distillation")
    _ax.legend(fontsize=9)
    # G : décodage spéculatif
    _ax = axes[1, 2]
    for _nom in ("élève distillé", "élève sans maître"):
        _g = sorted(k[1] for k in resultats_spec if k[0] == _nom)
        _ax.plot(_g, [resultats_spec[(_nom, g)]["tokens_par_appel"] for g in _g], "o-", label=_nom)
    _ax.set(xlabel="γ (tokens proposés par le brouillon)", ylabel="tokens produits par appel au maître",
            title="G. Décodage spéculatif")
    _ax.legend(fontsize=9)
    fig.tight_layout()
    _chemins = sauvegarder_figure(fig, "resultats-smoke" if SMOKE else "resultats", mo.notebook_dir() / "figures")
    print(f"Figure : {_chemins[0]}")
    fig


@app.cell
def _(
    SMOKE,
    cache_qwen,
    courbe_distille,
    courbe_dur,
    distributions,
    ecart_flash,
    mesures_kv,
    mo,
    np,
    octets_blocs,
    octets_cache,
    octets_standard,
    perte_maitre,
    resultats_ech,
    resultats_gqa,
    resultats_quant,
    resultats_spec,
    temps_simple,
    vt_brouillon,
    vt_speculatif,
):
    # --- Résultats bruts pour les figures du site (site/scripts/figures/ch28_*.py) ---
    _dossier = mo.notebook_dir() / "figures"
    _dossier.mkdir(exist_ok=True)
    _q = list(resultats_quant)
    _e = list(resultats_ech)
    _s = list(resultats_spec)
    np.savez(_dossier / ("resultats-smoke.npz" if SMOKE else "resultats.npz"),
             perte_maitre=perte_maitre, ecart_flash=ecart_flash, octets_standard=octets_standard, octets_blocs=octets_blocs,
             octets_cache=octets_cache,
             kv_cles=np.array([f"{d}|{int(a)}" for d, a in mesures_kv]),
             kv_temps=np.array([mesures_kv[k]["temps"] for k in mesures_kv]),
             cache_qwen_noms=np.array(list(cache_qwen)), cache_qwen_octets=np.array(list(cache_qwen.values())),
             gqa_groupes=np.array(list(resultats_gqa)),
             gqa_avant=np.array([r["avant"] for r in resultats_gqa.values()]),
             gqa_apres=np.array([r["apres"] for r in resultats_gqa.values()]),
             ech_noms=np.array(_e), ech_connus=np.array([resultats_ech[k]["mots_connus"] for k in _e]),
             ech_distincts=np.array([resultats_ech[k]["trigrammes_distincts"] for k in _e]),
             ech_perte=np.array([resultats_ech[k]["perte"] for k in _e]),
             distrib_p=np.array([distributions[t]["p"] for t in (0.5, 1.0, 1.5)]),
             distrib_noyau=np.array([distributions[t]["noyau"] for t in (0.5, 1.0, 1.5)]),
             quant_noms=np.array(_q), quant_bits=np.array([resultats_quant[k]["bits"] for k in _q]),
             quant_perte=np.array([resultats_quant[k]["perte"] for k in _q]),
             kd_dur=np.array(courbe_dur), kd_distille=np.array(courbe_distille), kd_distille_haut=np.array(courbe_distille_haut),
             spec_noms=np.array([f"{n}|{g}" for n, g in _s]),
             spec_acceptation=np.array([resultats_spec[k]["acceptation"] for k in _s]),
             spec_tokens_par_appel=np.array([resultats_spec[k]["tokens_par_appel"] for k in _s]),
             spec_acceleration=np.array([resultats_spec[k]["acceleration"] for k in _s]),
             temps_simple=temps_simple, vt_speculatif=vt_speculatif, vt_brouillon=vt_brouillon)
    print(f"Résultats bruts : {_dossier / ('resultats-smoke.npz' if SMOKE else 'resultats.npz')}")


@app.cell
def _(
    SMOKE,
    config,
    ecart_flash,
    ecart_kv,
    ecart_prefill,
    mesures_kv,
    mo,
    perte_eleve_distille,
    perte_eleve_dur,
    perte_maitre,
    resultats_ech,
    resultats_gqa,
    resultats_quant,
    resultats_spec,
    vt_speculatif,
):
    # --- Bilan : seuils du README, vérifiés en mode complet ---
    _acc_kv = sum(mesures_kv[("mps", False)]["temps"]) / sum(mesures_kv[("mps", True)]["temps"]) \
        if ("mps", True) in mesures_kv else float("nan")
    _acc_cpu = sum(mesures_kv[("cpu", False)]["temps"]) / sum(mesures_kv[("cpu", True)]["temps"])
    _g = config["quant_groupe"]
    _spec = resultats_spec[("élève distillé", 4)]
    seuils = {
        "attention par blocs = standard (< 1e-12)": ecart_flash < 1e-12,
        "cache KV = passage complet (< 1e-3)": max(ecart_kv, ecart_prefill) < 1e-3,
        "maître : perte de validation ≤ 1,60": perte_maitre <= 1.60,
        "cache KV : génération ≥ 1,5 × plus rapide sur CPU": _acc_cpu >= 1.5,
        "GQA 2 têtes K/V après uptraining : ≤ 0,05 au-dessus du MHA réentraîné": resultats_gqa[2]["apres"] <= resultats_gqa[4]["apres"] + 0.05,
        "MQA : perte juste après la moyenne > perte après uptraining": resultats_gqa[1]["avant"] > resultats_gqa[1]["apres"],
        "glouton : diversité plus faible que top-p 0,9": resultats_ech["glouton"]["trigrammes_distincts"] < resultats_ech["top-p 0,9"]["trigrammes_distincts"],
        "τ = 2 : moins de mots existants que top-p 0,9": resultats_ech["τ = 2"]["mots_connus"] < resultats_ech["top-p 0,9"]["mots_connus"],
        "int8 par ligne : ≤ 0,01 au-dessus du float32": resultats_quant["int8, une échelle par ligne"]["perte"] <= perte_maitre + 0.01,
        f"int4 groupes de {_g} meilleur que int4 par tenseur": resultats_quant[f"int4, groupes de {_g}"]["perte"] < resultats_quant["int4, une échelle par tenseur"]["perte"],
        "ternaire après coup : perte ≥ 0,5 au-dessus du float32": resultats_quant["ternaire (1,58 bit), après coup"]["perte"] >= perte_maitre + 0.5,
        "distillation (τ = 1) : élève ≥ 0,02 nat meilleur que sans maître": perte_eleve_distille <= perte_eleve_dur - 0.02,
        "spéculatif : distribution exacte (variation totale < 0,01)": vt_speculatif < 0.01,
        "spéculatif γ = 4, élève distillé : ≥ 2 tokens par appel au maître": _spec["tokens_par_appel"] >= 2.0,
    }
    for _k, _v in seuils.items():
        print(f"  [{'ok' if _v else 'ÉCHEC'}] {_k}")
    print(f"Accélération du cache KV : MPS × {_acc_kv:.2f}, CPU × {_acc_cpu:.2f}")
    _echecs = [k for k, v in seuils.items() if not v]
    if not SMOKE and _echecs:
        raise SystemExit("Seuils non atteints : " + " ; ".join(_echecs))
    mo.md(f"**Bilan** : {sum(seuils.values())}/{len(seuils)} seuils atteints" + (" (non vérifiés en smoke)" if SMOKE else "") + ".")


if __name__ == "__main__":
    app.run()
