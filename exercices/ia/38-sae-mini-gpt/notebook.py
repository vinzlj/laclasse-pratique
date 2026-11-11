# Exercice 38 : regarder dans la boîte noire.
#
#   make ex NUM=38              → éditeur marimo (bouton « Lancer »)
#   make ex NUM=38 MODE=run     → script complet, vérifie les seuils du README
#   make ex NUM=38 MODE=smoke   → test rapide (< 30 s)
#
# Du plus simple au plus fin :
#   A. superposition jouet : 5 caractéristiques rangées dans 2 dimensions (Elhage et al., 2022) ;
#   B. sondes linéaires sur le flux résiduel du mini-GPT de l'exercice 20a, modèle entraîné
#      contre modèle aléatoire ;
#   C. autoencodeur parcimonieux (SAE) sur les neurones du MLP de la couche 2 du mini-GPT :
#      reconstruction, parcimonie, perte récupérée, neurones contre caractéristiques ;
#   D. têtes d'induction : absentes du mini-GPT, puis apparition brutale dans un
#      Transformer à 2 couches entraîné à recopier ;
#   E. attributions sur un modèle tabulaire : valeurs de Shapley exactes (NumPy) contre la
#      bibliothèque shap, un raccourci démasqué, gradients intégrés, instabilité de LIME.
#
# Recharge data/checkpoints/mini-gpt-shakespeare.pt (exercice 20a) ; s'il manque, le
# réentraîne avec la même recette (≈ 3 min).

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
    from torch import nn

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
    # Exercice 38 : regarder dans la boîte noire

    Un réseau entraîné prédit bien, mais **pourquoi** ? On essaie ici, en miniature, les
    principales façons de répondre, de la plus extérieure (attribuer une prédiction aux
    entrées) à la plus intérieure (retrouver les « caractéristiques » et les circuits que le
    réseau a construits).

    | Partie | Méthode | Ce qu'on mesure |
    |---|---|---|
    | A | modèle jouet de superposition | combien de caractéristiques 2 dimensions peuvent ranger, selon leur rareté |
    | B | sondes linéaires | ce qu'on peut lire, couche par couche, dans le flux résiduel du mini-GPT |
    | C | autoencodeur parcimonieux (SAE) | reconstruction, parcimonie, perte récupérée ; neurones contre caractéristiques |
    | D | têtes d'induction | un circuit de copie : absent du mini-GPT, apparition brutale dans un modèle à 2 couches |
    | E | Shapley, SHAP, gradients intégrés, LIME | attributions exactes, raccourci démasqué, axiomes vérifiés |

    **Pertes optimisées.** A : erreur quadratique pondérée par l'importance de chaque
    caractéristique. B : entropie croisée (régression logistique). C : **MSE + λ·L1** du SAE,
    $\mathcal{L} = \lVert x - \hat x\rVert_2^2 + \lambda \lVert f(x)\rVert_1$, où
    $f(x) = \mathrm{ReLU}\big(W_e (x - b_d) + b_e\big)$ et $\hat x = W_d f(x) + b_d$, colonnes de
    $W_d$ de norme 1. D : entropie croisée du token suivant (chapitre 20). E : erreur
    quadratique (gradient boosting, petit MLP) ; les attributions elles-mêmes n'optimisent rien.
    Le mini-GPT n'est **jamais** modifié : on l'observe.
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
        # A. superposition : n caractéristiques, m dimensions, importance 0,9^i (i = 0 à 4)
        sup_n=5, sup_m=2, sup_sparsites=[0.0, 0.7, 0.9, 0.97], sup_pas=200 if SMOKE else 4000,
        sup_lot=1024, sup_lr=1e-2,
        # B. sondes : fenêtres de validation, positions tirées par fenêtre
        sonde_fenetres=40 if SMOKE else 400, sonde_positions=24,
        # C. SAE sur les 1 024 neurones du MLP de la couche 2 (indice 1), dictionnaire × 2
        sae_couche=1, sae_facteur=2, sae_lambda=3.0, sae_pas=20 if SMOKE else 2500,
        sae_fenetres_lot=16, sae_lr=1e-3, sae_fenetres_eval=16 if SMOKE else 200, sae_top=30,
        # D. têtes d'induction : séquence aléatoire de L tokens répétée deux fois
        ind_vocab=64, ind_d=64, ind_tetes=4, ind_pas=60 if SMOKE else 4000,
        ind_lot=64, ind_lr=1e-3, ind_mesure_tous=10 if SMOKE else 50,
        # E. attributions tabulaires
        shap_fond=20 if SMOKE else 100, shap_points=3 if SMOKE else 40, lime_graines=3 if SMOKE else 10,
        lime_echantillons=500, lime_petit=30, ig_pas=64, mlp_epoques=50 if SMOKE else 600,
        # Réentraînement du mini-GPT si le checkpoint manque (recette de l'exercice 20a)
        maitre_n_pas=30 if SMOKE else 1500,
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, périphérique {device}")
    print(config)
    return SEED, SMOKE, config, device


@app.cell
def _(mo):
    bouton = mo.ui.run_button(label="Lancer")
    bouton
    return (bouton,)


@app.cell
def _(mo):
    mo.md(r"""
    ## A. Superposition : plus de caractéristiques que de dimensions

    Modèle jouet d'Elhage et al. (2022) : $n = 5$ caractéristiques $x_i$, compressées en
    $m = 2$ dimensions par $h = Wx$, puis reconstruites par $x' = \mathrm{ReLU}(W^\top W x + b)$.
    Chaque $x_i$ vaut 0 avec la probabilité $S$ (la **rareté**), sinon un nombre uniforme
    entre 0 et 1 ; l'erreur de la caractéristique $i$ compte avec le poids $0{,}9^i$.
    Avec des caractéristiques fréquentes ($S = 0$), le mieux est de n'en garder que deux,
    orthogonales, comme le ferait une ACP. Quand elles deviennent rares, deux caractéristiques
    sont rarement actives en même temps : le réseau en range **plus que de dimensions**, au
    prix d'interférences que le ReLU et le biais négatif filtrent. Une caractéristique est
    « représentée » quand sa colonne $W_i$ a une norme supérieure à 0,5.
    """)


@app.cell
def _(bouton, config, mo, np, time, torch):
    mo.stop(mo.running_in_notebook() and not bouton.value, mo.md("*Cliquez sur « Lancer ».*"))

    def entrainer_superposition(S, graine=42):
        """Renvoie W (m, n), b (n) et la perte finale du modèle ReLU(WᵀWx + b)."""
        _g = torch.Generator().manual_seed(graine)
        n, m = config["sup_n"], config["sup_m"]
        W = torch.nn.Parameter(torch.randn(m, n, generator=_g) * 0.1)
        b = torch.nn.Parameter(torch.zeros(n))
        importance = 0.9 ** torch.arange(n, dtype=torch.float32)
        opt = torch.optim.AdamW([W, b], lr=config["sup_lr"], weight_decay=0.0)
        perte = None
        for _pas in range(config["sup_pas"]):
            # x_i = 0 avec la probabilité S, sinon uniforme sur [0, 1]
            x = torch.rand(config["sup_lot"], n, generator=_g) * (torch.rand(config["sup_lot"], n, generator=_g) >= S)
            x_rec = torch.relu(x @ W.T @ W + b)
            perte = (importance * (x - x_rec) ** 2).sum(-1).mean()
            opt.zero_grad()
            perte.backward()
            opt.step()
        return W.detach().numpy(), b.detach().numpy(), perte.item()

    _t0 = time.perf_counter()
    sup_W, sup_b, sup_representees = [], [], []
    for _S in config["sup_sparsites"]:
        _W, _b, _p = entrainer_superposition(_S)
        _normes = np.linalg.norm(_W, axis=0)
        sup_W.append(_W)
        sup_b.append(_b)
        sup_representees.append(int((_normes > 0.5).sum()))
        print(f"S = {_S:4.2f} : perte {_p:.4f}, normes ‖W_i‖ = {np.round(_normes, 2)}, "
              f"biais = {np.round(_b, 2)} → {sup_representees[-1]} caractéristiques représentées")
    sup_W, sup_b = np.array(sup_W), np.array(sup_b)
    duree_sup = time.perf_counter() - _t0
    print(f"Partie A : {duree_sup:.1f} s")
    return duree_sup, sup_W, sup_b, sup_representees


@app.cell
def _(mo):
    mo.md(r"""
    ## Le mini-GPT de l'exercice 20a, et un passage avant « instrumenté »

    `passe_detaillee` refait exactement le calcul de `MiniGPT.forward` en gardant au passage le
    flux résiduel après chaque couche, les poids d'attention de chaque tête et les activations
    des 1 024 neurones de chaque MLP. Elle accepte aussi une fonction `remplacer` qui modifie
    les neurones d'une couche (pour brancher le SAE ou les mettre à zéro).
    """)


@app.cell
def _(
    ConfigGPT,
    F,
    MiniGPT,
    SMOKE,
    bouton,
    charger_tinyshakespeare,
    checkpoints_dir,
    config,
    device,
    lot_aleatoire,
    math,
    mo,
    time,
    torch,
):
    mo.stop(mo.running_in_notebook() and not bouton.value)
    texte = charger_tinyshakespeare()
    caracteres = sorted(set(texte))
    vers_id = {c: i for i, c in enumerate(caracteres)}
    _n = int(config["part_train"] * len(texte))
    texte_val = texte[_n:]
    ids_train = torch.tensor([vers_id[c] for c in texte[:_n]], dtype=torch.long)
    ids_val = torch.tensor([vers_id[c] for c in texte_val], dtype=torch.long)

    def reentrainer(chemin):
        """Recette de l'exercice 20a (4 couches, 256 dimensions, 1 500 pas de 64 × 128)."""
        print(f"Checkpoint absent : réentraînement du mini-GPT ({config['maitre_n_pas']} pas, ≈ 3 min)…")
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
    if _chemin.exists():
        gpt, _infos = MiniGPT.charger(_chemin, device=device)
        assert _infos["caracteres"] == caracteres, "vocabulaire du checkpoint différent"
        print(f"Mini-GPT rechargé : {_chemin.name} (pas {_infos.get('pas')}, val {_infos.get('val_loss', float('nan')):.3f})")
    else:
        gpt = reentrainer(_chemin)
    gpt.eval()
    for _p in gpt.parameters():
        _p.requires_grad_(False)

    @torch.no_grad()
    def passe_detaillee(modele, idx, couche=None, remplacer=None):
        """Même calcul que MiniGPT.forward (mode évaluation), en gardant les intermédiaires.

        Renvoie (logits, résidus, attentions, neurones) :
          résidus[l] = flux résiduel avant la couche l (résidus[0] = plongements), longueur L + 1 ;
          attentions[l] = poids d'attention (B, têtes, T, T) de la couche l ;
          neurones[l] = activations GELU du MLP de la couche l (B, T, 4 × d).
        Si `remplacer` est donné, les neurones de la couche `couche` sont remplacés par
        remplacer(neurones) avant la projection de sortie du MLP.
        """
        B, T = idx.shape
        x = modele.emb_tokens(idx) + modele.emb_positions(torch.arange(T, device=idx.device))
        residus, attentions, neurones = [x], [], []
        causal = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
        for l, bloc in enumerate(modele.blocs):
            a = bloc.attn
            q, k, v = a.qkv(bloc.ln1(x)).split(x.size(-1), dim=2)
            q, k, v = (t.view(B, T, a.n_tetes, -1).transpose(1, 2) for t in (q, k, v))
            att = (q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))).masked_fill(~causal, float("-inf")).softmax(-1)
            attentions.append(att)
            x = x + a.proj((att @ v).transpose(1, 2).reshape(B, T, -1))
            h = F.gelu(bloc.mlp.fc(bloc.ln2(x)))
            if remplacer is not None and l == couche:
                h = remplacer(h)
            neurones.append(h)
            x = x + bloc.mlp.proj(h)
            residus.append(x)
        return modele.tete(modele.ln_final(x)), residus, attentions, neurones

    _idx = ids_val[:128].unsqueeze(0).to(device)
    _l1 = passe_detaillee(gpt, _idx)[0]
    _l2, _ = gpt(_idx)
    ecart_passe = (_l1 - _l2).abs().max().item()
    print(f"passe_detaillee = MiniGPT.forward : écart maximal des logits {ecart_passe:.1e}")
    _t0 = time.perf_counter()
    return caracteres, ecart_passe, gpt, ids_train, ids_val, passe_detaillee, texte_val, vers_id


@app.cell
def _(mo):
    mo.md(r"""
    ## B. Sondes linéaires : qu'est-ce qui est lisible, et où ?

    Une **sonde** (Alain et Bengio, 2016) est un classifieur linéaire entraîné sur les
    activations figées d'une couche. Si elle réussit, l'information est présente **et**
    lisible linéairement à cet endroit ; cela ne prouve pas que le modèle s'en sert. Deux
    propriétés, mesurées en précision équilibrée (50 % = hasard) :

    1. **le caractère précédent est une voyelle** : à la couche 0 (plongements du caractère
       courant et de sa position), l'information n'y est pas ; il faut qu'une tête d'attention
       aille la chercher ;
    2. **on est dans la première ligne d'une réplique** (la ligne qui suit « ROMEO: » ou
       « First Citizen: »), mesurée sur les minuscules seulement : rien dans le caractère
       courant ne le dit, il faut regarder plusieurs dizaines de caractères en arrière.

    Témoin indispensable : le **même réseau aux poids aléatoires**. Une sonde assez riche
    trouve parfois de l'information même dans un réseau qui n'a rien appris.
    """)


@app.cell
def _(
    ConfigGPT,
    MiniGPT,
    caracteres,
    config,
    device,
    gpt,
    ids_val,
    np,
    passe_detaillee,
    texte_val,
    time,
    torch,
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    _t0 = time.perf_counter()
    # Étiquettes calculées sur le texte de validation, caractère par caractère
    _voyelles = set("aeiouyAEIOUY")
    _prec_voyelle = np.zeros(len(texte_val), dtype=bool)
    _prec_voyelle[1:] = [c in _voyelles for c in texte_val[:-1]]
    # « Première ligne de réplique » : la ligne qui suit un nom de personnage (ligne terminée
    # par « : », précédée d'une ligne vide), par exemple « Hark, who comes here? » après « ROMEO: »
    _premiere = np.zeros(len(texte_val), dtype=bool)
    _debut = 0
    _lignes = texte_val.split("\n")
    for _i, _ligne in enumerate(_lignes):
        _nom_avant = _i >= 1 and _lignes[_i - 1].endswith(":") and 0 < len(_lignes[_i - 1]) <= 40 \
            and (_i == 1 or _lignes[_i - 2] == "")
        if _nom_avant:
            _premiere[_debut:_debut + len(_ligne)] = True
        _debut += len(_ligne) + 1
    # Minuscules seulement : la casse du caractère courant ne trahit rien
    _minuscule = np.array([c.islower() for c in texte_val])

    # Fenêtres de validation fixes ; dans chacune, positions tirées au hasard (≥ 32 : assez de contexte)
    _g = np.random.default_rng(42)
    _T = config["taille_bloc"]
    _debuts = _g.choice(len(ids_val) - _T - 1, config["sonde_fenetres"], replace=False)
    _pos = np.stack([_g.choice(np.arange(32, _T), config["sonde_positions"], replace=False)
                     for _ in _debuts])
    _X_idx = torch.stack([ids_val[d:d + _T] for d in _debuts]).to(device)
    _abs = _debuts[:, None] + _pos                       # indice dans texte_val de chaque position sondée
    _coupe = int(0.75 * len(_debuts))                    # 75 % des fenêtres pour entraîner la sonde

    def activations_residus(modele):
        _res = []
        for _i in range(0, len(_X_idx), 50):
            _r = passe_detaillee(modele, _X_idx[_i:_i + 50])[1]
            _res.append(torch.stack(_r).cpu())           # (L + 1, b, T, d)
        _res = torch.cat(_res, dim=1).numpy()
        return np.stack([_res[:, f, _pos[f]] for f in range(len(_debuts))], axis=1)  # (L+1, fenêtres, positions, d)

    def sonder(res, etiquettes, masque):
        """Précision équilibrée de la sonde sur les fenêtres de test, pour chaque couche."""
        y = etiquettes[_abs]
        m = masque[_abs]
        tr = np.zeros_like(m)
        tr[:_coupe] = True
        scores = []
        for couche in range(res.shape[0]):
            X = res[couche]
            Xtr, ytr = X[tr & m], y[tr & m]
            Xte, yte = X[~tr & m], y[~tr & m]
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
            clf.fit(sc.transform(Xtr), ytr)
            scores.append(balanced_accuracy_score(yte, clf.predict(sc.transform(Xte))))
        return np.array(scores)

    torch.manual_seed(42)
    gpt_aleatoire = MiniGPT(ConfigGPT(taille_vocab=len(caracteres))).to(device).eval()
    _proprietes = {"précédent = voyelle": (_prec_voyelle, np.ones_like(_minuscule)),
                   "première ligne de réplique": (_premiere, _minuscule)}
    sondes = {}
    for _nom_modele, _modele in [("entraîné", gpt), ("aléatoire", gpt_aleatoire)]:
        _res = activations_residus(_modele)
        for _prop, (_etiq, _masque) in _proprietes.items():
            sondes[(_prop, _nom_modele)] = sonder(_res, _etiq, _masque)
            print(f"{_prop:28s} modèle {_nom_modele:9s} : précision équilibrée par couche "
                  + " ".join(f"{100 * s:5.1f} %" for s in sondes[(_prop, _nom_modele)]))
    print(f"Part de « première ligne de réplique » parmi les minuscules sondées : "
          f"{100 * _premiere[_abs][_minuscule[_abs]].mean():.1f} %")
    duree_sondes = time.perf_counter() - _t0
    print(f"Partie B : {duree_sondes:.1f} s")
    return duree_sondes, sondes


@app.cell
def _(mo):
    mo.md(r"""
    ## C. Un autoencodeur parcimonieux (SAE) sur les neurones du mini-GPT

    Les 1 024 neurones du MLP de la couche 2 sont **polysémantiques** : chacun réagit à
    plusieurs choses sans rapport, signe de superposition (partie A). Le SAE (Bricken et
    al., 2023 ; Cunningham et al., 2023) réécrit chaque vecteur d'activations comme une
    somme **parcimonieuse** de 2 048 directions apprises (les « caractéristiques ») :

    $$f = \mathrm{ReLU}\big(W_e(x - b_d) + b_e\big), \qquad \hat x = W_d\, f + b_d, \qquad
    \mathcal{L} = \lVert x - \hat x\rVert_2^2 + \lambda\,\lVert f\rVert_1$$

    La MSE force à tout reconstruire, la pénalité L1 à n'allumer que peu de
    caractéristiques à la fois. Les colonnes de $W_d$ sont renormalisées à 1 après chaque pas
    (sinon le SAE tricherait : $f$ petit, $W_d$ grand). Les activations sont divisées par une
    échelle fixe pour que leur norme quadratique moyenne vaille la dimension (1 024).

    Mesures : part de variance non expliquée (FVU), nombre moyen de caractéristiques
    actives par token ($L_0$), caractéristiques mortes, et surtout la **perte récupérée** :
    on remplace les neurones par leur reconstruction dans le mini-GPT et on compare
    l'entropie croisée à celle du modèle intact (100 %) et à celle du modèle dont ces neurones
    sont mis à zéro (0 %).
    """)


@app.cell
def _(
    F,
    config,
    device,
    gpt,
    ids_train,
    lot_aleatoire,
    math,
    passe_detaillee,
    time,
    torch,
):
    _t0 = time.perf_counter()
    COUCHE = config["sae_couche"]
    _D = 4 * gpt.cfg.n_embd
    _Fn = config["sae_facteur"] * _D
    _gen = torch.Generator().manual_seed(42)

    def neurones_de(idx):
        return passe_detaillee(gpt, idx)[3][COUCHE].reshape(-1, _D)

    # Échelle fixe : norme quadratique moyenne = dimension
    _x0 = neurones_de(lot_aleatoire(ids_train, 128, 64, device, _gen)[0])
    echelle_sae = (_x0.pow(2).sum(-1).mean() / _D).sqrt().item()
    densite_neurones = (_x0 > 0).float().mean().item()
    print(f"Échelle {echelle_sae:.3f} ; un neurone est positif sur {100 * densite_neurones:.1f} % des tokens")

    torch.manual_seed(42)
    sae = dict(
        W_e=torch.nn.Parameter(torch.randn(_D, _Fn, device=device) / math.sqrt(_D)),
        b_e=torch.nn.Parameter(torch.zeros(_Fn, device=device)),
        b_d=torch.nn.Parameter((_x0.mean(0) / echelle_sae).clone()),
    )
    sae["W_d"] = torch.nn.Parameter(sae["W_e"].data.T.clone())   # décodeur initialisé = encodeur transposé
    with torch.no_grad():
        sae["W_d"].data /= sae["W_d"].data.norm(dim=1, keepdim=True)

    def encoder_sae(x):
        return F.relu((x - sae["b_d"]) @ sae["W_e"] + sae["b_e"])

    def decoder_sae(f):
        return f @ sae["W_d"] + sae["b_d"]

    _opt = torch.optim.Adam(sae.values(), lr=config["sae_lr"])
    historique_sae = []
    _n = config["sae_pas"]
    for _pas in range(_n):
        _x = neurones_de(lot_aleatoire(ids_train, 128, config["sae_fenetres_lot"], device, _gen)[0]) / echelle_sae
        _f = encoder_sae(_x)
        _mse = (decoder_sae(_f) - _x).pow(2).sum(-1).mean()
        _l1 = _f.sum(-1).mean()                                   # f ≥ 0 : |f| = f
        _lam = config["sae_lambda"] * min(1.0, _pas / (0.05 * _n))  # λ monte en 5 % des pas
        _perte = _mse + _lam * _l1
        _opt.zero_grad()
        _perte.backward()
        _opt.step()
        with torch.no_grad():
            sae["W_d"].data /= sae["W_d"].data.norm(dim=1, keepdim=True)
        if _pas % 250 == 0 or _pas == _n - 1:
            _fvu = (_mse / (_x - _x.mean(0)).pow(2).sum(-1).mean()).item()
            historique_sae.append((_pas, _mse.item(), _l1.item(), (_f > 0).float().sum(-1).mean().item(), _fvu))
            print(f"pas {_pas:5d}  MSE {_mse.item():7.1f}  L1 {_l1.item():6.1f}  L0 {historique_sae[-1][3]:6.1f}  FVU {_fvu:.3f}")
    duree_sae = time.perf_counter() - _t0
    print(f"Entraînement du SAE ({_D} → {_Fn}) : {duree_sae:.1f} s")
    return COUCHE, decoder_sae, densite_neurones, duree_sae, echelle_sae, encoder_sae, historique_sae


@app.cell
def _(
    COUCHE,
    F,
    caracteres,
    config,
    decoder_sae,
    device,
    echelle_sae,
    encoder_sae,
    gpt,
    ids_val,
    np,
    passe_detaillee,
    torch,
):
    # --- Évaluation sur des fenêtres de validation fixes ---
    _g = torch.Generator().manual_seed(7)
    _T = config["taille_bloc"]
    _debuts = torch.randint(len(ids_val) - _T - 1, (config["sae_fenetres_eval"],), generator=_g)
    _X = torch.stack([ids_val[d:d + _T] for d in _debuts]).to(device)
    _Y = torch.stack([ids_val[d + 1:d + 1 + _T] for d in _debuts]).to(device)

    def entropie(logits):
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), _Y.reshape(-1)).item()

    _nb = 50
    _ce = {"intact": 0.0, "SAE": 0.0, "zéro": 0.0}
    _neur, _feat = [], []
    _mse = _var = 0.0
    with torch.no_grad():
        for _i in range(0, len(_X), _nb):
            _x = _X[_i:_i + _nb]
            _w = len(_x) / len(_X)
            _lg, _, _, _h = passe_detaillee(gpt, _x)
            _ce["intact"] += _w * F.cross_entropy(_lg.reshape(-1, _lg.size(-1)), _Y[_i:_i + _nb].reshape(-1)).item()
            _hn = _h[COUCHE].reshape(-1, _h[COUCHE].size(-1)) / echelle_sae
            _f = encoder_sae(_hn)
            _mse += (decoder_sae(_f) - _hn).pow(2).sum().item()
            _var += (_hn - _hn.mean(0)).pow(2).sum().item()
            _neur.append(_hn.cpu())
            _feat.append(_f.cpu())
            _lg_sae = passe_detaillee(gpt, _x, COUCHE, lambda h: decoder_sae(encoder_sae(h / echelle_sae)) * echelle_sae)[0]
            _lg_zero = passe_detaillee(gpt, _x, COUCHE, torch.zeros_like)[0]
            _ce["SAE"] += _w * F.cross_entropy(_lg_sae.reshape(-1, _lg_sae.size(-1)), _Y[_i:_i + _nb].reshape(-1)).item()
            _ce["zéro"] += _w * F.cross_entropy(_lg_zero.reshape(-1, _lg_zero.size(-1)), _Y[_i:_i + _nb].reshape(-1)).item()
    act_neurones = torch.cat(_neur).numpy()
    act_features = torch.cat(_feat).numpy()
    ce_sae = _ce
    fvu = _mse / _var
    l0 = float((act_features > 0).sum(1).mean())
    vivantes = (act_features > 0).any(0)
    perte_recuperee = (ce_sae["zéro"] - ce_sae["SAE"]) / (ce_sae["zéro"] - ce_sae["intact"])
    print(f"FVU {fvu:.3f} (variance expliquée {100 * (1 - fvu):.1f} %), L0 = {l0:.1f} caractéristiques actives "
          f"par token sur {act_features.shape[1]}, mortes : {100 * (1 - vivantes.mean()):.1f} %")
    print(f"Entropie croisée : intact {ce_sae['intact']:.3f}, avec le SAE {ce_sae['SAE']:.3f}, "
          f"neurones à zéro {ce_sae['zéro']:.3f} → perte récupérée {100 * perte_recuperee:.1f} %")

    # Contexte de chaque token évalué : caractère courant et suivant
    _ids = _X.cpu().numpy().reshape(-1)
    _suiv = _Y.cpu().numpy().reshape(-1)
    contexte_eval = (_X.cpu().numpy(), _ids, _suiv)

    def purete(acts, k):
        """Pour chaque unité active au moins k fois : part de ses k plus fortes activations qui
        partagent le même caractère courant, ou le même caractère suivant (on garde le max)."""
        res = []
        for j in range(acts.shape[1]):
            a = acts[:, j]
            if (a > 0).sum() < k:
                continue
            top = np.argpartition(-a, k)[:k]
            p_cour = np.bincount(_ids[top], minlength=len(caracteres)).max() / k
            p_suiv = np.bincount(_suiv[top], minlength=len(caracteres)).max() / k
            res.append(max(p_cour, p_suiv))
        return np.array(res)

    purete_neurones = purete(act_neurones, config["sae_top"])
    purete_features = purete(act_features, config["sae_top"])
    for _nom, _p in [("neurones", purete_neurones), ("caractéristiques SAE", purete_features)]:
        print(f"{_nom:22s} : {len(_p)} unités, pureté médiane {100 * np.median(_p):.0f} %, "
              f"≥ 90 % pour {100 * (_p >= 0.9).mean():.0f} % d'entre elles")
    return (
        act_features,
        ce_sae,
        contexte_eval,
        fvu,
        l0,
        perte_recuperee,
        purete_features,
        purete_neurones,
        vivantes,
    )


@app.cell
def _(act_features, caracteres, config, contexte_eval, np, vivantes):
    # --- Inspection : 8 caractéristiques tirées au hasard parmi les vivantes ---
    _X, _ids, _suiv = contexte_eval
    _T = config["taille_bloc"]
    _g = np.random.default_rng(42)
    _candidates = np.flatnonzero(vivantes & ((act_features > 0).sum(0) >= 10))
    exemples_features = []
    for _j in sorted(_g.choice(_candidates, min(8, len(_candidates)), replace=False)):
        _a = act_features[:, _j]
        _top = np.argsort(-_a)[:5]
        _extraits = []
        for _t in _top:
            _f, _p = divmod(int(_t), _T)
            _avant = "".join(caracteres[i] for i in _X[_f, max(0, _p - 15):_p])
            _extraits.append((_avant + "[" + caracteres[_X[_f, _p]] + "]").replace("\n", "⏎"))
        _freq = 100 * (_a > 0).mean()
        exemples_features.append((int(_j), _freq, _extraits))
        print(f"caractéristique {_j:4d} (active sur {_freq:.2f} % des tokens) :")
        for _e in _extraits:
            print(f"    …{_e}")
    return (exemples_features,)


@app.cell
def _(mo):
    mo.md(r"""
    ## D. Têtes d'induction : un circuit de copie

    Une **tête d'induction** (Olsson et al., 2022) complète « … A B … A » par « B » : elle
    cherche l'occurrence précédente du token courant et recopie celui qui la suivait. Il
    faut **deux** couches d'attention : une tête de la couche 1 note dans chaque position
    « le token d'avant était A », une tête de la couche 2 s'en sert pour trouver où « A »
    était suivi de quelque chose. Test classique : une séquence aléatoire de $L$ tokens
    répétée deux fois. Sur la seconde copie, un modèle qui a des têtes d'induction prédit
    presque parfaitement ; le **score d'induction** d'une tête est le poids d'attention moyen
    qu'elle met, depuis la position $t$ de la seconde copie, sur la position $t - L + 1$.

    1. Sur le mini-GPT de l'exercice 20a.
    2. Sur un Transformer « attention seule » (ni MLP ni normalisation) entraîné sur des
       séquences de 64 tokens aléatoires dont un segment de longueur $L$ est recopié juste
       après lui-même, avec 2 couches puis 1 couche. $L$ (6 à 24) et la position du segment
       changent à chaque lot : sinon une seule couche suffit, en apprenant « regarde $L - 1$
       positions en arrière » (un raccourci positionnel, constaté dans une première version
       de cet exercice où $L$ était fixe).
    """)


@app.cell
def _(F, caracteres, config, device, gpt, ids_val, passe_detaillee, torch, vers_id):
    # --- 1. Le mini-GPT a-t-il des têtes d'induction ? ---
    def tester_induction(modele, seq, passe):
        """seq : (B, L) ; renvoie (perte 1re copie, perte 2de copie, scores par couche et par tête)."""
        L = seq.size(1)
        rep = torch.cat([seq, seq], 1).to(device)
        lg, att = passe(modele, rep)
        pertes = F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)), rep[:, 1:].reshape(-1),
                                 reduction="none").view(rep.size(0), -1)
        t = torch.arange(L, 2 * L - 1, device=device)
        scores = torch.stack([a[:, :, t, t - L + 1].mean((0, 2)) for a in att]).cpu().numpy()
        return pertes[:, :L - 1].mean().item(), pertes[:, L:].mean().item(), scores

    def passe_gpt(modele, idx):
        lg, _, att, _ = passe_detaillee(modele, idx)
        return lg, att

    _g = torch.Generator().manual_seed(42)
    _L = 60
    _deb = torch.randint(0, len(ids_val) - _L, (64,), generator=_g)
    _texte = torch.stack([ids_val[i:i + _L] for i in _deb])
    _minus = torch.tensor([vers_id[c] for c in "abcdefghijklmnopqrstuvwxyz"])
    _alea = _minus[torch.randint(0, 26, (64, _L), generator=_g)]
    induction_gpt = {}
    for _nom, _seq in [("extraits de validation", _texte), ("minuscules aléatoires", _alea)]:
        _p1, _p2, _sc = tester_induction(gpt, _seq, passe_gpt)
        induction_gpt[_nom] = (_p1, _p2, float(_sc.max()))
        print(f"Mini-GPT, {_nom} répétés : perte 1re copie {_p1:.2f}, 2de copie {_p2:.2f}, "
              f"meilleur score d'induction {_sc.max():.3f} (sur {_sc.size} têtes)")
    _ = caracteres
    return (induction_gpt,)


@app.cell
def _(config, device, math, nn, np, time, torch):
    # --- 2. Un Transformer « attention seule » entraîné sur des séquences répétées ---
    class AttentionSeule(nn.Module):
        """Plongements + n couches d'attention causale (sans MLP ni normalisation) + projection."""

        def __init__(self, n_couches, V, d, h, T):
            super().__init__()
            self.h = h
            self.emb = nn.Embedding(V, d)
            self.pos = nn.Embedding(T, d)
            self.qkv = nn.ModuleList([nn.Linear(d, 3 * d, bias=False) for _ in range(n_couches)])
            self.proj = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(n_couches)])
            self.sortie = nn.Linear(d, V, bias=False)

        def forward(self, idx):
            B, T = idx.shape
            x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))
            causal = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
            atts = []
            for qkv, proj in zip(self.qkv, self.proj):
                q, k, v = qkv(x).split(x.size(-1), dim=2)
                q, k, v = (t.view(B, T, self.h, -1).transpose(1, 2) for t in (q, k, v))
                att = (q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))).masked_fill(~causal, float("-inf")).softmax(-1)
                atts.append(att)
                x = x + proj((att @ v).transpose(1, 2).reshape(B, T, -1))
            return self.sortie(x), atts

    T_JOUET = 64

    def lot_repete(B, L, P, g):
        """B séquences de T_JOUET tokens aléatoires où x[P+L : P+2L] recopie x[P : P+L].

        L et P changent d'un lot à l'autre : la position seule ne dit pas où regarder,
        il faut retrouver l'occurrence précédente du token courant.
        """
        x = torch.randint(0, config["ind_vocab"], (B, T_JOUET), generator=g)
        x[:, P + L:P + 2 * L] = x[:, P:P + L]
        return x

    _g_test = torch.Generator().manual_seed(1)
    LOTS_TEST = []
    for _L in (8, 13, 18, 23):
        _P = int(torch.randint(0, T_JOUET - 2 * _L + 1, (1,), generator=_g_test))
        LOTS_TEST.append((lot_repete(64, _L, _P, _g_test).to(device), _L, _P))

    @torch.no_grad()
    def mesurer_jouet(m):
        """Perte sur la 1re et la 2de copie, score d'induction (B, couches, têtes) moyenné sur les lots test."""
        p1 = p2 = 0.0
        sc = 0.0
        for x, L, P in LOTS_TEST:
            lg, att = m(x)
            pertes = nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1),
                                                 reduction="none").view(x.size(0), -1)
            p1 += pertes[:, P:P + L - 1].mean().item() / len(LOTS_TEST)       # imprévisible
            p2 += pertes[:, P + L:P + 2 * L - 1].mean().item() / len(LOTS_TEST)  # prévisible par copie
            t = torch.arange(P + L, P + 2 * L - 1, device=device)
            sc = sc + torch.stack([a[:, :, t, t - L + 1].mean((0, 2)) for a in att]).cpu().numpy() / len(LOTS_TEST)
        return p1, p2, sc

    def entrainer_jouet(n_couches):
        torch.manual_seed(42)
        V = config["ind_vocab"]
        m = AttentionSeule(n_couches, V, config["ind_d"], config["ind_tetes"], T_JOUET).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=config["ind_lr"])
        g = torch.Generator().manual_seed(42)
        histo = []
        for pas in range(config["ind_pas"] + 1):
            if pas % config["ind_mesure_tous"] == 0:
                m.eval()
                p1, p2, sc = mesurer_jouet(m)
                histo.append((pas, p1, p2, float(sc[-1].max()), float(sc.max())))
                m.train()
            if pas == config["ind_pas"]:
                break
            L = int(torch.randint(6, 25, (1,), generator=g))
            P = int(torch.randint(0, T_JOUET - 2 * L + 1, (1,), generator=g))
            x = lot_repete(config["ind_lot"], L, P, g).to(device)
            lg, _ = m(x)
            perte = nn.functional.cross_entropy(lg[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
            opt.zero_grad()
            perte.backward()
            opt.step()
        return np.array(histo), sc

    _t0 = time.perf_counter()
    induction_2c, scores_2c = entrainer_jouet(2)
    induction_1c, scores_1c = entrainer_jouet(1)
    duree_induction = time.perf_counter() - _t0
    print(f"ln({config['ind_vocab']}) = {math.log(config['ind_vocab']):.2f} : perte d'une prédiction au hasard")
    for _nom, _h, _sc in [("2 couches", induction_2c, scores_2c), ("1 couche", induction_1c, scores_1c)]:
        _franchi = _h[_h[:, 2] < 2.0, 0]
        print(f"{_nom} : perte 2de copie {_h[0, 2]:.2f} → {_h[-1, 2]:.2f} ; 1re copie {_h[-1, 1]:.2f} ; "
              f"score d'induction de la dernière couche {_h[-1, 3]:.2f} ; "
              f"perte 2de copie < 2 à partir du pas {int(_franchi[0]) if len(_franchi) else '—'}")
        print("   scores par couche et par tête :", np.round(_sc, 2).tolist())
    print(f"Partie D : {duree_induction:.1f} s")
    return duree_induction, induction_1c, induction_2c, scores_2c


@app.cell
def _(mo):
    mo.md(r"""
    ## E. Attributions sur un modèle tabulaire : Shapley, SHAP, gradients intégrés, LIME

    Jeu `diabetes` de scikit-learn (442 patients, 10 variables standardisées, cible : une
    mesure de progression de la maladie un an après), modèle : gradient boosting. Pour une
    prédiction $f(x)$, la **valeur de Shapley** de la variable $j$ est sa contribution
    marginale moyenne sur tous les ordres d'arrivée des variables :

    $$\phi_j = \sum_{S \subseteq N\setminus\{j\}} \frac{|S|!\,(d-|S|-1)!}{d!}\,\big(v(S\cup\{j\}) - v(S)\big),
    \qquad v(S) = \mathbb{E}_{z \sim \text{fond}}\big[f(x_S, z_{\bar S})\big]$$

    Avec $d = 10$ variables, on peut **tout** énumérer ($2^{10} = 1\,024$ coalitions) : c'est la
    référence exacte, que la bibliothèque `shap` (Lundberg et Lee, 2017) doit retrouver. Puis :
    une colonne « fuite » glissée dans les données (un raccourci) ; les gradients intégrés
    (Sundararajan et al., 2017) sur un petit MLP et leur axiome de complétude ; la stabilité
    de LIME (Ribeiro et al., 2016) d'une graine à l'autre.
    """)


@app.cell
def _(config, np, time):
    import shap
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split

    _t0 = time.perf_counter()
    _donnees = load_diabetes(as_frame=True)
    noms_var = list(_donnees.data.columns)
    # Les arbres de scikit-learn comparent en float32 : on arrondit les données une fois pour
    # toutes, sinon un point tout près d'un seuil peut tomber d'un côté pour sklearn et de
    # l'autre pour shap (écarts de l'ordre de 0,1 constatés sans cet arrondi).
    _X = _donnees.data.values.astype(np.float32).astype(np.float64)
    _y = _donnees.target.values.astype(np.float64)
    Xtr, Xte, ytr, yte = train_test_split(_X, _y, test_size=0.25, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                   subsample=0.8, random_state=42).fit(Xtr, ytr)
    r2_gb = gb.score(Xte, yte)
    fond = Xtr[np.random.default_rng(42).choice(len(Xtr), config["shap_fond"], replace=False)]

    def shapley_exact(f, x, fond):
        """Valeurs de Shapley exactes de f en x, variables absentes tirées du fond (interventionnel).

        Coalition S = entier m sur d bits (bit j à 1 : la variable j prend la valeur de x).
        """
        d = len(x)
        m = np.arange(2 ** d)
        presentes = ((m[:, None] >> np.arange(d)) & 1).astype(bool)          # (2^d, d)
        Z = np.where(presentes[:, None, :], x, fond[None, :, :])              # (2^d, n_fond, d)
        v = f(Z.reshape(-1, d)).reshape(2 ** d, len(fond)).mean(1)            # v(S) pour chaque S
        taille = presentes.sum(1)
        fact = np.array([math_fact(k) for k in range(d + 1)], dtype=np.float64)
        phi = np.zeros(d)
        for j in range(d):
            sans_j = ~presentes[:, j]
            s = taille[sans_j]
            poids = fact[s] * fact[d - s - 1] / fact[d]
            phi[j] = (poids * (v[m[sans_j] | (1 << j)] - v[m[sans_j]])).sum()
        return phi, v[0]

    def math_fact(k):
        r = 1
        for i in range(2, k + 1):
            r *= i
        return r

    _pts = Xte[:config["shap_points"]]
    _t1 = time.perf_counter()
    phi_exact = np.array([shapley_exact(gb.predict, x, fond)[0] for x in _pts])
    duree_exact = time.perf_counter() - _t1
    _t1 = time.perf_counter()
    # Même définition dans la bibliothèque : explicateur exact (toutes les coalitions), variables
    # absentes remplacées par les individus du fond (masqueur « Independent »).
    _expl = shap.explainers.Exact(gb.predict, shap.maskers.Independent(fond, max_samples=len(fond)))
    phi_shap = _expl(_pts).values
    duree_shap = time.perf_counter() - _t1
    ecart_shap = float(np.abs(phi_shap - phi_exact).max())
    additivite = float(np.abs(phi_exact.sum(1) + gb.predict(fond).mean() - gb.predict(_pts)).max())
    print(f"Gradient boosting : R² test {r2_gb:.3f}")
    print(f"Shapley exact ({config['shap_points']} points × 1 024 coalitions × {len(fond)} individus de fond) : {duree_exact:.1f} s")
    print(f"shap.explainers.Exact : {duree_shap:.1f} s, écart maximal avec l'exact {ecart_shap:.1e} "
          f"(prédictions de l'ordre de {np.abs(gb.predict(_pts)).mean():.0f})")
    print(f"Efficacité : |Σφ + E[f] − f(x)| ≤ {additivite:.1e}")
    importance_globale = np.abs(phi_shap).mean(0)
    _ordre = np.argsort(-importance_globale)
    print("Importance moyenne |φ| : " + ", ".join(f"{noms_var[i]} {importance_globale[i]:.1f}" for i in _ordre))
    _p0 = _pts[0]
    print(f"Patient 0 : f(x) = {gb.predict(_p0[None])[0]:.1f}, E[f] = {gb.predict(fond).mean():.1f}, "
          + ", ".join(f"{noms_var[i]} {phi_exact[0, i]:+.1f}" for i in np.argsort(-np.abs(phi_exact[0]))[:4]))
    duree_tab = time.perf_counter() - _t0
    return (
        GradientBoostingRegressor,
        Xte,
        Xtr,
        additivite,
        duree_exact,
        ecart_shap,
        fond,
        gb,
        importance_globale,
        noms_var,
        phi_exact,
        r2_gb,
        shap,
        shapley_exact,
        yte,
        ytr,
    )


@app.cell
def _(GradientBoostingRegressor, Xte, Xtr, config, noms_var, np, shap, yte, ytr):
    # --- Un raccourci : une colonne qui « connaît » la cible pendant l'entraînement ---
    # Scénario : l'export des données a ajouté une variable mesurée APRÈS l'issue (ex. un
    # nombre de consultations de suivi), donc très corrélée à la cible. En production, au
    # moment de prédire, elle n'existe pas encore : on la remplace par une valeur sans lien.
    _g = np.random.default_rng(42)
    _norm = lambda y: (y - ytr.mean()) / ytr.std()  # noqa: E731
    Xtr_f = np.c_[Xtr, _norm(ytr) + 0.3 * _g.standard_normal(len(ytr))]
    Xte_f = np.c_[Xte, _norm(yte) + 0.3 * _g.standard_normal(len(yte))]            # même distribution
    Xprod = np.c_[Xte, _g.permutation(Xte_f[:, -1])]                                # fuite sans lien
    Xtr_f, Xte_f, Xprod = (z.astype(np.float32).astype(np.float64) for z in (Xtr_f, Xte_f, Xprod))
    gb_f = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, random_state=42).fit(Xtr_f, ytr)
    r2_fuite_test, r2_fuite_prod = gb_f.score(Xte_f, yte), gb_f.score(Xprod, yte)
    _fond = Xtr_f[np.random.default_rng(42).choice(len(Xtr_f), config["shap_fond"], replace=False)]
    _phi = shap.explainers.Exact(gb_f.predict, shap.maskers.Independent(_fond, max_samples=len(_fond)))(
        Xte_f[:config["shap_points"]]).values
    importance_fuite = np.abs(_phi).mean(0)
    part_fuite = importance_fuite[-1] / importance_fuite.sum()
    noms_fuite = noms_var + ["fuite"]
    print(f"Avec la fuite : R² test {r2_fuite_test:.3f} (même distribution) → R² en production {r2_fuite_prod:.3f}")
    print(f"SHAP : la colonne « fuite » porte {100 * part_fuite:.0f} % de l'importance totale ; "
          f"suivantes : " + ", ".join(f"{noms_fuite[i]} {importance_fuite[i]:.1f}"
                                     for i in np.argsort(-importance_fuite)[1:4]))
    return importance_fuite, noms_fuite, part_fuite, r2_fuite_prod, r2_fuite_test


@app.cell
def _(Xte, Xtr, config, nn, np, time, torch, ytr):
    # --- Gradients intégrés sur un petit MLP (CPU) ---
    _t0 = time.perf_counter()
    _mu, _sd = Xtr.mean(0), Xtr.std(0)
    _Xt = torch.tensor((Xtr - _mu) / _sd, dtype=torch.float32)
    _yt = torch.tensor((ytr - ytr.mean()) / ytr.std(), dtype=torch.float32)
    torch.manual_seed(42)
    mlp = nn.Sequential(nn.Linear(10, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
    _opt = torch.optim.Adam(mlp.parameters(), lr=3e-3, weight_decay=1e-3)
    for _e in range(config["mlp_epoques"]):
        _perte = (mlp(_Xt).squeeze(1) - _yt).pow(2).mean()
        _opt.zero_grad()
        _perte.backward()
        _opt.step()
    _Xe = torch.tensor((Xte[:config["shap_points"]] - _mu) / _sd, dtype=torch.float32)
    _base = torch.zeros(10)                     # point de référence : le patient moyen

    def f_mlp(x):
        return mlp(x).squeeze(-1)

    def gradients_integres(x, base, n):
        """∫₀¹ ∂f/∂x (base + α(x − base)) dα × (x − base), règle des trapèzes à n pas."""
        alphas = torch.linspace(0, 1, n + 1)
        chemin = (base + alphas[:, None] * (x - base)).requires_grad_(True)
        g, = torch.autograd.grad(f_mlp(chemin).sum(), chemin)
        moy = ((g[:-1] + g[1:]) / 2).mean(0)            # moyenne des trapèzes
        return moy * (x - base)

    _err_ig, _err_gxi = [], []
    for _x in _Xe:
        _ig = gradients_integres(_x, _base, config["ig_pas"])
        _x_g = _x.clone().requires_grad_(True)
        _g, = torch.autograd.grad(f_mlp(_x_g), _x_g)
        _delta = (f_mlp(_x) - f_mlp(_base)).item()
        _err_ig.append(abs(_ig.sum().item() - _delta) / abs(_delta))
        _err_gxi.append(abs((_g * (_x - _base)).sum().item() - _delta) / abs(_delta))
    erreur_ig, erreur_gxi = float(np.median(_err_ig)), float(np.median(_err_gxi))
    print(f"Complétude Σ attributions = f(x) − f(référence), écart relatif médian : "
          f"gradients intégrés ({config['ig_pas']} pas) {100 * erreur_ig:.2f} %, gradient × entrée {100 * erreur_gxi:.0f} %")
    print(f"(MLP : {time.perf_counter() - _t0:.1f} s)")
    return erreur_gxi, erreur_ig


@app.cell
def _(Xte, Xtr, config, fond, gb, noms_var, np, phi_exact, shapley_exact):
    # --- LIME écrit à la main : régression linéaire locale pondérée, plusieurs graines ---
    from sklearn.linear_model import Ridge

    _x0 = Xte[0]
    _sd = Xtr.std(0)
    _d = len(_x0)

    def lime(x, graine, n):
        g = np.random.default_rng(graine)
        Z = x + g.standard_normal((n, _d)) * _sd           # voisins tirés autour de x
        dist = np.linalg.norm((Z - x) / _sd, axis=1)
        poids = np.exp(-dist ** 2 / (0.75 * np.sqrt(_d)) ** 2)  # noyau de LIME (largeur 0,75·√d)
        return Ridge(alpha=1.0).fit((Z - x) / _sd, gb.predict(Z), sample_weight=poids).coef_

    def stabilite(n):
        coefs = np.array([lime(_x0, s, n) for s in range(config["lime_graines"])])
        top3 = [tuple(sorted(np.argsort(-np.abs(c))[:3])) for c in coefs]
        vals, comptes = np.unique(np.array(top3), axis=0, return_counts=True)
        return coefs, comptes.max() / len(top3), len(vals)

    stabilite_lime = {}
    for _n in (config["lime_petit"], config["lime_echantillons"]):
        _coefs, stabilite_lime[_n], _nb = stabilite(_n)
        print(f"LIME sur le patient 0, {config['lime_graines']} graines × {_n} voisins : le trio de tête "
              f"le plus fréquent revient {100 * stabilite_lime[_n]:.0f} % du temps ({_nb} trios différents)")
    _top_shap = tuple(sorted(np.argsort(-np.abs(phi_exact[0]))[:3]))
    for _s, _c in enumerate(_coefs[:3]):
        print(f"   graine {_s} : " + ", ".join(f"{noms_var[i]} {_c[i]:+.1f}" for i in np.argsort(-np.abs(_c))[:4]))
    print("   Shapley exact : " + ", ".join(f"{noms_var[i]} {phi_exact[0, i]:+.1f}" for i in np.argsort(-np.abs(phi_exact[0]))[:4]))
    _ = (fond, shapley_exact, _top_shap)
    return (stabilite_lime,)


@app.cell
def _(
    SMOKE,
    ce_sae,
    config,
    historique_sae,
    importance_fuite,
    importance_globale,
    induction_1c,
    induction_2c,
    mo,
    noms_fuite,
    noms_var,
    np,
    plt,
    purete_features,
    purete_neurones,
    sauvegarder_figure,
    scores_2c,
    sondes,
    sup_W,
    sup_b,
    sup_representees,
):
    # --- Figure récapitulative + résultats bruts pour les figures du site ---
    _fig, _ax = plt.subplots(2, 3, figsize=(15, 8.5))
    # A. superposition : colonnes de W pour S = 0 et S = 0,97
    _a = _ax[0, 0]
    for _coul, _i_s in [("#0072B2", 0), ("#D55E00", -1)]:
        _W = sup_W[_i_s]
        for _i in range(_W.shape[1]):
            _a.plot([0, _W[0, _i]], [0, _W[1, _i]], color=_coul, lw=2, alpha=0.8)
            _a.annotate(str(_i), _W[:, _i], color=_coul)
        _a.plot([], [], color=_coul, label=f"S = {config['sup_sparsites'][_i_s]}")
    _a.set(title="A. Colonnes W_i (5 caract. en 2D)", aspect="equal")
    _a.legend()
    # B. sondes
    _a = _ax[0, 1]
    for (_prop, _mod), _s in sondes.items():
        _a.plot(range(len(_s)), 100 * _s, marker="o", ls="-" if _mod == "entraîné" else "--", label=f"{_prop} ({_mod})")
    _a.set(title="B. Sondes linéaires", xlabel="couche (0 = plongements)", ylabel="précision équilibrée (%)", ylim=(40, 100))
    _a.legend(fontsize=7)
    # C. pureté
    _a = _ax[0, 2]
    _bins = np.linspace(0, 1, 21)
    _a.hist(purete_neurones, bins=_bins, alpha=0.6, label="neurones", density=True)
    _a.hist(purete_features, bins=_bins, alpha=0.6, label="caractéristiques SAE", density=True)
    _a.set(title=f"C. Pureté (top {config['sae_top']})", xlabel="part partageant un même caractère")
    _a.legend()
    # D. induction
    _a = _ax[1, 0]
    _a.plot(induction_2c[:, 0], induction_2c[:, 2], label="2 couches : perte 2de copie")
    _a.plot(induction_1c[:, 0], induction_1c[:, 2], label="1 couche : perte 2de copie")
    _a.plot(induction_2c[:, 0], induction_2c[:, 1], ls=":", label="2 couches : perte 1re copie")
    _a.set(title="D. Têtes d'induction", xlabel="pas", ylabel="entropie croisée")
    _a.legend(fontsize=8)
    # E. importances
    _a = _ax[1, 1]
    _o = np.argsort(importance_globale)
    _a.barh(np.array(noms_var)[_o], importance_globale[_o])
    _a.set(title="E. SHAP : importance moyenne |φ|")
    _a = _ax[1, 2]
    _o = np.argsort(importance_fuite)
    _a.barh(np.array(noms_fuite)[_o], importance_fuite[_o], color=["#D55E00" if n == "fuite" else "#0072B2" for n in np.array(noms_fuite)[_o]])
    _a.set(title="E. Avec une colonne « fuite »")
    _fig.tight_layout()
    _dossier = mo.notebook_dir() / "figures"
    _suffixe = "-smoke" if SMOKE else ""
    sauvegarder_figure(_fig, "resultats" + _suffixe, _dossier)
    np.savez(
        _dossier / f"resultats{_suffixe}.npz",
        sup_sparsites=np.array(config["sup_sparsites"]), sup_W=sup_W, sup_b=sup_b,
        sup_representees=np.array(sup_representees),
        sonde_proprietes=np.array([k[0] for k in sondes]), sonde_modeles=np.array([k[1] for k in sondes]),
        sonde_scores=np.array(list(sondes.values())),
        purete_neurones=purete_neurones, purete_features=purete_features,
        historique_sae=np.array(historique_sae),
        ce_sae=np.array([ce_sae["intact"], ce_sae["SAE"], ce_sae["zéro"]]),
        induction_2c=induction_2c, induction_1c=induction_1c, scores_2c=scores_2c,
        noms_var=np.array(noms_var), importance_globale=importance_globale,
        noms_fuite=np.array(noms_fuite), importance_fuite=importance_fuite,
    )
    print(f"Figure et résultats : {_dossier}/resultats{_suffixe}.png / .npz")
    _fig


@app.cell
def _(
    SMOKE,
    additivite,
    duree_induction,
    duree_sae,
    ecart_passe,
    ecart_shap,
    erreur_gxi,
    erreur_ig,
    fvu,
    induction_1c,
    induction_2c,
    induction_gpt,
    l0,
    mo,
    np,
    part_fuite,
    perte_recuperee,
    purete_features,
    purete_neurones,
    r2_fuite_prod,
    r2_fuite_test,
    sondes,
    sup_representees,
):
    # --- Bilan : seuils du README, vérifiés en mode complet ---
    _sv = sondes[("précédent = voyelle", "entraîné")]
    _sn = sondes[("première ligne de réplique", "entraîné")]
    _sn_alea = sondes[("première ligne de réplique", "aléatoire")]
    seuils = {
        "passe_detaillee = forward (écart < 1e-4)": ecart_passe < 1e-4,
        "A. S = 0 : au plus 2 caractéristiques représentées": sup_representees[0] <= 2,
        "A. S = 0,97 : les 5 caractéristiques représentées": sup_representees[-1] == 5,
        "B. voyelle précédente : ≥ 95 % à la meilleure couche, ≤ 75 % à la couche 0": _sv[1:].max() >= 0.95 and _sv[0] <= 0.75,
        "B. première ligne de réplique : entraîné ≥ 80 % et ≥ aléatoire + 5 points (meilleure couche)":
            _sn[1:].max() >= 0.80 and _sn[1:].max() >= _sn_alea[1:].max() + 0.05,
        "C. SAE : variance expliquée ≥ 80 %": fvu <= 0.20,
        "C. SAE : L0 ≤ 40 caractéristiques actives par token": l0 <= 40,
        "C. SAE : perte récupérée ≥ 80 %": perte_recuperee >= 0.80,
        "C. pureté médiane : caractéristiques ≥ neurones + 20 points": np.median(purete_features) >= np.median(purete_neurones) + 0.20,
        "D. mini-GPT : aucun score d'induction ≥ 0,1": max(v[2] for v in induction_gpt.values()) < 0.1,
        "D. 2 couches : perte 2de copie ≤ 1,5 (contre ln 64 = 4,16 au hasard) et score d'induction ≥ 0,3":
            induction_2c[-1, 2] <= 1.5 and induction_2c[-1, 3] >= 0.3,
        "D. 1 couche : perte 2de copie ≥ 3": induction_1c[-1, 2] >= 3.0,
        "E. Shapley exact = shap (écart < 1e-6)": ecart_shap < 1e-6,
        "E. efficacité Σφ + E[f] = f(x) (écart < 1e-6)": additivite < 1e-6,
        "E. fuite : ≥ 50 % de l'importance SHAP et R² chute d'au moins 0,3": part_fuite >= 0.5 and r2_fuite_test - r2_fuite_prod >= 0.3,
        "E. gradients intégrés : complétude à 1 % près, gradient × entrée au-delà de 5 %": erreur_ig <= 0.01 and erreur_gxi >= 0.05,
    }
    for _k, _v in seuils.items():
        print(("OK    " if _v else "ÉCHEC ") + _k)
    print(f"Durées : SAE {duree_sae:.0f} s, têtes d'induction {duree_induction:.0f} s")
    _echecs = [k for k, v in seuils.items() if not v]
    if _echecs and not SMOKE:
        raise SystemExit("Seuils non atteints : " + " ; ".join(_echecs))
    mo.md(f"**Bilan** : {sum(seuils.values())}/{len(seuils)} seuils atteints" + (" (non vérifiés en smoke)" if SMOKE else "") + ".")


if __name__ == "__main__":
    app.run()
