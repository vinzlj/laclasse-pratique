# Exercice 24a : un mini-CLIP entraîné de zéro sur des images synthétiques légendées.
#
#   make ex NUM=24a              → éditeur marimo (bouton « Entraîner »)
#   make ex NUM=24a MODE=run     → run complet (≈ 4 min sur M1 Pro, MPS)
#   make ex NUM=24a MODE=smoke   → test rapide (< 30 s)
#
# Partie A (NumPy) : découper une image en patchs comme un ViT, vérifier que la projection
# des patchs est une convolution, compter les tokens, calculer la perte de CLIP à la main.
# Partie B (PyTorch) : entraîner un encodeur d'image (ViT ou CNN) et un encodeur de texte
# avec la perte contrastive de CLIP, puis classer en zero-shot avec des phrases.

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

    return (
        F,
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
    # Exercice 24a : un mini-CLIP de zéro

    **Objectif.** Comprendre les deux idées du chapitre en les faisant tourner à petite
    échelle : (1) un **Vision Transformer** lit une image comme une suite de patchs ;
    (2) **CLIP** aligne un encodeur d'image et un encodeur de texte dans un même espace,
    ce qui permet ensuite de classer des images **sans exemple étiqueté** (zero-shot), en
    les comparant à des phrases.

    Les données sont synthétiques : des images 32×32 d'une forme (carré, cercle, triangle,
    croix) d'une couleur, d'une taille et à une position données, avec leur légende en
    français (« une petite croix verte en bas à droite »). Quatre associations forme +
    couleur ne sont **jamais** montrées à l'entraînement : le modèle saura-t-il les décrire ?

    **Perte optimisée** : la perte contrastive symétrique de CLIP (InfoNCE, expliquée au
    chapitre 23). Pour un lot de $N$ paires, avec $z_i$ et $t_j$ les plongements normalisés
    de l'image $i$ et du texte $j$ et $s$ une échelle apprise (inverse d'une température) :

    $$\mathcal{L} = \frac{1}{2N}\sum_{i=1}^{N}\Big[-\log\frac{e^{s\,z_i\cdot t_i}}{\sum_j e^{s\,z_i\cdot t_j}}
    \;-\;\log\frac{e^{s\,z_i\cdot t_i}}{\sum_j e^{s\,z_j\cdot t_i}}\Big]$$

    soit une entropie croisée sur les lignes (image → texte) et une sur les colonnes
    (texte → image) de la matrice des similarités.
    """)


@app.cell
def _(get_device, is_smoke, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    device = get_device()
    style_commun()
    config = dict(
        n_train=2000 if SMOKE else 20000,   # images d'entraînement (attributs tirés au hasard)
        n_test=400 if SMOKE else 2000,      # images de test (toutes les combinaisons, y compris exclues)
        pas=40 if SMOKE else 1500,          # pas d'optimisation par encodeur d'image
        lot=256,
        lr=1e-3,
        weight_decay=0.05,
        p_partiel=0.5,                      # probabilité d'une légende partielle (« un cercle rouge »)
        d_commun=64,                        # dimension de l'espace commun image-texte
        vit=dict(patch=4, d=64, couches=3, tetes=4),
        cnn=dict(canaux=(32, 64, 96)),
        texte=dict(d=64, couches=2, tetes=4),
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SEED, SMOKE, config, device


@app.cell
def _(mo):
    mo.md(r"""
    ## Partie A : le ViT et la perte de CLIP en NumPy

    **A1. Une image devient une suite de patchs.** Une image 3×32×32 coupée en carrés de
    4×4 pixels donne $8 \times 8 = 64$ patchs ; chacun, aplati, est un vecteur de
    $3 \times 4 \times 4 = 48$ nombres. Une couche linéaire (la même pour tous les patchs)
    les projette en dimension $d$ : ce sont les « tokens » du ViT. On vérifie que cette
    projection est exactement une **convolution** de noyau 4×4 et de pas 4 (c'est ainsi
    que les bibliothèques la codent).
    """)


@app.cell
def _(F, np, torch):
    # --- A1. Découper en patchs (NumPy) et vérifier contre une convolution PyTorch ---
    def en_patchs(image, p):
        """(C, H, W) → (H/p · W/p, C·p·p) : patchs lus ligne par ligne, de gauche à droite."""
        c, h, w = image.shape
        x = image.reshape(c, h // p, p, w // p, p)        # (C, lignes, p, colonnes, p)
        x = x.transpose(1, 3, 0, 2, 4)                     # (lignes, colonnes, C, p, p)
        return x.reshape((h // p) * (w // p), c * p * p)

    _rng = np.random.default_rng(0)
    _img = _rng.random((3, 32, 32))
    _patchs = en_patchs(_img, 4)
    _d = 64
    _E = _rng.normal(0, 0.1, (48, _d))                     # projection des patchs (apprise dans un ViT)
    _b = _rng.normal(0, 0.1, _d)
    _tokens = _patchs @ _E + _b                            # (64, d)
    # Même calcul avec une convolution : noyau 4×4, pas 4, d filtres
    _poids = torch.from_numpy(_E.T.reshape(_d, 3, 4, 4))
    _conv = F.conv2d(torch.from_numpy(_img)[None], _poids, torch.from_numpy(_b), stride=4)
    _tokens_conv = _conv[0].flatten(1).T.numpy()           # (64, d), même ordre de lecture
    ecart_patchs = float(np.abs(_tokens - _tokens_conv).max())
    print(f"{_patchs.shape[0]} patchs de {_patchs.shape[1]} nombres → tokens {_tokens.shape} ;"
          f" écart maximal avec la convolution : {ecart_patchs:.1e}")

    # Combien de tokens pour de vraies images ? (patchs + 1 token [CLS])
    tailles_tokens = {}
    for _nom, _taille, _p in [("ce mini-ViT", 32, 4), ("ViT-B/32 (exercice 24b)", 224, 32),
                              ("ViT-B/16", 224, 16), ("ViT-B/16 à 384 px", 384, 16)]:
        _n = (_taille // _p) ** 2 + 1
        tailles_tokens[_nom] = _n
        _cases = f"{_n * _n:,}".replace(",", " ")
        print(f"{_nom:26s}: image {_taille}×{_taille}, patchs {_p}×{_p} → {_n:4d} tokens,"
              f" matrice d'attention {_n}² = {_cases} cases par tête")
    return ecart_patchs, en_patchs, tailles_tokens


@app.cell
def _(mo):
    mo.md(r"""
    **A2. La perte de CLIP à la main.** Quatre images et leurs quatre légendes, déjà
    encodées (vecteurs de dimension 3 pour pouvoir suivre le calcul). La matrice
    $S = s \cdot Z T^\top$ compare chaque image à chaque légende ; la bonne paire est sur
    la diagonale. On calcule l'entropie croisée ligne par ligne et colonne par colonne,
    puis on compare à `F.cross_entropy`.
    """)


@app.cell
def _(F, np, torch):
    # --- A2. Perte contrastive symétrique de CLIP, en NumPy puis avec PyTorch ---
    def normaliser(v):
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    Z_jouet = normaliser(np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.2], [0.1, 0.0, 1.0], [0.7, 0.7, 0.0]]))
    T_jouet = normaliser(np.array([[0.9, 0.2, 0.1], [0.1, 0.8, 0.1], [0.0, 0.3, 0.9], [0.5, 0.6, 0.2]]))
    echelle_jouet = 10.0                                   # CLIP l'apprend ; elle plafonne à 100
    S_jouet = echelle_jouet * Z_jouet @ T_jouet.T          # (4 images, 4 légendes)

    def entropie_croisee_diag(S):
        # -log softmax(S)[i, i], moyenne sur les lignes (stabilisé en retirant le max)
        S = S - S.max(axis=1, keepdims=True)
        log_p = S - np.log(np.exp(S).sum(axis=1, keepdims=True))
        return -np.mean(np.diag(log_p))

    perte_img_txt = entropie_croisee_diag(S_jouet)         # image → texte (lignes)
    perte_txt_img = entropie_croisee_diag(S_jouet.T)       # texte → image (colonnes)
    perte_clip_numpy = (perte_img_txt + perte_txt_img) / 2
    _S = torch.from_numpy(S_jouet)
    _cible = torch.arange(4)
    perte_clip_torch = ((F.cross_entropy(_S, _cible) + F.cross_entropy(_S.T, _cible)) / 2).item()
    print("Similarités cosinus (lignes = images, colonnes = légendes) :")
    print(np.round(Z_jouet @ T_jouet.T, 2))
    print(f"image → texte {perte_img_txt:.4f}, texte → image {perte_txt_img:.4f},"
          f" moyenne {perte_clip_numpy:.4f} ; PyTorch {perte_clip_torch:.4f}")
    print(f"Hasard (4 paires) : log 4 = {np.log(4):.4f}")
    return perte_clip_numpy, perte_clip_torch


@app.cell
def _(mo):
    mo.md(r"""
    ## Partie B : un mini-CLIP de zéro

    **Les données.** Chaque image a quatre attributs : forme (4), couleur (4), taille (2),
    position (4), soit 128 combinaisons et autant de légendes complètes. Le fond, la teinte
    exacte, la taille et la position varient un peu d'une image à l'autre. La moitié des
    légendes d'entraînement sont **partielles** (« un objet rouge », « un grand cercle »),
    comme les légendes du web qui ne décrivent qu'une partie de l'image : c'est ce qui
    permettra de demander ensuite « un cercle » ou « un objet rouge » en zero-shot.

    Quatre couples forme + couleur sont **exclus de l'entraînement** (triangle bleu,
    cercle jaune, croix rouge, carré vert) mais présents en test.
    """)


@app.cell
def _(np, torch):
    # --- Données synthétiques : images, légendes et tokenisation par mots ---
    # (copie conforme dans site/scripts/figures/_ch24_commun.py pour les figures du site)
    FORMES = ["carré", "cercle", "triangle", "croix"]
    FEMININ = {"croix"}
    COULEURS = ["rouge", "vert", "bleu", "jaune"]
    COULEURS_FEM = {"rouge": "rouge", "vert": "verte", "bleu": "bleue", "jaune": "jaune"}
    RVB = np.array([[0.9, 0.15, 0.15], [0.15, 0.75, 0.2], [0.2, 0.35, 0.95], [0.95, 0.85, 0.1]],
                   dtype=np.float32)
    TAILLES = ["petit", "grand"]
    POSITIONS = ["en haut à gauche", "en haut à droite", "en bas à gauche", "en bas à droite"]
    CENTRES = np.array([(8, 8), (8, 24), (24, 8), (24, 24)], dtype=np.float32)   # (ligne, colonne)
    EXCLUS = {(2, 2), (1, 3), (3, 0), (0, 1)}   # (forme, couleur) jamais vus : triangle bleu…
    ATTRIBUTS = ["forme", "couleur", "taille", "position"]

    def legende(f, c, t, p, garder=(True, True, True, True)):
        """Légende française accordée ; `garder` choisit les attributs mentionnés."""
        gf, gc, gt, gp = garder
        fem = gf and FORMES[f] in FEMININ
        mots = ["une" if fem else "un"]
        if gt:
            mots.append(TAILLES[t] + ("e" if fem else ""))
        mots.append(FORMES[f] if gf else "objet")
        if gc:
            mots.append(COULEURS_FEM[COULEURS[c]] if fem else COULEURS[c])
        if gp:
            mots.append(POSITIONS[p])
        return " ".join(mots)

    def tirer_attributs(n, rng, exclure=True):
        """n quadruplets (forme, couleur, taille, position) ; sans les couples exclus si demandé."""
        sortie = np.zeros((0, 4), dtype=np.int64)
        while len(sortie) < n:
            a = np.stack([rng.integers(0, 4, n), rng.integers(0, 4, n),
                          rng.integers(0, 2, n), rng.integers(0, 4, n)], axis=1)
            if exclure:
                a = a[[(f, c) not in EXCLUS for f, c in a[:, :2]]]
            sortie = np.concatenate([sortie, a])
        return sortie[:n]

    def dessiner(attrs, rng, taille=32):
        """Images (n, 3, taille, taille) en float32, dessinées sans boucle sur les images."""
        n = len(attrs)
        f, c, t, p = attrs.T
        yy, xx = np.mgrid[0:taille, 0:taille].astype(np.float32)
        centre = CENTRES[p] + rng.uniform(-2, 2, (n, 2)).astype(np.float32)
        r = np.where(t == 1, 7.0, 4.0).astype(np.float32) * rng.uniform(0.9, 1.1, n).astype(np.float32)
        dy = (yy[None] - centre[:, 0, None, None]) / r[:, None, None]
        dx = (xx[None] - centre[:, 1, None, None]) / r[:, None, None]
        masques = np.stack([
            (np.abs(dy) <= 0.85) & (np.abs(dx) <= 0.85),                          # carré
            dy**2 + dx**2 <= 1,                                                    # cercle
            (dy >= -1) & (dy <= 0.8) & (np.abs(dx) <= (dy + 1) * 0.6),             # triangle
            ((np.abs(dx) <= 0.3) & (np.abs(dy) <= 1)) | ((np.abs(dy) <= 0.3) & (np.abs(dx) <= 1)),  # croix
        ])[f, np.arange(n)]
        fond = rng.uniform(0.25, 0.6, (n, 1, 1, 1)).astype(np.float32)
        img = fond + rng.normal(0, 0.05, (n, taille, taille, 3)).astype(np.float32)
        teinte = RVB[c] + rng.uniform(-0.08, 0.08, (n, 3)).astype(np.float32)
        img = np.where(masques[..., None], teinte[:, None, None, :], img)
        return np.clip(img, 0, 1).transpose(0, 3, 1, 2).astype(np.float32)

    def legendes_aleatoires(attrs, rng, p_partiel):
        """Une légende par image : complète, ou partielle avec probabilité p_partiel."""
        sortie = []
        for a in attrs:
            if rng.random() < p_partiel:
                g = rng.random(4) < 0.5
                if not g.any():
                    g[rng.integers(4)] = True
                sortie.append(legende(*a, garder=tuple(g)))
            else:
                sortie.append(legende(*a))
        return sortie

    # Vocabulaire : un token par mot (pas besoin de BPE pour 25 mots, voir chapitre 19)
    VOCAB = ["<pad>", "<cls>"]
    for _f in range(4):
        for _c in range(4):
            for _t in range(2):
                for _g in [(True,) * 4, (False,) * 4]:
                    for _m in legende(_f, _c, _t, 0, _g).split() + POSITIONS[1].split() + POSITIONS[2].split():
                        if _m not in VOCAB:
                            VOCAB.append(_m)
    INDEX = {m: i for i, m in enumerate(VOCAB)}
    LONGUEUR_MAX = 10   # <cls> + 8 mots au plus

    def tokeniser(phrases):
        ids = torch.zeros(len(phrases), LONGUEUR_MAX, dtype=torch.long)   # 0 = <pad>
        for i, s in enumerate(phrases):
            seq = [INDEX["<cls>"]] + [INDEX[m] for m in s.split()]
            ids[i, :len(seq)] = torch.tensor(seq)
        return ids

    COMBINAISONS = [(f, c, t, p) for f in range(4) for c in range(4) for t in range(2) for p in range(4)]
    LEGENDES_COMPLETES = [legende(*a) for a in COMBINAISONS]
    print(f"Vocabulaire ({len(VOCAB)} mots) : {' '.join(VOCAB[2:])}")
    print(f"{len(COMBINAISONS)} légendes complètes, dont "
          f"{sum((f, c) in EXCLUS for f, c, _, _ in COMBINAISONS)} jamais vues à l'entraînement")
    print("Exemples :", LEGENDES_COMPLETES[0], "|", legende(3, 1, 0, 3), "|",
          legende(2, 2, 1, 0, (False, True, False, True)))
    return (
        ATTRIBUTS,
        COMBINAISONS,
        COULEURS,
        EXCLUS,
        FORMES,
        LEGENDES_COMPLETES,
        LONGUEUR_MAX,
        POSITIONS,
        TAILLES,
        VOCAB,
        dessiner,
        legende,
        legendes_aleatoires,
        tirer_attributs,
        tokeniser,
    )


@app.cell
def _(COMBINAISONS, EXCLUS, SEED, config, dessiner, np, tirer_attributs, torch):
    # --- Jeux d'entraînement (sans les couples exclus) et de test (toutes les combinaisons) ---
    _rng = np.random.default_rng(SEED)
    attrs_train = tirer_attributs(config["n_train"], _rng, exclure=True)
    X_train = torch.from_numpy(dessiner(attrs_train, _rng))
    _rng_test = np.random.default_rng(SEED + 1)
    attrs_test = tirer_attributs(config["n_test"], _rng_test, exclure=False)
    X_test = torch.from_numpy(dessiner(attrs_test, _rng_test))
    vu_test = np.array([(f, c) not in EXCLUS for f, c in attrs_test[:, :2]])
    indice_test = np.array([COMBINAISONS.index(tuple(a)) for a in attrs_test])
    print(f"Entraînement : {len(X_train)} images ; test : {len(X_test)} images, dont "
          f"{(~vu_test).sum()} de couples forme + couleur jamais vus")
    return X_test, X_train, attrs_test, attrs_train, indice_test, vu_test


@app.cell
def _(SMOKE, X_train, attrs_train, legende, mo, plt, sauvegarder_figure):
    # --- Quelques images d'entraînement et leur légende complète ---
    _fig, _axes = plt.subplots(2, 4, figsize=(11, 6.6))
    for _k, _ax in enumerate(_axes.flat):
        _ax.imshow(X_train[_k].permute(1, 2, 0).numpy())
        _ax.set_title(legende(*attrs_train[_k]).replace(" en ", "\nen "), fontsize=9)
        _ax.axis("off")
    _fig.tight_layout()
    sauvegarder_figure(_fig, "exemples-smoke" if SMOKE else "exemples", mo.notebook_dir() / "figures")
    _fig


@app.cell
def _(mo):
    mo.md(r"""
    **Les modèles.** Deux encodeurs d'image, à comparer à budget égal :

    - un **ViT** : patchs 4×4 → projection linéaire → token `[CLS]` + plongements de
      position appris → 3 blocs Transformer (pré-normalisation, 4 têtes) → sortie du `[CLS]` ;
    - un **petit CNN** : trois convolutions 3×3 avec pooling, puis moyenne globale.

    L'encodeur de **texte** est un petit Transformer (2 blocs) sur les mots, avec masque de
    remplissage ; sa sortie est celle du token `<cls>`. Les deux sorties sont projetées
    dans un espace commun de dimension 64 puis normalisées ; l'échelle $s$ est apprise
    (départ $1/0{,}07 \approx 14{,}3$, plafond 100, comme CLIP).
    """)


@app.cell
def _(F, LONGUEUR_MAX, VOCAB, config, math, nn, torch):
    # --- Encodeurs : ViT, CNN, texte, et le modèle CLIP qui les relie ---
    class Bloc(nn.Module):
        """Bloc Transformer pré-normalisé : attention multi-têtes puis MLP, avec résidus."""

        def __init__(self, d, tetes):
            super().__init__()
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv, self.proj, self.tetes = nn.Linear(d, 3 * d), nn.Linear(d, d), tetes
            self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

        def forward(self, x, masque=None):
            B, T, D = x.shape
            q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.tetes, D // self.tetes).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=masque)   # pas de dropout (MPS)
            x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
            return x + self.mlp(self.ln2(x))

    class ViT(nn.Module):
        def __init__(self, patch, d, couches, tetes, taille=32, d_sortie=64):
            super().__init__()
            self.patch = patch
            n_patchs = (taille // patch) ** 2
            self.projection = nn.Linear(3 * patch * patch, d)          # la matrice E de la partie A
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            self.position = nn.Parameter(torch.randn(1, n_patchs + 1, d) * 0.02)
            self.blocs = nn.ModuleList([Bloc(d, tetes) for _ in range(couches)])
            self.ln, self.sortie = nn.LayerNorm(d), nn.Linear(d, d_sortie)

        def forward(self, x):
            B, C, H, W = x.shape
            P = self.patch
            # même découpage que en_patchs() de la partie A, pour tout un lot
            x = x.reshape(B, C, H // P, P, W // P, P).permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(B, (H // P) * (W // P), C * P * P)
            x = torch.cat([self.cls.expand(B, -1, -1), self.projection(x)], dim=1) + self.position
            for bloc in self.blocs:
                x = bloc(x)
            return self.sortie(self.ln(x[:, 0]))

    class CNN(nn.Module):
        def __init__(self, canaux, d_sortie=64):
            super().__init__()
            couches, c_in = [], 3
            for c in canaux:
                couches += [nn.Conv2d(c_in, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2)]
                c_in = c
            self.corps, self.sortie = nn.Sequential(*couches), nn.Linear(c_in, d_sortie)

        def forward(self, x):
            return self.sortie(self.corps(x).mean(dim=(2, 3)))

    class EncodeurTexte(nn.Module):
        def __init__(self, d, couches, tetes, d_sortie=64):
            super().__init__()
            self.mots = nn.Embedding(len(VOCAB), d)
            self.position = nn.Parameter(torch.randn(1, LONGUEUR_MAX, d) * 0.02)
            self.blocs = nn.ModuleList([Bloc(d, tetes) for _ in range(couches)])
            self.ln, self.sortie = nn.LayerNorm(d), nn.Linear(d, d_sortie)

        def forward(self, ids):
            x = self.mots(ids) + self.position
            masque = (ids != 0)[:, None, None, :]     # on n'écoute pas les <pad>
            for bloc in self.blocs:
                x = bloc(x, masque)
            return self.sortie(self.ln(x[:, 0]))

    class MiniCLIP(nn.Module):
        def __init__(self, encodeur_image):
            super().__init__()
            self.image = encodeur_image
            self.texte = EncodeurTexte(**config["texte"], d_sortie=config["d_commun"])
            self.log_echelle = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

        def encoder_image(self, x):
            return F.normalize(self.image(x), dim=-1)

        def encoder_texte(self, ids):
            return F.normalize(self.texte(ids), dim=-1)

        def echelle(self):
            return self.log_echelle.exp().clamp(max=100)

    def perte_clip(z_img, z_txt, echelle):
        """La perte de la partie A2, sur un lot : entropie croisée lignes + colonnes."""
        logits = echelle * z_img @ z_txt.T
        cible = torch.arange(len(z_img), device=z_img.device)
        return (F.cross_entropy(logits, cible) + F.cross_entropy(logits.T, cible)) / 2

    def creer_encodeur(nom):
        if nom == "ViT":
            return ViT(**config["vit"], d_sortie=config["d_commun"])
        return CNN(**config["cnn"], d_sortie=config["d_commun"])

    return MiniCLIP, creer_encodeur, perte_clip


@app.cell
def _(mo):
    # Le bouton empêche l'entraînement de se relancer à chaque modification d'une cellule amont.
    bouton = mo.ui.run_button(label="Entraîner")
    bouton
    return (bouton,)


@app.cell
def _(
    MiniCLIP,
    SEED,
    X_train,
    attrs_train,
    bouton,
    config,
    creer_encodeur,
    device,
    legendes_aleatoires,
    mo,
    np,
    perte_clip,
    time,
    tokeniser,
    torch,
):
    # En script (make ex MODE=run|smoke), il n'y a personne pour cliquer : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Entraîner » pour lancer les deux entraînements.*"))

    def entrainer(nom):
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        modele = MiniCLIP(creer_encodeur(nom)).to(device)
        optim = torch.optim.AdamW(modele.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        sched = torch.optim.lr_scheduler.OneCycleLR(optim, config["lr"], total_steps=config["pas"], pct_start=0.1)
        courbe = []
        t0 = time.perf_counter()
        for pas in range(config["pas"]):
            idx = rng.integers(0, len(X_train), config["lot"])
            x = X_train[idx].to(device)
            ids = tokeniser(legendes_aleatoires(attrs_train[idx], rng, config["p_partiel"])).to(device)
            perte = perte_clip(modele.encoder_image(x), modele.encoder_texte(ids), modele.echelle())
            optim.zero_grad()
            perte.backward()
            optim.step()
            sched.step()
            courbe.append(perte.item())
            if pas % 250 == 0 or pas == config["pas"] - 1:
                print(f"{nom} pas {pas:4d} : perte {perte.item():.3f}, échelle {modele.echelle().item():.1f}"
                      f" ({time.perf_counter() - t0:.0f} s)")
        duree = time.perf_counter() - t0
        n_param = sum(p.numel() for p in modele.image.parameters())
        print(f"{nom} : {n_param:,} paramètres d'image".replace(",", " ") + f", {duree:.0f} s")
        return modele.eval(), np.array(courbe), duree, n_param

    modeles, courbes, durees, n_params = {}, {}, {}, {}
    for _nom in ("ViT", "CNN"):
        modeles[_nom], courbes[_nom], durees[_nom], n_params[_nom] = entrainer(_nom)
    n_params["texte"] = sum(p.numel() for p in modeles["ViT"].texte.parameters())
    return courbes, durees, modeles, n_params


@app.cell
def _(mo):
    mo.md(r"""
    ## Évaluation : zero-shot avec des phrases

    1. **Légende complète parmi 128** : pour chaque image de test, on encode les 128
       légendes complètes possibles et on prend la plus proche. Hasard : 1/128 ≈ 0,8 %.
       On sépare les couples forme + couleur **vus** et **jamais vus** à l'entraînement.
    2. **Un attribut à la fois**, comme le « a photo of a {classe} » de CLIP : « un carré »,
       « un cercle »… pour la forme, « un objet rouge »… pour la couleur. Puis la variante
       **ensemble de gabarits** : le plongement d'une classe est la moyenne de ceux de
       toutes les légendes complètes qui la contiennent.
    """)


@app.cell
def _(
    ATTRIBUTS,
    COMBINAISONS,
    LEGENDES_COMPLETES,
    X_test,
    attrs_test,
    device,
    indice_test,
    legende,
    modeles,
    np,
    tokeniser,
    torch,
    vu_test,
):
    # --- Zero-shot : légendes complètes, puis un attribut à la fois ---
    _NB = {"forme": 4, "couleur": 4, "taille": 2, "position": 4}

    @torch.no_grad()
    def plongements(modele):
        z_img = torch.cat([modele.encoder_image(X_test[i:i + 500].to(device))
                           for i in range(0, len(X_test), 500)]).cpu()
        z_txt = modele.encoder_texte(tokeniser(LEGENDES_COMPLETES).to(device)).cpu()
        return z_img, z_txt

    @torch.no_grad()
    def prompts_attribut(modele, j):
        """Un prompt par valeur de l'attribut j (« un cercle »), et l'ensemble de gabarits."""
        simples, ensembles = [], []
        for v in range(_NB[ATTRIBUTS[j]]):
            a = [0, 0, 0, 0]
            a[j] = v
            garder = tuple(k == j for k in range(4))
            simples.append(legende(*a, garder=garder))
            idx = [i for i, comb in enumerate(COMBINAISONS) if comb[j] == v]
            z = modele.encoder_texte(tokeniser([LEGENDES_COMPLETES[i] for i in idx]).to(device)).mean(0)
            ensembles.append(torch.nn.functional.normalize(z, dim=0).cpu())
        z_simples = modele.encoder_texte(tokeniser(simples).to(device)).cpu()
        return simples, z_simples, torch.stack(ensembles)

    scores, similarites = {}, {}
    for _nom, _m in modeles.items():
        _zi, _zt = plongements(_m)
        _pred = (_zi @ _zt.T).argmax(1).numpy()
        _bon = _pred == indice_test
        _s = {"complete_vus": float(_bon[vu_test].mean()), "complete_jamais_vus": float(_bon[~vu_test].mean())}
        # Pour les images jamais vues : quel attribut le modèle se trompe-t-il à donner ?
        _pa = np.array([COMBINAISONS[i] for i in _pred])
        _s["jamais_vus_forme_juste"] = float((_pa[~vu_test, 0] == attrs_test[~vu_test, 0]).mean())
        _s["jamais_vus_couleur_juste"] = float((_pa[~vu_test, 1] == attrs_test[~vu_test, 1]).mean())
        for _j, _att in enumerate(ATTRIBUTS):
            _simples, _z1, _zens = prompts_attribut(_m, _j)
            _s[f"{_att}_prompt"] = float(((_zi @ _z1.T).argmax(1).numpy() == attrs_test[:, _j]).mean())
            _s[f"{_att}_ensemble"] = float(((_zi @ _zens.T).argmax(1).numpy() == attrs_test[:, _j]).mean())
        scores[_nom] = _s
        similarites[_nom] = (_zi, _zt)
        print(f"{_nom} : légende complète parmi 128 : {_s['complete_vus']:.1%} (couples vus),"
              f" {_s['complete_jamais_vus']:.1%} (jamais vus) ; hasard 0,8 %")
        print("      zero-shot par attribut (prompt seul / ensemble) : " + ", ".join(
            f"{_a} {_s[_a + '_prompt']:.0%} / {_s[_a + '_ensemble']:.0%}" for _a in ATTRIBUTS))
        print(f"      jamais vus : forme juste {_s['jamais_vus_forme_juste']:.0%},"
              f" couleur juste {_s['jamais_vus_couleur_juste']:.0%}")
    return scores, similarites


@app.cell
def _(LEGENDES_COMPLETES, X_test, indice_test, np, similarites, vu_test):
    # --- Matrice de similarité sur 8 images de test (une par combinaison vue, ViT) ---
    _zi, _zt = similarites["ViT"]
    _choix, _deja = [], set()
    for _i in np.flatnonzero(vu_test):
        if indice_test[_i] not in _deja and len(_choix) < 8:
            _choix.append(_i)
            _deja.add(indice_test[_i])
    choix_matrice = np.array(_choix)
    images_matrice = X_test[choix_matrice].permute(0, 2, 3, 1).numpy()
    legendes_matrice = [LEGENDES_COMPLETES[indice_test[i]] for i in choix_matrice]
    _zl = _zt[indice_test[choix_matrice]]
    matrice_sim = (_zi[choix_matrice] @ _zl.T).numpy()
    print(np.round(matrice_sim, 2))
    print("diagonale maximale sur chaque ligne :", bool((matrice_sim.argmax(1) == np.arange(8)).all()))
    return images_matrice, legendes_matrice, matrice_sim


@app.cell
def _(F, config, modeles):
    # --- Plongements de position appris par le ViT : similarité cosinus entre les 64 patchs ---
    _pos = modeles["ViT"].image.position[0, 1:].detach().cpu()        # sans le [CLS]
    _pos = F.normalize(_pos, dim=-1)
    sim_positions = (_pos @ _pos.T).numpy()                           # (64, 64)
    _cote = 32 // config["vit"]["patch"]
    _voisin = [sim_positions[i, i + 1] for i in range(len(_pos) - 1) if (i + 1) % _cote]
    _loin = [sim_positions[0, len(_pos) - 1], sim_positions[_cote - 1, len(_pos) - _cote]]
    print(f"Cosinus moyen entre patchs voisins sur une ligne : {sum(_voisin) / len(_voisin):.2f} ;"
          f" entre coins opposés : {_loin[0]:.2f} et {_loin[1]:.2f}")
    return (sim_positions,)


@app.cell
def _(
    ATTRIBUTS,
    SMOKE,
    config,
    courbes,
    images_matrice,
    legendes_matrice,
    matrice_sim,
    mo,
    np,
    plt,
    sauvegarder_figure,
    scores,
    sim_positions,
):
    # --- Figures de l'exercice (le site en refait des versions SVG à partir de resultats.npz) ---
    _dossier = mo.notebook_dir() / "figures"
    _fig, _ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for _nom, _c in courbes.items():
        _lisse = np.convolve(_c, np.ones(25) / 25, mode="valid") if len(_c) > 50 else _c
        _ax[0].plot(_lisse, label=_nom)
    _ax[0].axhline(np.log(config["lot"]), ls=":", c="gray", label="hasard (log 256)")
    _ax[0].set(xlabel="pas", ylabel="perte contrastive (moyenne glissante)", title="Entraînement")
    _ax[0].legend()
    _cles = ["complete_vus", "complete_jamais_vus"] + [f"{a}_prompt" for a in ATTRIBUTS]
    _noms = ["légende\n(vus)", "légende\n(jamais vus)"] + ATTRIBUTS
    _x = np.arange(len(_cles))
    for _k, (_nom, _s) in enumerate(scores.items()):
        _ax[1].bar(_x + (_k - 0.5) * 0.38, [100 * _s[c] for c in _cles], 0.38, label=_nom)
    _ax[1].set_xticks(_x, _noms)
    _ax[1].set(ylabel="exactitude zero-shot (%)", ylim=(0, 105), title="Zero-shot en test")
    _ax[1].legend()
    _fig.tight_layout()
    sauvegarder_figure(_fig, "bilan-smoke" if SMOKE else "bilan", _dossier)

    _fig2, _a2 = plt.subplots(figsize=(8, 7))
    _a2.imshow(matrice_sim, cmap="viridis")
    _a2.set_yticks(range(8), [""] * 8)
    _a2.set_xticks(range(8), [s.replace(" en ", "\nen ") for s in legendes_matrice], rotation=60,
                   ha="right", fontsize=7)
    for _i in range(8):
        _a2.imshow(images_matrice[_i], extent=(-1.6, -0.6, _i + 0.5, _i - 0.5))
        for _j in range(8):
            _a2.text(_j, _i, f"{matrice_sim[_i, _j]:.2f}", ha="center", va="center", fontsize=7,
                     color="white" if matrice_sim[_i, _j] < matrice_sim.max() * 0.6 else "black")
    _a2.set_xlim(-1.7, 7.5)
    _a2.set_ylim(7.5, -0.5)
    _a2.grid(False)
    _a2.set_title("Similarité cosinus image (ligne) / légende (colonne), ViT")
    sauvegarder_figure(_fig2, "similarites-smoke" if SMOKE else "similarites", _dossier)

    _cote = 32 // config["vit"]["patch"]
    _fig3, _a3 = plt.subplots(_cote, _cote, figsize=(7, 7))
    for _i, _a in enumerate(_a3.flat):
        _a.imshow(sim_positions[_i].reshape(_cote, _cote), cmap="viridis", vmin=-1, vmax=1)
        _a.axis("off")
    _fig3.suptitle("Plongements de position du ViT : similarité de chaque patch avec les autres")
    sauvegarder_figure(_fig3, "positions-smoke" if SMOKE else "positions", _dossier)
    plt.close("all")
    print(f"Figures : {_dossier}")
    return


@app.cell
def _(
    SEED,
    SMOKE,
    config,
    courbes,
    durees,
    ecart_patchs,
    images_matrice,
    json,
    legendes_matrice,
    matrice_sim,
    mo,
    n_params,
    np,
    perte_clip_numpy,
    perte_clip_torch,
    scores,
    sim_positions,
    tailles_tokens,
):
    # --- Résultats bruts pour les figures du site (site/scripts/figures/ch24_*.py) ---
    bilan = {
        "graine": SEED,
        "config": config,
        "ecart_patchs": ecart_patchs,
        "tailles_tokens": tailles_tokens,
        "perte_jouet": {"numpy": perte_clip_numpy, "torch": perte_clip_torch},
        "scores": scores,
        "durees": durees,
        "n_params": n_params,
        "legendes_matrice": legendes_matrice,
    }
    np.savez_compressed(
        mo.notebook_dir() / "figures" / ("resultats-smoke.npz" if SMOKE else "resultats.npz"),
        bilan=json.dumps(bilan, ensure_ascii=False),
        courbe_vit=courbes["ViT"],
        courbe_cnn=courbes["CNN"],
        matrice_sim=matrice_sim,
        images_matrice=images_matrice,
        sim_positions=sim_positions,
    )
    return (bilan,)


@app.cell
def _(SMOKE, bilan, mo):
    # --- Bilan : seuils chiffrés du README. En smoke, on vérifie seulement que tout tourne.
    _v, _c = bilan["scores"]["ViT"], bilan["scores"]["CNN"]
    seuils = {
        "patchs NumPy = convolution (écart < 1e-12)": bilan["ecart_patchs"] < 1e-12,
        "perte CLIP NumPy = PyTorch (écart < 1e-9)":
            abs(bilan["perte_jouet"]["numpy"] - bilan["perte_jouet"]["torch"]) < 1e-9,
        "ViT : légende complète ≥ 80 % (couples vus)": _v["complete_vus"] >= 0.80,
        "CNN : légende complète ≥ 95 % (couples vus)": _c["complete_vus"] >= 0.95,
        "ViT et CNN : couleur en zero-shot ≥ 95 %": min(_v["couleur_prompt"], _c["couleur_prompt"]) >= 0.95,
        "ViT : forme en zero-shot (ensemble) ≤ celle du CNN": _v["forme_ensemble"] <= _c["forme_ensemble"],
        "couples jamais vus : bien moins bons que les vus (écart ≥ 40 points)":
            max(_v["complete_jamais_vus"], _c["complete_jamais_vus"]) <= min(_v["complete_vus"], _c["complete_vus"]) - 0.40,
    }
    _lignes = "\n".join(f"- {'✅' if ok else '❌'} {nom}" for nom, ok in seuils.items())
    print(_lignes)
    if not SMOKE and not all(seuils.values()):
        raise SystemExit("Seuils non atteints :\n" + _lignes)
    mo.md(f"**Bilan**\n\n{_lignes}")


if __name__ == "__main__":
    app.run()
