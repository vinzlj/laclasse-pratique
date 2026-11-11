# Exercice 14 : détection et segmentation (IoU, NMS, U-Net, pertes focale et Dice).
#
#   make ex NUM=14              → éditeur marimo (bouton « Entraîner »)
#   make ex NUM=14 MODE=run     → script complet, vérifie les seuils du README
#   make ex NUM=14 MODE=smoke   → test rapide (< 30 s)
#
# Partie A en NumPy (boîtes, IoU, NMS, Dice, perte focale), vérifiée contre torchvision.
# Parties B à D en PyTorch (GPU Apple via MPS si disponible) sur des formes synthétiques
# générées par ce notebook : rien n'est téléchargé.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy import ndimage
    from torch import nn

    from ia_commun import get_device, is_smoke, sauvegarder_figure, seed_everything, style_commun

    return (
        F,
        get_device,
        is_smoke,
        json,
        mo,
        ndimage,
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
    # Exercice 14 : localiser et délimiter les objets

    Un classifieur (chapitre 13) répond « il y a un chat ». Ici on veut savoir **où** :
    une **boîte** autour de chaque objet (détection), ou la **classe de chaque pixel**
    (segmentation sémantique).

    Plan :

    - **A. Boîtes et masques à la main (NumPy), vérifiés contre torchvision** : IoU entre
      boîtes, suppression des non-maxima (NMS), IoU et Dice entre masques, perte focale.
    - **B. Données** : images 64×64 de cercles, carrés et triangles (tournés, de couleur
      quelconque) sur un fond en dégradé bruité. Le fond occupe ≈ 90 % des pixels.
    - **C. Segmentation (PyTorch)** : un U-Net réduit, le même réseau **sans ses ponts**
      (connexions de saut), puis le U-Net avec la perte focale et avec entropie croisée + Dice.
    - **D. Des masques aux boîtes** : chaque tache du masque prédit devient une boîte
      notée ; précision, rappel et mAP à IoU ≥ 0,5.

    **Pertes optimisées** (partie C), moyennées sur les pixels :

    - entropie croisée par pixel $\mathcal{L}_{CE} = -\log p_t$, où $p_t$ est la
      probabilité donnée à la bonne classe du pixel ;
    - perte focale $\mathcal{L}_{focale} = -(1-p_t)^\gamma \log p_t$ avec $\gamma = 2$
      (les pixels déjà bien classés, $p_t \to 1$, ne pèsent presque plus) ;
    - Dice « douce » $\mathcal{L}_{Dice} = 1 - \frac{1}{K}\sum_k
      \frac{2\sum_i p_{ik} y_{ik}}{\sum_i p_{ik} + \sum_i y_{ik}}$ sur les $K = 3$
      classes d'objets, ajoutée à l'entropie croisée.
    """)


@app.cell
def _(get_device, is_smoke, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    device = get_device()
    style_commun()
    config = dict(
        taille=64,                         # images carrées 64×64
        n_train=320 if SMOKE else 3000,
        n_val=64 if SMOKE else 500,
        epoques=1 if SMOKE else 6,
        lot=32,
        lr=2e-3,
        canaux=8 if SMOKE else 16,         # largeur du premier étage du U-Net
        gamma_focale=2.0,
    )
    CLASSES = ("fond", "cercle", "carré", "triangle")
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return CLASSES, SEED, SMOKE, config, device


@app.cell
def _(mo):
    mo.md(r"""
    ## A. Boîtes et masques à la main

    Une boîte est notée $(x_0, y_0, x_1, y_1)$ : coin haut gauche, coin bas droit.
    L'**IoU** (*intersection over union*, indice de Jaccard) de deux boîtes vaut
    $\text{aire}(A \cap B) / \text{aire}(A \cup B)$ : 1 si elles coïncident, 0 si elles
    sont disjointes. En détection, une prédiction compte comme correcte si son IoU avec
    une vraie boîte dépasse un seuil (0,5 dans PASCAL VOC).
    """)


@app.cell
def _(np):
    def iou_boites(a, b):
        """IoU de chaque boîte de a (N, 4) avec chaque boîte de b (M, 4) → (N, M)."""
        a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
        b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
        # Rectangle d'intersection : le plus grand des coins hauts gauches,
        # le plus petit des coins bas droits (diffusion NumPy : (N, 1) contre (1, M))
        x0 = np.maximum(a[:, None, 0], b[None, :, 0])
        y0 = np.maximum(a[:, None, 1], b[None, :, 1])
        x1 = np.minimum(a[:, None, 2], b[None, :, 2])
        y1 = np.minimum(a[:, None, 3], b[None, :, 3])
        inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)  # 0 si disjointes
        aire_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        aire_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        union = aire_a[:, None] + aire_b[None, :] - inter
        return inter / union

    def nms(boites, scores, seuil=0.5):
        """Suppression des non-maxima : garde la boîte la mieux notée, supprime celles
        qui la recouvrent à plus de `seuil`, recommence avec les restantes."""
        boites = np.asarray(boites, dtype=np.float64)
        ordre = np.argsort(-np.asarray(scores))
        gardees = []
        while ordre.size > 0:
            i = ordre[0]
            gardees.append(int(i))
            recouvrement = iou_boites(boites[i], boites[ordre[1:]])[0]
            ordre = ordre[1:][recouvrement <= seuil]
        return np.array(gardees, dtype=np.int64)

    def iou_masques(m1, m2):
        inter = np.logical_and(m1, m2).sum()
        return inter / np.logical_or(m1, m2).sum()

    def dice_masques(m1, m2):
        return 2 * np.logical_and(m1, m2).sum() / (m1.sum() + m2.sum())

    def focale(p_t, gamma):
        """Perte focale d'un pixel dont la bonne classe reçoit la probabilité p_t
        (gamma = 0 : entropie croisée)."""
        return -((1 - p_t) ** gamma) * np.log(p_t)

    return dice_masques, focale, iou_boites, iou_masques, nms


@app.cell
def _(dice_masques, focale, iou_boites, iou_masques, np):
    # --- Exemple vérifiable à la main ---
    # A = (0, 0, 10, 10) et B = (5, 5, 15, 15) : intersection 5 × 5 = 25,
    # union 100 + 100 − 25 = 175, IoU = 25 / 175 ≈ 0,143
    A = np.array([0, 0, 10, 10])
    exemples_iou = {
        "B = (5, 5, 15, 15)": iou_boites(A, [5, 5, 15, 15])[0, 0],
        "B = (2, 0, 12, 10)": iou_boites(A, [2, 0, 12, 10])[0, 0],   # décalée de 2 px
        "B = (0, 0, 5, 10)": iou_boites(A, [0, 0, 5, 10])[0, 0],     # moitié de A
        "B = (20, 20, 30, 30)": iou_boites(A, [20, 20, 30, 30])[0, 0],
    }
    for _nom, _v in exemples_iou.items():
        print(f"IoU(A, {_nom}) = {_v:.3f}")

    # IoU et Dice de deux masques : deux disques de rayon 10 dont les centres sont à 8 px
    _yy, _xx = np.mgrid[0:40, 0:40]
    _m1 = (_xx - 16) ** 2 + (_yy - 20) ** 2 <= 100
    _m2 = (_xx - 24) ** 2 + (_yy - 20) ** 2 <= 100
    iou_disques, dice_disques = iou_masques(_m1, _m2), dice_masques(_m1, _m2)
    print(f"Deux disques décalés : IoU = {iou_disques:.3f}, Dice = {dice_disques:.3f} "
          f"(Dice = 2·IoU/(1+IoU) = {2 * iou_disques / (1 + iou_disques):.3f})")

    # Perte focale : un pixel de fond « facile » (p_t = 0,9) contre un pixel difficile (0,1)
    tableau_focale = {g: [float(focale(p, g)) for p in (0.9, 0.5, 0.1)] for g in (0, 1, 2, 5)}
    print("Perte par pixel pour p_t = 0,9 / 0,5 / 0,1 :")
    for _g, _v in tableau_focale.items():
        print(f"  gamma = {_g} : " + " / ".join(f"{x:.4f}" for x in _v)
              + f"   (rapport facile/difficile : {_v[0] / _v[2]:.4f})")
    return dice_disques, exemples_iou, iou_disques, tableau_focale


@app.cell
def _(iou_boites, nms, np, torch):
    # --- Vérification contre torchvision (mêmes conventions de coordonnées) ---
    import torchvision

    _rng = np.random.default_rng(42)
    _xy = _rng.uniform(0, 50, (200, 2))
    _wh = _rng.uniform(2, 30, (200, 2))
    _b = np.concatenate([_xy, _xy + _wh], axis=1)
    _s = _rng.uniform(0, 1, 200)
    _ref_iou = torchvision.ops.box_iou(torch.from_numpy(_b), torch.from_numpy(_b)).numpy()
    ecart_iou_torchvision = float(np.abs(iou_boites(_b, _b) - _ref_iou).max())
    _ref_nms = torchvision.ops.nms(torch.from_numpy(_b), torch.from_numpy(_s), 0.5).numpy()
    nms_identique = bool(np.array_equal(nms(_b, _s, 0.5), _ref_nms))
    print(f"IoU : écart max avec torchvision.ops.box_iou = {ecart_iou_torchvision:.1e}")
    print(f"NMS : même liste que torchvision.ops.nms ({_ref_nms.size} boîtes gardées "
          f"sur 200) : {nms_identique}")
    return ecart_iou_torchvision, nms_identique


@app.cell
def _(mo):
    mo.md(r"""
    ## B. Les données : des formes générées

    Chaque image contient 1 à 5 objets qui ne se touchent pas, et 3 traits « leurres » qui
    font partie du fond. La **classe dépend de la
    forme seule** (la couleur est tirée au hasard) : pour classer un pixel, le réseau doit
    regarder autour de lui. Le générateur renvoie aussi les vraies boîtes, pour la partie D.
    """)


@app.cell
def _(np):
    def generer_formes(n, taille=64, graine=0, max_objets=5, n_traits=3):
        """Images (n, 3, T, T) float32 dans [0, 1], masques (n, T, T) int64
        (0 fond, 1 cercle, 2 carré, 3 triangle) et boîtes : liste de tableaux (k, 5)
        [x0, y0, x1, y1, classe] par image (x1, y1 exclus)."""
        rng = np.random.default_rng(graine)
        yy, xx = np.mgrid[0:taille, 0:taille] + 0.5           # centres des pixels
        images = np.empty((n, 3, taille, taille), dtype=np.float32)
        masques = np.zeros((n, taille, taille), dtype=np.int64)
        boites = []
        for k in range(n):
            # Fond : dégradé linéaire entre deux couleurs aléatoires, dans une direction aléatoire
            c0, c1 = rng.uniform(0, 1, 3), rng.uniform(0, 1, 3)
            a = rng.uniform(0, 2 * np.pi)
            t = np.clip(((xx - taille / 2) * np.cos(a) + (yy - taille / 2) * np.sin(a)) / taille + 0.5, 0, 1)
            img = c0[:, None, None] * (1 - t) + c1[:, None, None] * t
            # Leurres : quelques traits droits de couleur aléatoire, qui restent du « fond »
            for _t in range(n_traits):
                p0, p1 = rng.uniform(0, taille, 2), rng.uniform(0, taille, 2)
                d = p1 - p0
                s = np.clip(((xx - p0[0]) * d[0] + (yy - p0[1]) * d[1]) / (d @ d + 1e-9), 0, 1)
                dist = np.hypot(xx - p0[0] - s * d[0], yy - p0[1] - s * d[1])
                img[:, dist <= rng.uniform(0.6, 1.3)] = rng.uniform(0, 1, (3, 1))
            masque = np.zeros((taille, taille), dtype=np.int64)
            liste = []
            n_obj = rng.integers(1, max_objets + 1)
            for _essai in range(60):
                if len(liste) == n_obj:
                    break
                classe = int(rng.integers(1, 4))
                r = rng.uniform(4, 10)                          # « rayon » de l'objet
                cx, cy = rng.uniform(r + 1, taille - r - 1, 2)
                theta = rng.uniform(0, 2 * np.pi)
                dx, dy = xx - cx, yy - cy
                if classe == 1:                                 # disque
                    m = dx**2 + dy**2 <= r**2
                elif classe == 2:                               # carré tourné, même aire que le disque
                    u = dx * np.cos(theta) + dy * np.sin(theta)
                    v = -dx * np.sin(theta) + dy * np.cos(theta)
                    demi = r * np.sqrt(np.pi) / 2
                    m = (np.abs(u) <= demi) & (np.abs(v) <= demi)
                else:                                           # triangle équilatéral tourné
                    R = r * 1.25
                    angles = theta + 2 * np.pi * np.arange(3) / 3
                    px, py = cx + R * np.cos(angles), cy + R * np.sin(angles)
                    m = np.ones_like(masque, dtype=bool)
                    for i in range(3):                          # même côté des trois arêtes
                        j = (i + 1) % 3
                        m &= (px[j] - px[i]) * (yy - py[i]) - (py[j] - py[i]) * (xx - px[i]) >= 0
                if m.sum() < 12:
                    continue
                ys, xs = np.nonzero(m)
                boite = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, classe])
                # Pas de contact : boîtes séparées d'au moins 2 pixels
                if any(boite[0] < b[2] + 2 and b[0] < boite[2] + 2
                       and boite[1] < b[3] + 2 and b[1] < boite[3] + 2 for b in liste):
                    continue
                # Couleur aléatoire, assez différente du fond local
                fond_local = img[:, m].mean(axis=1)
                for _c in range(20):
                    couleur = rng.uniform(0, 1, 3)
                    if np.linalg.norm(couleur - fond_local) > 0.45:
                        break
                img[:, m] = couleur[:, None]
                masque[m] = classe
                liste.append(boite)
            img = img + rng.normal(0, 0.08, img.shape)          # bruit de capteur
            images[k] = np.clip(img, 0, 1)
            masques[k] = masque
            boites.append(np.array(liste, dtype=np.int64).reshape(-1, 5))
        return images, masques, boites

    return (generer_formes,)


@app.cell
def _(CLASSES, config, generer_formes, np, time):
    _t0 = time.perf_counter()
    X_train, M_train, B_train = generer_formes(config["n_train"], config["taille"], graine=42)
    X_val, M_val, B_val = generer_formes(config["n_val"], config["taille"], graine=4242)
    duree_generation = time.perf_counter() - _t0
    frequences = np.bincount(M_train.ravel(), minlength=4) / M_train.size
    n_objets_val = sum(len(b) for b in B_val)
    print(f"{len(X_train)} images d'entraînement, {len(X_val)} de validation "
          f"({n_objets_val} objets), générées en {duree_generation:.1f} s")
    print("Part des pixels : " + ", ".join(f"{c} {100 * f:.1f} %" for c, f in zip(CLASSES, frequences)))
    precision_tout_fond = float((M_val == 0).mean())
    print(f"Un « modèle » qui répond toujours « fond » a {100 * precision_tout_fond:.1f} % "
          "de pixels justes… et une IoU nulle sur chaque classe d'objet.")
    return (
        B_val,
        M_train,
        M_val,
        X_train,
        X_val,
        duree_generation,
        frequences,
        n_objets_val,
        precision_tout_fond,
    )


@app.cell
def _(B_val, CLASSES, M_val, X_val, plt):
    # Aperçu : quatre images de validation, leur masque et leurs boîtes
    _fig, _axes = plt.subplots(2, 4, figsize=(10, 5))
    for _k in range(4):
        _axes[0, _k].imshow(X_val[_k].transpose(1, 2, 0))
        for _b in B_val[_k]:
            _axes[0, _k].add_patch(plt.Rectangle((_b[0] - 0.5, _b[1] - 0.5), _b[2] - _b[0], _b[3] - _b[1],
                                                 fill=False, color="white", lw=1.2))
            _axes[0, _k].text(_b[0] - 0.5, _b[1] - 1.5, CLASSES[_b[4]], color="white", fontsize=7)
        _axes[1, _k].imshow(M_val[_k], cmap="viridis", vmin=0, vmax=3, interpolation="nearest")
    for _ax in _axes.ravel():
        _ax.set_axis_off()
    _axes[0, 0].set_title("image + boîtes", fontsize=9, loc="left")
    _axes[1, 0].set_title("masque (classe par pixel)", fontsize=9, loc="left")
    _fig


@app.cell
def _(mo):
    mo.md(r"""
    ## C. Segmentation : U-Net réduit

    L'**encodeur** divise la résolution par 2 à chaque étage (64 → 32 → 16 → 8) et double
    le nombre de canaux : il voit de plus en plus large mais de moins en moins net. Le
    **décodeur** remonte à 64×64. Les **ponts** du U-Net recopient, à chaque étage, les
    cartes de l'encodeur à côté de celles du décodeur (concaténation) : les détails fins
    perdus par le sous-échantillonnage reviennent. La variante « sans ponts » a
    exactement la même structure, ponts retirés.
    """)


@app.cell
def _(nn, torch):
    class Bloc(nn.Sequential):
        """Deux convolutions 3×3 (+ BatchNorm, chapitre 12, + ReLU) qui gardent la taille."""

        def __init__(self, entree, sortie):
            super().__init__(
                nn.Conv2d(entree, sortie, 3, padding=1, bias=False), nn.BatchNorm2d(sortie), nn.ReLU(inplace=True),
                nn.Conv2d(sortie, sortie, 3, padding=1, bias=False), nn.BatchNorm2d(sortie), nn.ReLU(inplace=True),
            )

    class UNet(nn.Module):
        """U-Net à 3 étages de sous-échantillonnage. ponts=False : même réseau sans
        connexions de saut (encodeur-décodeur simple)."""

        def __init__(self, n_classes=4, c=16, ponts=True):
            super().__init__()
            self.ponts = ponts
            self.enc1, self.enc2, self.enc3 = Bloc(3, c), Bloc(c, 2 * c), Bloc(2 * c, 4 * c)
            self.fond = Bloc(4 * c, 8 * c)                       # 8×8, le « bas du U »
            self.pool = nn.MaxPool2d(2)
            self.up3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, stride=2)   # 8 → 16
            self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2)   # 16 → 32
            self.up1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2)       # 32 → 64
            k = 2 if ponts else 1                                # la concaténation double les canaux
            self.dec3, self.dec2, self.dec1 = Bloc(k * 4 * c, 4 * c), Bloc(k * 2 * c, 2 * c), Bloc(k * c, c)
            self.tete = nn.Conv2d(c, n_classes, 1)               # un score par classe et par pixel

        def _monter(self, x, saut):
            return torch.cat([x, saut], dim=1) if self.ponts else x

        def forward(self, x):
            e1 = self.enc1(x)                                    # 64×64×c
            e2 = self.enc2(self.pool(e1))                        # 32×32×2c
            e3 = self.enc3(self.pool(e2))                        # 16×16×4c
            f = self.fond(self.pool(e3))                         # 8×8×8c
            d3 = self.dec3(self._monter(self.up3(f), e3))
            d2 = self.dec2(self._monter(self.up2(d3), e2))
            d1 = self.dec1(self._monter(self.up1(d2), e1))
            return self.tete(d1)                                 # (lot, n_classes, 64, 64)

    return (UNet,)


@app.cell
def _(F, torch):
    def perte_focale(logits, cible, gamma=2.0):
        """-(1 - p_t)^gamma · log p_t, moyennée sur tous les pixels."""
        log_p = F.log_softmax(logits, dim=1)
        log_pt = log_p.gather(1, cible.unsqueeze(1)).squeeze(1)  # log-proba de la bonne classe
        return (-(1 - log_pt.exp()) ** gamma * log_pt).mean()

    def perte_dice(logits, cible, eps=1.0):
        """1 − Dice « douce » moyenne sur les classes d'objets (le fond est exclu)."""
        p = logits.softmax(dim=1)[:, 1:]                         # (lot, 3, H, W)
        y = F.one_hot(cible, p.shape[1] + 1).permute(0, 3, 1, 2)[:, 1:].float()
        inter = (p * y).sum(dim=(0, 2, 3))
        somme = p.sum(dim=(0, 2, 3)) + y.sum(dim=(0, 2, 3))
        return 1 - ((2 * inter + eps) / (somme + eps)).mean()

    def matrice_confusion(pred, cible, n=4):
        """Matrice n×n : ligne = vraie classe, colonne = classe prédite (comptes de pixels)."""
        return torch.bincount(cible.ravel() * n + pred.ravel(), minlength=n * n).reshape(n, n)

    def scores_depuis_confusion(cm):
        cm = cm.double()
        tp = cm.diag()
        iou = tp / (cm.sum(0) + cm.sum(1) - tp)
        return {
            "precision_pixels": float(tp.sum() / cm.sum()),
            "iou_par_classe": [float(v) for v in iou],
            "miou_objets": float(iou[1:].mean()),               # moyenne sur cercle, carré, triangle
        }

    return matrice_confusion, perte_dice, perte_focale, scores_depuis_confusion


@app.cell
def _(mo):
    # Le bouton empêche l'entraînement de se relancer à chaque modification d'une cellule amont.
    bouton = mo.ui.run_button(label="Entraîner")
    bouton
    return (bouton,)


@app.cell
def _(
    F,
    M_train,
    M_val,
    UNet,
    X_train,
    X_val,
    bouton,
    config,
    device,
    matrice_confusion,
    mo,
    np,
    perte_dice,
    perte_focale,
    scores_depuis_confusion,
    time,
    torch,
):
    # En script (make ex MODE=run|smoke), il n'y a personne pour cliquer : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Entraîner » pour lancer la partie C.*"))

    Xt, Mt = torch.from_numpy(X_train).to(device), torch.from_numpy(M_train).to(device)
    Xv, Mv = torch.from_numpy(X_val).to(device), torch.from_numpy(M_val).to(device)

    def evaluer(modele):
        modele.eval()
        cm = torch.zeros(4, 4, dtype=torch.long, device=device)
        probas = []
        with torch.no_grad():
            for i in range(0, len(Xv), 100):
                logits = modele(Xv[i:i + 100])
                cm += matrice_confusion(logits.argmax(1), Mv[i:i + 100])
                probas.append(logits.softmax(1).cpu())
        modele.train()
        return scores_depuis_confusion(cm.cpu()), torch.cat(probas).numpy()

    VARIANTES = {
        # nom : (ponts, fonction de perte)
        "U-Net, entropie croisée": (True, lambda lo, y: F.cross_entropy(lo, y)),
        "sans ponts, entropie croisée": (False, lambda lo, y: F.cross_entropy(lo, y)),
        "U-Net, perte focale": (True, lambda lo, y: perte_focale(lo, y, config["gamma_focale"])),
        "U-Net, entropie croisée + Dice": (True, lambda lo, y: F.cross_entropy(lo, y) + perte_dice(lo, y)),
    }
    resultats, historiques, probas_val = {}, {}, {}
    _t_total = time.perf_counter()
    for nom, (ponts, critere) in VARIANTES.items():
        torch.manual_seed(42)                                    # même initialisation pour tous
        modele = UNet(c=config["canaux"], ponts=ponts).to(device)
        optim = torch.optim.Adam(modele.parameters(), lr=config["lr"])
        n_pas = config["epoques"] * (len(Xt) // config["lot"])
        planif = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=config["lr"], total_steps=n_pas)
        gen = torch.Generator().manual_seed(42)
        hist = {"perte": [], "miou_objets": [], "precision_pixels": []}
        t0 = time.perf_counter()
        for _ep in range(config["epoques"]):
            ordre = torch.randperm(len(Xt), generator=gen)
            pertes = []
            for i in range(0, len(Xt) - config["lot"] + 1, config["lot"]):
                idx = ordre[i:i + config["lot"]].to(device)
                loss = critere(modele(Xt[idx]), Mt[idx])
                optim.zero_grad()
                loss.backward()
                optim.step()
                planif.step()
                pertes.append(loss.item())
            sc, _ = evaluer(modele)
            hist["perte"].append(float(np.mean(pertes)))
            hist["miou_objets"].append(sc["miou_objets"])
            hist["precision_pixels"].append(sc["precision_pixels"])
            print(f"  [{nom}] époque {_ep + 1}/{config['epoques']} : perte {hist['perte'][-1]:.4f}, "
                  f"mIoU objets {100 * sc['miou_objets']:.1f} % ({time.perf_counter() - t0:.0f} s)", flush=True)
        duree = time.perf_counter() - t0
        sc, probas_val[nom] = evaluer(modele)
        sc["duree_s"] = duree
        sc["parametres"] = sum(p.numel() for p in modele.parameters())
        resultats[nom], historiques[nom] = sc, hist
        print(f"{nom:32s} {sc['parametres'] / 1e6:.2f} M param., {duree:5.1f} s : "
              f"mIoU objets {100 * sc['miou_objets']:.1f} %, pixels justes {100 * sc['precision_pixels']:.2f} %, "
              "IoU par classe " + " / ".join(f"{100 * v:.1f}" for v in sc["iou_par_classe"]))
    duree_c = time.perf_counter() - _t_total
    print(f"Partie C : {duree_c:.0f} s")
    return duree_c, historiques, probas_val, resultats


@app.cell
def _(M_val, np, probas_val):
    # Précision sur les bords : pixels à moins de 1 px d'un changement de classe dans la vérité.
    # C'est là que le sous-échantillonnage fait perdre de l'information.
    _m = M_val
    bord = np.zeros_like(_m, dtype=bool)
    bord[:, 1:, :] |= _m[:, 1:, :] != _m[:, :-1, :]
    bord[:, :-1, :] |= _m[:, 1:, :] != _m[:, :-1, :]
    bord[:, :, 1:] |= _m[:, :, 1:] != _m[:, :, :-1]
    bord[:, :, :-1] |= _m[:, :, 1:] != _m[:, :, :-1]
    precision_bords = {nom: float((p.argmax(1) == _m)[bord].mean()) for nom, p in probas_val.items()}
    precision_interieur = {nom: float((p.argmax(1) == _m)[~bord].mean()) for nom, p in probas_val.items()}
    print(f"Pixels de bord : {100 * bord.mean():.1f} % de l'image")
    for _nom in precision_bords:
        print(f"  {_nom:32s} bords {100 * precision_bords[_nom]:.1f} %, "
              f"ailleurs {100 * precision_interieur[_nom]:.2f} %")
    return precision_bords, precision_interieur


@app.cell
def _(mo):
    mo.md(r"""
    ## D. Des masques aux boîtes

    Chaque tache connexe d'une classe dans le masque prédit devient une boîte, notée par la
    probabilité moyenne de sa classe sur la tache. Une boîte prédite est un **vrai positif**
    si elle a la bonne classe et une IoU ≥ 0,5 avec une vraie boîte pas encore appariée
    (les boîtes sont examinées par score décroissant). On en tire précision, rappel et
    **AP** (*average precision*, aire sous la courbe précision-rappel) par classe ; leur
    moyenne est la **mAP@0,5**, la mesure de PASCAL VOC.
    """)


@app.cell
def _(B_val, iou_boites, ndimage, np, probas_val):
    def masque_vers_boites(proba, taille_min=6):
        """proba (4, H, W) → tableau (k, 6) [x0, y0, x1, y1, classe, score]."""
        pred = proba.argmax(0)
        sorties = []
        for classe in (1, 2, 3):
            etiquettes, n = ndimage.label(pred == classe)        # taches connexes (4-voisinage)
            for j, (sy, sx) in enumerate(ndimage.find_objects(etiquettes), start=1):
                tache = etiquettes == j
                if tache.sum() < taille_min:                     # miettes : ignorées
                    continue
                score = proba[classe][tache].mean()
                sorties.append([sx.start, sy.start, sx.stop, sy.stop, classe, score])
        return np.array(sorties, dtype=np.float64).reshape(-1, 6)

    def evaluer_detection(predictions, verites, seuil=0.5):
        """predictions, verites : listes (une entrée par image). Renvoie précision, rappel
        et AP par classe (aire sous la courbe précision-rappel, interpolation « tous points »)."""
        bilan = {}
        for classe in (1, 2, 3):
            lignes, n_vrais = [], 0
            for pred, vrai in zip(predictions, verites):
                p = pred[pred[:, 4] == classe]
                v = vrai[vrai[:, 4] == classe, :4]
                n_vrais += len(v)
                deja = np.zeros(len(v), dtype=bool)
                for b in p[np.argsort(-p[:, 5])]:
                    ok = False
                    if len(v):
                        ious = iou_boites(b[:4], v)[0]
                        ious[deja] = 0
                        j = ious.argmax()
                        if ious[j] >= seuil:
                            deja[j], ok = True, True
                    lignes.append((b[5], ok))
            lignes.sort(key=lambda t: -t[0])
            vp = np.cumsum([ok for _, ok in lignes])
            precision = vp / np.arange(1, len(lignes) + 1)
            rappel = vp / max(n_vrais, 1)
            # AP « tous points » : précision rendue décroissante, intégrée sur le rappel
            prec_env = np.maximum.accumulate(precision[::-1])[::-1] if len(lignes) else np.zeros(0)
            r = np.concatenate([[0.0], rappel])
            ap = float(np.sum((r[1:] - r[:-1]) * prec_env))
            bilan[classe] = dict(precision=float(vp[-1] / len(lignes)) if len(lignes) else 0.0,
                                 rappel=float(vp[-1] / max(n_vrais, 1)) if len(lignes) else 0.0,
                                 ap=ap, n_pred=len(lignes), n_vrais=n_vrais)
        return bilan

    detection = {}
    boites_predites = {}
    for _nom, _p in probas_val.items():
        boites_predites[_nom] = [masque_vers_boites(_x) for _x in _p]
        detection[_nom] = evaluer_detection(boites_predites[_nom], B_val)
        _d = detection[_nom]
        detection[_nom]["map50"] = float(np.mean([_d[c]["ap"] for c in (1, 2, 3)]))
        print(f"{_nom:32s} mAP@0,5 = {100 * _d['map50']:.1f} % ; "
              + " ; ".join(f"classe {c} : P {100 * _d[c]['precision']:.1f} % R {100 * _d[c]['rappel']:.1f} %"
                           for c in (1, 2, 3)))
    return boites_predites, detection


@app.cell
def _(B_val, boites_predites, nms, np):
    # NMS en situation : un détecteur à fenêtres glissantes renvoie souvent plusieurs boîtes
    # par objet. On simule ces doublons autour des vraies boîtes de 8 images de validation
    # (décalages de ±3 px, scores aléatoires) et on vérifie que la NMS n'en garde qu'une par objet.
    _rng = np.random.default_rng(42)
    _ref = "U-Net, entropie croisée"
    n_candidats = n_gardes = n_objets_nms = 0
    for _k in range(8):
        _cands, _scores = [], []
        for _b in B_val[_k]:
            for _r in range(6):
                _cands.append(_b[:4] + _rng.integers(-3, 4, 4))
                _scores.append(_rng.uniform(0.5, 1.0))
        _gardes = nms(np.array(_cands), np.array(_scores), seuil=0.3)
        n_candidats += len(_cands)
        n_gardes += len(_gardes)
        n_objets_nms += len(B_val[_k])
    print(f"NMS (seuil 0,3) : {n_candidats} boîtes candidates → {n_gardes} gardées "
          f"pour {n_objets_nms} objets")
    n_boites_unet = int(sum(len(b) for b in boites_predites[_ref]))
    return n_boites_unet, n_candidats, n_gardes, n_objets_nms


@app.cell
def _(
    B_val,
    CLASSES,
    M_val,
    SMOKE,
    X_val,
    boites_predites,
    historiques,
    mo,
    np,
    plt,
    probas_val,
    sauvegarder_figure,
):
    # --- Figures : exemples de segmentation et courbes d'apprentissage ---
    _noms = list(probas_val)
    _idx = [0, 1, 2, 3, 5, 7]
    _fig, _axes = plt.subplots(len(_idx), 2 + len(_noms), figsize=(2 * (2 + len(_noms)), 2 * len(_idx)))
    for _r, _k in enumerate(_idx):
        _axes[_r, 0].imshow(X_val[_k].transpose(1, 2, 0))
        _axes[_r, 1].imshow(M_val[_k], cmap="viridis", vmin=0, vmax=3, interpolation="nearest")
        for _c, _nom in enumerate(_noms):
            _axes[_r, 2 + _c].imshow(probas_val[_nom][_k].argmax(0), cmap="viridis", vmin=0, vmax=3,
                                     interpolation="nearest")
            for _b in boites_predites[_nom][_k]:
                _axes[_r, 2 + _c].add_patch(plt.Rectangle((_b[0] - 0.5, _b[1] - 0.5), _b[2] - _b[0], _b[3] - _b[1],
                                                          fill=False, color="white", lw=0.8))
        for _b in B_val[_k]:
            _axes[_r, 1].add_patch(plt.Rectangle((_b[0] - 0.5, _b[1] - 0.5), _b[2] - _b[0], _b[3] - _b[1],
                                                 fill=False, color="white", lw=0.8))
    for _c, _t in enumerate(["image", "vérité"] + _noms):
        _axes[0, _c].set_title(_t.replace(", ", "\n"), fontsize=8)
    for _ax in _axes.ravel():
        _ax.set_axis_off()
    _fig.suptitle("Classes : " + ", ".join(f"{i} {c}" for i, c in enumerate(CLASSES)), fontsize=9)
    _dossier = mo.notebook_dir() / "figures"
    _suffixe = "-smoke" if SMOKE else ""
    sauvegarder_figure(_fig, "exemples" + _suffixe, _dossier)

    _fig2, _ax2 = plt.subplots(figsize=(7, 4))
    for _nom, _h in historiques.items():
        _ax2.plot(np.arange(1, len(_h["miou_objets"]) + 1), 100 * np.array(_h["miou_objets"]),
                  marker="o", ms=3, label=_nom)
    _ax2.set(xlabel="époque", ylabel="mIoU objets en validation (%)", title="Apprentissage des quatre variantes")
    _ax2.legend(fontsize=8)
    sauvegarder_figure(_fig2, "courbes" + _suffixe, _dossier)
    _fig


@app.cell
def _(
    CLASSES,
    M_val,
    SMOKE,
    X_val,
    boites_predites,
    config,
    dice_disques,
    detection,
    duree_c,
    duree_generation,
    ecart_iou_torchvision,
    exemples_iou,
    frequences,
    historiques,
    iou_disques,
    json,
    mo,
    n_boites_unet,
    n_candidats,
    n_gardes,
    n_objets_nms,
    n_objets_val,
    nms_identique,
    np,
    precision_bords,
    precision_interieur,
    precision_tout_fond,
    probas_val,
    resultats,
    tableau_focale,
):
    # --- Sauvegarde des résultats (lus par les figures du site) ---
    _dossier = mo.notebook_dir() / "figures"
    _dossier.mkdir(exist_ok=True)
    _suffixe = "-smoke" if SMOKE else ""
    bilan = dict(
        config=config, classes=CLASSES, frequences=[float(f) for f in frequences],
        precision_tout_fond=precision_tout_fond, n_objets_val=n_objets_val,
        duree_generation_s=duree_generation, duree_c_s=duree_c,
        exemples_iou=exemples_iou, iou_disques=float(iou_disques), dice_disques=float(dice_disques),
        focale={str(g): v for g, v in tableau_focale.items()},
        ecart_iou_torchvision=ecart_iou_torchvision, nms_identique=nms_identique,
        nms=dict(candidats=n_candidats, gardes=n_gardes, objets=n_objets_nms),
        segmentation=resultats, historiques=historiques,
        precision_bords=precision_bords, precision_interieur=precision_interieur,
        detection={n: {str(k): v for k, v in d.items()} for n, d in detection.items()},
        n_boites_unet=n_boites_unet,
    )
    (_dossier / f"resultats{_suffixe}.json").write_text(json.dumps(bilan, indent=1, ensure_ascii=False,
                                                                    default=float))
    # Exemples pour les figures du site : 8 images, vérité, prédictions, boîtes
    _k = np.arange(8)
    np.savez_compressed(
        _dossier / f"exemples{_suffixe}.npz",
        images=X_val[_k], masques=M_val[_k],
        **{f"pred_{i}": probas_val[n][_k].argmax(1).astype(np.int8) for i, n in enumerate(probas_val)},
        noms=np.array(list(probas_val)),
        **{f"boites_{i}_{k}": boites_predites[n][k] for i, n in enumerate(probas_val) for k in _k},
    )
    print(f"Résultats : {_dossier / f'resultats{_suffixe}.json'}")
    return (bilan,)


@app.cell
def _(SMOKE, bilan, mo):
    # --- Bilan : seuils chiffrés du README. En smoke, on vérifie seulement que tout tourne.
    _s = bilan["segmentation"]
    _d = bilan["detection"]
    _ref, _sans = "U-Net, entropie croisée", "sans ponts, entropie croisée"
    _verifs = [
        ("IoU NumPy = torchvision (écart < 1e-9)", bilan["ecart_iou_torchvision"] < 1e-9),
        ("NMS NumPy = torchvision", bilan["nms_identique"]),
        ("U-Net (entropie croisée) : mIoU objets ≥ 80 %", _s[_ref]["miou_objets"] >= 0.80),
        ("U-Net (entropie croisée) : mAP@0,5 ≥ 85 %", _d[_ref]["map50"] >= 0.85),
        ("Ponts : + 4 points de mIoU au moins",
         _s[_ref]["miou_objets"] - _s[_sans]["miou_objets"] >= 0.04),
        ("Ponts : pixels de bord mieux classés (+ 5 points au moins)",
         bilan["precision_bords"][_ref] - bilan["precision_bords"][_sans] >= 0.05),
        ("U-Net (entropie croisée + Dice) : mIoU objets ≥ 85 %",
         _s["U-Net, entropie croisée + Dice"]["miou_objets"] >= 0.85),
    ]
    _lignes = "\n".join(f"- {'✅' if ok else '❌'} {txt}" for txt, ok in _verifs)
    _echecs = [txt for txt, ok in _verifs if not ok]
    if not SMOKE and _echecs:
        raise SystemExit("Seuils non atteints :\n" + "\n".join(_echecs))
    mo.md(f"**Bilan** ({bilan['duree_c_s']:.0f} s d'entraînement)\n\n{_lignes}")


if __name__ == "__main__":
    app.run()
