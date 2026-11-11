# Exercice 25a : diffusion (DDPM, DDIM, guidage sans classifieur) et flow matching sur MNIST.
#
#   make ex NUM=25a              → éditeur marimo (bouton « Entraîner »)
#   make ex NUM=25a MODE=run     → script complet, vérifie les seuils du README
#   make ex NUM=25a MODE=smoke   → test rapide (< 30 s)
#
# Partie A en NumPy (calendrier de bruit, formule fermée, diffusion et flow matching en 2D avec
# le débruiteur EXACT), parties B à E en PyTorch sur MNIST (GPU Apple via MPS si disponible).

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
    from scipy import linalg
    from torch import nn

    from ia_commun import checkpoints_dir, get_device, is_smoke, sauvegarder_figure, seed_everything, style_commun
    from ia_commun.donnees import mnist_tenseurs

    return (
        F,
        checkpoints_dir,
        get_device,
        is_smoke,
        json,
        linalg,
        math,
        mnist_tenseurs,
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
    # Exercice 25a : diffusion et flow matching sur MNIST

    Un **modèle de diffusion** apprend à retirer du bruit. On abîme une image $x_0$ en lui
    ajoutant du bruit gaussien en $T = 1\,000$ petites étapes ; grâce à une formule fermée, on
    saute directement à l'étape $t$ :
    $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$, avec
    $\varepsilon \sim \mathcal N(0, I)$ et $\bar\alpha_t = \prod_{s \le t} (1 - \beta_s)$.
    Un réseau $\varepsilon_\theta(x_t, t)$ apprend à retrouver $\varepsilon$. Pour générer, on part
    d'un bruit pur et on retire le bruit prédit, étape par étape.

    - **A. NumPy, sans apprentissage.** Le calendrier de bruit de DDPM ; la formule fermée
      contre 1 000 étapes de bruitage ; puis, sur un nuage 2D de huit gaussiennes dont on
      connaît le **débruiteur exact**, les échantillonneurs DDPM (1 000 pas), DDIM et flow
      matching (de 1 à 100 pas). Ce que le réseau devra approcher, et ce que coûte chaque
      façon de remonter le temps.
    - **B. DDPM sur MNIST.** Un petit U-Net (chapitre 14) prédit le bruit ; la classe du chiffre
      est donnée en condition, sauf dans 10 % des cas où on la remplace par « aucune classe ».
    - **C. Échantillonner.** DDPM en 1 000 pas contre DDIM en 100, 50, 20 et 10 pas, avec le
      même réseau, en demandant 100 images de chaque chiffre. Les images sont jugées par le même
      petit CNN que l'exercice 18 (distance de Fréchet entre caractéristiques, comme la FID).
    - **D. Guidage sans classifieur.** Demander un chiffre précis, avec un poids de guidage de
      0 à 7 : le chiffre demandé est-il obtenu, et que devient la diversité ?
    - **E. Flow matching.** Le même U-Net, entraîné à prédire une vitesse sur la ligne droite
      entre image et bruit ; échantillonné en 50 à 5 pas d'Euler.

    **Pertes optimisées.**
    - Diffusion (B) : $\mathbb E_{x_0, \varepsilon, t}\,\lVert \varepsilon - \varepsilon_\theta(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon,\ t,\ y)\rVert^2$
      (« ε-prédiction », la perte simplifiée de Ho et al., 2020).
    - Flow matching (E) : $\mathbb E_{x_0, \varepsilon, t}\,\lVert (\varepsilon - x_0) - v_\theta((1-t)\,x_0 + t\,\varepsilon,\ t,\ y)\rVert^2$, $t \sim \mathcal U[0, 1]$.
    - Juge : entropie croisée. Partie A : aucune (rien n'est appris, le débruiteur est exact).
    """)


@app.cell
def _(get_device, is_smoke, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    device = get_device()
    style_commun()
    config = dict(
        # Diffusion : calendrier linéaire de DDPM (Ho et al. 2020)
        T=1000, beta_1=1e-4, beta_T=0.02,
        # A. jouet 2D : 8 gaussiennes (σ = 0,1) sur un cercle de rayon 2
        n_jouet=1000 if SMOKE else 4000, pas_jouet=[1, 2, 5, 10, 20, 50, 100],
        # B et E. U-Net
        canaux=32, mult=(1, 2), lot=128, lr=2e-3,
        n_train=4000 if SMOKE else 60000, epoques=1 if SMOKE else 4, p_sans_classe=0.1,
        # Juge (même recette que l'exercice 18)
        epoques_juge=1 if SMOKE else 3,
        # C, D, E. échantillonnage
        n_ech=200 if SMOKE else 1000,
        pas_ddpm=50 if SMOKE else 1000,
        pas_ddim=[20, 5] if SMOKE else [100, 50, 20, 10],
        pas_fm=[20, 5] if SMOKE else [50, 20, 10, 5],
        guidages=[0.0, 1.0, 3.0, 7.0], pas_guidage=10 if SMOKE else 50,
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, périphérique {device}")
    print(config)
    return SEED, SMOKE, config, device


@app.cell
def _(mo):
    mo.md(r"""
    ## A. NumPy : le bruit, et le débruiteur exact d'un nuage 2D
    """)


@app.cell
def _(config, np):
    # --- A1. Calendrier de bruit : β_t linéaire, ᾱ_t = ∏(1 − β_s) ---
    T = config["T"]
    betas_np = np.linspace(config["beta_1"], config["beta_T"], T)
    alphas_barre_np = np.cumprod(1 - betas_np)
    resultats_a = {
        "alpha_barre_T": float(alphas_barre_np[-1]),
        "racine_alpha_barre_T": float(np.sqrt(alphas_barre_np[-1])),
        # Premier t (de 1 à T) où l'image et le bruit pèsent autant en variance : ᾱ_t < 0,5
        "t_moitie": int(np.argmax(alphas_barre_np < 0.5) + 1),
        "t_signal_10pc": int(np.argmax(np.sqrt(alphas_barre_np) < 0.1) + 1),
    }

    # --- A2. Formule fermée contre 1 000 étapes de bruitage, sur 20 000 « pixels » valant ±1 ---
    _rng = np.random.default_rng(42)
    _x0 = np.where(_rng.random(20000) < 0.5, -1.0, 1.0)
    _x = _x0.copy()
    _controle = {}
    for _t in range(T):
        _x = np.sqrt(1 - betas_np[_t]) * _x + np.sqrt(betas_np[_t]) * _rng.standard_normal(_x.shape)
        if _t + 1 in (100, 300, 1000):
            _ab = alphas_barre_np[_t]
            # Régression de x_t sur x_0 : pente attendue √ᾱ_t, variance résiduelle 1 − ᾱ_t
            _pente = float((_x * _x0).mean() / (_x0 ** 2).mean())
            _residu = float((_x - _pente * _x0).var())
            _controle[_t + 1] = {"pente": _pente, "pente_formule": float(np.sqrt(_ab)),
                                 "var_residu": _residu, "var_formule": float(1 - _ab)}
    resultats_a["ferme_contre_iteratif"] = _controle
    for _t, _c in _controle.items():
        print(f"t = {_t:4d} : pente {_c['pente']:.4f} (formule {_c['pente_formule']:.4f}), "
              f"variance du bruit {_c['var_residu']:.4f} (formule {_c['var_formule']:.4f})")
    print(f"ᾱ_T = {resultats_a['alpha_barre_T']:.2e} (√ᾱ_T = {resultats_a['racine_alpha_barre_T']:.4f}) ; "
          f"ᾱ_t passe sous 0,5 à t = {resultats_a['t_moitie']} ; √ᾱ_t sous 0,1 à t = {resultats_a['t_signal_10pc']}")
    return T, alphas_barre_np, betas_np, resultats_a


@app.cell
def _(T, alphas_barre_np, betas_np, config, np, resultats_a, time):
    # --- A3. Jouet 2D : huit gaussiennes, débruiteur EXACT (aucun réseau) ---
    # Les données suivent un mélange de 8 gaussiennes N(μ_k, σ²I). Bruité à l'instant t, le mélange
    # reste un mélange de gaussiennes : N(√ᾱ μ_k, (ᾱσ² + 1 − ᾱ) I). On sait donc calculer le bruit
    # moyen E[ε | x_t] : c'est exactement ce qu'un réseau parfait prédirait.
    K, RAYON, SIGMA = 8, 2.0, 0.1
    _angles = 2 * np.pi * np.arange(K) / K
    MU = np.stack([RAYON * np.cos(_angles), RAYON * np.sin(_angles)], 1)

    def _poids(x, centres, var):
        # Probabilité a posteriori de chaque composante (softmax stable)
        d = x[:, None, :] - centres[None]
        logw = -(d ** 2).sum(-1) / (2 * var)
        logw -= logw.max(1, keepdims=True)
        w = np.exp(logw)
        return w / w.sum(1, keepdims=True), d

    def eps_exact(x, ab):
        var = ab * SIGMA ** 2 + 1 - ab
        w, d = _poids(x, np.sqrt(ab) * MU, var)
        score = -(w[:, :, None] * d).sum(1) / var          # ∇ log p_t(x)
        return -np.sqrt(1 - ab) * score                     # ε* = −√(1 − ᾱ) · score

    def vitesse_exacte(x, t):
        # Flow matching : x_t = (1 − t) x0 + t ε ; vitesse cible ε − x0 ; v*(x, t) = E[ε − x0 | x_t]
        var = (1 - t) ** 2 * SIGMA ** 2 + t ** 2
        w, d = _poids(x, (1 - t) * MU, var)
        e_x0 = (w[:, :, None] * (MU[None] + ((1 - t) * SIGMA ** 2 / var) * d)).sum(1)
        e_eps = (x - (1 - t) * e_x0) / t
        return e_eps - e_x0

    def jouet_ddpm(x, rng, garder=()):
        traj = {}
        for t in range(T - 1, -1, -1):
            e = eps_exact(x, alphas_barre_np[t])
            x = (x - betas_np[t] / np.sqrt(1 - alphas_barre_np[t]) * e) / np.sqrt(1 - betas_np[t])
            if t > 0:
                x = x + np.sqrt(betas_np[t]) * rng.standard_normal(x.shape)
            if t in garder:
                traj[t] = x.copy()
        return x, traj

    def jouet_ddim(x, n_pas, trajectoire=False):
        ts = np.linspace(T - 1, 0, n_pas).round().astype(int)
        chemin = [x.copy()]
        for i, t in enumerate(ts):
            e = eps_exact(x, alphas_barre_np[t])
            x0 = (x - np.sqrt(1 - alphas_barre_np[t]) * e) / np.sqrt(alphas_barre_np[t])
            ab_suiv = alphas_barre_np[ts[i + 1]] if i + 1 < n_pas else 1.0
            x = np.sqrt(ab_suiv) * x0 + np.sqrt(1 - ab_suiv) * e
            chemin.append(x.copy())
        return (x, np.stack(chemin)) if trajectoire else x

    def jouet_fm(x, n_pas, trajectoire=False):
        ts = np.linspace(1, 0, n_pas + 1)
        chemin = [x.copy()]
        for i in range(n_pas):
            x = x - (ts[i] - ts[i + 1]) * vitesse_exacte(x, ts[i])   # un pas d'Euler vers t = 0
            chemin.append(x.copy())
        return (x, np.stack(chemin)) if trajectoire else x

    def qualite_jouet(x):
        d = np.linalg.norm(x[:, None] - MU[None], axis=-1)
        dans = d.min(1) < 3 * SIGMA                       # à moins de 3σ d'un centre
        comptes = np.bincount(d.argmin(1)[dans], minlength=K)
        return {"part_dans_modes": float(dans.mean()),
                "distance_moyenne": float(d.min(1).mean()),
                "mode_le_moins_peuple": float(comptes.min() / max(1, comptes.sum()) * K)}

    _rng = np.random.default_rng(42)
    _N = config["n_jouet"]
    _vraies = MU[_rng.integers(0, K, _N)] + SIGMA * _rng.standard_normal((_N, 2))
    _z = _rng.standard_normal((_N, 2))   # le même bruit de départ pour tous les échantillonneurs
    _t0 = time.perf_counter()
    _x_ddpm, traj_ddpm_jouet = jouet_ddpm(_z, _rng, garder=(999, 750, 500, 250, 100, 0))
    jouet = {"donnees": qualite_jouet(_vraies), "DDPM 1000 pas": qualite_jouet(_x_ddpm), "DDIM": {}, "FM": {}}
    for _n in config["pas_jouet"]:
        jouet["DDIM"][_n] = qualite_jouet(jouet_ddim(_z, _n))
        jouet["FM"][_n] = qualite_jouet(jouet_fm(_z, _n))
    # Trajectoires de 30 points pour la figure (DDIM et FM, 50 pas : quasi exactes)
    _, chemins_ddim_jouet = jouet_ddim(_z[:30], 50, trajectoire=True)
    _, chemins_fm_jouet = jouet_fm(_z[:30], 50, trajectoire=True)
    # Courbure : longueur du chemin / distance entre départ et arrivée (1 = ligne droite)
    _courbure = lambda c: float(np.mean(np.linalg.norm(np.diff(c, axis=0), axis=-1).sum(0)
                                        / np.linalg.norm(c[-1] - c[0], axis=-1)))
    jouet["courbure_ddim"] = _courbure(chemins_ddim_jouet)
    jouet["courbure_fm"] = _courbure(chemins_fm_jouet)
    duree_a = time.perf_counter() - _t0
    resultats_a["jouet"] = jouet
    print(f"données       : {jouet['donnees']}")
    print(f"DDPM 1000 pas : {jouet['DDPM 1000 pas']}")
    print("pas  | DDIM : dans les modes, distance | FM : dans les modes, distance")
    for _n in config["pas_jouet"]:
        _d, _f = jouet["DDIM"][_n], jouet["FM"][_n]
        print(f"{_n:4d} | {_d['part_dans_modes']:6.1%}  {_d['distance_moyenne']:.3f} | "
              f"{_f['part_dans_modes']:6.1%}  {_f['distance_moyenne']:.3f}")
    print(f"Courbure moyenne des trajectoires (1 = droite) : DDIM {jouet['courbure_ddim']:.3f}, "
          f"FM {jouet['courbure_fm']:.3f} ; partie A en {duree_a:.1f} s")
    return (
        MU,
        chemins_ddim_jouet,
        chemins_fm_jouet,
        jouet,
        traj_ddpm_jouet,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## B. Un U-Net qui prédit le bruit (MNIST)
    """)


@app.cell
def _(config, device, mnist_tenseurs, torch):
    # --- Données : MNIST (cache data/mnist/). Pixels ramenés dans [−1, 1] : même échelle que le bruit ---
    _Xtr, _ytr = mnist_tenseurs(train=True)
    _Xte, _yte = mnist_tenseurs(train=False)
    _perm = torch.randperm(60000, generator=torch.Generator().manual_seed(42))[: config["n_train"]]
    X_train01, y_train = _Xtr[_perm].to(device), _ytr[_perm].to(device)   # [0, 1] pour le juge
    X_test01, y_test = _Xte.to(device), _yte.to(device)
    X_train = X_train01 * 2 - 1
    X_test = X_test01 * 2 - 1
    print(f"train {tuple(X_train.shape)}, test {tuple(X_test.shape)}, pixels dans [{X_train.min():.0f}, {X_train.max():.0f}]")
    return X_test, X_test01, X_train, X_train01, y_test, y_train


@app.cell
def _(F, config, math, nn, torch):
    # --- Modèles ---
    def plongement_temps(t, dim):
        # Encodage sinusoïdal de l'instant t (même idée que l'encodage de position du Transformer)
        demi = dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(demi, device=t.device) / demi)
        a = t.float()[:, None] * freq[None]
        return torch.cat([a.sin(), a.cos()], 1)

    class BlocRes(nn.Module):
        """Bloc résiduel : deux convolutions 3×3 ; le plongement (temps + classe) est ajouté entre les deux."""

        def __init__(self, c_in, c_out, d_emb):
            super().__init__()
            self.n1, self.c1 = nn.GroupNorm(8, c_in), nn.Conv2d(c_in, c_out, 3, padding=1)
            self.emb = nn.Linear(d_emb, c_out)
            self.n2, self.c2 = nn.GroupNorm(8, c_out), nn.Conv2d(c_out, c_out, 3, padding=1)
            self.saut = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

        def forward(self, x, e):
            h = self.c1(F.silu(self.n1(x))) + self.emb(e)[:, :, None, None]
            return self.c2(F.silu(self.n2(h))) + self.saut(x)

    class UNet(nn.Module):
        """U-Net 28 → 14 → 7 → 14 → 28 avec ponts (chapitre 14). Entrée : image bruitée, instant, classe.

        Classe 10 = « aucune classe » (pour le guidage sans classifieur). Sortie : une image
        (le bruit prédit pour la diffusion, la vitesse pour le flow matching).
        """

        def __init__(self, c=32, mult=(1, 2), d_emb=128, echelle_t=1.0):
            super().__init__()
            c1, c2 = c * mult[0], c * mult[1]
            self.d_emb, self.echelle_t = d_emb, echelle_t
            self.mlp_t = nn.Sequential(nn.Linear(d_emb, d_emb), nn.SiLU(), nn.Linear(d_emb, d_emb))
            self.classe = nn.Embedding(11, d_emb)
            self.entree = nn.Conv2d(1, c, 3, padding=1)
            self.e1 = BlocRes(c, c, d_emb)
            self.bas1 = nn.Conv2d(c, c, 3, stride=2, padding=1)            # 28 → 14
            self.e2 = BlocRes(c, c1, d_emb)
            self.bas2 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)          # 14 → 7
            self.milieu = BlocRes(c1, c2, d_emb)
            self.haut2 = nn.ConvTranspose2d(c2, c2, 4, stride=2, padding=1)  # 7 → 14
            self.d2 = BlocRes(c2 + c1, c1, d_emb)
            self.haut1 = nn.ConvTranspose2d(c1, c1, 4, stride=2, padding=1)  # 14 → 28
            self.d1 = BlocRes(c1 + c, c, d_emb)
            self.sortie = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

        def forward(self, x, t, y):
            e = self.mlp_t(plongement_temps(t * self.echelle_t, self.d_emb)) + self.classe(y)
            h1 = self.e1(self.entree(x), e)
            h2 = self.e2(self.bas1(h1), e)
            m = self.milieu(self.bas2(h2), e)
            u = self.d2(torch.cat([self.haut2(m), h2], 1), e)
            u = self.d1(torch.cat([self.haut1(u), h1], 1), e)
            return self.sortie(u)

    class Juge(nn.Module):
        """Petit CNN classifieur de l'exercice 18 ; `caracteristiques` = avant-dernière couche (128 valeurs)."""

        def __init__(self):
            super().__init__()
            self.corps = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU())
            self.tete = nn.Linear(128, 10)

        def caracteristiques(self, x):
            return self.corps(x)

        def forward(self, x):
            return self.tete(self.corps(x))

    _n = sum(p.numel() for p in UNet(config["canaux"], config["mult"]).parameters())
    print(f"U-Net : {_n / 1e6:.2f} M de paramètres")
    return Juge, UNet


@app.cell
def _(mo):
    # Le bouton empêche les entraînements de se relancer à chaque modification d'une cellule amont.
    bouton = mo.ui.run_button(label="Entraîner")
    bouton
    return (bouton,)


@app.cell
def _(T, X_train, alphas_barre_np, betas_np, bouton, config, mo, torch):
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Entraîner » pour lancer l'apprentissage.*"))
    _dev = X_train.device
    betas = torch.tensor(betas_np, dtype=torch.float32, device=_dev)
    alphas_barre = torch.tensor(alphas_barre_np, dtype=torch.float32, device=_dev)

    def bruiter(x0, t, eps):
        # Formule fermée : x_t = √ᾱ_t · x0 + √(1 − ᾱ_t) · ε (vérifiée en A2)
        a = alphas_barre[t][:, None, None, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * eps

    def bruiter_fm(x0, t, eps):
        # Flow matching : ligne droite entre l'image (t = 0) et le bruit (t = 1)
        t = t[:, None, None, None]
        return (1 - t) * x0 + t * eps

    _ = T
    return alphas_barre, betas, bruiter, bruiter_fm


@app.cell
def _(
    SMOKE,
    T,
    UNet,
    X_test,
    checkpoints_dir,
    X_train,
    bruiter,
    bruiter_fm,
    config,
    seed_everything,
    time,
    torch,
    y_test,
    y_train,
):
    # --- B et E. Entraînement : même U-Net, même boucle ; seules la cible et l'entrée changent ---
    def entrainer(type_modele):
        seed_everything(42)
        dev = X_train.device
        # Diffusion : t entier de 0 à T − 1 ; flow matching : t réel dans [0, 1], mis à l'échelle × 1 000
        modele = UNet(config["canaux"], config["mult"], echelle_t=1.0 if type_modele == "ddpm" else 1000.0).to(dev)
        opt = torch.optim.Adam(modele.parameters(), lr=config["lr"])
        n_pas = config["epoques"] * (len(X_train) // config["lot"])
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=config["lr"], total_steps=n_pas, pct_start=0.05)
        gen = torch.Generator().manual_seed(42)
        pertes, t0, pas = [], time.perf_counter(), 0
        for _ep in range(config["epoques"]):
            ordre = torch.randperm(len(X_train), generator=gen).to(dev)
            for i in range(0, len(X_train) - config["lot"] + 1, config["lot"]):
                idx = ordre[i:i + config["lot"]]
                x0, y = X_train[idx], y_train[idx].clone()
                y[torch.rand(len(y), device=dev) < config["p_sans_classe"]] = 10   # « aucune classe »
                eps = torch.randn_like(x0)
                if type_modele == "ddpm":
                    t = torch.randint(0, T, (len(x0),), device=dev)
                    perte = (modele(bruiter(x0, t, eps), t, y) - eps).pow(2).mean()
                else:
                    t = torch.rand(len(x0), device=dev)
                    perte = (modele(bruiter_fm(x0, t, eps), t, y) - (eps - x0)).pow(2).mean()
                opt.zero_grad()
                perte.backward()
                opt.step()
                sched.step()
                pertes.append(perte.item())
                pas += 1
                if pas % 200 == 0:
                    print(f"  [{type_modele}] pas {pas}/{n_pas} perte {sum(pertes[-200:]) / 200:.4f} "
                          f"({time.perf_counter() - t0:.0f} s)", flush=True)
        duree = time.perf_counter() - t0
        modele.eval()
        # Perte de test par tranche de t (2 000 images, même bruit pour chaque tranche)
        g2 = torch.Generator(device="cpu").manual_seed(7)
        x0 = X_test[:2000]
        eps = torch.randn(x0.shape, generator=g2).to(dev)
        par_t = []
        with torch.no_grad():
            for frac in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
                if type_modele == "ddpm":
                    t = torch.full((len(x0),), int(frac * (T - 1)), device=dev)
                    err = (modele(bruiter(x0, t, eps), t, y_test[:2000]) - eps).pow(2).mean().item()
                else:
                    t = torch.full((len(x0),), frac, device=dev)
                    err = (modele(bruiter_fm(x0, t, eps), t, y_test[:2000]) - (eps - x0)).pow(2).mean().item()
                par_t.append((frac, err))
        print(f"[{type_modele}] {pas} pas en {duree:.0f} s ; perte finale (200 derniers pas) "
              f"{sum(pertes[-200:]) / min(200, len(pertes)):.4f}", flush=True)
        return modele, {"pertes": pertes, "duree": duree, "perte_par_t": par_t}

    unet_ddpm, histo_ddpm = entrainer("ddpm")
    if not SMOKE:   # data/checkpoints/ (non versionné) : pour réutiliser le réseau sans le réentraîner
        torch.save(unet_ddpm.state_dict(), checkpoints_dir() / "25a-unet-ddpm.pt")
    for _f, _e in histo_ddpm["perte_par_t"]:
        print(f"  t = {_f:4.2f}·T : MSE sur le bruit {_e:.4f}")
    return entrainer, histo_ddpm, unet_ddpm


@app.cell
def _(F, Juge, X_test01, X_train01, config, seed_everything, time, torch, unet_ddpm, y_test, y_train):
    # --- Le juge : même petit CNN et même recette que l'exercice 18 (images dans [0, 1]) ---
    # (lancé après la diffusion, donc derrière le bouton « Entraîner » : unet_ddpm sert de dépendance)
    _ = unet_ddpm
    seed_everything(42)
    juge = Juge().to(X_train01.device)
    _opt = torch.optim.Adam(juge.parameters(), lr=1e-3)
    _gen = torch.Generator().manual_seed(42)
    _t0 = time.perf_counter()
    for _ep in range(config["epoques_juge"]):
        _ordre = torch.randperm(len(X_train01), generator=_gen).to(X_train01.device)
        for _i in range(0, len(X_train01), 128):
            _idx = _ordre[_i:_i + 128]
            _p = F.cross_entropy(juge(X_train01[_idx]), y_train[_idx])
            _opt.zero_grad()
            _p.backward()
            _opt.step()
    juge.eval()
    with torch.no_grad():
        precision_juge = float((juge(X_test01).argmax(1) == y_test).float().mean())
    duree_juge = time.perf_counter() - _t0
    print(f"Juge : précision de test {precision_juge:.2%} ({duree_juge:.1f} s)")
    return duree_juge, juge, precision_juge


@app.cell
def _(X_test01, X_train01, juge, linalg, np, torch):
    # --- Mesures sur un lot d'images générées (dans [0, 1]) ---
    @torch.no_grad()
    def statistiques(images):
        f, p = [], []
        for i in range(0, len(images), 1000):
            f.append(juge.caracteristiques(images[i:i + 1000]).cpu().double().numpy())
            p.append(torch.softmax(juge(images[i:i + 1000]), 1).cpu().double().numpy())
        return np.concatenate(f), np.concatenate(p)

    def frechet(f1, f2):
        # Distance de Fréchet entre deux gaussiennes ajustées (formule de la FID)
        m1, m2 = f1.mean(0), f2.mean(0)
        _eps = 1e-6 * np.eye(f1.shape[1])
        c1, c2 = np.cov(f1, rowvar=False) + _eps, np.cov(f2, rowvar=False) + _eps
        racine = linalg.sqrtm(c1 @ c2)
        racine = racine.real if np.iscomplexobj(racine) else racine
        return float(((m1 - m2) ** 2).sum() + np.trace(c1 + c2 - 2 * racine))

    feat_test, _ = statistiques(X_test01)

    def mesurer(images, classes_voulues=None):
        f, p = statistiques(images)
        classes = np.bincount(p.argmax(1), minlength=10) / len(p)
        r = {"distance_frechet": frechet(f, feat_test),
             "confiance_moyenne": float(p.max(1).mean()),
             "part_min_classe": float(classes.min()),
             "pixels_gris": float(((images > 0.1) & (images < 0.9)).float().mean())}
        if classes_voulues is not None:
            cv = classes_voulues.cpu().numpy()
            r["classe_obtenue"] = float((p.argmax(1) == cv).mean())
            # Diversité : distance moyenne entre caractéristiques de deux images de la même classe demandée
            div = []
            for k in range(10):
                fk = f[cv == k]
                if len(fk) > 1:
                    d = np.sqrt(((fk[:, None] - fk[None]) ** 2).sum(-1))
                    div.append(d[np.triu_indices(len(fk), 1)].mean())
            r["diversite"] = float(np.mean(div))
        return r

    reference_vraies = mesurer(X_train01[: len(X_test01) // 10])
    print(f"Plancher : {len(X_test01) // 10} vraies images d'entraînement contre le test : "
          f"Fréchet {reference_vraies['distance_frechet']:.2f}")
    return mesurer, reference_vraies


@app.cell
def _(alphas_barre, betas, config, torch):
    # --- C. Échantillonneurs (mêmes formules que la partie A, avec le réseau à la place du débruiteur exact) ---
    @torch.no_grad()
    def predire_eps(modele, x, t, y, guidage):
        # Guidage sans classifieur : ε = ε_sans + w · (ε_avec − ε_sans) ; w = 0 : sans condition, w = 1 : condition seule
        tt = torch.full((len(x),), int(t), device=x.device) if not torch.is_tensor(t) else t
        if guidage == 0.0:
            return modele(x, tt, torch.full_like(y, 10))
        if guidage == 1.0:
            return modele(x, tt, y)
        e = modele(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([y, torch.full_like(y, 10)]))
        e_avec, e_sans = e.chunk(2)
        return e_sans + guidage * (e_avec - e_sans)

    @torch.no_grad()
    def ech_ddpm(modele, bruit, y, guidage=0.0, n_pas=None, gen=None):
        # DDPM (Ho et al.) : x_{t−1} = (x_t − β_t/√(1−ᾱ_t) · ε̂) / √(1−β_t) + √β_t · z, sur les T pas
        x = bruit.clone()
        n_pas = n_pas or config["T"]
        for t in range(n_pas - 1, -1, -1):   # en smoke, n_pas < T : seulement la fin de la chaîne
            e = predire_eps(modele, x, t, y, guidage)
            x = (x - betas[t] / (1 - alphas_barre[t]).sqrt() * e) / (1 - betas[t]).sqrt()
            if t > 0:
                x = x + betas[t].sqrt() * torch.randn(x.shape, generator=gen).to(x.device)
        return x

    @torch.no_grad()
    def ech_ddim(modele, bruit, y, n_pas, guidage=0.0, garder=False):
        # DDIM (Song et al., η = 0) : estimer x0, puis le rebruiter au niveau du pas suivant, sans hasard
        ts = torch.linspace(config["T"] - 1, 0, n_pas).round().long().tolist()
        x, etapes = bruit.clone(), []
        for i, t in enumerate(ts):
            e = predire_eps(modele, x, t, y, guidage)
            x0 = ((x - (1 - alphas_barre[t]).sqrt() * e) / alphas_barre[t].sqrt()).clamp(-1, 1)
            # x0 écrêté dans [−1, 1] (les pixels n'en sortent pas) ; on recalcule le bruit cohérent
            # avec lui, sinon un bruit extrapolé par le guidage s'accumule d'un pas à l'autre
            e = (x - alphas_barre[t].sqrt() * x0) / (1 - alphas_barre[t]).sqrt()
            ab_suiv = alphas_barre[ts[i + 1]] if i + 1 < n_pas else torch.tensor(1.0, device=x.device)
            if garder:
                etapes.append((t, x.clone(), e.clone(), x0.clone()))
            x = ab_suiv.sqrt() * x0 + (1 - ab_suiv).sqrt() * e
        return (x, etapes) if garder else x

    @torch.no_grad()
    def ech_fm(modele, bruit, y, n_pas, guidage=0.0):
        # Flow matching : pas d'Euler de t = 1 (bruit) à t = 0 (image) le long de la vitesse prédite
        ts = torch.linspace(1, 0, n_pas + 1).tolist()
        x = bruit.clone()
        for i in range(n_pas):
            tt = torch.full((len(x),), ts[i], device=x.device)
            v = predire_eps(modele, x, tt, y, guidage)
            x = x - (ts[i] - ts[i + 1]) * v
        return x

    def vers_01(x):
        return ((x.clamp(-1, 1) + 1) / 2)

    return ech_ddim, ech_ddpm, ech_fm, predire_eps, vers_01


@app.cell
def _(config, ech_ddim, ech_ddpm, mesurer, time, torch, unet_ddpm, vers_01):
    # --- C. DDPM 1 000 pas contre DDIM, même réseau. On demande 100 images de chaque chiffre
    # (condition seule, w = 1) : la distribution visée est celle du jeu de test, équilibré.
    _N = config["n_ech"]
    _dev = next(unet_ddpm.parameters()).device
    bruit_depart = torch.randn(_N, 1, 28, 28, generator=torch.Generator().manual_seed(2024)).to(_dev)
    classes_voulues = torch.arange(10, device=_dev).repeat_interleave(_N // 10)
    # Grilles de 64 images : 6 ou 7 de chaque chiffre, dans l'ordre 0, 1, …, 9, 0, 1…
    ordre_grille = torch.tensor([(k % 10) * (_N // 10) + k // 10 for k in range(64)], device=_dev)
    resultats_c, grilles_c = {}, {}

    _t0 = time.perf_counter()
    _x = ech_ddpm(unet_ddpm, bruit_depart, classes_voulues, guidage=1.0, n_pas=config["pas_ddpm"],
                  gen=torch.Generator().manual_seed(1))
    _nom = f"DDPM {config['pas_ddpm']} pas"
    resultats_c[_nom] = {**mesurer(vers_01(_x), classes_voulues), "duree": time.perf_counter() - _t0,
                         "pas": config["pas_ddpm"]}
    grilles_c[_nom] = vers_01(_x[ordre_grille]).cpu().numpy()
    for _n in config["pas_ddim"]:
        _t0 = time.perf_counter()
        _x = ech_ddim(unet_ddpm, bruit_depart, classes_voulues, _n, guidage=1.0)
        _nom = f"DDIM {_n} pas"
        resultats_c[_nom] = {**mesurer(vers_01(_x), classes_voulues), "duree": time.perf_counter() - _t0, "pas": _n}
        grilles_c[_nom] = vers_01(_x[ordre_grille]).cpu().numpy()
    for _nom, _r in resultats_c.items():
        print(f"{_nom:14s} Fréchet {_r['distance_frechet']:7.2f} | classe obtenue {_r['classe_obtenue']:.1%} | "
              f"classe la plus rare {_r['part_min_classe']:.1%} | pixels gris {_r['pixels_gris']:.1%} | "
              f"{_r['duree']:.1f} s", flush=True)
    return bruit_depart, classes_voulues, grilles_c, ordre_grille, resultats_c


@app.cell
def _(bruit_depart, classes_voulues, config, ech_ddim, mesurer, torch, unet_ddpm, vers_01):
    # --- D. Guidage sans classifieur : 100 images par classe demandée, DDIM, poids w variable ---
    _N = config["n_ech"]
    _dev = bruit_depart.device
    resultats_d, grilles_d = {}, {}
    for _w in config["guidages"]:
        _x = ech_ddim(unet_ddpm, bruit_depart[: len(classes_voulues)], classes_voulues, config["pas_guidage"], guidage=_w)
        _r = mesurer(vers_01(_x), classes_voulues)
        resultats_d[_w] = _r
        # Grille : 8 exemples de chaque classe (une ligne par classe)
        _par_classe = _N // 10
        _idx = torch.cat([torch.arange(8) + k * _par_classe for k in range(10)])
        grilles_d[_w] = vers_01(_x[_idx.to(_dev)]).cpu().numpy()
        print(f"w = {_w:3.1f} : classe demandée obtenue {_r['classe_obtenue']:.1%} | diversité {_r['diversite']:.2f} | "
              f"Fréchet {_r['distance_frechet']:7.2f} | confiance {_r['confiance_moyenne']:.3f} | "
              f"pixels gris {_r['pixels_gris']:.1%}", flush=True)
    return grilles_d, resultats_d


@app.cell
def _(
    SMOKE,
    bruit_depart,
    checkpoints_dir,
    classes_voulues,
    config,
    ech_ddim,
    ech_fm,
    entrainer,
    mesurer,
    ordre_grille,
    resultats_c,
    time,
    torch,
    unet_ddpm,
    vers_01,
):
    # --- E. Flow matching : même U-Net, même nombre de pas d'entraînement, cible = vitesse ---
    unet_fm, histo_fm = entrainer("fm")
    if not SMOKE:
        torch.save(unet_fm.state_dict(), checkpoints_dir() / "25a-unet-fm.pt")
    resultats_e, grilles_e = {}, {}
    for _n in config["pas_fm"]:
        _t0 = time.perf_counter()
        _x = ech_fm(unet_fm, bruit_depart, classes_voulues, _n, guidage=1.0)
        resultats_e[f"FM {_n} pas"] = {**mesurer(vers_01(_x), classes_voulues), "duree": time.perf_counter() - _t0,
                                       "pas": _n}
        grilles_e[f"FM {_n} pas"] = vers_01(_x[ordre_grille]).cpu().numpy()
        # DDIM au même nombre de pas (même bruit de départ, mêmes classes), pour comparer à coût égal
        if f"DDIM {_n} pas" in resultats_c:
            resultats_e[f"DDIM {_n} pas"] = resultats_c[f"DDIM {_n} pas"]
        else:
            _x = ech_ddim(unet_ddpm, bruit_depart, classes_voulues, _n, guidage=1.0)
            resultats_e[f"DDIM {_n} pas"] = {**mesurer(vers_01(_x), classes_voulues), "pas": _n}
            grilles_e[f"DDIM {_n} pas"] = vers_01(_x[ordre_grille]).cpu().numpy()
    for _nom in sorted(resultats_e, key=lambda k: (-resultats_e[k]["pas"], k)):
        _r = resultats_e[_nom]
        print(f"{_nom:12s} Fréchet {_r['distance_frechet']:7.2f} | pixels gris {_r['pixels_gris']:.1%} | "
              f"classe obtenue {_r['classe_obtenue']:.1%}", flush=True)
    return grilles_e, histo_fm, resultats_e, unet_fm


@app.cell
def _(X_test, alphas_barre, bruit_depart, ech_ddim, np, predire_eps, torch, unet_ddpm, y_test):
    # --- Trajectoires pour la démo du site et les figures ---
    # 1. Processus avant : un « 3 » de test, un bruit fixe, et à chaque t ce que le réseau en déduit
    _i3 = int((y_test == 3).nonzero()[0])
    _x0 = X_test[_i3:_i3 + 1]
    _eps = torch.randn(_x0.shape, generator=torch.Generator().manual_seed(11)).to(_x0.device)
    instants_avant = [0, 25, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 999]
    avant = {"x_t": [], "eps_pred": [], "x0_pred": [], "erreur_eps": []}
    with torch.no_grad():
        for _t in instants_avant:
            _ab = alphas_barre[_t]
            _xt = _ab.sqrt() * _x0 + (1 - _ab).sqrt() * _eps
            # Le réseau reçoit l'image bruitée, l'instant t et la classe (3)
            _e = predire_eps(unet_ddpm, _xt, _t, torch.full((1,), 3, device=_x0.device), 1.0)
            _x0p = ((_xt - (1 - _ab).sqrt() * _e) / _ab.sqrt()).clamp(-1, 1)
            avant["x_t"].append(_xt[0, 0].cpu().numpy())
            avant["eps_pred"].append(_e[0, 0].cpu().numpy())
            avant["x0_pred"].append(_x0p[0, 0].cpu().numpy())
            avant["erreur_eps"].append(float((_e - _eps).pow(2).mean()))
    avant = {k: np.array(v) for k, v in avant.items()}
    avant["x0"] = _x0[0, 0].cpu().numpy()
    avant["eps"] = _eps[0, 0].cpu().numpy()
    # 2. Génération : DDIM 50 pas depuis 8 bruits fixes (chiffres demandés 0 à 7), en gardant chaque étape
    _, _etapes = ech_ddim(unet_ddpm, bruit_depart[:8], torch.arange(8, device=bruit_depart.device), 50,
                          guidage=1.0, garder=True)
    generation = {"t": np.array([e[0] for e in _etapes]), "classes": np.arange(8),
                  "x_t": np.stack([e[1][:, 0].cpu().numpy() for e in _etapes]),
                  "eps_pred": np.stack([e[2][:, 0].cpu().numpy() for e in _etapes]),
                  "x0_pred": np.stack([e[3][:, 0].cpu().numpy() for e in _etapes])}
    print(f"Trajectoires : avant {avant['x_t'].shape}, génération {generation['x_t'].shape}")
    print("Erreur de prédiction du bruit sur le « 3 », par t : " +
          ", ".join(f"{t}: {e:.3f}" for t, e in zip(instants_avant, avant["erreur_eps"])))
    return avant, generation, instants_avant


@app.cell
def _(
    MU,
    SMOKE,
    avant,
    chemins_ddim_jouet,
    chemins_fm_jouet,
    config,
    generation,
    grilles_c,
    grilles_d,
    grilles_e,
    histo_ddpm,
    histo_fm,
    jouet,
    mo,
    np,
    plt,
    resultats_c,
    resultats_d,
    resultats_e,
    sauvegarder_figure,
):
    # --- Figures du notebook (PNG) ---
    dossier = mo.notebook_dir() / "figures"
    suffixe = "-smoke" if SMOKE else ""

    def mosaique(images, lignes, colonnes):
        im = np.asarray(images).reshape(-1, 28, 28)[: lignes * colonnes]
        return im.reshape(lignes, colonnes, 28, 28).transpose(0, 2, 1, 3).reshape(lignes * 28, colonnes * 28)

    # 1. Jouet 2D : trajectoires DDIM et FM, qualité selon le nombre de pas
    fig_a, axes_a = plt.subplots(1, 3, figsize=(15, 4.6))
    for _ax, _ch, _titre in [(axes_a[0], chemins_ddim_jouet, "DDIM (débruiteur exact)"),
                             (axes_a[1], chemins_fm_jouet, "Flow matching (vitesse exacte)")]:
        for _k in range(_ch.shape[1]):
            _ax.plot(_ch[:, _k, 0], _ch[:, _k, 1], lw=0.8, alpha=0.7)
        _ax.scatter(MU[:, 0], MU[:, 1], c="k", s=25, marker="x")
        _ax.set(title=_titre, aspect="equal", xlim=(-3.5, 3.5), ylim=(-3.5, 3.5))
    _pas = config["pas_jouet"]
    axes_a[2].plot(_pas, [jouet["DDIM"][n]["part_dans_modes"] for n in _pas], "o-", label="DDIM")
    axes_a[2].plot(_pas, [jouet["FM"][n]["part_dans_modes"] for n in _pas], "s-", label="flow matching")
    axes_a[2].axhline(jouet["DDPM 1000 pas"]["part_dans_modes"], ls="--", c="gray", label="DDPM 1 000 pas")
    axes_a[2].set(xscale="log", xlabel="nombre de pas", ylabel="part des points à moins de 3σ d'un centre",
                  title="Qualité selon le nombre de pas")
    axes_a[2].legend()
    sauvegarder_figure(fig_a, "jouet-2d" + suffixe, dossier)

    # 2. Pertes : courbes d'entraînement et MSE de test selon t
    fig_b, axes_b = plt.subplots(1, 2, figsize=(12, 4))
    for _h, _nom in [(histo_ddpm, "diffusion (ε)"), (histo_fm, "flow matching (v)")]:
        _p = np.convolve(_h["pertes"], np.ones(50) / 50, mode="valid")
        axes_b[0].plot(_p, label=_nom)
        axes_b[1].plot([f for f, _ in _h["perte_par_t"]], [e for _, e in _h["perte_par_t"]], "o-", label=_nom)
    axes_b[0].set(xlabel="pas", ylabel="MSE (moyenne sur 50 pas)", title="Entraînement", yscale="log")
    axes_b[1].set(xlabel="t / T", ylabel="MSE de test", title="Erreur selon le niveau de bruit", yscale="log")
    axes_b[0].legend()
    axes_b[1].legend()
    sauvegarder_figure(fig_b, "pertes" + suffixe, dossier)

    # 3. Échantillons : DDPM, DDIM, FM
    _toutes = {**grilles_c, **{k: v for k, v in grilles_e.items() if k.startswith("FM")}}
    _res = {**resultats_c, **resultats_e}
    fig_c, axes_c = plt.subplots(1, len(_toutes), figsize=(2.6 * len(_toutes), 3.2))
    for _ax, (_nom, _g) in zip(axes_c, _toutes.items()):
        _ax.imshow(mosaique(_g, 6, 6), cmap="gray_r", vmin=0, vmax=1)
        _ax.set_title(f"{_nom}\nFréchet {_res[_nom]['distance_frechet']:.1f}", fontsize=10)
        _ax.axis("off")
    sauvegarder_figure(fig_c, "echantillons" + suffixe, dossier)

    # 4. Guidage
    fig_d, axes_d = plt.subplots(1, len(grilles_d), figsize=(3.4 * len(grilles_d), 4.6))
    for _ax, (_w, _g) in zip(axes_d, grilles_d.items()):
        _ax.imshow(mosaique(_g, 10, 8), cmap="gray_r", vmin=0, vmax=1)
        _r = resultats_d[_w]
        _ax.set_title(f"w = {_w:g}\nclasse obtenue {_r['classe_obtenue']:.0%}, diversité {_r['diversite']:.1f}", fontsize=10)
        _ax.axis("off")
    sauvegarder_figure(fig_d, "guidage" + suffixe, dossier)

    # 5. Trajectoires : processus avant et génération
    fig_t, axes_t = plt.subplots(3, 8, figsize=(12, 4.8))
    _ia = np.linspace(0, len(avant["x_t"]) - 1, 8).round().astype(int)
    _ig = np.linspace(0, len(generation["t"]) - 1, 8).round().astype(int)
    for _j in range(8):
        axes_t[0, _j].imshow(avant["x_t"][_ia[_j]], cmap="gray_r", vmin=-1, vmax=1)
        axes_t[1, _j].imshow(generation["x_t"][_ig[_j], 0], cmap="gray_r", vmin=-1, vmax=1)
        axes_t[2, _j].imshow(generation["x0_pred"][_ig[_j], 0], cmap="gray_r", vmin=-1, vmax=1)
        axes_t[1, _j].set_title(f"t = {generation['t'][_ig[_j]]}", fontsize=8)
        for _i in range(3):
            axes_t[_i, _j].axis("off")
    for _i, _l in enumerate(["bruitage", "génération x_t", "x̂0 prédit"]):
        axes_t[_i, 0].text(-4, 14, _l, ha="right", va="center", fontsize=9)
    sauvegarder_figure(fig_t, "trajectoires" + suffixe, dossier)
    plt.close("all")
    print(f"Figures : {dossier}")


@app.cell
def _(
    SMOKE,
    avant,
    config,
    generation,
    grilles_c,
    grilles_d,
    grilles_e,
    histo_ddpm,
    histo_fm,
    instants_avant,
    json,
    mo,
    np,
    precision_juge,
    reference_vraies,
    resultats_a,
    resultats_c,
    resultats_d,
    resultats_e,
    traj_ddpm_jouet,
    chemins_ddim_jouet,
    chemins_fm_jouet,
):
    # --- Résultats bruts pour les figures et la démo du site (run complet seulement) ---
    if not SMOKE:
        _dossier = mo.notebook_dir() / "figures"
        _sous = lambda p: [float(np.mean(p[i:i + 50])) for i in range(0, len(p), 50)]
        _json = {
            "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in config.items()},
            "partie_a": {**resultats_a, "jouet": {**resultats_a["jouet"],
                         "DDIM": {str(k): v for k, v in resultats_a["jouet"]["DDIM"].items()},
                         "FM": {str(k): v for k, v in resultats_a["jouet"]["FM"].items()}}},
            "precision_juge": precision_juge, "reference_vraies": reference_vraies,
            "partie_c": resultats_c, "partie_d": {str(k): v for k, v in resultats_d.items()},
            "partie_e": resultats_e,
            "pertes_ddpm_par_50": _sous(histo_ddpm["pertes"]), "pertes_fm_par_50": _sous(histo_fm["pertes"]),
            "perte_par_t_ddpm": histo_ddpm["perte_par_t"], "perte_par_t_fm": histo_fm["perte_par_t"],
            "duree_ddpm": histo_ddpm["duree"], "duree_fm": histo_fm["duree"],
            "instants_avant": instants_avant, "erreur_eps_avant": avant["erreur_eps"].tolist(),
            "t_generation": generation["t"].tolist(),
        }
        (_dossier / "resultats.json").write_text(json.dumps(_json, ensure_ascii=False, indent=1))

        def _u8(a, bas=-1.0, haut=1.0):
            return np.round(np.clip((np.asarray(a) - bas) / (haut - bas), 0, 1) * 255).astype(np.uint8)

        np.savez_compressed(
            _dossier / "images.npz",
            avant_x0=_u8(avant["x0"]), avant_eps=_u8(avant["eps"], -3, 3),
            avant_x_t=_u8(avant["x_t"], -3, 3), avant_eps_pred=_u8(avant["eps_pred"], -3, 3),
            avant_x0_pred=_u8(avant["x0_pred"]),
            # Génération : trois des huit trajectoires seulement (chiffres demandés 0, 3 et 5)
            gen_classes=np.array([0, 3, 5]),
            gen_x_t=_u8(generation["x_t"][:, [0, 3, 5]], -3, 3),
            gen_x0_pred=_u8(generation["x0_pred"][:, [0, 3, 5]]),
            **{f"c_{k.replace(' ', '_')}": _u8(v, 0, 1) for k, v in grilles_c.items()},
            **{f"e_{k.replace(' ', '_')}": _u8(v, 0, 1) for k, v in grilles_e.items()},
            **{f"d_{w:g}": _u8(v, 0, 1) for w, v in grilles_d.items()},
            jouet_ddim=chemins_ddim_jouet.astype(np.float32), jouet_fm=chemins_fm_jouet.astype(np.float32),
            **{f"jouet_ddpm_{t}": v.astype(np.float32) for t, v in traj_ddpm_jouet.items()},
        )
        print(f"Résultats bruts : {_dossier / 'resultats.json'}, {_dossier / 'images.npz'}")


@app.cell
def _(
    SMOKE,
    config,
    duree_juge,
    histo_ddpm,
    histo_fm,
    mo,
    precision_juge,
    reference_vraies,
    resultats_a,
    resultats_c,
    resultats_d,
    resultats_e,
):
    # --- Bilan : seuils du README. En smoke, on vérifie seulement que tout tourne. ---
    _j = resultats_a["jouet"]
    _c, _d, _e = resultats_c, resultats_d, resultats_e
    _ddpm = _c.get(f"DDPM {config['pas_ddpm']} pas", {})
    seuils = {
        "A. formule fermée = 1 000 étapes (pente et variance à 0,01 près)": all(
            abs(v["pente"] - v["pente_formule"]) < 0.01 and abs(v["var_residu"] - v["var_formule"]) < 0.01
            for v in resultats_a["ferme_contre_iteratif"].values()),
        "A. jouet : DDPM 1 000 pas exact ≥ 95 % des points dans les modes": _j["DDPM 1000 pas"]["part_dans_modes"] >= 0.95,
        "A. jouet : FM 10 pas ≥ 90 % et DDIM 10 pas ≤ 85 %": (
            _j["FM"][10]["part_dans_modes"] >= 0.90 and _j["DDIM"][10]["part_dans_modes"] <= 0.85),
        "C. juge : précision de test ≥ 98,5 %": precision_juge >= 0.985,
        "C. DDPM 1 000 pas : Fréchet ≤ 15": _ddpm.get("distance_frechet", 1e9) <= 15,
        "C. DDIM 50 pas : Fréchet ≤ 30 (20 fois moins de pas)": _c.get("DDIM 50 pas", {}).get("distance_frechet", 1e9) <= 30,
        "C. DDIM 10 pas : Fréchet > DDIM 50 pas": (
            _c.get("DDIM 10 pas", {}).get("distance_frechet", 0) > _c.get("DDIM 50 pas", {}).get("distance_frechet", 1e9)),
        "D. guidage w = 1 : classe obtenue ≥ 80 %, w = 3 : ≥ 97 %": (
            _d[1.0]["classe_obtenue"] >= 0.80 and _d[3.0]["classe_obtenue"] >= 0.97),
        "D. guidage : diversité w = 7 < w = 3 < w = 1": _d[7.0]["diversite"] < _d[3.0]["diversite"] < _d[1.0]["diversite"],
        "D. guidage : Fréchet w = 7 > w = 1 (images moins variées que les vraies)": (
            _d[7.0]["distance_frechet"] > _d[1.0]["distance_frechet"]),
        "E. FM 10 pas : Fréchet ≤ 20 et < DDIM 10 pas": (
            _e.get("FM 10 pas", {}).get("distance_frechet", 1e9) <= 20
            and _e.get("FM 10 pas", {}).get("distance_frechet", 1e9) < _e.get("DDIM 10 pas", {}).get("distance_frechet", 0)),
        "E. FM 5 pas : Fréchet < DDIM 5 pas": (
            _e.get("FM 5 pas", {}).get("distance_frechet", 1e9) < _e.get("DDIM 5 pas", {}).get("distance_frechet", 0)),
    }
    _lignes = "\n".join(f"- {'✅' if ok else '❌'} {nom}" for nom, ok in seuils.items())
    _echecs = [nom for nom, ok in seuils.items() if not ok]
    print(_lignes)
    _total = histo_ddpm["duree"] + histo_fm["duree"] + duree_juge
    print(f"Entraînements : diffusion {histo_ddpm['duree']:.0f} s, flow matching {histo_fm['duree']:.0f} s, "
          f"juge {duree_juge:.0f} s ; plancher Fréchet (vraies images) {reference_vraies['distance_frechet']:.2f}")
    if SMOKE:
        print("(mode smoke : seuils non vérifiés, des ❌ sont attendus)")
    if not SMOKE and _echecs:
        raise SystemExit("Seuils non atteints :\n" + "\n".join(_echecs))
    _note = "\n\n*Mode smoke : seuils non vérifiés, des ❌ sont attendus.*" if SMOKE else ""
    mo.md(f"**Bilan** ({_total:.0f} s d'entraînement)\n\n{_lignes}{_note}")


if __name__ == "__main__":
    app.run()
