# Exercice 18 : modèles génératifs I sur MNIST (autoencodeur, autoencodeur débruitant,
# VAE, DCGAN), comparés avec un même juge (un petit classifieur).
#
#   make ex NUM=18              → éditeur marimo (bouton « Entraîner »)
#   make ex NUM=18 MODE=run     → script complet, vérifie les seuils du README
#   make ex NUM=18 MODE=smoke   → test rapide (< 30 s)
#
# Partie A en NumPy (autoencodeur linéaire = ACP, KL de deux gaussiennes), parties B à E
# en PyTorch sur MNIST (GPU Apple via MPS si disponible).

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
    from scipy import linalg
    from torch import nn

    from ia_commun import get_device, is_smoke, sauvegarder_figure, seed_everything, style_commun
    from ia_commun.donnees import mnist_tenseurs

    return (
        F,
        get_device,
        is_smoke,
        json,
        linalg,
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
    # Exercice 18 : autoencodeurs, VAE et GAN sur MNIST

    Un **modèle génératif** apprend la distribution des images pour en fabriquer de
    nouvelles. On construit trois familles du chapitre 18, puis on les juge toutes avec le
    même outil.

    - **A. NumPy.** Un autoencodeur **linéaire** à 2 neurones centraux, entraîné par descente
      de gradient (Adam recodé), retrouve le plan de l'ACP. On vérifie aussi la formule
      fermée de la KL entre deux gaussiennes, utilisée par le VAE.
    - **B. Autoencodeur débruitant.** Un même réseau convolutif *sur-complet* (plus de
      neurones au centre que de pixels), entraîné à recopier l'image propre, ou à
      reconstruire l'image propre depuis une version bruitée.
    - **C. Autoencodeur contre VAE, espace latent en 2D.** Même architecture ; le VAE ajoute
      le bruit de la reparamétrisation et le terme KL. Un VAE à 16 dimensions sert de point
      de comparaison pour la qualité.
    - **D. DCGAN réduit.** Un générateur convolutif contre un discriminateur.
    - **E. Le juge.** Un petit CNN entraîné sur MNIST classe 10 000 images de chaque
      modèle : confiance moyenne, classes couvertes, et distance de Fréchet entre les
      caractéristiques des vraies images et celles des images générées (même principe
      que la FID, avec ce CNN à la place d'Inception).

    **Pertes optimisées.**
    - Autoencodeur linéaire (A) : erreur quadratique moyenne de reconstruction
      $\frac1N\sum_i \lVert x_i - \hat x_i\rVert^2$.
    - Autoencodeurs B et C : entropie croisée binaire par pixel entre l'image cible et la
      reconstruction (vraisemblance de Bernoulli), la même que le terme de reconstruction du VAE.
    - VAE : l'opposé de l'ELBO (chapitre 09),
      $-\mathbb E_{q(z\mid x)}[\log p(x\mid z)] + D_{\mathrm{KL}}(q(z\mid x)\,\|\,\mathcal N(0, I))$,
      avec une vraisemblance de Bernoulli par pixel (entropie croisée binaire, sommée sur
      les 784 pixels) et la KL en formule fermée.
    - GAN : entropie croisée binaire du discriminateur $D$ (vraies images → 1, fausses → 0) ;
      le générateur minimise $-\log D(G(z))$ (version « non saturante » de Goodfellow et al.).
    - Juge : entropie croisée.
    """)


@app.cell
def _(get_device, is_smoke, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    device = get_device()
    style_commun()
    config = dict(
        n_train=3000 if SMOKE else 60000,   # images d'entraînement (MNIST complet en run)
        n_test=1000 if SMOKE else 10000,
        lot=128,
        # A. autoencodeur linéaire NumPy
        n_acp=2000 if SMOKE else 10000, pas_a=300 if SMOKE else 3000, lot_a=500, lr_a=3e-3,
        # B. autoencodeur débruitant
        sigma_bruit=0.5, epoques_b=1 if SMOKE else 3,
        # C. autoencodeur et VAE (MLP 784-512-256-d)
        epoques_c=1 if SMOKE else 20, lr_c=1e-3, latents_vae=[2, 16],
        # D. DCGAN (hyperparamètres de Radford et al. 2015)
        z_gan=64, epoques_d=1 if SMOKE else 6, lr_d=2e-4, beta1=0.5,
        fenetre_gan=5 if SMOKE else 50,     # pas par point des courbes du GAN (moyenne glissante)
        # E. juge
        epoques_juge=1 if SMOKE else 3,
        n_echantillons=1000 if SMOKE else 10000,
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, périphérique {device}")
    print(config)
    return SEED, SMOKE, config, device


@app.cell
def _(config, device, mnist_tenseurs, torch):
    # --- Données : MNIST (cache data/mnist/), pixels dans [0, 1] ---
    _Xtr, _ytr = mnist_tenseurs(train=True)
    _Xte, _yte = mnist_tenseurs(train=False)
    _perm = torch.randperm(60000, generator=torch.Generator().manual_seed(42))[: config["n_train"]]
    X_train, y_train = _Xtr[_perm].to(device), _ytr[_perm].to(device)
    X_test, y_test = _Xte[: config["n_test"]].to(device), _yte[: config["n_test"]].to(device)
    print(f"train {tuple(X_train.shape)}, test {tuple(X_test.shape)}")
    return X_test, X_train, y_test, y_train


@app.cell
def _(mo):
    mo.md(r"""
    ## A. Un autoencodeur linéaire retrouve l'ACP (NumPy)

    Encodeur $z = W_e^\top x$ ($784 \to 2$), décodeur $\hat x = W_d^\top z$ ($2 \to 784$),
    sans activation, sur des images centrées. On minimise l'erreur quadratique par mini-lots
    avec Adam recodé à la main. Théorie (Baldi et Hornik, 1989) : à l'optimum, le décodeur
    engendre le même plan que les deux premières composantes de l'ACP, donc la même erreur.
    On mesure les deux, et le cosinus des **angles principaux** entre les deux plans
    (1 = plans confondus).
    """)


@app.cell
def _(SEED, X_train, config, np):
    _rng = np.random.default_rng(SEED)
    Xa = X_train[: config["n_acp"]].reshape(-1, 784).cpu().numpy().astype(np.float64)
    moyenne_a = Xa.mean(0)
    Xc = Xa - moyenne_a

    # --- ACP par décomposition en valeurs singulières ---
    _U, _S, _Vt = np.linalg.svd(Xc, full_matrices=False)
    composantes = _Vt[:2]                                    # (2, 784)
    mse_acp = float(((Xc - Xc @ composantes.T @ composantes) ** 2).mean())

    # --- Autoencodeur linéaire + Adam en NumPy ---
    _params = {"We": _rng.normal(0, 0.01, (784, 2)), "Wd": _rng.normal(0, 0.01, (2, 784))}
    _m = {k: np.zeros_like(p) for k, p in _params.items()}
    _v = {k: np.zeros_like(p) for k, p in _params.items()}
    _b1, _b2, _eps = 0.9, 0.999, 1e-8
    _pertes_a = []
    for _t in range(1, config["pas_a"] + 1):
        _idx = _rng.integers(0, len(Xc), config["lot_a"])
        _x = Xc[_idx]
        _z = _x @ _params["We"]                                 # (lot, 2)
        _xh = _z @ _params["Wd"]                                # (lot, 784)
        _err = _xh - _x
        # Perte : moyenne sur le lot ET sur les pixels ; gradients par la règle de la chaîne
        _g_xh = 2 * _err / _err.size
        _grads = {"Wd": _z.T @ _g_xh, "We": _x.T @ (_g_xh @ _params["Wd"].T)}
        for _k in _params:
            _m[_k] = _b1 * _m[_k] + (1 - _b1) * _grads[_k]
            _v[_k] = _b2 * _v[_k] + (1 - _b2) * _grads[_k] ** 2
            _params[_k] -= config["lr_a"] * (_m[_k] / (1 - _b1**_t)) / (np.sqrt(_v[_k] / (1 - _b2**_t)) + _eps)
        if _t % 50 == 0 or _t == 1:
            _pertes_a.append((_t, float((_err**2).mean())))
    # Erreur finale sur toutes les images : on reprojette au mieux sur le plan du décodeur
    _zall = Xc @ _params["We"]
    mse_ae_lineaire = float(((Xc - _zall @ _params["Wd"]) ** 2).mean())
    # Angles principaux entre le plan du décodeur et celui de l'ACP
    _Q, _ = np.linalg.qr(_params["Wd"].T)                     # base orthonormée (784, 2)
    cos_angles = np.linalg.svd(_Q.T @ composantes.T, compute_uv=False)
    resultats_a = {
        "mse_acp_2": mse_acp,
        "mse_ae_lineaire_2": mse_ae_lineaire,
        "rapport": mse_ae_lineaire / mse_acp,
        "cos_angles_principaux": cos_angles.tolist(),
        "variance_expliquee_2": float((_S[:2] ** 2).sum() / (_S**2).sum()),
        "pertes": _pertes_a,
    }
    print(f"ACP 2 composantes : MSE {mse_acp:.5f} ({resultats_a['variance_expliquee_2']:.1%} de variance)")
    print(f"AE linéaire 2 neurones : MSE {mse_ae_lineaire:.5f} (rapport {resultats_a['rapport']:.4f}), "
          f"cos des angles principaux {np.round(cos_angles, 4).tolist()}")
    return moyenne_a, resultats_a


@app.cell
def _(SEED, np):
    # --- KL entre q = N(μ, σ²) et p = N(0, 1) : formule fermée contre Monte-Carlo ---
    def kl_gauss(mu, logvar):
        """KL(N(μ, σ²) ‖ N(0, 1)) = ½ (μ² + σ² − log σ² − 1), par dimension."""
        return 0.5 * (mu**2 + np.exp(logvar) - logvar - 1)

    _rng = np.random.default_rng(SEED)
    verif_kl = []
    for _mu, _sigma in [(0.0, 1.0), (1.0, 1.0), (0.0, 0.5), (2.0, 0.3)]:
        _z = _mu + _sigma * _rng.standard_normal(1_000_000)
        # log q(z) − log p(z), moyenné sur des tirages de q
        _mc = np.mean(-np.log(_sigma) - (_z - _mu) ** 2 / (2 * _sigma**2) + _z**2 / 2)
        verif_kl.append({"mu": _mu, "sigma": _sigma,
                         "fermee": float(kl_gauss(_mu, 2 * np.log(_sigma))), "monte_carlo": float(_mc)})
        print(f"μ = {_mu}, σ = {_sigma} : formule {verif_kl[-1]['fermee']:.4f} nat, "
              f"Monte-Carlo (10⁶ tirages) {_mc:.4f} nat")
    return (verif_kl,)


@app.cell
def _(nn, torch):
    # --- Modèles ---
    class AEConv(nn.Module):
        """Autoencodeur convolutif SUR-COMPLET : 784 pixels → 64 × 7 × 7 = 3 136 valeurs → 784."""

        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),     # 14 × 14
                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU())     # 7 × 7
            self.dec = nn.Sequential(
                nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), nn.ReLU(),   # 14 × 14
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),   # 28 × 28
                nn.Conv2d(32, 1, 3, padding=1))                        # logits des pixels

        def forward(self, x):
            return self.dec(self.enc(x))                              # logits ; sigmoïde → image

    class AEMLP(nn.Module):
        """Encodeur 784-512-256-d, décodeur d-256-512-784. Si variationnel, l'encodeur
        sort μ et log σ² (2d valeurs) et on tire z = μ + σ·ε (reparamétrisation)."""

        def __init__(self, d, variationnel):
            super().__init__()
            self.d, self.variationnel = d, variationnel
            self.enc = nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.ReLU(),
                                     nn.Linear(512, 256), nn.ReLU(),
                                     nn.Linear(256, 2 * d if variationnel else d))
            self.dec = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(),
                                     nn.Linear(512, 784))           # logits des 784 pixels

        def encoder(self, x):
            h = self.enc(x)
            if self.variationnel:
                return h[:, : self.d], h[:, self.d:]                 # μ, log σ²
            return h, None

        def decoder(self, z):
            return self.dec(z).view(-1, 1, 28, 28)                   # logits

        def forward(self, x):
            mu, logvar = self.encoder(x)
            z = mu + (0.5 * logvar).exp() * torch.randn_like(mu) if self.variationnel else mu
            return self.decoder(z), mu, logvar

    class Generateur(nn.Module):
        """DCGAN réduit : z (64) → 128 × 7 × 7 → 64 × 14 × 14 → 1 × 28 × 28 (tanh)."""

        def __init__(self, dz):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(dz, 128 * 7 * 7, bias=False), nn.BatchNorm1d(128 * 7 * 7), nn.ReLU())
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
                nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1), nn.Tanh())

        def forward(self, z):
            return self.conv(self.fc(z).view(-1, 128, 7, 7))

    class Discriminateur(nn.Module):
        """Convolutions à pas 2, LeakyReLU 0,2, BatchNorm sauf en entrée ; sortie : un logit."""

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),                       # 14
                nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),  # 7
                nn.Flatten(), nn.Linear(128 * 7 * 7, 1))

        def forward(self, x):
            return self.net(x)

    class Juge(nn.Module):
        """Petit CNN classifieur ; `caracteristiques` = avant-dernière couche (128 valeurs)."""

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

    return AEConv, AEMLP, Discriminateur, Generateur, Juge


@app.cell
def _(mo):
    # Le bouton empêche l'entraînement de se relancer à chaque modification d'une cellule amont.
    bouton = mo.ui.run_button(label="Entraîner")
    bouton
    return (bouton,)


@app.cell
def _(AEConv, F, X_test, X_train, bouton, config, mo, seed_everything, time, torch):
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Entraîner » pour lancer les parties B à E.*"))

    # --- B. Autoencodeur classique contre autoencodeur débruitant (même réseau) ---
    def bruiter(x, gen):
        # Bruit gaussien σ = 0,5 puis écrêtage dans [0, 1]
        b = torch.randn(x.shape, generator=gen).to(x.device)
        return (x + config["sigma_bruit"] * b).clamp(0, 1)

    def entrainer_ae_conv(debruitant):
        seed_everything(42)
        modele = AEConv().to(X_train.device)
        optim = torch.optim.Adam(modele.parameters(), lr=1e-3)
        gen = torch.Generator().manual_seed(42)
        n = len(X_train)
        for _ep in range(config["epoques_b"]):
            ordre = torch.randperm(n, generator=gen).to(X_train.device)
            for i in range(0, n, config["lot"]):
                x = X_train[ordre[i:i + config["lot"]]]
                entree = bruiter(x, gen) if debruitant else x
                # Entropie croisée binaire par pixel (vraisemblance de Bernoulli). Avec une MSE
                # après sigmoïde, ce réseau s'effondre sur une image presque noire (essayé :
                # MSE bloquée vers 0,11 après 300 pas, contre 0,002 avec la BCE).
                perte = F.binary_cross_entropy_with_logits(modele(entree), x)
                optim.zero_grad()
                perte.backward()
                optim.step()
        return modele.eval()

    _t0 = time.perf_counter()
    ae_classique = entrainer_ae_conv(False)
    ae_debruitant = entrainer_ae_conv(True)
    duree_b = time.perf_counter() - _t0

    @torch.no_grad()
    def psnr(a, b):
        # Rapport signal sur bruit de crête, en dB, pour des pixels dans [0, 1]
        return float(10 * torch.log10(1 / ((a - b) ** 2).mean()))

    _gen_test = torch.Generator().manual_seed(7)
    X_test_bruite = bruiter(X_test, _gen_test)
    with torch.no_grad():
        _sorties = {"bruitée": X_test_bruite,
                    "AE classique (propre → propre)": torch.sigmoid(ae_classique(X_test_bruite)),
                    "AE débruitant (bruitée → propre)": torch.sigmoid(ae_debruitant(X_test_bruite))}
        resultats_b = {k: {"mse": float(((s - X_test) ** 2).mean()), "psnr": psnr(s, X_test)}
                       for k, s in _sorties.items()}
        # Sur des images PROPRES, l'AE classique recopie presque parfaitement
        _propre = torch.sigmoid(ae_classique(X_test))
        resultats_b["AE classique sur image propre"] = {
            "mse": float(((_propre - X_test) ** 2).mean()), "psnr": psnr(_propre, X_test)}
        exemples_b = {"propre": X_test[:8].cpu().numpy(), "bruitee": X_test_bruite[:8].cpu().numpy(),
                      "ae": _sorties["AE classique (propre → propre)"][:8].cpu().numpy(),
                      "dae": _sorties["AE débruitant (bruitée → propre)"][:8].cpu().numpy()}
    for _k, _r in resultats_b.items():
        print(f"{_k:36s} MSE {_r['mse']:.4f}  PSNR {_r['psnr']:.1f} dB")
    print(f"Partie B : {duree_b:.1f} s", flush=True)
    return duree_b, exemples_b, resultats_b


@app.cell
def _(AEMLP, F, X_test, X_train, config, seed_everything, time, torch, y_test):
    # --- C. Autoencodeur (MLP, latent 2D) contre VAE (latent 2D et 16D) ---
    def perte_vae(logits, x, mu, logvar):
        """Opposé de l'ELBO par image : reconstruction (BCE sommée sur les pixels) + KL fermée."""
        rec = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / len(x)
        kl = (0.5 * (mu**2 + logvar.exp() - logvar - 1)).sum() / len(x)
        return rec + kl, rec, kl

    def entrainer_mlp(d, variationnel):
        seed_everything(42)
        modele = AEMLP(d, variationnel).to(X_train.device)
        optim = torch.optim.Adam(modele.parameters(), lr=config["lr_c"])
        gen = torch.Generator().manual_seed(42)
        n = len(X_train)
        histo = []
        for _ep in range(config["epoques_c"]):
            ordre = torch.randperm(n, generator=gen).to(X_train.device)
            somme = 0.0
            for i in range(0, n, config["lot"]):
                x = X_train[ordre[i:i + config["lot"]]]
                logits, mu, logvar = modele(x)
                if variationnel:
                    perte = perte_vae(logits, x, mu, logvar)[0]
                else:
                    # Même vraisemblance de Bernoulli, sans KL ni bruit : un autoencodeur classique
                    perte = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / len(x)
                optim.zero_grad()
                perte.backward()
                optim.step()
                somme += perte.item() * len(x)
            histo.append(somme / n)
        return modele.eval(), histo

    modeles_c, histos_c, resultats_c = {}, {}, {}
    _t0 = time.perf_counter()
    for _nom, _d, _var in [("AE 2D", 2, False), ("VAE 2D", 2, True), ("VAE 16D", 16, True)]:
        modeles_c[_nom], histos_c[_nom] = entrainer_mlp(_d, _var)
        _m = modeles_c[_nom]
        with torch.no_grad():
            _mu, _logvar = _m.encoder(X_test)
            _logits = _m.decoder(_mu)                      # reconstruction depuis la moyenne
            _r = {"mse_reconstruction": float(((torch.sigmoid(_logits) - X_test) ** 2).mean()),
                  "bce_par_image": float(F.binary_cross_entropy_with_logits(_logits, X_test, reduction="sum") / len(X_test))}
            if _var:
                # ELBO de test avec un tirage de z par image (estimation sans biais)
                _lg, _mu2, _lv2 = _m(X_test)
                _tot, _rec, _kl = perte_vae(_lg, X_test, _mu2, _lv2)
                _r.update({"moins_elbo": float(_tot), "reconstruction": float(_rec), "kl": float(_kl),
                           "kl_par_dimension": (0.5 * (_mu2**2 + _lv2.exp() - _lv2 - 1)).mean(0).tolist()})
            _r["codes_test"] = _mu.cpu().numpy()
            _r["etiquettes_test"] = y_test.cpu().numpy()
        resultats_c[_nom] = _r
        _txt = (f" | −ELBO {_r['moins_elbo']:.1f} nat = reconstruction {_r['reconstruction']:.1f} + KL {_r['kl']:.2f}"
                if _var else "")
        print(f"{_nom:8s} MSE {_r['mse_reconstruction']:.4f}, BCE {_r['bce_par_image']:.1f} nat/image{_txt}", flush=True)
    duree_c = time.perf_counter() - _t0
    print(f"Partie C : {duree_c:.1f} s")
    return duree_c, histos_c, modeles_c, resultats_c


@app.cell
def _(
    Discriminateur,
    F,
    Generateur,
    X_train,
    config,
    nn,
    seed_everything,
    time,
    torch,
):
    # --- D. DCGAN réduit ---
    def init_dcgan(m):
        # Initialisation de Radford et al. : loi normale de moyenne 0 et d'écart-type 0,02
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)

    seed_everything(42)
    _dev = X_train.device
    G = Generateur(config["z_gan"]).to(_dev)
    D = Discriminateur().to(_dev)
    G.apply(init_dcgan)
    D.apply(init_dcgan)
    _optG = torch.optim.Adam(G.parameters(), lr=config["lr_d"], betas=(config["beta1"], 0.999))
    _optD = torch.optim.Adam(D.parameters(), lr=config["lr_d"], betas=(config["beta1"], 0.999))
    _gen = torch.Generator().manual_seed(42)
    z_fixe = torch.randn(64, config["z_gan"], generator=torch.Generator().manual_seed(123)).to(_dev)
    _Xg = X_train * 2 - 1                                    # images dans [−1, 1] (sortie tanh)
    _n = len(_Xg)
    suivi_gan = {"pas": [], "perte_D": [], "perte_G": [], "D_vraies": [], "D_fausses": []}
    instantanes_gan = {}
    _acc, _k, _pas = {k: 0.0 for k in ("perte_D", "perte_G", "D_vraies", "D_fausses")}, 0, 0
    _t0 = time.perf_counter()
    for _ep in range(config["epoques_d"]):
        _ordre = torch.randperm(_n, generator=_gen).to(_dev)
        for _i in range(0, _n - config["lot"] + 1, config["lot"]):
            _x = _Xg[_ordre[_i:_i + config["lot"]]]
            _z = torch.randn(len(_x), config["z_gan"], generator=_gen).to(_dev)
            _fausses = G(_z)
            # 1) Discriminateur : vraies → 1, fausses → 0 (fausses détachées : G ne bouge pas)
            _lv, _lf = D(_x), D(_fausses.detach())
            _pD = (F.binary_cross_entropy_with_logits(_lv, torch.ones_like(_lv))
                   + F.binary_cross_entropy_with_logits(_lf, torch.zeros_like(_lf)))
            _optD.zero_grad()
            _pD.backward()
            _optD.step()
            # 2) Générateur : perte non saturante −log D(G(z)), c'est-à-dire « fausses → 1 »
            _lg = D(_fausses)
            _pG = F.binary_cross_entropy_with_logits(_lg, torch.ones_like(_lg))
            _optG.zero_grad()
            _pG.backward()
            _optG.step()
            _pas += 1
            for _cle, _val in (("perte_D", _pD), ("perte_G", _pG), ("D_vraies", torch.sigmoid(_lv).mean()),
                               ("D_fausses", torch.sigmoid(_lf).mean())):
                _acc[_cle] += _val.item()
            _k += 1
            if _k == config["fenetre_gan"]:                  # moyenne sur les derniers pas
                suivi_gan["pas"].append(_pas)
                for _cle, _somme in _acc.items():
                    suivi_gan[_cle].append(_somme / _k)
                _acc = dict.fromkeys(_acc, 0.0)
                _k = 0
        G.eval()
        with torch.no_grad():
            instantanes_gan[_ep + 1] = ((G(z_fixe) + 1) / 2).cpu().numpy()
        G.train()
        _dern = {c: suivi_gan[c][-1] if suivi_gan["pas"] else float("nan") for c in _acc}
        print(f"époque {_ep + 1} : perte D {_dern['perte_D']:.3f}, perte G {_dern['perte_G']:.3f}, "
              f"D(vraies) {_dern['D_vraies']:.2f}, D(fausses) {_dern['D_fausses']:.2f} "
              f"(moyennes des {config['fenetre_gan']} derniers pas)", flush=True)
    G.eval()
    duree_d = time.perf_counter() - _t0
    print(f"Partie D : {duree_d:.1f} s ({_pas} pas)")
    return G, duree_d, instantanes_gan, suivi_gan, z_fixe


@app.cell
def _(F, Juge, X_test, X_train, config, seed_everything, time, torch, y_test, y_train):
    # --- E. Le juge : un petit CNN classifieur entraîné sur les vraies images ---
    seed_everything(42)
    juge = Juge().to(X_train.device)
    _opt = torch.optim.Adam(juge.parameters(), lr=1e-3)
    _gen = torch.Generator().manual_seed(42)
    _t0 = time.perf_counter()
    for _ep in range(config["epoques_juge"]):
        _ordre = torch.randperm(len(X_train), generator=_gen).to(X_train.device)
        for _i in range(0, len(X_train), config["lot"]):
            _idx = _ordre[_i:_i + config["lot"]]
            _p = F.cross_entropy(juge(X_train[_idx]), y_train[_idx])
            _opt.zero_grad()
            _p.backward()
            _opt.step()
    juge.eval()
    with torch.no_grad():
        precision_juge = float((juge(X_test).argmax(1) == y_test).float().mean())
    duree_juge = time.perf_counter() - _t0
    print(f"Juge : exactitude de test {precision_juge:.2%} ({duree_juge:.1f} s)")
    return duree_juge, juge, precision_juge


@app.cell
def _(G, X_test, X_train, config, juge, linalg, modeles_c, np, resultats_c, torch):
    # --- E (suite). Échantillons de chaque modèle, jugés de la même façon ---
    _N = config["n_echantillons"]
    _gen = torch.Generator().manual_seed(2024)
    _dev = X_train.device

    @torch.no_grad()
    def echantillons_mlp(nom):
        m = modeles_c[nom]
        if m.variationnel:
            z = torch.randn(_N, m.d, generator=_gen)               # loi a priori N(0, I)
        else:
            # Pas de loi a priori pour un AE : on tire dans une gaussienne ajustée sur ses codes
            codes = torch.from_numpy(resultats_c[nom]["codes_test"]).double()
            cov = torch.cov(codes.T)
            L = torch.linalg.cholesky(cov)
            z = (codes.mean(0) + torch.randn(_N, m.d, generator=_gen).double() @ L.T).float()
        return torch.sigmoid(m.decoder(z.to(_dev)))

    @torch.no_grad()
    def echantillons_gan():
        sorties = [G(torch.randn(1000, config["z_gan"], generator=_gen).to(_dev)) for _ in range(_N // 1000 or 1)]
        return ((torch.cat(sorties)[:_N] + 1) / 2).clamp(0, 1)

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
        # + 10⁻⁶·I : des neurones ReLU toujours nuls rendent les covariances singulières
        _eps = 1e-6 * np.eye(f1.shape[1])
        c1, c2 = np.cov(f1, rowvar=False) + _eps, np.cov(f2, rowvar=False) + _eps
        racine = linalg.sqrtm(c1 @ c2)
        racine = racine.real if np.iscomplexobj(racine) else racine
        return float(((m1 - m2) ** 2).sum() + np.trace(c1 + c2 - 2 * racine))

    _f_test, _ = statistiques(X_test)
    sources = {
        "vraies (train)": X_train[:_N],
        "AE 2D": echantillons_mlp("AE 2D"),
        "VAE 2D": echantillons_mlp("VAE 2D"),
        "VAE 16D": echantillons_mlp("VAE 16D"),
        "DCGAN": echantillons_gan(),
    }
    resultats_e, grilles_e = {}, {}
    for _nom, _img in sources.items():
        _f, _p = statistiques(_img)
        _classes = np.bincount(_p.argmax(1), minlength=10) / len(_p)
        resultats_e[_nom] = {
            "confiance_moyenne": float(_p.max(1).mean()),
            "part_confiance_sup_0_9": float((_p.max(1) > 0.9).mean()),
            "parts_classes": _classes.tolist(),
            "part_min_classe": float(_classes.min()),
            "classes_couvertes_1pc": int((_classes >= 0.01).sum()),
            "distance_frechet": frechet(_f, _f_test),
            # « Netteté » : part des pixels ni noirs ni blancs (entre 0,1 et 0,9)
            "pixels_gris": float(((_img > 0.1) & (_img < 0.9)).float().mean()),
        }
        grilles_e[_nom] = _img[:64].cpu().numpy()
        _r = resultats_e[_nom]
        print(f"{_nom:15s} confiance {_r['confiance_moyenne']:.3f} (> 0,9 : {_r['part_confiance_sup_0_9']:.1%}) | "
              f"classe la plus rare {_r['part_min_classe']:.1%} | Fréchet {_r['distance_frechet']:7.2f} | "
              f"pixels gris {_r['pixels_gris']:.1%}", flush=True)
    return grilles_e, resultats_e


@app.cell
def _(X_test, modeles_c, np, torch, y_test):
    # --- Interpolation entre deux chiffres : pixels, AE 2D, VAE 2D, VAE 16D ---
    _yt = y_test.cpu().numpy()
    _ia, _ib = int(np.where(_yt == 1)[0][0]), int(np.where(_yt == 7)[0][1])
    _xa, _xb = X_test[_ia:_ia + 1], X_test[_ib:_ib + 1]
    _t = torch.linspace(0, 1, 9, device=X_test.device)[:, None]
    interpolations = {"pixels": ((1 - _t) * _xa.flatten(1) + _t * _xb.flatten(1)).view(-1, 28, 28).cpu().numpy()}
    with torch.no_grad():
        for _nom in ("AE 2D", "VAE 2D", "VAE 16D"):
            _m = modeles_c[_nom]
            _za, _zb = _m.encoder(_xa)[0], _m.encoder(_xb)[0]
            interpolations[_nom] = torch.sigmoid(_m.decoder((1 - _t) * _za + _t * _zb)).view(-1, 28, 28).cpu().numpy()
        # Carte de l'espace latent 2D du VAE : grille 15 × 15 sur les quantiles de N(0, 1)
        from scipy.stats import norm

        _q = torch.tensor(norm.ppf(np.linspace(0.03, 0.97, 15)), dtype=torch.float32)
        _gz = torch.stack(torch.meshgrid(_q, _q.flip(0), indexing="xy"), -1).view(-1, 2).to(X_test.device)
        carte_vae = torch.sigmoid(modeles_c["VAE 2D"].decoder(_gz)).view(15, 15, 28, 28).cpu().numpy()
    print(f"Interpolation entre l'image de test n° {_ia} (un 1) et n° {_ib} (un 7)")
    return carte_vae, interpolations


@app.cell
def _(
    SMOKE,
    carte_vae,
    exemples_b,
    grilles_e,
    histos_c,
    instantanes_gan,
    interpolations,
    mo,
    np,
    plt,
    resultats_c,
    resultats_e,
    sauvegarder_figure,
    suivi_gan,
):
    # --- Figures du notebook (PNG) ---
    dossier = mo.notebook_dir() / "figures"
    suffixe = "-smoke" if SMOKE else ""

    def mosaique(images, lignes, colonnes):
        im = np.asarray(images).reshape(-1, 28, 28)[: lignes * colonnes]
        return im.reshape(lignes, colonnes, 28, 28).transpose(0, 2, 1, 3).reshape(lignes * 28, colonnes * 28)

    # 1. Débruitage
    fig_b, axes_b = plt.subplots(4, 1, figsize=(8, 4.6))
    for _ax, (_cle, _titre) in zip(axes_b, [("propre", "originale"), ("bruitee", "bruitée (σ = 0,5)"),
                                           ("ae", "AE classique"), ("dae", "AE débruitant")]):
        _ax.imshow(mosaique(exemples_b[_cle], 1, 8), cmap="gray_r", vmin=0, vmax=1)
        _ax.set_ylabel(_titre, rotation=0, ha="right", va="center", fontsize=9)
        _ax.set_xticks([]), _ax.set_yticks([])
        _ax.grid(False)
    sauvegarder_figure(fig_b, "debruitage" + suffixe, dossier)

    # 2. Espaces latents 2D : AE contre VAE, et carte du VAE
    fig_c, axes_c = plt.subplots(1, 3, figsize=(15, 4.8))
    for _ax, _nom in zip(axes_c[:2], ("AE 2D", "VAE 2D")):
        _c, _y = resultats_c[_nom]["codes_test"], resultats_c[_nom]["etiquettes_test"]
        _sc = _ax.scatter(_c[:, 0], _c[:, 1], c=_y, cmap="tab10", s=2, vmin=-0.5, vmax=9.5)
        _ax.set(title=f"{_nom} : codes des images de test", xlabel="z₁", ylabel="z₂")
    fig_c.colorbar(_sc, ax=axes_c[1], ticks=range(10))
    axes_c[2].imshow(carte_vae.transpose(0, 2, 1, 3).reshape(15 * 28, 15 * 28), cmap="gray_r")
    axes_c[2].set(title="VAE 2D : décodage d'une grille de z", xticks=[], yticks=[])
    axes_c[2].grid(False)
    sauvegarder_figure(fig_c, "espace-latent" + suffixe, dossier)

    # 3. Échantillons des quatre modèles
    _noms = list(grilles_e)
    fig_e, axes_e = plt.subplots(1, len(_noms), figsize=(3.2 * len(_noms), 3.6))
    for _ax, _nom in zip(axes_e, _noms):
        _ax.imshow(mosaique(grilles_e[_nom], 8, 8), cmap="gray_r", vmin=0, vmax=1)
        _ax.set_title(f"{_nom}\nFréchet {resultats_e[_nom]['distance_frechet']:.1f}", fontsize=10)
        _ax.axis("off")
    sauvegarder_figure(fig_e, "echantillons" + suffixe, dossier)

    # 4. GAN : courbes et instantanés
    fig_d, axes_d = plt.subplots(1, 2, figsize=(12, 4))
    axes_d[0].plot(suivi_gan["pas"], suivi_gan["perte_D"], label="perte D")
    axes_d[0].plot(suivi_gan["pas"], suivi_gan["perte_G"], label="perte G")
    axes_d[0].set(xlabel="pas", title="Pertes (moyennes sur 50 pas)")
    axes_d[0].legend()
    axes_d[1].plot(suivi_gan["pas"], suivi_gan["D_vraies"], label="D(vraies)")
    axes_d[1].plot(suivi_gan["pas"], suivi_gan["D_fausses"], label="D(fausses)")
    axes_d[1].set(xlabel="pas", title="Sortie moyenne du discriminateur", ylim=(0, 1))
    axes_d[1].legend()
    sauvegarder_figure(fig_d, "gan-courbes" + suffixe, dossier)

    # 5. Interpolations
    fig_i, axes_i = plt.subplots(len(interpolations), 1, figsize=(8, 1.1 * len(interpolations)))
    for _ax, (_nom, _im) in zip(axes_i, interpolations.items()):
        _ax.imshow(mosaique(_im, 1, 9), cmap="gray_r", vmin=0, vmax=1)
        _ax.set_ylabel(_nom, rotation=0, ha="right", va="center", fontsize=9)
        _ax.set_xticks([]), _ax.set_yticks([])
        _ax.grid(False)
    sauvegarder_figure(fig_i, "interpolation" + suffixe, dossier)

    # 6. Courbes d'apprentissage C
    fig_h, ax_h = plt.subplots(figsize=(7, 4))
    for _nom, _h in histos_c.items():
        ax_h.plot(range(1, len(_h) + 1), _h, marker=".", label=_nom)
    ax_h.set(xlabel="époque", ylabel="perte d'entraînement (nat/image)", title="AE (BCE) et VAE (−ELBO)")
    ax_h.legend()
    sauvegarder_figure(fig_h, "apprentissage-c" + suffixe, dossier)
    plt.close("all")
    print(f"Figures : {dossier}")
    _ = instantanes_gan
    mo.md("Figures écrites dans `figures/`.")


@app.cell
def _(
    SMOKE,
    carte_vae,
    config,
    exemples_b,
    grilles_e,
    histos_c,
    instantanes_gan,
    interpolations,
    json,
    mo,
    np,
    precision_juge,
    resultats_a,
    resultats_b,
    resultats_c,
    resultats_e,
    suivi_gan,
    verif_kl,
):
    # --- Résultats bruts pour les figures du site (run complet seulement) ---
    if not SMOKE:
        _dossier = mo.notebook_dir() / "figures"
        _c = {k: {kk: vv for kk, vv in v.items() if kk not in ("codes_test", "etiquettes_test")}
              for k, v in resultats_c.items()}
        _json = {"config": config, "partie_a": resultats_a, "verif_kl": verif_kl, "partie_b": resultats_b,
                 "partie_c": _c, "histos_c": histos_c, "suivi_gan": suivi_gan,
                 "precision_juge": precision_juge, "partie_e": resultats_e}
        (_dossier / "resultats.json").write_text(json.dumps(_json, ensure_ascii=False, indent=1))
        # Images en uint8 (0-255) : quelques centaines de Ko au total
        def _u8(a):
            return np.round(np.clip(np.asarray(a), 0, 1) * 255).astype(np.uint8)

        np.savez_compressed(
            _dossier / "images.npz",
            codes_ae=resultats_c["AE 2D"]["codes_test"].astype(np.float32),
            codes_vae=resultats_c["VAE 2D"]["codes_test"].astype(np.float32),
            etiquettes=resultats_c["VAE 2D"]["etiquettes_test"].astype(np.uint8),
            carte_vae=_u8(carte_vae),
            **{f"debruitage_{k}": _u8(v) for k, v in exemples_b.items()},
            **{f"grille_{i}": _u8(grilles_e[k]) for i, k in enumerate(grilles_e)},
            noms_grilles=np.array(list(grilles_e)),
            **{f"interp_{i}": _u8(interpolations[k]) for i, k in enumerate(interpolations)},
            noms_interp=np.array(list(interpolations)),
            **{f"gan_epoque_{k}": _u8(v) for k, v in instantanes_gan.items()},
        )
        print(f"Résultats bruts : {_dossier / 'resultats.json'}, {_dossier / 'images.npz'}")


@app.cell
def _(
    SMOKE,
    duree_b,
    duree_c,
    duree_d,
    duree_juge,
    mo,
    precision_juge,
    resultats_a,
    resultats_b,
    resultats_c,
    resultats_e,
    verif_kl,
):
    # --- Bilan : seuils du README. En smoke, on vérifie seulement que tout tourne. ---
    _e = resultats_e
    _b = resultats_b
    seuils = {
        "A. AE linéaire : MSE ≤ 1,01 × MSE de l'ACP": resultats_a["rapport"] <= 1.01,
        "A. AE linéaire : cos des angles principaux avec l'ACP ≥ 0,99": min(resultats_a["cos_angles_principaux"]) >= 0.99,
        "A. KL fermée = Monte-Carlo à 0,01 nat près": all(abs(r["fermee"] - r["monte_carlo"]) < 0.01 for r in verif_kl),
        "B. AE débruitant : PSNR ≥ bruitée + 8 dB": _b["AE débruitant (bruitée → propre)"]["psnr"] >= _b["bruitée"]["psnr"] + 8,
        "B. AE débruitant : PSNR ≥ AE classique + 4 dB": (
            _b["AE débruitant (bruitée → propre)"]["psnr"] >= _b["AE classique (propre → propre)"]["psnr"] + 4),
        "B. AE classique sur image propre : PSNR ≥ 25 dB (il sait recopier)": _b["AE classique sur image propre"]["psnr"] >= 25,
        "C. VAE 2D : −ELBO de test ≤ 150 nat": resultats_c["VAE 2D"]["moins_elbo"] <= 150,
        "C. VAE 16D : −ELBO de test ≤ 105 nat": resultats_c["VAE 16D"]["moins_elbo"] <= 105,
        "E. juge : exactitude de test ≥ 98,5 %": precision_juge >= 0.985,
        "E. DCGAN : Fréchet < VAE 16D < VAE 2D": (
            _e["DCGAN"]["distance_frechet"] < _e["VAE 16D"]["distance_frechet"] < _e["VAE 2D"]["distance_frechet"]),
        "E. VAE 2D : Fréchet < AE 2D (la loi a priori sert)": _e["VAE 2D"]["distance_frechet"] < _e["AE 2D"]["distance_frechet"],
        "E. DCGAN : les 10 classes produites, chacune ≥ 2 %": _e["DCGAN"]["part_min_classe"] >= 0.02,
    }
    _lignes = "\n".join(f"- {'✅' if ok else '❌'} {nom}" for nom, ok in seuils.items())
    _echecs = [nom for nom, ok in seuils.items() if not ok]
    print(_lignes)
    _total = duree_b + duree_c + duree_d + duree_juge
    print(f"Entraînements : B {duree_b:.0f} s, C {duree_c:.0f} s, D {duree_d:.0f} s, juge {duree_juge:.0f} s ; total {_total:.0f} s")
    if SMOKE:
        print("(mode smoke : seuils non vérifiés, des ❌ sont attendus)")
    if not SMOKE and _echecs:
        raise SystemExit("Seuils non atteints :\n" + "\n".join(_echecs))
    _note = "\n\n*Mode smoke : seuils non vérifiés, des ❌ sont attendus.*" if SMOKE else ""
    mo.md(f"**Bilan** ({_total:.0f} s d'entraînement)\n\n{_lignes}{_note}")


if __name__ == "__main__":
    app.run()
