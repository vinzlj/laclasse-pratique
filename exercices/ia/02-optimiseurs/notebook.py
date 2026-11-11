# Exercice 02 : optimiseurs, de SGD à Adam, AdamW et Muon (chapitre 02. Optimisation).
#
#   make ex NUM=02              → éditeur marimo (curseurs, bouton « Entraîner »)
#   make ex NUM=02 MODE=run     → script complet, vérifie les seuils du README
#   make ex NUM=02 MODE=smoke   → test rapide (< 30 s)
#
# Progression : chaque optimiseur est d'abord écrit en numpy (quelques lignes), testé
# sur trois surfaces 2D, puis comparé à torch.optim (mêmes chiffres à 1e-10 près ;
# écart mesuré ~3e-15), et enfin utilisé sur un vrai réseau (MLP sur MNIST). Les
# formules suivent exactement les conventions de PyTorch 2.14, pour que la comparaison
# soit possible.

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
    from torch import nn

    from ia_commun import get_device, is_smoke, sauvegarder_figure, seed_everything, style_commun

    return (
        get_device,
        is_smoke,
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
    # Exercice 02 : optimiseurs, de SGD à Adam, AdamW et Muon

    **Objectif.** Écrire soi-même SGD, momentum, Nesterov, AdaGrad, RMSProp, Adam et
    AdamW en numpy, les regarder se déplacer sur trois surfaces difficiles (vallée
    étroite, point selle, fonction de Rosenbrock), puis vérifier qu'ils donnent
    exactement les mêmes chiffres que `torch.optim` avant de les utiliser sur un vrai
    réseau.

    **Ce qu'on minimise.** Aucune fonction de perte en propre : l'optimiseur minimise
    la fonction qu'on lui donne. Ici, d'abord des fonctions test $f(x, y)$ choisies
    pour leurs pièges, puis une erreur quadratique moyenne (régression linéaire, partie 4)
    et l'entropie croisée d'un classifieur MNIST (partie 7), deux pertes du chapitre 01.

    Plan : (1) surfaces ; (2) optimiseurs en numpy ; (3) trajectoires ;
    (4) taille de lot et planning du pas ; (5) Adam + L2 contre AdamW ;
    (6) Newton et L-BFGS ; (7) mêmes calculs avec `torch.optim`, puis un MLP sur MNIST ;
    (8) bonus : Muon.
    """)


@app.cell
def _(is_smoke, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        # nombre de pas sur chaque surface 2D (parties 2 et 3)
        pas_surface={"vallee": 300, "selle": 300, "rosenbrock": 300 if SMOKE else 3000},
        # régression linéaire par mini-lots (partie 4)
        epoques_regression=3 if SMOKE else 20,
        # MLP sur MNIST (partie 7)
        n_mnist=5000 if SMOKE else 60000,
        epoques_mnist=1 if SMOKE else 2,
        lot_mnist=128,
        # jouet Muon (partie 8)
        pas_muon=60 if SMOKE else 300,
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SEED, SMOKE, config


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Trois surfaces, trois pièges

    | Surface | Formule | Minimum | Piège |
    |---|---|---|---|
    | vallée étroite | $f = \tfrac12(x^2 + 50\,y^2)$ | $(0, 0)$ | courbure 50 fois plus forte en $y$ qu'en $x$ : un pas assez petit pour $y$ est minuscule pour $x$ |
    | point selle | $f = x^2 + \tfrac14 y^4 - \tfrac12 y^2$ | $(0, \pm 1)$, $f = -0{,}25$ | en $(0, 0)$ le gradient est nul sans que ce soit un minimum ; on part juste à côté ($y = 10^{-4}$) |
    | Rosenbrock (1960) | $f = (1 - x)^2 + 100\,(y - x^2)^2$ | $(1, 1)$ | une vallée courbe en forme de banane : facile d'y tomber, long de la suivre |
    """)


@app.cell
def _(np):
    # --- Surfaces : la fonction f et son gradient, écrit à la main ---
    # Chaque surface est un dict : f, grad, point de départ, cadre du tracé, minimum.
    def _f_vallee(p):
        x, y = p
        return 0.5 * (x**2 + 50.0 * y**2)

    def _g_vallee(p):
        x, y = p
        return np.array([x, 50.0 * y])

    def _f_selle(p):
        x, y = p
        return x**2 + 0.25 * y**4 - 0.5 * y**2

    def _g_selle(p):
        x, y = p
        return np.array([2.0 * x, y**3 - y])

    def _f_rosen(p):
        x, y = p
        return (1.0 - x) ** 2 + 100.0 * (y - x**2) ** 2

    def _g_rosen(p):
        x, y = p
        return np.array([-2.0 * (1.0 - x) - 400.0 * x * (y - x**2), 200.0 * (y - x**2)])

    SURFACES = {
        "vallee": dict(nom="vallée étroite", f=_f_vallee, grad=_g_vallee, depart=(-8.0, 1.0),
                       cadre=(-9, 3, -1.5, 1.5), f_min=0.0, seuil=1e-3),
        "selle": dict(nom="point selle", f=_f_selle, grad=_g_selle, depart=(1.5, 1e-4),
                      cadre=(-1.8, 1.8, -1.6, 1.6), f_min=-0.25, seuil=-0.24),
        "rosenbrock": dict(nom="Rosenbrock", f=_f_rosen, grad=_g_rosen, depart=(-1.5, 2.0),
                           cadre=(-2, 2, -1, 3), f_min=0.0, seuil=1e-3),
    }

    # Contrôle : le gradient écrit à la main = différences finies
    for _cle, _s in SURFACES.items():
        _p = np.array([0.3, -0.7])
        _h = 1e-6
        _num = np.array([(_s["f"](_p + _h * _e) - _s["f"](_p - _h * _e)) / (2 * _h) for _e in np.eye(2)])
        assert np.allclose(_num, _s["grad"](_p), atol=1e-5), _cle
    print("Gradients vérifiés par différences finies sur les trois surfaces.")
    return (SURFACES,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Les optimiseurs en numpy

    Tous ont la même interface : `pas(p, g)` reçoit les paramètres $p$ et le gradient
    $g$, renvoie les nouveaux paramètres. Les formules sont celles de PyTorch 2.14
    (voir la partie 7, qui le vérifie). Notations : $\eta$ pas d'apprentissage,
    $t$ numéro du pas, opérations élément par élément.

    | Optimiseur | Mise à jour | Mémoire |
    |---|---|---|
    | SGD | $p \leftarrow p - \eta g$ | aucune |
    | momentum (Polyak, 1964) | $b \leftarrow \mu b + g$ ; $p \leftarrow p - \eta b$ | vitesse $b$ |
    | Nesterov (1983) | $b \leftarrow \mu b + g$ ; $p \leftarrow p - \eta (g + \mu b)$ | vitesse $b$ |
    | AdaGrad (2011) | $s \leftarrow s + g^2$ ; $p \leftarrow p - \eta g / (\sqrt{s} + \varepsilon)$ | somme des $g^2$ |
    | RMSProp (Hinton, cours *Neural Networks for Machine Learning*, leçon 6e) | $s \leftarrow \alpha s + (1-\alpha) g^2$ ; $p \leftarrow p - \eta g / (\sqrt{s} + \varepsilon)$ | moyenne glissante des $g^2$ |
    | Adam (2014) | $m \leftarrow \beta_1 m + (1-\beta_1) g$ ; $v \leftarrow \beta_2 v + (1-\beta_2) g^2$ ; $p \leftarrow p - \eta \,\hat m / (\sqrt{\hat v} + \varepsilon)$ | $m$ et $v$, corrigés : $\hat m = m/(1-\beta_1^t)$, $\hat v = v/(1-\beta_2^t)$ |
    | AdamW (2017) | $p \leftarrow p(1 - \eta\lambda)$, puis le pas d'Adam | idem |

    RMSProp n'a pas d'article : il vient d'une diapositive de cours, avec $\alpha = 0{,}9$ ;
    PyTorch a choisi $\alpha = 0{,}99$ par défaut, que l'on garde ici.

    Deux conventions de PyTorch à connaître : la vitesse $b$ n'est **pas** multipliée par
    $\eta$ (avec un $\eta$ constant, c'est la même trajectoire que la forme
    « $v \leftarrow \mu v - \eta g$ » des manuels), et elle démarre à $b = g$ au premier pas.
    """)


@app.cell
def _(np):
    # --- Optimiseurs « à la main ». Chaque classe garde son état entre deux pas. ---
    class SGD:
        def __init__(self, lr):
            self.lr = lr

        def pas(self, p, g):
            return p - self.lr * g

    class Momentum:
        """Boule pesante de Polyak (nesterov=False) ou gradient accéléré de Nesterov."""

        def __init__(self, lr, mu=0.9, nesterov=False):
            self.lr, self.mu, self.nesterov = lr, mu, nesterov
            self.b = None  # vitesse

        def pas(self, p, g):
            # PyTorch : la vitesse démarre à g au premier pas, puis b = mu*b + g
            self.b = g.copy() if self.b is None else self.mu * self.b + g
            # Nesterov : on « regarde devant » en ajoutant mu*b au gradient courant
            direction = g + self.mu * self.b if self.nesterov else self.b
            return p - self.lr * direction

    class AdaGrad:
        def __init__(self, lr, eps=1e-10):
            self.lr, self.eps, self.s = lr, eps, 0.0

        def pas(self, p, g):
            self.s = self.s + g * g  # la somme ne fait que grandir : le pas effectif rétrécit
            return p - self.lr * g / (np.sqrt(self.s) + self.eps)

    class RMSProp:
        def __init__(self, lr, alpha=0.99, eps=1e-8):
            self.lr, self.alpha, self.eps, self.s = lr, alpha, eps, 0.0

        def pas(self, p, g):
            # moyenne glissante : on oublie les vieux gradients (contrairement à AdaGrad)
            self.s = self.alpha * self.s + (1 - self.alpha) * g * g
            return p - self.lr * g / (np.sqrt(self.s) + self.eps)

    class Adam:
        """Adam ; avec decouple=True et wd > 0 : AdamW. Avec decouple=False et wd > 0 :
        Adam + pénalité L2 ajoutée au gradient (ce que fait torch.optim.Adam(weight_decay=…))."""

        def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0, decouple=False):
            self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
            self.wd, self.decouple = wd, decouple
            self.m, self.v, self.t = 0.0, 0.0, 0

        def pas(self, p, g):
            self.t += 1
            if self.wd:
                if self.decouple:
                    p = p * (1 - self.lr * self.wd)  # AdamW : on rétrécit les poids à part
                else:
                    g = g + self.wd * p  # Adam + L2 : la pénalité passe DANS le gradient
            self.m = self.b1 * self.m + (1 - self.b1) * g  # moyenne des gradients (direction)
            self.v = self.b2 * self.v + (1 - self.b2) * g * g  # moyenne des carrés (échelle)
            # corrections de biais : m et v démarrent à 0, donc sous-estimés au début
            m_hat = self.m / (1 - self.b1**self.t)
            v_hat = self.v / (1 - self.b2**self.t)
            return p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def AdamW(lr, wd=0.01, **kw):
        return Adam(lr, wd=wd, decouple=True, **kw)

    def trajectoire(surface, optimiseur, n_pas, ecretage=None):
        """Applique n_pas de l'optimiseur depuis le départ de la surface.
        ecretage : norme maximale du gradient (gradient clipping), None = sans."""
        p = np.array(surface["depart"], dtype=float)
        points = [p]
        for _ in range(n_pas):
            g = surface["grad"](p)
            if ecretage is not None:
                n = np.linalg.norm(g)
                if n > ecretage:
                    g = g * (ecretage / n)
            p = optimiseur.pas(p, g)
            points.append(p)
            if not np.all(np.isfinite(p)) or np.abs(p).max() > 1e6:
                break  # divergence : on arrête là
        return np.array(points)

    return AdaGrad, Adam, AdamW, Momentum, RMSProp, SGD, trajectoire


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Trajectoires sur les trois surfaces

    Réglages : un même $\eta$ pour SGD, momentum et Nesterov sur une surface donnée
    (pour isoler l'effet de la mémoire), $\eta$ propre aux méthodes adaptatives (leur pas
    se mesure en unités de paramètre, pas en unités de gradient). $\mu = 0{,}9$,
    $\beta_1 = 0{,}9$, $\beta_2 = 0{,}999$, $\alpha = 0{,}99$ (valeurs par défaut de PyTorch).
    Le tableau donne le **nombre de pas pour passer sous le seuil** (vallée et Rosenbrock :
    $f < 10^{-3}$ ; selle : $f < -0{,}24$, c'est-à-dire sorti du col).
    """)


@app.cell
def _(AdaGrad, Adam, Momentum, RMSProp, SGD):
    # Pas d'apprentissage par surface et par optimiseur (repris tels quels par la démo
    # du site, composant Optimizers). Une fabrique par optimiseur : un état neuf à chaque appel.
    REGLAGES = {
        "vallee": {"SGD": 0.025, "momentum": 0.025, "Nesterov": 0.025, "AdaGrad": 0.5, "RMSProp": 0.1, "Adam": 0.5},
        "selle": {"SGD": 0.05, "momentum": 0.05, "Nesterov": 0.05, "AdaGrad": 0.05, "RMSProp": 0.05, "Adam": 0.05},
        "rosenbrock": {"SGD": 1e-3, "momentum": 1e-3, "Nesterov": 1e-3, "AdaGrad": 0.1, "RMSProp": 3e-3, "Adam": 0.02},
    }

    def fabrique(nom, lr):
        return {
            "SGD": lambda: SGD(lr),
            "momentum": lambda: Momentum(lr, mu=0.9),
            "Nesterov": lambda: Momentum(lr, mu=0.9, nesterov=True),
            "AdaGrad": lambda: AdaGrad(lr),
            "RMSProp": lambda: RMSProp(lr),
            "Adam": lambda: Adam(lr),
        }[nom]()

    return REGLAGES, fabrique


@app.cell
def _(REGLAGES, SURFACES, config, fabrique, mo, np, trajectoire):
    # Toutes les trajectoires, et le tableau « pas pour passer sous le seuil »
    TRAJ = {}
    lignes_tableau = []
    for _cle, _s in SURFACES.items():
        TRAJ[_cle] = {}
        for _nom, _lr in REGLAGES[_cle].items():
            _tr = trajectoire(_s, fabrique(_nom, _lr), config["pas_surface"][_cle])
            TRAJ[_cle][_nom] = _tr
            _fs = np.array([_s["f"](_q) for _q in _tr])
            _sous = np.nonzero(_fs < _s["seuil"])[0]
            lignes_tableau.append(dict(
                surface=_s["nom"], optimiseur=_nom, eta=_lr,
                pas_sous_seuil=int(_sous[0]) if len(_sous) else None,
                f_final=float(_fs[-1]), point_final=np.round(_tr[-1], 4).tolist(),
            ))
    for _l in lignes_tableau:
        print(f"{_l['surface']:15s} {_l['optimiseur']:9s} η={_l['eta']:<6} "
              f"sous le seuil au pas {_l['pas_sous_seuil']!s:>5}   f final {_l['f_final']:.2e}")
    mo.ui.table(lignes_tableau, selection=None)
    return TRAJ, lignes_tableau


@app.cell
def _(SURFACES, config, mo):
    # Curseurs (éditeur) : choisir la surface et « rejouer » les trajectoires pas à pas
    choix_surface = mo.ui.dropdown(
        options={_s["nom"]: _k for _k, _s in SURFACES.items()}, value="Rosenbrock", label="Surface"
    )
    curseur_pas = mo.ui.slider(1, max(config["pas_surface"].values()), value=max(config["pas_surface"].values()),
                               label="Pas affichés", full_width=True)
    mo.hstack([choix_surface, curseur_pas])
    return choix_surface, curseur_pas


@app.cell
def _(SURFACES, np):
    # Tracé commun : lignes de niveau (échelle log pour voir le fond des vallées) + trajectoires
    COULEURS = {"SGD": "#0072B2", "momentum": "#E69F00", "Nesterov": "#CC79A7",
                "AdaGrad": "#56B4E9", "RMSProp": "#009E73", "Adam": "#222222"}

    def tracer_surface(ax, cle, trajs, n_max=None, noms=None):
        s = SURFACES[cle]
        x0, x1, y0, y1 = s["cadre"]
        xx, yy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
        zz = s["f"]((xx, yy)) - s["f_min"]
        ax.contour(xx, yy, np.log10(zz + 1e-3), levels=20, colors="0.7", linewidths=0.6, linestyles="solid")
        for nom, tr in trajs.items():
            if noms and nom not in noms:
                continue
            t = tr[: (n_max or len(tr)) + 1]
            ax.plot(t[:, 0], t[:, 1], "-", color=COULEURS[nom], lw=1.4, label=nom)
            ax.plot(t[-1, 0], t[-1, 1], "o", color=COULEURS[nom], ms=4)
        ax.plot(*s["depart"], "k^", ms=7)
        ax.set(xlim=(x0, x1), ylim=(y0, y1), title=s["nom"])
        ax.grid(False)

    return (tracer_surface,)


@app.cell
def _(TRAJ, choix_surface, curseur_pas, plt, tracer_surface):
    # Vue interactive (éditeur) : une surface, les n premiers pas
    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    tracer_surface(_ax, choix_surface.value, TRAJ[choix_surface.value], n_max=curseur_pas.value)
    _ax.legend(loc="upper right", fontsize=8)
    _fig


@app.cell
def _(SMOKE, TRAJ, mo, plt, sauvegarder_figure, tracer_surface):
    # Figure fixe : les trois surfaces côte à côte (sans AdaGrad, pour la lisibilité)
    # Pas affichés : assez peu pour que les points d'arrivée se distinguent
    _affiches = {"vallee": 60, "selle": 60, "rosenbrock": 1000}
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for _ax, _cle in zip(_axes, ["vallee", "selle", "rosenbrock"]):
        tracer_surface(_ax, _cle, TRAJ[_cle], n_max=_affiches[_cle],
                       noms=["SGD", "momentum", "Nesterov", "RMSProp", "Adam"])
        _ax.set_title(f"{_ax.get_title()} : {_affiches[_cle]} pas")
    _axes[0].legend(loc="lower right", fontsize=8)
    _chemin = sauvegarder_figure(_fig, "trajectoires-smoke" if SMOKE else "trajectoires",
                                 mo.notebook_dir() / "figures")
    print(f"Figure : {_chemin[0]}")
    _fig


@app.cell
def _(mo):
    mo.md(r"""
    ### Pas trop grand, pas trop petit, et écrêtage du gradient

    Sur la vallée étroite, la plus forte courbure vaut $L = 50$ : SGD est stable si
    $\eta < 2/L = 0{,}04$. Juste au-dessus, il diverge ; très en dessous, il rampe.
    Sur Rosenbrock, un $\eta$ un peu trop grand fait exploser SGD en quelques pas ;
    l'**écrêtage** (*gradient clipping*) plafonne la norme du gradient et suffit à le
    garder stable, au prix d'un pas plus court dans les zones raides.
    """)


@app.cell
def _(SGD, SURFACES, np, trajectoire):
    # Balayage de η sur la vallée étroite (SGD seul, 300 pas)
    resultats_eta = {}
    for _lr in [0.005, 0.025, 0.039, 0.041]:
        _tr = trajectoire(SURFACES["vallee"], SGD(_lr), 300)
        _f = SURFACES["vallee"]["f"](_tr[-1])
        resultats_eta[_lr] = (len(_tr) - 1, float(_f))
        print(f"vallée, SGD η={_lr:<6}: {len(_tr) - 1:3d} pas, f final = {_f:.3e}"
              + ("   ← diverge" if not np.isfinite(_f) or _f > 1e3 else ""))

    # Écrêtage sur Rosenbrock : η = 3e-3, 3000 pas, norme du gradient plafonnée à 10
    _s = SURFACES["rosenbrock"]
    _sans = trajectoire(_s, SGD(3e-3), 3000)
    _avec = trajectoire(_s, SGD(3e-3), 3000, ecretage=10.0)
    ecretage = dict(pas_avant_divergence=len(_sans) - 1, final_avec=_avec[-1], f_avec=float(_s["f"](_avec[-1])))
    print(f"Rosenbrock, SGD η=3e-3 sans écrêtage : divergence après {len(_sans) - 1} pas")
    print(f"Rosenbrock, SGD η=3e-3 avec écrêtage (norme ≤ 10) : point final {np.round(_avec[-1], 3)}, "
          f"f = {ecretage['f_avec']:.3f}")
    return ecretage, resultats_eta


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Taille de lot et planning du pas

    Un réseau ne voit jamais « la » surface : il en estime la pente sur un **mini-lot**
    d'exemples. On le simule sur une régression linéaire (4 000 exemples, 20 variables
    d'échelles différentes, perte = erreur quadratique moyenne $\tfrac12\,\overline{(Xw-y)^2}$).
    On trace l'**excès de perte** $\mathcal L(w) - \mathcal L(w^\star)$, où $w^\star$ est
    la solution exacte des moindres carrés, en fonction du nombre de passes sur les
    données (époques), donc à calcul égal.

    - **plein lot** : un pas par époque, gradient exact ;
    - **mini-lot de 32** à $\eta$ constant : 125 pas par époque, gradient bruité ;
    - **mini-lot de 32** avec **échauffement** (*warmup*, 5 % des pas, $\eta$ monte
      linéairement) puis **décroissance en cosinus** jusqu'à 0.
    """)


@app.cell
def _(config, math, np):
    # --- Régression linéaire par mini-lots, en numpy ---
    _rng = np.random.default_rng(42)
    N_REG, D_REG = 4000, 20
    X_reg = _rng.normal(size=(N_REG, D_REG)) * np.linspace(0.2, 2.0, D_REG)  # échelles de 0,2 à 2
    _w_vrai = _rng.normal(size=D_REG)
    y_reg = X_reg @ _w_vrai + 0.5 * _rng.normal(size=N_REG)
    _w_opt = np.linalg.lstsq(X_reg, y_reg, rcond=None)[0]

    def perte_reg(w):
        r = X_reg @ w - y_reg
        return 0.5 * np.mean(r * r)

    PERTE_OPT = perte_reg(_w_opt)

    def eta_planning(t, total, eta_max, echauffement=0.05):
        """Échauffement linéaire puis décroissance en cosinus jusqu'à 0."""
        t_ech = max(1, int(echauffement * total))
        if t < t_ech:
            return eta_max * (t + 1) / t_ech
        return eta_max * 0.5 * (1 + math.cos(math.pi * (t - t_ech) / (total - t_ech)))

    def entrainer_reg(taille_lot, eta, planning=False, epoques=config["epoques_regression"]):
        rng = np.random.default_rng(42)
        w = np.zeros(D_REG)
        exces = [perte_reg(w) - PERTE_OPT]
        pas_par_epoque = N_REG // taille_lot
        total, t = epoques * pas_par_epoque, 0
        for _ in range(epoques):
            ordre = rng.permutation(N_REG)
            for i in range(pas_par_epoque):
                idx = ordre[i * taille_lot:(i + 1) * taille_lot]
                g = X_reg[idx].T @ (X_reg[idx] @ w - y_reg[idx]) / taille_lot  # gradient du mini-lot
                w = w - (eta_planning(t, total, eta) if planning else eta) * g
                t += 1
            exces.append(perte_reg(w) - PERTE_OPT)
        return np.array(exces)

    _H = np.linalg.eigvalsh(X_reg.T @ X_reg / N_REG)
    print(f"Courbures de la perte : de {_H.min():.3f} à {_H.max():.2f} (rapport {_H.max() / _H.min():.0f}) "
          f"→ plein lot stable si η < {2 / _H.max():.2f}")

    COURBES_LOTS = {
        "plein lot (1 pas par époque), η = 0,4": entrainer_reg(N_REG, 0.4),
        "lot de 32, η = 0,1 constant": entrainer_reg(32, 0.1),
        "lot de 32, η = 0,4 constant": entrainer_reg(32, 0.4),
        "lot de 32, η max 0,4, échauffement + cosinus": entrainer_reg(32, 0.4, planning=True),
    }
    for _nom, _c in COURBES_LOTS.items():
        print(f"{_nom:46s} excès après 1 époque {_c[1]:.2e}, à la fin {_c[-1]:.2e}")
    return COURBES_LOTS, eta_planning


@app.cell
def _(COURBES_LOTS, SMOKE, eta_planning, mo, np, plt, sauvegarder_figure):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]})
    for _nom, _c in COURBES_LOTS.items():
        _ax1.semilogy(np.arange(len(_c)), _c, "o-", ms=3, label=_nom)
    _ax1.set(xlabel="époques (passes sur les données)", ylabel="excès de perte  L(w) − L(w*)",
             title="Mini-lots : rapides au début, puis un plancher de bruit")
    _ax1.legend(fontsize=8)
    _total = 2500
    _ax2.plot([eta_planning(_t, _total, 0.4) for _t in range(_total)])
    _ax2.set(xlabel="pas", ylabel="η", title="Échauffement puis cosinus")
    _chemin = sauvegarder_figure(_fig, "mini-lots-smoke" if SMOKE else "mini-lots", mo.notebook_dir() / "figures")
    print(f"Figure : {_chemin[0]}")
    _fig


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Adam + L2 contre AdamW

    Jouet à deux poids : $w_1$ sert (perte $\tfrac12 (w_1 - 1)^2$), $w_2$ ne sert à rien
    (aucun gradient venant de la perte), et on veut que la régularisation le ramène vers 0
    **à une vitesse réglée par** $\lambda$.

    - **Adam + L2** : on ajoute $\lambda w$ au gradient. Pour $w_2$, le gradient vaut
      $\lambda w_2$, puis Adam le divise par $\sqrt{\hat v}$, du même ordre que $\lambda |w_2|$ :
      $\lambda$ **disparaît**, $w_2$ descend d'environ $\eta$ par pas au début, puis
      ralentit, **exactement de la même façon** pour $\lambda = 0{,}01$ et $\lambda = 0{,}1$.
    - **AdamW** : $w \leftarrow w(1 - \eta\lambda)$ à part. $w_2$ décroît géométriquement,
      d'autant plus vite que $\lambda$ est grand : $\lambda$ garde son sens.
    """)


@app.cell
def _(Adam, np):
    def jouet_l2(decouple, lam, eta=1e-2, n=500):
        opt = Adam(eta, wd=lam, decouple=decouple)
        w = np.array([0.0, 1.0])
        hist = [w]
        for _ in range(n):
            g = np.array([w[0] - 1.0, 0.0])  # gradient de la perte seule
            w = opt.pas(w, g)
            hist.append(w)
        return np.array(hist)

    JOUET_L2 = {(d, lam): jouet_l2(d, lam) for d in (False, True) for lam in (0.01, 0.1)}
    for (_d, _lam), _h in JOUET_L2.items():
        print(f"{'AdamW    ' if _d else 'Adam + L2'} λ={_lam:<5}: w2 au pas 100 = {_h[100, 1]:.4f}, "
              f"au pas 500 = {_h[-1, 1]:.4f} ; w1 final = {_h[-1, 0]:.4f}")
    return (JOUET_L2,)


@app.cell
def _(JOUET_L2, SMOKE, mo, plt, sauvegarder_figure):
    _fig, _ax = plt.subplots(figsize=(7, 4))
    for (_d, _lam), _h in JOUET_L2.items():
        _ax.plot(_h[:, 1], ("-" if _d else "--"), lw=2 if _d else 1.5,
                 label=f"{'AdamW' if _d else 'Adam + L2'}, λ = {_lam}".replace(".", ","))
    _ax.set(xlabel="pas", ylabel="w₂ (poids inutile)", title="La pénalité L2 perd son λ dans Adam")
    _ax.legend(fontsize=9)
    _chemin = sauvegarder_figure(_fig, "adam-l2-adamw-smoke" if SMOKE else "adam-l2-adamw", mo.notebook_dir() / "figures")
    print(f"Figure : {_chemin[0]}")
    _fig


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Second ordre : Newton et L-BFGS

    Newton utilise la courbure exacte (la matrice hessienne $H$) :
    $p \leftarrow p - H^{-1} g$. En 2D, $H$ est une matrice 2 × 2 ; pour un réseau
    de $10^9$ paramètres, elle aurait $10^{18}$ cases. **L-BFGS** (1989) estime
    $H^{-1}$ à partir des derniers déplacements et gradients (ici `scipy`, mémoire de 10 paires).
    """)


@app.cell
def _(SURFACES, np):
    from scipy.optimize import minimize

    _s = SURFACES["rosenbrock"]

    def hessienne_rosen(p):
        x, y = p
        return np.array([[2 - 400 * (y - x**2) + 800 * x**2, -400 * x], [-400 * x, 200.0]])

    # Newton « pur » : pas complet, aucune recherche linéaire
    _p = np.array(_s["depart"])
    newton_iter = None
    for _i in range(1, 101):
        _p = _p - np.linalg.solve(hessienne_rosen(_p), _s["grad"](_p))
        if np.linalg.norm(_p - 1.0) < 1e-8:
            newton_iter = _i
            break
    print(f"Newton : {newton_iter} itérations pour |p − (1, 1)| < 1e-8")

    _res = minimize(_s["f"], np.array(_s["depart"]), jac=_s["grad"], method="L-BFGS-B",
                    options=dict(gtol=1e-10, ftol=1e-15))
    lbfgs = dict(iterations=int(_res.nit), evaluations=int(_res.nfev),
                 distance=float(np.linalg.norm(_res.x - 1.0)))
    print(f"L-BFGS : {lbfgs['iterations']} itérations, {lbfgs['evaluations']} évaluations de f et du gradient, "
          f"|p − (1, 1)| = {lbfgs['distance']:.1e}")
    return lbfgs, newton_iter


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. La même chose avec `torch.optim`, puis un vrai réseau

    D'abord, la preuve que nos classes numpy sont **les mêmes algorithmes** que ceux de
    PyTorch : 500 pas sur Rosenbrock en `float64`, sur CPU, et on compare les points.
    """)


@app.cell
def _(
    AdaGrad,
    Adam,
    AdamW,
    Momentum,
    RMSProp,
    SGD,
    SURFACES,
    np,
    torch,
    trajectoire,
):
    _s = SURFACES["rosenbrock"]

    def f_torch(p):
        return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2

    _paires = {
        "SGD": (lambda: SGD(1e-3), lambda q: torch.optim.SGD([q], lr=1e-3)),
        "momentum": (lambda: Momentum(1e-3), lambda q: torch.optim.SGD([q], lr=1e-3, momentum=0.9)),
        "Nesterov": (lambda: Momentum(1e-3, nesterov=True),
                     lambda q: torch.optim.SGD([q], lr=1e-3, momentum=0.9, nesterov=True)),
        "AdaGrad": (lambda: AdaGrad(0.1), lambda q: torch.optim.Adagrad([q], lr=0.1)),
        "RMSProp": (lambda: RMSProp(3e-3), lambda q: torch.optim.RMSprop([q], lr=3e-3)),
        "Adam": (lambda: Adam(0.02), lambda q: torch.optim.Adam([q], lr=0.02)),
        "AdamW": (lambda: AdamW(0.02, wd=0.01), lambda q: torch.optim.AdamW([q], lr=0.02, weight_decay=0.01)),
    }
    ecarts_torch = {}
    for _nom, (_fab_np, _fab_torch) in _paires.items():
        _tr = trajectoire(_s, _fab_np(), 500)
        _q = torch.tensor(_s["depart"], dtype=torch.float64, requires_grad=True)
        _opt = _fab_torch(_q)
        _pts = [_q.detach().numpy().copy()]
        for _ in range(500):
            _opt.zero_grad()
            f_torch(_q).backward()
            _opt.step()
            _pts.append(_q.detach().numpy().copy())
        ecarts_torch[_nom] = float(np.abs(np.array(_pts) - _tr).max())
        print(f"{_nom:9s} écart maximal numpy / torch.optim sur 500 pas : {ecarts_torch[_nom]:.1e}")
    return (ecarts_torch,)


@app.cell
def _(mo):
    mo.md(r"""
    Puis un vrai réseau : un MLP 784-256-256-10 sur MNIST, perte = entropie croisée,
    lots de 128, même initialisation pour tous. On compare SGD, SGD + Nesterov, Adam,
    AdamW et Muon (couche cachée 256 × 256) + AdamW (le reste), avec les pas usuels
    de chacun et sans planning.
    """)


@app.cell
def _(mo):
    # Le bouton empêche l'entraînement de se relancer à chaque modification d'une cellule amont.
    bouton = mo.ui.run_button(label="Entraîner les cinq MLP")
    bouton
    return (bouton,)


@app.cell
def _(SEED, bouton, config, get_device, mo, nn, np, seed_everything, time, torch):
    # En script (make ex MODE=run|smoke), il n'y a personne pour cliquer : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Entraîner les cinq MLP » pour lancer la comparaison.*"))
    from ia_commun.donnees import mnist_tenseurs

    device = get_device()
    _Xtr, _ytr = mnist_tenseurs(train=True)
    _Xte, _yte = mnist_tenseurs(train=False)
    _Xtr, _ytr = _Xtr[: config["n_mnist"]], _ytr[: config["n_mnist"]]
    # aplatir, normaliser (moyenne 0, écart type 1) ; déjà en float32 (MPS refuse le float64)
    _moy, _et = _Xtr.mean(), _Xtr.std()
    Xtr = ((_Xtr.view(len(_Xtr), -1) - _moy) / _et).to(device)
    Xte = ((_Xte.view(len(_Xte), -1) - _moy) / _et).to(device)
    ytr, yte = _ytr.to(device), _yte.to(device)

    def creer_mlp():
        return nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                             nn.Linear(256, 10)).to(device)

    def optimiseurs_torch(nom, m):
        if nom == "SGD":
            return [torch.optim.SGD(m.parameters(), lr=0.05)]
        if nom == "SGD + Nesterov":
            return [torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9, nesterov=True)]
        if nom == "Adam":
            return [torch.optim.Adam(m.parameters(), lr=1e-3)]
        if nom == "AdamW":
            return [torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)]
        # Muon ne s'applique qu'aux matrices des couches cachées ; entrée, sortie et biais : AdamW
        cachee = m[2].weight
        autres = [p for p in m.parameters() if p is not cachee]
        return [torch.optim.Muon([cachee], lr=0.02, momentum=0.95, weight_decay=0.01),
                torch.optim.AdamW(autres, lr=1e-3, weight_decay=0.01)]

    resultats_mnist = {}
    _t0 = time.perf_counter()
    for _nom in ["SGD", "SGD + Nesterov", "Adam", "AdamW", "Muon + AdamW"]:
        seed_everything(SEED)  # même initialisation et même ordre des lots pour tous
        _m = creer_mlp()
        _opts = optimiseurs_torch(_nom, _m)
        _gen = torch.Generator().manual_seed(SEED)
        _pertes = []
        _t1 = time.perf_counter()
        for _ in range(config["epoques_mnist"]):
            _ordre = torch.randperm(len(Xtr), generator=_gen).to(device)
            for _i in range(0, len(Xtr) - config["lot_mnist"] + 1, config["lot_mnist"]):
                _idx = _ordre[_i:_i + config["lot_mnist"]]
                _perte = nn.functional.cross_entropy(_m(Xtr[_idx]), ytr[_idx])
                for _o in _opts:
                    _o.zero_grad()
                _perte.backward()
                for _o in _opts:
                    _o.step()
                _pertes.append(_perte.item())
        with torch.no_grad():
            _acc = (_m(Xte).argmax(1) == yte).float().mean().item()
        resultats_mnist[_nom] = dict(pertes=np.array(_pertes), precision=_acc, duree=time.perf_counter() - _t1)
        print(f"{_nom:15s} précision test {_acc:.2%}, perte moyenne des 100 derniers lots "
              f"{np.mean(_pertes[-100:]):.3f}, {resultats_mnist[_nom]['duree']:.1f} s")
    duree_mnist = time.perf_counter() - _t0
    print(f"Cinq entraînements : {duree_mnist:.1f} s")
    return duree_mnist, resultats_mnist


@app.cell
def _(SMOKE, mo, np, plt, resultats_mnist, sauvegarder_figure):
    _fig, _ax = plt.subplots(figsize=(8, 4.2))
    _k = 25  # moyenne glissante sur 25 lots pour lisser le bruit des mini-lots
    for _nom, _r in resultats_mnist.items():
        _p = np.convolve(_r["pertes"], np.ones(_k) / _k, mode="valid")
        _ax.semilogy(_p, label=f"{_nom} ({_r['precision']:.1%})".replace(".", ","))
    _ax.set(xlabel="pas (lots de 128)", ylabel="perte d'entraînement (entropie croisée)",
            title="MLP sur MNIST : précision test entre parenthèses")
    _ax.legend(fontsize=9)
    _chemin = sauvegarder_figure(_fig, "mnist-smoke" if SMOKE else "mnist", mo.notebook_dir() / "figures")
    print(f"Figure : {_chemin[0]}")
    _fig


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Bonus : Muon, l'orthogonalisation en jouet

    Muon (K. Jordan, 2024) prend la mise à jour de SGD + Nesterov d'une **matrice** de
    poids et la remplace par la matrice orthogonale la plus proche ($U V^\top$ si
    $G = U S V^\top$) : toutes les directions avancent à la même vitesse, quelle que soit
    la taille de leur valeur singulière. Plutôt qu'une SVD, coûteuse, il fait 5 itérations
    de Newton-Schulz, de simples produits de matrices :
    $A = X X^\top$, $X \leftarrow aX + (bA + cA^2)X$ avec $(a, b, c) = (3{,}4445 ; -4{,}7750 ; 2{,}0315)$.
    Ces coefficients ne convergent pas exactement vers 1 : les valeurs singulières
    finissent quelque part entre 0,5 et 1,5 environ, ce qui suffit en pratique.
    """)


@app.cell
def _(config, np, torch):
    def newton_schulz(G, n_iter=5):
        """Orthogonalisation approchée de G (version numpy, en float64)."""
        a, b, c = 3.4445, -4.7750, 2.0315
        haute = G.shape[0] > G.shape[1]
        X = G.T if haute else G  # on travaille avec la forme « large »
        X = X / (np.linalg.norm(X) + 1e-7)  # norme de Frobenius : valeurs singulières ≤ 1
        for _ in range(n_iter):
            A = X @ X.T
            X = a * X + (b * A + c * A @ A) @ X
        return X.T if haute else X

    # Jouet : régression matricielle W (64 × 32), perte ½ tr((W − W*) Σ (W − W*)ᵀ),
    # avec Σ mal conditionnée (valeurs propres de 1 à 10^-2,5) et tournée au hasard.
    _rng = np.random.default_rng(42)
    _m, _n = 64, 32
    _U, _ = np.linalg.qr(_rng.normal(size=(_n, _n)))
    _Sigma = _U @ np.diag(np.logspace(0, -2.5, _n)) @ _U.T
    _Wstar = _rng.normal(size=(_m, _n))

    def perte_mat(W):
        D = W - _Wstar
        return 0.5 * np.trace(D @ _Sigma @ D.T)

    def grad_mat(W):
        return (W - _Wstar) @ _Sigma

    # Ce que fait Newton-Schulz aux valeurs singulières du gradient
    _G0 = grad_mat(np.zeros((_m, _n)))
    sv_avant = np.linalg.svd(_G0, compute_uv=False)
    sv_apres = np.linalg.svd(newton_schulz(_G0), compute_uv=False)
    print(f"Valeurs singulières du gradient : de {sv_avant.min():.3f} à {sv_avant.max():.1f} "
          f"(rapport {sv_avant.max() / sv_avant.min():.0f})")
    print(f"Après Newton-Schulz : de {sv_apres.min():.2f} à {sv_apres.max():.2f}")

    # Un pas de notre Muon numpy contre torch.optim.Muon (qui calcule en bfloat16)
    def pas_muon(W, g, buf, lr, mu=0.95):
        buf = buf + (1 - mu) * (g - buf)  # moyenne glissante (lerp, comme PyTorch)
        u = g + mu * (buf - g)  # Nesterov
        echelle = np.sqrt(max(1.0, W.shape[0] / W.shape[1]))  # réglage « original » de Jordan
        return W - lr * echelle * newton_schulz(u), buf

    _W1, _ = pas_muon(np.zeros((_m, _n)), _G0, np.zeros((_m, _n)), lr=0.1)
    _q = torch.zeros(_m, _n, requires_grad=True)
    _opt = torch.optim.Muon([_q], lr=0.1, weight_decay=0.0, momentum=0.95, nesterov=True)
    _q.grad = torch.tensor(_G0, dtype=torch.float32)
    _opt.step()
    # torch calcule Newton-Schulz en bfloat16 (8 bits de mantisse) : on compare donc la
    # direction des deux mises à jour (cosinus) et leur écart relatif global (Frobenius).
    _a, _b = _q.detach().numpy().ravel().astype(float), _W1.ravel()
    cos_muon_torch = float(_a @ _b / (np.linalg.norm(_a) * np.linalg.norm(_b)))
    ecart_muon_torch = float(np.linalg.norm(_a - _b) / np.linalg.norm(_b))
    print(f"numpy (float64) / torch.optim.Muon (bfloat16), un pas : cosinus {cos_muon_torch:.4f}, "
          f"écart relatif {ecart_muon_torch:.1%}")

    # Comparaison sur le jouet : chaque méthode avec son meilleur η sur une petite grille,
    # η décroissant linéairement jusqu'à 0 (Muon fait des pas de taille fixe : sans
    # décroissance, il tourne autour du minimum).
    def entrainer_mat(methode, eta0, n=config["pas_muon"]):
        W = np.zeros((_m, _n))
        buf, mm, vv = np.zeros_like(W), np.zeros_like(W), np.zeros_like(W)
        for t in range(1, n + 1):
            g = grad_mat(W)
            eta = eta0 * (1 - (t - 1) / n)
            if methode == "Nesterov":
                buf = 0.9 * buf + g
                W = W - eta * (g + 0.9 * buf)
            elif methode == "Adam":
                mm = 0.9 * mm + 0.1 * g
                vv = 0.999 * vv + 0.001 * g * g
                W = W - eta * (mm / (1 - 0.9**t)) / (np.sqrt(vv / (1 - 0.999**t)) + 1e-8)
            else:
                W, buf = pas_muon(W, g, buf, eta)
        return perte_mat(W)

    GRILLES = {"Nesterov": [0.1, 0.2, 0.4], "Adam": [0.1, 0.3, 1.0], "Muon": [0.1, 0.2, 0.4, 0.8]}
    jouet_muon = {}
    for _meth, _grille in GRILLES.items():
        _res = {_e: entrainer_mat(_meth, _e) for _e in _grille}
        _best = min(_res, key=_res.get)
        jouet_muon[_meth] = (_best, _res[_best])
        print(f"{_meth:9s} meilleur η = {_best:<4} → perte finale {_res[_best]:.2e} (départ {perte_mat(np.zeros((_m, _n))):.0f})")
    return cos_muon_torch, ecart_muon_torch, jouet_muon, sv_apres, sv_avant


@app.cell
def _(SMOKE, mo, np, plt, sauvegarder_figure, sv_apres, sv_avant):
    _fig, _ax = plt.subplots(figsize=(7, 3.8))
    _i = np.arange(1, len(sv_avant) + 1)
    _ax.semilogy(_i, sv_avant / sv_avant.max(), "o-", ms=3, label="gradient (normalisé par sa plus grande valeur)")
    _ax.semilogy(_i, sv_apres, "s-", ms=3, label="après 5 itérations de Newton-Schulz")
    _ax.set(xlabel="rang de la valeur singulière", ylabel="valeur singulière",
            title="Muon égalise les directions d'une mise à jour")
    _ax.legend(fontsize=9)
    _chemin = sauvegarder_figure(_fig, "muon-smoke" if SMOKE else "muon", mo.notebook_dir() / "figures")
    print(f"Figure : {_chemin[0]}")
    _fig


@app.cell
def _(
    SMOKE,
    cos_muon_torch,
    ecarts_torch,
    ecretage,
    jouet_muon,
    lbfgs,
    lignes_tableau,
    mo,
    newton_iter,
    np,
    resultats_eta,
    resultats_mnist,
    sv_apres,
    COURBES_LOTS,
    JOUET_L2,
    TRAJ,
):
    # --- Bilan : les seuils du README. En smoke, on vérifie seulement que tout tourne. ---
    def _pas(surface, opt):
        return next(l["pas_sous_seuil"] for l in lignes_tableau if l["surface"] == surface and l["optimiseur"] == opt)

    _dist = {k: float(np.linalg.norm(TRAJ["rosenbrock"][k][-1] - 1.0)) for k in TRAJ["rosenbrock"]}
    _c = COURBES_LOTS
    _cos = _c["lot de 32, η max 0,4, échauffement + cosinus"][-1]
    _const = _c["lot de 32, η = 0,1 constant"][-1]
    _acc = {k: r["precision"] for k, r in resultats_mnist.items()}
    controles = {
        "Rosenbrock 3000 pas : SGD à plus de 0,1 de (1, 1)": _dist["SGD"] > 0.1,
        "Rosenbrock 3000 pas : momentum, Nesterov, Adam à moins de 0,01": max(_dist["momentum"], _dist["Nesterov"], _dist["Adam"]) < 0.01,
        "vallée : momentum au moins 2 fois plus rapide que SGD (même η)": _pas("vallée étroite", "momentum") * 2 <= _pas("vallée étroite", "SGD"),
        "selle : Adam et RMSProp sortent au moins 3 fois plus vite que SGD": max(_pas("point selle", "Adam"), _pas("point selle", "RMSProp")) * 3 <= _pas("point selle", "SGD"),
        "vallée : SGD diverge à η = 0,041 et converge à 0,039": resultats_eta[0.041][0] < 300 and resultats_eta[0.039][1] < 1e-6,
        "écrêtage : stable à η = 3e-3, f < 0,2": ecretage["pas_avant_divergence"] < 100 and ecretage["f_avec"] < 0.2,
        "mini-lots : cosinus 100 fois sous le plancher à η constant": _cos * 100 < _const,
        "Adam + L2 : w2 identique pour λ = 0,01 et 0,1": abs(JOUET_L2[(False, 0.01)][100, 1] - JOUET_L2[(False, 0.1)][100, 1]) < 1e-3,
        "AdamW : w2 dépend de λ": JOUET_L2[(True, 0.1)][-1, 1] < JOUET_L2[(True, 0.01)][-1, 1] - 0.2,
        "Newton < 10 itérations, L-BFGS < 60": newton_iter is not None and newton_iter < 10 and lbfgs["iterations"] < 60 and lbfgs["distance"] < 1e-6,
        "numpy = torch.optim à 1e-10 près": max(ecarts_torch.values()) < 1e-10,
        "MNIST : Adam ≥ 96,5 %, SGD + Nesterov > SGD": _acc["Adam"] >= 0.965 and _acc["SGD + Nesterov"] > _acc["SGD"],
        "Muon : valeurs singulières dans [0,5 ; 1,5], cosinus avec torch > 0,99": 0.5 <= sv_apres.min() and sv_apres.max() <= 1.5 and cos_muon_torch > 0.99,
        "jouet matriciel : Muon et Adam sous 1e-3": jouet_muon["Muon"][1] < 1e-3 and jouet_muon["Adam"][1] < 1e-3,
    }
    for _k, _ok in controles.items():
        print(("OK    " if _ok else ("—     " if SMOKE else "ÉCHEC ")) + _k)
    _echecs = [k for k, ok in controles.items() if not ok]
    if not SMOKE and _echecs:
        raise SystemExit(f"Seuils non atteints : {_echecs}")
    mo.md(f"**Bilan** : {len(controles) - len(_echecs)}/{len(controles)} contrôles passés"
          + (" (mode smoke : seuils non exigés)." if SMOKE else "."))


if __name__ == "__main__":
    app.run()
