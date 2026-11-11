# Notebook 39-autodiff-maison (chapitre 39, jacobienne, règle de la chaîne multivariée, graphe
# de calcul, différentiation automatique).
#
# Une différentiation automatique en mode inverse écrite en quelques dizaines de lignes : une
# classe Valeur qui enregistre le graphe de calcul pendant la passe avant, puis backward() qui
# le parcourt à rebours (tri topologique) en additionnant les contributions des enfants. On la
# vérifie sur les exemples du cours, puis par différences centrées, et l’on entraîne le réseau
# 2-2-1 du cours (sigmoïde partout) sur le problème du « ou exclusif ».
#
#   make nb NUM=39              → éditeur marimo (bouton « Lancer le calcul »)
#   make nb NUM=39 MODE=run     → script complet
#   make nb NUM=39 MODE=smoke   → test rapide (moins de 30 s)

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
    import sympy as sp
    from math_commun import is_smoke, sauvegarder_figure, seed_everything, style_commun

    return is_smoke, math, mo, np, plt, sauvegarder_figure, seed_everything, sp, style_commun, time


@app.cell
def _(mo):
    mo.md(r"""
    # Une différentiation automatique maison

    **Chapitre 39, sections 3, 4 et 9.** Une bibliothèque comme PyTorch enregistre le **graphe
    de calcul** pendant la passe avant, puis `loss.backward()` le parcourt à l’envers. On refait
    la même chose sur des nombres, avec une classe `Valeur` :

    1. chaque opération (`+`, `-`, `*`, `**`, sigmoïde, $-\ln$) crée un nœud qui retient ses
       parents et ses **dérivées locales** ;
    2. `backward()` range les nœuds dans un ordre où chacun vient après ses parents (tri
       topologique), pose $\bar{L} = 1$, puis remonte en ajoutant à chaque parent
       $\bar{u} \times \partial u / \partial v$ : un nœud qui sert deux fois reçoit la **somme**
       de deux contributions.

    On vérifie ensuite le moteur sur les exemples du cours ($\bar{w} = -18$, $\bar{w} = -22$,
    le neurone, le réseau 2-2-1), puis par **différences centrées** sur des réseaux tirés au
    hasard, et l’on s’en sert pour entraîner le réseau 2-2-1 sur le « ou exclusif ».
    """)


@app.cell
def _(is_smoke, np, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        n_reseaux=5 if SMOKE else 50,     # réseaux 2-2-1 tirés au hasard pour la vérification
        h=1e-5,                           # pas des différences centrées
        epoques=1_500 if SMOKE else 6_000,  # pas de descente sur le « ou exclusif »
        eta=2.0,                          # taux d’apprentissage
    )
    rng = np.random.default_rng(SEED)
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SMOKE, config, rng


@app.cell
def _(math):
    class Valeur:
        """Un nœud du graphe de calcul : une valeur, ses parents et ses dérivées locales."""

        def __init__(self, donnee, parents=(), locales=()):
            self.donnee = float(donnee)
            self.grad = 0.0           # l’adjoint v̄ = ∂L/∂v, rempli par backward()
            self.parents = parents    # nœuds dont dépend celui-ci
            self.locales = locales    # ∂(ce nœud)/∂(parent), dans le même ordre

        @staticmethod
        def _v(x):
            return x if isinstance(x, Valeur) else Valeur(x)

        def __add__(self, autre):
            autre = Valeur._v(autre)
            return Valeur(self.donnee + autre.donnee, (self, autre), (1.0, 1.0))

        __radd__ = __add__

        def __mul__(self, autre):
            autre = Valeur._v(autre)
            # produit : le facteur échangé
            return Valeur(self.donnee * autre.donnee, (self, autre), (autre.donnee, self.donnee))

        __rmul__ = __mul__

        def __neg__(self):
            return self * -1.0

        def __sub__(self, autre):
            return self + (-Valeur._v(autre))

        def __rsub__(self, autre):
            return Valeur._v(autre) + (-self)

        def __pow__(self, k):
            return Valeur(self.donnee**k, (self,), (k * self.donnee ** (k - 1),))

        def sigmoide(self):
            s = 1.0 / (1.0 + math.exp(-self.donnee))
            return Valeur(s, (self,), (s * (1.0 - s),))

        def moinsln(self):
            return Valeur(-math.log(self.donnee), (self,), (-1.0 / self.donnee,))

        def backward(self):
            # Tri topologique : chaque nœud après ses parents.
            ordre, vus = [], set()

            def visiter(v):
                if id(v) not in vus:
                    vus.add(id(v))
                    for p in v.parents:
                        visiter(p)
                    ordre.append(v)

            visiter(self)
            for v in ordre:
                v.grad = 0.0
            self.grad = 1.0  # L̄ = 1
            for v in reversed(ordre):
                for p, d in zip(v.parents, v.locales, strict=True):
                    p.grad += v.grad * d  # les contributions des enfants s’additionnent

    return (Valeur,)


@app.cell
def _(Valeur, math):
    # --- Les exemples du cours ---
    def ouverture():
        w, x, b, y = Valeur(2), Valeur(3), Valeur(1), Valeur(10)
        L = ((w * x + b) - y) ** 2
        L.backward()
        return L.donnee, w.grad, b.grad

    def penalite():
        w, x, y = Valeur(2), Valeur(3), Valeur(10)
        L = (w * x - y) ** 2 + 0.5 * w**2
        L.backward()
        return L.donnee, w.grad

    def neurone():
        w, x, b = Valeur(1), Valeur(2), Valeur(-1)
        L = (w * x + b).sigmoide().moinsln()
        L.backward()
        return L.donnee, w.grad, b.grad

    exemples = dict(ouverture=ouverture(), penalite=penalite(), neurone=neurone())
    attendus = dict(
        ouverture=(9.0, -18.0, -6.0),
        penalite=(18.0, -22.0),
        neurone=(-math.log(1 / (1 + math.exp(-1))), 2 * (1 / (1 + math.exp(-1)) - 1),
                 1 / (1 + math.exp(-1)) - 1),
    )
    ecart_cours = max(abs(a - b) for _k in exemples for a, b in zip(exemples[_k], attendus[_k], strict=True))
    for _k, _v in exemples.items():
        print(_k, "→", ", ".join(f"{t:.4f}" for t in _v))
    print(f"Écart maximal avec le cours : {ecart_cours:.1e}")
    return (ecart_cours,)


@app.cell
def _(Valeur, np):
    def reseau_221(W, b, v, c, x, y):
        """Réseau 2-2-1, convention colonne : z = Wx + b, h = σ(z), ŷ = σ(vᵀh + c), L = log-loss."""
        h = [(W[i][0] * x[0] + W[i][1] * x[1] + b[i]).sigmoide() for i in range(2)]
        yh = (v[0] * h[0] + v[1] * h[1] + c).sigmoide()
        if y == 1:
            return yh.moinsln()
        return (1 - yh).moinsln()

    def gradients_valeur(params, x, y):
        """Gradients par la classe Valeur : (W, b, v, c) à plat, 9 nombres."""
        P = [Valeur(p) for p in params]
        W, b, v, c = [[P[0], P[1]], [P[2], P[3]]], [P[4], P[5]], [P[6], P[7]], P[8]
        L = reseau_221(W, b, v, c, x, y)
        L.backward()
        return L.donnee, np.array([p.grad for p in P])

    def perte_numpy(params, x, y):
        W = np.array(params[:4]).reshape(2, 2)
        b, v, c = np.array(params[4:6]), np.array(params[6:8]), params[8]
        h = 1 / (1 + np.exp(-(W @ x + b)))
        yh = 1 / (1 + np.exp(-(v @ h + c)))
        return -np.log(yh) if y == 1 else -np.log(1 - yh)

    def gradients_formules(params, x, y):
        """Formules de la section 6 : δ₂ = ŷ − y, ∂L/∂v = δ₂h, δ₁ = (vδ₂) ⊙ h(1 − h), ∂L/∂W = δ₁xᵀ."""
        W = np.array(params[:4]).reshape(2, 2)
        b, v, c = np.array(params[4:6]), np.array(params[6:8]), params[8]
        h = 1 / (1 + np.exp(-(W @ x + b)))
        yh = 1 / (1 + np.exp(-(v @ h + c)))
        d2 = yh - y
        d1 = v * d2 * h * (1 - h)
        return np.concatenate([np.outer(d1, x).ravel(), d1, d2 * h, [d2]])

    return gradients_formules, gradients_valeur, perte_numpy


@app.cell
def _(gradients_formules, gradients_valeur, np):
    # --- Le réseau 2-2-1 du cours (section 6) ---
    params_cours = [0.5, 0.3, -0.5, 0.8, 0.0, 0.0, 1.0, -1.0, 0.0]
    x_cours = np.array([1.0, 0.0])
    L_cours, g_cours = gradients_valeur(params_cours, x_cours, 1)
    ecart_221 = float(np.max(np.abs(g_cours - gradients_formules(params_cours, x_cours, 1))))
    print(f"2-2-1 du cours : L = {L_cours:.4f} ; ∂L/∂W = {np.round(g_cours[:4], 4)} ; "
          f"∂L/∂v = {np.round(g_cours[6:8], 4)} ; écart aux formules {ecart_221:.1e}")
    return L_cours, ecart_221, g_cours


@app.cell
def _(mo):
    # Le bouton empêche le calcul de se relancer à chaque modification d’une cellule amont.
    bouton = mo.ui.run_button(label="Lancer le calcul")
    bouton  # noqa: B018 (marimo affiche la dernière expression)
    return (bouton,)


@app.cell
def _(bouton, config, gradients_valeur, mo, np, perte_numpy, rng, time):
    # En script (make nb MODE=run|smoke), personne ne clique : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value, mo.md("*Cliquez sur « Lancer le calcul ».*"))

    t0 = time.perf_counter()
    # 1. Vérification par différences centrées sur des réseaux tirés au hasard.
    _h = config["h"]
    _erreurs = []
    for _ in range(config["n_reseaux"]):
        _params = rng.normal(0, 1.5, 9)
        _x = rng.normal(0, 1, 2)
        _y = int(rng.integers(0, 2))
        _, _g = gradients_valeur(list(_params), _x, _y)
        _g_num = np.empty(9)
        for _k in range(9):
            _plus, _moins = _params.copy(), _params.copy()
            _plus[_k] += _h
            _moins[_k] -= _h
            _g_num[_k] = (perte_numpy(_plus, _x, _y) - perte_numpy(_moins, _x, _y)) / (2 * _h)
        _erreurs.append(np.max(np.abs(_g - _g_num) / np.maximum(np.maximum(np.abs(_g), np.abs(_g_num)), 1e-8)))
    erreur_rel_max = float(np.max(_erreurs))

    # 2. Entraînement du 2-2-1 sur le « ou exclusif » (XOR), descente de gradient sur la perte moyenne.
    _X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    _Y = [0, 1, 1, 0]
    _theta = rng.normal(0, 1, 9)
    pertes = []
    for _ in range(config["epoques"]):
        _total, _grad = 0.0, np.zeros(9)
        for _xi, _yi in zip(_X, _Y, strict=True):
            _Li, _gi = gradients_valeur(list(_theta), _xi, _yi)
            _total += _Li / 4
            _grad += _gi / 4
        pertes.append(_total)
        _theta = _theta - config["eta"] * _grad
    _sorties = [1 / (1 + np.exp(-(_theta[6:8] @ (1 / (1 + np.exp(-(_theta[:4].reshape(2, 2) @ _xi + _theta[4:6]))))
                                  + _theta[8]))) for _xi in _X]
    justes = sum(int((_s > 0.5) == bool(_yi)) for _s, _yi in zip(_sorties, _Y, strict=True))
    duree = time.perf_counter() - t0
    print(f"Différences centrées : erreur relative maximale {erreur_rel_max:.1e} sur {config['n_reseaux']} réseaux")
    print(f"XOR : perte {pertes[0]:.4f} → {pertes[-1]:.4f}, sorties {np.round(_sorties, 3)}, "
          f"{justes}/4 justes ; {duree:.1f} s")
    return duree, erreur_rel_max, justes, pertes


@app.cell
def _(sp):
    # --- Vérification exacte (sympy) : les trois dérivées de la section 7 ---
    _z, _y = sp.symbols("z y", real=True)
    _sig = 1 / (1 + sp.exp(-_z))
    assert sp.simplify(sp.diff(_sig, _z) - _sig * (1 - _sig)) == 0
    _logloss = -_y * sp.log(_sig) - (1 - _y) * sp.log(1 - _sig)
    assert sp.simplify(sp.diff(_logloss, _z) - (_sig - _y)) == 0
    _zs = sp.symbols("z1:4", real=True)
    _S = sum(sp.exp(_t) for _t in _zs)
    _p = [sp.exp(_t) / _S for _t in _zs]
    for _c in range(3):
        _L = -sp.log(_p[_c])
        for _j in range(3):
            assert sp.simplify(sp.diff(_L, _zs[_j]) - (_p[_j] - (1 if _j == _c else 0))) == 0
    print("sympy : σ′ = σ(1 − σ), dL/dz = σ(z) − y et ∂(−ln p_c)/∂z = p − y vérifiés")
    return


@app.cell
def _(SMOKE, mo, pertes, plt, sauvegarder_figure):
    # --- Figure, sauvegardée dans figures/ (suffixe -smoke : n’écrase pas celle du run complet) ---
    _fig, _ax = plt.subplots()
    _ax.semilogy(range(1, len(pertes) + 1), pertes)
    _ax.set(xlabel="pas de descente", ylabel="perte moyenne (échelle log)",
           title="Le réseau 2-2-1 apprend le « ou exclusif »")
    _chemins = sauvegarder_figure(_fig, "perte-xor-smoke" if SMOKE else "perte-xor", mo.notebook_dir() / "figures")
    print(f"Figure : {_chemins[0].name}")
    _fig  # noqa: B018 (marimo affiche la dernière expression)
    return


@app.cell
def _(L_cours, SMOKE, duree, ecart_221, ecart_cours, erreur_rel_max, g_cours, justes, mo, pertes):
    # --- Bilan : seuils du README, vérifiés aussi en smoke ---
    _echecs = []
    if ecart_cours > 1e-12:
        _echecs.append(f"exemples du cours : écart {ecart_cours:.1e}")
    if abs(L_cours - 0.5782) > 5e-5 or abs(g_cours[0] + 0.1032) > 5e-5 or ecart_221 > 1e-12:
        _echecs.append(f"2-2-1 du cours : L = {L_cours:.5f}, ∂L/∂W₁₁ = {g_cours[0]:.5f}, écart {ecart_221:.1e}")
    if erreur_rel_max > 1e-6:
        _echecs.append(f"différences centrées : erreur relative {erreur_rel_max:.1e} (attendue ≤ 1e-6)")
    _seuil = 0.2 if SMOKE else 0.05
    if not pertes[-1] < _seuil:
        _echecs.append(f"XOR : perte finale {pertes[-1]:.3f} (attendue < {_seuil})")
    if justes != 4:
        _echecs.append(f"XOR : {justes}/4 justes")
    if _echecs:
        raise SystemExit("Bilan : ÉCHEC : " + " ; ".join(_echecs))
    print(f"Bilan : OK (exemples du cours exacts, 2-2-1 : L = {L_cours:.4f}, erreur relative "
          f"{erreur_rel_max:.1e}, XOR {justes}/4 avec une perte de {pertes[-1]:.3f}, {duree:.1f} s)")
    _err_fr = f"{erreur_rel_max:.1e}".replace(".", ",")
    _perte_fr = f"{pertes[-1]:.3f}".replace(".", ",")
    mo.md(f"**Bilan** : moteur exact sur les exemples du cours ; erreur relative maximale "
          f"{_err_fr} contre les différences centrées ; XOR appris ({justes}/4), perte finale {_perte_fr}.")
    return


if __name__ == "__main__":
    app.run()
