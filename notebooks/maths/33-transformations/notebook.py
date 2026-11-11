# Notebook 33-transformations (chapitre 33, Matrices comme transformations du plan, déterminant).
#
# Les transformations du chapitre en numpy : matrices de rotation, de cisaillement et de
# symétrie, image d’une grille de points et d’une « maison », composition dans les deux
# ordres (BA contre AB), np.linalg.det comparé à l’aire du polygone image (formule du
# lacet), déterminant nul qui écrase le plan, et rotations par blocs de RoPE.
#
#   make nb NUM=33              → éditeur marimo (bouton « Lancer le calcul »)
#   make nb NUM=33 MODE=run     → script complet
#   make nb NUM=33 MODE=smoke   → test rapide (moins de 30 s)

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import sympy as sp
    from math_commun import is_smoke, sauvegarder_figure, seed_everything, style_commun

    return is_smoke, mo, np, plt, sauvegarder_figure, seed_everything, sp, style_commun, time


@app.cell
def _(mo):
    mo.md(r"""
    # Transformations du plan en numpy : colonnes, composition, déterminant

    **Chapitre 33, sections 1 à 7.** Une matrice $2 \times 2$ transforme tout le plan ; ses
    colonnes sont les images de $\mathbf{e}_1$ et $\mathbf{e}_2$. Cinq expériences :

    1. **La grille.** Image d’une grille de points et d’une maison par une rotation, un
       cisaillement et une symétrie ; les colonnes sont `A @ e1` et `A @ e2`.
    2. **L’ordre.** Rotation d’un quart de tour $A$ et symétrie d’axe $(Ox)$ $B$ :
       « $A$ puis $B$ » donne $BA$, différent de $AB$.
    3. **Le déterminant mesure l’aire.** Pour des matrices et des polygones tirés au hasard,
       l’aire de l’image (formule du lacet) vaut $\lvert \det A \rvert$ fois l’aire de départ.
    4. **Le déterminant nul.** Une matrice de colonnes colinéaires envoie tous les points sur
       une droite.
    5. **RoPE.** Avec des rotations par blocs, le score $(R_m\mathbf{q}) \cdot (R_n\mathbf{k})$
       ne dépend que de l’écart $n - m$.
    """)


@app.cell
def _(is_smoke, np, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        n_essais=200 if SMOKE else 5_000,   # matrices et polygones tirés pour l’expérience 3
        n_sommets=5,                        # sommets des polygones tirés (étoilés autour d’un centre)
        dimension_rope=8 if SMOKE else 64,  # dimension d des vecteurs de l’expérience 5
        positions=50 if SMOKE else 500,     # couples (m, n) testés
    )
    rng = np.random.default_rng(SEED)
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SMOKE, config, rng


@app.cell
def _(np):
    # --- Outils : matrices usuelles, aire d’un polygone (formule du lacet) ---
    def rotation(theta):
        """Matrice R_θ : colonnes (cos θ ; sin θ) et (−sin θ ; cos θ)."""
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])

    def cisaillement(k):
        return np.array([[1.0, k], [0.0, 1.0]])

    def aire_orientee(points):
        """Aire orientée d’un polygone (sommets en lignes) : formule du lacet."""
        x, y = points[:, 0], points[:, 1]
        return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    MAISON = np.array([[0, 0], [1, 0], [1, 1], [0.5, 1.6], [0, 1]])   # sens direct
    return MAISON, aire_orientee, cisaillement, rotation


@app.cell
def _(MAISON, aire_orientee, cisaillement, np, rotation):
    # --- Expérience 1 : colonnes et images ---
    e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    A_ex = np.array([[2, -1], [1, 1]])
    colonnes = dict(
        Ae1=A_ex @ e1, Ae2=A_ex @ e2,
        image_32=A_ex @ np.array([3, 2]),            # (4 ; 5) dans le cours
    )
    transfos = {
        "rotation π/6": rotation(np.pi / 6),
        "cisaillement k = 1": cisaillement(1.0),
        "symétrie d’axe (Ox)": np.array([[1.0, 0.0], [0.0, -1.0]]),
    }
    maisons = {nom: MAISON @ M.T for nom, M in transfos.items()}   # un point par ligne : p ↦ M p
    aires_maison = {nom: aire_orientee(P) for nom, P in maisons.items()}
    print("A e1 =", colonnes["Ae1"], " A e2 =", colonnes["Ae2"], " A (3 ; 2) =", colonnes["image_32"])
    print(f"Aire orientée de la maison : {aire_orientee(MAISON):.3f}")
    for _nom, _a in aires_maison.items():
        print(f"  image par {_nom} : {_a:+.3f} (det = {np.linalg.det(transfos[_nom]):+.3f})")
    return aires_maison, colonnes, transfos


@app.cell
def _(mo):
    # Le bouton empêche le calcul de se relancer à chaque modification d’une cellule amont.
    bouton = mo.ui.run_button(label="Lancer le calcul")
    bouton  # noqa: B018 (marimo affiche la dernière expression)
    return (bouton,)


@app.cell
def _(aire_orientee, bouton, config, mo, np, rng, rotation, time):
    # En script (make nb MODE=run|smoke), personne ne clique : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer le calcul ».*"))

    t0 = time.perf_counter()
    # --- Expérience 2 : l’ordre de composition ---
    _A = rotation(np.pi / 2)                       # A : quart de tour
    _B = np.array([[1.0, 0.0], [0.0, -1.0]])       # B : symétrie d’axe (Ox)
    _x = np.array([1.0, 0.0])
    ordre = dict(
        A_puis_B=_B @ (_A @ _x), BA=np.round(_B @ _A, 12), AB=np.round(_A @ _B, 12),
    )
    # --- Expérience 3 : |det A| facteur d’aire, sur des polygones étoilés tirés au hasard ---
    _ecarts = []
    for _ in range(config["n_essais"]):
        _M = rng.normal(size=(2, 2))
        _angles = np.sort(rng.uniform(0, 2 * np.pi, config["n_sommets"]))
        _rayons = rng.uniform(0.5, 2.0, config["n_sommets"])
        _P = np.c_[_rayons * np.cos(_angles), _rayons * np.sin(_angles)]   # sens direct
        _avant, _apres = aire_orientee(_P), aire_orientee(_P @ _M.T)
        _ecarts.append(abs(_apres - np.linalg.det(_M) * _avant) / max(1.0, abs(_apres)))
    aire = dict(ecart_max=float(np.max(_ecarts)), n=config["n_essais"])
    # --- Expérience 4 : déterminant nul ---
    _C = np.array([[1.0, 2.0], [2.0, 4.0]])
    _pts = rng.uniform(-3, 3, size=(200, 2))
    _img = _pts @ _C.T
    nul = dict(
        det=float(np.linalg.det(_C)),
        hors_droite=float(np.abs(_img[:, 1] - 2 * _img[:, 0]).max()),   # images sur y = 2x
        meme_image=(_C @ np.array([2.0, 0.0])).tolist(), meme_image_2=(_C @ np.array([0.0, 1.0])).tolist(),
    )
    # --- Expérience 5 : RoPE, rotations par blocs ---
    _d = config["dimension_rope"]
    _theta = 10_000.0 ** (-2 * np.arange(_d // 2) / _d)   # θᵢ = 10000^(−2(i−1)/d), i = 1…d/2

    def _rope(v, pos):
        _paires = v.reshape(-1, 2)
        _c, _s = np.cos(pos * _theta), np.sin(pos * _theta)
        return np.c_[_c * _paires[:, 0] - _s * _paires[:, 1], _s * _paires[:, 0] + _c * _paires[:, 1]].ravel()

    _q, _k = rng.normal(size=_d), rng.normal(size=_d)
    _ecarts_rope = []
    for _ in range(config["positions"]):
        _m, _n = rng.integers(0, 4_096, size=2)
        _decalage = rng.integers(0, 4_096)
        _s1 = _rope(_q, _m) @ _rope(_k, _n)
        _s2 = _rope(_q, _m + _decalage) @ _rope(_k, _n + _decalage)
        _ecarts_rope.append(abs(_s1 - _s2))
    _p = rng.normal(size=_d)
    rope = dict(
        ecart_max=float(np.max(_ecarts_rope)),
        norme=float(abs(np.linalg.norm(_rope(_p, 123)) - np.linalg.norm(_p))),
        periode_lente=float(2 * np.pi / (10_000.0 ** (-(128 - 2) / 128))),   # d = 128, paire la plus lente
    )
    duree = time.perf_counter() - t0
    print(f"A puis B envoie e1 sur {ordre['A_puis_B']} ; BA =\n{ordre['BA']}\nAB =\n{ordre['AB']}")
    print(f"Aire : {aire['n']} essais, écart relatif max à |det A| × aire : {aire['ecart_max']:.1e}")
    print(f"Déterminant nul : det = {nul['det']:.1e}, distance max à la droite y = 2x : {nul['hors_droite']:.1e} ; "
          f"(2 ; 0) ↦ {nul['meme_image']} et (0 ; 1) ↦ {nul['meme_image_2']}")
    print(f"RoPE (d = {_d}) : écart max du score après décalage {rope['ecart_max']:.1e}, "
          f"norme conservée à {rope['norme']:.1e} près ; période de la paire la plus lente (d = 128) "
          f"≈ {rope['periode_lente']:,.0f} positions ; calcul {duree:.2f} s")
    return aire, duree, nul, ordre, rope


@app.cell
def _(sp):
    # --- Vérification exacte (sympy) ---
    _a, _b, _c, _d, _e, _f, _g, _h = sp.symbols("a b c d e f g h")
    _A = sp.Matrix([[_a, _b], [_c, _d]])
    _B = sp.Matrix([[_e, _f], [_g, _h]])
    assert sp.expand((_A * _B).det() - _A.det() * _B.det()) == 0            # det(AB) = det A det B
    _s, _t = sp.symbols("s t", real=True)
    def _R(x):
        return sp.Matrix([[sp.cos(x), -sp.sin(x)], [sp.sin(x), sp.cos(x)]])
    assert sp.simplify(sp.expand_trig(_R(_s) * _R(_t) - _R(_s + _t))) == sp.zeros(2, 2)   # R_a R_b = R_{a+b}
    assert sp.simplify(_R(_s).T - _R(-_s)) == sp.zeros(2, 2)                 # R_θᵀ = R_{−θ}
    assert sp.simplify(_R(_s).det()) == 1
    _u = sp.Matrix([2, 1])
    _P = _u * _u.T / (_u.T * _u)[0]
    assert _P * sp.Matrix([5, 0]) == sp.Matrix([4, 2]) and _P * _u == _u      # projection du cours
    assert sp.Matrix([[2, 1, 5], [0, 3, -1], [0, 0, sp.Rational(1, 2)]]).det() == 3
    print("sympy : det(AB) = det A det B, R_a R_b = R_{a+b}, R_θᵀ = R_{−θ}, det R_θ = 1, "
          "projection sur (2 ; 1), déterminant triangulaire = 3")
    return


@app.cell
def _(MAISON, SMOKE, mo, np, plt, sauvegarder_figure, transfos):
    # --- Figure (suffixe -smoke : n’écrase pas celle du run complet) ---
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 4.2))
    _k = np.arange(-3, 4)
    for _ax, (_nom, _M) in zip(_axes, transfos.items()):
        for _v in _k:                                   # grille transformée
            _l1 = np.c_[np.full(50, _v), np.linspace(-3, 3, 50)] @ _M.T
            _l2 = np.c_[np.linspace(-3, 3, 50), np.full(50, _v)] @ _M.T
            _ax.plot(_l1[:, 0], _l1[:, 1], color="0.8", lw=0.8)
            _ax.plot(_l2[:, 0], _l2[:, 1], color="0.8", lw=0.8)
        _ax.fill(*MAISON.T, alpha=0.15, ls="--", ec="C0", label="départ")
        _img = MAISON @ _M.T
        _ax.fill(*_img.T, alpha=0.4, color="C1", label="image")
        _ax.annotate("", xy=_M[:, 0], xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="C3", lw=2))
        _ax.annotate("", xy=_M[:, 1], xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="C2", lw=2))
        _ax.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), aspect="equal",
                title=f"{_nom}\ndet = {np.linalg.det(_M):+.2f}")
    _axes[0].legend(loc="lower left")
    _chemins = sauvegarder_figure(_fig, "transformations-smoke" if SMOKE else "transformations",
                                 mo.notebook_dir() / "figures")
    print(f"Figure : {_chemins[0].name}")
    _fig  # noqa: B018 (marimo affiche la dernière expression)
    return


@app.cell
def _(aire, aires_maison, colonnes, duree, mo, np, nul, ordre, rope):
    # --- Bilan : seuils du README, vérifiés aussi en smoke ---
    _echecs = []
    if not (np.allclose(colonnes["Ae1"], [2, 1]) and np.allclose(colonnes["Ae2"], [-1, 1])
            and np.allclose(colonnes["image_32"], [4, 5])):
        _echecs.append(f"colonnes et image de (3 ; 2) : {colonnes}")
    if not (np.isclose(aires_maison["rotation π/6"], 1.3) and np.isclose(aires_maison["cisaillement k = 1"], 1.3)
            and np.isclose(aires_maison["symétrie d’axe (Ox)"], -1.3)):
        _echecs.append(f"aires de la maison (1,3 ; 1,3 ; −1,3 attendues) : {aires_maison}")
    if not (np.allclose(ordre["A_puis_B"], [0, -1]) and np.array_equal(ordre["BA"], [[0, -1], [-1, 0]])
            and np.array_equal(ordre["AB"], [[0, 1], [1, 0]])):
        _echecs.append(f"ordre de composition : {ordre}")
    if aire["ecart_max"] > 1e-9:
        _echecs.append(f"aire de l’image ≠ |det| × aire : écart {aire['ecart_max']:.1e}")
    if abs(nul["det"]) > 1e-12 or nul["hors_droite"] > 1e-9 or nul["meme_image"] != nul["meme_image_2"]:
        _echecs.append(f"déterminant nul : {nul}")
    if rope["ecart_max"] > 1e-8 or rope["norme"] > 1e-9 or not 54_000 < rope["periode_lente"] < 55_000:
        _echecs.append(f"RoPE : {rope}")
    if _echecs:
        raise SystemExit("Bilan : ÉCHEC : " + " ; ".join(_echecs))
    print(f"Bilan : OK (colonnes (2 ; 1) et (−1 ; 1), image (4 ; 5) ; BA ≠ AB ; aire = |det| × aire "
          f"à {aire['ecart_max']:.0e} près sur {aire['n']} essais ; det nul : tout sur y = 2x ; "
          f"RoPE invariant par décalage à {rope['ecart_max']:.0e} près ; {duree:.2f} s)")
    mo.md(f"**Bilan** : sur {aire['n']} matrices et polygones tirés au hasard, l’aire de l’image vaut "
          f"$\\lvert\\det A\\rvert$ fois l’aire de départ (écart relatif maximal {aire['ecart_max']:.0e}) ; "
          f"avec RoPE, décaler les deux positions change le score d’au plus {rope['ecart_max']:.0e}.")
    return


if __name__ == "__main__":
    app.run()
