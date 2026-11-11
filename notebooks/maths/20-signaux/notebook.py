# Notebook 20-signaux (chapitre 20, Trigonométrie : cercle, radians, sinus et cosinus).
#
# Signaux sinusoïdaux : période d’une somme (sin 2t + sin 3t), amplitude et phase retrouvées
# par la formule d’addition ; encodage de position sinusoïdal d’un transformer (carte p × i,
# périodes en progression géométrique, décalage = rotation) ; erreur de sin θ ≈ θ ;
# approximation d’un créneau par des sommes de sinusoïdes.
#
#   make nb NUM=20              → éditeur marimo (bouton « Lancer le calcul »)
#   make nb NUM=20 MODE=run     → script complet
#   make nb NUM=20 MODE=smoke   → test rapide (moins de 30 s)

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
    # Signaux : sinusoïdes, encodage de position, petits angles, créneau

    **Chapitre 20, sections 3, 5, 6, 7, 8 et 9.** Quatre expériences, tous les angles en radians.

    1. **Période d’une somme.** $s(t) = \sin(2t) + \sin(3t)$ : on cherche numériquement la plus
       petite durée $T \gt 0$ telle que $s(t + T) = s(t)$ ; le cours annonce $2\pi$.
    2. **Amplitude et phase.** Un signal bruité $A \sin(\omega t + \varphi)$ s’écrit, par la formule
       d’addition, $a \sin(\omega t) + b \cos(\omega t)$ avec $a = A \cos \varphi$ et
       $b = A \sin \varphi$ : on retrouve $a$ et $b$ par moindres carrés, donc $A$ et $\varphi$.
    3. **Encodage de position.** $PE(p, 2i) = \sin(p / 10\,000^{2i/d})$,
       $PE(p, 2i + 1) = \cos(p / 10\,000^{2i/d})$ : carte des valeurs, périodes de chaque paire,
       et vérification que $PE(p + k)$ est $PE(p)$ tourné d’un angle qui ne dépend que de $k$.
    4. **Petits angles et créneau.** Erreur relative de $\sin \theta \approx \theta$ ; sommes
       $\sin t + \sin(3t)/3 + \sin(5t)/5 + \cdots$ comparées au créneau de hauteur $\pi/4$.
    """)


@app.cell
def _(is_smoke, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)  # le bruit de l’expérience 2 est tiré avec cette graine
    style_commun()
    config = dict(
        pas_t=1e-3 if SMOKE else 1e-4,            # grille pour chercher la période
        d=64 if SMOKE else 512,                     # dimension de l’encodage de position
        positions=256 if SMOKE else 2048,           # positions p = 0 … positions − 1
        termes=[1, 2, 5, 20, 50] if SMOKE else [1, 2, 5, 20, 50, 200, 1000],
        bruit=0.1,
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SEED, SMOKE, config


@app.cell
def _(mo):
    # Le bouton empêche le calcul de se relancer à chaque modification d’une cellule amont.
    bouton = mo.ui.run_button(label="Lancer le calcul")
    bouton  # noqa: B018 (marimo affiche la dernière expression)
    return (bouton,)


@app.cell
def _(SEED, bouton, config, mo, np, time):
    # En script (make nb MODE=run|smoke), personne ne clique : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer le calcul ».*"))
    t0 = time.perf_counter()

    # --- 1. Période de sin 2t + sin 3t : plus petit T de la grille où s(t + T) ≈ s(t) partout.
    # On part de T = 0,1 : un décalage minuscule change très peu s, sans être une période.
    def s(t):
        return np.sin(2 * t) + np.sin(3 * t)

    _t = np.linspace(0, 2 * np.pi, 400)
    _candidats = np.arange(0.1, 2 * np.pi + 0.01, config["pas_t"])
    _ecarts = np.array([np.max(np.abs(s(_t + T) - s(_t))) for T in _candidats])
    periode = float(_candidats[np.argmax(_ecarts < 1e-2)])
    ecart_pi = float(np.max(np.abs(s(_t + np.pi) - s(_t))))  # π n’est pas une période

    # --- 2. Amplitude et phase par moindres carrés : A sin(ωt + φ) = a sin ωt + b cos ωt ---
    rng = np.random.default_rng(SEED)
    A_vrai, f_vrai, phi_vrai = 1.5, 2.0, np.pi / 3
    omega = 2 * np.pi * f_vrai
    _tt = np.linspace(0, 2, 500)
    mesure = A_vrai * np.sin(omega * _tt + phi_vrai) + config["bruit"] * rng.standard_normal(_tt.size)
    _X = np.column_stack([np.sin(omega * _tt), np.cos(omega * _tt)])
    (a, b), *_ = np.linalg.lstsq(_X, mesure, rcond=None)
    A_est, phi_est = float(np.hypot(a, b)), float(np.arctan2(b, a))

    # --- 3. Encodage de position ---
    d, P = config["d"], config["positions"]
    p = np.arange(P)[:, None]
    i = np.arange(d // 2)[None, :]
    lam = 10000.0 ** (2 * i / d)
    PE = np.empty((P, d))
    PE[:, 0::2] = np.sin(p / lam)
    PE[:, 1::2] = np.cos(p / lam)
    periodes = (2 * np.pi * lam).ravel()
    raison = float(periodes[1] / periodes[0])  # progression géométrique : rapport constant
    raisons_ok = bool(np.allclose(periodes[1:] / periodes[:-1], raison))
    # Décaler de k = tourner chaque paire (cos, sin) de l’angle k / λ_i
    k = 7
    _beta = (k / lam).ravel()
    _sin, _cos = PE[:-k, 0::2], PE[:-k, 1::2]
    rot_sin = _sin * np.cos(_beta) + _cos * np.sin(_beta)
    rot_cos = _cos * np.cos(_beta) - _sin * np.sin(_beta)
    err_rotation = float(max(np.max(np.abs(rot_sin - PE[k:, 0::2])), np.max(np.abs(rot_cos - PE[k:, 1::2]))))
    # RoPE : rotation relative entre positions m et n = (n − m) θ
    theta = 0.1
    angle_3_10 = float((10 - 3) * theta)

    # --- 4a. Petits angles ---
    def err_rel(th):
        return float((th - np.sin(th)) / np.sin(th))

    erreurs_angles = {"0,1 rad": err_rel(0.1), "10°": err_rel(np.radians(10)), "30°": err_rel(np.radians(30))}

    # --- 4b. Créneau : sommes partielles de sin((2j + 1) t) / (2j + 1) ---
    _tc = np.linspace(0.3, np.pi - 0.3, 2001)  # palier, loin des sauts
    _tf = np.linspace(1e-4, np.pi / 2, 20001)   # pour mesurer le dépassement près du saut
    ecarts_palier, maximums = {}, {}
    for _n in config["termes"]:
        _j = np.arange(_n)[:, None]
        _som = (np.sin((2 * _j + 1) * _tc) / (2 * _j + 1)).sum(axis=0)
        ecarts_palier[_n] = float(np.max(np.abs(_som - np.pi / 4)))
        _somf = (np.sin((2 * _j + 1) * _tf) / (2 * _j + 1)).sum(axis=0)
        maximums[_n] = float(_somf.max())
    duree = time.perf_counter() - t0

    print(f"1. période de sin 2t + sin 3t : {periode:.4f} (2π = {2 * np.pi:.4f}) ; "
          f"écart max pour T = π : {ecart_pi:.3f}")
    print(f"2. A = {A_est:.3f} (vrai {A_vrai}), φ = {phi_est:.3f} (vrai π/3 = {phi_vrai:.3f})")
    print(f"3. d = {d} : périodes de {periodes[0]:.3f} à {periodes[-1]:.0f} positions, "
          f"raison {raison:.5f} ; erreur de la rotation pour k = {k} : {err_rotation:.2e} ; "
          f"RoPE positions 3 et 10 : {angle_3_10:.1f} rad = {np.degrees(angle_3_10):.1f}°")
    print(f"4. erreurs relatives de sin θ ≈ θ : { {n: f'{100 * e:.3f} %' for n, e in erreurs_angles.items()} }")
    print(f"   créneau, écart max sur le palier : { {n: round(e, 4) for n, e in ecarts_palier.items()} }")
    print(f"   maximum près du saut : { {n: round(m, 4) for n, m in maximums.items()} } (π/4 = {np.pi / 4:.4f})")
    print(f"Calcul : {duree:.3f} s")
    return (A_est, A_vrai, PE, angle_3_10, d, duree, ecart_pi, ecarts_palier, err_rotation, erreurs_angles,
            maximums, mesure, periode, periodes, phi_est, phi_vrai, raison, raisons_ok, s)


@app.cell
def _(sp):
    # --- Vérification exacte (sympy) : formules du cours ---
    _a, _b, _t, _w, _f = sp.symbols("a b t omega phi", real=True)
    assert sp.simplify(sp.expand_trig(sp.cos(_a + _b)) - (sp.cos(_a) * sp.cos(_b) - sp.sin(_a) * sp.sin(_b))) == 0
    assert sp.simplify(sp.expand_trig(sp.sin(_a + _b)) - (sp.sin(_a) * sp.cos(_b) + sp.cos(_a) * sp.sin(_b))) == 0
    assert sp.simplify(sp.diff(sp.sin(_w * _t + _f), _t) - _w * sp.cos(_w * _t + _f)) == 0
    assert sp.simplify(sp.cos(sp.pi / 12) - (sp.sqrt(2) + sp.sqrt(6)) / 4) == 0
    assert sp.simplify(sp.cos(5 * sp.pi / 12) - (sp.sqrt(6) - sp.sqrt(2)) / 4) == 0
    assert sp.limit(sp.sin(_t) / _t, _t, 0) == 1
    # Période de sin 2t + sin 3t : 2π convient
    _decale = sp.sin(2 * (_t + 2 * sp.pi)) + sp.sin(3 * (_t + 2 * sp.pi))
    assert sp.simplify(_decale - sp.sin(2 * _t) - sp.sin(3 * _t)) == 0
    print("sympy : formules d’addition, dérivée de sin(ωt + φ), cos(π/12), cos(5π/12), sin t / t → 1, période 2π")
    return


@app.cell
def _(A_est, PE, SMOKE, d, ecarts_palier, mesure, mo, np, phi_est, plt, s, sauvegarder_figure):
    # --- Figure : 4 panneaux ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    _t = np.linspace(0, 4 * np.pi, 1000)
    axes[0, 0].plot(_t, s(_t), lw=1.5)
    for _x in (0, 2 * np.pi, 4 * np.pi):
        axes[0, 0].axvline(_x, color="gray", ls="--", lw=1)
    axes[0, 0].set(title="sin 2t + sin 3t : période 2π", xlabel="t")
    _tt = np.linspace(0, 2, mesure.size)
    axes[0, 1].plot(_tt, mesure, ".", ms=2, alpha=0.6, label="mesure bruitée")
    axes[0, 1].plot(_tt, A_est * np.sin(4 * np.pi * _tt + phi_est), lw=2,
                    label=f"ajustement : A = {A_est:.2f}, φ = {phi_est:.2f}".replace(".", ","))
    axes[0, 1].set(title="Signal 1,5 sin(4πt + π/3) bruité", xlabel="t (s)")
    axes[0, 1].legend(fontsize=8)
    _im = axes[1, 0].imshow(PE[:200].T, aspect="auto", origin="lower", cmap="RdBu", vmin=-1, vmax=1)
    axes[1, 0].set(title=f"Encodage de position (d = {d})", xlabel="position p", ylabel="composante")
    fig.colorbar(_im, ax=axes[1, 0])
    _n = list(ecarts_palier)
    axes[1, 1].loglog(_n, [ecarts_palier[k] for k in _n], "o-")
    axes[1, 1].set(title="Créneau : écart max sur le palier", xlabel="nombre de termes", ylabel="écart")
    fig.tight_layout()
    chemins = sauvegarder_figure(fig, "signaux-smoke" if SMOKE else "signaux", mo.notebook_dir() / "figures")
    print(f"Figure : {chemins[0].name}")
    fig  # noqa: B018 (marimo affiche la dernière expression)
    return


@app.cell
def _(A_est, A_vrai, angle_3_10, d, duree, ecart_pi, ecarts_palier, err_rotation, erreurs_angles,
      maximums, mo, np, periode, periodes, phi_est, phi_vrai, raison, raisons_ok):
    # --- Bilan : seuils chiffrés du README, vérifiés aussi en smoke ---
    echecs = []
    if abs(periode - 2 * np.pi) > 2e-3:
        echecs.append(f"période {periode:.4f} au lieu de 2π")
    if ecart_pi < 1:
        echecs.append(f"π semble être une période (écart {ecart_pi:.3f})")
    if abs(A_est - A_vrai) > 0.05 or abs(phi_est - phi_vrai) > 0.05:
        echecs.append(f"amplitude {A_est:.3f} ou phase {phi_est:.3f} mal retrouvée")
    if abs(periodes[0] - 2 * np.pi) > 1e-12 or not raisons_ok:
        echecs.append("périodes de l’encodage : pas une progression géométrique de premier terme 2π")
    if abs(raison - 10000 ** (2 / d)) > 1e-12:
        echecs.append(f"raison {raison} ≠ 10000^(2/d)")
    if d == 512 and abs(periodes[-1] - 60611.5) > 1:
        echecs.append(f"période de la dernière paire {periodes[-1]:.1f} au lieu d’environ 60 611")
    if err_rotation > 1e-9:
        echecs.append(f"décalage ≠ rotation (erreur {err_rotation:.2e})")
    if abs(angle_3_10 - 0.7) > 1e-12:
        echecs.append("angle RoPE")
    if not (0.00166 < erreurs_angles["0,1 rad"] < 0.00168 and 0.0050 < erreurs_angles["10°"] < 0.0052
            and 0.046 < erreurs_angles["30°"] < 0.048):
        echecs.append(f"erreurs des petits angles {erreurs_angles}")
    _n = list(ecarts_palier)
    if not all(ecarts_palier[a] > ecarts_palier[b] for a, b in zip(_n[1:], _n[2:], strict=False)):
        echecs.append(f"l’écart sur le palier ne diminue pas : {ecarts_palier}")
    if ecarts_palier[50] > 0.02:
        echecs.append(f"écart sur le palier avec 50 termes : {ecarts_palier[50]:.4f} (attendu ≤ 0,02)")
    if abs(maximums[50] - 0.926) > 0.01:
        echecs.append(f"dépassement près du saut avec 50 termes : max {maximums[50]:.4f} (attendu ≈ 0,926)")
    if echecs:
        raise SystemExit("Bilan : ÉCHEC : " + " ; ".join(echecs))
    print(f"Bilan : OK (période {periode:.4f}, A = {A_est:.3f}, φ = {phi_est:.3f}, raison {raison:.5f}, "
          f"rotation {err_rotation:.1e}, erreurs {[round(100 * e, 3) for e in erreurs_angles.values()]} %, "
          f"palier 50 termes {ecarts_palier[50]:.4f}, max {maximums[50]:.4f}, {duree:.2f} s)")
    mo.md("**Bilan** : la somme $\\sin 2t + \\sin 3t$ a pour période $2\\pi$ (pas $\\pi$) ; la formule "
          "d’addition retrouve amplitude et phase ; les périodes de l’encodage forment une progression "
          "géométrique, et décaler une position revient à tourner chaque paire ; $\\sin \\theta \\approx \\theta$ "
          "se trompe de $0{,}17\\ \\%$ à $0{,}1$ rad ; les sommes de sinusoïdes collent au créneau, sauf un "
          "dépassement d’environ $0{,}14$ près des sauts, qui ne disparaît pas.")
    return


if __name__ == "__main__":
    app.run()
