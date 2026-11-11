# Notebook 51-convolution (chapitre 51, Convolution discrète 1D et 2D).
#
# La convolution à la main et avec numpy : corrélation croisée écrite avec des tranches,
# comparée à np.correlate et à np.convolve (noyau retourné) ; moyenne glissante d’un signal
# bruité ; convolution 2D par sliding_window_view sur une image synthétique, avec les filtres
# du cours ; matrice T de la convolution 1D ; équivariance par translation (et sa perte avec
# un pas 2) ; loi de la somme de deux dés par np.convolve. numpy seul (pas de scipy).
#
#   make nb NUM=51              → éditeur marimo (bouton « Lancer le calcul »)
#   make nb NUM=51 MODE=run     → script complet
#   make nb NUM=51 MODE=smoke   → test rapide (moins de 30 s)

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
    from numpy.lib.stride_tricks import sliding_window_view

    return is_smoke, mo, np, plt, sauvegarder_figure, seed_everything, sliding_window_view, sp, style_commun, time


@app.cell
def _(mo):
    mo.md(r"""
    # La convolution, à la main et avec numpy

    **Chapitre 51, sections 1 à 7.** Indices à partir de 0, comme numpy. Pour un signal
    $x$ de taille $n$ et un noyau $w$ de taille $k$ :

    - **corrélation croisée** (le calcul des réseaux de neurones) :
      $y_i = \sum_{j=0}^{k-1} w_j\, x_{i+j}$, $i$ de $0$ à $n - k$ ;
    - **convolution** des mathématiques : $(x \ast w)_i = \sum_j w_j\, x_{i-j}$, la même
      chose avec le noyau retourné.

    Six expériences :

    1. la corrélation écrite avec des tranches de tableau, comparée à `np.correlate`, puis à
       `np.convolve` (qui retourne le noyau) ;
    2. une moyenne glissante qui lisse un signal bruité ;
    3. la convolution 2D par `sliding_window_view` et les filtres du cours sur une image
       synthétique, avec la taille de sortie $\lfloor (n + 2p - k)/s \rfloor + 1$ ;
    4. la matrice $T$ de la convolution : $T x$ redonne la corrélation ;
    5. l’équivariance : décaler puis convoluer = convoluer puis décaler, sauf avec un pas 2 ;
    6. la loi de la somme de deux dés, par `np.convolve`, vérifiée en fractions exactes.
    """)


@app.cell
def _(is_smoke, np, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        n_tests=200 if SMOKE else 5_000,    # tirages (signal, noyau) comparés à numpy
        n_signal=200 if SMOKE else 2_000,   # longueur du signal bruité de l’expérience 2
        cote_image=32 if SMOKE else 96,     # côté de l’image synthétique
    )
    rng = np.random.default_rng(SEED)
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SMOKE, config, rng


@app.cell
def _(mo):
    # Le bouton empêche le calcul de se relancer à chaque modification d’une cellule amont.
    bouton = mo.ui.run_button(label="Lancer le calcul")
    bouton  # noqa: B018 (marimo affiche la dernière expression)
    return (bouton,)


@app.cell
def _(bouton, config, mo, np, rng, sliding_window_view, time):
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer le calcul ».*"))
    t0 = time.perf_counter()

    def correlation(x, w, s=1):
        """yᵢ = Σⱼ wⱼ xᵢ₊ⱼ, en ajoutant k tranches décalées (aucune boucle sur i)."""
        n, k = len(x), len(w)
        y = sum(w[j] * x[j:n - k + 1 + j] for j in range(k))
        return y[::s]

    # 1. Corrélation à la main contre numpy
    ecart_corr = 0.0
    ecart_conv = 0.0
    for _ in range(config["n_tests"]):
        n = int(rng.integers(5, 40))
        k = int(rng.integers(1, min(n, 8) + 1))
        x = rng.normal(size=n)
        w = rng.normal(size=k)
        ecart_corr = max(ecart_corr, float(np.abs(correlation(x, w) - np.correlate(x, w, "valid")).max()))
        ecart_conv = max(ecart_conv, float(np.abs(correlation(x, w[::-1]) - np.convolve(x, w, "valid")).max()))
    exemple = correlation(np.array([0, 0, 1, 3, 2, 0, 0, 0, 0]), np.array([1, 0, -1]))
    exemple_conv = np.convolve([0, 0, 1, 3, 2, 0, 0, 0, 0], [1, 0, -1], "valid")
    ventes = correlation(np.array([120, 90, 150, 60, 180.0]), np.full(3, 1 / 3))

    # 2. Moyenne glissante d’un signal bruité
    t = np.arange(config["n_signal"])
    propre = np.sin(2 * np.pi * t / 200)
    bruite = propre + rng.normal(scale=0.5, size=t.size)
    k_lisse = 21
    lisse = correlation(bruite, np.full(k_lisse, 1 / k_lisse))
    centre = propre[k_lisse // 2: k_lisse // 2 + lisse.size]
    err_bruite = float(np.sqrt(np.mean((bruite - propre) ** 2)))
    err_lisse = float(np.sqrt(np.mean((lisse - centre) ** 2)))

    # 3. Convolution 2D et filtres
    def conv2d(img, noyau, p=0, s=1):
        if p:
            img = np.pad(img, p)
        fenetres = sliding_window_view(img, noyau.shape)[::s, ::s]
        return np.einsum("ijuv,uv->ij", fenetres, noyau)

    c = config["cote_image"]
    img = np.full((c, c), 0.2)
    img[c // 6: c // 2, c // 8: c // 2] = 0.9
    filtres = {
        "flou": np.ones((3, 3)) / 9,
        "flou gaussien": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
        "Sobel x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        "laplacien": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
        "netteté": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    }
    sorties = {nom: conv2d(img, f, p=1) for nom, f in filtres.items()}
    sobel_max = float(sorties["Sobel x"].max())
    zone_unie = {nom: float(conv2d(np.full((5, 5), 0.2), f)[1, 1]) for nom, f in filtres.items()}
    tailles = {(p, s): conv2d(np.zeros((64, 64)), np.ones((3, 3)), p, s).shape[0]
               for p, s in [(0, 1), (1, 1), (1, 2)]}
    bord = np.zeros((6, 6))
    bord[:, 3:] = 1
    prewitt = conv2d(bord, np.array([[-1, 0, 1]] * 3))

    # 4. Matrice T de la convolution
    w_t = np.array([0.25, 0.5, 0.25])
    x_t = np.array([4, 8, 0, 4, 12.0])
    T = np.zeros((3, 5))
    for i in range(3):
        T[i, i:i + 3] = w_t
    Tx = T @ x_t

    # 5. Équivariance
    def decaler(x, d):
        return np.concatenate([np.zeros(d), x[:len(x) - d]])

    x_e = rng.normal(size=60)
    w_e = rng.normal(size=5)
    d = 7
    y_e = correlation(x_e, w_e)
    y_dec = correlation(decaler(x_e, d), w_e)
    ecart_equi = float(np.abs(y_dec[d:] - y_e[:-d]).max())
    impulsion = np.array([0, 0, 1, 0, 0, 0, 0, 0.0])
    pas2 = [correlation(decaler(impulsion, dd), np.array([1, 1.0]), s=2).tolist() for dd in (0, 1, 2)]

    # 6. Deux dés
    de = np.full(6, 1 / 6)
    loi = np.convolve(de, de)

    duree = time.perf_counter() - t0
    print(f"Calcul : {duree:.2f} s")
    print(f"1. écarts à numpy : corrélation {ecart_corr:.1e}, convolution {ecart_conv:.1e} ; "
          f"exemple (1, 0, −1) : {exemple.tolist()} ; np.convolve : {exemple_conv.tolist()} ; "
          f"ventes {ventes.round(6).tolist()}")
    print(f"2. erreur quadratique : bruité {err_bruite:.3f}, lissé sur {k_lisse} points {err_lisse:.3f}")
    print(f"3. tailles 64, k = 3 : {tailles} ; Sobel x max {sobel_max:.3f} ; zone unie 0,2 : {zone_unie}")
    print(f"4. T x = {Tx.tolist()}")
    print(f"5. équivariance (pas 1) : écart {ecart_equi:.1e} ; pas 2, décalages 0, 1, 2 : {pas2}")
    print(f"6. deux dés × 36 : {(loi * 36).round(6).tolist()}")
    return (bord, bruite, centre, duree, ecart_conv, ecart_corr, ecart_equi, exemple, exemple_conv, img, k_lisse,
            lisse, loi, pas2, prewitt, propre, sobel_max, sorties, tailles, Tx, ventes, zone_unie, err_bruite,
            err_lisse)


@app.cell
def _(sp):
    # --- Vérification exacte (sympy) : loi de la somme de deux dés, et noyau symétrique ---
    _de = [sp.Rational(1, 6)] * 6
    loi_exacte = [sum(_de[a] * _de[t - a] for a in range(6) if 0 <= t - a < 6) for t in range(11)]
    assert [36 * p for p in loi_exacte] == [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1], loi_exacte
    assert sum(loi_exacte) == 1
    # corrélation et convolution coïncident pour un noyau symétrique
    _xs = sp.symbols("x0:6")
    _w = [sp.Rational(1, 4), sp.Rational(1, 2), sp.Rational(1, 4)]
    _corr = [sum(_w[j] * _xs[i + j] for j in range(3)) for i in range(4)]
    _conv = [sum(_w[j] * _xs[i + 2 - j] for j in range(3)) for i in range(4)]
    assert all(sp.simplify(a - b) == 0 for a, b in zip(_corr, _conv, strict=True))
    print(f"sympy : P(X + Y = 7) = {loi_exacte[5]}, somme des 11 probabilités = {sum(loi_exacte)}")
    return (loi_exacte,)


@app.cell
def _(SMOKE, bruite, centre, img, k_lisse, lisse, loi, mo, np, plt, propre, sauvegarder_figure, sorties):
    # --- Figure, sauvegardée dans figures/ (suffixe -smoke en mode smoke) ---
    fig = plt.figure(figsize=(9, 8))
    ax1 = fig.add_subplot(3, 1, 1)
    debut = k_lisse // 2
    ax1.plot(bruite, color="0.75", lw=0.8, label="signal bruité")
    ax1.plot(np.arange(debut, debut + lisse.size), lisse, lw=1.6, label=f"moyenne glissante ({k_lisse} points)")
    ax1.plot(propre, "--", lw=1, color="k", label="signal sans bruit")
    ax1.set(title="Une moyenne glissante lisse le bruit", xlabel="indice i")
    ax1.legend(fontsize=8, loc="upper right")
    noms = ["image", "flou gaussien", "Sobel x", "laplacien"]
    images = [img, sorties["flou gaussien"], sorties["Sobel x"], sorties["laplacien"]]
    for position, (nom, z) in enumerate(zip(noms, images, strict=True)):
        ax = fig.add_subplot(3, 4, 5 + position)
        signe = nom in ("Sobel x", "laplacien")
        m = float(np.abs(z).max())
        ax.imshow(z, cmap="RdBu_r" if signe else "gray", vmin=-m if signe else 0, vmax=m if signe else 1,
                  interpolation="nearest")
        ax.set_title(nom, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.bar(np.arange(2, 13), loi, color="#3f74c4")
    ax3.set(title="Loi de la somme de deux dés (np.convolve)", xlabel="somme t", ylabel="probabilité",
            xticks=range(2, 13))
    fig.tight_layout()
    chemins = sauvegarder_figure(fig, "convolution-smoke" if SMOKE else "convolution", mo.notebook_dir() / "figures")
    print(f"Figure : {chemins[0].name}")
    fig  # noqa: B018 (marimo affiche la dernière expression)
    return


@app.cell
def _(Tx, duree, ecart_conv, ecart_corr, ecart_equi, err_bruite, err_lisse, exemple, exemple_conv, loi, mo, np,
      pas2, prewitt, sobel_max, tailles, ventes, zone_unie):
    # --- Bilan : seuils chiffrés du README, vérifiés aussi en smoke (calculs exacts ou presque) ---
    echecs = []
    if max(ecart_corr, ecart_conv) > 1e-12:
        echecs.append(f"écart à numpy {max(ecart_corr, ecart_conv):.1e}")
    if exemple.tolist() != [-1, -3, -1, 3, 2, 0, 0] or exemple_conv.tolist() != [1, 3, 1, -3, -2, 0, 0]:
        echecs.append(f"exemple (1, 0, −1) : {exemple.tolist()} / {exemple_conv.tolist()}")
    if not np.allclose(ventes, [120, 100, 130]):
        echecs.append(f"ventes {ventes}")
    if err_lisse > err_bruite / 3:
        echecs.append(f"lissage insuffisant : {err_lisse:.3f} contre {err_bruite:.3f}")
    if tailles != {(0, 1): 62, (1, 1): 64, (1, 2): 32}:
        echecs.append(f"tailles {tailles}")
    if not np.array_equal(prewitt, np.tile([0, 3, 3, 0], (4, 1))):
        echecs.append(f"Prewitt {prewitt.tolist()}")
    if abs(sobel_max - 4 * 0.7) > 1e-9:
        echecs.append(f"Sobel x max {sobel_max}")
    attendu_unie = {"flou": 0.2, "flou gaussien": 0.2, "Sobel x": 0.0, "laplacien": 0.0, "netteté": 0.2}
    if any(abs(zone_unie[nom] - v) > 1e-12 for nom, v in attendu_unie.items()):
        echecs.append(f"zone unie {zone_unie}")
    if not np.allclose(Tx, [5, 3, 5]):
        echecs.append(f"T x = {Tx}")
    if ecart_equi > 1e-12:
        echecs.append(f"équivariance {ecart_equi:.1e}")
    if pas2 != [[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]:
        echecs.append(f"pas 2 {pas2}")
    if not np.allclose(loi * 36, [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]):
        echecs.append(f"deux dés {loi * 36}")
    if echecs:
        raise SystemExit("Bilan : ÉCHEC : " + " ; ".join(echecs))
    print(f"Bilan : OK (écart à numpy {max(ecart_corr, ecart_conv):.1e}, lissage {err_bruite:.3f} → "
          f"{err_lisse:.3f}, tailles 62, 64, 32, T x = (5, 3, 5), équivariance {ecart_equi:.1e}, "
          f"P(7) = {loi[5]:.4f}, {duree:.2f} s)")
    mo.md(f"**Bilan** : la corrélation à la main égale `np.correlate` (écart {max(ecart_corr, ecart_conv):.1e}) ; "
          f"le lissage divise l’erreur par {err_bruite / err_lisse:.1f} ; un pas 2 casse l’équivariance.")
    return


if __name__ == "__main__":
    app.run()
