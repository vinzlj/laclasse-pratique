# Notebook 48-flottants (chapitre 48, Calcul numérique : flottants, quantification, stabilité,
# différences finies).
#
# Les flottants vus de numpy : constantes de float16, bfloat16 (simulé en arrondissant les
# bits d’un float32, sans dépendance nouvelle) et float32, absorption d’un compteur et d’une
# longue somme, softmax naïve contre log-sum-exp, quantification int8 par groupes d’une
# matrice de poids aléatoires, et vérification du gradient d’une petite perte par différence
# centrée, en float64 puis en float32.
#
#   make nb NUM=48              → éditeur marimo (bouton « Lancer le calcul »)
#   make nb NUM=48 MODE=run     → script complet
#   make nb NUM=48 MODE=smoke   → test rapide (moins de 30 s)

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
    # Flottants en numpy : formats, absorption, softmax stable, quantification, gradient

    **Chapitre 48, sections 1 à 7.** Un flottant garde un nombre fixe de chiffres binaires :
    $x = (-1)^s \times (1{,}b_1 \dots b_m)_2 \times 2^{E - \text{biais}}$. Cinq expériences :

    1. **Les formats.** Plus grand fini, plus petit normalisé et epsilon machine
       $\varepsilon = 2^{-m}$ de float16, bfloat16 et float32 ; ce que devient $0{,}1$.
    2. **L’absorption.** Un compteur float16 bloqué à 2 048 ; une longue somme de $0{,}1$ en
       float32 qui dérive, et la même somme accumulée en float64.
    3. **La softmax stable.** Sur des scores de l’ordre de 1 000, la formule naïve renvoie NaN ;
       la log-sum-exp, $\mathrm{LSE}(\mathbf{z}) = m + \ln \sum e^{z_i - m}$, non.
    4. **La quantification int8.** Une matrice de poids aléatoires, une échelle
       $s = \max|x_i| / 127$ par groupe de 32 poids : l’erreur ne dépasse jamais $s/2$.
    5. **La vérification de gradient.** Le gradient d’une perte logistique, comparé à une
       différence centrée : erreur relative minuscule s’il est juste, énorme s’il est faux,
       et peu fiable en float32.
    """)


@app.cell
def _(is_smoke, np, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        n_somme=100_000 if SMOKE else 10_000_000,   # termes 0,1 de la longue somme
        lignes_poids=64 if SMOKE else 1_024,        # matrice de poids lignes × 256
        groupe=32,                                  # poids par échelle de quantification
        n_exemples=50 if SMOKE else 500,            # exemples de la perte logistique
        dim=5 if SMOKE else 20,                     # paramètres vérifiés
        h=1e-5,                                     # pas de la différence centrée (float64)
    )
    rng = np.random.default_rng(SEED)
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SMOKE, config, rng


@app.cell
def _(np):
    # --- bfloat16 simulé : les 16 bits hauts d’un float32, arrondis au plus proche (pair) ---
    def vers_bfloat16(x):
        """Arrondit des float32 au bfloat16 le plus proche ; renvoie des float32."""
        bits = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
        return bits.astype(np.uint32).view(np.float32)

    def constantes():
        lignes = []
        for nom, t in (("float16", np.float16), ("float32", np.float32)):
            fi = np.finfo(t)
            lignes.append((nom, float(fi.max), float(fi.tiny), float(fi.eps), float(t(0.1))))
        lignes.append(("bfloat16", (2 - 2**-7) * 2.0**127, 2.0**-126, 2.0**-7,
                       float(vers_bfloat16(np.float32(0.1)))))
        return lignes

    for nom, grand, petit, eps, un_dixieme in constantes():
        print(f"{nom:9s} plus grand {grand:.4g}, plus petit normalisé {petit:.3g}, "
              f"ε = {eps:.3g}, 0,1 devient {un_dixieme!r}")
    return constantes, vers_bfloat16


@app.cell
def _(mo):
    # Le bouton empêche le calcul de se relancer à chaque modification d’une cellule amont.
    bouton = mo.ui.run_button(label="Lancer le calcul")
    bouton  # noqa: B018 (marimo affiche la dernière expression)
    return (bouton,)


@app.cell
def _(bouton, config, math, mo, np, rng, time, vers_bfloat16):
    # En script (make nb MODE=run|smoke), personne ne clique : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer le calcul ».*"))
    t0 = time.perf_counter()
    res = {}

    # 2. Absorption : compteur float16, longue somme float32 contre float64.
    compteur = np.float16(0)
    for _ in range(3_000):
        compteur = np.float16(compteur + np.float16(1))
    res["compteur16"] = float(compteur)
    res["bf16_257"] = float(vers_bfloat16(np.float32(257)))
    termes = np.full(config["n_somme"], 0.1, dtype=np.float32)
    res["somme32"] = float(np.cumsum(termes, dtype=np.float32)[-1])   # accumulation séquentielle
    res["somme64"] = float(np.cumsum(termes.astype(np.float64))[-1])
    res["somme_exacte"] = config["n_somme"] * float(np.float32(0.1))  # 0,1 déjà arrondi en float32

    # 3. Softmax naïve contre stable.
    z = np.array([1000.0, 999.0, 998.0])
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp(z) / np.exp(z).sum()
    m = z.max()
    lse = m + math.log(np.exp(z - m).sum())
    stable = np.exp(z - lse)
    res["naive_nan"] = bool(np.isnan(naive).all())
    res["lse"] = lse
    res["stable"] = stable

    # 4. Quantification int8 symétrique, une échelle par groupe de `groupe` poids.
    W = rng.normal(0, 0.02, size=(config["lignes_poids"], 256)).astype(np.float32)
    W[0, 0] = 0.5                                       # un poids isolé, vingt-cinq fois plus grand
    G = W.reshape(-1, config["groupe"]).astype(np.float64)
    s = np.abs(G).max(axis=1, keepdims=True) / 127
    k = np.clip(np.rint(G / s), -127, 127)
    erreur = np.abs(G - s * k)
    res["ratio_max"] = float((erreur / (s / 2)).max())  # ≤ 1 attendu
    res["erreur_relative_s"] = (erreur / s).ravel()
    s_tenseur = np.abs(G).max() / 127                   # une seule échelle pour toute la matrice
    k_t = np.rint(G / s_tenseur)
    res["nuls_tenseur"] = float((k_t == 0).mean())
    res["nuls_groupes"] = float((k == 0).mean())
    res["octets"] = (W.size * 4, W.size * 1 + s.size * 2)  # float32 ; int8 + échelles float16

    # 5. Vérification de gradient : perte logistique moyenne, L(w) = moyenne de ln(1 + e^(−y x·w)).
    X = rng.normal(size=(config["n_exemples"], config["dim"]))
    y = rng.choice([-1.0, 1.0], size=config["n_exemples"])
    w0 = rng.normal(size=config["dim"])

    def perte(w, dtype=np.float64):
        Xd, yd, wd = X.astype(dtype), y.astype(dtype), w.astype(dtype)
        return np.logaddexp(dtype(0), -yd * (Xd @ wd)).mean(dtype=dtype)

    def gradient(w, bug=False):
        marge = y * (X @ w)
        sig = np.exp(-np.logaddexp(0, marge))           # σ(−marge), sans débordement
        g = -(X * (y * sig)[:, None]).mean(axis=0)
        return g / 2 if bug else g                      # le bug : un facteur 2 perdu

    def gradient_numerique(w, h, dtype=np.float64):
        g = np.zeros_like(w)
        for i in range(len(w)):
            e = np.zeros_like(w)
            e[i] = h
            g[i] = (perte(w + e, dtype) - perte(w - e, dtype)) / (2 * h)
        return g

    def erreur_rel(g, gh):
        return float(np.max(np.abs(g - gh) / np.maximum(np.maximum(np.abs(g), np.abs(gh)), 1e-300)))

    g_juste = gradient(w0)
    gh64 = gradient_numerique(w0, config["h"])
    res["err_juste"] = erreur_rel(g_juste, gh64)
    res["err_bug"] = erreur_rel(gradient(w0, bug=True), gh64)
    res["err_float32"] = erreur_rel(g_juste, gradient_numerique(w0, 1e-3, np.float32))
    pas = np.logspace(-12, -1, 23)
    res["pas"] = pas
    res["err_pas"] = np.array([erreur_rel(g_juste, gradient_numerique(w0, h)) for h in pas])
    duree = time.perf_counter() - t0
    print(f"Calcul : {duree:.2f} s")
    for cle in ("compteur16", "bf16_257", "somme32", "somme64", "somme_exacte", "naive_nan", "lse",
                "ratio_max", "nuls_tenseur", "nuls_groupes", "octets", "err_juste", "err_bug", "err_float32"):
        print(f"  {cle} = {res[cle]}")
    return duree, res


@app.cell
def _(sp):
    # --- Vérifications exactes (sympy) ---
    a, h = sp.symbols("a h", positive=True)
    f = a**3
    centree = sp.expand(((a + h) ** 3 - (a - h) ** 3) / (2 * h))
    avant = sp.expand(((a + h) ** 3 - a**3) / h)
    assert sp.simplify(centree - (3 * a**2 + h**2)) == 0           # erreur h² f‴/6 = h², exacte
    assert sp.simplify(avant - (3 * a**2 + 3 * a * h + h**2)) == 0  # erreur h f″/2 + h²
    assert sp.diff(f, a, 3) == 6
    z1, z2, z3, cs = sp.symbols("z1 z2 z3 c", real=True)
    lse_s = sp.log(sp.exp(z1) + sp.exp(z2) + sp.exp(z3))
    decale = sp.log(sp.exp(z1 - cs) + sp.exp(z2 - cs) + sp.exp(z3 - cs))
    ecart = decale - (lse_s - cs)
    for valeurs in ({z1: 1000, z2: 999, z3: 998, cs: 1000}, {z1: -3, z2: 2, z3: 7, cs: 5}):
        assert abs(sp.N(ecart.subs(valeurs), 50)) < sp.Float("1e-40")
    assert sp.Rational(2) ** -10 == sp.Rational(1, 1024)
    assert (2 - sp.Rational(2) ** -10) * 2**15 == 65504
    assert (2 - sp.Rational(1, 4)) * 2**15 == 57344 and sp.Rational(7, 4) * 2**8 == 448
    print("sympy : différences finies de x³, invariance LSE(z − c) = LSE(z) − c, constantes des formats OK")
    return


@app.cell
def _(SMOKE, mo, np, plt, res, sauvegarder_figure, vers_bfloat16):
    # --- Figure : écart entre flottants voisins selon x, et erreurs de quantification ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    xs = np.logspace(-4, 4, 200)
    ax1.loglog(xs, np.spacing(xs.astype(np.float16)).astype(float), label="float16")
    x32 = xs.astype(np.float32)
    ecart_bf = (vers_bfloat16(x32 * (1 + 2.0**-7)) - vers_bfloat16(x32)).astype(float)
    ax1.loglog(xs, np.where(ecart_bf > 0, ecart_bf, np.nan), label="bfloat16")
    ax1.loglog(xs, np.spacing(x32).astype(float), label="float32")
    ax1.set(xlabel="x", ylabel="écart avec le flottant suivant", title="L’écart grandit avec x")
    ax1.legend()
    ax2.hist(res["erreur_relative_s"], bins=40, color="C2")
    ax2.axvline(0.5, color="k", ls="--", label="borne s/2")
    ax2.set(xlabel="erreur / s", ylabel="nombre de poids", title="Quantification int8 par groupes")
    ax2.legend()
    chemins = sauvegarder_figure(fig, "flottants-smoke" if SMOKE else "flottants", mo.notebook_dir() / "figures")
    print(f"Figure : {chemins[0].name}")
    fig  # noqa: B018 (marimo affiche la dernière expression)
    return


@app.cell
def _(constantes, duree, mo, np, res):
    # --- Bilan : seuils du README, vérifiés aussi en smoke ---
    echecs = []
    cst = {nom: (g, p, e, d) for nom, g, p, e, d in constantes()}
    if cst["float16"][0] != 65504 or cst["float16"][2] != 2**-10 or cst["float32"][2] != 2**-23:
        echecs.append("constantes float16 ou float32")
    if cst["bfloat16"][3] != 0.10009765625 or cst["float16"][3] != 0.0999755859375:
        echecs.append("0,1 en bfloat16 ou en float16")
    if res["compteur16"] != 2048 or res["bf16_257"] != 256:
        echecs.append(f"absorption : compteur {res['compteur16']}, 257 en bfloat16 {res['bf16_257']}")
    derive32 = abs(res["somme32"] - res["somme_exacte"]) / res["somme_exacte"]
    derive64 = abs(res["somme64"] - res["somme_exacte"]) / res["somme_exacte"]
    if not (derive32 > 1e-4 and derive64 < 1e-9):
        echecs.append(f"longue somme : dérive float32 {derive32:.1e}, float64 {derive64:.1e}")
    if not res["naive_nan"] or abs(res["lse"] - 1000.4076059644444) > 1e-9:
        echecs.append(f"softmax : naïve NaN {res['naive_nan']}, LSE {res['lse']}")
    if not np.allclose(res["stable"], [0.66524096, 0.24472847, 0.09003057], atol=1e-8):
        echecs.append("softmax stable")
    if res["ratio_max"] > 1 + 1e-9:
        echecs.append(f"erreur de quantification {res['ratio_max']:.3f} × s/2")
    if not res["nuls_tenseur"] > res["nuls_groupes"]:
        echecs.append("le poids isolé devrait annuler plus de poids avec une seule échelle")
    if not (res["err_juste"] < 1e-7 and res["err_bug"] > 1e-2 and res["err_float32"] > 1e-6):
        echecs.append(f"vérification de gradient : juste {res['err_juste']:.1e}, bug {res['err_bug']:.1e}, "
                      f"float32 {res['err_float32']:.1e}")
    if echecs:
        raise SystemExit("Bilan : ÉCHEC : " + " ; ".join(echecs))
    o32, o8 = res["octets"]
    print(f"Bilan : OK (compteur float16 bloqué à 2 048, dérive float32 {derive32:.1e} contre {derive64:.1e}, "
          f"LSE {res['lse']:.4f}, erreur de quantification ≤ {res['ratio_max']:.3f} × s/2, "
          f"poids nuls {res['nuls_tenseur']:.0%} avec une échelle contre {res['nuls_groupes']:.0%} par groupes, "
          f"{o32} octets → {o8}, gradient juste {res['err_juste']:.1e}, faux {res['err_bug']:.1e}, "
          f"float32 {res['err_float32']:.1e} ; {duree:.2f} s)")
    def fr(x):
        return f"{x:.1e}".replace(".", ",")

    mo.md(f"**Bilan** : erreur relative du gradient juste {fr(res['err_juste'])}, du gradient faux "
          f"{fr(res['err_bug'])}, en float32 {fr(res['err_float32'])}.")
    return


if __name__ == "__main__":
    app.run()
