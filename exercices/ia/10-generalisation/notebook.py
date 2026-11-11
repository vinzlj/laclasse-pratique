# Exercice 10 : généraliser, la théorie (pulvérisation, dimension VC, bornes, no free lunch).
#
#   make ex NUM=10              → éditeur marimo (bouton « Lancer les calculs »)
#   make ex NUM=10 MODE=run     → script complet, vérifie les seuils du README
#   make ex NUM=10 MODE=smoke   → test rapide (< 30 s)
#
# Tout tourne sur le CPU (NumPy, SciPy, scikit-learn) : les modèles sont minuscules,
# ce qui compte ici, c'est de compter (des étiquetages, des écarts, des fonctions cibles).

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import itertools
    import time
    import warnings
    from fractions import Fraction

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import linprog
    from scipy.special import comb, gammaln

    from ia_commun import is_smoke, sauvegarder_figure, seed_everything, style_commun

    return (
        Fraction,
        comb,
        gammaln,
        is_smoke,
        itertools,
        linprog,
        mo,
        np,
        plt,
        sauvegarder_figure,
        seed_everything,
        style_commun,
        time,
        warnings,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Exercice 10 : quand l'erreur d'entraînement dit-elle quelque chose du test ?

    On entraîne un classifieur sur $N$ exemples et on mesure son **risque empirique**
    $\hat R(h)$ (taux d'erreur sur ces $N$ exemples). Ce qui nous intéresse vraiment est son
    **risque réel** $R(h)$ : le taux d'erreur sur de *nouvelles* données tirées de la même
    distribution. La théorie de Vapnik et Chervonenkis borne l'écart $R - \hat R$.

    **Perte** : la perte 0-1, $\ell(h(x), y) = \mathbb 1[h(x) \ne y]$.
    $\hat R(h) = \frac1N\sum_i \mathbb 1[h(x_i)\ne y_i]$ et $R(h) = \mathbb P[h(x)\ne y]$.
    Les classifieurs de la partie C sont entraînés par régression logistique (entropie
    croisée, plus facile à minimiser), mais **mesurés** en perte 0-1 : la borne VC vaut pour
    *toute* fonction de la classe, quel que soit l'algorithme qui l'a choisie.

    Plan :

    - **A. Pulvériser.** Compter, par force brute, combien d'étiquetages une classe de
      classifieurs peut réaliser sur $n$ points ; en déduire sa **dimension VC** et comparer
      au lemme de Sauer.
    - **B. Classe finie.** Hoeffding + borne de l'union : la garantie tient-elle vraiment
      dans 95 % des tirages ? Combien est-elle pessimiste ?
    - **C. Classe infinie.** Polynômes de degré 1 à 10 en 2D, $N$ de 20 à 10 000 : écart
      train/test mesuré contre la borne VC.
    - **D. No free lunch.** En moyenne sur *toutes* les fonctions cibles, aucun algorithme
      ne fait mieux que le hasard hors de l'ensemble d'entraînement.
    """)


@app.cell
def _(is_smoke, seed_everything, style_commun):
    # --- Configuration : le SEUL endroit où le mode smoke réduit le travail ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        delta=0.05,                       # niveau de confiance 1 − δ = 95 %
        # A : pulvérisation
        n_max_droites=8 if SMOKE else 10,  # nombre max. de points pour la croissance des demi-plans
        # B : classe finie de seuils
        k_seuils=101,                     # |H| = 101 seuils régulièrement espacés dans [0, 1]
        bruit_b=0.2,                      # 20 % d'étiquettes inversées
        n_b=[20, 100, 500, 2000],
        tirages_b=200 if SMOKE else 2000,
        # C : polynômes en 2D
        degres=[1, 2, 5] if SMOKE else [1, 2, 3, 4, 6, 8, 10],
        tailles=[20, 200, 2000] if SMOKE else [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
        repetitions=2 if SMOKE else 10,
        n_test=20_000 if SMOKE else 100_000,
        rayon2=0.5,                       # frontière vraie : cercle x1² + x2² = 0,5
        bruit_c=0.1,                      # 10 % d'étiquettes inversées : erreur de Bayes = 0,10
        C_logistique=1e4,                 # régularisation quasi nulle : on veut un ERM
        # D : no free lunch
        taille_domaine=10,
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SEED, SMOKE, config


@app.cell
def _(mo):
    mo.md(r"""
    ## A. Pulvériser des points

    Une classe $\mathcal H$ **pulvérise** (*shatters*) un ensemble de $n$ points si elle
    réalise ses $2^n$ étiquetages. La **fonction de croissance** $m_{\mathcal H}(n)$ est le
    nombre maximal d'étiquetages réalisables sur $n$ points ; la **dimension VC**
    $d_{VC}$ est le plus grand $n$ tel que $m_{\mathcal H}(n) = 2^n$.

    **Lemme de Sauer** : si $d_{VC} = d$, alors
    $m_{\mathcal H}(n) \le \sum_{i=0}^{d} \binom{n}{i}$, un polynôme de degré $d$ en $n$
    au lieu de $2^n$. C'est ce qui rend les bornes possibles.

    D'abord en 1D, à la main : seuils $h(x) = \mathbb 1[x \ge t]$ et intervalles
    $h(x) = \mathbb 1[a \le x \le b]$. Puis en 2D, des demi-plans : un étiquetage est
    réalisable s'il existe $(w, b)$ avec $s_i (w\cdot x_i + b) \ge 1$ pour tout $i$, où
    $s_i = 2y_i - 1 \in \{-1, +1\}$ est l'étiquette en signe, ce que l'on teste par programmation linéaire (`scipy.optimize.linprog`).
    """)


@app.cell
def _(comb, itertools, linprog, np):
    def etiquetages_seuils(x):
        """Étiquetages réalisés par les seuils 1[x ≥ t] sur les points x (1D), par force brute."""
        xs = np.sort(x)
        candidats = np.concatenate([[xs[0] - 1], (xs[:-1] + xs[1:]) / 2, [xs[-1] + 1]])
        return {tuple((x >= t).astype(int)) for t in candidats}

    def etiquetages_intervalles(x):
        """Étiquetages réalisés par les intervalles 1[a ≤ x ≤ b] (intervalle vide compris)."""
        xs = np.sort(x)
        bornes = np.concatenate([[xs[0] - 1], (xs[:-1] + xs[1:]) / 2, [xs[-1] + 1]])
        res = {tuple(np.zeros(len(x), dtype=int))}
        for a, b in itertools.combinations(bornes, 2):
            res.add(tuple(((x >= a) & (x <= b)).astype(int)))
        return res

    def separable(phi, y):
        """Existe-t-il w (biais compris dans phi) tel que s_i · (w · phi_i) ≥ 1, avec s_i = 2y_i − 1 ? Programme linéaire."""
        signes = 2 * np.asarray(y) - 1
        A = -(signes[:, None] * phi)            # contrainte : −s_i (w·phi_i) ≤ −1
        res = linprog(np.zeros(phi.shape[1]), A_ub=A, b_ub=-np.ones(len(y)),
                      bounds=[(None, None)] * phi.shape[1], method="highs")
        return res.status == 0

    def croissance(phi):
        """Nombre d'étiquetages (parmi les 2^n) réalisables par un séparateur linéaire en phi."""
        n = phi.shape[0]
        return sum(separable(phi, y) for y in itertools.product([0, 1], repeat=n))

    def sauer(n, d):
        """Borne de Sauer : somme des C(n, i) pour i ≤ d."""
        return int(sum(comb(n, i, exact=True) for i in range(min(d, n) + 1)))

    return croissance, etiquetages_intervalles, etiquetages_seuils, sauer, separable


@app.cell
def _(croissance, etiquetages_intervalles, etiquetages_seuils, np, sauer):
    rng_a = np.random.default_rng(42)
    print("1D : nombre d'étiquetages réalisés sur n points (sur 2^n possibles)")
    print(f"{'n':>3} {'2^n':>5} {'seuils':>7} {'Sauer d=1':>10} {'intervalles':>12} {'Sauer d=2':>10}")
    resultats_1d = []
    for n1 in range(1, 7):
        x1d = rng_a.uniform(0, 1, n1)
        s, i2 = len(etiquetages_seuils(x1d)), len(etiquetages_intervalles(x1d))
        resultats_1d.append((n1, s, i2))
        print(f"{n1:>3} {2**n1:>5} {s:>7} {sauer(n1, 1):>10} {i2:>12} {sauer(n1, 2):>10}")

    # 2D, demi-plans : phi(x) = (1, x1, x2)
    def phi_affine(p):
        return np.c_[np.ones(len(p)), p]

    triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.3, 0.9]])
    alignes = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    carre = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    m_triangle = croissance(phi_affine(triangle))
    m_alignes = croissance(phi_affine(alignes))
    m_carre = croissance(phi_affine(carre))
    print(f"\n2D, demi-plans : triangle {m_triangle}/8, trois points alignés {m_alignes}/8, "
          f"carré {m_carre}/16 (Sauer, d = 3 : {sauer(4, 3)})")
    return m_alignes, m_carre, m_triangle, phi_affine, resultats_1d, rng_a


@app.cell
def _(config, croissance, np, phi_affine, rng_a, sauer, time):
    # Fonction de croissance des demi-plans : on prend le maximum sur quelques nuages aléatoires
    t0_a = time.perf_counter()
    croissance_demi_plans = {}
    for n2 in range(1, config["n_max_droites"] + 1):
        croissance_demi_plans[n2] = max(croissance(phi_affine(rng_a.uniform(-1, 1, (n2, 2))))
                                        for _ in range(3))
    print(f"{'n':>3} {'2^n':>6} {'demi-plans':>11} {'Sauer d=3':>10}  (formule exacte n² − n + 2)")
    for n2, m in croissance_demi_plans.items():
        print(f"{n2:>3} {2**n2:>6} {m:>11} {sauer(n2, 3):>10}  ({n2 * n2 - n2 + 2})")
    print(f"({time.perf_counter() - t0_a:.1f} s)")
    return (croissance_demi_plans,)


@app.cell
def _(mo):
    mo.md(r"""
    **Coniques.** Un classifieur polynomial de degré 2 en 2D est un séparateur linéaire
    dans l'espace des $D = 6$ monômes $(1, x_1, x_2, x_1^2, x_1x_2, x_2^2)$. Sa dimension VC
    vaut au plus $D$ (séparateur affine dans $\mathbb R^{5}$) ; on vérifie ci-dessous qu'il
    pulvérise 6 points pris au hasard, et pas les 7 points tirés ensuite (la borne 6 vient
    de l'argument affine). En degré $p$ :
    $D = (p+1)(p+2)/2$, la valeur utilisée dans la partie C.

    On construit les monômes à la main, puis on compare avec
    `sklearn.preprocessing.PolynomialFeatures` (mêmes colonnes, même ordre).
    """)


@app.cell
def _(np):
    def monomes(X, p):
        """Tous les monômes x1^a · x2^b avec a + b ≤ p, dans l'ordre de scikit-learn."""
        colonnes = []
        for k in range(p + 1):          # degré total k
            for a in range(k, -1, -1):  # x1^a x2^(k−a), a décroissant
                colonnes.append(X[:, 0] ** a * X[:, 1] ** (k - a))
        return np.stack(colonnes, axis=1)

    def dimension_vc_poly(p):
        return (p + 1) * (p + 2) // 2

    return dimension_vc_poly, monomes


@app.cell
def _(croissance, monomes, np, rng_a):
    from sklearn.preprocessing import PolynomialFeatures

    _X = rng_a.uniform(-1, 1, (5, 2))
    ecart_monomes = float(np.abs(monomes(_X, 4) - PolynomialFeatures(4).fit_transform(_X)).max())
    print(f"Monômes à la main contre PolynomialFeatures (degré 4) : écart max {ecart_monomes:.1e}")

    m_conique_6 = croissance(monomes(rng_a.uniform(-1, 1, (6, 2)), 2))
    m_conique_7 = croissance(monomes(rng_a.uniform(-1, 1, (7, 2)), 2))
    print(f"Coniques : 6 points → {m_conique_6}/64 étiquetages ; 7 points → {m_conique_7}/128")
    return ecart_monomes, m_conique_6, m_conique_7


@app.cell
def _(mo):
    mo.md(r"""
    ## B. Une classe finie : Hoeffding et borne de l'union

    $\mathcal H$ = 101 seuils $h_t(x) = \mathbb 1[x \ge t]$, $t \in \{0; 0{,}01; \dots; 1\}$.
    Données : $x$ uniforme sur $[0, 1]$, vraie étiquette $\mathbb 1[x \ge 0{,}3]$, inversée
    avec probabilité 0,2. Le risque réel se calcule exactement :
    $R(h_t) = 0{,}2 + 0{,}6\,|t - 0{,}3|$.

    **Pour une seule** hypothèse fixée à l'avance, l'inégalité de Hoeffding donne
    $\mathbb P[|R - \hat R| > \gamma] \le 2e^{-2\gamma^2 N}$. Mais l'algorithme *choisit*
    $h$ après avoir vu les données ; il faut donc que l'écart soit petit **pour toutes** les
    hypothèses à la fois. Borne de l'union sur les $k = |\mathcal H|$ hypothèses : avec
    probabilité au moins $1 - \delta$,
    $$\forall h \in \mathcal H,\quad |R(h) - \hat R(h)| \le \gamma = \sqrt{\tfrac{1}{2N}\ln\tfrac{2k}{\delta}}.$$

    On le vérifie sur 2 000 jeux d'entraînement par taille $N$.
    """)


@app.cell
def _(config, np, time):
    k = config["k_seuils"]
    seuils = np.linspace(0, 1, k)
    risque_reel_b = config["bruit_b"] + (1 - 2 * config["bruit_b"]) * np.abs(seuils - 0.3)
    rng_b = np.random.default_rng(42)
    t0_b = time.perf_counter()
    lignes_b = []
    for n_b in config["n_b"]:
        gamma = np.sqrt(np.log(2 * k / config["delta"]) / (2 * n_b))
        gamma_seul = np.sqrt(np.log(2 / config["delta"]) / (2 * n_b))  # Hoeffding, une hypothèse
        ecarts_max, ecarts_erm, risques_erm = [], [], []
        for _ in range(config["tirages_b"]):
            x = rng_b.uniform(0, 1, n_b)
            y = (x >= 0.3) ^ (rng_b.uniform(0, 1, n_b) < config["bruit_b"])
            # risque empirique des 101 seuils d'un coup : matrice (k, N)
            r_emp_b = ((x[None, :] >= seuils[:, None]) != y[None, :]).mean(axis=1)
            ecarts = np.abs(risque_reel_b - r_emp_b)
            j = int(np.argmin(r_emp_b))              # ERM : le seuil de plus faible erreur
            ecarts_max.append(ecarts.max())
            ecarts_erm.append(risque_reel_b[j] - r_emp_b[j])
            risques_erm.append(risque_reel_b[j])
        ecarts_max = np.array(ecarts_max)
        lignes_b.append(dict(
            N=n_b, gamma=gamma, gamma_seul=gamma_seul,
            echec=float((ecarts_max > gamma).mean()),
            ecart_max_q95=float(np.quantile(ecarts_max, 0.95)),
            ecart_erm_moyen=float(np.mean(ecarts_erm)),
            risque_erm=float(np.mean(risques_erm)),
        ))
    print(f"{'N':>5} {'borne γ':>8} {'γ (1 hyp.)':>11} {'q95 écart max':>14} "
          f"{'écart ERM moyen':>16} {'R(ERM)':>7} {'tirages où borne violée':>24}")
    for l in lignes_b:
        print(f"{l['N']:>5} {l['gamma']:>8.3f} {l['gamma_seul']:>11.3f} {l['ecart_max_q95']:>14.3f} "
              f"{l['ecart_erm_moyen']:>16.3f} {l['risque_erm']:>7.3f} {l['echec']:>23.2%}")
    print(f"({time.perf_counter() - t0_b:.1f} s ; R minimal possible = {risque_reel_b.min():.2f})")
    return (lignes_b,)


@app.cell
def _(mo):
    mo.md(r"""
    ## C. Une classe infinie : écart train/test contre la borne VC

    Données : $x$ uniforme sur $[-1, 1]^2$, étiquette $\mathbb 1[x_1^2 + x_2^2 \ge 0{,}5]$
    inversée avec probabilité 0,1 (le meilleur classifieur possible a donc un risque de
    0,10). Classifieurs : séparateurs linéaires sur les monômes de degré $\le p$, appris par
    régression logistique presque sans régularisation. Le degré 1 ne peut pas tracer un
    cercle ; le degré 2 le peut ; au-delà, la classe est inutilement riche.

    Attention : la régression logistique minimise l'entropie croisée, pas la perte 0-1.
    Même avec beaucoup de données, le degré 2 plafonne un peu au-dessus de 0,10 : le
    meilleur classifieur au sens de l'entropie croisée n'est pas exactement le cercle. La
    borne VC n'en souffre pas : elle vaut pour toutes les fonctions de la classe.

    **Borne VC** (forme du cours *Learning from Data* d'Abu-Mostafa, Caltech, cours 7) :
    avec probabilité au moins $1-\delta$, pour toute $h \in \mathcal H$,
    $$R(h) \le \hat R(h) + \Omega, \qquad
    \Omega = \sqrt{\frac{8}{N}\ln\frac{4\,m_{\mathcal H}(2N)}{\delta}},$$
    avec $m_{\mathcal H}(2N) \le \sum_{i=0}^{d_{VC}}\binom{2N}{i}$ (Sauer), calculé en
    logarithme pour éviter les dépassements.
    """)


@app.cell
def _(gammaln, np):
    def log_sauer(n, d):
        """ln de la somme des C(n, i), i ≤ d (log-somme-exp de log-binomiaux)."""
        i = np.arange(0, min(d, n) + 1)
        logs = gammaln(n + 1) - gammaln(i + 1) - gammaln(n - i + 1)
        m = logs.max()
        return float(m + np.log(np.exp(logs - m).sum()))

    def omega_vc(N, d, delta):
        """Terme de la borne VC : sqrt(8/N · ln(4 m_H(2N) / δ))."""
        return float(np.sqrt(8 / N * (np.log(4 / delta) + log_sauer(2 * N, d))))

    return log_sauer, omega_vc


@app.cell
def _(mo):
    bouton = mo.ui.run_button(label="Lancer les calculs (partie C)")
    bouton
    return (bouton,)


@app.cell
def _(bouton, config, dimension_vc_poly, mo, monomes, np, omega_vc, time, warnings):
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer les calculs » pour la partie C.*"))
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def tirer(rng, n):
        X = rng.uniform(-1, 1, (n, 2))
        y = ((X ** 2).sum(axis=1) >= config["rayon2"]).astype(int)
        y ^= (rng.uniform(0, 1, n) < config["bruit_c"]).astype(int)
        return X, y

    rng_c = np.random.default_rng(42)
    X_test, y_test = tirer(rng_c, config["n_test"])
    phi_test = {p: monomes(X_test, p)[:, 1:] for p in config["degres"]}  # biais : géré par sklearn
    t0_c = time.perf_counter()
    resultats_c = []   # une ligne par (degré, N, répétition)
    for N in config["tailles"]:
        for rep in range(config["repetitions"]):
            X_tr, y_tr = tirer(rng_c, N)
            for p in config["degres"]:
                modele = make_pipeline(StandardScaler(),
                                       LogisticRegression(C=config["C_logistique"], max_iter=5000))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    modele.fit(monomes(X_tr, p)[:, 1:], y_tr)
                r_emp = float((modele.predict(monomes(X_tr, p)[:, 1:]) != y_tr).mean())
                r_test = float((modele.predict(phi_test[p]) != y_test).mean())
                resultats_c.append(dict(p=p, N=N, rep=rep, d=dimension_vc_poly(p),
                                        r_emp=r_emp, r_test=r_test,
                                        omega=omega_vc(N, dimension_vc_poly(p), config["delta"])))
    duree_c = time.perf_counter() - t0_c
    print(f"{len(resultats_c)} entraînements en {duree_c:.1f} s")
    return duree_c, resultats_c


@app.cell
def _(config, np, resultats_c):
    def moyenne(p, N, cle):
        return float(np.mean([r[cle] for r in resultats_c if r["p"] == p and r["N"] == N]))

    # Tableau : moyenne sur les répétitions
    print(f"{'degré':>5} {'d_VC':>4} {'N':>6} {'R_emp':>7} {'R_test':>7} {'écart':>7} {'Ω (borne)':>10} {'Ω/écart':>8}")
    for _p in config["degres"]:
        for _N in config["tailles"]:
            _re, _rt, _om = moyenne(_p, _N, "r_emp"), moyenne(_p, _N, "r_test"), moyenne(_p, _N, "omega")
            _ec = _rt - _re
            _rapport = f"{_om / _ec:8.0f}" if _ec > 1e-3 else "       —"
            print(f"{_p:>5} {(_p + 1) * (_p + 2) // 2:>4} {_N:>6} {_re:>7.3f} {_rt:>7.3f} {_ec:>7.3f} {_om:>10.3f} {_rapport}")
    respect = float(np.mean([r["r_test"] <= r["r_emp"] + r["omega"] for r in resultats_c]))
    print(f"\nBorne respectée dans {respect:.1%} des {len(resultats_c)} entraînements (garantie : ≥ {1 - config['delta']:.0%})")

    return moyenne, respect


@app.cell
def _(config, omega_vc):
    # Combien d'exemples faut-il pour que Ω descende sous 0,1 ? (recherche par dichotomie)
    def n_pour_omega(d, cible, delta=config["delta"]):
        bas, haut = 1, 10**12
        while bas < haut:
            mil = (bas + haut) // 2
            if omega_vc(mil, d, delta) <= cible:
                haut = mil
            else:
                bas = mil + 1
        return bas

    n_requis = {d: n_pour_omega(d, 0.1) for d in (3, 6, 10, 66)}
    for d_, n_ in n_requis.items():  # (Ω ≤ 0,1)
        print(f"d_VC = {d_:>2} : Ω ≤ 0.1 dès N = {n_:>10,}".replace(",", " ") + f"  (≈ {n_ / d_:,.0f} × d_VC)".replace(",", " "))
    return (n_requis,)


@app.cell
def _(SMOKE, config, mo, moyenne, np, plt, sauvegarder_figure):
    # --- Figure : (gauche) degré 2 selon N ; (droite) N = 200 selon le degré ---
    fig_c, (ax_n, ax_d) = plt.subplots(1, 2, figsize=(11, 4.2))
    tailles = np.array(config["tailles"])
    p_ref = 2
    ax_n.plot(tailles, [moyenne(p_ref, N, "r_emp") for N in tailles], "o-", label="erreur d'entraînement")
    ax_n.plot(tailles, [moyenne(p_ref, N, "r_test") for N in tailles], "s-", label="erreur de test")
    ax_n.plot(tailles, [moyenne(p_ref, N, "r_emp") + moyenne(p_ref, N, "omega") for N in tailles],
              "D--", color="black", label="borne VC (δ = 0,05)")
    ax_n.axhline(config["bruit_c"], color="0.5", ls=":", label="erreur de Bayes 0,10")
    ax_n.set(xscale="log", ylim=(0, 1.05), xlabel="N (exemples d'entraînement)",
             ylabel="taux d'erreur (perte 0-1)", title=f"Degré {p_ref} (d_VC = 6)")
    ax_n.legend(fontsize=8)
    N_ref = 200 if 200 in config["tailles"] else config["tailles"][1]
    degres = config["degres"]
    ax_d.plot(degres, [moyenne(p, N_ref, "r_emp") for p in degres], "o-", label="erreur d'entraînement")
    ax_d.plot(degres, [moyenne(p, N_ref, "r_test") for p in degres], "s-", label="erreur de test")
    ax_d.axhline(config["bruit_c"], color="0.5", ls=":")
    ax_d.set(ylim=(0, 0.6), xlabel="degré du polynôme", ylabel="taux d'erreur",
             title=f"N = {N_ref} (la borne VC dépasse 1 partout)")
    ax_d.legend(fontsize=8)
    fig_c.tight_layout()
    chemins_c = sauvegarder_figure(fig_c, "borne-vc-smoke" if SMOKE else "borne-vc", mo.notebook_dir() / "figures")
    print(f"Figure : {chemins_c[0]}")
    fig_c


@app.cell
def _(mo):
    mo.md(r"""
    ## D. No free lunch

    Domaine de 10 points $x = 0, 1, \dots, 9$. On apprend sur les points pairs
    (étiquettes connues) et on prédit les points impairs (**hors de l'ensemble
    d'entraînement**). On énumère les $2^{10} = 1\,024$ fonctions cibles possibles et on
    calcule, en fractions exactes, la précision moyenne de quatre algorithmes :

    - **plus proche voisin** (copie l'étiquette du point pair le plus proche à gauche) ;
    - **majorité** (prédit l'étiquette la plus fréquente de l'entraînement) ;
    - **toujours 0** ;
    - **anti-voisin** (fait exactement l'inverse du plus proche voisin).

    Puis on recommence en ne gardant que les cibles « régulières » : les 11 seuils
    $\mathbb 1[x \ge t]$.
    """)


@app.cell
def _(Fraction, config, itertools):
    taille = config["taille_domaine"]
    entrainement = list(range(0, taille, 2))
    hors = list(range(1, taille, 2))

    def plus_proche_voisin(etiq, x):
        return etiq[x - 1]                 # voisin pair à gauche (x impair ≥ 1)

    def majorite(etiq, x):
        uns = sum(etiq[i] for i in entrainement)
        return int(2 * uns > len(entrainement))

    def toujours_zero(etiq, x):
        return 0

    def anti_voisin(etiq, x):
        return 1 - etiq[x - 1]

    algorithmes = {"plus proche voisin": plus_proche_voisin, "majorité": majorite,
                   "toujours 0": toujours_zero, "anti-voisin": anti_voisin}

    def precision_hors(cibles):
        res = {}
        for nom, algo in algorithmes.items():
            total = Fraction(0)
            for f in cibles:
                # l'algorithme ne « voit » que les étiquettes des points pairs
                vues = {i: f[i] for i in entrainement}
                total += Fraction(sum(algo(vues, x) == f[x] for x in hors), len(hors))
            res[nom] = total / len(cibles)
        return res

    toutes = list(itertools.product([0, 1], repeat=taille))
    seuils_d = [tuple(int(x >= t) for x in range(taille)) for t in range(taille + 1)]
    nfl_toutes = precision_hors(toutes)
    nfl_seuils = precision_hors(seuils_d)
    print(f"{'algorithme':>20} {'toutes les cibles (1 024)':>26} {'cibles seuils (11)':>20}")
    for nom in algorithmes:
        print(f"{nom:>20} {nfl_toutes[nom]!s:>26} {float(nfl_seuils[nom]):>20.3f}  ({nfl_seuils[nom]})")
    return nfl_seuils, nfl_toutes


@app.cell
def _(
    Fraction,
    SMOKE,
    config,
    croissance_demi_plans,
    duree_c,
    ecart_monomes,
    lignes_b,
    m_alignes,
    m_carre,
    m_conique_6,
    m_conique_7,
    m_triangle,
    mo,
    moyenne,
    nfl_seuils,
    nfl_toutes,
    resultats_1d,
    respect,
):
    # --- Bilan : seuils chiffrés du README. En smoke, on vérifie seulement que tout tourne.
    controles = {
        "A. seuils : n + 1 étiquetages, intervalles : n(n+1)/2 + 1":
            all(s == n + 1 and i == n * (n + 1) // 2 + 1 for n, s, i in resultats_1d),
        "A. triangle pulvérisé (8/8), points alignés non (6/8)": m_triangle == 8 and m_alignes == 6,
        "A. carré : 14/16 (les deux « XOR » impossibles)": m_carre == 14,
        "A. demi-plans : croissance = n² − n + 2 pour n ≥ 3": all(m == n * n - n + 2 for n, m in croissance_demi_plans.items() if n >= 3),
        "A. monômes à la main = PolynomialFeatures": ecart_monomes < 1e-12,
        "A. coniques : 6 points pulvérisés, 7 non": m_conique_6 == 64 and m_conique_7 < 128,
        "B. borne de l'union violée dans ≤ δ des tirages": all(l["echec"] <= config["delta"] for l in lignes_b),
        "D. no free lunch : 1/2 exactement pour les 4 algorithmes": all(v == Fraction(1, 2) for v in nfl_toutes.values()),
        "D. cibles seuils : plus proche voisin ≥ 0,9": nfl_seuils["plus proche voisin"] >= Fraction(9, 10),
    }
    if not SMOKE:
        controles.update({
            "C. borne VC respectée dans ≥ 95 % des entraînements": respect >= 1 - config["delta"],
            "C. degré 2, N = 10 000 : erreur de test ≤ 0,12": moyenne(2, 10000, "r_test") <= 0.12,
            "C. degré 10, N = 50 : écart train/test ≥ 0,2": moyenne(10, 50, "r_test") - moyenne(10, 50, "r_emp") >= 0.2,
            "C. borne vide de sens (Ω > 1) pour N ≤ 100, tous degrés": all(moyenne(p, N, "omega") > 1 for p in config["degres"] for N in config["tailles"] if N <= 100),
            "C. Ω au moins 5 fois l'écart train/test mesuré": all(
                moyenne(p, N, "omega") >= 5 * (moyenne(p, N, "r_test") - moyenne(p, N, "r_emp"))
                for p in config["degres"] for N in config["tailles"]),
            "C. degré 2, N = 10 000 : borne ≤ 0,5": moyenne(2, 10000, "r_emp") + moyenne(2, 10000, "omega") <= 0.5,
        })
    for nom_c, ok in controles.items():
        print(f"[{'ok' if ok else 'ÉCHEC'}] {nom_c}")
    echecs = [n for n, ok in controles.items() if not ok]
    if echecs and not SMOKE:
        raise SystemExit(f"Seuils non atteints : {echecs}")
    mo.md(f"**Bilan** : {len(controles) - len(echecs)}/{len(controles)} contrôles passés ; partie C en {duree_c:.1f} s.")


if __name__ == "__main__":
    app.run()
