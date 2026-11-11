# Notebook 50-markov (chapitre 50, chaînes de Markov et équation de Bellman).
#
# La chaîne « site » du cours (accueil, catalogue, blog) en numpy : loi π₀Pⁿ par
# np.linalg.matrix_power, distribution invariante par np.linalg.eig sur Pᵀ et par un système
# linéaire, marche simulée et fréquences de visite ; PageRank par itération sur le petit
# graphe de liens du cours et sur un graphe aléatoire ; évaluation d’une politique par
# np.linalg.solve(I − γP, R), comparée à l’itération de Bellman ; générateur de texte par une
# chaîne de lettres estimée sur un court texte écrit pour le notebook. Vérifications exactes
# par sympy. numpy seul (pas de scipy).
#
#   make nb NUM=50              → éditeur marimo (bouton « Lancer le calcul »)
#   make nb NUM=50 MODE=run     → script complet
#   make nb NUM=50 MODE=smoke   → test rapide (moins de 30 s)

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
    # Chaînes de Markov, PageRank et équation de Bellman

    **Chapitre 50, sections 3 à 5, 7 et 8.** Une chaîne de Markov est décrite par sa matrice
    de transition $P$ (ligne = départ, colonne = arrivée, lignes de somme 1). La loi après
    $n$ pas est le vecteur ligne $\pi_n = \pi_0 P^n$, la distribution invariante vérifie
    $\pi P = \pi$, et, pour une politique fixée, les valeurs vérifient
    $V = R + \gamma P V$, soit $V = (I - \gamma P)^{-1} R$.

    Cinq expériences :

    1. la chaîne du site (accueil, catalogue, blog) : $\pi_0 P^n$ pour $n = 0$ à $30$, et
       l’invariante $(0{,}4 \,;\, 0{,}4 \,;\, 0{,}2)$ trouvée de deux façons ;
    2. une marche simulée : les fréquences de visite tendent vers l’invariante ;
    3. PageRank par itération $r \leftarrow rG$, sur les quatre pages du cours puis sur un
       graphe aléatoire, comparé à la résolution directe ;
    4. la valeur d’une politique sur la grille $2 \times 3$ du cours et pour le glacier de la
       section 7 : système linéaire contre itération de Bellman ;
    5. une chaîne de lettres estimée sur un court texte, et le texte qu’elle génère.
    """)


@app.cell
def _(is_smoke, np, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        n_marche=20_000 if SMOKE else 200_000,   # pas de la marche simulée (expérience 2)
        n_pages=60 if SMOKE else 500,            # pages du graphe aléatoire (expérience 3)
        n_lettres=200 if SMOKE else 600,         # lettres générées (expérience 5)
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
def _(bouton, config, mo, np, rng, time):
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer le calcul ».*"))
    t0 = time.perf_counter()

    # 1. La chaîne du site : 0 accueil, 1 catalogue, 2 blog
    P = np.array([[0.0, 0.7, 0.3], [0.8, 0.0, 0.2], [0.4, 0.6, 0.0]])
    pi0 = np.array([1.0, 0.0, 0.0])
    lois = np.array([pi0 @ np.linalg.matrix_power(P, n) for n in range(31)])
    # invariante, méthode 1 : vecteur propre de Pᵀ pour la valeur propre 1, divisé par sa somme
    vals, vecs = np.linalg.eig(P.T)
    j1 = int(np.argmin(np.abs(vals - 1)))
    pi_eig = np.real(vecs[:, j1]) / np.real(vecs[:, j1]).sum()
    # invariante, méthode 2 : (Pᵀ − I) πᵀ = 0, une équation remplacée par la somme = 1
    A_sys = P.T - np.eye(3)
    A_sys[-1] = 1.0
    pi_sys = np.linalg.solve(A_sys, np.array([0.0, 0.0, 1.0]))
    ecart_invariante = float(np.max(np.abs(pi_eig - pi_sys)))
    rayon = float(sorted(np.abs(vals))[-2])          # plus grand module des autres valeurs propres

    # 2. Marche simulée : on tire chaque pas par la ligne de l’état courant
    cumul = np.cumsum(P, axis=1)
    u = rng.random(config["n_marche"])
    etat, visites = 0, np.zeros(3)
    trajet = np.empty(config["n_marche"], dtype=int)
    for t in range(config["n_marche"]):
        etat = int(np.searchsorted(cumul[etat], u[t], side="right"))
        trajet[t] = etat
    visites = np.bincount(trajet, minlength=3) / config["n_marche"]
    ecart_marche = float(np.max(np.abs(visites - pi_sys)))
    checkpoints = np.unique(np.logspace(1, np.log10(config["n_marche"]), 40).astype(int))
    frequences = np.array([np.bincount(trajet[:c], minlength=3) / c for c in checkpoints])

    # 3. PageRank : les quatre pages du cours (liens 1 → 2, 1 → 3, 2 → 3, 3 → 1, 4 → 3)
    def google(A, d):
        n = A.shape[0]
        sortants = A.sum(axis=1)
        M = np.where(sortants[:, None] > 0, A / np.maximum(sortants, 1)[:, None], 1.0 / n)
        return d * M + (1 - d) / n * np.ones((n, n))

    def pagerank_iteration(G, tol=1e-12):
        r = np.full(G.shape[0], 1.0 / G.shape[0])
        for k in range(1, 10_000):
            r_suiv = r @ G
            if np.abs(r_suiv - r).sum() < tol:
                return r_suiv, k
            r = r_suiv
        return r, k

    A4 = np.zeros((4, 4))
    for (i, j) in [(0, 1), (0, 2), (1, 2), (2, 0), (3, 2)]:
        A4[i, j] = 1.0
    r4, iter4 = pagerank_iteration(google(A4, 0.85))
    # graphe aléatoire : chaque page cite quelques pages, certaines aucune
    n_p = config["n_pages"]
    A_rand = (rng.random((n_p, n_p)) < 3.0 / n_p).astype(float)
    np.fill_diagonal(A_rand, 0.0)
    A_rand[rng.random(n_p) < 0.05] = 0.0               # quelques pages sans lien sortant
    G_rand = google(A_rand, 0.85)
    r_rand, iter_rand = pagerank_iteration(G_rand)
    A_s = G_rand.T - np.eye(n_p)
    A_s[-1] = 1.0
    b_s = np.zeros(n_p)
    b_s[-1] = 1.0
    r_direct = np.linalg.solve(A_s, b_s)
    ecart_pagerank = float(np.max(np.abs(r_rand - r_direct)))

    # 4. Valeur d’une politique. Grille 2 × 3 (cases 1 à 6) : 1 → 2 → 3 → 6, 4 → 5 → 6, 6 absorbant
    suivant = [1, 2, 5, 4, 5, 5]
    P_grille = np.zeros((6, 6))
    for i, j in enumerate(suivant):
        P_grille[i, j] = 1.0
    R_grille = np.array([0, 0, 1, 0, 1, 0], dtype=float)

    def evaluer(Pm, R, gamma, tol=1e-10):
        V_exact = np.linalg.solve(np.eye(len(R)) - gamma * Pm, R)
        V, ecarts = np.zeros(len(R)), []
        while True:
            V = R + gamma * Pm @ V
            ecarts.append(float(np.max(np.abs(V - V_exact))))
            if ecarts[-1] < tol:
                return V_exact, V, ecarts

    V_grille, _V_it, ecarts_grille = evaluer(P_grille, R_grille, 0.9)
    P_meteo = np.array([[0.8, 0.2], [0.4, 0.6]])
    V_glacier, _V_it2, ecarts_glacier = evaluer(P_meteo, np.array([1.0, 0.0]), 0.9)
    _e = np.array(ecarts_glacier)
    _e = _e[_e > 1e-7]                                  # loin des erreurs d’arrondi
    rapports = _e[1:] / _e[:-1]

    # 5. Chaîne de lettres estimée sur un court texte (écrit pour ce notebook, aucune donnée externe)
    texte = (
        "une chaine de markov oublie le passe et ne garde que le present le visiteur passe de page "
        "en page le serveur passe de calme a charge la meteo passe du soleil a la pluie et chaque "
        "fois la question est la meme quelle est la probabilite de la suite quand on connait le "
        "present on compte les paires de lettres qui se suivent dans un texte on divise par le "
        "nombre de departs et on obtient une matrice de transition dont chaque ligne fait un "
        "quelques lignes suffisent pour que la chaine produise des mots presque francais"
    )
    alphabet = sorted(set(texte))
    index = {c: i for i, c in enumerate(alphabet)}
    comptes = np.zeros((len(alphabet), len(alphabet)))
    for a, b in zip(texte, texte[1:], strict=False):
        comptes[index[a], index[b]] += 1
    P_lettres = comptes / comptes.sum(axis=1, keepdims=True)
    lettre = "q"
    genere = [lettre]
    for _k in range(config["n_lettres"] - 1):
        lettre = alphabet[int(rng.choice(len(alphabet), p=P_lettres[index[lettre]]))]
        genere.append(lettre)
    texte_genere = "".join(genere)
    apres_q = alphabet[int(np.argmax(P_lettres[index["q"]]))]

    duree = time.perf_counter() - t0
    print(f"Calcul : {duree:.2f} s")
    print(f"1. π₁₀ = {np.round(lois[10], 4)}, invariante {np.round(pi_sys, 4)} (eig et système à "
          f"{ecart_invariante:.1e} près), rayon des autres valeurs propres {rayon:.4f}")
    print(f"2. {config['n_marche']} pas : fréquences {np.round(visites, 4)}, écart maximal {ecart_marche:.4f}")
    print(f"3. PageRank 4 pages : {np.round(r4, 4)} en {iter4} itérations ; graphe de {n_p} pages : "
          f"{iter_rand} itérations, écart à la résolution directe {ecart_pagerank:.1e}")
    print(f"4. grille : V = {np.round(V_grille, 4)} ({len(ecarts_grille)} balayages) ; glacier : "
          f"V = {np.round(V_glacier, 4)} ({len(ecarts_glacier)} balayages, rapport des écarts ≤ {rapports.max():.3f})")
    print(f"5. après « q » : « {apres_q} » ; texte généré : {texte_genere[:120]}…")
    return (P, P_lettres, V_glacier, V_grille, alphabet, apres_q, checkpoints, duree, ecart_invariante,
            ecart_marche, ecart_pagerank, ecarts_glacier, ecarts_grille, frequences, iter4, lois, pi_sys,
            r4, rapports, rayon, texte_genere)


@app.cell
def _(sp):
    # --- Vérifications exactes (sympy) : les nombres du cours ---
    _R = sp.Rational
    _P = sp.Matrix([[0, _R(7, 10), _R(3, 10)], [_R(8, 10), 0, _R(2, 10)], [_R(4, 10), _R(6, 10), 0]])
    _pi = sp.Matrix([[_R(2, 5), _R(2, 5), _R(1, 5)]])
    assert _pi * _P == _pi
    assert sp.Matrix([[1, 0, 0]]) * _P**2 == sp.Matrix([[_R(17, 25), _R(9, 50), _R(7, 50)]])
    _M = sp.Matrix([[0, _R(1, 2), _R(1, 2), 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0]])
    _d = _R(17, 20)
    _G = _d * _M + (1 - _d) / 4 * sp.ones(4, 4)
    _r = sp.Matrix([[_R(659, 1769), _R(27713, 141520), _R(2789, 7076), _R(3, 80)]])
    assert _r * _G == _r and sum(_r) == 1
    _W = sp.Matrix([[_R(4, 5), _R(1, 5)], [_R(2, 5), _R(3, 5)]])
    _V = (sp.eye(2) - _R(9, 10) * _W).inv() * sp.Matrix([1, 0])
    assert _V == sp.Matrix([_R(115, 16), _R(45, 8)])
    print("sympy : invariante (2/5, 2/5, 1/5), π₂ = (17/25, 9/50, 7/50), PageRank exact pour d = 17/20, "
          "valeurs du glacier (115/16, 45/8)")
    return


@app.cell
def _(SMOKE, checkpoints, ecarts_glacier, frequences, lois, mo, np, pi_sys, plt, r4, sauvegarder_figure):
    # --- Figure, sauvegardée dans figures/ (suffixe -smoke en mode smoke) ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    noms = ["accueil", "catalogue", "blog"]
    ax = axes[0, 0]
    for _j in range(3):
        ax.plot(lois[:, _j], marker="o", ms=3, label=noms[_j])
        ax.axhline(pi_sys[_j], color="gray", linestyle=":", lw=1)
    ax.set(title="Loi πₙ = π₀Pⁿ depuis l’accueil", xlabel="n (clics)", ylabel="probabilité")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for _j in range(3):
        ax.semilogx(checkpoints, frequences[:, _j], label=noms[_j])
        ax.axhline(pi_sys[_j], color="gray", linestyle=":", lw=1)
    ax.set(title="Fréquences de visite d’une marche simulée", xlabel="nombre de pas", ylabel="part du temps")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.bar(["1", "2", "3", "4"], r4, color="tab:blue")
    ax.set(title="PageRank des quatre pages (d = 0,85)", xlabel="page", ylabel="rang")
    ax = axes[1, 1]
    ax.semilogy(np.arange(1, len(ecarts_glacier) + 1), ecarts_glacier, label="écart à V (glacier)")
    ax.semilogy(np.arange(1, len(ecarts_glacier) + 1), 0.9 ** np.arange(1, len(ecarts_glacier) + 1) * 10,
                linestyle="--", label="10 × 0,9ᵏ")
    ax.set(title="Itération de Bellman, γ = 0,9", xlabel="balayage k", ylabel="plus grand écart")
    ax.legend(fontsize=8)
    fig.tight_layout()
    chemins = sauvegarder_figure(fig, "markov-smoke" if SMOKE else "markov", mo.notebook_dir() / "figures")
    print(f"Figure : {chemins[0].name}")
    fig  # noqa: B018 (marimo affiche la dernière expression)
    return


@app.cell
def _(SMOKE, V_glacier, V_grille, apres_q, duree, ecart_invariante, ecart_marche, ecart_pagerank,
      ecarts_grille, lois, mo, np, pi_sys, r4, rapports, rayon):
    # --- Bilan : seuils chiffrés du README, vérifiés aussi en smoke ---
    echecs = []
    if np.max(np.abs(pi_sys - [0.4, 0.4, 0.2])) > 1e-12 or ecart_invariante > 1e-12:
        echecs.append(f"invariante {pi_sys} (écart eig/système {ecart_invariante:.1e})")
    if np.max(np.abs(lois[2] - [0.68, 0.18, 0.14])) > 1e-12:
        echecs.append(f"π₂ = {lois[2]}")
    if np.max(np.abs(lois[30] - pi_sys)) > 1e-3:
        echecs.append(f"π₃₀ trop loin de l’invariante : {lois[30]}")
    if abs(rayon - (0.5 + np.sqrt(5) / 10)) > 1e-12:
        echecs.append(f"rayon des autres valeurs propres {rayon}")
    tol_marche = 0.03 if SMOKE else 0.01
    if ecart_marche > tol_marche:
        echecs.append(f"fréquences de visite à {ecart_marche:.4f} de l’invariante (seuil {tol_marche})")
    if np.max(np.abs(r4 - [659 / 1769, 27713 / 141520, 2789 / 7076, 3 / 80])) > 1e-9 or ecart_pagerank > 1e-9:
        echecs.append(f"PageRank : {r4}, écart {ecart_pagerank:.1e}")
    if np.max(np.abs(V_grille - [0.81, 0.9, 1, 0.9, 1, 0])) > 1e-12 or len(ecarts_grille) > 4:
        echecs.append(f"grille : {V_grille} en {len(ecarts_grille)} balayages")
    if np.max(np.abs(V_glacier - [7.1875, 5.625])) > 1e-12 or rapports.max() > 0.9 + 1e-6:
        echecs.append(f"glacier : {V_glacier}, rapport {rapports.max():.4f}")
    if apres_q != "u":
        echecs.append(f"lettre après q : {apres_q}")
    if echecs:
        raise SystemExit("Bilan : ÉCHEC : " + " ; ".join(echecs))
    print(f"Bilan : OK (invariante (0,4 ; 0,4 ; 0,2) par eig et par système, fréquences à {ecart_marche:.4f}, "
          f"PageRank exact à 1e-9, grille en {len(ecarts_grille)} balayages, glacier (7,1875 ; 5,625), "
          f"écart de Bellman multiplié par au plus 0,9 à chaque balayage, {duree:.2f} s)")
    _f = f"{ecart_marche:.4f}".replace(".", ",")
    mo.md(f"**Bilan** : l’invariante vaut $(0{{,}}4 \\,;\\, 0{{,}}4 \\,;\\, 0{{,}}2)$ ; les fréquences "
          f"de visite s’en écartent de {_f} au plus ; les valeurs du glacier sont $(7{{,}}1875 \\,;\\, 5{{,}}625)$.")
    return


if __name__ == "__main__":
    app.run()
