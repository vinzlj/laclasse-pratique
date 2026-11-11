# Notebook 46-entropie-croisee (chapitre 46, Information I : surprise, entropie, entropie
# croisée, KL).
#
# Un tout petit modèle de langage par caractères, en numpy : loi des caractères (unigrammes),
# puis loi du caractère suivant sachant le précédent (bigrammes), apprises sur le début d’un
# court texte et évaluées sur la fin. On mesure l’entropie croisée en bits par caractère et
# la perplexité, on vérifie H(p, q) = H(p) + KL(p ‖ q) et on compare les deux sens de la KL.
#
#   make nb NUM=46              → éditeur marimo (bouton « Lancer le calcul »)
#   make nb NUM=46 MODE=run     → script complet
#   make nb NUM=46 MODE=smoke   → test rapide (moins de 30 s)

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
    # Entropie croisée et perplexité d’un modèle de caractères

    **Chapitre 46, sections 3 à 6.** Un modèle de langage donne, à chaque position, une loi $q$
    sur le caractère suivant ; sa perte est la moyenne des surprises $-\log_2 q(\text{vrai caractère})$,
    en bits par caractère : c’est l’**entropie croisée**. Sa **perplexité** vaut $2$ puissance
    cette moyenne. On compare trois modèles, appris sur les 80 premiers pour cent d’un court
    texte et évalués sur les 20 derniers :

    1. **uniforme** : tous les caractères de l’alphabet ont la même probabilité ;
    2. **unigrammes** : la fréquence de chaque caractère dans le texte d’apprentissage ;
    3. **bigrammes** : la fréquence du caractère suivant **sachant le précédent**, lissée par
       $\alpha$ (on ajoute $\alpha$ à chaque compte, pour qu’aucune probabilité ne soit nulle :
       une seule probabilité nulle rendrait l’entropie croisée infinie).

    On vérifie aussi, sur les lois de caractères, que $H(p, q) = H(p) + D_{\mathrm{KL}}(p \,\|\, q)$
    et que les deux sens de la KL diffèrent.
    """)


@app.cell
def _(is_smoke, np, seed_everything, style_commun):
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    # Texte écrit pour ce notebook (aucune donnée téléchargée), en minuscules sans accents
    # pour garder un petit alphabet : 26 lettres, l’espace, le point et la virgule.
    TEXTE = (
        "le chat dort sur le canape. le chien dort sous la table. le chat mange, puis le chat dort. "
        "la souris passe sous la table, le chat ouvre un oeil et la souris repart. le chien mange "
        "dans la cuisine, puis le chien dort pres de la porte. le soir, le chat monte sur le lit et "
        "dort encore. la souris mange une miette sous la table. le matin, le chien sort, le chat "
        "reste sur le canape, et la souris dort dans le mur. quand le chien rentre, le chat se leve, "
        "passe sous la table et repart dormir sur le lit. la souris ne sort plus. le chat dort, le "
        "chien dort, la souris dort : la maison dort. "
    )
    config = dict(
        part_apprentissage=0.8,
        alphas=np.geomspace(0.001, 10, 6 if SMOKE else 200),   # lissages essayés
        n_genere=80,                                            # caractères tirés du modèle
    )
    rng = np.random.default_rng(SEED)
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, {len(TEXTE)} caractères, "
          f"{len(config['alphas'])} valeurs de alpha")
    return SMOKE, TEXTE, config, rng


@app.cell
def _(np):
    def entropie_bits(p):
        p = np.asarray(p, dtype=float)
        p = p[p > 0]                      # convention 0 log 0 = 0
        return float(-(p * np.log2(p)).sum())

    def entropie_croisee_bits(p, q):
        p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
        m = p > 0
        return float(-(p[m] * np.log2(q[m])).sum())

    def kl_bits(p, q):
        p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
        m = p > 0
        return float((p[m] * np.log2(p[m] / q[m])).sum())

    return entropie_bits, entropie_croisee_bits, kl_bits


@app.cell
def _(TEXTE, config, np):
    alphabet = sorted(set(TEXTE))
    V = len(alphabet)
    indice = {c: i for i, c in enumerate(alphabet)}
    codes = np.array([indice[c] for c in TEXTE])
    coupe = int(config["part_apprentissage"] * len(codes))
    appr, test = codes[:coupe], codes[coupe:]
    p_appr = np.bincount(appr, minlength=V) / len(appr)
    p_test = np.bincount(test, minlength=V) / len(test)
    comptes_bi = np.zeros((V, V))
    np.add.at(comptes_bi, (appr[:-1], appr[1:]), 1)
    print(f"Alphabet de {V} caractères, {len(appr)} pour apprendre, {len(test)} pour évaluer")
    return V, alphabet, appr, comptes_bi, p_appr, p_test, test


@app.cell
def _(mo):
    # Le bouton empêche le calcul de se relancer à chaque modification d’une cellule amont.
    bouton = mo.ui.run_button(label="Lancer le calcul")
    bouton  # noqa: B018 (marimo affiche la dernière expression)
    return (bouton,)


@app.cell
def _(V, appr, bouton, comptes_bi, config, mo, np, p_appr, test, time):
    # En script (make nb MODE=run|smoke), personne ne clique : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer le calcul ».*"))

    t0 = time.perf_counter()

    def perte_bigrammes(alpha):
        """Entropie croisée (bits par caractère) des bigrammes lissés sur le texte de test."""
        q = (comptes_bi + alpha) / (comptes_bi + alpha).sum(axis=1, keepdims=True)
        return float(-np.log2(q[test[:-1], test[1:]]).mean()), q

    perte_uniforme = float(np.log2(V))
    # Unigrammes lissés de Laplace (+1 à chaque compte) : un caractère absent de l’apprentissage
    # garde une probabilité non nulle.
    q_uni = (p_appr * len(appr) + 1) / (len(appr) + V)
    perte_unigrammes = float(-np.log2(q_uni[test[1:]]).mean())
    pertes_bi = np.array([perte_bigrammes(a)[0] for a in config["alphas"]])
    k = int(np.argmin(pertes_bi))
    alpha_opt = float(config["alphas"][k])
    perte_bi, q_bi = perte_bigrammes(alpha_opt)
    duree = time.perf_counter() - t0
    for _nom, _L in (("uniforme", perte_uniforme), ("unigrammes", perte_unigrammes),
                     (f"bigrammes (alpha = {alpha_opt:.3g})", perte_bi)):
        print(f"{_nom:<28} {_L:.3f} bits par caractère, perplexité {2 ** _L:.2f}")
    print(f"calcul {duree:.2f} s")
    return alpha_opt, duree, perte_bi, perte_unigrammes, perte_uniforme, pertes_bi, q_bi


@app.cell
def _(entropie_bits, entropie_croisee_bits, kl_bits, np, p_appr, p_test):
    # --- Décomposition H(p, q) = H(p) + KL(p ‖ q), sur les lois de caractères ---
    # p : loi des caractères du test ; q : loi apprise, lissée pour n’avoir aucun zéro.
    q_lisse = (p_appr + 1e-3) / (p_appr + 1e-3).sum()
    H_p = entropie_bits(p_test)
    H_pq = entropie_croisee_bits(p_test, q_lisse)
    D_pq = kl_bits(p_test, q_lisse)
    uniforme = np.full(len(p_appr), 1 / len(p_appr))
    D_pu, D_up = kl_bits(p_appr, uniforme), kl_bits(uniforme, q_lisse)
    print(f"H(p) = {H_p:.4f}, H(p, q) = {H_pq:.4f}, KL(p ‖ q) = {D_pq:.4f} bit ; "
          f"écart à la décomposition {abs(H_pq - H_p - D_pq):.1e}")
    print(f"KL(caractères ‖ uniforme) = {D_pu:.3f} bit ; KL(uniforme ‖ caractères) = {D_up:.3f} bit")
    return D_pq, D_pu, D_up, H_p, H_pq


@app.cell
def _(sp):
    # --- Vérification exacte (sympy) : l’exemple météo du cours, en bits ---
    _p = [sp.Rational(1, 2), sp.Rational(1, 4), sp.Rational(1, 4)]
    _q = [sp.Rational(1, 4), sp.Rational(1, 4), sp.Rational(1, 2)]

    def _log2(_x):
        return sp.log(_x, 2)

    _H = sp.nsimplify(-sum(_a * _log2(_a) for _a in _p))
    _Hpq = sp.nsimplify(-sum(_a * _log2(_b) for _a, _b in zip(_p, _q, strict=True)))
    _D = sp.nsimplify(sum(_a * _log2(_a / _b) for _a, _b in zip(_p, _q, strict=True)))
    assert (_H, _Hpq, _D) == (sp.Rational(3, 2), sp.Rational(7, 4), sp.Rational(1, 4)), (_H, _Hpq, _D)
    assert sp.simplify(_Hpq - _H - _D) == 0
    # Perplexité du modèle A de l’ouverture : 2^2 = 4, et (1/2 · 1/4 · 1/8)^(−1/3) = 4
    assert sp.nsimplify((sp.Rational(1, 64)) ** sp.Rational(-1, 3)) == 4
    print("sympy : H(p) = 3/2, H(p, q) = 7/4, KL(p ‖ q) = 1/4 bit ; perplexité du modèle A = 4")
    return


@app.cell
def _(alphabet, alpha_opt, config, np, q_bi, rng, test):
    # --- Quelques caractères tirés du modèle de bigrammes (graine 42) ---
    _c = int(test[0])
    _sortie = [alphabet[_c]]
    for _ in range(config["n_genere"] - 1):
        _c = int(rng.choice(len(alphabet), p=q_bi[_c]))
        _sortie.append(alphabet[_c])
    texte_genere = "".join(_sortie)
    print(f"Tiré des bigrammes (alpha = {alpha_opt:.3g}) : « {texte_genere} »")
    return (texte_genere,)


@app.cell
def _(
    SMOKE,
    alpha_opt,
    config,
    mo,
    perte_bi,
    perte_unigrammes,
    perte_uniforme,
    pertes_bi,
    plt,
    sauvegarder_figure,
):
    # --- Figure (suffixe -smoke : n’écrase pas celle du run complet) ---
    _fig, _ax = plt.subplots()
    _ax.semilogx(config["alphas"], pertes_bi, label="bigrammes lissés")
    _ax.axhline(perte_uniforme, ls="--", color="gray", lw=1, label="uniforme")
    _ax.axhline(perte_unigrammes, ls=":", color="black", lw=1, label="unigrammes")
    _ax.scatter([alpha_opt], [perte_bi], color="C3", zorder=3, label="meilleur lissage")
    _ax.set(xlabel="lissage alpha (échelle logarithmique)", ylabel="entropie croisée (bits par caractère)",
            title="Perte des trois modèles sur le texte de test")
    _ax.legend(fontsize=9)
    _chemins = sauvegarder_figure(_fig, "entropie-croisee-smoke" if SMOKE else "entropie-croisee",
                                 mo.notebook_dir() / "figures")
    print(f"Figure : {_chemins[0].name}")
    _fig  # noqa: B018 (marimo affiche la dernière expression)
    return


@app.cell
def _(D_pq, D_pu, D_up, H_p, H_pq, V, duree, mo, np, perte_bi, perte_unigrammes, perte_uniforme):
    # --- Bilan : seuils du README, vérifiés aussi en smoke ---
    _echecs = []
    if abs(perte_uniforme - np.log2(V)) > 1e-12:
        _echecs.append("perte du modèle uniforme différente de log₂ V")
    if not perte_bi < perte_unigrammes < perte_uniforme:
        _echecs.append(f"ordre des pertes inattendu : {perte_bi:.3f}, {perte_unigrammes:.3f}, {perte_uniforme:.3f}")
    if perte_bi > 3.0:
        _echecs.append(f"bigrammes : {perte_bi:.3f} bits par caractère (attendu ≤ 3)")
    if abs(H_pq - H_p - D_pq) > 1e-12 or D_pq < 0:
        _echecs.append("décomposition H(p, q) = H(p) + KL(p ‖ q) non vérifiée")
    if abs(D_pu - D_up) < 0.05:
        _echecs.append(f"les deux sens de la KL sont trop proches ({D_pu:.3f} et {D_up:.3f})")
    if _echecs:
        raise SystemExit("Bilan : ÉCHEC : " + " ; ".join(_echecs))
    _msg = (f"Bilan : OK (perplexités uniforme {2 ** perte_uniforme:.1f}, unigrammes {2 ** perte_unigrammes:.2f}, "
            f"bigrammes {2 ** perte_bi:.2f} ; décomposition à 1e-12 près ; KL {D_pu:.3f} et {D_up:.3f} bit ; "
            f"calcul {duree:.2f} s)")
    print(_msg)
    mo.md(_msg.replace(".", ","))
    return


if __name__ == "__main__":
    app.run()
