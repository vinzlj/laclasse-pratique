# Exercice 37 : l'effondrement des modèles (model collapse), chapitre 37.
# Un modèle réajusté, génération après génération, sur ses propres échantillons perd
# d'abord la queue de la distribution, puis converge vers un point.
# Partie 1 : une gaussienne (numpy pur). Partie 2 : des n-grammes de mots sur TinyShakespeare.
# Dans les deux cas, on compare « remplacer » les données par les échantillons du modèle,
# « garder un peu de vraies données » et « accumuler » (ne jamais rien jeter).
#
#   make ex NUM=37              → éditeur marimo (interactif)
#   make ex NUM=37 MODE=run     → script complet (vérifie les seuils du README)
#   make ex NUM=37 MODE=smoke   → test rapide (< 30 s)
#
# Aucun réseau de neurones, aucun GPU : numpy seulement. Les figures du site
# (site/scripts/figures/ch37_*.py) lisent figures/resultats.json, écrit par le run complet.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import json
    import re
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.special import digamma, erfc

    from ia_commun import PALETTE, is_smoke, sauvegarder_figure, seed_everything, style_commun


@app.cell
def _():
    mo.md(r"""
    # Exercice 37 : l'effondrement des modèles

    Les grands modèles écrivent désormais une part du texte, des images et du code qu'on
    trouve en ligne, et l'on se sert de leurs sorties pour entraîner les modèles suivants
    (données synthétiques). Que se passe-t-il si chaque génération de modèle apprend
    **seulement** sur ce qu'a produit la précédente ?

    Shumailov et al. (2023, puis *Nature* 2024) appellent le résultat **effondrement des
    modèles** (*model collapse*) : d'abord la **queue** de la distribution disparaît (les
    événements rares ne sont plus tirés, donc plus appris), puis la distribution se resserre
    jusqu'à un point. L'article commence par deux cas sans aucun réseau de neurones, que
    l'on reproduit ici : une **gaussienne** et une **distribution discrète** (des mots).

    La boucle, à chaque génération $g$ :

    $$
    \text{données}_g \;\xrightarrow{\ \text{ajuster}\ }\; \text{modèle}_g
    \;\xrightarrow{\ \text{échantillonner}\ }\; \text{données}_{g+1}
    $$

    **Ce qui est optimisé** : à chaque génération, la **vraisemblance** des données du
    moment (maximum de vraisemblance, chapitre 00). Pour une gaussienne, cela revient à
    prendre la moyenne et l'écart-type empiriques ; pour des n-grammes, à compter. Pas de
    descente de gradient : le phénomène vient de l'échantillonnage, pas de l'optimiseur.

    Trois régimes sont comparés :

    - **remplacer** : la génération $g+1$ n'apprend que sur les échantillons du modèle $g$
      (le cadre de Shumailov et al.) ;
    - **10 % de réel** : 90 % d'échantillons, 10 % de vraies données neuves ;
    - **accumuler** : on garde les vraies données de départ et tous les échantillons
      précédents, on ajoute les nouveaux (le cadre de Gerstgrasser et al., 2024).
    """)
    return


@app.cell
def _():
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        # Partie 1 : gaussienne N(0, 1), n points par génération, chaînes indépendantes
        n_chaines=40 if SMOKE else 200,
        generations_gauss=100 if SMOKE else 1000,
        tailles=[10, 100, 1000],          # n : nombre de points tirés à chaque génération
        n_reference=100,                  # n des régimes « sans biais », « 10 % de réel », « accumuler »
        part_reel=0.10,
        # Partie 2 : n-grammes de mots sur TinyShakespeare (90 % entraînement, 10 % test)
        generations_ngram=4 if SMOKE else 30,
        generations_unigramme=20 if SMOKE else 300,
        n_flux=2_000,                     # flux échantillonnés en parallèle (bigrammes)
        temperatures=[1.0, 0.7],
        lambda_interp=0.7,                # entropie croisée test : λ·bigramme + (1−λ)·unigramme lissé
        gens_rang=[0, 5, 30],             # générations montrées sur la courbe rang-fréquence
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}\nconfig {config}")
    return SEED, SMOKE, config


@app.cell
def _():
    mo.md(r"""
    ## 1. Une gaussienne qui apprend sur elle-même

    Génération 0 : on tire $n$ points d'une vraie loi $\mathcal{N}(0, 1)$ et on ajuste une
    gaussienne (moyenne $\hat\mu$, écart-type $\hat\sigma$ empiriques). Génération 1 : on
    tire $n$ points de **ce modèle**, on réajuste. Et ainsi de suite.

    Pourquoi $\hat\sigma$ diminuerait-il ? Chaque génération multiplie la variance par un
    facteur aléatoire :

    $$
    \hat\sigma_{g+1}^2 = \hat\sigma_g^2 \cdot \frac{\chi^2_{n-1}}{n}
    \qquad\text{(estimateur du maximum de vraisemblance)}
    $$

    En moyenne, ce facteur vaut $(n-1)/n$, un peu moins de 1 : la variance fond de
    $1/n$ par génération. Corriger le biais (diviser par $n-1$, estimateur « sans biais »)
    rend la **moyenne** du facteur égale à 1, mais pas sa **médiane** : le logarithme du
    facteur a une espérance négative, environ $-1/(n-1)$, et $\log\hat\sigma^2$ fait une
    marche aléatoire qui dérive vers $-\infty$. Une variance finit par s'écraser presque
    sûrement, même « sans biais ».

    Calcul à la main pour $n = 100$ et 1 000 générations (estimateur sans biais) :
    $\log \hat\sigma^2 \approx 1000 \times (-1/99) \approx -10{,}1$, soit un écart-type
    médian de $e^{-5{,}05} \approx 0{,}006$. La cellule suivante compare ce calcul à la
    simulation.
    """)
    return


@app.function
def queue_gaussienne(mu, sigma, seuil=2.0):
    """P(|X| > seuil) sous le modèle N(mu, sigma²). Vaut 0,0455 pour N(0, 1) et seuil 2."""
    # Avec n = 10, l'écart-type finit par valoir exactement 0 en virgule flottante (un
    # « Dirac ») : la queue vaut alors 1 si ce point est au-delà du seuil, 0 sinon.
    mu, sigma = np.asarray(mu, float), np.asarray(sigma, float)
    r2 = np.sqrt(2.0) * np.where(sigma > 0, sigma, 1.0)
    q = 0.5 * erfc((seuil - mu) / r2) + 0.5 * erfc((seuil + mu) / r2)
    return np.where(sigma > 0, q, (np.abs(mu) > seuil).astype(float))


@app.function
def generations_gaussiennes(rng, n_chaines, n, generations, regime="remplacer", part_reel=0.1):
    """Ajuste une gaussienne, échantillonne, réajuste, `generations` fois, sur `n_chaines` chaînes
    indépendantes à la fois (une ligne par chaîne).

    regime : « remplacer » (écart-type du maximum de vraisemblance, ddof = 0),
             « sans-biais » (ddof = 1), « reel » (une part `part_reel` de vraies données neuves
             à chaque génération), « accumuler » (toutes les données depuis le début).
    Renvoie mu, sigma de forme (generations + 1, n_chaines) ; la ligne 0 est le modèle ajusté
    sur les vraies données.
    """
    mu = np.empty((generations + 1, n_chaines))
    sigma = np.empty_like(mu)
    ddof = 1 if regime == "sans-biais" else 0
    x = rng.standard_normal((n_chaines, n))            # génération 0 : vraies données N(0, 1)
    # Pour « accumuler » : sommes courantes (somme, somme des carrés, effectif) de toutes les données
    s1, s2, cnt = x.sum(1), (x**2).sum(1), n
    mu[0], sigma[0] = x.mean(1), x.std(1, ddof=ddof)
    n_reel = round(part_reel * n) if regime == "reel" else 0
    for g in range(1, generations + 1):
        # Échantillonner le modèle courant (une gaussienne par chaîne)
        x = mu[g - 1][:, None] + sigma[g - 1][:, None] * rng.standard_normal((n_chaines, n - n_reel))
        if n_reel:
            x = np.concatenate([x, rng.standard_normal((n_chaines, n_reel))], axis=1)
        if regime == "accumuler":
            s1, s2, cnt = s1 + x.sum(1), s2 + (x**2).sum(1), cnt + n
            mu[g] = s1 / cnt
            sigma[g] = np.sqrt(np.maximum(s2 / cnt - mu[g] ** 2, 0.0))
        else:
            mu[g], sigma[g] = x.mean(1), x.std(1, ddof=ddof)
    return mu, sigma


@app.function
def sigma_median_theorique(n, generations, sans_biais):
    """Écart-type médian prévu après `generations` générations : exp(G/2 · E[log facteur]).

    facteur = chi²_{n-1} / n (maximum de vraisemblance) ou chi²_{n-1} / (n-1) (sans biais) ;
    E[log chi²_k] = digamma(k/2) + log 2. La somme de G logarithmes indépendants est presque
    gaussienne : sa médiane est proche de sa moyenne.
    """
    k = n - 1
    e_log = digamma(k / 2) + np.log(2.0) - np.log(k if sans_biais else n)
    return float(np.exp(0.5 * generations * e_log))


@app.cell
def _(config):
    _rng = np.random.default_rng(42)
    _t0 = time.perf_counter()
    _G, _C, _nr = config["generations_gauss"], config["n_chaines"], config["n_reference"]
    gauss = {}
    for _n in config["tailles"]:
        gauss[f"remplacer-{_n}"] = generations_gaussiennes(_rng, _C, _n, _G, "remplacer")
    gauss["sans-biais"] = generations_gaussiennes(_rng, _C, _nr, _G, "sans-biais")
    gauss["reel"] = generations_gaussiennes(_rng, _C, _nr, _G, "reel", config["part_reel"])
    gauss["accumuler"] = generations_gaussiennes(_rng, _C, _nr, _G, "accumuler")
    duree_gauss = time.perf_counter() - _t0

    # Bilan par régime : écart-type médian (et 10e-90e centiles), masse de la queue |x| > 2,
    # part des chaînes « effondrées » (écart-type < 0,1)
    bilan_gauss = {}
    for _k, (_mu, _sg) in gauss.items():
        _q = queue_gaussienne(_mu[-1], _sg[-1])
        bilan_gauss[_k] = dict(
            sigma_median=float(np.median(_sg[-1])),
            sigma_p10=float(np.percentile(_sg[-1], 10)), sigma_p90=float(np.percentile(_sg[-1], 90)),
            queue2_mediane=float(np.median(_q)),
            part_effondrees=float(np.mean(_sg[-1] < 0.1)),
            part_nulles=float(np.mean(_sg[-1] == 0)),     # écart-type exactement 0 (n = 10)
            # première génération où l'écart-type médian passe sous 0,5 puis sous 0,01 (−1 : jamais)
            gen_sous_0_5=int(np.argmax(np.median(_sg, 1) < 0.5)) if (np.median(_sg, 1) < 0.5).any() else -1,
            gen_sous_0_01=int(np.argmax(np.median(_sg, 1) < 0.01)) if (np.median(_sg, 1) < 0.01).any() else -1,
            mu_ecart_type=float(_mu[-1].std()),
        )
    theorie = {
        "remplacer-100": sigma_median_theorique(100, _G, sans_biais=False),
        "sans-biais": sigma_median_theorique(_nr, _G, sans_biais=True),
        "remplacer-1000": sigma_median_theorique(1000, _G, sans_biais=False),
    }
    print(f"{_C} chaînes × {_G} générations, {duree_gauss:.1f} s")
    print(f"{'régime':>16} | σ médian | σ prévu | σ p10–p90        | P(|x|>2) médiane | σ < 0,1 | σ méd. < 0,5 / < 0,01 à la gén.")
    for _k, _b in bilan_gauss.items():
        _th = f"{theorie[_k]:.2g}" if _k in theorie else "—"
        print(f"{_k:>16} | {_b['sigma_median']:8.3g} | {_th:>7} | {_b['sigma_p10']:.3g} – {_b['sigma_p90']:.3g}"
              f" | {_b['queue2_mediane']:.3g} | {_b['part_effondrees']:.0%} | {_b['gen_sous_0_5']} / {_b['gen_sous_0_01']}")
    print("Pour N(0, 1) : σ = 1, P(|x| > 2) = 0,0455")
    return bilan_gauss, duree_gauss, gauss, theorie


@app.cell
def _(SMOKE, config, gauss):
    # Figure 1 : écart-type au fil des générations (médiane et 10e-90e centiles des chaînes),
    # puis densités ajustées d'une chaîne (n = 100, remplacer) à quelques générations.
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    _gen = np.arange(config["generations_gauss"] + 1)
    _series = [(f"remplacer-{n}", f"remplacer, n = {n}") for n in config["tailles"]] + [
        ("sans-biais", "remplacer, n = 100, sans biais"),
        ("reel", "n = 100, 10 % de réel"), ("accumuler", "accumuler, n = 100")]
    for _i, (_k, _lib) in enumerate(_series):
        _s = gauss[_k][1]
        _a1.plot(_gen, np.median(_s, 1), color=PALETTE[_i], lw=1.8, label=_lib)
        _a1.fill_between(_gen, np.percentile(_s, 10, 1), np.percentile(_s, 90, 1),
                         color=PALETTE[_i], alpha=0.12, lw=0)
    _a1.set(yscale="log", ylim=(1e-6, 3), xlabel="génération", ylabel="écart-type ajusté σ̂",
            title="Écart-type médian (bande : 10e–90e centiles)")
    _a1.legend(fontsize=8)
    _x = np.linspace(-4, 4, 801)
    _mu, _sg = gauss["remplacer-100"]
    _a2.plot(_x, np.exp(-_x**2 / 2) / np.sqrt(2 * np.pi), color="k", ls=":", label="vraie loi N(0, 1)")
    for _i, _g in enumerate([g for g in (0, 50, 200, 500) if g <= config["generations_gauss"]]):
        _m, _s = _mu[_g, 0], _sg[_g, 0]
        _a2.plot(_x, np.exp(-((_x - _m) / _s) ** 2 / 2) / (_s * np.sqrt(2 * np.pi)),
                 color=PALETTE[_i], label=f"génération {_g} (σ̂ = {_s:.2f})")
    _a2.axvspan(-4, -2, color="grey", alpha=0.1)
    _a2.axvspan(2, 4, color="grey", alpha=0.1)
    _a2.set(ylim=(0, 1.6), xlabel="x", ylabel="densité", title="Une chaîne (n = 100, remplacer)")
    _a2.legend(fontsize=8)
    _chemin = sauvegarder_figure(_fig, "gaussienne-smoke" if SMOKE else "gaussienne",
                                 mo.notebook_dir() / "figures")
    print(f"Figure : {_chemin[0]}")
    _fig
    return


@app.cell
def _():
    mo.md(r"""
    ## 2. Des mots qui disparaissent

    Même boucle sur du texte : TinyShakespeare découpé en mots (minuscules) et en signes de
    ponctuation, 90 % pour l'entraînement, 10 % gardés comme **texte réel de test**.

    ### 2.a Unigrammes : la loi des mots rares

    Le modèle le plus simple : la fréquence de chaque mot. Ajuster = compter ; échantillonner
    = tirer $M$ mots selon ces fréquences ($M$ = taille du corpus d'entraînement). Un mot vu
    $c$ fois sur $M$ est tiré à la génération suivante avec la probabilité
    $1 - (1 - c/M)^M \approx 1 - e^{-c}$. Un mot vu **une seule fois** (un *hapax*) disparaît
    donc avec une probabilité $e^{-1} \approx 37\,\%$ dès la première génération, et un mot
    disparu ne revient **jamais** : le modèle lui donne une probabilité nulle. C'est le cas
    « distribution discrète » de Shumailov et al. : une chaîne de Markov dont les seuls états
    absorbants sont les distributions réduites à un seul mot.

    ### 2.b Bigrammes : la boucle d'un vrai petit modèle de langue

    Un modèle de bigrammes prédit le mot suivant d’après le mot courant (au chapitre 20, le bigramme de caractères sert de témoin au mini-GPT ; ici, ce sont des mots).
    Échantillonner = générer du texte, 2 000 flux en parallèle ; ajuster = compter les paires.
    On mesure à chaque génération :

    - le **vocabulaire** (mots de probabilité non nulle) et le nombre de **paires distinctes** ;
    - l'**entropie** de la distribution des mots générés (en bits : la diversité) ;
    - l'**entropie croisée sur le texte réel de test** (bits par mot, plus bas = mieux),
      avec un modèle lissé $\lambda\,P_{\text{bigramme}} + (1-\lambda)\,P_{\text{unigramme}}$
      ($\lambda = 0{,}7$, unigramme lissé par add-one sur le vocabulaire réel complet) pour
      qu'elle reste finie quand un mot du test a disparu.

    Deux régimes de plus : la **température** $T = 0{,}7$ à l'échantillonnage (les
    probabilités sont élevées à la puissance $1/T$, puis renormalisées : on favorise les
    suites fréquentes, comme le font les réglages de décodage des LLM, chapitre 28), et
    les régimes « 10 % de réel » et « accumuler » de la partie 1.
    """)
    return


@app.function
def decouper_mots(texte):
    """Mots en minuscules (apostrophe interne gardée) et ponctuation forte comme jetons."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[.,;:!?]", texte.lower())


@app.function
def compter_paires(flux, V):
    """Comptes des bigrammes d'un corpus, lu comme un anneau (le dernier jeton est suivi du
    premier) : tout jeton présent a ainsi au moins un successeur.
    Renvoie (cles, comptes) triés, cle = a * V + b."""
    flux = np.asarray(flux, dtype=np.int64)
    cles = flux * V + np.roll(flux, -1)
    return np.unique(cles, return_counts=True)


@app.function
def fusionner_comptes(cles_a, cnt_a, cles_b, cnt_b):
    """Somme de deux tables de comptes (régime « accumuler »)."""
    cles, inv = np.unique(np.concatenate([cles_a, cles_b]), return_inverse=True)
    return cles, np.bincount(inv, weights=np.concatenate([cnt_a, cnt_b])).astype(np.int64)


@app.function
def table_echantillonnage(cles, comptes, V, temperature=1.0):
    """Prépare l'échantillonnage du mot suivant pour tous les mots à la fois.

    Les paires sont triées par mot courant a : chaque ligne a est un segment contigu. On y
    stocke a + (probabilité cumulée dans la ligne), une suite croissante sur tout le tableau :
    tirer le successeur de a revient à chercher a + u (u uniforme dans [0, 1)) avec
    np.searchsorted, pour 2 000 flux d'un coup."""
    a, b = cles // V, cles % V
    p = comptes.astype(float) ** (1.0 / temperature)
    debuts = np.flatnonzero(np.r_[True, a[1:] != a[:-1]])
    p /= np.repeat(np.add.reduceat(p, debuts), np.diff(np.r_[debuts, len(a)]))
    cum = np.cumsum(p)
    cum -= np.repeat(np.r_[0.0, cum[debuts[1:] - 1]], np.diff(np.r_[debuts, len(a)]))
    fins = np.r_[debuts[1:] - 1, len(a) - 1]
    cum[fins] = 1.0                      # dernier élément de chaque ligne : exactement 1
    return a + cum, b


@app.function
def generer(rng, table, departs, longueur):
    """Génère len(departs) flux de `longueur` jetons avec le modèle de bigrammes `table`."""
    cumul, succ = table
    flux = np.empty((len(departs), longueur), dtype=np.int64)
    flux[:, 0] = departs
    for t in range(1, longueur):
        i = np.searchsorted(cumul, flux[:, t - 1] + rng.random(len(departs)), side="right")
        flux[:, t] = succ[i]
    return flux.ravel()


@app.function
def entropie_croisee_test(cles, comptes, V, test, unigramme_lisse, lam):
    """Bits par mot du texte réel de test sous λ·P_bigramme + (1 − λ)·P_unigramme lissé."""
    test = np.asarray(test, dtype=np.int64)
    a, b = test[:-1], test[1:]
    total_a = np.bincount(cles // V, weights=comptes, minlength=V)
    q = a * V + b
    pos = np.clip(np.searchsorted(cles, q), 0, len(cles) - 1)
    c_ab = np.where(cles[pos] == q, comptes[pos], 0)
    p_bi = np.divide(c_ab, total_a[a], out=np.zeros(len(a)), where=total_a[a] > 0)
    p = lam * p_bi + (1 - lam) * unigramme_lisse[b]
    return float(-np.mean(np.log2(p)))


@app.function
def entropie_bits(comptes):
    p = comptes[comptes > 0] / comptes.sum()
    return float(-(p * np.log2(p)).sum())


@app.cell
def _():
    from ia_commun.donnees import charger_tinyshakespeare

    _jetons = decouper_mots(charger_tinyshakespeare())
    vocab = sorted(set(_jetons))
    _index = {m: i for i, m in enumerate(vocab)}
    _ids = np.array([_index[m] for m in _jetons], dtype=np.int64)
    _coupe = int(0.9 * len(_ids))
    train, test = _ids[:_coupe], _ids[_coupe:]
    V = len(vocab)
    # Unigramme lissé (add-one) sur le vocabulaire réel complet : ne dépend que du réel, sert de
    # filet de sécurité commun à toutes les générations dans l'entropie croisée de test.
    unigramme_lisse = (np.bincount(train, minlength=V) + 1.0) / (len(train) + V)
    _f = lambda v: f"{v:,}".replace(",", "\u202f")
    print(f"{_f(len(_ids))} jetons, vocabulaire {_f(V)} ; entraînement {_f(len(train))}, test {_f(len(test))}")
    return V, test, train, unigramme_lisse, vocab


@app.cell
def _(config, train, V):
    # --- 2.a Unigrammes : tirage multinomial de M mots, génération après génération ---
    _rng = np.random.default_rng(43)
    _M = len(train)
    _c = np.bincount(train, minlength=V)
    hapax0 = _c == 1
    _t0 = time.perf_counter()
    vocab_uni, entropie_uni = [int((_c > 0).sum())], [entropie_bits(_c)]
    comptes_uni = [_c]
    for _g in range(1, config["generations_unigramme"] + 1):
        _c = _rng.multinomial(_M, _c / _M)
        vocab_uni.append(int((_c > 0).sum()))
        entropie_uni.append(entropie_bits(_c))
        if _g <= 10:
            comptes_uni.append(_c)
    duree_uni = time.perf_counter() - _t0
    # Part des hapax perdus dès la génération 1, et selon le compte initial c (prévu : e^{-c})
    _c0, _c1 = comptes_uni[0], comptes_uni[1]
    perte_par_compte = {c: float(np.mean(_c1[_c0 == c] == 0)) for c in (1, 2, 3, 5)}
    print(f"Unigrammes, {config['generations_unigramme']} générations en {duree_uni:.1f} s")
    print(f"Hapax : {hapax0.sum()} ; perdus à la génération 1 : {perte_par_compte[1]:.1%} (prévu e^-1 = 36,8 %)")
    for _cc, _pp in perte_par_compte.items():
        print(f"  mot vu {_cc} fois : perdu avec la proportion {_pp:.1%} (prévu {np.exp(-_cc):.1%})")
    _gs = [g for g in (0, 1, 10, 100, 300) if g < len(vocab_uni)]
    print("Vocabulaire : " + ", ".join(f"g{g} = {vocab_uni[g]}" for g in _gs))
    print("Entropie (bits) : " + ", ".join(f"g{g} = {entropie_uni[g]:.2f}" for g in _gs))
    return duree_uni, entropie_uni, perte_par_compte, vocab_uni


@app.cell
def _(V, config, test, train, unigramme_lisse):
    # --- 2.b Bigrammes : quatre régimes ---
    _M, _B = len(train), config["n_flux"]
    _L = _M // _B
    _G = config["generations_ngram"]
    _regimes = {"T=1": (1.0, "remplacer"), "T=0,7": (0.7, "remplacer"),
                "reel": (1.0, "reel"), "accumuler": (1.0, "accumuler")}
    suivi = {}
    corpus_exemples = {}
    _t0 = time.perf_counter()
    for _nom, (_temp, _mode) in _regimes.items():
        _rng = np.random.default_rng(44)
        _cles, _cnt = compter_paires(train, V)
        _corpus = train
        _s = {k: [] for k in ("vocab", "paires", "entropie", "xent_test", "frequences")}
        for _g in range(_G + 1):
            # Mesures du modèle de la génération g
            _uni = np.bincount(_cles // V, weights=_cnt, minlength=V)
            _s["vocab"].append(int((_uni > 0).sum()))
            _s["paires"].append(int(len(_cles)))
            _s["entropie"].append(entropie_bits(np.bincount(_corpus, minlength=V)))
            _s["xent_test"].append(entropie_croisee_test(_cles, _cnt, V, test, unigramme_lisse,
                                                          config["lambda_interp"]))
            if _g in config["gens_rang"]:
                _s["frequences"].append(np.sort(np.bincount(_corpus, minlength=V))[::-1])
            if _g == _G:
                break
            # Échantillonner le corpus suivant, puis réajuster selon le régime
            _table = table_echantillonnage(_cles, _cnt, V, _temp)
            _n_reel = int(config["part_reel"] * _M) if _mode == "reel" else 0
            _b = (_M - _n_reel) // _L
            _departs = _corpus[_rng.integers(0, len(_corpus), _b)]
            _corpus = generer(_rng, _table, _departs, _L)
            if _n_reel:
                _o = _rng.integers(0, _M - _n_reel)
                _corpus = np.concatenate([_corpus, train[_o:_o + _n_reel]])
            _c2, _n2 = compter_paires(_corpus, V)
            if _mode == "accumuler":
                _cles, _cnt = fusionner_comptes(_cles, _cnt, _c2, _n2)
            else:
                _cles, _cnt = _c2, _n2
            if _g == 0 or _g == _G - 1:
                corpus_exemples[(_nom, _g + 1)] = _corpus[:60]
        suivi[_nom] = _s
    duree_bi = time.perf_counter() - _t0
    print(f"Bigrammes, 4 régimes × {_G} générations en {duree_bi:.1f} s")
    print(f"{'régime':>10} | vocabulaire g0 → g{_G} | paires g0 → g{_G} | entropie (bits) | entropie croisée test")
    for _nom, _s in suivi.items():
        print(f"{_nom:>10} | {_s['vocab'][0]:6d} → {_s['vocab'][-1]:6d} ({_s['vocab'][-1] / _s['vocab'][0]:.0%})"
              f" | {_s['paires'][0]:6d} → {_s['paires'][-1]:6d} | {_s['entropie'][0]:.2f} → {_s['entropie'][-1]:.2f}"
              f" | {_s['xent_test'][0]:.2f} → {_s['xent_test'][-1]:.2f}")
    return corpus_exemples, duree_bi, suivi


@app.cell
def _(config, corpus_exemples, vocab):
    # Un extrait de texte produit par le modèle de la génération 1 et par celui de la dernière
    _G = config["generations_ngram"]
    for (_nom, _g), _ids in sorted(corpus_exemples.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if _nom in ("T=1", "T=0,7"):
            print(f"[{_nom}, génération {_g}] " + " ".join(vocab[i] for i in _ids[:40]))
    return


@app.cell
def _(SMOKE, config, suivi, vocab_uni):
    # Figure 2 : vocabulaire et entropie croisée de test au fil des générations,
    # puis courbe rang-fréquence (loi de Zipf) qui se fait couper la queue.
    _fig, (_a1, _a2, _a3) = plt.subplots(1, 3, figsize=(15, 4.3))
    _noms = {"T=1": "remplacer, T = 1", "T=0,7": "remplacer, T = 0,7",
             "reel": "10 % de réel", "accumuler": "accumuler"}
    for _i, (_k, _lib) in enumerate(_noms.items()):
        _s = suivi[_k]
        _a1.plot(np.array(_s["vocab"]) / _s["vocab"][0], color=PALETTE[_i], lw=1.8, label=_lib)
        _a2.plot(_s["xent_test"], color=PALETTE[_i], lw=1.8, label=_lib)
    _a1.plot(np.array(vocab_uni[: config["generations_ngram"] + 1]) / vocab_uni[0], color="grey",
             ls=":", label="unigrammes, remplacer")
    _a1.set(xlabel="génération", ylabel="vocabulaire restant (part du départ)", ylim=(0, 1.05),
            title="Les mots rares disparaissent")
    _a1.legend(fontsize=8)
    _a2.set(xlabel="génération", ylabel="bits par mot (texte réel de test)",
            title="Entropie croisée sur du vrai texte")
    for _i, _g in enumerate(config["gens_rang"][: len(suivi["T=1"]["frequences"])]):
        _f = suivi["T=1"]["frequences"][_i]
        _f = _f[_f > 0]
        _a3.loglog(np.arange(1, len(_f) + 1), _f, color=PALETTE[_i], lw=1.6, label=f"génération {_g}")
    _a3.set(xlabel="rang du mot", ylabel="nombre d'occurrences", title="Rang-fréquence (T = 1, remplacer)")
    _a3.legend(fontsize=8)
    _chemin = sauvegarder_figure(_fig, "ngrammes-smoke" if SMOKE else "ngrammes",
                                 mo.notebook_dir() / "figures")
    print(f"Figure : {_chemin[0]}")
    _fig
    return


@app.cell
def _(
    SMOKE,
    bilan_gauss,
    config,
    duree_bi,
    duree_gauss,
    duree_uni,
    entropie_uni,
    gauss,
    perte_par_compte,
    suivi,
    theorie,
    vocab_uni,
):
    # Résultats chiffrés pour le README et les figures du site (site/scripts/figures/ch37_*.py).
    # Pour la gaussienne, on sous-échantillonne les générations (une sur cinq) pour alléger le JSON.
    _pas = 5
    _g = {k: dict(mediane=np.median(s, 1)[::_pas].tolist(), p10=np.percentile(s, 10, 1)[::_pas].tolist(),
                  p90=np.percentile(s, 90, 1)[::_pas].tolist()) for k, (_m, s) in gauss.items()}
    _mu, _sg = gauss["remplacer-100"]
    resume = dict(
        config=config,
        gauss=dict(pas=_pas, sigma=_g, bilan=bilan_gauss, theorie=theorie,
                   chaine0=dict(mu=_mu[:, 0].tolist(), sigma=_sg[:, 0].tolist())),
        unigrammes=dict(vocab=vocab_uni, entropie=entropie_uni,
                        perte_par_compte={str(k): v for k, v in perte_par_compte.items()}),
        bigrammes={k: {kk: (vv if kk != "frequences" else [f[f > 0].tolist() for f in vv])
                       for kk, vv in s.items()} for k, s in suivi.items()},
        durees=dict(gauss=duree_gauss, unigrammes=duree_uni, bigrammes=duree_bi),
    )
    _chemin = mo.notebook_dir() / "figures" / ("resultats-smoke.json" if SMOKE else "resultats.json")
    _chemin.write_text(json.dumps(resume, ensure_ascii=False, separators=(",", ":"),
                                  default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"Résultats : {_chemin}")
    return


@app.cell
def _(SMOKE, bilan_gauss, perte_par_compte, suivi, vocab_uni):
    _b, _s1, _s7 = bilan_gauss, suivi["T=1"], suivi["T=0,7"]
    _sr, _sa = suivi["reel"], suivi["accumuler"]
    verifs = {
        "Gaussienne, remplacer, n = 100 : écart-type médian < 0,01 après 1 000 générations":
            _b["remplacer-100"]["sigma_median"] < 0.01,
        "Gaussienne, estimateur sans biais : s'effondre aussi (écart-type médian < 0,05)":
            _b["sans-biais"]["sigma_median"] < 0.05,
        "Gaussienne, remplacer, n = 100 : masse médiane de la queue |x| > 2 < 1e-6 (vraie : 0,0455)":
            _b["remplacer-100"]["queue2_mediane"] < 1e-6,
        "Gaussienne, plus n est grand, plus c'est lent : σ médian n = 10 < n = 100 < n = 1 000":
            _b["remplacer-10"]["sigma_median"] < _b["remplacer-100"]["sigma_median"]
            < _b["remplacer-1000"]["sigma_median"],
        "Gaussienne, accumuler : écart-type médian entre 0,9 et 1,1, 10e centile au-dessus de 0,8":
            0.9 <= _b["accumuler"]["sigma_median"] <= 1.1 and _b["accumuler"]["sigma_p10"] > 0.8,
        "Gaussienne, 10 % de réel : écart-type médian ≥ 0,5":
            _b["reel"]["sigma_median"] >= 0.5,
        "Unigrammes : hapax perdus à la génération 1 entre 33 % et 41 % (prévu e^-1 = 36,8 %)":
            0.33 <= perte_par_compte[1] <= 0.41,
        "Unigrammes : le vocabulaire ne remonte jamais": bool(np.all(np.diff(vocab_uni) <= 0)),
        "Bigrammes T = 1 : vocabulaire final ≤ 75 % du départ, sans jamais remonter":
            _s1["vocab"][-1] <= 0.75 * _s1["vocab"][0] and bool(np.all(np.diff(_s1["vocab"]) <= 0)),
        "Bigrammes T = 0,7 : vocabulaire final ≤ 25 % du départ (la température accélère)":
            _s7["vocab"][-1] <= 0.25 * _s7["vocab"][0],
        "Bigrammes T = 1 : entropie croisée sur le texte réel de test en hausse d'au moins 0,5 bit":
            _s1["xent_test"][-1] >= _s1["xent_test"][0] + 0.5,
        "Bigrammes, accumuler : entropie croisée de test à moins de 0,3 bit du départ (vocabulaire intact : garanti par construction, contrôle de cohérence)":
            _sa["xent_test"][-1] <= _sa["xent_test"][0] + 0.3 and _sa["vocab"][-1] == _sa["vocab"][0],
        "Bigrammes, 10 % de réel : garde plus de vocabulaire que T = 1":
            _sr["vocab"][-1] > _s1["vocab"][-1],
    }
    for _k, _ok in verifs.items():
        print(("OK    " if _ok else ("--    " if SMOKE else "ÉCHEC ")) + _k
              + ("" if _ok or not SMOKE else " (non vérifié en smoke)"))
    return (verifs,)


@app.cell
def _(SMOKE, verifs):
    _echecs = [k for k, ok in verifs.items() if not ok]
    if not SMOKE and _echecs:
        raise SystemExit("Seuils non atteints : " + " ; ".join(_echecs))
    if SMOKE:
        _msg = (f"**Bilan (smoke)** : le code tourne de bout en bout ; seuils non vérifiés "
                f"({sum(verifs.values())} sur {len(verifs)} déjà atteints).")
    else:
        _msg = f"**Bilan** : {sum(verifs.values())} vérifications sur {len(verifs)} passent."
    mo.md(_msg)
    return


if __name__ == "__main__":
    app.run()
