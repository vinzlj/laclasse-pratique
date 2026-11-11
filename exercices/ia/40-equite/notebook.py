# Exercice 40 : mesurer et corriger l'inéquité d'un classifieur, puis protéger les données
# (chapitre 40).
#
# Données réelles : recensement américain (ACS PUMS 2018, Californie), tâche « ACSIncome » de
# Folktables (Ding et al., 2021) : prédire si le revenu annuel dépasse 50 000 $.
# Partie 1 en numpy : les taux par groupe (sélection, vrais positifs, faux positifs, précision).
# Partie 2 avec scikit-learn : régression logistique avec et sans l'attribut sensible, proxys.
# Partie 3 en numpy : post-traitement par seuils propres à chaque groupe (Hardt et al., 2016).
# Partie 4 : l'incompatibilité des critères, vérifiée sur les chiffres (Chouldechova, 2017).
# Partie 5 en numpy : confidentialité différentielle (mécanisme de Laplace, descente de
# gradient différentiellement privée, comptabilité zCDP de Bun et Steinke, 2016).
#
#   make ex NUM=40              → éditeur marimo (interactif)
#   make ex NUM=40 MODE=run     → script complet (vérifie les seuils du README)
#   make ex NUM=40 MODE=smoke   → test rapide (< 30 s)
#
# Les figures du site (site/scripts/figures/ch40_*.py) lisent figures/resultats.json,
# écrit par le run complet.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import fcntl
    import json
    import os
    import time
    import urllib.request
    import zipfile

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from ia_commun import (
        PALETTE,
        data_dir,
        is_smoke,
        sauvegarder_figure,
        seed_everything,
        style_commun,
    )


@app.cell
def _():
    mo.md(r"""
    # Exercice 40 : équité d'un classifieur et confidentialité des données

    Un modèle peut être exact **en moyenne** et traiter deux groupes très différemment. On
    le mesure ici sur de vraies données : le recensement américain de 2018 (enquête ACS,
    Californie, 195 665 personnes de plus de 16 ans qui travaillent), dans la version
    « ACSIncome » de **Folktables** (Ding, Hardt, Miller et Schmidt, 2021), écrite pour
    remplacer le vieux jeu « Adult » de 1994. La tâche : prédire si le revenu annuel
    dépasse 50 000 $. L'attribut sensible : le sexe déclaré (variable `SEX` de l'enquête,
    binaire en 2018 : c'est une limite des données, pas un choix de l'exercice).

    **Ce qui est optimisé.** Le modèle est une régression logistique : il minimise
    l'**entropie croisée binaire** (chapitre 01). Aucune contrainte d'équité n'entre dans
    l'entraînement : l'équité est corrigée **après coup**, en choisissant un seuil de
    décision par groupe (post-traitement de Hardt, Price et Srebro, 2016). En partie 5,
    la même entropie croisée est minimisée par une descente de gradient **bruitée** pour
    protéger chaque personne du jeu d'entraînement.

    Plan :

    1. **Mesurer (numpy).** Matrice de confusion par groupe, puis quatre taux : sélection,
       vrais positifs, faux positifs, précision (valeur prédictive positive).
    2. **Entraîner (scikit-learn).** Avec l'attribut sensible, puis sans (« l'équité par
       ignorance ») ; mesurer à quel point les autres variables le trahissent (proxys).
    3. **Corriger (numpy).** Un seuil par groupe pour égaliser soit le taux de sélection
       (parité démographique), soit le taux de vrais positifs (égalité des chances).
    4. **Constater l'incompatibilité.** Quand les taux de base diffèrent, aucune politique
       n'égalise tout à la fois ; l'identité de Chouldechova le vérifie au chiffre près.
    5. **Protéger (numpy).** Mécanisme de Laplace sur des comptages, puis régression
       logistique entraînée avec écrêtage et bruit gaussien (le principe de DP-SGD), avec
       un budget $\varepsilon$ calculé exactement.
    """)
    return


@app.cell
def _():
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        # smoke : sous-échantillon de 20 000 personnes (mêmes étapes, mêmes fonctions)
        n_max=20_000 if SMOKE else None,
        # grille des cibles pour le post-traitement (taux de sélection ou de vrais positifs)
        n_cibles=41 if SMOKE else 201,
        # descente de gradient privée
        dp_pas=60 if SMOKE else 200,
        dp_lr=1.0,
        dp_inertie=0.9,  # momentum : simple post-traitement des gradients bruités
        dp_ecretage=1.0,
        dp_delta=1e-6,
        dp_epsilons=[0.1, 1.0, 10.0] if SMOKE else [0.05, 0.1, 0.3, 1.0, 3.0, 10.0],
        dp_n_petit=2_000,
        dp_repetitions=2 if SMOKE else 5,
        laplace_tirages=2_000 if SMOKE else 20_000,
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SEED, SMOKE, config


@app.cell
def _():
    mo.md(r"""
    ## Les données

    Le fichier « personnes » de l'enquête ACS 2018 sur un an pour la Californie (67 Mo
    compressé, téléchargé une fois dans `data/folktables/`). On applique exactement la
    définition de la tâche ACSIncome du paquet Folktables (licence MIT) : dix variables
    (âge, catégorie d'employeur, diplôme, situation matrimoniale, profession, lieu de
    naissance, lien avec la personne de référence du ménage, heures travaillées par
    semaine, sexe, « race » au sens du recensement américain), et les filtres hérités
    d'Adult : plus de 16 ans, revenu supérieur à 100 $, au moins une heure travaillée par
    semaine, poids de sondage positif.
    """)
    return


@app.function
def telecharger_acs_californie_2018():
    """Télécharge (une fois) le fichier personnes ACS PUMS 2018 1 an de Californie.

    Source : https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_pca.zip
    (Census Bureau ; conditions : https://www.census.gov/data/developers/about/terms-of-service.html).
    Verrou de fichier pour que deux processus ne téléchargent pas en même temps.
    """
    url = "https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_pca.zip"
    dossier = data_dir("folktables")
    chemin = dossier / "csv_pca.zip"
    if chemin.exists():
        return chemin
    with open(dossier / ".csv_pca.lock", "w") as verrou:
        fcntl.flock(verrou, fcntl.LOCK_EX)
        if not chemin.exists():  # un autre processus a pu finir pendant l'attente
            print(f"Téléchargement de {url} (67 Mo) → {chemin}")
            requete = urllib.request.Request(url, headers={"User-Agent": "ia-experiments"})
            tmp = chemin.with_name(f"csv_pca.zip.{os.getpid()}.part")
            with urllib.request.urlopen(requete, timeout=300) as reponse, open(tmp, "wb") as f:
                while bloc := reponse.read(1 << 20):
                    f.write(bloc)
            tmp.replace(chemin)  # jamais de fichier à moitié écrit dans le cache
        fcntl.flock(verrou, fcntl.LOCK_UN)
    return chemin


@app.function
def charger_acs_income():
    """Tâche ACSIncome de Folktables (Ding et al., 2021), Californie 2018.

    Renvoie un DataFrame avec les 10 variables explicatives, la cible y (revenu > 50 000 $)
    et le revenu brut. Cache : data/folktables/acsincome_ca_2018.parquet.
    """
    cache = data_dir("folktables") / "acsincome_ca_2018.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    zip_ = telecharger_acs_californie_2018()
    colonnes = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P",
                "PINCP", "PWGTP"]
    with zipfile.ZipFile(zip_) as z, z.open("psam_p06.csv") as f:
        df = pd.read_csv(f, usecols=colonnes)
    # Filtres de Folktables, repris d'Adult : AGEP > 16, PINCP > 100, WKHP > 0, PWGTP >= 1
    df = df[(df.AGEP > 16) & (df.PINCP > 100) & (df.WKHP > 0) & (df.PWGTP >= 1)]
    df = df.drop(columns="PWGTP").astype({c: "int64" for c in colonnes if c not in ("PINCP", "PWGTP")})
    df = df.reset_index(drop=True)
    df["y"] = (df.PINCP > 50_000).astype("int8")
    tmp = cache.with_name(f"{cache.name}.{os.getpid()}.part")
    df.to_parquet(tmp)
    tmp.replace(cache)
    return df


@app.cell
def _(SEED, config):
    _t0 = time.perf_counter()
    df_complet = charger_acs_income()
    df = df_complet if config["n_max"] is None else df_complet.sample(config["n_max"], random_state=SEED)
    df = df.reset_index(drop=True)
    y = df["y"].to_numpy().astype(int)
    femme = (df["SEX"] == 2).to_numpy().astype(int)  # 1 = femme, 0 = homme (codes 2 et 1 de l'ACS)
    # Découpage 60 / 20 / 20, stratifié sur (cible, groupe) : mêmes proportions partout
    _strate = y * 2 + femme
    i_app, _reste = train_test_split(np.arange(len(df)), test_size=0.4, random_state=SEED, stratify=_strate)
    i_val, i_test = train_test_split(_reste, test_size=0.5, random_state=SEED, stratify=_strate[_reste])
    duree_donnees = time.perf_counter() - _t0
    print(f"{len(df):,} personnes ({len(i_app):,} apprentissage, {len(i_val):,} validation, "
          f"{len(i_test):,} test), chargées en {duree_donnees:.1f} s")
    print(f"Revenu > 50 000 $ : {y.mean():.1%} au total ; hommes {y[femme == 0].mean():.1%}, "
          f"femmes {y[femme == 1].mean():.1%}")
    return df, df_complet, duree_donnees, femme, i_app, i_test, i_val, y


@app.cell
def _():
    mo.md(r"""
    ## Partie 1 : mesurer, en numpy

    Pour chaque groupe $a$ (hommes, femmes), on compte les vrais positifs (VP), faux
    positifs (FP), faux négatifs (FN) et vrais négatifs (VN) de la décision $\hat Y$, puis :

    | Taux | Formule | Critère d'équité qui l'égalise entre groupes |
    |---|---|---|
    | sélection | $P(\hat Y = 1 \mid A = a) = \frac{VP + FP}{N_a}$ | **parité démographique** |
    | vrais positifs (rappel) | $P(\hat Y = 1 \mid Y = 1, A = a) = \frac{VP}{VP + FN}$ | **égalité des chances** |
    | faux positifs | $P(\hat Y = 1 \mid Y = 0, A = a) = \frac{FP}{FP + VN}$ | avec le précédent : **chances égalisées** |
    | précision (VPP) | $P(Y = 1 \mid \hat Y = 1, A = a) = \frac{VP}{VP + FP}$ | **parité prédictive** (proche de la calibration) |

    Le **taux de base** $p_a = P(Y = 1 \mid A = a)$ ne dépend pas du modèle : c'est la
    réalité mesurée par l'enquête, avec ses propres inégalités.
    """)
    return


@app.function
def taux_par_groupe(y, decision, groupe):
    """Matrice de confusion et taux usuels pour chaque valeur de `groupe` (numpy pur)."""
    res = {}
    for g in (0, 1):
        m = groupe == g
        t, d = y[m], decision[m]
        vp = int(((t == 1) & (d == 1)).sum())
        fp = int(((t == 0) & (d == 1)).sum())
        fn = int(((t == 1) & (d == 0)).sum())
        vn = int(((t == 0) & (d == 0)).sum())
        res[g] = dict(
            vp=vp, fp=fp, fn=fn, vn=vn, n=int(m.sum()),
            base=(vp + fn) / m.sum(),
            selection=(vp + fp) / m.sum(),
            tvp=vp / (vp + fn),
            tfp=fp / (fp + vn),
            vpp=vp / (vp + fp),
            exactitude=(vp + vn) / m.sum(),
        )
    return res


@app.function
def ecarts(taux):
    """Écarts femmes − hommes, en points de pourcentage (0,01 = 1 point)."""
    return {k: taux[1][k] - taux[0][k] for k in ("selection", "tvp", "tfp", "vpp", "exactitude")}


@app.function
def afficher(nom, y, decision, groupe):
    t = taux_par_groupe(y, decision, groupe)
    e = ecarts(t)
    exact = float((y == decision).mean())
    print(f"{nom:<38} exactitude {exact:.1%} | "
          + " | ".join(f"{k} H {t[0][k]:.3f} F {t[1][k]:.3f} (écart {e[k]:+.3f})"
                       for k in ("selection", "tvp", "tfp", "vpp")))
    return dict(exactitude=exact, taux=t, ecarts=e)


@app.cell
def _():
    # Le bouton empêche les calculs de se relancer à chaque modification d'une cellule amont.
    bouton = mo.ui.run_button(label="Lancer les calculs")
    bouton
    return (bouton,)


@app.cell
def _():
    mo.md(r"""
    ## Partie 2 : entraîner, avec scikit-learn

    Variables numériques (âge, diplôme, heures) centrées-réduites, variables catégorielles
    (catégorie d'employeur, situation matrimoniale, profession, lieu de naissance, lien
    dans le ménage, « race ») en indicatrices (*one-hot*), les modalités de moins de
    20 personnes regroupées. Deux modèles :

    - **avec** l'attribut sensible `SEX` parmi les variables ;
    - **sans** (« l'équité par ignorance ») : le modèle ne voit jamais le sexe.

    Puis un troisième modèle prédit le **sexe** à partir des variables du modèle « sans » :
    s'il y arrive bien, retirer la colonne ne retire pas l'information (variables
    **proxys** : profession, heures travaillées…).
    """)
    return


@app.cell
def _(bouton, df, femme, i_app, i_test, i_val, y):
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer les calculs ».*"))
    NUMERIQUES = ["AGEP", "SCHL", "WKHP"]
    CATEGORIELLES = ["COW", "MAR", "OCCP", "POBP", "RELP", "RAC1P"]

    def preparer(avec_sexe):
        cat = CATEGORIELLES + (["SEX"] if avec_sexe else [])
        return ColumnTransformer([
            ("num", StandardScaler(), NUMERIQUES),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), cat),
        ])

    _t0 = time.perf_counter()
    modeles, scores = {}, {}
    for _nom, _avec in (("avec", True), ("sans", False)):
        _prep = preparer(_avec)
        _Xa = _prep.fit_transform(df.iloc[i_app])
        _clf = LogisticRegression(max_iter=3000).fit(_Xa, y[i_app])
        modeles[_nom] = (_prep, _clf)
        scores[_nom] = {
            "val": _clf.predict_proba(_prep.transform(df.iloc[i_val]))[:, 1],
            "test": _clf.predict_proba(_prep.transform(df.iloc[i_test]))[:, 1],
        }
        print(f"Modèle « {_nom} » : {_Xa.shape[1]} colonnes après encodage")

    # Proxy : prédire le sexe sans la colonne SEX
    _prep = preparer(False)
    _clf = LogisticRegression(max_iter=3000).fit(_prep.fit_transform(df.iloc[i_app]), femme[i_app])
    _p = _clf.predict_proba(_prep.transform(df.iloc[i_test]))[:, 1]
    proxy = dict(auc=float(roc_auc_score(femme[i_test], _p)),
                 exactitude=float(((_p > 0.5) == femme[i_test]).mean()))
    duree_modeles = time.perf_counter() - _t0
    print(f"Prédire le sexe sans la colonne SEX : AUC {proxy['auc']:.3f}, "
          f"exactitude {proxy['exactitude']:.1%} (hasard : 50 %, majorité : {1 - femme.mean():.1%})")
    print(f"Entraînements : {duree_modeles:.1f} s")
    return duree_modeles, proxy, scores


@app.cell
def _(femme, i_test, scores, y):
    # Seuil unique 0,5 pour tout le monde : la décision « naturelle » d'un classifieur
    yt, gt = y[i_test], femme[i_test]
    res_base = {
        "avec": afficher("avec SEX, seuil 0,5", yt, (scores["avec"]["test"] > 0.5).astype(int), gt),
        "sans": afficher("sans SEX, seuil 0,5", yt, (scores["sans"]["test"] > 0.5).astype(int), gt),
    }
    return gt, res_base, yt


@app.cell
def _():
    mo.md(r"""
    ## Partie 3 : corriger après coup, un seuil par groupe

    Le post-traitement de Hardt, Price et Srebro (2016) ne réentraîne rien : il prend les
    scores $s(x) \in [0, 1]$ du modèle et choisit un **seuil par groupe** $\tau_a$
    (décision $\hat Y = 1$ si $s(x) > \tau_a$).

    - **Parité démographique** : pour une cible $c$, $\tau_a$ est le quantile $1 - c$ des
      scores du groupe $a$ (une proportion $c$ de chaque groupe est sélectionnée).
    - **Égalité des chances** : pour une cible $c$, $\tau_a$ est le quantile $1 - c$ des
      scores des **positifs** du groupe $a$ (une proportion $c$ des positifs de chaque
      groupe est sélectionnée).

    La cible $c$ est choisie sur l'ensemble de **validation** (celle qui y donne la
    meilleure exactitude), puis tout est mesuré sur le **test**. L'article original
    autorise aussi un tirage au sort entre deux seuils pour égaliser exactement deux taux
    à la fois (chances égalisées) ; on s'en tient ici aux seuils déterministes.
    """)
    return


@app.function
def seuils_parite(score, groupe, cible):
    """Un seuil par groupe tel que la proportion sélectionnée vaille `cible` dans chaque groupe."""
    return {g: float(np.quantile(score[groupe == g], 1 - cible)) for g in (0, 1)}


@app.function
def seuils_egalite_chances(score, y, groupe, cible):
    """Un seuil par groupe tel que le taux de vrais positifs vaille `cible` dans chaque groupe."""
    return {g: float(np.quantile(score[(groupe == g) & (y == 1)], 1 - cible)) for g in (0, 1)}


@app.function
def decider(score, groupe, seuils):
    return np.where(groupe == 1, score > seuils[1], score > seuils[0]).astype(int)


@app.cell
def _(config, femme, gt, i_val, scores, y, yt):
    _sv, _yv, _gv = scores["avec"]["val"], y[i_val], femme[i_val]
    _st = scores["avec"]["test"]
    _cibles = np.linspace(0.05, 0.95, config["n_cibles"])
    politiques = {}
    for _nom, _fabrique in (
        ("parite", lambda c: seuils_parite(_sv, _gv, c)),
        ("egalite_chances", lambda c: seuils_egalite_chances(_sv, _yv, _gv, c)),
    ):
        # Exactitude de validation pour chaque cible ; on garde la meilleure
        _exact = [float((decider(_sv, _gv, _fabrique(c)) == _yv).mean()) for c in _cibles]
        _c = float(_cibles[int(np.argmax(_exact))])
        _seuils = _fabrique(_c)
        _r = afficher(f"{_nom} (cible {_c:.3f})", yt, decider(_st, gt, _seuils), gt)
        _r.update(cible=_c, seuils=_seuils)
        politiques[_nom] = _r
    print("Seuils : " + " ; ".join(f"{k} H {v['seuils'][0]:.3f} F {v['seuils'][1]:.3f}"
                                   for k, v in politiques.items()))
    return (politiques,)


@app.cell
def _():
    mo.md(r"""
    ## Partie 4 : les critères sont incompatibles

    Dans chaque groupe, quatre nombres sont liés par une identité comptable (Chouldechova,
    2017), où $p$ est le taux de base du groupe :

    $$
    \text{TFP} \;=\; \frac{p}{1 - p}\cdot\frac{1 - \text{VPP}}{\text{VPP}}\cdot \text{TVP}
    $$

    Si les taux de base diffèrent ($p_F \neq p_H$), on ne peut donc pas avoir à la fois
    même VPP, même TVP **et** même TFP dans les deux groupes (sauf classifieur parfait).
    La cellule vérifie l'identité sur chaque politique, puis montre qu'aucune n'égalise
    tous les taux.
    """)
    return


@app.cell
def _(politiques, res_base):
    _toutes = {"seuil unique (avec SEX)": res_base["avec"], "seuil unique (sans SEX)": res_base["sans"],
               "parité démographique": politiques["parite"],
               "égalité des chances": politiques["egalite_chances"]}
    erreur_identite = 0.0
    for _nom, _r in _toutes.items():
        for _g in (0, 1):
            _t = _r["taux"][_g]
            _p = _t["base"]
            _tfp = _p / (1 - _p) * (1 - _t["vpp"]) / _t["vpp"] * _t["tvp"]
            erreur_identite = max(erreur_identite, abs(_tfp - _t["tfp"]))
    print(f"Identité de Chouldechova : écart maximal {erreur_identite:.1e} (calcul exact, aux arrondis près)")
    print(f"{'politique':<26}{'exactitude':>11}{'Δ sélection':>13}{'Δ TVP':>9}{'Δ TFP':>9}{'Δ VPP':>9}")
    for _nom, _r in _toutes.items():
        _e = _r["ecarts"]
        print(f"{_nom:<26}{_r['exactitude']:>11.3f}{_e['selection']:>+13.3f}{_e['tvp']:>+9.3f}"
              f"{_e['tfp']:>+9.3f}{_e['vpp']:>+9.3f}")
    return (erreur_identite,)


@app.function
def courbe_roc(y, score, n_points=101):
    """Courbe ROC calculée à la main : (TFP, TVP) pour des seuils régulièrement espacés en quantiles."""
    seuils = np.quantile(score, np.linspace(0, 1, n_points))
    tfp = np.array([(score[y == 0] > s).mean() for s in seuils])
    tvp = np.array([(score[y == 1] > s).mean() for s in seuils])
    return tfp, tvp


@app.cell
def _(gt, scores, yt):
    _s = scores["avec"]["test"]
    roc = {g: courbe_roc(yt[gt == g], _s[gt == g]) for g in (0, 1)}
    return (roc,)


@app.cell
def _():
    mo.md(r"""
    ## Partie 5 : protéger les personnes du jeu de données

    **Confidentialité différentielle** (Dwork, McSherry, Nissim et Smith, 2006). Un
    algorithme $M$ est $(\varepsilon, \delta)$-différentiellement privé si, pour deux jeux
    de données $D$ et $D'$ qui ne diffèrent que par **une** personne, et tout ensemble de
    résultats $S$ : $P(M(D) \in S) \le e^{\varepsilon}\, P(M(D') \in S) + \delta$.
    Ce que l'algorithme publie change à peine qu'on y soit ou non.

    **5.a Mécanisme de Laplace.** Un comptage change d'au plus 1 quand on remplace une
    personne (sensibilité 1) ; publier le comptage plus un bruit de Laplace d'échelle
    $1/\varepsilon$ est $\varepsilon$-privé. Le même bruit est négligeable sur un grand
    groupe et écrase un petit groupe.

    **5.b Descente de gradient privée.** Le principe de DP-SGD (Abadi et al., 2016) :
    **écrêter** le gradient de chaque personne à une norme $C$, les **sommer**, ajouter un
    bruit gaussien $\mathcal{N}(0, \sigma^2 I)$, puis faire le pas. Ici le gradient est
    calculé sur tout le jeu à chaque pas (pas de sous-échantillonnage), ce qui permet une
    comptabilité exacte et simple : remplacer une personne change la somme d'au plus $2C$,
    chaque pas est donc $\rho$-zCDP avec $\rho = (2C)^2 / (2\sigma^2)$, les pas
    s'additionnent, et $\rho$-zCDP entraîne $(\rho + 2\sqrt{\rho \ln(1/\delta)}, \delta)$-DP
    (Bun et Steinke, 2016). On fixe $\varepsilon$ et $\delta$, on en déduit $\sigma$.

    En simplifiant : on met à l'échelle les variables numériques avec des bornes
    publiques (âge sur 99, diplôme sur 24, heures sur 99) et l'on suppose publique la
    liste des modalités (elle l'est : dictionnaire des données de l'enquête), pour que
    seule la descente de gradient touche aux données.
    """)
    return


@app.function
def bruit_pour_epsilon(epsilon, delta, n_pas, ecretage):
    """Écart type du bruit gaussien (sur la somme des gradients) pour un budget (ε, δ) donné.

    Comptabilité zCDP (Bun et Steinke, 2016) : chaque pas coûte ρ₁ = (2C)² / (2σ²)
    (sensibilité 2C quand on remplace une personne), n_pas pas coûtent n_pas · ρ₁, et
    ρ-zCDP ⇒ (ρ + 2 √(ρ ln(1/δ)), δ)-DP. On inverse cette dernière relation.
    """
    L = np.log(1 / delta)
    rho = (np.sqrt(L + epsilon) - np.sqrt(L)) ** 2  # ρ total autorisé
    return float(np.sqrt(n_pas * (2 * ecretage) ** 2 / (2 * rho)))


@app.function
def epsilon_pour_bruit(sigma, delta, n_pas, ecretage):
    rho = n_pas * (2 * ecretage) ** 2 / (2 * sigma**2)
    return float(rho + 2 * np.sqrt(rho * np.log(1 / delta)))


@app.function
def descente_privee(X, y, n_pas, lr, ecretage, sigma, rng, inertie=0.9):
    """Régression logistique par descente de gradient écrêtée et bruitée (numpy, X creux).

    Gradient de l'entropie croisée pour la personne i : (σ(w·xᵢ) − yᵢ) xᵢ. Sa norme vaut
    |σ(w·xᵢ) − yᵢ| · ‖xᵢ‖ : l'écrêtage par personne ne coûte qu'un produit.
    sigma = 0 et ecretage = inf donnent la descente de gradient ordinaire.
    L'inertie (momentum) ne lit que les gradients déjà bruités : c'est un post-traitement,
    qui ne coûte rien en confidentialité.
    """
    n, d = X.shape
    w = np.zeros(d)
    v = np.zeros(d)
    normes_x = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    for _ in range(n_pas):
        p = 1 / (1 + np.exp(-(X @ w)))
        r = p - y
        facteur = np.minimum(1.0, ecretage / np.maximum(np.abs(r) * normes_x, 1e-12))
        somme = X.T @ (r * facteur)  # somme des gradients écrêtés
        if sigma > 0:
            somme = somme + rng.normal(0.0, sigma, size=d)
        v = inertie * v + somme / n
        w -= lr * v
    return w


@app.cell
def _(config, df, df_complet, femme, i_app, i_test, y):
    _t0 = time.perf_counter()
    # --- 5.a Laplace sur des comptages : nombre de personnes à plus de 50 000 $ par « race »
    _rng = np.random.default_rng(42)
    _comptes = df_complet.groupby("RAC1P")["y"].agg(["sum", "size"])
    laplace = {}
    for _eps in (0.1, 1.0):
        for _code, (_vrai, _taille) in _comptes.iterrows():
            _bruits = _rng.laplace(0.0, 1.0 / _eps, size=config["laplace_tirages"])
            laplace[f"{_eps}-{_code}"] = dict(
                epsilon=_eps, code=int(_code), vrai=int(_vrai), taille=int(_taille),
                erreur_rel_moy=float(np.mean(np.abs(_bruits)) / max(_vrai, 1)),
            )
    for _l in laplace.values():
        if _l["code"] in (1, 4):
            print(f"Laplace ε = {_l['epsilon']} : RAC1P {_l['code']} ({_l['taille']} personnes, "
                  f"{_l['vrai']} à > 50 000 $) → erreur relative moyenne {_l['erreur_rel_moy']:.2%}")

    # --- 5.b Descente de gradient privée
    _num = np.stack([df.AGEP / 99, df.SCHL / 24, df.WKHP / 99], axis=1)  # bornes publiques
    _enc = OneHotEncoder(categories=[sorted(df_complet[c].unique()) for c in
                                     ["COW", "MAR", "OCCP", "POBP", "RELP", "RAC1P", "SEX"]],
                         handle_unknown="ignore")
    _cat = _enc.fit_transform(df[["COW", "MAR", "OCCP", "POBP", "RELP", "RAC1P", "SEX"]])
    _X = sparse.hstack([sparse.csr_matrix(_num), _cat, np.ones((len(df), 1))]).tocsr()
    _Xa, _ya, _Xt, _yt = _X[i_app], y[i_app].astype(float), _X[i_test], y[i_test]
    _gt = femme[i_test]
    _T, _lr, _C, _delta = config["dp_pas"], config["dp_lr"], config["dp_ecretage"], config["dp_delta"]
    _m = config["dp_inertie"]

    def _evaluer(w):
        _d = (_Xt @ w > 0).astype(int)
        _t = taux_par_groupe(_yt, _d, _gt)
        return float((_d == _yt).mean()), ecarts(_t)

    _w0 = descente_privee(_Xa, _ya, _T, _lr, np.inf, 0.0, None, _m)
    _exact0, _ec0 = _evaluer(_w0)
    print(f"Sans confidentialité ({_T} pas) : exactitude test {_exact0:.1%}")
    _rng_sous = np.random.default_rng(42)
    _petit = _rng_sous.choice(_Xa.shape[0], size=min(config["dp_n_petit"], _Xa.shape[0]), replace=False)
    _w0p = descente_privee(_Xa[_petit], _ya[_petit], _T, _lr, np.inf, 0.0, None, _m)
    _exact0p, _ = _evaluer(_w0p)
    print(f"Sans confidentialité, {len(_petit)} personnes : exactitude test {_exact0p:.1%}")

    dp = dict(sans_bruit=dict(complet=_exact0, petit=_exact0p, ecarts_complet=_ec0),
              n_complet=int(_Xa.shape[0]), n_petit=len(_petit), delta=_delta, pas=_T,
              ecretage=_C, lr=_lr, inertie=_m, points=[])
    for _eps in config["dp_epsilons"]:
        _sigma = bruit_pour_epsilon(_eps, _delta, _T, _C)
        assert abs(epsilon_pour_bruit(_sigma, _delta, _T, _C) - _eps) < 1e-9
        for _taille, _Xs, _ys in (("complet", _Xa, _ya), ("petit", _Xa[_petit], _ya[_petit])):
            _ex = []
            for _rep in range(config["dp_repetitions"]):
                _rng_b = np.random.default_rng(1000 * _rep + 42)
                _ex.append(_evaluer(descente_privee(_Xs, _ys, _T, _lr, _C, _sigma, _rng_b, _m))[0])
            dp["points"].append(dict(epsilon=_eps, taille=_taille, sigma=_sigma,
                                     moyenne=float(np.mean(_ex)), min=float(np.min(_ex)),
                                     max=float(np.max(_ex))))
            print(f"ε = {_eps:<5} σ = {_sigma:8.1f}  {_taille:<8} exactitude {np.mean(_ex):.1%} "
                  f"(min {np.min(_ex):.1%}, max {np.max(_ex):.1%})")
    duree_dp = time.perf_counter() - _t0
    print(f"Partie 5 : {duree_dp:.1f} s")
    return dp, duree_dp, laplace


@app.cell
def _(SMOKE, dp, gt, politiques, res_base, roc, scores, yt):
    # --- Figures de l'exercice : taux par groupe pour chaque politique, et exactitude selon ε
    _pol = [("seuil 0,5\navec SEX", res_base["avec"]), ("seuil 0,5\nsans SEX", res_base["sans"]),
            ("parité\ndémographique", politiques["parite"]),
            ("égalité des\nchances", politiques["egalite_chances"])]
    _fig, _axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    for _ax, _cle, _titre in zip(_axes, ["selection", "tvp", "tfp", "vpp"],
                                 ["taux de sélection", "taux de vrais positifs",
                                  "taux de faux positifs", "précision (VPP)"]):
        _x = np.arange(len(_pol))
        _ax.bar(_x - 0.2, [r["taux"][0][_cle] for _, r in _pol], 0.4, color=PALETTE[0], label="hommes")
        _ax.bar(_x + 0.2, [r["taux"][1][_cle] for _, r in _pol], 0.4, color=PALETTE[1], label="femmes")
        _ax.set_xticks(_x, [n for n, _ in _pol], fontsize=8)
        _ax.set_title(_titre)
    _axes[0].legend()
    _fig.tight_layout()
    _dossier = mo.notebook_dir() / "figures"
    sauvegarder_figure(_fig, "equite-smoke" if SMOKE else "equite", _dossier)

    _fig2, _ax2 = plt.subplots(figsize=(6.5, 4))
    for _taille, _coul in (("complet", PALETTE[0]), ("petit", PALETTE[3])):
        _pts = [p for p in dp["points"] if p["taille"] == _taille]
        _ax2.errorbar([p["epsilon"] for p in _pts], [p["moyenne"] for p in _pts],
                      yerr=[[p["moyenne"] - p["min"] for p in _pts], [p["max"] - p["moyenne"] for p in _pts]],
                      marker="o", color=_coul, capsize=3,
                      label=f"{dp['n_' + _taille]:,} personnes".replace(",", " "))
        _ax2.axhline(dp["sans_bruit"][_taille], color=_coul, ls=":", lw=1)
    _ax2.set(xscale="log", xlabel="budget de confidentialité ε (δ = 10⁻⁶)", ylabel="exactitude (test)",
             title="Régression logistique différentiellement privée")
    _ax2.legend()
    sauvegarder_figure(_fig2, "confidentialite-smoke" if SMOKE else "confidentialite", _dossier)

    # Matrices de confusion et ROC pour les figures du site
    _st = scores["avec"]["test"]
    _pts_op = {}
    for _nom, _r in (("seuil", res_base["avec"]), ("parite", politiques["parite"]),
                     ("egalite_chances", politiques["egalite_chances"])):
        _pts_op[_nom] = {g: dict(tfp=_r["taux"][g]["tfp"], tvp=_r["taux"][g]["tvp"]) for g in (0, 1)}
    figures_donnees = dict(
        roc={str(g): dict(tfp=[round(v, 4) for v in roc[g][0]], tvp=[round(v, 4) for v in roc[g][1]])
             for g in (0, 1)},
        points_fonctionnement=_pts_op,
        auc={str(g): float(roc_auc_score(yt[gt == g], _st[gt == g])) for g in (0, 1)},
    )
    _fig2
    return (figures_donnees,)


@app.cell
def _(
    SMOKE,
    df,
    dp,
    duree_donnees,
    duree_dp,
    duree_modeles,
    erreur_identite,
    femme,
    figures_donnees,
    laplace,
    politiques,
    proxy,
    res_base,
    y,
):
    # --- Résultats chiffrés (lus par les figures du site) puis bilan avec les seuils du README
    def _nettoyer(o):
        if isinstance(o, dict):
            return {str(k): _nettoyer(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_nettoyer(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    resume = _nettoyer(dict(
        n=len(df), base=dict(hommes=float(y[femme == 0].mean()), femmes=float(y[femme == 1].mean())),
        base_par_groupe={"0": int((femme == 0).sum()), "1": int((femme == 1).sum())},
        modeles=res_base, politiques=politiques, proxy=proxy, erreur_identite=erreur_identite,
        laplace=laplace, dp=dp, figures=figures_donnees,
        durees=dict(donnees=duree_donnees, modeles=duree_modeles, confidentialite=duree_dp),
    ))
    _chemin = mo.notebook_dir() / "figures" / ("resultats-smoke.json" if SMOKE else "resultats.json")
    _chemin.write_text(json.dumps(resume, ensure_ascii=False, indent=1))

    _a, _s = res_base["avec"], res_base["sans"]
    _pa, _ec = politiques["parite"], politiques["egalite_chances"]
    _dp_c = {p["epsilon"]: p for p in dp["points"] if p["taille"] == "complet"}
    _dp_p = {p["epsilon"]: p for p in dp["points"] if p["taille"] == "petit"}
    _l = {(v["epsilon"], v["code"]): v for v in laplace.values()}
    verifs = [
        ("exactitude du modèle avec SEX ≥ 0,80", _a["exactitude"] >= 0.80),
        ("écart de sélection (avec SEX, seuil 0,5) ≤ −0,08", _a["ecarts"]["selection"] <= -0.08),
        ("sans SEX : l'écart de sélection reste ≤ −0,05", _s["ecarts"]["selection"] <= -0.05),
        ("le sexe se prédit sans la colonne SEX : AUC ≥ 0,75", proxy["auc"] >= 0.75),
        ("parité démographique : |écart de sélection| ≤ 0,01", abs(_pa["ecarts"]["selection"]) <= 0.01),
        ("égalité des chances : |écart de TVP| ≤ 0,01", abs(_ec["ecarts"]["tvp"]) <= 0.01),
        ("post-traitements : perte d'exactitude ≤ 2 points",
         max(_a["exactitude"] - _pa["exactitude"], _a["exactitude"] - _ec["exactitude"]) <= 0.02),
        ("égalité des chances : écart de VPP ≥ 0,03 en valeur absolue", abs(_ec["ecarts"]["vpp"]) >= 0.03),
        ("identité de Chouldechova vérifiée à 1e-9 près", erreur_identite <= 1e-9),
        ("Laplace ε = 1 : erreur relative < 0,01 % sur le grand groupe, > 10 % sur le plus petit",
         _l[(1.0, 1)]["erreur_rel_moy"] < 1e-4 and _l[(1.0, 4)]["erreur_rel_moy"] > 0.10),
        ("DP ε = 1, jeu complet : exactitude à moins de 1 point du modèle sans bruit",
         dp["sans_bruit"]["complet"] - _dp_c.get(1.0, {"moyenne": 0})["moyenne"] <= 0.01),
        ("DP ε = 0,1, 2 000 personnes : perte d'au moins 5 points",
         dp["sans_bruit"]["petit"] - _dp_p.get(0.1, {"moyenne": 1})["moyenne"] >= 0.05),
    ]
    for _nom, _ok in verifs:
        # En smoke (20 000 personnes, 60 pas), les seuils ne sont pas vérifiés
        print(f"[{'OK' if _ok else ('non atteint (smoke)' if SMOKE else 'ÉCHEC')}] {_nom}")
    _echecs = [n for n, ok in verifs if not ok]
    if not SMOKE and _echecs:
        raise SystemExit("Seuils non atteints : " + " ; ".join(_echecs))
    mo.md(f"**Bilan** : {len(verifs) - len(_echecs)} seuils sur {len(verifs)} atteints"
          + (" (mode smoke : non vérifiés)" if SMOKE else "") + ".")
    return


if __name__ == "__main__":
    app.run()
