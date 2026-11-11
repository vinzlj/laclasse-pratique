# Exercice 30c : apprendre des préférences, en NumPy (chapitre 30).
# Modèle de récompense de Bradley-Terry, sur-optimisation d'une récompense apprise
# (best-of-n), puis DPO comparé à la solution exacte du problème que résout le RLHF.
#
#   make ex NUM=30c              → éditeur marimo (interactif)
#   make ex NUM=30c MODE=run     → script complet (vérifie les seuils du README)
#   make ex NUM=30c MODE=smoke   → test rapide (< 30 s)
#
# NumPy seul, sur CPU : aucun réseau, aucun téléchargement. Les versions « bibliothèque »
# (LoRA avec PEFT, DPO avec TRL) sont les exercices cloud 30a et 30b.
# Les figures du site (site/scripts/figures/ch30_*.py) lisent figures/resultats.json,
# écrit par le run complet.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import json
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from ia_commun import PALETTE, is_smoke, sauvegarder_figure, seed_everything, style_commun


@app.cell
def _():
    mo.md(r"""
    # Exercice 30c : apprendre des préférences humaines, en NumPy

    Un LLM réglé par instructions (SFT) sait répondre ; reste à lui faire préférer les
    **bonnes** réponses. Les humains ne savent pas donner une note absolue fiable, mais ils
    savent dire laquelle de deux réponses ils préfèrent. On part donc de **paires**
    (réponse préférée $y_w$, réponse rejetée $y_l$).

    Pour voir tous les mécanismes sans GPU, une « réponse » est ici résumée par deux nombres :
    sa **qualité** $q$ (exactitude, utilité) et sa **longueur** $L$ (centrée réduite : 0 =
    longueur moyenne). Le « modèle SFT » produit des réponses où les deux sont corrélées
    (une réponse plus longue est en moyenne plus détaillée). Les vrais goûts humains, inconnus
    de l'algorithme, sont
    $$u^\star(y) = q + 0{,}5\,L - 0{,}3\,L^2 :$$
    un peu de longueur plaît, trop de longueur déplaît (maximum en $L \approx 0{,}83$).

    **Partie A. Modèle de récompense (Bradley-Terry, 1952).** On suppose
    $P(y_w \succ y_l) = \sigma\big(r(y_w) - r(y_l)\big)$ et l'on ajuste $r$ en minimisant
    $$\mathcal{L}_{\text{BT}} = -\,\mathbb{E}\big[\log \sigma\big(r(y_w) - r(y_l)\big)\big],$$
    une entropie croisée binaire sur la différence des scores. Le modèle de récompense ne voit
    la qualité qu'à travers un bruit (il la perçoit mal), mais il voit la longueur exactement.

    **Partie B. Sur-optimisation.** On choisit la meilleure de $n$ réponses selon la
    récompense **apprise** (best-of-n) et l'on mesure la vraie utilité $u^\star$ : quand on
    pousse fort, la récompense apprise monte toujours, la vraie finit par baisser
    (Gao et al., 2022 ; loi de Goodhart).

    **Partie C. DPO (Rafailov et al., 2023).** Le RLHF cherche la politique qui maximise
    $\mathbb{E}_\pi[r] - \beta\,\mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})$ ; la solution exacte
    est $\pi^\star(y) \propto \pi_{\text{ref}}(y)\,e^{r(y)/\beta}$. DPO saute le modèle de
    récompense et minimise directement
    $$\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}\Big[\log \sigma\Big(\beta \log \tfrac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)}
    - \beta \log \tfrac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\Big)\Big].$$
    On vérifie numériquement qu'il retombe sur $\pi^\star$.
    """)


@app.cell
def _():
    # --- Configuration : tout ce qui change entre mode complet et mode smoke est ici ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        n_paires=500 if SMOKE else 2000,         # paires étiquetées pour le modèle de récompense
        n_paires_grand=2000 if SMOKE else 20000,  # même modèle, dix fois plus de données
        n_test=500 if SMOKE else 5000,            # paires de test
        bruit_qualite=1.5,                        # écart type du bruit sur la qualité perçue
        n_best_of=[1, 2, 4, 8, 16, 32, 64, 128, 256] + ([] if SMOKE else [512, 1024, 2048, 4096]),
        n_requetes=200 if SMOKE else 1000,        # tirages indépendants par valeur de n
        n_reponses=8,                             # partie C : réponses candidates
        n_paires_dpo=500 if SMOKE else 3000,
        betas=[0.1, 0.3, 1.0, 3.0] if SMOKE else [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0],
        beta_principal=0.5,
        n_iter=1000 if SMOKE else 4000,           # descente de gradient (problèmes convexes)
    )
    rng = np.random.default_rng(SEED)
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SMOKE, config, rng


@app.cell
def _():
    # Le bouton empêche les calculs (≈ 7 s) de se relancer à chaque modification d'une cellule amont.
    bouton = mo.ui.run_button(label="Lancer les calculs")
    bouton
    return (bouton,)


@app.function
def sigmoide(z):
    """σ(z) = 1 / (1 + e^(−z)), écrite de façon stable pour les grands |z|."""
    return np.exp(-np.logaddexp(0.0, -z))


@app.function
def log_sigmoide(z):
    """log σ(z) = −log(1 + e^(−z)), stable."""
    return -np.logaddexp(0.0, -z)


@app.function
def tirer_reponses(rng, n):
    """Réponses du « modèle SFT » : qualité q ~ N(0, 1), longueur L corrélée à q (ρ = 0,5)."""
    q = rng.normal(size=n)
    L = 0.5 * q + np.sqrt(0.75) * rng.normal(size=n)
    return q, L


@app.function
def utilite_humaine(q, L):
    """Vrais goûts humains (inconnus des algorithmes) : u* = q + 0,5·L − 0,3·L²."""
    return q + 0.5 * L - 0.3 * L**2


@app.function
def ajuster_bradley_terry(D, y, n_iter, lr=1.0):
    """Régression logistique sur des différences : P(a ≻ b) = σ(w · (φ(a) − φ(b))).

    D : (N, d) différences de caractéristiques, y : 1 si a est préférée. Descente de
    gradient sur la perte de Bradley-Terry (problème convexe). Renvoie w et la perte finale.
    """
    w = np.zeros(D.shape[1])
    for _ in range(n_iter):
        p = sigmoide(D @ w)
        w -= lr * D.T @ (p - y) / len(y)
    z = D @ w
    perte = -np.mean(y * log_sigmoide(z) + (1 - y) * log_sigmoide(-z))
    return w, float(perte)


@app.function
def kl_best_of_n(n):
    """KL(best-of-n ‖ politique de départ) = log n − (n − 1)/n (Stiennon et al., 2020).

    Exacte quand les n réponses tirées sont toutes distinctes (c'est le cas ici : elles sont
    continues) ; sinon, borne supérieure (Beirami et al., 2024).
    """
    return np.log(n) - (n - 1) / n


@app.cell
def _():
    mo.md(r"""
    ## Partie A : un modèle de récompense de Bradley-Terry

    Exemple à la main : si $r(y_w) = 1{,}2$ et $r(y_l) = 0{,}4$, le modèle donne
    $P(y_w \succ y_l) = \sigma(0{,}8) = 0{,}690$ et la perte de cette paire vaut
    $-\log 0{,}690 = 0{,}371$. Deux réponses à égalité ($r(y_w) = r(y_l)$) coûtent
    $\log 2 = 0{,}693$.
    """)


@app.cell
def _(bouton, config, rng):
    # En script (make ex MODE=run|smoke), il n'y a personne pour cliquer : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer les calculs » (au-dessus) pour lancer les parties A et B.*"))

    def _calcul():
        # Exemple chiffré de la cellule précédente
        print(f"σ(0,8) = {sigmoide(0.8):.3f} ; perte = {-log_sigmoide(0.8):.3f} ; égalité : {-log_sigmoide(0.0):.3f}")

        def paires(n):
            """n paires tirées du modèle SFT, étiquetées par un humain simulé (Bradley-Terry sur u*)."""
            qa, La = tirer_reponses(rng, n)
            qb, Lb = tirer_reponses(rng, n)
            diff_vraie = utilite_humaine(qa, La) - utilite_humaine(qb, Lb)
            y = (rng.random(n) < sigmoide(diff_vraie)).astype(float)   # l'humain hésite parfois
            # Le modèle de récompense perçoit la qualité avec un bruit, la longueur exactement
            qa_vue = qa + config["bruit_qualite"] * rng.normal(size=n)
            qb_vue = qb + config["bruit_qualite"] * rng.normal(size=n)
            return dict(qa=qa_vue, La=La, qb=qb_vue, Lb=Lb, y=y, diff_vraie=diff_vraie)

        def caracteristiques(q_vue, L, forme):
            # « lineaire » : (q perçue, L) ; « quadratique » : on ajoute L², la forme des vrais goûts
            cols = [q_vue, L] + ([L**2] if forme == "quadratique" else [])
            return np.stack(cols, 1)

        def differences(P, forme):
            return caracteristiques(P["qa"], P["La"], forme) - caracteristiques(P["qb"], P["Lb"], forme)

        t0 = time.perf_counter()
        entrainement = paires(config["n_paires"])
        entrainement_grand = paires(config["n_paires_grand"])
        test = paires(config["n_test"])
        modeles_r = {}
        for nom, donnees, forme in [
            ("lineaire", entrainement, "lineaire"),
            ("lineaire-x10", entrainement_grand, "lineaire"),
            ("quadratique", entrainement, "quadratique"),
        ]:
            w, perte = ajuster_bradley_terry(differences(donnees, forme), donnees["y"], config["n_iter"])
            precision = float(np.mean((differences(test, forme) @ w > 0) == (test["y"] == 1)))
            modeles_r[nom] = dict(w=w.tolist(), forme=forme, perte=perte, precision_test=precision)
            print(f"{nom:13s} w = {np.round(w, 3)}  perte d'entraînement {perte:.3f}  précision test {precision:.1%}")
        # Plafond : un juge qui connaîtrait u* exactement se trompe encore quand l'humain hésite
        plafond = float(np.mean((test["diff_vraie"] > 0) == (test["y"] == 1)))
        print(f"Plafond (juge qui connaît u*) : {plafond:.1%} ; partie A en {time.perf_counter() - t0:.1f} s")
        return caracteristiques, modeles_r, plafond

    caracteristiques, modeles_r, plafond = _calcul()
    return caracteristiques, modeles_r, plafond


@app.cell
def _():
    mo.md(r"""
    ## Partie B : pousser la récompense apprise jusqu'à la casser

    Best-of-n : on tire $n$ réponses et l'on garde celle que le modèle de récompense préfère.
    C'est la façon la plus simple d'optimiser une récompense, et son écart à la politique de
    départ vaut $\mathrm{KL} = \log n - (n-1)/n$ : exactement ici, parce que les réponses tirées
    (continues) sont toutes distinctes ; en général, cette formule n'est qu'une borne supérieure
    (Beirami et al., 2024). Plus $n$ est grand, plus on optimise fort.
    """)


@app.cell
def _(caracteristiques, config, modeles_r, rng):
    def _calcul():
        t0_b = time.perf_counter()
        courbes = {nom: dict(recompense=[], vraie=[], longueur=[]) for nom in [*modeles_r, "oracle"]}
        for n in config["n_best_of"]:
            M = config["n_requetes"]
            q, L = tirer_reponses(rng, M * n)
            q, L = q.reshape(M, n), L.reshape(M, n)
            q_vue = q + config["bruit_qualite"] * rng.normal(size=q.shape)
            u = utilite_humaine(q, L)
            lignes = np.arange(M)
            for nom, m in modeles_r.items():
                r = caracteristiques(q_vue.ravel(), L.ravel(), m["forme"]) @ np.array(m["w"])
                r = r.reshape(M, n)
                k = r.argmax(1)                        # la réponse que la récompense apprise préfère
                courbes[nom]["recompense"].append(float(r[lignes, k].mean()))
                courbes[nom]["vraie"].append(float(u[lignes, k].mean()))
                courbes[nom]["longueur"].append(float(L[lignes, k].mean()))
            k = u.argmax(1)                            # référence : choisir avec les vrais goûts
            courbes["oracle"]["recompense"].append(float(u[lignes, k].mean()))
            courbes["oracle"]["vraie"].append(float(u[lignes, k].mean()))
            courbes["oracle"]["longueur"].append(float(L[lignes, k].mean()))
        kl_bon = [float(kl_best_of_n(n)) for n in config["n_best_of"]]

        print("   n   KL    | lineaire : récompense apprise, vraie utilité, longueur | quadratique : vraie | oracle : vraie")
        for i, n in enumerate(config["n_best_of"]):
            c = courbes["lineaire"]
            print(f"{n:5d} {kl_bon[i]:5.2f} | {c['recompense'][i]:6.2f} {c['vraie'][i]:6.2f} {c['longueur'][i]:5.2f}"
                  f" | {courbes['quadratique']['vraie'][i]:6.2f} | {courbes['oracle']['vraie'][i]:6.2f}")
        bilan_b = {}
        for nom in courbes:
            v = courbes[nom]["vraie"]
            i_max = int(np.argmax(v))
            bilan_b[nom] = dict(n_pic=config["n_best_of"][i_max], vraie_pic=v[i_max], vraie_fin=v[-1],
                                vraie_depart=v[0])
            print(f"{nom:13s} vraie utilité : départ {v[0]:.2f}, pic {v[i_max]:.2f} à n = {config['n_best_of'][i_max]},"
                  f" fin {v[-1]:.2f} (n = {config['n_best_of'][-1]})")
        print(f"Partie B en {time.perf_counter() - t0_b:.1f} s")
        return bilan_b, courbes, kl_bon

    bilan_b, courbes, kl_bon = _calcul()
    return bilan_b, courbes, kl_bon


@app.cell
def _():
    mo.md(r"""
    ## Partie C : DPO contre la solution exacte du RLHF

    Huit réponses candidates à une même question. La politique de référence
    $\pi_{\text{ref}}$ (le modèle SFT) leur donne des probabilités inégales ; les paires
    étiquetées sont tirées de $\pi_{\text{ref}}$, comme dans un vrai jeu de préférences.

    - **Chemin RLHF** : on ajuste une récompense $\hat r$ (Bradley-Terry, un score par
      réponse), puis on calcule la politique optimale **exacte**
      $\pi^\star \propto \pi_{\text{ref}}\, e^{\hat r/\beta}$. Un vrai RLHF l'approche par
      PPO, avec quatre réseaux en mémoire (politique, référence, récompense, critique).
    - **Chemin DPO** : on part de $\pi_\theta = \pi_{\text{ref}}$ et l'on descend le gradient
      de $\mathcal{L}_{\text{DPO}}$ sur les mêmes paires. Aucun modèle de récompense.
    - **Témoin SFT sur les réponses préférées** : maximiser $\log \pi(y_w)$ seulement, sans
      jamais regarder $y_l$.

    Au départ, $\pi_\theta = \pi_{\text{ref}}$ : chaque log-ratio est nul et la perte DPO vaut
    $\log 2 = 0{,}693$ par paire.
    """)


@app.cell
def _(bouton, config, rng):
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer les calculs » (en haut) pour lancer la partie C.*"))

    def _calcul():
        t0_c = time.perf_counter()
        K = config["n_reponses"]
        q_c, L_c = tirer_reponses(rng, K)
        u_c = utilite_humaine(q_c, L_c)                     # vraie utilité de chaque réponse
        z_ref = rng.normal(size=K)                          # logits de la politique de référence
        log_pi_ref = z_ref - np.logaddexp.reduce(z_ref)
        pi_ref = np.exp(log_pi_ref)

        # Jeu de préférences : deux réponses tirées de π_ref, étiquetées par l'humain simulé
        a = rng.choice(K, size=4 * config["n_paires_dpo"], p=pi_ref)
        b = rng.choice(K, size=4 * config["n_paires_dpo"], p=pi_ref)
        garder = a != b
        a, b = a[garder][: config["n_paires_dpo"]], b[garder][: config["n_paires_dpo"]]
        a_gagne = rng.random(len(a)) < sigmoide(u_c[a] - u_c[b])
        gagnante = np.where(a_gagne, a, b)
        perdante = np.where(a_gagne, b, a)
        E = np.eye(K)
        D_paires = E[gagnante] - E[perdante]                # une ligne par paire : +1 / −1

        def log_softmax(z):
            return z - np.logaddexp.reduce(z)

        def kl(p, q):
            # max(0, …) : une KL est positive ; sans lui, l'arrondi peut afficher −7e-17.
            return max(0.0, float(np.sum(p * (np.log(p) - np.log(q)))))

        # --- Chemin RLHF : récompense de Bradley-Terry (score par réponse, y = 1 : D = gagnante − perdante)
        r_chapeau, perte_bt = ajuster_bradley_terry(D_paires, np.ones(len(gagnante)), config["n_iter"], lr=2.0)
        r_chapeau -= r_chapeau.mean()                      # défini à une constante près

        def dpo(beta, n_iter, lr=0.05):
            """DPO à la main : θ = logits de la politique, initialisés à ceux de π_ref.

            Optimiseur volontairement différent de celui du modèle de récompense (Adam ici,
            descente de gradient simple là-bas) : si les deux chemins tombent sur la même
            politique, c'est que leurs optimums coïncident, pas que leurs calculs sont recopiés.
            """
            theta = z_ref.copy()
            m1, m2 = np.zeros(K), np.zeros(K)
            pertes = []
            for it in range(n_iter):
                lp = log_softmax(theta)
                # marge implicite : β·[log π/π_ref (y_w) − log π/π_ref (y_l)]
                marge = beta * ((lp - log_pi_ref)[gagnante] - (lp - log_pi_ref)[perdante])
                if it % 50 == 0 or it == n_iter - 1:
                    pertes.append(float(-np.mean(log_sigmoide(marge))))
                # ∂ marge / ∂θ = β·(e_w − e_l) : le terme log-somme-exp du softmax s'annule
                # dans la différence, c'est pourquoi DPO n'a pas besoin de la constante Z(x).
                coef = -(1 - sigmoide(marge)) * beta       # ∂(−log σ(m))/∂m · β
                g = (D_paires.T @ coef) / len(coef)
                # Adam (β1 = 0,9, β2 = 0,999)
                m1 = 0.9 * m1 + 0.1 * g
                m2 = 0.999 * m2 + 0.001 * g**2
                theta -= lr * (m1 / (1 - 0.9 ** (it + 1))) / (np.sqrt(m2 / (1 - 0.999 ** (it + 1))) + 1e-8)
            return np.exp(log_softmax(theta)), pertes

        resultats_c = dict(pi_ref=pi_ref.tolist(), utilite=u_c.tolist(), longueur=L_c.tolist(),
                           r_chapeau=r_chapeau.tolist(), perte_bt=perte_bt, betas={})
        beta0 = config["beta_principal"]
        for beta in sorted(set(config["betas"]) | {beta0}):
            pi_dpo, pertes_dpo = dpo(beta, config["n_iter"])
            pi_star = np.exp(log_softmax(z_ref + r_chapeau / beta))     # solution exacte, récompense apprise
            pi_oracle = np.exp(log_softmax(z_ref + u_c / beta))         # solution exacte, vrais goûts
            r_implicite = beta * (np.log(pi_dpo) - log_pi_ref)
            r_implicite -= r_implicite.mean()
            resultats_c["betas"][f"{beta:g}"] = dict(
                pi_dpo=pi_dpo.tolist(), pi_star=pi_star.tolist(),
                kl_dpo_star=kl(pi_dpo, pi_star),
                ecart_recompense=float(np.max(np.abs(r_implicite - r_chapeau))),
                kl_dpo_ref=kl(pi_dpo, pi_ref), utilite_dpo=float(pi_dpo @ u_c),
                kl_oracle_ref=kl(pi_oracle, pi_ref), utilite_oracle=float(pi_oracle @ u_c),
                perte_initiale=pertes_dpo[0], perte_finale=pertes_dpo[-1],
            )
        # Témoin : SFT sur les seules réponses préférées = fréquence de chaque réponse parmi les gagnantes
        pi_sft = np.bincount(gagnante, minlength=K) / len(gagnante)
        pi_sft = np.clip(pi_sft, 1e-12, None)
        resultats_c["sft_preferees"] = dict(pi=pi_sft.tolist(), kl_ref=kl(pi_sft, pi_ref),
                                            utilite=float(pi_sft @ u_c))
        resultats_c["utilite_ref"] = float(pi_ref @ u_c)

        B = resultats_c["betas"][f"{beta0:g}"]
        print("réponse :      " + " ".join(f"{k:6d}" for k in range(K)))
        print("utilité u* :   " + " ".join(f"{v:6.2f}" for v in u_c))
        print("π_ref :        " + " ".join(f"{v:6.3f}" for v in pi_ref))
        print(f"π_DPO (β={beta0:g}) : " + " ".join(f"{v:6.3f}" for v in B["pi_dpo"]))
        print(f"π* (β={beta0:g}) :    " + " ".join(f"{v:6.3f}" for v in B["pi_star"]))
        print("SFT préférées :" + " ".join(f"{v:6.3f}" for v in pi_sft))
        print(f"Perte DPO : {B['perte_initiale']:.3f} au départ (log 2 = {np.log(2):.3f}) → {B['perte_finale']:.3f}")
        print(f"KL(π_DPO ‖ π*) = {B['kl_dpo_star']:.2e} ; écart max entre récompense implicite et r̂ : {B['ecart_recompense']:.2e}")
        print(f"Vraie utilité moyenne : π_ref {resultats_c['utilite_ref']:.3f}, SFT préférées {pi_sft @ u_c:.3f} "
              f"(KL {resultats_c['sft_preferees']['kl_ref']:.3f}), DPO β={beta0:g} {B['utilite_dpo']:.3f} (KL {B['kl_dpo_ref']:.3f})")
        print("β       KL(π_DPO‖π_ref)  utilité DPO  utilité optimum (vrais goûts, même β)")
        for cle, v in resultats_c["betas"].items():
            print(f"{cle:6s}  {v['kl_dpo_ref']:8.3f}        {v['utilite_dpo']:7.3f}      {v['utilite_oracle']:7.3f}")
        print(f"Partie C en {time.perf_counter() - t0_c:.1f} s")
        return B, beta0, resultats_c

    B, beta0, resultats_c = _calcul()
    return B, beta0, resultats_c


@app.cell
def _(
    B,
    SMOKE,
    beta0,
    bilan_b,
    config,
    courbes,
    kl_bon,
    modeles_r,
    plafond,
    resultats_c,
):
    # --- Figure : sur-optimisation (B) et DPO contre π* (C) ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.4))
    _x = np.sqrt(kl_bon)
    _c = courbes["lineaire"]
    ax1.plot(_x, _c["recompense"], "o--", color=PALETTE[0], label="récompense apprise (linéaire)")
    ax1.plot(_x, _c["vraie"], "o-", color=PALETTE[3], label="vraie utilité (linéaire, 2 000 paires)")
    ax1.plot(_x, courbes["lineaire-x10"]["vraie"], "s-", color=PALETTE[4], ms=4,
             label="vraie utilité (linéaire, 20 000 paires)")
    ax1.plot(_x, courbes["quadratique"]["vraie"], "^-", color=PALETTE[2], ms=4, label="vraie utilité (avec L²)")
    ax1.plot(_x, courbes["oracle"]["vraie"], ":", color="grey", label="choix avec les vrais goûts")
    ax1.set(xlabel="√KL(best-of-n ‖ départ)", ylabel="moyenne de la réponse retenue",
            title="Partie B : sur-optimisation")
    ax1.legend(fontsize=8)
    ax2.plot(_x, _c["longueur"], "o-", color=PALETTE[3], label="récompense linéaire")
    ax2.plot(_x, courbes["quadratique"]["longueur"], "^-", color=PALETTE[2], label="récompense avec L²")
    ax2.axhline(0.5 / 0.6, color="grey", ls=":", label="longueur préférée (u* maximal)")
    ax2.set(xlabel="√KL(best-of-n ‖ départ)", ylabel="longueur moyenne retenue", title="Ce que l'on optimise vraiment")
    ax2.legend(fontsize=8)
    _k = np.arange(config["n_reponses"])
    _larg = 0.27
    ax3.bar(_k - _larg, resultats_c["pi_ref"], _larg, color="grey", label="π_ref")
    ax3.bar(_k, B["pi_star"], _larg, color=PALETTE[0], label=f"π* (RLHF exact, β = {beta0:g})")
    ax3.bar(_k + _larg, B["pi_dpo"], _larg, color=PALETTE[1], label="π_DPO")
    ax3.set(xlabel="réponse candidate", ylabel="probabilité", title="Partie C : DPO retombe sur π*")
    ax3.set_xticks(_k, [f"{i}\nu*={v:.1f}" for i, v in zip(_k, resultats_c["utilite"])], fontsize=8)
    ax3.legend(fontsize=8)
    fig.tight_layout()
    _dossier = mo.notebook_dir() / "figures"
    _chemins = sauvegarder_figure(fig, "preferences-smoke" if SMOKE else "preferences", _dossier)
    print(f"Figure : {_chemins[0]}")

    _export = dict(config=config, modeles_recompense=modeles_r, plafond=plafond,
                   best_of_n=dict(n=config["n_best_of"], kl=kl_bon, courbes=courbes, bilan=bilan_b),
                   dpo=resultats_c)
    _chemin = _dossier / ("resultats-smoke.json" if SMOKE else "resultats.json")
    _chemin.write_text(json.dumps(_export, ensure_ascii=False, indent=1))
    print(f"Résultats : {_chemin}")
    fig


@app.cell
def _(B, SMOKE, bilan_b, courbes, modeles_r, plafond):
    # --- Bilan : seuils chiffrés du README. En smoke, on vérifie seulement que tout tourne.
    lin = bilan_b["lineaire"]
    verifs = {
        "A. précision du modèle linéaire ≥ 65 % et sous le plafond":
            0.65 <= modeles_r["lineaire"]["precision_test"] <= plafond + 0.01,
        "B. récompense apprise croissante avec n":
            bool(np.all(np.diff(courbes["lineaire"]["recompense"]) > 0)),
        "B. vraie utilité (linéaire) : chute ≥ 0,3 après le pic":
            lin["vraie_pic"] - lin["vraie_fin"] >= 0.3,
        "B. modèle avec L² : pas de chute de plus de 0,1":
            bilan_b["quadratique"]["vraie_pic"] - bilan_b["quadratique"]["vraie_fin"] <= 0.1,
        "C. KL(π_DPO ‖ π*) < 10⁻³": B["kl_dpo_star"] < 1e-3,
    }
    for nom, ok in verifs.items():
        # En smoke (n s'arrête à 256, peu de paires), un seuil manqué n'est pas une panne.
        etat = "OK   " if ok else ("(non vérifié en smoke) " if SMOKE else "ÉCHEC ")
        print(etat + nom)
    if not SMOKE and not all(verifs.values()):
        raise SystemExit("Un seuil du README n'est pas atteint (voir ci-dessus).")
    mo.md(f"**Bilan** : précision du modèle de récompense {modeles_r['lineaire']['precision_test']:.1%} "
          f"(plafond {plafond:.1%}) ; vraie utilité du best-of-n : pic {lin['vraie_pic']:.2f} à n = {lin['n_pic']}, "
          f"puis {lin['vraie_fin']:.2f} ; KL(π_DPO ‖ π*) = {B['kl_dpo_star']:.1e}.")


if __name__ == "__main__":
    app.run()
