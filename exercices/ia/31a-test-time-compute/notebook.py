# Exercice 31a : dépenser du calcul à l'inférence (réponse directe, chaîne de pensée, vote).
#
#   make ex NUM=31a              → éditeur marimo (bouton « Lancer les générations »)
#   make ex NUM=31a MODE=run     → run complet en script (30 problèmes, N = 4), sous verrou GPU
#   make ex NUM=31a MODE=smoke   → test rapide (2 problèmes, N = 2), < 30 s
#
# Prérequis : Ollama lancé en local (`ollama serve`, ou l'application) et le modèle
# téléchargé une fois : `ollama pull qwen3.5:2b` (≈ 3 Go). Aucun entraînement ici : on
# interroge un petit LLM figé et on compare des façons de dépenser des tokens.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import re
    import time
    from collections import Counter

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from ia_commun import PALETTE, is_smoke, sauvegarder_figure, seed_everything, style_commun

    return (
        Counter,
        PALETTE,
        is_smoke,
        json,
        mo,
        np,
        plt,
        re,
        sauvegarder_figure,
        seed_everything,
        style_commun,
        time,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Exercice 31a : calcul à l'inférence sur des problèmes de GSM8K

    **Objectif.** Mesurer ce que rapporte, en précision, chaque token de plus dépensé
    **après** l'entraînement, sur des problèmes d'arithmétique en plusieurs étapes
    (GSM8K, Cobbe et al. 2021) résolus par un petit LLM local (`qwen3.5:2b`, 2,2 milliards
    de paramètres, via Ollama). Quatre stratégies :

    1. **réponse directe** : « donne seulement le nombre » (quelques tokens) ;
    2. **chaîne de pensée courte** (*chain of thought*) : le modèle écrit son raisonnement
       en 6 lignes au plus, puis « Answer: n » (une centaine de tokens) ;
    3. **chaîne de pensée libre** : même chose sans limite de lignes (plus de tokens dans
       **une seule** chaîne : on allonge le calcul « en série ») ;
    4. **vote majoritaire** (*self-consistency*) : N chaînes courtes échantillonnées
       (température 0,7, ici N = 4), on garde la réponse la plus fréquente (N fois le coût :
       on élargit le calcul « en parallèle »).

    On calcule aussi la précision d'un **vérificateur parfait** (« au moins une des N
    réponses est juste », pass@N) : c'est le plafond d'une stratégie best-of-N, et l'écart
    avec le vote dit ce qu'un bon vérificateur pourrait encore gagner.

    Enfin, on calcule en numpy l'**avantage de groupe de GRPO** sur ces mêmes N réponses
    (récompense vérifiable 1 si juste, 0 sinon) : c'est le signal qu'utiliserait un
    entraînement par renforcement (exercice 31b, dans le cloud).

    **Fonction de perte : aucune** dans cet exercice (rien n'est entraîné). Ce qui est
    mesuré : la précision (réponse finale exacte) en fonction du nombre de tokens générés.
    """)


@app.cell
def _(is_smoke, seed_everything, style_commun):
    # --- Configuration : seul endroit où le mode smoke réduit le travail ---
    SMOKE = is_smoke()
    SEED = seed_everything(42)
    style_commun()
    config = dict(
        modele="qwen3.5:2b",
        n_problemes=2 if SMOKE else 30,   # problèmes tirés au hasard (graine 42) dans le test de GSM8K
        n_echantillons=2 if SMOKE else 4,  # N du vote majoritaire
        tokens_direct=24,                  # budget de la réponse directe
        tokens_cot=96 if SMOKE else 384,   # budget d'une chaîne courte (au-delà : coupée)
        tokens_libre=96 if SMOKE else 768,  # budget de la chaîne libre
        temperature_cot=0.7,               # échantillonnage des chaînes de pensée
    )
    print(f"Mode {'smoke' if SMOKE else 'complet'}, graine {SEED}, config {config}")
    return SEED, SMOKE, config


@app.cell
def _(SEED, config, np):
    # --- Données : GSM8K (openai/gsm8k, licence MIT), partie test, 1 319 problèmes ---
    # Téléchargé une fois depuis le Hub Hugging Face dans data/hf/ (≈ 0,4 Mo).
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    chemin_gsm8k = hf_hub_download(
        "openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset",
        revision="740312add88f781978c0658806c59bc2815b9866",
    )
    tous = pq.read_table(chemin_gsm8k).to_pylist()

    def reponse_attendue(solution: str) -> float:
        # La solution de référence se termine par « #### 18 » (virgules de milliers possibles)
        return float(solution.split("####")[-1].strip().replace(",", ""))

    indices = np.random.default_rng(SEED).choice(len(tous), config["n_problemes"], replace=False)
    problemes = [
        {"id": int(i), "question": tous[i]["question"], "attendu": reponse_attendue(tous[i]["answer"])}
        for i in indices
    ]
    print(f"{len(tous)} problèmes dans le test de GSM8K ; {len(problemes)} retenus : "
          f"{[p['id'] for p in problemes]}")
    print("Exemple :", problemes[0]["question"][:200], "→", problemes[0]["attendu"])
    return (problemes,)


@app.cell
def _(config, mo):
    # --- Le LLM local : vérifier qu'Ollama répond et que le modèle est là ---
    import ollama

    client = ollama.Client()
    try:
        presents = {m.model for m in client.list().models}
    except Exception as erreur:  # serveur absent : message clair plutôt qu'une trace
        raise SystemExit(
            "Ollama ne répond pas sur http://127.0.0.1:11434. Lancez l'application Ollama "
            f"(ou `ollama serve`), puis relancez. Détail : {erreur}"
        ) from erreur
    if config["modele"] not in presents:
        raise SystemExit(f"Modèle {config['modele']} absent : `ollama pull {config['modele']}` "
                         f"(≈ 3 Go), puis relancez. Modèles présents : {sorted(presents)}")
    mo.md(f"Ollama répond ; modèle `{config['modele']}` présent.")
    return (client,)


@app.cell
def _(re):
    # --- Consignes (en anglais, comme les problèmes) et lecture de la réponse finale ---
    CONSIGNE_DIRECTE = ("Answer the math problem with the final number only. "
                        "No words, no explanation, no units.")
    # Consigne de chaîne de pensée volontairement cadrée : sans « 6 lignes au plus », le
    # modèle écrit des titres et du LaTeX, ~2,5 fois plus de tokens, et dépasse souvent le budget.
    CONSIGNE_COT = ("Solve the math problem. Write a short step-by-step solution: at most 6 lines, "
                    "one calculation per line, no headings, no LaTeX. "
                    "Then write the final line as 'Answer: <number>'.")
    # Chaîne libre : le modèle choisit sa longueur (souvent titres, étapes détaillées, LaTeX)
    CONSIGNE_LIBRE = ("Solve the math problem. Reason step by step, "
                      "then write the final line as 'Answer: <number>'.")

    NOMBRE = r"-?\d+(?:\.\d+)?"

    def extraire_nombre(texte: str) -> float | None:
        """Nombre après « Answer: » s'il existe, sinon le dernier nombre du texte."""
        t = texte.replace(",", "")  # 1,000 → 1000
        apres = re.findall(r"answer\s*:\s*[^\d\-]{0,12}(" + NOMBRE + ")", t, flags=re.IGNORECASE)
        if apres:
            return float(apres[-1])
        tous_nombres = re.findall(NOMBRE, t)
        return float(tous_nombres[-1]) if tous_nombres else None

    # Petits tests de l'extraction (vérifiables à la main)
    assert extraire_nombre("9 * 2 = 18.\nAnswer: 18") == 18
    assert extraire_nombre("Answer: $1,250") == 1250
    assert extraire_nombre("The total is **42**") == 42
    assert extraire_nombre("je ne sais pas") is None
    return CONSIGNE_COT, CONSIGNE_DIRECTE, CONSIGNE_LIBRE, extraire_nombre


@app.cell
def _(client, config, extraire_nombre):
    def generer(question: str, consigne: str, n_tokens: int, temperature: float, graine: int):
        """Un appel au LLM local. think=False : pas de « réflexion » cachée (mode thinking
        de Qwen 3.5 désactivé), tout le raisonnement est dans la réponse visible."""
        r = client.chat(
            model=config["modele"],
            messages=[{"role": "system", "content": consigne}, {"role": "user", "content": question}],
            think=False,
            # presence_penalty : Ollama applique 1,5 par défaut à qwen3.5, ce qui pénalise
            # tout token déjà écrit (donc les nombres qu'un calcul doit recopier) : on l'annule.
            options={"temperature": temperature, "seed": graine, "num_predict": n_tokens,
                     "presence_penalty": 0.0},
        )
        texte = r.message.content
        coupee = r.done_reason == "length"             # budget de tokens atteint
        # Texte coupé sans ligne « Answer: » : pas de réponse (le dernier nombre écrit
        # serait un résultat intermédiaire, juste par hasard au mieux)
        sans_reponse = coupee and "answer" not in texte.lower()
        return {
            "texte": texte,
            "reponse": None if sans_reponse else extraire_nombre(texte),
            "tokens": r.eval_count,                    # tokens générés (le coût)
            "secondes": r.eval_duration / 1e9,         # temps de génération seul
            "coupee": coupee,
            "chargement_s": (r.load_duration or 0) / 1e9,
        }

    return (generer,)


@app.cell
def _(mo):
    bouton = mo.ui.run_button(label="Lancer les générations")
    bouton
    return (bouton,)


@app.cell
def _(
    CONSIGNE_COT,
    CONSIGNE_DIRECTE,
    CONSIGNE_LIBRE,
    SEED,
    bouton,
    config,
    generer,
    mo,
    problemes,
    time,
):
    # En script (make ex MODE=run|smoke), personne ne clique : on lance directement.
    mo.stop(mo.running_in_notebook() and not bouton.value,
            mo.md("*Cliquez sur « Lancer les générations » (≈ 8 min en mode complet).*"))

    t0 = time.perf_counter()
    # Premier appel : charge le modèle en mémoire (non compté dans les mesures)
    echauffement = generer("1 + 1 = ?", CONSIGNE_DIRECTE, 4, 0.0, SEED)
    print(f"Chargement du modèle : {echauffement['chargement_s']:.1f} s")

    resultats = []
    for _k, _p in enumerate(problemes):
        # 1. Réponse directe, déterministe (température 0)
        directe = generer(_p["question"], CONSIGNE_DIRECTE, config["tokens_direct"], 0.0, SEED)
        # 2-3. N chaînes de pensée échantillonnées, graines 42, 43, 44…
        cots = [generer(_p["question"], CONSIGNE_COT, config["tokens_cot"],
                        config["temperature_cot"], SEED + j)
                for j in range(config["n_echantillons"])]
        # 4. Une chaîne libre, plus longue (même graine que la première chaîne courte)
        libre = generer(_p["question"], CONSIGNE_LIBRE, config["tokens_libre"],
                        config["temperature_cot"], SEED)
        resultats.append({**_p, "directe": directe, "cot": cots, "libre": libre})
        justes = [c["reponse"] == _p["attendu"] for c in cots]
        print(f"[{_k + 1:2d}/{len(problemes)}] attendu {_p['attendu']:g} | directe "
              f"{directe['reponse']} | CoT {[c['reponse'] for c in cots]} {justes} "
              f"| {sum(c['tokens'] for c in cots)} tokens | libre {libre['reponse']} "
              f"({libre['tokens']} tokens)")
    duree = time.perf_counter() - t0
    appels = [a for r in resultats for a in [r["directe"], r["libre"], *r["cot"]]]
    n_tok = sum(a["tokens"] for a in appels)
    s_gen = sum(a["secondes"] for a in appels)
    debit = n_tok / s_gen
    print(f"Générations : {duree:.0f} s au total, {n_tok} tokens, débit {debit:.1f} tokens/s")
    return debit, duree, resultats


@app.cell
def _(Counter, config, np, resultats):
    # --- Analyse : précision et coût de chaque stratégie (numpy) ---
    N = config["n_echantillons"]
    attendu = np.array([r["attendu"] for r in resultats])
    rep_directe = np.array([np.nan if r["directe"]["reponse"] is None else r["directe"]["reponse"]
                            for r in resultats])
    rep_cot = np.array([[np.nan if c["reponse"] is None else c["reponse"] for c in r["cot"]]
                        for r in resultats])                       # (problèmes, N)
    tok_directe = np.array([r["directe"]["tokens"] for r in resultats])
    tok_cot = np.array([[c["tokens"] for c in r["cot"]] for r in resultats])
    coupees = np.array([[c["coupee"] for c in r["cot"]] for r in resultats])
    rep_libre = np.array([np.nan if r["libre"]["reponse"] is None else r["libre"]["reponse"]
                          for r in resultats])
    tok_libre = np.array([r["libre"]["tokens"] for r in resultats])
    juste_libre = rep_libre == attendu

    juste_directe = rep_directe == attendu
    juste_cot = rep_cot == attendu[:, None]                        # récompense vérifiable 0/1

    def vote(reponses: np.ndarray) -> float:
        """Réponse la plus fréquente ; à égalité, la première rencontrée (ordre des graines)."""
        valides = [float(x) for x in reponses if not np.isnan(x)]
        return Counter(valides).most_common(1)[0][0] if valides else np.nan

    # Vote majoritaire avec les k premiers échantillons, k = 1…N
    precision_vote = {k: float(np.mean([vote(rep_cot[i, :k]) == attendu[i]
                                        for i in range(len(attendu))]))
                      for k in range(1, N + 1)}
    bilan = {
        "n_problemes": len(attendu),
        "n_echantillons": int(N),
        "directe": {"precision": float(juste_directe.mean()), "tokens": float(tok_directe.mean())},
        # Une chaîne courte : moyenne sur les N échantillons (estimation de pass@1)
        "cot_1": {"precision": float(juste_cot.mean()), "tokens": float(tok_cot.mean())},
        "libre": {"precision": float(juste_libre.mean()), "tokens": float(tok_libre.mean()),
                  "part_coupees": float(np.mean([r["libre"]["coupee"] for r in resultats]))},
        "vote": {str(k): {"precision": precision_vote[k], "tokens": float(tok_cot[:, :k].sum(1).mean())}
                 for k in precision_vote},
        # Vérificateur parfait : juste si au moins une des N chaînes est juste
        "pass_n": {"precision": float(juste_cot.any(1).mean()), "tokens": float(tok_cot.sum(1).mean())},
        "part_coupees": float(coupees.mean()),
        "justes_par_probleme": juste_cot.sum(1).tolist(),
    }
    print(f"Réponse directe       : {bilan['directe']['precision']:.0%} "
          f"({bilan['directe']['tokens']:.0f} tokens/problème)")
    print(f"Une chaîne libre      : {bilan['libre']['precision']:.0%} "
          f"({bilan['libre']['tokens']:.0f} tokens/problème, "
          f"{bilan['libre']['part_coupees']:.0%} coupées au budget)")
    print(f"Une chaîne courte     : {bilan['cot_1']['precision']:.0%} "
          f"({bilan['cot_1']['tokens']:.0f} tokens/problème, "
          f"{bilan['part_coupees']:.0%} coupées au budget)")
    for _k, _v in bilan["vote"].items():
        print(f"Vote sur {_k} chaîne(s)  : {_v['precision']:.0%} ({_v['tokens']:.0f} tokens/problème)")
    print(f"Vérificateur parfait  : {bilan['pass_n']['precision']:.0%} (pass@{N})")
    return N, bilan, juste_cot


@app.cell
def _(N, bilan, juste_cot, mo, np):
    # --- De l'inférence à l'entraînement : l'avantage de groupe de GRPO, en numpy ---
    # Pour chaque problème (le « groupe »), les N réponses reçoivent une récompense
    # r = 1 si juste, 0 sinon. GRPO remplace le critique de PPO par la moyenne du groupe :
    #     A_i = (r_i − moyenne(r)) / (écart-type(r) + ε)
    r = juste_cot.astype(np.float64)
    moyenne = r.mean(1, keepdims=True)
    ecart = r.std(1, keepdims=True)
    avantages = (r - moyenne) / (ecart + 1e-4)
    sans_signal = (ecart[:, 0] == 0)              # toutes justes ou toutes fausses
    exemple = int(np.argmax(~sans_signal)) if (~sans_signal).any() else 0
    grpo = {"part_sans_signal": float(sans_signal.mean()),
                     "exemple_recompenses": r[exemple].tolist(),
                     "exemple_avantages": [round(float(a), 3) for a in avantages[exemple]]}
    print(f"Groupes sans signal (avantage nul pour les {N} réponses) : {sans_signal.mean():.0%}")
    print(f"Exemple : récompenses {r[exemple].tolist()} → avantages "
          f"{np.round(avantages[exemple], 2).tolist()}")
    mo.md(f"""
    **Avantage de groupe (GRPO).** Sur {len(r)} problèmes, **{sans_signal.mean():.0%}** des groupes
    ont toutes leurs réponses justes ou toutes fausses : avantage nul, aucun gradient.
    Exemple : récompenses {r[exemple].tolist()} → avantages {np.round(avantages[exemple], 2).tolist()}.
    """)
    return (grpo,)


@app.cell
def _(
    N,
    PALETTE,
    SMOKE,
    bilan,
    grpo,
    json,
    mo,
    np,
    plt,
    sauvegarder_figure,
):
    # --- Figure : précision en fonction du coût, et répartition des groupes ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    xs = [bilan["directe"]["tokens"], bilan["cot_1"]["tokens"]]
    ys = [bilan["directe"]["precision"], bilan["cot_1"]["precision"]]
    ax1.plot(xs, np.array(ys) * 100, "o-", color=PALETTE[0], label="directe → une chaîne courte")
    ax1.plot([bilan["cot_1"]["tokens"], bilan["libre"]["tokens"]],
             [bilan["cot_1"]["precision"] * 100, bilan["libre"]["precision"] * 100], "^:",
             color=PALETTE[4], label="une chaîne courte → une chaîne libre (en série)")
    vx = [bilan["vote"][k]["tokens"] for k in bilan["vote"]]
    vy = [bilan["vote"][k]["precision"] * 100 for k in bilan["vote"]]
    ax1.plot(vx, vy, "s-", color=PALETTE[1], label="vote sur k chaînes courtes (en parallèle)")
    ax1.plot([bilan["pass_n"]["tokens"]], [bilan["pass_n"]["precision"] * 100], "D",
             color=PALETTE[2], label=f"vérificateur parfait (pass@{N})")
    ax1.annotate("directe", (xs[0], ys[0] * 100), textcoords="offset points", xytext=(6, -12))
    ax1.set(xscale="log", xlabel="tokens générés par problème (échelle log)",
            ylabel="précision (%)", ylim=(0, 105), title="Précision contre calcul à l'inférence")
    ax1.legend(fontsize=9, loc="lower right")

    comptes = np.bincount(bilan["justes_par_probleme"], minlength=N + 1)
    couleurs = [PALETTE[3] if k in (0, N) else PALETTE[0] for k in range(N + 1)]
    ax2.bar(range(N + 1), comptes, color=couleurs)
    ax2.set(xlabel=f"réponses justes sur {N} (un groupe GRPO)", ylabel="problèmes",
            title="Groupes sans signal (rouge)", xticks=range(N + 1))
    fig.tight_layout()

    dossier = mo.notebook_dir() / "figures"
    nom = "precision-cout-smoke" if SMOKE else "precision-cout"
    chemins = sauvegarder_figure(fig, nom, dossier)
    # Résultats chiffrés pour le site (site/scripts/figures/ch31_*.py les relit)
    (dossier / f"resultats{'-smoke' if SMOKE else ''}.json").write_text(
        json.dumps({**bilan, "grpo": grpo}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Figure : {chemins[0]}")
    fig


@app.cell
def _(SMOKE, bilan, client, config, debit, duree, mo):
    # --- Bilan : seuils du README (vérifiés en mode complet seulement) ---
    client.generate(model=config["modele"], keep_alive=0)  # libère la mémoire du GPU partagé
    gain_cot = bilan["cot_1"]["precision"] - bilan["directe"]["precision"]
    if not SMOKE:
        vote_n = bilan["vote"][str(config["n_echantillons"])]["precision"]
        if gain_cot < 0.25:
            raise SystemExit(f"Gain de la chaîne courte {gain_cot:.0%}, sous 25 points")
        if vote_n < bilan["cot_1"]["precision"] + 0.05:
            raise SystemExit(f"Vote {vote_n:.0%} : moins de 5 points au-dessus d'une chaîne")
    mo.md(f"""
    **Bilan** : réponse directe {bilan['directe']['precision']:.0%}, une chaîne courte
    {bilan['cot_1']['precision']:.0%} (+{gain_cot * 100:.0f} points, seuil +25), une chaîne
    libre {bilan['libre']['precision']:.0%},
    vote sur {config['n_echantillons']} {bilan['vote'][str(config['n_echantillons'])]['precision']:.0%},
    vérificateur parfait {bilan['pass_n']['precision']:.0%}. Débit {debit:.0f} tokens/s,
    {duree:.0f} s de génération.
    """)


if __name__ == "__main__":
    app.run()
