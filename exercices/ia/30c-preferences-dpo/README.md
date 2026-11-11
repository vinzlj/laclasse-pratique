# Exercice 30c : apprendre des préférences, en NumPy (Bradley-Terry, sur-optimisation, DPO)

| | |
|---|---|
| **Chapitre lié** | [30. Adaptation et alignement](https://laclasse.poneyponey.fr/ia/p08-llm/30-adaptation-alignement/) |
| **Exécution** | locale (CPU, NumPy seul) |
| **Commande** | `make ex NUM=30c` (éditeur), `MODE=run` (complet), `MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 9 s (M1 Pro, CPU, 15/10/2026, attente du verrou exclue), dont ≈ 6,5 s de calcul (partie A : 2,5 s ; B : 0,3 s ; C : 3,6 s) ; smoke : 3 s |
| **Travail estimé** | ~1 h |
| **Seed** | 42 (`seed_everything(42)` et `np.random.default_rng(42)`) |
| **Données** | générées (humains simulés), rien n'est téléchargé |

## Objectif

Voir sur un exemple jouet, chiffres à l'appui, les trois mécanismes de l'alignement par préférences : un modèle de récompense de Bradley-Terry appris sur des paires, ce qui arrive quand on optimise trop fort une récompense apprise (sur-optimisation, Gao et al. 2022), et le fait que DPO retrouve exactement la politique que le RLHF cherche, sans modèle de récompense ni renforcement. Les versions « bibliothèque » sur un vrai LLM sont les exercices cloud [30a](../30a-cloud-lora/README.md) (SFT, LoRA, QLoRA) et [30b](../30b-cloud-dpo/README.md) (DPO avec TRL).

## Ce que vous allez faire

Une « réponse » est résumée par deux nombres : sa qualité $q$ et sa longueur $L$ (centrée réduite), corrélées ($\rho = 0{,}5$) dans les réponses du « modèle SFT ». Les vrais goûts des humains simulés sont $u^\star = q + 0{,}5\,L - 0{,}3\,L^2$ : un peu de longueur plaît, trop déplaît (optimum en $L = 0{,}83$). Un humain préfère $a$ à $b$ avec la probabilité $\sigma(u^\star(a) - u^\star(b))$ : il hésite quand les réponses se valent.

1. **Partie A, modèle de récompense.** Ajuster par descente de gradient un modèle de Bradley-Terry linéaire $r = w_1 \tilde q + w_2 L$ sur 2 000 paires étiquetées. Il ne voit la qualité qu'avec un bruit d'écart type 1,5 ($\tilde q$), mais voit la longueur exactement. Deux variantes : 20 000 paires ; et un modèle qui a aussi la caractéristique $L^2$ (la bonne forme). Mesurer la précision sur 5 000 paires de test, et la comparer au plafond (un juge qui connaîtrait $u^\star$).
2. **Partie B, sur-optimisation.** Best-of-n : tirer $n$ réponses ($n$ = 1 à 4 096), garder celle que le modèle de récompense préfère, mesurer sa récompense apprise, sa vraie utilité et sa longueur (moyenne sur 1 000 tirages). L'écart à la politique de départ vaut $\mathrm{KL} = \log n - (n-1)/n$, exactement ici parce que les réponses tirées, continues, sont toutes distinctes (en général, c'est une borne supérieure : Beirami et al., 2024).
3. **Partie C, DPO.** Huit réponses candidates, une politique de référence $\pi_{\text{ref}}$, 3 000 paires tirées de $\pi_{\text{ref}}$ et étiquetées. Chemin RLHF : ajuster une récompense de Bradley-Terry $\hat r$ (un score par réponse, descente de gradient), puis calculer la solution exacte $\pi^\star \propto \pi_{\text{ref}}\,e^{\hat r/\beta}$. Chemin DPO : optimiser directement les logits de la politique avec la perte DPO (gradient écrit à la main, optimiseur Adam). Comparer, pour $\beta$ de 0,05 à 5, et ajouter un témoin « SFT sur les seules réponses préférées ».

**Pertes optimisées** : Bradley-Terry $-\log\sigma\big(r(y_w) - r(y_l)\big)$ (parties A et C) ; DPO $-\log\sigma\big(\beta\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\big)$ (partie C). La partie B n'optimise rien par gradient : elle choisit.

## Résultat attendu

Le run complet vérifie ces seuils et s'arrête en erreur s'ils ne sont pas atteints (valeurs du run du 15/10/2026, graine 42) :

| Vérification | Seuil | Obtenu |
|---|---|---|
| A. Précision du modèle linéaire sur les paires de test | ≥ 65 % et sous le plafond | **70,9 %** (plafond 76,8 %) ; 20 000 paires : 71,0 % ; avec $L^2$ : 71,7 % |
| B. Récompense apprise | croissante avec $n$ | 0,00 → **3,05** (n = 4 096) |
| B. Vraie utilité, modèle linéaire | chute ≥ 0,3 après son pic | −0,28 → **1,02 au pic (n = 64)** → **0,46** (n = 4 096) |
| B. Vraie utilité, modèle avec $L^2$ | pas de chute de plus de 0,1 | 2,14 à n = 4 096, sans chute |
| C. KL($\pi_{\text{DPO}}$ ‖ $\pi^\star$), β = 0,5 | < 10⁻³ | **2 × 10⁻¹⁴** |

Autres valeurs du run complet :

- Partie A : poids appris $w = (0{,}18 ; 0{,}69)$ : le modèle s'appuie surtout sur la longueur. Perte d'entraînement 0,575 (log 2 = 0,693 pour un modèle qui ne sait rien).
- Partie B : longueur moyenne retenue 0,0 (n = 1) → 3,37 écarts types (n = 4 096), contre 0,83 préférée. Avec 20 000 paires : pic 1,04 à n = 32, puis 0,53. Plafond (choix avec les vrais goûts) : 3,60 à n = 4 096.
- Partie C (β = 0,5) : perte DPO 0,693 au départ (= log 2) → 0,508 ; écart maximal entre la récompense implicite $\beta \log \pi_{\text{DPO}}/\pi_{\text{ref}}$ (centrée) et $\hat r$ : 2,6 × 10⁻⁷. Vraie utilité moyenne : $\pi_{\text{ref}}$ 0,072 ; SFT sur les préférées 0,552 (KL 0,126) ; DPO 0,890 (KL 0,494). Pour chaque β, l'utilité de DPO est à moins de 0,025 de l'optimum calculé avec les vrais goûts au même β (ex. β = 1 : 0,688 contre 0,703 ; β = 0,1 : 1,210 contre 1,211).

En mode smoke (`n` s'arrête à 256, moins de paires), les seuils ne sont pas exigés : un seuil manqué s'affiche « (non vérifié en smoke) », pas « ÉCHEC ». Dans l'éditeur (`make ex NUM=30c`), les calculs attendent un clic sur le bouton « Lancer les calculs ».

Figure : `figures/preferences.png` ; résultats : `figures/resultats.json` (lu par les figures du site `ch30-sur-optimisation` et `ch30-dpo-optimum`).

## Ce qu'il faut observer

- **Une récompense apprise n'est qu'un indicateur.** Le modèle de récompense classe 71 % des paires de test comme les humains (le plafond est 77 %, car les humains simulés hésitent). C'est bon, et pourtant il valorise la longueur pour elle-même : dans la zone où il a été entraîné, plus long voulait souvent dire meilleur.
- **Optimiser fort casse l'indicateur (loi de Goodhart).** Jusqu'à n ≈ 64 (KL ≈ 3 nats), récompense apprise et vraie utilité montent ensemble. Au-delà, la première monte toujours, la seconde baisse : best-of-n trouve des réponses de plus en plus longues, hors de la zone où le modèle de récompense avait vu des données. Dix fois plus de paires n'y changent presque rien (pic 1,04 contre 1,02) : le défaut vient de la **forme** du modèle, pas du manque de données. C'est la raison d'être de la pénalité KL du RLHF et du β de DPO : rester près de la politique de départ, là où la récompense est fiable.
- **DPO = RLHF sans les intermédiaires.** Deux chemins, deux optimiseurs différents, la même politique au bout (barres identiques, KL ≈ 10⁻¹⁴). La récompense implicite $\beta\log\pi/\pi_{\text{ref}}$ de DPO est, à une constante près, la récompense de Bradley-Terry. La constante de normalisation $Z(x)$, incalculable sur un vrai LLM, disparaît parce que la perte ne regarde que des **différences** entre deux réponses au même prompt.
- **β règle le compromis.** β petit : la politique se concentre sur la meilleure réponse (KL ≈ 1,7) ; β grand : elle reste près de $\pi_{\text{ref}}$. Tous les points DPO suivent la frontière optimale. Le témoin SFT sur les réponses préférées tombe lui aussi près de cette frontière ici (0,552 pour une KL de 0,126), mais sans réglage possible : il ne regarde jamais la réponse rejetée.

## Questions de réflexion

1. Une paire où les deux réponses ont la même récompense coûte $\log 2 = 0{,}693$. Que vaut la perte de Bradley-Terry pour une marge $r(y_w) - r(y_l) = 0{,}8$ ? Et pour une marge de −0,8 (le modèle se trompe) ? Pourquoi la perte ne s'annule-t-elle jamais tout à fait ?
2. Dans la partie B, on a supposé que l'on connaissait $u^\star$ pour mesurer la « vraie » utilité. Dans un vrai projet, comment Gao et al. (2022) ont-ils fait pour disposer d'une « vraie » récompense ? (Indice : un grand modèle de récompense joue le rôle de l'humain.)
3. Pourquoi la pénalité KL du RLHF (ou β dans DPO) limite-t-elle la sur-optimisation ? Reliez la réponse au pic observé vers KL ≈ 3 nats.
4. DPO n'utilise jamais de réponse générée pendant l'entraînement, contrairement au RLHF par PPO. Qu'est-ce que cela fait gagner ? Qu'est-ce que cela peut faire perdre si les paires ont été produites par un autre modèle que $\pi_{\text{ref}}$ ?

## Pour aller plus loin

- Réduire `bruit_qualite` à 0,5 : le modèle de récompense perçoit mieux la qualité ; le pic de la partie B arrive-t-il plus tard ?
- Remplacer le best-of-n de la partie B par la politique exacte $\pi \propto \pi_{\text{ref}}\,e^{r/\beta}$ sur un grand échantillon (pondération), et tracer la vraie utilité en fonction de β.
- Articles : Bradley et Terry, *Biometrika* 1952, [doi:10.2307/2334029](https://doi.org/10.2307/2334029) ; Gao, Schulman et Hilton, [arXiv:2210.10760](https://arxiv.org/abs/2210.10760) ; Rafailov et al., DPO, [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) ; Stiennon et al. (formule de la KL du best-of-n), [arXiv:2009.01325](https://arxiv.org/abs/2009.01325) ; Beirami et al. (cette formule est en général une borne supérieure), [arXiv:2401.01879](https://arxiv.org/abs/2401.01879).
