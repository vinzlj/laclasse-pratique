# Exercice 40 : mesurer et corriger l’inéquité d’un classifieur, puis protéger les données

| | |
|---|---|
| **Chapitre lié** | [40. Équité, éthique, impacts et limites ouvertes](https://laclasse.poneyponey.fr/ia/p10-fiabilite/40-equite-ethique/) |
| **Exécution** | locale, CPU (numpy, pandas, scikit-learn ; aucun réseau de neurones, pas de GPU) |
| **Commande** | `make ex NUM=40` (éditeur marimo), `make ex NUM=40 MODE=run` (complet), `make ex NUM=40 MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 24 s (attente du verrou exclue), dont modèles 2 s et confidentialité différentielle 16 s ; premier lancement : + 3 s de téléchargement et 4 s de lecture du CSV ; smoke : 5 s (M1 Pro, CPU, 16/10/2026) |
| **Travail estimé** | 1 h 30 |
| **Seed** | 42 (`seed_everything(42)`, découpages `random_state=42`, bruits `default_rng(42)` et `default_rng(1000·k + 42)`) |
| **Données** | fichier « personnes » de l’enquête ACS 2018 sur un an, Californie (US Census Bureau, `csv_pca.zip`, 67 Mo), téléchargé une fois dans `data/folktables/` ; cache de 1,6 Mo `data/folktables/acsincome_ca_2018.parquet`. Conditions d’utilisation : https://www.census.gov/data/developers/about/terms-of-service.html |

## Objectif

Mesurer sur de vraies données ce que veulent dire « parité démographique », « égalité des chances » et « parité prédictive », corriger un classifieur par un seuil propre à chaque groupe, constater qu’on ne peut pas satisfaire tous les critères à la fois, puis entraîner le même modèle avec une garantie de confidentialité différentielle et voir ce qu’elle coûte.

Ce qui est optimisé : l’**entropie croisée binaire** d’une régression logistique. Aucune contrainte d’équité n’entre dans l’entraînement : l’équité est corrigée **après coup** par le choix des seuils (post-traitement). En partie 5, la même entropie croisée est minimisée par une descente de gradient écrêtée et bruitée.

## Ce que vous allez faire

1. **Les données.** La tâche ACSIncome de Folktables (Ding et al., 2021) reconstruite en quelques lignes de pandas : 10 variables, filtres hérités d’Adult (âge > 16, revenu > 100 $, au moins une heure travaillée, poids de sondage ≥ 1), cible « revenu > 50 000 $ ». 195 665 personnes, découpées 60 / 20 / 20 (apprentissage, validation, test), stratifiées sur (cible, sexe). Attribut sensible : `SEX` (1 homme, 2 femme ; binaire dans l’enquête 2018).
2. **Mesurer en numpy** (`taux_par_groupe`) : matrice de confusion de chaque groupe, puis taux de sélection, de vrais positifs (TVP), de faux positifs (TFP) et précision (VPP).
3. **Entraîner avec scikit-learn** deux régressions logistiques, avec et sans la colonne `SEX` (623 et 625 colonnes après encodage), et une troisième qui prédit le sexe sans cette colonne (proxys).
4. **Corriger en numpy** (`seuils_parite`, `seuils_egalite_chances`) : un seuil par groupe, la cible commune choisie sur la validation (meilleure exactitude, grille de 201 valeurs), tout mesuré sur le test.
5. **Vérifier l’incompatibilité** : identité de Chouldechova $\text{TFP} = \frac{p}{1-p}\cdot\frac{1-\text{VPP}}{\text{VPP}}\cdot\text{TVP}$ dans chaque groupe et chaque politique, tableau des écarts.
6. **Protéger** : (a) mécanisme de Laplace sur le nombre de personnes à plus de 50 000 $ par catégorie « race » du recensement (`RAC1P`), 20 000 tirages ; (b) `descente_privee` : régression logistique, gradient de chaque personne écrêté à la norme $C = 1$, somme bruitée par $\mathcal{N}(0, \sigma^2 I)$, 200 pas avec inertie 0,9 (post-traitement, gratuit en confidentialité), $\delta = 10^{-6}$ ; $\sigma$ déduit de $\varepsilon$ par la comptabilité zCDP de Bun et Steinke (2016) : un pas coûte $\rho_1 = (2C)^2/(2\sigma^2)$, les pas s’additionnent, $\rho$-zCDP ⇒ $(\rho + 2\sqrt{\rho\ln(1/\delta)}, \delta)$-DP. Six budgets $\varepsilon$ de 0,05 à 10, sur le jeu complet (117 399 personnes) et sur 2 000 personnes, 5 tirages du bruit chacun.

## Résultat attendu

Le run complet vérifie 12 seuils et s’arrête avec une erreur si l’un manque. Valeurs mesurées le 16/10/2026 (identiques d’un lancement à l’autre, hors durées : contrôlé à la relecture par deux lancements supplémentaires ; CPU, graine 42), sur les 39 133 personnes du test.

| Seuil vérifié | Mesuré |
|---|---|
| exactitude du modèle avec `SEX` ≥ 0,80 | **81,4 %** |
| écart de sélection femmes − hommes (seuil 0,5) ≤ −8 points | **−12,2 points** (34,8 % contre 47,0 %) |
| sans `SEX`, l’écart de sélection reste ≤ −5 points | **−7,9 points** (37,1 % contre 45,0 %) |
| le sexe se prédit sans la colonne `SEX` : AUC ≥ 0,75 | **AUC 0,820**, exactitude 74,0 % (classe majoritaire : 52,8 %) |
| parité démographique : écart de sélection ≤ 1 point | **−0,1 point** (41,6 % et 41,5 %) |
| égalité des chances : écart de TVP ≤ 1 point | **+0,2 point** (77,1 % et 77,3 %) |
| post-traitements : perte d’exactitude ≤ 2 points | parité **−0,6 point** (80,8 %), égalité des chances **−0,1 point** (81,3 %) |
| égalité des chances : écart de VPP ≥ 3 points | **−8,4 points** (80,9 % contre 72,5 %) |
| identité de Chouldechova à 10⁻⁹ près | écart maximal **8·10⁻¹⁷** |
| Laplace ε = 1 : erreur relative < 0,01 % sur le grand groupe, > 10 % sur le plus petit | **0,002 %** (53 650 sur 121 006) et **19,9 %** (5 sur 13) |
| DP ε = 1, jeu complet : moins d’un point de perte | **80,3 %** contre 81,3 % sans bruit |
| DP ε = 0,1, 2 000 personnes : au moins 5 points de perte | **49,6 %** contre 79,7 % sans bruit |

Taux de base : 46,6 % des hommes et 34,9 % des femmes dépassent 50 000 $.

Taux sur le test, hommes / femmes (écart femmes − hommes en points) :

| Politique | Exactitude | Sélection | TVP | TFP | VPP |
|---|---|---|---|---|---|
| seuil 0,5, avec `SEX` | 81,4 % | 47,0 / 34,8 (−12,2) | 79,8 / 74,0 (−5,8) | 18,3 / 13,7 (−4,6) | 79,1 / 74,3 (−4,8) |
| seuil 0,5, sans `SEX` | 81,3 % | 45,0 / 37,1 (−7,9) | 77,8 / 77,1 (−0,7) | 16,5 / 15,7 (−0,8) | 80,4 / 72,5 (−7,9) |
| parité démographique (seuils 0,598 / 0,388) | 80,8 % | 41,6 / 41,5 (−0,1) | 73,6 / 82,4 (+8,7) | 13,6 / 19,6 (+6,0) | 82,5 / 69,3 (−13,2) |
| égalité des chances (seuils 0,549 / 0,457) | 81,3 % | 44,4 / 37,2 (−7,2) | 77,1 / 77,3 (+0,2) | 15,9 / 15,8 (−0,1) | 80,9 / 72,5 (−8,4) |

Confidentialité différentielle (exactitude moyenne sur le test, 5 tirages du bruit ; sans bruit : 81,3 % sur le jeu complet, 79,7 % sur 2 000 personnes) :

| ε | 0,05 | 0,1 | 0,3 | 1 | 3 | 10 |
|---|---|---|---|---|---|---|
| σ (bruit sur la somme des gradients) | 2 976 | 1 490 | 498 | 151 | 52 | 17 |
| 117 399 personnes | 70,5 % | 75,5 % | 79,5 % | 80,3 % | 80,4 % | 80,4 % |
| 2 000 personnes | 47,1 % | 49,6 % | 55,4 % | 62,4 % | 69,7 % | 76,6 % |

Mécanisme de Laplace, erreur relative moyenne : ε = 0,1 → 0,02 % sur le groupe « White alone » (53 650 personnes à plus de 50 000 $), 198 % sur « Alaska Native alone » (5 personnes) ; ε = 1 → 0,002 % et 19,9 %.

Figures produites dans `figures/` : `equite.png` (quatre taux par groupe pour chaque politique), `confidentialite.png` (exactitude selon ε) ; résultats chiffrés dans `figures/resultats.json`, lus par les figures du site (`site/scripts/figures/ch40_*.py`).

## Ce qu’il faut observer

- **Exact en moyenne, inégal par groupe.** Le modèle a la même exactitude pour les deux groupes (80,8 % et 82,0 %) mais sélectionne 12 points de femmes en moins et retrouve moins bien celles qui dépassent 50 000 $ (TVP 74,0 % contre 79,8 %). Une partie de l’écart de sélection vient des données elles-mêmes (taux de base 34,9 % contre 46,6 %) : le modèle reproduit une inégalité réelle de revenus.
- **Retirer la colonne ne retire pas l’information.** Sans `SEX`, un autre modèle retrouve le sexe avec une AUC de 0,82 à partir de la profession, des heures travaillées, etc. L’écart de sélection baisse (−7,9 points) sans disparaître. Et il se déplace : l’égalité des chances est presque atteinte par hasard (−0,7 point), mais la précision chute chez les femmes (VPP 72,5 % contre 80,4 %) : un « oui » du modèle vaut moins pour une femme.
- **Un seuil par groupe suffit pour un critère.** Le post-traitement égalise la sélection ou le TVP à 0,2 point près, sans réentraîner, pour moins d’un point d’exactitude. Sur la figure ROC du chapitre, cela revient à choisir un point par courbe.
- **Mais jamais tous les critères.** La parité démographique *creuse* l’écart de VPP (−13,2 points) et inverse celui de TVP (+8,7). L’égalité des chances laisse 7 points d’écart de sélection et 8 points de VPP. L’identité de Chouldechova explique pourquoi : avec des taux de base différents, TVP et TFP égaux imposent des VPP différentes.
- **La confidentialité coûte peu sur un grand jeu, beaucoup sur un petit.** À ε = 1, le bruit nécessaire est le même pour 117 399 ou 2 000 personnes (σ = 151 sur la somme), mais il est divisé par $n$ : un point perdu dans le premier cas, 17 dans le second. Même effet avec Laplace : invisible sur 53 650 personnes, il écrase un groupe de 5. Les petites populations, souvent les minorités, paient le plus.
- **L’écrêtage seul a un prix.** À ε = 10, le bruit est négligeable et l’on plafonne pourtant à 80,4 % au lieu de 81,3 % : écrêter chaque gradient à $C = 1$ biaise la descente.

## Questions de réflexion

1. Dans quel contexte réel (crédit, embauche, détection de fraude, dépistage médical) choisiriez-vous la parité démographique plutôt que l’égalité des chances, et pourquoi ? Que suppose chaque critère sur la qualité des étiquettes $Y$ ?
2. Le post-traitement utilise le sexe **au moment de décider**. Pourquoi est-ce gênant juridiquement dans certains pays, et que se passe-t-il si l’on ne dispose que d’un modèle « sans `SEX` » et d’un seul seuil ?
3. Avec les chiffres du tableau, vérifiez à la main l’identité de Chouldechova pour les femmes sous l’égalité des chances ($p = 0{,}349$, VPP = 0,725, TVP = 0,773) : quel TFP obtenez-vous ?
4. Pourquoi la sensibilité est-elle $2C$ et non $C$ dans la comptabilité de la partie 5 ? Que changerait le sous-échantillonnage aléatoire de DP-SGD (lots tirés au hasard) au budget dépensé par pas ?

## Pour aller plus loin

- Égaliser *aussi* les TFP (chances égalisées) avec le tirage au sort entre deux seuils de l’article original, ou refaire l’exercice avec `RAC1P` comme attribut sensible (neuf groupes, dont plusieurs très petits).
- La bibliothèque Fairlearn (`ThresholdOptimizer`) implémente ce post-traitement ; Opacus implémente DP-SGD avec un comptable de Rényi. Aucune des deux n’est installée dans le projet.
- Articles : Hardt, Price et Srebro, *Equality of Opportunity in Supervised Learning*, NIPS 2016, https://arxiv.org/abs/1610.02413 ; Chouldechova, *Fair Prediction with Disparate Impact: A Study of Bias in Recidivism Prediction Instruments*, Big Data 5(2), 2017, https://doi.org/10.1089/big.2016.0047 ; Kleinberg, Mullainathan et Raghavan, *Inherent Trade-Offs in the Fair Determination of Risk Scores*, 2016, https://arxiv.org/abs/1609.05807 ; Ding, Hardt, Miller et Schmidt, *Retiring Adult: New Datasets for Fair Machine Learning*, 2021, https://arxiv.org/abs/2108.04884 (code Folktables, licence MIT : https://github.com/socialfoundations/folktables) ; Abadi et al., *Deep Learning with Differential Privacy*, ACM CCS 2016, https://arxiv.org/abs/1607.00133 ; Bun et Steinke, *Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds*, TCC 2016, https://arxiv.org/abs/1605.02065 (propositions 1.3 et 1.6, lemme 1.7).
- Livre : Barocas, Hardt et Narayanan, *Fairness and Machine Learning*, https://fairmlbook.org/.
