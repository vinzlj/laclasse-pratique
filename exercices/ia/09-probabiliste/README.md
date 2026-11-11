# Exercice 09 : réseau bayésien, EM, HMM, MCMC et inférence variationnelle en numpy

| | |
|---|---|
| **Chapitre lié** | [09. Modèles probabilistes et variables latentes](https://laclasse.poneyponey.fr/ia/p04-statistique/09-modeles-probabilistes/) |
| **Exécution** | locale, CPU, numpy (aucun GPU) |
| **Commande** | `make ex NUM=09` (éditeur), `make ex NUM=09 MODE=run` (complet), `make ex NUM=09 MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 20 s, dont 13,5 s pour Baum-Welch (3 × 200 itérations sur 5 000 jours) ; smoke : 5 s (M1 Pro, CPU, mesures du 14/10/2026 confirmées le 15/10/2026) |
| **Travail estimé** | 1 h 30 |
| **Seed** | 42 (`seed_everything(42)` ; chaque partie tire ses données avec `np.random.default_rng(42 + k)`) |
| **Données** | toutes simulées dans le notebook ; rien n'est téléchargé |

## Objectif

Programmer les algorithmes qui raisonnent avec des **variables cachées** (inférence exacte dans un réseau bayésien, EM, forward, Viterbi, Baum-Welch, Metropolis-Hastings, inférence variationnelle) et vérifier chiffres à l'appui la propriété centrale de chacun, notamment le lien entre EM, log-vraisemblance et **ELBO**.

Ce qui est optimisé : la log-vraisemblance $\log p(x \mid \theta)$ (parties 2 et 3, via l'ELBO) et l'ELBO d'une loi approchée $q$ (partie 5). Parties 1 et 4 : aucune perte, on calcule ou on échantillonne une loi.

## Ce que vous allez faire

1. **Réseau bayésien** pluie → arroseur, (pluie, arroseur) → herbe mouillée (valeurs du schéma `File:SimpleBayesNet.svg` de Wikimedia Commons) :
   - inférence exacte par énumération de la loi jointe ;
   - la même chose dans une **variante sans flèche pluie → arroseur** (arroseur à 0,4 qu'il pleuve ou non), qui isole l'effet d'explication ;
   - échantillonnage ancestral avec rejet (100 000 tirages), comparé à l'exact.
2. **Mélange de 3 gaussiennes et EM** (900 points : deux bandes allongées parallèles, un amas rond) :
   - EM from scratch (covariances pleines), 10 initialisations aléatoires ;
   - à chaque itération : log-vraisemblance, ELBO après l'étape M, KL calculée directement ;
   - comparaison avec `sklearn.mixture.GaussianMixture` et `sklearn.cluster.KMeans` ;
   - dans l'éditeur, un **curseur** rejoue EM itération par itération.
3. **HMM météo** (états cachés soleil / nuages / pluie ; observations vélo / transports / télétravail) :
   - forward et Viterbi contrôlés par énumération des $3^8 = 6\,561$ chemins sur 8 jours ;
   - décodage d'une année (365 jours) : jour par jour sans dynamique, Viterbi, forward-backward ;
   - Baum-Welch : réapprendre $A$ et $B$ à partir des seules observations (5 000 jours, 3 initialisations). Dans l'éditeur, cette étape (≈ 15 s) attend un clic sur le bouton « Lancer Baum-Welch » ; en script, elle part directement.
4. **Metropolis-Hastings** sur une densité 2D à deux bosses (poids 0,3 et 0,7), départ en (−6 ; 6), 30 000 pas, trois tailles de pas (0,1 ; 1,5 ; 15) ; taux d'acceptation, taille d'échantillon effective (ESS, estimateur de Geyer), part du temps passée dans chaque bosse.
5. **Régression logistique bayésienne** (2 paramètres, 40 points, a priori $\mathcal{N}(0, 3^2 I)$) :
   - loi a posteriori exacte sur une grille 400 × 400, qui donne aussi l'évidence $\log p(\mathcal{D})$ ;
   - Metropolis-Hastings (30 000 pas) ;
   - inférence variationnelle champ moyen : gaussienne diagonale, gradient reparamétré (16 tirages par pas), Adam écrit en numpy, 3 000 pas.

## Résultat attendu

Le run complet s'arrête avec une erreur si l'un des 22 contrôles échoue. Valeurs mesurées (déterministes à graine fixée) :

| Contrôle | Seuil | Mesuré |
|---|---|---|
| Réseau de la figure | $P(\text{pluie} \mid h, a) < P(\text{pluie}) < P(\text{pluie} \mid h)$ et $P(\text{pluie} \mid a) < 0{,}01$ | 0,0068 < 0,2 < 0,3577 ; 0,0062 |
| Variante indépendante : effet d'explication | $P(\text{pluie} \mid a) = P(\text{pluie}) < P(\text{pluie} \mid h, a) < P(\text{pluie} \mid h)$ | 0,2 = 0,2 < 0,2157 < 0,3782 |
| Rejet contre exact, $P(\text{pluie} \mid h)$ | écart < 0,01 | 0,3547 contre 0,3577 (44,8 % des tirages gardés) |
| EM : log-vraisemblance | jamais en baisse | oui, −4 817,1 → −2 870,2 en 114 itérations |
| EM : ELBO ≤ log-vraisemblance et log-vraisemblance = ELBO + KL | écart < 10⁻⁶ | écart max 9·10⁻¹³ |
| EM numpy contre scikit-learn (log-vraisemblance moyenne par point) | écart < 10⁻³ | −3,18913 et −3,18913 |
| Indice de Rand ajusté, EM | **≥ 0,95** | 0,997 (scikit-learn : 0,997) |
| Indice de Rand ajusté, k-means | ≤ 0,8 | 0,499 |
| HMM : forward contre énumération (8 jours) | écart < 10⁻⁹ | −8,973231 des deux côtés |
| HMM : Viterbi contre énumération | même probabilité | oui (même chemin) |
| Viterbi contre décodage jour par jour (365 jours) | + 5 points au moins | 77,8 % contre 69,9 % |
| Baum-Welch : log-vraisemblance | jamais en baisse | oui |
| Baum-Welch : $A$ et $B$ retrouvés | **écart max ≤ 0,1** | 0,074 sur $A$, 0,055 sur $B$ |
| MH, pas 1,5 : erreur sur la moyenne | < 0,15 | 0,02 |
| MH, pas 1,5 : acceptation | entre 20 et 60 % | 28,2 % |
| MH, pas 0,1 | acceptation > 90 %, ESS ≤ 1/10 de celui du pas 1,5 | 92,4 %, ESS 9 contre 424 |
| MH, pas 15 : acceptation | < 5 % | 0,8 % |
| Grille : masse de probabilité sur le bord | < 10⁻⁴ | 4,9·10⁻⁶ |
| MH a posteriori : moyenne | < 0,1 écart-type de l'exacte | (−2,751 ; 3,642) contre (−2,803 ; 3,700) |
| VI : ELBO sous $\log p(\mathcal{D})$ | au-delà de 3 erreurs types | −14,463 ± 0,003 contre −13,603 : KL = 0,86 nat |
| VI : moyenne | < 0,25 écart-type de l'exacte | (−2,861 ; 3,548) |
| VI champ moyen : écarts-types | < 90 % des exacts | (0,52 ; 0,47) contre (1,12 ; 1,11) |

Autres valeurs mesurées :

| Grandeur | Valeur |
|---|---|
| $P(\text{herbe})$ ; $P(\text{arroseur} \mid h)$ ; $P(\text{pluie} \mid h, \neg a)$ | 0,4484 ; 0,6467 ; 1,0 |
| EM : log-vraisemblances finales des 10 initialisations | 9 fois −2 870,2, une fois −3 189,0 (optimum local) |
| HMM, 365 jours : forward-backward (état le plus probable chaque jour) | 78,1 % |
| Baum-Welch : log p(obs) appris / vrai modèle | −5 158,3 / −5 163,5 |
| MH pas 0,1 / 1,5 / 15 : part du temps dans la bosse de droite (vrai 70 %) | 84,0 % / 70,1 % / 71,8 % |
| MH pas 0,1 / 1,5 / 15 : moyenne estimée (vrai (0,8 ; 0,4)) | (1,31 ; 0,69) / (0,78 ; 0,40) / (0,87 ; 0,43) |
| Corrélation a posteriori de $(w_0, w_1)$ | −0,88 |
| MH a posteriori : acceptation, ESS | 43,9 %, 745 |

Figures produites dans `figures/` : `em.png` (instantanés d'EM et courbes), `kmeans-em.png`, `hmm.png` (vraie météo, Viterbi, probabilités a posteriori ; Baum-Welch), `mcmc.png` (trois chaînes), `vi.png` (exacte, MCMC, VI ; courbe d'ELBO).

## Ce qu'il faut observer

- **L'effet d'explication.** Savoir l'herbe mouillée fait monter la pluie de 20 % à 36 % ; apprendre en plus que l'arroseur tournait la fait tomber à 0,7 %. Mais dans ce réseau, l'arroseur dépend déjà de la pluie : sans regarder l'herbe, $P(\text{pluie} \mid a)$ vaut 0,6 %, et la chute vient surtout de cette flèche. La variante à causes indépendantes montre l'effet d'explication pur : 20 % a priori (et toujours 20 % si l'on sait seulement que l'arroseur a tourné), 37,8 % sachant l'herbe mouillée, 21,6 % sachant en plus l'arroseur. Deux causes indépendantes d'un même effet deviennent dépendantes dès qu'on observe l'effet.
- **Le rejet gaspille.** 55 % des tirages sont jetés parce que l'herbe y est sèche. Plus l'évidence est rare, plus le gaspillage est grand : c'est la motivation des méthodes MCMC.
- **EM ne baisse jamais, mais peut stagner.** La courbe de log-vraisemblance monte toujours (garantie théorique vérifiée), et l'ELBO mesurée après l'étape M reste dessous. Avec la meilleure initialisation, la courbe reste presque plate pendant environ 85 itérations avant la remontée finale. Une initialisation sur dix finit dans un optimum local (−3 189 contre −2 870) : on relance EM plusieurs fois et on garde la meilleure.
- **k-means contre EM.** k-means suppose des groupes ronds de même taille : il coupe les bandes en travers (indice 0,50). Le mélange à covariances pleines les retrouve (0,997).
- **La dynamique aide.** Deviner la météo jour par jour d'après la seule observation donne 69,9 % ; Viterbi, qui tient compte de la persistance de la météo, 77,8 %. Le treillis évite d'énumérer $3^{365}$ chemins.
- **Les étiquettes apprises n'ont pas de nom.** Baum-Welch retrouve $A$ et $B$ à 0,07 près, mais il faut chercher quelle permutation des états appris correspond à « soleil », « nuages », « pluie ». Le modèle appris a une vraisemblance un peu **plus haute** que le vrai : il colle aussi au bruit de cet échantillon.
- **La taille du pas de MH.** Pas trop petit : 92 % d'acceptation mais la chaîne avance à petits pas et ne change de bosse que très rarement (elle reste à gauche pendant au moins ses 2 000 premiers pas), d'où 84 % du temps à droite au lieu de 70 % et une moyenne fausse ; 27 000 pas ne valent que 9 tirages indépendants. Pas trop grand : 99 % de refus. La théorie (Roberts, Gelman et Gilks, 1997) recommande environ 23 % d'acceptation en grande dimension, davantage en petite dimension (environ 44 % en dimension 1) ; ici, le bon pas accepte 28 %. Un taux d'acceptation élevé n'est pas un signe de qualité.
- **VI contre MCMC.** La gaussienne champ moyen est bien centrée mais deux fois trop étroite : elle ne peut pas représenter la corrélation −0,88 et la KL(q ‖ p) qu'elle minimise pénalise surtout la masse que q met là où p est faible. L'ELBO plafonne 0,86 nat sous l'évidence ; c'est exactement cette KL. MCMC, lui, retrouve moyenne et écarts-types à quelques pour cent près, mais au prix de 30 000 évaluations corrélées.

## Questions de réflexion

1. Dans le réseau bayésien, pourquoi $P(\text{pluie} \mid h, \neg a)$ vaut-il exactement 1 ? Quelle valeur de la table faut-il changer pour que ce ne soit plus le cas, et que devient alors l'effet d'explication ?
2. EM garantit que la log-vraisemblance ne baisse jamais. Pourquoi cela ne garantit-il pas de trouver le maximum global ? Que faudrait-il observer pour décider combien d'initialisations lancer ?
3. Sur 365 jours, forward-backward (état le plus probable chaque jour) fait un peu mieux que Viterbi en nombre de jours justes. Pourquoi ? Dans quel cas préférer quand même Viterbi (pensez à un chemin impossible, une transition de probabilité nulle) ?
4. Remplacez le champ moyen par une gaussienne à covariance pleine (paramétrée par une matrice triangulaire $L$, $w = \mu + L\varepsilon$). Que devient l'écart entre l'ELBO et $\log p(\mathcal{D})$ ? Pourquoi ne peut-il pas devenir nul ici ?

## Pour aller plus loin

- Ajouter une proposition non symétrique à Metropolis-Hastings (par exemple une marche dans l'espace des logarithmes) et vérifier qu'oublier la correction de Hastings biaise la moyenne.
- Échantillonnage de Gibbs sur la gaussienne a posteriori corrélée : comparer l'ESS à celui de MH quand la corrélation passe de 0 à 0,99.
- Remplacer le HMM par un filtre de Kalman (états et observations continus) : même forward, avec des gaussiennes au lieu de tables.
