# Exercice 25a : diffusion (DDPM, DDIM, guidage) et flow matching sur MNIST

| | |
|---|---|
| **Chapitre lié** | [25. Modèles génératifs II : diffusion, flow matching, vidéo](https://laclasse.poneyponey.fr/ia/p07-transformer/25-generatifs-2-diffusion/) |
| **Exécution** | locale, MPS (NumPy pour la partie A, PyTorch pour B à E ; tourne aussi sur CPU, beaucoup plus lentement) |
| **Commande** | `make ex NUM=25a` (éditeur), `make ex NUM=25a MODE=run` (complet), `MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 8 min 16 s à 8 min 59 s selon la charge de la machine (trois runs, M1 Pro, MPS, 15/10/2026, attente du verrou exclue), dont 2 min 19 s pour entraîner le U-Net de diffusion, 2 min 17 s pour celui du flow matching, 14 à 21 s pour le juge, 1 min 52 s à 2 min 21 s pour les 1 000 images de DDPM en 1 000 pas ; smoke : 23 à 26 s |
| **Travail estimé** | ~3 h |
| **Seed** | 42 (`seed_everything(42)` avant chaque modèle) ; ordre des lots fixé par `torch.Generator().manual_seed(42)` ; bruits de départ des échantillons tirés avec la graine 2024, bruit de DDPM avec la graine 1. Trois runs complets ont donné exactement les mêmes pertes ; les distances de Fréchet varient au plus de 0,1 (calcul sur MPS, pas reproductible au bit près) |
| **Données** | MNIST via `ia_commun.donnees.mnist_tenseurs()`, cache `data/mnist/` (≈ 55 Mo, téléchargé au premier lancement) ; 60 000 images d'entraînement, 10 000 de test. Les deux U-Net sont sauvegardés dans `data/checkpoints/25a-unet-ddpm.pt` et `25a-unet-fm.pt` (non versionnés) |

## Objectif

Construire un modèle de diffusion complet sur de petites images : bruiter, apprendre à prédire le bruit, puis générer en remontant le temps ; mesurer ce que coûtent et valent les échantillonneurs (DDPM, DDIM), ce que fait le guidage sans classifieur, et ce que change le flow matching, avec le même réseau et le même juge chiffré.

## Ce que vous allez faire

1. **A. NumPy, sans apprentissage.**
   - A1. Calendrier de bruit de DDPM : βₜ linéaire de 10⁻⁴ à 0,02 sur T = 1 000 pas, ᾱₜ = ∏(1 − βₛ).
   - A2. Vérifier la formule fermée xₜ = √ᾱₜ·x₀ + √(1 − ᾱₜ)·ε contre 1 000 étapes de bruitage successives, sur 20 000 « pixels » valant ±1 : pente de xₜ sur x₀ et variance du bruit restant.
   - A3. Un nuage 2D de huit gaussiennes (σ = 0,1) sur un cercle de rayon 2. Bruité, il reste un mélange de gaussiennes : on calcule **exactement** le bruit moyen E[ε | xₜ] (ce qu'un réseau parfait prédirait) et la vitesse moyenne E[ε − x₀ | xₜ] du flow matching. On échantillonne 4 000 points par DDPM (1 000 pas), DDIM et flow matching (1 à 100 pas), depuis les mêmes bruits de départ, et on compte la part de points arrivés à moins de 3σ d'un centre.
2. **B. Un U-Net qui prédit le bruit.** U-Net 28 → 14 → 7 → 14 → 28 avec ponts (chapitre 14), blocs résiduels, GroupNorm, 32 puis 64 canaux, 0,33 million de paramètres. Il reçoit l'image bruitée, l'instant t (encodage sinusoïdal) et la classe du chiffre, remplacée par « aucune classe » dans 10 % des cas. 4 époques (1 872 pas), lots de 128, Adam, taux en « un cycle » jusqu'à 2·10⁻³. On mesure ensuite l'erreur de test selon le niveau de bruit.
3. **Le juge** : le même petit CNN et la même recette que l'exercice 18 (98,85 % de précision de test), dont l'avant-dernière couche sert à calculer une distance de Fréchet (principe de la FID) entre 1 000 images générées et les 10 000 images de test.
4. **C. Échantillonner.** On demande 100 images de chaque chiffre (condition seule, sans guidage), depuis les mêmes 1 000 bruits : DDPM en 1 000 pas (stochastique), puis DDIM en 100, 50, 20 et 10 pas (déterministe), avec le même réseau.
5. **D. Guidage sans classifieur.** ε̂ = ε̂_sans + w·(ε̂_avec − ε̂_sans), DDIM 50 pas, w = 0 (sans condition), 1 (condition seule), 3 et 7. On mesure la part d'images que le juge range dans la classe demandée et la **diversité** (distance moyenne entre les caractéristiques de deux images d'une même classe demandée).
6. **E. Flow matching.** Le même U-Net, le même nombre de pas d'entraînement, mais la cible est la vitesse ε − x₀ au point xₜ = (1 − t)·x₀ + t·ε, t tiré uniformément dans [0, 1]. On génère par la méthode d'Euler en 50, 20, 10 et 5 pas, et on compare à DDIM au même nombre de pas.
7. Les trajectoires (un « 3 » bruité pas à pas ; une génération DDIM de 50 pas) sont exportées pour la démo du site.

**Pertes optimisées** : diffusion, ‖ε − ε_θ(√ᾱₜ·x₀ + √(1 − ᾱₜ)·ε, t, y)‖² (« ε-prédiction », perte simplifiée de Ho et al., 2020) ; flow matching, ‖(ε − x₀) − v_θ((1 − t)·x₀ + t·ε, t, y)‖² ; juge, entropie croisée. Partie A : aucune (rien n'est appris).

## Résultat attendu

En mode complet, le notebook vérifie douze seuils et s'arrête avec une erreur si l'un d'eux n'est pas atteint. Valeurs du dernier run du 15/10/2026 (M1 Pro, MPS, seed 42), celles de `figures/resultats.json`.

**A. NumPy**

| Mesure | Obtenu |
|---|---|
| ᾱ_T (t = 1 000) | 4,04·10⁻⁵ (√ᾱ_T = 0,0064 : il reste 0,6 % de l'image) |
| ᾱₜ passe sous 0,5 (autant de variance d'image que de bruit) | t = 260 |
| √ᾱₜ passe sous 0,1 | t = 674 |
| Formule fermée contre 1 000 étapes, t = 100 / 300 / 1 000 : pente | 0,9490 / 0,6359 / 0,0036 (formule 0,9471 / 0,6296 / 0,0064) |
| Idem, variance du bruit restant | 0,1034 / 0,6099 / 1,0099 (formule 0,1030 / 0,6036 / 1,0000) |

Seuil : écarts inférieurs à 0,01.

Jouet 2D, débruiteur exact, 4 000 points, part arrivée à moins de 3σ d'un centre (vraies données : 98,7 % ; DDPM 1 000 pas : 99,0 %) :

| Pas | 1 | 2 | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| DDIM | 0 % | 0 % | 16,0 % | **77,2 %** | 97,2 % | 99,1 % | 99,1 % |
| Flow matching (Euler) | 0 % | 5,2 % | 88,9 % | **98,1 %** | 98,5 % | 98,8 % | 98,9 % |

En 1 pas, tous les points tombent au centre (distance moyenne à un centre 1,98 et 2,00 : la moyenne des données). Rapport longueur du chemin / distance entre départ et arrivée (1 = ligne droite), 30 trajectoires de 50 pas : DDIM 1,47, flow matching 1,93. Seuils : DDPM ≥ 95 % ; flow matching 10 pas ≥ 90 % et DDIM 10 pas ≤ 85 %.

**B. Entraînement** : perte de diffusion 0,024 en fin d'entraînement (moyenne des 200 derniers pas), perte de flow matching 0,169 (les deux cibles n'ont pas la même échelle : ne pas comparer ces deux nombres). Erreur de test selon le niveau de bruit (2 000 images) :

| t / T | 0,01 | 0,1 | 0,3 | 0,5 | 0,7 | 0,9 | 0,99 |
|---|---|---|---|---|---|---|---|
| diffusion : MSE sur ε | 0,147 | 0,064 | 0,030 | 0,012 | 0,0017 | 0,0003 | 0,0002 |
| flow matching : MSE sur ε − x₀ | 0,219 | 0,134 | 0,110 | 0,128 | 0,198 | 0,242 | 0,216 |

**C et E. Échantillonneurs** (1 000 images, 100 par chiffre demandé ; plancher : 1 000 vraies images d'entraînement contre le test donnent 6,2)

| Échantillonneur | Pas | Distance de Fréchet | Classe demandée obtenue | Durée (1 000 images) |
|---|---|---|---|---|
| DDPM | 1 000 | **9,8** | 89,7 % | 1 min 52 s |
| DDIM | 100 | 19,9 | 86,9 % | 11,1 s |
| DDIM | 50 | 20,2 | 87,0 % | 5,6 s |
| DDIM | 20 | 22,3 | 88,0 % | 2,3 s |
| DDIM | 10 | 28,6 | 89,5 % | 1,1 s |
| DDIM | 5 | 73,0 | 93,2 % | — |
| Flow matching | 50 | **11,0** | 94,5 % | — |
| Flow matching | 20 | 11,8 | 94,9 % | — |
| Flow matching | 10 | **14,3** | 95,5 % | — |
| Flow matching | 5 | 25,5 | 96,1 % | — |

Seuils : juge ≥ 98,5 % ; DDPM ≤ 15 ; DDIM 50 pas ≤ 30 ; DDIM 10 pas moins bon que 50 pas ; flow matching 10 pas ≤ 20 et meilleur que DDIM 10 pas ; flow matching 5 pas meilleur que DDIM 5 pas.

**D. Guidage sans classifieur** (DDIM 50 pas, 1 000 images)

| w | Classe demandée obtenue | Diversité | Distance de Fréchet | Confiance du juge |
|---|---|---|---|---|
| 0 (sans condition) | 9,5 % (le hasard) | 35,4 | 90,8 | 0,827 |
| 1 (condition seule) | 87,0 % | 25,9 | **20,2** | 0,940 |
| 3 | **99,8 %** | 18,1 | 55,6 | 0,998 |
| 7 | 100,0 % | 15,9 | 132,5 | 1,000 |

Seuils : w = 1 ≥ 80 % et w = 3 ≥ 97 % ; diversité décroissante de w = 1 à w = 7 ; Fréchet plus grande à w = 7 qu'à w = 1.

Figures produites dans `figures/` : `jouet-2d.png`, `pertes.png`, `echantillons.png`, `guidage.png`, `trajectoires.png` ; résultats bruts : `figures/resultats.json` et `figures/images.npz` (≈ 720 Ko, lus par les figures du site `site/scripts/figures/ch25_*.py` et par la démo `DiffusionSlider`).

## Ce qu'il faut observer

- **Le bruitage n'a rien d'appris.** La formule fermée donne en une opération ce que 1 000 petites étapes donnent (pente et variance identiques à 0,01 près). C'est ce qui rend l'entraînement simple : on tire un t au hasard et on fabrique xₜ directement, sans simuler la chaîne.
- **Tout le travail du réseau est d'approcher le débruiteur exact.** Dans le jouet 2D, sans aucun réseau, DDPM en 1 000 pas reproduit les huit nuages (99 % des points). En un seul pas, DDIM et flow matching envoient tout au centre : la meilleure estimation de x₀ quand on ne voit que du bruit est la moyenne des données, et la moyenne de huit nuages n'est aucun d'eux. Il faut plusieurs pas pour « choisir » un nuage.
- **Prédire le bruit est facile quand il y en a beaucoup, et inversement.** L'erreur de la diffusion passe de 0,147 à t = 0,01·T à 0,0002 à t = 0,99·T : quand xₜ est presque tout entier du bruit, le bruit, c'est xₜ. Pour le « 3 » de la démo, l'erreur va de 0,32 (t = 0) à 0,000 (t = 999). Le flow matching prédit une vitesse ε − x₀ : son erreur reste entre 0,11 et 0,24 à tous les niveaux, plus forte aux deux bouts.
- **DDPM est lent, DDIM échange un peu de qualité contre beaucoup de temps.** Même réseau, sans réentraînement : DDIM en 50 pas est 20 fois plus rapide que DDPM en 1 000 pas, pour une distance de Fréchet de 20,2 au lieu de 9,8. En dessous d'une dizaine de pas, la qualité s'effondre (73,0 à 5 pas).
- **Le flow matching fait mieux à nombre de pas égal**, ici à tous les nombres de pas : 14,3 en 10 pas, mieux que DDIM en 100 pas (19,9). Ce n'est pas que ses trajectoires soient droites : dans le jouet 2D, elles sont même plus courbes que celles de DDIM (1,93 contre 1,47). Seuls les chemins **conditionnels** (une image donnée, un bruit donné) sont droits ; la vitesse apprise en est la moyenne, et la moyenne de lignes droites qui se croisent n'est pas droite. Le flux rectifié (*reflow*) redresse ces trajectoires en réentraînant sur des couples (bruit, image générée).
- **Le guidage échange diversité contre fidélité.** Sans condition (w = 0), le réseau ne produit des chiffres que d'une qualité médiocre (Fréchet 90,8) : il n'a vu « aucune classe » que dans 10 % des pas d'entraînement. La condition seule (w = 1) donne le bon chiffre 87 fois sur 100 et la meilleure distance. À w = 3, le chiffre est juste à 99,8 % et le juge est sûr de lui (0,998), mais les images d'une même classe se ressemblent (diversité 25,9 → 18,1) et la distance aux vraies images remonte (55,6) ; à w = 7, des chiffres épais et stéréotypés (132,5). Un fort guidage donne des images « plus typiques que nature ».
- **La silhouette se décide tôt.** Dans la trajectoire DDIM, l'estimation x̂₀ montre la forme du chiffre dès t ≈ 700, alors que xₜ n'est encore que du bruit à l'œil ; les derniers pas ne font qu'affiner le trait.

## Questions de réflexion

1. Le notebook écrête x̂₀ dans [−1, 1] à chaque pas de DDIM, puis recalcule le bruit compatible avec cette estimation. Retirez la ligne qui recalcule `e` et relancez la partie D : que deviennent les images à w = 3 et 7 ? (Une première version de l'exercice avait ce défaut : 11,9 % et 0,2 % de classes justes à w = 3 et 7.)
2. ᾱₜ passe sous 0,5 à t = 260 et √ᾱₜ sous 0,1 à t = 674 : un tiers des 1 000 pas se passe dans un bruit presque total. Que proposerait un calendrier « cosinus » (Nichol et Dhariwal, 2021) ?
3. DDPM ajoute du bruit neuf à chaque pas, DDIM non : le même bruit de départ donne toujours la même image avec DDIM. À quoi sert cette propriété (pensez à l'interpolation entre deux bruits de départ) ?
4. Pourquoi la perte de diffusion (0,024) est-elle bien plus petite que celle du flow matching (0,169), alors que le second modèle génère de meilleures images ? Que mesure chacune ?

## Pour aller plus loin

- Partie C : remplacer `guidage=1.0` par `0.0` et comparer DDPM et DDIM sans condition.
- Partie E : ajouter le guidage au flow matching (`ech_fm(..., guidage=3.0)`) : même compromis qu'en D ?
- Entraîner 8 époques au lieu de 4 (≈ 4 min 40 s de plus) et mesurer ce que gagnent DDPM, DDIM et le flow matching.
- Articles : Ho, Jain et Abbeel (DDPM), [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) ; Song, Meng et Ermon (DDIM), [arXiv:2010.02502](https://arxiv.org/abs/2010.02502) ; Ho et Salimans (guidage sans classifieur), [arXiv:2207.12598](https://arxiv.org/abs/2207.12598) ; Lipman et al. (flow matching), [arXiv:2210.02747](https://arxiv.org/abs/2210.02747) ; Liu, Gong et Liu (flux rectifié), [arXiv:2209.03003](https://arxiv.org/abs/2209.03003).
