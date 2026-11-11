# 45 · Tirer, estimer, dériver une moyenne

| | |
|---|---|
| **Chapitre lié** | [45. Échantillonnage : Monte-Carlo, réparamétrisation, score, gradient de politique](https://laclasse.poneyponey.fr/maths/p7-probas-ia/45-echantillonnage/), sections 1 (Monte-Carlo), 2 (transformée inverse), 3 (échantillonnage préférentiel), 4 (mélanges), 5 et 6 (réparamétrisation, astuce du logarithme, REINFORCE) et 7 (Langevin) |
| **Commande** | `make nb NUM=45` (éditeur), `make nb NUM=45 MODE=run` (complet), `make nb NUM=45 MODE=smoke` (moins de 30 s) |
| **Durée mesurée** | run complet : 8 s, dont 3,3 s de calcul ; smoke : 2 s (machine de travail, 27/10/2026) |
| **Travail estimé** | 30 min |
| **Graine** | 42 (`seed_everything(42)`, variable `SEED` du Makefile), un seul générateur numpy pour tout le notebook |
| **Données** | aucune : tirages pseudo-aléatoires, rien n’est téléchargé |

## Objectif

Retrouver par la simulation les résultats du chapitre 45 : l’erreur de Monte-Carlo décroît comme $\frac{1}{\sqrt{n}}$ ; la transformée inverse produit la bonne loi ; un bon échantillonnage préférentiel divise la variance par des centaines ; un mélange n’est pas une combinaison linéaire ; la réparamétrisation et l’astuce du logarithme estiment le même gradient avec des variances différentes ; la ligne de base de REINFORCE réduit la variance sans changer la moyenne ; la dynamique de Langevin, avec son bruit, finit par suivre la densité visée.

## Ce que vous allez faire

1. Lire la cellule de configuration : elle seule change entre mode complet et mode smoke (nombre de fléchettes, de simulations, de tirages et de répétitions).
2. Lire la vérification sympy : $F^{-1}(u) = -m \ln(1 - u)$ pour l’exponentielle ; $\mathbb{E}[(U_1 + \dots + U_{10})^2] = \frac{155}{6} \approx 25{,}833$ et l’écart-type de cette variable, $\approx 9{,}200$ ; la variance du mélange, $\frac{109}{25} = 4{,}36$ ; $\mathbb{E}[2(\mu + \sigma\varepsilon)\varepsilon] = 2\sigma$ ; $\mathbb{E}\big[z^2 \frac{z - \mu}{\sigma^2}\big] = 2\mu$ ; le score $-\frac{x - \mu}{\sigma^2}$ de la loi normale.
3. Cliquer sur « Lancer le calcul » dans l’éditeur marimo, puis lire les trois graphiques.

## Résultat attendu

Run complet (seuils vérifiés aussi en smoke) :

- **π par fléchettes**, 100 simulations d’un million de fléchettes : la marge à 95 % $1{,}96 \times 1{,}642/\sqrt{n}$ contient l’erreur dans **95 %** des cas (seuil : au moins 85 %), et la pente de l’erreur moyenne en échelles logarithmiques vaut **−0,50** (seuil : entre −0,65 et −0,35).
- **Transformée inverse**, un million de tirages $-200 \ln(1 - U)$ : moyenne 199,8 (écart toléré 2) ; 99ᵉ centile estimé **920,9** pour une valeur exacte $-200 \ln 0{,}01 \approx 921{,}0$ (écart toléré 3 %) ; 1,00 % des tirages au-delà.
- **$P(Z > 3) \approx 0{,}00135$**, 1 000 estimations de 10 000 tirages : erreur relative 27,6 % en tirage direct, **1,8 %** avec $q = \mathcal{N}(3 \,;\, 1)$ ; variance divisée par **226** (valeur théorique ≈ 218, seuil : entre 120 et 400).
- **Mélange** $0{,}7\,\mathcal{N}(0 \,;\, 1) + 0{,}3\,\mathcal{N}(4 \,;\, 1)$ : moyenne 1,20, variance **4,35** (exacte 4,36) ; combinaison $0{,}7\,X_1 + 0{,}3\,X_2$ : moyenne 1,20, variance **0,58**.
- **Gradient de $\mathbb{E}[z^2]$ en $\mu$**, $z \sim \mathcal{N}(1 \,;\, 2^2)$, valeur exacte 2 : la réparamétrisation donne 2,00 avec une variance par tirage de **16** ($4\sigma^2$) ; l’astuce du logarithme donne aussi 2,00, mais avec une variance de **74,4** (exacte 74,25) ; avec une ligne de base égale à $\mathbb{E}[z^2] = 5$, elle descend à 48,1.
- **Bandit à deux bras** en $\theta = 0$ : l’estimateur REINFORCE a pour moyenne **0,25** avec ou sans ligne de base, mais une variance de 0,0625 sans ligne de base et **0** avec $b = 0{,}5$. En 2 000 pas ($\eta = 0{,}1$), la probabilité de jouer le bon bras passe de 0,5 à **0,993**.
- **Langevin**, 2 000 particules parties de $x = 4$, $\eta = 0{,}1$ : part des particules à gauche de 0 après 100, 500 et 1 000 pas : 0,17 ; 0,58 ; **0,67** (seuils : moins de 0,25, plus de 0,5, et à 0,05 de 0,7) ; sans bruit, toutes se figent au sommet de droite, en $x = 2$.

Le notebook s’arrête par `SystemExit` si un seuil n’est pas atteint. Figure : `figures/monte-carlo.png`.

## Ce qu’il faut observer

- Sur le graphique de π, les erreurs des simulations zigzaguent, mais restent presque toutes sous la droite de la marge, de pente −1/2 : cent fois plus de fléchettes pour un chiffre de plus.
- L’histogramme des temps tirés épouse la densité exponentielle : la transformée inverse ne fait qu’appliquer $F^{-1}$ à des uniformes.
- L’histogramme des particules de Langevin a deux bosses, dans les proportions du mélange ; sans le bruit, toutes les particules seraient restées dans la petite bosse de droite, la plus proche du départ.

## Questions de réflexion

1. Dans l’expérience 3, remplacez $q = \mathcal{N}(3 \,;\, 1)$ par $q = \mathcal{N}(1 \,;\, 1)$, puis par $q = \mathcal{N}(5 \,;\, 1)$ : la variance baisse-t-elle encore ? Pourquoi un $q$ trop décalé finit-il par nuire ?
2. Pourquoi la réparamétrisation a-t-elle ici une variance plus petite que l’astuce du logarithme ? Quelle information sur $f$ utilise-t-elle que l’autre méthode ignore ?
3. Passez le pas de Langevin de 0,1 à 0,01 : faut-il plus ou moins de pas pour que 70 % des particules rejoignent la grosse bosse ? Et que se passe-t-il avec un pas de 1 ?
