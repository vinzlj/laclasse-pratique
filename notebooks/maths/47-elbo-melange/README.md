# 47 · Jensen, information mutuelle et ELBO, par le calcul

| | |
|---|---|
| **Chapitre lié** | [47. Information II : inégalité de Jensen, information mutuelle, ELBO](https://laclasse.poneyponey.fr/maths/p7-probas-ia/47-information-elbo/), sections 2 (Jensen), 4 (information mutuelle), 5 et 6 (ELBO, EM, VAE) et 7 (distribution de Gibbs) |
| **Commande** | `make nb NUM=47` (éditeur), `make nb NUM=47 MODE=run` (complet), `make nb NUM=47 MODE=smoke` (moins de 30 s) |
| **Durée mesurée** | run complet : 3 s, dont 0,3 s de calcul ; smoke : 2 s (machine de travail, 27/10/2026) |
| **Travail estimé** | 25 min |
| **Graine** | 42 (`seed_everything(42)`, variable `SEED` du Makefile) ; les 300 points du mélange sont tirés par un générateur à part (graine 47), identiques dans les deux modes |
| **Données** | aucune : tirages pseudo-aléatoires, rien n’est téléchargé |

## Objectif

Retrouver par le calcul les résultats du chapitre 47 : la moyenne des logarithmes est sous le logarithme de la moyenne (Jensen) ; l’information mutuelle du tableau $(0{,}4 \,;\, 0{,}1 \,;\, 0{,}1 \,;\, 0{,}4)$ vaut 0,278 bit, et son estimation sur des tirages s’en approche ; pendant l’algorithme EM, $\ln p$ ne baisse jamais et l’ELBO la touche après chaque étape E ; la formule fermée de la KL d’un encodeur gaussien est juste ; la distribution de Gibbs maximise « récompense − β·KL », avec la valeur $\beta \ln Z$.

## Ce que vous allez faire

1. Lire la cellule de configuration : elle seule change entre mode complet et mode smoke (nombre de tirages, de répétitions, d’itérations d’EM, de politiques tirées au hasard).
2. Lire la vérification sympy : la KL entre $\mathcal{N}(\mu_1 \,;\, \sigma^2)$ et $\mathcal{N}(\mu_2 \,;\, \sigma^2)$ vaut $\frac{(\mu_1 - \mu_2)^2}{2\sigma^2}$ ; la formule générale redonne celle du VAE ; $J(\pi^*) = \ln Z$ sur l’exemple de l’ouverture.
3. Cliquer sur « Lancer le calcul » dans l’éditeur marimo, puis lire les trois graphiques.

## Résultat attendu

Run complet (seuils vérifiés aussi en smoke) :

- Jensen, $X$ uniforme sur $[1 \,;\, 4]$, 500 000 tirages : $\ln \mathbb{E}[X] - \mathbb{E}[\ln X] \approx$ **0,0679** nat, pour une valeur exacte $\ln 2{,}5 - \frac{4 \ln 4 - 3}{3} \approx 0{,}0679$ (écart toléré 0,005).
- Information mutuelle exacte **0,2781** bit ; estimée sur 100 000 tirages : 0,278 (écart toléré 0,005). Pour le tableau indépendant $(0{,}12 \,;\, 0{,}18 \,;\, 0{,}28 \,;\, 0{,}42)$, l’estimation est positive mais petite (0,007 bit sur 100 tirages, moins de 0,001 au-delà de 10 000) : l’estimateur « par les fréquences » est biaisé vers le haut.
- EM, 300 points, 100 itérations : $\ln p$ passe de **−633,6** à **−585,2** nats sans jamais baisser ; après chaque étape E, l’écart entre $\ln p$ et l’ELBO est au plus **10⁻⁹** (mesuré : 1,1 × 10⁻¹³, l’arrondi des flottants).
- KL contre $\mathcal{N}(0 \,;\, 1)$ : 0 ; 0,5 ; **0,3181** ; **2,7490** nats pour $(\mu \,;\, \sigma) = (0 \,;\, 1)$, $(1 \,;\, 1)$, $(0 \,;\, 0{,}5)$, $(2 \,;\, 0{,}3)$, par la formule et par la somme de Riemann (écart toléré 10⁻⁴).
- Gibbs, $\pi_{\mathrm{ref}} = (0{,}5 \,;\, 0{,}3 \,;\, 0{,}2)$, $r = (0 \,;\, 1 \,;\, 2)$ : pour $\beta = 1$, $\pi^* \approx (0{,}179 \,;\, 0{,}292 \,;\, 0{,}529)$ et $\beta \ln Z \approx$ **1,0272** ; pour chaque $\beta$ testé, $J(\pi^*) = \beta \ln Z$ et aucune des 20 000 politiques tirées au hasard ne fait mieux.

Le notebook s’arrête par `SystemExit` si un seuil n’est pas atteint. Figure : `figures/elbo-melange.png`.

## Ce qu’il faut observer

- Sur le graphique d’EM, l’ELBO part loin sous $\ln p$ (la loi $q$ de départ est uniforme), la rejoint à l’étape E, puis les deux montent ensemble à l’étape M, avec un petit écart qui réapparaît.
- L’information mutuelle estimée d’un tableau indépendant n’est jamais négative et pas tout à fait nulle : avec peu de tirages, le hasard crée une dépendance apparente.
- Quand $\beta$ diminue, la politique de Gibbs se concentre sur la réponse de récompense 2 ; quand $\beta$ grandit, elle revient vers $\pi_{\mathrm{ref}}$.

## Questions de réflexion

1. Remplacez $X$ uniforme sur $[1 \,;\, 4]$ par $X$ uniforme sur $[2 \,;\, 3]$ : l’écart de Jensen augmente-t-il ou diminue-t-il ? Pourquoi ?
2. Lancez EM avec une autre valeur de départ (par exemple les deux moyennes à 0) : atteint-on la même valeur finale de $\ln p$ ?
3. Pour $\beta = 0{,}1$, $\pi^*$ met presque toute sa masse sur une seule réponse : pourquoi des politiques tirées au hasard (loi de Dirichlet uniforme) tombent-elles rarement aussi près d’un sommet ?
