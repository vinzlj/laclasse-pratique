# 00. Apprendre pour une machine : les fondations

Repère historique (1805 (Legendre), 1809 (Gauss), 1847 (Cauchy)) : Tirer d'observations bruitées les paramètres d'un modèle : choisir ceux qui minimisent une somme d'erreurs, puis savoir les calculer quand la formule exacte devient impraticable.

Au début du XIXe siècle, les astronomes ont plus d'observations que d'inconnues : une orbite de comète se décrit par quelques éléments, mais on la mesure bien plus souvent, et chaque mesure est un peu fausse. Aucun jeu de paramètres ne satisfait toutes les équations à la fois. Legendre publie en 1805 une règle simple : « rendre minimum la somme des quarrés des erreurs ». Gauss la justifie en 1809 par une loi de probabilité des erreurs, et affirme s'en servir depuis 1795. En 1847, Cauchy, face à des systèmes d'équations (non linéaires) que l'élimination ne permet pas de résoudre, propose de faire décroître une fonction pas à pas en suivant l'opposé de ses dérivées : c'est la descente de gradient. Deux siècles plus tard, entraîner un réseau de plusieurs milliards de paramètres repose encore sur ces deux idées.

## L'idée en une image

Vous voulez prédire le temps de chargement d'une page web à partir de son poids. Vous avez mesuré 40 pages. Plutôt que d'écrire la règle vous-même, vous choisissez une famille de règles (« le temps est une droite du poids ») et vous laissez la machine choisir la meilleure droite de la famille, celle qui se trompe le moins sur vos 40 mesures.

C'est tout. Apprendre, pour une machine, c'est :

1. des données : des exemples (entrée $x$, réponse $y$) ;
2. un modèle : une fonction $f_\theta(x)$ dont les paramètres $\theta$ sont des boutons réglables ;
3. une fonction de perte : un nombre qui dit à quel point le modèle se trompe sur les données ;
4. une optimisation : une procédure qui tourne les boutons pour faire baisser ce nombre.

La boucle tourne jusqu'à ce que la perte ne baisse plus. Une question reste, et c'est la plus importante : la règle trouvée marche-t-elle sur des pages qu'on n'a pas mesurées ? Une règle qui récite les 40 exemples par cœur n'a rien appris. Tout le chapitre tourne autour de ces deux temps : minimiser l'erreur sur les données, puis vérifier que le résultat se généralise.

| | Avant : programmer la règle | Après : apprendre la règle |
|---|---|---|
| Qui écrit la règle | le développeur, à la main (`if poids > 2 Mo…`) | l'optimisation, à partir d'exemples |
| Ce qu'on fournit | la logique | une famille de modèles, une perte, des données |
| Quand la réalité change | on réécrit le code | on réentraîne sur de nouvelles données |
| Comment on sait que ça marche | tests unitaires sur des cas prévus | erreur mesurée sur des données jamais vues |
| Ce qui coince | les cas qu'on n'a pas prévus | le bruit qu'on apprend par cœur (surapprentissage) |

## Le mécanisme

Le chapitre suit la régression linéaire du début à la fin, puis la complique juste assez pour voir apparaître le surapprentissage. Les nombres cités viennent de l'exercice du chapitre (seed 42) et de la démo, qui utilisent les mêmes données.

### Données, modèle, perte : la régression linéaire

Les 40 pages ont un poids $x$ tiré entre 0 et 2 Mo. Le « vrai » temps de chargement, que la machine ne connaît pas, vaut $y = 0{,}8 + 1{,}5\,x$ secondes, plus un bruit aléatoire d'écart-type 0,3 s (réseau, cache, serveur). Le modèle est une droite :

(Formule : Régression linéaire (une entrée).)

C'est la régression linéaire. Avec $d$ entrées (poids, nombre de requêtes, latence…), elle devient $\hat y = b + w_1 x_1 + \dots + w_d x_d$.

Pour mesurer l'erreur, on prend l'erreur quadratique moyenne : la moyenne des carrés des résidus $\hat y_i - y_i$.

(Formule : Perte MSE.)

Point clé : une fois les données fixées, la perte est une fonction des paramètres. Apprendre, c'est chercher $\theta^\star = \argmin_\theta \Loss(\theta)$.

Exemple à la main. Trois pages : $(x, y) = (0, 1), (1, 3), (2, 4)$.

- Avec $\theta = (0, 0)$ (on prédit toujours 0) : résidus $-1, -3, -4$, donc $\Loss = (1 + 9 + 16)/3 \approx 8{,}67$.
- La meilleure droite est $\hat y = 1{,}17 + 1{,}5\,x$ (calcul ci-dessous) : prédictions 1,17 ; 2,67 ; 4,17, résidus 0,17 ; −0,33 ; 0,17, et $\Loss = 0{,}056$. Aucune droite ne fait mieux, mais aucune n'atteint zéro : les trois points ne sont pas alignés.

#### Résoudre exactement : les équations normales

Pour la MSE d'un modèle linéaire, le minimum se calcule directement. On empile les exemples dans une matrice de conception $X$ de taille $N \times 2$ (une colonne de 1 pour $b$, une colonne des $x_i$ pour $w$) ; les prédictions s'écrivent alors $X\theta$ et la perte $\Loss(\theta) = \frac{1}{N}\lVert X\theta - y\rVert^2$. Au minimum, le gradient (section « Descente de gradient » plus bas) s'annule :

(Formule : Équations normales (Legendre, 1805).)

C'est la méthode des moindres carrés. Sur l'exemple à trois points, $X^\top X = \begin{pmatrix} 3 & 3 \\ 3 & 5 \end{pmatrix}$ et $X^\top y = \begin{pmatrix} 8 \\ 11 \end{pmatrix}$ ; la solution du système est $b = 7/6 \approx 1{,}17$, $w = 1{,}5$.

Sur les 40 pages de l'exercice, la formule fermée donne $b = 0{,}898$ et $w = 1{,}412$ (la vérité était 0,8 et 1,5 : le bruit et le petit nombre de points expliquent l'écart), pour une MSE de 0,044. scikit-learn (`LinearRegression`) trouve la même droite à $10^{-15}$ près.

```python
import numpy as np

rng = np.random.default_rng(42)
x = rng.uniform(0, 2, 40)                       # poids des pages (Mo)
y = 0.8 + 1.5 * x + rng.normal(0, 0.3, 40)      # temps mesurés (s)

X = np.column_stack([np.ones_like(x), x])       # matrice de conception, forme (40, 2)
theta = np.linalg.solve(X.T @ X, X.T @ y)       # équations normales (sans inverser X^T X)
mse = np.mean((X @ theta - y)  2)
print(theta, mse)                               # [0.898 1.412] 0.0442
```

#### Ce que la méthode optimise

La régression linéaire minimise la MSE. Ce choix n'est pas neutre : élever au carré pénalise fortement les grosses erreurs, donc un seul point aberrant tire toute la droite. Pourquoi le carré plutôt que la valeur absolue, et ce que change ce choix, c'est l'objet du chapitre 01, qui compare toutes les pertes.

### Algèbre linéaire minimale : vecteurs, matrices, dimensions

Tout le cours manipule des tableaux de nombres. Il suffit de quatre opérations et d'un réflexe : vérifier les dimensions.

| Objet | Notation | Forme (`shape`) | Exemple |
|---|---|---|---|
| scalaire | $a$ | `()` | la perte $\Loss = 0{,}044$ |
| vecteur | $x \in \R^d$ | `(d,)` | les paramètres $\theta = (b, w)$, `(2,)` |
| matrice | $X \in \R^{N \times d}$ | `(N, d)` | la matrice de conception, `(40, 2)` |
| tenseur | — | `(B, C, H, W)` | un lot de 32 images RVB 224 × 224 : `(32, 3, 224, 224)` |

- Produit scalaire : $u^\top v = \sum_j u_j v_j$. C'est une somme pondérée : $\hat y = \theta^\top (1, x) = b + w\,x$. Il mesure aussi à quel point deux vecteurs « pointent dans le même sens », idée reprise par les plongements (chapitre 15).
- Produit matrice-vecteur : $X\theta$ calcule les 40 prédictions d'un coup. Formes : `(40, 2) @ (2,) → (40,)`.
- Produit de matrices : `(N, d) @ (d, k) → (N, k)`. La dimension intérieure $d$ doit être la même des deux côtés et disparaît. Une couche de réseau qui prend 32 images MNIST aplaties (784 pixels) et sort 10 scores fait `(32, 784) @ (784, 10) → (32, 10)` : la matrice de poids a $784 \times 10 = 7\,840$ paramètres.
- Transposée : $X^\top$ échange lignes et colonnes, `(40, 2) → (2, 40)`. Dans $X^\top X$, `(2, 40) @ (40, 2) → (2, 2)` : une petite matrice qui résume les 40 exemples.

Quand un calcul « ne passe pas », c'est presque toujours une forme mal alignée. Piège spécifique à NumPy et PyTorch : la diffusion (*broadcasting*) étend silencieusement les axes de taille 1. `y_pred` de forme `(40, 1)` moins `y` de forme `(40,)` donne une matrice `(40, 40)`, sans erreur, et une MSE fausse.

### Probabilités : trois lois qui reviennent partout

Un modèle ne donne pas toujours un nombre : souvent, il donne une loi de probabilité. Trois lois suffisent pour l'essentiel du cours.

- Bernoulli, pour un résultat oui/non. Un paramètre $p \in [0, 1]$ : $P(y{=}1) = p$, $P(y{=}0) = 1 - p$. Spam ou non, clic ou non (régression logistique, ch. 01).
- Catégorielle, pour une classe parmi $K$. Paramètres $p_1, \dots, p_K$, de somme 1 : $P(y{=}k) = p_k$. Chiffre de 0 à 9 (ch. 07), token suivant parmi des dizaines de milliers (ch. 20).
- Gaussienne (ou normale), pour un nombre réel bruité. Paramètres : moyenne $\mu$ et écart-type $\sigma$ ; densité $\frac{1}{\sqrt{2\pi\sigma^2}}\, e^{-(y-\mu)^2 / 2\sigma^2}$, la « courbe en cloche ». Bruit de mesure, initialisation des poids, modèles de diffusion (ch. 25).

Deux résumés d'une loi : l'espérance $\E[Y]$ (la valeur moyenne, $\mu$ pour la gaussienne, $p$ pour Bernoulli) et la variance $\mathrm{Var}(Y) = \E[(Y - \E[Y])^2]$ (la dispersion, $\sigma^2$ pour la gaussienne, $p(1-p)$ pour Bernoulli). Dans notre régression, on suppose $y = b + w\,x + \varepsilon$ avec un bruit $\varepsilon$ gaussien de moyenne 0 et d'écart-type 0,3 : même la vraie droite fait une erreur quadratique moyenne de $0{,}3^2 = 0{,}09$ sur des données nouvelles. En moyenne sur des données nouvelles, aucun modèle ne descend sous ce plancher, le bruit irréductible (une mesure sur quelques points peut tomber en dessous par hasard).

### Maximum de vraisemblance et MAP

La vraisemblance. Prenons une pièce truquée de paramètre $p$ inconnu, lancée 10 fois : 7 piles. La probabilité de cette séquence précise, vue comme une fonction de $p$, est la vraisemblance : $L(p) = p^7 (1-p)^3$. On travaille avec son logarithme (ici népérien, $\ln$), plus commode (les produits deviennent des sommes, et on évite des nombres minuscules) :

| $p$ | 0,5 | 0,7 | 0,9 |
|---|---|---|---|
| $\ln L(p) = 7 \ln p + 3 \ln(1-p)$ | −6,93 | −6,11 | −7,65 |

Le maximum de vraisemblance choisit le $p$ qui rend les données observées les plus probables : en annulant la dérivée, $\hat p = 7/10$. Le principe est général : on maximise $\sum_i \log p_\theta(y_i \mid x_i)$, ou de façon équivalente on minimise $-\sum_i \log p_\theta(y_i \mid x_i)$. C'est déjà une perte.

Lien avec la MSE. Si l'on suppose que $y_i$ suit une gaussienne centrée sur la prédiction $f_\theta(x_i)$, d'écart-type $\sigma$ fixé, alors $-\log$ vraisemblance $= \frac{1}{2\sigma^2}\sum_i (y_i - f_\theta(x_i))^2 + \text{constante}$ : maximiser la vraisemblance revient exactement aux moindres carrés. C'est ce que Gauss montrait en 1809. La correspondance complète (gaussienne → MSE, Laplace → MAE, catégorielle → entropie croisée) est développée au chapitre 01.

Le MAP : ajouter ce qu'on croit avant de voir les données. Le maximum de vraisemblance ne fait confiance qu'aux données. Avec peu de points et beaucoup de paramètres, il peut choisir des valeurs absurdes (on le verra avec des coefficients de plusieurs milliers). Le maximum a posteriori ajoute une loi a priori sur les paramètres et maximise, par la règle de Bayes, $p(\theta \mid \text{données}) \propto p(\text{données} \mid \theta)\, p(\theta)$.

Avec un a priori gaussien $w_j \sim \mathcal{N}(0, \tau^2)$ (« les poids sont probablement petits, de l'ordre de $\tau$ »), on prend $-\log$ et on multiplie par $2\sigma^2$ :

(Formule : MAP avec a priori gaussien = moindres carrés + pénalité L2.)

Avec $\sigma = 0{,}3$ : un a priori $\tau = 1$ donne $\lambda = 0{,}09$ ; un a priori plus large, $\tau = 10$, donne $\lambda = 0{,}0009$. Plus on croit les poids petits, plus on pénalise. Cette pénalité s'appelle L2 ou Ridge (section « Surapprentissage » ci-dessous). Un a priori de Laplace, $p(w_j) \propto e^{-|w_j|/s}$ (d'échelle $s$), donne de même une pénalité L1, $\lambda \sum_j |w_j|$ avec $\lambda = 2\sigma^2/s$.

### Entropie, entropie croisée, KL, information mutuelle

Ces quatre quantités de la théorie de l'information (Shannon, 1948) reviennent dans les pertes de classification (ch. 01), les modèles génératifs (ch. 18, 25) et l'apprentissage contrastif (ch. 23). On les définit ici une fois ; les logarithmes sont en base 2, donc les résultats en bits.

L'entropie mesure l'incertitude d'une loi $p$. En simplifiant : le nombre moyen de questions oui/non nécessaires pour deviner le résultat (plus exactement, la limite inférieure du nombre moyen de bits par résultat, atteinte en codant de longues séquences : coder un seul lancer coûte toujours au moins 1 bit).

Une pièce équilibrée : $H = 1$ bit. Une pièce qui tombe sur pile 9 fois sur 10 : $H = -(0{,}9 \log_2 0{,}9 + 0{,}1 \log_2 0{,}1) = 0{,}47$ bit. On est moins surpris en moyenne.

L'entropie croisée $H(p, q) = -\sum_x p(x) \log_2 q(x)$ mesure la surprise moyenne quand les données suivent $p$ mais qu'on prédit avec $q$. Pour la pièce à 90 % ($p = (0{,}9 ; 0{,}1)$) :

- si le modèle prédit $q = (0{,}5 ; 0{,}5)$ : $H(p, q) = 1$ bit ;
- si le modèle prédit $q = (0{,}8 ; 0{,}2)$ : $H(p, q) = 0{,}52$ bit ;
- si le modèle prédit $q = p$ : $H(p, q) = H(p) = 0{,}47$ bit, le minimum.

La divergence de Kullback-Leibler est l'excès : $D_{\mathrm{KL}}(p \,\|\, q) = H(p, q) - H(p) = \sum_x p(x) \log_2 \frac{p(x)}{q(x)}$. Ici 0,53 bit pour $q = (0{,}5 ; 0{,}5)$ et 0,05 bit pour $q = (0{,}8 ; 0{,}2)$. Elle est toujours positive ou nulle, nulle seulement si $q = p$, et pas symétrique ($D_{\mathrm{KL}}(p\|q) \neq D_{\mathrm{KL}}(q\|p)$ en général) : ce n'est pas une distance. Comme $H(p)$ ne dépend pas du modèle, minimiser l'entropie croisée ou la KL par rapport à $q$, c'est la même chose. L'entropie croisée comme perte de classification est traitée au chapitre 01.

L'information mutuelle mesure ce qu'une variable apprend sur une autre. Exemple : $X$ = « page lourde ou légère », $Y$ = « chargement lent ou rapide », avec la loi jointe :

| | lent | rapide |
|---|---|---|
| lourde | 0,4 | 0,1 |
| légère | 0,1 | 0,4 |

Ici $I = 1 + 1 - 1{,}72 = 0{,}28$ bit : connaître le poids de la page lève un peu plus d'un quart de l'incertitude sur la vitesse. Si les quatre cases valaient 0,25, $I$ serait nul (indépendance). L'information mutuelle ne suppose aucune relation linéaire, contrairement à la corrélation. Le chapitre 23 montre que la perte InfoNCE de l'apprentissage contrastif en maximise une borne inférieure.

### Descente de gradient en 1D puis en 2D

Les équations normales ne marchent que pour la MSE d'un modèle linéaire. Pour presque tout le reste du cours (un réseau de neurones, une autre perte), il n'existe pas de formule. On descend alors pas à pas.

Le gradient $\nabla\Loss(\theta)$ est le vecteur des dérivées partielles $\left(\frac{\partial\Loss}{\partial b}, \frac{\partial\Loss}{\partial w}\right)$. Il pointe vers la montée la plus raide ; on fait donc un petit pas dans la direction opposée :

(Formule : Descente de gradient (Cauchy, 1847).)

C'est la descente de gradient, et $\eta$ est le taux d’apprentissage. Cauchy l'écrit en 1847 pour une fonction $u = f(x, y, z, \dots)$ jamais négative, qu'il cherche à annuler : prendre les accroissements $\alpha = -\theta X$, $\beta = -\theta Y$… (où $X, Y$ sont les dérivées partielles et $\theta$ un petit nombre positif : son $\theta$ est notre $\eta$), ce qui fait baisser $u$ de $\theta(X^2 + Y^2 + \dots)$ au premier ordre.

Un pas à la main, sur les trois pages de tout à l'heure, depuis $\theta_0 = (0, 0)$ avec $\eta = 0{,}1$ :

1. résidus $X\theta_0 - y = (-1, -3, -4)$ ;
2. gradient $\frac{2}{3} X^\top (X\theta_0 - y) = \frac{2}{3}\,(-8, -11) = (-5{,}33 ; -7{,}33)$ ;
3. mise à jour $\theta_1 = (0, 0) - 0{,}1 \times (-5{,}33 ; -7{,}33) = (0{,}53 ; 0{,}73)$.

La perte passe de 8,67 à 2,41, puis à 0,70 au pas suivant. Elle continue de baisser vers 0,056.

```python
def gradient_mse(theta, X, y):
    return 2 / len(y) * X.T @ (X @ theta - y)

theta = np.array([2.0, 0.0])            # départ (b, w) = (2, 0)
for t in range(1000):
    theta -= 0.3 * gradient_mse(theta, X, y)   # η = 0,3
```

Vérifier son gradient. Avant de faire confiance à une formule de gradient, on la compare aux différences finies $\frac{\Loss(\theta + h e_j) - \Loss(\theta - h e_j)}{2h}$ avec $h = 10^{-6}$. Dans l'exercice, l'écart est de $2{,}7 \times 10^{-10}$. Les bibliothèques modernes calculent le gradient automatiquement (rétropropagation, chapitre 07), mais le réflexe reste utile.

#### En 1D : le pas critique

Fixons $b$ à sa valeur optimale et laissons seulement $w$ apprendre. La perte devient une parabole $\Loss(w) = \frac{a}{2}(w - w^\star)^2 + \text{cte}$, de courbure $a = \frac{2}{N}\sum_i x_i^2$. Un pas donne $w_{t+1} - w^\star = (1 - \eta a)(w_t - w^\star)$ : l'écart au minimum est multiplié à chaque pas par $1 - \eta a$.

- $0 \lt \eta a \lt 1$ : facteur entre 0 et 1, on glisse vers le minimum sans le dépasser ;
- $1 \lt \eta a \lt 2$ : facteur entre −1 et 0, on saute par-dessus le minimum à chaque pas, mais de moins en moins loin ;
- $\eta a \gt 2$ : facteur inférieur à −1, chaque saut est plus long que le précédent : divergence.

Le pas critique est donc $\eta_c = 2/a$. Sur les 40 pages, $a = 2{,}93$ et $\eta_c = 0{,}683$. Essayez de part et d'autre :

#### En 2D : la vallée et sa courbure

Quand $b$ et $w$ apprennent ensemble, la perte est une cuvette dont les lignes de niveau sont des ellipses. Sa courbure dans chaque direction est donnée par la hessienne, la matrice des dérivées secondes ; pour la MSE linéaire, elle est constante : $H = \frac{2}{N} X^\top X$. Pour une matrice 2 × 2 comme celle-ci, il existe deux directions particulières, perpendiculaires, les axes des ellipses (les directions propres) ; la courbure le long de chacune est une valeur propre de $H$. Sur les 40 pages, ces deux valeurs propres sont $\lambda_{\min} = 0{,}281$ (direction peu courbée, le fond plat de la vallée) et $\lambda_{\max} = 4{,}650$ (direction raide, les flancs). Leur rapport, le conditionnement, vaut 16,6 : la vallée est 16,6 fois plus courbée dans un sens que dans l'autre.

Le raisonnement 1D s'applique dans chaque direction propre, avec un facteur $1 - \eta\lambda$ par direction : $1 - \eta\lambda_{\min}$ pour la direction plate, $1 - \eta\lambda_{\max}$ pour la direction raide. La direction raide impose le pas critique, $\eta_c = 2/\lambda_{\max} = 0{,}430$ ; la direction plate impose la lenteur.

| $\eta$ | Facteur plat | Facteur raide | Résultat (départ (2, 0), écart au minimum $\lt 10^{-6}$) |
|---|---|---|---|
| 0,05 | 0,986 | 0,767 | 1 000 itérations : lent, freiné par la direction plate |
| 0,3 | 0,916 | −0,395 | 161 itérations |
| 0,42 | 0,882 | −0,953 | 263 itérations : oscille d'un flanc à l'autre |
| 0,45 | 0,874 | −1,093 | diverge : MSE au-delà de $10^8$ en 110 itérations |

Le cas $\eta = 0{,}42$ surprend : des pas plus grands que 0,3, et pourtant une convergence plus lente. L'écart dans la direction raide n'est réduit que de 5 % par pas ($|1 - \eta\lambda_{\max}| = 0{,}953$), en changeant de signe à chaque fois. Le meilleur pas fixe équilibre les deux directions, $\eta = 2/(\lambda_{\min} + \lambda_{\max}) \approx 0{,}41$, et le pire des deux facteurs vaut alors $(\kappa - 1)/(\kappa + 1) = 0{,}89$ avec $\kappa = 16{,}6$. Plus le conditionnement est grand, plus ce facteur approche 1 : c'est le problème que le momentum et Adam cherchent à contourner (chapitre 02).

À vous : choisissez un pas avec les préréglages ou le curseur, faites des pas un par un, et déplacez le point de départ (clic ou toucher dans le graphique). Le pas critique ne dépend pas du point de départ.

Deux simplifications de cette section : la perte est ici quadratique et convexe, avec un seul minimum et une courbure constante. Pour un réseau de neurones, la courbure change d'un point à l'autre, il y a des points selles (minimum dans une direction, maximum dans une autre) et de nombreux minima ; la règle $\eta \lt 2/\lambda_{\max}$ ne vaut alors que localement. Et l'on a calculé le gradient sur toutes les données à chaque pas (descente « batch »). Avec des millions d'exemples, on l'estime sur un petit lot tiré au hasard : c'est la descente de gradient stochastique, qui remonte à l'approximation stochastique de Robbins et Monro (1951), traitée au chapitre 02.

### Surapprentissage, biais-variance, régularisation

Changeons de problème pour rendre le modèle plus souple. La vraie fonction est maintenant une courbe, $y = \sin(2\pi x)$ plus un bruit d'écart-type 0,3 ; on a 20 points d'entraînement, 20 de validation et 1 000 de test. Le modèle est un polynôme de degré $d$ : $\hat y = \theta_0 + \theta_1 u + \dots + \theta_d u^d$, avec $u = 2x - 1$ ramené dans $[-1, 1]$. C'est toujours une régression linéaire, au sens « linéaire par rapport aux paramètres » : on a simplement remplacé la colonne $x$ par $d$ colonnes $u, u^2, \dots, u^d$.

| Degré | Erreur d'entraînement | Erreur de test |
|---|---|---|
| 0 (constante) | 0,614 | 0,627 : sous-apprentissage |
| 3 | 0,048 | 0,131, le minimum |
| 12 | 0,016 | 39 |
| 15 | 0,006 | $5{,}1 \times 10^5$ |

L'erreur d'entraînement baisse toujours avec le degré : un polynôme plus souple peut toujours faire au moins aussi bien sur les points qu'il voit. L'erreur de test, elle, remonte : c'est le surapprentissage. Le polynôme de degré 12 passe près de chacun des 20 points en oscillant violemment entre eux, avec des coefficients énormes ($\lVert w \rVert = 6\,765$).

#### Le compromis biais-variance

Imaginons qu'on puisse tirer beaucoup de jeux d'entraînement différents et réentraîner à chaque fois. En un point $x$, l'erreur quadratique attendue se décompose exactement en trois termes :

(Formule : Décomposition biais-variance (perte quadratique).)

Le biais mesure l'erreur systématique d'un modèle trop rigide (une droite ne suivra jamais une sinusoïde, quel que soit le jeu tiré). La variance mesure à quel point le modèle change d'un jeu à l'autre (un polynôme de degré 11 sur 12 points épouse chaque fois un bruit différent). Augmenter la complexité baisse le biais et monte la variance : c'est le compromis biais-variance. La figure suivante est mesurée, pas dessinée : 500 jeux simulés de 12 points, un polynôme de chaque degré ajusté sur chacun.

Le minimum (0,12 au degré 3) est proche du plancher 0,09. Cette courbe en U est la vision classique. Elle est exacte ici, mais les réseaux de neurones modernes, avec bien plus de paramètres que d'exemples, généralisent souvent bien quand même : la « double descente » du chapitre 27 la nuance. La théorie de la généralisation (PAC, dimension VC) est au chapitre 10.

#### Pénalité L2 (Ridge) : rapetisser tous les poids

La régularisation contraint le modèle pour réduire la variance, au prix d'un peu de biais. La plus courante ajoute à la perte la somme des carrés des poids. C'est la régression ridge, et c'est exactement le MAP avec a priori gaussien vu plus haut :

(Formule : Ridge : perte pénalisée et solution exacte.)

Le $\lambda$ n'est pas appris par la descente : c'est un hyperparamètre, fixé avant l'entraînement. Au degré 12, $\lambda = 10^{-3}$ fait passer l'erreur de test de 39 à 0,137. Ajouter $\lambda P$ rend aussi la matrice à inverser mieux conditionnée : c'était la motivation de Hoerl et Kennard en 1970, pour qui les moindres carrés donnent des estimations peu fiables quand les variables d'entrée sont fortement corrélées. Ils donnent son nom et son cadre statistique à une pénalité déjà connue ailleurs : c'est la régularisation de Tikhonov, utilisée pour les équations intégrales.

#### Pénalité L1 (Lasso) : annuler des poids

Le Lasso (Tibshirani, 1996 ; la même idée était apparue en géophysique dès 1986) remplace le carré par la valeur absolue : $\lVert A\theta - y \rVert^2 + \lambda \sum_j |w_j|$ (scikit-learn écrit $\frac{1}{2N}\lVert A\theta - y\rVert^2 + \alpha \sum_j |w_j|$, d'où un $\alpha$ sur une autre échelle). Différence décisive : la pénalité L2 a une dérivée $2\lambda w_j$ qui s'évanouit près de zéro, elle rapetisse les poids sans jamais les annuler ; la pénalité L1 a une dérivée de taille constante $\lambda$, qui pousse un poids peu utile jusqu'à exactement zéro. Avec $\alpha = 10^{-2}$, le Lasso annule 9 des 12 coefficients, pour une erreur de test de 0,108. Il fait donc aussi une sélection de variables : un modèle à 3 termes est plus facile à lire qu'un modèle à 12.

#### Arrêt précoce : régulariser sans toucher à la perte

Troisième façon de faire : garder la MSE sans pénalité, lancer la descente de gradient sur le polynôme de degré 12, et surveiller l'erreur sur le jeu de validation à mesure qu'on avance. On garde les paramètres du moment où elle était la plus basse : c'est l'arrêt précoce.

| | Itération | Erreur de test |
|---|---|---|
| Meilleure validation (mesurée à 300 instants) | 4 182 | 0,136 |
| Fin de la descente | 500 000 | 0,216 |
| Pour comparer : Ridge, λ = 10⁻³ | — | 0,137 |

Sans aucune pénalité, s'arrêter tôt donne un modèle presque aussi bon que Ridge. L'explication est celle de la section 2D : la descente progresse vite dans les directions très courbées (celles que les données déterminent bien) et lentement dans les directions plates (celles que le bruit détermine). S'arrêter tôt revient à ne pas laisser le temps aux secondes de s'installer. Pour une perte quadratique, on peut montrer que l'arrêt précoce agit comme une pénalité L2 dont la force décroît avec le nombre de pas (en simplifiant : l'équivalence n'est qu'approchée). En deep learning, c'est une régularisation très répandue, souvent appliquée sans qu'on la nomme.

### Entraînement, validation, test et validation croisée

Pour choisir $\lambda$ ou le degré, il faut une mesure honnête de l'erreur sur des données nouvelles. D'où trois jeux aux rôles bien séparés :

| Jeu | Sert à | Ce qu'il ne faut pas faire |
|---|---|---|
| entraînement | ajuster les paramètres $\theta$ (descente, formule fermée) | juger le modèle dessus : l'erreur d'entraînement est toujours optimiste |
| validation | choisir les hyperparamètres ($\lambda$, degré, moment d'arrêt) | le réutiliser comme mesure finale : on l'a « consulté » pour décider |
| jeu de test | mesurer une seule fois, à la fin, la performance attendue | décider quoi que ce soit en le regardant |

Le jeu de validation de 20 points a servi à l'arrêt précoce. Pour choisir $\lambda$, plaçons-nous dans le cas courant où l'on n'a que les 20 points d'entraînement : en garder 5 de côté gâche des données, et le score dépend beaucoup des 5 points tirés. La validation croisée à $k$ plis utilise chaque point tour à tour pour entraîner et pour valider ; à chaque tour, on entraîne sur les 4 autres plis :

```text
            pli 1   pli 2   pli 3   pli 4   pli 5
tour 1     [ VAL ] [entr.] [entr.] [entr.] [entr.]   → MSE₁
tour 2     [entr.] [ VAL ] [entr.] [entr.] [entr.]   → MSE₂
tour 3     [entr.] [entr.] [ VAL ] [entr.] [entr.]   → MSE₃
tour 4     [entr.] [entr.] [entr.] [ VAL ] [entr.]   → MSE₄
tour 5     [entr.] [entr.] [entr.] [entr.] [ VAL ]   → MSE₅
score CV = (MSE₁ + … + MSE₅) / 5
```

Dans l'exercice, sur une grille de 41 valeurs de $\lambda$ entre $10^{-8}$ et $10^2$, la validation croisée écrite à la main choisit $\lambda = 5{,}6 \times 10^{-3}$ (score CV 0,063, plus bas que le plancher 0,09 : sur 4 points par pli, l'estimation est très bruitée, et retenir le minimum parmi 41 valeurs la rend optimiste ; la question 5 du quiz revient sur ce biais, et c'est pour cela que la mesure finale se fait sur le test) ; `cross_val_score` de scikit-learn, sur les mêmes plis, donne les mêmes scores à $10^{-8}$ près. Réentraîné sur les 20 points, ce modèle obtient une MSE de 0,141 sur le test, mesurée une seule fois.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Mêmes plis que la version à la main : 5 paquets de 4 points tirés au hasard
perm = np.random.default_rng(42).permutation(20)
plis = [(np.setdiff1d(perm, p), p) for p in np.array_split(perm, 5)]
# A_tr[:, 1:] : sans la colonne de 1, Ridge ajoute lui-même la constante (non pénalisée)
scores = {lam: -cross_val_score(Ridge(alpha=lam), A_tr[:, 1:], y_tr, cv=plis,
                                scoring="neg_mean_squared_error").mean()
          for lam in np.logspace(-8, 2, 41)}
lam = min(scores, key=scores.get)               # choisi SANS regarder le test
```

Deux pièges classiques. La fuite de données : normaliser les entrées avec la moyenne calculée sur toutes les données (test compris) avant de découper, c'est laisser l'information du test entrer dans l'entraînement ; on calcule la normalisation sur l'entraînement seul, à l'intérieur de chaque pli. Les données ordonnées dans le temps : pour prédire le trafic de demain, on ne valide pas sur des jours tirés au hasard parmi le passé, mais sur la période qui suit l'entraînement.

La validation croisée n'a pas un inventeur unique : Mosteller et Tukey en donnent une première formulation en 1968, et la variante « un point à la fois » (*leave-one-out*) est introduite indépendamment par Stone (1974), Allen (1974) et Geisser (1975), d'après la synthèse d'Arlot et Celisse (§4.4).

### Métriques : mesurer ce qui compte

La perte est ce qu'on optimise ; la métrique est ce qu'on rapporte, et elles diffèrent souvent (la distinction est développée au chapitre 01).

En régression. La MSE (en unités au carré), sa racine RMSE (dans l'unité de $y$ : 0,21 s pour la droite de l'exercice), la MAE (erreur absolue moyenne), et le coefficient de détermination :

La droite des 40 pages a $R^2 = 0{,}936$ ; le polynôme Ridge choisi par validation croisée a $R^2 = 0{,}765$ sur le test.

En classification, tout part de la matrice de confusion. Exemple : un détecteur de robots sur 1 000 visites, dont 50 robots. Il en signale 60, dont 40 vrais robots.

| | prédit robot | prédit humain |
|---|---|---|
| vrai robot (50) | VP = 40 | FN = 10 |
| vrai humain (950) | FP = 20 | VN = 930 |

- exactitude : $(40 + 930)/1\,000 = 97\ \%$. Mais un détecteur qui répond toujours « humain » obtient déjà 95 % sans détecter aucun robot. Avec des classes déséquilibrées, l'exactitude seule ne dit presque rien.
- précision : $\text{VP}/(\text{VP} + \text{FP}) = 40/60 = 0{,}67$. Quand il crie « robot », il a raison deux fois sur trois.
- rappel : $\text{VP}/(\text{VP} + \text{FN}) = 40/50 = 0{,}80$. Il attrape 80 % des robots.
- score F1 : $2PR/(P + R) = 2 \times 0{,}67 \times 0{,}80 / 1{,}47 = 0{,}73$. La moyenne harmonique punit le déséquilibre : précision 1 et rappel 0,01 donnent F1 = 0,02, pas 0,5.

Précision et rappel dépendent du seuil : le modèle sort un score entre 0 et 1, et l'on déclare « robot » au-delà de 0,5, ou de 0,9. Monter le seuil fait en général gagner en précision et perdre en rappel. La courbe ROC trace le taux de vrais positifs (le rappel) contre le taux de faux positifs $\text{FP}/(\text{FP} + \text{VN})$ pour tous les seuils, et son aire, l'aire sous la courbe ROC, résume le classement sans choisir de seuil : c'est la probabilité qu'un robot tiré au hasard ait un score plus haut qu'un humain tiré au hasard. Exemple : robots notés 0,9 ; 0,8 ; 0,4, humains notés 0,7 ; 0,3 ; 0,2. Sur les 9 paires (robot, humain), le robot a le score le plus haut dans 8 (seule la paire 0,4 contre 0,7 est mal classée) : AUC = 8/9 ≈ 0,89. Une AUC de 0,5 correspond au hasard.

Quelle métrique choisir dépend du coût des erreurs : rater une fraude (FN) ou bloquer un vrai client (FP) ne coûtent pas la même chose.

## Ce qui coinçait

Ce chapitre pose le cadre, mais chacun de ses choix a des limites que la suite du cours lève :

- La MSE n'est qu'un choix parmi d'autres. Elle suppose un bruit gaussien et se laisse tirer par les points aberrants ; pour classer, elle convient mal. Changer de perte change le modèle obtenu : chapitre 01.
- La descente de gradient « batch » est lente et capricieuse. Chaque pas parcourt toutes les données, le pas critique dépend de la courbure, et un mauvais conditionnement la ralentit. Mini-lots, momentum et Adam : chapitre 02.
- Le modèle linéaire ne sait faire que ce que ses colonnes permettent. Pour la sinusoïde, il a fallu inventer à la main les colonnes $u^2, u^3, \dots$ Apprendre ces représentations au lieu de les fabriquer est la promesse des réseaux de neurones : chapitre 07.
- La formule fermée ne passe pas à l'échelle. Résoudre $(X^\top X)\theta = X^\top y$ coûte de l'ordre de $N d^2 + d^3$ opérations pour $N$ exemples et $d$ paramètres (former $X^\top X$, puis résoudre) : impensable pour des millions de poids, et inexistante pour un réseau non linéaire.
- La courbe en U ne raconte pas toute l'histoire. Elle sera nuancée par la double descente au chapitre 27, et la question « pourquoi ça généralise ? » reçoit un traitement théorique au chapitre 10.

- Moindres carrés (Legendre 1805, Gauss 1809) : Choisir les paramètres qui minimisent la somme des carrés des erreurs ; pour un modèle linéaire, une formule exacte.
- Descente de gradient (Cauchy 1847) : Quand il n’y a pas de formule, descendre la pente pas à pas ; le pas doit rester sous 2/λ_max.
- Maximum de vraisemblance, MAP : Une perte, c’est un modèle probabiliste du bruit ; une pénalité, c’est un a priori sur les poids.
- Ridge (1970), Lasso (1996), arrêt précoce : Contraindre le modèle pour qu’il apprenne la tendance plutôt que le bruit.
- Validation croisée (Stone, Allen 1974 ; Geisser 1975) : Choisir les réglages sur des données que le modèle n’a pas vues, et garder le test pour la fin.
- Fonctions de perte : La perte choisie décide de ce que le modèle apprend.

Et ensuite ? Tout ce chapitre a minimisé la MSE sans se demander si c'était le bon critère. Ce n'est qu'un choix, et ce choix change le modèle : avec quelques points aberrants, la droite des moindres carrés bascule, celle de la valeur absolue tient bon ; pour classer, on minimise tout autre chose. C'est le sujet du chapitre 01 : fonctions de perte.

## Exercice

En trois parties : la droite (formule fermée, descente de gradient avec quatre pas, scikit-learn), la courbe (polynômes de degré 0 à 15, Ridge, Lasso, arrêt précoce), puis la validation croisée à la main et avec scikit-learn. Ce qu'il faut observer :

- les trois pas inférieurs à $\eta_c = 0{,}430$ retrouvent la formule fermée à $10^{-6}$ près, et $\eta = 0{,}45$, à peine 5 % au-dessus, diverge ;
- l'erreur d'entraînement baisse toujours avec le degré quand l'erreur de test explose (39 au degré 12) ; Ridge, avec $\lambda$ choisi par validation croisée, la ramène à 0,141, soit 277 fois moins ;
- le jeu de test ne sert qu'une fois, à la toute fin.

Questions de réflexion : que deviendrait le conditionnement de $H$ si l'on centrait les poids $x$ (moyenne nulle) avant d'apprendre ? Et si l'on choisissait $\lambda$ en regardant directement l'erreur de test, que vaudrait encore ce chiffre ?

## Auto-évaluation

Question : Une matrice X a la forme (64, 128) et une matrice W la forme (128, 10). Quelle est la forme de X @ W, et combien W a-t-il de paramètres ?

Réponse : La dimension intérieure (128) est commune et disparaît : `X @ W` a la forme `(64, 10)`, par exemple 10 scores pour chacun des 64 exemples d'un lot. W a $128 \times 10 = 1\,280$ paramètres. `W @ X`, lui, lèverait une erreur : `(128, 10) @ (64, 128)` n'aligne pas 10 et 64.

Question : Sur une perte quadratique dont la hessienne a pour valeurs propres 0,5 et 8, quel est le pas critique de la descente de gradient ? Que se passe-t-il avec η = 0,2 ?

Réponse : $\eta_c = 2/\lambda_{\max} = 2/8 = 0{,}25$. Avec $\eta = 0{,}2$, le facteur dans la direction raide vaut $1 - 0{,}2 \times 8 = -0{,}6$ : la descente oscille d'un flanc à l'autre mais converge. Dans la direction plate, il vaut $1 - 0{,}2 \times 0{,}5 = 0{,}9$ : c'est elle qui fixe la vitesse, environ 10 % d'écart en moins par pas.

Question : Pourquoi la pénalité L2 correspond-elle à un a priori gaussien sur les poids, et que devient λ si l'on croit les poids très petits ?

Réponse : Un a priori $w_j \sim \mathcal{N}(0, \tau^2)$ ajoute $\frac{1}{2\tau^2}\sum_j w_j^2$ à $-\log p(\theta \mid \text{données})$ ; combiné au terme $\frac{1}{2\sigma^2}\sum_i(y_i - \hat y_i)^2$ du bruit gaussien, cela donne les moindres carrés plus $\lambda \lVert w \rVert^2$ avec $\lambda = \sigma^2/\tau^2$. Croire les poids très petits, c'est prendre $\tau$ petit, donc $\lambda$ grand : on pénalise plus fort.

Question : Un détecteur de fraude affiche une exactitude de 99,2 % sur des données où 1 % des transactions sont frauduleuses. Est-ce un bon modèle ?

Réponse : On ne peut pas le dire. Répondre toujours « pas de fraude » donne déjà 99 % d'exactitude avec un rappel nul. Il faut regarder la précision et le rappel sur la classe « fraude » (ou le F1, ou l'AUC-ROC), et le coût relatif d'une fraude ratée face à un client bloqué à tort.

Question : Vous testez 50 valeurs de λ et gardez celle qui donne la plus petite erreur sur le jeu de test (0,12). Pourquoi 0,12 est-il une estimation trop optimiste ?

Réponse : Parce que le jeu de test a servi à choisir : parmi 50 essais, on a retenu celui que le hasard de ces données-là favorisait le plus. Le test est devenu un jeu de validation. Il faut choisir λ par validation (ou validation croisée) sur les données d'entraînement, puis mesurer le test une seule fois, comme dans l'exercice (0,141).

## Pour aller plus loin

Textes fondateurs (voir les fiches ci-dessus) : l'appendice de Legendre (p. 72-80 de l'édition de 1805), en français, se lit encore très bien (transcription de l'université d'York, d'après la réédition de 1806) ; la note de Cauchy (1847) tient en trois pages (traduction anglaise de R. J. Pulskamp).

Cours et vidéos

- 3Blue1Brown, *Essence of linear algebra* (vidéos, en anglais) : vecteurs, matrices et produits vus géométriquement.
- 3Blue1Brown, *Gradient descent, how neural networks learn* : la descente de gradient sur un vrai réseau, chapitre 2 de la série sur les réseaux de neurones.
- Stanford CS229, notes de cours d'Andrew Ng et collègues (PDF) : §1.1 à 1.3 (LMS, équations normales, interprétation probabiliste), §8.1 (biais-variance), §9.1 à 9.4 (régularisation, validation croisée, lien bayésien).
- Chris Olah, *Visual Information Theory* : entropie, entropie croisée et KL en images.
- Google, *Classification: ROC and AUC* (Machine Learning Crash Course) : courbe ROC et AUC, interactif.

Livres en accès libre

- Deisenroth, Faisal et Ong, *Mathematics for Machine Learning* (Cambridge University Press, 2020) : algèbre linéaire, probabilités et optimisation, exactement les prérequis de ce chapitre.
- Hastie, Tibshirani et Friedman, *The Elements of Statistical Learning* (2e éd., 2009) : §3.4 (Ridge et Lasso), §7.3 (décomposition biais-variance), §7.10 (validation croisée).

Documentation

- scikit-learn, *Linear Models* (moindres carrés, Ridge, Lasso) et *Cross-validation*.
