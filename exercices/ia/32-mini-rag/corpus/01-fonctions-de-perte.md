# 01. Fonctions de perte (loss functions)

Repère historique (1922 (Fisher) à 1995 (Cortes et Vapnik)) : Choisir ce que le modèle doit minimiser : c'est ce choix, et non l'algorithme d'optimisation, qui décide de ce que le modèle apprend. Et ne pas confondre ce qu'on minimise (la loss) avec ce qu'on mesure (la métrique).

Au début du XIXe siècle, Legendre et Gauss ajustent des orbites en minimisant la somme des carrés des écarts (chapitre 00). Gauss (1809) les relie déjà à une loi normale des erreurs. En 1912 puis 1922, Ronald Fisher, statisticien à la station agronomique de Rothamsted, généralise cette idée en une méthode valable pour tout modèle : le maximum de vraisemblance. Chaque perte devient alors l'aveu d'une hypothèse sur le bruit des données. Les décennies suivantes ajoutent des pertes taillées pour d'autres besoins : la divergence de Kullback-Leibler (Kullback et Leibler, 1951), la régression logistique (Berkson 1944, Cox 1958), la perte robuste de Huber (1964), la sortie softmax associée à l'entropie croisée pour les réseaux de neurones (Bridle 1989), la marge souple des SVM (Cortes et Vapnik 1995). Aujourd'hui, presque tout le deep learning repose sur l'une d'elles.

## L'idée en un exemple

Une fonction de perte est un barème de pénalités : pour chaque exemple, elle dit combien coûte l'écart entre ce que le modèle prédit et ce qu'il aurait dû prédire. L'entraînement cherche les paramètres qui rendent la pénalité moyenne la plus petite possible. Le modèle n'a pas d'autre boussole : il apprend exactement ce que le barème récompense, ni plus ni moins.

Exemple, le modèle le plus simple qui soit : résumer par un seul nombre les salaires mensuels d'une équipe de 11 personnes, dont la dirigeante.

```text
2 000  2 100  2 300  2 400  2 500  2 600  2 700  2 900  3 100  3 300  25 000   (euros)
```

| Barème (perte) | Ce qu'il pénalise | Meilleur nombre | Pourquoi |
|---|---|---|---|
| Carré de l'écart : erreur quadratique moyenne (MSE) | les gros écarts, très fort | 4 627 € | c'est la moyenne : le salaire de 25 000 € tire tout vers le haut |
| Valeur absolue de l'écart : erreur absolue moyenne (MAE) | chaque euro d'écart, pareil | 2 600 € | c'est la médiane : la dirigeante ne compte que comme « un point au-dessus » |
| perte de Huber (δ = 1 000 €) | carré jusqu'à 1 000 €, puis linéaire | 2 690 € | un compromis, presque insensible au salaire extrême |

Mêmes données, même « modèle » (une constante), trois réponses. Aucune n'est fausse : chacune est la bonne réponse à la question posée par sa perte. Si vous voulez estimer la masse salariale, la moyenne est la bonne ; si vous voulez décrire le salaire « typique », c'est la médiane. Choisir la perte, c'est choisir la question.

Deuxième idée du chapitre : la perte n'est pas forcément ce qu'on veut mesurer à la fin. Un classifieur de spam est jugé sur son taux d'erreur, mais on l'entraîne presque toujours avec une autre fonction, l'entropie croisée. La perte sert à apprendre ; la métrique sert à juger.

| | Avant : « on minimise l'erreur » | Après : on choisit sa perte |
|---|---|---|
| Ce qu'on fait | MSE pour tout, par habitude ; on optimise directement ce qu'on mesure | une perte adaptée au bruit et à la tâche (MSE, MAE, Huber, log-loss, hinge, entropie croisée…), distincte de la métrique |
| Effet sur le modèle | quelques valeurs aberrantes suffisent à fausser une régression ; en classification, des exemples « trop bien classés » déplacent la frontière | le modèle apprend ce qu'on veut vraiment ; les points aberrants ou trop faciles pèsent peu |
| Lien avec la théorie | aucun : le carré semble « naturel » | chaque perte correspond à une hypothèse sur le bruit (maximum de vraisemblance) |

## Le mécanisme

Ce que la méthode optimise. Toutes les pertes du parcours : ce chapitre est la page carrefour des fonctions de perte. Les autres chapitres y renvoient pour la fiche complète de chacune.

### Loss contre métrique

Le circuit complet d'un apprentissage supervisé : la perte est la seule chose dont le gradient remonte jusqu'aux paramètres. La métrique est calculée à côté, sur d'autres données, et n'influence rien directement.

Pourquoi ne pas optimiser directement la métrique ? Parce que la plupart des métriques sont en escalier. Le taux d'erreur d'un classifieur est la moyenne de la perte 0-1 : 1 si la classe prédite est fausse, 0 sinon. Bouger un peu les paramètres ne change en général aucune prédiction, donc le taux d'erreur ne bouge pas : sa dérivée est nulle presque partout, et la descente de gradient (chapitre 02) n'a aucune pente à suivre. On minimise donc une perte de substitution (*surrogate loss*) lisse, dont la petite valeur entraîne en général une bonne métrique.

L'exercice du chapitre montre que les deux peuvent diverger nettement. Trois droites sont ajustées sur les mêmes 100 points, dont 10 mesures faussées :

| Perte minimisée | MSE d'entraînement (la perte du seul modèle MSE) | Métrique : erreur absolue moyenne sur 1 000 points propres |
|---|---|---|
| MSE | 44,09 (la meilleure) | 4,13 |
| MAE | 62,36 | 0,844 |
| Huber (δ = 1) | 62,36 | 0,843 |

Le modèle MSE a fait exactement ce qu'on lui demandait : il a la plus petite MSE. Sur ce qui compte ici, il est cinq fois moins bon. Le problème ne vient pas de l'optimisation, mais de la question posée.

### Chaque perte est une hypothèse sur le bruit

Le lien qui organise tout le chapitre : minimiser une perte revient presque toujours à maximiser une vraisemblance. Le principe du maximum de vraisemblance (choisir les paramètres qui rendent les données observées les plus probables) est présenté au chapitre 00 ; on ne garde ici que sa conséquence pour les pertes.

On suppose que la réponse observée $y$ est la prédiction $\hat y = f_\theta(x)$ plus un bruit tiré d'une certaine loi. Maximiser le produit des probabilités des $n$ exemples revient à minimiser la somme des opposés de leurs logarithmes (le logarithme transforme le produit en somme et ne change pas où se trouve le maximum) :

(Formule : Perte = log-vraisemblance négative.)

Exemple complet avec un bruit gaussien d'écart-type $\sigma$ fixé : $p(y \mid x) = \frac{1}{\sqrt{2\pi}\,\sigma} \exp\!\bigl(-\frac{(y - \hat y)^2}{2\sigma^2}\bigr)$. Son opposé du logarithme vaut

$$
-\log p(y \mid x) = \frac{(y - \hat y)^2}{2\sigma^2} + \log\bigl(\sqrt{2\pi}\,\sigma\bigr).
$$

Le second terme ne dépend pas de $\theta$, le facteur $\frac{1}{2\sigma^2}$ non plus : minimiser cette quantité revient à minimiser $(y - \hat y)^2$. La MSE, c'est le maximum de vraisemblance sous un bruit gaussien. Les autres correspondances se démontrent de la même façon :

| Loi supposée pour $y$ | $-\log p$, à des constantes près | Perte obtenue | Meilleure constante |
|---|---|---|---|
| Gaussienne autour de $\hat y$ | $(y - \hat y)^2$ | MSE | la moyenne |
| Laplace autour de $\hat y$ (densité $\propto e^{-\lvert y - \hat y\rvert / b}$) | $\lvert y - \hat y\rvert$ | MAE | la médiane |
| Bernoulli de paramètre $p$ (deux classes) | $-[y \log p + (1-y)\log(1-p)]$ | entropie croisée binaire | la fréquence de la classe 1 |
| Catégorielle de paramètres $p_1, …, p_K$ | $-\log p_c$ ($c$ = bonne classe) | entropie croisée | les fréquences des classes |

Ce tableau dit quand une perte est le bon choix : la MSE suppose un bruit gaussien, aux queues très fines, pour lequel un écart de 10 écarts-types est quasi impossible. Face à un tel écart, elle préfère déplacer tout le modèle plutôt que de l'« admettre ». La loi de Laplace a des queues plus épaisses, et la MAE tolère donc mieux les écarts extrêmes. La perte de Huber est elle aussi un $-\log p$ : celui de la loi « la moins favorable » que Huber exhibe en 1964, gaussienne au centre et à queues exponentielles (de Laplace) au-delà du seuil. Ce choix rend l'estimateur minimax face à une gaussienne « contaminée » par une proportion $\varepsilon$ de valeurs quelconques : sa plus mauvaise performance, sur toutes ces contaminations, est la meilleure possible.

### Régression : MSE, MAE, Huber

Notation : le résidu $r = \hat y - y$, écart entre la prédiction et la réponse attendue. Une phrase résume la section : la dérivée de la perte par rapport à $\hat y$ dit avec quelle force chaque exemple tire sur le modèle. Dans tout le chapitre, on appelle force l'opposé de cette dérivée, $-\ell'$ : la descente de gradient déplace la prédiction dans ce sens, et la mise à jour des paramètres est une somme de ces forces.

#### Erreur quadratique moyenne (MSE)

L'erreur quadratique moyenne est lisse, sa dérivée est simple, et pour un modèle linéaire elle a une solution exacte (les moindres carrés, chapitre 00). Son défaut tient dans sa dérivée $2r$ : un point dont le résidu vaut 25 tire 25 fois plus fort qu'un point dont le résidu vaut 1 (force de 50 contre 2, en valeur absolue). Une valeur aberrante domine donc le gradient. Pour un modèle constant $c$, annuler la somme des forces $\sum_i 2(y_i - c) = 0$ donne $c = \frac{1}{n}\sum_i y_i$ : la moyenne.

#### Erreur absolue moyenne (MAE, perte L1)

Avec l'erreur absolue moyenne, chaque point tire avec la même force, quelle que soit son erreur : il ne vote que pour le sens (« monte » ou « descends »). Le point d'équilibre d'un modèle constant est celui où il y a autant de points au-dessus qu'en dessous : la médiane. Pour les salaires, 2 600 €, que la dirigeante gagne 25 000 € ou 250 000 €. Le prix à payer : la dérivée saute de −1 à +1 en zéro (on se contente alors d'un *sous-gradient* (*subgradient*) : en zéro, n'importe quelle pente entre −1 et +1 fait l'affaire), et elle ne diminue pas à l'approche de l'optimum, donc la descente de gradient doit réduire son pas pour converger.

#### Perte de Huber

La perte de Huber recolle une parabole et deux demi-droites de façon que la courbe et sa dérivée soient continues. Près de zéro, elle se comporte comme la MSE (lisse, convergence douce) ; loin, comme la MAE (force bornée à δ). Le facteur ½ sert justement à raccorder les deux morceaux : en $r = \delta$, les deux formules valent $\tfrac12\delta^2$ et ont la même pente $\delta$.

Calcul à la main pour les salaires, avec δ = 1 000 € : à l'optimum (vers 2 700 €), les 10 salaires « normaux » sont à moins de 1 000 € de la constante $c$ et exercent la force $y_i - c$ ; la dirigeante est au-delà du seuil et exerce la force écrêtée +1 000 (elle tire vers le haut). L'équilibre s'écrit $\sum_{i=1}^{10}(y_i - c) + 1\,000 = 0$, soit $25\,900 - 10\,c + 1\,000 = 0$, donc $c = 2\,690$ €. Le salaire extrême ne pèse que comme un écart de 1 000 €, quelle que soit sa valeur.

#### Ce que ça donne sur une régression

Dans l'exercice, 100 points suivent $y = 2x + 1$ plus un petit bruit, mais les 10 points de plus grand $x$ ont une mesure faussée de −25 (un capteur défaillant). On ajuste la même droite avec les trois pertes, par descente de gradient en numpy :

| Perte | Pente (vraie : 2) | Ordonnée à l'origine (vraie : 1) |
|---|---|---|
| MSE | 0,589 | 5,35 |
| MAE | 1,923 | 1,11 |
| Huber (δ = 1) | 1,918 | 1,17 |

Dix points sur cent ont suffi à coucher la droite MSE : pour réduire leurs résidus de 25, elle accepte des résidus moyens sur les 90 autres. MAE et Huber retrouvent la vraie droite. scikit-learn donne les mêmes pentes (0,589 avec `LinearRegression`, 1,923 avec `QuantileRegressor` à la médiane). `HuberRegressor` donne 1,914 : son seuil `epsilon=1.35` s'applique à des résidus divisés par une échelle de bruit qu'il estime lui-même, ce n'est donc pas exactement δ = 1.

### Classification binaire : de la perte 0-1 à la log-loss et à la hinge

Deux classes, étiquetées $y \in \{-1, +1\}$. Le modèle produit un score réel $f(x)$ (par exemple $f(x) = w \cdot x + b$) et prédit la classe du signe de ce score. La quantité clé est la marge :

$$
m = y \, f(x)
$$

$m > 0$ : l'exemple est du bon côté de la frontière ; $m \le 0$ : il est mal classé ; plus $m$ est grand, plus le modèle est sûr de lui et a raison. Toutes les pertes de classification binaire de cette section s'écrivent comme une fonction de la marge seule, ce qui permet de les comparer sur un même axe.

| Perte | En fonction de la marge $m$ | Force sur un exemple ($-\ell'(m)$) | Exemple à $m = 5$ (très bien classé) |
|---|---|---|---|
| 0-1 (la métrique) | 1 si $m \le 0$, sinon 0 | 0 partout (pas de gradient) | perte 0 |
| carré (moindres carrés sur ±1) | $(1 - m)^2$ | $2(1 - m)$ : négative si $m > 1$ | perte 16, force −8 |
| log-loss | $\log(1 + e^{-m})$ | $\sigma(-m)$, entre 0 et 1 | perte 0,0067, force 0,0067 |
| hinge | $\max(0, 1 - m)$ | 1 si $m < 1$, sinon 0 | perte 0, force 0 |

Ici $\sigma(z) = \frac{1}{1 + e^{-z}}$ est la sigmoïde (son rôle de fonction d'activation est traité au chapitre 07).

Le carré, c'est-à-dire la régression linéaire appliquée aux étiquettes ±1, semble raisonnable. Mais comme $y^2 = 1$, $(f - y)^2 = (y f - 1)^2 = (m - 1)^2$ : il pénalise l'écart à une marge de exactement 1, des deux côtés. Un point très bien classé ($m = 5$) coûte 16, autant qu'un point franchement mal classé ($m = -3$). Pour réduire ce coût, l'optimisation rapproche ce point de la frontière, quitte à la tordre.

#### Log-loss et régression logistique

La régression logistique transforme le score en probabilité par la sigmoïde, $p = \sigma(z)$ avec $z = w \cdot x + b$, et minimise l'opposé de la log-vraisemblance de Bernoulli : l'entropie croisée binaire, ou log-loss. Le score $z$ est un logit : on retrouve $z = \log\frac{p}{1 - p}$, le logarithme de la cote (*odds*, le rapport $p/(1-p)$ : une cote de 3 veut dire « trois chances contre une »).

Deux propriétés expliquent sa domination :

1. Le gradient est simple et borné. $\frac{\partial \ell}{\partial z} = p - y$ est compris entre −1 et 1, et le gradient par rapport aux poids vaut $(p - y)\,x$. Un exemple mal classé avec assurance ($p \approx 0$ alors que $y = 1$) tire avec la force maximale, 1 ; un exemple bien classé avec assurance ne tire presque plus.
2. La sortie peut être une probabilité bien calibrée, c'est-à-dire fiable au sens des fréquences : si le modèle dit 0,8 sur beaucoup d'exemples, environ 80 % d'entre eux devraient être de la classe 1 (sous réserve que le modèle soit bien spécifié et non surajusté).

Vérification avec les deux codages : si $y = 1$ (marge $m = z$), $-\log \sigma(z) = \log(1 + e^{-z})$ ; si $y = 0$ (étiquette −1, marge $m = -z$), $-\log(1 - \sigma(z)) = -\log \sigma(-z) = \log(1 + e^{z})$. C'est bien $\log(1 + e^{-m})$ dans les deux cas.

#### Hinge, la perte des SVM

La perte charnière fait comme la log-loss loin à gauche (linéaire pour les erreurs), mais devient exactement nulle dès que $m \ge 1$. Conséquence : seuls les exemples mal classés ou trop proches de la frontière (les vecteurs de support, *support vectors*) influencent le modèle ; tous les autres pourraient être déplacés sans rien changer, tant qu'ils restent au-delà de la marge ($m > 1$). C'est la perte des machines à vecteurs de support, dont la géométrie (marge maximale, noyaux) est expliquée au chapitre 08. Cortes et Vapnik ne parlent pas de « hinge » en 1995 : ils écrivent des contraintes avec des variables d'écart $\xi_i \ge 1 - y_i f(x_i)$, $\xi_i \ge 0$, dont la somme minimale est exactement la somme des pertes charnières. L'article pénalise en fait $F\bigl(\sum_i \xi_i\bigr)$, avec $F$ convexe croissante : le corps du texte et les expériences prennent $F(u) = u^2$, l'annexe traite aussi $F(u) = u$, et c'est ce dernier cas, $C \sum_i \xi_i$, qui est devenu la forme standard.

#### Ce que ça donne sur une classification

L'exercice entraîne un classifieur linéaire sur deux nuages de 100 points (centres (−2, 0) et (2, 0)), puis sur les mêmes données plus 20 points de classe +1 placés très loin, autour de (12, 3), du bon côté. Ces points ne sont pas des erreurs : ils sont faciles. Log-loss et hinge sont entraînées par descente de gradient avec une petite pénalité L2 sur les poids (λ = 0,01) ; les moindres carrés, par leur solution exacte, sans pénalité. On mesure l'exactitude (proportion de bonnes réponses) sur 2 000 points de test :

| Perte | Sans les points lointains | Avec | Force moyenne des points lointains ($-\ell'(m)$) |
|---|---|---|---|
| Moindres carrés | 98,0 % | 88,8 % | −1,55 : les ramène vers la frontière |
| Log-loss | 98,0 % | 98,0 % | 1,5 × 10⁻¹² |
| Hinge | 98,0 % | 97,9 % | 0 |

Les moindres carrés punissent les exemples trop bien classés et inclinent la frontière vers eux. La log-loss leur donne un poids $\sigma(-m)$ quasi nul, la hinge un poids nul. Avec la hinge, 18 points sur 220 ont une marge inférieure à 1 : ce sont eux, et eux seuls, qui fixent la frontière. `SVC(kernel="linear")` de scikit-learn trouve aussi 18 vecteurs de support. C'est une bonne raison de préférer, en classification, des pertes conçues pour elle.

### Plusieurs classes : softmax, température, entropie croisée

Avec $K$ classes, le modèle produit un vecteur de $K$ scores, les logits $z_1, …, z_K$. La softmax les transforme en distribution de probabilité, et l'entropie croisée pénalise la probabilité donnée à la bonne classe :

Calcul complet avec les logits de l'exercice, $z = (2 ;\ 1 ;\ 0{,}1 ;\ -1)$ pour (chat, chien, lapin, poisson), la bonne classe étant « chat » :

```text
exponentielles : e^2 = 7,389   e^1 = 2,718   e^0,1 = 1,105   e^-1 = 0,368    somme = 11,580
probabilités   : p = (0,638 ; 0,235 ; 0,095 ; 0,032)
perte          : -log(0,638) = 0,449
gradient p - y : (0,638 - 1 ; 0,235 ; 0,095 ; 0,032) = (-0,362 ; 0,235 ; 0,095 ; 0,032)
```

Le gradient se lit directement : monter le logit du chat (composante négative, donc la descente l'augmente), baisser les autres en proportion de la probabilité qu'ils volent. L'exercice vérifie cette formule par différences finies (écart 8,6 × 10⁻¹¹) et retrouve les valeurs de PyTorch au dernier chiffre près. Ce « sortie moins cible » n'est pas un hasard. On le savait déjà pour des sorties sigmoïdes indépendantes ; John Bridle l'établit en 1989 pour la sortie softmax et conclut que « softmax output stages and 1-from-N RE scoring are natural partners » (RE pour *relative entropy* ; avec une cible « 1 parmi N », ce score se réduit à $-\log$ de la probabilité de la bonne classe, c'est-à-dire l'entropie croisée). C'est la même forme que pour la régression logistique ($K = 2$).

La température $T$ divise les logits avant la softmax. Avec les mêmes logits :

| Température | Probabilités | Entropie de la distribution |
|---|---|---|
| T = 0,5 (plus pointue) | (0,862 ; 0,117 ; 0,019 ; 0,002) | 0,468 nat |
| T = 1 | (0,638 ; 0,235 ; 0,095 ; 0,032) | 0,961 nat |
| T = 2 (plus plate) | (0,451 ; 0,274 ; 0,174 ; 0,101) | 1,249 nat |

Ici, le logarithme est népérien : les entropies sont en nats, l'unité naturelle des pertes calculées avec `log` (1 nat = 1/ln 2 ≈ 1,443 bit ; le chapitre 00 comptait en bits, avec $\log_2$). Ainsi 0,961 nat = 1,386 bit.

L'entropie (chapitre 00) mesure l'incertitude : elle vaut $\log 4 \approx 1{,}386$ nat pour une distribution uniforme sur 4 classes, 0 pour une certitude. Quand $T \to 0$, la softmax devient un « argmax » (tout sur le plus grand logit) ; quand $T \to \infty$, elle tend vers l'uniforme. La température revient au chapitre 28 pour l'échantillonnage des LLM et la distillation, et dans les pertes contrastives (chapitre 23).

Piège d'implémentation :
En PyTorch, `F.cross_entropy` (et `nn.CrossEntropyLoss`) attend les logits, pas les probabilités : elle applique elle-même la softmax, sous une forme numériquement stable. Lui donner des probabilités revient à appliquer deux softmax, et l'entraînement marche mal sans planter. De même, `F.binary_cross_entropy_with_logits` est préférable à une sigmoïde suivie de `F.binary_cross_entropy`.

### Divergence de Kullback-Leibler

Quand la cible n'est pas une classe unique mais une distribution $P$ (par exemple les probabilités d'un modèle « maître » en distillation), on compare deux distributions avec la divergence KL. Rappel du chapitre 00, où elle est définie : $D_{\mathrm{KL}}(P\|Q) = H(P, Q) - H(P)$, positive ou nulle, non symétrique, et, comme $H(P)$ ne dépend pas du modèle, la minimiser par rapport à $Q$ revient à minimiser l'entropie croisée. C'est pourquoi on entraîne avec l'entropie croisée tout en parlant de « rapprocher deux distributions ».

Ce qui compte pour l'usage comme perte, c'est le sens. Avec $P$ = softmax des logits ci-dessus et $Q$ uniforme sur 4 classes, l'exercice obtient $D_{\mathrm{KL}}(P\|Q) = 0{,}426$ et $D_{\mathrm{KL}}(Q\|P) = 0{,}538$ nat : deux pertes différentes, qui ne poussent pas le modèle de la même façon. La distillation (chapitre 28), l'ELBO des VAE (chapitre 09) et la pénalité qui retient un LLM près de son modèle de départ en RLHF (chapitre 30) choisissent chacune un sens. Vérification à la main du premier chiffre : $H(P, Q) = \log 4 = 1{,}386$ et $H(P) = 0{,}961$, d'où $1{,}386 - 0{,}961 = 0{,}425$, aux arrondis près.

### Le code : toutes les pertes en quelques lignes

Les pertes et leurs dérivées en numpy, dans une version simplifiée de l'exercice : celui-ci centre-réduit $x$ et fait décroître le pas de la MAE, ce qui donne les mêmes pentes à 0,002 près. La boucle d'entraînement ne change pas d'une perte à l'autre : seule la dérivée change.

```python
import numpy as np

# --- Régression : r = y_pred - y -------------------------------------------
def mse(r):          return np.mean(r2),        2 * r
def mae(r):          return np.mean(np.abs(r)),   np.sign(r)
def huber(r, d=1.0):
    a = np.abs(r)
    perte = np.where(a <= d, 0.5 * r2, d * (a - 0.5 * d))
    return np.mean(perte), np.clip(r, -d, d)        # dérivée = résidu écrêté

# --- Classification binaire : m = y * f(x), y dans {-1, +1} -----------------
def log_loss(m):     return np.mean(np.logaddexp(0, -m)), -np.exp(-np.logaddexp(0, m))  # stable : -σ(-m)
def hinge(m):        return np.mean(np.maximum(0, 1 - m)), -(m < 1).astype(float)

# --- Multiclasse : logits z de forme (n, K), classes c de forme (n,) ---------
def entropie_croisee(z, c):
    z = z - z.max(axis=1, keepdims=True)              # stabilité : softmax inchangée
    log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))   # log-softmax
    p = np.exp(log_p)
    y = np.eye(z.shape[1])[c]                          # cibles one-hot
    return -np.mean(log_p[np.arange(len(c)), c]), (p - y) / len(c)

# --- Une seule boucle pour toutes les pertes de régression -----------------
def ajuster(x, y, perte, pas=0.01, iterations=5000):
    a, b = 0.0, 0.0                                    # droite y = a x + b
    for _ in range(iterations):
        _, derivee = perte(a * x + b - y)              # dérivée par point
        a -= pas * np.mean(derivee * x)                # règle de la chaîne
        b -= pas * np.mean(derivee)
    return a, b
```

Retirer le maximum des logits avant l'exponentielle ne change pas la softmax (le facteur $e^{-\max}$ se simplifie) mais évite les débordements : $e^{1000}$ vaut l'infini en virgule flottante. `np.logaddexp(0, -m)` calcule $\log(1 + e^{-m})$ sans débordement pour les grandes marges négatives ; de même, $\sigma(-m) = e^{-\log(1 + e^{m})}$ évite le débordement de `np.exp(m)` pour les grandes marges positives.

### Les autres pertes du parcours

Le chapitre 01 sert de carrefour : les pertes suivantes sont expliquées dans leur chapitre propriétaire. Beaucoup sont des variations sur l'entropie croisée ou la MSE vues plus haut.

| Perte | Idée en une ligne | Parente | Chapitre |
|---|---|---|---|
| Focal loss | entropie croisée multipliée par $(1 - p)^\gamma$ : les exemples faciles ne comptent presque plus | entropie croisée | 14 |
| Contrastive, InfoNCE | entropie croisée où les « classes » sont les autres exemples du lot ; sensible à la température | entropie croisée | 15, 23 |
| Erreur TD, gradient de politique | MSE vers une cible estimée ; $-A \log \pi(a \mid s)$, une log-loss pondérée par l'avantage ($\pi$ : la politique, probabilité de l'action $a$ dans l'état $s$ ; $A$ : l'avantage, combien l'action a fait mieux que prévu) | MSE, log-loss | 11, 17 |
| Objectif clipé PPO | gain $\rho A$, avec le ratio $\rho = \pi_{\text{nouveau}}(a \mid s) / \pi_{\text{ancien}}(a \mid s)$ (noté $r$ dans l'article), écrêté quand $\rho$ sort de $[1 - \varepsilon, 1 + \varepsilon]$ dans le sens qui augmenterait le gain | — | 17 |
| Reconstruction, ELBO | reconstruction (MSE ou entropie croisée) plus une KL vers une loi a priori | MSE, KL | 18 (ELBO : 09) |
| Adversariale (GAN) | entropie croisée binaire d'un discriminateur, que le générateur cherche à tromper | entropie croisée binaire | 18 |
| Token suivant, perplexité | entropie croisée sur le vocabulaire ; perplexité = exponentielle de sa moyenne | entropie croisée | 20 |
| ε-prédiction, flow matching | MSE entre le bruit (ou la vitesse) prédit et le vrai | MSE | 25 |
| JEPA | distance entre représentations prédites et cibles, dans un espace latent | MSE, L1 | 26 |
| Distillation | KL entre maître et élève, tous deux à température T, multipliée par T² | KL | 28 |
| Bradley-Terry, RLHF, DPO | $-\log \sigma(\text{marge})$ entre réponse préférée et rejetée | log-loss | 30 |

Fiches courtes des pertes dont le chapitre propriétaire est déjà écrit :

Six des pertes du tableau, tracées sur des exemples jouets :

## Les pertes côte à côte

La figure qui résume le chapitre : à gauche les pertes, à droite leurs dérivées, dont l'opposé est la force exercée par un exemple en fonction de son erreur. En régression, la dérivée de la MSE grandit sans limite, celles de la MAE et de Huber sont bornées. En classification, la dérivée du carré redevient positive au-delà d'une marge de 1, ce qui punit les exemples trop bien classés ; celles de la log-loss et de la hinge s'annulent.

À vous de manipuler. Cochez les pertes, affichez leurs dérivées, déplacez le curseur (ou touchez le graphique) pour lire les valeurs, et faites varier le seuil δ de Huber : avec δ grand, Huber rejoint la MSE (à un facteur ½ près) ; avec δ petit, elle ressemble à une MAE multipliée par δ.

La même démo pour la classification, en fonction de la marge $m$ :

## Ce qui coinçait

Choisir une perte règle la question « quoi apprendre », pas toutes les difficultés :

- La perte n'est qu'un substitut de l'objectif réel. On minimise l'entropie croisée en espérant un bon taux d'erreur, ou la perplexité en espérant un bon assistant. L'écart entre les deux ne se voit qu'avec une métrique mesurée à part, sur des données de validation (chapitre 00). Aux chapitres 30 et 39, ce décalage devient central : un modèle peut maximiser sa récompense sans faire ce qu'on voulait.
- Les pertes robustes sont moins commodes. La MAE n'est pas dérivable en zéro et sa force ne diminue pas près de l'optimum ; Huber demande de régler δ, qui dépend de l'échelle du bruit.
- La log-loss pousse à l'excès de confiance. Sur des données séparables, elle continue de diminuer tant que les poids grandissent : sans régularisation (L2, arrêt précoce, chapitre 00), les poids partent à l'infini. Guo et al. ont montré en 2017 (« On Calibration of Modern Neural Networks », ICML) que les réseaux profonds modernes entraînés à l'entropie croisée sont souvent mal calibrés, en général trop sûrs d'eux : leur confiance affichée dépasse leur taux de réussite réel. Le lissage des étiquettes (chapitre 12) atténue ce défaut ; la mesure de la calibration est au chapitre 39.
- Les classes rares disparaissent dans la moyenne. Une perte moyenne sur 10 000 exemples de fond et 10 objets à détecter est dominée par le fond. La focal loss (chapitre 14) répond à ce problème.
- La hinge ne donne pas de probabilité, seulement un score et une décision.
- Choisir la perte ne dit pas comment la minimiser. Sur un réseau profond, la perte n'est pas convexe, a des millions de paramètres, et on ne peut la calculer que sur des petits lots de données : c'est le travail de l'optimisation.

- Moindres carrés (Legendre, Gauss, 1805-1809) : Minimiser la somme des carrés des écarts : simple, exact pour un modèle linéaire, mais sensible aux valeurs aberrantes.
- Maximum de vraisemblance (Fisher, 1922) : Une perte = −log de la probabilité des données : chaque perte avoue une hypothèse sur le bruit (gaussien → MSE, Laplace → MAE).
- Régression logistique (Berkson 1944, Cox 1958) : Probabilité par sigmoïde et log-loss : gradient p − y, les exemples faciles ne pèsent plus.
- Divergence KL (Kullback et Leibler, 1951) : Comparer deux distributions ; à cible fixée, la minimiser revient à minimiser l’entropie croisée.
- Perte de Huber (1964) : Carré près de zéro, linéaire au-delà : la force de chaque point est bornée.
- Softmax et entropie croisée (Bridle, 1989) : K logits deviennent K probabilités ; le gradient vaut encore sortie − cible.
- Hinge et SVM à marge souple (Cortes et Vapnik, 1995) : Perte nulle au-delà d’une marge de 1 : seuls les vecteurs de support comptent.
- Pertes modernes (InfoNCE, DPO, diffusion…) : Presque toutes recombinent entropie croisée, MSE et KL.

Et ensuite ? Une perte étant choisie, il reste à trouver les paramètres qui la rendent minimale. Avec une droite et la MSE, une formule suffit ; avec un réseau de millions de paramètres, il faut descendre la pente pas à pas, avec des gradients bruités et des surfaces pleines de vallées étroites et de points selles. C'est l'objet du chapitre 02 : Optimisation.

## Exercice

Ce qu'il faut observer :

- Moyenne contre médiane : un seul salaire extrême fait passer la meilleure constante MSE de 2 590 € (moyenne des dix autres) à 4 627 € ; la MAE reste à 2 600 €.
- La droite MSE couchée par 10 % d'aberrants (pente 0,589 au lieu de 2), alors que MAE et Huber restent vers 1,92 ; et pourtant le modèle MSE a la meilleure MSE d'entraînement.
- Les moindres carrés punissent les exemples trop bien classés : 98,0 % → 88,8 % d'exactitude quand on ajoute 20 points faciles, contre 98,0 % → 98,0 % (log-loss) et 98,0 % → 97,9 % (hinge).
- Le gradient de l'entropie croisée vaut p − y, vérifié par différences finies, et six pertes numpy coïncident avec celles de PyTorch.

Questions de réflexion : quelle constante obtiendrait Huber si δ devenait très grand, ou très petit ? Dans quelle situation réelle préférer malgré tout la MSE ? Pourquoi le gradient $p - y$, borné entre −1 et 1, est-il préférable à celui d'une MSE appliquée aux probabilités pour un exemple très mal classé ?

## Auto-évaluation

Question : Pourquoi n'entraîne-t-on pas un classifieur en minimisant directement son taux d'erreur ?

Réponse : Le taux d'erreur est la moyenne de la perte 0-1, une fonction en escalier : une petite modification des poids ne change en général aucune prédiction, donc sa dérivée est nulle presque partout (et non définie aux sauts). La descente de gradient n'a aucune pente à suivre. On minimise une perte de substitution lisse (log-loss, entropie croisée, hinge) et on garde le taux d'erreur comme métrique de validation.

Question : On résume les salaires 2 000, 2 500, 3 000 et 20 000 € par une seule valeur. Que donnent la MSE et la MAE, et pourquoi ?

Réponse : La MSE donne la moyenne, 6 875 € : sa dérivée $2(c - y_i)$ fait tirer chaque salaire en proportion de son écart, et 20 000 € tire très fort. La MAE donne la médiane : avec 4 valeurs, toute constante entre 2 500 et 3 000 € convient (autant de points au-dessus qu'en dessous, chaque point tirant avec une force ±1). Le salaire extrême n'y compte que comme « un point au-dessus ».

Question : Quelle hypothèse sur les données fait-on, sans toujours le savoir, en entraînant une régression avec la MSE ?

Réponse : Que le bruit autour de la prédiction est gaussien (de variance constante) : la MSE est l'opposé de la log-vraisemblance gaussienne, à des constantes près. Une gaussienne rend les très grands écarts quasi impossibles, donc la MSE déplace tout le modèle pour les réduire. Avec des valeurs aberrantes, un bruit à queues plus épaisses (Laplace → MAE) ou la perte de Huber sont plus réalistes.

Question : Logits (3 ; 1 ; 0) et bonne classe 1 (la première). Que vaut le gradient de l'entropie croisée par rapport aux logits ? Et si on divise les logits par une température T = 10 ?

Réponse : $e^3 = 20{,}09$, $e^1 = 2{,}72$, $e^0 = 1$, somme $23{,}80$ : $p \approx (0{,}844 ;\ 0{,}114 ;\ 0{,}042)$. Le gradient vaut $p - y \approx (-0{,}156 ;\ 0{,}114 ;\ 0{,}042)$ : la descente monte le premier logit et baisse les autres. Avec $T = 10$, les logits deviennent (0,3 ; 0,1 ; 0) et la distribution est presque uniforme, environ (0,39 ; 0,32 ; 0,29) : la température aplatit la softmax.

Question : Pourquoi peut-on dire indifféremment qu'on minimise l'entropie croisée ou la divergence KL entre les données et le modèle ?

Réponse : Parce que $D_{\mathrm{KL}}(P\|Q) = H(P, Q) - H(P)$, et que l'entropie $H(P)$ des données ne dépend pas du modèle $Q$. Les deux quantités diffèrent d'une constante et ont le même minimum. Attention au sens : la KL n'est pas symétrique ($D_{\mathrm{KL}}(Q\|P)$ est une autre perte, aux propriétés différentes).

## Pour aller plus loin

- Articles originaux : Fisher (1922) pour le maximum de vraisemblance, Huber (1964), Kullback et Leibler (1951), Berkson (1944) et Cox (1958) pour la régression logistique, Bridle (1989) pour la softmax, Cortes et Vapnik (1995) pour la marge souple : liens dans les fiches ci-dessus.
- Stanford CS231n, « Linear classification » : perte des SVM multiclasses et classifieur softmax comparés sur le même exemple, avec une démo interactive.
- Stanford CS229, notes de cours (A. Ng, T. Ma), chapitres 1 à 3 : moindres carrés et vraisemblance gaussienne, régression logistique, modèles linéaires généralisés et softmax.
- Dive into Deep Learning, « Softmax Regression » : softmax, entropie croisée et son gradient, avec le code.
- K. Murphy, *Probabilistic Machine Learning: An Introduction* (livre en ligne, MIT Press 2022) : maximum de vraisemblance, régression robuste et pertes de classification, en détail.
- A. Karpathy, « The spelled-out intro to language modeling: building makemore » (vidéo de la série Neural Networks: Zero to Hero) : la log-vraisemblance négative construite à la main, puis comme perte d'un réseau.
- S. Stigler, « The Epic Story of Maximum Likelihood » (*Statistical Science*, 2007) : l'histoire du maximum de vraisemblance, de Laplace et Gauss à Fisher.
- Documentation PyTorch des pertes et exemple scikit-learn « HuberRegressor vs Ridge » sur des données avec valeurs aberrantes.
