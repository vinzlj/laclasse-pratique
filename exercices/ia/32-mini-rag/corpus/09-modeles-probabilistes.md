# 09. Modèles probabilistes et variables latentes

Repère historique (1953 (Metropolis) à 1999 (inférence variationnelle) ; en IA, années 1980 (Pearl)) : Raisonner sous incertitude et modéliser des causes cachées, là où les règles des systèmes experts échouaient : combiner des indices qui se contredisent ou dépendent les uns des autres, et deviner ce qu'on n'observe pas (le groupe d'un point, la météo derrière un comportement) à partir de ce qu'on observe.

Au milieu des années 1980, les systèmes experts du chapitre 05 gèrent l'incertitude avec des facteurs de certitude : des conventions de calcul commodes, mais incohérentes dès que deux indices dépendent l'un de l'autre. Judea Pearl, à l'université de Californie à Los Angeles (UCLA), propose de prendre les probabilités au sérieux et de les rendre calculables grâce à un graphe : ce sont les réseaux bayésiens, réunis dans son livre de 1988. L'IA découvre alors des outils que la physique, l'ingénierie et la statistique avaient forgés pour leurs propres besoins : l'échantillonnage de Metropolis (1953, Los Alamos), le filtre de Kalman (1960), l'algorithme de Viterbi (1967, codes correcteurs), l'algorithme de Baum-Welch (1970), l'algorithme EM, pour « espérance-maximisation » (1977, Harvard), l'échantillonneur de Gibbs (1984, restauration d'images). Au cours des années 1990 (premières procédures dès 1987, synthèse de Jordan et ses coauteurs en 1999), l'inférence variationnelle transforme le calcul de probabilités en problème d'optimisation. En 2011, Pearl reçoit le prix Turing « pour ses contributions fondamentales à l'intelligence artificielle par le développement d'un calcul du raisonnement probabiliste et causal ».

## L'idée en une image

Un matin, la pelouse est mouillée. Il a plu ? Ou l'arroseur automatique a tourné ? Vous ne voyez ni la pluie ni l'arroseur : seulement l'herbe. Pourtant vous raisonnez sans effort. L'herbe mouillée rend la pluie plus probable ; si le voisin vous dit que son arroseur a tourné cette nuit, la pluie redevient moins probable : l'arroseur suffit à expliquer l'herbe mouillée. Ce petit raisonnement contient tout le chapitre :

- il y a des causes cachées (pluie, arroseur) et des effets observés (herbe mouillée). On appelle variables latentes ce qu'on ne voit jamais directement ;
- chaque relation est incertaine : la pluie mouille l'herbe 8 fois sur 10, pas toujours ;
- on veut inverser les flèches : passer de « la cause produit l'effet » (facile à décrire) à « vu l'effet, quelle cause ? » (ce qu'on veut savoir).

L'outil qui fait l'inversion est la règle de Bayes, déjà croisée au chapitre 00 pour le maximum a posteriori :

$$
P(\text{cause} \mid \text{effet}) = \frac{P(\text{effet} \mid \text{cause})\; P(\text{cause})}{P(\text{effet})}.
$$

Le résultat s'appelle la loi a posteriori ; le dénominateur, la probabilité de ce qu'on a observé toutes causes confondues, s'appelle l'évidence. Le chapitre raconte une seule difficulté sous plusieurs formes : ce dénominateur est une somme sur toutes les combinaisons de causes cachées, et ce nombre de combinaisons explose. Quatre familles d'algorithmes s'en accommodent.

| Question | Exemple du chapitre | Algorithme | Idée |
|---|---|---|---|
| Quelle cause, vu l'effet ? | pluie ou arroseur ? | énumération, propagation dans un graphe | la structure du graphe évite de tout énumérer |
| Quels groupes cachés dans ces données ? | 900 points, 3 groupes inconnus | EM (détaillé plus bas) sur un mélange de gaussiennes | deviner les groupes, ajuster, recommencer |
| Quelle suite de causes, vu une suite d'effets ? | la météo d'après vos trajets | forward-backward, Viterbi, Baum-Welch | programmation dynamique sur le temps |
| Et quand le calcul exact est hors de portée ? | loi des paramètres d'une régression | méthodes MCMC (détaillées plus bas) ou inférence variationnelle | tirer des exemples, ou approcher par une loi simple |

Le tableau « avant / après » de ce chapitre oppose deux façons de gérer l'incertitude :

| | Avant : facteurs de certitude (MYCIN, années 1970) | Après : modèles probabilistes (Pearl, 1988 ; EM, HMM…) |
|---|---|---|
| Représentation | des règles « SI… ALORS… » avec un coefficient entre −1 et 1 | une loi de probabilité sur toutes les variables, décrite par un graphe |
| Combiner deux indices | une formule fixe, qui suppose les indices indépendants | la règle de Bayes, qui tient compte des dépendances |
| Raisonner « à l'envers » (de l'effet à la cause) | une autre règle, écrite à la main | automatique : c'est la même loi, interrogée dans l'autre sens |
| Variables jamais observées | impossible | au cœur du modèle (groupes, états cachés, thèmes) |
| Apprendre des données | non : un expert écrit tout | oui : maximum de vraisemblance, EM, Baum-Welch |
| Ce qui coince | incohérences, explosion du nombre de règles | coût du calcul exact, structure du graphe souvent choisie à la main |

## Le mécanisme

### Réseaux bayésiens : factoriser la loi jointe

Trois variables oui/non (pluie $P$, arroseur $A$, herbe mouillée $H$) ont $2^3 = 8$ combinaisons. Donner la probabilité de chacune, la loi jointe, suffit à répondre à toutes les questions. Avec 30 variables, il faudrait plus d'un milliard de nombres. Un réseau bayésien évite ce mur : il dessine qui influence qui, et ne stocke pour chaque variable que sa probabilité sachant ses parents.

Ici, l'arroseur dépend de la pluie (programmé pour ne pas tourner s'il pleut), et l'herbe des deux. La loi jointe est le produit des tables, $P(p, a, h) = P(p)\,P(a \mid p)\,P(h \mid a, p)$, cas particulier de la règle générale :

(Formule : Factorisation d'un réseau bayésien.)

Ce que le graphe ne dessine pas compte autant que ce qu'il dessine : une flèche absente est une hypothèse d'indépendance. Un modèle graphique probabiliste sert précisément à ça : rendre une loi sur beaucoup de variables compacte, lisible et calculable.

Exemple complet à la main, avec les tables de la figure. Les 8 produits $P(p)\,P(a \mid p)\,P(h \mid a, p)$ :

```text
pluie  arroseur  herbe   produit                        probabilité
 oui     oui      oui    0,2 × 0,01 × 0,99            = 0,00198
 oui     non      oui    0,2 × 0,99 × 0,8             = 0,15840
 non     oui      oui    0,8 × 0,4  × 0,9             = 0,28800
 non     non      oui    0,8 × 0,6  × 0,0             = 0
 (les 4 lignes « herbe sèche » complètent le total à 1)
```

- $P(\text{herbe}) = 0{,}00198 + 0{,}1584 + 0{,}288 = 0{,}44838$ : c'est l'évidence.
- $P(\text{pluie} \mid \text{herbe}) = (0{,}00198 + 0{,}1584) / 0{,}44838 = 0{,}3577$. Voir l'herbe mouillée fait passer la pluie de 20 % à 36 %.
- $P(\text{pluie} \mid \text{herbe}, \text{arroseur}) = 0{,}00198 / (0{,}00198 + 0{,}288) = 0{,}0068$. Apprendre que l'arroseur a tourné fait retomber la pluie à 0,7 %.

Attention, ce dernier chiffre mélange deux effets. Dans ce réseau, l'arroseur dépend déjà de la pluie (il ne tourne presque jamais quand il pleut) : sans même regarder l'herbe, savoir qu'il a tourné ramène la pluie à $P(\text{pluie} \mid \text{arroseur}) = 0{,}002 / 0{,}322 = 0{,}0062$. La chute à 0,7 % vient presque entièrement de cette flèche pluie → arroseur.

Pour voir le phénomène qui nous intéresse à l'état pur, supprimons cette flèche : l'arroseur tourne avec la probabilité 0,4, qu'il pleuve ou non. Pluie et arroseur sont alors indépendants a priori. Avec la même table pour l'herbe :

| Ce qu'on sait | $P(\text{pluie} \mid \dots)$ |
|---|---|
| rien | 0,2 |
| l'arroseur a tourné (rien sur l'herbe) | 0,2 : l'arroseur ne dit rien de la pluie |
| l'herbe est mouillée | $0{,}1752 / 0{,}4632 = 0{,}378$ |
| l'herbe est mouillée et l'arroseur a tourné | $0{,}0792 / 0{,}3672 = 0{,}216$ |

C'est l'effet d’explication : deux causes indépendantes d'un même effet deviennent dépendantes dès qu'on observe cet effet, parce que l'une suffit à « expliquer » l'observation. Savoir l'arroseur en marche ne disait rien de la pluie ; sachant l'herbe mouillée, cela la fait retomber de 38 % à 22 %, presque à son niveau de départ. Aucune combinaison de facteurs de certitude ne reproduit ce comportement sans une règle écrite exprès.

Inférence exacte et son coût. Calculer une probabilité conditionnelle, c'est sommer la loi jointe sur les variables qu'on ne connaît pas et qui ne nous intéressent pas. L'énumération coûte $2^n$ termes pour $n$ variables cachées binaires. Sur un graphe en arbre, Pearl a montré qu'on peut faire circuler des messages locaux le long des arêtes (la propagation de croyances, *belief propagation*) et obtenir le résultat exact en un temps proportionnel à la taille du graphe ; sur un graphe quelconque, le calcul exact reste exponentiel dans le pire cas, et l'on passe aux méthodes approchées de la fin du chapitre.

Échantillonnage avec rejet. Autre idée : simuler le monde au lieu de calculer. On tire la pluie, puis l'arroseur sachant la pluie, puis l'herbe (on tire les parents avant les enfants), et l'on jette les tirages où l'herbe est sèche. L'échantillonnage avec rejet sur 100 000 tirages en garde 44,8 % et estime $P(\text{pluie} \mid \text{herbe}) \approx 0{,}3547$, contre 0,3577 exactement. Il gaspille 55 % du travail, et ce serait bien pire si l'observation était rare : avec une observation de probabilité $10^{-6}$, il faudrait un million de tirages pour en garder un seul. C'est ce gaspillage que les méthodes MCMC (détaillées plus bas) éviteront.

Champs de Markov. Toutes les dépendances n'ont pas de sens naturel : deux pixels voisins d'une image se ressemblent, sans que l'un « cause » l'autre. Un champ de Markov utilise un graphe non orienté et écrit la loi comme un produit de potentiels positifs $\psi_c$ sur des groupes de voisins $c$, divisé par une constante de normalisation $Z$ :

$$
p(x) = \frac{1}{Z} \prod_{c} \psi_c(x_c), \qquad Z = \sum_{x} \prod_c \psi_c(x_c).
$$

$Z$ est une somme sur toutes les configurations (sur toutes les images possibles), donc incalculable en pratique ; on retrouvera ce problème avec la machine de Boltzmann (chapitre 07) et les modèles à énergie (chapitre 26). Les méthodes MCMC de ce chapitre ont l'avantage de ne jamais avoir besoin de $Z$.

Corrélation n'est pas causalité :
Un réseau bayésien appris sur des données observées décrit des dépendances, pas des causes. Les ventes de glaces et les noyades montent ensemble, parce qu'un facteur de confusion (la chaleur) agit sur les deux ; interdire les glaces ne sauverait personne. Distinguer observer (« sachant qu'on a vendu beaucoup de glaces ») et intervenir (« si l'on force les ventes à zéro ») demande un modèle causal explicite. Pearl a formalisé cette distinction avec son opérateur $do(\cdot)$ : $P(\text{noyade} \mid do(\text{glaces} = 0))$ n'est pas $P(\text{noyade} \mid \text{glaces} = 0)$. Référence : J. Pearl, *Causality* (Cambridge University Press, 2e édition 2009).

### Mélanges de gaussiennes et algorithme EM

Le problème. 900 points en 2D forment visiblement trois groupes : deux bandes allongées et inclinées, et un amas rond. On ne connaît pas le groupe de chaque point. Le partitionnement par k-moyennes (k-means, chapitre 08) affecte chaque point au centre le plus proche : il suppose des groupes ronds et de même taille, et coupe les bandes en travers.

Le modèle génératif d'un mélange de gaussiennes raconte comment les points auraient été produits : pour chaque point, on tire un groupe caché $z \in \{1, \dots, K\}$ avec les probabilités $\pi_1, \dots, \pi_K$, puis on tire le point d'une gaussienne propre à ce groupe, de moyenne $\mu_k$ et de covariance $\Sigma_k$ (qui code la forme et l'orientation d'une ellipse).

(Formule : Mélange de K gaussiennes.)

Si l'on connaissait les groupes, ce serait facile : moyenne et covariance des points de chaque groupe. Si l'on connaissait les paramètres, ce serait facile aussi : la règle de Bayes donne pour chaque point la probabilité d'appartenir à chaque groupe. On ne connaît ni l'un ni l'autre. L'algorithme espérance-maximisation (on l'abrège EM) alterne les deux calculs faciles :

(Formule : Étape E d'EM pour un mélange de gaussiennes.)

(Formule : Étape M.)

La responsabilité est une affectation douce : un point à mi-chemin de deux groupes compte pour moitié dans chacun. C'est pourquoi on présente souvent EM comme des « k-moyennes douces ». Les k-moyennes sont d'ailleurs la limite d'EM quand toutes les covariances sont égales à $\sigma^2 I$ avec $\sigma \to 0$ : les responsabilités deviennent 0 ou 1.

Une itération à la main. Quatre points sur une droite, $x = (0 ;\ 1 ;\ 3 ;\ 4)$ ; deux composantes de poids ½ et de variance fixée à 1, initialisées en $\mu_1 = 1$ et $\mu_2 = 3$. Pour garder le calcul faisable à la main, on ne met à jour que les moyennes : c'est l'étape M ci-dessus, sans ses lignes $\pi_k$ et $\Sigma_k$.

```text
Étape E   point 0 : r1 = e^(-0,5) / (e^(-0,5) + e^(-4,5)) = 0,982   r2 = 0,018
          point 1 : r1 = e^0 / (e^0 + e^(-2))           = 0,881   r2 = 0,119
          point 3 : r1 = 0,119   r2 = 0,881     (par symétrie)
          point 4 : r1 = 0,018   r2 = 0,982
Étape M   n1 = 0,982 + 0,881 + 0,119 + 0,018 = 2,0
          μ1 = (0 × 0,982 + 1 × 0,881 + 3 × 0,119 + 4 × 0,018) / 2,0 = 0,655
          μ2 = 3,345                                (par symétrie)
Log-vraisemblance : −7,158 avant, −6,856 après    (elle a monté ; poids et variances fixés)
```

En répétant, les moyennes se stabilisent vers 0,56 et 3,44, un peu en deçà des centres des deux paires : chaque point garde une petite responsabilité dans l'autre composante. Si l'on met aussi à jour les variances, comme dans l'étape M complète, elles convergent vers 0,5 et 3,5, les centres des deux paires (les variances tombent alors à 0,25).

#### Pourquoi EM ne peut pas faire baisser la vraisemblance : l'ELBO

La justification d'EM est l'identité centrale de ce chapitre, qui reviendra pour l'inférence variationnelle, les VAE (chapitre 18) et la diffusion (chapitre 25). Pour n'importe quelle loi $q(z)$ sur les variables cachées :

(Formule : Log-vraisemblance = ELBO + KL.)

La démonstration tient en une ligne : $\log p(x, z) - \log q(z) = \log p(x) + \log p(z \mid x) - \log q(z)$ ; on prend l'espérance sous $q$, et le terme $\E_q[\log p(z \mid x) - \log q(z)]$ est exactement $-D_{\mathrm{KL}}(q \,\|\, p(z \mid x))$. La divergence de Kullback-Leibler (chapitre 01) étant positive, la borne inférieure de l’évidence, abrégée ELBO, est toujours en dessous de la log-vraisemblance. EM est une montée par coordonnées sur l'ELBO :

1. Étape E : à $\theta$ fixé, on choisit $q(z) = p(z \mid x, \theta)$ (les responsabilités). La KL tombe à zéro : l'ELBO touche la log-vraisemblance.
2. Étape M : à $q$ fixé, on maximise l'ELBO en $\theta$. Elle monte, donc la log-vraisemblance, qui est au-dessus, monte au moins autant.

```text
log p(x | θ)   ─────────●═════════════●══════════════●═══════  (ne baisse jamais)
                        │E touche     ↑ M fait monter │E touche
ELBO(q, θ)     ─────────●─────────────●              ●
                   θ_t, q_t      θ_{t+1}, q_t    θ_{t+1}, q_{t+1}
```

L'exercice vérifie cette identité à chaque itération : l'écart entre la log-vraisemblance et ELBO + KL ne dépasse pas $9 \cdot 10^{-13}$ (erreur d'arrondi), et la log-vraisemblance ne baisse jamais.

Ce que donne l'exercice. Sur les 900 points, 10 initialisations aléatoires : 9 finissent à la même log-vraisemblance, −2 870,2, et une reste bloquée dans un optimum local, à −3 189,0. EM ne garantit qu'une montée, pas le maximum global : on relance plusieurs fois et on garde la meilleure. La meilleure passe de −4 817,1 à −2 870,2 en 114 itérations, avec un long plateau presque plat de l'itération 10 environ à l'itération 95, le temps que deux composantes se décident à se partager les deux bandes. scikit-learn (`GaussianMixture`) trouve la même solution : log-vraisemblance moyenne −3,18913 par point des deux côtés.

Les groupes retrouvés se mesurent par l'indice de Rand ajusté (1 = partition identique à la vraie, 0 = pas mieux que le hasard) : 0,997 pour EM, 0,499 pour les k-moyennes, qui coupent les bandes en travers.

Le cœur de l'algorithme en numpy (extrait simplifié ; `log_gauss`, qui calcule $\log \mathcal{N}$ via une décomposition de Cholesky, et `logsumexp`, qui évite les débordements, sont définies dans `exercices/09-probabiliste/notebook.py`) :

```python
import numpy as np

def em_melange(X, K, rng, n_iter=300):
    n, d = X.shape
    mu  = X[rng.choice(n, K, replace=False)]        # K points au hasard comme centres
    cov = np.array([np.cov(X.T)] * K)               # covariance globale pour chacun
    pi  = np.full(K, 1 / K)
    for _ in range(n_iter):
        # Étape E : log[π_k N(x_i | μ_k, Σ_k)], puis normalisation par ligne
        lj = np.stack([np.log(pi[k]) + log_gauss(X, mu[k], cov[k]) for k in range(K)], axis=1)
        r = np.exp(lj - logsumexp(lj, axis=1)[:, None])           # responsabilités (n, K)
        # Étape M : statistiques pondérées par les responsabilités
        nk  = r.sum(axis=0)
        pi  = nk / n
        mu  = (r.T @ X) / nk[:, None]
        cov = np.stack([((r[:, k, None] * (X - mu[k])).T @ (X - mu[k])) / nk[k]
                        + 1e-6 * np.eye(d) for k in range(K)])    # petit terme : évite Σ singulière
    return pi, mu, cov, r
```

Le terme $10^{-6} I$ n'est pas décoratif : si une composante se réduit à un seul point, sa covariance tend vers zéro et la vraisemblance vers l'infini. C'est une vraie pathologie du maximum de vraisemblance pour les mélanges, pas un bogue.

### Modèles de Markov cachés : des causes cachées dans le temps

Vous ne voyez pas la météo depuis votre bureau sans fenêtre, mais vous savez comment votre collègue vient travailler : à vélo, en transports, ou elle télétravaille. Son choix dépend du temps qu'il fait, et le temps qu'il fait aujourd'hui dépend de celui d'hier. C'est un modèle de Markov caché, abrégé HMM : une chaîne de Markov d'états cachés (la météo), dont chacun émet une observation (le trajet).

Les paramètres de l'exercice : la loi du premier jour $\pi_0 = (0{,}5 ;\ 0{,}3 ;\ 0{,}2)$, une matrice de transition $A$ et une matrice d'émission $B$.

| $A$ : météo de demain | soleil | nuages | pluie |
|---|---|---|---|
| si soleil aujourd'hui | 0,80 | 0,15 | 0,05 |
| si nuages | 0,20 | 0,60 | 0,20 |
| si pluie | 0,10 | 0,20 | 0,70 |

| $B$ : trajet observé | vélo | transports | télétravail |
|---|---|---|---|
| par soleil | 0,75 | 0,20 | 0,05 |
| par nuages | 0,20 | 0,60 | 0,20 |
| par pluie | 0,05 | 0,30 | 0,65 |

Trois questions, trois algorithmes, tous fondés sur la même remarque : la météo d'aujourd'hui résume tout le passé (propriété de Markov). Au lieu d'énumérer les $3^T$ suites de météo possibles ($3^{365} \approx 10^{174}$ sur un an), on avance jour par jour en ne gardant que 3 nombres.

1. Quelle est la probabilité de ces observations ? Passe forward. On note $\alpha_t(j)$ la probabilité d'avoir vu les observations $o_1, \dots, o_t$ et d'être dans l'état $j$ au jour $t$.

(Formule : Algorithme forward.)

À la main, deux jours : vélo puis transports.

```text
jour 1 (vélo)        α1 = π0 × B[:, vélo] = (0,5×0,75 ; 0,3×0,2 ; 0,2×0,05) = (0,375 ; 0,060 ; 0,010)
                     prévision du jour 2 : α1 · A = (0,313 ; 0,094 ; 0,038)
jour 2 (transports)  α2 = (0,313×0,2 ; 0,094×0,6 ; 0,038×0,3)  = (0,0626 ; 0,0566 ; 0,0113)
p(vélo, transports) = 0,0626 + 0,0566 + 0,0113 = 0,1305
P(météo du jour 2 | les deux trajets) = α2 / 0,1305 = (0,48 ; 0,43 ; 0,09)
```

Les transports seuls suggèrent des nuages (émission 0,6 contre 0,2 au soleil). Mais il faisait sans doute beau la veille, et le beau temps persiste : tout compte fait, soleil (0,48) passe légèrement devant nuages (0,43). Coût : $T \times K^2$ opérations, soit $365 \times 9 = 3\,285$ pour un an au lieu de $10^{174}$. En pratique, on renormalise $\alpha_t$ à chaque pas (les produits de probabilités deviennent vite trop petits pour la machine) et l'on additionne les logarithmes des facteurs.

2. Quelle est la suite de météo la plus probable ? Viterbi. L'algorithme de Viterbi remplace la somme par un maximum : $\delta_t(j)$ est la probabilité du meilleur chemin qui arrive en $j$ au jour $t$, et l'on retient d'où il vient.

(Formule : Algorithme de Viterbi.)

Au dernier jour, on prend le meilleur état et l'on remonte les pointeurs. Sur l'exemple : $\delta_2 = (0{,}375 \times 0{,}8 \times 0{,}2 ;\ 0{,}375 \times 0{,}15 \times 0{,}6 ;\ \dots) = (0{,}060 ;\ 0{,}034 ;\ 0{,}006)$, et le meilleur chemin est soleil, soleil. C'est de la programmation dynamique, la même idée que les plus courts chemins du chapitre 04 : le treillis ci-dessous est un graphe, et le chemin de Viterbi est le plus court chemin pour des coûts $-\log$ probabilité. Andrew Viterbi l'a publié en 1967 pour décoder des codes correcteurs d'erreurs ; il sert aujourd'hui encore dans les modems et les téléphones.

3. Quelle était la météo chaque jour, en tenant compte de toute la semaine ? Forward-backward. Une passe backward, symétrique de la passe forward, calcule la probabilité des observations futures sachant l'état de chaque jour. Le produit des deux, normalisé, donne la loi a posteriori de l'état au jour $t$ sachant toutes les observations : $P(s_t = j \mid o_{1:T}) \propto \alpha_t(j)\,\beta_t(j)$. C'est l'algorithme forward-backward.

Sur une année simulée de 365 jours, l'exercice compare trois façons de deviner la météo :

| Méthode | Jours retrouvés | Ce qu'elle utilise |
|---|---|---|
| Jour par jour, sans dynamique | 69,9 % | seulement le trajet du jour |
| Viterbi | 77,8 % | la meilleure suite cohérente |
| Forward-backward | 78,1 % | le meilleur état pour chaque jour séparément |

La dynamique fait gagner 8 points. Forward-backward fait un peu mieux que Viterbi en nombre de jours justes, parce que c'est exactement ce qu'il maximise ; Viterbi, lui, garantit une suite cohérente (jamais une transition de probabilité nulle), ce qui compte en reconnaissance vocale, où la suite de sons doit former un mot. L'exercice contrôle aussi forward et Viterbi sur 8 jours contre l'énumération brute des $3^8 = 6\,561$ chemins : même log-probabilité, −8,973231.

Apprendre $A$ et $B$ sans jamais voir la météo : Baum-Welch. Si l'on ne connaît que les trajets, on peut réestimer les paramètres par EM : forward-backward donne les probabilités a posteriori de chaque état et de chaque transition (étape E), puis on compte ces transitions et émissions « douces » et on normalise (étape M). C'est l'algorithme de Baum-Welch, publié en 1970, sept ans avant que Dempster, Laird et Rubin ne donnent son nom et sa théorie générale à EM. Sur 5 000 jours, avec 3 initialisations de 200 itérations, l'exercice retrouve $A$ à 0,074 près et $B$ à 0,055 près, une fois trouvée la permutation des états : l'algorithme découvre trois régimes, mais rien ne lui dit lequel s'appelle « soleil ». Sa log-vraisemblance (−5 158,3) est même légèrement meilleure que celle du vrai modèle (−5 163,5) : il colle aussi au hasard de cet échantillon.

```python
def viterbi(obs, pi0, A, B):
    """Suite d'états la plus probable. Calcul en log pour éviter les sous-dépassements."""
    T, K = len(obs), len(pi0)
    lA, lB = np.log(A), np.log(B)
    delta = np.log(pi0) + lB[:, obs[0]]                 # δ_1
    psi = np.zeros((T, K), dtype=int)
    for t in range(1, T):
        scores = delta[:, None] + lA                    # scores[i, j] : venir de i pour aller en j
        psi[t] = scores.argmax(axis=0)                  # meilleure provenance de chaque j
        delta = scores.max(axis=0) + lB[:, obs[t]]
    chemin = [int(delta.argmax())]
    for t in range(T - 1, 0, -1):                       # remontée des pointeurs
        chemin.append(int(psi[t, chemin[-1]]))
    return chemin[::-1]
```

États continus : Kalman et filtres particulaires. Remplacez la météo par la position et la vitesse d'un véhicule, et les trajets par des mesures GPS bruitées. Si l'état évolue linéairement avec un bruit gaussien et que les mesures sont linéaires et gaussiennes, la passe forward se calcule exactement avec des gaussiennes au lieu de tables : c'est le filtre de Kalman (1960), qui alterne prédiction (l'état avance, l'incertitude grandit) et correction (la mesure arrive, l'incertitude diminue). Hors de ce cadre linéaire et gaussien, le filtre particulaire (Gordon, Salmond et Smith, 1993) représente la loi de l'état par un nuage d'échantillons pondérés, qu'on propage, repondère par la vraisemblance de chaque mesure, puis rééchantillonne. L'exercice ne les programme pas : ce sont deux pistes de son « pour aller plus loin ».

### MCMC : échantillonner ce qu'on ne sait pas calculer

Le dénominateur de Bayes est souvent hors de portée : une somme sur trop de configurations, ou une intégrale sans formule. Mais on connaît presque toujours la loi à une constante près, $p(x) = \tilde p(x) / Z$, avec $\tilde p$ facile à évaluer (a priori × vraisemblance) et $Z$ inconnu. Les méthodes Monte-Carlo par chaînes de Markov construisent une marche aléatoire qui visite chaque région proportionnellement à sa probabilité ; les moyennes le long de la marche approchent les espérances voulues.

L'algorithme de Metropolis-Hastings tient en trois lignes :

(Formule : Metropolis-Hastings (proposition symétrique : algorithme de Metropolis).)

Monter vers une région plus probable est toujours accepté ; descendre l'est avec la probabilité $\tilde p(x') / \tilde p(x)$. Exemple : la cible a deux bosses de poids 0,3 et 0,7. Proposer de sauter du centre de la petite bosse au centre de la grande donne un rapport d'environ $0{,}7 / 0{,}3 = 2{,}3$ : accepté. Le saut inverse a un rapport de 0,43 : accepté 43 fois sur 100. C'est ce déséquilibre qui fait passer la chaîne 70 % du temps dans la grande bosse. Hastings (1970) a ajouté le facteur $q(x \mid x') / q(x' \mid x)$, qui corrige les propositions non symétriques ; sans lui, la chaîne converge vers une mauvaise loi.

La règle rappelle le recuit simulé du chapitre 04, qui en est issu : accepter parfois un mouvement « vers le bas » avec une probabilité contrôlée. Mais le recuit fait baisser une température pour trouver un optimum ; Metropolis-Hastings, à température fixe, décrit toute la loi.

```python
def metropolis_hastings(log_p_tilde, x0, pas, n, rng):
    x, lp = np.array(x0, float), log_p_tilde(np.array(x0, float))
    chaine, acceptes = np.empty((n, len(x))), 0
    for i in range(n):
        x_prop = x + pas * rng.standard_normal(len(x))    # proposition symétrique
        lp_prop = log_p_tilde(x_prop)
        if np.log(rng.random()) < lp_prop - lp:           # accepter avec proba min(1, p̃(x')/p̃(x))
            x, lp, acceptes = x_prop, lp_prop, acceptes + 1
        chaine[i] = x                                     # refus : on recopie l'ancien point
    return chaine, acceptes / n
```

Tout se joue sur la taille du pas. L'exercice lance la chaîne depuis (−6 ; 6) pour 30 000 pas sur la cible à deux bosses (moyenne exacte (0,8 ; 0,4)), en jetant les 3 000 premiers pas (le rodage, *burn-in*, avant que la chaîne n'ait oublié son point de départ).

| Pas | Acceptation | Taille effective (sur 27 000 pas) | Temps dans la grande bosse (vrai : 70 %) | Erreur sur la moyenne |
|---|---|---|---|---|
| 0,1 | 92,4 % | 9 | 84 % | 0,59 |
| 1,5 | 28,2 % | 424 | 70,1 % | 0,02 |
| 15 | 0,8 % | 96 | 71,8 % | 0,08 |

Un pas minuscule fait accepter presque tout, mais la chaîne rampe et ne change presque jamais de bosse : 27 000 pas valent 9 tirages indépendants. Un pas énorme fait presque tout refuser : la chaîne reste immobile. Pour une proposition gaussienne, les résultats théoriques (Roberts, Gelman et Gilks, 1997) recommandent environ 23 % d'acceptation en grande dimension, davantage en petite dimension (environ 44 % en dimension 1) ; ici, le bon pas accepte 28 %. Un taux d'acceptation élevé n'est pas un signe de qualité. La taille effective d’échantillon, abrégée ESS, mesure ce que valent vraiment des tirages corrélés :

$$
\mathrm{ESS} = \frac{n}{1 + 2 \sum_{k \ge 1} \rho_k},
$$

où $\rho_k$ est l'autocorrélation de la chaîne à $k$ pas d'écart (l'exercice tronque la somme selon la règle de Geyer, 1992). Plus la chaîne « se souvient » longtemps de sa position, plus l'ESS est petite.

Gibbs. Quand on sait tirer chaque variable sachant toutes les autres, on peut les mettre à jour une à une : c'est l'échantillonnage de Gibbs, introduit en statistique et nommé en 1984 par les frères Geman (la même mise à jour est connue en physique statistique sous le nom d'algorithme du bain thermique, *heat bath*) pour restaurer des images bruitées avec un champ de Markov (chaque pixel ne dépend que de ses voisins, sa loi conditionnelle est donc facile). C'est un cas particulier de Metropolis-Hastings dont toutes les propositions sont acceptées. Il devient lent quand les variables sont très corrélées : il avance en escalier, un axe à la fois, dans une vallée diagonale.

L'algorithme porte le nom du premier auteur, mais sa paternité est discutée. D'après le témoignage de Marshall Rosenbluth en 2003, rapporté par J. E. Gubernatis (*Physics of Plasmas*, 2005), l'idée de départ vient d'Edward Teller, Marshall Rosenbluth a mis au point la méthode et Arianna Rosenbluth l'a programmée sur le MANIAC ; Metropolis, qui avait dirigé la construction de la machine, aurait surtout fourni le temps de calcul. Les mémoires de Teller décrivent au contraire un travail commun des cinq auteurs.

### Inférence variationnelle : remplacer l'échantillonnage par une optimisation

MCMC est exact « à la limite », mais lent et difficile à diagnostiquer : quand s'arrêter ? L'inférence variationnelle fait un autre pari : choisir une famille de lois simples $q_\phi$ (par exemple des gaussiennes) et chercher celle qui ressemble le plus à la loi a posteriori. L'identité vue avec EM dit comment, sans connaître l'évidence :

$$
\log p(x) = \mathrm{ELBO}(q_\phi) + D_{\mathrm{KL}}\bigl(q_\phi \,\|\, p(z \mid x)\bigr).
$$

$\log p(x)$ ne dépend pas de $\phi$ : maximiser l'ELBO revient à minimiser la KL entre l'approximation et la vraie loi. L'inférence devient un problème d'optimisation, qu'on résout avec les outils du chapitre 02.

L'exemple de l'exercice : une régression logistique bayésienne à deux paramètres $w = (w_0, w_1)$, c'est-à-dire le modèle $P(y = 1 \mid x) = \sigma(w_0 + w_1 x)$ du chapitre 01 dont on cherche cette fois toute la loi des paramètres, et non une seule valeur ; 40 points, a priori $\mathcal{N}(0, 3^2)$ sur chaque paramètre. La loi a posteriori $p(w \mid \mathcal{D}) \propto p(\mathcal{D} \mid w)\, p(w)$ n'a pas de formule, mais avec deux paramètres on peut la calculer sur une grille de 400 × 400 points : c'est la référence exacte, avec une évidence $\log p(\mathcal{D}) = -13{,}6027$ et une forte corrélation, −0,88, entre $w_0$ et $w_1$ (les $x$ ne sont pas centrés : augmenter la pente oblige à baisser l'ordonnée à l'origine).

L'approximation la plus simple est le champ moyen : une gaussienne à axes parallèles, $q(w) = \mathcal{N}(w_0 \mid \mu_0, \sigma_0^2)\,\mathcal{N}(w_1 \mid \mu_1, \sigma_1^2)$, soit quatre nombres à régler. Son ELBO :

(Formule : ELBO d'une approximation gaussienne.)

L'écriture $w = \mu + \sigma \odot \varepsilon$ est l'astuce de reparamétrisation : le hasard ne dépend plus des paramètres, on peut donc dériver sous l'espérance et estimer le gradient avec quelques tirages. Le premier terme pousse $q$ vers les régions probables ; le second, l'entropie, l'empêche de s'effondrer sur le seul maximum. La même astuce fera tourner les autoencodeurs variationnels du chapitre 18.

Extrait simplifié ; `grad_log_p_tilde` (gradient de la log-densité non normalisée) et `adam_monter` (un pas d'Adam en montée) sont des fonctions auxiliaires, dont la version complète se trouve dans `exercices/09-probabiliste/notebook.py` :

```python
mu, rho = np.zeros(2), np.zeros(2)                  # σ = exp(ρ) reste positif
for t in range(3000):
    eps = rng.standard_normal((16, 2))              # 16 tirages par pas
    w = mu + np.exp(rho) * eps                      # astuce de reparamétrisation
    g = grad_log_p_tilde(w)                         # ∇_w log p̃(w), forme (16, 2)
    g_mu  = g.mean(axis=0)
    g_rho = (g * eps * np.exp(rho)).mean(axis=0) + 1.0   # +1 : dérivée de l'entropie Σ ρ_j
    mu, rho = adam_monter(mu, rho, g_mu, g_rho)     # Adam en MONTÉE : on maximise l'ELBO
```

Le verdict de l'exercice, sur la même loi a posteriori :

| | Moyenne $(w_0 ;\ w_1)$ | Écarts-types | Coût, défaut |
|---|---|---|---|
| Exacte (grille 400 × 400) | (−2,803 ; 3,700) | (1,12 ; 1,11), corrélation −0,88 | impossible au-delà de quelques paramètres |
| Metropolis-Hastings | (−2,751 ; 3,642) | (1,08 ; 1,07) | 30 000 évaluations corrélées, ESS 745 |
| VI champ moyen (3 000 pas d'Adam) | (−2,861 ; 3,548) | (0,52 ; 0,47) | rapide, mais deux fois trop sûre d'elle |

L'ELBO finale vaut −14,463 (± 0,003), contre $\log p(\mathcal{D}) = -13{,}603$ : l'écart, 0,86 nat (le nat est l'unité des logarithmes népériens, comme le bit pour $\log_2$ ; voir le chapitre 01), est exactement la KL entre l'approximation et la vraie loi. Il ne peut pas tomber à zéro : aucune gaussienne à axes parallèles ne représente une corrélation de −0,88.

Pourquoi trop étroite plutôt que trop large ? Parce que $D_{\mathrm{KL}}(q \,\|\, p) = \E_q[\log q - \log p]$ est une moyenne sous $q$ : elle punit lourdement $q$ de mettre de la masse là où $p$ est presque nulle, et ne coûte rien là où $q$ ne va pas. Face à une vallée diagonale, la meilleure ellipse à axes parallèles se loge à l'intérieur, sans déborder. En général, l'inférence variationnelle (avec $D_{\mathrm{KL}}(q \,\|\, p)$) sous-estime l'incertitude : c'est une conséquence de son objectif, pas une loi sans exception, et ce n'est pas toujours gênant en pratique. La courbe de la fiche ci-dessous le montre sur un cas d'école : pour approcher $p = \mathcal{N}(0, 1)$ par $q = \mathcal{N}(0, s^2)$, la pénalité $D_{\mathrm{KL}}(q \,\|\, p)$ est bien plus faible pour un $q$ trop étroit ($s = 0{,}5$ : 0,32 nat) que pour un $q$ trop large ($s = 2$ : 0,81 nat).

| | MCMC | Inférence variationnelle |
|---|---|---|
| Principe | marche aléatoire dont la loi stationnaire est la cible | optimisation de l'ELBO dans une famille simple |
| Résultat | des échantillons | une loi approchée $q$, qu'on peut évaluer et tirer |
| Exact ? | oui si on attend assez longtemps | non : biais fixé par la famille choisie |
| Coût | beaucoup d'évaluations, corrélées ; difficile de savoir quand s'arrêter | rapide, s'arrête quand l'ELBO plafonne ; passe à l'échelle (minibatchs, GPU) |
| Héritiers | inférence bayésienne en statistique (Stan, PyMC) | VAE (ch. 18), diffusion vue comme un VAE hiérarchique (ch. 25) |

### Ce que la méthode optimise

Les modèles de ce chapitre maximisent la log-vraisemblance des observations, $\log p(x \mid \theta)$, ou minimisent son opposé, la log-vraisemblance négative (le lien entre perte et vraisemblance est au chapitre 01). La difficulté nouvelle est la somme sur les variables cachées à l'intérieur du logarithme. D'où la seconde quantité du chapitre, dont il est propriétaire : l'ELBO, qu'EM maximise par coordonnées (étape E en $q$, étape M en $\theta$) et que l'inférence variationnelle maximise par gradient. Les parties « réseau bayésien » et « MCMC » n'optimisent rien : elles calculent ou échantillonnent une loi.

### Processus gaussiens (en bref)

Un processus gaussien place une loi de probabilité directement sur les fonctions. On choisit un noyau $k(x, x')$ qui dit à quel point les valeurs en $x$ et $x'$ se ressemblent (par exemple $k(x, x') = \exp(-\lVert x - x' \rVert^2 / 2\ell^2)$, proche de 1 pour des points voisins). Pour toute famille de points, les valeurs de la fonction suivent alors une gaussienne de covariance donnée par le noyau. Conditionner sur $n$ mesures bruitées $y$ (bruit de variance $\sigma^2$) donne en un point nouveau $x_*$ :

$$
\text{moyenne} = k_*^\top (K + \sigma^2 I)^{-1} y, \qquad \text{variance} = k(x_*, x_*) - k_*^\top (K + \sigma^2 I)^{-1} k_*,
$$

où $K$ est la matrice $n \times n$ des $k(x_i, x_j)$ et $k_*$ le vecteur des $k(x_i, x_*)$. Le résultat est une prédiction et une barre d'erreur, qui s'élargit loin des données. Le prix : inverser une matrice $n \times n$, soit un coût en $n^3$, ce qui limite la méthode à quelques milliers de points sans approximation. On les retrouve en optimisation bayésienne d'hyperparamètres (chapitre 08). Référence : Rasmussen et Williams, *Gaussian Processes for Machine Learning* (MIT Press, 2006), en accès libre.

### Une ligne sur LDA

L'allocation de Dirichlet latente (Blei, Ng et Jordan, 2003) applique toutes les idées du chapitre aux textes : chaque document est un mélange de thèmes cachés, chaque thème une loi sur les mots, et l'on ajuste le tout par inférence variationnelle. Ce fut longtemps un outil courant de « découverte de thèmes », avant les plongements de mots (chapitre 15).

## Trois réponses à une même question

La figure de l'inférence variationnelle, plus haut, résume le chapitre, parce qu'elle montre côte à côte les trois façons d'obtenir une loi a posteriori :

1. calculer exactement (la grille, les contours noirs) : parfait, mais seulement en très petite dimension ou quand la structure le permet (réseau en arbre, HMM) ;
2. échantillonner (MCMC, les points) : exact à la limite, au prix de beaucoup de calcul et de tirages corrélés ;
3. approcher par optimisation (VI, les cercles) : rapide, mais limité par la famille choisie, et trop confiant.

EM se place entre les deux derniers : son étape E est une inférence exacte (les responsabilités) quand elle est calculable, et l'ELBO est le pont entre tous ces algorithmes. Reprenez les figures du chapitre avec cette grille de lecture : le treillis de Viterbi est une inférence exacte rendue possible par la structure en chaîne ; les trois chaînes de Metropolis-Hastings montrent le prix de l'échantillonnage ; les ellipses d'EM sont des lois $q$ qui touchent la vraie loi a posteriori à chaque étape E.

## Ce qui coinçait

- La structure est choisie à la main. Le graphe d'un réseau bayésien, le nombre de groupes $K$ d'un mélange, le nombre d'états d'un HMM sont des hypothèses de l'auteur. Apprendre la structure elle-même est beaucoup plus difficile. Les réseaux profonds (chapitre 12) apprendront des représentations sans qu'on dessine les dépendances.
- L'inférence exacte explose. Exponentielle dans le pire cas sur un graphe quelconque ; les méthodes approchées ont chacune leur défaut (MCMC lent et difficile à diagnostiquer, VI trop confiante).
- EM trouve un optimum local. Une initialisation sur dix restait bloquée dans l'exercice ; il faut relancer, et choisir $K$ par un critère externe.
- Les HMM ont la mémoire courte. La propriété de Markov résume tout le passé par un état parmi $K$ : impossible de se souvenir du sujet d'une phrase vingt mots plus tôt sans multiplier les états. Les réseaux récurrents (chapitre 16) remplaceront l'état discret par un vecteur appris.
- Corrélation n'est pas causalité. Un modèle appris sur des observations ne dit pas ce qui se passe quand on intervient.
- Les familles de lois sont simples. Gaussiennes, tables, mélanges : les VAE (chapitre 18) garderont l'ELBO mais remplaceront ces lois par des réseaux de neurones.

- Facteurs de certitude (MYCIN, années 1970) : Des coefficients combinés par des formules fixes : commode, incohérent dès que les indices dépendent les uns des autres.
- Réseaux bayésiens (Pearl, 1988) : Un graphe factorise la loi jointe ; la règle de Bayes inverse les flèches et produit l’effet d’explication.
- Metropolis (1953), Hastings (1970), Gibbs (1984) : On ne sait pas normaliser la loi ? On marche au hasard de façon à visiter chaque région en proportion de sa probabilité.
- HMM : Viterbi (1967), Baum-Welch (1970) : Un état caché qui évolue dans le temps ; la programmation dynamique remplace 3^T chemins par T × 9 opérations.
- EM (Dempster, Laird et Rubin, 1977) : Deviner les variables cachées, réajuster les paramètres, recommencer : la vraisemblance ne baisse jamais.
- Inférence variationnelle, ELBO (années 1990) : Remplacer le calcul d’une loi par l’optimisation d’une borne : rapide, mais trop confiant.
- VAE (2013) : La même ELBO, avec des réseaux de neurones à la place des lois simples.

Et ensuite ? Les modèles de ce chapitre et du précédent s'ajustent tous sur un échantillon fini : EM maximise la vraisemblance des 900 points qu'il voit, Baum-Welch colle même un peu mieux à ses 5 000 jours que le vrai modèle. Qu'est-ce qui garantit qu'un modèle performant sur les données d'entraînement le restera sur des données nouvelles ? Le chapitre 10 : Généraliser, la théorie pose cette question en toute rigueur (cadre PAC, dimension VC).

## Exercice

Ce qu'il faut observer :

- L'effet d'explication : la pluie passe de 20 % à 36 % quand on voit l'herbe mouillée, puis tombe à 0,7 % quand on apprend que l'arroseur tournait, surtout parce que l'arroseur dépend déjà de la pluie (0,6 % sans regarder l'herbe). Dans la variante à causes indépendantes, l'effet d'explication pur : 20 %, puis 37,8 %, puis 21,6 %. Le rejet jette 55 % des tirages.
- EM ne baisse jamais, mais peut stagner ou se bloquer : plateau de l'itération 10 à 95 environ, une initialisation sur dix dans un optimum local (−3 189,0 contre −2 870,2). L'ELBO reste sous la log-vraisemblance, l'écart étant la KL, vérifié à $10^{-12}$ près.
- La dynamique aide : 69,9 % de jours retrouvés sans elle, 77,8 % avec Viterbi, 78,1 % avec forward-backward.
- La taille du pas de Metropolis-Hastings : 92 % d'acceptation et une ESS de 9 avec un pas de 0,1, contre 28 % et 424 avec un pas de 1,5.
- VI contre MCMC : la gaussienne champ moyen est bien centrée mais deux fois trop étroite ; l'ELBO plafonne 0,86 nat sous l'évidence.

Questions de réflexion : pourquoi $P(\text{pluie} \mid \text{herbe}, \text{pas d'arroseur})$ vaut-il exactement 1, et quelle valeur de la table changer pour que ce ne soit plus le cas ? Combien d'initialisations d'EM lancer, et qu'observer pour en décider ? Si l'on remplace le champ moyen par une gaussienne à covariance pleine, que devient l'écart entre l'ELBO et $\log p(\mathcal{D})$, et pourquoi ne peut-il toujours pas s'annuler ?

## Auto-évaluation

Question : Dans la variante sans flèche pluie → arroseur (arroseur en marche avec la probabilité 0,4, qu'il pleuve ou non), savoir que l'arroseur a tourné ne change rien à la probabilité de pluie. Pourquoi cela change-t-il quand on sait aussi que l'herbe est mouillée ?

Réponse : Parce qu'on observe alors un effet commun. Sachant l'herbe mouillée, il faut au moins une cause ; si l'arroseur suffit à l'expliquer, la pluie n'est plus nécessaire et redescend vers son niveau a priori : $P(\text{pluie} \mid h) = 0{,}378$, mais $P(\text{pluie} \mid h, a) = 0{,}216$, contre 0,2 a priori. C'est l'effet d'explication : deux causes indépendantes deviennent dépendantes dès qu'on conditionne sur leur effet. Dans le réseau de la figure, la chute est bien plus forte (0,36 → 0,0068), mais pour une autre raison surtout : l'arroseur y dépend déjà de la pluie, et $P(\text{pluie} \mid a) = 0{,}0062$ sans même regarder l'herbe.

Question : EM garantit que la log-vraisemblance ne baisse jamais. Expliquez pourquoi avec l'ELBO, et dites ce que la garantie ne promet pas.

Réponse : $\log p(x \mid \theta) = \mathrm{ELBO}(q, \theta) + D_{\mathrm{KL}}(q \,\|\, p(z \mid x, \theta))$ avec une KL positive. L'étape E choisit $q = p(z \mid x, \theta)$ : la KL vaut 0 et l'ELBO touche la log-vraisemblance. L'étape M augmente l'ELBO en $\theta$ ; la log-vraisemblance, toujours au-dessus, augmente donc au moins autant. La garantie ne promet pas le maximum global : EM peut s'arrêter dans un optimum local (une initialisation sur dix dans l'exercice), d'où les relances.

Question : HMM météo : sur deux jours, la collègue vient à vélo puis en transports. Pourquoi l'algorithme forward juge-t-il « soleil » un peu plus probable que « nuages » le deuxième jour, alors que les transports sont trois fois plus fréquents par temps nuageux ?

Réponse : Parce qu'il combine l'observation avec la dynamique. Le vélo du jour 1 rend le soleil très probable ce jour-là, et le soleil persiste (0,8). La prévision du jour 2, avant de voir le trajet, est (0,313 ; 0,094 ; 0,038) ; multipliée par les émissions des transports (0,2 ; 0,6 ; 0,3), elle donne (0,063 ; 0,057 ; 0,011), soit 0,48 pour le soleil contre 0,43 pour les nuages après normalisation. Un décodage jour par jour, sans dynamique, aurait choisi les nuages.

Question : Une chaîne de Metropolis-Hastings accepte 92 % de ses propositions. Est-ce une bonne nouvelle ?

Réponse : Pas forcément. Un taux d'acceptation très élevé signale souvent un pas trop petit : la chaîne accepte tout parce qu'elle ne bouge presque pas, ses tirages sont très corrélés et elle explore lentement. Dans l'exercice, le pas de 0,1 donne 92 % d'acceptation mais une taille effective de 9 sur 27 000 pas et 84 % du temps dans la grande bosse au lieu de 70 %. Le pas de 1,5, avec 28 % d'acceptation, donne une ESS de 424 et les bonnes proportions.

Question : Pourquoi l'approximation variationnelle champ moyen de l'exercice est-elle deux fois trop étroite, et que mesure l'écart de 0,86 nat entre l'ELBO et log p(D) ?

Réponse : La VI minimise $D_{\mathrm{KL}}(q \,\|\, p)$, une moyenne sous $q$ qui pénalise fortement la masse que $q$ met là où $p$ est faible. La vraie loi est une ellipse diagonale (corrélation −0,88) ; une gaussienne à axes parallèles ne peut pas la suivre et se loge à l'intérieur, d'où des écarts-types de 0,5 au lieu de 1,1. L'écart $\log p(\mathcal{D}) - \mathrm{ELBO} = 0{,}86$ nat est exactement $D_{\mathrm{KL}}(q \,\|\, p(w \mid \mathcal{D}))$ pour la meilleure $q$ de la famille : il ne peut pas s'annuler tant que la famille ne contient pas la vraie loi.

## Pour aller plus loin

- Articles originaux : Pearl (1988) pour les réseaux bayésiens, Metropolis et al. (1953) et Hastings (1970) pour MCMC, Dempster, Laird et Rubin (1977) pour EM, Rabiner (1989) pour les HMM, Jordan et al. (1999) pour l'inférence variationnelle : liens dans les fiches ci-dessus. Aussi : Viterbi, 1967 ; Baum et al., 1970 ; Kalman, 1960 ; Geman et Geman, 1984 ; Gordon, Salmond et Smith, 1993 ; Geyer, « Practical Markov Chain Monte Carlo » (1992) ; Blei, Ng et Jordan, LDA (2003) ; Pearl, *Causality* (2009).
- D. Blei, A. Kucukelbir et J. McAuliffe, « Variational Inference: A Review for Statisticians » (*JASA*, 2017) : l'inférence variationnelle moderne, ELBO, champ moyen, comparaison avec MCMC.
- Stanford CS229, notes de cours, chapitre 11 « EM algorithms » : EM pour les mélanges de gaussiennes et sa dérivation par l'ELBO (notations proches, avec $\phi_j$ pour les poids des composantes).
- Stanford CS228, notes sur les modèles graphiques probabilistes (V. Kuleshov et S. Ermon) : réseaux bayésiens, propagation de croyances, échantillonnage, inférence variationnelle, EM, jusqu'au VAE.
- C. Bishop, *Pattern Recognition and Machine Learning* (Springer, 2006, PDF gratuit) : chapitres 8 (modèles graphiques), 9 (mélanges et EM), 10 (inférence variationnelle), 11 (échantillonnage), 13 (HMM et Kalman).
- K. Murphy, *Probabilistic Machine Learning: Advanced Topics* (MIT Press, 2023, brouillon en ligne) : la référence moderne et complète sur tout le chapitre.
- D. MacKay, *Information Theory, Inference, and Learning Algorithms* (Cambridge University Press, 2003, téléchargeable) : chapitres sur les méthodes de Monte-Carlo et les méthodes variationnelles, très intuitifs.
- D. Jurafsky et J. H. Martin, *Speech and Language Processing* (3e édition en ligne), annexe A « Hidden Markov Models » : forward, Viterbi et Baum-Welch pas à pas, sur un exemple proche du nôtre.
- C. Feng, « The Markov-chain Monte Carlo Interactive Gallery » : regardez Metropolis-Hastings, Gibbs et leurs variantes explorer des lois en 2D, dans le navigateur.
- J. Görtler, R. Kehlbeck et O. Deussen, « A Visual Exploration of Gaussian Processes » (*Distill*, 2019) : les processus gaussiens et le rôle du noyau, de façon interactive. Le livre de Rasmussen et Williams est en accès libre.
- 3Blue1Brown, « Bayes theorem, the geometry of changing beliefs » : la règle de Bayes, visuellement.
