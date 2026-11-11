# 10. Généraliser : la théorie (PAC, VC)

Repère historique (1968-1971 (Vapnik et Chervonenkis), 1984 (Valiant), 1996 (Wolpert)) : Garantir qu'un modèle entraîné sur un échantillon fini se comportera bien sur de nouvelles données, sans connaître la loi qui les produit.

À Moscou, à l'Institut d'automatique et de télémécanique (devenu en 1969 l'Institut des problèmes de commande), Vladimir Vapnik et Alexeï Tchervonenkis travaillent depuis le début des années 1960 sur la reconnaissance de formes. Une question les arrête : un classifieur qui fait peu d'erreurs sur ses exemples d'apprentissage en fera-t-il peu sur les suivants ? Pour une règle fixée d'avance, la loi des grands nombres répond oui. Mais l'apprentissage choisit sa règle parmi une infinité, en regardant les données. En 1968, une note des *Doklady* annonce leur réponse, détaillée en 1971 dans *Theory of Probability and Its Applications* : la convergence vaut pour toute une classe de règles à la fois si l'on peut compter ce que cette classe sait faire sur n points. En 1984, à Harvard, l'informaticien Leslie Valiant pose la question autrement : apprendre, c'est trouver en temps raisonnable, avec forte probabilité, une règle presque juste. En 1989, Blumer, Ehrenfeucht, Haussler et Warmuth montrent que les deux cadres se rejoignent. En 1996, David Wolpert, au Santa Fe Institute, rappelle le revers de la médaille : sans hypothèse sur ce qu'on cherche, aucun algorithme ne généralise mieux qu'un autre.

## L'idée en une image

Un étudiant prépare un examen avec 20 sujets d'annales corrigés. Il obtient 19/20 en s'entraînant dessus. Deux scénarios :

- il a appris trois règles de calcul qui suffisent à résoudre ces sujets ; son 19/20 dit quelque chose de sa note à l'examen ;
- il a appris par cœur les 20 corrigés ; son 19/20 ne dit rien, et un 21e sujet le laissera démuni.

Dans les deux cas, la note d'entraînement est la même. Ce qui change, c'est la quantité de choses qu'il aurait pu apprendre : avec trois règles, on ne peut pas coller à n'importe quels corrigés ; avec une mémoire illimitée, si. Plus l'élève peut tout expliquer après coup, moins sa réussite passée prouve quelque chose.

La généralisation d'un modèle suit la même logique. Le modèle est choisi dans une classe d’hypothèses (toutes les droites du plan, tous les polynômes de degré 2…). Si cette classe est assez petite au regard du nombre d'exemples, une faible erreur d'entraînement garantit, avec forte probabilité, une faible erreur sur de nouvelles données. Toute la théorie de ce chapitre sert à donner un sens précis à « petite » : un nombre d'hypothèses quand elles sont en nombre fini, une dimension VC quand elles sont en nombre infini. Elle a un revers, le théorème du « no free lunch » : la garantie ne vient pas de rien, elle vient d'un pari sur la forme de la cible.

| | Avant : mesurer (chapitre 00) | Après : garantir (ce chapitre) |
|---|---|---|
| Outil | un jeu de test mis de côté, la validation croisée | une borne qui relie erreur d'entraînement, taille de la classe et nombre d'exemples |
| Ce qu'on obtient | une estimation de l'erreur de ce modèle, après coup | une garantie avant l'entraînement, pour toute règle de la classe et toute distribution |
| Ce qui explique le surapprentissage | une courbe en U mesurée (biais-variance) | une classe trop riche pour le nombre d'exemples (dimension VC grande devant N) |
| Ce qui coince | il faut sacrifier des données ; rien ne dit pourquoi ça marche | des bornes justes mais très pessimistes : 54 000 exemples pour garantir un écart de 0,1 avec des coniques, alors que l'écart mesuré dans l'exercice est déjà sous 0,1 à N = 20 et sous 0,01 vers 1 000 exemples |

## Le mécanisme

Le cadre tient en un schéma. Une distribution inconnue produit des exemples ; l'algorithme n'en voit que N, et choisit sa règle dans une classe fixée d'avance. La question est l'écart entre ce qu'il mesure et ce qu'il obtiendra.

### 1. Rappel : biais et variance

Le chapitre 00 a montré le surapprentissage et le compromis biais-variance sur une régression polynomiale : en montant le degré, l'erreur d'entraînement baisse toujours, l'erreur de test baisse puis remonte (minimum 0,131 au degré 3, puis 39 au degré 12). Ce chapitre ne refait pas cette décomposition ; il en donne la version « garantie ». Le rôle de la variance y est tenu par l'écart entre erreur mesurée et erreur réelle, et celui de la complexité du modèle par la taille de la classe. Une différence de fond : la décomposition biais-variance est une moyenne sur les jeux d'entraînement possibles, pour une perte quadratique, alors que les bornes de ce chapitre disent ce qui arrive sur le jeu qu'on a, avec une probabilité contrôlée.

### 2. Risque empirique, risque réel

On classe en deux classes, $y \in \{0, 1\}$, et on compte les erreurs avec la perte 0-1 : 1 si la règle se trompe, 0 sinon. Deux moyennes de cette perte comptent :

(Formule : Risque réel et risque empirique.)

Le risque empirique est une estimation du risque réel. La différence $R(h) - \hat R(h)$ est l'écart de généralisation. Le principe le plus simple, la minimisation du risque empirique (ERM), choisit dans la classe $\mathcal H$ la règle de plus faible erreur d'entraînement :

$$
\hat h = \argmin_{h \in \mathcal H} \hat R(h).
$$

Le piège de la sélection. Pour une règle $h$ fixée avant de voir les données, $\hat R(h)$ est une moyenne de N tirages indépendants : elle est sans biais, et elle se concentre autour de $R(h)$. Mais $\hat h$ est choisie parce que son erreur d'entraînement est la plus basse : on sélectionne précisément la règle que le hasard de l'échantillon a le plus avantagée. Dans l'exercice (101 seuils, N = 20), la règle choisie par l'ERM a un risque réel de 0,256 en moyenne, et son erreur d'entraînement est en moyenne 0,087 plus basse. Pour garantir quoi que ce soit sur $\hat h$, il faut que $\hat R$ soit proche de $R$ pour toutes les règles de la classe à la fois. C'est exactement la « convergence uniforme » du titre de 1971.

### Ce que la méthode optimise

La théorie raisonne sur la perte 0-1 : le risque empirique (ce qu'on mesure) contre le risque réel (ce qu'on veut). Les bornes portent sur toute règle de la classe, quel que soit l'algorithme qui l'a choisie. En pratique on n'optimise jamais directement la perte 0-1, plate presque partout : on minimise une perte de substitution, comme l'entropie croisée de la régression logistique, utilisée dans l'exercice. Le détail de ces pertes et la raison de ce choix sont au chapitre 01.

### 3. Une seule règle : l'inégalité de Hoeffding

Commençons par une règle $h$ fixée d'avance. Chaque exemple donne une erreur (1) ou non (0) avec probabilité $R(h)$ : $\hat R(h)$ est la fréquence de « pile » dans N lancers d'une pièce truquée. L'inégalité de Hoeffding (1963) borne la probabilité que cette fréquence s'éloigne de la vraie probabilité :

(Formule : Inégalité de Hoeffding (variables dans [0, 1]).)

On retourne la formule : on fixe un risque d'échec $\delta$ et on cherche l'écart garanti. En posant $2e^{-2N\gamma^2} = \delta$, il vient $\gamma = \sqrt{\ln(2/\delta)/(2N)}$.

Exemple chiffré. $N = 100$, $\delta = 0{,}05$ : $\ln(2/0{,}05) = \ln 40 = 3{,}689$, divisé par $2N = 200$ donne $0{,}0184$, dont la racine vaut $\gamma = 0{,}136$. Avec 95 % de confiance, l'erreur mesurée sur 100 exemples est à moins de 13,6 points de l'erreur réelle. Pour diviser cet écart par 2, il faut 4 fois plus d'exemples : la précision progresse en $1/\sqrt N$.

### 4. Un nombre fini de règles : la borne de l'union

Avec $k$ règles, chacune peut être malchanceuse. La borne de l’union dit que la probabilité qu'au moins une le soit est au plus la somme : $k$ fois $2e^{-2N\gamma^2}$. En recommençant le calcul avec $\delta$ :

(Formule : Borne de généralisation pour une classe finie.)

Le nombre de règles n'entre que par son logarithme. Avec $k = 101$, $N = 100$ : $\ln(2 \times 101 / 0{,}05) = \ln 4\,040 = 8{,}30$, d'où $\gamma = \sqrt{8{,}30/200} = 0{,}204$, à peine 1,5 fois l'écart d'une règle seule.

On en tire la complexité d’échantillon : pour garantir un écart $\gamma$, il suffit de $N \ge \ln(2k/\delta)/(2\gamma^2)$ exemples. Pour $\gamma = 0{,}05$ et $\delta = 0{,}05$, cela fait 1 661 exemples avec 101 règles, 3 501 avec un million de règles. Multiplier la classe par 10 000 ne fait que doubler le besoin en données.

Comme la borne vaut pour toutes les règles, elle vaut pour celle de l'ERM, et on en déduit qu'elle est presque aussi bonne que la meilleure de la classe : $R(\hat h) \le \min_{h \in \mathcal H} R(h) + 2\gamma$ (l'ERM bat la meilleure règle sur l'échantillon, et chacune des deux est à moins de $\gamma$ de son risque réel).

Ce que mesure l'exercice. 101 seuils sur $[0, 1]$, des étiquettes données par le seuil 0,3 puis inversées dans 20 % des cas. Le risque réel de chaque seuil se calcule exactement, $R(h_t) = 0{,}2 + 0{,}6\,|t - 0{,}3|$, et l'on tire 2 000 échantillons pour chaque N :

| N | γ, une règle seule | γ, 101 règles (borne de l'union) | écart maximal mesuré (95 % des tirages en dessous) | tirages où la borne est violée |
|---|---|---|---|---|
| 20 | 0,304 | 0,456 | 0,304 | 0,10 % |
| 100 | 0,136 | 0,204 | 0,140 | 0 % |
| 500 | 0,061 | 0,091 | 0,060 | 0 % |
| 2 000 | 0,030 | 0,046 | 0,031 | 0 % |

La borne tient largement (au plus 0,10 % de violations pour 5 % autorisés). Mais la colonne mesurée colle à celle d'une seule règle : les 101 seuils voisins se trompent presque sur les mêmes points, ils sont si corrélés que la classe se comporte comme une seule hypothèse. La borne de l'union, qui les traite comme indépendants, paie pour rien. Retenez ce mécanisme : la théorie compte les règles, alors que ce qui compte vraiment, c'est le nombre de comportements différents qu'elles ont sur les données. C'est l'idée de la dimension VC.

### 5. Le cadre PAC de Valiant (1984)

En 1984, Leslie Valiant, informaticien théoricien à Harvard, propose de définir l'apprentissage comme on a défini le calcul. Son article, *A theory of the learnable*, demande qu'un algorithme, à partir d'exemples tirés selon une distribution quelconque, produise en temps polynomial une règle dont l'erreur est faible avec forte probabilité. Valiant y parle d'une règle déduite « avec probabilité au moins $1 - 1/h$ », d'erreur au plus $1/h$, pour un paramètre $h$ réglable. Le nom d'apprentissage PAC (*probably approximately correct* : probablement, approximativement juste) n'est pas dans l'article ; il s'est imposé ensuite, avec deux paramètres séparés :

(Formule : Apprenabilité PAC (en simplifiant).)

En simplifiant : cette forme suppose que la cible est dans la classe (cas « réalisable »), ce qui demande moins d'exemples que la borne de l'union ci-dessus (voir l'encadré). L'article de Valiant applique ce cadre à des classes de formules logiques (par exemple les formules en forme normale conjonctive à au plus $k$ littéraux par clause) et insiste sur le coût de calcul autant que sur le nombre d'exemples : c'est la marque de l'informatique théorique. Valiant reçoit le prix Turing pour l'année 2010, pour ses contributions à la théorie du calcul, dont la théorie de l'apprentissage PAC.

Pour les curieux : le cas réalisable, en 1/ε :
Quand la cible est dans la classe, une règle parfaite existe et le calcul est plus favorable que la borne de l'union. Une règle d'erreur réelle supérieure à $\varepsilon$ passe sans faute un exemple avec probabilité au plus $1 - \varepsilon$, donc N exemples avec probabilité au plus $(1-\varepsilon)^N \le e^{-\varepsilon N}$. Sur $k$ règles, la probabilité qu'une « mauvaise » règle passe tous les exemples est au plus $k\,e^{-\varepsilon N}$, d'où $N \ge (\ln k + \ln(1/\delta))/\varepsilon$, en $1/\varepsilon$ et non en $1/\varepsilon^2$. Avec 101 règles, $\varepsilon = \delta = 0{,}05$ : $(4{,}62 + 3{,}00)/0{,}05 \approx$ 153 exemples, contre 1 661 quand le bruit empêche une règle parfaite.

### 6. Une infinité de règles : pulvérisation et dimension VC

Les droites du plan forment une classe infinie : $\ln k = \infty$, la borne précédente ne dit rien. Pourtant, sur un échantillon donné, deux droites qui classent les points de la même façon sont indiscernables. Il faut compter les étiquetages réalisables : de combien de façons la classe peut-elle étiqueter n points ?

Sur une droite, à la main. Les seuils $h_t(x) = \mathbf{1}[x \ge t]$ sur n points ordonnés ne réalisent que les étiquetages « des 0 puis des 1 » : le seuil tombe dans l'un des n + 1 intervalles entre les points. Avec 3 points, 4 étiquetages sur 8 : 000, 001, 011, 111. Les intervalles $\mathbf{1}[a \le x \le b]$ réalisent « des 0, un bloc de 1, des 0 » : choisir le début et la fin du bloc parmi n + 1 positions, soit $n(n+1)/2$, plus l'étiquetage tout à 0, donc $n(n+1)/2 + 1$. Avec 3 points, 7 sur 8 : seul 101 est impossible. C'est d'ailleurs le premier exemple de l'article de 1971 (les demi-droites $x \le a$, qui donnent $r + 1$ sous-ensembles sur $r$ points).

Une classe pulvérise un ensemble de points si elle réalise tous ses étiquetages. La dimension VC d'une classe, notée $d_{VC}$, est la taille du plus grand ensemble qu'elle pulvérise :

(Formule : Dimension VC.)

Les seuils pulvérisent 1 point mais aucune paire (impossible de mettre 1 à gauche et 0 à droite) : $d_{VC} = 1$. Les intervalles pulvérisent 2 points mais aucun triplet (101) : $d_{VC} = 2$.

Les demi-plans du plan (une droite, des 1 d'un côté). Trois points en triangle : les 8 étiquetages sont réalisés, la figure le montre. Trois points alignés : seulement 6 sur 8, car le point du milieu ne peut pas être seul de son côté. La dimension vaut pourtant 3, puisqu'il suffit d'un triplet pulvérisé. Quatre points : jamais. Pour un carré, les deux étiquetages en diagonale (le XOR du chapitre 06) sont impossibles, 14 sur 16. Pour toute autre position de 4 points, un argument de géométrie (non détaillé ici) exhibe aussi un étiquetage impossible. Donc $d_{VC} = 3$. L'exercice le vérifie en testant chaque étiquetage par programmation linéaire : existe-t-il $(w, b)$ tel que $s_i(w \cdot x_i + b) \ge 1$ pour tous les points, avec $s_i = 2y_i - 1 \in \{-1, +1\}$ ?

Plus généralement, les séparateurs affines de $\R^D$ (un hyperplan) ont une dimension VC de $D + 1$, leur nombre de paramètres. Les coniques (frontières polynomiales de degré 2 du plan) s'écrivent comme un séparateur linéaire sur les 6 monômes $1, x_1, x_2, x_1^2, x_1x_2, x_2^2$ (dont la constante 1), c'est-à-dire un séparateur affine de $\R^5$, ce qui borne leur dimension à 6 ; l'exercice trouve 6 points pulvérisés (64 étiquetages sur 64), et pas le septuplet qu'il tire (126 sur 128) ; la borne 6 vient de l'argument affine. Pour un degré $p$, on retient $d_{VC} = (p+1)(p+2)/2$, le nombre de monômes : 3 pour les droites, 6 pour les coniques, 66 au degré 10.

Dimension VC n'est pas nombre de paramètres :
Pour les classes linéaires, les deux coïncident, d'où la règle approximative « dimension VC ≈ nombre de paramètres ». Elle peut être très fausse. La classe $\mathbf{1}[\sin(\omega x) \ge 0]$ n'a qu'un paramètre, la fréquence $\omega$, mais une fréquence assez élevée peut épouser n'importe quel étiquetage de points bien choisis : sa dimension VC est infinie. À l'inverse, une classe à beaucoup de paramètres contraints peut avoir une petite dimension. Ce qui compte, c'est le nombre de comportements, pas le nombre de boutons.

### 7. La fonction de croissance et le lemme de Sauer

La fonction de croissance $m_{\mathcal H}(n)$ est le nombre maximal d'étiquetages que la classe réalise sur n points. Elle vaut $2^n$ tant que $n \le d_{VC}$. Ensuite, elle décroche de l'exponentielle et ne croît plus qu'au plus comme un polynôme de degré $d_{VC}$ :

(Formule : Lemme de Sauer-Shelah.)

Vapnik et Chervonenkis établissent en 1971 (lemme 1 et théorème 1 de l'article) une borne de cette nature : une somme de coefficients binomiaux, qu'ils majorent ensuite, si bien que la fonction de croissance est « soit identiquement égale à $2^r$, soit majorée par $r^n + 1$ ». Attention aux notations de l'article : $r$ y est le nombre de points et $n = d_{VC} + 1$ le premier nombre de points qui n'est plus pulvérisé ; la borne est donc un polynôme de degré $d_{VC} + 1$ en $r$. Leur somme va jusqu'au rang $d_{VC} + 1$, un terme de plus que nécessaire. La forme serrée ci-dessus, arrêtée au rang $d_{VC}$, est publiée indépendamment en 1972 par Norbert Sauer et par Saharon Shelah (qui en crédite aussi Micha Perles).

Vérification à la main : pour les seuils ($d_{VC} = 1$), $\binom{n}{0} + \binom{n}{1} = 1 + n$, exactement le compte trouvé plus haut ; pour les intervalles ($d_{VC} = 2$), $1 + n + n(n-1)/2 = n(n+1)/2 + 1$, exact aussi. Pour les demi-plans ($d_{VC} = 3$), le compte par force brute de l'exercice donne $n^2 - n + 2$ à partir de 3 points : 14, 22, 32, 44, 58, 74, 92 pour n = 4 à 10, sous la borne de Sauer (15, 26, 42, 64, 93, 130, 176). À 10 points, 92 étiquetages réalisables sur 1 024 : les droites ne savent produire que 9 % des étiquetages possibles. C'est ce manque de liberté qui rend une garantie possible.

### 8. La borne VC

Il reste à remplacer $k$ par $m_{\mathcal H}$ dans la borne de l'union. L'obstacle : le risque réel fait intervenir une infinité de points, pas seulement les N de l'échantillon. L'astuce de 1971 (le lemme 2 de l'article) est la symétrisation : on imagine un second échantillon fantôme de N points, et on montre que l'écart entre erreur d'entraînement et erreur réelle est contrôlé par l'écart entre les erreurs sur les deux moitiés d'un échantillon de 2N points. Sur 2N points, la classe n'a que $m_{\mathcal H}(2N)$ comportements distincts, et la borne de l'union redevient finie. On obtient la borne suivante, avec les constantes du théorème 2 de l'article de 1971, sous la forme qu'en donne le cours de Yaser Abu-Mostafa (Caltech) et qu'utilise l'exercice :

(Formule : Borne de généralisation VC.)

Avec le lemme de Sauer, $\ln m_{\mathcal H}(2N) \approx d_{VC} \ln(2N)$, donc en simplifiant $\Omega \approx \sqrt{8\,d_{VC}\ln(2N)/N}$. C'est la formule à retenir du chapitre : l'écart garanti grandit comme la racine de la dimension VC et diminue à peu près comme $1/\sqrt N$. Elle ne dépend ni de la distribution, ni de l'algorithme : seulement de la classe et de N.

Exemple chiffré : des droites, 100 exemples. $d_{VC} = 3$, $N = 100$, $\delta = 0{,}05$. Lemme de Sauer sur $2N = 200$ points : $1 + 200 + 19\,900 + 1\,313\,400 = 1\,333\,501$ étiquetages au plus, contre $2^{200}$ possibles. Puis $\ln(4 \times 1\,333\,501 / 0{,}05) = 18{,}49$ ; multiplié par $8/100$, cela donne $1{,}479$, dont la racine vaut $\Omega = 1{,}22$. L'erreur réelle est garantie inférieure à l'erreur d'entraînement plus 1,22 : une garantie vide, puisqu'une erreur vaut au plus 1. Il faudrait environ 28 000 exemples pour descendre à 0,1.

Le calcul complet tient en quelques lignes ; c'est celui de l'exercice :

```python
import numpy as np
from scipy.special import gammaln

def log_sauer(n, d):
    """ln de la somme des C(n, i) pour i ≤ d (borne de Sauer sur m_H(n)), sans débordement."""
    i = np.arange(0, min(d, n) + 1)
    logs = gammaln(n + 1) - gammaln(i + 1) - gammaln(n - i + 1)   # ln C(n, i)
    m = logs.max()
    return m + np.log(np.exp(logs - m).sum())                      # log-somme-exp

def omega_vc(N, d_vc, delta=0.05):
    """Pénalité de la borne VC : sqrt(8/N · ln(4 m_H(2N) / δ))."""
    return np.sqrt(8 / N * (np.log(4 / delta) + log_sauer(2 * N, d_vc)))

print(omega_vc(100, 3))      # 1.216 : garantie vide (> 1)
print(omega_vc(10_000, 6))   # 0.214 : coniques, 10 000 exemples
```

Ce que mesure l'exercice. Points uniformes dans le carré $[-1, 1]^2$, étiquetés par le cercle $x_1^2 + x_2^2 = 0{,}5$, avec 10 % d'étiquettes inversées : aucune règle ne fait mieux que 0,10 d'erreur. On entraîne des classifieurs polynomiaux de degré 1 à 10 (régression logistique sur les monômes) pour N de 20 à 10 000, 10 répétitions, et l'on compare l'erreur d'entraînement, l'erreur de test (sur 100 000 points) et la borne $\hat R + \Omega$ :

| Degré 2 ($d_{VC}$ = 6) | erreur d'entraînement | erreur de test | écart mesuré | borne $\hat R + \Omega$ |
|---|---|---|---|---|
| N = 20 | 0,185 | 0,279 | 0,094 | 2,99 |
| N = 200 | 0,126 | 0,142 | 0,016 | 1,29 |
| N = 10 000 | 0,114 | 0,114 | 0,000 | 0,33 |

Trois constats.

- La borne est juste. Elle tient dans les 630 entraînements (100 %), quel que soit le degré ou N. C'est un théorème, pas une estimation.
- Elle a la bonne forme. À N = 200, quand le degré monte, l'erreur d'entraînement baisse (0,05 au degré 10) et l'erreur de test remonte (0,14 aux degrés 2 à 4, 0,19 au degré 10) : l'écart se creuse avec la richesse de la classe, comme $\Omega$. Au pire, degré 10 et N = 50 : 0,008 à l'entraînement, 0,244 en test, un écart de 0,236. Et à degré fixé, plus N grandit, plus l'écart se referme.
- Elle est très pessimiste. $\Omega$ dépasse 1 (garantie vide) pour tout N ≤ 100. Elle vaut entre 9,8 et plus de 500 fois l'écart mesuré. Pour garantir $\Omega \le 0{,}1$, il faudrait 28 341 exemples avec des droites ($d_{VC}$ = 3), 53 862 avec des coniques (6), 88 053 au degré 3 (10), 568 960 au degré 10 (66) : environ $9\,000 \times d_{VC}$. L'écart mesuré, lui, passe sous 0,01 vers 500 à 2 000 exemples. Abu-Mostafa propose d'ailleurs dans son cours une règle empirique bien plus modeste, $N \ge 10\,d_{VC}$.

Pourquoi un tel écart ? La borne doit valoir pour toutes les distributions, y compris les pires imaginables, et pour toutes les règles de la classe, y compris les plus absurdes ; ses constantes (8, 4) viennent d'une preuve qui accumule les majorations (symétrisation, borne de l'union sur $m_{\mathcal H}(2N)$, lemme de Sauer). Sur une distribution « gentille » comme un cercle bruité, elle est forcément lâche. Sa valeur est ailleurs : elle prouve que l'apprentissage est possible avec une classe infinie, et dit de quoi dépend la difficulté.

En simplifiant : pourquoi le degré 2 plafonne à 0,114 et non 0,10 :
Le cercle est une conique : le degré 2 contient la règle parfaite, d'erreur 0,10. Avec 10 000 exemples, l'exercice obtient 0,114. Ce n'est pas un défaut de la borne, qui porte sur toute règle de la classe quel que soit l'algorithme : c'est que la régression logistique minimise l'entropie croisée, pas la perte 0-1, et que le meilleur modèle pour l'une n'est pas exactement le meilleur pour l'autre. La théorie analyse l'ERM sur la perte 0-1, que l'on n'optimise pas directement en pratique.

En 1989, Anselm Blumer, Andrzej Ehrenfeucht, David Haussler et Manfred Warmuth réunissent les deux traditions : la condition essentielle pour qu'une classe soit apprenable sans hypothèse sur la distribution, au sens du nombre d'exemples nécessaires, est que sa dimension VC soit finie. Une classe de dimension infinie, comme les sinus ci-dessus, ne l'est pas : pour tout N, une distribution adverse la met en échec. L'apprenabilité au sens complet de Valiant, en temps polynomial, demande en plus un algorithme efficace, une question que l'article traite à part.

### 9. Pas de repas gratuit

Les bornes précédentes disent que l'erreur réelle est proche de l'erreur d'entraînement si la classe est petite. Elles ne disent pas que l'erreur d'entraînement sera faible : si la cible n'a rien à voir avec la classe choisie, on obtiendra une garantie solide… d'un mauvais résultat. Le théorème du théorème du « no free lunch » de David Wolpert (1996) pousse cette remarque à l'extrême, en regardant l'erreur hors de l'ensemble d'entraînement :

> En moyenne sur toutes les fonctions cibles possibles, tous les algorithmes d'apprentissage ont la même erreur sur les points qu'ils n'ont pas vus.

L'argument tient en une ligne sur un domaine fini. Fixez les étiquettes des points vus. Pour un point non vu, la moitié des cibles compatibles l'étiquettent 0, l'autre moitié 1. Quelle que soit la prédiction de l'algorithme, elle est juste pour exactement la moitié de ces cibles : précision moyenne 1/2, pour tout algorithme.

L'exercice le calcule exactement. Un domaine de 10 points $x = 0, \dots, 9$ ; on apprend sur les 5 pairs, on prédit les 5 impairs ; on énumère les $2^{10} = 1\,024$ cibles, en fractions exactes. Quatre algorithmes : le plus proche voisin (copie l'étiquette du point pair à gauche), la majorité, « toujours 0 » et l'anti-voisin, qui prédit l'inverse du plus proche voisin, une absurdité délibérée.

| Algorithme | toutes les cibles (1 024) | cibles « seuil » seulement (11) |
|---|---|---|
| plus proche voisin | 1/2 | 10/11 ≈ 0,91 |
| majorité | 1/2 | 42/55 ≈ 0,76 |
| toujours 0 | 1/2 | 5/11 ≈ 0,45 |
| anti-voisin | 1/2 | 1/11 ≈ 0,09 |

Sur toutes les cibles, l'anti-voisin vaut le plus proche voisin. Dès qu'on suppose quelque chose (« la cible est un seuil » : les points voisins ont souvent la même étiquette), les algorithmes se départagent, et l'ordre dépend entièrement de cette supposition : sur des cibles qui alternent 0 et 1, l'anti-voisin serait le meilleur. Ce que l'algorithme suppose de la cible s'appelle son biais inductif. Le choix d'une classe d'hypothèses en est un : en prenant des coniques, on parie que la frontière est une conique.

Ce que le théorème ne dit pas :
Il ne dit pas que l'apprentissage est impossible, ni que tous les algorithmes se valent sur les problèmes réels : ceux-ci ne sont pas des cibles tirées uniformément parmi toutes les fonctions possibles, ils ont de la régularité (des images voisines ont souvent la même classe). Il dit que cette régularité est une hypothèse, et que la supériorité d'un algorithme se mesure toujours relativement à une famille de problèmes. Formulation exacte, conditions et version « optimisation » (Wolpert et Macready, 1997) : voir l'article. La moyenne uniforme sur toutes les cibles est une simplification de notre présentation ; Wolpert énonce des résultats plus généraux, sur des lois a priori sur les cibles.

On peut relire la dimension VC à cette lumière. Une classe qui pulvérise tout (dimension infinie) ne fait aucun pari : elle peut expliquer n'importe quoi après coup, comme l'étudiant qui apprend les corrigés par cœur, et c'est pour cela qu'elle ne garantit rien. Une classe de petite dimension fait un pari fort ; si le pari est bon, peu d'exemples suffisent.

## Ce qui coinçait

- Des bornes vides en pratique. Avec les constantes de la preuve, la borne VC est vide ($\Omega > 1$) en dessous d'environ 50 exemples par unité de dimension VC (160 pour les droites, 288 pour les coniques), et il en faut environ 9 000 par unité pour garantir un écart de 0,1 ; sur les problèmes réels, on continue de mesurer sur un jeu de test. Des outils plus fins, qui tiennent compte de la distribution (complexité de Rademacher, marges, stabilité, bornes PAC-bayésiennes), resserrent l'écart sans le combler ; ils dépassent ce parcours.
- Le pire cas sur la distribution. La borne doit valoir pour toute distribution ; les données réelles ne sont pas les pires possibles. C'est la source principale de son pessimisme.
- Le calcul est ignoré. La théorie analyse l'ERM sur la perte 0-1, plate presque partout et qu'on n'optimise pas directement ; les algorithmes réels minimisent autre chose (entropie croisée, perte charnière) avec une descente de gradient.
- Le paradoxe des réseaux profonds. Un réseau moderne a bien plus de paramètres que d'exemples, et les bornes connues sur sa dimension VC croissent avec ce nombre de paramètres (pour les réseaux ReLU, au moins proportionnellement : Bartlett, Harvey, Liaw et Mehrabian, JMLR 2019, arXiv:1703.02930). La borne VC prédit alors un écart énorme, ou ne dit rien. Or ces réseaux généralisent souvent très bien, alors même qu'ils sont capables d'apprendre par cœur des étiquettes tirées au hasard (Zhang et al., arXiv:1611.03530, 2016). Ce n'est pas que la théorie soit fausse : elle est juste mais ne voit que la classe, alors que ce qui compte semble être quelle règle l'algorithme trouve dans cette classe. C'est la partie empirique du chapitre 27 : double descente, régularisation implicite, grokking.
- Pas de repas gratuit. Toute garantie repose sur une hypothèse sur la cible ; aucune théorie ne dispense de choisir un bon biais inductif. Les architectures des chapitres suivants (convolutions, attention) sont autant de paris de ce type.

- Biais-variance (rappel) : Trop rigide : biais ; trop souple : variance. Mesuré après coup sur un jeu de test.
- Hoeffding (1963) : Pour UNE règle fixée, l’erreur mesurée sur N exemples est à ±√(ln(2/δ)/(2N)) de la vraie.
- Borne de l’union, classe finie : k règles coûtent seulement ln k : N ≥ ln(2k/δ)/(2γ²).
- Convergence uniforme et dimension VC (Vapnik et Chervonenkis, 1968-1971) : Compter les comportements, pas les règles : au-delà de d_VC points, polynomial au lieu de 2ⁿ ; écart ≈ √(d_VC ln N / N).
- PAC (Valiant, 1984) : Apprendre = trouver en temps polynomial, avec probabilité 1 − δ, une règle d’erreur ≤ ε.
- PAC et VC (Blumer et al., 1989) : Apprenable sans hypothèse sur la distribution (en nombre d’exemples) exactement quand d_VC est finie ; le temps de calcul est une question à part.
- No free lunch (Wolpert, 1996) : En moyenne sur toutes les cibles, aucun algorithme ne bat le hasard hors des exemples : la généralisation vient du biais inductif.
- Généralisation des réseaux profonds : Bien plus de paramètres que d’exemples, et pourtant ça généralise : la théorie classique ne suffit plus.

Et ensuite ? Toute cette partie a supposé un professeur : des exemples étiquetés, tirés indépendamment, et une erreur mesurable exemple par exemple. Beaucoup de problèmes n'ont rien de tel : un programme qui joue aux dames ou pilote un robot n'a pas d'étiquette « bonne action » ; il agit, observe une récompense, souvent bien plus tard, et doit en déduire ce qui a marché. Les exemples ne sont plus indépendants, ils dépendent de ses propres choix. Apprendre par essais et erreurs est l'objet du chapitre 11 : apprentissage par renforcement classique.

## Exercice

Ce qu'il faut observer :

- Du 2ⁿ au polynôme. Seuils et intervalles réalisent exactement la borne de Sauer ($n + 1$ et $n(n+1)/2 + 1$) ; les demi-plans pulvérisent un triangle (8/8) mais pas trois points alignés (6/8) ni un carré (14/16), et ne réalisent que 92 étiquetages sur 1 024 à 10 points.
- Une borne juste mais payée trop cher. Pour 101 seuils, la borne de l'union n'est violée que dans 0,10 % des tirages au pire, mais l'écart mesuré colle à celui d'une seule règle (0,140 contre 0,136 à N = 100).
- Une borne VC qui a la bonne forme et la mauvaise taille. Respectée dans 100 % des 630 entraînements, elle suit la tendance (l'écart croît avec le degré, décroît avec N) mais vaut de 9,8 à plus de 500 fois l'écart mesuré.
- Aucun repas gratuit. Sur les 1 024 cibles, les quatre algorithmes font exactement 1/2 ; sur les seuils, le plus proche voisin monte à 10/11 et l'anti-voisin tombe à 1/11.

Questions de réflexion : la classe $\mathbf{1}[\sin(\omega x) \ge 0]$ n'a qu'un paramètre ; combien de points peut-elle pulvériser ? La borne VC demande environ 54 000 exemples pour garantir un écart de 0,1 avec des coniques ; l'expérience mesure un écart de 0,016 dès 200 exemples : laquelle des deux affirmations est « vraie », et que sait la borne que l'expérience ignore ?

## Auto-évaluation

Question : Pourquoi l'erreur d'entraînement de la règle choisie par l'ERM est-elle en moyenne trop optimiste, alors que l'erreur d'entraînement d'une règle fixée d'avance ne l'est pas ?

Réponse : Pour une règle fixée d'avance, $\hat R(h)$ est la moyenne de N erreurs indépendantes : son espérance est exactement $R(h)$. L'ERM, elle, choisit la règle dont l'erreur d'entraînement est la plus basse : parmi des règles proches, elle retient celle que le hasard de l'échantillon a le plus favorisée. C'est un biais de sélection (dans l'exercice : 0,087 d'optimisme en moyenne à N = 20). D'où le besoin d'une borne uniforme, valable pour toutes les règles de la classe à la fois.

Question : Avec δ = 0,05 et une classe de 101 règles, combien d'exemples faut-il pour garantir un écart γ = 0,05 entre erreur réelle et erreur d'entraînement ? Et avec un million de règles ?

Réponse : $N \ge \ln(2k/\delta)/(2\gamma^2)$. Pour $k = 101$ : $\ln(4\,040) = 8{,}30$, divisé par $2 \times 0{,}0025 = 0{,}005$, soit 1 661 exemples. Pour $k = 10^6$ : $\ln(4 \times 10^7) = 17{,}5$, soit 3 501. La taille de la classe n'intervient que par son logarithme : 10 000 fois plus de règles, à peine 2 fois plus d'exemples.

Question : Les intervalles [a, b] de la droite réelle (étiquette 1 dedans, 0 dehors) : quelle est leur dimension VC ? Justifiez les deux sens.

Réponse : 2. Deux points sont pulvérisés : 00 (intervalle qui ne contient aucun des deux), 10, 01 (un intervalle autour d'un seul point), 11 (un intervalle qui contient les deux). Trois points $x_1 \lt x_2 \lt x_3$ ne le sont jamais : l'étiquetage 1, 0, 1 exigerait un intervalle contenant $x_1$ et $x_3$ mais pas $x_2$, impossible puisqu'un intervalle contenant les deux extrémités contient tout ce qui est entre. Vérification par Sauer : sur 3 points, $1 + 3 + 3 = 7$ étiquetages, soit 8 moins 1.

Question : Coniques (d_VC = 6), N = 10 000, erreur d'entraînement 0,114 : la borne VC vaut 0,33 alors que l'erreur de test mesurée est 0,114. La borne est-elle fausse ? Pourquoi est-elle si lâche ?

Réponse : Non : elle affirme seulement que l'erreur réelle est inférieure à 0,33 avec probabilité au moins 95 %, ce qui est vrai (et vrai dans les 630 entraînements de l'exercice). Elle est lâche parce qu'elle vaut pour toute distribution, y compris les pires, et pour toute règle de la classe, quel que soit l'algorithme ; ses constantes (8 et 4) et ses étapes (symétrisation, borne de l'union sur $m_{\mathcal H}(2N)$, lemme de Sauer) accumulent des majorations. Sur un cercle bruité, distribution très « gentille », elle ne peut pas être serrée.

Question : Le théorème du no free lunch dit que tous les algorithmes ont la même erreur hors entraînement en moyenne. Pourquoi l'apprentissage marche-t-il quand même ?

Réponse : Parce que la moyenne porte sur toutes les fonctions cibles, uniformément, y compris celles où des entrées voisines ont des étiquettes sans aucun rapport. Les problèmes réels ont de la régularité ; un algorithme dont le biais inductif correspond à cette régularité bat le hasard (plus proche voisin à 10/11 sur les cibles « seuil » de l'exercice). Le théorème rappelle que cette supériorité est relative à une hypothèse sur la cible, jamais absolue : sur des cibles qui alternent 0 et 1, l'anti-voisin gagnerait.

## Pour aller plus loin

- Articles originaux : Vapnik et Chervonenkis (1971), Valiant (1984), Blumer et al. (1989), Wolpert (1996) : liens dans les fiches ci-dessus. Le livre de V. Vapnik, *The Nature of Statistical Learning Theory* (Springer, 1995, doi:10.1007/978-1-4757-2440-0), présente la théorie par son auteur, avec peu de preuves.
- Y. Abu-Mostafa, *Learning from Data* (Caltech) : les cours 5 « Training versus Testing », 6 « Theory of Generalization » et 7 « The VC Dimension » (page des cours, vidéos et transparents ; cours 7 en vidéo ; transparents du cours 7). La borne VC de ce chapitre et la règle « N ≥ 10 d_VC » en viennent.
- Stanford CS229, notes de cours de T. Ma et A. Ng (PDF) : §8.3 « Sample complexity bounds » (borne de l'union, Hoeffding, classe finie, dimension VC) ; le §8.2 traite de la double descente, prélude au chapitre 27.
- F. Bach, *Learning Theory from First Principles* (MIT Press, 2024 ; version PDF de l'auteur, Inria et ENS) : chapitre 2 pour les théorèmes « no free lunch », chapitre 4 pour la décomposition approximation-estimation et la complexité de Rademacher, qui affine la dimension VC.
- J. Kun, « Probably Approximately Correct — a Formal Theory of Learning » (blog *Math ∩ Programming*, 2014) : le cadre PAC expliqué par un mathématicien à des programmeurs, avec du code.
