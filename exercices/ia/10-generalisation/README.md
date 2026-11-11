# Exercice 10 : pulvérisation, dimension VC, bornes de généralisation et no free lunch

| | |
|---|---|
| **Chapitre lié** | [10. Généraliser : la théorie (PAC, VC)](https://laclasse.poneyponey.fr/ia/p04-statistique/10-generalisation-theorie/) |
| **Exécution** | locale, CPU (NumPy, SciPy, scikit-learn ; aucun GPU nécessaire) |
| **Commande** | `make ex NUM=10` (éditeur marimo), `make ex NUM=10 MODE=run` (complet), `make ex NUM=10 MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 35 s au total (deux mesures, 35 et 35 s hors attente du verrou), dont 27 à 29 s pour les 630 entraînements de la partie C ; smoke : 5 s (M1 Pro, 14/10/2026) |
| **Travail estimé** | 45 min |
| **Seed** | 42 (`seed_everything(42)` et `np.random.default_rng(42)`) |
| **Données** | générées par le notebook (points uniformes, étiquettes bruitées) ; rien n'est téléchargé |

## Objectif

Mesurer ce que garantit la théorie de Vapnik-Chervonenkis : compter les étiquetages qu'une classe de classifieurs peut réaliser (pulvérisation, dimension VC, lemme de Sauer), vérifier qu'une borne de généralisation tient réellement dans 95 % des tirages, constater qu'elle est juste mais très pessimiste, puis voir par un calcul exact qu'aucun algorithme ne généralise sans hypothèse sur la cible (no free lunch).

Perte : la **perte 0-1** (taux d'erreur). Risque empirique $\hat R(h)$ = erreur sur les $N$ exemples d'entraînement ; risque réel $R(h)$ = erreur sur de nouvelles données (calculé exactement en partie B, estimé sur 100 000 points de test en partie C).

## Ce que vous allez faire

**Partie A : pulvériser** (« from scratch », puis bibliothèque).

1. Compter à la main les étiquetages réalisés sur $n$ points de la droite par les seuils $\mathbb 1[x \ge t]$ et les intervalles $\mathbb 1[a \le x \le b]$ ; comparer à $2^n$ et à la borne de Sauer $\sum_{i \le d}\binom{n}{i}$.
2. En 2D, tester chaque étiquetage par programmation linéaire (`scipy.optimize.linprog`) : existe-t-il $(w, b)$ avec $s_i(w \cdot x_i + b) \ge 1$, où $s_i = 2y_i - 1 \in \{-1, +1\}$ ? Triangle, trois points alignés, carré.
3. Tracer la fonction de croissance des demi-plans pour $n = 1$ à 10.
4. Construire les monômes de degré $\le p$ à la main, vérifier contre `PolynomialFeatures`, puis montrer que les coniques (degré 2, $D = 6$ monômes) pulvérisent 6 points tirés au hasard, mais pas les 7 points tirés ensuite.

**Partie B : une classe finie** (101 seuils sur $[0, 1]$, 20 % d'étiquettes inversées, risque réel exact $R(h_t) = 0{,}2 + 0{,}6\,|t - 0{,}3|$).

5. Pour $N$ = 20, 100, 500, 2 000, tirer 2 000 jeux d'entraînement ; mesurer le plus grand écart $|R - \hat R|$ sur les 101 hypothèses et le comparer à la borne Hoeffding + union $\gamma = \sqrt{\ln(2k/\delta)/(2N)}$ (notes CS229, § 8.3).

**Partie C : une classe infinie** (2D, cercle $x_1^2 + x_2^2 = 0{,}5$, 10 % d'étiquettes inversées, donc erreur minimale possible 0,10).

6. Pour les degrés 1, 2, 3, 4, 6, 8, 10 ($d_{VC} = (p+1)(p+2)/2$ = 3 à 66) et $N$ de 20 à 10 000, 10 répétitions : régression logistique quasi sans régularisation, erreurs d'entraînement et de test en perte 0-1.
7. Calculer la borne VC $R \le \hat R + \Omega$, $\Omega = \sqrt{\tfrac{8}{N}\ln\tfrac{4\,m_{\mathcal H}(2N)}{\delta}}$ (forme du cours d'Abu-Mostafa, *Learning from Data*, Caltech, cours 7), avec $m_{\mathcal H}(2N) \le \sum_{i \le d_{VC}}\binom{2N}{i}$ calculé en logarithme.
8. Chercher le $N$ à partir duquel $\Omega \le 0{,}1$.

**Partie D : no free lunch** (domaine de 10 points, entraînement sur les 5 pairs, prédiction des 5 impairs).

9. Énumérer les 1 024 fonctions cibles et calculer en fractions exactes la précision hors entraînement de quatre algorithmes : plus proche voisin, majorité, toujours 0, anti-voisin. Recommencer avec les seules 11 cibles « seuil ».

Figure produite : `figures/borne-vc.png` (gauche : degré 2 selon $N$ ; droite : $N$ = 200 selon le degré).

## Résultat attendu

Le notebook vérifie ces seuils en mode complet et s'arrête avec une erreur si l'un manque :

| Contrôle | Seuil | Obtenu (14/10/2026) |
|---|---|---|
| Seuils / intervalles en 1D | $n+1$ et $n(n+1)/2 + 1$ étiquetages | exact pour $n$ = 1 à 6 (= Sauer avec $d$ = 1 et 2) |
| Demi-plans : triangle, alignés, carré | 8/8, 6/8, 14/16 | 8/8, 6/8, 14/16 (Sauer : 15) |
| Croissance des demi-plans | $n^2 - n + 2$ pour $n \ge 3$ | 14, 22, 32, 44, 58, 74, 92 pour $n$ = 4 à 10 (Sauer : 15, 26, 42, 64, 93, 130, 176 ; $2^n$ : jusqu'à 1 024) |
| Monômes à la main contre scikit-learn | écart < 10⁻¹² | ≈ 10⁻¹⁶ |
| Coniques | 6 points pulvérisés, 7 non | 64/64 ; 126/128 |
| B. Borne de l'union violée | dans ≤ 5 % des tirages | 0,10 % pour N = 20 (2 tirages sur 2 000), 0 % pour N = 100, 500, 2 000 |
| C. Borne VC respectée | dans ≥ 95 % des 630 entraînements | 100 % |
| C. Degré 2, N = 10 000 | erreur de test ≤ 0,12 | 0,114 |
| C. Degré 10, N = 50 | écart train/test ≥ 0,2 | 0,236 (0,008 contre 0,244) |
| C. Borne vide de sens | $\Omega > 1$ pour tout $N \le 100$ | $\Omega$ de 1,22 (degré 1, N = 100) à 3,58 |
| C. Pessimisme de la borne | $\Omega \ge 5 \times$ l'écart mesuré | rapport de 9,8 (degré 10, N = 20) à plus de 500 |
| C. Degré 2, N = 10 000 | borne $\hat R + \Omega \le 0{,}5$ | 0,33 |
| D. No free lunch | 1/2 exactement pour les 4 algorithmes | 1/2, 1/2, 1/2, 1/2 |
| D. Cibles seuils | plus proche voisin ≥ 0,9 | 10/11 ≈ 0,909 (majorité 42/55 ≈ 0,764 ; toujours 0 : 5/11 ; anti-voisin : 1/11) |

Autres valeurs affichées :

- **Partie B** : $\gamma$ (union, 101 hypothèses) = 0,456, 0,204, 0,091, 0,046 pour N = 20, 100, 500, 2 000 ; le quantile à 95 % de l'écart maximal mesuré vaut 0,304, 0,140, 0,060, 0,031, presque exactement le $\gamma$ de Hoeffding pour **une seule** hypothèse (0,304, 0,136, 0,061, 0,030). L'ERM choisit un seuil dont le risque réel moyen vaut 0,256 (N = 20 ; son erreur d'entraînement est en moyenne 0,087 plus basse) puis 0,200 (N = 2 000, l'optimum).
- **Partie C**, degré 2 : erreurs entraînement / test 0,185 / 0,279 (N = 20), 0,126 / 0,142 (N = 200), 0,114 / 0,114 (N = 10 000) ; borne $\hat R + \Omega$ = 2,99, 1,29, 0,33. À N = 200 : erreur de test minimale aux degrés 2 à 4 (0,14), 0,19 au degré 10 où l'erreur d'entraînement tombe à 0,05.
- **Taille nécessaire** pour $\Omega \le 0{,}1$ : 28 341 exemples pour $d_{VC}$ = 3, 53 862 pour 6, 88 053 pour 10, 568 960 pour 66, soit environ 9 000 × $d_{VC}$, alors que l'écart mesuré est déjà inférieur à 0,01 vers 500 à 2 000 exemples.
- À N = 20, $\Omega$ plafonne à 3,58 dès $d_{VC} \ge 40 = 2N$ : la borne de Sauer vaut alors $2^{2N}$, tous les étiquetages.

## Ce qu'il faut observer

- **Partie A.** Tant que $n \le d_{VC}$, une classe réalise les $2^n$ étiquetages ; au-delà, son nombre d'étiquetages ne croît plus que comme un polynôme de degré $d_{VC}$ (lemme de Sauer). C'est ce passage de l'exponentiel au polynomial qui rend une garantie possible. La dimension VC demande **un** ensemble pulvérisé, pas tous : trois points alignés ne le sont pas, la dimension vaut quand même 3.
- **Partie B.** La borne tient (au plus 0,10 % de violations pour 5 % autorisés) mais elle paie la borne de l'union : les 101 seuils sont tellement corrélés que l'écart maximal se comporte comme celui d'une seule hypothèse.
- **Partie C.** Plus le degré monte, plus l'erreur d'entraînement baisse et plus l'écart avec le test se creuse à $N$ fixé ; il faut d'autant plus d'exemples pour le refermer. La borne VC capte cette tendance (Ω croît avec $d_{VC}$ et décroît comme $\sqrt{\ln N / N}$) mais elle est 10 à plusieurs centaines de fois plus grande que l'écart mesuré, et au-dessus de 1 (donc inutile) pour tous les petits $N$. Elle vaut pour toutes les distributions, y compris les pires : sur une distribution gentille, elle est forcément lâche.
- **Partie C, détail.** Le degré 2 plafonne à 0,114 et non à 0,10 : la régression logistique minimise l'entropie croisée, pas la perte 0-1. Ce n'est pas un défaut de la borne, qui porte sur toute fonction de la classe, quel que soit l'algorithme.
- **Partie D.** En moyenne sur toutes les cibles, chaque algorithme fait exactement 1/2 hors de l'ensemble d'entraînement, y compris l'anti-voisin absurde. Dès qu'on suppose quelque chose (« la cible est un seuil »), les algorithmes se départagent : c'est l'hypothèse, pas l'algorithme seul, qui permet de généraliser.

## Questions de réflexion

1. Les demi-plans du plan ont 3 paramètres et une dimension VC de 3, les coniques 6 et 6. La dimension VC est-elle toujours le nombre de paramètres ? (Indice : la classe $\mathbb 1[\sin(\omega x) \ge 0]$ n'a qu'un paramètre $\omega$ ; combien de points peut-elle pulvériser ?)
2. En partie B, pourquoi l'écart de l'hypothèse choisie par l'ERM est-il en moyenne positif (son erreur réelle dépasse son erreur d'entraînement) ? Que se passerait-il si l'on mesurait l'écart d'un seuil fixé à l'avance ?
3. La borne VC demande environ 54 000 exemples pour garantir un écart de 0,1 avec des coniques ; l'expérience mesure un écart de 0,016 dès 200 exemples. Laquelle de ces deux affirmations est « vraie » ? Qu'est-ce que la borne sait que l'expérience ignore ?
4. Un réseau de neurones moderne a des millions de paramètres et s'entraîne sur moins d'exemples. Que dirait la borne VC de son erreur de test ? (Le chapitre 27 y revient.)

## Pour aller plus loin

- Remplacer les 10 % de bruit par 0 % (cas réalisable) : l'écart train/test se referme-t-il plus vite ? (Les bornes du cas réalisable sont en $1/N$ au lieu de $1/\sqrt N$.)
- Ajouter une régularisation (C = 1 au lieu de 10⁴) au degré 10 : l'erreur de test à N = 200 baisse-t-elle ? La borne VC, elle, ne change pas : pourquoi ?
- Notes du cours CS229 (Stanford), partie « Learning theory » (§ 8.3) : https://cs229.stanford.edu/main_notes.pdf
- Y. Abu-Mostafa, *Learning from Data*, cours 7 « The VC dimension » (Caltech) : https://work.caltech.edu/slides/slides07.pdf
