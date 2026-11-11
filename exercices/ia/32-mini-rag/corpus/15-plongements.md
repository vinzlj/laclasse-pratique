# 15. Plongements avant les Transformers et préhistoire contrastive

Repère historique (1954 à 2015 (word2vec : 2013)) : Représenter le sens d'un mot par un vecteur de nombres, de façon que deux mots de sens voisin aient des vecteurs proches, et apprendre ces vecteurs à partir de textes bruts, sans dictionnaire ni étiquette.

Pour un ordinateur, un mot n'est qu'une suite de lettres : « roi » et « reine » n'ont pas plus en commun que « roi » et « trois ». En 1954, le linguiste Zellig Harris (Université de Pennsylvanie) avance que la différence de sens va avec la différence des contextes où les mots apparaissent. La recherche d'information en tire des outils de comptage : la pondération IDF de Karen Spärck Jones (Cambridge, 1972), le modèle vectoriel de Gerard Salton et de son équipe de Cornell (système SMART, années 1960 et 1970, article de référence en 1975), puis l'analyse sémantique latente (Bellcore et Université de Chicago, 1990), qui compresse les comptages en une centaine de dimensions. En 2003, Yoshua Bengio et ses collègues (Université de Montréal) font apprendre un vecteur par mot à un réseau de neurones qui prédit le mot suivant, mais l'entraînement prend des semaines. En 2013, Tomáš Mikolov et ses collègues chez Google simplifient tout : word2vec apprend des vecteurs de qualité sur des milliards de mots en une journée. La même année, on remarque que vecteur(« king ») − vecteur(« man ») + vecteur(« woman ») tombe près de « queen ». En parallèle, les réseaux « siamois » des Bell Labs (1993), la perte contrastive (2006) et la perte triplet (née de l'apprentissage de métrique en 2005, popularisée par FaceNet en 2015) appliquent la même idée aux signatures et aux visages.

## L'idée en une image

« You shall know a word by the company it keeps » : on connaît un mot à ses fréquentations, écrivait le linguiste britannique J. R. Firth en 1957. Si vous lisez « Le *glorb* a aboyé toute la nuit, puis le *glorb* a rongé son os », vous savez à peu près ce qu'est un *glorb* sans l'avoir jamais vu. C'est l'hypothèse distributionnelle : des mots qui apparaissent dans des contextes semblables ont des sens semblables. Tout ce chapitre en tire des méthodes de plus en plus efficaces.

Le point de départ est l'encodage one-hot : avec un vocabulaire de 3 251 mots (celui de l'exercice, sur les pièces de Shakespeare), chaque mot est un vecteur de 3 251 cases avec un seul 1. Tous les mots sont alors à la même distance les uns des autres. L'arrivée est un plongement : chaque mot est un point dans un espace de quelques dizaines de dimensions (50 dans l'exercice, 300 dans les modèles publiés), placé de façon que les mots de sens voisin soient voisins. Imaginez une carte où « three » est à côté de « four », « father » à côté de « son », et où le trajet de « he » à « she » est le même que celui de « his » à « her ».

Pour un développeur web, la comparaison la plus proche est celle des index de recherche : une requête et des documents transformés en vecteurs, puis triés par ressemblance. C'est d'ailleurs de là que vient la moitié de l'histoire.

| | Avant : one-hot | Après : plongement word2vec (exercice 15) |
|---|---|---|
| Taille d'un vecteur | 3 251 (un par mot du vocabulaire) | 50 |
| Nombres non nuls | 1 | 50 |
| Cosinus entre « three » et « four » | 0 | 0,90 |
| Plus proches voisins de « king » | aucun sens : tous à égalité | henry, edward, richard, lewis, warwick, vi |
| Analogies réussies (16 testées) | 0 | 8 (dont lord : lady :: king : queen) |
| Ce qui coince | ne sait rien de la ressemblance | un seul vecteur par mot, quel que soit le contexte ; il faut beaucoup de texte |

Entre les deux, une étape : compter. Décrire un mot par les documents ou les mots qui l'entourent (TF-IDF, analyse sémantique latente ou LSA, sections 2 et 3) donne déjà une partie du résultat, sans aucun apprentissage. L'exercice montre que, sur un petit corpus, cette étape fait presque jeu égal avec word2vec ; l'avantage de word2vec tient surtout à sa capacité à avaler des milliards de mots.

## Le mécanisme

Le fil de l'exercice 15, du texte brut aux vecteurs de mots :

### 1. One-hot : tous les mots à égale distance

Numérotons les $V$ mots du vocabulaire. Le mot numéro $i$ devient le vecteur $e_i$ de $\R^V$, avec un 1 en position $i$ et des 0 ailleurs. Pour comparer deux vecteurs, on utilisera tout au long du chapitre la similarité cosinus :

(Formule : Similarité cosinus.)

Le cosinus vaut 1 si les deux vecteurs pointent dans la même direction, 0 s'ils sont perpendiculaires et −1 s'ils sont opposés. Il ignore la longueur : un document deux fois plus long, avec les mêmes proportions de mots, a le même cosinus. Pour deux mots différents en one-hot, $e_i \cdot e_j = 0$ : le cosinus vaut 0 entre « king » et « queen » comme entre « king » et « three ». L'exercice le vérifie sur ses 3 251 mots. Ce n'est pas faux, c'est vide : le codage ne contient aucune information sur le sens, et un modèle qui le reçoit en entrée doit tout apprendre mot par mot. Ce qu'il apprend sur « king » ne lui sert à rien pour « queen ».

### 2. Compter : sac de mots et TF-IDF

Le sac de mots. En recherche d'information, on représente un document par le nombre de fois qu'il contient chaque mot du vocabulaire : c'est le sac de mots. L'ordre est perdu, et l'exercice le vérifie avec scikit-learn : « the dog bit the man » et « the man bit the dog » ont exactement le même vecteur.

La pondération TF-IDF. Tous les mots ne se valent pas : « the » est partout, « poison » est rare et dit beaucoup sur une réplique. En 1972, Karen Spärck Jones propose de pondérer chaque terme par sa rareté parmi les documents. Le poids d'un mot $t$ dans un document $d$ :

(Formule : TF-IDF (forme de l'exercice).)

Sur les répliques de TinyShakespeare, chiffres de l'exercice :

| Mot | Occurrences | Répliques qui le contiennent (df) | IDF = ln(7 222 / df) |
|---|---|---|---|
| the | 6 285 | 2 833 | 0,94 |
| king | 448 | 357 | 3,01 |
| daughter | 119 | 105 | 4,23 |
| poison | 27 | 22 | 5,79 |

« the » n'a pas un IDF nul : beaucoup de répliques sont courtes (« Of love? ») et ne le contiennent pas. Il pèse quand même six fois moins que « poison ». Il existe de nombreuses variantes (logarithme du TF, IDF lissé). Avec `smooth_idf=False` et sans normalisation (`norm=None`), scikit-learn calcule ln(N/df) + 1 : le + 1 évite d'annuler un mot. Par défaut, il lisse en plus l'IDF (ln((1 + N)/(1 + df)) + 1) et ramène chaque vecteur à la norme 1. L'exercice retrouve exactement la matrice de `TfidfVectorizer` réglé ainsi (écart 0), une fois le + 1 pris en compte.

Le modèle vectoriel. Gerard Salton et ses collègues (Cornell, système SMART, années 1960 et 1970) font de chaque document et de chaque requête un vecteur de termes pondérés (TF-IDF dans l'article de 1975, devenu la référence du modèle vectoriel), et classent les documents par cosinus avec la requête. C'est le principe des moteurs de recherche pendant des décennies, et l'ancêtre direct des pondérations qui les équipent encore. Le moteur de l'exercice, sur 7 222 répliques :

```text
requête « love daughter »
  0,586  KING RICHARD III : Then know, that from my soul I love thy daughter.
  0,541  BENVOLIO         : Of love?
  0,525  BENVOLIO         : In love?

requête « crown king england »
  0,482  WARWICK          : What good is this to England and himself!
  0,457  Second Keeper    : But, if thou be a king, where is thy crown?
  0,452  KING HENRY VI    : Where did you dwell when I was King of England?
```

Deux répliques de deux mots arrivent en tête pour « love » : courtes, elles ont presque toute leur « masse » sur ce mot, et le cosinus ne pénalise pas la brièveté. C'est un défaut connu, que corrigent les pondérations ultérieures.

De la recherche de documents aux vecteurs de mots. Rangeons les comptages pondérés dans une matrice $X$ de taille (mots × répliques). Une colonne décrit une réplique : c'est le vecteur du moteur de recherche. Une ligne décrit un mot par les répliques où il apparaît : c'est déjà un plongement, très creux (7 222 dimensions, presque toutes nulles). Deux mots qui apparaissent dans les mêmes répliques ont des lignes proches. Le premier pas vers le sens est fait, sans rien apprendre.

Mais les voisins de « king » dans cette représentation sont *and, the, of, to, henry, his* : une longue réplique qui parle d'un roi contient aussi beaucoup de mots outils, et la réplique entière est un contexte trop large.

Encadré : un découpage naïf :
Avant de compter, il faut découper le texte en mots. L'exercice utilise une expression régulière, `[a-z]+(?:'[a-z]+)?`, après passage en minuscules : des lettres, avec au plus une apostrophe interne (« i'll », « o'er »). La ponctuation disparaît, « king's » et « king » deviennent deux mots sans rapport, et l'on jette tout mot vu moins de 5 fois. Résultat : 12 350 mots distincts, dont 3 251 gardés ; 7,7 % des occurrences du texte tombent hors du vocabulaire (mot hors vocabulaire) et sont ignorées. Sur des langues plus riches en formes (le français conjugué, le finnois, l'allemand et ses mots composés), le problème s'aggrave. La tokenisation en sous-mots (BPE, WordPiece) qui règle ce problème est le sujet du chapitre 19.

### 3. Compresser : l'analyse sémantique latente (LSA)

Deux problèmes de la recherche par mots-clés : la synonymie (un document sur les « automobiles » ne répond pas à la requête « voiture ») et la polysémie (« avocat » le fruit et « avocat » le métier). En 1990, Scott Deerwester, Susan Dumais et leurs collègues proposent l'analyse sémantique latente : remplacer la matrice termes × documents par une approximation de petit rang, qui force les mots apparaissant dans des documents semblables à partager les mêmes quelques « dimensions latentes ».

L'outil est la décomposition en valeurs singulières :

(Formule : SVD tronquée (LSA).)

Parmi toutes les matrices de rang $k$, $X_k$ est celle qui minimise l'erreur de reconstruction $\lVert X - X_k \rVert^2$ (somme des carrés des écarts, case par case) : c'est le théorème d'Eckart et Young. C'est donc la même mathématique que l'analyse en composantes principales du chapitre 08, appliquée à la matrice de comptages sans la centrer.

L'exercice calcule la SVD « à la main » : les vecteurs singuliers à gauche de $X$ sont les vecteurs propres de $X X^\top$ (une matrice mots × mots de cooccurrences pondérées), et $\sigma_i = \sqrt{\lambda_i}$. Les 100 valeurs singulières coïncident avec celles de `TruncatedSVD` de scikit-learn à $1{,}5 \times 10^{-7}$ près (relatif). Les 100 dimensions gardent 35,6 % de l'« énergie » de la matrice ($\sum_{i \le 100} \sigma_i^2 / \sum_i \sigma_i^2$), et le calcul prend 7 s.

L'effet : les voisins de « king » deviennent *regal, henry's, seated, henry, lewis, london*. Les mots outils ont disparu. Dans l'article de 1990, sur la collection médicale MED (1 033 documents, 5 823 termes), la recherche par LSA bat la recherche par mots-clés sur la plupart des niveaux de rappel.

La limite : une SVD d'une matrice de vocabulaire × corpus coûte cher, et tout est à refaire quand le corpus change. Il faudra près de 25 ans pour comprendre que word2vec fait, en fait, presque la même chose (section 5).

### 4. Apprendre les vecteurs : le modèle de langage neuronal (2003)

La représentation distribuée est une idée connexionniste des années 1980 (chapitre 07) : un concept est un motif d'activité sur de nombreuses unités. Yoshua Bengio, Réjean Ducharme, Pascal Vincent et Christian Jauvin l'appliquent aux mots en 2003, dans un modèle de langage : un modèle qui donne la probabilité du mot suivant connaissant les $n - 1$ précédents.

Les modèles de l'époque, les *n*-grammes, comptent les suites de mots vues dans le corpus. Ils butent sur la malédiction de la dimension : avec 100 000 mots de vocabulaire, il existe $10^{50}$ suites de 10 mots, et presque toutes les phrases de test contiennent des suites jamais vues. L'idée de Bengio : si « chat » et « chien » ont des vecteurs proches, ce que le modèle apprend sur « le chat dort » se transfère à « le chien dort ».

La table $C$ (un vecteur de $m$ nombres par mot, $m$ = 30, 60 ou 100 dans l'article) est partagée par toutes les positions du contexte et apprise en même temps que le reste du réseau, par rétropropagation de l'entropie croisée du mot suivant (chapitre 01). Les plongements sont un sous-produit : personne ne les a demandés, ils émergent parce qu'ils aident à prédire. Résultat : le meilleur *n*-gramme a une perplexité supérieure d'environ 24 % à celle du réseau sur le corpus Brown (312 contre 252, sur environ 1,2 million de mots), et l'écart est d'environ 8 % sur AP News (117 contre 109, sur 14 millions de mots). Le prix : sur AP News, 5 époques ont pris trois semaines sur 40 processeurs, surtout à cause de la softmax finale, qui calcule un score pour chacun des 17 964 mots du vocabulaire à chaque exemple. La perplexité et le modèle de langage moderne sont l'objet du chapitre 20 ; le petit modèle de Bengio est reconstruit pas à pas dans la vidéo de Karpathy citée en fin de chapitre.

En 2008, Ronan Collobert et Jason Weston (NEC Labs) entraînent des plongements sur tout Wikipédia (631 millions de mots) sans softmax : le réseau doit donner un score plus élevé à une vraie fenêtre de texte qu'à la même fenêtre dont le mot central a été remplacé par un mot au hasard. Ces plongements pré-entraînés améliorent ensuite nettement l'étiquetage des rôles sémantiques (qui fait quoi à qui dans la phrase), la tâche principale de l'article : l'erreur passe de 17,3 % à 14,3 %. On y voit déjà deux idées de word2vec : se débarrasser de la softmax, et apprendre en opposant le vrai au faux.

### 5. word2vec : skip-gram et échantillonnage négatif (2013)

Tomáš Mikolov, Kai Chen, Greg Corrado et Jeffrey Dean (Google) font le pari inverse de la profondeur : un modèle presque linéaire, mais entraîné sur beaucoup plus de texte. word2vec propose deux architectures :

- sac de mots continu : prédire le mot central à partir de la moyenne des vecteurs de ses voisins ;
- skip-gram : à partir du mot central, prédire chacun de ses voisins.

Il n'y a pas de couche cachée non linéaire : un mot est un vecteur, et le score d'un couple (mot, voisin) est un simple produit scalaire. Dans l'article, un CBOW apprend des vecteurs de 300 dimensions sur 1,6 milliard de mots en moins d'une journée sur un seul processeur (0,6 jour ; deux jours pour skip-gram) ; le second article parle de plus de 100 milliards de mots par jour avec une implémentation optimisée. C'est une affaire d'échelle : on ne gagne pas par un modèle plus malin, mais par un modèle assez simple pour avaler le web.

Les données d'entraînement. On n'a besoin d'aucune étiquette : le texte fournit lui-même les questions et les réponses. C'est une tâche prétexte d'apprentissage auto-supervisé (chapitre 03). Pour chaque mot du corpus, on forme les paires (mot central, voisin) dans une fenêtre de quelques mots :

```text
fenêtre de 2 autour de « king » :   ... the  good  KING  of  england ...
paires (centre, voisin) :  (king, the) (king, good) (king, of) (king, england)
```

Dans l'exercice, la taille de la fenêtre est tirée au hasard entre 1 et 4 pour chaque mot (comme dans le code C d'origine) : les voisins immédiats sont vus plus souvent que les lointains. Une époque produit environ 643 000 paires.

L'objectif de départ : une softmax. Chaque mot $w$ a deux vecteurs : $\mathbf{v}_w$ quand il est au centre, $\mathbf{u}_w$ quand il est voisin. Skip-gram modélise la probabilité d'un voisin $c$ sachant le centre $w$ :

(Formule : Skip-gram avec softmax complète.)

C'est une softmax ordinaire sur $V$ classes (chapitre 01), et c'est tout le problème : chaque paire demande $V$ produits scalaires, soit un million dans l'article de 2013, qui garde le million de mots les plus fréquents de Google News. Le premier article contourne le coût par une softmax hiérarchique (environ $\log_2 V$ décisions binaires le long d'un arbre de mots). Le second en propose une plus simple.

L'échantillonnage négatif. Au lieu de demander « quel est le voisin parmi un million de mots ? », on pose une question oui/non : « ce couple (mot, voisin) est-il un vrai couple du texte, ou un couple fabriqué au hasard ? ». Pour chaque vraie paire $(w, c)$, on tire $K$ mots au hasard $n_1, \dots, n_K$ (les « négatifs »), et l'on entraîne une régression logistique à répondre 1 pour la vraie paire et 0 pour les fausses. C'est l'échantillonnage négatif :

(Formule : Perte du skip-gram avec échantillonnage négatif, pour une paire.)

C'est exactement l'entropie croisée binaire du chapitre 01, sommée sur une paire positive et $K$ négatives. Le coût d'une paire passe de $V$ à $K + 1$ produits scalaires : 6 au lieu de 3 251 dans l'exercice, au lieu d'un million dans l'article de 2013. L'idée vient de l'estimation contrastive par bruit (*noise-contrastive estimation*, NCE) de Michael Gutmann et Aapo Hyvärinen (Helsinki, 2010) : apprendre une distribution en apprenant à la distinguer d'un bruit connu. L'échantillonnage négatif en est une version simplifiée, qui renonce à estimer des probabilités exactes et ne garde que ce qu'il faut pour de bons vecteurs. En simplifiant : on n'apprend plus un modèle de langage, on apprend des vecteurs.

Deux astuces de fréquence. Elles comptent autant que l'architecture :

- Les négatifs sont tirés selon la fréquence à la puissance 3/4. Avec la fréquence brute, « the » serait tiré 3,5 % du temps dans l'exercice ; avec la puissance 3/4, 1,5 %, et un mot vu 5 fois est tiré 2,6 fois plus souvent qu'avec la fréquence brute. L'article dit avoir essayé plusieurs distributions et trouvé celle-ci nettement meilleure ; le choix est empirique.
- Le sous-échantillonnage des mots fréquents : avant chaque époque, chaque occurrence d'un mot de fréquence relative $f$ est jetée avec une probabilité qui croît avec $f$. L'article écrit $P(\text{jeter}) = 1 - \sqrt{t/f}$ avec $t \approx 10^{-5}$ ; le code C publié, que suit l'exercice, garde le mot avec la probabilité $\min\bigl(1, \sqrt{t/f} + t/f\bigr)$ et $t = 10^{-3}$. « the » ($f$ = 3,5 %) n'est gardé que dans 19,7 % des cas, « king » toujours. Les fenêtres s'en trouvent élargies d'autant (le voisin de « king » n'est plus « the » mais le mot d'après), l'entraînement va plus vite (2 à 10 fois selon l'article) et les mots rares comptent plus.

Le gradient, à la main. Avec $s_+ = \sigma(\mathbf{v}_w \cdot \mathbf{u}_c)$ et $s_k = \sigma(\mathbf{v}_w \cdot \mathbf{u}_{n_k})$, et puisque $\frac{d}{dx}\log\sigma(x) = 1 - \sigma(x)$ :

$$
\frac{\partial \ell}{\partial \mathbf{v}_w} = (s_+ - 1)\,\mathbf{u}_c + \sum_{k=1}^{K} s_k\,\mathbf{u}_{n_k},
\qquad
\frac{\partial \ell}{\partial \mathbf{u}_c} = (s_+ - 1)\,\mathbf{v}_w,
\qquad
\frac{\partial \ell}{\partial \mathbf{u}_{n_k}} = s_k\,\mathbf{v}_w .
$$

Lecture : $s_+ - 1$ est négatif, donc un pas de descente de gradient rapproche $\mathbf{v}_w$ de $\mathbf{u}_c$, d'autant plus que le modèle se trompait ($s_+$ loin de 1). Chaque $s_k$ est positif : le pas éloigne $\mathbf{v}_w$ des négatifs, d'autant plus qu'ils semblaient plausibles. L'exercice compare ce gradient à celui de l'autograd de PyTorch : écart de $5{,}6 \times 10^{-17}$.

Exemple à la main, en 2 dimensions, avec un seul négatif : $\mathbf{v}_w = (1 ;\ 0{,}5)$, $\mathbf{u}_c = (0{,}8 ;\ 0{,}4)$, $\mathbf{u}_n = (-0{,}5 ;\ 1)$.

```text
score vrai couple    v·u_c = 0,8 + 0,2 = 1,0    σ(1,0) = 0,731   −ln 0,731 = 0,313
score faux couple    v·u_n = −0,5 + 0,5 = 0     σ(0)   = 0,5     −ln 0,5   = 0,693
perte = 0,313 + 0,693 = 1,006

∂ℓ/∂v = (0,731 − 1) × (0,8 ; 0,4) + 0,5 × (−0,5 ; 1)
      = (−0,215 ; −0,108) + (−0,25 ; 0,5) = (−0,465 ; 0,392)
pas de 0,025 :  v ← v − 0,025 × ∂ℓ/∂v = (1,012 ; 0,490)
```

Le score du vrai couple monte ($\mathbf{v}_w \cdot \mathbf{u}_c$ : 1,000 → 1,005) et celui du faux couple descend ($\mathbf{v}_w \cdot \mathbf{u}_n$ : 0 → −0,016). (Ici, $\mathbf{v}_w$ et $\mathbf{u}_c$ pointent déjà dans la même direction : le pas ne peut pas les « aligner » davantage, il allonge surtout $\mathbf{v}_w$ le long de $\mathbf{u}_c$.)

Le code de l'exercice, sans la boucle d'époques. Un lot de paires à la fois, gradients appliqués avec `np.add.at` (qui additionne correctement quand un même mot apparaît plusieurs fois dans le lot) :

```python
import numpy as np

def sigmoide(x):
    return 1.0 / (1.0 + np.exp(-x))

def perte_et_gradients(W, C, w, o, neg):
    """W : vecteurs « centre » (V, D) ; C : vecteurs « contexte » (V, D).
    w, o : indices des mots centraux et des vrais voisins (B,) ; neg : négatifs (B, K)."""
    vw, uc, un = W[w], C[o], C[neg]                  # (B, D), (B, D), (B, K, D)
    sp = sigmoide((vw * uc).sum(1))                  # s+ : un score par paire
    sn = sigmoide(np.einsum("bd,bkd->bk", vw, un))   # s_k : K scores par paire
    perte = -(np.log(sp).sum() + np.log(1 - sn).sum()) / len(w)
    g_vw = (sp - 1)[:, None] * uc + (sn[:, :, None] * un).sum(1)
    g_uc = (sp - 1)[:, None] * vw
    g_un = sn[:, :, None] * vw[:, None, :]
    return perte, g_vw, g_uc, g_un

# un pas de descente de gradient stochastique sur le lot
# D = dimension des vecteurs, lr = pas d'apprentissage (voir les réglages ci-dessous)
perte, g_vw, g_uc, g_un = perte_et_gradients(W, C, w, o, neg)
np.add.at(C, o, -lr * g_uc)
np.add.at(C, neg.ravel(), -lr * g_un.reshape(-1, D))
np.add.at(W, w, -lr * g_vw)
```

Réglages de l'exercice : 50 dimensions, 10 époques, 5 négatifs, pas initial de 0,025 qui décroît linéairement jusqu'à presque 0. Comme dans le code C de word2vec : 5 négatifs, puissance 3/4, seuil de sous-échantillonnage $10^{-3}$, mots vus au moins 5 fois, pas initial de 0,025 (sa valeur pour skip-gram ; 0,05 pour CBOW) et même initialisation. Le code C diffère par la fenêtre (5), la dimension (100) et le nombre d'époques (5). Initialisation : $\mathbf{v}$ uniforme dans $[-0{,}5/D ;\ 0{,}5/D]$, $\mathbf{u} = 0$. Au premier pas, tous les scores valent donc 0, toutes les probabilités 1/2, et la perte vaut exactement $(1 + K)\ln 2 = 6 \ln 2 \approx 4{,}159$. Le gradient par rapport à $\mathbf{v}$ est nul ; seul $\mathbf{u}$ bouge, ce qui suffit à briser la symétrie. 27 s d'entraînement en numpy sur le processeur d'un Mac M1 Pro.

La perte descend à 3,00 en moyenne sur la première époque et 2,54 sur la dernière. Elle ne descendra jamais près de 0 : un mot a des dizaines de voisins possibles, et les négatifs tirés au hasard sont parfois de vrais voisins. Une perte qui baisse ne dit pas grand-chose de la qualité des vecteurs : il faut des tests extérieurs (section 7).

word2vec, c'est (presque) compter. En 2014, Omer Levy et Yoav Goldberg (Université Bar-Ilan) montrent que, à l'optimum, skip-gram avec échantillonnage négatif fait en sorte que $\mathbf{v}_w \cdot \mathbf{u}_c$ égale l'information mutuelle ponctuelle du couple, décalée d'une constante :

$$
\mathbf{v}_w \cdot \mathbf{u}_c = \log \frac{P(w, c)}{P(w)\,P(c)} - \log K .
$$

Autrement dit, word2vec factorise implicitement une matrice mots × contextes de cooccurrences, comme la LSA factorise une matrice mots × documents. Les deux écoles, compter et prédire, se rejoignent. En simplifiant : ce résultat vaut si les vecteurs ont assez de dimensions pour atteindre l'optimum de chaque case ; avec 50 ou 300 dimensions, word2vec en est une approximation de rang réduit, pondérée par la fréquence des paires.

### 6. GloVe et fastText

GloVe (*Global Vectors*, Stanford, 2014). Jeffrey Pennington, Richard Socher et Christopher Manning prennent acte de ce lien et l'écrivent directement : compter d'abord les cooccurrences $X_{ij}$ (combien de fois le mot $j$ apparaît dans la fenêtre du mot $i$) sur tout le corpus, puis apprendre des vecteurs dont le produit scalaire approche $\log X_{ij}$. Leur argument est un tableau de leur article, mesuré sur 6 milliards de mots :

| | k = solid | k = gas | k = water | k = fashion |
|---|---|---|---|---|
| P(k ∣ ice) | 1,9 × 10⁻⁴ | 6,6 × 10⁻⁵ | 3,0 × 10⁻³ | 1,7 × 10⁻⁵ |
| P(k ∣ steam) | 2,2 × 10⁻⁵ | 7,8 × 10⁻⁴ | 2,2 × 10⁻³ | 1,8 × 10⁻⁵ |
| rapport | 8,9 | 0,085 | 1,36 | 0,96 |

Les probabilités seules sont difficiles à lire ; leur rapport sépare net ce qui est propre à la glace (solide, 8,9), propre à la vapeur (gaz, 0,085), et ce qui concerne les deux ou aucun (eau, mode : près de 1). Un rapport devient une différence de logarithmes, d'où des vecteurs dont les différences portent le sens. La perte est des moindres carrés pondérés :

(Formule : Objectif de GloVe.)

Comme pour l'échantillonnage négatif, l'exposant 3/4 est un choix empirique (« une amélioration modeste » sur l'exposant 1, selon l'article). GloVe a été publié avec des vecteurs pré-entraînés (jusqu'à 840 milliards de mots du web) très utilisés en traitement du langage jusqu'à l'arrivée des modèles contextuels.

fastText (Facebook, 2016-2017). Piotr Bojanowski, Edouard Grave, Armand Joulin et Tomáš Mikolov (passé de Google à Facebook) s'attaquent aux mots rares et inconnus. Dans skip-gram, chaque mot a son vecteur, indépendant de son orthographe. Dans fastText, le vecteur d'un mot est la somme des vecteurs de ses <em>n</em>-grammes de caractères (de 3 à 6 lettres), plus celui du mot entier. Avec $n = 3$ et les marques de début et de fin `<` et `>`, « where » devient :

```text
<wh   whe   her   ere   re>     plus le mot entier <where>
```

(Le trigramme `her` de « where » est distinct du mot `<her>`.) « king's » partage alors l'essentiel de ses morceaux avec « king », « kingdom » aussi, et un mot jamais vu reçoit un vecteur à partir de ses morceaux. C'est la réponse à la troisième limite du découpage naïf, avant la tokenisation en sous-mots des Transformers (chapitre 19).

### 7. Cosinus, voisins et analogies : ce que contiennent les vecteurs

Voisins. Pour le voisinage, on normalise les vecteurs et l'on trie par cosinus. Les voisins de quelques mots dans les trois représentations de l'exercice :

| Mot | TF-IDF (7 222 dim.) | LSA (100 dim.) | word2vec (50 dim.) |
|---|---|---|---|
| king | and, the, of, to, henry, his | regal, henry's, seated, henry, lewis, london | henry, edward, richard, lewis, warwick, vi |
| three | wenches, themselves, pence, four, pound | pound, money, four, five, master | four, two, five, twelve, ago, years |
| night | black, lovers, lodging, unseen, robes | apace, cords, lodging, whip, lovers | day, morning, wedding, morrow, church |
| rome | marcius, senators, coriolanus, corioli | marcius, caius, coriolanus, senators | coriolanus, aufidius, tribunes, volsces, people |

Les voisins TF-IDF et LSA sont thématiques : des mots qui apparaissent dans les mêmes répliques (« night » avec « lovers », dans *Roméo et Juliette*). Les voisins word2vec sont plutôt substituables : des mots qui peuvent prendre la même place dans une phrase (« three » et « four », « night » et « day »). C'est l'effet de la fenêtre étroite : deux mots qu'on peut échanger ont les mêmes voisins immédiats. Pour « rome », les trois représentations donnent surtout la pièce *Coriolan* : sur un petit corpus, le sujet d'une pièce domine.

Essayez par vous-même. La démo contient les vecteurs word2vec de l'exercice pour les 1 004 mots les plus fréquents. Tapez un mot (ou touchez un exemple), comparez ses voisins à 50 dimensions (au cosinus) et ses voisins sur la carte 2D, et testez des analogies. La démo cherche parmi ces 1 004 mots seulement : ses voisins peuvent différer un peu de ceux de l'exercice (pas de *lewis* pour « king »), et l'analogie king : queen :: prince : princess en est absente (« princess » n'est pas exporté).

Cherchez : un mot dont les voisins sont bons à 50 dimensions mais dispersés sur la carte (les carrés orange) ; une analogie qui marche et une qui rate ; la différence entre la carte ACP et la carte t-SNE pour « sword ».

Les cartes 2D mentent. Pour dessiner 50 dimensions sur un écran, il faut une réduction de dimension : l'ACP ou t-SNE, toutes deux expliquées au chapitre 08. L'exercice mesure ce qu'elles conservent : la part des 10 plus proches voisins de chaque mot (à 50 dimensions) qui restent parmi ses 10 plus proches voisins sur la carte.

L'ACP garde 21 % de la variance mais seulement 7 % des voisins ; t-SNE en garde 35 % et dessine de beaux amas, mais la taille des amas et les distances entre eux n'ont pas de sens. Deux mots proches sur la carte ne le sont pas forcément dans l'espace d'origine. Ne raisonnez jamais sur une carte 2D seule : vérifiez au cosinus dans l'espace complet.

Les analogies. Le résultat qui a rendu ces vecteurs célèbres (Mikolov, Yih et Zweig, 2013, observé d'abord sur les vecteurs d'un modèle de langage récurrent, puis mesuré à grande échelle avec word2vec) : les relations entre mots sont à peu près des translations (analogie vectorielle). Pour répondre à « a est à b ce que c est à ? », on cherche le mot $d$ dont le vecteur est le plus proche de $\mathbf{b} - \mathbf{a} + \mathbf{c}$ :

(Formule : Analogie vectorielle (3CosAdd).)

L'exclusion de $a$, $b$, $c$ n'est pas un détail : sans elle, la réponse est très souvent $c$ lui-même (« king »), car $\mathbf{x}_b - \mathbf{x}_a$ est un petit décalage. L'article de word2vec et son code l'appliquent, ce qui rend les résultats moins magiques qu'il n'y paraît (Nissim, van Noord et van der Goot en font la critique en 2020, voir « Pour aller plus loin »). Pour mesurer cela, l'article de janvier 2013 publie un jeu de 19 544 analogies (8 869 « sémantiques », comme pays : capitale, et 10 675 « syntaxiques », comme singulier : pluriel) ; une réponse ne compte que si le mot le plus proche est exactement le bon.

Sur nos 200 000 mots, word2vec réussit 8 analogies sur 16 : toutes celles qui portent sur des mots grammaticaux très fréquents (he : she :: his : her, i : my :: he : his, we : our :: they : their, eye : eyes :: hand : hands…) et lord : lady :: king : queen. Mais man : woman :: king : ? donne *england*, et father : mother :: son : ? donne *wife*. L'analogie emblématique demande des corpus des milliers de fois plus grands (la figure de similarités, plus bas, compare les quatre représentations sur dix mots).

Les vecteurs apprennent aussi les biais. Si le corpus associe plus souvent « infirmière » à « elle » et « ingénieur » à « il », les vecteurs le font aussi. En 2016, Tolga Bolukbasi et ses collègues (Université de Boston et Microsoft Research) montrent que les vecteurs word2vec de Google News complètent « man : computer programmer :: woman : ? » par *homemaker* (femme au foyer), et proposent de retirer la « direction du genre » des mots qui ne devraient pas en avoir. Ce n'est pas une bizarrerie : le plongement fait exactement ce qu'on lui demande, résumer les cooccurrences du texte. Les questions d'équité sont traitées au chapitre 40.

### 8. Préhistoire contrastive : réseaux siamois, perte contrastive, perte triplet

Les plongements de mots ne sont pas les seuls. Dès les années 1990, une autre lignée apprend des vecteurs pour comparer : deux signatures sont-elles de la même personne ? deux photos montrent-elles le même visage ? Il y a trop de personnes, et pas assez d'exemples par personne, pour en faire une classe chacune. On apprend plutôt un espace où la distance dit la ressemblance : c'est l'apprentissage de métrique. L'idée « rapprocher les vrais couples, éloigner les faux » est celle de l'échantillonnage négatif ; elle deviendra au chapitre 23 le moteur de l'apprentissage auto-supervisé des images.

Le réseau siamois (Bell Labs, 1993). Jane Bromley, Isabelle Guyon, Yann LeCun et leurs collègues veulent vérifier des signatures tracées sur une tablette. Leur réseau siamois : deux copies d'un même réseau convolutif (mêmes poids) transforment chacune une signature en vecteur ; la sortie est le cosinus entre les deux vecteurs.

À l'entraînement, la cible est un cosinus de 1 pour deux vraies signatures, de −0,9 ou −1 si l'une est une imitation. À l'usage, on ne stocke que le vecteur de la signature de référence. Dans la version présentée à NIPS en 1993, avec un seuil réglé pour détecter 80 % des imitations, le système accepte 95,5 % des vraies signatures. Le même schéma sert aujourd'hui partout où l'on compare : visages, empreintes, recherche d'images et de textes semblables (chapitre 22).

La perte contrastive à marge (2006). Raia Hadsell, Sumit Chopra et Yann LeCun (New York University) écrivent une perte générale pour des paires étiquetées « semblables » ou « dissemblables », la perte contrastive :

(Formule : Perte contrastive (Hadsell, Chopra et LeCun, 2006).)

Une paire semblable est tirée l'une vers l'autre comme par un ressort ($\tfrac12 D^2$). Une paire dissemblable est repoussée, mais seulement si elle est plus proche que $m$. Avec $m = 1$ : une paire semblable à distance 0,4 coûte $0{,}5 \times 0{,}16 = 0{,}08$ ; une paire dissemblable à distance 0,4 coûte $0{,}5 \times 0{,}6^2 = 0{,}18$ ; à distance 1,2, elle ne coûte rien.

Pourquoi faut-il des paires dissemblables ? L'article le dit nettement : si l'on ne fait que rapprocher les paires semblables, la solution optimale est un réseau qui envoie tout sur le même point. La perte est nulle, et les vecteurs sont inutiles. C'est l'effondrement (*collapse*) ; le terme « contrastif » (qui oppose positifs et négatifs) désigne justement ce qui l'empêche. Toute la famille des méthodes auto-supervisées du chapitre 23 et du chapitre 26 tourne autour de cette question : comment éviter l'effondrement, avec ou sans négatifs.

La perte triplet et FaceNet (Google, 2015). La perte triplet vient de l'apprentissage de métrique : en 2005, Kilian Weinberger, John Blitzer et Lawrence Saul (méthode LMNN, NIPS 2005) règlent une distance pour que chaque exemple soit plus proche de ses voisins de même classe que des autres, avec une marge. FaceNet la popularise en l'appliquant à un réseau profond. Florian Schroff, Dmitry Kalenichenko et James Philbin ne comparent plus des paires mais des triplets : une ancre $a$ (une photo de Marie), un positif $p$ (une autre photo de Marie), un négatif $n$ (une photo de Paul). La perte triplet exige seulement que l'ancre soit plus proche du positif que du négatif, avec une marge $\alpha$ :

(Formule : Perte triplet (FaceNet).)

Exemple : si $\lVert f(a) - f(p)\rVert^2 = 0{,}6$ et $\lVert f(a) - f(n)\rVert^2 = 0{,}7$, la perte vaut $\max(0 ;\ 0{,}6 - 0{,}7 + 0{,}2) = 0{,}1$ : le triplet est dans le bon ordre, mais pas avec assez de marge. Si $\lVert f(a) - f(n)\rVert^2 = 1{,}0$, la perte est nulle et le triplet n'apprend plus rien. D'où l'importance du choix des triplets : un négatif tiré au hasard est presque toujours déjà loin. FaceNet choisit, dans chaque lot, des négatifs difficiles, mais pas les plus difficiles de tous (qui, selon l'article, mènent à de mauvais minima) : des négatifs « semi-difficiles », plus loin que le positif mais encore dans la marge. Résultat annoncé : 99,63 % de précision sur le jeu de vérification de visages LFW, avec 128 octets par visage.

| | word2vec, échantillonnage négatif | Perte contrastive (2006) | Perte triplet (2005, FaceNet 2015) |
|---|---|---|---|
| Unité d'apprentissage | (mot, voisin) + K mots au hasard | une paire étiquetée | (ancre, positif, négatif) |
| D'où viennent les positifs | du texte : mots voisins | étiquettes humaines | étiquettes (même personne) |
| Mesure | produit scalaire | distance euclidienne | distance euclidienne au carré |
| Forme de la perte | entropie croisée binaire | ressort + marge | marge sur une différence de distances |
| Descendance | InfoNCE, CLIP (ch. 23) | méthodes auto-supervisées | recherche d'images, de personnes |

### 9. Recommandation : la factorisation matricielle

Remplacez « mots » et « documents » par « utilisateurs » et « films », et les comptages par des notes : vous obtenez le problème de la recommandation. En 2006, Netflix publie plus de 100 millions de notes (1 à 5 étoiles) données par environ 500 000 clients anonymes à plus de 17 000 films, et promet un million de dollars à la première équipe qui fera 10 % de mieux que son propre système, mesuré en erreur quadratique (RMSE 0,9514 pour Netflix, 0,8563 exigé). La matrice utilisateurs × films est presque vide : chacun n'a noté qu'une petite partie des films.

La factorisation matricielle, décrite par Yehuda Koren (Yahoo! Research), Robert Bell et Chris Volinsky (AT&T Labs), membres de l'équipe en tête du concours, donne à chaque utilisateur $u$ un vecteur $\mathbf{p}_u$ et à chaque film $i$ un vecteur $\mathbf{q}_i$, et prédit la note par leur produit scalaire :

(Formule : Factorisation matricielle régularisée (Koren, Bell et Volinsky, 2009).)

C'est une erreur quadratique (chapitre 01), mais calculée seulement sur les cases connues : c'est ce qui la distingue d'une SVD, qui aurait besoin de la matrice entière. On l'optimise par descente de gradient stochastique ou par moindres carrés alternés. Exemple jouet à deux dimensions, qui se trouveraient correspondre à « action » et « romance » : Marie a $\mathbf{p} = (0{,}2 ;\ 1{,}8)$, un film d'action $\mathbf{q} = (2{,}0 ;\ 0{,}3)$, une comédie romantique $\mathbf{q}' = (0{,}4 ;\ 2{,}2)$. Notes prédites : $0{,}4 + 0{,}54 = 0{,}94$ et $0{,}08 + 3{,}96 = 4{,}04$. En pratique, les dimensions apprises ne portent pas d'étiquette ; sur les données Netflix, l'article lit la première comme une opposition entre comédies potaches ou films d'horreur et drames à héroïnes fortes, la seconde entre films indépendants salués par la critique et films grand public formatés, mais c'est une interprétation après coup. Le modèle complet de l'article ajoute une moyenne générale et des biais par utilisateur et par film (certains notent large, certains films plaisent à tous), puis des effets temporels.

Le lien avec le reste du chapitre : un film est décrit par les gens qui l'ont aimé, comme un mot par les mots qui l'entourent. C'est le filtrage collaboratif ; LSA, word2vec (qui factorise implicitement une matrice de PMI) et la recommandation sont trois variantes de « factoriser une matrice de cooccurrences ». Sa limite est le démarrage à froid : un film qui vient de sortir n'a pas de note, donc pas de vecteur. On y répond en ajoutant des descriptions du contenu (genre, texte, image), exactement comme fastText ajoute l'orthographe aux mots.

### Ce que les méthodes optimisent

| Méthode | Ce qui est optimisé | Comment |
|---|---|---|
| One-hot, TF-IDF | rien : on compte | — |
| LSA | erreur de reconstruction $\lVert X - X_k \rVert^2$ de la matrice TF-IDF par une matrice de rang $k$ | formule fermée : SVD |
| Modèle de langage neuronal (2003) | entropie croisée du mot suivant (chapitre 01) | rétropropagation |
| word2vec skip-gram | entropie croisée binaire « vrai voisin contre K mots au hasard » | descente de gradient stochastique |
| GloVe | moindres carrés pondérés sur le log des cooccurrences | AdaGrad dans l'article |
| Réseau siamois, perte contrastive | distance faible pour les paires semblables, au-delà d'une marge pour les autres | rétropropagation |
| Perte triplet | ordre des distances ancre-positif et ancre-négatif, avec marge | rétropropagation, négatifs semi-difficiles |
| Factorisation matricielle | erreur quadratique sur les notes connues + régularisation L2 | descente de gradient stochastique ou moindres carrés alternés |

La perte qui caractérise ce chapitre est celle de l'échantillonnage négatif. Sa courbe montre les deux termes en fonction du score $s = \mathbf{v} \cdot \mathbf{u}$ : le vrai voisin est poussé vers les scores positifs, le mot tiré au hasard vers les scores négatifs. Fiche complète des pertes, avec l'entropie croisée binaire dont elle dérive, au chapitre 01. InfoNCE, qui généralise cette idée à une softmax sur un lot, appartient au chapitre 23.

## Quatre représentations sur le même corpus

L'exercice 15 compare les quatre représentations sur les mêmes mots de TinyShakespeare (194 017 mots, 3 251 dans le vocabulaire). Deux tests, en plus des voisins :

- test de groupes : 50 mots répartis en 5 groupes (famille, nombres, corps, titres, temps) ; part des mots dont le plus proche voisin, parmi ces 50, est du même groupe. Le hasard donne environ 20 % ;
- analogies : 16 analogies fixées à l'avance, résolues par 3CosAdd.

| Représentation | Dimension | Test de groupes | Analogies, top 1 (top 5) |
|---|---|---|---|
| one-hot | 3 251 | — (tous les cosinus sont nuls) | 0 |
| TF-IDF (lignes de la matrice) | 7 222 | 78 % | 6/16 (7) |
| LSA | 100 | 82 % | 0/16 (2) |
| word2vec skip-gram | 50 | 80 % | 8/16 (9) |

Ce qu'il faut y lire :

- Sur un petit corpus, compter fait presque aussi bien qu'apprendre. Au test de groupes, 78, 82 et 80 % : les trois sont au coude à coude, et un écart de 2 points sur 50 mots, c'est un mot. L'exercice le montre honnêtement : word2vec n'est pas magique, il exploite la même information que les comptages.
- word2vec gagne en compacité et aux analogies grammaticales : 50 nombres denses contre 7 222 presque tous nuls, 8 analogies contre 6. Son avantage historique est ailleurs : il passe à des milliards de mots là où une SVD de la matrice complète devient impraticable.
- La LSA réussit le test de groupes mais aucune analogie : ses dimensions captent les thèmes des répliques, pas les relations grammaticales, qui demandent des contextes très proches.
- TF-IDF réussit 6 analogies grâce aux mots grammaticaux (he/she, his/her) qui apparaissent dans des répliques différentes selon qu'on parle d'un homme ou d'une femme ; ses voisins restent pollués par les mots outils.

## Ce qui coinçait

- Un seul vecteur par mot. « avocat » (le fruit) et « avocat » (le métier) partagent le même vecteur, une moyenne de leurs contextes. En anglais shakespearien, « fair » (beau, juste, blond) aussi. Le sens d'un mot dépend de la phrase, et un plongement statique l'ignore. Les plongements contextuels (ELMo, au chapitre 16, puis BERT au chapitre 22) calculent un vecteur par occurrence.
- Pas d'ordre. Le sac de mots l'ignore complètement ; word2vec ne le voit qu'à l'intérieur d'une petite fenêtre, sans savoir qui est à gauche. Pour représenter une phrase, on fait souvent la moyenne des vecteurs de ses mots : « the dog bit the man » et « the man bit the dog » restent identiques.
- Les mots rares et inconnus. Un mot vu moins de 5 fois n'a pas de vecteur (7,7 % des occurrences de l'exercice). fastText atténue le problème ; la tokenisation en sous-mots le règle (chapitre 19).
- Il faut beaucoup de texte. Les analogies célèbres demandent des milliards de mots ; sur 200 000, man : woman :: king : ? donne *england*.
- Les vecteurs reproduisent les biais du corpus, stéréotypes compris (chapitre 40).
- Pour les pertes contrastives : choisir les négatifs. Un négatif facile n'apprend rien, un négatif trop difficile déstabilise, et sans négatif tout s'effondre. Ce problème occupera l'apprentissage auto-supervisé jusqu'aux années 2020 (chapitre 23).

- One-hot : Un mot = une case ; tous les mots sont à égale distance, rien n’est partagé.
- Hypothèse distributionnelle (Harris, 1954) : Des mots dans des contextes semblables ont des sens semblables.
- TF-IDF, modèle vectoriel (1972-1975) : Compter les mots, pondérer par la rareté, comparer au cosinus.
- LSA (1990) : Compresser la matrice mots × documents par SVD : les synonymes se rapprochent.
- Modèle de langage neuronal (2003) : Un vecteur appris par mot, sous-produit de la prédiction du mot suivant.
- word2vec (2013) : Un modèle presque linéaire, vrai voisin contre mots au hasard, sur des milliards de mots.
- GloVe, fastText (2014-2017) : Factoriser les cooccurrences ; composer un mot à partir de ses morceaux.
- Siamois (1993), pertes contrastive (2006) et triplet (2005, FaceNet 2015) : Apprendre un espace où la distance dit la ressemblance, en opposant positifs et négatifs.
- Factorisation matricielle (2009) : Un vecteur par utilisateur, un par film, la note est leur produit scalaire.
- RNN, LSTM, attention : Lire la phrase dans l’ordre et calculer un vecteur qui dépend du contexte.

Et ensuite ? Un vecteur par mot ne dit rien de l'ordre des mots ni du contexte : « bank » a le même vecteur au bord de la rivière et au guichet, et une phrase n'est qu'un sac. Pour traduire, transcrire la parole ou prédire la suite d'un texte, il faut un modèle qui lise la séquence et garde une mémoire de ce qu'il a lu. Les plongements de ce chapitre deviennent la première couche de ces modèles : c'est l'objet du chapitre 16 : Séquences avant les Transformers.

## Exercice

Ce qu'il faut observer :

- One-hot ne sait rien, compter sait déjà beaucoup : tous les cosinus one-hot valent 0 ; dès que les mots sont décrits par leurs répliques (TF-IDF), « three » ressemble à « four ». La matrice TF-IDF en numpy est identique à celle de scikit-learn (écart 0), la LSA aussi ($1{,}5 \times 10^{-7}$).
- Le skip-gram écrit à la main : gradient identique à l'autograd ($5{,}6 \times 10^{-17}$), perte qui part exactement de $6 \ln 2 = 4{,}159$ et finit à 2,54 ; voisins de « king » : *henry, edward, richard, lewis, warwick*.
- Le match des représentations : 78 / 82 / 80 % au test de groupes, 6 / 0 / 8 analogies ; man : woman :: king : ? rate.
- Les cartes 2D : 7 % des voisins conservés par l'ACP, 35 % par t-SNE.

Questions de réflexion : pourquoi l'IDF d'un mot présent dans toutes les répliques vaut-il 0, et est-ce souhaitable pour des vecteurs de mots ? word2vec donne deux vecteurs par mot ; que se passerait-il si l'on imposait $\mathbf{v} = \mathbf{u}$ (pensez au score d'un mot avec lui-même) ? Quel test départagerait TF-IDF, LSA et word2vec sur ce corpus, si l'on peut en trouver un ?

## Auto-évaluation

Question : Dans un corpus de 7 222 répliques, un mot apparaît dans 2 833 répliques, un autre dans 22. Quels sont leurs IDF (ln N/df) ? Que vaudrait l'IDF d'un mot présent dans toutes les répliques, et pourquoi scikit-learn ajoute-t-il 1 ?

Réponse : $\ln(7\,222 / 2\,833) = \ln 2{,}55 \approx 0{,}94$ (c'est « the » dans l'exercice) et $\ln(7\,222 / 22) = \ln 328 \approx 5{,}79$ (« poison »). Un mot présent partout a $\mathrm{df} = N$, donc un IDF de $\ln 1 = 0$ : il disparaît de tous les vecteurs. Pour un moteur de recherche, c'est souhaitable (il ne distingue aucun document) ; mais une requête faite uniquement de tels mots donnerait un vecteur nul, et l'on perd une information parfois utile. scikit-learn ajoute 1 à l'IDF pour qu'aucun mot ne soit totalement annulé.

Question : Au début de l'entraînement de l'exercice, tous les vecteurs de contexte u sont nuls et il y a 5 négatifs par paire. Pourquoi la perte vaut-elle exactement 6 ln 2 ≈ 4,16 ? Et pourquoi l'apprentissage peut-il quand même démarrer ?

Réponse : Tous les produits scalaires $\mathbf{v}_w \cdot \mathbf{u}$ valent 0, donc toutes les sigmoïdes valent $\sigma(0) = 1/2$. La perte d'une paire est $-\log \tfrac12$ pour la vraie paire plus $5 \times (-\log \tfrac12)$ pour les négatifs, soit $6 \ln 2 \approx 4{,}159$. Le gradient par rapport à $\mathbf{v}_w$ est nul (il ne contient que des $\mathbf{u}$), mais celui par rapport à $\mathbf{u}_c$ vaut $(s_+ - 1)\,\mathbf{v}_w = -\tfrac12 \mathbf{v}_w$, non nul puisque les $\mathbf{v}$ sont initialisés au hasard : les vecteurs de contexte bougent dès le premier pas, et les $\mathbf{v}$ suivent au pas suivant.

Question : Pourquoi l'échantillonnage négatif rend-il word2vec des centaines de fois plus rapide qu'une softmax complète ? Qu'est-ce qu'on y perd ?

Réponse : Avec la softmax, chaque paire (mot, voisin) demande un score pour tous les mots du vocabulaire (le dénominateur) : $V$ produits scalaires, un million dans l'article de 2013, 3 251 dans l'exercice. Avec l'échantillonnage négatif, on ne calcule que $K + 1$ produits scalaires (6 dans l'exercice) : une vraie paire et $K$ paires tirées au hasard, chacune jugée par une sigmoïde. On perd un vrai modèle de langage : les scores ne sont plus des probabilités normalisées du voisin, et l'on ne peut pas s'en servir pour générer du texte ou calculer une perplexité. Pour apprendre des vecteurs, cela ne gêne pas.

Question : Sur TinyShakespeare, word2vec réussit lord : lady :: king : queen mais pas man : woman :: king : queen (il répond england). Proposez une explication, puis dites pourquoi les analogies excluent a, b et c des réponses.

Réponse : L'analogie demande que le décalage $\mathbf{x}_{woman} - \mathbf{x}_{man}$ soit à peu près le même que $\mathbf{x}_{queen} - \mathbf{x}_{king}$. « lord » et « lady » s'emploient comme « king » et « queen » (titres, formules d'adresse), donc leurs décalages se ressemblent. « man » et « woman » apparaissent dans d'autres contextes (descriptions, proverbes) et sont moins fréquents : sur 200 000 mots, leur décalage est bruité, et le vecteur cherché tombe près des mots associés à « king » dans les pièces historiques, comme *england*. Il faut des corpus des milliers de fois plus grands pour que la relation masculin/féminin devienne régulière. Quant à l'exclusion : $\mathbf{x}_{b} - \mathbf{x}_{a}$ est un petit décalage, donc $\mathbf{x}_b - \mathbf{x}_a + \mathbf{x}_c$ reste très proche de $\mathbf{x}_c$ ; sans exclusion, la réponse serait presque toujours « king » lui-même.

Question : Un réseau siamois est entraîné uniquement sur des paires de visages de la même personne, avec la perte ½D². Que se passe-t-il ? Comment la perte contrastive et la perte triplet l'évitent-elles ?

Réponse : La perte est minimale (nulle) si le réseau envoie toutes les images sur le même vecteur : $D = 0$ pour toutes les paires. C'est l'effondrement : la perte est parfaite et les vecteurs ne servent à rien. La perte contrastive ajoute des paires dissemblables, pénalisées tant que leur distance est inférieure à la marge $m$ ($\tfrac12 \max(0, m - D)^2$) : envoyer tout au même point coûte alors $\tfrac12 m^2$ par paire dissemblable. La perte triplet exige que l'ancre soit plus proche du positif que du négatif d'au moins $\alpha$ : avec tout au même point, chaque triplet coûte $\alpha$. Dans les deux cas, ce sont les négatifs qui empêchent l'effondrement.

## Pour aller plus loin

- Articles originaux : Harris (1954), Spärck Jones (1972), Salton et al. (1975), Deerwester et al. (1990), Bengio et al. (2003), Mikolov et al. (2013, les deux articles, courts et lisibles), Levy et Goldberg (2014), Pennington et al. (2014), Bojanowski et al. (2017), Bromley et al. (1993), Hadsell et al. (2006), Schroff et al. (2015), Koren et al. (2009) : liens dans les fiches ci-dessus. Le code C d'origine de word2vec, dont l'exercice reprend les réglages : archive Google Code et copie sur GitHub. Vecteurs pré-entraînés : GloVe, fastText.
- Y. Goldberg et O. Levy, « word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method » (2014) : la dérivation de la perte, que l'article d'origine donne sans beaucoup d'explications.
- Stanford CS224n, *Natural Language Processing with Deep Learning* : le site du cours, dont les premières séances portent sur les vecteurs de mots, word2vec et GloVe.
- D. Jurafsky et J. H. Martin, *Speech and Language Processing*, 3<sup>e</sup> édition en ligne, chapitre « Embeddings » : TF-IDF, PMI, word2vec, biais, avec des exemples soignés.
- Dive into Deep Learning : « Word Embedding (word2vec) », « Word Embedding with Global Vectors (GloVe) » et « Matrix Factorization », avec du code.
- J. Alammar, « The Illustrated Word2vec » : skip-gram et échantillonnage négatif en dessins ; C. McCormick, « Word2Vec Tutorial - The Skip-Gram Model ».
- 3Blue1Brown, « Transformers, the tech behind LLMs » (chapitre 5 de la série sur l'apprentissage profond) : la partie sur les plongements montre les directions de sens (masculin/féminin) dans un espace de mots, avant l'attention.
- A. Karpathy, « Building makemore Part 2: MLP », de la série Neural Networks: Zero to Hero : le modèle de Bengio et al. (2003) reconstruit pas à pas, avec ses plongements.
- C. Olah, « Deep Learning, NLP, and Representations » (2014) : pourquoi des représentations apprises pour une tâche servent à d'autres.
- M. Nissim, R. van Noord et R. van der Goot, « Fair Is Better than Sensational: Man Is to Doctor as Woman Is to Doctor », *Computational Linguistics* 46(2), 2020 : ce que l'exclusion de a, b et c fait aux analogies, et pourquoi il faut lire avec prudence les analogies « biaisées » comme les analogies célèbres.
- K. Weinberger, J. Blitzer et L. Saul, « Distance Metric Learning for Large Margin Nearest Neighbor Classification » (NIPS 2005) : la perte à marge sur des triplets dont s'inspire FaceNet.
- L. Weng, « Contrastive Representation Learning » (2021) : de la perte contrastive et triplet jusqu'à InfoNCE, la suite de la section 8.
- Embedding Projector (TensorFlow) : explorer dans le navigateur des vecteurs word2vec appris sur un grand corpus, en ACP, t-SNE ou UMAP ; à comparer avec la démo de ce chapitre.
- M. Brunila et J. LaViolette, « What company do words keep? Revisiting the distributional semantics of J.R. Firth & Zellig Harris » (2022) : ce que Firth et Harris voulaient vraiment dire, au-delà de la citation.
- B. Sagot au Collège de France : chaire annuelle Informatique et sciences numériques (2023-2024), « Apprendre les langues aux machines », en français : des comptages aux modèles de langue.
