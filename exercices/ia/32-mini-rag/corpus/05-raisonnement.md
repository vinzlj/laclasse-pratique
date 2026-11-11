# 05. Raisonner : logique, CSP, planification, systèmes experts

Repère historique (1958-1987) : Encoder l’expertise humaine en règles et faits, et en déduire des conclusions.

En novembre 1958, à un symposium du National Physical Laboratory (Teddington), John McCarthy propose un programme qui reçoit ses connaissances sous forme d’énoncés logiques et en tire lui-même les conséquences : il dira plus tard que c’était sans doute le premier article d’IA logique. En 1965, Robinson réduit la démonstration automatique à une seule règle, la résolution ; en 1972, Prolog en fait un langage de programmation. À Stanford, l’équipe de DENDRAL (chimie, à partir du milieu des années 1960) puis de MYCIN (antibiotiques, thèse de Shortliffe en 1975) montre qu’un programme devient compétent quand on lui donne beaucoup de connaissances spécialisées, écrites avec des experts sous forme de règles. Dans les années 1980, ces « systèmes experts » deviennent une industrie. Elle s’effondre à la fin de la décennie : c’est le second hiver de l’IA.

## L’idée en une image

Au chapitre 04, le programme cherchait un chemin dans un espace d’états, et toute l’intelligence tenait dans l’heuristique. Ici, on change de question : que sait le programme, et que peut-il en déduire ? L’IA symbolique répond en séparant deux choses que le code habituel mélange :

- une base de connaissances, écrite par un humain : des faits (« le serveur renvoie une erreur 502 ») et des règles (« SI erreur 502 ET serveur web joignable ALORS l’application derrière le proxy est en panne ») ;
- un moteur d’inférence, écrit une fois pour toutes, qui applique les règles sans rien savoir du domaine.

Un développeur web connaît déjà ce découpage. Une feuille CSS ne dit pas comment calculer le style de chaque nœud du DOM : elle déclare des règles, et le moteur du navigateur les applique. Une requête SQL décrit le résultat voulu, et l’optimiseur choisit l’algorithme. L’IA symbolique pousse la même idée jusqu’au raisonnement : on déclare ce qui est vrai, un moteur générique déduit le reste.

Ce chapitre présente trois moteurs, qui correspondent à trois façons de raisonner :

| Moteur | On écrit | Le moteur calcule | Exemple de l’exercice |
|---|---|---|---|
| Règles (logique) | faits + règles SI… ALORS | les conclusions qui en découlent | diagnostic de panne d’un site web |
| Contraintes (CSP) | variables, valeurs possibles, interdits | une affectation qui respecte tout | sudoku |
| Planification (STRIPS) | état de départ, but, actions possibles | une suite d’actions qui atteint le but | empiler des blocs avec un bras robot |

La flèche « pourquoi ? » compte : un système à règles peut expliquer sa conclusion en citant les règles qui y ont mené. C’était l’un de ses arguments majeurs, et c’est ce qui manquera aux réseaux de neurones des chapitres suivants.

| | Avant : programme écrit pour un cas précis | Après : base de connaissances + moteur générique |
|---|---|---|
| Où vit l’expertise | dans le code, mêlée au contrôle (`if` imbriqués) | dans des règles lisibles, séparées du moteur |
| Ajouter un savoir | modifier et retester le programme | ajouter une règle (MYCIN en a compté environ 500) |
| Expliquer une réponse | relire le code | afficher la chaîne de règles déclenchées |
| Poser des questions | toutes d’avance | seulement celles qui servent : 2 questions sur 12 pour tester « certificat expiré » dans l’exercice |
| Ce qui coince encore | — | rien n’est appris : chaque cas nouveau exige une règle écrite par un humain ; hors de ses règles, le système se tait |

## Le mécanisme

### Ce que la méthode optimise

Aucune fonction de perte. Rien n’est appris à partir de données, donc rien n’est minimisé : on cherche une preuve (une conclusion qui découle des règles), une affectation qui respecte toutes les contraintes, ou un plan (une suite d’actions). Le seul « coût » de ce chapitre est la longueur du plan en planification, comme le coût de chemin du chapitre 04. Les fonctions de perte, et ce qu’elles changent quand on apprend, sont au chapitre 01.

### 1. La logique comme langage de connaissances

Logique propositionnelle. La logique propositionnelle manipule des énoncés vrais ou faux, notés par des lettres : $p$ = « le site renvoie un 502 », $q$ = « le serveur web répond ». On les combine avec $\wedge$ (ET), $\vee$ (OU), $\neg$ (NON) et $\Rightarrow$ (IMPLIQUE). Une affectation vrai/faux de toutes les lettres s’appelle un modèle ; une base de connaissances $B$ (l’ensemble des règles et des faits) restreint les modèles possibles. Ce que la base « sait », c’est ce qui est vrai dans tous ses modèles :

C’est la démonstration par l’absurde, mécanisée : si supposer $q$ faux contredit la base, alors $q$ est une conséquence de la base. Décider si une formule a un modèle s’appelle le problème SAT. Cook a montré en 1971 que c’est le premier problème NP-complet (Leonid Levin a obtenu le même résultat de son côté en URSS, publié en 1973 : c’est le théorème de Cook-Levin) : aucun algorithme connu ne le résout en temps polynomial dans le pire cas. L’exercice s’en sert quand même comme juge de paix : le solveur SAT de `sympy` retrouve exactement les conclusions du moteur de règles sur les 5 cas de test.

Clauses de Horn. Heureusement, les règles d’un système expert ont une forme simple, « SI $a_1$ ET … ET $a_k$ ALORS $c$ », c’est-à-dire $a_1 \wedge \dots \wedge a_k \Rightarrow c$, avec une seule conclusion positive. C’est une clause de Horn (du nom du logicien Alfred Horn, 1951). Plus exactement, une clause de Horn contient au plus une conclusion positive ; les règles SI… ALORS en ont exactement une, ce qu’on appelle des clauses définies. Sur ces règles-là, pas besoin d’un solveur SAT : appliquer les règles dans le sens de la flèche, jusqu’à ce que plus rien ne change, déduit exactement les faits que la base implique. Cette garantie tombe dès qu’une règle conclut « $a$ OU $b$ » (question de réflexion 2 de l’exercice).

Logique du premier ordre. La logique propositionnelle oblige à écrire une lettre par fait : « le serveur 1 répond », « le serveur 2 répond »… La logique du premier ordre ajoute des objets, des prédicats sur ces objets et des variables quantifiées :

Une règle couvre alors tous les serveurs. Pour l’appliquer à un fait précis, il faut trouver la valeur de $x$ qui rend la règle et le fait identiques : c’est l’unification. En 1965, Robinson montre qu’une seule règle d’inférence, la résolution, combinée à l’unification, suffit à démontrer par l’absurde tout ce qui découle d’une base en logique du premier ordre. C’est le moteur de la démonstration automatique.

Prolog : la logique comme langage de programmation. À Marseille, Alain Colmerauer et Philippe Roussel cherchent à faire dialoguer un ordinateur en français ; le projet donne une version préliminaire de Prolog fin 1971 et une version définitive fin 1972. Un programme Prolog est une base de clauses de Horn ; l’exécuter, c’est poser une question et laisser le moteur chercher une preuve :

```prolog
% Faits
parent(alice, bob).
parent(bob, carole).
% Règle : X est grand-parent de Z s'il existe Y tel que X parent de Y et Y parent de Z
grand_parent(X, Z) :- parent(X, Y), parent(Y, Z).

% Question
?- grand_parent(alice, Qui).
Qui = carole.
```

Pas à pas : le moteur cherche une règle dont la tête s’unifie avec `grand_parent(alice, Qui)` ; il trouve la règle avec `X = alice`, `Z = Qui`. Il doit alors prouver `parent(alice, Y)` : le premier fait s’unifie avec `Y = bob`. Reste `parent(bob, Qui)` : le second fait donne `Qui = carole`. Si une étape échouait, Prolog reviendrait au dernier choix pour essayer le fait suivant (c’est le retour arrière, revu en section 3). Prolog raisonne à rebours, du but vers les faits : c’est du chaînage arrière, comme MYCIN (section 2).

### 2. Les systèmes experts : chaînage avant, chaînage arrière

Un système expert est une base de règles écrites avec un spécialiste, plus un moteur. L’exercice en construit un en miniature : 15 règles pour diagnostiquer la panne d’un site web, 12 faits observables, 6 diagnostics, 5 cas de test. Extrait :

```text
R1  SI statut_502 ET serveur_web_repond          ALORS proxy_ok_appli_ko
R2  SI proxy_ok_appli_ko ET deploiement_recent   ALORS diagnostic:appli_plante_au_demarrage
R3  SI proxy_ok_appli_ko ET memoire_saturee      ALORS diagnostic:appli_tuee_par_manque_de_memoire
R8  SI alerte_securite_navigateur ET date_certificat_depassee ALORS diagnostic:certificat_expire
R10 SI diagnostic:appli_plante_au_demarrage      ALORS action:revenir_a_la_version_precedente
```

Les faits qu’aucune règle ne conclut (`statut_502`, `deploiement_recent`…) sont les observables : on les demande à l’utilisateur. Les autres sont déduits. Deux façons de faire tourner la même base. Le chaînage avant, sur le cas « déploiement raté » (faits observés en haut, règles dans les hexagones) :

Le chaînage avant part des faits observés et applique toute règle dont les prémisses sont vraies, en boucle, jusqu’au point fixe (plus aucun fait nouveau). Le cœur tient en quelques lignes (version simplifiée ; celle de l’exercice saute en plus les règles déjà déclenchées et compte les tests) :

```python
def chainage_avant(faits, regles):
    faits = set(faits)
    nouveau = True
    while nouveau:                      # un passage = un « cycle »
        nouveau = False
        for nom, premisses, conclusion in regles:
            if conclusion not in faits and all(p in faits for p in premisses):
                faits.add(conclusion)   # la règle se déclenche
                nouveau = True
    return faits
```

Sur le cas « déploiement raté » (502, serveur web qui répond, déploiement récent), la trace de l’exercice est : R1 donne `proxy_ok_appli_ko`, R2 donne le diagnostic, R10 l’action, puis un dernier passage ne trouve rien de nouveau. Total, en sautant les règles déjà déclenchées : 27 tests de règles. Le chaînage avant est simple, mais il veut toutes les observations d’avance, ici les 12.

Le chaînage arrière part dans l’autre sens. Pour établir une hypothèse, il cherche les règles qui la concluent, puis tente d’établir leurs prémisses, récursivement ; il ne pose une question que lorsqu’il arrive à un fait observable :

```python
def etablir(but, regles, demander, memo):
    if but in memo:                       # déjà établi ou réfuté
        return memo[but]
    utiles = [premisses for nom, premisses, conclusion in regles
              if conclusion == but]
    if not utiles:                        # fait observable : on pose la question
        memo[but] = demander(but)
        return memo[but]
    memo[but] = False                     # protège contre les boucles
    for premisses in utiles:
        if all(etablir(p, regles, demander, memo) for p in premisses):
            memo[but] = True
            break
    return memo[but]
```

Mesures de l’exercice : pour tester tous les diagnostics, le chaînage arrière pose 5 à 8 questions sur 12 selon le cas ; pour vérifier une seule hypothèse, « le certificat a-t-il expiré ? », il en pose 2. Les deux moteurs donnent les mêmes diagnostics sur les 5 cas, et le solveur SAT les confirme.

MYCIN. Ce second sens est celui de MYCIN, le système de Stanford qui conseillait l’antibiotique à prescrire en cas d’infection grave (bactériémie, puis méningite). Le bilan publié en 1984 par Buchanan et Shortliffe en décrit le fonctionnement : « la structure de contrôle est un chaînage arrière des règles, dirigé par le but » (ch. 4). Le chaînage arrière se justifiait par le coût des questions : chacune prend du temps au médecin, et certaines réponses exigent des analyses. Une règle typique, traduite en anglais par le programme lui-même (sa forme interne était du code LISP) :

```text
RULE036
IF:   1) The gram stain of the organism is gramneg, and
      2) The morphology of the organism is rod, and
      3) The aerobicity of the organism is anaerobic
THEN: There is suggestive evidence (.6) that the identity
      of the organism is bacteroides
```

Le « .6 » est un facteur de certitude, un nombre entre −1 (sûrement faux) et 1 (sûrement vrai) : la médecine raisonne rarement en certitudes. Les auteurs précisent que ce ne sont pas des probabilités, seulement des degrés de croyance « informellement fondés » sur elles. Leur règle de combinaison, pour deux indices qui soutiennent la même hypothèse :

Exemple à la main : deux règles concluent chacune « bacteroides » à 0,6. Après la première, la croyance vaut 0,6 et il reste 0,4 de doute ; la seconde en comble 60 %, soit 0,24. Total : $0{,}6 + 0{,}6 \times 0{,}4 = 0{,}84$. La croyance monte sans jamais dépasser 1. Le facteur de certitude final est la croyance moins le doute, $CF = MB - MD$. Une prémisse composée (« A ET B ») prend le minimum des facteurs de ses parties.

En simplifiant : ces combinaisons sont des conventions commodes, pas des calculs de probabilités. Elles supposent les indices indépendants, ce qui est rarement vrai ; les réseaux bayésiens (années 1980) traiteront l’incertitude de façon cohérente, au chapitre 09.

Les chiffres de MYCIN, tirés du même livre :
- la base a fini par compter environ 500 règles ;
- lors d’une évaluation en aveugle sur 10 cas de méningite, 8 spécialistes extérieurs à Stanford ont noté, sans savoir qui était qui, dix prescriptions par cas : celle de MYCIN, celle réellement donnée au patient, et celles d’un étudiant en médecine et de sept médecins de Stanford (dont cinq enseignants en infectiologie). 65 % des notes données à MYCIN étaient « acceptable », contre 42,5 % à 62,5 % pour les cinq enseignants (ch. 31) ;
- et pourtant, « MYCIN n’a jamais été utilisé en routine auprès des patients » (ch. 32). Les auteurs l’expliquent par des raisons pratiques : les médecins ne ressentaient pas le besoin d’un conseil, et il aurait fallu trouver un terminal, se connecter, retaper des résultats d’analyses déjà stockés sur d’autres ordinateurs, et une consultation complète pouvait durer jusqu’à 30 minutes ou une heure quand la machine de recherche partagée était chargée. Fin 1978, MYCIN est figé et ne sert plus qu’aux démonstrations.

MYCIN descendait de DENDRAL, lancé au milieu des années 1960 par Joshua Lederberg et Edward Feigenbaum pour proposer la structure d’une molécule à partir de données de spectrométrie de masse. Leçon commune des deux projets : la puissance vient des connaissances spécialisées plus que de méthodes de résolution générales. Feigenbaum en tire l’« ingénierie de la connaissance » (*knowledge engineering*), métier qui consiste à faire parler l’expert et à transcrire son savoir en règles. Les mêmes auteurs nomment aussi le problème : dès DENDRAL, « il était évident que faire entrer les connaissances du domaine dans un programme était un goulot d’étranglement » (Buchanan et Shortliffe, 1984, ch. 7). C’est le goulot d’acquisition des connaissances.

Le succès industriel. R1 (appelé XCON chez Digital Equipment Corporation), écrit à Carnegie-Mellon par John McDermott et décrit en 1982, configure les ordinateurs VAX-11/780 : à partir de la commande d’un client, il détermine les modifications nécessaires pour que le système fonctionne et la disposition des composants. Selon l’article, l’usine de DEC s’en sert alors régulièrement, et le programme cherche peu : à chaque étape, il « reconnaît simplement quoi faire ». Dans les années 1980, d’autres systèmes suivent. On vend des « coquilles » (*expert system shells*) : un moteur et une interface livrés vides, à remplir de règles. La première, EMYCIN, a été tirée de MYCIN dès la fin des années 1970 (van Melle, IJCAI 1979). On vend aussi des machines spécialisées pour exécuter LISP.

ELIZA : des règles sans connaissances :
Bien avant, dès 1966, Joseph Weizenbaum (MIT) avait montré l’autre face des règles. ELIZA ne contient aucune connaissance du monde : des mots-clés déclenchent des règles de décomposition de la phrase, puis des règles de recomposition fabriquent la réponse (à « He says I’m depressed much of the time », elle répond « I AM SORRY TO HEAR YOU ARE DEPRESSED »). Avec un script qui imite un psychothérapeute, les utilisateurs lui prêtaient pourtant de la compréhension. Weizenbaum écrit que l’élégance d’ELIZA tient à ce qu’elle « entretient l’illusion de la compréhension avec si peu de machinerie », et rappelle que le vrai test de la compréhension n’est pas de continuer une conversation mais d’en tirer des conclusions valides. La question reviendra avec les modèles de langage, aux chapitres 31 et 33.

### 3. Les contraintes : éliminer plutôt qu’essayer

Beaucoup de problèmes ne se décrivent pas par des règles SI… ALORS mais par des interdits : un emploi du temps (deux cours dans la même salle, non), une carte à colorier (deux pays voisins, pas la même couleur), un sudoku. C’est un problème de satisfaction de contraintes (CSP) :

Le sudoku a 81 variables. Chaque case a 20 « pairs » (8 sur sa ligne, 8 sur sa colonne, 4 de plus dans son bloc), et la seule contrainte est « valeur différente de celle de chaque pair ».

Essayer : le retour arrière. La méthode naïve est le retour arrière : remplir les cases vides dans l’ordre, essayer 1, puis 2…, et revenir sur le dernier choix dès qu’un pair a déjà la valeur. C’est une recherche en profondeur, comme au chapitre 04. Elle finit toujours, mais sur la grille difficile de l’exercice elle essaie 445 778 affectations.

Éliminer : l’arc-consistance. Mackworth (1977) formalise une idée plus économe : avant de choisir, retirer les valeurs qui ne peuvent mener nulle part. Un « arc » est une contrainte vue depuis l’une de ses variables. Il est consistant quand chaque valeur restante a un partenaire compatible de l’autre côté :

L’arc-consistance s’obtient avec l’algorithme AC-3 : une file d’arcs à « réviser » ; réviser $(X_i, X_j)$, c’est retirer de $D_i$ les valeurs sans support. Si $D_i$ a rétréci, les arcs qui pointent vers $X_i$ doivent être revus, car leurs supports ont pu disparaître. C’est la propagation de contraintes : un retrait en entraîne d’autres, de proche en proche. Le code de l’exercice, générique (il ne sait rien du sudoku) :

```python
from collections import deque

def ac3(domaines, voisins, compatible):
    file = deque((xi, xj) for xi in domaines for xj in voisins[xi])
    while file:
        xi, xj = file.popleft()
        # réviser l'arc : retirer de D_i les valeurs sans support dans D_j
        retirees = {v for v in domaines[xi]
                    if not any(compatible(xi, v, xj, w) for w in domaines[xj])}
        if retirees:
            domaines[xi] -= retirees
            if not domaines[xi]:
                return False             # domaine vide : aucune solution
            file.extend((xk, xi) for xk in voisins[xi] if xk != xj)
    return True
```

Exemple à la main, celui de l’exercice : $X \lt Y$, $Y \lt Z$, chaque domaine vaut $\{1, 2, 3\}$.

| Révision | Arc | Valeurs retirées | Pourquoi | Domaines après |
|---|---|---|---|---|
| 1 | (X, Y) | 3 de X | aucun Y supérieur à 3 | X = {1, 2} |
| 2 | (Y, X) | 1 de Y | aucun X inférieur à 1 | Y = {2, 3} |
| 3 | (Y, Z) | 3 de Y | aucun Z supérieur à 3 | Y = {2} |
| 4 | (Z, Y) | 1 et 2 de Z | Z doit dépasser Y = 2 | Z = {3} |
| 5 | (Z, Y) | rien | arc déjà consistant (ajouté deux fois à la file) | |
| 6 | (X, Y) | 2 de X | X doit être inférieur à Y = 2 | X = {1} |

Six révisions, aucun essai : la solution X = 1, Y = 2, Z = 3 tombe par élimination.

Sur le sudoku, AC-3 seul fait la même chose qu’un joueur qui raye les candidats impossibles. Sur la grille facile, il trouve la solution sans aucun choix (51 cases sur 51, les candidats passent de 459 à 51). Sur la grille difficile, il ne fixe aucune case, mais divise les candidats par deux (540 → 254).

Combiner les deux. La bonne méthode alterne éliminer et essayer. Deux améliorations du retour arrière, mesurées dans l’exercice :
- MRV (*minimum remaining values*) : choisir d’abord la case qui a le moins de candidats (un coup forcé s’il n’en reste qu’un), et après chaque choix retirer la valeur des domaines des pairs (*vérification en avant*, en anglais *forward checking*) ;
- MAC (*maintaining arc consistency*) : après chaque choix, relancer AC-3 sur les arcs touchés.

| Grille | Retour arrière naïf | MRV + vérification en avant | MAC |
|---|---|---|---|
| Facile (51 cases vides) | 37 652 affectations, ≈ 24 ms | 51 affectations, ≈ 1 ms | 0 affectation, ≈ 8 ms |
| Difficile (60 cases vides) | 445 778 affectations, ≈ 285 ms | 10 101 affectations, ≈ 170 ms | 1 850 affectations, ≈ 440 ms (516 257 révisions d’arcs) |

Les durées sont indicatives (M1 Pro, CPU) et varient de quelques pour cent d’un run à l’autre. Sur la grille difficile, MAC essaie 241 fois moins d’affectations que le naïf. Mais il est le plus lent des trois en temps : chaque affectation déclenche des centaines de révisions d’arcs. Moins de nœuds n’est pas toujours moins de secondes ; c’est un arbitrage qu’on retrouve partout en recherche (question de réflexion 3).

### 4. La planification : STRIPS, puis PDDL

Un robot doit agir : prendre, poser, empiler. Il faut trouver une suite d’actions qui mène de la situation présente à un but. C’est la planification automatique. En logique pure, la difficulté est de dire tout ce qu’une action ne change pas (empiler un bloc ne repeint pas les murs, ne déplace pas les autres blocs…) : c’est le problème du cadre. En 1971, Fikes et Nilsson, au Stanford Research Institute, écrivent STRIPS pour le robot Shakey et contournent le problème par une convention : un état est un ensemble de faits, et une action ne touche que les faits qu’elle cite.

Exemple, l’action `empiler(A, B)` du monde des blocs (*blocks world*) :

| Préconditions | Ajouts | Retraits |
|---|---|---|
| tient(A), libre(B) | sur(A, B), libre(A), main_vide | tient(A), libre(B) |

Partant d’un état où le bras tient A et où B est libre, l’action retire ces deux faits, ajoute les trois autres, et ne touche à rien d’autre. Avec cette représentation, planifier peut se ramener à une recherche dans l’espace des états, celle du chapitre 04 : chaque état est un nœud, chaque action applicable une arête.

L’anomalie de Sussman. Trois blocs : C est posé sur A, B est seul sur la table ; but, « A sur B et B sur C ». Les planificateurs du début des années 1970, dont STRIPS lui-même avec sa pile de buts, traitaient les sous-buts l’un après l’autre. Or ici, les deux ordres échouent : viser d’abord « A sur B » oblige plus tard à démonter A pour poser B sur C ; viser d’abord « B sur C » enterre A sous la pile. L’exercice le mesure : chaque ordre produit 8 actions et défait le premier sous-but. Gerald Sussman a décrit le cas dans sa thèse au MIT (1973). Une recherche dans l’espace des états complets n’a pas ce problème : la recherche en largeur trouve le plan optimal de 6 actions après avoir développé 18 états.

L’explosion combinatoire. Le nombre d’états atteignables grandit plus vite qu’exponentiellement avec le nombre de blocs (mesures de l’exercice, 5 instances tirées au hasard par taille) :

| Blocs | Actions | États | Dév. (larg. / glout.) | Plan (larg. / glout.) |
| --- | --- | --- | --- | --- |
| 3 | 18 | 22 | 5 / 5 | 3,6 / 3,6 |
| 5 | 50 | 866 | 285 / 23 | 8,0 / 9,2 |
| 7 | 98 | 65 990 | 9 270 / 72 | 10,8 / 13,6 |
| 8 | 128 | 695 417 | 66 977 / 76 | 13,2 / 15,2 |

Le tableau compte les actions instanciées, les états atteignables, les états développés par chaque recherche (moyenne sur 5 instances) et la longueur moyenne des plans. La recherche gloutonne (*greedy best-first search*), guidée par l’heuristique « nombre de faits du but encore faux », développe 76 états au lieu de 66 977 à 8 blocs, mais rend des plans plus longs (15,2 actions contre 13,2). Cette heuristique est pourtant admissible : chaque action ajoute au plus un fait du but, donc elle ne surestime jamais le nombre d’actions restantes. Le défaut vient de la recherche gloutonne elle-même : elle ne regarde que $h$ et ignore le coût déjà payé $g$, si bien qu’elle ne garantit jamais le plan le plus court. A\* ($f = g + h$, chapitre 04) le garantirait avec cette même heuristique.

PDDL. En 1998, pour la première compétition internationale de planification, un comité présidé par Drew McDermott (Yale) fixe un langage commun, PDDL, qui reprend la notation de STRIPS. Le même `empiler` s’écrit :

```lisp
(:action empiler
  :parameters (?x ?y)
  :precondition (and (tient ?x) (libre ?y))
  :effect (and (sur ?x ?y) (libre ?x) (main-vide)
               (not (tient ?x)) (not (libre ?y))))
```

Les `(not …)` sont les retraits. On décrit d’un côté le domaine (les actions), de l’autre le problème (objets, état initial, but), puis n’importe quel planificateur conforme cherche le plan. Ce format reste le langage d’échange des planificateurs ; le chapitre 33 y revient à propos des agents à base de LLM, qui doivent eux aussi enchaîner des actions vers un but.

### 5. Représenter les connaissances : ontologies et graphes

Règles, contraintes et actions supposent un vocabulaire : quels objets existent, quelles relations les lient. Une ontologie fixe ce vocabulaire (« un serveur web est un logiciel », « un logiciel tourne sur une machine ») ; les faits s’écrivent alors en triplets sujet, relation, objet :

```text
(nginx,        est_un,       serveur_web)
(serveur_web,  sorte_de,     logiciel)
(nginx,        tourne_sur,   vm-12)
(vm-12,        situee_dans,  eu-west-1)
```

Lus comme un graphe (les sujets et objets sont des nœuds, les relations des arêtes), ces triplets forment un graphe de connaissances. Une règle simple, « si $x$ tourne sur $y$ et $y$ est située dans $z$, alors $x$ est hébergé dans $z$ », déduit un fait que personne n’a écrit : nginx est hébergé dans eu-west-1. Le chaînage avant de la section 2 s’applique tel quel.

Deux ambitions opposées ont marqué cette lignée :
- tout écrire à la main. Le projet Cyc, lancé en 1984 par Douglas Lenat, veut donner aux programmes le sens commun qui leur manque. Dix ans plus tard, d’après Lenat, « un siècle-personne » de travail avait produit environ $10^5$ concepts et $10^6$ axiomes saisis à la main. Le sens commun s’est révélé sans fond ;
- assembler à grande échelle. En 2012, Google annonce son Knowledge Graph (« things, not strings ») : plus de 500 millions d’objets et 3,5 milliards de faits au lancement, tirés de sources publiques comme Freebase, Wikipédia et le CIA World Factbook. Wikidata (Vrandečić et Krötzsch, 2014) en est l’équivalent ouvert et collaboratif.

Les graphes de connaissances n’ont jamais disparu : on les retrouve aujourd’hui à côté des modèles de langage, par exemple dans GraphRAG, au chapitre 32.

### 6. Encadré : la logique floue

Vrai à 0,8 :
En 1965, Lotfi Zadeh (Berkeley) propose les ensembles flous : l’appartenance à un ensemble n’est plus oui ou non, mais un degré $\mu(x)$ entre 0 et 1. « 28 °C » peut appartenir à l’ensemble « chaud » au degré 0,8 et à « tiède » au degré 0,3. Les opérations classiques se généralisent : le complément vaut $1 - \mu$, l’union prend le maximum des degrés, l’intersection le minimum. Avec $\mu_{\text{chaud}}(28) = 0{,}8$ et $\mu_{\text{humide}}(70\,\%) = 0{,}5$ (pour 70 % d’humidité), la prémisse « chaud ET humide » est vraie à $\min(0{,}8 ; 0{,}5) = 0{,}5$.

C’est de là qu’est née la logique floue, qui a trouvé sa place en commande automatique parce qu’elle traduit directement des règles d’opérateur du type « si la température est un peu haute, baisser un peu le chauffage » ; Mamdani et Assilian décrivent dès 1975 un régulateur à logique floue. Elle reste une logique de règles écrites à la main : elle traite le vague des mots, pas l’absence de connaissances. Le plan du cours ne lui consacre donc pas de chapitre.

## Ce qui coinçait : le second hiver

Les systèmes experts ont tenu leurs promesses dans des domaines étroits et stables. Ailleurs, cinq limites se sont accumulées :

- La fragilité. Hors de ses règles, un système expert ne dégrade pas sa réponse : il se tait, ou se trompe avec assurance. Dans l’exercice, le cas « hors domaine » (images absentes, lenteur sur mobile) ne produit aucun diagnostic et aucun message : le système ne sait pas qu’il ne sait pas.
- Le goulot d’acquisition. Chaque règle s’obtient en interrogeant un expert, souvent incapable d’expliciter ce qu’il fait d’instinct. Et une base de centaines de règles qui interagissent devient difficile à maintenir : ajouter une règle peut en contredire dix (question de réflexion 1 de l’exercice).
- L’explosion combinatoire. SAT est NP-complet, l’espace d’états du monde des blocs passe de 22 états à 3 blocs à 695 417 à 8. Les heuristiques repoussent le mur sans le supprimer.
- L’incertitude. Les facteurs de certitude de MYCIN sont des conventions ; elles ne disent pas comment réviser une croyance quand les indices se contredisent ou dépendent les uns des autres. Les modèles probabilistes (chapitre 09) prendront le relais.
- Le sens commun. Tout ce qu’un enfant sait sans qu’on le lui dise doit être écrit. Cyc a montré l’ampleur de la tâche.

L’effondrement. D’après les histoires de l’IA de Daniel Crevier (1993), Pamela McCorduck (2004) et Harvey Newquist (1994), citées par la page « AI winter » de Wikipédia, plusieurs chocs se suivent à la fin des années 1980. En 1987, le marché des machines LISP s’effondre : les stations de travail généralistes (Sun) et même les ordinateurs de bureau font aussi bien pour beaucoup moins cher. Les systèmes experts déployés, XCON compris, se révèlent coûteux à maintenir. En 1988, la DARPA coupe les crédits de l’IA dans son programme Strategic Computing, et le projet japonais de « cinquième génération », lancé au début des années 1980, s’achève en 1992 sans avoir atteint ses objectifs. Wikipédia date ce second hiver de 1987 à 2000 environ ; les sources ne s’accordent pas sur les bornes (le premier hiver, celui des années 1970, est raconté au chapitre 06).

Ce qui a survécu n’est pas mince : les solveurs SAT et de contraintes, la planification en PDDL, Prolog, les ontologies et les graphes de connaissances restent enseignés (les chapitres 6 à 11 du manuel de Russell et Norvig leur sont consacrés) et utilisés. Ce qui a disparu, c’est l’idée qu’on atteindrait une intelligence générale en écrivant assez de règles.

- Recherche dans un espace d’états (1956-1968) : Explorer les possibilités ; l’heuristique dit où regarder d’abord.
- Logique et résolution (1958-1965) : Écrire ce qui est vrai ; un moteur générique en déduit le reste.
- Prolog (1972) : Un programme est une base de règles ; l’exécuter, c’est chercher une preuve.
- STRIPS (1971), PDDL (1998) : Une action ne change que les faits qu’elle cite ; planifier, c’est chercher.
- Arc-consistance, AC-3 (1977) : Éliminer les valeurs impossibles avant d’essayer.
- Systèmes experts, MYCIN (1976) : Beaucoup de règles d’un domaine étroit battent une méthode générale ; mais il faut les écrire.
- Perceptron (1958) : Ne plus écrire les règles : régler des poids à partir d’exemples.

Et ensuite ? Tout ce chapitre bute sur la même pierre : les règles sont écrites à la main, et elles ne passent pas à l’échelle du monde réel. Une autre voie existait depuis 1943, en parallèle : au lieu de programmer la connaissance, construire des neurones artificiels qui apprennent leurs réglages à partir d’exemples. Elle commence au chapitre 06, avec le neurone formel et le perceptron, et avec son propre hiver.

## Exercice

En Python pur, sur CPU (20 s pour le run complet), vous programmez les trois moteurs du chapitre : un système expert de 15 règles interrogé en chaînage avant puis arrière et vérifié par un solveur SAT, AC-3 et trois solveurs de sudoku, et un planificateur STRIPS du monde des blocs.

Ce qu’il faut observer :

- le chaînage arrière pose 5 à 8 questions sur 12 pour tous les diagnostics, et 2 pour une hypothèse unique ; le cas hors domaine ne produit rien, sans erreur ;
- AC-3 seul résout la grille facile sans aucun choix et aucune case de la grille difficile ; MAC essaie 241 fois moins d’affectations que le naïf, mais prend plus de temps que MRV ;
- sur l’anomalie de Sussman, viser un sous-but après l’autre échoue dans les deux ordres ; à 8 blocs, l’espace compte 695 417 états.

Questions de réflexion : que devient la base si l’on ajoute « SI statut_502 ALORS l’application plante au démarrage » ? Pourquoi l’accord entre le chaînage avant et le solveur SAT disparaît-il si une règle conclut « A OU B » ?

## Auto-évaluation

Question : Une base contient les règles R1 : « 502 ET serveur_web_repond ⇒ proxy_ok_appli_ko » et R2 : « proxy_ok_appli_ko ET deploiement_recent ⇒ appli_plante ». Faits : 502, serveur_web_repond, deploiement_recent. Que déduit le chaînage avant, et dans quel ordre ?

Réponse : Premier passage : R1 a ses deux prémisses vraies, elle ajoute `proxy_ok_appli_ko` ; R2 a maintenant ses deux prémisses vraies (selon l’ordre de parcours, dans le même passage ou au suivant), elle ajoute `appli_plante`. Un passage de plus ne trouve aucun fait nouveau : c’est le point fixe. Comme les règles sont des clauses de Horn, ces deux faits sont exactement ce que la base implique logiquement.

Question : Pourquoi MYCIN raisonnait-il en chaînage arrière plutôt qu’en chaînage avant ?

Réponse : Le chaînage avant exige toutes les observations d’avance. Le chaînage arrière part d’une hypothèse (« l’organisme est-il un bacteroides ? »), remonte aux règles qui la concluent et ne pose une question que lorsqu’un fait manquant est nécessaire. Chaque question coûte du temps au médecin, parfois une analyse : MYCIN n’en posait donc que les utiles. Dans l’exercice, tester l’hypothèse « certificat expiré » demande 2 questions sur 12.

Question : Deux règles de MYCIN soutiennent la même hypothèse avec une croyance de 0,6 chacune. Quelle croyance combinée obtient-on ? Est-ce une probabilité ?

Réponse : $MB = 0{,}6 + 0{,}6 \times (1 - 0{,}6) = 0{,}84$ : le second indice comble 60 % du doute restant. Ce n’est pas une probabilité : c’est une convention qui suppose les indices indépendants et ne dit pas comment traiter des indices liés ; les auteurs eux-mêmes la présentent comme « informellement fondée » sur les probabilités.

Question : Domaines X, Y, Z = {1, 2, 3}, contraintes X < Y et Y < Z. Que donne AC-3, et combien d’affectations faut-il essayer ensuite ?

Réponse : AC-3 retire 3 de X (pas de Y plus grand), 1 de Y (pas de X plus petit), 3 de Y (pas de Z plus grand) ; Y = {2} entraîne Z = {3} puis X = {1}. Il ne reste qu’une valeur par variable : zéro essai. Sur un sudoku facile c’est pareil (les 51 cases sont déduites) ; sur une grille difficile, AC-3 ne fixe aucune case et il faut combiner propagation et recherche (MAC).

Question : Dans l’anomalie de Sussman (C sur A, B sur la table ; but : A sur B et B sur C), pourquoi traiter les sous-buts l’un après l’autre échoue-t-il, et qu’est-ce qui le résout ?

Réponse : Viser « A sur B » d’abord pose A sur B ; pour poser ensuite B sur C, il faut démonter A. Viser « B sur C » d’abord met B sur C alors que C couvre encore A, qui se retrouve sous la pile. Les sous-buts interagissent. Une recherche en largeur dans l’espace des états complets, avec les actions décrites à la manière de STRIPS, considère les deux faits ensemble et trouve le plan de 6 actions : dépiler C de A, poser C, prendre B, empiler B sur C, prendre A, empiler A sur B. L’algorithme d’origine de STRIPS, qui empilait les sous-buts, n’y parvenait pas.

## Pour aller plus loin

- Articles originaux (fiches ci-dessus) : McCarthy, « Programs with Common Sense » (1958) ; Robinson, « A Machine-Oriented Logic Based on the Resolution Principle » (1965) ; Weizenbaum, « ELIZA » (1966, PDF hébergé par le cours CS124 de Stanford) ; Fikes et Nilsson, « STRIPS » (1971) ; Mackworth, « Consistency in networks of relations » (1977) ; Colmerauer et Roussel, « The birth of Prolog » (1993).
- MYCIN en entier : B. G. Buchanan et E. H. Shortliffe (dir.), *Rule-Based Expert Systems: The MYCIN Experiments of the Stanford Heuristic Programming Project* (1984), tous les chapitres en PDF ; commencer par le ch. 4 (structure du système), le ch. 11 (facteurs de certitude) et le ch. 31 (évaluation).
- Manuel de référence : S. Russell et P. Norvig, *Artificial Intelligence: A Modern Approach*, 4e édition : ch. 6 (satisfaction de contraintes), 7 à 9 (logique et inférence), 10 (représentation des connaissances), 11 (planification automatique).
- Cours en vidéo : P. H. Winston, MIT 6.034 *Artificial Intelligence* (automne 2010), leçon 3 (arbres de buts et systèmes experts à règles) et leçons 7 à 9 (contraintes).
- Sudoku par propagation : P. Norvig, « Solving Every Sudoku Puzzle », la même combinaison propagation + recherche (MRV) en une page de Python.
- Essayer : elizabot, la réimplémentation d’ELIZA de N. Landsteiner (celle de la capture d’écran) ; SWI-Prolog pour jouer avec Prolog ; planning.wiki, référence de PDDL et de ses versions, et le manuel PDDL de 1998.
