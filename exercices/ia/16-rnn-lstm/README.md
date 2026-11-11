# Exercice 16 : RNN, LSTM, recherche en faisceau et attention

| | |
|---|---|
| **Chapitre lié** | [16. Séquences avant les Transformers : RNN, LSTM, attention](https://laclasse.poneyponey.fr/ia/p06-deep-learning/16-sequences/) |
| **Exécution** | locale : numpy et PyTorch sur CPU (parties 1, 2 et 4, petits réseaux), MPS pour le modèle de langage (partie 3) |
| **Commande** | `make ex NUM=16` (éditeur), `make ex NUM=16 MODE=run` (complet), `MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 6 min 06 s (processus entier, attente du verrou exclue), dont 3 s pour le RNN numpy, 29 s pour la tâche de mémoire, 3 min 28 s pour les trois modèles de langage, 3 s de décodage et 2 min pour les deux modèles séquence à séquence ; smoke : 21 s (M1 Pro, MPS, 15/10/2026) |
| **Travail estimé** | ~1 h 30 |
| **Seed** | 42 (`seed_everything(42)`, `np.random.default_rng(42)`, `torch.manual_seed(42)` avant chaque modèle). Les parties CPU sont reproductibles à l'identique ; sur MPS, « même graine = même courbe à quelques décimales près » |
| **Données** | TinyShakespeare (1,1 Mo), téléchargé une fois dans `data/input.txt` par `ia_commun.donnees.charger_tinyshakespeare()` ; les autres données sont générées |

## Objectif

Voir de ses yeux les trois obstacles que les modèles de séquences ont dû franchir avant le Transformer : le gradient qui s'éteint en remontant le temps (et le LSTM qui le fait passer), le choix d'une règle de décodage (glouton, recherche en faisceau, tirage), et le goulot du vecteur unique d'un modèle séquence à séquence, levé par l'attention.

## Ce que vous allez faire

1. **Un RNN en numpy.** Écrire la propagation $h_t = \tanh(x_t W_x + h_{t-1} W_h + b)$ et la rétropropagation dans le temps (BPTT) à la main, puis la vérifier contre l'autograd de PyTorch en float64.
2. **Mesurer le gradient qui remonte le temps** : norme de ∂ℒ/∂h_t sur une séquence de 60 pas, pour quatre rayons spectraux de W_h (0,5 ; 0,9 ; 1,2 ; 2,0).
3. **La tâche de mémoire** : une séquence de T symboles parmi 4, le premier marqué « à retenir », les autres du bruit ; à la fin, redonner le premier (hasard : 25 %). Entraîner le RNN numpy (Adam écrit à la main) pour T = 5, 10, 20, puis `nn.RNN` contre `nn.LSTM` (32 unités, au plus 2 000 pas) pour T = 5, 10, 20, 30, 50. Le biais de la porte d'oubli du LSTM est initialisé à 3 (σ(3) ≈ 0,95).
4. **Un modèle de langage par caractère** sur TinyShakespeare : RNN, GRU et LSTM de 2 couches × 256 unités, 1 000 pas de 64 fenêtres de 128 caractères, avec forçage par l'enseignant (l'entrée au pas t est le vrai caractère t).
5. **Décoder** 200 caractères après « ROMEO: » avec le LSTM : glouton, recherche en faisceau (k = 2, 4, 8, 16), tirage à température 1 et 0,7 ; mesurer la log-probabilité moyenne du texte sous le modèle et la part de 4-grammes de caractères distincts.
6. **Séquence à séquence** : inverser une suite de 4 à 30 lettres (20 lettres possibles) avec un encodeur GRU bidirectionnel et un décodeur GRU, sans attention (le décodeur ne voit que le dernier état de l'encodeur) puis avec l'attention additive de Bahdanau ; mesurer la part de suites exactes selon la longueur et tracer la carte d'alignement.

Ce qui est optimisé, partout : l'entropie croisée de l'élément à prédire, ℒ = −(1/N) Σᵢ log p(yᵢ | contexteᵢ) ; pour le modèle de langage et le modèle séquence à séquence, le contexte est formé des vrais éléments précédents (forçage par l'enseignant).

## Résultat attendu

En mode complet, le notebook vérifie neuf seuils et s'arrête avec une erreur si l'un d'eux n'est pas atteint. Valeurs obtenues le 15/10/2026 (seed 42) :

| Seuil | Valeur obtenue |
|---|---|
| BPTT numpy = autograd (écart < 10⁻¹⁰) | 8,3 × 10⁻¹⁷ |
| RNN numpy : précision ≥ 99 % à T = 10 | 100 % (aussi à T = 5 et 20) |
| `nn.RNN` : précision ≤ 35 % au plus grand retard (T = 50) | 25,5 % |
| `nn.LSTM` : précision ≥ 99 % au plus grand retard (T = 50) | 100 % (résolu en 145 pas) |
| LSTM : perte de validation ≤ 1,60 nats/caractère | 1,553 |
| LSTM meilleur que le RNN (validation) | 1,553 contre 1,601 |
| faisceau k = 8 au moins aussi probable que le glouton | −0,682 contre −0,807 nats/car. |
| avec attention : ≥ 95 % de suites exactes à 30 lettres | 100 % |
| sans attention : ≤ 50 % de suites exactes à 30 lettres | 0 % |

**Gradient dans le temps** (RNN non entraîné, norme de ∂ℒ/∂h relative au dernier pas) :

| Rayon spectral de W_h | 10 pas avant la fin | 30 pas | 60 pas |
|---|---|---|---|
| 0,5 | 2,3 × 10⁻⁴ | 3,5 × 10⁻¹¹ | 2,3 × 10⁻²¹ |
| 0,9 | 2,9 × 10⁻² | 6,7 × 10⁻⁵ | 1,2 × 10⁻⁸ |
| 1,2 | 0,15 | 8,9 × 10⁻³ | 2,6 × 10⁻⁴ |
| 2,0 | 2,3 | 11 | 160 |

**Tâche de mémoire** (précision sur 2 000 séquences de test ; pas où la perte du lot passe sous 0,05) :

| Retard T | 5 | 10 | 20 | 30 | 50 |
|---|---|---|---|---|---|
| RNN numpy | 100 % | 100 % | 100 % | — | — |
| `nn.RNN` | 100 % (79) | 100 % (100) | 73,9 % | 63,2 % | 25,5 % |
| `nn.LSTM` | 100 % (57) | 100 % (74) | 100 % (97) | 100 % (126) | 100 % (145) |

Le RNN numpy réussit T = 20 alors que `nn.RNN` y plafonne à 74 % : même calcul, mais pas la même initialisation (PyTorch tire les poids uniformément dans ±1/√32, notre RNN numpy en loi normale) ni les mêmes séquences d'entraînement ; à T = 20 le RNN est à la limite, et un détail suffit à le faire basculer.

**Modèle de langage par caractère** (65 caractères, hasard = ln 65 ≈ 4,17 nats) :

| Cellule | Paramètres | Validation après 1 000 pas | Perplexité | Durée |
|---|---|---|---|---|
| RNN | 0,23 M | 1,601 | 4,96 | 65 s |
| GRU | 0,66 M | 1,512 | 4,54 | 100 s |
| LSTM | 0,88 M | 1,553 | 4,73 | 42 s |

**Décodage** (LSTM, 200 caractères après « ROMEO: ») :

| Règle | Log-prob. moyenne (nats/car.) | 4-grammes distincts | Début du texte |
|---|---|---|---|
| glouton | −0,807 | 31 % | *What shall be the strange of the strange of the state, / That she shall be the strange of the streats,* |
| faisceau k = 2 | −0,742 | 64 % | *Where is this is the strange of this father,* |
| faisceau k = 4 | −0,680 | 60 % | *What, will you have been to the strange of thee, / And therefore that thou shalt thou art thee,* |
| faisceau k = 8 | −0,682 | 35 % | *My lord, I will not be done to thee, / And therefore that thou shalt thou art thee,* (trois fois) |
| faisceau k = 16 | −0,544 | 78 % | *My lord; and therefore that thou shalt have you. // DUKE VINCENTIO: / My lord, I will not be gone.* |
| tirage τ = 0,7 | −1,098 | 86 % | *You loud so did you for the sun of the nobleman.* |
| tirage τ = 1 | −1,401 | 98 % | *Yes, the savethary we such crond, one anquire to the either.* |

**Séquence à séquence** (suites exactes sur 200 suites par longueur, décodage glouton) :

| Longueur | 4 | 6 | 8 | 10 | 12 | 14 | 16 à 30 |
|---|---|---|---|---|---|---|---|
| sans attention | 98,5 % | 88,5 % | 59,5 % | 25,5 % | 8 % | 2 % | 0 % |
| avec attention | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % (99,5 % à 28) |

Exemple : « girhaaafbtol » → « lotbfaaahrig » ; l'attention met en moyenne 55 % de son poids sur la lettre à recopier, le reste presque entièrement sur sa voisine.

Figures produites : `figures/bilan.png` (quatre panneaux), `figures/alignement.png` ; résultats bruts pour le site : `figures/resultats.npz` (lu par `site/scripts/figures/ch16_*.py`).

## Ce qu'il faut observer

- **Le gradient ne se conserve presque jamais.** À chaque pas en arrière, il est multiplié par W_hᵀ et par la dérivée de tanh (inférieure à 1). Avec un rayon spectral de 0,9, il perd un facteur 34 en 10 pas et plus de 80 millions en 60 ; avec 2,0, il explose (×160). Même à 1,2, la saturation de tanh le fait disparaître. Le RNN apprend mal ce qui dépend d'un événement lointain : il réussit T = 10 en 100 pas, plafonne à 74 % à T = 20 et retombe au hasard à T = 50.
- **Le LSTM passe parce que sa cellule est additive** : c_t = f_t ⊙ c_{t−1} + i_t ⊙ c̃_t. Avec f_t ≈ 1, le gradient traverse la cellule presque intact. Avec un biais d'oubli de 1 au lieu de 3, il résout encore T = 20 (en 126 pas) mais échoue à T = 30 dans ce budget (25 % après 2 000 pas, le hasard) : il faut que la cellule commence par retenir.
- **Sur du texte, l'écart est modeste** (1,60 contre 1,55 et 1,51) : pour prédire la lettre suivante, les quelques caractères précédents font l'essentiel. Le GRU apprend le plus vite à nombre de pas égal, avec environ 25 % de paramètres de moins que le LSTM ; sur MPS, il est pourtant le plus lent par pas.
- **Le texte le plus probable n'est pas le meilleur texte.** Le glouton tourne en boucle (« the strange of the strange »). Le faisceau trouve des suites plus probables mais pas forcément moins répétitives, et une largeur plus grande ne garantit rien : k = 8 fait même un peu moins bien que k = 4 (−0,682 contre −0,680) et se répète davantage. Le tirage produit un texte varié mais moins vraisemblable (mots inventés à τ = 1). C'est pour cela que la recherche en faisceau a servi en traduction (une seule bonne réponse) et que les modèles de langage actuels échantillonnent.
- **Le goulot du vecteur unique** : sans attention, tout passe par 128 nombres ; le modèle sait inverser 4 lettres, mais plus 12 (8 %). Avec attention, la longueur ne compte plus. La carte d'alignement montre l'anti-diagonale, que personne n'a programmée.
- **Forçage par l'enseignant** : à l'entraînement, le décodeur reçoit toujours la bonne lettre précédente ; au test, il reçoit la sienne. Une erreur commise tôt devient donc l'entrée des pas suivants, situation que le modèle n'a jamais rencontrée à l'entraînement.

## Questions de réflexion

1. Dans la BPTT, pourquoi le gradient de W_h est-il une **somme** sur tous les pas ? Que se passerait-il si on coupait la séquence en morceaux de 20 pas sans transmettre le gradient d'un morceau à l'autre (BPTT tronquée) pour la tâche de mémoire à T = 50 ?
2. L'écrêtage de la norme du gradient (à 1 ici) règle-t-il l'explosion ? la disparition ? Pourquoi l'un des deux problèmes est-il bien plus facile à traiter que l'autre ?
3. Le LSTM a une porte d'oubli, pas le RNN. Quelle valeur devrait prendre f_t pendant les T − 1 pas de bruit de la tâche de mémoire ? et la porte d'entrée i_t ?
4. La recherche en faisceau maximise la probabilité de la suite entière. Pourquoi un texte de probabilité maximale est-il souvent répétitif ? Pour quelle tâche voudrait-on quand même la suite la plus probable ?

## Pour aller plus loin

- Mettre `biais_oubli` à 1 dans la configuration : le LSTM résout T = 20 en 126 pas mais échoue à T = 30 et à T = 50 (25 % après 2 000 pas, essai du 15/10/2026, même graine). À 3, il résout aussi T = 70 en un peu plus de 350 pas dans certains essais, mais pas de façon fiable avec cette graine.
- Passer `lm_pas` à 3 000 (non mesuré ici : comptez environ trois fois la durée de la partie 3) et comparer la perte finale et les textes tirés.
- Remplacer l'encodeur bidirectionnel par un GRU simple (`bidirectional=False`, en adaptant les dimensions) : la carte d'alignement devient-elle plus nette ?
- Ajouter une recherche en faisceau au décodeur séquence à séquence et comparer au glouton sur les suites de 12 lettres sans attention.
