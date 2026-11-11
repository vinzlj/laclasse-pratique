# Exercice 07 : rétropropagation, MLP et fonctions d'activation

| | |
|---|---|
| **Chapitre lié** | [07. Rétropropagation, MLP et fonctions d'activation](https://laclasse.poneyponey.fr/ia/p03-connexionnisme/07-retropropagation/) |
| **Exécution** | locale : NumPy sur CPU (parties 1 à 5), PyTorch sur MPS (partie 6) ; tourne aussi sur CPU seul |
| **Commande** | `make ex NUM=07` (éditeur marimo), `make ex NUM=07 MODE=run` (complet), `make ex NUM=07 MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 20 à 29 s au total (trois mesures), dont MNIST numpy 0,34 à 0,6 s par époque et PyTorch 0,62 à 0,73 s par époque (0,87 à 1,35 s pour la première) ; smoke : 4 à 7 s (M1 Pro, MPS, 14 et 15/10/2026) |
| **Travail estimé** | ~2 h |
| **Seed** | 42 (`seed_everything(42)`, `np.random.default_rng(42)`, `torch.manual_seed(42)`) |
| **Données** | MNIST (téléchargé une fois par `ia_commun.donnees.mnist_tenseurs`, ~63 Mo dans `data/mnist/`) ; XOR bruité généré par le notebook |

## Objectif

Calculer soi-même le gradient de la perte par rapport à tous les poids d'un réseau multicouche (passe avant, puis passe arrière par la règle de la chaîne), le vérifier par différences finies, voir ce qu'une couche cachée apporte (XOR) et ce que le choix de l'activation change en profondeur, puis constater que l'autograd de PyTorch fait exactement le même calcul.

## Ce que vous allez faire

1. **Réseau 2-2-1 à la main** (sigmoïdes, entropie croisée binaire, x = (1, 0), y = 1) : afficher chaque valeur de la passe avant et de la passe arrière, faire un pas de descente de gradient, puis refaire le calcul avec autograd et lire le graphe de calcul que PyTorch a enregistré.
2. **MLP numpy de taille quelconque** : écrire `propager` (passe avant, avec cache des $z$ et des activations) et `retropropager` (passe arrière, $\partial\mathcal{L}/\partial W_\ell = a_{\ell-1}^\top \delta_\ell$, $\delta_{\ell-1} = \delta_\ell W_\ell^\top \odot f'(z_{\ell-1})$), pour la sigmoïde, tanh et ReLU, avec softmax et entropie croisée en sortie.
3. **Contrôle du gradient** : comparer chacune des 59 dérivées partielles d'un réseau 3-5-4-3 à des différences finies centrées ; recommencer avec une passe arrière volontairement fausse (dérivée de l'activation oubliée).
4. **XOR bruité** (400 points en quatre nuages) : modèle sans couche cachée contre MLP 2-16-2 (tanh), descente de gradient sur tout le lot (3 000 pas, taux 0,5).
5. **Profondeur** : 8 couches cachées de 32 neurones, même initialisation (poids ~ N(0, 1/n_entrée)) pour la sigmoïde, tanh et ReLU ; mesurer la norme du gradient de chaque couche à l'initialisation, puis entraîner 2 000 pas (taux 0,05).
6. **MNIST en numpy** : MLP 784-256-256-10 (ReLU), SGD par mini-lots de 64, taux 0,1, 10 époques, en float32.
7. **Même modèle en PyTorch** : vérifier sur un lot que les gradients d'autograd (float64, CPU) coïncident avec ceux de `retropropager`, puis entraîner avec `nn.Sequential`, `cross_entropy` et `torch.optim.SGD` (mêmes réglages, sur MPS).

Figures produites : `figures/xor.png` (frontières de décision), `figures/profondeur.png` (gradient par couche et courbes de perte), `figures/mnist.png` (numpy contre PyTorch).

## Résultat attendu

Le notebook vérifie ces seuils en mode complet et s'arrête avec une erreur si l'un manque :

| Contrôle | Seuil | Obtenu (14 et 15/10/2026) |
|---|---|---|
| 2-2-1 : un pas de descente fait baisser la perte | oui | 0,5782 → 0,4636 |
| 2-2-1 : nos gradients contre autograd | écart < 10⁻¹² | 5,6 × 10⁻¹⁷ |
| Différences finies (sigmoïde, tanh, ReLU) | erreur relative < 10⁻⁷ | 1,1 × 10⁻⁹ ; 7,9 × 10⁻¹⁰ ; 8,3 × 10⁻¹⁰ |
| Même contrôle, f′(z) oubliée | erreur détectée (> 10⁻²) | 0,41 ; 0,19 ; 0,42 |
| XOR, sans couche cachée | exactitude test ≤ 60 % | 45,8 % (perte bloquée à 0,692 ≈ ln 2) |
| XOR, MLP 2-16-2 | exactitude test ≥ 97 % | 99,5 % (perte finale 0,014) |
| Profond, sigmoïde : gradient couche 1 / couche 9 | < 10⁻⁴ | 7,8 × 10⁻⁷ |
| Profond, tanh et ReLU : même rapport | > 10⁻² | 0,22 et 0,87 |
| Profond, perte finale après 2 000 pas | sigmoïde > 0,65 ; tanh et ReLU < 0,1 | 0,693 ; 0,012 ; 0,010 |
| MNIST numpy, exactitude test après 10 époques | ≥ 97,5 % | 98,16 % |
| MNIST PyTorch, exactitude test après 10 époques | ≥ 97,5 % | 97,83 % |
| Gradients numpy contre autograd, MLP MNIST (269 322 poids) | écart < 10⁻¹⁰ | 8,7 × 10⁻¹⁸ |

Valeurs du réseau 2-2-1 (W1 = [[0,5 ; −0,5], [0,3 ; 0,8]], W2 = (1 ; −1), biais nuls) : z1 = (0,5 ; −0,5), h = (0,6225 ; 0,3775), z2 = 0,2449, ŷ = 0,5609, ℒ = 0,5782 ; δ2 = −0,4391, ∂ℒ/∂W2 = (−0,2733 ; −0,1658), σ′(z1) = (0,2350 ; 0,2350), δ1 = (−0,1032 ; 0,1032), ∂ℒ/∂W1 = [[−0,1032 ; 0,1032], [0 ; 0]].

## Ce qu'il faut observer

- **Le gradient d'un poids = activation en amont × erreur en aval.** Dans le 2-2-1, l'entrée x₂ vaut 0 : les poids qui en partent ont un gradient nul, quel que soit le reste du réseau.
- **Le contrôle par différences finies est impitoyable** : un gradient juste donne 10⁻⁹ environ (l'erreur d'arrondi), un oubli de f′(z) donne 0,2 à 0,4. C'est le premier test à écrire quand on code une passe arrière à la main. Il coûte deux passes avant **par paramètre** (118 pour 59 paramètres), la rétropropagation une passe arrière pour tous.
- **Sans couche cachée, le XOR est hors d'atteinte** : la perte reste à ln 2 (le modèle répond 50/50 partout) et l'exactitude est celle du hasard. Une couche cachée de 16 neurones suffit.
- **En profondeur, la sigmoïde éteint le gradient** : à chaque couche il est multiplié par σ′(z) ≤ 0,25, et la première couche reçoit un gradient un million de fois plus petit que la dernière. Le réseau ne bouge pas en 2 000 pas. Avec tanh (dérivée 1 en 0) et ReLU (dérivée 1 pour z > 0), le gradient garde le même ordre de grandeur d'une couche à l'autre et la perte passe sous 0,1 en 86 pas (tanh) et 106 pas (ReLU).
- **numpy et PyTorch font le même calcul** : écart de 10⁻¹⁷ sur 269 322 gradients. Les courbes MNIST diffèrent seulement parce que l'initialisation et l'ordre des mini-lots ne sont pas les mêmes.
- **Le GPU n'aide pas ici** : pour un MLP de cette taille et des lots de 64, numpy sur CPU (0,35 à 0,6 s par époque) va plus vite que PyTorch sur MPS (environ 0,65 s) ; le coût fixe de chaque lancement de calcul sur le GPU domine.

## Questions de réflexion

1. Dans le 2-2-1, pourquoi δ1 a-t-il deux composantes de même valeur absolue et de signes opposés ? Que deviendrait ∂ℒ/∂W1 avec x = (1 ; 0,5) ?
2. Pour un réseau d'un million de poids, combien de passes avant demanderait le gradient par différences finies ? Pourquoi la rétropropagation n'en demande-t-elle qu'une, plus une passe arrière ?
3. Dans la partie 4, le gradient de la couche 1 avec la sigmoïde est d'environ 10⁻⁶ fois (7,8 × 10⁻⁷) celui de la couche 9 (la couche de sortie). Quel facteur moyen par couche cela représente-t-il ? Comparez à 0,25.
4. ReLU a une dérivée nulle pour z inférieur à 0. Quel problème cela pose-t-il si un neurone reçoit z négatif pour tous les exemples ? Quelle variante du chapitre le corrige ?

## Pour aller plus loin

- Remplacer tanh par ReLU dans le MLP 2-16-2 du XOR : la frontière change-t-elle d'allure ?
- Dans la partie 4, passer le taux d'apprentissage à 0,1 : avec ReLU, la perte fait un bond brutal vers 10 vers le pas 1 150 avant de redescendre (instabilité de la descente de gradient à pas fixe).
- Écrire un petit autograd scalaire (chaque opération enregistre ses parents et sa dérivée locale), comme micrograd dans la première vidéo de la série « Neural Networks: Zero to Hero » d'Andrej Karpathy : https://karpathy.ai/zero-to-hero.html
