# Exercice 02 : optimiseurs, de SGD à Adam, AdamW et Muon

| | |
|---|---|
| **Chapitre lié** | [02. Optimisation](https://laclasse.poneyponey.fr/ia/p00-outils/02-optimisation/) |
| **Exécution** | locale ; numpy sur CPU pour les parties 1 à 6 et 8, MPS pour le MLP de la partie 7 |
| **Commande** | `make ex NUM=02` (éditeur, curseurs et bouton), `make ex NUM=02 MODE=run` (complet), `make ex NUM=02 MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 18 à 25 s (trois runs), dont 12 à 16 s pour les cinq MLP sur MNIST ; smoke : 6 à 7 s (M1 Pro, MPS, 14/10/2026) |
| **Travail estimé** | ~1 h 30 |
| **Seed** | 42 (`seed_everything(42)`, `np.random.default_rng(42)`) |
| **Données** | surfaces et régression générées ; MNIST (≈ 63 Mo) via `ia_commun.donnees.mnist_tenseurs`, en cache dans `data/mnist/` |

## Objectif

Écrire soi-même SGD, momentum, Nesterov, AdaGrad, RMSProp, Adam et AdamW en numpy, voir sur des surfaces 2D ce que chacun corrige chez le précédent, puis vérifier que `torch.optim` fait exactement les mêmes calculs avant de les comparer sur un vrai réseau. Bonus : l'orthogonalisation de Muon.

Aucune fonction de perte propre au chapitre : l'optimiseur minimise ce qu'on lui donne (fonctions test 2D, erreur quadratique moyenne d'une régression, entropie croisée d'un classifieur MNIST ; pertes du chapitre 01).

## Ce que vous allez faire

1. **Surfaces** : coder trois fonctions test et leur gradient (vérifié par différences finies) :
   vallée étroite $\tfrac12(x^2 + 50y^2)$, point selle $x^2 + \tfrac14y^4 - \tfrac12y^2$, Rosenbrock $(1-x)^2 + 100(y-x^2)^2$.
2. **Optimiseurs en numpy** : sept classes de quelques lignes, aux conventions de PyTorch 2.14.
3. **Trajectoires** : les lancer depuis le même point de départ, compter les pas pour passer sous un seuil, rejouer les trajectoires au curseur (éditeur). Balayer le pas η sur la vallée ; tester l'écrêtage du gradient sur Rosenbrock.
4. **Taille de lot et planning** : régression linéaire (4 000 exemples, 20 variables) en plein lot, en mini-lots de 32 à pas constant, puis avec échauffement + cosinus.
5. **Adam + L2 contre AdamW** sur un jouet à deux poids.
6. **Second ordre** : Newton (hessienne exacte) et L-BFGS (`scipy`) sur Rosenbrock.
7. **Bibliothèque** : refaire 500 pas avec `torch.optim` et comparer aux classes numpy ; puis entraîner cinq fois le même MLP 784-256-256-10 sur MNIST (SGD, SGD + Nesterov, Adam, AdamW, Muon + AdamW), 2 époques, lots de 128.
8. **Bonus Muon** : Newton-Schulz en numpy, effet sur les valeurs singulières, comparaison à `torch.optim.Muon`, jouet de régression matricielle.

## Résultat attendu

Le run complet vérifie quatorze contrôles et s'arrête avec une erreur si l'un échoue. Résultats obtenus le 14/10/2026 (M1 Pro) :

| Contrôle (seuil) | Résultat mesuré |
|---|---|
| Rosenbrock, 3 000 pas depuis (−1,5 ; 2) : SGD à plus de 0,1 de (1, 1) | SGD finit en (0,78 ; 0,61), f = 0,049 |
| … momentum, Nesterov et Adam à moins de 0,01 | f < 10⁻³ atteint aux pas 666 (momentum), 557 (Nesterov), 1 564 (Adam) ; f final ≈ 10⁻¹¹ |
| Vallée étroite, même η = 0,025 : momentum au moins 2 fois plus rapide que SGD | f < 10⁻³ au pas 73 (momentum), 53 (Nesterov) contre 205 (SGD) |
| Point selle, départ (1,5 ; 10⁻⁴) : Adam et RMSProp sortent au moins 3 fois plus vite que SGD | f < −0,24 au pas 35 (Adam), 9 (RMSProp), 54 (momentum), 49 (Nesterov), 202 (SGD) |
| Vallée : SGD diverge à η = 0,041, converge à η = 0,039 (limite théorique 2/50 = 0,04) | η = 0,041 : f = 2,7·10¹³ au pas 284 ; η = 0,039 : f = 1,4·10⁻⁹ |
| Écrêtage : SGD à η = 3·10⁻³ diverge sans, reste stable avec (norme ≤ 10), f < 0,2 | sans : divergence en 5 pas ; avec : (0,92 ; 0,82), f = 0,105 |
| Mini-lots : échauffement + cosinus 100 fois sous le plancher du pas constant | excès de perte final 2,8·10⁻⁵ contre 6,5·10⁻³ (lot 32, η = 0,1) |
| Adam + L2 : w₂ identique pour λ = 0,01 et λ = 0,1 | 0,2244 au pas 100 dans les deux cas |
| AdamW : w₂ dépend de λ | 0,951 (λ = 0,01) et 0,606 (λ = 0,1) au pas 500 |
| Newton < 10 itérations, L-BFGS < 60 | Newton : 7 ; L-BFGS : 44 itérations (54 évaluations), erreur 1,1·10⁻¹¹ |
| numpy = `torch.optim` à 10⁻¹⁰ près (500 pas, float64) | écart maximal 3,3·10⁻¹⁵ (sept optimiseurs) |
| MNIST : Adam ≥ 96,5 % en test, SGD + Nesterov > SGD | SGD 95,04 %, SGD + Nesterov 97,17 %, Adam 97,17 %, AdamW 97,03 %, Muon + AdamW 97,11 % |
| Muon : valeurs singulières après Newton-Schulz dans [0,5 ; 1,5] ; cosinus avec `torch.optim.Muon` > 0,99 | de 0,56 à 1,19 (avant : de 0,018 à 8,4) ; cosinus 0,9965 |
| Jouet matriciel : Muon et Adam sous 10⁻³ | Adam 7,2·10⁻⁷ (η = 1), Muon 1,3·10⁻⁵ (η = 0,2), Nesterov 3,7·10⁻³ (η = 0,4) |

Les chiffres MNIST se sont reproduits à l'identique sur deux runs ; sur MPS, « même graine » garantit la même courbe à quelques décimales près, pas au bit près.

Figures produites dans `figures/` : `trajectoires.png`, `mini-lots.png`, `adam-l2-adamw.png`, `mnist.png`, `muon.png`.

## Ce qu'il faut observer

- **Vallée étroite** : SGD est bridé par la direction raide ($y$, courbure 50) et rampe dans la direction plate ($x$, courbure 1) : à η = 0,025, $x$ ne perd que 2,5 % par pas. Le momentum accumule la vitesse le long de $x$ (3 fois moins de pas) mais zigzague en $y$ ; Adam et RMSProp ramènent les deux axes à la même échelle (parce que la vallée est alignée sur les axes : voir la question 1).
- **Point selle** : le gradient y est presque nul (y = 10⁻⁴), donc SGD fait des pas presque nuls et y reste 200 pas. RMSProp et Adam divisent par la taille du gradient : un gradient minuscule donne quand même un pas de l'ordre de η (10 η au premier pas pour RMSProp, faute de correction de biais : sa moyenne $s$ part de 0, donc $\sqrt{s_1} = 0{,}1\,|g|$).
- **Rosenbrock** : tout le monde tombe vite dans la vallée en banane ; c'est de la **suivre** qui coûte. SGD et RMSProp n'arrivent pas en 3 000 pas, momentum et Nesterov si.
- **AdaGrad** (dans le tableau) ralentit puis s'arrête : sa somme de $g^2$ ne fait que grandir. Il ne sort pas du col en 300 pas. C'est exactement ce que RMSProp corrige en oubliant les vieux gradients.
- **Mini-lots** : à calcul égal (une époque), le mini-lot de 32 à η = 0,1 est déjà 200 fois plus bas que le plein lot (1,6·10⁻² contre 3,65) ; mais à pas constant il stagne sur un plancher de bruit, d'autant plus haut que η est grand. Faire décroître η (cosinus) supprime ce plancher.
- **Adam + L2** : le λ disparaît dans la normalisation d'Adam. AdamW rend au weight decay son sens.
- **Newton** atteint (1, 1) à 10⁻⁸ près en 7 itérations, là où Adam met 1 564 pas pour passer sous f = 10⁻³, mais chaque itération résout un système avec la hessienne : impossible à $10^9$ paramètres. L-BFGS est le compromis, efficace sur des problèmes déterministes de taille moyenne.
- **MNIST** : sur ce petit MLP, momentum, Adam, AdamW et Muon finissent à 0,15 point les uns des autres ; seul SGD sans momentum est nettement derrière. Les différences entre optimiseurs modernes se voient surtout à grande échelle.
- **Muon** : une mise à jour dont les valeurs singulières s'étalent sur un rapport de 462 devient une matrice où toutes les directions comptent à peu près autant. Sur notre jouet quadratique, Adam reste devant : Muon n'est pas fait pour ce type de problème.

## Questions de réflexion

1. Tournez la vallée étroite de 45° ($f = \tfrac12 (u^2 + 50 v^2)$ avec $u = (x+y)/\sqrt2$, $v = (x-y)/\sqrt2$). Pour comparer à problème égal, tournez aussi le départ : (−8 ; 1) sur la vallée tournée correspond à (−4,95 ; −6,36) sur la vallée alignée. Comptez les pas de SGD, du momentum, de Nesterov, de RMSProp et d'Adam dans les deux cas. Lesquels ralentissent, et pourquoi la normalisation « coordonnée par coordonnée » d'Adam dépend-elle des axes ?
2. À η = 0,028, momentum converge encore sur la vallée mais Nesterov diverge (essayez : il explose vers le pas 157). Nesterov n'est donc pas « momentum en mieux » dans tous les cas : qu'est-ce qui change dans sa mise à jour ?
3. Pourquoi Adam sort-il du point selle 6 fois plus vite que SGD, alors que son pas η = 0,05 est le même ?
4. Le mini-lot de 32 à η = 0,4 constant finit plus haut que celui à η = 0,1. Qu'est-ce que le cosinus garde de chacun des deux ?

## Pour aller plus loin

- Remplacer Adam par AdamW avec λ = 0,1 sur MNIST pendant 10 époques et suivre la norme des poids.
- Essayer `torch.optim.Muon(..., adjust_lr_fn="match_rms_adamw")` pour réutiliser le pas d'AdamW (option documentée de PyTorch 2.14).
- Références : Kingma & Ba, « Adam: A Method for Stochastic Optimization », [arXiv:1412.6980](https://arxiv.org/abs/1412.6980) ; Loshchilov & Hutter, « Decoupled Weight Decay Regularization », [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) ; K. Jordan, « Muon: An optimizer for hidden layers in neural networks », [billet du 08/12/2024](https://kellerjordan.github.io/posts/muon/) et [code](https://github.com/KellerJordan/Muon) ; G. Hinton, *Neural Networks for Machine Learning*, [leçon 6e (RMSProp)](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) ; H. H. Rosenbrock, « An Automatic Method for Finding the Greatest or Least Value of a Function », *The Computer Journal* 3(3):175-184, 1960, [doi:10.1093/comjnl/3.3.175](https://doi.org/10.1093/comjnl/3.3.175).
