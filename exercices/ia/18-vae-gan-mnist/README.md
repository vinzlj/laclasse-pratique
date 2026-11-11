# Exercice 18 : autoencodeurs, VAE et DCGAN sur MNIST

| | |
|---|---|
| **Chapitre lié** | [18. Modèles génératifs I : autoencodeurs, VAE, GAN, flows, autorégressifs](https://laclasse.poneyponey.fr/ia/p06-deep-learning/18-generatifs-1/) |
| **Exécution** | locale, MPS (NumPy pour la partie A, PyTorch pour B à E ; tourne aussi sur CPU, plus lentement) |
| **Commande** | `make ex NUM=18` (éditeur), `make ex NUM=18 MODE=run` (complet), `MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 5 min 15 s (M1 Pro, MPS, 15/10/2026, attente du verrou exclue), dont 59 s pour la partie B (deux autoencodeurs convolutifs), 2 min 06 s pour la partie C (trois MLP, 20 époques chacun), 1 min 34 s pour le DCGAN et 13 s pour le juge ; smoke : environ 10 s |
| **Travail estimé** | ~2 h |
| **Seed** | 42 (`seed_everything(42)` avant chaque modèle) ; ordre des lots fixé par `torch.Generator().manual_seed(42)` ; bruit de test, z fixes et échantillons tirés avec des générateurs à graine fixe (7, 123, 2024) |
| **Données** | MNIST via `ia_commun.donnees.mnist_tenseurs()`, cache `data/mnist/` (≈ 55 Mo, téléchargé au premier lancement) ; 60 000 images d'entraînement, 10 000 de test |

## Objectif

Construire trois familles de modèles génératifs du chapitre 18 (autoencodeur et sa variante débruitante, autoencodeur variationnel, GAN convolutif) sur les mêmes images, voir ce que change chaque ingrédient (le bruit en entrée, le terme KL, l'adversaire), et juger leurs images avec un même outil chiffré plutôt qu'à l'œil.

## Ce que vous allez faire

1. **A. NumPy : un autoencodeur linéaire est une ACP.** Sur 10 000 images centrées, entraîner un autoencodeur 784 → 2 → 784 sans activation, par mini-lots de 500 et un Adam recodé à la main (3 000 pas), puis comparer son erreur de reconstruction à celle de l'ACP à 2 composantes (SVD) et mesurer les cosinus des angles principaux entre les deux plans. Vérifier aussi la formule fermée $D_{\mathrm{KL}}(\mathcal N(\mu, \sigma^2) \,\|\, \mathcal N(0, 1)) = \tfrac12(\mu^2 + \sigma^2 - \log \sigma^2 - 1)$ contre une estimation Monte-Carlo (10⁶ tirages).
2. **B. Autoencodeur classique contre autoencodeur débruitant.** Un réseau convolutif *sur-complet* : 784 pixels → 64 × 7 × 7 = 3 136 valeurs → 784 pixels. Le même réseau est entraîné deux fois, 3 époques, Adam 10⁻³ : à recopier des images propres, puis à reconstruire l'image propre à partir d'une image bruitée (bruit gaussien σ = 0,5, écrêté dans [0, 1]). On compare le rapport signal sur bruit de crête (PSNR) sur les 10 000 images de test bruitées.
3. **C. Autoencodeur contre VAE.** Encodeur 784-512-256-d, décodeur symétrique, vraisemblance de Bernoulli par pixel, 20 époques, Adam 10⁻³, lots de 128. Trois modèles : autoencodeur classique en 2D, VAE en 2D, VAE en 16D. On mesure la reconstruction, l'ELBO de test, la KL par dimension, et on trace les codes des images de test.
4. **D. DCGAN réduit.** Générateur z (64) → 128 × 7 × 7 → 64 × 14 × 14 → 1 × 28 × 28 (tanh), discriminateur à deux convolutions de pas 2 ; hyperparamètres de Radford et al. (2015) : poids initialisés selon une loi normale de moyenne 0 et d'écart-type 0,02, Adam lr 2·10⁻⁴ et β₁ = 0,5, LeakyReLU de pente 0,2, BatchNorm sauf en sortie de G et en entrée de D, lots de 128. Six époques ; on suit les pertes, la sortie moyenne de D et 64 images issues des mêmes z à chaque époque.
5. **E. Le juge.** Un petit CNN (deux convolutions, 3 époques) entraîné sur les vraies images classe 10 000 images de chaque source : vraies images, autoencodeur 2D (codes tirés dans une gaussienne ajustée sur ses codes, faute de loi a priori), VAE 2D et VAE 16D (codes tirés dans N(0, I)), DCGAN. On relève sa confiance moyenne, la part de chaque classe, la part de pixels « gris » (entre 0,1 et 0,9, ni encre ni fond) et la **distance de Fréchet** entre les caractéristiques de son avant-dernière couche (128 valeurs) pour ces images et pour les 10 000 images de test : la formule de la FID (Heusel et al., 2017), avec ce CNN à la place d'Inception, donc des valeurs qui ne se comparent pas aux FID publiées.
6. **Interpolation.** Passer d'un 1 à un 7 du jeu de test en neuf étapes : en mélangeant les pixels, puis en décodant des codes intermédiaires de chaque modèle.

**Pertes optimisées** : erreur quadratique (A) ; entropie croisée binaire par pixel (B et autoencodeur de C) ; opposé de l'ELBO, reconstruction de Bernoulli plus KL fermée (VAE) ; entropie croisée binaire du discriminateur et perte non saturante −log D(G(z)) du générateur (D) ; entropie croisée (juge).

## Résultat attendu

En mode complet, le notebook vérifie douze seuils et s'arrête avec une erreur si l'un d'eux n'est pas atteint. Valeurs du run du 15/10/2026 (M1 Pro, MPS, seed 42 ; deux runs complets ont donné les mêmes valeurs pour C, D et E à l'affichage).

**A. Autoencodeur linéaire et ACP (10 000 images)**

| Mesure | Seuil | Obtenu |
|---|---|---|
| MSE de l'ACP à 2 composantes | — | 0,05614 (16,7 % de la variance expliquée) |
| MSE de l'autoencodeur linéaire | ≤ 1,01 × ACP | 0,05615 (rapport 1,0003) |
| Cosinus des angles principaux entre les deux plans | ≥ 0,99 | 0,9994 et 0,9982 |
| KL fermée contre Monte-Carlo (μ, σ) = (0, 1), (1, 1), (0 ; 0,5), (2 ; 0,3) | écart < 0,01 nat | 0 / 0, 0,5000 / 0,5010, 0,3181 / 0,3184, 2,7490 / 2,7493 nat |

**B. Débruitage (10 000 images de test, bruit σ = 0,5)**

| Entrée → sortie | MSE | PSNR |
|---|---|---|
| image bruitée (sans traitement) | 0,1156 | 9,4 dB |
| autoencodeur classique appliqué à l'image bruitée | 0,0866 | 10,6 dB |
| **autoencodeur débruitant** appliqué à l'image bruitée | **0,0119** | **19,2 dB** |
| autoencodeur classique appliqué à l'image **propre** | 0,0007 | 31,5 dB |

Seuils : débruitant ≥ bruitée + 8 dB (obtenu + 9,8 dB) et ≥ classique + 4 dB (+ 8,6 dB) ; classique sur image propre ≥ 25 dB (il sait recopier).

**C. Autoencodeur et VAE (10 000 images de test, nat par image)**

| Modèle | MSE de reconstruction (décodage de μ) | BCE (décodage de μ) | Reconstruction de l'ELBO (z tiré) | KL | **−ELBO** |
|---|---|---|---|---|---|
| autoencodeur 2D | 0,0356 | 131,7 | — | — | — |
| VAE 2D | 0,0367 | 134,4 | 135,4 | 6,6 (3,5 + 3,1) | **142,0** (seuil ≤ 150) |
| VAE 16D | 0,0117 | 76,3 | 81,5 | 19,3 | **100,7** (seuil ≤ 105) |

Les deux premières colonnes décodent la moyenne μ du code, sans bruit, comme pour l'autoencodeur : c'est la comparaison équitable entre les trois modèles. La colonne « Reconstruction de l'ELBO » est le terme de reconstruction de l'ELBO, estimé avec un code z tiré dans q(z | x) ; elle s'additionne à la KL pour donner −ELBO.

Dans le VAE 16D, 5 dimensions sur 16 ont une KL nulle (0,00 nat) : le modèle n'en a pas l'usage et les laisse égales à la loi a priori ; les 11 autres portent entre 0,8 et 2,6 nat chacune.

**E. Le juge (exactitude de test 98,85 %, seuil ≥ 98,5 %) et les échantillons (10 000 par source)**

| Source | **Distance de Fréchet** | Confiance moyenne du juge | Classe la plus rare | Pixels gris |
|---|---|---|---|---|
| vraies images (entraînement) | 1,7 | 0,988 | 9,1 % | 8,7 % |
| autoencodeur 2D | 507,5 | 0,899 | 1,8 % (les 4) | 24,6 % |
| VAE 2D | 470,7 | 0,890 | 6,0 % | 26,0 % |
| VAE 16D | 172,9 | 0,853 | 6,8 % | 16,8 % |
| **DCGAN** | **62,2** | 0,859 | 4,1 % (les 4) | 8,2 % |

Seuils : DCGAN < VAE 16D < VAE 2D en distance de Fréchet ; VAE 2D < autoencodeur 2D ; le DCGAN produit les 10 classes, chacune ≥ 2 %.

**D. DCGAN (2 808 pas)** : perte de G de 3,92 (50 premiers pas) à environ 1,2 dès le pas 500, puis entre 1,2 et 1,4 ; perte de D de 0,14 à environ 0,9 ; D(vraies) de 0,94 à 0,66, D(fausses) de 0,05 à 0,34, stables ensuite.

Figures produites dans `figures/` : `debruitage.png`, `espace-latent.png`, `echantillons.png`, `gan-courbes.png`, `interpolation.png`, `apprentissage-c.png` ; résultats bruts : `figures/resultats.json` et `figures/images.npz` (≈ 450 Ko, lus par les figures du site, `site/scripts/figures/ch18_*.py`).

## Ce qu'il faut observer

- **Un autoencodeur linéaire ne fait rien de plus qu'une ACP.** Même erreur à 0,03 % près, même plan (cosinus 0,999). Ce qui rend un autoencodeur intéressant, ce sont les couches non linéaires : en 2D, l'autoencodeur MLP de la partie C reconstruit les images de test avec une MSE de 0,036, contre 0,056 pour l'ACP au même nombre de dimensions (mesurée, elle, sur les 10 000 images de la partie A).
- **Un autoencodeur sur-complet apprend l'identité.** Avec 3 136 valeurs au centre pour 784 pixels, rien n'oblige à résumer : l'autoencodeur classique recopie les images propres presque parfaitement (31,5 dB) et recopie aussi le bruit (10,6 dB contre 9,4). Le même réseau, entraîné avec du bruit en entrée, ne peut plus tricher : il doit savoir à quoi ressemble un chiffre pour l'effacer (19,2 dB). La contrainte vient de la tâche, pas de la taille du code.
- **L'autoencodeur reconstruit mieux, le VAE génère mieux.** En 2D, l'autoencodeur gagne en reconstruction (131,7 nat contre 134,4 pour le VAE, en décodant dans les deux cas le code moyen) mais ses codes s'étalent sur des dizaines d'unités, en bras séparés par du vide ; tirer un code au hasard tombe souvent dans un trou (distance 507,5, et le chiffre 4 n'apparaît presque plus : 1,8 %). Le terme KL du VAE (6,6 nat) tasse les codes autour de l'origine, et un code tiré dans N(0, I) donne presque toujours un chiffre (470,7, toutes les classes entre 6 et 15 %).
- **Un code plus grand aide beaucoup, et le VAE choisit ce qu'il utilise.** Passer de 2 à 16 dimensions réduit fortement la reconstruction (terme de l'ELBO : 135,4 → 81,5 nat) et la distance de Fréchet (470,7 → 172,9). Cinq dimensions restent inutilisées (KL nulle) : leur coût KL dépasserait ce qu'elles apporteraient à la reconstruction.
- **Les images du VAE sont floues, celles du GAN sont nettes.** La vraisemblance de Bernoulli récompense une valeur moyenne quand le modèle hésite sur un pixel : d'où 17 à 26 % de pixels gris contre 8,7 % sur les vraies images. Le DCGAN, jugé par un discriminateur qui repère le flou, a 8,2 % de pixels gris et la meilleure distance (62,2), avec des chiffres parfois mal formés.
- **La confiance du juge est un mauvais critère seul.** Les échantillons flous du VAE 2D obtiennent une confiance moyenne plus haute (0,890) que ceux du DCGAN (0,859) : un chiffre flou et « moyen » est facile à classer. La distance de Fréchet compare la distribution entière (moyenne et covariance des caractéristiques), elle pénalise le manque de variété et le flou.
- **Les pertes d'un GAN ne disent pas si les images s'améliorent.** Pertes et sorties de D sont plates dès 500 pas environ, alors que les images issues des mêmes z continuent de progresser jusqu'à l'époque 6. La perte mesure un équilibre entre deux réseaux, pas une qualité. Le GAN ne s'est pas effondré sur quelques modes ici, mais ses classes sont déséquilibrées (4 : 4,1 %, 9 : 15,6 %).
- **Interpoler dans l'espace latent fabrique des chiffres, interpoler les pixels les superpose.** Avec 16 dimensions, le 1 se courbe progressivement en 7 ; avec 2 dimensions, le chemin traverse d'autres chiffres ou bascule brusquement.

## Questions de réflexion

1. La partie A montre que l'autoencodeur linéaire retrouve le **plan** de l'ACP. Retrouve-t-il aussi les deux **axes** de l'ACP (première et deuxième composante) ? Ajoutez dans la cellule de la partie A le produit des lignes du décodeur (`_params["Wd"]`, normalisées) par les deux premières composantes (`composantes`), et expliquez pourquoi la perte ne peut pas distinguer les axes à l'intérieur du plan.
2. Remplacez la vraisemblance de Bernoulli du VAE 2D par une erreur quadratique multipliée par un coefficient β (essayez 1, 10, 100). Que deviennent l'équilibre reconstruction / KL et le flou ? Qu'est-ce que ce coefficient représente du point de vue probabiliste ?
3. Pourquoi le VAE peut-il calculer (une borne de) la vraisemblance de ses images, et pas le GAN ? Que faudrait-il pour comparer les deux sur ce critère ?
4. Le discriminateur termine à D(vraies) ≈ 0,66 et D(fausses) ≈ 0,34, loin de l'équilibre théorique 0,5 / 0,5. Que se passerait-il si on entraînait D cinq fois par pas de G ? Essayez et regardez les images.

## Pour aller plus loin

- Partie B : remplacer le bruit gaussien par le bruit de masquage de Vincent et al. (2008) : mettre à zéro une fraction des pixels (25 %, 50 %, 75 %).
- Partie C : entraîner l'autoencodeur 2D avec une erreur quadratique après sigmoïde au lieu de l'entropie croisée binaire, et regarder les premières époques. Le même piège a été rencontré en écrivant la partie B : avec une MSE après sigmoïde, le réseau convolutif restait bloqué sur une image presque noire (MSE ≈ 0,11 après 300 pas, contre 0,002 avec l'entropie croisée).
- Partie D : passer à la perte « saturante » log(1 − D(G(z))) de l'article original (Goodfellow et al., 2014) et comparer les 500 premiers pas.
- Articles : Baldi et Hornik, « Neural networks and principal component analysis: Learning from examples without local minima », *Neural Networks* 2(1), 1989, [doi:10.1016/0893-6080(89)90014-2](https://doi.org/10.1016/0893-6080(89)90014-2) ; Vincent et al., ICML 2008, [doi:10.1145/1390156.1390294](https://doi.org/10.1145/1390156.1390294) ; Kingma et Welling, [arXiv:1312.6114](https://arxiv.org/abs/1312.6114) ; Goodfellow et al., [arXiv:1406.2661](https://arxiv.org/abs/1406.2661) ; Radford et al., [arXiv:1511.06434](https://arxiv.org/abs/1511.06434) ; Heusel et al. (FID), [arXiv:1706.08500](https://arxiv.org/abs/1706.08500).
