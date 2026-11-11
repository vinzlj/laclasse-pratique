# Exercice 37 : l'effondrement des modèles, une gaussienne et des mots qui apprennent sur eux-mêmes

| | |
|---|---|
| **Chapitre lié** | [37. Synthèse : comment le Transformer a absorbé domaine après domaine](https://laclasse.poneyponey.fr/ia/p09-absorption/37-synthese/) |
| **Exécution** | locale, CPU, numpy seulement (aucun réseau de neurones, aucun GPU) |
| **Commande** | `make ex NUM=37` (éditeur marimo), `make ex NUM=37 MODE=run` (complet), `make ex NUM=37 MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 9 s (trois lancements, attente du verrou exclue), dont gaussienne 2,6 à 2,8 s, unigrammes 0,1 s, bigrammes 3,5 à 3,7 s, le reste en imports et figures ; smoke : 4 s (M1 Pro, CPU, 16/10/2026) |
| **Travail estimé** | 30 min à 1 h |
| **Bouton** | aucun : le calcul complet prend 9 s, tout part à l'ouverture dans l'éditeur |
| **Seed** | 42 (`seed_everything(42)` ; générateurs numpy `default_rng(42)` pour la gaussienne, 43 pour les unigrammes, 44 pour chaque régime des bigrammes) |
| **Données** | TinyShakespeare (1,1 Mo), téléchargé une fois dans `data/input.txt` par `ia_commun.donnees.charger_tinyshakespeare` ; la partie 1 n'utilise que des tirages aléatoires |

## Objectif

Voir, sur les deux exemples les plus simples possibles, ce qui arrive à un modèle qu'on réentraîne génération après génération sur ce que produit la génération précédente : la **queue** de la distribution disparaît d'abord, puis tout se resserre vers un point. Mesurer ensuite ce qui l'empêche : garder une part de vraies données, ou accumuler au lieu de remplacer.

Ce qui est optimisé : à chaque génération, la **vraisemblance** des données du moment (maximum de vraisemblance, chapitre 00). Pour une gaussienne, $\hat\mu$ et $\hat\sigma$ empiriques ; pour des n-grammes, des comptes divisés par des totaux. Il n'y a ni gradient ni réseau : l'effondrement vient de l'**échantillonnage fini**, pas de l'optimiseur.

## Ce que vous allez faire

1. **Une gaussienne** (`generations_gaussiennes`). On tire $n$ points de $\mathcal{N}(0,1)$, on ajuste $\hat\mu, \hat\sigma$, on tire $n$ points de ce modèle, on réajuste, 1 000 fois ; 200 chaînes indépendantes en parallèle. Six régimes : remplacer avec $n = 10$, 100 et 1 000 ; $n = 100$ avec l'estimateur sans biais ($n-1$ au dénominateur) ; $n = 100$ dont 10 % de vraies données neuves à chaque génération ; $n = 100$ en accumulant toutes les données depuis le début. On compare l'écart-type médian à la prévision $\exp\big(\tfrac{G}{2}\,\mathbb{E}[\log \chi^2_{n-1}/n]\big)$ (`sigma_median_theorique`).
2. **Des mots, version unigramme.** TinyShakespeare découpé en mots et ponctuation (250 148 jetons, 90 % pour l'entraînement : 225 133 jetons, 11 690 mots distincts). Le modèle = la fréquence de chaque mot ; une génération = un tirage multinomial de 225 133 mots ; 300 générations. On vérifie la loi « un mot vu $c$ fois disparaît avec une probabilité $\approx e^{-c}$ ».
3. **Des mots, version bigramme** (`compter_paires`, `table_echantillonnage`, `generer`). Le modèle prédit le mot suivant d'après le mot courant. Une génération = générer environ autant de mots (2 000 flux de 112 mots en parallèle, soit 224 000 mots) puis recompter les paires ; 30 générations. Quatre régimes : remplacer à la température 1 ; remplacer à la température 0,7 (probabilités élevées à la puissance $1/T$ puis renormalisées) ; 10 % de vrai texte (un passage contigu du corpus d'entraînement tiré au hasard) à chaque génération ; accumuler. On mesure le vocabulaire, le nombre de paires distinctes, l'entropie des mots générés et l'**entropie croisée sur les 10 % de texte réel de test** (bits par mot), avec un modèle lissé $0{,}7\,P_{\text{bigramme}} + 0{,}3\,P_{\text{unigramme}}$ (unigramme lissé par add-one sur le vocabulaire réel) pour qu'elle reste finie.

## Résultat attendu

Le run complet vérifie 13 seuils et s'arrête avec une erreur si l'un manque. Valeurs mesurées le 16/10/2026, identiques au chiffre près sur trois lancements (graine 42) :

| Partie | Seuil vérifié | Mesuré |
|---|---|---|
| Gaussienne, remplacer, $n = 100$ | écart-type médian **< 0,01** après 1 000 générations | **3,3·10⁻⁵** (prévu 4,1·10⁻⁵) ; 100 % des chaînes sous 0,1 |
| Gaussienne, sans biais, $n = 100$ | s'effondre aussi : écart-type médian **< 0,05** | **0,008** (prévu 0,006) ; 89 % des chaînes sous 0,1 |
| Gaussienne, remplacer, $n = 100$ | masse médiane de la queue $\lvert x\rvert > 2$ **< 10⁻⁶** (vraie loi : 0,0455) | **0** (en virgule flottante) |
| Gaussienne | plus $n$ est grand, plus c'est lent | écart-type médian final : $n = 10$ : **0** (en virgule flottante) ; $n = 100$ : 3,3·10⁻⁵ ; $n = 1\,000$ : **0,375** (prévu 0,37) |
| Gaussienne, accumuler | écart-type médian entre 0,9 et 1,1, 10e centile > 0,8 | **0,974** (10e–90e centiles : 0,87–1,08) |
| Gaussienne, 10 % de réel | écart-type médian **≥ 0,5** | **0,936** (0,80–1,15) |
| Unigrammes | hapax perdus dès la génération 1 entre 33 et 41 % | **36,8 %** (prévu $e^{-1}$ = 36,8 %) |
| Unigrammes | le vocabulaire ne remonte jamais | oui : 11 690 → 9 459 (g1) → 4 595 (g10) → 1 253 (g100) → 590 (g300) |
| Bigrammes, $T = 1$ | vocabulaire final **≤ 75 %** du départ, sans jamais remonter | **2 666 mots (23 %)** |
| Bigrammes, $T = 0{,}7$ | vocabulaire final **≤ 25 %** | **2** (« , » et « and ») |
| Bigrammes, $T = 1$ | entropie croisée de test en hausse d'au moins 0,5 bit | **8,51 → 9,51** bits par mot |
| Bigrammes, accumuler | entropie croisée de test à moins de 0,3 bit du départ (le vocabulaire, lui, reste intact par construction : on n'enlève jamais de compte) | **8,51 → 8,55** |
| Bigrammes, 10 % de réel | garde plus de vocabulaire que $T = 1$ | **6 997 mots (60 %)** |

Autres valeurs utiles :

- Génération où l'écart-type médian passe sous 0,5, puis sous 0,01 : $n = 10$ : 6 et 41 ; $n = 100$ : 72 et 445 ; sans biais : 136 et 951 ; $n = 1\,000$ : 730 et jamais. À $n = 10$, l'écart-type finit à 0 en virgule flottante dans 199 chaînes sur 200 (mathématiquement, $\hat\sigma$ reste positif ; mais sous la précision des nombres autour de $\hat\mu$, environ $10^{-16}$, les $n$ tirages sortent identiques et l'écart-type calculé vaut exactement 0), et la moyenne s'est figée en un point **aléatoire** (écart-type des moyennes finales entre chaînes : 0,86) : le modèle ne converge pas vers la vraie moyenne 0, il converge vers « n'importe où ».
- Une chaîne ($n = 100$, remplacer) : $\hat\sigma$ = 0,99 (g0), 0,76 (g50), 0,18 (g200), 0,002 (g500), 0,00 (g1 000) ; $\hat\mu$ = −0,04, 0,96, 1,31, 1,24.
- Unigrammes, part des mots perdus dès la génération 1 selon leur compte initial $c$ : 36,8 % ($c = 1$, prévu 36,8 %), 13,6 % ($c = 2$, prévu 13,5 %), 3,7 % ($c = 3$, prévu 5,0 % : peu de mots, la statistique est bruitée), 0,7 % ($c = 5$, prévu 0,7 %). Entropie : 9,01 → 8,17 bits (g100) → 7,67 bits (g300).
- Bigrammes, paires distinctes à la génération 30 : 86 953 au départ ; $T = 1$ : 16 486 ; $T = 0{,}7$ : 4 ; 10 % de réel : 42 736 ; accumuler : 117 603 (les paires inventées s'ajoutent). Entropie des mots générés : $T = 1$ : 9,01 → 8,57 bits ; $T = 0{,}7$ : 9,01 → 3,51 (g5) → 1,00 bit (deux mots équiprobables, « , » et « and ») ; entropie croisée de test à $T = 0{,}7$ : 10,89 bits.
- Bigrammes, vocabulaire à $T = 0{,}7$ : 11 690, 6 967, 3 430, 1 267, 420, 173 (g5), 79, 34, 21 (g8), 12 (g10), 2 (g20).

Extraits de texte (40 premiers mots d'un flux) :

```text
[T = 1, génération 1]   and england's king , i with thy head , and pluck aufidius : and , and show the princess by the watch ? …
[T = 1, génération 30]  ? are , my pretext to him for edward : rise , after son aumerle . angelo a fowl , appear'd , so must i now , peevish …
[T = 0,7, génération 1] and do i am i would they gone , and put in the blood , and stain'd this is not sell with his , forgive ; …
[T = 0,7, génération 30] and , and , and , and , and , and , and , and , and , and , and , …
```

Figures produites dans `figures/` : `gaussienne.png`, `ngrammes.png` ; résultats chiffrés dans `figures/resultats.json` (lus par les figures du site, `site/scripts/figures/ch37_*.py`).

## Ce qu'il faut observer

- **La queue part en premier.** Pour la gaussienne comme pour les mots, ce sont les événements rares qui disparaissent d'abord : un mot vu une seule fois a 37 % de chances de ne pas être tiré à la génération suivante, et un mot non tiré a une probabilité nulle pour toujours. La queue $\lvert x \rvert > 2$ d'une gaussienne (4,6 % des tirages) tombe à zéro quand $\hat\sigma$ s'écrase. Sur la courbe rang-fréquence, les mille mots les plus fréquents ne bougent pas : c'est la longue traîne qui est coupée. Un modèle qui imite bien la moyenne peut ainsi oublier tout ce qui est rare (dialectes, cas limites, minorités) sans que ses performances « moyennes » ne bougent beaucoup.
- **Ce n'est pas un défaut d'estimateur.** Corriger le biais de l'écart-type rend juste l'espérance de chaque étape ; la médiane, elle, dérive quand même, parce que $\log \hat\sigma^2$ fait une marche aléatoire de pente moyenne $\approx -1/(n-1)$. Le calcul à la main ($e^{-1000/(2 \times 99)} \approx 0{,}006$) prédit la simulation (0,008). La seule cause est l'échantillonnage fini ; plus d'échantillons par génération ne fait que ralentir.
- **La température accélère tout.** Échantillonner à $T = 0{,}7$, réglage courant des LLM pour produire un texte « propre », coupe déjà la queue à chaque tirage : en 20 générations, le modèle n'écrit plus que « and , and , ». Des données synthétiques produites avec un décodage conservateur sont moins diverses que le modèle lui-même.
- **Accumuler suffit, ici.** Garder les vraies données et ajouter les synthétiques au lieu de remplacer maintient $\hat\sigma$ à 0,97 et l'entropie croisée de test à 8,55 bits (le résultat de Gerstgrasser et al., 2024) ; le vocabulaire, lui, reste intact par construction, puisqu'on n'enlève jamais de compte. Garder 10 % de vraies données neuves stabilise aussi la gaussienne (0,94) ; pour les mots, il stabilise le vocabulaire vers 60 % : un passage de 22 500 mots ne contient pas les mots rares des autres passages, qui restent perdus jusqu'à ce qu'un passage les ramène.
- **Le modèle se dégrade sur le vrai texte.** L'entropie croisée sur du texte réel qu'il n'a jamais vu monte de 8,51 à 9,51 bits par mot (deux fois moins probable par mot, en moyenne), alors que l'entropie de ce qu'il génère baisse : il devient plus sûr de lui et plus loin de la réalité.

## Questions de réflexion

1. Pour $n = 100$, combien de générations faut-il, d'après la formule, pour que l'écart-type médian (estimateur du maximum de vraisemblance) soit divisé par 2 ? Comparez avec la génération 72 mesurée. Et pour $n = 1\,000$ ?
2. Pourquoi un mot qui disparaît d'un modèle d'unigrammes ou de bigrammes ne peut-il jamais revenir ? Quelle modification du modèle (lissage, mélange avec un autre modèle) l'empêcherait, et pourquoi cela ne résout-il pas tout ?
3. AlphaFold 2 a été réentraîné sur les plus sûres de ses propres prédictions, faites pour environ 350 000 séquences (« auto-distillation »), et y a gagné en précision. Qu'est-ce qui distingue cette situation de la boucle de l'exercice ? (Indice : regardez ce qui est filtré et ce qui est gardé.)
4. Sur le Web, la part de texte écrit par des modèles augmente, et les corpus d'entraînement suivants en contiennent. Lequel des trois régimes de l'exercice ressemble le plus à cette situation ? Quelles précautions un constructeur de jeu de données peut-il prendre (chapitre 28) ?

## Pour aller plus loin

- Remplacer l'échantillonnage à température par un échantillonnage *top-k* ou *top-p* (chapitre 28) dans `table_echantillonnage` : lequel coupe la queue le plus vite ?
- Passer aux trigrammes (paires → triplets) : le vocabulaire disparaît-il plus vite ou moins vite ?
- Dans le régime « 10 % de réel », tirer 10 % de **phrases** au hasard dans tout le corpus au lieu d'un passage contigu : le vocabulaire se stabilise-t-il plus haut ?
- Articles : Shumailov, Shumaylov, Zhao, Gal, Papernot et Anderson, *The Curse of Recursion: Training on Generated Data Makes Models Forget*, 2023, https://arxiv.org/abs/2305.17493 (sections 4.1 « Discrete distributions with exact approximation » et 4.2 « Single dimensional Gaussian ») ; version publiée : Shumailov, Shumaylov, Zhao, Papernot, Anderson et Gal, *AI models collapse when trained on recursively generated data*, Nature 631:755-759, 2024, https://doi.org/10.1038/s41586-024-07566-y ; Gerstgrasser et al., *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data*, 2024, https://arxiv.org/abs/2404.01413 ; Alemohammad et al., *Self-Consuming Generative Models Go MAD*, 2023, https://arxiv.org/abs/2307.01850 (la même boucle sur des modèles d'images).
