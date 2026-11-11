# Exercice 12 : rendre les réseaux profonds entraînables

| | |
|---|---|
| **Chapitre lié** | [12. Rendre les réseaux profonds entraînables](https://laclasse.poneyponey.fr/ia/p06-deep-learning/12-reseaux-profonds/) |
| **Exécution** | locale, MPS (NumPy pour les parties A et B, PyTorch pour C et D ; tourne aussi sur CPU, plus lentement) |
| **Commande** | `make ex NUM=12` (éditeur), `make ex NUM=12 MODE=run` (complet), `MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 1 min 03 s à 1 min 25 s (trois mesures le 14/10/2026 ; parties C et D : 55 s et 9 s au run du 15/10), dont 50 à 60 s pour la partie C (21 entraînements) et 8 à 16 s pour la partie D ; smoke : 6 à 8 s (M1 Pro, MPS, 14 et 15/10/2026, attente du verrou exclue) |
| **Travail estimé** | ~1 h 15 |
| **Seed** | 42 (`seed_everything(42)`) ; partie C : graines 42, 43 et 44, résultats moyennés ; découpage des données fixé par `torch.Generator().manual_seed(42)` |
| **Données** | MNIST via `ia_commun.donnees.mnist_tenseurs()`, cache `data/mnist/` (≈ 55 Mo, téléchargé au premier lancement) ; parties A et B : tableaux aléatoires générés |

## Objectif

Voir, chiffres à l'appui, les deux obstacles qui empêchaient d'entraîner des réseaux profonds (le signal qui s'éteint ou explose d'une couche à l'autre, et le surapprentissage) et mesurer ce que changent l'initialisation Xavier/He, le dropout, la normalisation par lot (BatchNorm), la pénalité L2, le label smoothing et mixup.

## Ce que vous allez faire

1. **A. Initialisation (NumPy).** Propager un lot de 512 vecteurs gaussiens dans un réseau de 30 couches de largeur 256 (sans biais), pour cinq couples activation / initialisation : tanh avec σ = 0,01 (« petite ») ou σ = 1 (« grande »), tanh + Xavier (σ² = 1/n), ReLU + Xavier, ReLU + He (σ² = 2/n). Mesurer l'écart type des activations couche par couche, puis rétropropager un gradient aléatoire et mesurer ‖∂L/∂z‖ (le gradient qui redescend) et ‖∂L/∂W‖.
2. **B. Les mécanismes à la main (NumPy), comparés à PyTorch.**
   - dropout inversé : proportion de zéros, facteur 1/(1 − p) des survivants, espérance conservée, identité en mode `eval()` ;
   - BatchNorm : statistiques du lot à l'entraînement, moyennes glissantes à l'inférence, comparées à `nn.BatchNorm1d` (y compris le détail : la moyenne glissante prend la variance **non biaisée**) ;
   - label smoothing (ε = 0,1) : cible lissée et perte, comparées à `F.cross_entropy(..., label_smoothing=0.1)` ;
   - mixup (α = 0,2) : mélange de deux images et de leurs cibles, distribution de λ.
3. **C. Régularisation sur MNIST (PyTorch).** Le MLP 784-256-256-10, entraîné sur **5 000 images seulement** (pour qu'il surapprenne en quelques époques), 30 époques, Adam (lr 10⁻³), lots de 128. Sept configurations, trois graines chacune : sans régularisation, dropout (p = 0,5 après chaque couche cachée), BatchNorm, dropout + BatchNorm, pénalité L2 (λ = 5·10⁻⁴ sur les matrices de poids), label smoothing (ε = 0,1), mixup (α = 0,2). Validation : 10 000 autres images du jeu d'entraînement ; test : les 10 000 images officielles. On mesure aussi la précision de test du modèle de la meilleure époque de validation (arrêt précoce).
4. **Pourquoi la perte de validation remonte, et le dropout à l'inférence.** Sur le modèle sans régularisation (graine 42), comparer la meilleure époque de validation et la dernière : nombre d'erreurs, confiance et perte moyennes sur ces erreurs. Sur le modèle « dropout », mesurer la précision en mode `eval()`, avec un seul sous-réseau tiré au hasard (mode `train()`) et avec la moyenne des probabilités de 100 sous-réseaux.
5. **D. Profondeur (PyTorch).** Un MLP de **20 couches cachées** de 256 neurones, SGD avec momentum 0,9 (lr 0,01), 5 époques sur les mêmes 5 000 images, six variantes : tanh + σ = 0,01, tanh + Xavier, ReLU + Xavier, ReLU + initialisation par défaut de PyTorch, ReLU + He, ReLU + défaut de PyTorch + BatchNorm. On note la norme du gradient de chaque couche au premier pas, la courbe de perte et la précision de validation.

## Résultat attendu

En mode complet, le notebook vérifie dix-huit seuils et s'arrête avec une erreur si l'un d'eux n'est pas atteint. Valeurs obtenues le 14/10/2026 (M1 Pro, MPS, seed 42 ; trois runs complets, valeurs identiques à l'affichage), puis le 15/10/2026 après le passage de l'écart type entre graines à la version d'échantillon (n − 1) ; le « ± » de la partie C est cet écart type, les moyennes n'ont pas changé :

**A. Initialisation (30 couches, largeur 256)**

| Variante | σ activations couche 1 | couche 10 | couche 30 | ‖∂L/∂z‖ couche 30 | ‖∂L/∂z‖ couche 1 |
|---|---|---|---|---|---|
| tanh + petite (σ = 0,01) | 0,157 | 1,1·10⁻⁸ | **1,4·10⁻²⁴** | 0,71 | **5,9·10⁻²⁴** |
| tanh + grande (σ = 1) | 0,975 | 0,974 | 0,975 (saturé) | 0,128 | **5,8·10¹²** |
| tanh + Xavier | 0,629 | 0,231 | 0,133 | 0,697 | 0,112 |
| ReLU + Xavier | 0,589 | 0,022 | **3,7·10⁻⁵** | 0,497 | **2,5·10⁻⁵** |
| ReLU + He | 0,833 | 0,714 | 1,22 | 0,497 | 0,578 |

Seuils : tanh petite σ(couche 30) < 10⁻¹⁰ et ‖∂L/∂z‖(couche 1) < 10⁻¹⁰ ; tanh grande : rapport couche 1 / couche 30 > 10⁶ ; ReLU + Xavier σ(couche 30) < 10⁻³ ; ReLU + He σ(couche 30) > 0,1 et rapport des gradients entre 0,1 et 10.

**B. Mécanismes à la main**

| Vérification | Seuil | Obtenu |
|---|---|---|
| Dropout p = 0,5 : proportion de zéros (NumPy / PyTorch) | entre 49 et 51 % | 49,98 % / 49,97 % |
| Dropout : espérance conservée | à 1 % près | écart 0,03 % ; facteur des survivants chez PyTorch : 2,000 |
| Dropout en `eval()` | identité | identité |
| BatchNorm NumPy = PyTorch (entraînement / inférence) | écart < 10⁻⁴ | 1,4·10⁻⁶ / 1,2·10⁻⁶ |
| Label smoothing NumPy = PyTorch | écart < 10⁻⁵ | 4,8·10⁻⁷ |

Autres valeurs : cible lissée (ε = 0,1, K = 10) = 0,91 sur la bonne classe et 0,01 ailleurs ; perte minimale atteignable avec cette cible : 0,500 (son entropie), et non 0 ; avec α = 0,2, λ tombe sous 0,1 ou au-dessus de 0,9 dans 67 % des tirages (la plupart des mélanges restent proches d'une seule image).

**C. Régularisation sur MNIST (5 000 images, 30 époques, moyenne de 3 graines)**

| Configuration | Précision train | Perte val finale (minimum, époque) | Précision val | **Test** | Test avec arrêt précoce |
|---|---|---|---|---|---|
| sans régularisation | 100,00 % | **0,308** (0,248, époques 9 à 11) | 93,88 % | **94,13 % ± 0,13** | 93,57 % |
| dropout | 99,83 % | 0,226 (0,215, ép. 19 à 23) | 94,29 % | 94,83 % ± 0,03 | 94,81 % |
| BatchNorm | 100,00 % | 0,204 (0,190, ép. 6 à 11) | 94,67 % | 95,16 % ± 0,03 | 95,16 % |
| dropout + BatchNorm | 99,92 % | 0,200 (0,195, ép. 21 à 25) | 94,41 % | 95,01 % ± 0,06 | 94,92 % |
| pénalité L2 | 99,97 % | 0,217 (0,217, ép. 22 à 30) | 94,06 % | 94,40 % ± 0,03 | 94,51 % |
| label smoothing | 100,00 % | 0,259 (0,254) | **95,58 %** | **95,77 % ± 0,12** | 95,80 % |
| mixup | 99,95 % | 0,209 (0,195) | 94,49 % | 95,09 % ± 0,24 | 95,13 % |

Seuils : sans régularisation, perte de validation finale ≥ 1,15 × son minimum (obtenu : 0,308 / 0,248 = 1,24 sur la courbe moyenne) ; dropout, BatchNorm, les deux et label smoothing : test ≥ base + 0,4 point (obtenu : + 0,7 à + 1,6 point).

**Pourquoi la perte de validation remonte (modèle sans régularisation, graine 42, validation)**

| Modèle | Perte val | Erreurs | Perte moyenne par erreur | Confiance moyenne sur les erreurs | Part de la perte due aux erreurs |
|---|---|---|---|---|---|
| meilleure époque de validation (arrêt précoce) | 0,245 | 666 | 3,05 | 72 % | 83 % |
| dernière époque (30) | 0,316 | **625** | **4,82** | **82 %** | **95 %** |

Seuil : à la fin, au plus autant d'erreurs qu'à l'arrêt précoce **et** une confiance moyenne plus haute sur ces erreurs.

**Dropout à l'inférence (modèle « dropout », graine 42, validation)**

| Mode | Précision |
|---|---|
| `eval()` (réseau complet) | 94,15 % |
| un sous-réseau tiré au hasard (`train()`) | 92,43 % ; deux passes successives sont en désaccord sur **6,2 %** des images |
| moyenne des probabilités de 100 sous-réseaux | 94,12 %, même prédiction que `eval()` sur **99,5 %** des images (seuil ≥ 98 %) |

**D. MLP de 20 couches cachées (5 époques, SGD + momentum)**

| Variante | ‖∂L/∂W‖ couche 1 au 1er pas | couche 20 | Perte 1er pas → fin | Précision val |
|---|---|---|---|---|
| tanh + petite (σ = 0,01) | 6,7·10⁻¹⁶ | 5,5·10⁻¹⁶ | 2,301 → 2,301 | **11,29 %** |
| tanh + Xavier | 0,30 | 0,18 | 2,310 → 0,374 | 81,26 % |
| ReLU + Xavier | 4,7·10⁻⁴ | 2,4·10⁻⁴ | 2,301 → 2,299 | **11,29 %** |
| ReLU + défaut PyTorch | 1,8·10⁻⁸ | 1,2·10⁻² | 2,300 → 2,301 | **11,29 %** |
| ReLU + He | 0,35 | 0,16 | 2,319 → 0,553 | 71,17 % |
| ReLU + défaut PyTorch + BatchNorm | 68,7 | 1,00 | 2,383 → 0,588 | **87,50 %** |

Seuils : les trois premières variantes « mortes » sous 20 % ; tanh + Xavier, ReLU + He et BatchNorm au-dessus de 60 % ; écart type de l'initialisation par défaut de PyTorch × √n = 0,578 (attendu 1/√3 ≈ 0,577).

Figures produites : `figures/initialisation.png`, `figures/regularisation.png`, `figures/profondeur.png` ; résultats bruts : `figures/resultats.json` (lu par les scripts des figures du site, `site/scripts/figures/ch12_*.py`).

## Ce qu'il faut observer

- **Le signal se multiplie à chaque couche.** Avec σ = 0,01, chaque couche multiplie l'écart type par ≈ σ·√n = 0,16 : après 30 couches il reste 10⁻²⁴, et le gradient qui redescend s'éteint de la même façon (0,71 en haut, 6·10⁻²⁴ en bas). Avec σ = 1, tanh sature (0,975 partout) et le gradient **explose** en descendant (0,13 → 6·10¹²). Xavier (σ² = 1/n) garde tanh à peu près stable ; pour ReLU, qui coupe la moitié des valeurs, il faut doubler (He, σ² = 2/n) : avec Xavier, ReLU perd un facteur √2 par couche (3,7·10⁻⁵ à la couche 30).
- **Un réseau profond mal initialisé n'apprend rien du tout.** Partie D : trois variantes restent à la perte ln 10 ≈ 2,303 (celle d'une réponse uniforme sur 10 classes) et répondent toujours la même classe : 11,29 % est exactement la part des « 1 » dans la validation (1 129 sur 10 000). L'initialisation **par défaut** de PyTorch (uniforme sur ±1/√n, donc σ² = 1/(3n), pensée pour des réseaux peu profonds) en fait partie : le gradient de la première couche vaut 1,8·10⁻⁸.
- **BatchNorm rend l'initialisation presque indifférente.** Même départ par défaut, mais chaque couche renormalise ses entrées : 87,5 % en 5 époques, mieux que He sans normalisation (71 %). La perte de « ReLU + He » fait un pic vers le 30ᵉ pas : un réseau profond non normalisé reste instable avec ce pas d'apprentissage.
- **Le surapprentissage se lit sur la perte de validation, pas sur la précision d'entraînement.** Sans régularisation, la précision d'entraînement atteint 100 % et la perte d'entraînement 0,002 ; la perte de validation passe par un minimum (0,248 vers l'époque 11 sur la courbe moyenne) puis remonte jusqu'à 0,308. Avec dropout, BatchNorm ou les deux, elle reste vers 0,20.
- **Perte et précision ne disent pas la même chose.** La perte de validation remonte surtout parce que le modèle devient **trop sûr de lui** sur ses erreurs, alors que sa précision continue de monter lentement : arrêter à la meilleure perte de validation donne ici 93,57 % au test, **moins** que de continuer (94,13 %). Le détail (graine 42) : entre la meilleure époque et la dernière, le nombre d'erreurs **baisse** (666 → 625) mais la confiance moyenne sur ces erreurs monte de 72 à 82 % et la perte moyenne par erreur de 3,05 à 4,82 ; à la fin, 95 % de la perte vient des 6 % d'images mal classées. L'arrêt précoce sur la perte n'est pas toujours le bon critère.
- **Label smoothing : meilleure précision, perte plus haute.** C'est la meilleure configuration au test (95,77 %), mais sa perte de validation (0,259) est plus élevée que celle des autres régularisations : la cible n'est plus « 100 % sur la bonne classe », donc le modèle n'atteint jamais une perte nulle (minimum 0,50 par exemple pour ε = 0,1). Comparer des pertes entre configurations qui n'optimisent pas la même cible est trompeur.
- **Les gains sont modestes et ne s'additionnent pas.** Dropout + BatchNorm (95,01 %) ne fait pas mieux que BatchNorm seul (95,16 %). Les écarts entre graines (± 0,03 à ± 0,24 point) sont plus petits que les gains mesurés pour dropout, BatchNorm et label smoothing (+ 0,7 à + 1,6 point), d'où l'intérêt de répéter avec plusieurs graines avant de conclure.
- **Dropout = moyenne d'un ensemble de sous-réseaux.** Un seul sous-réseau tiré au hasard fait moins bien (92,4 %) et change d'avis sur 6 % des images d'une passe à l'autre ; la moyenne de 100 sous-réseaux et le réseau complet en mode `eval()` donnent la même réponse sur 99,5 % des images. Oublier `model.eval()` à l'inférence, c'est prédire avec un seul sous-réseau tiré au sort.

## Questions de réflexion

1. Pourquoi le gradient d'une couche est-il un **produit** de termes (un par couche au-dessus d'elle) ? Qu'en déduire sur la profondeur à partir de laquelle un facteur 0,7 par couche devient un problème ?
2. Partie D : avec BatchNorm, la norme du gradient de la première couche vaut 68,7 au premier pas, alors que les poids sont minuscules. Montrez que si l'on multiplie les poids d'une couche suivie de BatchNorm par une constante c, la sortie ne change pas, et que le gradient est divisé par c. Qu'est-ce que cela implique pour la pénalité L2 sur ces poids ?
3. Le label smoothing améliore la précision mais augmente la perte de validation. Quelle quantité mesureriez-vous pour vérifier qu'il rend le modèle moins trop sûr de lui (indice : chapitre 39, calibration) ?
4. Mixup avec α = 0,2 produit surtout des mélanges très déséquilibrés (λ proche de 0 ou de 1). Que se passerait-il avec α = 1 (λ uniforme) sur un réseau aussi petit ? Essayez.

## Pour aller plus loin

- Entraîner sur les 60 000 images (`n_train=50000` pour garder la validation séparée) : que deviennent les écarts entre configurations ?
- Passer `couches_d` à 50 : He seul tient-il encore ? Et BatchNorm ?
- Remplacer Adam par SGD dans la partie C : le classement des régularisations change-t-il ?
- Placer le dropout avant la BatchNorm au lieu d'après la ReLU, et comparer.
