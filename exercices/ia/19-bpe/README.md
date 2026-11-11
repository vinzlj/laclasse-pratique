# Exercice 19 : BPE à la main, puis comparé à tiktoken

| | |
|---|---|
| **Chapitre lié** | [19. Tokenisation](https://laclasse.poneyponey.fr/ia/p07-transformer/19-tokenisation/) |
| **Exécution** | locale, CPU (Python pur ; aucun GPU, aucun PyTorch) |
| **Commande** | `make ex NUM=19` (éditeur), `make ex NUM=19 MODE=run` (complet), `MODE=smoke` (< 30 s) |
| **Durée mesurée** | run complet : 20 à 34 s selon la charge de la machine (processus entier, attente du verrou exclue), dont 7 à 10 s pour la version naïve de BPE ; smoke : 5 s (13 s au premier lancement, téléchargements compris) (M1 Pro, CPU, 15/10/2026) |
| **Travail estimé** | ~1 h 30 |
| **Seed** | 42 (`seed_everything(42)`), par principe : BPE est déterministe (à égalité d'effectif, la plus petite paire d'identifiants gagne), les résultats sont identiques d'un lancement à l'autre ; seuls les temps varient |
| **Données** | *Le tour du monde en quatre-vingts jours* de Jules Verne (1873), en français (Gutenberg n° 800, 462 Ko) et dans la traduction anglaise de George M. Towle (Gutenberg n° 103, 404 Ko), téléchargés une fois dans `data/gutenberg/` ; fichiers de rangs de `tiktoken` (GPT-2, GPT-4, GPT-4o, ≈ 7 Mo) téléchargés une fois dans `data/tiktoken/` |

## Objectif

Programmer l'algorithme BPE sur octets (celui de GPT-2) en Python pur, l'accélérer sans changer son résultat, vérifier qu'avec les fusions de GPT-2 notre encodeur donne exactement les mêmes tokens que `tiktoken`, puis mesurer ce que coûte une langue absente du corpus d'apprentissage du tokenizer, en particulier le français.

## Ce que vous allez faire

1. **Corpus.** Le même roman en français et en anglais ; les 20 derniers pour cent de chaque livre servent de texte de test, jamais vu par BPE. Les apostrophes et guillemets anglais typographiques (`’ “ ”`, 3 octets en UTF-8) sont ramenés à l'ASCII, comme dans le texte français ; les guillemets `« »` restent.
2. **Trois découpages naïfs.** Caractères (103 symboles en français), octets UTF-8 (256, rien n'est inconnu, mais « é » prend 2 octets), mots entiers (chaque signe de ponctuation comptant comme un mot : 8 610 mots et signes distincts en français, et 6,0 % des mots et signes du test absents du vocabulaire : *malheureusement*, *quotidiennement*, *transatlantiques*…).
3. **BPE naïf.** Pré-découpage par l'expression régulière de GPT-2 (un mot avec l'espace qui le précède, un nombre, une ponctuation), puis boucle : compter toutes les paires de symboles voisins, fusionner la plus fréquente (identifiant 256 + k), recommencer. 256 fusions sur 100 Ko de français.
4. **Deux optimisations du comptage des paires.** (a) Compter les morceaux distincts une seule fois, pondérés par leur nombre d'occurrences ; (b) tenir à jour les effectifs des paires et un index « paire → morceaux qui la contiennent », pour ne retoucher que les morceaux concernés par chaque fusion. Les trois versions doivent produire exactement la même liste de fusions.
5. **Apprendre 2 000 fusions** sur chaque livre (vocabulaire de 2 256 symboles), encoder et décoder le texte de test, tracer la compression en fonction du nombre de fusions, et suivre une phrase inédite fusion après fusion.
6. **Reproduire tiktoken.** Avec les rangs publiés pour GPT-2 (`tiktoken.get_encoding("gpt2")._mergeable_ranks`), notre encodeur doit sortir les mêmes identifiants que `tiktoken`, token par token, sur 40 000 caractères de test.
7. **Comparer les langues.** Tokens par mot et surcoût du français pour GPT-2 (`r50k_base`), GPT-4 (`cl100k_base`), GPT-4o (`o200k_base`) et nos deux BPE.
8. **Compter les lettres.** Regarder ce que le modèle voit de « strawberry ».

Ce qui est optimisé : **rien**. BPE n'a pas de fonction de perte ; chaque fusion est un choix glouton fondé sur un comptage.

## Résultat attendu

En mode complet, le notebook vérifie huit seuils et s'arrête avec une erreur si l'un d'eux n'est pas atteint. Valeurs obtenues le 15/10/2026 :

| Seuil | Valeur obtenue |
|---|---|
| les trois versions de BPE donnent les mêmes 256 fusions | oui |
| `decoder(encoder(x)) == x` (test français, test anglais, texte avec €, emoji, chinois, œ) | oui |
| notre encodeur avec les rangs de GPT-2 = `tiktoken` (12 759 tokens sur 40 001 caractères ; clé `comparaison_tiktoken`) | identique |
| version incrémentale ≥ 10 fois plus rapide que la naïve (256 fusions, 100 Ko) | × 30 à × 47 selon la charge (7 à 10 s contre 0,22 à 0,24 s ; version dédoublonnée : 2,3 à 2,5 s) |
| BPE-fr, 2 000 fusions : ≤ 350 tokens pour 1 000 caractères du test français | 339 |
| BPE-fr fait au moins 15 % de tokens de moins que BPE-en sur le livre français | 141 767 contre 224 864 (− 37 %) |
| GPT-2 : à longueur égale en octets, le français coûte ≥ 1,3 fois plus de tokens | × 1,45 |
| GPT-4o réduit ce surcoût par rapport à GPT-2 | × 1,11 |

**Compression** (tokens pour 1 000 caractères du texte de test) :

| Fusions | 0 | 10 | 50 | 100 | 200 | 500 | 1 000 | 2 000 |
|---|---|---|---|---|---|---|---|---|
| BPE-fr sur le français | 1 027 | 898 | 703 | 614 | 527 | 437 | 384 | 339 |
| BPE-en sur le français | 1 027 | 971 | 807 | 736 | 660 | 598 | 566 | 537 |
| BPE-en sur l'anglais | 1 001 | 865 | 681 | 591 | 507 | 418 | 365 | 322 |
| BPE-fr sur l'anglais | 1 001 | 945 | 793 | 728 | 660 | 600 | 561 | 529 |

Repères : mots entiers (ponctuation comprise), ≈ 226 (français) et 218 (anglais) tokens pour 1 000 caractères, mais 6,0 % et 5,3 % de mots inconnus.

**Langues** (livres entiers ; le texte français fait 433 028 octets et 67 429 mots, la traduction 369 582 octets et 63 334 mots, soit un rapport de 1,17 en octets) :

| Tokenizer | Tokens (fr) | Tokens (en) | Tokens/mot fr | Tokens/mot en | fr / en | Surcoût à longueur égale |
|---|---|---|---|---|---|---|
| GPT-2 (`r50k_base`, 50 257) | 161 584 | 95 356 | 2,40 | 1,51 | 1,69 | × 1,45 |
| GPT-4 (`cl100k_base`, 100 277) | 130 802 | 90 157 | 1,94 | 1,42 | 1,45 | × 1,24 |
| GPT-4o (`o200k_base`, 200 019) | 115 913 | 89 458 | 1,72 | 1,41 | 1,30 | × 1,11 |
| BPE-fr de l'exercice (2 256) | 141 767 | 193 620 | 2,10 | 3,06 | 0,73 | × 0,62 |
| BPE-en de l'exercice (2 256) | 224 864 | 117 190 | 3,33 | 1,85 | 1,92 | × 1,64 |

**Une phrase inédite**, « Les voyageurs arrivèrent enfin à Londres. », avec le BPE-fr : 43 tokens à 0 fusion (« è » et « à » font chacun 2 octets), 30 à 50 fusions, 21 à 200, 10 à 1 000, 8 à 2 000 : `Les | ␣voyageurs | ␣arriv | èrent | ␣enfin | ␣à | ␣Londres | .`

**Premières fusions.** Français : `␣d`, `es`, `␣l`, `é`, `en`, `ai`, `on`, `␣s`… (la 4ᵉ fusion recolle les deux octets de « é ») ; anglais : `␣t`, `he`, `␣a`, `␣the`, `in`, `␣w`…

**Lettres.** « ␣strawberry » est **un seul token** pour GPT-2 comme pour GPT-4o (identifiant 101830 dans `o200k_base`) ; « ␣anticonstitutionnellement » en fait 6 avec GPT-2 et 4 avec GPT-4o (`␣anticon | stitution | nel | lement`) ; « ␣1234567 » devient `␣123 | 45 | 67` (GPT-2) et `␣ | 123 | 456 | 7` (GPT-4o).

Figures produites : `figures/compression.png` ; résultats bruts pour le site : `figures/resultats.json` (lu par `site/scripts/figures/ch19_*.py`).

## Ce qu'il faut observer

- **Le vocabulaire d'octets est fermé.** Le texte avec €, emoji et caractères chinois s'encode et se décode sans perte, alors qu'aucun de ces caractères n'apparaît dans le corpus : au pire, ils restent en octets (`⟨F0⟩⟨9F⟩⟨9A⟩⟨82⟩` pour 🚂). Plus de mot inconnu, contrairement au découpage en mots (6 % du test).
- **Les fusions suivent les fréquences de la langue.** En français, `␣d`, `es`, `␣l`, puis « é » recollé dès la 4ᵉ fusion ; en anglais, `␣t`, `he`, `␣the`. Les noms propres fréquents du livre ont vite leur token : dans le BPE français, `␣Fogg` dès la 104ᵉ fusion, `␣Passepartout` dès la 168ᵉ, `␣Phileas` dès la 227ᵉ, `␣Londres` à la 727ᵉ (clé `rangs_noms_propres` de `resultats.json`). Un vocabulaire appris reflète son corpus.
- **Rendements décroissants.** Les 200 premières fusions divisent le nombre de tokens par deux ; les 1 800 suivantes ne gagnent plus qu'un tiers. C'est pourquoi les vocabulaires des LLM font des dizaines ou des centaines de milliers de tokens, appris sur des corpus énormes.
- **La langue du corpus décide.** Un BPE appris sur l'anglais coûte 58 % de tokens en plus sur le français (537 contre 339 pour 1 000 caractères), et réciproquement. GPT-2, appris sur un corpus anglais, fait payer le français 1,45 fois plus cher à texte égal ; GPT-4o, avec un vocabulaire quatre fois plus grand, 1,11 fois : plus de tokens, c'est plus de calcul, une fenêtre de contexte qui se remplit plus vite et, pour une API, une facture plus élevée.
- **Optimiser sans changer le résultat.** Les trois versions donnent la même liste de fusions ; seule la manière de compter change. La version naïve relit tout le corpus à chaque fusion ; la version incrémentale ne touche que les morceaux qui contiennent la paire fusionnée.
- **Un LLM ne voit pas les lettres.** Pour GPT-4o, « ␣strawberry » est l'entier 101830 : savoir qu'il contient trois « r » est une connaissance apprise, pas une lecture.

## Questions de réflexion

1. Pourquoi le pré-découpage par expression régulière (le mot avec l'espace qui le précède) est-il utile ? Que se passerait-il sans lui sur un texte où « chat. », « chat, » et « chat! » sont fréquents ?
2. Le BPE-fr n'a toujours pas de token pour « arrivèrent » après 2 000 fusions, mais en a un pour « ␣Londres ». Pourquoi ? Qu'est-ce que cela dit des mots rares d'une langue très fléchie ?
3. Si on remplace les octets par des caractères Unicode comme alphabet de départ, que gagne-t-on et que perd-on (pensez au chinois et aux emoji) ?
4. GPT-4o découpe « 1234567 » en `123 | 456 | 7`. Quel problème cela pose-t-il pour l'addition de grands nombres ? Comment pourrait-on découper les nombres autrement ?

## Pour aller plus loin

- Passer `fusions` à 5 000 dans la cellule de configuration : la courbe de compression continue-t-elle de descendre ? Le vocabulaire commence-t-il à apprendre des mots entiers rares, voire des morceaux de phrases ?
- Remplacer le pré-découpage de GPT-2 par celui de GPT-4 (`tiktoken.get_encoding("cl100k_base")._pat_str`, qui groupe les chiffres par trois au plus) et comparer les fusions.
- Unigram (Kudo, 2018) part d'un très grand vocabulaire et le réduit en maximisant la vraisemblance du corpus par l'algorithme EM : contrairement à BPE et à WordPiece, qui choisissent une fusion à la fois, elle optimise un objectif global. La bibliothèque `sentencepiece` (non installée dans le projet) l'implémente.
