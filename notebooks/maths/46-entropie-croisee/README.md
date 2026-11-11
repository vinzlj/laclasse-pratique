# 46 · Entropie croisée et perplexité d’un modèle de caractères

| | |
|---|---|
| **Chapitre lié** | [46. Information I : surprise, entropie, entropie croisée, KL](https://laclasse.poneyponey.fr/maths/p7-probas-ia/46-information-entropie/), sections 3 à 6 |
| **Commande** | `make nb NUM=46` (éditeur), `make nb NUM=46 MODE=run` (complet), `make nb NUM=46 MODE=smoke` (moins de 30 s) |
| **Durée mesurée** | run complet : 2 s, dont moins de 0,01 s de calcul ; smoke : 2 s (machine de travail, 27/10/2026) |
| **Travail estimé** | 25 min |
| **Graine** | 42 (`seed_everything(42)`, variable `SEED` du Makefile) ; elle ne sert qu’au texte tiré du modèle |
| **Données** | aucune téléchargée : un court texte de 586 caractères écrit pour le notebook, dans la cellule de configuration |

## Objectif

Mesurer la perte d’un petit modèle de langage par caractères, en bits par caractère, et sa perplexité : uniforme, unigrammes (fréquences des caractères), bigrammes (caractère suivant sachant le précédent, lissés). Vérifier sur des lois de caractères que $H(p, q) = H(p) + D_{\mathrm{KL}}(p \,\|\, q)$ et que la KL n’est pas symétrique.

## Ce que vous allez faire

1. Lire la cellule de configuration : le texte, la part d’apprentissage (80 %) et les lissages $\alpha$ essayés (6 en smoke, 200 en complet).
2. Cliquer sur « Lancer le calcul » : les trois modèles sont évalués sur les 20 % de fin du texte, qu’ils n’ont pas vus.
3. Lire la décomposition de l’entropie croisée, les deux sens de la KL, puis quelques caractères tirés du modèle de bigrammes.

## Résultat attendu

- Modèle uniforme sur les 23 caractères de l’alphabet : $\log_2 23 \approx 4{,}524$ bits par caractère, perplexité 23 (exactement, à $10^{-12}$ près).
- Unigrammes : environ 3,955 bits par caractère, perplexité 15,5.
- Bigrammes au meilleur lissage : **au plus 3 bits** par caractère (mesuré : 2,495 bits, perplexité 5,64 en complet avec $\alpha \approx 0{,}089$ ; 2,515 bits en smoke) ; l’ordre bigrammes < unigrammes < uniforme est exigé.
- $H(p, q) - H(p) - D_{\mathrm{KL}}(p \,\|\, q)$ nul à $10^{-12}$ près (mesuré : $2{,}5 \times 10^{-16}$), avec $D_{\mathrm{KL}}(p \,\|\, q) \approx 0{,}150$ bit.
- Les deux sens de la KL entre la loi des caractères et la loi uniforme diffèrent d’au moins 0,05 bit (mesuré : 0,631 et 0,777 bit).
- sympy vérifie l’exemple météo du cours ($H(p) = \frac{3}{2}$, $H(p, q) = \frac{7}{4}$, KL $= \frac{1}{4}$ bit) et la perplexité 4 du modèle A de l’ouverture.

Le notebook s’arrête par `SystemExit` si un seuil n’est pas atteint. Figure : `figures/entropie-croisee.png` (perte des bigrammes selon $\alpha$, avec les deux autres modèles en repère).

## Ce qu’il faut observer

- Connaître le caractère précédent divise la perplexité par près de trois : après une espace, le modèle sait qu’un « l » ou un « c » est probable ; après « q », il a appris « u ».
- Sans lissage ($\alpha \to 0$), la perte remonte : un bigramme jamais vu à l’apprentissage reçoit une probabilité presque nulle, et sa surprise, presque infinie, pèse lourd dans la moyenne. Trop de lissage ($\alpha$ grand) rapproche le modèle de l’uniforme.
- Le texte tiré des bigrammes ressemble à du français de loin (« le », « la », « sorie ») mais n’a aucun sens : le modèle ne voit qu’un caractère en arrière.

## Questions de réflexion

1. Pourquoi évalue-t-on sur la fin du texte, et pas sur le texte d’apprentissage ? Que deviendrait la perte des bigrammes sans lissage sur le texte d’apprentissage lui-même ?
2. La perte est en bits par caractère. Que vaut-elle en nats ? Et la perplexité change-t-elle ?
3. Pourquoi $D_{\mathrm{KL}}(\text{uniforme} \,\|\, \text{caractères})$ est-elle plus grande que l’autre sens ? Pensez aux caractères rares.

## Pour aller plus loin

- Passer aux trigrammes (les deux caractères précédents) : la perte baisse-t-elle encore, avec un texte aussi court ?
- Remplacer le texte par un autre, dans une autre langue : comparer l’entropie des lois de caractères.
