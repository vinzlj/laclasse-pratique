# 44 · Vraisemblance, Bayes et estimateurs, par la simulation

| | |
|---|---|
| **Chapitre lié** | [44. Inférence : Bayes, vraisemblance, MAP, estimation](https://laclasse.poneyponey.fr/maths/p7-probas-ia/44-inference/), sections 1 et 2 (vraisemblance), 3 (conjugaison Bêta-Bernoulli), 5 (biais, effondrement), 6 (intervalle de confiance, bootstrap) et 7 (Hoeffding) |
| **Commande** | `make nb NUM=44` (éditeur), `make nb NUM=44 MODE=run` (complet), `make nb NUM=44 MODE=smoke` (moins de 30 s) |
| **Durée mesurée** | run complet : 5 s, dont 2 s de calcul ; smoke : 4 s (machine de travail, 27/10/2026) |
| **Travail estimé** | 25 min |
| **Graine** | 42 (`seed_everything(42)`, variable `SEED` du Makefile) |
| **Données** | aucune : tirages pseudo-aléatoires et dix temps de réponse écrits dans le code, rien n’est téléchargé |

## Objectif

Retrouver par le calcul les résultats du chapitre 44 : le maximum de la log-vraisemblance d’une Bernoulli est la fréquence $\frac{k}{n}$ ; une loi $\text{Bêta}$ mise à jour clic après clic rejoint la fréquence observée ; diviser par $n$ sous-estime la variance d’un facteur $\frac{n - 1}{n}$ ; l’intervalle $\bar x \pm 1{,}96\,\frac{\sigma}{\sqrt n}$ contient la vraie moyenne 95 fois sur 100 ; le bootstrap donne un intervalle sans formule ; la borne de Hoeffding est respectée (et large) ; une gaussienne réestimée génération après génération s’effondre.

## Ce que vous allez faire

1. Lire la cellule de configuration : elle seule change entre mode complet et mode smoke (nombre d’échantillons, de rééchantillonnages, de chaînes).
2. Lire la vérification sympy : $\ell'(p) = 0$ en $\frac{k}{n}$, $\int_0^1 p^7 (1 - p)^3 \, \mathrm{d}p = \frac{1}{1\,320}$, moyenne de $\text{Bêta}(5 \,;\, 5)$ égale à $\frac{1}{2}$, $\gamma \approx 0{,}136$ pour $N = 100$ et $\delta = 0{,}05$.
3. Cliquer sur « Lancer le calcul » dans l’éditeur marimo, puis lire les trois graphiques.

## Résultat attendu

Run complet (seuils du smoke entre parenthèses) :

- 60 succès sur 200 lancers : maximum de la grille en **0,300** $= \frac{k}{n}$ (écart toléré 0,001).
- $\text{Bêta}(2 \,;\, 18)$ puis 500 affichages (147 clics) : moyenne a posteriori **0,2865**, pour une fréquence 0,294 ; aire de la densité 1,000.
- $n = 5$, $\sigma^2 = 1$, 200 000 échantillons : $\mathbb{E}[\hat\sigma^2] \approx$ **0,801** pour 0,8 (écart toléré 0,004 ; 0,01 en smoke), $\mathbb{E}[s^2] \approx$ **1,002**.
- Couverture de $\bar x \pm 1{,}96 \times \frac{50}{5}$ ($n = 25$) : **0,950** (écart toléré 0,004 ; 0,01).
- Bootstrap des dix temps de réponse (moyenne 139,8 ms, médiane 122,5 ms), 4 000 rééchantillonnages : moyenne dans **[111,6 ; 180,5]**, médiane dans **[105,0 ; 145,0]** ; la formule $\bar x \pm 1{,}96\,\frac{s}{\sqrt n}$ donne [100,8 ; 178,8].
- Erreurs de taux 0,2 sur $N = 100$ exemples : écart supérieur à 0,1 dans **0,8 %** des cas, sous la borne de Hoeffding $2e^{-2} \approx 0{,}271$.
- Effondrement, $n = 100$, 300 générations : $\sigma$ médian **0,044** en divisant par $n$ (prévu environ $e^{-3} \approx 0{,}050$), **0,216** en divisant par $n - 1$ (prévu $e^{-300/198} \approx 0{,}22$), à un facteur 1,6 près.

Le notebook s’arrête par `SystemExit` si un seuil n’est pas atteint. Figure : `figures/vraisemblance.png`.

## Ce qu’il faut observer

- La moyenne a posteriori part de 0,1 (l’a priori), monte vite, puis colle à la fréquence observée : l’a priori pèse comme 20 affichages fictifs, et s’efface quand les vrais affichages se comptent par centaines.
- L’intervalle bootstrap de la moyenne n’est pas symétrique autour de 139,8 : la valeur aberrante 310 étire sa borne haute. Celui de la médiane est plus étroit.
- La borne de Hoeffding est vraie mais très prudente ici (0,271 contre 0,008 observé) : elle ne suppose rien d’autre que des valeurs entre 0 et 1.

## Questions de réflexion

1. Remplacez l’a priori $\text{Bêta}(2 \,;\, 18)$ par $\text{Bêta}(20 \,;\, 180)$ : même moyenne, force 200. Combien d’affichages faut-il pour que la moyenne a posteriori dépasse 0,25 ?
2. Recommencez le calcul du biais avec $n = 50$ : quel rapport attendez-vous ? Pourquoi le biais compte-t-il surtout pour les petits échantillons ?
3. Dans l’effondrement, ajoutez à chaque génération 10 % de données tirées de la vraie loi $\mathcal{N}(0 \,;\, 1)$ : que devient $\sigma$ ?
