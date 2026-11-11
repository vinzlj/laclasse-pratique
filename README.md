# laclasse-pratique

Les exercices Python et les notebooks qui accompagnent deux parcours du site « laclasse » :

- **[Mécanismes de l’IA](https://laclasse.poneyponey.fr/ia/)** : une cinquantaine d’exercices, du perceptron aux petits GPT, dans `exercices/ia/` ;
- **[Les maths, du collège à l’IA](https://laclasse.poneyponey.fr/maths/)** : 26 notebooks où le code éclaire une notion du cours, dans `notebooks/maths/`.

Les cours eux-mêmes sont sur le site ; ce dépôt ne contient que la pratique. Chaque dossier a un `README.md` qui donne l’objectif, le chapitre lié, la commande, la durée mesurée, le résultat attendu et des questions de réflexion.

## Installer

Il vous faut un terminal, `git`, `make` et [uv](https://docs.astral.sh/uv/), le gestionnaire de projets Python qui installe Python et les versions exactes des bibliothèques (celles du fichier `uv.lock`, qui ont servi à mesurer les résultats des README).

```sh
# uv, une seule fois (macOS ou Linux ; sous Windows, voir la documentation d’uv)
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/vinzlj/laclasse-pratique
cd laclasse-pratique
make install          # tout : exercices IA (PyTorch, scikit-learn…) et notebooks des maths
make install-maths    # ou seulement les notebooks des maths : numpy, sympy, matplotlib, marimo
```

Une fois installé, l’environnement complet occupe environ 1,3 Go (PyTorch et les bibliothèques de traitement du langage), l’environnement des maths environ 350 Mo. Python 3.13 est choisi par uv (`.python-version`) : inutile de l’installer vous-même.

Les exercices ont été écrits et mesurés sur un Mac Apple Silicon (GPU MPS). Ils tournent aussi sur processeur, sous Linux ou Windows, en plus de temps. Ceux qui demandent un vrai GPU s’ouvrent dans Google Colab ou Kaggle (gratuits), en suivant leur `README.md`.

## Lancer un notebook des maths

```sh
make nb NUM=12              # ouvre le notebook du chapitre 12 dans l’éditeur marimo (navigateur)
make nb NUM=12 MODE=run     # l’exécute d’un bloc, comme un script
make nb NUM=12 MODE=smoke   # version courte, moins de 30 s, pour vérifier que tout marche
```

Les notebooks sont des fichiers [marimo](https://docs.marimo.io/) : des cellules réactives dans un fichier `.py`. Liste et conventions : [notebooks/README.md](notebooks/README.md).

## Lancer un exercice du parcours IA

```sh
make ex NUM=06              # ouvre l’exercice 06 dans l’éditeur marimo
make ex NUM=13a MODE=run    # l’exécute en entier, comme un script
make ex NUM=13a MODE=smoke  # version réduite, en moins de 30 s
```

Les données (MNIST, CIFAR-10, TinyShakespeare, modèles Hugging Face) sont téléchargées au premier besoin dans `data/`, qui n’est jamais versionné. Quelques exercices demandent [Ollama](https://ollama.com/) (leur README le dit). Les exercices « cloud » (`13b`, `20b`, `23b`, `24b`, `25b`, `30a`, `30b`, `31b`) n’ont pas de `notebook.py` mais un `notebook.ipynb` à importer dans Colab ou Kaggle.

Le dossier `figures/` de chaque exercice contient les figures et résultats du run complet : comparez-les aux vôtres.

## Toutes les commandes

| Commande | Effet |
|---|---|
| `make help` | liste des cibles |
| `make smoke-nb` | tous les notebooks des maths en version courte, l’un après l’autre |
| `make smoke-ia` | tous les exercices IA locaux en version courte (télécharge les données au premier passage) |
| `make smoke` | les deux |
| `make test` | tests Python des paquets communs et des scripts |
| `make check` | `uv.lock` à jour, tests, cohérence des dossiers (README, notebook, lien vers le chapitre) |

Les modes `run` et `smoke` passent par `scripts/run_locked.sh` : un seul entraînement à la fois, pour que deux calculs ne se disputent pas le GPU et que les durées mesurées restent justes.

## Organisation

```
├── exercices/ia/NN-nom/     un exercice du parcours IA : README.md, notebook.py (ou .ipynb), figures/
├── notebooks/maths/NN-nom/  un notebook du parcours de maths : README.md, notebook.py, figures/
├── src/ia_commun/           utilitaires des exercices IA : graine, GPU, données, tracés, mini-GPT
├── math_commun/             utilitaires des notebooks des maths : graine, mode smoke, figures
├── scripts/                 verrou des calculs, contrôle des dossiers
├── tests/                   tests pytest
└── pyproject.toml, uv.lock  dépendances (groupes ia, rl, nlp, misc, dev)
```

Les gabarits `exercices/ia/_gabarit/` et `notebooks/maths/_gabarit/` montrent la structure commune d’un exercice.

## Licences

Code sous licence [MIT](LICENSE) ; énoncés, README et autres textes sous [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.fr) ; jeux de données et modèles de tiers sous leur propre licence. Détail : [LICENCES.md](LICENCES.md).
