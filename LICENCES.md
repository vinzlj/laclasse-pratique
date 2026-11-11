# Licences

Ce dépôt mélange du code, des textes et des données. Chacun a sa licence.

| Contenu | Licence | Fichiers |
|---|---|---|
| **Code** : notebooks marimo et Jupyter, paquets communs, scripts, tests | [MIT](LICENSE) | `exercices/**/notebook.py`, `exercices/**/notebook.ipynb`, `notebooks/**/notebook.py`, `src/`, `math_commun/`, `scripts/`, `tests/`, `Makefile`, `pyproject.toml` |
| **Textes** : énoncés, README des exercices et des notebooks, documentation | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.fr) | `*.md`, dont le corpus de l’exercice 32 (`exercices/ia/32-mini-rag/corpus/`, extrait des chapitres du site), et les données écrites pour les exercices (`questions.json`, `taches.json`, `juge.json`, `documents.json`, documentation fictive de l’exercice 33) |
| **Résultats et figures produits par les exercices** (runs complets, pour comparer avec les vôtres) | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.fr) | `exercices/**/figures/`, `notebooks/**/figures/` |
| **Jeux de données et modèles de tiers** (MNIST, CIFAR-10, TinyShakespeare, modèles Hugging Face, Ollama…) | **Leur propre licence** | Pas dans le dépôt : téléchargés au premier lancement dans `data/` ; licence indiquée dans le README de chaque exercice qui les utilise |

Attribution demandée pour les textes et figures : « Les maths, du collège à l’IA » ou « Mécanismes de l’IA », vinzlj, CC BY-NC-SA 4.0.

CC BY-NC-SA 4.0 : reprise et modification libres, avec attribution, sans usage commercial (vente, publicité, formation payante…) sans accord écrit de l’auteur, et partage des versions modifiées sous la même licence.

Certaines figures de résultats montrent des images tirées de jeux de données de tiers (chiffres de MNIST, images de CIFAR-10, Oxford-IIIT Pet…) ou générées par des modèles entraînés sur eux : ces images restent soumises aux conditions de leur jeu de données d’origine.
