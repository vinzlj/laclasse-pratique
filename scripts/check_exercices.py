#!/usr/bin/env python3
"""Cohérence des dossiers d’exercices et de notebooks (sans réseau).

Pour chaque dossier exercices/ia/NN…/ et notebooks/maths/NN-…/ (hors dossiers « _ ») :
  1. README.md présent, avec le notebook attendu : notebook.py (exercice local, notebook
     des maths) ou un fichier .ipynb (exercice IA à lancer dans le cloud) ;
  2. le README renvoie à la page de son chapitre sur le site
     (https://laclasse.poneyponey.fr/ia/pNN-…/NN-…/ ou /maths/pN-…/NN-…/), et le numéro du
     chapitre est celui du dossier ;
  3. le README contient les rubriques obligatoires : Objectif, Résultat attendu, Ce qu’il
     faut observer, Questions de réflexion.

Usage : uv run python scripts/check_exercices.py [--racine .]
Code de sortie : 0 si tout est cohérent, 1 sinon.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

SITE = "https://laclasse.poneyponey.fr"
RUBRIQUES = {
    "objectif": "Objectif",
    "resultat attendu": "Résultat attendu",
    "ce qu'il faut observer": "Ce qu’il faut observer",
    "questions de reflexion": "Questions de réflexion",
}
PARTIES = {
    # dossier → (parcours du site, motif du lien vers le chapitre)
    "exercices/ia": ("ia", re.compile(re.escape(SITE) + r"/ia/p\d{2}-[a-z0-9-]+/(\d{2})-[a-z0-9-]+/")),
    "notebooks/maths": ("maths", re.compile(re.escape(SITE) + r"/maths/p\d-[a-z0-9-]+/(\d{2})-[a-z0-9-]+/")),
}


def _normaliser(texte: str) -> str:
    texte = unicodedata.normalize("NFKD", texte)
    return "".join(c for c in texte if not unicodedata.combining(c)).lower().replace("’", "'")


def verifier(racine: Path) -> tuple[list[str], int]:
    erreurs: list[str] = []
    n = 0
    for rel, (parcours, motif) in PARTIES.items():
        base = racine / rel
        if not base.is_dir():
            erreurs.append(f"{rel}/ : dossier absent")
            continue
        for d in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
            n += 1
            nom = f"{rel}/{d.name}"
            readme = d / "README.md"
            if not readme.is_file():
                erreurs.append(f"{nom}/ : README.md manquant")
                continue
            if not (d / "notebook.py").is_file() and not list(d.glob("*.ipynb")):
                erreurs.append(f"{nom}/ : ni notebook.py ni notebook .ipynb")
            if parcours == "maths" and not (d / "notebook.py").is_file():
                erreurs.append(f"{nom}/ : notebook.py manquant")
            texte = readme.read_text(encoding="utf-8")
            liens = motif.findall(texte)
            if not liens:
                erreurs.append(f"{nom}/README.md : aucun lien vers la page du chapitre ({SITE}/{parcours}/…)")
            elif liens[0] != d.name[:2]:
                erreurs.append(f"{nom}/README.md : lien vers le chapitre {liens[0]}, attendu {d.name[:2]}")
            titres = {_normaliser(t).strip() for t in re.findall(r"^#{2,3}\s+(.+)$", texte, re.MULTILINE)}
            for cle, libelle in RUBRIQUES.items():
                if not any(t.startswith(cle) for t in titres):
                    erreurs.append(f"{nom}/README.md : rubrique « {libelle} » absente")
    return erreurs, n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vérifie les dossiers d’exercices et de notebooks.")
    parser.add_argument("--racine", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="racine du dépôt (défaut : dossier parent de scripts/)")
    args = parser.parse_args(argv)
    erreurs, n = verifier(args.racine)
    for e in erreurs:
        print(f"ERREUR : {e}")
    print(f"check_exercices : {n} dossier(s), {len(erreurs)} erreur(s).")
    return 1 if erreurs else 0


if __name__ == "__main__":
    sys.exit(main())
