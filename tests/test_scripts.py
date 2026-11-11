"""Tests rapides de math_commun, du verrou et du contrôle des dossiers (sans réseau)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
SCRIPTS = RACINE / "scripts"


def charger(nom: str):
    spec = importlib.util.spec_from_file_location(nom, SCRIPTS / f"{nom}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[nom] = module
    spec.loader.exec_module(module)
    return module


def test_math_commun(tmp_path, monkeypatch):
    import math_commun as mc

    assert mc.is_smoke(["notebook.py", "--smoke"]) and not mc.is_smoke(["notebook.py"])
    monkeypatch.setenv("MATH_SMOKE", "1")
    assert mc.is_smoke([])
    monkeypatch.setenv("SEED", "7")
    assert mc.seed_everything() == 7
    import numpy as np

    a = np.random.rand()
    mc.seed_everything(7)
    assert np.random.rand() == a
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mc.style_commun()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    chemins = mc.sauvegarder_figure(fig, "essai", tmp_path / "figures", formats=("png", "svg"))
    assert [c.name for c in chemins] == ["essai.png", "essai.svg"] and all(c.stat().st_size > 0 for c in chemins)
    plt.close(fig)
    assert (mc.NOTEBOOKS / "_gabarit" / "notebook.py").is_file()


def test_run_locked(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "IA_GPU_LOCK": str(tmp_path / "verrou")}
    sh = str(SCRIPTS / "run_locked.sh")
    aide = subprocess.run([sh, "--help"], check=False, capture_output=True, env=env, text=True)
    assert aide.returncode == 0 and "Usage" in aide.stdout and "set -u" not in aide.stdout
    r = subprocess.run([sh, "--", "sh", "-c", "test -d \"$IA_GPU_LOCK\" && exit 3"],
                       check=False, capture_output=True, env=env)
    assert r.returncode == 3  # le verrou existe pendant la commande, le code est propagé
    assert not (tmp_path / "verrou").exists()  # et il est libéré après


def test_check_exercices_depot():
    m = charger("check_exercices")
    erreurs, n = m.verifier(RACINE)
    assert erreurs == [] and n > 0


def test_check_exercices_incoherent(tmp_path):
    m = charger("check_exercices")
    (tmp_path / "notebooks/maths").mkdir(parents=True)
    d = tmp_path / "exercices/ia/06-essai"
    d.mkdir(parents=True)
    (d / "README.md").write_text(f"# Essai\n\n[Chapitre]({m.SITE}/ia/p02-x/07-y/)\n\n## Objectif\n", encoding="utf-8")
    erreurs, n = m.verifier(tmp_path)
    assert n == 1
    texte = "\n".join(erreurs)
    assert "ni notebook.py" in texte and "chapitre 07, attendu 06" in texte and "Résultat attendu" in texte
