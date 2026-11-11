# Exercices Python du parcours IA et notebooks marimo des maths.
#
#   make help                        liste des cibles
#   make install                     environnement Python complet (uv sync --locked)
#   make install-maths               seulement ce qu’il faut aux notebooks des maths
#   make ex NUM=13a [MODE=…]         exercice du parcours IA (edit | run | smoke)
#   make nb NUM=12 [MODE=…]          notebook du parcours de maths (edit | run | smoke)
#   make smoke                       tous les exercices locaux et tous les notebooks en mode smoke
#   make test | check                tests Python ; lock, tests et cohérence des dossiers
#
# Attention : /usr/bin/make de macOS est GNU Make 3.81 (pas de .ONESHELL) : les recettes
# multilignes sont enchaînées par des « \ ».

SHELL := /bin/bash
.DEFAULT_GOAL := help

UV   ?= uv
NUM  ?=
MODE ?= edit
# Limite du mode smoke (secondes) : au-delà, avertissement (exercices) ou échec (notebooks).
SMOKE_MAX ?= 30

# --- Environnement des exercices IA (exporté vers toutes les recettes) ---
export PYTORCH_ENABLE_MPS_FALLBACK := 1
# DATA_DIR peut viser un autre dossier déjà rempli : make ex NUM=13a DATA_DIR=/autre/data
export DATA_DIR           ?= $(CURDIR)/data
export HF_HOME            := $(DATA_DIR)/hf
export TIKTOKEN_CACHE_DIR := $(DATA_DIR)/tiktoken
export SEED               ?= 42

# Dossier de l’exercice IA : nom exact (_gabarit), puis « NUM-… » (13a-…), puis
# « NUMx-… » (make ex NUM=13 ouvre 13a-…).
EX_DIR := $(if $(NUM),$(firstword $(wildcard exercices/ia/$(NUM)) \
                                  $(wildcard exercices/ia/$(NUM)-*) \
                                  $(wildcard exercices/ia/$(NUM)[a-z]-*)))
# Dossier du notebook des maths : notebooks/maths/NUM-… (NUM sur deux chiffres) ou _gabarit.
NB_DIR := $(if $(NUM),$(firstword $(wildcard notebooks/maths/$(NUM)-* notebooks/maths/$(NUM))))

# Verrous : un exercice IA à la fois (GPU), un notebook des maths à la fois (durées mesurées).
RUN_LOCKED := scripts/run_locked.sh
NB_LOCKED  := IA_GPU_LOCK=/tmp/laclasse-nb.lock IA_LOCK_LIBELLE="verrou des notebooks" scripts/run_locked.sh
# Notebooks des maths : dépendances communes seulement (numpy, sympy, matplotlib, marimo),
# paquet math_commun/ trouvé par PYTHONPATH.
NB_RUN := env PYTHONPATH=$(CURDIR)$${PYTHONPATH:+:$$PYTHONPATH} $(UV) run --no-default-groups

.PHONY: help install install-maths lock-check test check check-exercices ex nb smoke smoke-ia smoke-nb

help: ## Affiche cette aide
	@echo "Exercices du parcours IA et notebooks des maths : cibles disponibles"
	@echo
	@grep -hE '^[a-zA-Z0-9_.-]+:.*## ' $(MAKEFILE_LIST) | sort -u \
	  | awk 'BEGIN {FS = ":.*## "} {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Variables : NUM=13a (exercice IA) ou NUM=12 (notebook)  MODE=edit|run|smoke  SEED=42"

# ------------------------------------------------------------------ Installation

install: ## Environnement Python complet (exercices IA et notebooks), selon uv.lock
	$(UV) sync --locked

install-maths: ## Environnement léger : numpy, sympy, matplotlib, marimo (notebooks des maths)
	$(UV) sync --locked --no-default-groups --group dev

lock-check: ## Vérifie que uv.lock correspond à pyproject.toml
	$(UV) lock --check

# ------------------------------------------------------------------ Contrôles

test: ## Tests Python (ia_commun, math_commun, cohérence des dossiers)
	$(UV) run pytest

check-exercices: ## README, notebook et lien vers le chapitre de chaque dossier
	$(UV) run --no-default-groups python scripts/check_exercices.py

check: lock-check test check-exercices ## Tout vérifier (lock, tests, cohérence)

# ------------------------------------------------------------------ Exercices du parcours IA

ex: ## make ex NUM=13a [MODE=edit|run|smoke] : exercice du parcours IA
	@test -n "$(NUM)" || { echo "Préciser NUM, ex. : make ex NUM=13a"; exit 1; }
	@test -n "$(EX_DIR)" || { echo "Exercice $(NUM) introuvable dans exercices/ia/"; exit 1; }
	@test -f "$(EX_DIR)/notebook.py" || { echo "$(EX_DIR) est un exercice cloud : voir son README.md (Colab ou Kaggle)"; exit 1; }
	@case "$(MODE)" in \
	  edit)  echo "Éditeur marimo (pas de verrou GPU : fermez l’éditeur avant un make smoke)"; \
	         $(UV) run marimo edit "$(EX_DIR)/notebook.py" ;; \
	  run)   MPLBACKEND=Agg $(RUN_LOCKED) -- $(UV) run python "$(EX_DIR)/notebook.py" ;; \
	  smoke) chrono=$$(mktemp); \
	         MPLBACKEND=Agg $(RUN_LOCKED) --chrono "$$chrono" -- $(UV) run python "$(EX_DIR)/notebook.py" --smoke \
	           || { c=$$?; rm -f "$$chrono"; exit $$c; }; \
	         duree=$$(cat "$$chrono"); rm -f "$$chrono"; \
	         echo "Smoke $(EX_DIR) : $${duree} s (attente du verrou exclue)"; \
	         [ $$duree -le $(SMOKE_MAX) ] || echo "ATTENTION : plus de $(SMOKE_MAX) s" ;; \
	  *)     echo "MODE inconnu : $(MODE) (edit, run ou smoke)"; exit 1 ;; \
	esac

smoke-ia: ## Tous les exercices IA locaux en mode smoke, en série (télécharge les données au besoin)
	@echecs=(); lents=(); n=0; \
	for d in exercices/ia/[0-9]*/; do \
	  d=$${d%/}; [ -f "$$d/notebook.py" ] || continue; \
	  n=$$((n + 1)); chrono=$$(mktemp); echo "=== $$d"; \
	  if MPLBACKEND=Agg $(RUN_LOCKED) --chrono "$$chrono" -- $(UV) run python "$$d/notebook.py" --smoke; then \
	    duree=$$(cat "$$chrono"); echo "    ok en $${duree} s"; \
	    [ $$duree -le $(SMOKE_MAX) ] || lents+=("$$d ($${duree} s)"); \
	  else echecs+=("$$d"); fi; \
	  rm -f "$$chrono"; \
	done; \
	echo "--- $$n exercice(s) local(aux) testé(s)"; \
	[ $${#lents[@]} -eq 0 ] || printf 'Plus de $(SMOKE_MAX) s : %s\n' "$${lents[@]}"; \
	[ $${#echecs[@]} -eq 0 ] || { printf 'ÉCHEC : %s\n' "$${echecs[@]}"; exit 1; }

# ------------------------------------------------------------------ Notebooks des maths

nb: ## make nb NUM=12 [MODE=edit|run|smoke] : notebook du parcours de maths
	@test -n "$(NUM)" || { echo "Préciser NUM, ex. : make nb NUM=12"; exit 1; }
	@test -n "$(NB_DIR)" && test -f "$(NB_DIR)/notebook.py" || { echo "Notebook $(NUM) introuvable : notebooks/maths/$(NUM)-<nom>/notebook.py absent"; exit 1; }
	@case "$(MODE)" in \
	  edit)  $(NB_RUN) marimo edit "$(NB_DIR)/notebook.py" ;; \
	  run)   MPLBACKEND=Agg $(NB_LOCKED) -- $(NB_RUN) python "$(NB_DIR)/notebook.py" ;; \
	  smoke) chrono=$$(mktemp); \
	         MPLBACKEND=Agg $(NB_LOCKED) --chrono "$$chrono" -- $(NB_RUN) python "$(NB_DIR)/notebook.py" --smoke \
	           || { c=$$?; rm -f "$$chrono"; exit $$c; }; \
	         duree=$$(cat "$$chrono"); rm -f "$$chrono"; \
	         echo "Smoke $(NB_DIR) : $${duree} s (attente du verrou exclue)"; \
	         [ $$duree -le $(SMOKE_MAX) ] || { echo "ÉCHEC : plus de $(SMOKE_MAX) s"; exit 1; } ;; \
	  *)     echo "MODE inconnu : $(MODE) (edit, run ou smoke)"; exit 1 ;; \
	esac

smoke-nb: ## Tous les notebooks des maths en mode smoke, en série
	@n=0; echecs=(); \
	for d in notebooks/maths/[0-9][0-9]-*/; do \
	  d=$${d%/}; [ -f "$$d/notebook.py" ] || continue; n=$$((n + 1)); echo "=== $$d"; \
	  MPLBACKEND=Agg $(NB_LOCKED) -- $(NB_RUN) python "$$d/notebook.py" --smoke || echecs+=("$$d"); \
	done; \
	echo "--- $$n notebook(s) testé(s)"; \
	[ $${#echecs[@]} -eq 0 ] || { printf 'ÉCHEC : %s\n' "$${echecs[@]}"; exit 1; }

smoke: smoke-nb smoke-ia ## Tous les notebooks des maths, puis tous les exercices IA locaux, en mode smoke
