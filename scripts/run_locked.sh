#!/usr/bin/env bash
# Verrou GPU global : exécute une commande quand aucun autre entraînement ne tourne.
#
# Un portable n'a qu'un GPU : deux entraînements simultanés faussent les durées
# mesurées et peuvent saturer la mémoire. `make ex`, `make nb` et les cibles smoke
# passent par ce script.
#
# Principe : `mkdir` est atomique ; le premier qui crée /tmp/ia-gpu.lock détient
# le verrou, les autres attendent. Le détenteur écrit son PID et sa commande dans
# le dossier, puis le PID de la commande lancée (fichier « enfant »). Le verrou n'est
# orphelin, donc récupéré, que si les DEUX processus ont disparu : après un kill -9 de
# ce script, l'entraînement qui tourne encore garde le GPU jusqu'à sa fin.
#
# Usage : scripts/run_locked.sh [--attente-max S] [--chrono FICHIER] -- commande arg1 arg2…
#         scripts/run_locked.sh --help
# --chrono FICHIER : écrit dans FICHIER la durée de la commande seule, en secondes
#                    (attente du verrou exclue) ; sert à `make ex MODE=smoke`.
# Variables : IA_GPU_LOCK (chemin du verrou, défaut /tmp/ia-gpu.lock ; les notebooks
#               des maths prennent /tmp/laclasse-nb.lock)
#             IA_LOCK_LIBELLE (nom affiché dans les messages, défaut « verrou GPU »)
#             IA_GPU_LOCK_DESACTIVE=1 (exécute sans verrou, déconseillé)

set -u

VERROU="${IA_GPU_LOCK:-/tmp/ia-gpu.lock}"
LIBELLE="${IA_LOCK_LIBELLE:-verrou GPU}"
ATTENTE_MAX=7200   # 2 h : au-delà, on abandonne (code 75)
INTERVALLE=2
CHRONO=""

aide() {
  sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) aide; exit 0 ;;
    --attente-max) ATTENTE_MAX="$2"; shift 2 ;;
    --chrono) CHRONO="$2"; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done

if [ $# -eq 0 ]; then
  echo "run_locked.sh : aucune commande à exécuter (voir --help)" >&2
  exit 2
fi

# Exécution de la commande, chronométrée si --chrono (sans exec, pour mesurer).
lancer_simple() {
  local t0 c
  t0=$(date +%s)
  "$@"; c=$?
  [ -z "$CHRONO" ] || echo $(( $(date +%s) - t0 )) > "$CHRONO"
  return "$c"
}

if [ "${IA_GPU_LOCK_DESACTIVE:-0}" = 1 ]; then
  lancer_simple "$@"; exit $?
fi

# Verrou réentrant : un `make smoke` qui détient déjà le verrou peut relancer
# run_locked.sh pour chaque exercice sans se bloquer lui-même.
if [ "${IA_GPU_LOCK_DETENU:-}" = "$VERROU" ]; then
  lancer_simple "$@"; exit $?
fi

detenu=0
enfant=""
liberer() {
  if [ "$detenu" = 1 ]; then
    rm -rf "$VERROU"
    detenu=0
  fi
}
# Ctrl-C, kill, fermeture du terminal : la commande en cours est arrêtée (SIGTERM, puis
# SIGKILL au bout de 10 s), puis le verrou est libéré. Le verrou n'est jamais rendu
# tant que la commande tourne.
interrompre() {
  local code=$1 n=0
  if [ -n "$enfant" ] && kill -0 "$enfant" 2>/dev/null; then
    kill -TERM "$enfant" 2>/dev/null
    while kill -0 "$enfant" 2>/dev/null && [ $n -lt 50 ]; do sleep 0.2; n=$((n + 1)); done
    kill -0 "$enfant" 2>/dev/null && kill -KILL "$enfant" 2>/dev/null
    wait "$enfant" 2>/dev/null
  fi
  liberer
  exit "$code"
}
trap liberer EXIT
trap 'interrompre 130' INT
trap 'interrompre 143' TERM
trap 'interrompre 129' HUP

debut=$(date +%s)
annonce=0
# Âge d'un dossier en secondes (stat BSD sur macOS, GNU sous Linux).
age() {
  local m
  m=$(stat -f %m "$1" 2>/dev/null || stat -c %Y "$1" 2>/dev/null) || { echo 0; return; }
  echo $(( $(date +%s) - m ))
}

# Un verrou est orphelin si son PID n'existe plus ET que sa commande (PID « enfant »)
# non plus, ou s'il n'a pas de PID depuis plus de 10 s (détenteur tué entre le mkdir
# et l'écriture du PID).
orphelin() {
  local p e
  p=$(cat "$VERROU/pid" 2>/dev/null || true)
  e=$(cat "$VERROU/enfant" 2>/dev/null || true)
  if [ -n "$p" ]; then
    ! kill -0 "$p" 2>/dev/null || return 1
    [ -z "$e" ] || ! kill -0 "$e" 2>/dev/null
  else
    [ -d "$VERROU" ] && [ "$(age "$VERROU")" -gt 10 ]
  fi
}

# Récupération d'un verrou orphelin, sous un second verrou (« .recup »). Sans lui, deux
# processus qui voient le même PID mort feraient chacun un rm -rf, et le second
# pourrait effacer le verrou que le premier vient de reprendre. Sous .recup, l'état
# est relu : un verrou neuf (PID vivant, ou pas encore de PID) n'est jamais effacé.
recuperer() {
  local recup="$VERROU.recup"
  if ! mkdir "$recup" 2>/dev/null; then
    # .recup laissé par un processus tué pendant la récupération (fenêtre très courte)
    [ "$(age "$recup")" -gt 30 ] && rmdir "$recup" 2>/dev/null
    return 1
  fi
  if orphelin; then
    echo "[$LIBELLE] verrou orphelin (PID $(cat "$VERROU/pid" 2>/dev/null || echo '?') disparu$( [ -f "$VERROU/enfant" ] && echo ", commande $(cat "$VERROU/enfant") terminée")) : récupération" >&2
    rm -rf "$VERROU"
  fi
  rmdir "$recup" 2>/dev/null
}

while ! mkdir "$VERROU" 2>/dev/null; do
  pid=$(cat "$VERROU/pid" 2>/dev/null || true)
  if orphelin; then
    recuperer || sleep 0.2
    continue
  fi
  if [ "$annonce" = 0 ]; then
    cmd=$(cat "$VERROU/commande" 2>/dev/null || echo "?")
    echo "[$LIBELLE] occupé par PID ${pid:-?} ($cmd) : attente…" >&2
    annonce=1
  fi
  if [ $(( $(date +%s) - debut )) -ge "$ATTENTE_MAX" ]; then
    echo "[$LIBELLE] abandon après ${ATTENTE_MAX} s d'attente" >&2
    exit 75
  fi
  sleep "$INTERVALLE"
done
detenu=1
echo $$ > "$VERROU/pid"
echo "$*" > "$VERROU/commande"
date '+%Y-%m-%d %H:%M:%S' > "$VERROU/depuis"
if [ "$annonce" = 1 ]; then
  echo "[$LIBELLE] obtenu après $(( $(date +%s) - debut )) s" >&2
fi

export IA_GPU_LOCK_DETENU="$VERROU"
# La commande tourne en arrière-plan pour que son PID soit inscrit dans le verrou
# (fichier « enfant ») : si ce script est tué par kill -9, le verrou reste pris tant
# qu'elle tourne. « <&0 » garde l'entrée standard (sinon /dev/null en arrière-plan).
# Pas d'exec : le shell doit survivre pour libérer le verrou à la fin.
t0=$(date +%s)
"$@" <&0 &
enfant=$!
echo "$enfant" > "$VERROU/enfant"
# Le premier wait rend le code de sortie (même si bash a déjà récupéré l'enfant) ;
# la boucle couvre un wait interrompu par un signal.
wait "$enfant"; code=$?
while kill -0 "$enfant" 2>/dev/null; do
  wait "$enfant"; code=$?
done
[ -z "$CHRONO" ] || echo $(( $(date +%s) - t0 )) > "$CHRONO"
exit "$code"
