#!/usr/bin/env bash
# Avvia / ferma / controlla un SuperLink per GPU.
#
#   bash superlinks.sh start          # ne avvia uno per ogni GPU in GPUS
#   bash superlinks.sh status         # run attive e processi di ogni SuperLink
#   bash superlinks.sh wait-idle G    # aspetta che il SuperLink della GPU G
#                                     # non abbia piu' run ne' processi vivi
#   bash superlinks.sh stop
#
# PERCHE' UNO PER GPU
# [B] la GPU si sceglie all'AVVIO DEL SUPERLINK. flwr run non esegue nulla:
# consegna la run al SuperLink, che la fa girare in un sottoprocesso figlio.
# Quel figlio eredita l'ambiente del SuperLink, non quello della shell da cui
# lanci flwr run -- per questo CUDA_VISIBLE_DEVICES davanti a flwr run non ha
# alcun effetto. Un SuperLink per GPU, avviato con la sua variabile, e ogni
# shard che parla col proprio.
#
# [B] DATABASE SEPARATI E NUOVI. La coda delle run vive in state.db. Con un
# database condiviso i tre SuperLink si contenderebbero la stessa coda. E la
# coda SOPRAVVIVE: se il SuperLink muore, le run lanciate nel frattempo
# restano dentro e partono tutte insieme al riavvio successivo (11/09). Per
# questo ogni avvio parte da un database vuoto (il vecchio resta in .prev).

set -u
cd "$(dirname "$0")"

# di default le GPU di experiment.toml, cosi' start/status/stop a mano usano
# le stesse porte di run.sh
GPUS=${GPUS:-$(python3 -c "import tomllib; print(*tomllib.load(open('experiment.toml', 'rb'))['campaign']['gpus'])")}
PORT_BASE=${PORT_BASE:-39101}
STATE_DIR=${STATE_DIR:-$HOME/.flwr/sweep}
CONF=${CONF:-$HOME/.flwr/config.toml}
PROJECT=${PROJECT:-$PWD}
WAIT_MAX=${WAIT_MAX:-300}   # wait-idle: dopo quanti secondi ferma i residui

read -ra GPU_ARR <<< "$GPUS"

name_of() { echo "gpu$1"; }
port_of() { local i=0; for g in "${GPU_ARR[@]}"; do
    [ "$g" = "$1" ] && { echo $((PORT_BASE + i)); return; }; i=$((i+1)); done; }
sl_pid() { pgrep -f "control-api-address 127.0.0.1:$(port_of "$1")" | head -1; }

descendants() { local c; for c in $(pgrep -P "$1"); do echo "$c"; descendants "$c"; done; }

# processi flwr-simulation discendenti del SuperLink della GPU $1: uno per
# run in corso. Solo i SUOI, non quelli di altri SuperLink o di altri utenti.
sim_procs() {
  local root p; root=$(sl_pid "$1"); [ -n "$root" ] || return 0
  for p in $(descendants "$root"); do
    ps -o args= -p "$p" 2>/dev/null | grep -q "flwr-simulation" && echo "$p"
  done
}

# id delle run non finite (pending, starting, running) sul SuperLink della
# GPU $1, una per riga; "?" se flwr ls non risponde
active_runs() {
  timeout 30 flwr ls "$(name_of "$1")" --format json 2>/dev/null | python3 -c '
import json, sys
s = sys.stdin.read()
try:
    runs = json.loads(s[s.index("{"):])["runs"]
except Exception:
    print("?"); sys.exit()
for r in runs:
    if not str(r.get("status", "")).lower().startswith("finished"):
        print(r["run-id"], r.get("status", ""), sep=":")'
}
# solo gli id, per flwr stop
run_ids() { echo "$1" | cut -d: -f1; }

count() { [ "$1" = "?" ] && echo "?" || echo "$1" | grep -c .; }

case "${1:-status}" in

start)
  mkdir -p "$STATE_DIR"
  # --- voci in config.toml, aggiunte una sola volta
  # [B] insecure = true e' obbligatorio: il SuperLink parte con --insecure,
  # ma flwr senza quella riga usa TLS (default) e non si collega. Sul vecchio
  # server le voci erano scritte a mano; sul nuovo le crea questo script.
  for gpu in "${GPU_ARR[@]}"; do
    n=$(name_of "$gpu"); p=$(port_of "$gpu")
    if ! grep -q "^\[superlink\.${n}\]" "$CONF" 2>/dev/null; then
      printf '\n[superlink.%s]\naddress = "127.0.0.1:%s"\ninsecure = true\n' "$n" "$p" >> "$CONF"
      echo "  config.toml: aggiunta [superlink.${n}] -> 127.0.0.1:${p}"
    else
      sect=$(awk -v s="[superlink.${n}]" '$0 == s {f = 1; next} /^\[/ {f = 0} f' "$CONF")
      echo "$sect" | grep -q "127.0.0.1:${p}" \
        || echo "  ATTENZIONE: [superlink.${n}] in ${CONF} non punta a 127.0.0.1:${p}"
      echo "$sect" | grep -Eq "^insecure *= *true" \
        || echo "  ATTENZIONE: [superlink.${n}] in ${CONF} senza 'insecure = true'"
    fi
  done

  for gpu in "${GPU_ARR[@]}"; do
    n=$(name_of "$gpu"); p=$(port_of "$gpu")
    if [ -n "$(sl_pid "$gpu")" ]; then
      echo "  ${n}: gia' attivo sulla porta ${p}"
      continue
    fi
    db="${STATE_DIR}/state_${n}.db"
    if [ -f "$db" ]; then
      mv -f "$db" "${db}.prev"; rm -f "${db}-wal" "${db}-shm"
      echo "  ${n}: coda vecchia spostata in ${db}.prev"
    fi
    ( cd "$PROJECT" && \
      export CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 && \
      nohup flower-superlink --insecure --simulation --isolation subprocess \
        --control-api-address "127.0.0.1:${p}" \
        --serverappio-api-address "127.0.0.1:0" \
        --database "$db" \
        > "/tmp/superlink_${n}.log" 2>&1 & )
    sleep 4
    if [ -n "$(sl_pid "$gpu")" ]; then
      echo "  ${n}: avviato su GPU ${gpu}, porta ${p}"
    else
      echo "  ${n}: AVVIO FALLITO -- vedi /tmp/superlink_${n}.log"
      tail -5 "/tmp/superlink_${n}.log"
    fi
  done
  ;;

status)
  # [B] prima qui c'era "task avviati" = righe "Started task" nel log: un
  # contatore cumulativo, che per costruzione cresceva a ogni run e non
  # poteva mai scendere. Ora: run non finite (flwr ls) e processi di
  # simulazione vivi. A riposo devono essere 0, durante una run 1.
  for gpu in "${GPU_ARR[@]}"; do
    n=$(name_of "$gpu"); p=$(port_of "$gpu")
    pid=$(sl_pid "$gpu")
    if [ -n "$pid" ]; then
      echo "  ${n}  GPU ${gpu}  porta ${p}  pid ${pid}  run attive: $(count "$(active_runs "$gpu")")" \
           " processi di simulazione: $(count "$(sim_procs "$gpu")")"
    else
      echo "  ${n}  GPU ${gpu}  porta ${p}  NON ATTIVO"
    fi
  done
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
  ;;

wait-idle)
  # [B] UNA RUN ALLA VOLTA PER GPU. Il CSV di una run arriva prima che la run
  # sia davvero finita (valutazione finale, Ray che si spegne): lanciando
  # subito la successiva le due si sovrapponevano sulla GPU e le run attive
  # crescevano a ogni run. _shard.sh chiama questo prima di ogni run; dopo
  # WAIT_MAX secondi i residui vengono fermati.
  gpu="${2:?uso: bash superlinks.sh wait-idle <gpu>}"
  n=$(name_of "$gpu"); t0=$(date +%s); t=0; last=-60
  [ -n "$(sl_pid "$gpu")" ] || exit 0
  while :; do
    runs=$(active_runs "$gpu"); procs=$(sim_procs "$gpu")
    [ "$runs" = "?" ] && runs=""          # flwr ls muto: contano i processi
    [ -z "$runs" ] && [ -z "$procs" ] && break
    t=$(( $(date +%s) - t0 ))             # tempo vero, flwr ls non e' istantaneo
    if [ $(( t - last )) -ge 60 ]; then   # un segno di vita al minuto
      echo "  ${n}: aspetto la chiusura della run precedente (${t}s/${WAIT_MAX}s):" \
           "run [$(echo $runs)] processi [$(echo $procs)]"
      last=$t
    fi
    if [ "$t" -ge "$WAIT_MAX" ]; then
      echo "  ${n}: dopo ${t}s la fermo"
      for id in $(run_ids "$runs"); do flwr stop "$id" "$n" >/dev/null 2>&1; done
      sleep 5
      for pr in $(sim_procs "$gpu"); do kill "$pr" 2>/dev/null; done
      sleep 5
      for pr in $(sim_procs "$gpu"); do kill -9 "$pr" 2>/dev/null; done
      break
    fi
    sleep 5
  done
  [ "$t" -gt 0 ] && echo "  ${n}: chiusa dopo ${t}s"
  exit 0
  ;;

stop)
  for gpu in "${GPU_ARR[@]}"; do
    n=$(name_of "$gpu")
    pid=$(sl_pid "$gpu")
    [ -n "$pid" ] || continue
    # i figli non muoiono col padre: vanno chiusi a mano (solo i suoi)
    procs=$(sim_procs "$gpu")
    kill "$pid"; echo "  ${n}: fermato (pid ${pid})"
    sleep 3
    for pr in $procs; do kill "$pr" 2>/dev/null && echo "  ${n}: simulazione residua ${pr} chiusa"; done
  done
  ;;

*)
  echo "uso: bash superlinks.sh {start|status|wait-idle <gpu>|stop}"
  exit 1
  ;;
esac
