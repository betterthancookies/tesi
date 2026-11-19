#!/usr/bin/env bash

set -u
cd "$(dirname "$0")"

GPUS=${GPUS:-$(python3 -c "import os, tomllib; print(*tomllib.load(open(os.environ.get('EXP_TOML') or 'experiment.toml', 'rb'))['campaign']['gpus'])")}
PORT_BASE=${PORT_BASE:-39101}
STATE_DIR=${STATE_DIR:-$HOME/.flwr/sweep}
CONF=${CONF:-$HOME/.flwr/config.toml}
PROJECT=${PROJECT:-$PWD}
WAIT_MAX=${WAIT_MAX:-300}
WAIT_ORPHAN=${WAIT_ORPHAN:-30}

read -ra GPU_ARR <<< "$GPUS"

name_of() { echo "gpu$1"; }
port_of() { local i=0; for g in "${GPU_ARR[@]}"; do
    [ "$g" = "$1" ] && { echo $((PORT_BASE + i)); return; }; i=$((i+1)); done; }
sl_pid() { pgrep -f "control-api-address 127.0.0.1:$(port_of "$1")" | head -1; }

descendants() { local c; for c in $(pgrep -P "$1"); do echo "$c"; descendants "$c"; done; }

sim_procs() {
  local root p st; root=$(sl_pid "$1"); [ -n "$root" ] || return 0
  for p in $(descendants "$root"); do
    ps -o args= -p "$p" 2>/dev/null | grep -q "flwr-simulation" || continue
    st=$(ps -o stat= -p "$p" 2>/dev/null)
    case "$st" in
      Z*) [ "${2:-}" = "zombie" ] && echo "$p" ;;
      *)  [ "${2:-}" = "zombie" ] || echo "$p" ;;
    esac
  done
}

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
run_ids() { echo "$1" | cut -d: -f1; }

count() { [ "$1" = "?" ] && echo "?" || echo "$1" | grep -c .; }

case "${1:-status}" in

start)
  mkdir -p "$STATE_DIR"
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
      exec nohup flower-superlink --insecure --simulation --isolation subprocess \
        --control-api-address "127.0.0.1:${p}" \
        --serverappio-api-address "127.0.0.1:0" \
        --database "$db" \
        > "/tmp/superlink_${n}.log" 2>&1 < /dev/null & )
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
  for gpu in "${GPU_ARR[@]}"; do
    n=$(name_of "$gpu"); p=$(port_of "$gpu")
    pid=$(sl_pid "$gpu")
    if [ -n "$pid" ]; then
      echo "  ${n}  GPU ${gpu}  porta ${p}  pid ${pid}  run attive: $(count "$(active_runs "$gpu")")" \
           " processi di simulazione: $(count "$(sim_procs "$gpu")")" \
           " (zombie innocui: $(count "$(sim_procs "$gpu" zombie)"))"
    else
      echo "  ${n}  GPU ${gpu}  porta ${p}  NON ATTIVO"
    fi
  done
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
  ;;

wait-idle)
  gpu="${2:?uso: bash superlinks.sh wait-idle <gpu>}"
  n=$(name_of "$gpu"); t0=$(date +%s); t=0; last=-60
  [ -n "$(sl_pid "$gpu")" ] || exit 0
  while :; do
    runs=$(active_runs "$gpu"); procs=$(sim_procs "$gpu")
    known=1
    [ "$runs" = "?" ] && { runs=""; known=0; }
    [ -z "$runs" ] && [ -z "$procs" ] && break
    t=$(( $(date +%s) - t0 ))
    limit=$WAIT_MAX
    [ -z "$runs" ] && [ "$known" = "1" ] && limit=$WAIT_ORPHAN
    if [ $(( t - last )) -ge 60 ]; then
      echo "  ${n}: aspetto la chiusura della run precedente (${t}s/${limit}s):" \
           "run [$(echo $runs)] processi [$(echo $procs)]"
      last=$t
    fi
    if [ "$t" -ge "$limit" ]; then
      if [ -n "$runs" ]; then
        echo "  ${n}: run ancora attiva dopo ${t}s, la fermo"
        for id in $(run_ids "$runs"); do flwr stop "$id" "$n" >/dev/null 2>&1; done
        sleep 5
      else
        echo "  ${n}: run finita ma processo ancora vivo dopo ${t}s, lo chiudo:"
      fi
      for pr in $(sim_procs "$gpu"); do
        ps -o pid=,etime=,args= -p "$pr" 2>/dev/null | cut -c1-160 | sed 's/^/      /'
        kill "$pr" 2>/dev/null
      done
      sleep 5
      for pr in $(sim_procs "$gpu"); do kill -9 "$pr" 2>/dev/null; done
      sleep 1
      left=$(sim_procs "$gpu")
      [ -n "$left" ] && echo "  ${n}: ATTENZIONE, processi ancora vivi: $(echo $left)"
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
