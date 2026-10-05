#!/usr/bin/env bash
# Avvia / ferma / controlla un SuperLink per GPU.
#
#   bash superlinks.sh start    # ne avvia uno per ogni GPU in GPUS
#   bash superlinks.sh status
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
# [B] DATABASE SEPARATI. La coda delle run vive in state.db. Con un database
# condiviso i tre SuperLink si contenderebbero la stessa coda. E la coda
# SOPRAVVIVE: se il SuperLink muore, le run lanciate nel frattempo restano
# dentro e partono tutte insieme al riavvio successivo (11/09).

set -u

GPUS=${GPUS:-"1 2 3"}
PORT_BASE=${PORT_BASE:-39101}
STATE_DIR=${STATE_DIR:-$HOME/.flwr/sweep}
CONF=${CONF:-$HOME/.flwr/config.toml}
PROJECT=${PROJECT:-$PWD}

read -ra GPU_ARR <<< "$GPUS"

name_of() { echo "gpu$1"; }
port_of() { local i=0; for g in "${GPU_ARR[@]}"; do
    [ "$g" = "$1" ] && { echo $((PORT_BASE + i)); return; }; i=$((i+1)); done; }

case "${1:-status}" in

start)
  mkdir -p "$STATE_DIR"
  # --- voci in config.toml, aggiunte una sola volta
  for gpu in "${GPU_ARR[@]}"; do
    n=$(name_of "$gpu"); p=$(port_of "$gpu")
    if ! grep -q "^\[superlink\.${n}\]" "$CONF" 2>/dev/null; then
      printf '\n[superlink.%s]\naddress = "127.0.0.1:%s"\n' "$n" "$p" >> "$CONF"
      echo "  config.toml: aggiunta [superlink.${n}] -> 127.0.0.1:${p}"
    fi
  done

  for gpu in "${GPU_ARR[@]}"; do
    n=$(name_of "$gpu"); p=$(port_of "$gpu")
    if pgrep -f "control-api-address 127.0.0.1:${p}" >/dev/null; then
      echo "  ${n}: gia' attivo sulla porta ${p}"
      continue
    fi
    ( cd "$PROJECT" && \
      export CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 && \
      nohup flower-superlink --insecure --simulation --isolation subprocess \
        --control-api-address "127.0.0.1:${p}" \
        --serverappio-api-address "127.0.0.1:0" \
        --database "${STATE_DIR}/state_${n}.db" \
        > "/tmp/superlink_${n}.log" 2>&1 & )
    sleep 4
    if pgrep -f "control-api-address 127.0.0.1:${p}" >/dev/null; then
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
    pid=$(pgrep -f "control-api-address 127.0.0.1:${p}" | head -1)
    if [ -n "$pid" ]; then
      # [B] il numero di run in coda smaschera un SuperLink che accetta ma non
      # esegue: se cresce e non scende, l'esecutore e' morto.
      q=$(grep -c "Started task" "/tmp/superlink_${n}.log" 2>/dev/null || echo 0)
      echo "  ${n}  GPU ${gpu}  porta ${p}  pid ${pid}  task avviati: ${q}"
    else
      echo "  ${n}  GPU ${gpu}  porta ${p}  NON ATTIVO"
    fi
  done
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
  ;;

stop)
  for gpu in "${GPU_ARR[@]}"; do
    n=$(name_of "$gpu"); p=$(port_of "$gpu")
    pid=$(pgrep -f "control-api-address 127.0.0.1:${p}" | head -1)
    [ -n "$pid" ] && { kill "$pid"; echo "  ${n}: fermato (pid ${pid})"; }
  done
  sleep 3
  # i figli non muoiono col padre: vanno chiusi a mano
  pkill -f "flwr-simulation" 2>/dev/null && echo "  simulazioni residue chiuse"
  ;;

*)
  echo "uso: bash superlinks.sh {start|status|stop}"
  exit 1
  ;;
esac