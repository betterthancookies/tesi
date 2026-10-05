#!/usr/bin/env bash
# Uno shard = un beta su una GPU. Lanciato da run.sh, non a mano.
# Riceve tutto per ambiente: BETA GPU ROT FORCE RESULTS NAME ROUNDS CLIENTS K
# SEEDS ALGOS TIMEOUT WORLD ALGOCFG
set -u

export HF_HUB_OFFLINE=1 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
SUPERLINK="gpu${GPU}"
WDIR="${RESULTS}/_gpu${GPU}"
TSV="${RESULTS}/runs_gpu${GPU}.tsv"
BTAG=$(echo "$BETA" | tr -d '.')
mkdir -p "$WDIR"
[ -f "$TSV" ] || printf "seed\tbeta\talgo\trun_id\n" > "$TSV"

read -ra ALGO_ARR <<< "$ALGOS"
# [B] rotazione: con piu' shard in parallelo evita che due lavorino sullo
# stesso algoritmo nello stesso istante, contendendosi la GPU.
n=${#ALGO_ARR[@]}
off=$(( ROT % n ))
ALGO_ARR=( "${ALGO_ARR[@]:$off}" "${ALGO_ARR[@]:0:$off}" )

# pesi di SAGE per beta, letti da experiment.toml
read -r SA SB SC QA QB <<< "$(python3 - "$BETA" <<'PY'
import sys, tomllib
b = sys.argv[1]
c = tomllib.load(open("experiment.toml", "rb"))["algo"]
abc = c["sage_abc"].get(b, [0.5, 0.2, 0.3])
ab = c["sage_ab"].get(b, [0.5, 0.5])
print(*abc, *ab)
PY
)"

DONE=0; SKIP=0; FAIL=0
echo "=== shard beta=${BETA} su GPU ${GPU} | ${#ALGO_ARR[@]} algoritmi ==="

for seed in $SEEDS; do
  for label in "${ALGO_ARR[@]}"; do
    target="${RESULTS}/${label}_b${BTAG}_seed${seed}.csv"
    if [ "$FORCE" != "1" ] && [ -f "$target" ]; then
      echo "--- ${label} b=${BETA} s=${seed}: gia' presente, salto"
      SKIP=$((SKIP + 1)); continue
    fi

    # label -> algoritmo + extra specifici
    algo="$label"; extra=""
    case "$label" in
      fedprox)        algo="fedprox" ;;
      sage)           algo="sage";      extra="sage-a=${SA} sage-b=${SB} sage-c=${SC}" ;;
      sage_soc)       algo="sage_soc";  extra="sage-a=${QA} sage-b=${QB}" ;;
      sage_smart)     algo="sage_smart";      extra="sage-a=${QA} sage-b=${QB}" ;;
      escs_sd)        algo="escs-sd" ;;
      escs_sp)        algo="escs-sp" ;;
      escs_md)        algo="escs-md" ;;
      escs_mp)        algo="escs-mp" ;;
      escs_sd_lin)    algo="escs-sd";  extra="escs-battery='energy'" ;;
      escs_sp_lin)    algo="escs-sp";  extra="escs-battery='energy'" ;;
      escs_md_lin)    algo="escs-md";  extra="escs-battery='energy'" ;;
      escs_mp_lin)    algo="escs-mp";  extra="escs-battery='energy'" ;;
    esac

    cfg="num-server-rounds=${ROUNDS} algorithm='${algo}' seed=${seed} beta=${BETA}"
    cfg="$cfg clients-per-round=${K} results-dir='${WDIR}'"
    cfg="$cfg ${WORLD} ${ALGOCFG}"
    [ -n "$extra" ] && cfg="$cfg $extra"

    echo ""
    echo "--- ${label} b=${BETA} seed=${seed} ---"
    before=$(date +%s)
    out=$(flwr run . "${SUPERLINK}" \
            --federation-config "num-supernodes=${CLIENTS}" \
            --run-config "$cfg" 2>&1)
    rid=$(echo "$out" | grep -oP 'run [0-9]+' | head -1 | awk '{print $2}')
    if [ -z "$rid" ]; then
      echo "  ERRORE all'avvio:"; echo "$out" | tail -12
      FAIL=$((FAIL + 1)); continue
    fi
    echo "  run_id=${rid}"
    printf "%s\t%s\t%s\t%s\n" "$seed" "$BETA" "$label" "$rid" >> "$TSV"

    # attende il CSV
    ok=0
    while [ $(( $(date +%s) - before )) -lt "$TIMEOUT" ]; do
      newest=$(ls -t "${WDIR}"/*.csv 2>/dev/null | head -1)
      if [ -n "$newest" ] && [ "$(stat -c %Y "$newest")" -ge "$before" ]; then
        mv "$newest" "$target"; ok=1; break
      fi
      sleep 20
    done
    if [ "$ok" = "1" ]; then
      echo "  -> ${target} ($(( $(date +%s) - before ))s)"
      DONE=$((DONE + 1))
    else
      echo "  TIMEOUT dopo ${TIMEOUT}s: nessun CSV"
      FAIL=$((FAIL + 1))
    fi
  done
done

echo ""
echo "=== shard beta=${BETA}: ${DONE} fatte, ${SKIP} saltate, ${FAIL} fallite ==="