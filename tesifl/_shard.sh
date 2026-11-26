#!/usr/bin/env bash
set -u

export HF_HUB_OFFLINE=1 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
SUPERLINK="gpu${GPU}"
WDIR="${RESULTS}/_gpu${GPU}"
TSV="${RESULTS}/runs_gpu${GPU}.tsv"
BTAG=$(echo "$BETA" | tr -d '.')
mkdir -p "$WDIR"
[ -f "$TSV" ] || printf "seed\tbeta\talgo\trun_id\n" > "$TSV"

read -ra ALGO_ARR <<< "$ALGOS"
n=${#ALGO_ARR[@]}
off=$(( ROT % n ))
ALGO_ARR=( "${ALGO_ARR[@]:$off}" "${ALGO_ARR[@]:0:$off}" )

read -r SA SB SC QA QB <<< "$(python3 - "$BETA" "${EXP_TOML:-experiment.toml}" <<'PY'
import sys, tomllib
b = sys.argv[1]
c = tomllib.load(open(sys.argv[2], "rb"))["algo"]
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

    wscale=""
    core="$label"
    if [[ "$label" =~ ^(.+)_x([0-9]+(p[0-9]+)?)$ ]]; then
      core="${BASH_REMATCH[1]}"
      wscale="${BASH_REMATCH[2]/p/.}"
    fi
    lrate=""
    if [[ "$core" =~ ^(.+)_lr([0-9]+(p[0-9]+)?)$ ]]; then
      core="${BASH_REMATCH[1]}"
      lrate="${BASH_REMATCH[2]/p/.}"
    fi
    bsize=""
    if [[ "$core" =~ ^(.+)_B([0-9]+)$ ]]; then
      core="${BASH_REMATCH[1]}"
      bsize="${BASH_REMATCH[2]}"
    fi

    case "$core" in
      *_lin)  battery="lin" ;;
      *_peuk) battery="peuk" ;;
      *_nm)   battery="nm" ;;
      *)      battery="lin" ;;
    esac
    base="${core%_lin}"; base="${base%_peuk}"; base="${base%_nm}"
    algo=""; extra=""
    case "$base" in
      fedavg)     algo="fedavg" ;;
      fedavg_guard) algo="fedavg"; extra="soc-guard=0.2" ;;
      fedprox)    algo="fedprox" ;;
      sage)       if [ "$battery" = "lin" ]; then
                    algo="sage";     extra="sage-a=${SA} sage-b=${SB} sage-c=${SC}"
                  else
                    algo="sage_soc"; extra="sage-a=${QA} sage-b=${QB}"
                  fi ;;
      sage_smart) algo="sage_smart"; extra="sage-a=${QA} sage-b=${QB}" ;;
      sage_smart_nostale|sage_smart_noest|sage_smart_noneg|sage_smart_e5|sage_smart_b128|sage_smart_nopartial|sage_smart_srvlin|sage_smart_bsteps|sage_smart_bgrow)
                  algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='${base#sage_smart_}'" ;;
      sage_smart_cum0) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='nostale+noest+noneg+e5+b128+nopartial'" ;;
      sage_smart_cum1) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='noest+noneg+e5+b128+nopartial'" ;;
      sage_smart_cum2p) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='noest+noneg+b128'" ;;
      sage_smart_pcum4) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='noest+noneg'" ;;
      sage_smart2_nofc) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='noest'" ;;
      sage_smart2_rev2) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='e5+b128+nopartial'" ;;
      sage_smart2_rev3) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='b128'" ;;
      sage_smart2_socfc) algo="sage_smart"; extra="sage-a=${QA} sage-b=${QB}" ;;
      escs_sd)    algo="escs-sd" ;;
      escs_sp)    algo="escs-sp" ;;
      escs_md)    algo="escs-md" ;;
      escs_mp)    algo="escs-mp" ;;
    esac
    if [ -z "$algo" ]; then
      echo "--- ${label}: etichetta sconosciuta, salto (vedi experiment.toml)"
      FAIL=$((FAIL + 1)); continue
    fi
    extra="world-battery='${battery}'${extra:+ $extra}${wscale:+ workload-scale=${wscale}}"

    RDIR="${WDIR}/${label}_b${BTAG}_s${seed}"
    rm -rf "${RDIR}"; mkdir -p "${RDIR}"
    cfg="num-server-rounds=${ROUNDS} algorithm='${algo}' seed=${seed} beta=${BETA}"
    cfg="$cfg clients-per-round=${K} results-dir='${RDIR}'"
    algocfg="$ALGOCFG"
    [ -n "$bsize" ] && algocfg=$(echo "$algocfg" | sed -E "s/(^| )batch-size=[0-9]+/\1batch-size=${bsize}/")
    [ -n "$lrate" ] && algocfg=$(echo "$algocfg" | sed -E "s/(^| )learning-rate=[0-9.eE+-]+/\1learning-rate=${lrate}/")
    cfg="$cfg ${WORLD} ${algocfg}"
    [ -n "$extra" ] && cfg="$cfg $extra"

    echo ""
    echo "--- ${label} b=${BETA} seed=${seed} ---"
    bash superlinks.sh wait-idle "${GPU}"
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

    ok=0
    while [ $(( $(date +%s) - before )) -lt "$TIMEOUT" ]; do
      newest=$(ls -t "${RDIR}"/*.csv 2>/dev/null | head -1)
      if [ -n "$newest" ]; then
        mv "$newest" "$target"; ok=1
        for ext in devsoc devinfo; do
          f=$(ls "${RDIR}"/*."${ext}" 2>/dev/null | head -1)
          [ -n "$f" ] && mv "$f" "${target%.csv}.${ext}"
        done
        break
      fi
      sleep 20
    done
    if [ "$ok" = "1" ]; then
      echo "  -> ${target} ($(( $(date +%s) - before ))s)"
      rm -rf "${RDIR}"
      DONE=$((DONE + 1))
    else
      echo "  TIMEOUT dopo ${TIMEOUT}s: nessun CSV, fermo la run ${rid}"
      flwr stop "${rid}" "${SUPERLINK}" >/dev/null 2>&1 \
        || echo "  (flwr stop ${rid} ${SUPERLINK} non riuscito: fermala a mano)"
      FAIL=$((FAIL + 1))
    fi
  done
done

bash superlinks.sh wait-idle "${GPU}"
echo ""
echo "=== shard beta=${BETA}: ${DONE} fatte, ${SKIP} saltate, ${FAIL} fallite ==="
