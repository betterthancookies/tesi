#!/usr/bin/env bash
# Unico punto di lancio. Configura tutto in experiment.toml, poi:
#
#   bash run.sh              esegue la campagna
#   bash run.sh --dry        stampa cosa farebbe e si ferma
#   bash run.sh --resume     salta i CSV gia' presenti (default: attivo)
#   bash run.sh --force      rifa' tutto anche se i CSV ci sono
#
# COSA FA
#   - avvia un SuperLink per GPU (superlinks.sh) e verifica che rispondano;
#   - lancia un shard per beta, in parallelo, sfalsati e con l'ordine degli
#     algoritmi ruotato;
#   - archivia i CSV in results_<name>/ con nomi deterministici;
#   - a fine corsa unisce i .tsv e rimuove le cartelle di transito.
#
# [B] SFALSAMENTO E ROTAZIONE. Il 09/09 i tre shard partirono insieme sulla
# stessa prima configurazione e si contesero la GPU: ~20% dei round di FedAvg
# non addestro' nessun client, in silenzio e senza errori nei log. Lo
# scoprimmo solo dai CSV. Da qui stagger_s e la rotazione.

set -u
cd "$(dirname "$0")"

DRY=0; FORCE=0
for a in "$@"; do
  case "$a" in
    --dry) DRY=1 ;;
    --force) FORCE=1 ;;
    --resume) FORCE=0 ;;
    *) echo "opzione sconosciuta: $a"; exit 1 ;;
  esac
done

CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-tesifl}

# ---- legge experiment.toml una volta sola ---------------------------
eval "$(python3 - <<'PY'
import tomllib, shlex
c = tomllib.load(open("experiment.toml", "rb"))
cam, w, al, mo = c["campaign"], c["world"], c["algo"], c["model"]
k = max(1, round(cam["k_fraction"] * cam["clients"]))
out = {
    "NAME": cam["name"], "ROUNDS": cam["rounds"], "CLIENTS": cam["clients"],
    "K": k, "STAGGER": cam["stagger_s"], "TIMEOUT": cam["timeout_s"],
    "SEEDS": " ".join(str(s) for s in cam["seeds"]),
    "BETAS": " ".join(str(b) for b in cam["betas"]),
    "GPUS": " ".join(str(g) for g in cam["gpus"]),
    "ALGOS": " ".join(cam["algorithms"]),
    "WORLD": (f"tiers-enabled={str(w['tiers']).lower()} "
              f"workload-enabled={str(w['workload']).lower()} "
              f"idle-enabled={str(w['idle']).lower()} "
              f"recharge-enabled={str(w['recharge']).lower()} "
              f"recharge-available={str(w['recharge_available']).lower()}"),
        "ALGOCFG": (f'model-name="{mo["name"]}" '
                f"local-epochs={al['local_epochs']} batch-size={al['batch_size']} "
                f"learning-rate={al['learning_rate']} "
                f"fraction-evaluate={al['fraction_evaluate']} "
                f"proximal-mu={al['proximal_mu']} "
                f"smart-stale-weight={al['smart_stale_weight']} "
                f"smart-stale-max={al['smart_stale_max']} "
                f"escs-min-battery={al['escs_min_battery']} "
                f"escs-min-nq={al['escs_min_nq']} "
                f"escs-first-round-all={str(al['escs_first_round_all']).lower()} "
                f"escs-cap-probabilistic={str(al['escs_cap_probabilistic']).lower()}"),
}
for key, val in out.items():
    print(f"{key}={shlex.quote(str(val))}")
PY
)"

RESULTS="results_${NAME}"
read -ra GPU_ARR <<< "$GPUS"
read -ra BETA_ARR <<< "$BETAS"
read -ra ALGO_ARR <<< "$ALGOS"

if [ ${#BETA_ARR[@]} -gt ${#GPU_ARR[@]} ]; then
  echo "ERRORE: ${#BETA_ARR[@]} beta ma ${#GPU_ARR[@]} GPU."; exit 1
fi

N_SEEDS=$(echo "$SEEDS" | wc -w)
TOTAL=$(( ${#ALGO_ARR[@]} * ${#BETA_ARR[@]} * N_SEEDS ))
echo "=== campagna '${NAME}' -> ${RESULTS}/ ==="
echo "    ${CLIENTS} client (k=${K}) | ${ROUNDS} round | beta [${BETAS}] | seed [${SEEDS}]"
echo "    ${#ALGO_ARR[@]} algoritmi x ${#BETA_ARR[@]} beta x ${N_SEEDS} seed = ${TOTAL} run"
echo "    mondo: ${WORLD}"
python3 -c "import sys;sys.path.insert(0,'.');from device.constants import describe;print('    fisica:',describe())"
echo ""

if [ "$DRY" = "1" ]; then
  for i in "${!BETA_ARR[@]}"; do
    echo "  [dry] beta=${BETA_ARR[$i]} -> GPU ${GPU_ARR[$i]}, rotazione ${i}"
  done
  exit 0
fi

# ---- SuperLink, uno per GPU -----------------------------------------
GPUS="$GPUS" bash superlinks.sh start
GPUS="$GPUS" bash superlinks.sh status
for i in "${!GPU_ARR[@]}"; do
  if ! pgrep -f "control-api-address 127.0.0.1:$((39101 + i))" >/dev/null; then
    echo "ERRORE: SuperLink della GPU ${GPU_ARR[$i]} non attivo. Annullo."
    echo "        vedi /tmp/superlink_gpu${GPU_ARR[$i]}.log"; exit 1
  fi
done
echo ""

mkdir -p "$RESULTS"
sessions=()
for i in "${!BETA_ARR[@]}"; do
  beta="${BETA_ARR[$i]}"; gpu="${GPU_ARR[$i]}"
  btag=$(echo "$beta" | tr -d '.')
  sess="${NAME}_b${btag}"; log="log_${NAME}_b${btag}.log"
  tmux kill-session -t "$sess" 2>/dev/null
  tmux new -d -s "$sess" "source ${CONDA_SH} && conda activate ${CONDA_ENV} && \
cd $(pwd) && BETA=${beta} GPU=${gpu} ROT=${i} FORCE=${FORCE} \
RESULTS='${RESULTS}' NAME='${NAME}' ROUNDS=${ROUNDS} CLIENTS=${CLIENTS} K=${K} \
SEEDS='${SEEDS}' ALGOS='${ALGOS}' TIMEOUT=${TIMEOUT} \
WORLD='${WORLD}' ALGOCFG='${ALGOCFG}' bash _shard.sh 2>&1 | tee ${log}"
  sessions+=("$sess")
  echo "  beta=${beta} -> GPU ${gpu} | sessione ${sess} | ${log}"
  [ "$i" -lt $(( ${#BETA_ARR[@]} - 1 )) ] && sleep "$STAGGER"
done

echo ""
echo "  in corso. Monitor:  tail -f log_${NAME}_b*.log"
while true; do
  alive=0
  for s in "${sessions[@]}"; do
    tmux has-session -t "$s" 2>/dev/null && alive=$((alive + 1))
  done
  [ "$alive" -eq 0 ] && break
  sleep 60
done

# ---- pulizia --------------------------------------------------------
for g in "${GPU_ARR[@]}"; do
  left=$(ls "${RESULTS}/_gpu${g}"/*.csv 2>/dev/null | wc -l)
  if [ "$left" -gt 0 ]; then
    echo "  ATTENZIONE: ${left} CSV non rinominati in ${RESULTS}/_gpu${g}/"
  else
    rm -rf "${RESULTS}/_gpu${g}"
  fi
done
tsvs=$(ls "${RESULTS}"/runs_gpu*.tsv 2>/dev/null)
if [ -n "$tsvs" ]; then
  printf "seed\tbeta\talgo\trun_id\n" > "${RESULTS}/runs.tsv"
  cat $tsvs | grep -v "^seed" >> "${RESULTS}/runs.tsv" 2>/dev/null
  rm -f $tsvs
fi
cp experiment.toml "${RESULTS}/experiment.toml"   # com'era configurata la campagna

echo ""
echo "=== fatto: $(ls ${RESULTS}/*_b*_seed*.csv 2>/dev/null | wc -l) CSV su ${TOTAL} ==="
echo "Verifica:  python check_runs.py --dir ${RESULTS}"
echo "Analisi :  python analyze_rounds.py --dir ${RESULTS} && python plot_curves.py --dir ${RESULTS}"