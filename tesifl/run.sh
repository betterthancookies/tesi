#!/usr/bin/env bash
# Unico punto di lancio. Configura tutto in experiment.toml, poi:
#
#   bash run.sh              esegue la campagna
#   bash run.sh --fastTest   run minuscola (FedAvg, beta 0.5, 3 round) che
#                            verifica ambiente, GPU, dataset e valori; scrive
#                            in results_fasttest/, non tocca la campagna
#   FT_ALGO=sage_smart_nm bash run.sh --fastTest   la stessa con un'altra
#                            etichetta (es. per vedere i B scelti da SAGE-smart)
#   bash run.sh --dry        stampa cosa farebbe e si ferma
#   bash run.sh --resume     salta i CSV gia' presenti (default: attivo)
#   bash run.sh --force      rifa' tutto anche se i CSV ci sono
#
#   EXP_TOML=altro.toml bash run.sh   campagna, algoritmi e [algo] da un
#       altro file (lo usa sweep_batch.sh). La FISICA viene sempre da
#       experiment.toml: la legge device/constants.py nei processi Flower.
#
# COSA FA
#   - controlla l'ambiente (comandi, GPU vista da torch, CIFAR-10 in cache);
#   - avvia un SuperLink per GPU (superlinks.sh) e verifica che rispondano;
#   - assegna i beta alle GPU a giro (beta i -> GPU i mod n_gpu): con una GPU
#     per beta girano in parallelo, sfalsati e con l'ordine degli algoritmi
#     ruotato; i beta che condividono una GPU girano IN SEQUENZA nella stessa
#     sessione tmux. Con una GPU sola l'intera campagna e' una fila unica;
#   - archivia i CSV in results_<name>/ con nomi deterministici;
#   - a fine corsa unisce i .tsv e rimuove le cartelle di transito.
#
# [B] SFALSAMENTO E ROTAZIONE. Il 09/09 i tre shard partirono insieme sulla
# stessa prima configurazione e si contesero la GPU: ~20% dei round di FedAvg
# non addestro' nessun client, in silenzio e senza errori nei log. Lo
# scoprimmo solo dai CSV. Da qui stagger_s e la rotazione, e la regola che
# due run non condividono mai una GPU.

set -u
cd "$(dirname "$0")"

DRY=0; FORCE=0; FAST=0
for a in "$@"; do
  case "$a" in
    --dry) DRY=1 ;;
    --force) FORCE=1 ;;
    --resume) FORCE=0 ;;
    --fastTest|--fasttest|--fast-test) FAST=1 ;;
    *) echo "opzione sconosciuta: $a"; exit 1 ;;
  esac
done

CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-tesifl}
export EXP_TOML=${EXP_TOML:-experiment.toml}
[ -f "$EXP_TOML" ] || { echo "ERRORE: $EXP_TOML non esiste"; exit 1; }

# ---- legge experiment.toml una volta sola ---------------------------
eval "$(python3 - <<'PY'
import os, tomllib, shlex
c = tomllib.load(open(os.environ["EXP_TOML"], "rb"))
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
                f"smart-soc-min={al['smart_soc_min']} "
                f"smart-epochs-min={al['smart_epochs_min']} "
                f"smart-deadline-mult={al['smart_deadline_mult']} "
                f"smart-probe-samples={al['smart_probe_samples']} "
                f"smart-max-waves={al['smart_max_waves']} "
                f"smart-batch-min={al['smart_batch_min']} "
                f"smart-batch-max={al['smart_batch_max']} "
                f"escs-min-battery={al['escs_min_battery']} "
                f"escs-min-nq={al['escs_min_nq']} "
                f"escs-first-round-all={str(al['escs_first_round_all']).lower()} "
                f"escs-cap-probabilistic={str(al['escs_cap_probabilistic']).lower()}"),
}
for key, val in out.items():
    print(f"{key}={shlex.quote(str(val))}")
PY
)" || { echo "ERRORE: experiment.toml non leggibile (serve Python >= 3.11)"; exit 1; }

RESULTS="results_${NAME}"
read -ra GPU_ARR <<< "$GPUS"
read -ra BETA_ARR <<< "$BETAS"
read -ra ALGO_ARR <<< "$ALGOS"
NG=${#GPU_ARR[@]}
[ "$NG" -gt 0 ] || { echo "ERRORE: nessuna GPU in experiment.toml"; exit 1; }

# ---- controlli d'ambiente -------------------------------------------
# [B] cambio server (ottobre 2026): ambiente, dataset in cache e config.toml
# di flwr vanno ricreati, e ogni pezzo mancante fallisce in un punto diverso
# (spesso dentro il SuperLink, dove non lo vedi). Meglio fermarsi qui.
# tmux e conda.sh servono solo alla campagna: nel fastTest sono avvisi.
WARN=""
preflight() {
  local ok=1 gpu="${GPU_ARR[0]}" cmd
  echo "=== controlli d'ambiente ==="
  for cmd in flwr flower-superlink; do
    if command -v "$cmd" >/dev/null 2>&1; then
      echo "  [OK]   comando ${cmd}"
    else
      echo "  [FAIL] comando ${cmd} non trovato: conda activate ${CONDA_ENV}? pip install -e .?"
      ok=0
    fi
  done
  for cmd in tmux; do
    if command -v "$cmd" >/dev/null 2>&1; then
      echo "  [OK]   comando ${cmd}"
    elif [ "$FAST" = "1" ]; then
      echo "  [WARN] comando ${cmd} non trovato: serve per la campagna"
      WARN="${WARN}|tmux non installato: serve per la campagna (conda install -c conda-forge tmux)"
    else
      echo "  [FAIL] comando ${cmd} non trovato (conda install -c conda-forge tmux)"; ok=0
    fi
  done
  if [ -f "$CONDA_SH" ]; then
    echo "  [OK]   ${CONDA_SH}"
  elif [ "$FAST" = "1" ]; then
    echo "  [WARN] ${CONDA_SH} non esiste: serve per la campagna"
    WARN="${WARN}|CONDA_SH non trovato: export CONDA_SH=\$(conda info --base)/etc/profile.d/conda.sh"
  else
    echo "  [FAIL] ${CONDA_SH} non esiste:"
    echo "         export CONDA_SH=\$(conda info --base)/etc/profile.d/conda.sh"; ok=0
  fi
  if CUDA_VISIBLE_DEVICES="$gpu" python3 -c \
      "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "  [OK]   torch vede la GPU ${gpu} ($(CUDA_VISIBLE_DEVICES="$gpu" python3 -c \
      'import torch; print(torch.__version__, torch.cuda.get_device_name(0))' 2>/dev/null))"
  else
    echo "  [FAIL] torch non vede la GPU ${gpu}: indice giusto in experiment.toml?"
    echo "         torch compilato per una CUDA piu' nuova del driver? (nvidia-smi)"; ok=0
  fi
  if HF_HUB_OFFLINE=1 python3 -c \
      "from datasets import load_dataset; load_dataset('uoft-cs/cifar10', split='test')" \
      >/dev/null 2>&1; then
    echo "  [OK]   CIFAR-10 in cache"
  else
    echo "  [FAIL] CIFAR-10 non in cache (i SuperLink girano con HF_HUB_OFFLINE=1)."
    echo "         Scaricalo una volta:"
    echo "         python -c \"from datasets import load_dataset; load_dataset('uoft-cs/cifar10')\""
    ok=0
  fi
  echo ""
  [ "$ok" = "1" ]
}

# avvia i SuperLink delle GPU date e controlla che rispondano
start_superlinks() {
  local gpus="$1" i=0 g
  GPUS="$gpus" bash superlinks.sh start
  GPUS="$gpus" bash superlinks.sh status
  for g in $gpus; do
    if ! pgrep -f "control-api-address 127.0.0.1:$((39101 + i))" >/dev/null; then
      echo "ERRORE: SuperLink della GPU ${g} non attivo. Annullo."
      echo "        vedi /tmp/superlink_gpu${g}.log"; return 1
    fi
    i=$((i + 1))
  done
  echo ""
}

# ---- fastTest -------------------------------------------------------
if [ "$FAST" = "1" ]; then
  FT_RESULTS="results_fasttest"; FT_ROUNDS=3; FT_BETA=0.5; FT_TIMEOUT=1800
  FT_ALGO=${FT_ALGO:-fedavg}
  gpu="${GPU_ARR[0]}"; seed="${SEEDS%% *}"
  echo "=== fastTest: ${FT_ALGO}, beta ${FT_BETA}, seed ${seed}, ${FT_ROUNDS} round, ${CLIENTS} client (k=${K}), GPU ${gpu} ==="
  python3 -c "import sys;sys.path.insert(0,'.');from device.constants import describe;print('    fisica:',describe())"
  echo ""
  preflight || { echo "=== PROBLEMI nell'ambiente: NON procedere ==="; exit 1; }
  [ "$DRY" = "1" ] && exit 0
  # il SuperLink e' lo stesso della campagna (GPU in prima posizione)
  start_superlinks "$GPUS" || exit 1
  rm -rf "$FT_RESULTS"; mkdir -p "$FT_RESULTS"
  t0=$(date +%s)
  BETA=${FT_BETA} GPU=${gpu} ROT=0 FORCE=1 RESULTS="${FT_RESULTS}" NAME=fasttest \
  ROUNDS=${FT_ROUNDS} CLIENTS=${CLIENTS} K=${K} SEEDS="${seed}" ALGOS="${FT_ALGO}" \
  TIMEOUT=${FT_TIMEOUT} WORLD="${WORLD}" ALGOCFG="${ALGOCFG}" \
    bash _shard.sh 2>&1 | tee "${FT_RESULTS}/fasttest.log"
  dt=$(( $(date +%s) - t0 ))
  btag=$(echo "$FT_BETA" | tr -d '.')
  python3 check_fasttest.py --csv "${FT_RESULTS}/${FT_ALGO}_b${btag}_seed${seed}.csv" \
    --label "$FT_ALGO" \
    --rounds "$FT_ROUNDS" --clients "$CLIENTS" --seconds "$dt" --warn "$WARN"
  rc=$?
  [ "$rc" = "0" ] || echo "  log: ${FT_RESULTS}/fasttest.log  e  /tmp/superlink_gpu${gpu}.log"
  exit "$rc"
fi

# ---- beta -> GPU ----------------------------------------------------
# GPU_BETAS[j] = "beta:rotazione ..." dei beta assegnati alla GPU j
declare -a GPU_BETAS
for i in "${!BETA_ARR[@]}"; do
  j=$(( i % NG ))
  GPU_BETAS[$j]="${GPU_BETAS[$j]:-}${GPU_BETAS[$j]:+ }${BETA_ARR[$i]}:${i}"
done

N_SEEDS=$(echo "$SEEDS" | wc -w)
TOTAL=$(( ${#ALGO_ARR[@]} * ${#BETA_ARR[@]} * N_SEEDS ))
echo "=== campagna '${NAME}' -> ${RESULTS}/ ==="
echo "    ${CLIENTS} client (k=${K}) | ${ROUNDS} round | beta [${BETAS}] | seed [${SEEDS}]"
echo "    ${#ALGO_ARR[@]} algoritmi x ${#BETA_ARR[@]} beta x ${N_SEEDS} seed = ${TOTAL} run"
echo "    mondo: ${WORLD}"
python3 -c "import sys;sys.path.insert(0,'.');from device.constants import describe;print('    fisica:',describe())"
for j in "${!GPU_ARR[@]}"; do
  [ -n "${GPU_BETAS[$j]:-}" ] || continue
  bs=""; for pair in ${GPU_BETAS[$j]}; do bs="${bs} ${pair%%:*}"; done
  n_b=$(echo $bs | wc -w)
  echo "    GPU ${GPU_ARR[$j]}: beta [${bs# }] in sequenza -> $(( n_b * ${#ALGO_ARR[@]} * N_SEEDS )) run in fila"
done
echo ""

preflight || { echo "=== PROBLEMI nell'ambiente: annullo ==="; exit 1; }
[ "$DRY" = "1" ] && exit 0

# ---- SuperLink, uno per GPU -----------------------------------------
start_superlinks "$GPUS" || exit 1

mkdir -p "$RESULTS"
sessions=(); first=1
for j in "${!GPU_ARR[@]}"; do
  [ -n "${GPU_BETAS[$j]:-}" ] || continue
  gpu="${GPU_ARR[$j]}"
  sess="${NAME}_gpu${gpu}"
  # gli shard della stessa GPU in fila: ";" e non "&&", cosi' uno shard con
  # run fallite non blocca i beta successivi
  shards=""
  for pair in ${GPU_BETAS[$j]}; do
    beta="${pair%%:*}"; rot="${pair##*:}"
    btag=$(echo "$beta" | tr -d '.')
    log="log_${NAME}_b${btag}.log"
    shards="${shards}${shards:+; }EXP_TOML='${EXP_TOML}' BETA=${beta} GPU=${gpu} ROT=${rot} FORCE=${FORCE} \
RESULTS='${RESULTS}' NAME='${NAME}' ROUNDS=${ROUNDS} CLIENTS=${CLIENTS} K=${K} \
SEEDS='${SEEDS}' ALGOS='${ALGOS}' TIMEOUT=${TIMEOUT} \
WORLD='${WORLD}' ALGOCFG='${ALGOCFG}' bash _shard.sh 2>&1 | tee ${log}"
    echo "  beta=${beta} -> GPU ${gpu} | sessione ${sess} | ${log}"
  done
  [ "$first" = "1" ] || sleep "$STAGGER"
  first=0
  tmux kill-session -t "$sess" 2>/dev/null
  tmux new -d -s "$sess" "source ${CONDA_SH} && conda activate ${CONDA_ENV} && \
cd $(pwd) && { ${shards}; }"
  sessions+=("$sess")
done

echo ""
echo "  in corso. Monitor:  tail -f log_${NAME}_b*.log"
echo "            sessioni: ${sessions[*]}  (tmux attach -t <sessione>)"
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
  # [B] ogni run ha la sua sottocartella: un CSV rimasto li' e' di una run
  # andata in timeout, e il nome della cartella dice quale
  left=$(find "${RESULTS}/_gpu${g}" -name '*.csv' 2>/dev/null | wc -l)
  if [ "$left" -gt 0 ]; then
    echo "  ATTENZIONE: ${left} CSV di run in timeout in ${RESULTS}/_gpu${g}/:"
    find "${RESULTS}/_gpu${g}" -name '*.csv' | sed 's/^/    /'
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
cp "$EXP_TOML" "${RESULTS}/experiment.toml"   # com'era configurata la campagna

echo ""
echo "=== fatto: $(ls ${RESULTS}/*_b*_seed*.csv 2>/dev/null | wc -l) CSV su ${TOTAL} ==="
echo "Verifica:  python check_runs.py --dir ${RESULTS}"
echo "Analisi :  python analyze_rounds.py --dir ${RESULTS} && python plot_curves.py --dir ${RESULTS}"
