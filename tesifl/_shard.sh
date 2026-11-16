#!/usr/bin/env bash
# Uno shard = un beta su una GPU. Lanciato da run.sh, non a mano.
# Con meno GPU che beta, run.sh mette piu' shard in fila sulla stessa GPU.
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

# pesi di SAGE per beta, letti da experiment.toml (o da EXP_TOML, vedi run.sh)
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

    # [B] 04/11, esperimento sul workload (workload_sweep.sh): un suffisso
    # finale _x<m> moltiplica gli stati del workload per m, con "p" al posto
    # del punto (sage_smart_nm_x3, fedavg_nm_x1p5). Si toglie prima di leggere
    # la batteria. Senza suffisso: workload-scale di pyproject (1.0).
    wscale=""
    core="$label"
    if [[ "$label" =~ ^(.+)_x([0-9]+(p[0-9]+)?)$ ]]; then
      core="${BASH_REMATCH[1]}"
      wscale="${BASH_REMATCH[2]/p/.}"
    fi

    # label -> batteria dei device + algoritmo + extra specifici.
    # Il suffisso _lin / _peuk / _nm e' la batteria FISICA del mondo (lineare,
    # Peukert, datasheet); senza suffisso (fedavg, fedprox) e' lineare.
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
      fedprox)    algo="fedprox" ;;
      # SAGE: nel mondo lineare quello del paper (a, b, c, rinnovabile),
      # negli altri l'ablazione (a, b, niente rinnovabile ne' soglia)
      sage)       if [ "$battery" = "lin" ]; then
                    algo="sage";     extra="sage-a=${SA} sage-b=${SB} sage-c=${SC}"
                  else
                    algo="sage_soc"; extra="sage-a=${QA} sage-b=${QB}"
                  fi ;;
      # [B] 16/11: sage_smart e' l'ex SAGE-smart 2 (previsione del SoC,
      # negoziazione con rosa e scelta fresca, niente broadcast, lavoro
      # parziale), l'unico SAGE-smart: fra _lin, _peuk e _nm cambia solo la
      # batteria dei device. Le altre etichette SPENGONO componenti
      # (smart-ablation in server_app): nostale, noest, noneg, e5, b128,
      # nopartial.
      sage_smart) algo="sage_smart"; extra="sage-a=${QA} sage-b=${QB}" ;;
      # ablazione leave-one-out: un componente spento, il nome e' il token
      sage_smart_nostale|sage_smart_noest|sage_smart_noneg|sage_smart_e5|sage_smart_b128|sage_smart_nopartial)
                  algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='${base#sage_smart_}'" ;;
      # ablazione CUMULATA (ablation_cum.sh), nomi delle run gia' fatte.
      # Ordine diritto: 0 cum0, 1 cum1 (+ anzianita'), 2 cum2p (+ E adattivo
      # col lavoro parziale), 3 pcum4 (+ B adattivo), 4 sage_smart2_nofc
      # (+ negoziazione), 5 sage_smart (+ previsione del SoC).
      # Ordine inverso: 0 cum0, 1 cum1, 2 sage_smart2_rev2 (+ negoziazione e
      # previsione, E = 5 e B = 128 fissi), 3 sage_smart2_rev3 (+ E
      # adattivo), 4 sage_smart (+ B adattivo).
      # [B] 16/11: cum2p e pcum4 erano girate col carico MASSIMO
      # all'iscrizione (tolto): col codice di oggi non tornano identiche.
      # Le altre sono uguali alle run fatte fino al 12/11.
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
      # nome di sage_smart nelle cumulate fatte fino al 12/11 (stessa run)
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

    # [B] UNA CARTELLA PER RUN. Prima tutte le run dello shard scrivevano in
    # ${WDIR} e lo shard prendeva il CSV piu' recente: una run oltre il
    # timeout continuava a girare, il suo CSV arrivava mentre lo shard
    # aspettava la run SUCCESSIVA e veniva salvato col nome sbagliato, e da li'
    # tutte le etichette successive slittavano di una posizione (campagna
    # rechargeONdeviceON_ResNet20_totale, beta 0.1). Con una cartella propria
    # un CSV in ritardo non puo' finire sotto un'altra etichetta. Il beta e'
    # nel nome perche' con una GPU sola piu' shard usano la stessa ${WDIR}.
    RDIR="${WDIR}/${label}_b${BTAG}_s${seed}"
    rm -rf "${RDIR}"; mkdir -p "${RDIR}"
    cfg="num-server-rounds=${ROUNDS} algorithm='${algo}' seed=${seed} beta=${BETA}"
    cfg="$cfg clients-per-round=${K} results-dir='${RDIR}'"
    cfg="$cfg ${WORLD} ${ALGOCFG}"
    [ -n "$extra" ] && cfg="$cfg $extra"

    echo ""
    echo "--- ${label} b=${BETA} seed=${seed} ---"
    # una run alla volta: la precedente deve aver chiuso (vedi superlinks.sh)
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

    # attende il CSV di QUESTA run, nella sua cartella
    ok=0
    while [ $(( $(date +%s) - before )) -lt "$TIMEOUT" ]; do
      newest=$(ls -t "${RDIR}"/*.csv 2>/dev/null | head -1)
      if [ -n "$newest" ]; then
        mv "$newest" "$target"; ok=1
        # [B] 30/10: file per device (server_app li scrive prima del .csv),
        # con lo stesso nome della run: <label>_b<beta>_seed<s>.devsoc/.devinfo
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
      # [B] la run va FERMATA: lasciata andare contende la GPU alle successive
      # e le rallenta, che e' il modo in cui scattano altri timeout
      echo "  TIMEOUT dopo ${TIMEOUT}s: nessun CSV, fermo la run ${rid}"
      flwr stop "${rid}" "${SUPERLINK}" >/dev/null 2>&1 \
        || echo "  (flwr stop ${rid} ${SUPERLINK} non riuscito: fermala a mano)"
      FAIL=$((FAIL + 1))
    fi
  done
done

bash superlinks.sh wait-idle "${GPU}"   # nessun residuo per lo shard successivo
echo ""
echo "=== shard beta=${BETA}: ${DONE} fatte, ${SKIP} saltate, ${FAIL} fallite ==="