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
      sage_smart) algo="sage_smart"; extra="sage-a=${QA} sage-b=${QB}" ;;
      # ablazione di SAGE-smart (ablation.sh): toglie un solo componente,
      # vedi smart-ablation in server_app
      sage_smart_nostale|sage_smart_noest|sage_smart_noneg|sage_smart_e5|\
sage_smart_nonova|sage_smart_b128)
                  algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='${base#sage_smart_}'" ;;
      # ablazione CUMULATA (ablation_cum.sh): dal gradino 0, tutto spento, si
      # accende un componente per gradino: anzianita', E adattivo, B piccolo,
      # negoziazione, stima del SoC. Il gradino 6 (+ FedNova) e' sage_smart.
      sage_smart_cum0) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='nostale+noest+noneg+e5+nonova+b128'" ;;
      sage_smart_cum1) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='noest+noneg+e5+nonova+b128'" ;;
      sage_smart_cum2) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='noest+noneg+nonova+b128'" ;;
      # [B] 10/11 gradino 2 col lavoro parziale dentro l'E adattivo (il client
      # decide quanto lavorare, anche meno di un'epoca se E = 2 non sta nella
      # deadline), per la cumulata di SAGE-smart 2 a sei gradini: 2 = cum2p,
      # 3 = pcum4 (= cum3 + lavoro parziale), 4 = nofc, 5 = socfc
      sage_smart_cum2p) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='noest+noneg+nonova+b128+partial'" ;;
      sage_smart_cum3) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='noest+noneg+nonova'" ;;
      sage_smart_cum4) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='noest+nonova'" ;;
      sage_smart_cum5) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='nonova'" ;;
      # esperimento sul workload (workload_sweep.sh): negoziazione e stima del
      # SoC spente insieme, che i relatori vedono come un componente solo
      sage_smart_noinfo) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='noneg+noest'" ;;
      # [B] 05/11 SAGE-smart "plus": il client rifiuta la proposta anche se il
      # lavoro minimo non sta nella deadline, e la stima del SoC usa il
      # workload medio riportato. Le varianti tolgono negoziazione o stima.
      # (senza entrambe = sage_smart_noinfo: i due interruttori non contano)
      sage_smart_plus) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='dlrefuse+loadavg'" ;;
      sage_smart_plus_noneg) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='noneg+dlrefuse+loadavg'" ;;
      sage_smart_plus_noest) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='noest+dlrefuse+loadavg'" ;;
      # [B] 06/11 "plus2": come plus, ma il rifiuto per deadline scatta solo
      # se a riposo il client ci starebbe (colpa del carico del momento).
      # Col plus i telefoni lenti con molti dati rifiutavano sempre.
      sage_smart_plus2) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='dlload+loadavg'" ;;
      # [B] 06/11 "plus3": plus2 + lavoro parziale. Chi non sta nella deadline
      # nemmeno con E_lo (lento + molti dati) fa i passi che ci stanno invece
      # di arrivare in ritardo; nella media pesa per i campioni visti.
      sage_smart_plus3) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='dlload+loadavg+partial'" ;;
      # [B] 09/11 "plus4" = SAGE-smart del paper: plus3 + all'iscrizione il
      # server assume il workload MEDIO della popolazione (0.118 a x1) invece
      # del massimo. Col massimo, a beta 1.0, chi non era mai contattato veniva
      # stimato scarico e non veniva piu' interrogato (6 mai scelti a x1).
      sage_smart_plus4) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='dlload+loadavg+partial+enrollmean'" ;;
      # [B] 09/11 SAGE-smart 2: la stima PREVEDE (SoC con le ricariche
      # attese, workload che torna verso la media del client) e la
      # negoziazione RETTIFICA: proposte a una rosa di 1.5 k, il server sceglie
      # i k migliori sullo stato fresco pesando per 1 - w; chi resta fuori e'
      # rinviato senza perdere anzianita'. Il client rifiuta solo per energia
      # (niente dlload), lavoro parziale come plus4. Broadcast spento: chi e'
      # raggiungibile si scopre dalle risposte (10% non risponde).
      #   sage_smart2          completo
      #   sage_smart2_nopick   senza rosa: proposte ai soli k (negoziazione 1)
      #   sage_smart2_nofc     senza previsione: classifica sull'ultimo report
      #   sage_smart2_noinfo   senza entrambe: ultimo report, niente proposte
      sage_smart2) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='partial+enrollmean+forecast+pick+nobcast'" ;;
      sage_smart2_nopick) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='partial+enrollmean+forecast+nobcast'" ;;
      sage_smart2_nofc) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='noest+partial+enrollmean+pick+nobcast'" ;;
      sage_smart2_noinfo) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='noest+noneg+partial+enrollmean+nobcast'" ;;
      # [B] 10/11 SAGE-smart 2 CORRETTO: previsione del solo SoC in classifica
      # (token socfc). sage_smart2 pesava la classifica per 1 - w atteso ed
      # escludeva la catena di workload high (-1.5 punti a beta 0.1, 4
      # telefoni mai scelti); qui il workload pesa solo nella scelta finale.
      sage_smart2_socfc) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='partial+enrollmean+socfc+pick+nobcast'" ;;
      # [B] 12/11 cumulata di SAGE-smart 2 in ORDINE INVERSO (l'ordine conta:
      # dopo l'E adattivo negoziazione e stima hanno poco da correggere).
      # 0 cum0, 1 cum1, 2 = cum1 + negoziazione e stima (E = 5, B = 128
      # fissi), 3 = + E adattivo col lavoro parziale, 4 = + B adattivo =
      # sage_smart2_socfc (stessi token)
      sage_smart2_rev2) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='e5+nonova+b128+enrollmean+socfc+pick+nobcast'" ;;
      sage_smart2_rev3) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='nonova+b128+partial+enrollmean+socfc+pick+nobcast'" ;;
      # [B] 06/11 ablazione CUMULATA del plus3 (ABL_PLUS3=1 bash
      # ablation_cum.sh). I gradini 0-3 sono sage_smart_cum0..3 (gia' fatti
      # a x1), poi: 4 = gradino 3 + lavoro parziale; 5 = + negoziazione, col
      # rifiuto per deadline solo se colpa del carico; 6 = + stima del SoC col
      # workload medio = sage_smart_plus3. FedNova e' spento di default.
      sage_smart_pcum4) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='noest+noneg+partial'" ;;
      sage_smart_pcum5) algo="sage_smart"
                  extra="sage-a=${QA} sage-b=${QB} smart-ablation='noest+dlload+partial'" ;;
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