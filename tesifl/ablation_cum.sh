#!/usr/bin/env bash
# Ablazione CUMULATA di SAGE-smart: dal gradino 0 (tutto spento) si accende un
# componente per gradino, nell'ordine di importanza dell'ablazione
# leave-one-out. Mondo nm, seed 2026, beta di experiment.toml: 6 gradini x 3
# beta = 18 run in fila sulla GPU di ABL_GPUS (default 2, Corinna).
#
#   nohup bash ablation_cum.sh > ablation_cum.log 2>&1 &     lancio consigliato
#   bash ablation_cum.sh --dry                                cosa farebbe
#   FT_ALGO=sage_smart_cum0_nm bash ablation_cum.sh --fastTest   run minuscola
#
#   gradino  etichetta            componenti accesi
#   0        sage_smart_cum0_nm   nessuno: a*SoC+b*D, E=5 e B=128 fissi, niente
#                                 negoziazione, SoC = ultimo report, media FedAvg
#   1        sage_smart_cum1_nm   + anzianita' (c)
#   2        sage_smart_cum2_nm   + E adattivo (scelto dal client)
#   3        sage_smart_cum3_nm   + B piccolo (scelto dal client, 16-128)
#   4        sage_smart_cum4_nm   + negoziazione
#   5        sage_smart_cum5_nm   + stima del SoC lato server
#   6        sage_smart_nm        + FedNova = SAGE-smart completo: GIA' FATTO
#                                 (campagna cap_aug e ablazione, seed 2026)
#   riferimenti gia' fatti: fedavg_nm (ablazione), sage_nm (cap_aug)
#
# [B] 06/11 CUMULATA DEL PLUS3, ABL_PLUS3=1: i gradini 0-3 restano quelli
# sopra (gia' fatti, x1); si aggiungono
#   4        sage_smart_pcum4_nm  gradino 3 + lavoro parziale
#   5        sage_smart_pcum5_nm  + negoziazione (rifiuto per deadline solo se
#                                 colpa del carico)
#   6        sage_smart_plus3_nm  + stima del SoC col workload medio = plus3
# a ogni carico del workload (_x3 _x2 _x1p5 _x1, ABL_MULTS per cambiarli):
# 36 run in results_ablation_cum_plus3_ResNet18_C100/.
#   ABL_PLUS3=1 nohup bash ablation_cum.sh > ablation_plus3.log 2>&1 &
#   ABL_PLUS3=1 FT_ALGO=sage_smart_pcum5_nm_x3 bash ablation_cum.sh --fastTest
#
# Come ablation.sh: experiment.toml derivato (sweeps/experiment_ablation_cum.toml)
# che cambia SOLO nome, GPU, seed e algoritmi; fisica, [algo] e [model] restano
# quelli della campagna principale. Risultati in results_<nome>/.
# [B] la lista degli algoritmi e' sostituita riga per riga fino alla "]" sola
# su una riga: un commento con "[B]" dentro la lista non la tronca.

set -u
cd "$(dirname "$0")"

if [ "${ABL_SMART2:-0}" = "1" ]; then
  # [B] 09/11 CUMULATA SAGE-smart CONTRO SAGE-smart 2, come quella del report
  # 13 ma senza il gradino di FedNova (scartato) e tutta col codice di oggi:
  # mondo nm, x1, seed 2026, i 3 beta di experiment.toml.
  #   0  sage_smart_cum0      nessun componente: E=5 e B=128 fissi, niente
  #                           negoziazione, SoC = ultimo report
  #   1  sage_smart_cum1      + anzianita' (c)
  #   2  sage_smart_cum2      + E adattivo (scelto dal client)
  #   3  sage_smart_cum3      + B piccolo (scelto dal client, 16-128)
  #   4  sage_smart_pcum4     + lavoro parziale                  (comune)
  #   SAGE-smart (plus4, quella della campagna principale):
  #   5  sage_smart_pcum5     + negoziazione (rifiuto se colpa del carico)
  #   6  sage_smart_plus4     + stima del SoC (workload medio, anche
  #                           all'iscrizione)
  #   SAGE-smart 2:
  #   5' sage_smart2_noinfo   - broadcast (10% non raggiungibili)
  #   6' sage_smart2_nofc     + negoziazione 2 (rosa 1.5 k, scelta fresca)
  #   7' sage_smart2_socfc    + previsione del solo SoC (ricariche attese,
  #                           consumo); il workload pesa solo nella scelta
  #                           finale. [B] 10/11: sostituisce sage_smart2, che
  #                           pesava la classifica per 1 - w atteso ed
  #                           escludeva la catena high (-1.5 punti a beta 0.1)
  # 10 gradini x 3 beta = 30 run. Etichette col suffisso _x1 (identico a
  # senza). ABL_BETAS="0.1" per un solo beta (le run gia' presenti nella
  # cartella si saltano, --force le rifa').
  # [B] 10/11 figura a sei gradini di SAGE-smart 2 (0 base, 1 +c, 2 +E
  # adattivo, 3 +B adattivo, 4 +negoziazione, 5 +stima SoC): il lavoro
  # parziale entra nell'E adattivo, quindi serve anche sage_smart_cum2p
  # (cum2 + lavoro parziale); il 3 e' pcum4, il 4 nofc, il 5 socfc.
  DEF_NAME="ablation_cum_smart2_ResNet18_C100"
  DEF_ALGOS=""
  for v in sage_smart_cum0_nm sage_smart_cum1_nm sage_smart_cum2_nm sage_smart_cum2p_nm sage_smart_cum3_nm \
           sage_smart_pcum4_nm sage_smart_pcum5_nm sage_smart_plus4_nm \
           sage_smart2_noinfo_nm sage_smart2_nofc_nm sage_smart2_socfc_nm; do
    DEF_ALGOS="${DEF_ALGOS:+$DEF_ALGOS }${v}_x1"
  done
elif [ "${ABL_PLUS3:-0}" = "1" ]; then
  # [B] 06/11 cumulata del PLUS3: solo i gradini nuovi, a ogni carico del
  # workload (dal piu' alto), seed 2026, i 3 beta di experiment.toml.
  # 3 gradini x 4 carichi x 3 beta = 36 run. Gradini 0-3: vecchia cartella.
  DEF_NAME="ablation_cum_plus3_ResNet18_C100"
  DEF_ALGOS=""
  for m in ${ABL_MULTS:-x3 x2 x1p5 x1}; do
    for v in sage_smart_pcum4_nm sage_smart_pcum5_nm sage_smart_plus3_nm; do
      DEF_ALGOS="${DEF_ALGOS:+$DEF_ALGOS }${v}_${m}"
    done
  done
else
  DEF_NAME="ablation_cum_ResNet18_C100_cap_aug"
  DEF_ALGOS="sage_smart_cum0_nm sage_smart_cum1_nm sage_smart_cum2_nm sage_smart_cum3_nm sage_smart_cum4_nm sage_smart_cum5_nm"
fi
NAME=${ABL_NAME:-"$DEF_NAME"}
GPUS_ABL=${ABL_GPUS:-"2"}
SEEDS_ABL=${ABL_SEEDS:-"2026"}
ALGOS=${ABL_ALGOS:-"$DEF_ALGOS"}
# [B] 10/11: ABL_BETAS="0.1" restringe i beta (vuoto = quelli di experiment.toml)
BETAS_ABL=${ABL_BETAS:-""}
F="sweeps/experiment_ablation_cum.toml"

mkdir -p sweeps
python3 - "$NAME" "$F" "$ALGOS" "$GPUS_ABL" "$SEEDS_ABL" "$BETAS_ABL" <<'PY' || exit 1
import re, sys, tomllib
name, out, algos = sys.argv[1], sys.argv[2], sys.argv[3].split()
gpus = [int(g) for g in sys.argv[4].split()]
seeds = [int(s) for s in sys.argv[5].split()]
# repr del float: 1.0 resta "1.0" (con "1" il beta diventerebbe intero e il
# nome dei file "_b1_" invece di "_b10_")
betas = [float(b) for b in sys.argv[6].split()] or None
src = open("experiment.toml", encoding="utf-8").read()
camp, rest = src.split("[world]", 1)          # nome, GPU, seed e algoritmi stanno prima
camp = re.sub(r'(?m)^name\s*=\s*"[^"]*"', f'name = "{name}"', camp, count=1)
camp = re.sub(r'(?m)^gpus\s*=\s*\[[^\]\n]*\]',
              "gpus = [" + ", ".join(map(str, gpus)) + "]", camp, count=1)
camp = re.sub(r'(?m)^seeds\s*=\s*\[[^\]\n]*\]',
              "seeds = [" + ", ".join(map(str, seeds)) + "]", camp, count=1)
if betas:
    camp = re.sub(r'(?m)^betas\s*=\s*\[[^\]\n]*\]',
                  "betas = [" + ", ".join(repr(b) for b in betas) + "]", camp, count=1)
# lista degli algoritmi: dalla riga "algorithms = [" alla prima riga che e'
# solo "]", sostituita per righe (niente regex non greedy sui commenti)
lines = camp.split("\n")
i0 = next(i for i, l in enumerate(lines) if re.match(r"\s*algorithms\s*=\s*\[", l))
i1 = next(i for i in range(i0, len(lines)) if lines[i].strip() == "]")
lines[i0:i1 + 1] = ["algorithms = [" + ", ".join(f'"{a}"' for a in algos) + "]"]
txt = "\n".join(lines) + "[world]" + rest
c, orig = tomllib.loads(txt), tomllib.loads(src)
# solo questi campi devono essere cambiati
changed = {("campaign", "name"), ("campaign", "gpus"), ("campaign", "seeds"),
           ("campaign", "algorithms")}
assert c["campaign"]["name"] == name
assert c["campaign"]["gpus"] == gpus
assert c["campaign"]["seeds"] == seeds
assert c["campaign"]["algorithms"] == algos
if betas:
    changed.add(("campaign", "betas"))
    assert c["campaign"]["betas"] == betas
    assert all(isinstance(b, float) for b in c["campaign"]["betas"])
for sec in orig:
    for k, v in orig[sec].items():
        if (sec, k) not in changed:
            assert c[sec][k] == v, (sec, k)
open(out, "w", encoding="utf-8", newline="\n").write(txt)
print(f"  {out}: campagna {name}, GPU {gpus}, seed {seeds}, beta {c['campaign']['betas']}, "
      f"{len(algos)} etichette")
PY

EXP_TOML="$F" bash run.sh "$@"
