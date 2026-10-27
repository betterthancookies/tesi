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
# Come ablation.sh: experiment.toml derivato (sweeps/experiment_ablation_cum.toml)
# che cambia SOLO nome, GPU, seed e algoritmi; fisica, [algo] e [model] restano
# quelli della campagna principale. Risultati in results_<nome>/.
# [B] la lista degli algoritmi e' sostituita riga per riga fino alla "]" sola
# su una riga: un commento con "[B]" dentro la lista non la tronca.

set -u
cd "$(dirname "$0")"

NAME=${ABL_NAME:-"ablation_cum_ResNet18_C100_cap_aug"}
GPUS_ABL=${ABL_GPUS:-"2"}
SEEDS_ABL=${ABL_SEEDS:-"2026"}
ALGOS=${ABL_ALGOS:-"sage_smart_cum0_nm sage_smart_cum1_nm sage_smart_cum2_nm sage_smart_cum3_nm sage_smart_cum4_nm sage_smart_cum5_nm"}
F="sweeps/experiment_ablation_cum.toml"

mkdir -p sweeps
python3 - "$NAME" "$F" "$ALGOS" "$GPUS_ABL" "$SEEDS_ABL" <<'PY' || exit 1
import re, sys, tomllib
name, out, algos = sys.argv[1], sys.argv[2], sys.argv[3].split()
gpus = [int(g) for g in sys.argv[4].split()]
seeds = [int(s) for s in sys.argv[5].split()]
src = open("experiment.toml", encoding="utf-8").read()
camp, rest = src.split("[world]", 1)          # nome, GPU, seed e algoritmi stanno prima
camp = re.sub(r'(?m)^name\s*=\s*"[^"]*"', f'name = "{name}"', camp, count=1)
camp = re.sub(r'(?m)^gpus\s*=\s*\[[^\]\n]*\]',
              "gpus = [" + ", ".join(map(str, gpus)) + "]", camp, count=1)
camp = re.sub(r'(?m)^seeds\s*=\s*\[[^\]\n]*\]',
              "seeds = [" + ", ".join(map(str, seeds)) + "]", camp, count=1)
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
for sec in orig:
    for k, v in orig[sec].items():
        if (sec, k) not in changed:
            assert c[sec][k] == v, (sec, k)
open(out, "w", encoding="utf-8", newline="\n").write(txt)
print(f"  {out}: campagna {name}, GPU {gpus}, seed {seeds}, {len(algos)} etichette")
PY

EXP_TOML="$F" bash run.sh "$@"
