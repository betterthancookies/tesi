#!/usr/bin/env bash
# Ablazione leave-one-out di SAGE-smart: quanto pesa ogni componente.
# Mondo nm, beta e seed di experiment.toml, 8 etichette x 3 beta = 24 run in
# fila sulla GPU di ABL_GPUS (default 1, server Stanis).
#
#   nohup bash ablation.sh > ablation.log 2>&1 &       lancio consigliato
#   bash ablation.sh --dry                             cosa farebbe
#   FT_ALGO=sage_smart_noneg_nm bash ablation.sh --fastTest   run minuscola
#
#   sage_smart_nm           riferimento: SAGE-smart completo
#   sage_smart_nostale_nm   senza anzianita' (c = 0 nello score)
#   sage_smart_noest_nm     senza stima del SoC: ultimo SoC riportato
#   sage_smart_noneg_nm     senza negoziazione: niente proposte/rifiuti
#   sage_smart_e5_nm        senza epoche adattive: E = 5 fisso (B libero)
#   sage_smart_nonova_nm    senza FedNova: media pesata per i dati
#   sage_smart_b128_nm      senza batch adattivo: B = 128 fisso (E libero)
#   fedavg_nm               riferimento basso, stesso mondo e stessa GPU
#
# La campagna e' una normale campagna di run.sh con un experiment.toml
# derivato (sweeps/experiment_ablation.toml) che cambia SOLO nome, algoritmi
# e GPU: fisica, [algo] e [model] restano quelli di experiment.toml, quindi
# la stessa configurazione della campagna principale (cap_aug).
# Risultati in results_<nome>/.
#
# [B] experiment.toml NON va toccato su Stanis: e' lo stesso file della
# campagna di Corinna (gpus = [2]); le GPU di qui stanno solo nel derivato.
# Per i comandi a mano:  EXP_TOML=sweeps/experiment_ablation.toml bash superlinks.sh status

set -u
cd "$(dirname "$0")"

NAME=${ABL_NAME:-"ablation_ResNet18_C100_cap_aug"}
GPUS_ABL=${ABL_GPUS:-"1"}
ALGOS=${ABL_ALGOS:-"sage_smart_nm sage_smart_nostale_nm sage_smart_noest_nm sage_smart_noneg_nm sage_smart_e5_nm sage_smart_nonova_nm sage_smart_b128_nm fedavg_nm"}
F="sweeps/experiment_ablation.toml"

mkdir -p sweeps
python3 - "$NAME" "$F" "$ALGOS" "$GPUS_ABL" <<'PY' || exit 1
import re, sys, tomllib
name, out, algos = sys.argv[1], sys.argv[2], sys.argv[3].split()
gpus = [int(g) for g in sys.argv[4].split()]
src = open("experiment.toml", encoding="utf-8").read()
camp, rest = src.split("[world]", 1)          # nome, GPU e algoritmi stanno prima
camp = re.sub(r'(?m)^name\s*=\s*"[^"]*"', f'name = "{name}"', camp, count=1)
camp = re.sub(r'(?m)^gpus\s*=\s*\[[^\]]*\]',
              "gpus = [" + ", ".join(map(str, gpus)) + "]", camp, count=1)
camp = re.sub(r'(?ms)^algorithms\s*=\s*\[.*?\]',
              "algorithms = [" + ", ".join(f'"{a}"' for a in algos) + "]",
              camp, count=1)
txt = camp + "[world]" + rest
c, orig = tomllib.loads(txt), tomllib.loads(src)
# solo questi campi devono essere cambiati
changed = {("campaign", "name"), ("campaign", "gpus"), ("campaign", "algorithms")}
assert c["campaign"]["name"] == name
assert c["campaign"]["gpus"] == gpus
assert c["campaign"]["algorithms"] == algos
for sec in orig:
    for k, v in orig[sec].items():
        if (sec, k) not in changed:
            assert c[sec][k] == v, (sec, k)
open(out, "w", encoding="utf-8", newline="\n").write(txt)
print(f"  {out}: campagna {name}, GPU {gpus}, {len(algos)} etichette")
PY

EXP_TOML="$F" bash run.sh "$@"
