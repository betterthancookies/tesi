#!/usr/bin/env bash

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
camp, rest = src.split("[world]", 1)
camp = re.sub(r'(?m)^name\s*=\s*"[^"]*"', f'name = "{name}"', camp, count=1)
camp = re.sub(r'(?m)^gpus\s*=\s*\[[^\]]*\]',
              "gpus = [" + ", ".join(map(str, gpus)) + "]", camp, count=1)
camp = re.sub(r'(?ms)^algorithms\s*=\s*\[.*?\]',
              "algorithms = [" + ", ".join(f'"{a}"' for a in algos) + "]",
              camp, count=1)
txt = camp + "[world]" + rest
c, orig = tomllib.loads(txt), tomllib.loads(src)
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
