#!/usr/bin/env bash

set -u
cd "$(dirname "$0")"

if [ "${ABL_SMART2:-0}" = "1" ]; then
  DEF_NAME="ablation_cum_smart2_ResNet18_C100"
  DEF_ALGOS=""
  for v in sage_smart_cum0_nm sage_smart_cum1_nm sage_smart_cum2_nm sage_smart_cum2p_nm sage_smart_cum3_nm \
           sage_smart_pcum4_nm sage_smart_pcum5_nm sage_smart_plus4_nm \
           sage_smart2_noinfo_nm sage_smart2_nofc_nm sage_smart2_socfc_nm; do
    DEF_ALGOS="${DEF_ALGOS:+$DEF_ALGOS }${v}_x1"
  done
elif [ "${ABL_PLUS3:-0}" = "1" ]; then
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
BETAS_ABL=${ABL_BETAS:-""}
F="sweeps/experiment_ablation_cum.toml"

mkdir -p sweeps
python3 - "$NAME" "$F" "$ALGOS" "$GPUS_ABL" "$SEEDS_ABL" "$BETAS_ABL" <<'PY' || exit 1
import re, sys, tomllib
name, out, algos = sys.argv[1], sys.argv[2], sys.argv[3].split()
gpus = [int(g) for g in sys.argv[4].split()]
seeds = [int(s) for s in sys.argv[5].split()]
betas = [float(b) for b in sys.argv[6].split()] or None
src = open("experiment.toml", encoding="utf-8").read()
camp, rest = src.split("[world]", 1)
camp = re.sub(r'(?m)^name\s*=\s*"[^"]*"', f'name = "{name}"', camp, count=1)
camp = re.sub(r'(?m)^gpus\s*=\s*\[[^\]\n]*\]',
              "gpus = [" + ", ".join(map(str, gpus)) + "]", camp, count=1)
camp = re.sub(r'(?m)^seeds\s*=\s*\[[^\]\n]*\]',
              "seeds = [" + ", ".join(map(str, seeds)) + "]", camp, count=1)
if betas:
    camp = re.sub(r'(?m)^betas\s*=\s*\[[^\]\n]*\]',
                  "betas = [" + ", ".join(repr(b) for b in betas) + "]", camp, count=1)
lines = camp.split("\n")
i0 = next(i for i, l in enumerate(lines) if re.match(r"\s*algorithms\s*=\s*\[", l))
i1 = next(i for i in range(i0, len(lines)) if lines[i].strip() == "]")
lines[i0:i1 + 1] = ["algorithms = [" + ", ".join(f'"{a}"' for a in algos) + "]"]
txt = "\n".join(lines) + "[world]" + rest
c, orig = tomllib.loads(txt), tomllib.loads(src)
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
