#!/usr/bin/env bash

set -u
cd "$(dirname "$0")"

NAME=${WL_NAME:-"workload_ResNet18_C100"}
GPUS_WL=${WL_GPUS:-"2"}
SEEDS_WL=${WL_SEEDS:-"2026"}
BETAS_WL=${WL_BETAS:-"0.1"}
MULTS=${WL_MULTS:-"x3 x2 x1p5 x1"}
if [ "${WL_SMART2:-0}" = "1" ]; then
  NAME=${WL_NAME:-"smart2_ResNet18_C100"}
  BETAS_WL=${WL_BETAS:-"0.1 0.5 1.0"}
  MULTS=${WL_MULTS:-"x1 x3"}
  VARIANTS=${WL_VARIANTS:-"sage_smart2_nm sage_smart2_noinfo_nm"}
elif [ "${WL_PLUS:-0}" = "3" ]; then
  MULTS=${WL_MULTS:-"x3 x1"}
  VARIANTS=${WL_VARIANTS:-"sage_smart_plus3_nm"}
elif [ "${WL_PLUS:-0}" = "2" ]; then
  MULTS=${WL_MULTS:-"x3 x1"}
  VARIANTS=${WL_VARIANTS:-"sage_smart_plus2_nm"}
elif [ "${WL_PLUS:-0}" = "1" ]; then
  VARIANTS=${WL_VARIANTS:-"sage_smart_plus_nm sage_smart_plus_noneg_nm sage_smart_plus_noest_nm"}
else
  VARIANTS=${WL_VARIANTS:-"sage_smart_nm sage_smart_noneg_nm sage_smart_noest_nm sage_smart_noinfo_nm fedavg_nm"}
fi
if [ -z "${WL_ALGOS:-}" ]; then
  WL_ALGOS=""
  for m in $MULTS; do
    for v in $VARIANTS; do
      [ "$v" = "sage_smart_nm" ] && [ "$m" = "x1" ] && continue
      WL_ALGOS="${WL_ALGOS:+$WL_ALGOS }${v}_${m}"
    done
  done
fi
F="sweeps/experiment_workload.toml"

mkdir -p sweeps
python3 - "$NAME" "$F" "$WL_ALGOS" "$GPUS_WL" "$SEEDS_WL" "$BETAS_WL" <<'PY' || exit 1
import re, sys, tomllib
name, out, algos = sys.argv[1], sys.argv[2], sys.argv[3].split()
gpus = [int(g) for g in sys.argv[4].split()]
seeds = [int(s) for s in sys.argv[5].split()]
betas = [float(b) for b in sys.argv[6].split()]
src = open("experiment.toml", encoding="utf-8").read()
camp, rest = src.split("[world]", 1)
camp = re.sub(r'(?m)^name\s*=\s*"[^"]*"', f'name = "{name}"', camp, count=1)
camp = re.sub(r'(?m)^gpus\s*=\s*\[[^\]\n]*\]',
              "gpus = [" + ", ".join(map(str, gpus)) + "]", camp, count=1)
camp = re.sub(r'(?m)^seeds\s*=\s*\[[^\]\n]*\]',
              "seeds = [" + ", ".join(map(str, seeds)) + "]", camp, count=1)
camp = re.sub(r'(?m)^betas\s*=\s*\[[^\]\n]*\]',
              "betas = [" + ", ".join(map(str, betas)) + "]", camp, count=1)
lines = camp.split("\n")
i0 = next(i for i, l in enumerate(lines) if re.match(r"\s*algorithms\s*=\s*\[", l))
i1 = next(i for i in range(i0, len(lines)) if lines[i].strip() == "]")
lines[i0:i1 + 1] = ["algorithms = [" + ", ".join(f'"{a}"' for a in algos) + "]"]
txt = "\n".join(lines) + "[world]" + rest
c, orig = tomllib.loads(txt), tomllib.loads(src)
changed = {("campaign", "name"), ("campaign", "gpus"), ("campaign", "seeds"),
           ("campaign", "betas"), ("campaign", "algorithms")}
assert c["campaign"]["name"] == name
assert c["campaign"]["gpus"] == gpus
assert c["campaign"]["seeds"] == seeds
assert c["campaign"]["betas"] == betas
assert c["campaign"]["algorithms"] == algos
for sec in orig:
    for k, v in orig[sec].items():
        if (sec, k) not in changed:
            assert c[sec][k] == v, (sec, k)
bad = [a for a in algos if not re.search(r"_x[0-9]+(p[0-9]+)?$", a)]
assert not bad, f"etichette senza _x<m>: {bad}"
open(out, "w", encoding="utf-8", newline="\n").write(txt)
print(f"  {out}: campagna {name}, GPU {gpus}, seed {seeds}, beta {betas}, {len(algos)} etichette")
PY

EXP_TOML="$F" bash run.sh "$@"
