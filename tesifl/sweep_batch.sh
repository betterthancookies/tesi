#!/usr/bin/env bash

set -u
cd "$(dirname "$0")"

BATCHES=${BATCHES:-"32 64 128"}
ALGOS=${ALGOS:-"fedavg sage_smart_nm"}
ARGS=("$@")

mkdir -p sweeps
for b in $BATCHES; do
  f="sweeps/experiment_B${b}.toml"
  python3 - "$b" "$f" "$ALGOS" <<'PY' || exit 1
import re, sys, tomllib
b, out, algos = int(sys.argv[1]), sys.argv[2], sys.argv[3].split()
src = open("experiment.toml", encoding="utf-8").read()
camp, rest = src.split("[world]", 1)
camp = re.sub(r'(?m)^name\s*=\s*"[^"]*"', f'name = "sweepB{b}"', camp, count=1)
camp = re.sub(r'(?ms)^algorithms\s*=\s*\[.*?\]',
              "algorithms = [" + ", ".join(f'"{a}"' for a in algos) + "]",
              camp, count=1)
rest = re.sub(r'(?m)^batch_size\s*=\s*\d+', f"batch_size = {b}", rest, count=1)
for k in ("smart_batch_min", "smart_batch_max"):
    rest = re.sub(rf'(?m)^{k}\s*=\s*\d+', f"{k} = {b}", rest, count=1)
txt = camp + "[world]" + rest
c, orig = tomllib.loads(txt), tomllib.loads(src)
changed = {("campaign", "name"), ("campaign", "algorithms"), ("algo", "batch_size"),
           ("algo", "smart_batch_min"), ("algo", "smart_batch_max")}
assert c["campaign"]["name"] == f"sweepB{b}"
assert c["campaign"]["algorithms"] == algos
assert c["algo"]["batch_size"] == b
assert c["algo"]["smart_batch_min"] == c["algo"]["smart_batch_max"] == b
for sec in orig:
    for k, v in orig[sec].items():
        if (sec, k) not in changed:
            assert c[sec][k] == v, (sec, k)
open(out, "w", encoding="utf-8", newline="\n").write(txt)
print(f"  {out}: B={b}, algoritmi {algos}")
PY
done

for b in $BATCHES; do
  echo ""
  echo "################ sweep: B = ${b} ################"
  EXP_TOML="sweeps/experiment_B${b}.toml" bash run.sh ${ARGS[@]+"${ARGS[@]}"} || {
    echo "sweep: run.sh fallito per B=${b}, mi fermo"; exit 1; }
done

echo ""
echo "=== sweep finito. Confronto:  python compare_batch.py ==="
