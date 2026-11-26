#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"

NAME=${VAL_NAME:-"validation_final"}
GPUS_VAL=${VAL_GPUS:-"2"}
F="sweeps/experiment_validations.toml"
PHASES=(
  "sage_smart_lin sage_smart_peuk sage_smart_nm"
  "sage_smart_srvlin_nm"
  "fedavg_guard_nm_B16"
  "fedavg_nm_B16"
  "fedavg_nm fedavg_nm_lr0p04 fedavg_nm_lr0p08"
  "fedprox_nm_B16 sage_nm_B16 escs_sd_nm_B16 escs_sp_nm_B16 escs_md_nm_B16 escs_mp_nm_B16"
)

FT=0
for a in "$@"; do
  case "$a" in --fastTest|--fasttest|--fast-test) FT=1 ;; esac
done

write_toml() {
  mkdir -p sweeps
  python3 - "$NAME" "$F" "$1" "$GPUS_VAL" <<'PY'
import re, sys, tomllib
name, out, algos = sys.argv[1], sys.argv[2], sys.argv[3].split()
gpus = [int(g) for g in sys.argv[4].split()]
src = open("experiment.toml", encoding="utf-8").read()
camp, rest = src.split("[world]", 1)
camp = re.sub(r'(?m)^name\s*=\s*"[^"]*"', f'name = "{name}"', camp, count=1)
camp = re.sub(r'(?m)^gpus\s*=\s*\[[^\]\n]*\]',
              "gpus = [" + ", ".join(map(str, gpus)) + "]", camp, count=1)
lines = camp.split("\n")
i0 = next(i for i, l in enumerate(lines) if re.match(r"\s*algorithms\s*=\s*\[", l))
i1 = next(i for i in range(i0, len(lines)) if lines[i].strip() == "]")
lines[i0:i1 + 1] = ["algorithms = [" + ", ".join(f'"{a}"' for a in algos) + "]"]
txt = "\n".join(lines) + "[world]" + rest
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
print(f"  {out}: campagna {name}, GPU {gpus}, seed {c['campaign']['seeds']}, "
      f"beta {c['campaign']['betas']}, {len(algos)} etichette: {' '.join(algos)}")
PY
}

if [ "$FT" = "1" ]; then
  write_toml "${PHASES[0]}" || exit 1
  EXP_TOML="$F" bash run.sh "$@"
  exit $?
fi

N=${#PHASES[@]}
for i in "${!PHASES[@]}"; do
  p=$((i + 1))
  if [ -n "${VAL_PHASES:-}" ] && [[ " ${VAL_PHASES} " != *" ${p} "* ]]; then
    continue
  fi
  echo ""
  echo "################ fase ${p}/${N}: ${PHASES[$i]} ################"
  write_toml "${PHASES[$i]}" || exit 1
  EXP_TOML="$F" bash run.sh "$@"
  if [ -f "results_${NAME}/runs.tsv" ]; then
    cp "results_${NAME}/runs.tsv" "results_${NAME}/runs_fase${p}.tsv"
  fi
done
echo ""
echo "=== validations: fasi finite, risultati in results_${NAME}/ ==="
