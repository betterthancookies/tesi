#!/usr/bin/env bash
# Esperimento sul WORKLOAD (call del 30/10): da quale carico di altre app la
# negoziazione e la stima del SoC di SAGE-smart cominciano a contare. Gli
# stati del workload (0 / 0.10 / 0.30 in experiment.toml) sono moltiplicati per
# m, con le stesse catene: con lo stesso seed la sequenza degli stati e' la
# stessa, cambia solo quanto pesa ogni stato (tetto w = 0.95).
# Dal carico piu' alto al piu' basso, come chiesto dai relatori.
#
#   nohup bash workload_sweep.sh > workload_sweep.log 2>&1 &     lancio consigliato
#   bash workload_sweep.sh --dry                                  cosa farebbe
#   FT_ALGO=sage_smart_noinfo_nm_x3 bash workload_sweep.sh --fastTest   run minuscola
#
#   m      stato pesante   etichette (mondo nm, beta 0.1, seed 2026)
#   x3     w = 0.90        sage_smart_nm  completo
#   x2     w = 0.60        sage_smart_noneg_nm   senza negoziazione
#   x1p5   w = 0.45        sage_smart_noest_nm   senza stima (SoC = ultimo report)
#   x1     w = 0.30        sage_smart_noinfo_nm  senza entrambe
#                          fedavg_nm             riferimento
#   [B] sage_smart_nm_x1 non c'e': e' sage_smart_nm b01 seed 2026 della
#   campagna v2 (zero ritiri, quindi identica anche col codice del 04/11).
#   4 x 5 - 1 = 19 run in fila sulla GPU di WL_GPUS (default 2, Corinna).
#
# Come ablation_cum.sh: experiment.toml derivato (sweeps/experiment_workload.toml)
# che cambia SOLO nome, GPU, seed, beta e algoritmi; fisica, [algo] e [model]
# restano quelli della campagna principale. Il moltiplicatore viaggia nel nome
# dell'etichetta (_x<m>, "p" = punto), letto da _shard.sh. Risultati in
# results_<nome>/.

set -u
cd "$(dirname "$0")"

NAME=${WL_NAME:-"workload_ResNet18_C100"}
GPUS_WL=${WL_GPUS:-"2"}
SEEDS_WL=${WL_SEEDS:-"2026"}
BETAS_WL=${WL_BETAS:-"0.1"}
MULTS=${WL_MULTS:-"x3 x2 x1p5 x1"}
VARIANTS=${WL_VARIANTS:-"sage_smart_nm sage_smart_noneg_nm sage_smart_noest_nm sage_smart_noinfo_nm fedavg_nm"}
if [ -z "${WL_ALGOS:-}" ]; then
  WL_ALGOS=""
  for m in $MULTS; do
    for v in $VARIANTS; do
      [ "$v" = "sage_smart_nm" ] && [ "$m" = "x1" ] && continue   # = campagna v2
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
camp, rest = src.split("[world]", 1)          # nome, GPU, seed, beta e algoritmi stanno prima
camp = re.sub(r'(?m)^name\s*=\s*"[^"]*"', f'name = "{name}"', camp, count=1)
camp = re.sub(r'(?m)^gpus\s*=\s*\[[^\]\n]*\]',
              "gpus = [" + ", ".join(map(str, gpus)) + "]", camp, count=1)
camp = re.sub(r'(?m)^seeds\s*=\s*\[[^\]\n]*\]',
              "seeds = [" + ", ".join(map(str, seeds)) + "]", camp, count=1)
camp = re.sub(r'(?m)^betas\s*=\s*\[[^\]\n]*\]',
              "betas = [" + ", ".join(map(str, betas)) + "]", camp, count=1)
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
# ogni etichetta deve avere il moltiplicatore in coda
bad = [a for a in algos if not re.search(r"_x[0-9]+(p[0-9]+)?$", a)]
assert not bad, f"etichette senza _x<m>: {bad}"
open(out, "w", encoding="utf-8", newline="\n").write(txt)
print(f"  {out}: campagna {name}, GPU {gpus}, seed {seeds}, beta {betas}, {len(algos)} etichette")
PY

EXP_TOML="$F" bash run.sh "$@"
