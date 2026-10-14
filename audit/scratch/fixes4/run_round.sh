#!/bin/zsh
set -eu
cd /Users/navneetbavineni/STEGO
export WANDB_MODE=offline HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
for candidate in D E; do
  for seed in 0 1 2; do
    logfile="audit/scratch/fixes4/${candidate}_seed${seed}_train.log"
    if [[ -e "$logfile" ]]; then
      print -u2 "Refusing to overwrite existing run log: $logfile"
      exit 1
    fi
    print "START candidate=$candidate seed=$seed"
    caffeinate -i python3 -B audit/scratch/fixes4/run.py --candidate "$candidate" --seed "$seed" > "$logfile" 2>&1
    print "FINISHED candidate=$candidate seed=$seed"
  done
done
