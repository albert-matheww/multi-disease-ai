#!/bin/zsh
# Every experiment that needs TabPFN, in one go. Requires TABPFN_TOKEN (in .env or the shell),
# which the account holder obtains at https://ux.priorlabs.ai after accepting the licence.
# Nothing here is a substitute for real runs: without a token each step exits with an error.
#
#   zsh paper-draft/experiments/run_tabpfn_all.sh          # ~ CPU-hours; resumable per stage
#
# Outputs land in paper-draft/results/raw/*tabpfn*.json and are picked up by the two analyze
# stages at the end. Then: python paper-draft/experiments/worked_example.py > paper-draft/results/worked_examples.md
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python; X=paper-draft/experiments/run_experiments.py; A=paper-draft/experiments/run_ablations.py
LOG=paper-draft/experiments/logs/tabpfn_all.log
: > $LOG
run() { echo "##### $* ($(date +%H:%M:%S))" | tee -a $LOG; "$@" 2>&1 | grep -E "^\[|Error|Traceback|error" | tee -a $LOG; }

# 1. the shipped pipeline (writes models/*_tabpfn.joblib, reports/*_metrics.json and figures)
run $PY main.py --skip-download --stages train,evaluate,explain
# 2. paper harness, same folds/seeds as the baselines
run $PY $X --stage main     --models tabpfn
run $PY $X --stage cv       --models tabpfn
run $PY $X --stage coverage --models tabpfn
run $PY $X --stage shap
# 3. design-choice ablations with TabPFN as the model (one repeat: cost)
for s in preproc threshold conformal; do run $PY $A --stage $s --models tabpfn --repeats 1; done
# 4. tables
run $PY $X --stage analyze
run $PY $A --stage analyze
# 5. real held-out example rows scored by the served predictor
$PY paper-draft/experiments/worked_example.py > paper-draft/results/worked_examples.md
echo "##### ALL DONE ($(date +%H:%M:%S))" | tee -a $LOG
