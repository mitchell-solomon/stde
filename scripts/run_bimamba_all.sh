#!/usr/bin/env bash

# Run train_bimamba.py for every PDE defined in EqnConfig.
# For each PDE we run once with sparse_stde and once with backward-mode AD
# (stacked Hessian calculation).  Logs are written to logs/ and runs continue
# even if an error occurs.

set -u

# gather PDE names from the config dataclass
PDE_NAMES=$(python - <<'PY'
from typing import get_args
from stde.config import EqnConfig
print(' '.join(get_args(EqnConfig.__annotations__['name'])))
PY
)

mkdir -p logs

for PDE in $PDE_NAMES; do
    # query if the equation is time dependent to set spatial dimension
    IS_TIME_DEP=$(python - <<'PY'
import sys
from stde import equations as eqns
print('1' if getattr(eqns, sys.argv[1]).time_dependent else '0')
PY
    "$PDE")

    if [ "$IS_TIME_DEP" = "1" ]; then
        DIM=1
    else
        DIM=2
    fi

    for METHOD in sparse_stde stacked; do
        LOG_FILE="logs/${PDE}_${METHOD}.log"
        RUN_NAME="${PDE}_${METHOD}"
        echo "Running $PDE with $METHOD" | tee -a "$LOG_FILE"
        if python train_bimamba.py \
            --eqn_name "$PDE" \
            --dim "$DIM" \
            --hess_diag_method "$METHOD" \
            --get_mem \
            --run_name "$RUN_NAME" >> "$LOG_FILE" 2>&1; then
            echo "Completed $PDE with $METHOD" >> "$LOG_FILE"
        else
            echo "ERROR running $PDE with $METHOD" >> "$LOG_FILE"
        fi
    done

done

