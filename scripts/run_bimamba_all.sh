#!/usr/bin/env bash

# Run train_bimamba.py for every PDE defined in EqnConfig.
# For each PDE we run once with sparse_stde and once with backward-mode AD
# (stacked Hessian calculation).  Logs are written to logs/ and runs continue
# even if an error occurs.

set -u

# Main arguments for train_bimamba.py (edit as needed)
EPOCHS=10000
EVAL_EVERY=5000
LR=1e-3
N_TEST=2000
TEST_BATCH_SIZE=20
SEQ_LEN=3
SEED_FRAC=0.01

# gather PDE names from the config dataclass
PDE_NAMES=$(python - <<'PY'
from typing import get_args
from stde.config import EqnConfig
print(' '.join(get_args(EqnConfig.__annotations__['name'])))
PY
)

mkdir -p logs

for PDE in $PDE_NAMES; do
    # Set dims for special cases
    if [[ "$PDE" == "SemilinearHeatTime" || "$PDE" == "SineGordonTime" || "$PDE" == "AllenCahnTime" ]]; then
        DIMS=(10 100 1000)
    elif [[ "$PDE" == *Threebody* ]]; then
        DIMS=(3 4 5)
    else
        # query if the equation is time dependent to set spatial dimension
        IS_TIME_DEP=$(python - "$PDE" <<'PY'
import sys
from stde import equations as eqns
print('1' if getattr(eqns, sys.argv[1]).time_dependent else '0')
PY
        )
        if [ "$IS_TIME_DEP" = "1" ]; then
            DIMS=(1)
        else
            DIMS=(2)
        fi
    fi

    for DIM in "${DIMS[@]}"; do
        for METHOD in sparse_stde stacked; do
            LOG_FILE="logs/${PDE}_${METHOD}_d${DIM}.log"
            RUN_NAME="${PDE}_${METHOD}_d${DIM}"
            echo "Running $PDE with $METHOD and dim $DIM" | tee -a "$LOG_FILE"
            if python train_bimamba.py \
                --eqn_name "$PDE" \
                --spatial_dim "$DIM" \
                --hess_diag_method "$METHOD" \
                --run_name "$RUN_NAME" \
                --epochs "$EPOCHS" \
                --eval_every "$EVAL_EVERY" \
                --lr "$LR" \
                --N_test "$N_TEST" \
                --test_batch_size "$TEST_BATCH_SIZE" \
                --seq_len "$SEQ_LEN" \
                --seed_frac "$SEED_FRAC" \
                >> "$LOG_FILE" 2>&1; then
                echo "Completed $PDE with $METHOD and dim $DIM" >> "$LOG_FILE"
            else
                echo "ERROR running $PDE with $METHOD and dim $DIM" >> "$LOG_FILE"
            fi
        done
    done

done

