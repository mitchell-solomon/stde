#!/usr/bin/env bash

# Run low-dimensional high-order PDE sweeps.
# Sweeps benchmark equations defined in BENCHMARKS and logs results.
# Modeled after scripts/run_bimamba_all.sh

set -u

# Main arguments for run_lowD_highord.py (edit as needed)
EPOCHS=10000
EVAL_EVERY=5000
LR=1e-4
N_TEST=2000
TEST_BATCH_SIZE=20
SEQ_LEN=5
SEED_FRAC=0.05
SEEDS=1

BENCHMARKS=(KdV2d highord1d)
HIGHORD1D_VARIANTS=(sg gkdv gkdv_high)

mkdir -p logs

for B in "${BENCHMARKS[@]}"; do
    if [ "$B" = "highord1d" ]; then
        for VAR in "${HIGHORD1D_VARIANTS[@]}"; do
            case "$VAR" in
                sg)
                    export HIGHORD1D_CASE=2
                    export HIGHORD1D_EQ=1
                    ;;
                gkdv)
                    export HIGHORD1D_CASE=3
                    export HIGHORD1D_EQ=2
                    ;;
                gkdv_high)
                    export HIGHORD1D_CASE=4
                    export HIGHORD1D_EQ=3
                    ;;
                *)
                    echo "Unknown variant $VAR" >&2
                    continue
                    ;;
            esac
            LOG_FILE="logs/${B}_${VAR}.log"
            echo "Running $B variant $VAR" | tee -a "$LOG_FILE"
            if python run_lowD_highord.py \
                --benchmarks "$B" \
                --seeds "$SEEDS" \
                --epochs "$EPOCHS" \
                --eval_every "$EVAL_EVERY" \
                --lr "$LR" \
                --n_test "$N_TEST" \
                --test_batch_size "$TEST_BATCH_SIZE" \
                --seq_len "$SEQ_LEN" \
                --seed_frac "$SEED_FRAC" \
                >> "$LOG_FILE" 2>&1; then
                echo "Completed $B variant $VAR" >> "$LOG_FILE"
            else
                echo "ERROR running $B variant $VAR" >> "$LOG_FILE"
            fi
        done
    else
        # KdV2d benchmark
        export KDV2D_CASE=4
        LOG_FILE="logs/${B}.log"
        echo "Running $B" | tee -a "$LOG_FILE"
        if python run_lowD_highord.py \
            --benchmarks "$B" \
            --seeds "$SEEDS" \
            --epochs "$EPOCHS" \
            --eval_every "$EVAL_EVERY" \
            --lr "$LR" \
            --n_test "$N_TEST" \
            --test_batch_size "$TEST_BATCH_SIZE" \
            --seq_len "$SEQ_LEN" \
            --seed_frac "$SEED_FRAC" \
            >> "$LOG_FILE" 2>&1; then
            echo "Completed $B" >> "$LOG_FILE"
        else
            echo "ERROR running $B" >> "$LOG_FILE"
        fi
    fi
done
