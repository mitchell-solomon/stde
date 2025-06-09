#!/usr/bin/env bash
set -euo pipefail

# 1) Define arrays of values (around each default) for parameters without explicit choices:
seeds=(0 1 2)
dims=(2)
epochs=(4000)
lrs=(0.001)
N_tests=(2000)
test_bss=(20)
seq_lens=(5)
rand_bss=(2)
num_blocks=(1 2)
hidden_feats=(32)
expansion_factors=(2.0)
dense_expansions=(2)

# 2) Use the explicit ‘choices’ from your parser:
x_orderings=("radial")
activations=("silu" "gelu" "tanh" "wave")
ssm_activations=("silu")

# 3) For boolean flags, include both “off” (empty) and “on” (the flag itself):
sparse_flags=("" "--sparse")
complement_flags=("")
tie_in_proj_flags=("")
tie_gate_flags=("")
bidirectional_flags=("" "--bidirectional")

# 4) Path to your training script:
CMD="python train_bimamba_sine_gordon.py"

# 5) Nested loops for grid search:
for seed in "${seeds[@]}"; do
  for dim in "${dims[@]}"; do
    for epoch in "${epochs[@]}"; do
      for lr in "${lrs[@]}"; do
        for N_test in "${N_tests[@]}"; do
          for test_bs in "${test_bss[@]}"; do
            for seq_len in "${seq_lens[@]}"; do
              for rand_bs in "${rand_bss[@]}"; do
                for xb in "${x_orderings[@]}"; do
                  for act in "${activations[@]}"; do
                    for ssm_act in "${ssm_activations[@]}"; do
                      for nblk in "${num_blocks[@]}"; do
                        for hf in "${hidden_feats[@]}"; do
                          for ef in "${expansion_factors[@]}"; do
                            for de in "${dense_expansions[@]}"; do
                              for sparse in "${sparse_flags[@]}"; do
                                for comp in "${complement_flags[@]}"; do
                                  for tip in "${tie_in_proj_flags[@]}"; do
                                    for tig in "${tie_gate_flags[@]}"; do
                                      for bid in "${bidirectional_flags[@]}"; do

                                        # build a descriptive run name:
                                        run_name=gs_s${seed}_d${dim}_e${epoch}_lr${lr}_\
xord${xb}_act${act}_ssm${ssm_act}_nb${nblk}_hf${hf}_ef${ef}_de${de}\
${sparse//--/}_\
${comp//--/}_\
${tip//--/}_\
${tig//--/}_\
${bid//--/}

                                        # invoke your python script:
                                        $CMD \
                                          --SEED "$seed" \
                                          --dim "$dim" \
                                          --epochs "$epoch" \
                                          --lr "$lr" \
                                          --N_test "$N_test" \
                                          --test_batch_size "$test_bs" \
                                          --seq_len "$seq_len" \
                                          --rand_batch_size "$rand_bs" \
                                          --x_ordering "$xb" \
                                          --activation "$act" \
                                          --ssm_activation "$ssm_act" \
                                          --num_mamba_blocks "$nblk" \
                                          --hidden_features "$hf" \
                                          --expansion_factor "$ef" \
                                          --dense_expansion "$de" \
                                          $sparse \
                                          $comp \
                                          $tip \
                                          $tig \
                                          $bid \
                                          --run_name "$run_name"

                                      done
                                    done
                                  done
                                done
                              done
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
