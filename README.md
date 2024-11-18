This repo provides the official implementation of the NeurIPS2024 paper [Stochastic Taylor Derivative Estimator: Efficient amortization for arbitrary differential operators](https://openreview.net/forum?id=J2wI2rCG2u).

# Installation
Simply run
``` shell
pip install .
```

# How to run the minimal example script
For reader who do not wish to go through the entire repo, the script `sine_gordon.py` provides a minimal implementation of the sine-gordon equation described in Appendix I.1. The default hyperparmeter setting follows the description in Appendix H. To run sparse STDE for 100kD Sine-Gordon with randomization batch of `16`: 
``` shell
python sine_gordon.py --sparse --dim 100000 --rand_batch_size 16
```

# How to reproduce results shown in the paper
## Inseparable and effectively high-dimensional PDEs
To run the 100kD two-body Allen-Cahn equation described in Appendix I.1. with sparse STDE: 
``` shell
./scripts/insep.sh --config.eqn_cfg.rand_batch_size 16 --config.eqn_cfg.hess_diag_method sparse_stde --config.eqn_cfg.dim 100000 --config.eqn_cfg.name AllenCahnTwobody
```
To run other equations, change the flag `--config.eqn_cfg.name`. See the list of equation name in `stde/config.py`.

To get memory usage, add the following flags `--get_mem --n_runs 1 --config.test_cfg.n_points 200`, which runs a few epochs to determine the peak GPU memory usage.

You will find the experiment summary and saved checkpoints in the `_results` folder.

## Semilinear Parabolic PDEs
To run the 10kD Semilinear Heat equation described in Appendix I.2. with sparse STDE: 
``` shell
./scripts/semilinear_parabolic.sh --config.eqn_cfg.name SemilinearHeatTime --config.eqn_cfg.dim 10000 --config.eqn_cfg.hess_diag_method sparse_stde --config.eqn_cfg.rand_batch_size 16
```

## Weight sharing
To enable weight sharing described in Appendix G and I.3, add the `--config.model_cfg.block_size` flag. For example:
``` shell
./scripts/insep.sh --config.eqn_cfg.rand_batch_size 16 --config.eqn_cfg.hess_diag_method sparse_stde --config.eqn_cfg.dim 100000 --config.eqn_cfg.name AllenCahnTwobody --config.model_cfg.block_size 50
```

## High-order PDEs
To run the high-order low-dimensional PDEs described in Appendix I.4.1, change the '--config.eqn_cfg.name' flag accordingly. For example, to run the Gradient-enhanced 1D Korteweg-de Vries (g-KdV) equation:
``` shell
./scripts/insep.sh --config.eqn_cfg.rand_batch_size 0 --config.eqn_cfg.hess_diag_method sparse_stde --config.eqn_cfg.dim 1 --config.eqn_cfg.name highord1d 
```
To run the amortized gradient-enhanced PINN described in Appendix I.4.2, change the '--config.eqn_cfg.name' flag accordingly. For example, to run two-body Allen-Cahn equation with amoritzed gPINN:
``` shell
./scripts/insep.sh --config.eqn_cfg.rand_batch_size 16 --config.eqn_cfg.hess_diag_method sparse_stde --config.eqn_cfg.dim 100000 --config.eqn_cfg.name AllenCahnTwobodyG --config.eqn_cfg.gpinn_weight 0.1
```
