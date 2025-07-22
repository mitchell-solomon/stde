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

> **Note**
> Some "threebody" equations (e.g. `SineGordonThreebody`) require a spatial
> dimension of at least 3. Running them with `--dim < 3` will produce invalid
> reference solutions and lead to infinite relative errors.

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

## Code overview
This repository provides utilities for physics-informed neural network (PINN)
experiments built around stochastic Taylor derivative estimators (STDE).  The
core functionality lives inside the `stde/` package while scripts such as
`train_bimamba.py` offer ready-to-run training setups.

`train_bimamba.py` trains a PINN composed of bidirectional MAMBA blocks. Key
methods include:

- **`sample_domain_seq_fn`** – samples sequences of domain points via the
  equation-specific samplers. With the `--use_seed_seq` flag sequences are

  drawn in a small neighbourhood around random seed points. The neighbourhood
  size scales with the domain width and is controlled by `--seed_frac`
  (default 1 %). If the equation has

  a temporal dimension the sequence axis corresponds to time.
- **`residual_fn`** – computes residuals by delegating to the selected equation
  object.
- **`BiMambaPINN`** – a Flax module stacking Bi‑MAMBA blocks and enforcing
  boundary conditions.
- **`train_step`** – performs a single optimization step (JAX jit/vmap) and
  computes domain and boundary losses.

Results and model checkpoints are stored under `_results/`.

## Available PDEs
The table below summarises the equations defined in `EqnConfig` together with
information relevant to the implementation.

| Name | Time dep.? | Brownian traj.? | Equation | Boundary condition |
|-----|:----------:|:---------------:|---------|-------------------|
| `HJB_LIN` | ✓ | ✗ | \(u_t + \Delta u - d^{-1}\|\nabla u\|^c = -2\) | \(u(x,T)=\sum_i x_i\) |
| `HJB_LQG` | ✓ | ✗ | \(u_t + \Delta u - \mu\|\nabla u\|^2 = 0\) | \(u(x,T)=\log((1+\|x\|^2)/2)\) |
| `BSB` | ✓ | ✗ | \(u_t + \tfrac{1}{2}\sigma^2 x^2\cdot\nabla^2 u - r(u- x\cdot\nabla u)=0\) | \(u(x,T)=\sum_i x_i^2\) |
| `Wave` | ✓ | ✗ | \(u_{tt}-\Delta u=0\) | \(u(x,0)=\sum_i\sinh x_i,\ u_t(x,0)=0\) |
| `Poisson` | ✗ | ✗ | \(\Delta u = g(x)\) | \(u(x)=\sum_i e^{x_i}/d\) |
| `PoissonHouman` | ✗ | ✗ | \(\Delta u = g(x)\) | same as `Poisson` |
| `PoissonTwobody` | ✗ | ✗ | \(\Delta u = g_{2b}(x)\) | Dirichlet on unit sphere |
| `PoissonTwobodyG` | ✗ | ✗ | same as above with gPINN | Dirichlet on unit sphere |
| `PoissonThreebody` | ✗ | ✗ | \(\Delta u = g_{3b}(x)\) | Dirichlet on unit sphere |
| `AllenCahnTwobody` | ✗ | ✗ | \(\Delta u + u - u^3 = g_{2b}(x)\) | Dirichlet on unit sphere |
| `AllenCahnTwobodyG` | ✗ | ✗ | same as above with gPINN | Dirichlet on unit sphere |
| `AllenCahnThreebody` | ✗ | ✗ | \(\Delta u + u - u^3 = g_{3b}(x)\) | Dirichlet on unit sphere |
| `SineGordonTwobody` | ✗ | ✗ | \(\Delta u + \sin u = g_{2b}(x)\) | Dirichlet on unit sphere |
| `SineGordonTwobodyG` | ✗ | ✗ | same as above with gPINN | Dirichlet on unit sphere |
| `SineGordonThreebody` | ✗ | ✗ | \(\Delta u + \sin u = g_{3b}(x)\) | Dirichlet on unit sphere |
| `AllenCahnTime` | ✓ | ✓ | \(u_t + \Delta u + u - u^3 = 0\) | terminal \(\arctan(\max_i x_i)\) |
| `SineGordonTime` | ✓ | ✓ | \(u_t + \Delta u + d\,\sin(u/d)=0\) | terminal \(5/(10+2\|x\|^2)\,d\) |
| `SemilinearHeatTime` | ✓ | ✓ | \(u_t + \Delta u + \tfrac{1-u^2}{1+u^2}=0\) | terminal \(5/(10+2\|x\|^2)\) |
| `KdV2d` | ✓ | ✗ | high order KdV-type | \(u(x,0)=\sum_i\sinh x_i\) |
| `highord1d` | ✓ | ✗ | high-order 1D PDE | \(u(x,0)=\sum_i\sinh x_i\) |

