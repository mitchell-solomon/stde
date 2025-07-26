#!/usr/bin/env python
import argparse
import os
import json
import logging
import pickle
import time
from functools import partial
from typing import Callable, Tuple, Optional, get_args

import haiku as hk
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float
from jax import config
# config.update("jax_enable_x64", True)

import numpy as np

from flax.training import train_state
import flax.linen as nn

import optax
from tqdm import tqdm

import matplotlib.pyplot as plt
import pprint
import re

from stde.model import BidirectionalMamba, MambaConfig, DiagnosticsConfig, SSMConfig
from stde.mlp import MlpBackbone, MlpConfig
from stde.config import EqnConfig, ModelConfig, GDConfig
from stde import equations as eqns

def count_params(params):
    flat = jax.tree_util.tree_leaves(params)
    return int(sum(np.prod(p.shape) for p in flat))


def save_params(
    params,
    step: int,
    save_dir: str,
    save_every: int,
    epochs: int,
    *,
    is_best: bool = False,
):
    if step == 0:
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    if step == epochs - 1:
        with open(os.path.join(save_dir, "params_final.pkl"), "wb") as f:
            pickle.dump(params, f)
    if step % save_every == 0:
        with open(os.path.join(save_dir, f"params_{step}.pkl"), "wb") as f:
            pickle.dump(params, f)
    if is_best:
        with open(os.path.join(save_dir, "params_lowest_loss.pkl"), "wb") as f:
            pickle.dump(params, f)



# -----------------------------------------------------------------------------
# CLI arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="PINN Training with Bi-MAMBA")

# -- existing args --
parser.add_argument("--SEED", type=int, default=0)
parser.add_argument(
    "--dim",
    dest="spatial_dim",
    type=int,
    default=2,
    help="number of spatial dimensions of the problem",
)
parser.add_argument(
    "--spatial_dim",
    type=int,
    dest="spatial_dim",
    help=argparse.SUPPRESS,
)
parser.add_argument("--epochs", type=int, default=10000)
parser.add_argument("--eval_every", type=int, default=5000)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr_decay", type=str, default="cosine", choices=["none", "piecewise", "cosine", "linear", "exponential"])
parser.add_argument("--gamma", type=float, default=0.9995)
parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd", "rmsprop"])
parser.add_argument("--n_fgd_vec", type=int, default=0)
parser.add_argument("--N_test", type=int, default=2000)
parser.add_argument("--test_batch_size", type=int, default=20)
parser.add_argument("--N_val", type=int, default=200, help="size of validation set")
parser.add_argument("--val_batch_size", type=int, default=20, help="validation batch size")
parser.add_argument("--seq_len", type=int, default=3, help="sequence length for Bi-MAMBA")
parser.add_argument(
    "--use_seed_seq",
    type=bool,
    default=True,
    help="Sample each sequence around random seed points rather than independently",
)
parser.add_argument(
    "--seed_frac",
    type=float,
    default=0.01,
    help="Relative neighborhood size for --use_seed_seq as a fraction of the domain width",
)

parser.add_argument("--x_radius", type=float, default=1.0)
parser.add_argument("--x_ordering", type=str, choices=["none", "coordinate", "radial"], default="none", 
                    help="How to order your spatial sequence: `none` (leave random), `coordinate` (sort by x[0]), `radial` (sort by ∥x∥).")

parser.add_argument(
    "--hess_diag_method",
    type=str,
    choices=[
        "stacked",
        "forward",
        "sparse_stde",
        "dense_stde",
        "scan"
    ],
    default="sparse_stde",
    help="method for computing the Hessian diagonal",
)
parser.add_argument(
    "--stde_dist",
    type=str,
    choices=["normal", "rademacher"],
    default="rademacher",
    help="distribution for dense STDE",
)
parser.add_argument(
    "--backbone",
    type=str,
    choices=["Mamba", "MLP"],
    default="Mamba",
    help="network backbone to use",
)
parser.add_argument(
    "--no_stde",
    action="store_true",
    help="disable the STDE estimator",
)
parser.add_argument(
    "--ad_mode",
    type=str,
    choices=["forward", "reverse"],
    default="reverse",
    help="AD mode when STDE is disabled",
)
eqn_choices = get_args(EqnConfig.__annotations__["name"])
parser.add_argument(
    "--eqn_name",
    type=str,
    choices=eqn_choices,
    default="SineGordonTwobody",
    help="PDE to solve",
)

# numberof bidirectional mamba blocks
parser.add_argument("--num_mamba_blocks", type=int, default=1, help="number of bidirectional mamba blocks")
parser.add_argument("--block_size", type=int, default=-1, help="weight sharing block size for MLP")
parser.add_argument("--mlp_width", type=int, default=128, help="width of hidden layers in MLP backbone")
parser.add_argument("--mlp_depth", type=int, default=4, help="number of layers in MLP backbone")

# -- arguments for MambaConfig --
parser.add_argument("--hidden_features",    type=int,    default=8,      help="hidden_features in each Mamba block")
parser.add_argument("--expansion_factor",   type=float,  default=2.0,     help="expansion factor in Mamba MLP")
parser.add_argument("--dt_rank",            type=str,    default="auto", choices=["auto","full","low"], help="dt_rank setting")
parser.add_argument("--activation",         type=str,    default="tanh",  choices=["silu","relu","gelu","tanh", "wave"], help="activation fn")
parser.add_argument("--norm_type",          type=str,    default="none",  choices=["none","batch","layer", "group"], help="type of normalization")
parser.add_argument("--dense_expansion",    type=int,    default=2,       help="dense expansion ratio")
parser.add_argument("--complement",         action="store_true", help="use complement flag")
parser.add_argument("--tie_in_proj",        action="store_true", help="tie input projection weights")
parser.add_argument("--tie_gate",           action="store_true", help="tie gating weights")
parser.add_argument("--bidirectional",      action="store_true", help="use bidirectional mamba")

# -- arguments for DiagnosticsConfig --
parser.add_argument("--diag_skip",     action="store_true", help="enable skip diagnostics")
parser.add_argument("--diag_gate",     action="store_true", help="enable gate diagnostics")
parser.add_argument("--diag_gated",    action="store_true", help="enable gated diagnostics")
parser.add_argument("--diag_residual", action="store_true", help="enable residual diagnostics")

# -- arguments for SSMConfig --
parser.add_argument("--recursive_scan", action="store_true", help="use recursive scan")
parser.add_argument("--min_recursion_length", type=int, default=100, help="minimum recursion length")
parser.add_argument("--recursive_split", action="store_true", help="use recursive split")
parser.add_argument("--custom_vjp_scan", action="store_true", help="use custom vjp scan")
parser.add_argument("--ssm_activation", type=str, default="silu", choices=["silu","relu","gelu","tanh", "wave"], help="activation fn for SSM")

# -- run arguments --
parser.add_argument("--run_name", type=str, default="test_run")
parser.add_argument("--save_every", type=int, default=10000,
                    help="save parameters every n steps")

args = parser.parse_args()

if args.no_stde:
    if args.ad_mode == "forward":
        args.hess_diag_method = "forward"
    else:
        args.hess_diag_method = "stacked"

# derive rand_batch_size from dimension (order of magnitude lower)
rand_batch_size = max(1, args.spatial_dim // 10)
args.rand_batch_size = rand_batch_size


# ---------------------------------------------------------------------------
# logging setup
# ---------------------------------------------------------------------------
# create a dir with the run name where we save all results
save_dir = f"_results/{args.run_name}"
os.makedirs(save_dir, exist_ok=True)

log_file = os.path.join(save_dir, "train.log")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(log_file, mode="w")
fh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(fh)

# --- Spatial dimension requirements checks ---
# These equations require spatial_dim == 1
one_dim_eqns = ["SemilinearHeatTime", "SineGordonTime", "AllenCahnTime"]
allowed_dims = {10, 100, 1000, 10000}
if args.eqn_name in one_dim_eqns and args.spatial_dim not in allowed_dims:
    logger.error(f"ERROR: {args.eqn_name} only supports spatial_dim in {sorted(allowed_dims)}, but got spatial_dim={args.spatial_dim}")
    exit(1)
if "Threebody" in args.eqn_name and args.spatial_dim < 3:
    logger.error(f"ERROR: {args.eqn_name} requires spatial_dim >= 3, but got spatial_dim={args.spatial_dim}")
    exit(1)


# set up Haiku PRNG sequence for stde.operators
rng_seq = hk.PRNGSequence(args.SEED)
hk.next_rng_key = lambda: next(rng_seq)


np.random.seed(args.SEED)

# log args
args_str = pprint.pformat(vars(args), indent=2)
logger.info(f"Args:\n{args_str}\n")

# -----------------------------------------------------------------------------
# Hessian‐trace estimator
# -----------------------------------------------------------------------------

coeffs_ = np.random.randn(1, args.spatial_dim)

eqn_cfg = EqnConfig(
    name=args.eqn_name,
    dim=args.spatial_dim,
    max_radius=args.x_radius,
    rand_batch_size=rand_batch_size,
    hess_diag_method=args.hess_diag_method,
    stde_dist=args.stde_dist,
)
if args.backbone == "MLP":
    model_width = args.mlp_width
    model_depth = args.mlp_depth
else:
    model_width = args.hidden_features
    model_depth = args.num_mamba_blocks

model_cfg = ModelConfig(
    net=args.backbone,
    width=model_width,
    depth=model_depth,
    block_size=args.block_size,
)

eqn = getattr(eqns, eqn_cfg.name)
if eqn.random_coeff:
    eqn_cfg.coeffs = coeffs_
else:
    eqn_cfg.coeffs = None

# sampler for boundary points using equation-specific sampling
sample_domain_fn = eqn.get_sample_domain_fn(eqn_cfg)
sample_boundary_fn = sample_domain_fn

# estimate typical domain extents to scale neighbourhood sampling
span_rng = jax.random.PRNGKey(args.SEED + 1)
x_tmp, t_tmp, _, _, _ = sample_domain_fn(1024, 8, span_rng)
x_span = float(jnp.max(x_tmp) - jnp.min(x_tmp))
if eqn.time_dependent and t_tmp is not None:
    t_span = float(jnp.max(t_tmp) - jnp.min(t_tmp))
else:
    t_span = 0.0

seed_x_sigma = args.seed_frac * x_span
seed_t_sigma = args.seed_frac * t_span

# -----------------------------------------------------------------------------
# Domain sampler utilising equation specific sampler
# -----------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("batch_size", "seq_len", "use_seed"))
def sample_domain_seq_fn(
    batch_size: int,
    rng: jax.Array,
    seq_len: int,
    use_seed: bool,
) -> Tuple[jnp.ndarray, jax.Array]:
    """Sample ``batch_size``\*``seq_len`` points and reshape to sequences.

    If the equation is time dependent, the returned sequence dimension
    corresponds to the temporal axis. The sampled ``x`` and ``t`` are
    concatenated such that the model receives ``(x, t)`` as features."""

    if use_seed and not eqn.is_traj:
        # sample seed points then draw a short sequence around each seed
        x_seed, t_seed, _, _, rng = sample_domain_fn(batch_size, 0, rng)
        keys = jax.random.split(
            rng, 3 if eqn.time_dependent and t_seed is not None else 2
        )
        rng = keys[-1]
        x_noise = seed_x_sigma * jax.random.normal(
            keys[0], (batch_size, seq_len, args.spatial_dim)
        )
        x_seq = x_seed[:, None, :] + x_noise
        if eqn.time_dependent and t_seed is not None:
            t_noise = seed_t_sigma * jax.random.normal(
                keys[1], (batch_size, seq_len, 1)
            )
            t_seq = t_seed[:, None, :] + t_noise
            sort_idx = jnp.argsort(t_seq[..., 0], axis=1)
            x_seq = jnp.take_along_axis(x_seq, sort_idx[..., None], axis=1)
            t_seq = jnp.take_along_axis(t_seq, sort_idx[..., None], axis=1)
            x_seq = jnp.concatenate([x_seq, t_seq], axis=-1)
        return x_seq, rng
    else:
        if eqn.is_traj:
            x, t, _, _, rng = sample_domain_fn(batch_size, seq_len - 1, rng)
        else:
            x, t, _, _, rng = sample_domain_fn(batch_size * seq_len, 0, rng)

        # reshape to sequences
        x_seq = x.reshape((batch_size, seq_len, -1))
        if eqn.time_dependent and t is not None:
            t_seq = t.reshape((batch_size, seq_len, -1))
            # order each sequence by increasing time so that seq axis is temporal
            sort_idx = jnp.argsort(t_seq[..., 0], axis=1)
            x_seq = jnp.take_along_axis(x_seq, sort_idx[..., None], axis=1)
            t_seq = jnp.take_along_axis(t_seq, sort_idx[..., None], axis=1)
            x_seq = jnp.concatenate([x_seq, t_seq], axis=-1)
        return x_seq, rng

if eqn.time_dependent:
    sol_fn = lambda xt: eqn.sol(
        xt[..., : args.spatial_dim], xt[..., args.spatial_dim :], eqn_cfg
    )

    def residual_fn(xt, u_fn: Callable, key: jax.Array) -> Float[Array, "xt_dim"]:
        x_part = xt[..., : args.spatial_dim]
        t_part = xt[..., args.spatial_dim :]
        res = eqn.res(
            x_part,
            t_part,
            lambda xi, ti: u_fn(jnp.concatenate([xi, ti], axis=-1)),
            eqn_cfg,
            key,
        )
        if isinstance(res, tuple):
            res = res[0]
        return res
else:
    sol_fn = lambda x: eqn.sol(x, None, eqn_cfg)

    def residual_fn(x, u_fn: Callable, key: jax.Array) -> Float[Array, "xt_dim"]:
        res = eqn.res(x, None, lambda xi, _t: u_fn(xi), eqn_cfg, key)
        if isinstance(res, tuple):
            res = res[0]
        return res

def eval_model(model, params, eqn, eqn_cfg, seqs, truths, y_true_l1, y_true_l2, args):
    """Evaluate the model on a dataset."""
    if not eqn.is_traj:
        l1_total, l2_total_sqr = 0.0, 0.0
        for b in range(seqs.shape[0]):
            x_seq = seqs[b]
            y_true = truths[b]
            y_pred = model.apply({"params": params}, x_seq)
            if y_pred.ndim == 3 and y_pred.shape[-1] == 1:
                y_pred = y_pred.squeeze(-1)
            err = y_pred - y_true
            l1_total += jnp.sum(jnp.abs(err))
            l2_total_sqr += jnp.sum(err**2)

        l1_rel = float(l1_total / y_true_l1)
        l2_rel = float(jnp.sqrt(l2_total_sqr) / y_true_l2)

    else:
        xt_zero = jnp.zeros((1, args.seq_len, args.spatial_dim + 1))
        y_pred = model.apply({"params": params}, xt_zero)[0, 0]
        y_true = eqn.sol(jnp.zeros((args.spatial_dim,)), jnp.zeros((1,)), eqn_cfg)
        l1_rel = float(jnp.abs(y_pred - y_true) / jnp.abs(y_true))
        l2_rel = l1_rel


    return l1_rel, l2_rel



# -----------------------------------------------------------------------------
# Training state
# -----------------------------------------------------------------------------
class MambaTrainState(train_state.TrainState):
    rng: jax.Array


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # rng setup
    master_rng = jax.random.PRNGKey(args.SEED)
    rng_train, rng_test = jax.random.split(master_rng)

    
    # make a class for the PINN, supporting either an MLP or Bi-MAMBA backbone
    class PINN(nn.Module):
        eqn: eqns.Equation
        eqn_cfg: EqnConfig
        model_cfg: ModelConfig
        backbone: str = "Mamba"
        @nn.compact
        def __call__(self, x):
            # Ensure input shape is (B, L, D)
            B = x.shape[0]
            D = x.shape[-1]
            L = x.shape[-2]
            x_in = x

            if self.backbone == "Mamba":
                # instantiate Mamba
                mamba_cfg = MambaConfig(
                    hidden_features       = args.hidden_features,
                    expansion_factor      = args.expansion_factor,
                    dt_rank               = args.dt_rank,
                    activation            = args.activation,
                    norm_type             = args.norm_type,
                    dense_expansion       = args.dense_expansion,
                    complement            = args.complement,
                    tie_in_proj           = args.tie_in_proj,
                    tie_gate              = args.tie_gate,
                    radius                = args.x_radius,
                    bidirectional         = args.bidirectional,
                    diagnostics           = DiagnosticsConfig(
                        skip     = args.diag_skip,
                        gate     = args.diag_gate,
                        gated    = args.diag_gated,
                        residual = args.diag_residual,
                    ),
                )

                ssm_cfg = SSMConfig(
                    recursive_scan      = args.recursive_scan,
                    min_recursion_length   = args.min_recursion_length,
                    recursive_split     = args.recursive_split,
                    custom_vjp_scan     = args.custom_vjp_scan,
                    activation           = args.ssm_activation,
                )
                # Apply the Mamba model
                for _ in range(self.model_cfg.depth):
                    x = BidirectionalMamba(**vars(mamba_cfg), ssm_args=vars(ssm_cfg))(x)

                x_out = nn.Dense(args.dense_expansion*D, name="mlp", kernel_init=nn.initializers.lecun_normal())(x)
                x_out = nn.gelu(x_out)
                x_out = nn.Dense(1, name="mlp_proj", kernel_init=nn.initializers.lecun_normal())(x_out)
                x_out = x_out.squeeze(-1)
            else:
                # instantiate MLP
                mlp_cfg = MlpConfig(
                    width=self.model_cfg.width,
                    depth=self.model_cfg.depth,
                    w_init=self.model_cfg.w_init,
                    b_init=self.model_cfg.b_init,
                    block_size=self.model_cfg.block_size,
                    use_conv=self.model_cfg.use_conv,
                    hidden_sizes=self.model_cfg.hidden_sizes,
                    activation=args.activation,
                )
                x_out = MlpBackbone(mlp_cfg, time_dependent=self.eqn.time_dependent)(x)
            
            # enforce PDE-specific boundary condition
            if self.eqn.time_dependent:
                x_part, t_part = x_in[..., : args.spatial_dim], x_in[..., args.spatial_dim :]
                x_out = self.eqn.enforce_boundary(x_part, t_part, x_out, self.eqn_cfg)
            else:
                x_out = self.eqn.enforce_boundary(x_in, None, x_out, self.eqn_cfg)

            return x_out

        def tabulate_model(self,
                           n_pts: int = 4,
                           dim: int = args.spatial_dim,
                           radius: float = args.x_radius,
                           seq_len: int = args.seq_len,
                           rng: Optional[jax.Array] = None):
            """Print a tabulated summary of the model architecture."""

            if rng is None:
                rng = jax.random.PRNGKey(0)

            x, _ = sample_domain_seq_fn(
                batch_size=n_pts,
                rng=rng,
                seq_len=seq_len,
                use_seed=args.use_seed_seq,
            )

            table = nn.tabulate(
                self,
                rngs={"params": rng},
                mutable=["params", "diagnostics", "intermediates"],
            )(x)
            return table

        def loss_fn(self, params, rng, mamba_apply, sample_domain_seq_fn, sample_boundary_fn, rand_batch_size, args, eqn, eqn_cfg):
            batch_rng, new_rng = jax.random.split(rng)
            x_seq, batch_rng = sample_domain_seq_fn(
                batch_size=rand_batch_size,
                rng=batch_rng,
                seq_len=args.seq_len,
                use_seed=args.use_seed_seq,
            )
            def y_at_l(xt_i, l, full_seq):
                seq2 = lax.dynamic_update_slice(full_seq, xt_i[None, :], (l, 0))
                y2 = mamba_apply({"params": params}, seq2[None, ...]).squeeze(0)
                return y2[l]
            def residuals_for_one_sequence(full_seq, key):
                L = full_seq.shape[0]
                def one_step_res(l, xt_l, key):
                    def u_fn(xt_i):
                        return y_at_l(xt_i, l, full_seq)
                    res_val = residual_fn(xt_l, u_fn, key)
                    return res_val, key
                keys = jax.random.split(key, L)
                resids, _ = jax.vmap(one_step_res, in_axes=(0, 0, 0), out_axes=(0, 0))(
                    jnp.arange(L), full_seq, keys
                )
                return resids, _
            outer_keys = jax.random.split(batch_rng, x_seq.shape[0])
            all_resids, _ = jax.vmap(
                residuals_for_one_sequence, in_axes=(0, 0), out_axes=(0, 0)
            )(x_seq, outer_keys)
            domain_loss = jnp.mean(all_resids ** 2)
            _, _, x_b, t_b, batch_rng = sample_boundary_fn(
                rand_batch_size, rand_batch_size, batch_rng
            )
            if eqn.time_dependent:
                xt_b = jnp.concatenate([x_b, t_b], axis=-1)
                u_b = mamba_apply({"params": params}, xt_b[:, None, :]).squeeze()
            else:
                u_b = mamba_apply({"params": params}, x_b[:, None, :]).squeeze()
            g_b = eqn.boundary_cond(x_b, t_b, eqn_cfg)
            boundary_loss = jnp.mean((g_b - u_b) ** 2)
            loss = (
                eqn_cfg.domain_weight * domain_loss
                + eqn_cfg.boundary_weight * boundary_loss
            )
            return loss, new_rng

        def err_norms_fn(self, params, x_seq, y_true, mamba_apply):
            y_pred = mamba_apply({"params": params}, x_seq)
            if y_pred.ndim == 3 and y_pred.shape[-1] == 1:
                y_pred = y_pred.squeeze(-1)
            err = y_true - y_pred
            l1 = jnp.abs(err).sum()
            l2 = (err ** 2).sum()
            return l1, l2

    
    logger.info(f"Instantiating model with backbone {args.backbone}")   
    # And then proceed to instantiate your model as before:
    model = PINN(eqn=eqn, eqn_cfg=eqn_cfg, model_cfg=model_cfg, backbone=args.backbone)

    
    # initialize parameters on a dummy sequence
    rng_train, init_rng = jax.random.split(rng_train)
    x_dummy, rng = sample_domain_seq_fn(
        batch_size=2,
        rng=init_rng,
        seq_len=args.seq_len,
        use_seed=args.use_seed_seq,
    )
    flax_vars = model.init(init_rng, x_dummy)
    logger.info("Model instantiated")

    # log input and output shapes
    logger.info(f"Input shape: {x_dummy.shape}")
    logger.info(f"Output shape: {model.apply(flax_vars, x_dummy).shape}")
    

    # Tabulate the model architecture for reference
    table = model.tabulate_model()

    def strip_ansi(text):
        ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        return ansi_escape.sub('', text)

    table = strip_ansi(table)
    logger.info(f"Model architecture:{table}")

    cfg = GDConfig(
        lr=args.lr,
        lr_decay=args.lr_decay,
        gamma=args.gamma,
        optimizer=args.optimizer,
        epochs=args.epochs,
        n_fgd_vec=args.n_fgd_vec,
    )

    # init exponential decay learning rate schedule
    if cfg.lr_decay == "piecewise":
        lr = optax.piecewise_constant_schedule(
        init_value=cfg.lr,
        boundaries_and_scales={
            int(cfg.epochs * 0.5): 0.5,
            int(cfg.epochs * 0.75): 0.25,
            int(cfg.epochs * 0.825): 0.125,
        }
        )
    elif cfg.lr_decay == "linear":
        lr = optax.linear_schedule(
        init_value=cfg.lr, end_value=0., transition_steps=cfg.epochs
        )
    elif cfg.lr_decay == "exponential":
        lr = optax.exponential_decay(
        init_value=cfg.lr, transition_steps=cfg.epochs, decay_rate=cfg.gamma
        )
    elif cfg.lr_decay == "cosine":
        lr = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1.0,
        warmup_steps=50,
        decay_steps=cfg.epochs - 50,
        end_value=0.0,
        )
    else:
        lr = cfg.lr
    
    optimizer = getattr(optax, cfg.optimizer)(learning_rate=lr)

    state = MambaTrainState.create(
        apply_fn=model.apply,
        params=flax_vars["params"],
        tx=optimizer,
        rng=rng_train,
    )
    num_params = count_params(state.params)
    logger.info(f"num params: {num_params}")
    gpu_mems = []

    # prepare test and validation sets (once)
    test_seqs = test_truths = None
    val_seqs = val_truths = None
    if not eqn.is_traj:
        n_test_batches = args.N_test // args.test_batch_size
        test_seqs = []
        test_truths = []

        # add tqdm prgress bar to test set generation
        for _ in tqdm(range(n_test_batches)):
            rng_test, sample_rng = jax.random.split(rng_test)
            x_test_seq, _ = sample_domain_seq_fn(
                batch_size=args.test_batch_size,
                rng=sample_rng,
                seq_len=args.seq_len,
                use_seed=args.use_seed_seq,
            )
            # collapse batch & seq dims to analytical solver
            B, L, D = x_test_seq.shape
            x_flat = x_test_seq.reshape((B * L, D))
            y_flat = jax.vmap(sol_fn)(x_flat)
            test_seqs.append(x_test_seq)
            test_truths.append(y_flat.reshape((B, L)))
        test_seqs = jnp.stack(test_seqs)       # (n_batches, B, L, D)
        test_truths = jnp.stack(test_truths)   # (n_batches, B, L)

        # validation set generation
        n_val_batches = args.N_val // args.val_batch_size
        val_seqs = []
        val_truths = []
        for _ in tqdm(range(n_val_batches)):
            rng_test, sample_rng = jax.random.split(rng_test)
            x_val_seq, _ = sample_domain_seq_fn(
                batch_size=args.val_batch_size,
                rng=sample_rng,
                seq_len=args.seq_len,
                use_seed=args.use_seed_seq,
            )
            B, L, D = x_val_seq.shape
            x_flat = x_val_seq.reshape((B * L, D))
            y_flat = jax.vmap(sol_fn)(x_flat)
            val_seqs.append(x_val_seq)
            val_truths.append(y_flat.reshape((B, L)))
        val_seqs = jnp.stack(val_seqs)
        val_truths = jnp.stack(val_truths)

        # flatten all test ground-truth values into a single vector
        y_true_all = test_truths.reshape(-1)
        y_val_all = val_truths.reshape(-1)

        # L1 norm of the entire test set
        y_true_l1 = float(jnp.sum(jnp.abs(y_true_all)))
        y_val_l1 = float(jnp.sum(jnp.abs(y_val_all)))

        # L2 norm of the entire test set
        y_true_l2 = float(jnp.linalg.norm(y_true_all))
        y_val_l2 = float(jnp.linalg.norm(y_val_all))
    else:
        # reference value only defined at x=0,t=0
        y_ref = eqn.sol(jnp.zeros((args.spatial_dim,)), jnp.zeros((1,)), eqn_cfg)
        y_true_l1 = jnp.abs(y_ref)
        y_true_l2 = jnp.abs(y_ref)
        y_val_l1 = y_true_l1
        y_val_l2 = y_true_l2

    @jax.jit
    def train_step(state: MambaTrainState) -> MambaTrainState:
        (loss, new_rng), grads = jax.value_and_grad(model.loss_fn, has_aux=True)(
            state.params, state.rng, model.apply, sample_domain_seq_fn, sample_boundary_fn, rand_batch_size, args, eqn, eqn_cfg
        )
        state = state.apply_gradients(grads=grads, rng=new_rng)
        return state, loss, grads
    
    losses = []
    best_loss = float("inf")
    iters = tqdm(range(args.epochs), desc=f"training eqn {args.eqn_name}\n")
    epoch_times = []
    start_time = time.time()
    for step in iters:
        step_start = time.time()
        state, train_loss, grads = train_step(state)
        train_loss_f = float(train_loss)
        losses.append(train_loss_f)
        is_best = False
        if train_loss_f < best_loss:
            best_loss = train_loss_f
            is_best = True
        save_params(
            state.params,
            step,
            save_dir,
            args.save_every,
            args.epochs,
            is_best=is_best,
        )

        if step % args.eval_every == 0 or step == args.epochs - 1:
            l1_rel, l2_rel = eval_model(
                model, state.params, eqn, eqn_cfg, val_seqs, val_truths, y_val_l1, y_val_l2, args)
            grad_norm = float(jnp.sqrt(
                sum(jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grads))
            ))
            desc_str = (
                f"iter={step} | "
                f"l1_rel={l1_rel:.2e} | l2_rel={l2_rel:.2e} | "
                f"loss={train_loss:.2e} | grad_norm={grad_norm:.2e}"
            )
            # save desc_str to log file
            logger.info(desc_str)
            iters.set_description(desc_str)
            mem_stats = jax.local_devices()[0].memory_stats()
            peak_mem = mem_stats['peak_bytes_in_use'] / 1024**2
            gpu_mems.append(peak_mem)

        epoch_times.append(time.time() - step_start)

    # read iter/s from log file
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
        iter_per_s = float(lines[-3].strip().split(', ')[-1].split('it/s')[0])
    except Exception:
        iter_per_s = 0.0

    total_time = time.time() - start_time
    time_per_epoch = sum(epoch_times) / len(epoch_times)

    # --- Plot training loss curve ---
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Training Step')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curve')
    plt.savefig(f"{save_dir}/training_loss_curve.png", dpi=300)
    # plt.show()

    # --- Final evaluation on the full test set ---
    best_params_path = os.path.join(save_dir, "params_lowest_loss.pkl")
    if os.path.exists(best_params_path):
        with open(best_params_path, "rb") as f:
            best_params = pickle.load(f)
        state = state.replace(params=best_params)
    print("\n=== Final evaluation on test set ===")
    l1_rel, l2_rel = eval_model(
        model,
        state.params,
        eqn,
        eqn_cfg,
        test_seqs,
        test_truths,
        y_true_l1,
        y_true_l2,
        args,
    )
    if not eqn.is_traj:
        print(f"Final → l1_rel={l1_rel:.3e} | l2_rel={l2_rel:.3e}")
        logger.info(f"Final → l1_rel={l1_rel:.3e} | l2_rel={l2_rel:.3e}")
    else:
        print(f"Final → l1_rel={l1_rel:.3e}")
        logger.info(f"Final → l1_rel={l1_rel:.3e}")

    with open(f"{save_dir}/final_eval_results.json", "w") as f:
        json.dump(
            {
                "l1_rel": l1_rel,
                "l2_rel": l2_rel,
                "iter_per_s": iter_per_s,
                "time_per_epoch": time_per_epoch,
                "total_time": total_time,
                "peak_gpu_mem": max(gpu_mems),
                "num_params": num_params,
                "final_loss": float(losses[-1]) if losses else 0.0,
                "best_loss": best_loss,
            },
            f,
            indent=2,
        )


    # --- Plotting ---
    def _title_for_eqn(name: str, cfg: EqnConfig) -> str:
        params = []
        if name == "HJB_LQG":
            params.append(f"mu={cfg.mu}")
        if name == "HJB_LIN":
            params.append(f"c={cfg.c}")
        if name == "BSB":
            params.append(f"sigma={cfg.sigma}")
            params.append(f"r={cfg.r}")
        if hasattr(cfg, "max_radius"):
            params.append(f"max_r={cfg.max_radius}")
        parts = ", ".join(params)
        bc = eqn.boundary_cond.__name__
        return f"{name} ({parts})\nBC: {bc}"

    def plot_solution(x_flat, u_true, u_pred,
                      dim: int | None = None,
                      cmap: str = 'viridis',
                      eqn_name: str = args.eqn_name):
        """Plot true, predicted and diff solution with PDE meta info."""

        diff = u_pred - u_true

        if eqn.time_dependent or eqn.is_traj:
            xi = np.array(x_flat[:, 0])
            yi = np.array(x_flat[:, -1])
            xlabel, ylabel = 'x', 't'

            vmin = min(np.min(u_true), np.min(u_pred))
            vmax = max(np.max(u_true), np.max(u_pred))
            # For diff, use symmetric colorbar spanning the superset range
            diff_abs_max = max(abs(vmin), abs(vmax), np.max(np.abs(diff)))
            diff_vmin, diff_vmax = -diff_abs_max, diff_abs_max

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            sc0 = axes[0].scatter(xi, yi, c=u_true, cmap=cmap,
                                  vmin=vmin, vmax=vmax, s=20)
            axes[0].set_title('True u')
            axes[0].set_xlabel(xlabel)
            axes[0].set_ylabel(ylabel)

            sc1 = axes[1].scatter(xi, yi, c=u_pred, cmap=cmap,
                                  vmin=vmin, vmax=vmax, s=20)
            axes[1].set_title('Predicted u')
            axes[1].set_xlabel(xlabel)
            axes[1].set_ylabel(ylabel)

            sc2 = axes[2].scatter(xi, yi, c=diff, cmap='coolwarm', s=20,
                                 vmin=diff_vmin, vmax=diff_vmax)
            axes[2].set_title('Difference')
            axes[2].set_xlabel(xlabel)
            axes[2].set_ylabel(ylabel)

            fig.colorbar(sc0, ax=axes[:2], shrink=0.8, pad=0.02, label='u')
            fig.colorbar(sc2, ax=axes[2], shrink=0.8, pad=0.02, label='Δu')

        else:
            if dim is None:
                D = x_flat.shape[1] if x_flat.ndim > 1 else 1
            else:
                D = dim

            if D > 2:
                print(f"Warning: embedding dim={D} > 2; plotting only first two dims.")
                D = 2

            if D == 1:
                x = x_flat if x_flat.ndim == 1 else x_flat[:, 0]
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                axes[0].plot(x, u_true, '.', label='true')
                axes[0].set_title('True u(x)')
                axes[0].set_xlabel('x')
                axes[0].set_ylabel('u')

                axes[1].plot(x, u_pred, '.', color='C1')
                axes[1].set_title('Predicted u(x)')
                axes[1].set_xlabel('x')
                axes[1].set_ylabel('u')

                axes[2].plot(x, diff, '.', color='C2')
                axes[2].set_title('Difference')
                axes[2].set_xlabel('x')
                axes[2].set_ylabel('Δu')

            elif D == 2:
                xi = x_flat[:, 0]
                yi = x_flat[:, 1]

                vmin = min(np.min(u_true), np.min(u_pred))
                vmax = max(np.max(u_true), np.max(u_pred))
                diff_abs_max = max(abs(vmin), abs(vmax), np.max(np.abs(diff)))
                diff_vmin, diff_vmax = -diff_abs_max, diff_abs_max

                fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                                       subplot_kw={'aspect': 'equal'})

                sc0 = axes[0].scatter(xi, yi, c=u_true, cmap=cmap,
                                      vmin=vmin, vmax=vmax, s=20)
                axes[0].set_title('True u')
                axes[0].set_xlabel('dim0')
                axes[0].set_ylabel('dim1')

                sc1 = axes[1].scatter(xi, yi, c=u_pred, cmap=cmap,
                                      vmin=vmin, vmax=vmax, s=20)
                axes[1].set_title('Predicted u')
                axes[1].set_xlabel('dim0')
                axes[1].set_ylabel('dim1')

                sc2 = axes[2].scatter(xi, yi, c=diff, cmap='coolwarm', s=20,
                                     vmin=diff_vmin, vmax=diff_vmax)
                axes[2].set_title('Difference')
                axes[2].set_xlabel('dim0')
                axes[2].set_ylabel('dim1')

                fig.colorbar(sc0, ax=axes[:2], shrink=0.8, pad=0.02, label='u')
                fig.colorbar(sc2, ax=axes[2], shrink=0.8, pad=0.02, label='Δu')

            else:
                raise ValueError('plot only supports 1D or 2D embeddings for visualization')

        fig.suptitle(_title_for_eqn(eqn_name, eqn_cfg))
        plt.savefig(f"{save_dir}/{eqn_name}_solution.png", dpi=300)
    def plot_all_solutions(test_seqs, test_truths, params, model, dim=None,
                           xlabel='x', ylabel='u', cmap='viridis',
                           eqn_name: str = args.eqn_name):
        """
        Run the model over every test batch, flatten, and plot true vs. predicted.
        
        Args:
        test_seqs:   array (n_batches, B, L, D)
        test_truths: array (n_batches, B, L)
        params:      your trained params to pass into model.apply
        model:       your model instance
        dim:         spatial dimension (if None, inferred from D)
        """
        # flatten batches
        n_batches, B, L, D = test_seqs.shape
        if dim is None:
            dim = D

        # reshape to (n_batches*B, L, D) for a big batch
        seqs_flat = test_seqs.reshape((n_batches * B, L, D))
        # run model in one go
        u_pred_seq = model.apply({"params": params},
                                seqs_flat)      # → (n_batches*B, L, 1) or (nB, L)
        
        # flatten spatial points and truths
        x_flat = test_seqs.reshape((n_batches * B * L, D))
        u_true = test_truths.reshape((n_batches * B * L,))
        u_pred = u_pred_seq.reshape((n_batches * B * L,))

        # convert to NumPy for plotting
        x_flat_np = np.array(x_flat)
        u_true_np  = np.array(u_true)
        u_pred_np  = np.array(u_pred)

        # now call the existing plot fn
        plot_solution(x_flat_np,
                      u_true_np,
                      u_pred_np,
                      dim=dim,
                      cmap=cmap,
                      eqn_name=eqn_name)

    if not eqn.is_traj:
        plot_all_solutions(test_seqs,
                           test_truths,
                           state.params,
                           model,
                           dim=args.spatial_dim)
    def visualize_sequences(x_seq, x_ordering="none"):
        """
        Scatter‐plot each 2D “sequence” in x_seq, coloring by sequence index.
        If x_ordering="radius" or "coordinate", also draw lines to show the ordering.

        Args:
        x_seq:       numpy array of shape (B, L, 2)
        x_ordering:  "none", "coordinate", or "radial"
        """
        B, L, D = x_seq.shape
        assert D == 2, "visualize_sequences only supports 2D inputs"

        cmap = plt.get_cmap("tab10")
        plt.figure(figsize=(6,6))

        for b in range(B):
            pts = x_seq[b]  # (L, 2)

            # apply ordering if requested
            if x_ordering == "coordinate":
                idx = np.argsort(pts[:, 0])
                pts = pts[idx]
            elif x_ordering == "radial":
                idx = np.argsort(np.linalg.norm(pts, axis=1))
                pts = pts[idx]
            # else "none": leave as is

            color = cmap(b % 10)
            plt.scatter(pts[:,0], pts[:,1], color=color, s=30, label=f"seq {b}")

            # if ordered, draw lines to reveal the path
            if x_ordering in ("coordinate", "radial"):
                plt.plot(pts[:,0], pts[:,1], color=color, alpha=0.5)

        plt.title(f"Sequences on 2D ball (ordering={x_ordering})")
        plt.gca().set_aspect("equal", "box")
        plt.xlabel("x₀")
        plt.ylabel("x₁")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/sequences_{x_ordering}.png", dpi=300)
    if not eqn.is_traj:
        seq_vis = test_seqs[0]
        if seq_vis.shape[-1] > 2:
            seq_vis = seq_vis[..., :2]
        visualize_sequences(seq_vis, x_ordering=args.x_ordering)

if __name__ == "__main__":
    main()
