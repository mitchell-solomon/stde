#!/usr/bin/env python
import argparse
import os
import json
from functools import partial
from typing import Callable, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float
from jax import config

config.update("jax_enable_x64", True)
import numpy as np

from flax.training import train_state
import flax.linen as nn


import optax
from tqdm import tqdm

import matplotlib.pyplot as plt
from pprint import pprint


from my_mamba import BidirectionalMamba, MambaConfig, DiagnosticsConfig, SSMConfig
from stde.config import EqnConfig
from stde.equations import (
    twobody_sol as eq_twobody_sol,
    SineGordon_twobody_inhomo_exact,
    threebody_sol as eq_threebody_sol,
    SineGordon_threebody_inhomo_exact,
)
from stde.operators import hte
# -----------------------------------------------------------------------------
# CLI arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="PINN Training with Bi-MAMBA")

# -- existing args --
parser.add_argument("--SEED", type=int, default=0)
parser.add_argument("--dim", type=int, default=2, help="spatial dimension of the problem")
parser.add_argument("--epochs", type=int, default=10000)
parser.add_argument("--eval_every", type=int, default=10000000)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--N_test", type=int, default=2000)
parser.add_argument("--test_batch_size", type=int, default=20)
parser.add_argument("--seq_len", type=int, default=3, help="sequence length for Bi-MAMBA")
parser.add_argument("--rand_batch_size", type=int, default=16)
parser.add_argument("--x_radius", type=float, default=1.0)
parser.add_argument("--x_ordering", type=str, choices=["none", "coordinate", "radial"], default="radial", help="How to order your spatial sequence: `none` (leave random), `coordinate` (sort by x[0]), `radial` (sort by ∥x∥).")

parser.add_argument(
    "--sampling_mode", type=str,
    choices=["random", "grid", "radial"],
    default="random",
    help="How to sample points: `random`=Monte‐Carlo; `grid`=even grid; "
         "`radial`=random‐start radial lines"
)

parser.add_argument(
    "--hess_diag_method",
    type=str,
    choices=[
        "stacked",
        "forward",
        "sparse_stde",
        "dense_stde",
        "scan",
        "folx",
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
    "--problem",
    type=str,
    choices=["twobody", "threebody"],
    default="twobody",
    help="choose sine-gordon variant",
)

# numberof bidirectional mamba blocks
parser.add_argument("--num_mamba_blocks", type=int, default=1, help="number of bidirectional mamba blocks")

# -- arguments for MambaConfig --
parser.add_argument("--hidden_features",    type=int,    default=64,      help="hidden_features in each Mamba block")
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

args = parser.parse_args()
pprint(args)

# set up Haiku PRNG sequence for stde.operators
rng_seq = hk.PRNGSequence(args.SEED)
hk.next_rng_key = lambda: next(rng_seq)


np.random.seed(args.SEED)

# create a dir with the run name where we save all results
save_dir = f"_results/{args.run_name}"
os.makedirs(save_dir, exist_ok=True)

# save args as a json file to save dir
with open(f"{save_dir}/args.json", "w") as f:
    json.dump(vars(args), f, indent=2)


# -----------------------------------------------------------------------------
# Domain sampler: can return single points or time sequences of points
# -----------------------------------------------------------------------------

@partial(jax.jit, static_argnames=(
    "batch_size","seq_len","radius","dim",
    "sampling_mode","x_ordering"
))
def sample_domain_fn(batch_size: int,
                     rng: jax.Array,
                     radius: float,
                     dim: int,
                     seq_len: int,
                     sampling_mode: str = "random",
                     x_ordering:   str = "none",
                     ) -> Tuple[jnp.ndarray, jax.Array]:
    # 1) Sampling
    if sampling_mode == "random":
        # your Monte‐Carlo interior sampler
        keys = jax.random.split(rng, seq_len + 1)
        out = []
        for i in range(seq_len):
            r = jax.random.uniform(keys[i], (batch_size, 1), minval=0.0, maxval=radius)
            x = jax.random.normal(keys[i+1], (batch_size, dim))
            x = x / jnp.linalg.norm(x, axis=-1, keepdims=True) * r
            out.append(x)
        x_seq, new_rng = jnp.stack(out, axis=1), keys[0]

    elif sampling_mode == "grid":
        # even grid projected onto sphere
        n = int(round(batch_size ** (1/ dim)))
        coords = jnp.linspace(-radius, radius, n)
        meshes = jnp.meshgrid(*([coords]*dim), indexing="ij")
        pts = jnp.stack([m.flatten() for m in meshes], axis=-1)  # (B, D)
        norm = jnp.linalg.norm(pts, axis=-1, keepdims=True)
        pts = pts / norm * radius
        x_seq = jnp.broadcast_to(pts[:,None,:], (batch_size, seq_len, dim))
        new_rng = rng

    elif sampling_mode == "radial":
        # Random‐start & end radial segments:
        # each sequence is a straight ray from r_start → r_end, with RANDOM spacing.

        # 1) split RNG for four draws
        rng, sub_r0, sub_r1, sub_dir, sub_t = jax.random.split(rng, 5)

        # 2) sample two radii ∈ [0, R/4] and [R, R] (so r0<r1)
        r0 = jax.random.uniform(sub_r0, (batch_size,1), minval=0.0,        maxval=radius/4)
        r1 = jax.random.uniform(sub_r1, (batch_size,1), minval=radius/4,   maxval=radius)
        r_start = jnp.minimum(r0, r1)   # (B,1)
        r_end   = jnp.maximum(r0, r1)   # (B,1)

        # 3) sample seq_len random t’s ∈ [0,1], then sort to enforce monotonicity
        t_rand    = jax.random.uniform(sub_t, (batch_size, seq_len, 1))
        t_sorted  = jnp.sort(t_rand, axis=1)             # (B, L, 1)

        # 4) compute per‐step radii: r = (1−t)·r_start + t·r_end
        radii = (1.0 - t_sorted) * r_start[:,None,:] \
              +            t_sorted * r_end[:,None,:]   # (B, L, 1)

        # 5) sample random directions on unit‐sphere
        dirs = jax.random.normal(sub_dir, (batch_size, dim))
        dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)  # (B, D)

        # 6) assemble sequences
        x_seq, new_rng = dirs[:, None, :] * radii, rng           # → (B, L, D)
        return x_seq, new_rng

    else:
        raise ValueError(f"Unknown sampling_mode: {sampling_mode}")

    # 2) Ordering
    if x_ordering == "coordinate":
        idx = jnp.argsort(x_seq[...,0], axis=1)
    elif x_ordering == "radial":
        idx = jnp.argsort(jnp.linalg.norm(x_seq, axis=-1), axis=1)
    elif x_ordering == "none":
        return x_seq, new_rng
    else:
        raise ValueError(f"Unknown x_ordering: {x_ordering}")

    batch_idx = jnp.arange(batch_size)[:,None]
    x_seq = x_seq[batch_idx, idx]
    return x_seq, new_rng

# -----------------------------------------------------------------------------
# Hessian‐trace estimator
# -----------------------------------------------------------------------------

# STDE using utilities from stde.operators

def hess_trace(fn: Callable, cfg: EqnConfig) -> Callable:
    """Return a Hessian-trace estimator for ``fn`` using :func:`hte`."""

    ht = hte(fn, cfg, argnums=0)

    def fn_trace(x_i):
        _, f_val, trace_est = ht(x_i)
        return f_val, trace_est

    return fn_trace

def SineGordon_op(x, u_fn: Callable) -> Float[Array, "xt_dim"]:
    r"""
    .. math::
    \nabla u(x) + sin(u(x)) = g(x)
    """
    # run the Hessian‐trace estimator
    u_, lap = hess_trace(u_fn, eqn_cfg)(x)
    return lap + jnp.sin(u_)


coeffs_ = np.random.randn(1, args.dim)

eqn_cfg = EqnConfig(
    dim=args.dim,
    max_radius=args.x_radius,
    rand_batch_size=args.rand_batch_size,
    hess_diag_method=args.hess_diag_method,
    stde_dist=args.stde_dist,
)
eqn_cfg.coeffs = coeffs_
if args.problem == "twobody":
    sol_fn = lambda x: eq_twobody_sol(x, None, eqn_cfg)
    g_exact_fn = lambda x: SineGordon_twobody_inhomo_exact(x, eqn_cfg)
else:
    sol_fn = lambda x: eq_threebody_sol(x, None, eqn_cfg)
    g_exact_fn = lambda x: SineGordon_threebody_inhomo_exact(x, eqn_cfg)

def SineGordon_res_fn(x, u_fn: Callable) -> Float[Array, "xt_dim"]:
    r"""
    .. math::
    L u(x) = g(x)
    """
    Lu = SineGordon_op(x, u_fn)
    g = g_exact_fn(x)
    return Lu - g

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

    # instantiate Bi-MAMBA
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
    class ZeroOnUnitBall(nn.Module):
        @nn.compact
        def __call__(self, x_in, u_val, radius=1.0):
            return (radius**2 - jnp.sum(x_in**2, -1)) * u_val
    
    # make a class for the PINN, which is a stack of Bi-MAMBA blocks
    class PINN(nn.Module):
        @nn.compact
        def __call__(self, x):
            # Ensure input shape is (B, L, D)
            B = x.shape[0]
            D = x.shape[-1]
            L = x.shape[-2]
            x_in = x

            # Apply the Mamba model
            for i in range(args.num_mamba_blocks):
                x = BidirectionalMamba(**vars(mamba_cfg), ssm_args=vars(ssm_cfg))(x)
            
            x_out = nn.Dense(args.dense_expansion*D, name="mlp", kernel_init=nn.initializers.lecun_normal())(x)
            x_out = nn.gelu(x_out)
            x_out = nn.Dense(1, name="mlp_proj", kernel_init=nn.initializers.lecun_normal())(x_out) # (B, L, D) --> (B, L, 1)
            x_out = ZeroOnUnitBall(name="zero_unit_ball")(x_in, x_out.squeeze(-1), radius=args.x_radius)

            return x_out

    # And then proceed to instantiate your model as before:
    mamba = PINN()

    # initialize parameters on a dummy sequence
    rng_train, init_rng = jax.random.split(rng_train)
    x_dummy, rng = sample_domain_fn(batch_size=2,
                                    rng=init_rng,
                                    radius=args.x_radius,
                                    dim=args.dim,
                                    seq_len=args.seq_len,
                                    x_ordering=args.x_ordering,
                                    sampling_mode=args.sampling_mode)
    flax_vars = mamba.init(init_rng, x_dummy)

    # print input and output shapes
    print("Input shape:", x_dummy.shape)
    print("Output shape:", mamba.apply(flax_vars, x_dummy).shape)
    
    # init exponential decay learning rate schedule
    
    lr = args.lr
    # lr = optax.exponential_decay(
    #     init_value=args.lr,
    #     transition_steps=args.epochs,
    #     transition_begin=args.epochs//3,
    #     decay_rate=0.0001
    # )
    
    # # plot lr vs epoch
    # epochs = list(range(args.epochs))
    # lrs = [lr(epoch) for epoch in epochs]
    # plt.plot(epochs, lrs)
    # plt.xlabel('Epoch')
    # plt.ylabel('Learning Rate')
    # plt.yscale('log')
    # plt.title('Learning Rate Schedule')
    # plt.show()


    # create optimizer + train state
    optimizer = optax.adam(learning_rate=lr)
    state = MambaTrainState.create(
        apply_fn=mamba.apply,
        params=flax_vars["params"],
        tx=optimizer,
        rng=rng_train,
    )

    # prepare test set (once)
    n_test_batches = args.N_test // args.test_batch_size
    test_seqs = []
    test_truths = []

    for _ in range(n_test_batches):
        rng_test, sample_rng = jax.random.split(rng_test)
        x_test_seq, _ = sample_domain_fn(args.test_batch_size,
                                        rng=sample_rng,
                                        radius=args.x_radius,
                                        dim=args.dim,
                                        seq_len=args.seq_len,
                                        x_ordering=args.x_ordering,
                                        sampling_mode=args.sampling_mode)
        # collapse batch & seq dims to analytical solver
        B, L, D = x_test_seq.shape
        x_flat = x_test_seq.reshape((B * L, D))
        y_flat = jax.vmap(sol_fn)(x_flat)
        test_seqs.append(x_test_seq)
        test_truths.append(y_flat.reshape((B, L)))
    test_seqs = jnp.stack(test_seqs)       # (n_batches, B, L, D)
    test_truths = jnp.stack(test_truths)   # (n_batches, B, L)
    
    # flatten all test ground-truth values into a single vector
    y_true_all = test_truths.reshape(-1)  

    # L1 norm of the entire test set
    y_true_l1 = float(jnp.sum(jnp.abs(y_true_all)))     

    # L2 norm of the entire test set
    y_true_l2 = float(jnp.linalg.norm(y_true_all))   

    @jax.jit
    def train_step(state: MambaTrainState) -> MambaTrainState:
        # 1) split off one rng for sampling, one to carry forward
        batch_rng, next_rng = jax.random.split(state.rng)
        x_seq, batch_rng = sample_domain_fn(
            batch_size=args.rand_batch_size,
            rng=batch_rng,
            radius=args.x_radius,
            dim=args.dim,
            seq_len=args.seq_len,
            x_ordering=args.x_ordering,
            sampling_mode=args.sampling_mode
        )  # x_seq: (B, L, D)

        def loss_fn(params, rng):
            """
            Computes the mean-squared PDE residual loss for a batch given parameters.

            Returns:
            loss:       scalar MSE of all residuals
            new_rng:    PRNGKey to carry forward
            """
            # 1) split RNG for sampling vs. carry-forward
            batch_rng, new_rng = jax.random.split(rng)

            # 2) sample a fresh batch of input sequences
            x_seq, batch_rng = sample_domain_fn(
                batch_size=args.rand_batch_size,
                rng=batch_rng,
                radius=args.x_radius,
                dim=args.dim,
                seq_len=args.seq_len,
                x_ordering=args.x_ordering,
                sampling_mode=args.sampling_mode
            )  # x_seq shape: (B, L, D)

            # 3) helper: model output at time index l
            def y_at_l(x_i, l, full_seq):
                # replace the l-th entry of full_seq with x_i
                seq2 = lax.dynamic_update_slice(full_seq, x_i[None, :], (l, 0))
                # apply your Bi-MAMBA with the passed-in params
                y2 = mamba.apply({"params": params}, seq2[None, ...]).squeeze(0)
                return y2[l]

            # 4) compute the vector of residuals for one sequence
        def residuals_for_one_sequence(full_seq):
            L = full_seq.shape[0]

            def one_step_res(l, x_l):
                # build a u_fn that closes over `params` and this full_seq
                def u_fn(xi):
                    return y_at_l(xi, l, full_seq)

                # STDE Hessian-trace at x_l
                y0, lap = hess_trace(u_fn, eqn_cfg)(x_l)

                # analytic inhomogeneity
                g0 = g_exact_fn(x_l)

                return lap + jnp.sin(y0) - g0

            resids = jax.vmap(one_step_res)(jnp.arange(L), full_seq)
            return resids

        all_resids = jax.vmap(residuals_for_one_sequence)(x_seq)  # (B, L)

            # 6) mean-squared residual loss
            loss = jnp.mean(all_resids ** 2)
            return loss, new_rng

        # inside your jitted train_step
        (loss, new_rng), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, state.rng
        )
        state = state.apply_gradients(grads=grads, rng=new_rng)

        return state, loss, grads
    losses = []
    for step in tqdm(range(args.epochs), desc="training"):
        # let train_step now return (new_state, train_loss, grads)
        state, train_loss, grads = train_step(state)
        losses.append(train_loss)


        if step % args.eval_every == 0:
            # compute relative test errors
            l1_total, l2_total_sqr = 0., 0.
            for b in range(test_seqs.shape[0]):
                x_seq = test_seqs[b]
                y_true = test_truths[b]        # (B, L)
                y_pred = mamba.apply({"params": state.params}, x_seq) # .squeeze(-1)
                err = y_pred - y_true
                l1_total += jnp.sum(jnp.abs(err))
                l2_total_sqr += jnp.sum(err**2)
            l1_rel = float(l1_total / y_true_l1)
            l2_rel = float(jnp.sqrt(l2_total_sqr) / y_true_l2)

            # gradient norm
            grad_norm = jnp.sqrt(
                sum(jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grads))
            )

            print(f"step={step:4d} | "
                f"train_loss={train_loss:.3e} | "
                f"grad_norm={grad_norm:.3e} | "
                f"l1_rel={l1_rel:.3e} | "
                f"l2_rel={l2_rel:.3e}")
    
    # --- Plot training loss curve ---
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Training Step')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curve')
    plt.savefig(f"{save_dir}/training_loss_curve.png", dpi=300)
    # plt.show()

    # --- Final evaluation on the full test set ---
    print("\n=== Final evaluation on test set ===")
    l1_total, l2_total_sqr = 0.0, 0.0
    for b in range(test_seqs.shape[0]):
        x_seq = test_seqs[b]               # (B, L, D)
        y_true = test_truths[b]            # (B, L)

        y_pred = mamba.apply({"params": state.params}, x_seq)
        # drop trailing dim if present
        if y_pred.ndim == 3 and y_pred.shape[-1] == 1:
            y_pred = y_pred.squeeze(-1)

        err = y_pred - y_true
        l1_total       += jnp.sum(jnp.abs(err))
        l2_total_sqr   += jnp.sum(err**2)

    l1_rel = float(l1_total / y_true_l1)
    l2_rel = float(jnp.sqrt(l2_total_sqr) / y_true_l2)
    print(f"Final → l1_rel={l1_rel:.3e} | l2_rel={l2_rel:.3e}")

    # save final eval results to a json file
    with open(f"{save_dir}/final_eval_results.json", "w") as f:
        json.dump({
            "l1_rel": l1_rel,
            "l2_rel": l2_rel
        }, f, indent=2)


    # --- Plotting ---


    def plot_sine_gordon_solution(x_flat, u_true, u_pred,
                                dim: int = None,
                                xlabel: str = 'x',
                                ylabel: str = 'u',
                                cmap: str = 'viridis'):
        """
        Plot true vs. predicted Sine–Gordon solutions.
        If the embedding dimension D > 2, only the first two dims are used for plotting.
        
        Args:
        x_flat:   array of shape (N, D) or (N,) for 1D
        u_true:   array of shape (N,)
        u_pred:   array of shape (N,)
        dim:      override spatial dimension (1 or 2). If None, inferred from x_flat.
        """
        # Infer dimension if not provided
        if dim is None:
            if x_flat.ndim == 1:
                D = 1
            else:
                D = x_flat.shape[1]
        else:
            D = dim

        # Cap at 2 dimensions for plotting
        if D > 2:
            print(f"Warning: embedding dim={D} > 2; plotting only first two dims.")
            D = 2

        # 1D plot
        if D == 1:
            # collapse to 1D coordinate
            x = x_flat if x_flat.ndim == 1 else x_flat[:, 0]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            ax1.plot(x, u_true, '.', label='true')
            ax1.set_title('True u(x)')
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel(ylabel)
            ax2.plot(x, u_pred, '.', label='pred', color='C1')
            ax2.set_title('Predicted u(x)')
            ax2.set_xlabel(xlabel)
            ax2.set_ylabel(ylabel)

        # 2D scatter with shared colorbar
        elif D == 2:
            # pick first two coords
            xi = x_flat[:, 0]
            yi = x_flat[:, 1]

            # common color limits
            vmin = min(np.min(u_true), np.min(u_pred))
            vmax = max(np.max(u_true), np.max(u_pred))

            fig, (ax1, ax2) = plt.subplots(
                1, 2, figsize=(12, 5), subplot_kw={'aspect': 'equal'}
            )

            sc1 = ax1.scatter(xi, yi,
                            c=u_true, cmap=cmap,
                            vmin=vmin, vmax=vmax,
                            s=20)
            ax1.set_title('True u')
            ax1.set_xlabel('dim0')
            ax1.set_ylabel('dim1')

            sc2 = ax2.scatter(xi, yi,
                            c=u_pred, cmap=cmap,
                            vmin=vmin, vmax=vmax,
                            s=20)
            ax2.set_title('Predicted u')
            ax2.set_xlabel('dim0')
            ax2.set_ylabel('dim1')

            # one shared colorbar anchored to the bottom outside the x axis
            cbar = fig.colorbar(sc1, ax=[ax1, ax2],
                                shrink=0.8, pad=0.02,
                                label=ylabel,
                                orientation='vertical',
                                location='right',
                                )

            # plt.tight_layout()

        else:
            raise ValueError("plot only supports 1D or 2D embeddings for visualization")

        # plt.tight_layout()
        plt.savefig(f"{save_dir}/sine_gordon_solution.png", dpi=300)
        # plt.show()

    def plot_all_sine_gordon_solutions(test_seqs, test_truths, params, mamba, dim=None,
                                    xlabel='x', ylabel='u', cmap='viridis'):
        """
        Run the model over every test batch, flatten, and plot true vs. predicted.
        
        Args:
        test_seqs:   array (n_batches, B, L, D)
        test_truths: array (n_batches, B, L)
        params:      your trained params to pass into mamba.apply
        mamba:       your BidirectionalMamba instance
        dim:         spatial dimension (if None, inferred from D)
        """
        # flatten batches
        n_batches, B, L, D = test_seqs.shape
        if dim is None:
            dim = D

        # reshape to (n_batches*B, L, D) for a big batch
        seqs_flat = test_seqs.reshape((n_batches * B, L, D))
        # run model in one go
        u_pred_seq = mamba.apply({"params": params},
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
        plot_sine_gordon_solution(x_flat_np,
                                u_true_np,
                                u_pred_np,
                                dim=dim,
                                xlabel=xlabel,
                                ylabel=ylabel,
                                cmap=cmap)
    
    plot_all_sine_gordon_solutions(test_seqs,
                               test_truths,
                               state.params,
                               mamba,
                               dim=args.dim)
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
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/sequences_{x_ordering}.png", dpi=300)
        # plt.show()
    visualize_sequences(test_seqs[0], x_ordering=args.x_ordering)



if __name__ == "__main__":
    main()
