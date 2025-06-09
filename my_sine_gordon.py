import argparse
from collections import namedtuple
from functools import partial
from typing import Callable, NamedTuple, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random
import optax
from jax.experimental import jet
from jaxtyping import Array, Float
from tqdm import tqdm

from mamba import BidirectionalMamba, DiagnosticsConfig, MambaConfig, sample_domain_fn
from stde_mamba import BidirectionalSTDEMamba
from mamba_utils import create_mamba_model
from dataclasses import dataclass
from typing import Any
from flax.core import FrozenDict



parser = argparse.ArgumentParser(description='PINN Training')
parser.add_argument('--SEED', type=int, default=0)
parser.add_argument('--dim', type=int, default=10)  # dimension of the problem
parser.add_argument('--epochs', type=int, default=10000)  # Adam epochs
parser.add_argument('--lr', type=float, default=1e-3)  # Adam lr

# BidirectionalMamba specific arguments
parser.add_argument('--hidden_features', type=int, default=32)  # N (hidden dimension)
parser.add_argument('--expansion_factor', type=float, default=2.0)  # E
parser.add_argument('--dt_rank', default='auto')  # R
parser.add_argument('--activation', type=str, default='gelu', choices=['gelu', 'silu'])
parser.add_argument('--norm_type', type=str, default='layer', choices=['layer', 'rms'])
parser.add_argument('--dense_expansion', type=int, default=4)
parser.add_argument('--complement', action='store_true', help='whether to use complement mode')
parser.add_argument('--tie_in_proj', action='store_true', help='whether to tie input projections')
parser.add_argument('--tie_gate', action='store_true', help='whether to tie gates')
parser.add_argument('--concatenate_fwd_rev', action='store_true', help='whether to concatenate forward and reverse')

# Other training arguments
parser.add_argument('--N_f', type=int, default=100)  # num of residual points
parser.add_argument('--N_test', type=int, default=20000)  # num of test points
parser.add_argument('--test_batch_size', type=int, default=200)  # num of test points
parser.add_argument('--x_radius', type=float, default=1.0)
parser.add_argument('--rand_batch_size', type=int, default=10)
parser.add_argument('--sparse', action=argparse.BooleanOptionalAction,
                    help='whether to use sparse or dense stde')
parser.add_argument('--eval_every', type=int, default=1000)
args = parser.parse_args()
print(args)

###########################
# Seeded coefficient initialization using JAX
###########################
# Use JAX's PRNG to generate coefficients deterministically.
rng_key = jax.random.PRNGKey(args.SEED)
rng_key, subkey = jax.random.split(rng_key)
coeffs_ = jax.random.normal(subkey, (1, args.dim))

###########################
# STDE
###########################

def hess_trace(fn: Callable) -> Callable:
    """
    Given a function fn, returns a function that computes fn(x) and an
    estimated Hessian trace via jax.experimental.jet.jet.
    An explicit rng key is required.
    """
    def fn_trace(x_i, rng):
        if args.sparse:
            rng, subkey = jax.random.split(rng)
            idx_set = jax.random.choice(subkey, args.dim, shape=(args.rand_batch_size,), replace=False)
            rand_vec = jax.vmap(lambda i: jnp.eye(args.dim)[i])(idx_set)
        else:
            rng, subkey = jax.random.split(rng)
            rand_vec = 2 * (jax.random.randint(subkey, shape=(args.rand_batch_size, args.dim), minval=0, maxval=2) - 0.5)
        taylor_2 = lambda v: jet.jet(fun=fn, primals=(x_i,), series=((v, jnp.zeros(args.dim)),))
        f_vals, (_, hvps) = jax.vmap(taylor_2)(rand_vec)
        trace_est = jnp.mean(hvps)
        if args.sparse:
            trace_est *= args.dim
        return f_vals[0], trace_est
    return fn_trace

###########################
# Equation and helper functions
###########################

@partial(jax.jit, static_argnames=('u_fn',))
def SineGordon_op(x, u_fn: Callable, rng) -> Float[Array, "xt_dim"]:
    # Ensure x maintains (B, L, D) shape
    if len(x.shape) != 3:
        x = x.reshape(-1, 1, x.shape[-1])
    u_val = u_fn(x)
    u_, u_xx = hess_trace(u_fn)(x, rng)
    return u_xx + jnp.sin(u_)

def twobody_sol(x) -> Float[Array, "*batch"]:
    t1 = args.x_radius**2 - jnp.sum(x**2, -1)
    x1, x2 = x[..., :-1], x[..., 1:]
    t2 = coeffs_[:, :-1] * jnp.sin(x1 + jnp.cos(x2) + x2 * jnp.cos(x1))
    t2 = jnp.sum(t2, -1)
    u_exact = jnp.squeeze(t1 * t2)
    return u_exact

def twobody_lapl_analytical(x):
    coeffs = coeffs_[:, :-1]
    const_2 = 1
    u1 = 1 - jnp.sum(x**2)
    du1_dx = -2 * x
    d2u1_dx2 = -2

    x1, x2 = x[:-1], x[1:]
    coeffs = coeffs.reshape(-1)
    u2 = coeffs * jnp.sin(const_2 * (x1 + jnp.cos(x2) + x2 * jnp.cos(x1)))
    u2 = jnp.sum(u2)
    du2_dx_part1 = coeffs * jnp.cos(const_2 * (x1 + jnp.cos(x2) + x2 * jnp.cos(x1))) * const_2 * (1 - x2 * jnp.sin(x1))
    du2_dx_part2 = coeffs * jnp.cos(const_2 * (x1 + jnp.cos(x2) + x2 * jnp.cos(x1))) * const_2 * (-jnp.sin(x2) + jnp.cos(x1))
    du2_dx = jnp.zeros((args.dim,))
    du2_dx = du2_dx.at[:-1].add(du2_dx_part1)
    du2_dx = du2_dx.at[1:].add(du2_dx_part2)
    d2u2_dx2_part1 = -coeffs * jnp.sin(const_2 * (x1 + jnp.cos(x2) + x2 * jnp.cos(x1))) * const_2**2 * (1 - x2 * jnp.sin(x1))**2 + coeffs * const_2 * jnp.cos(const_2 * (x1 + jnp.cos(x2) + x2 * jnp.cos(x1))) * (- x2 * jnp.cos(x1))
    d2u2_dx2_part2 = -coeffs * jnp.sin(const_2 * (x1 + jnp.cos(x2) + x2 * jnp.cos(x1))) * const_2**2 * (-jnp.sin(x2) + jnp.cos(x1))**2 + coeffs * const_2 * jnp.cos(const_2 * (x1 + jnp.cos(x2) + x2 * jnp.cos(x1))) * (-jnp.cos(x2))
    d2u2_dx2 = jnp.zeros((args.dim,))
    d2u2_dx2 = d2u2_dx2.at[:-1].add(d2u2_dx2_part1)
    d2u2_dx2 = d2u2_dx2.at[1:].add(d2u2_dx2_part2)
    ff = u1 * d2u2_dx2 + 2 * du1_dx * du2_dx + u2 * d2u1_dx2
    ff = jnp.sum(ff)
    u = u1 * u2
    return ff, u

def SineGordon_twobody_inhomo_exact(x):
    u_exact_lapl, u_exact = twobody_lapl_analytical(x)
    g_exact = u_exact_lapl + jnp.sin(u_exact)
    return g_exact

@partial(jax.jit, static_argnames=('u_fn',))
def SineGordon_res_fn(x, u_fn: Callable, rng) -> Float[Array, "xt_dim"]:
    # Ensure x maintains (B, L, D) shape
    if len(x.shape) != 3:
        x = x.reshape(-1, 1, x.shape[-1])
    Lu = SineGordon_op(x, u_fn, rng)
    g = SineGordon_twobody_inhomo_exact(x)
    return Lu - g

###########################
# Loss and error functions (jitted)
###########################

from functools import partial

class PINN(nn.Module):
    """PINN implementation using Flax linen."""

    @nn.compact
    def __call__(self, x) -> jax.Array:
        # Ensure input shape is (B, L, D)
        B = x.shape[0]
        D = x.shape[-1]
        x = x.reshape(B, 1, D)
        y = self.net(x)
        return y

    def net(self, x: jax.Array) -> jax.Array:
        # Ensure input shape is (B, L, D)
        B = x.shape[0]
        D = x.shape[-1]
        x = x.reshape(B, 1, D)

        model_config = MambaConfig(
            hidden_features=args.hidden_features,
            expansion_factor=args.expansion_factor,
            dt_rank=args.dt_rank,
            activation=args.activation,
            norm_type=args.norm_type,
            mlp_layer=True,
            dense_expansion=args.dense_expansion,
            complement=args.complement,
            tie_in_proj=args.tie_in_proj,
            tie_gate=args.tie_gate,
            concatenate_fwd_rev=args.concatenate_fwd_rev,
            diagnostics=DiagnosticsConfig()
        )

        # Use STDE-compatible Mamba model for Taylor automatic differentiation
        mamba = create_mamba_model(model_config, use_stde_compatible=True)
        return mamba(x)

    @partial(jax.jit, static_argnames=('self',))
    def loss_fn(self, params, rng, x):
        def u_fn(x_in):
            B = x_in.shape[0]
            D = x_in.shape[-1]
            x_in = x_in.reshape(B, 1, D)
            return self.apply(params, x_in)

        B = x.shape[0]
        D = x.shape[-1]
        x = x.reshape(B, 1, D)

        rngs = jax.random.split(rng, x.shape[0])
        domain_res = jax.vmap(lambda x_i, r: SineGordon_res_fn(x_i, u_fn, r))(x, rngs)
        domain_loss = jnp.mean(domain_res**2)

        return domain_loss, {"domain_loss": domain_loss}

    @partial(jax.jit, static_argnames=('self',))
    def err_norms_fn(self, params, rng, x, y_true):
        """Compute error norms between predicted and true solutions."""
        # Fix: Remove extra params nesting
        y_pred = self.apply(params, x)
        err = y_true - y_pred

        l1_norm = jnp.mean(jnp.abs(err))
        l2_norm = jnp.mean(err**2)

        return l1_norm, l2_norm

    def init_for_multitransform(self):
        """Initialize model for multiple transformations."""
        return (
            self.__call__,
            namedtuple("PINN", ["u", "loss_fn", "err_norms_fn"])(
                self.__call__, self.loss_fn, self.err_norms_fn
            ),
        )

###########################
# Tabulate function
###########################

def tabulate_model():
    # Use parsed arguments for dimensions
    dummy_x, _, rng = sample_domain_fn(args.rand_batch_size, args.dim, args.x_radius, jax.random.PRNGKey(args.SEED))

    model_config = MambaConfig(
        hidden_features=args.hidden_features,
        expansion_factor=args.expansion_factor,
        dt_rank=args.dt_rank,
        activation=args.activation,
        norm_type=args.norm_type,
        mlp_layer=True,
        dense_expansion=args.dense_expansion,
        complement=args.complement,
        tie_in_proj=args.tie_in_proj,
        tie_gate=args.tie_gate,
        concatenate_fwd_rev=args.concatenate_fwd_rev,
        diagnostics=DiagnosticsConfig()
    )

    # Create both standard and STDE-compatible models for comparison
    model_standard = BidirectionalMamba(**vars(model_config))
    model_stde = BidirectionalSTDEMamba(**vars(model_config))

    print("Standard Mamba Model:")
    print(nn.tabulate(model_standard, rngs={"params": jax.random.PRNGKey(args.SEED)})(dummy_x))

    print("\nSTDE-Compatible Mamba Model:")
    print(nn.tabulate(model_stde, rngs={"params": jax.random.PRNGKey(args.SEED)})(dummy_x))

###########################
# Training
###########################

class TrainingState(NamedTuple):
    params: any
    opt_state: optax.OptState
    rng_key: jax.Array

def train():
    rng = jax.random.PRNGKey(args.SEED)
    dummy_x, _, rng = sample_domain_fn(args.test_batch_size, args.dim, args.x_radius, rng)
    print(dummy_x.shape)
    model = PINN()
    params = model.init(rng, dummy_x)

    # Create optimizer
    optimizer = optax.adam(learning_rate=args.lr)
    opt_state = optimizer.init(params)

    # Initialize training state
    state = TrainingState(params, opt_state, rng)

    @partial(jax.jit)
    def update(state: TrainingState):
        key, sample_key = jax.random.split(state.rng_key)
        x_sample, _, _ = sample_domain_fn(args.N_f, args.dim, args.x_radius, sample_key)

        # Ensure x_sample has correct shape (B, L, D)
        B = x_sample.shape[0]
        D = x_sample.shape[-1]
        x_sample = x_sample.reshape(B, 1, D)

        print(f"x_sample shape in update: {x_sample.shape}")  # Debug print

        (loss, aux), grad = jax.value_and_grad(model.loss_fn, has_aux=True)(
            state.params, key, x_sample)
        updates, new_opt_state = optimizer.update(grad, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)

        return loss, TrainingState(new_params, new_opt_state, key), aux

    err_norms_jit = jax.jit(lambda params, rng, x, y: model.err_norms_fn(params, rng, x, y))

    n_test_batches = args.N_test // args.test_batch_size
    assert n_test_batches > 0
    x_tests, y_trues = [], []
    for _ in tqdm(range(n_test_batches), desc="generating test data..."):
        x_test_i, _, rng = sample_domain_fn(args.test_batch_size, args.dim, args.x_radius, rng)
        y_true_i = twobody_sol(x_test_i)

        x_tests.append(jnp.array(x_test_i))
        y_trues.append(jnp.array(y_true_i))
    y_true = jnp.hstack(jnp.array(y_trues))
    y_true_l1, y_true_l2 = [jnp.linalg.norm(y_true, ord=ord) for ord in [1, 2]]

    iters = tqdm(range(args.epochs))
    for step in iters:
        loss, state, aux = update(state)
        if step % args.eval_every == 0:
            l1_total, l2_total_sqr = 0., 0.
            for i in range(n_test_batches):
                l1, l2_sqr = err_norms_jit(
                    state.params,
                    state.rng_key,
                    x_tests[i],
                    y_trues[i]
                )
                l1_total += l1
                l2_total_sqr += l2_sqr
            l1_rel = float(l1_total / y_true_l1)
            l2_rel = float(jnp.sqrt(l2_total_sqr) / y_true_l2)
            desc_str = f"{l1_rel=:.2E} | {l2_rel=:.2E} | {loss=:.2E} | "
            desc_str += " | ".join([f"{k}={v:.2E}" for k, v in aux.items() if v != 0.0])
            print(desc_str)

def test_bidirectional_mamba():
    """Test BidirectionalMamba initialization and basic components."""
    # Test parameters
    n_pts = 4
    dim = 1
    radius = 1.0

    # Initialize model
    model = PINN()

    # Create input using sample_domain_fn
    key = jax.random.PRNGKey(0)
    x, _, _ = sample_domain_fn(n_pts, dim, radius, key)

    # Verify input shape
    assert len(x.shape) == 3, f"Expected shape (B, L, D), got shape {x.shape}"
    assert x.shape[1] == 1, f"Expected L=1, got shape {x.shape}"

    # Initialize parameters
    variables = model.init(key, x)

    # Check if model components exist
    params = variables['params']
    assert 'BidirectionalMamba_0' in params, "Missing Mamba component"
    assert isinstance(params['BidirectionalMamba_0'], dict), "Mamba params should be a dict"
    mamba_params = params['BidirectionalMamba_0']
    assert 'in_proj' in mamba_params, "Missing input projection layer"
    assert 'out_proj' in mamba_params, "Missing output projection layer"

def test_bidirectional_mamba_forward():
    """Test forward pass of BidirectionalMamba."""
    # Test parameters
    n_pts = 4
    dim = 1
    radius = 1.0

    # Initialize model
    model = PINN()

    # Create input using sample_domain_fn
    key = jax.random.PRNGKey(0)
    x, _, _ = sample_domain_fn(n_pts, dim, radius, key)

    # Verify input shape
    assert len(x.shape) == 3, f"Expected shape (B, L, D), got shape {x.shape}"
    assert x.shape[1] == 1, f"Expected L=1, got shape {x.shape}"

    # Initialize parameters and run forward pass
    variables = model.init(key, x)
    output = model.apply(variables, x)

    # Test output shape
    # expected_shape = (x.shape[0],)  # Should be (B,) after squeeze
    # assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"

    # Test output type and values
    assert isinstance(output, jnp.ndarray), f"Expected output to be jnp.ndarray, but got {type(output)}"
    assert not jnp.any(jnp.isnan(output)), "Output contains NaN values"

def test_pinn_loss():
    """Test PINN loss computation."""
    # Test parameters
    n_pts = 4
    dim = 1
    radius = 1.0

    # Initialize model
    model = PINN()

    # Create input
    key = jax.random.PRNGKey(0)
    x, _, _ = sample_domain_fn(n_pts, dim, radius, key)

    # Initialize parameters
    variables = model.init(key, x)

    # Compute loss
    loss, aux = model.loss_fn(variables['params'], key, x)  # Note: passing params directly

    # Test loss type and value
    assert isinstance(loss, jnp.ndarray), f"Expected loss to be jnp.ndarray, but got {type(loss)}"
    assert loss.ndim == 0, f"Expected loss to be scalar, but got shape {loss.shape}"
    assert not jnp.isnan(loss), "Loss is NaN"
    assert loss >= 0, "Loss should be non-negative"

    # Test aux dictionary
    assert "domain_loss" in aux, "Missing domain_loss in aux dictionary"
    assert isinstance(aux["domain_loss"], jnp.ndarray), "domain_loss should be an array"

if __name__ == "__main__":
    # Run tests first
    tabulate_model()
    # print("Running tests...")
    # test_bidirectional_mamba()
    test_bidirectional_mamba_forward()
    # test_pinn_loss()
    # print("All tests passed!")

    # Then run training
    print("\nStarting training...")

    train()
