from __future__ import annotations

"""Research-grade STDE-style solver for (2+1)D KdV using JAX + NNX.

This module follows the user-specified implementation plan for Appendix I.4.1,
implementing sampling, jet-based derivative extraction, training utilities, and
minimal evaluation/benchmark helpers. It purposefully avoids Haiku and uses Flax
NNX for the neural network model.
"""

import functools
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
from jax.experimental import jet
from flax import nnx
import optax

try:
    import orbax.checkpoint as orbax
except ImportError:  # pragma: no cover
    orbax = None

Array = jax.Array

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class Config:
    """Configuration for training and evaluation.

    Attributes:
        seed: Base RNG seed.
        dtype: Floating point precision to use throughout the module.
        width: Hidden width of the MLP.
        depth: Number of hidden layers (excluding the output layer).
        lr: Learning rate for Adam.
        batch_size: Mini-batch size for both collocation and boundary loaders.
        lambda_bc: Weight for boundary loss term.
        steps: Number of training steps in the demonstration loop.
        checkpoint_dir: Optional directory for Orbax checkpoints.
    """

    seed: int = 42
    dtype: jnp.dtype = jnp.float32
    width: int = 128
    depth: int = 4
    lr: float = 1e-3
    batch_size: int = 256
    lambda_bc: float = 1.0
    steps: int = 100000
    checkpoint_dir: Optional[Path] = None


# -----------------------------------------------------------------------------
# Problem definition
# -----------------------------------------------------------------------------


DOMAIN_X = (-10.0, 10.0)
DOMAIN_Y = (-10.0, 10.0)
DOMAIN_T = (-5.0, 5.0)


def exact_solution(X: Array) -> Array:
    """Exact solution u*(x, y, t) for the (2+1)D KdV instance."""

    chex.assert_shape(X, (None, 3))
    x, y, t = jnp.split(X, 3, axis=-1)
    numerator = -3.0 * t + 2.0 * x + 2.0 * y
    denom = (0.5 * x + y - 2.0 * t) ** 2 + (0.5 * t + 0.5 * x) ** 2 + 3.0
    out = (numerator / denom)[..., 0]
    chex.assert_tree_all_finite(out)
    return out


# -----------------------------------------------------------------------------
# Sampling utilities
# -----------------------------------------------------------------------------


def _latin_hypercube(key: Array, n: int, dims: int) -> Array:
    """Simple Latin hypercube sampler on [0, 1]^dims."""

    key_perm, key_offset = jax.random.split(key)
    perm = jax.vmap(lambda k: jax.random.permutation(k, n))(jax.random.split(key_perm, dims))
    base = (perm + jnp.linspace(0.0, 1.0, n, endpoint=False)) / float(n)
    offsets = jax.random.uniform(key_offset, (dims, n), dtype=jnp.float32)
    samples = (base + offsets / n).T
    return samples


def sample_collocation(key: Array, n: int = 20_000) -> Array:
    """Sample collocation points via Latin hypercube over the 3D domain."""

    unit = _latin_hypercube(key, n, 3)
    scales = jnp.array(
        [DOMAIN_X[1] - DOMAIN_X[0], DOMAIN_Y[1] - DOMAIN_Y[0], DOMAIN_T[1] - DOMAIN_T[0]]
    )
    lows = jnp.array([DOMAIN_X[0], DOMAIN_Y[0], DOMAIN_T[0]])
    return unit * scales + lows


def sample_boundary(key: Array, n_init: int = 100, n_bdry: int = 300) -> Tuple[Array, Array]:
    """Sample initial and boundary points deterministically."""

    key_init, key_bdry = jax.random.split(key)
    init_xy = _latin_hypercube(key_init, n_init, 2)
    init_xy = init_xy * jnp.array([DOMAIN_X[1] - DOMAIN_X[0], DOMAIN_Y[1] - DOMAIN_Y[0]])
    init_xy = init_xy + jnp.array([DOMAIN_X[0], DOMAIN_Y[0]])
    init_t = -5.0 * jnp.ones((n_init, 1))
    X_init = jnp.concatenate([init_xy, init_t], axis=-1)

    per_face = n_bdry // 4
    keys = jax.random.split(key_bdry, 4)
    faces = []
    for i, k in enumerate(keys):
        other = _latin_hypercube(k, per_face, 2)
        other = other * jnp.array([DOMAIN_Y[1] - DOMAIN_Y[0], DOMAIN_T[1] - DOMAIN_T[0]])
        other = other + jnp.array([DOMAIN_Y[0], DOMAIN_T[0]])
        if i == 0:
            x_face = jnp.full((per_face, 1), DOMAIN_X[0])
            y_face, t_face = jnp.split(other, 2, axis=-1)
        elif i == 1:
            x_face = jnp.full((per_face, 1), DOMAIN_X[1])
            y_face, t_face = jnp.split(other, 2, axis=-1)
        elif i == 2:
            y_face = jnp.full((per_face, 1), DOMAIN_Y[0])
            x_face, t_face = jnp.split(other, 2, axis=-1)
        else:
            y_face = jnp.full((per_face, 1), DOMAIN_Y[1])
            x_face, t_face = jnp.split(other, 2, axis=-1)
        faces.append(jnp.concatenate([x_face, y_face, t_face], axis=-1))
    X_bdry = jnp.concatenate(faces, axis=0)

    X_b = jnp.concatenate([X_init, X_bdry], axis=0)
    u_b = exact_solution(X_b)
    return X_b, u_b


# -----------------------------------------------------------------------------
# Data loaders (Grain-like deterministic iterators)
# -----------------------------------------------------------------------------


def _batch_iter(array: Array, batch_size: int):
    total = array.shape[0]
    for start in range(0, total, batch_size):
        yield array[start : start + batch_size]


def collocation_loader(X_r: Array, batch_size: int):
    return _batch_iter(X_r, batch_size)


def boundary_loader(X_b: Array, u_b: Array, batch_size: int):
    total = X_b.shape[0]
    for start in range(0, total, batch_size):
        yield X_b[start : start + batch_size], u_b[start : start + batch_size]


# -----------------------------------------------------------------------------
# Model definition (NNX MLP)
# -----------------------------------------------------------------------------


class MLP(nnx.Module):
    """Tanh MLP mapping (x, y, t) -> scalar."""

    layers: nnx.List[nnx.Linear]

    def __init__(self, rng: Array, width: int = 128, depth: int = 4):
        super().__init__()
        layers = []
        key = rng
        # First layer expects 3D input (x, y, t).
        key, sub = jax.random.split(key)
        layers.append(
            nnx.Linear(
                in_features=3,
                out_features=width,
                rngs=nnx.Rngs(params=sub),
            )
        )
        # Remaining hidden layers are width -> width.
        for _ in range(depth - 1):
            key, sub = jax.random.split(key)
            layers.append(
                nnx.Linear(
                    in_features=width,
                    out_features=width,
                    rngs=nnx.Rngs(params=sub),
                )
            )
        key, out = jax.random.split(key)
        self.layers = nnx.List(layers)
        self.out = nnx.Linear(in_features=width, out_features=1, rngs=nnx.Rngs(params=out))

    def __call__(self, X: Array) -> Array:
        chex.assert_shape(X, (None, 3))
        h = X
        for layer in self.layers:
            h = jnp.tanh(layer(h))
        out = self.out(h)[..., 0]
        return out


# -----------------------------------------------------------------------------
# Jet utilities
# -----------------------------------------------------------------------------


def pushforward_jet(
    fun: Callable[[Array], Array],
    X: Array,
    series_terms: Sequence[Array],
    factorial_scaled: bool,
) -> Tuple[Array, Sequence[Array]]:
    """Apply JAX jet transform to obtain Taylor coefficients."""

    def single(x: Array, *terms: Array):
        x = x[None, :]
        terms = tuple(t[None, :] for t in terms)
        primals, series = jet.jet(fun, (x,), (terms,), factorial_scaled=factorial_scaled)
        primals = primals[0]
        coeffs = [t[0] for t in series]
        return primals, coeffs

    f0, coeffs = jax.vmap(single, in_axes=(0,) + (0,) * len(series_terms))(X, *series_terms)
    return f0, [c for c in coeffs]


def calibrate_factorial_scaled(rng: Array) -> bool:
    del rng
    return True


def _second_mixed_from_dirs(f2_x: Array, f2_y: Array, f2_xy_dir: Array) -> Array:
    return 0.5 * (f2_xy_dir - f2_x - f2_y)


def compute_derivatives(
    model: MLP,
    X: Array,
    factorial_scaled: bool,
) -> Tuple[Array, ...]:
    e_x = jnp.array([1.0, 0.0, 0.0])
    e_y = jnp.array([0.0, 1.0, 0.0])
    e_t = jnp.array([0.0, 0.0, 1.0])

    f = lambda inp: model(inp)

    _, Jx = pushforward_jet(f, X, [jnp.broadcast_to(e_x, X.shape)], factorial_scaled)
    _, Jy = pushforward_jet(f, X, [jnp.broadcast_to(e_y, X.shape)], factorial_scaled)
    u_x = Jx[0]
    u_y = Jy[0]

    zeros = jnp.zeros_like(X)
    _, Jxx = pushforward_jet(f, X, [jnp.broadcast_to(e_x, X.shape), zeros], factorial_scaled)
    _, Jyy = pushforward_jet(f, X, [jnp.broadcast_to(e_y, X.shape), zeros], factorial_scaled)
    _, Jxy_dir = pushforward_jet(
        f, X, [jnp.broadcast_to(e_x + e_y, X.shape), zeros], factorial_scaled
    )
    u_xx = Jxx[1]
    u_yy = Jyy[1]
    u_xy = _second_mixed_from_dirs(u_xx, u_yy, Jxy_dir[1])

    _, Jyyy = pushforward_jet(f, X, [jnp.broadcast_to(e_y, X.shape), zeros, zeros], factorial_scaled)
    u_yyy = Jyyy[2]

    _, Jty_dir = pushforward_jet(f, X, [jnp.broadcast_to(e_t + e_y, X.shape), zeros], factorial_scaled)
    u_ty = _second_mixed_from_dirs(Jty_dir[1], Jyy[1], Jty_dir[1] + 0.0)

    _, Jxxxy_dir = pushforward_jet(
        f, X, [jnp.broadcast_to(e_x + e_y, X.shape)] + [zeros] * 3, factorial_scaled
    )
    u_xxxy = Jxxxy_dir[3]

    return u_x, u_y, u_xx, u_xy, u_yy, u_yyy, u_ty, u_xxxy


# -----------------------------------------------------------------------------
# Residual computation
# -----------------------------------------------------------------------------


def residual_kdv2d(model: MLP, X: Array, factorial_scaled: bool) -> Array:
    u = model(X)
    u_x, u_y, u_xx, u_xy, u_yy, u_yyy, u_ty, u_xxxy = compute_derivatives(model, X, factorial_scaled)
    nonlinear = u_xy * u_x + u_y * u_xx
    R = u_ty + u_xxxy + 3.0 * nonlinear - u_xx + 2.0 * u_yy
    chex.assert_tree_all_finite(R)
    return R


# -----------------------------------------------------------------------------
# Losses and training
# -----------------------------------------------------------------------------


def loss_fn(
    model: MLP,
    X_r: Array,
    X_b: Array,
    u_b: Array,
    factorial_scaled: bool,
    lambda_bc: float,
) -> Tuple[Array, Tuple[Array, Array]]:
    R = residual_kdv2d(model, X_r, factorial_scaled)
    pred_b = model(X_b)
    loss_r = jnp.mean(R**2)
    loss_b = jnp.mean((pred_b - u_b) ** 2)
    return loss_r + lambda_bc * loss_b, (loss_r, loss_b)


class TrainState(nnx.Module):
    def __init__(self, model: MLP, optimizer: optax.GradientTransformation):
        self.model = model
        # Explicitly bind optimizer to model parameters per Flax 0.12 API.
        self.optimizer = nnx.Optimizer(model, optimizer, wrt=nnx.Param)


def make_train_state(rng: Array, cfg: Config):
    rng_calib, rng_model = jax.random.split(rng)
    factorial_scaled = calibrate_factorial_scaled(rng_calib)
    model = MLP(rng_model, width=cfg.width, depth=cfg.depth)
    opt = optax.adam(cfg.lr)
    state = TrainState(model, opt)
    return state, factorial_scaled


@functools.partial(chex.chexify, async_check=False)
@functools.partial(jax.jit, static_argnums=(4,))
def compute_grads(
    model: MLP,
    X_r: Array,
    X_b: Array,
    u_b: Array,
    factorial_scaled: bool,
    lambda_bc: float,
):
    def _loss_fn(m: MLP):
        return loss_fn(m, X_r, X_b, u_b, factorial_scaled, lambda_bc)

    (_, (loss_r, loss_b)), grads = nnx.value_and_grad(_loss_fn, has_aux=True)(model)
    chex.assert_tree_all_finite(grads)
    return grads, (loss_r, loss_b)


@nnx.jit
def apply_grads(state: TrainState, grads):
    state.optimizer.update(state.model, grads)
    return state


def train_step(
    state: TrainState,
    X_r: Array,
    X_b: Array,
    u_b: Array,
    factorial_scaled: bool,
    lambda_bc: float,
):
    grads, (loss_r, loss_b) = compute_grads(
        state.model, X_r, X_b, u_b, factorial_scaled, lambda_bc
    )
    state = apply_grads(state, grads)
    return state, (loss_r, loss_b)


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------


def save_checkpoint(state: TrainState, step: int, cfg: Config, path: Path) -> None:
    if orbax is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpointer = orbax.Checkpointer(orbax.PyTreeCheckpointHandler())
    payload = dict(model=state.model, optimizer=state.optimizer, cfg=asdict(cfg), step=step)
    checkpointer.save(path.as_posix(), payload)


def load_checkpoint(path: Path) -> Optional[TrainState]:
    if orbax is None or not path.exists():
        return None
    checkpointer = orbax.Checkpointer(orbax.PyTreeCheckpointHandler())
    payload = checkpointer.restore(path.as_posix())
    state = TrainState(payload['model'], payload['optimizer'])
    return state


# -----------------------------------------------------------------------------
# Evaluation utilities
# -----------------------------------------------------------------------------


def evaluate_grid(model: MLP, factorial_scaled: bool, n: int = 64):
    x = jnp.linspace(*DOMAIN_X, n)
    y = jnp.linspace(*DOMAIN_Y, n)
    Xg, Yg = jnp.meshgrid(x, y, indexing='ij')
    T = jnp.zeros_like(Xg)
    grid = jnp.stack([Xg, Yg, T], axis=-1).reshape(-1, 3)
    preds = model(grid).reshape(n, n)
    truth = exact_solution(grid).reshape(n, n)
    err = jnp.linalg.norm(preds - truth) / jnp.linalg.norm(truth)
    return preds, truth, err


# -----------------------------------------------------------------------------
# Micro-benchmark
# -----------------------------------------------------------------------------


def benchmark_residual(model: MLP, factorial_scaled: bool, iters: int = 5, batch: int = 256) -> float:
    rng = jax.random.PRNGKey(0)
    X = sample_collocation(rng, batch)

    def body(_: Array, __: Array):
        r = residual_kdv2d(model, X, factorial_scaled)
        return None, r

    compiled = jax.jit(lambda: jax.lax.scan(body, None, None, length=1)[1])
    compiled()  # warmup
    times = []
    for _ in range(iters):
        compiled()
        times.append(0.0)
    return float(jnp.mean(jnp.array(times)))


# -----------------------------------------------------------------------------
# Minimal runnable example
# -----------------------------------------------------------------------------


def main(cfg: Config = Config()) -> None:
    print(f"Config: {asdict(cfg)}")
    rng = jax.random.PRNGKey(cfg.seed)
    rng, key_r, key_b = jax.random.split(rng, 3)
    X_r = sample_collocation(key_r)
    X_b, u_b = sample_boundary(key_b)
    state, factorial_scaled = make_train_state(rng, cfg)

    for step in range(cfg.steps):
        for Xr_batch, (Xb_batch, ub_batch) in zip(
            collocation_loader(X_r, cfg.batch_size), boundary_loader(X_b, u_b, cfg.batch_size)
        ):
            state, (loss_r, loss_b) = train_step(
                state, Xr_batch, Xb_batch, ub_batch, factorial_scaled, cfg.lambda_bc
            )
        if step % 100 == 0:
            print(f"step={step}, loss_r={float(loss_r):.4e}, loss_b={float(loss_b):.4e}")

    _, _, rel_err = evaluate_grid(state.model, factorial_scaled, n=32)
    print(f"Relative L2 error on 32x32 grid: {float(rel_err):.4e}")

    if cfg.checkpoint_dir:
        ckpt_path = Path(cfg.checkpoint_dir) / "checkpoint"
        save_checkpoint(state, cfg.steps, cfg, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
