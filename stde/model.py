import dataclasses
import math
from collections import namedtuple
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jax.random

import stde.types as types
from stde.config import EqnConfig, ModelConfig
from stde.equations import pinn_loss_fn
from stde.types import Equation


def calculate_gain_leaky_relu(negative_slope: float) -> float:
  """This is what pytorch uses by default for the gain in
  kaiming uniform initialization for the Linear layer.

  Effective this samples from
  uniform(-1/sqrt(in_features), 1/sqrt(in_features))
  """
  return math.sqrt(2.0 / (1 + negative_slope**2))


INITIALIZERS = dict(
  kaiming_uniform=hk.initializers.VarianceScaling(
    scale=calculate_gain_leaky_relu(math.sqrt(5))**2,
    mode='fan_in',
    distribution='uniform'
  ),
  xavier_normal=hk.initializers.VarianceScaling(
    scale=1., mode='fan_avg', distribution='normal'
  )
)


@dataclasses.dataclass
class PINN(hk.Module):
  eqn: Equation
  eqn_cfg: EqnConfig
  model_cfg: ModelConfig

  def __call__(self, x: types.X, t: types.T) -> jax.Array:
    """ansatze for space-time domain scalar function"""
    # squeeze to ensure scalar output
    inputs = x if not self.eqn.time_dependent else jnp.concatenate(
      [x, t], axis=-1
    )
    pred = jnp.squeeze(self.net(inputs))
    if self.eqn_cfg.enforce_boundary:
      return self.eqn.enforce_boundary(x, t, pred, self.eqn_cfg)
    else:
      return pred

  def net(self, xt: jax.Array) -> jax.Array:
    """NOTE: if eqn.time_dependent, the last dim is t."""
    if len(self.model_cfg.hidden_sizes) == 0:
      hidden_sizes = [self.model_cfg.width] * (self.model_cfg.depth - 1) + [1]
    else:
      hidden_sizes = list(self.model_cfg.hidden_sizes) + [1]

    init_kwargs = dict()
    if self.model_cfg.w_init in INITIALIZERS:
      init_kwargs['w_init'] = INITIALIZERS[self.model_cfg.w_init]
    if self.model_cfg.b_init in INITIALIZERS:
      init_kwargs['b_init'] = INITIALIZERS[self.model_cfg.b_init]

    if self.model_cfg.block_size != -1:  # WEIGHT SHARING VIA CONV
      if self.eqn.time_dependent:
        x = xt[..., :-1]
      else:
        x = xt

      if self.model_cfg.use_conv:
        x = hk.Conv1D(
          output_channels=1,
          kernel_shape=self.model_cfg.block_size,
          stride=self.model_cfg.block_size,
          padding="VALID"
        )(
          x[..., None]  # add channel dim
        )
        x = x[..., 0]  # remove channel dim

      else:  # use MLP
        x = x.reshape(xt.shape[:-1] + (-1, self.model_cfg.block_size))
        layer_fn = hk.Linear(1, name=f"linear_first", **init_kwargs)
        x = jax.nn.tanh(layer_fn(x))
        x = x[..., 0]  # remove channel dim

      if self.eqn.time_dependent:
        xt = jnp.concatenate([x, xt[..., -1:]], axis=-1)
      else:
        xt = x

    for i, h_i in enumerate(hidden_sizes):
      layer_fn = hk.Linear(h_i, name=f"linear_{i}", **init_kwargs)
      xt = layer_fn(xt)
      if i != len(hidden_sizes) - 1:  # activation
        xt = jax.nn.tanh(xt)
    return xt

  def loss_fn(self, x, t, x_boundary, t_boundary):
    return pinn_loss_fn(
      x, t, x_boundary, t_boundary, self.__call__, self.eqn_cfg, self.eqn
    )

  def err_norms_fn(self, x, t, y, y_t):
    y_pred = self.__call__(x, t)
    err = y - y_pred
    l1 = jnp.abs(err).sum()
    l2 = (err**2).sum()
    w1_t = 0.
    if self.eqn.time_dependent and self.model_cfg.compute_w1_loss:
      u_t = jax.vmap(jax.grad(self.__call__, argnums=1))(x, t)
      w1_t = ((u_t - y_t)**2).sum()
    return l1, l2, w1_t

  def init_for_multitransform(self):
    return (
      self.__call__,
      namedtuple("PINN", ["u", "loss_fn", "err_norms_fn"
                         ])(self.__call__, self.loss_fn, self.err_norms_fn),
    )




import logging
from functools import partial
from typing import Any, Callable, Union, Tuple, Dict
from dataclasses import field, dataclass
import math
import einops
import flax.linen as nn
from flax import struct

import jax
from jax import jit, numpy as jnp

from jax.experimental.jet import jet
from jax import config
config.update("jax_enable_x64", True)

@struct.dataclass
class DiagnosticsConfig:
    skip: bool = False
    gate: bool = False
    gated: bool = False
    residual: bool = False
    ssm_input_norm: bool = False
    ssm_coeffs: bool = False
    ssm_residual: bool = False
    ssm_output_norm: bool = False


@dataclass(frozen=True)
class MambaConfig:
    hidden_features: int
    expansion_factor: float
    dt_rank: Any
    activation: str
    norm_type: str
    dense_expansion: int
    complement: bool
    tie_in_proj: bool
    tie_gate: bool
    diagnostics: DiagnosticsConfig
    radius: float = 1.0
    bidirectional: bool = True

@dataclass(frozen=True)
class SSMConfig:
    recursive_scan: bool = False
    min_recursion_length: int = 2
    recursive_split: int = 2
    custom_vjp_scan: bool = False
    activation: str = "silu"

@dataclass(frozen=True)
class PINNConfig:
    num_mamba_blocks: int



# -----------------------------------------------------------------------------
# Domain sampler: can return single points or time sequences of points
# -----------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("batch_size", "seq_len", "radius", "dim"))
def sample_domain_fn(batch_size: int,
                     rng: jax.Array,
                     radius: float,
                     dim: int,
                     seq_len: int,
                     ) -> Tuple[jnp.ndarray, jax.Array]:
    """
    Sample `batch_size` points in R^dim.  If seq_len > 1, returns
    shape (batch_size, seq_len, dim), otherwise (batch_size, 1, dim).
    """
    keys = jax.random.split(rng, seq_len + 1)
    out = []
    for i in range(seq_len):
        # radius in [0, x_radius]
        r = jax.random.uniform(keys[i], (batch_size, 1),
                               minval=0.0, maxval=radius)
        x = jax.random.normal(keys[i + 1], (batch_size, dim))
        # project onto sphere of radius r
        x = x / jnp.linalg.norm(x, axis=-1, keepdims=True) * r
        out.append(x)
    x_seq = jnp.stack(out, axis=1)  # (B, seq_len, dim)
    return x_seq, keys[0]

# alpha = exp(A*Delta) [zero-order hold], beta = B*Delta*x [Euler step]
@jit
def compute_alpha (Acoeff, Delta_chunk):
    return jnp.exp (jnp.einsum ('dn,lbd->lbdn', Acoeff, Delta_chunk))  # (chunk_size, B, D, N)

# The zero-order hold is empirically only really necessary for alpha, since it has a gating effect
@jit
def compute_alpha_beta (x_chunk, Acoeff, B_chunk, Delta_chunk):
    alpha = compute_alpha (Acoeff, Delta_chunk)  # (chunk_size, B, D, N)
    beta = jnp.einsum ('lbn,lbd,lbd->lbdn', B_chunk, x_chunk, Delta_chunk)  # (chunk_size, B, D, N)
    return alpha, beta

def largest_factor_up_to(b,n):
    if n < 2:
        return n
    k = b
    while n % k != 0:
        k -= 1
    return k

# ─────────── NEW FUNCTION ───────────
@jit
def ssm_parallel_scan(x, Acoeff, Bcoeff, Ccoeff, Delta):
    """
    x:      (B, L, D)
    Acoeff: (D, N)
    Bcoeff: (B, L, N)
    Ccoeff: (B, L, N)
    Delta:  (B, L, D)
    returns y: (B, L, D)
    """

    # 1) compute α and β directly in batch‑major shape
    #    α, β: (B, L, D, N)
    α = jnp.exp(jnp.einsum('dn,bld->bldn', Acoeff, Delta))
    β = jnp.einsum('bln,bld,bld->bldn', Bcoeff, x, Delta)

    # 2) prefix‐product along the time axis (axis=1)
    P    = jnp.cumprod(α, axis=1)         # (B, L, D, N)
    invP = 1.0 / P                        # (B, L, D, N)

    # 3) weighted prefix‐sum of β/g
    S    = jnp.cumsum(β * invP, axis=1)   # (B, L, D, N)

    # 4) h = P * S
    h    = P * S                          # (B, L, D, N)

    # 5) project through C
    #    note: Ccoeff is (B, L, N) so we align dims with h
    y    = jnp.einsum('bln,bldn->bld', Ccoeff, h)  # (B, L, D)

    return y

# ──────────────────────────────────────

@jit
def inverse_softplus(x):
    return x + jnp.log(1 - jnp.exp(-x))


class SelectiveSSM(nn.Module):
    """ A variation on MAMBA: https://arxiv.org/pdf/2312.00752.pdf """

    reverse: bool = False
    complement: bool = False  # only checked if reverse is true

    hidden_features: int = 16  # N
    chunk_size: int = None
    n_channel_groups: int = None

    dt_rank: Union[int, str] = 'auto'  # R
    dt_proj: bool = True   # whether to use a linear projection (vs broadcast) to map dt_rank to D

    dt_min: float = 0.001  # 1/(long-range context length)
    dt_max: float = 0.1    # 1/(short-range context length)

    a_init_scale: float = 1.0

    l2_scale: float = 0.0

    shift_conv_size: int = 3

    activation: str = "silu"

    diagnostics: DiagnosticsConfig = field(default_factory=lambda: DiagnosticsConfig())

    recursive_scan: bool = False
    min_recursion_length: int = 2
    recursive_split: int = 2

    custom_vjp_scan: bool = False

    @nn.compact
    def __call__(
        self,
        x,  # (B, L, D)
        train: bool = False,
    ):
        B = x.shape[-3]
        L = x.shape[-2]
        D = x.shape[-1]  # if called by BidirectionalMamba, this is actually E*D

        N = self.hidden_features

        if self.dt_rank == 'auto':
            dt_rank = math.ceil(D / 16)
        else:
            dt_rank = self.dt_rank

        if self.reverse:
            x = jnp.flip (x, axis=(-2,-1) if self.complement else -2)
        # shift conv
        u = nn.Conv (features=D, feature_group_count=D, kernel_size=(self.shift_conv_size,), strides=(1,), padding="SAME", use_bias=False, name="shift_conv", kernel_init=nn.initializers.lecun_normal()) (x)  # (B, L, D)

        if self.activation == "gelu":
            u = nn.gelu(u)
        elif self.activation == "relu":
            u = nn.relu(u)
        elif self.activation == "silu":
            u = nn.silu(u)
        elif self.activation == "wave":
            u = WaveAct()(u)
        elif self.activation == "tanh":
            u = nn.tanh(u)
        elif self.activation is not None:
            raise Exception(f"Unknown activation: {self.activation}")

        # Initialize A nonrandomly with evenly spaced eigenvalues; keep parameterization in log space to guarantee A<0
        Acoeff = -jnp.exp (self.param ('A_log', lambda rng: jnp.log (jnp.repeat (jnp.arange(start=1,stop=N+1,dtype=jnp.float32)[None,:], D, axis=0))))  # (D, N)
        Bcoeff, Ccoeff = jnp.split (nn.Dense (features=2*N, name='BC', use_bias=True, kernel_init=nn.initializers.lecun_normal()) (u), 2, axis=-1)  # (B, L, N) *2
        Dcoeff = self.param ('D', lambda rng: jnp.ones((D,)))  # (D,)

        dt_bias_init = lambda rng, shape, dtype: inverse_softplus (jax.random.uniform (rng, shape=shape, dtype=dtype, minval=self.dt_min, maxval=self.dt_max))
        dt = nn.Dense (features=dt_rank, use_bias=True, name='dt',
                       kernel_init=nn.initializers.lecun_normal(),
                       bias_init=nn.initializers.zeros if self.dt_proj else dt_bias_init) (u)  # (B, L, dt_rank)

        if self.dt_proj:
            dt = nn.Dense (features=D, use_bias=True, kernel_init=nn.initializers.lecun_normal(), bias_init=dt_bias_init, name='dt_proj') (dt)  # (B, L, D)
        else:
            if dt_rank > 1:  # if dt_rank is 1, we can just rely on broadcasting, and save memory
                if D % dt_rank != 0:
                    raise ValueError(f"dt_rank={dt_rank} must divide D={D}")
                dt = jnp.repeat (dt, D // dt_rank, axis=-1)  # (B, L, D)
    
        dt = jnp.log1p(jnp.exp(dt)) # low-level softplus

        # ─── Vectorized prefix-scan ───
        y = ssm_parallel_scan(x, Acoeff, Bcoeff, Ccoeff, dt)

        if self.reverse:
            y = jnp.flip (y, axis=(-2,-1) if self.complement else -2)

        # Add in the skip connection term
        y = y + jnp.einsum ('bld,d->bld', x, Dcoeff)

        return y

def relu(x):
    return jnp.maximum(x, 0)


class WaveAct(nn.Module):
    @nn.compact
    def __call__(self, x):
        # initialize w1, w2 to 1.0
        w1 = self.param('w1', nn.initializers.ones, ())
        w2 = self.param('w2', nn.initializers.ones, ())
        # broadcast to x’s shape when multiplying
        return w1 * jnp.sin(x) + w2 * jnp.cos(x)


class ZeroOnUnitBall(nn.Module):
    @nn.compact
    def __call__(self, x_in, u_val, radius=1.0):
        return (radius**2 - jnp.sum(x_in**2, -1)) * u_val


class BidirectionalMamba(nn.Module):
    hidden_features: int   # N
    expansion_factor: float  # E
    dt_rank: Union[int, str] = 'auto'
    complement: bool = False
    tie_in_proj: bool = False
    tie_gate: bool = False
    concatenate_fwd_rev: bool = True
    activation: str = "gelu"
    norm_type: str = "none"
    bn_momentum: float = 0.9
    dense_expansion: int = 2
    mlp_dropout_rate: float = 0.1
    ssm_args: Dict[str, Any] = field(default_factory=dict)
    diagnostics: DiagnosticsConfig = field(default_factory=lambda: DiagnosticsConfig())
    l2_scale: float = 1e-9
    radius: float = 1.0
    bidirectional: bool = True

    @nn.compact
    def __call__(self, x, train: bool = False):
        # Ensure input is 3D: (batch_size, sequence_length, features)
        B = x.shape[0]
        D = x.shape[-1]
        L = x.shape[-2]
        x = x.reshape(B, L, D)

        input_features = x.shape[-1]  # D

        if self.dt_rank == 'auto':
            dt_rank = math.ceil(input_features / 16)
        else:
            dt_rank = self.dt_rank

        if self.activation == "gelu":
            activate = nn.gelu
        elif self.activation == "silu":
            activate = nn.silu
        elif self.activation == "relu":
            activate = relu
        elif self.activation == "tanh":
            activate = nn.tanh
        elif self.activation == "wave":
            activate = WaveAct()
        else:
            raise Exception(f"Unknown activation: {self.activation}")

        skip = x

        # normalize
        if self.norm_type == "batch":
            x = nn.BatchNorm(momentum=self.bn_momentum, use_running_average=not train)(x)
        elif self.norm_type == "layer":
            x = nn.LayerNorm()(x)
        elif self.norm_type == "group":
            x = nn.GroupNorm()(x)
        elif self.norm_type == "rms":
            x = nn.RMSNorm()(x)
        elif self.norm_type == "none":
            pass
        else:
            raise Exception(f"Unknown norm type: {self.norm_type}")

        ED = math.ceil (self.expansion_factor * input_features)
        
        
        # project to expanded dimension
        n_in_proj = 1 if self.tie_in_proj else 2
        n_gate = 1 if self.tie_gate else 2
        
        [xf, _xr, zf, _zr] = jnp.split(nn.Dense (features=((n_in_proj+n_gate)*ED), name='in_proj', kernel_init=nn.initializers.lecun_normal()) (x), [k*ED for k in [1,n_in_proj,n_in_proj+1]], axis=-1)
        
        xr = xf if self.tie_in_proj else _xr
        zr = zf if self.tie_gate else _zr

        # forward and backward SSM
        ssm = SelectiveSSM
        xf = ssm(hidden_features=self.hidden_features, reverse=False, dt_rank=dt_rank, diagnostics=self.diagnostics, **self.ssm_args) (xf, train)
        
        if self.bidirectional:
            xr = ssm(hidden_features=self.hidden_features, reverse=True, complement=self.complement, dt_rank=dt_rank, diagnostics=self.diagnostics, **self.ssm_args) (xr, train)
            x = xf * activate(zf) + xr * activate(zr)
        else:
            x = xf * activate(zf)
        
        # project back down
        x = nn.Dense (features=input_features, name='out_proj', kernel_init=nn.initializers.lecun_normal()) (x)
        x = activate(x)

        x_out = skip + x

        return x_out

def test_bidirectional_mamba_initialization():
    print("\n=== Testing Bidirectional Mamba Initialization ===")
    # Test parameters
    n_pts = 8
    dim = 5
    radius = 1.0
    seq_len = 10

    print(f"Test parameters: n_pts={n_pts}, dim={dim}, seq_len={seq_len}, radius={radius}")

    model_config = MambaConfig(
        hidden_features=32,
        expansion_factor=2.0,
        dt_rank='auto',
        activation='gelu',
        norm_type='layer',
        mlp_layer=True,
        dense_expansion=4,
        complement=True,
        tie_in_proj=True,
        tie_gate=True,
        concatenate_fwd_rev=True,
        diagnostics=DiagnosticsConfig()
    )

    print(f"Model config: hidden_features={model_config.hidden_features}, expansion_factor={model_config.expansion_factor}")

    # Initialize model
    model = BidirectionalMamba(**vars(model_config))
    print("Model initialized successfully")

    # Create input using sample_domain_fn
    key = jax.random.PRNGKey(0)
    x, key = sample_domain_fn(n_pts, key, seq_len=seq_len, radius=radius, dim=dim)
    print(f"Input shape: {x.shape}")

    # Initialize parameters
    variables = model.init(key, x)
    print(f"Model parameters initialized successfully")

    # Check if model components exist
    params = variables['params']
    assert 'in_proj' in params, "Missing input projection layer"
    assert 'out_proj' in params, "Missing output projection layer"

    # Print parameter shapes
    print("Key parameter shapes:")
    print(f"  in_proj/kernel: {params['in_proj']['kernel'].shape}")
    print(f"  out_proj/kernel: {params['out_proj']['kernel'].shape}")

    print("Bidirectional Mamba initialization test passed!")

def test_bidirectional_mamba_forward():
    print("\n=== Testing Bidirectional Mamba Forward Pass ===")
    # Test parameters
    n_pts = 4
    dim = 5
    radius = 1.0
    seq_len = 10

    print(f"Test parameters: n_pts={n_pts}, dim={dim}, seq_len={seq_len}, radius={radius}")

    model_config = MambaConfig(
        hidden_features=16,
        expansion_factor=2.0,
        dt_rank='auto',
        activation='gelu',
        norm_type='layer',
        mlp_layer=True,
        dense_expansion=4,
        complement=True,
        tie_in_proj=True,
        tie_gate=True,
        concatenate_fwd_rev=True,
        diagnostics=DiagnosticsConfig()
    )

    print(f"Model config: hidden_features={model_config.hidden_features}, expansion_factor={model_config.expansion_factor}")

    # Initialize model
    model = BidirectionalMamba(**vars(model_config))
    print("Model initialized successfully")

    # Create input using sample_domain_fn
    key = jax.random.PRNGKey(0)
    x, key = sample_domain_fn(n_pts, key, seq_len=seq_len, radius=radius, dim=dim)
    print(f"Input shape: {x.shape}")

    # Initialize parameters and run forward pass
    variables = model.init(key, x)
    print("Parameters initialized successfully")

    output = model.apply(variables, x)
    print(f"Forward pass completed successfully")
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[:2]}")

    # Test output shape
    print(f"  Forward output shape: {output.shape}")
    expected_shape = x.shape[:2]
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"Output shape test passed: {output.shape} matches expected {expected_shape}")

    # Test output type
    assert isinstance(output, jnp.ndarray), f"Expected output to be jnp.ndarray, but got {type(output)}"
    print(f"Output type test passed: {type(output).__name__}")

    # Test that output contains no NaN values
    has_nans = jnp.any(jnp.isnan(output))
    assert not has_nans, "Output contains NaN values"
    print(f"NaN check passed: Output contains no NaN values")

    print("Bidirectional Mamba forward pass test passed!")

def test_bidirectional_mamba_configurations():
    print("\n=== Testing Bidirectional Mamba with Different Configurations ===")
    n_pts = 2
    dim = 5
    radius = 1.0
    seq_len = 10

    print(f"Test parameters: n_pts={n_pts}, dim={dim}, seq_len={seq_len}, radius={radius}")

    # Test different configurations
    base_config = dict(
        hidden_features=16,
        expansion_factor=2.0,
        dt_rank='auto',
        activation='gelu',
        norm_type='layer',
        mlp_layer=True,
        dense_expansion=4,
        complement=True,
        tie_in_proj=True,
        tie_gate=True,
        concatenate_fwd_rev=True,
        diagnostics=DiagnosticsConfig()
    )

    configs = [
        # Test complement and tie configurations
        {**base_config, 'complement': True, 'tie_in_proj': True, 'tie_gate': True, 'concatenate_fwd_rev': True},
        {**base_config, 'complement': False, 'tie_in_proj': False, 'tie_gate': False, 'concatenate_fwd_rev': False},

        # Test different activations and norms
        {**base_config, 'activation': 'silu', 'norm_type': 'layer'},
        {**base_config, 'activation': 'gelu', 'norm_type': 'rms'},

        # Test MLP configurations
        {**base_config, 'mlp_layer': True, 'dense_expansion': 8},
    ]

    print(f"Testing {len(configs)} different configurations")

    key = jax.random.PRNGKey(0)
    x, key = sample_domain_fn(n_pts, key, seq_len=seq_len, radius=radius, dim=dim)
    
    print(f"Input shape: {x.shape}")

    for i, config in enumerate(configs):
        print(f"\nTesting configuration {i+1}:")
        print(f"  activation: {config['activation']}")
        print(f"  norm_type: {config['norm_type']}")
        print(f"  complement: {config['complement']}")
        print(f"  tie_in_proj: {config['tie_in_proj']}")
        print(f"  tie_gate: {config['tie_gate']}")
        print(f"  concatenate_fwd_rev: {config['concatenate_fwd_rev']}")

        model_config = MambaConfig(**config)
        model = BidirectionalMamba(**vars(model_config))
        variables = model.init(key, x)
        output = model.apply(variables, x, train=True)

        # Test output shape
        print(f"  Forward output shape: {output.shape}")
        expected_shape = x.shape[:2]
        assert output.shape == expected_shape, f"Shape mismatch with config {config}"
        assert not jnp.any(jnp.isnan(output)), f"NaN values found with config {config}"

        print(f"  Output shape: {output.shape}")
        print(f"  Output sample: {output[0]}")
        print(f"  Configuration {i+1} test passed!")

    print("\nAll configuration tests passed!")

def test_bidirectional_mamba_training_mode():
    print("\n=== Testing Bidirectional Mamba in Training Mode with Diagnostics ===")
    # Test training mode with diagnostics
    n_pts = 4
    dim = 5
    radius = 1.0
    seq_len = 10

    print(f"Test parameters: n_pts={n_pts}, dim={dim}, seq_len={seq_len}, radius={radius}")

    model_config = MambaConfig(
        hidden_features=32,
        expansion_factor=2.0,
        dt_rank='auto',
        activation='gelu',
        norm_type='layer',
        mlp_layer=True,
        dense_expansion=4,
        complement=True,
        tie_in_proj=True,
        tie_gate=True,
        concatenate_fwd_rev=True,
        diagnostics=DiagnosticsConfig(
            skip=True,
            gate=True,
            gated=True,
            residual=True,
            ssm_input_norm=False,
            ssm_coeffs=False,
            ssm_residual=False,
            ssm_output_norm=False
        )
    )

    print("Diagnostics enabled for: skip, gate, gated, residual")

    model = BidirectionalMamba(**vars(model_config))
    print("Model initialized successfully")

    key = jax.random.PRNGKey(0)
    x, key = sample_domain_fn(n_pts, key, seq_len=seq_len, radius=radius, dim=dim)
    print(f"Input shape: {x.shape}")

    variables = model.init(key, x)
    print("Parameters initialized successfully")

    # Test forward pass in training mode
    output, diagnostics = model.apply(
        variables, x, train=True,
        mutable=['diagnostics']
    )
    print("Forward pass in training mode completed successfully")
    print(f"Output shape: {output.shape}")

    assert 'diagnostics' in diagnostics
    assert 'skip_mean' in diagnostics['diagnostics']
    assert 'gate_fwd_mean' in diagnostics['diagnostics']
    assert 'gated_mean' in diagnostics['diagnostics']
    assert 'residual_mean' in diagnostics['diagnostics']

    print("\nDiagnostics collected:")
    print(f"  skip_mean: {diagnostics['diagnostics']['skip_mean']}")
    print(f"  gate_fwd_mean: {diagnostics['diagnostics']['gate_fwd_mean']}")
    print(f"  gated_mean: {diagnostics['diagnostics']['gated_mean']}")
    print(f"  residual_mean: {diagnostics['diagnostics']['residual_mean']}")

    print("Training mode test with diagnostics passed!")

def tabulate_bidirectional_mamba():
    print("\n=== Tabulating Bidirectional Mamba Model Architecture ===")
    # Define sample input dimensions
    n_pts = 4
    dim = 5
    radius = 1.0
    seq_len = 10

    print(f"Input parameters: n_pts={n_pts}, dim={dim}, seq_len={seq_len}, radius={radius}")

    # Create input using sample_domain_fn
    key = jax.random.PRNGKey(0)
    x, key = sample_domain_fn(n_pts, key, seq_len=seq_len, radius=radius, dim=dim)
    print(f"Input shape: {x.shape}")

    model_config = MambaConfig(
        hidden_features=32,
        expansion_factor=2.0,
        dt_rank='auto',
        activation='gelu',
        norm_type='layer',
        mlp_layer=True,
        dense_expansion=4,
        complement=True,
        tie_in_proj=True,
        tie_gate=True,
        concatenate_fwd_rev=True,
        diagnostics=DiagnosticsConfig()
    )

    print(f"Model configuration:")
    print(f"  hidden_features: {model_config.hidden_features}")
    print(f"  expansion_factor: {model_config.expansion_factor}")
    print(f"  dt_rank: {model_config.dt_rank}")
    print(f"  activation: {model_config.activation}")
    print(f"  norm_type: {model_config.norm_type}")
    print(f"  mlp_layer: {model_config.mlp_layer}")

    # Initialize model with configuration
    model = BidirectionalMamba(**vars(model_config))
    print("Model initialized successfully")

    # Print the tabulated model summary
    print("\nDetailed Model Architecture:")
    print(nn.tabulate(
        model,
        rngs={"params": jax.random.PRNGKey(0)},
        mutable=['params', 'diagnostics', 'intermediates'],
    )(x, train=True))

    print("\nModel tabulation completed successfully!")

    # Jet (Taylor‐mode) vs jvp
def test_ssm_parallel_scan_jet_forward_mode():
    B, L, D, N = 2, 5, 3, 4
    key = jax.random.PRNGKey(0)
    x  = jax.random.normal(key, (B, L, D))
    A  = jax.random.normal(key, (D, N))
    Bc = jax.random.normal(key, (B, L, N))
    Cc = jax.random.normal(key, (B, L, N))
    Δ  = jax.random.uniform(key, (B, L, D))

    def f(x):
        return ssm_parallel_scan(x, A, Bc, Cc, Δ)

    v = jax.random.normal(key, x.shape)

    # y_primal, y_tangent = jet(f, (x,), (v,))
    y_primal, y_tangent_list = jet(f, (x,), ([v],))
    y_tangent = y_tangent_list[0]
    y_ref,    y_jvp     = jax.jvp(f, (x,), (v,))

    # print diagnostics
    print("=== test_ssm_parallel_scan_jet_forward_mode ===")
    print(f"primal.shape = {y_primal.shape}")
    p_diff = jnp.abs(y_primal - y_ref)
    t_diff = jnp.abs(y_tangent - y_jvp)
    print(f"max|primal - ref|   = {jnp.max(p_diff):.3e}")
    print(f"max|tangent - jvp| = {jnp.max(t_diff):.3e}")

    assert jnp.allclose(y_primal, y_ref, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(y_tangent, y_jvp, atol=1e-6, rtol=1e-6)

    print("Test passed!")




def test_stde_on_minimal_mamba():
    sparse = True
    key = jax.random.PRNGKey(0)
    # toy 1-step sequence in dim=3
    x = jax.random.normal(key, (1,2,3))
    rand_batch_size = x.shape[0]
    dim = x.shape[-1]
    
    # STDE
    def hess_trace(fn: Callable) -> Callable:

        def fn_trace(x_i, key):
            key, subkey = jax.random.split(key)

            if sparse:
                key, subkey = jax.random.split(subkey)
                idx_set = jax.random.choice(
                    subkey, dim, shape=(rand_batch_size,), replace=False
                )
                rand_vec = jax.vmap(lambda i: jnp.eye(dim)[i])(idx_set)

            else:
                key, subkey = jax.random.split(subkey)
                rand_vec = 2 * (
                    jax.random.randint(
                    subkey, shape=(rand_batch_size, dim), minval=0, maxval=2
                    ) - 0.5
                )
            # perform the jvp‐via‐Taylor‐series
            taylor_2 = lambda v: jet(
            fun=fn, primals=(x_i,), series=((v, jnp.zeros(dim)),)
            )
            f_vals, (_, hvps) = jax.vmap(taylor_2)(rand_vec)
            trace_est = jnp.mean(hvps)
            if sparse:
                trace_est *= dim
            return f_vals[0], trace_est, key

        return fn_trace
    
    
    cfg = MambaConfig(
        hidden_features=16, expansion_factor=2.0, dt_rank='auto',
        activation='gelu', norm_type='layer',
        dense_expansion=2, complement=True,
        tie_in_proj=True, tie_gate=True,
        concatenate_fwd_rev=True,
        diagnostics=DiagnosticsConfig()  # no sow
    )
    model = BidirectionalMamba(**vars(cfg))
    vars_ = model.init(key, x, train=False)
    params = vars_['params']

    # wrap the model in the same hess_trace you use
    def u_fn(xi):
      # xi: (3,)
      # bump it into shape (1,1,3)
      y = model.apply({'params':params}, xi[None,None,:], train=False)
      return jnp.squeeze(y)

    ht = hess_trace(u_fn)
    # run it once to force Jet to kick in
    y0, trace_est, _ = ht(x[0,0], key)
    print("y0, trace_est =", y0, trace_est)


if __name__ == "__main__":
    print("\n===================================================")
    print("RUNNING MAMBA MODEL TESTS")
    print("===================================================")

    test_bidirectional_mamba_initialization()
    test_bidirectional_mamba_forward()
    test_bidirectional_mamba_configurations()
    test_bidirectional_mamba_training_mode()
    tabulate_bidirectional_mamba()
    test_ssm_parallel_scan_jet_forward_mode()
    test_stde_on_minimal_mamba()

    print("\n===================================================")
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("===================================================")

