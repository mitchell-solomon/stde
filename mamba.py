
import logging
from functools import partial, reduce
from typing import Any, Callable, Sequence, Union, Tuple, Dict
from dataclasses import field, dataclass

import math
from functools import reduce, partial

import einops
import flax.linen as nn
from flax import struct

import jax
from jax import custom_vjp, jit, numpy as jnp
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
    mlp_layer: bool
    dense_expansion: int
    complement: bool
    tie_in_proj: bool
    tie_gate: bool
    concatenate_fwd_rev: bool
    diagnostics: DiagnosticsConfig



# The associative scan is a product of matrices of the form ((g,h),(0,1)) where g_i=exp(A*Delta)x_i and h_i=B*Delta*x_i
# Since matrices of this form are are closed under multiplication, we can represent all intermediate products in the same way
@jax.remat
def associative_scan_fn (l, r):
    g_l, h_l = l
    g_r, h_r = r
    return tuple((g_l*g_r, g_r*h_l + h_r))

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

# x: (B, L, D)
# Acoeff: (D, N)
# Bcoeff: (B, L, N)
# Ccoeff: (B, L, N)
# Delta: (B, L, D) or (B, L, 1);  can assume (B, L, D) and rely on broadcasting
def ssm_recursive_scan (x, Acoeff, Bcoeff, Ccoeff, Delta, min_recursion_length: int = 2, recursive_split: int = 2):
    B = x.shape[-3]
    L = x.shape[-2]
    D = x.shape[-1]
    N = Acoeff.shape[-1]

    # Transpose length & batch dimensions to make the scan over length, and split into chunks
    # This is a bit inefficient, but taking dynamic slices appears to be worse in terms of GPU memory usage
    x = einops.rearrange (x, 'b l d -> l b d')
    Bcoeff = einops.rearrange (Bcoeff, 'b l n -> l b n')
    Ccoeff = einops.rearrange (Ccoeff, 'b l n -> l b n')
    Delta = einops.rearrange (Delta, 'b l d -> l b d')

    # Recursive function to do associative scan
    @jax.remat
    def scan_chunk (carry, chunk):
        g_init, h_init = carry  # (B, D, N)  (B, D, N)
        x_chunk, B_chunk, C_chunk, Delta_chunk = chunk
        chunk_size = x_chunk.shape[0]

        if chunk_size > min_recursion_length and chunk_size % recursive_split == 0:
            # Split inputs into chunks, scan each chunk, and concatenate results
            # Again, this seems inefficient, but empirically uses less GPU memory than passing an index range and doing dynamic slicing
            x_chunk = einops.rearrange (x_chunk, '(c l) b d -> c l b d', c=recursive_split)
            B_chunk = einops.rearrange (B_chunk, '(c l) b n -> c l b n', c=recursive_split)
            C_chunk = einops.rearrange (C_chunk, '(c l) b n -> c l b n', c=recursive_split)
            Delta_chunk = einops.rearrange (Delta_chunk, '(c l) b d -> c l b d', c=recursive_split)
            (g_init, h_init), y_chunk = jax.lax.scan (scan_chunk, (g_init, h_init), (x_chunk, B_chunk, C_chunk, Delta_chunk))
            y_chunk = einops.rearrange (y_chunk, 'c l b d -> (c l) b d')
            return (g_init, h_init), y_chunk

        alpha, beta = compute_alpha_beta (x_chunk, Acoeff, B_chunk, Delta_chunk)  # (chunk_size, B, D, N)  (chunk_size, B, D, N)
        gs, hs = jax.lax.associative_scan (associative_scan_fn, (alpha, beta))  # (chunk_size, B, D, N)  (chunk_size, B, D, N)
        hs = gs * h_init + hs  # Incorporate h_init here so that it is reflected in y_chunk
        # We only need to keep the last state of gs, so we can discard the rest. Otherwise we would incorporate g_init here, like so:
        # gs = g_init * As
        y_chunk = jnp.einsum ('lbn,lbdn->lbd', C_chunk, hs)  # (chunk_size, B, D)
        return (gs[-1,...] * g_init, hs[-1,...]), y_chunk  # note g_init incorporated here

    (_A_final, _h_final), y = scan_chunk ((jnp.ones((B,D,N)), jnp.zeros((B,D,N))), (x, Bcoeff, Ccoeff, Delta))

    return einops.rearrange (y, 'l b d -> b l d')  # (B, L, D)


@partial(custom_vjp, nondiff_argnums=(5,6))
@partial(jit, static_argnames=('min_recursion_length','recursive_split',))
def ssm_scan (x, Acoeff, Bcoeff, Ccoeff, Delta, min_recursion_length: int = 2, recursive_split: int = 2):
    return ssm_recursive_scan (x, Acoeff, Bcoeff, Ccoeff, Delta, min_recursion_length, recursive_split)

@partial(jit, static_argnames=('min_recursion_length','recursive_split',))
def ssm_scan_forward (x, Acoeff, Bcoeff, Ccoeff, Delta, min_recursion_length, recursive_split):
    y = ssm_recursive_scan (x, Acoeff, Bcoeff, Ccoeff, Delta, min_recursion_length, recursive_split)
    return y, (x, Acoeff, Bcoeff, Ccoeff, Delta)

@jit
def forward_scan_fn (h_tMinus1, chunk):
    alpha_t, beta_t = chunk
    h_t = h_tMinus1 * alpha_t + beta_t
    return h_t, h_tMinus1

@jit
def backward_scan_fn (f_alpha_tPlus1, chunk):
    alpha_t, C_dy_t = chunk
    f_t = f_alpha_tPlus1 + C_dy_t
    return f_t * alpha_t, f_t


# x_chunk: (L, B, D)
# Acoeff: (D, N)
# B_chunk: (L, B, N)
# C_chunk: (L, B, N)
# Delta_chunk: (L, B, D)
# dy_chunk: (L, B, D)
# h_left: (B, D, N)
# f_alpha_right: (B, D, N)
@partial(jit, static_argnames=('min_recursion_length','recursive_split',))
def ssm_scan_backward_recursive (x_chunk, Acoeff, B_chunk, C_chunk, Delta_chunk, dy_chunk, h_left, f_alpha_right, min_recursion_length: int = 2, recursive_split: int = 2):
    L = x_chunk.shape[0]
    if L > min_recursion_length and L % recursive_split == 0:
        mid = jnp.ceil (L // 2)
        x_chunk = einops.rearrange (x_chunk, '(c l) b d -> c l b d', c=recursive_split)
        B_chunk = einops.rearrange (B_chunk, '(c l) b n -> c l b n', c=recursive_split)
        C_chunk = einops.rearrange (C_chunk, '(c l) b n -> c l b n', c=recursive_split)
        Delta_chunk = einops.rearrange (Delta_chunk, '(c l) b d -> c l b d', c=recursive_split)
        dy_chunk = einops.rearrange (dy_chunk, '(c l) b d -> c l b d', c=recursive_split)
        @jit
        def slim_backward_scan_fn (f_alpha_tPlus1, chunk):
            C_t, Delta_t, dy_t = chunk
            alpha_t = jnp.exp (jnp.einsum ('dn,bd->bdn', Acoeff, Delta_t))
            C_dy_t = jnp.einsum ('bn,bd->bdn', C_t, dy_t)
            f_t = f_alpha_tPlus1 + C_dy_t
            return f_t * alpha_t, None
        @jit
        def backward_scan_chunks (f_alpha, chunk):
            C, Delta, dy = chunk
            next_f_alpha, _ = jax.lax.scan (slim_backward_scan_fn, f_alpha, (C, Delta, dy), reverse=True)
            return next_f_alpha, f_alpha
        _f_alpha_left, f_alphas = jax.lax.scan (backward_scan_chunks, f_alpha_right, (C_chunk, Delta_chunk, dy_chunk), reverse=True)
        @jit
        def forward_scan_chunks (carry, chunk):
            dA, h_left = carry
            x, B, C, Delta, dy, f_alpha_right = chunk
            dx_chunk, dA_chunk, dB_chunk, dC_chunk, dDelta_chunk, h_right = ssm_scan_backward_recursive (x, Acoeff, B, C, Delta, dy, h_left, f_alpha_right, min_recursion_length=min_recursion_length, recursive_split=recursive_split)
            dA = dA + dA_chunk
            return (dA, h_right), (dx_chunk, dB_chunk, dC_chunk, dDelta_chunk)
        (dA, h_right), (dxs, dBs, dCs, dDeltas) = jax.lax.scan (forward_scan_chunks,
                                                                (jnp.zeros_like(Acoeff), h_left),
                                                                (x_chunk, B_chunk, C_chunk, Delta_chunk, dy_chunk, f_alphas))
        dxs = einops.rearrange (dxs, 'c l b d -> (c l) b d')
        dBs = einops.rearrange (dBs, 'c l b n -> (c l) b n')
        dCs = einops.rearrange (dCs, 'c l b n -> (c l) b n')
        dDeltas = einops.rearrange (dDeltas, 'c l b d -> (c l) b d')
        return dxs, dA, dBs, dCs, dDeltas, h_right
    else:
        alpha, beta = compute_alpha_beta (x_chunk, Acoeff, B_chunk, Delta_chunk)   # (L,B,D,N) (L,B,D,N)
        C_dy = jnp.einsum ('lbn,lbd->lbdn', C_chunk, dy_chunk)  # (L,B,D,N)
        h_right, hs = jax.lax.scan (forward_scan_fn, h_left, (alpha, beta))  # (B,D,N) (L,B,D,N)
        _f_alpha_left, fs = jax.lax.scan (backward_scan_fn, f_alpha_right, (alpha, C_dy), reverse=True)  # (B,D,N) (L,B,D,N)
        Delta_fs = jnp.einsum ('lbd,lbdn->lbdn', Delta_chunk, fs)
        alpha_hs = jnp.einsum ('lbdn,lbdn->lbdn', alpha, hs)
        dx = jnp.einsum ('lbdn,lbn->lbd', Delta_fs, B_chunk)
        dA = jnp.einsum ('lbdn,lbdn->dn', Delta_fs, alpha_hs)
        dB = jnp.einsum ('lbdn,lbd->lbn', Delta_fs, x_chunk)
        dC = jnp.einsum ('lbd,lbdn->lbn', dy_chunk, jnp.concatenate ([hs[1:,...], h_right[None,...]], axis=0))
        dDelta = jnp.einsum ('lbdn,lbdn->lbd', fs, jnp.einsum('dn,lbdn->lbdn', Acoeff, alpha_hs) + jnp.einsum('lbn,lbd->lbdn', B_chunk, x_chunk))
        return dx, dA, dB, dC, dDelta, h_right

@partial(jit, static_argnames=('min_recursion_length','recursive_split',))
def ssm_scan_backward (min_recursion_length, recursive_split, res, dy):
    x, Acoeff, Bcoeff, Ccoeff, Delta = res
    B = x.shape[-3]
    D = x.shape[-1]
    N = Acoeff.shape[-1]
    x = einops.rearrange (x, 'b l d -> l b d')
    Bcoeff = einops.rearrange (Bcoeff, 'b l n -> l b n')
    Ccoeff = einops.rearrange (Ccoeff, 'b l n -> l b n')
    Delta = einops.rearrange (Delta, 'b l d -> l b d')
    dy = einops.rearrange (dy, 'b l d -> l b d')
    h_left = jnp.zeros ((B, D, N))
    f_alpha_right = jnp.zeros ((B, D, N))
    dx, dA, dB, dC, dDelta, _h_right = ssm_scan_backward_recursive (x, Acoeff, Bcoeff, Ccoeff, Delta, dy, h_left, f_alpha_right, min_recursion_length=min_recursion_length, recursive_split=recursive_split)
    dx = einops.rearrange (dx, 'l b d -> b l d')
    dB = einops.rearrange (dB, 'l b n -> b l n')
    dC = einops.rearrange (dC, 'l b n -> b l n')
    dDelta = einops.rearrange (dDelta, 'l b d -> b l d')
    return dx, dA, dB, dC, dDelta

ssm_scan.defvjp (ssm_scan_forward, ssm_scan_backward)

def l2_norm(params, alpha = 1.):
    return alpha * jnp.sum (jnp.array ([jnp.sum(x*x) for x in jax.tree_util.tree_leaves(params)]))

def inverse_softplus(x):
    return x + jnp.log(1 - jnp.exp(-x))

def debug_log(fmt: str, *args, **kwargs):
  jax.debug.callback(
      lambda *args, **kwargs: logging.warning(fmt.format(*args, **kwargs)),
      *args, **kwargs)

def largest_factor_up_to(b,n):
    if n < 2:
        return n
    k = b
    while n % k != 0:
        k -= 1
    return k

# x: (B, L, D)
# Acoeff: (D, N)
# Bcoeff: (B, L, N)
# Ccoeff: (B, L, N)
# dt: (B, L, D) or (B, L, 1);  can assume (B, L, D) and rely on broadcasting
def ssm_chunked_scan (x, Acoeff, Bcoeff, Ccoeff, dt, chunk_size: int = None, n_channel_groups: int = 1):
    B = x.shape[-3]
    L = x.shape[-2]
    D = x.shape[-1]
    N = Acoeff.shape[-1]

    if n_channel_groups is not None:
        K = n_channel_groups
    else:
        K = 1
    if D % K != 0:
        raise ValueError(f"n_channel_groups={n_channel_groups} must divide D={D}")

    if chunk_size is None:
        chunk_size = largest_factor_up_to(int(math.sqrt(K*L)),L)

    if L % chunk_size != 0:
        raise ValueError(f"chunk_size={chunk_size} must divide L={L}")
    n_chunks = L // chunk_size

    # Transpose length & batch dimensions to make the scan over length, and split into chunks
    # This is a bit inefficient, but taking dynamic slices appears to be worse
    x_chunks = einops.rearrange (x, 'b (c l) (k d) -> c k l b d', c=n_chunks, k=K)
    A_blocks = einops.rearrange (Acoeff, '(k d) n -> k d n', k=K)
    B_chunks = einops.rearrange (Bcoeff, 'b (c l) n -> c l b n', c=n_chunks)
    C_chunks = einops.rearrange (Ccoeff, 'b (c l) n -> c l b n', c=n_chunks)
    dt_chunks = einops.rearrange (dt, 'b (c l) (k d) -> c k l b d', c=n_chunks, k=K)

    # Function to do an associative scan for a single chunk
    # We decorate this with @jax.remat to flag that we are OK with re-performing this scan whenever needed
    @jax.remat
    def scan_chunk (carry, chunk):
        # For the purposes of shape annotation within this code we write D instead of D/K
        g_init, h_init = carry  # (1, B, D, N)  (1, B, D, N)

        x_chunk, A_block, B_chunk, C_chunk, dt_chunk = chunk
        # dA = exp(A*dt) [zero-order hold], dB = B*dt*x [Euler step]
        dA = jnp.exp (jnp.einsum ('dn,lbd->lbdn', A_block, dt_chunk))  # (chunk_size, B, D, N)
        dB = jnp.einsum ('lbn,lbd,lbd->lbdn', B_chunk, x_chunk, dt_chunk)  # (chunk_size, B, D, N)
        # The associative scan is a product of matrices of the form ((g,h),(0,1)) where g_i=exp(A*dt)x_i and h_i=B*dt*x_i
        # Since matrices of this form are are closed under multiplication, we can represent all intermediate products in the same way
        @jax.remat
        def associative_scan_fn (l, r):  # l, r, and return value are tuples of the form ((B,D,N), (B,D,N))
            g_l, h_l = l
            g_r, h_r = r
            return tuple((g_l*g_r, g_r*h_l + h_r))
        gs, hs = jax.lax.associative_scan (associative_scan_fn, (dA, dB))  # (chunk_size, B, D, N)  (chunk_size, B, D, N)
        hs = gs * h_init + hs  # Incorporate h_init here so that it is reflected in y_chunk
        # We only need to keep the last state of gs, so we can discard the rest. Otherwise we would incorporate g_init here, like so:
        # gs = g_init * gs
        y_chunk = jnp.einsum ('lbn,lbdn->lbd', C_chunk, hs)  # (chunk_size, B, D)
        return (gs[-1:,...] * g_init, hs[-1:,...]), y_chunk  # note g_init incorporated here

    # A wrapper that splits the dimensions into K blocks and does the inner associative scan for each block, re-using B and C (which don't change across dimensions)
    @jax.remat
    def scan_chunk_mapped (carry, chunk):
        g_init, h_init = carry  # (K,1,B,D/K,N) (K,1,B,D/K,N)

        x_chunk, B_chunk, C_chunk, dt_chunk = chunk   # (K,B,L,D/K), (B,L,N), (B,L,N), (K,B,L,D/K)
        @jax.remat
        def scan_chunk_wrapper (block):
            dA_init_block, dB_init_block, x_chunk_block, A_block, dt_chunk_block = block
            return scan_chunk ((dA_init_block, dB_init_block), (x_chunk_block, A_block, B_chunk, C_chunk, dt_chunk_block))
        return jax.lax.map (scan_chunk_wrapper, (g_init, h_init, x_chunk, A_blocks, dt_chunk))


    # Perform the scan over chunks recurrently (with rematerialization as noted above), with each chunk being an associative scan
    (_A_final, _h_final), y_chunks = jax.lax.scan (scan_chunk_mapped, (jnp.ones((K,1,B,D//K,N)), jnp.zeros((K,1,B,D//K,N))), (x_chunks, B_chunks, C_chunks, dt_chunks))  # (K, n_chunks, B, D//K)

    return einops.rearrange (y_chunks, 'c k l b d -> b (c l) (k d)')  # (B, L, D)


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

        if train and self.diagnostics.ssm_input_norm:
            self.sow("diagnostics", "ssm_input_mean", jnp.mean(x))
            self.sow("diagnostics", "ssm_input_sd", jnp.std(x))

        if self.reverse:
            x = jnp.flip (x, axis=(-2,-1) if self.complement else -2)

        u = nn.Conv (features=D, feature_group_count=D, kernel_size=(self.shift_conv_size,), strides=(1,), padding="SAME", use_bias=False, name="shift_conv", kernel_init=nn.initializers.lecun_normal()) (x)  # (B, L, D)

        if train and self.diagnostics.ssm_coeffs:
            self.sow("diagnostics", "conv_mean", jnp.mean(u))
            self.sow("diagnostics", "conv_sd", jnp.std(u))

        if self.activation == "gelu":
            u = nn.gelu(u)
        elif self.activation == "relu":
            u = nn.relu(u)
        elif self.activation == "silu":
            u = nn.silu(u)
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

        if train and self.diagnostics.ssm_coeffs:
            self.sow("diagnostics", "dt_lowrank_mean", jnp.mean(dt))
            self.sow("diagnostics", "dt_lowrank_sd", jnp.std(dt))

        if self.dt_proj:
            dt = nn.Dense (features=D, use_bias=True, kernel_init=nn.initializers.lecun_normal(), bias_init=dt_bias_init, name='dt_proj') (dt)  # (B, L, D)
        else:
            if dt_rank > 1:  # if dt_rank is 1, we can just rely on broadcasting, and save memory
                if D % dt_rank != 0:
                    raise ValueError(f"dt_rank={dt_rank} must divide D={D}")
                dt = jnp.repeat (dt, D // dt_rank, axis=-1)  # (B, L, D)
        # dt = nn.activation.softplus (dt)  # (B, L, D) or (B, L, 1)
        dt = jnp.log1p(jnp.exp(dt))

        if train and self.diagnostics.ssm_coeffs:
            self.sow("diagnostics", "activated_conv_mean", jnp.mean(u))
            self.sow("diagnostics", "activated_conv_sd", jnp.std(u))
            self.sow("diagnostics", "dt_mean", jnp.mean(dt))
            self.sow("diagnostics", "dt_sd", jnp.std(dt))
            self.sow("diagnostics", "A_mean", jnp.mean(Acoeff))
            self.sow("diagnostics", "A_sd", jnp.std(Acoeff))
            self.sow("diagnostics", "B_sd", jnp.std(Bcoeff))
            self.sow("diagnostics", "C_sd", jnp.std(Ccoeff))

        # Perform SSM scan
        if self.custom_vjp_scan:
            y = ssm_scan (x, Acoeff, Bcoeff, Ccoeff, dt, min_recursion_length=self.min_recursion_length, recursive_split=self.recursive_split)  # (B, L, D)
        elif self.recursive_scan:
            y = ssm_recursive_scan (x, Acoeff, Bcoeff, Ccoeff, dt, min_recursion_length=self.min_recursion_length, recursive_split=self.recursive_split)  # (B, L, D)
        else:
            y = ssm_chunked_scan (x, Acoeff, Bcoeff, Ccoeff, dt, chunk_size=self.chunk_size, n_channel_groups=self.n_channel_groups)  # (B, L, D)

        if self.reverse:
            y = jnp.flip (y, axis=(-2,-1) if self.complement else -2)

        if train and self.diagnostics.ssm_residual:
            self.sow("diagnostics", "ssm_residual_mean", jnp.mean(y))
            self.sow("diagnostics", "ssm_residual_sd", jnp.std(y))

        # Add in the skip connection term
        y = y + jnp.einsum ('bld,d->bld', x, Dcoeff)

        # Regularizers
        if train:
            # add l2 norm for params
            self.sow("losses", "ssm_regularizer", l2_norm (self.variables['params'], self.l2_scale))

        if train and self.diagnostics.ssm_output_norm:
            self.sow("diagnostics", "ssm_output_mean", jnp.mean(y))
            self.sow("diagnostics", "ssm_output_sd", jnp.std(y))

        return y



class ZeroOnUnitBall(nn.Module):
    @nn.compact
    def __call__(self, x_in, u_val):
        # Remove the sequence dimension (B, 1, D) -> (B, D) for the boundary calculation
        # x = x_in.squeeze(1)
        return (1**2 - jnp.sum(x_in**2, -1)) * u_val

class BidirectionalMamba(nn.Module):
    hidden_features: int   # N
    expansion_factor: float  # E
    dt_rank: Union[int, str] = 'auto'
    complement: bool = False
    tie_in_proj: bool = False
    tie_gate: bool = False
    concatenate_fwd_rev: bool = True
    activation: str = "silu"
    norm_type: str = "rms"
    bn_momentum: float = 0.9
    mlp_layer: bool = False
    dense_expansion: int = 2
    mlp_dropout_rate: float = 0.1
    ssm_args: Dict[str, Any] = field(default_factory=dict)
    diagnostics: DiagnosticsConfig = field(default_factory=lambda: DiagnosticsConfig())
    l2_scale: float = 1e-6

    @nn.compact
    def __call__(self, x, train: bool = False):
        # Ensure input is 3D: (batch_size, sequence_length, features)
        B = x.shape[0]
        D = x.shape[-1]
        L = x.shape[-2]
        x = x.reshape(B, L, D)

        # Apply layer normalization
        # x = nn.LayerNorm()(x)
        # x = x

        input_features = x.shape[-1]  # D
        x_in = x

        if self.dt_rank == 'auto':
            dt_rank = math.ceil(input_features / 16)
        else:
            dt_rank = self.dt_rank

        if self.activation == "gelu":
            activate = nn.gelu
        elif self.activation == "silu":
            activate = nn.silu
        elif self.activation == "relu":
            activate = nn.relu
        elif self.activation == "tanh":
            activate = nn.tanh
        else:
            raise Exception(f"Unknown activation: {self.activation}")

        skip = x
        if self.diagnostics.skip and train:
            self.sow("diagnostics", "skip_mean", jnp.mean(skip))
            self.sow("diagnostics", "skip_sd", jnp.std(skip))

        # normalize
        if self.norm_type == "batch":
            x = nn.BatchNorm(momentum=self.bn_momentum, use_running_average=not train)(x)
        elif self.norm_type == "layer":
            x = nn.LayerNorm()(x)
            # x = x
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
        xr = ssm(hidden_features=self.hidden_features, reverse=True, complement=self.complement, dt_rank=dt_rank, diagnostics=self.diagnostics, **self.ssm_args) (xr, train)

        if self.diagnostics.gate and train:
            self.sow("diagnostics", "gate_fwd_mean", jnp.mean(zf))
            self.sow("diagnostics", "gate_fwd_sd", jnp.std(zf))
            self.sow("diagnostics", "gate_rev_mean", jnp.mean(zr))
            self.sow("diagnostics", "gate_rev_sd", jnp.std(zr))

        # concatenate (or add) forward and backward channels, multiplied by respective activated gates
        if self.concatenate_fwd_rev:
            x = jnp.concatenate ([xf * activate(zf), xr * activate(zr)], axis=-1)
        else:
            x = xf * activate(zf) + xr * activate(zr)

        if self.diagnostics.gated and train:
            self.sow("diagnostics", "gated_mean", jnp.mean(x))
            self.sow("diagnostics", "gated_sd", jnp.std(x))

        # project back down
        x = nn.Dense (features=input_features, name='out_proj', kernel_init=nn.initializers.lecun_normal()) (x)

        # residual add
        if self.diagnostics.residual and train:
            self.sow("diagnostics", "residual_mean", jnp.mean(x))
            self.sow("diagnostics", "residual_sd", jnp.std(x))

        x_out = skip + x

        # MLP layer (optional)
        if self.mlp_layer:
            x_out = nn.Dense(self.dense_expansion*input_features, name="mlp", kernel_init=nn.initializers.lecun_normal())(x_out)
            x_out = activate(x_out)
            x_out = nn.Dense(1, name="mlp_proj", kernel_init=nn.initializers.lecun_normal())(x_out)
            x_out = ZeroOnUnitBall(name="zero_unit_ball")(x_in, x_out.squeeze(-1))


        # Regularizers
        if train:
            self.sow("losses", "mamba_regularizer", l2_norm (self.variables['params'], self.l2_scale))

        return x_out


@partial(jax.jit, static_argnames=['n_pts', 'dim', 'radius'])
def sample_domain_fn(n_pts: int, dim: int, radius: float, rng: jax.Array):
    keys = jax.random.split(rng, 6)
    r = jax.random.uniform(keys[0], (n_pts, 1)) * radius
    x = jax.random.normal(keys[1], (n_pts, dim))
    x = x / jnp.linalg.norm(x, axis=-1, keepdims=True) * r
    # Add unit dimension in the middle to make it (B, 1, D)
    x = jnp.expand_dims(x, axis=1)
    t = jax.random.uniform(keys[2], (n_pts, 1))
    return x, t, keys[5]

def test_bidirectional_mamba_initialization():
    print("\n=== Testing Bidirectional Mamba Initialization ===")
    # Test parameters
    n_pts = 8
    dim = 1
    radius = 1.0

    print(f"Test parameters: n_pts={n_pts}, dim={dim}, radius={radius}")

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
    x, t, _ = sample_domain_fn(n_pts, dim, radius, key)
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
    dim = 1
    radius = 1.0

    print(f"Test parameters: n_pts={n_pts}, dim={dim}, radius={radius}")

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
    x, t, _ = sample_domain_fn(n_pts, dim, radius, key)
    print(f"Input shape: {x.shape}")

    # Initialize parameters and run forward pass
    variables = model.init(key, x)
    print("Parameters initialized successfully")

    output = model.apply(variables, x)
    print(f"Forward pass completed successfully")
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[:2]}")

    # Test output shape
    expected_shape = x.squeeze(-1).shape
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"
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
    dim = 1
    radius = 1.0

    print(f"Test parameters: n_pts={n_pts}, dim={dim}, radius={radius}")

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
    x, t, _ = sample_domain_fn(n_pts, dim, radius, key)
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

        expected_shape = x.squeeze(1).shape
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
    dim = 1
    radius = 1.0

    print(f"Test parameters: n_pts={n_pts}, dim={dim}, radius={radius}")

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
    x, t, _ = sample_domain_fn(n_pts, dim, radius, key)
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
    n_pts = 10
    dim = 1
    radius = 1.0

    print(f"Input parameters: n_pts={n_pts}, dim={dim}, radius={radius}")

    # Create input using sample_domain_fn
    key = jax.random.PRNGKey(0)
    x, t, _ = sample_domain_fn(n_pts, dim, radius, key)
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


if __name__ == "__main__":
    print("\n===================================================")
    print("RUNNING MAMBA MODEL TESTS")
    print("===================================================")

    test_bidirectional_mamba_initialization()
    test_bidirectional_mamba_forward()
    test_bidirectional_mamba_configurations()
    test_bidirectional_mamba_training_mode()
    tabulate_bidirectional_mamba()

    print("\n===================================================")
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("===================================================")

