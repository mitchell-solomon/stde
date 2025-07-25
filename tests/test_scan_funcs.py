import numpy as np
import time
import math
from functools import partial
import einops
import jax
import jax.numpy as jnp
from jax import custom_vjp, jit
from jax.experimental.jet import jet
from jax import config
# config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import seaborn as sns

def largest_factor_up_to(b,n):
    if n < 2:
        return n
    k = b
    while n % k != 0:
        k -= 1
    return k

@jax.remat
def associative_scan_fn (l, r):
    g_l, h_l = l
    g_r, h_r = r
    return tuple((g_l*g_r, g_r*h_l + h_r))

@jit
def compute_alpha (Acoeff, Delta_chunk):
    return jnp.exp (jnp.einsum ('dn,lbd->lbdn', Acoeff, Delta_chunk))  # (chunk_size, B, D, N)

@jit
def compute_alpha_beta (x_chunk, Acoeff, B_chunk, Delta_chunk):
    alpha = compute_alpha (Acoeff, Delta_chunk)  # (chunk_size, B, D, N)
    beta = jnp.einsum ('lbn,lbd,lbd->lbdn', B_chunk, x_chunk, Delta_chunk)  # (chunk_size, B, D, N)
    return alpha, beta

# # ─────────── OLD FUNCTION ───────────
# @jit
# def ssm_parallel_scan(x, Acoeff, Bcoeff, Ccoeff, Delta):
#     """
#     Replace all lax.scan logic with one-shot cumprod/cumsum.
#     x:   (B, L, D)
#     Acoeff: (D, N)
#     Bcoeff: (B, L, N)
#     Ccoeff: (B, L, N)
#     Delta:  (B, L, D)
#     returns y: (B, L, D)
#     """
#     # 1) transpose into time-major
#     x_t     = einops.rearrange(x,     'b l d -> l b d')       # (L, B, D)
#     B_t     = einops.rearrange(Bcoeff,'b l n -> l b n')       # (L, B, N)
#     C_t     = einops.rearrange(Ccoeff,'b l n -> l b n')       # (L, B, N)
#     Δ_t     = einops.rearrange(Delta, 'b l d -> l b d')       # (L, B, D)

#     # 2) compute per-step α, β
#     α, β    = compute_alpha_beta(x_t, Acoeff, B_t, Δ_t)        # each (L, B, D, N)

#     # 3) prefix-product of α
#     G       = jnp.cumprod(α, axis=0)                           # (L, B, D, N)
#     ones    = jnp.ones_like(α[:1])                             # (1, B, D, N)
#     G0      = jnp.concatenate([ones, G[:-1]], axis=0)         # (L, B, D, N)

#     # 4) weighted prefix-sum of β
#     S       = jnp.cumsum(β * G0, axis=0)                       # (L, B, D, N)

#     # 5) recover h_t = G * h0 + S
#     h0      = jnp.ones_like(α[0])                              # (B, D, N)
#     h       = G * h0[None, ...] + S                            # (L, B, D, N)

#     # 6) project through C
#     y_t     = jnp.einsum('lbn,lbdn->lbd', C_t, h)              # (L, B, D)

#     # 7) back to (B, L, D)
#     return einops.rearrange(y_t, 'l b d -> b l d')
# # ──────────────────────────────────────

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

# Copy of ssm_chunked_scan from mamba.py
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

def test_scan_functions():
    """Test and visualize different scan functions."""
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Define test parameters
    B = 2       # Batch size
    L = 100     # Sequence length
    D = 4       # Input dimension
    N = 8       # Hidden dimension
    
    # Generate random inputs
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (B, L, D))
    
    key, subkey = jax.random.split(key)
    Acoeff = -jnp.exp(jax.random.normal(subkey, (D, N)))  # Negative to ensure stability
    
    key, subkey = jax.random.split(key)
    Bcoeff = jax.random.normal(subkey, (B, L, N))
    
    key, subkey = jax.random.split(key)
    Ccoeff = jax.random.normal(subkey, (B, L, N))
    
    key, subkey = jax.random.split(key)
    Delta = jnp.abs(jax.random.normal(subkey, (B, L, D))) * 0.1  # Small positive values
    
    # JIT-compile the scan functions
    parallel_scan_jit = jax.jit(ssm_parallel_scan)
    recursive_scan_jit = jax.jit(ssm_recursive_scan)
    chunked_scan_jit = jax.jit(ssm_chunked_scan)
    
    # Warm-up JIT
    _ = parallel_scan_jit(x, Acoeff, Bcoeff, Ccoeff, Delta)
    _ = recursive_scan_jit(x, Acoeff, Bcoeff, Ccoeff, Delta)
    _ = chunked_scan_jit(x, Acoeff, Bcoeff, Ccoeff, Delta)
    
    # Time the functions
    start_time = time.time()
    y_parallel = parallel_scan_jit(x, Acoeff, Bcoeff, Ccoeff, Delta)
    parallel_time = time.time() - start_time
    
    start_time = time.time()
    y_recursive = recursive_scan_jit(x, Acoeff, Bcoeff, Ccoeff, Delta)
    recursive_time = time.time() - start_time
    
    start_time = time.time()
    y_chunked = chunked_scan_jit(x, Acoeff, Bcoeff, Ccoeff, Delta)
    chunked_time = time.time() - start_time
    
    # Convert to numpy for plotting
    y_parallel_np = np.array(y_parallel)
    assert not np.isnan(y_parallel_np).any(), "NaN values found in parallel scan output"
    y_recursive_np = np.array(y_recursive)
    assert not np.isnan(y_recursive_np).any(), "NaN values found in recursive scan output"
    y_chunked_np = np.array(y_chunked)
    assert not np.isnan(y_chunked_np).any(), "NaN values found in chunked scan output"
    
    # Calculate differences
    diff_parallel_recursive = np.abs(y_parallel_np - y_recursive_np)
    diff_parallel_chunked = np.abs(y_parallel_np - y_chunked_np)
    diff_recursive_chunked = np.abs(y_recursive_np - y_chunked_np)
    
    # Create plots
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Output comparison for a single channel
    channel = 0
    plt.subplot(3, 2, 1)
    plt.plot(y_parallel_np[0, :, channel], label='Parallel')
    plt.plot(y_recursive_np[0, :, channel], label='Recursive', linestyle='--')
    plt.plot(y_chunked_np[0, :, channel], label='Chunked', linestyle=':')
    plt.title(f'Output Comparison (Channel {channel})')
    plt.xlabel('Sequence Position')
    plt.ylabel('Output Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Difference between implementations
    plt.subplot(3, 2, 2)
    plt.semilogy(np.mean(diff_parallel_recursive, axis=(0, 2)), label='|Parallel - Recursive|')
    plt.semilogy(np.mean(diff_parallel_chunked, axis=(0, 2)), label='|Parallel - Chunked|')
    plt.semilogy(np.mean(diff_recursive_chunked, axis=(0, 2)), label='|Recursive - Chunked|')
    plt.title('Mean Absolute Difference (Log Scale)')
    plt.xlabel('Sequence Position')
    plt.ylabel('Mean Abs Difference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Heatmap of outputs for each implementation
    plt.subplot(3, 2, 3)
    sns.heatmap(y_parallel_np[0, :, :], cmap='viridis')
    plt.title('Parallel Scan Output Heatmap')
    plt.xlabel('Channel')
    plt.ylabel('Sequence Position')
    
    plt.subplot(3, 2, 4)
    sns.heatmap(y_recursive_np[0, :, :], cmap='viridis')
    plt.title('Recursive Scan Output Heatmap')
    plt.xlabel('Channel')
    plt.ylabel('Sequence Position')
    
    plt.subplot(3, 2, 5)
    sns.heatmap(y_chunked_np[0, :, :], cmap='viridis')
    plt.title('Chunked Scan Output Heatmap')
    plt.xlabel('Channel')
    plt.ylabel('Sequence Position')
    
    # Plot 6: Performance comparison
    plt.subplot(3, 2, 6)
    methods = ['Parallel', 'Recursive', 'Chunked']
    times = [parallel_time, recursive_time, chunked_time]
    plt.bar(methods, times)
    plt.title('Execution Time Comparison')
    plt.ylabel('Time (seconds)')
    for i, v in enumerate(times):
        plt.text(i, v + 0.01, f'{v:.4f}s', ha='center')
    
    plt.tight_layout()
    plt.savefig('scan_comparison.png', dpi=300)
    plt.show()
    
    # Test forward-mode AD compatibility
    print("\nTesting forward-mode AD compatibility:")
    
    # Define a simple function that uses the scan
    def f_parallel(x):
        return jnp.sum(ssm_parallel_scan(x, Acoeff, Bcoeff, Ccoeff, Delta))
    
    def f_recursive(x):
        return jnp.sum(ssm_recursive_scan(x, Acoeff, Bcoeff, Ccoeff, Delta))
    
    # Create a tangent vector
    v = jax.random.normal(key, x.shape)
    # Test jet with parallel scan
    try:
        print("Testing jet with parallel_scan...")
        primals, series = jet(f_parallel, (x,), ((v,),))
        print("✓ Parallel scan is compatible with jax.experimental.jet")
        jet_parallel_success = True
    except Exception as e:
        print("✗ Parallel scan failed with jet:", str(e))
        jet_parallel_success = False
    
    # Test jet with recursive scan
    try:
        print("\nTesting jet with recursive_scan...")
        primals, series = jet(f_recursive, (x,), ((v,),))
        print("✓ Recursive scan is compatible with jax.experimental.jet")
        jet_recursive_success = True
    except Exception as e:
        print("✗ Recursive scan failed with jet:", str(e))
        jet_recursive_success = False
    
    # Test JVP with both implementations
    print("\nTesting jvp with both implementations:")
    
    try:
        jvp_parallel = jax.jvp(f_parallel, (x,), (v,))
        print("✓ Parallel scan is compatible with jax.jvp")
        jvp_parallel_success = True
    except Exception as e:
        print("✗ Parallel scan failed with jvp:", str(e))
        jvp_parallel_success = False
    
    try:
        jvp_recursive = jax.jvp(f_recursive, (x,), (v,))
        print("✓ Recursive scan is compatible with jax.jvp")
        jvp_recursive_success = True
    except Exception as e:
        print("✗ Recursive scan failed with jvp:", str(e))
        jvp_recursive_success = False
    
    # Compare jet and jvp results if both succeeded
    if jet_parallel_success and jvp_parallel_success:
        print("\nComparing jet and jvp results for parallel scan:")
        jet_result = series[0]
        _, jvp_result = jvp_parallel
        diff = jnp.abs(jet_result - jvp_result)
        print(f"Max absolute difference: {jnp.max(diff):.2e}")
        print(f"Mean absolute difference: {jnp.mean(diff):.2e}")
    
    # Create a more detailed visualization of differences
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Error accumulation along sequence
    plt.subplot(2, 2, 1)
    seq_error_pr = np.mean(diff_parallel_recursive, axis=(0, 2))
    seq_error_pc = np.mean(diff_parallel_chunked, axis=(0, 2))
    seq_error_rc = np.mean(diff_recursive_chunked, axis=(0, 2))
    
    plt.semilogy(seq_error_pr, label='|Parallel - Recursive|')
    plt.semilogy(seq_error_pc, label='|Parallel - Chunked|')
    plt.semilogy(seq_error_rc, label='|Recursive - Chunked|')
    plt.title('Error Accumulation Along Sequence')
    plt.xlabel('Sequence Position')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Error distribution across channels
    plt.subplot(2, 2, 2)
    channel_error_pr = np.mean(diff_parallel_recursive, axis=(0, 1))
    channel_error_pc = np.mean(diff_parallel_chunked, axis=(0, 1))
    channel_error_rc = np.mean(diff_recursive_chunked, axis=(0, 1))
    
    plt.bar(np.arange(D) - 0.2, channel_error_pr, width=0.2, label='|Parallel - Recursive|')
    plt.bar(np.arange(D), channel_error_pc, width=0.2, label='|Parallel - Chunked|')
    plt.bar(np.arange(D) + 0.2, channel_error_rc, width=0.2, label='|Recursive - Chunked|')
    plt.title('Error Distribution Across Channels')
    plt.xlabel('Channel')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Detailed view of a specific sequence segment
    segment_start = 40
    segment_length = 20
    segment_end = segment_start + segment_length
    
    plt.subplot(2, 2, 3)
    for ch in range(min(D, 3)):  # Plot first 3 channels
        plt.plot(range(segment_start, segment_end), 
                 y_parallel_np[0, segment_start:segment_end, ch], 
                 label=f'Parallel Ch{ch}', 
                 marker='o')
        plt.plot(range(segment_start, segment_end), 
                 y_recursive_np[0, segment_start:segment_end, ch], 
                 label=f'Recursive Ch{ch}', 
                 marker='x', 
                 linestyle='--')
    
    plt.title(f'Detailed View of Sequence Segment [{segment_start}:{segment_end}]')
    plt.xlabel('Sequence Position')
    plt.ylabel('Output Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Relative error distribution (histogram)
    plt.subplot(2, 2, 4)
    
    # Calculate relative errors where values are significant
    mask = np.abs(y_recursive_np) > 1e-10
    rel_error_pr = np.abs(y_parallel_np - y_recursive_np) / (np.abs(y_recursive_np) + 1e-10) * mask
    rel_error_pc = np.abs(y_parallel_np - y_chunked_np) / (np.abs(y_chunked_np) + 1e-10) * mask
    
    plt.hist(rel_error_pr.flatten(), bins=50, alpha=0.5, label='Parallel vs Recursive')
    plt.hist(rel_error_pc.flatten(), bins=50, alpha=0.5, label='Parallel vs Chunked')
    plt.title('Relative Error Distribution')
    plt.xlabel('Relative Error')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig('scan_detailed_comparison.png', dpi=300)
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"{'Method':<10} {'Mean Error':<15} {'Max Error':<15} {'Time (s)':<10}")
    print("-" * 50)
    print(f"{'Par vs Rec':<10} {np.mean(diff_parallel_recursive):<15.2e} {np.max(diff_parallel_recursive):<15.2e} {parallel_time:<10.4f}")
    print(f"{'Par vs Chk':<10} {np.mean(diff_parallel_chunked):<15.2e} {np.max(diff_parallel_chunked):<15.2e} {recursive_time:<10.4f}")
    print(f"{'Rec vs Chk':<10} {np.mean(diff_recursive_chunked):<15.2e} {np.max(diff_recursive_chunked):<15.2e} {chunked_time:<10.4f}")
    
    # Test with longer sequences to see how errors accumulate
    print("\nTesting with longer sequences...")
    
    # Define sequence lengths to test
    seq_lengths = [10, 100, 1000]
    max_errors = []
    mean_errors = []
    
    for seq_len in seq_lengths:
        print(f"\nSequence length: {seq_len}")
        
        # Generate new random inputs with the current sequence length
        key, subkey = jax.random.split(key)
        x_long = jax.random.normal(subkey, (B, seq_len, D))
        
        key, subkey = jax.random.split(key)
        Bcoeff_long = jax.random.normal(subkey, (B, seq_len, N))
        
        key, subkey = jax.random.split(key)
        Ccoeff_long = jax.random.normal(subkey, (B, seq_len, N))
        
        key, subkey = jax.random.split(key)
        Delta_long = jnp.abs(jax.random.normal(subkey, (B, seq_len, D))) * 0.1
        
        # Run the scan functions
        y_parallel_long = parallel_scan_jit(x_long, Acoeff, Bcoeff_long, Ccoeff_long, Delta_long)
        y_recursive_long = recursive_scan_jit(x_long, Acoeff, Bcoeff_long, Ccoeff_long, Delta_long)
        
        # Calculate differences
        diff_long = np.abs(np.array(y_parallel_long) - np.array(y_recursive_long))
        max_err = np.max(diff_long)
        mean_err = np.mean(diff_long)
        
        max_errors.append(max_err)
        mean_errors.append(mean_err)
        
        print(f"Max error: {max_err:.2e}")
        print(f"Mean error: {mean_err:.2e}")
    
    # Plot error growth with sequence length
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(seq_lengths, max_errors, marker='o')
    plt.title('Maximum Error vs Sequence Length')
    plt.xlabel('Sequence Length')
    plt.ylabel('Maximum Absolute Error')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(seq_lengths, mean_errors, marker='o')
    plt.title('Mean Error vs Sequence Length')
    plt.xlabel('Sequence Length')
    plt.ylabel('Mean Absolute Error')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scan_error_vs_length.png', dpi=300)
    plt.show()
    
    # Test with different hidden dimensions to see impact on performance
    print("\nTesting with different hidden dimensions...")
    
    # Define hidden dimensions to test
    hidden_dims = [4, 8, 16, 32]
    parallel_times = []
    recursive_times = []
    chunked_times = []
    
    for N_test in hidden_dims:
        print(f"\nHidden dimension: {N_test}")
        
        # Generate new random inputs with the current hidden dimension
        key, subkey = jax.random.split(key)
        Acoeff_N = -jnp.exp(jax.random.normal(subkey, (D, N_test)))
        
        key, subkey = jax.random.split(key)
        Bcoeff_N = jax.random.normal(subkey, (B, L, N_test))
        
        key, subkey = jax.random.split(key)
        Ccoeff_N = jax.random.normal(subkey, (B, L, N_test))
        
        # Time the functions
        start_time = time.time()
        _ = parallel_scan_jit(x, Acoeff_N, Bcoeff_N, Ccoeff_N, Delta)
        parallel_time_N = time.time() - start_time
        parallel_times.append(parallel_time_N)
        
        start_time = time.time()
        _ = recursive_scan_jit(x, Acoeff_N, Bcoeff_N, Ccoeff_N, Delta)
        recursive_time_N = time.time() - start_time
        recursive_times.append(recursive_time_N)
        
        start_time = time.time()
        _ = chunked_scan_jit(x, Acoeff_N, Bcoeff_N, Ccoeff_N, Delta)
        chunked_time_N = time.time() - start_time
        chunked_times.append(chunked_time_N)
        
        print(f"Parallel time: {parallel_time_N:.4f}s")
        print(f"Recursive time: {recursive_time_N:.4f}s")
        print(f"Chunked time: {chunked_time_N:.4f}s")
    
    # Plot performance vs hidden dimension
    plt.figure(figsize=(10, 6))
    
    plt.plot(hidden_dims, parallel_times, marker='o', label='Parallel')
    plt.plot(hidden_dims, recursive_times, marker='s', label='Recursive')
    plt.plot(hidden_dims, chunked_times, marker='^', label='Chunked')
    
    plt.title('Execution Time vs Hidden Dimension')
    plt.xlabel('Hidden Dimension (N)')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scan_time_vs_dimension.png', dpi=300)
    plt.show()
    
    return {
        'parallel_output': y_parallel_np,
        'recursive_output': y_recursive_np,
        'chunked_output': y_chunked_np,
        'parallel_time': parallel_time,
        'recursive_time': recursive_time,
        'chunked_time': chunked_time,
        'max_errors': max_errors,
        'mean_errors': mean_errors,
        'hidden_dims': hidden_dims,
        'parallel_times': parallel_times,
        'recursive_times': recursive_times,
        'chunked_times': chunked_times,
    }

if __name__ == "__main__":
    results = test_scan_functions()
    print("\nTest completed successfully!")