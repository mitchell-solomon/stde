import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def random_unit_vectors(key, batch_size, dim):
    v = jax.random.normal(key, (batch_size, dim))
    return v / jnp.linalg.norm(v, axis=-1, keepdims=True)

def sample_dirichlet_segments(key, batch_size, seq_len, radius, dim):
    # Straight ray with Dirichlet‐distributed breakpoints
    sub_r, sub_w, sub_dir = jax.random.split(key, 3)
    r_start, r_end = 0.0, radius
    # Dirichlet weights via Gamma(1.0)
    w = jax.random.gamma(sub_w, 1.0, (batch_size, seq_len))
    w = w / jnp.sum(w, axis=1, keepdims=True)
    t = jnp.cumsum(w, axis=1).reshape(batch_size, seq_len, 1)
    dirs = random_unit_vectors(sub_dir, batch_size, dim)
    radii = (1 - t) * r_start + t * r_end
    return dirs[:, None, :] * radii  # (B, L, D)

def sample_brownian_bridge(key, batch_size, seq_len, radius, dim):
    # Brownian‐bridge around a straight interpolation
    key0, key1, key_noise = jax.random.split(key, 3)
    dirs0 = random_unit_vectors(key0, batch_size, dim)
    dirs1 = random_unit_vectors(key1, batch_size, dim)
    r0, r1 = 0.0, radius
    t_lin = jnp.linspace(0, 1, seq_len).reshape(1, seq_len, 1)
    interp_dirs = (1 - t_lin) * dirs0[:, None, :] + t_lin * dirs1[:, None, :]
    interp_r = (1 - t_lin) * r0 + t_lin * r1
    x_lin = interp_dirs * interp_r
    sigma = radius * 0.1
    noise = jax.random.normal(key_noise, (batch_size, seq_len, dim))
    scale = jnp.sqrt(t_lin * (1 - t_lin))
    return x_lin + sigma * scale * noise

def sample_random_walk(key, batch_size, seq_len, radius, dim, step_scale=0.05):
    subkeys = jax.random.split(key, seq_len + 1)
    key0, step_keys = subkeys[0], subkeys[1:]
    # start uniformly in ball
    u = jax.random.uniform(key0, (batch_size, 1))
    r0 = radius * (u ** (1.0 / dim))
    dirs0 = random_unit_vectors(key0, batch_size, dim)
    x0 = dirs0 * r0
    def step_fn(prev, subkey):
        k1, k2 = jax.random.split(subkey)
        dv = jax.random.normal(k1, (batch_size, dim))
        dv = dv / jnp.linalg.norm(dv, axis=-1, keepdims=True)
        dr = jax.random.uniform(k2, (batch_size, 1), minval=0.0, maxval=step_scale * radius)
        nxt = prev + dv * dr
        norm = jnp.linalg.norm(nxt, axis=-1, keepdims=True)
        # project back into ball
        nxt = jnp.where(norm > radius, nxt / norm * radius, nxt)
        return nxt, nxt
    _, seq = jax.lax.scan(step_fn, x0, step_keys)
    return jnp.transpose(seq, (1, 0, 2))  # (B, L, D)

def sample_slerp_arc(key, batch_size, seq_len, radius, dim):
    key0, key1 = jax.random.split(key)
    u0 = random_unit_vectors(key0, batch_size, dim)
    u1 = random_unit_vectors(key1, batch_size, dim)
    dot = jnp.sum(u0 * u1, axis=-1, keepdims=True)
    omega = jnp.arccos(jnp.clip(dot, -1+1e-6, 1-1e-6))
    t_lin = jnp.linspace(0, 1, seq_len).reshape(1, seq_len, 1)
    sin_omega = jnp.sin(omega)
    dirs = (jnp.sin((1 - t_lin) * omega) / sin_omega)[:, :, None] * u0[:, None, :] + \
           (jnp.sin(t_lin * omega) / sin_omega)[:, :, None] * u1[:, None, :]
    radii = radius * jnp.ones((batch_size, seq_len, 1))
    return dirs * radii

def plot_sampling_methods(batch_size=1, seq_len=100, radius=1.0, dim=2, seed=42):
    key = jax.random.PRNGKey(seed)
    methods = [
        ("Dirichlet Segments", sample_dirichlet_segments),
        ("Brownian Bridge", sample_brownian_bridge),
        ("Random Walk", sample_random_walk),
        ("Slerp Arc", sample_slerp_arc),
    ]
    for name, sampler in methods:
        key, subkey = jax.random.split(key)
        x_seq = sampler(subkey, batch_size, seq_len, radius, dim)
        traj = x_seq[0]  # first sequence in batch, shape (L, D)
        plt.figure()
        plt.plot(traj[:, 0], traj[:, 1])
        plt.title(name)
        plt.gca().set_aspect('equal', 'box')
    plt.show()

# Run a comparison for dim=2:
plot_sampling_methods(batch_size=1, seq_len=100, radius=1.0, dim=2, seed=0)
