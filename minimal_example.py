import jax
import jax.numpy as jnp
from jax.experimental import jet

# General parameters
dim = 50000               # Arbitrary dimension (change as needed)
rand_batch_size = 1  # Number of random probes
sparse = False          # Use dense sampling (Rademacher distribution)

###########################################################
# Define the scalar function u: R^dim -> R.
# We choose u(x) = sum(x^2), whose Hessian is 2*I.
# Hence, the exact Laplacian is 2*dim.
###########################################################
def u(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(x**2)

###########################################################
# STDE using the jet API.
#
# Given a function fn, hess_trace returns a function that,
# for an input x_i and RNG key, does the following:
#
# 1. Samples rand_batch_size random vectors v in R^dim.
#    When sparse is False, each component of v is independently
#    sampled as -1 or 1 (Rademacher distribution).
#
# 2. For each v, computes the Taylor expansion of fn at x_i:
#       fn(x_i + t*v) = fn(x_i) + t <grad fn(x_i), v> + ½ t^2 (vᵀ H fn(x_i) v) + O(t³)
#
# 3. Extracts the second-order coefficient (vᵀ H fn(x_i) v)
#    and averages over all v to estimate trace(H fn(x_i)).
###########################################################
def hess_trace(fn) -> callable:
    def fn_trace(x_i, rng):
        if sparse:
            rng, subkey = jax.random.split(rng)
            idx_set = jax.random.choice(subkey, dim, shape=(rand_batch_size,), replace=False)
            # For sparse sampling, use one-hot vectors.
            rand_vec = jax.vmap(lambda i: jnp.eye(dim)[i])(idx_set)
        else:
            rng, subkey = jax.random.split(rng)
            # Sample v from a Rademacher distribution:
            rand_vec = 2 * (jax.random.randint(subkey, shape=(rand_batch_size, dim), minval=0, maxval=2) - 0.5)
        
        # For each random vector v, compute the Taylor series of fn at x_i.
        # The series argument tells jet to consider v as the directional perturbation.
        taylor_2 = lambda v: jet.jet(fun=fn, primals=(x_i,), series=((v, jnp.zeros(dim)),))
        # f_vals are the function values (all equal to fn(x_i)) and hvps are the second-order terms.
        f_vals, (_, hvps) = jax.vmap(taylor_2)(rand_vec)
        trace_est = jnp.mean(hvps)
        if sparse:
            trace_est *= dim
        return f_vals[0], trace_est
    return fn_trace

###########################################################
# STDE Laplacian Estimator
#
# Given a function u_fn, this function returns an estimate of
# the Laplacian (i.e. the trace of the Hessian) at point x.
###########################################################
def stde_laplacian(u_fn, x: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
    fn_trace = hess_trace(u_fn)
    _, trace_est = fn_trace(x, key)
    return trace_est

###########################################################
# Main: Generalized to arbitrary dimension.
#
# We generate a test point x as a vector of ones with length 'dim'
# and compare the STDE Laplacian estimate to the exact value (2*dim).
###########################################################
def main():
    # Create a test point x in R^dim.
    x = jnp.ones((dim,))
    key = jax.random.PRNGKey(0)
    
    approx_lap = stde_laplacian(u, x, key)
    exact_lap = 2.0 * dim
    
    print(f"Dimension: {dim}")
    print(f"STDE Approx Laplacian: {approx_lap:.4f}")
    print(f"Exact Laplacian:       {exact_lap:.4f}")

if __name__ == "__main__":
    main()
