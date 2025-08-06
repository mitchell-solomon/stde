"""
PINNMamba: Physics-Informed Neural Networks with Mamba architecture for the Sine-Gordon equation.

This implementation follows the sequence-to-sequence, encoder-only model approach from the PINNMamba paper.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import math
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from mamba import BidirectionalMamba, MambaConfig, DiagnosticsConfig, sample_domain_fn


class PINNMambaEncoder(nn.Module):
    """
    Encoder module for PINNMamba architecture.

    This module processes input sequences using a Bidirectional Mamba model.
    """
    hidden_dim: int
    num_layers: int
    hidden_d_ff: int = 256
    heads: int = 2
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, train: bool = False):
        """
        Apply the encoder to the input sequence.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            train: Whether the model is in training mode

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        # Create Mamba configuration
        model_config = MambaConfig(
            hidden_features=self.hidden_dim,
            expansion_factor=2.0,
            dt_rank='auto',
            activation='gelu',
            norm_type='layer',
            mlp_layer=True,
            dense_expansion=self.hidden_d_ff // self.hidden_dim,
            complement=True,
            tie_in_proj=True,
            tie_gate=True,
            concatenate_fwd_rev=True,
            diagnostics=DiagnosticsConfig()
        )

        # Apply multiple layers of Bidirectional Mamba
        for i in range(self.num_layers):
            # Apply layer normalization before each layer
            x = nn.LayerNorm()(x)

            # Apply Bidirectional Mamba layer
            # We need to handle the sequence dimension differently
            # The original Mamba expects (B, 1, D) but we have (B, seq_len, D)
            B, seq_len, D = x.shape

            # Process each sequence element separately
            mamba_layer = BidirectionalMamba(**vars(model_config))

            # Process each element in the sequence separately
            outputs = []
            for j in range(seq_len):
                # Extract the j-th element from each sequence
                x_j = x[:, j, :]  # (B, D)
                # Reshape for Mamba
                x_j_reshaped = x_j.reshape(B, 1, D)
                # Apply Mamba
                out_j = mamba_layer(x_j_reshaped, train=train)
                # Store output
                outputs.append(out_j)

            # Stack outputs along sequence dimension
            mamba_out = jnp.concatenate(outputs, axis=1)  # (B, seq_len, D or 1)

            # Ensure mamba_out has the same shape as x for the residual connection
            if mamba_out.shape[-1] != x.shape[-1]:
                # Project mamba_out to match x's dimension
                mamba_out = nn.Dense(features=D)(mamba_out)

            # Apply dropout during training
            if train:
                mamba_out = nn.Dropout(rate=self.dropout_rate)(mamba_out, deterministic=not train)

            # Residual connection
            x = x + mamba_out

        # Final layer normalization
        x = nn.LayerNorm()(x)

        return x


class PINNMamba(nn.Module):
    """
    PINNMamba model for solving PDEs using sequence-to-sequence approach.

    This model follows the encoder-only architecture from the PINNMamba paper.
    """
    in_dim: int
    out_dim: int
    hidden_dim: int
    num_layer: int
    hidden_d_ff: int = 256
    heads: int = 2
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, t, train: bool = False):
        """
        Forward pass of the PINNMamba model.

        Args:
            x: Spatial coordinates tensor of shape (batch_size, seq_len, x_dim)
            t: Time coordinates tensor of shape (batch_size, seq_len, 1)
            train: Whether the model is in training mode

        Returns:
            Output tensor of shape (batch_size, seq_len, out_dim)
        """
        # Concatenate spatial and temporal coordinates
        inputs = jnp.concatenate([x, t], axis=-1)  # (batch_size, seq_len, in_dim)

        # Input embedding
        x = nn.Dense(features=self.hidden_dim, name="linear_emb")(inputs)

        # Apply encoder
        encoder = PINNMambaEncoder(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layer,
            hidden_d_ff=self.hidden_d_ff,
            heads=self.heads,
            dropout_rate=self.dropout_rate
        )
        x = encoder(x, train=train)

        # Output projection with multiple layers
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=self.hidden_dim // 2)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=self.out_dim, name="linear_out")(x)

        return x


def create_sequence_data(x_points, t_points, batch_size=32, seq_len=128):
    """
    Create sequence data for training PINNMamba.

    Args:
        x_points: Spatial points of shape (n_x, x_dim)
        t_points: Temporal points of shape (n_t, 1)
        batch_size: Number of sequences in a batch
        seq_len: Length of each sequence

    Returns:
        Batched sequences of shape (batch_size, seq_len, x_dim + 1)
    """
    n_x = x_points.shape[0]
    n_t = t_points.shape[0]
    x_dim = x_points.shape[1]

    # Create random indices for sampling
    key = jax.random.PRNGKey(0)
    key, subkey1, subkey2 = jax.random.split(key, 3)

    # Initialize batch arrays
    x_batch = jnp.zeros((batch_size, seq_len, x_dim))
    t_batch = jnp.zeros((batch_size, seq_len, 1))

    for i in range(batch_size):
        # For each batch, create a sequence by sampling random points
        x_indices = jax.random.randint(subkey1, (seq_len,), 0, n_x)
        t_indices = jax.random.randint(subkey2, (seq_len,), 0, n_t)

        # Sample points
        x_seq = x_points[x_indices]
        t_seq = t_points[t_indices]

        # Store in batch
        x_batch = x_batch.at[i].set(x_seq)
        t_batch = t_batch.at[i].set(t_seq)

        # Update keys for next batch
        key, subkey1, subkey2 = jax.random.split(key, 3)

    return x_batch, t_batch


def create_sine_gordon_data(n_x=100, n_t=50, x_radius=1.0, t_max=1.0, batch_size=32, seq_len=128):
    """
    Create data for the Sine-Gordon equation.

    Args:
        n_x: Number of spatial points
        n_t: Number of temporal points
        x_radius: Radius of the spatial domain
        t_max: Maximum time
        batch_size: Number of sequences in a batch
        seq_len: Length of each sequence

    Returns:
        Batched sequences for training
    """
    # Create spatial points in a ball
    key = jax.random.PRNGKey(0)
    x_points, _, key = sample_domain_fn(n_x, 2, x_radius, key)
    x_points = x_points.reshape(n_x, 2)  # Remove sequence dimension

    # Create temporal points
    t_points = jnp.linspace(0, t_max, n_t).reshape(n_t, 1)

    # Create sequence data
    return create_sequence_data(x_points, t_points, batch_size, seq_len)


@partial(jax.jit, static_argnames=('u_fn',))
def sine_gordon_pde(x, t, u_fn: Callable, rng):
    """
    Compute the Sine-Gordon PDE residual.

    Args:
        x: Spatial coordinates tensor of shape (batch_size, seq_len, x_dim)
        t: Time coordinates tensor of shape (batch_size, seq_len, 1)
        u_fn: Function that computes u(x, t)
        rng: Random key for stochastic operations

    Returns:
        PDE residual
    """
    # Concatenate inputs
    inputs = jnp.concatenate([x, t], axis=-1)

    # Define a function for computing derivatives
    def u(xt):
        return u_fn(xt[..., :-1], xt[..., -1:])

    # Compute u and its derivatives
    u_val = u(inputs)

    # Compute derivatives using JAX's automatic differentiation
    u_t = jax.grad(lambda xt: u(xt).sum(), 1)(inputs)
    u_tt = jax.grad(lambda xt: jax.grad(lambda xt: u(xt).sum(), 1)(xt).sum(), 1)(inputs)

    u_x = jax.grad(lambda xt: u(xt).sum(), 0)(inputs)
    u_xx = jax.grad(lambda xt: jax.grad(lambda xt: u(xt).sum(), 0)(xt).sum(), 0)(inputs)

    # Compute the Sine-Gordon PDE residual: u_tt - u_xx + sin(u) = 0
    residual = u_tt - u_xx + jnp.sin(u_val)

    return residual


def test_pinnmamba_model():
    """Test the PINNMamba model."""
    print("\n=== Testing PINNMamba Model ===")

    # Test parameters
    batch_size = 4
    seq_len = 10
    in_dim = 2
    out_dim = 1
    hidden_dim = 32
    num_layers = 2

    print(f"Test parameters: batch_size={batch_size}, seq_len={seq_len}, in_dim={in_dim}, out_dim={out_dim}")
    print(f"Model config: hidden_dim={hidden_dim}, num_layers={num_layers}")

    # Initialize model
    model = PINNMamba(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        num_layer=num_layers
    )
    print("Model initialized successfully")

    # Create dummy inputs
    key = jax.random.PRNGKey(0)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    x = jax.random.normal(subkey1, (batch_size, seq_len, in_dim-1))
    t = jax.random.normal(subkey2, (batch_size, seq_len, 1))
    print(f"Input shapes: x={x.shape}, t={t.shape}")

    # Initialize parameters
    variables = model.init(key, x, t)
    print("Parameters initialized successfully")

    # Run forward pass
    output = model.apply(variables, x, t)
    print(f"Forward pass completed successfully")
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[0, 0]}")

    # Test output shape
    expected_shape = (batch_size, seq_len, out_dim)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"
    print(f"Output shape test passed: {output.shape} matches expected {expected_shape}")

    # Test output type
    assert isinstance(output, jnp.ndarray), f"Expected output to be jnp.ndarray, but got {type(output)}"
    print(f"Output type test passed: {type(output).__name__}")

    # Test that output contains no NaN values
    has_nans = jnp.any(jnp.isnan(output))
    assert not has_nans, "Output contains NaN values"
    print(f"NaN check passed: Output contains no NaN values")

    print("PINNMamba model test passed!")


def test_sequence_data_creation():
    """Test the sequence data creation functions."""
    print("\n=== Testing Sequence Data Creation ===")

    # Test parameters
    n_x = 100
    n_t = 50
    x_radius = 1.0
    t_max = 1.0
    batch_size = 8
    seq_len = 16

    print(f"Test parameters: n_x={n_x}, n_t={n_t}, batch_size={batch_size}, seq_len={seq_len}")

    # Create data
    x_batch, t_batch = create_sine_gordon_data(
        n_x=n_x,
        n_t=n_t,
        x_radius=x_radius,
        t_max=t_max,
        batch_size=batch_size,
        seq_len=seq_len
    )
    print("Data created successfully")

    # Check shapes
    assert x_batch.shape == (batch_size, seq_len, 2), f"Expected x_batch shape (batch_size, seq_len, 2), got {x_batch.shape}"
    assert t_batch.shape == (batch_size, seq_len, 1), f"Expected t_batch shape (batch_size, seq_len, 1), got {t_batch.shape}"
    print(f"Shape test passed: x_batch={x_batch.shape}, t_batch={t_batch.shape}")

    # Check values
    assert jnp.all(jnp.linalg.norm(x_batch, axis=-1) <= x_radius), "Some x values are outside the domain radius"
    assert jnp.all((t_batch >= 0) & (t_batch <= t_max)), "Some t values are outside the time range"
    print(f"Value range test passed")

    # Print sample data
    print(f"Sample x values: {x_batch[0, 0:2]}")
    print(f"Sample t values: {t_batch[0, 0:2]}")

    print("Sequence data creation test passed!")


def tabulate_pinnmamba_model():
    """Tabulate the PINNMamba model architecture."""
    print("\n=== Tabulating PINNMamba Model Architecture ===")

    # Test parameters
    batch_size = 4
    seq_len = 10
    in_dim = 3
    out_dim = 1
    hidden_dim = 32
    num_layers = 2

    print(f"Model parameters: in_dim={in_dim}, out_dim={out_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}")

    # Create dummy inputs
    key = jax.random.PRNGKey(0)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    x = jax.random.normal(subkey1, (batch_size, seq_len, in_dim-1))
    t = jax.random.normal(subkey2, (batch_size, seq_len, 1))
    print(f"Input shapes: x={x.shape}, t={t.shape}")

    # Initialize model
    model = PINNMamba(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        num_layer=num_layers
    )
    print("Model initialized successfully")

    # Print the tabulated model summary
    print("\nDetailed Model Architecture:")
    print(nn.tabulate(
        model,
        rngs={"params": jax.random.PRNGKey(0)},
        mutable=['params', 'diagnostics', 'intermediates'],
    )(x, t, train=True))

    print("\nModel tabulation completed successfully!")


if __name__ == "__main__":
    print("\n===================================================")
    print("RUNNING PINNMAMBA MODEL TESTS")
    print("===================================================")

    test_pinnmamba_model()
    test_sequence_data_creation()
    tabulate_pinnmamba_model()

    print("\n===================================================")
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("===================================================")
