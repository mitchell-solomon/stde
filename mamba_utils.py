"""
Utility functions for working with Mamba models in a PINN framework.

This module provides a compatibility layer between the original Mamba implementation
and the STDE-compatible implementation.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Union, Callable
import flax.linen as nn

from mamba import MambaConfig, DiagnosticsConfig, BidirectionalMamba
from stde_mamba import BidirectionalSTDEMamba


def create_mamba_model(
    config: MambaConfig,
    use_stde_compatible: bool = False,
    **kwargs
) -> Union[BidirectionalMamba, BidirectionalSTDEMamba]:
    """
    Create a Mamba model with the given configuration.

    Args:
        config: The Mamba configuration.
        use_stde_compatible: Whether to use the STDE-compatible implementation.
        **kwargs: Additional arguments to pass to the model constructor.

    Returns:
        A Mamba model.
    """
    if use_stde_compatible:
        return BidirectionalSTDEMamba(
            hidden_features=config.hidden_features,
            expansion_factor=config.expansion_factor,
            dt_rank=config.dt_rank,
            activation=config.activation,
            norm_type=config.norm_type,
            mlp_layer=config.mlp_layer,
            dense_expansion=config.dense_expansion,
            complement=config.complement,
            tie_in_proj=config.tie_in_proj,
            tie_gate=config.tie_gate,
            concatenate_fwd_rev=config.concatenate_fwd_rev,
            diagnostics=config.diagnostics,
            **kwargs
        )
    else:
        return BidirectionalMamba(**vars(config), **kwargs)


def with_taylor_ad(
    fn: Callable,
    primals,
    series,
    use_stde_compatible: bool = True
):
    """
    Apply Taylor automatic differentiation to a function.

    This function provides a compatibility layer for using Taylor automatic
    differentiation with functions that may or may not be compatible with
    jax.experimental.jet.

    Args:
        fn: The function to differentiate.
        primals: The primal values.
        series: The series values.
        use_stde_compatible: Whether to use the STDE-compatible implementation.

    Returns:
        The result of applying Taylor automatic differentiation to the function.
    """
    if use_stde_compatible:
        from jax.experimental import jet
        return jet.jet(fn, primals, series)
    else:
        # Fallback to standard JAX autodiff
        # This won't work with lax.scan, but we provide it for completeness
        primal_out = fn(*primals)

        # Compute first-order derivatives using jvp
        tangent_out = jax.jvp(fn, primals, series[0])[1]

        # For higher-order derivatives, you would need to use higher-order jvp
        # or other techniques, but this is just a fallback
        return primal_out, (tangent_out,)


def create_mamba_pinn_model(
    config: MambaConfig,
    use_stde_compatible: bool = False,
    **kwargs
):
    """
    Create a PINN model using Mamba.

    Args:
        config: The Mamba configuration.
        use_stde_compatible: Whether to use the STDE-compatible implementation.
        **kwargs: Additional arguments to pass to the model constructor.

    Returns:
        A PINN model using Mamba.
    """
    mamba_model = create_mamba_model(config, use_stde_compatible, **kwargs)

    # Define a PINN model that uses the Mamba model
    class MambaPINN(nn.Module):
        mamba: Union[BidirectionalMamba, BidirectionalSTDEMamba]

        def setup(self):
            self.mamba = mamba_model

        def __call__(self, x, train=False):
            # Ensure input shape is (B, L, D)
            B = x.shape[0]
            D = x.shape[-1]
            x = x.reshape(B, 1, D)

            # Apply the Mamba model
            y = self.mamba(x, train)

            # Reshape output to (B,) or (B, 1)
            return y.reshape(B, -1)

    return MambaPINN()
