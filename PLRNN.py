import jax
from jax import random
from jax import numpy as jnp
from functools import partial
from flax.linen import initializers
from flax.linen import RNNCellBase, compact
from flax.linen.module import compact, nowrap
from flax.linen.linear import Dense, default_kernel_init
from typing import (Any, Optional, Tuple, TypeVar)


A = TypeVar('A')
PRNGKey = jax.Array
Shape = Tuple[int, ...]
Dtype = Any
Array = jax.Array
Carry = Any
CarryHistory = Any
Output = Any


class basicPLRNNCell(RNNCellBase):
    features: int
    kernel_init: initializers.Initializer = default_kernel_init
    bias_init: initializers.Initializer = initializers.zeros_init()
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float64
    carry_init: initializers.Initializer = initializers.zeros_init()

    rng = jax.random.PRNGKey(1234)
    R = random.normal(rng, (15, 15))
    epsilon = 1e-8

    # Create matrix A (Diagonal Matrix)

    def init_A(self, key, shape, dtype):
        D = shape[0]
        scale = 10
        q = (self.R.T@self.R/D + scale*jnp.eye(D))
        w, v = jnp.linalg.eigh(q)
        H = q / jnp.max(w.real)
        A = jnp.diag(H)
        return A

    # Create matrix W (Off-Diagonal Matrix)

    def init_W(self, key, shape, dtype):
        D = shape[0]
        scale = 10
        q = (self.R.T@self.R/D + scale*jnp.eye(D))
        w, v = jnp.linalg.eigh(q)
        H = q / jnp.max(w.real)
        A = jnp.diag(H)
        W = H - A*jnp.eye(D)
        return W

    @compact
    def __call__(self, carry, inputs):
        z_prev = carry

        # Diagonal matrix A - implemented as a dense layer with diagonal weights
        A = partial(
            Dense,
            features=self.features,
            use_bias=False,
            kernel_init=self.init_A,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        # Off-diagonal weight matrix W with bias
        W = partial(
            Dense,
            features=self.features,
            use_bias=True,
            kernel_init=self.init_W,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        # Activation function (ReLU)
        def phi(x): return jnp.maximum(0, x)

        # Latent model equation
        z_t = A(name='A')(z_prev) + W(name='W')(phi(z_prev))

        return z_t, z_t

    @nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]) -> Array:
        batch_dims = input_shape[:-1]
        key = random.split(rng)[0]
        mem_shape = batch_dims + (self.features,)
        z = self.carry_init(key, mem_shape, self.param_dtype)
        return z

    @property
    def num_feature_axes(self) -> int:
        return 1
