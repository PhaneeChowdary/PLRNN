import jax
from typing import Any
from jax import random
import jax.numpy as jnp
from flax.linen.linear import Dense
from flax.linen import RNNCellBase, compact


class basicPLRNNCell(RNNCellBase):

    """
    Implementation of Piecewise Linear Recurrent Neural Network (PLRNN) using JAX

    Latent Model (z_t = A*z_{t-1} + W*phi(z_{t-1})):

    z_t: The latent state at time t.
    A: A diagonal matrix that influences the latent state from the previous time step (z_{t-1}).
    W: An off-diagonal weight matrix applied to the nonlinear function phi, which is a ReLU (max(0, z)).
    phi: A non-linear function (ReLU) applied to the latent state.

    Observation Model (x_t = B*z_t):

    x_t: The observed output at time t.
    B: A matrix that maps the latent state to the observed space.
    """

    D: int  # Number of variables in latent space
    N: int  # Number of observed neuronal firing rates
    dtype: Any = jnp.float32
    rng = jax.random.PRNGKey(1234)
    R = random.normal(rng, (15, 15))

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

    # Initialize A, W, and B
    def setup(self):
        # Diagonal matrix A
        self.A = Dense(1, use_bias=False, dtype=self.dtype,
                       kernel_init=self.init_A)

        # W - Off-diagonal matrix + h (bias)
        self.W = Dense(self.D, use_bias=True, dtype=self.dtype,
                       kernel_init=self.init_W)

        # B Matrix
        self.B = Dense(self.N, use_bias=False, dtype=self.dtype)

    # Forward pass
    @compact
    def __call__(self, carry, inputs):
        z_prev = carry
        s_t = inputs

        # A * z_{t-1}
        A_z = self.A(z_prev)

        # W * phi(z_{t-1})
        W_z = self.W(jnp.maximum(0, z_prev))

        # Latent Model
        z_t = A_z + W_z + s_t*z_prev

        # Observation Model
        x_t = self.B(z_t)

        return z_t, x_t

    def initialize_carry(self, rng):
        z_shape = (self.D,)
        z_init = random.normal(rng, z_shape, dtype=self.dtype)
        return z_init
