import jax
import optax
import numpy as np
from typing import Any
from jax import random
import jax.numpy as jnp
from scipy import integrate
from flax.linen.linear import Dense
from matplotlib import pyplot as plt
from flax.linen import RNNCellBase, compact


# Rossler dynamical system, used for generating synthetic data.
def rossler(u, t, p):
    x, y, z = u
    a, b, c = p
    x_ = -y-z
    y_ = x+a*y
    z_ = b+z*(x-c)
    return [x_, y_, z_]


good_u0 = np.array([-4.53741665, 2.17898447, 0.02006643])


# Generates data based on the Rossler system.
def get_data(tt, a=0.2, b=0.2, c=5.7, u0=good_u0):
    p = np.array([a, b, c])
    sol = integrate.odeint(rossler, u0, tt, args=(p,))
    return sol


class basicPLRNNCell(RNNCellBase):

    """
    Implementation of Piecewise Linear Recurrent Neural Network (PLRNN) using JAX

    Latent Model (z_t = A*z_{t-1} + W*phi(z_{t-1}) + C*s_t):

    z_t: The latent state at time t.
    A: A diagonal matrix that influences the latent state from the previous time step (z_{t-1}).
    W: An off-diagonal weight matrix applied to the nonlinear function phi, which is a ReLU (max(0, z)).
    C: A matrix that incorporates the external input s_t into the latent state.
    phi: A non-linear function (ReLU) applied to the latent state.

    Observation Model (x_t = B*z_t):

    x_t: The observed output at time t.
    B: A matrix that maps the latent state to the observed space.
    """

    D: int  # Number of variables in latent space
    N: int  # Number of observed neuronal firing rates
    dtype: Any = jnp.float32

    # Create matrix A (Diagonal Matrix)
    def init_A(self, key, shape, dtype):
        D = shape[0]
        scale = 10
        R = random.normal(key, (D, D))
        q = (R.T@R/D + scale*jnp.eye(D))
        w, v = jnp.linalg.eigh(q)
        H = q / jnp.max(w.real)
        A = jnp.diag(H)
        return A

    # Create matrix W (Off-Diagonal Matrix)
    def init_W(self, key, shape, dtype):
        D = shape[0]
        scale = 10
        R = random.normal(key, (D, D))
        q = (R.T@R/D + scale*jnp.eye(D))
        w, v = jnp.linalg.eigh(q)
        H = q / jnp.max(w.real)
        A = jnp.diag(H)
        W = H - A*jnp.eye(D)
        return W

    # Initialize A, W, C, and B
    def setup(self):
        # Diagonal matrix A
        self.A = Dense(1, use_bias=False, dtype=self.dtype,
                       kernel_init=self.init_A)

        # W - Off-diagonal matrix + h (bias)
        self.W = Dense(self.D, use_bias=True, dtype=self.dtype,
                       kernel_init=self.init_W)

        # C Matrix
        self.C = Dense(self.D, use_bias=False, dtype=self.dtype)

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

        # C*s_t
        C_s = self.C(s_t)

        # Latent Model z_t
        z_t = A_z + W_z + C_s

        # Observation Model x_t
        x_t = self.B(z_t)

        return z_t, x_t

    def initialize_carry(self, rng):
        z_shape = (self.D,)

        # Initializing z0 as a trainable parameter
        z_init = random.normal(rng, z_shape, dtype=self.dtype)
        return z_init


# Loss function that computes MSE
@jax.jit
def compute_loss_(params, carry, s, obs):
    _, x = n1.apply(params, carry, s)
    error = x - obs
    e2 = error*error
    return jnp.mean(e2)


# Reset the diagonal elements of W to Zero
def reset_W_diagonal(params):
    W = params['params']['W']['kernel']
    W = W.at[jnp.diag_indices(W.shape[0])].set(0)
    params['params']['W']['kernel'] = W
    return params


# Main block execution
if __name__ == '__main__':
    D = 15
    N = 3

    # Generate data
    time = 123
    time_steps = 200
    obs = get_data(np.linspace(0, time, time*10))
    obs = jnp.array(obs)[-time_steps:, :N][None, :]

    rng = jax.random.PRNGKey(1234)
    s = jnp.zeros((1, time_steps, 1))

    # Initialize the model
    n1 = basicPLRNNCell(D=D, N=N)

    # Initialize the carry (latent state), parameters
    carry = n1.initialize_carry(rng)
    params = n1.init(rng, carry, s)

    # Set up the optimizer
    optimizer = optax.chain(optax.adamw(learning_rate=1e-3))
    opt_state = optimizer.init(params)

    # Training loop
    for epoch in range(10):
        loss, grads = jax.value_and_grad(compute_loss_)(params, carry, s, obs)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Reset the diagonal elements of W to zero
        params = reset_W_diagonal(params)
        # print(f"Epoch {epoch}, Loss: {loss}")

    xi = n1.apply(params, carry, s)

    for epoch in range(1000):
        loss, grads = jax.value_and_grad(compute_loss_)(params, carry, s, obs)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Reset the diagonal elements of W to zero
        params = reset_W_diagonal(params)

    xe = n1.apply(params, carry, s)

    # print(params['params']['B']['kernel'])
    # print(params['params']['A']['kernel'])
    # print(params['params']['W']['kernel'])

    # q = jax.tree_util.tree_map(lambda x: x.shape, params)
    # print(q['params'])

    # Plot the xe and obs
    xe = np.array(xe[1][0])
    obs = np.array(obs)[0]

    plt.figure(figsize=(12, 8))
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(obs[:, i], color='C0')
        plt.plot(xe[:, i], '--', color='C1')
    plt.tight_layout()
    plt.savefig('FinalTrajectory.pdf')
    plt.close()
