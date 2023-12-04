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


def rossler(u, t, p):
    x, y, z = u
    a, b, c = p
    x_ = -y-z
    y_ = x+a*y
    z_ = b+z*(x-c)
    return [x_, y_, z_]


good_u0 = np.array([-4.53741665, 2.17898447, 0.02006643])


def get_data(tt, a=0.2, b=0.2, c=5.7, u0=good_u0):
    p = np.array([a, b, c])
    sol = integrate.odeint(rossler, u0, tt, args=(p,))
    return sol


class basicPLRNNCell(RNNCellBase):
    D: int  # Number of variables in latent space
    N: int  # Number of observed neuronal firing rates
    dtype: Any = jnp.float32

    def diag_init(self, key, shape, dtype=jnp.float32):
        # Initialize a diagonal matrix
        assert shape[0] == shape[1]
        diag = jax.random.normal(key, (shape[0],))
        return jnp.diag(diag)

    def off_diag_init(self, key, shape, dtype=jnp.float32):
        # Initialize an off-diagonal matrix
        assert shape[0] == shape[1]
        mat = jax.random.normal(key, shape)
        return mat * (1 - jnp.eye(shape[0]))

    def setup(self):
        # Diagonal matrix A
        self.A = Dense(1, use_bias=False, dtype=self.dtype)

        # W - Off-diagonal matrix + h (bias)
        self.W = Dense(self.D, use_bias=True, dtype=self.dtype,
                       kernel_init=self.off_diag_init)

        # C Matrix
        self.C = Dense(self.D, use_bias=False, dtype=self.dtype)

        # B Matrix
        self.B = Dense(self.N, use_bias=False, dtype=self.dtype)

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


@jax.jit
def compute_loss_(params, carry, s, obs):
    _, x = n1.apply(params, carry, s)
    error = x - obs
    e2 = error*error
    return jnp.mean(e2)


def reset_W_diagonal(params):
    W_matrix = params['params']['W']['kernel']

    # Set the diagonal elements to zero
    W_matrix = W_matrix.at[jnp.diag_indices(W_matrix.shape[0])].set(0)

    # Update the parameters dictionary
    params['params']['W']['kernel'] = W_matrix
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

    # q=jax.tree_util.tree_map(lambda x: x.shape, params)
    # print(q['params'])

    # Inspect the shape and type of xe
    print("Type of xe:", type(xe))
    if isinstance(xe, tuple):
        print("Length of tuple:", len(xe))
        for i, item in enumerate(xe):
            print(f"Shape of item {i} in tuple:", item.shape)
    elif isinstance(xe, jnp.ndarray):
        print("Shape of xe as ndarray:", xe.shape)
    else:
        print("xe is neither a tuple nor an ndarray")

    # Plot the xe and obs
    xe = np.array(xe[1][0])
    obs = np.array(obs)[0]

    ax1 = plt.subplot(311)
    ax1.plot(obs[:, 0], color='C0')
    ax1.plot(xe[:, 0], color='C1')

    ax2 = plt.subplot(312)
    ax2.plot(obs[:, 1], color='C0')
    ax2.plot(xe[:, 1], '--', color='C1')

    ax3 = plt.subplot(313)
    ax3.plot(obs[:, 2], color='C0')
    ax3.plot(xe[:, 2], '--', color='C1')

    plt.savefig('FinalTrajectory.pdf')
    plt.close()