import jax
import optax
import numpy as np
from jax import random
from scipy import integrate
from jax import numpy as jnp
from flax import linen as nn
from functools import partial
from matplotlib import pyplot as plt
from basicPLRNNCell import basicPLRNNCell


# Rossler dynamical system, used for generating synthetic data.
def rossler(u, t, p):
    x, y, z = u
    a, b, c = p
    x_ = -y - z
    y_ = x + a * y
    z_ = b + z * (x - c)
    return [x_, y_, z_]


# Good initial condition for the Rossler attractor
good_u0 = np.array([-4.53741665, 2.17898447, 0.02006643])


# Generates data based on the Rossler system.
def get_data(tt, a=0.2, b=0.2, c=5.7, u0=good_u0):
    p = np.array([a, b, c])
    sol = integrate.odeint(rossler, u0, tt, args=(p,))
    return sol


# Neural network definition
class Net(nn.Module):
    latent_size: int
    num_neurons: int

    @nn.compact
    def __call__(self, carry, s):
        plrnn = basicPLRNNCell(self.latent_size, self.num_neurons)
        carry, outputs = plrnn(carry, s)
        return carry, outputs


# Loss function that computes MSE
@jax.jit
def compute_loss_(params, carry, s, obs):
    _, x = model.apply(params, carry, s)
    error = obs - x
    e2 = error * error
    return jnp.mean(e2)


# Reset the diagonal elements of W to Zero
def reset_W_diagonal(params):
    W = params['params']['basicPLRNNCell_0']['W']['kernel']
    W = W.at[jnp.diag_indices(W.shape[0])].set(0)
    params['params']['basicPLRNNCell_0']['W']['kernel'] = W
    return params


# Main execution block
if __name__ == '__main__':

    # Generate data
    Time = 123
    T = 200
    my_dims = 3
    tt = np.linspace(0, Time, Time * 10)
    obs = get_data(tt)[-T:, :my_dims][None, :]

    # Input for the neural network
    dz = 15
    input_dims = 1
    batch_size = 1
    s = jnp.zeros((batch_size, T, input_dims))

    # Neural network setup
    model = Net(dz, my_dims)

    # Initialize carry
    plrnn_cell = basicPLRNNCell(dz, my_dims)
    carry = plrnn_cell.initialize_carry(random.PRNGKey(1234))

    # Initialize model parameters
    key, skey = random.split(random.PRNGKey(1234), 2)
    params = model.init(skey, carry, s)
    x = model.apply(params, carry, s)

    # Loss function and optimizer
    compute_loss = partial(compute_loss_, obs=jnp.array(obs))
    loss_grad = jax.grad(compute_loss)
    optimizer = optax.adamw(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    # Training loop
    for epoch in range(10):
        # print("Epoch: ", epoch)
        grads = loss_grad(params, carry, s)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        # Reset the diagonal elements of W to zero
        params = reset_W_diagonal(params)

    carry, predicted = model.apply(params, carry, s)
    carry = carry[0][0]

    # Further training
    for epoch in range(1000):
        # print("Epoch: ", epoch)
        grads = loss_grad(params, carry, s)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        # Reset the diagonal elements of W to zero
        params = reset_W_diagonal(params)

    carry, xe = model.apply(params, carry, s)

    print("A Matrix: ", params['params']['basicPLRNNCell_0']['A']['kernel'])
    print("\nB Matrix: ", params['params']['basicPLRNNCell_0']['B']['kernel'])
    print("\nW Matrix: ", params['params']['basicPLRNNCell_0']['W']['kernel'])

    # Plot the results
    xe = np.array(xe)[0]
    obs = np.array(obs)[0]

    print("\nShape of xe: ", xe.shape)
    print("Shape of obs: ", obs.shape)

    ax1 = plt.subplot(311)
    ax1.plot(obs[:, 0], color='C0')
    ax1.plot(xe[:, 0], '--', color='C1')

    ax2 = plt.subplot(312)
    ax2.plot(obs[:, 1], color='C0')
    ax2.plot(xe[:, 1], '--', color='C1')

    ax3 = plt.subplot(313)
    ax3.plot(obs[:, 2], color='C0')
    ax3.plot(xe[:, 2], '--', color='C1')

    plt.savefig('RNNTrajectory.pdf')
    plt.close()
