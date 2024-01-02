import jax
import optax
import numpy as np
from jax import random
from scipy import integrate
from jax import numpy as jnp
from flax import linen as nn
from functools import partial
from matplotlib import pyplot as plt
from PLRNN import basicPLRNNCell


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
    def __call__(self, s):
        plrnn = basicPLRNNCell(self.latent_size)
        latentModel = nn.RNN(plrnn, return_carry=False, name="LatentModel")
        obsModel = nn.Dense(self.num_neurons, use_bias=False, name="ObsModel")
        z = latentModel(s)
        x = obsModel(z)
        print(z.shape, x.shape, s.shape)
        return x


# Loss function that computes MSE
@jax.jit
def compute_loss_(params, s, obs):
    x = model.apply(params, s)
    error = obs - x
    e2 = error * error
    return jnp.mean(e2)


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

    # Initialize model parameters
    key, skey = random.split(random.PRNGKey(1234), 2)
    params = model.init(skey, s)
    x = model.apply(params, s)

    # Loss function and optimizer
    compute_loss = partial(compute_loss_, s=jnp.array(s), obs=jnp.array(obs))
    loss_grad = jax.grad(compute_loss)
    optimizer = optax.chain(optax.clip(0.2), optax.adamw(learning_rate=1e-3,))
    opt_state = optimizer.init(params)

    print("Initial Parameters:", params)

    # # Before training, the initial loss and gradients
    # initial_loss = compute_loss(params)
    # initial_grads = loss_grad(params)
    # initial_grads = jax.tree_map(
    #     lambda x: jnp.nan_to_num(x, nan=0.0), initial_grads)
    # print("Initial Loss:", initial_loss)
    # print("Initial Gradients:", initial_grads)
    # print(params)

    # Training loop
    for epoch in range(10):
        grads = loss_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = reset_W_diagonal(params)
        # Check in which iterarion we are getting NaN values?
        # print("Epoch ", epoch, params)

    predicted = model.apply(params, s)
    # print(params)

    # Further training
    for epoch in range(100):
        grads = loss_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = reset_W_diagonal(params)

    xe = model.apply(params, s)

    # plot
    xe = np.array(xe)[0]
    obs = np.array(obs)[0]

    ax1 = plt.subplot(311)
    ax1.plot(obs[:, 0], color='C0')
    ax1.plot(xe[:, 0], '--', color='C1')

    ax2 = plt.subplot(312)
    ax2.plot(obs[:, 1], color='C0')
    ax2.plot(xe[:, 1], '--', color='C1')

    ax3 = plt.subplot(313)
    ax3.plot(obs[:, 2], color='C0')
    ax3.plot(xe[:, 2], '--', color='C1')

    plt.savefig('RNNtrajectory.pdf')
    plt.close()
