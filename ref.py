import jax
import numpy as np

def A(x):
    '''
        Left part of initial equation
    '''
    return x + (1. + 3.*x**2) / (1. + x + x**3)


def B(x):
    '''
        Right part of initial equation
    '''
    return x**3 + 2.*x + x**2 * ((1. + 3.*x**2) / (1. + x + x**3))


def f(x, psy):
    '''
        d(psy)/dx = f(x, psy)
        This is f() function on the right
    '''
    return B(x) - psy * A(x)


def psy_analytic(x):
    '''
        Analytical solution of current problem
    '''
    return (np.exp((-x**2)/2.)) / (1. + x + x**3) + x**2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return jax.nn.sigmoid(x) * (1 - jax.nn.sigmoid(x))


def neural_network(W, x):
    a1 = jax.nn.sigmoid(np.dot(x, W[0]))
    return np.dot(a1, W[1])

def d_neural_network_dx(W, x, k=1):
    return np.dot(np.dot(W[1].T, W[0].T**k), sigmoid_grad(x))

def loss_function(W, x):
    loss_sum = 0.
    for xi in x:
        net_out = neural_network(W, xi)[0][0]
        psy_t = 1. + xi * net_out
        d_net_out = d_neural_network_dx(W, xi)[0][0]
        d_psy_t = net_out + xi * d_net_out
        func = f(xi, psy_t)       
        err_sqr = (d_psy_t - func)**2

        loss_sum += err_sqr
    return loss_sum

W = [np.array([[-0.75268114, -0.02114583, -1.11579788,  0.78253833,  1.12619648,
         1.61703103, -0.25205003, -0.50347778, -0.90865558,  0.32148524]]), np.array([[-0.22212057],
       [-0.22813992],
       [-0.81672296],
       [ 2.21553157],
       [ 0.95053005],
       [ 0.1161893 ],
       [-0.13836353],
       [-1.63998191],
       [-1.23521537],
       [-1.21106445]])]

if __name__ == "__main__":
    loss_function(W, [0])