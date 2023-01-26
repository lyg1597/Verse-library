import jax
import jax.numpy as jnp 
import optax 
import copy
jax.config.update('jax_platform_name', 'cpu')

# Test the ability of using neural networks for solving differential equation for the example they provided

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

def f_x(x,t):
    x_dot = t**3+2*t+t**2*(1+3*t**2)/(1+t+t**3) - (t+(1+3*t**2)/(1+t+t**3))*x
    # z_dot = x[:,1]
    # vz_dot = x[:,2]
    # az_dot = jnp.zeros(100)
    # y_dot = x[:,4]
    # vy_dot = -0.01*x[:,4]+x[:,5]
    # ay_dot = jnp.zeros(100)
    return x_dot

def init_weight():
    prng = jax.random.PRNGKey(42)
    prng, w1key, w2key= jax.random.split(prng, 3)
    params = dict(
        w1 = jnp.array([[-0.75268114, -0.02114583, -1.11579788,  0.78253833,  1.12619648, 1.61703103, -0.25205003, -0.50347778, -0.90865558,  0.32148524]]),
        w2 = jnp.array([[-0.22212057],
       [-0.22813992],
       [-0.81672296],
       [ 2.21553157],
       [ 0.95053005],
       [ 0.1161893 ],
       [-0.13836353],
       [-1.63998191],
       [-1.23521537],
       [-1.21106445]]),
    )
    return params

def forward(params, t):
    a1 = jax.nn.sigmoid(t@params["w1"])
    x = a1@params["w2"]
    return x

def neural_network(W, x):
    a1 = jax.nn.sigmoid(jnp.dot(x, W['w1']))
    return jnp.dot(a1, W['w2'])

dnn_dt = jax.jacfwd(forward, argnums=1)

ref_traj = None

def sigmoid_grad(x):
    return jax.nn.sigmoid(x) * (1 - jax.nn.sigmoid(x))

def d_neural_network_dx(W, x, k=1):
    return jnp.dot(W['w2'].T, W['w1'].T**k)*sigmoid_grad(x)

# @jax.jit 
def xloss(params, t, x0):
    x = forward(params, t)
    xt = x0 + t*x
    x_ref = forward(params, t)
    dforwarddt = jnp.diagonal(dnn_dt(params, t)[:,:,:,0],axis1 = 0, axis2 = 2).T
    d_net_out = d_neural_network_dx(params, t)
    dx_dt = x + t*dforwarddt
    fx_t = f_x(xt, t)
    f_ref = f(t[0], xt[0])
    grad_loss = jnp.sum(optax.l2_loss(dx_dt, fx_t))*5

    # x_traj = x0 + t*x 
    # traj_loss = jnp.sum(optax.l2_loss(ref_traj, x_traj))

    return grad_loss 

def nn_solution(x0, T, params, num=1000):
    x0 = jnp.reshape(jnp.array(x0), (1, -1))
    steps = jnp.linspace(0,T,num)
    res = []
    for step in steps:
        x_t = x0 + step*forward(params, jnp.array([step]))
        res.append(jnp.squeeze(x_t))
    return jnp.array(res) 

def dynamic(x, t):
    # z,vz,az,y,vy,ay = x 
    x_dot = t**3+2*t+t**2*(1+3*t**2)/(1+t+t**3)-(t+(1+3*t**2)/(1+t+t**3))*x[0]
    return [x_dot]

def integrate_solution(x0, T, num=1000):
    import numpy as np 
    from scipy.integrate import odeint 

    t = [round(T/num*i, 10) for i in range(num)]
    traj = odeint(dynamic, x0, t)
    return traj

def compute_ref_traj(T, num_pts, x0 = [1]):
    res = integrate_solution(x0, T, num_pts)
    return jnp.array(res)

if __name__ == "__main__":
    T = 1
    num_pts = 11
    x0 = [1]
    x0_extend = jnp.array([x0 for _ in range(num_pts)])

    params = init_weight()
    t = jnp.linspace(0,T,num_pts)
    t = jnp.reshape(t, (-1,1))
    # t = jnp.array([np.linspace]).T

    schedlue = optax.linear_schedule(1.0,0.0001,0.0001,10000)
    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(params)

    xloss_grad = jax.jit(jax.grad(xloss, argnums=0))

    final_res = params 
    best_loss = float('inf')

    ref_traj = compute_ref_traj(T, num_pts, x0 = x0)

    for i in range(20000):
        # x = forward(params, t)
        val = xloss(params, t, x0_extend)
        if val < best_loss:
            best_loss = val 
            final_res = copy.deepcopy(params) 
        print(i, val)
        grads = xloss_grad(params, t, x0_extend)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
    params = final_res

    
    loss_val = xloss(final_res, t, x0_extend)
    print(loss_val)

    traj_nn = nn_solution([1.0], 1, final_res)

    traj_gt = integrate_solution([1.0], 1)

    import matplotlib.pyplot as plt 
    plt.plot(jnp.linspace(0,1,1000), traj_nn)
    plt.plot(jnp.linspace(0,1,1000), traj_gt[:,0])
    plt.show()