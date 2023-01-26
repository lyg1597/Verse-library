import jax
import jax.numpy as jnp 
import optax 
import copy
jax.config.update('jax_platform_name', 'cpu')

# Test the ability of using neural networks for solving differential equation

@jax.jit
def f_x(x):
    z_dot = x[:,1]
    vz_dot = x[:,2]
    az_dot = jnp.zeros(x[:,1].shape)
    y_dot = x[:,4]
    vy_dot = -0.01*x[:,4]+x[:,5]
    ay_dot = jnp.zeros(x[:,1].shape)
    return jnp.array([z_dot, vz_dot, az_dot, y_dot, vy_dot, ay_dot])

def init_weight():
    prng = jax.random.PRNGKey(42)
    prng, w1key, b1key, w2key, b2key= jax.random.split(prng, 5)
    params = dict(
        w1 = 1e-2*jax.random.normal(w1key, (1, 128)),
        b1 = 1e-2*jax.random.normal(b1key, (1, 128)),
        w2 = 1e-2*jax.random.normal(w2key, (128, 6)),
        b2 = 1e-2*jax.random.normal(b2key, (1, 6)),
    )
    return params

@jax.jit
def forward(params, t):
    x = jax.nn.sigmoid(t@params["w1"]+params['b1'])
    x = x@params["w2"] + params["b2"]
    return x

dnn_dt = jax.jit(jax.jacfwd(forward, argnums=1))

ref_traj = None

# @jax.jit 
def xloss(params, t, x0):
    x = forward(params, t)
    xt = x0 + t*x
    dforwarddt = jnp.diagonal(dnn_dt(params, t)[:,:,:,0],axis1 = 0, axis2 = 2).T
    dx_dt = x + t*dforwarddt
    fx_t = f_x(xt).T
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
    z,vz,az,y,vy,ay = x 
    z_dot = vz
    vz_dot = az
    az_dot = jnp.zeros(vz.shape)
    y_dot = vy
    vy_dot = -0.01*vy+ay
    ay_dot = jnp.zeros(vz.shape)
    return [z_dot, vz_dot, az_dot, y_dot, vy_dot, ay_dot]

def integrate_solution(x0, T, num=1000):
    import numpy as np 
    from scipy.integrate import odeint 

    t = [round(T/num*i, 10) for i in range(num)]
    traj = odeint(dynamic, x0, t)
    return traj

def compute_ref_traj(T, num_pts, x0 = [0,0,0,0,0,0,0]):
    traj = integrate_solution(x0, T, num_pts)
    return jnp.array(traj)

if __name__ == "__main__":
    T = 7
    num_pts = 200
    x0 = [800, 0, -0.1, 0, 82, 0]
    x0_extend = jnp.array([x0 for _ in range(num_pts)])

    params = init_weight()
    t = jnp.linspace(0,T,num_pts)
    t = jnp.reshape(t, (-1,1))
    # t = jnp.array([np.linspace]).T

    schedlue = optax.linear_schedule(0.001,0.00001,0.000000001,2000)
    optimizer = optax.adam(0.1)
    opt_state = optimizer.init(params)

    xloss_grad = jax.jit(jax.grad(xloss, argnums=0))

    final_res = None 
    best_loss = float('inf')

    ref_traj = compute_ref_traj(T, num_pts, x0 = x0)

    for i in range(40000):
        # x = forward(params, t)
        val = xloss(params, t, x0_extend)
        if val < best_loss:
            best_loss = val 
            final_res = copy.deepcopy(params) 
        print(i, val)
        grads = xloss_grad(params, t, x0_extend)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
    # params = final_res

    
    loss_val = xloss(final_res, t, x0_extend)
    print(loss_val)

    num_test_pts = 1000

    traj_nn = nn_solution(x0, T, final_res, num_test_pts)

    traj_gt = integrate_solution(x0, T, num_test_pts)

    import matplotlib.pyplot as plt 
    # plt.plot(jnp.linspace(0,T,num_test_pts), traj_nn[:,3])
    # plt.plot(jnp.linspace(0,T,num_test_pts), traj_gt[:,3])
    # plt.plot(jnp.linspace(0,T,num_pts), ref_traj[:,3])
    plt.plot(traj_nn[:,3], traj_nn[:,0])
    plt.plot(traj_gt[:,3], traj_gt[:,0])
    # plt.plot(ref_traj[:,3], ref_traj[:,0])
    plt.show()