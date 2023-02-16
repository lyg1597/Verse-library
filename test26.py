import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax import jacfwd, jacrev, jit
from jax.experimental.ode import odeint 
import optax
import matplotlib
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt
import copy
import numpy as np
jax.config.update('jax_platform_name', 'cpu')

# A small LQR example

def func(t,x,args):
    A = jnp.array([[2,1],[1,2]])
    B = jnp.array([[1,0],[0,1]])
    K = args

    x_dot = A@x-B@K@x

    return x_dot

# The first way of solving it is approximate the integration with Riemann sum 
# Solve ode using ode solver
def ode_solve(func, y0, t1, arg=None, sample_num=10000):
    term = ODETerm(func)
    solver = Dopri5()
    sample_array = jnp.linspace(0, t1, sample_num)
    saveat = SaveAt(ts = sample_array)
    dt = 0.1
    sol = diffeqsolve(term, solver, t0=0, t1=t1, dt0=dt, y0=y0, saveat=saveat, args=arg)
    return sol.ys

def loss(K, x0, t1):
    sample_num = 10000
    traj = ode_solve(func, x0, t1, K, sample_num=sample_num)
    traj = jnp.squeeze(traj)
    delta = t1/sample_num 
    # tmp = traj.reshape((traj.shape[0]*traj.shape[2],-1))
    J = jnp.sum((jnp.diag(traj@traj.T)+jnp.diag(traj@K.T@jnp.array([[3,-1],[-1,2]])@K@traj.T))*delta)
    return J

# The second way is to use ground truth solution and actually integrate the result

if __name__ == "__main__":
    schedule = optax.linear_schedule(1.0, 0.1, 0.0001, 2000)
    start_learning_rate = 5.0
    optimizer = optax.adam(schedule)

    # Initialize parameters of the model + optimizer.
    params = 4*jnp.eye(2,2)
    # params = jnp.array([[4.0648, 1.6719],[2.4455, 4.1895]])
    opt_state = optimizer.init(params)

    # x0 = jnp.array(np.random.uniform(-1,1,(2,2))).T
    x0 = jnp.array([[1],[1]])
    t1 = 20
    res = loss(params, x0, t1)
    
    best_loss = float('inf')
    final_res = None
    
    res = jax.xla_computation(loss)(params, x0, t1)
    with open('t.dot','w+') as f:
        f.write(res.as_hlo_dot_graph())

    # print(res)
    # res_grad = jax.grad(loss)(params, x0, t1)
    # print(res_grad)

    # loss_batch = jax.vmap(loss, (None, 1, None), 1)

    # tmp_loss = jit(loss) 
    # tmp_grad = jit(jax.grad(loss))
    # tmp_loss_batch = jit(loss_batch)
    # tmp_grad_batch = jit(jax.grad(loss_batch))
    # for i in range(500):
    #     val = tmp_loss(params, x0, t1)
    #     print(i, val, params)
    #     if val < best_loss:
    #         best_loss = val 
    #         final_res = copy.deepcopy(params) 
        
    #     grads = tmp_grad(params, x0, t1)
    #     updates, opt_state = optimizer.update(grads, opt_state)
    #     params = optax.apply_updates(params, updates)

    # # theta_t1 = final_res[0]
    # # theta_t2 = final_res[1]
    # # theta_t3 = final_res[2]
    # # theta_t4 = final_res[3]
    # # theta_t5 = final_res[4]
    # # theta_t6 = final_res[5]
    # # theta_t7 = final_res[6]
    # print(final_res)
    # print(tmp_loss(final_res, x0, t1))

