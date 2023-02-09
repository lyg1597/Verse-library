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
from scipy.linalg import solve_continuous_are, expm 
from scipy.integrate import quad
import pickle
jax.config.update('jax_platform_name', 'cpu')

DIM = 5

# More complicated LQR example

def func(t,x,args):
    A, B, K = args
    # A = jnp.array([[2,1],[1,2]])
    # B = jnp.array([[1,0],[0,1]])
    # K = args

    x_dot = A@x-B@K@x

    return x_dot

# The first way of solving it is approximate the integration with Riemann sum 
# Solve ode using ode solver
def ode_solve(func, y0, t1, arg=None, sample_num=10000):
    term = ODETerm(func)
    solver = Dopri5()
    sample_array = jnp.linspace(0, t1, sample_num)
    saveat = SaveAt(ts = sample_array)
    dt = 0.005
    sol = diffeqsolve(term, solver, t0=0, t1=t1, dt0=dt, y0=y0, saveat=saveat, args=arg)
    return sol.ys

def loss(K, x0, t1, A, B, R):
    sample_num = 5000
    traj = ode_solve(func, x0, t1, (A,B,K), sample_num=sample_num)
    traj = jnp.squeeze(traj)
    delta = t1/sample_num 
    # tmp = traj.reshape((traj.shape[0]*traj.shape[2],-1))
    J = jnp.sum((jnp.diag(traj@traj.T)+jnp.diag(traj@K.T@R@K@traj.T))*delta)
    return J

def func_check(t, args):
    A,B,K,R,x0 = args
    xt = expm((A-B@K)*t)@x0
    res = xt.T@xt + xt.T@K.T@R@K@xt 
    return res 

def non_diff_cost(t1, A, B, K, R, x0):
    res = quad(func_check, 0, t1, args = ((A,B,K,R,x0),))
    return res[0]


# The second way is to use ground truth solution and actually integrate the result

if __name__ == "__main__":
    NUM = 10
    
    loss_diff_list = []
    loss_diff_gt_list = []
    K_list = []
    K_gt_list = []
    loss_true_list = []
    loss_true_gt_list = []
    for j in range(2,NUM):
        DIM = j
        schedule = optax.linear_schedule(1.0, 0.1, 0.0001, 2000)
        start_learning_rate = 5.0
        optimizer = optax.adam(0.1)

        # Initialize parameters of the model + optimizer.
        A_base = jnp.ones((DIM,DIM))
        # for i in range(DIM):
        #     for j in range(DIM):
        #         A_base = A_base.at[i,j].set(i+j+1)
        A = A_base@A_base.T
        B = jnp.eye(DIM)
        Q = jnp.eye(DIM)
        R = 2*jnp.eye(DIM,DIM)
        for i in range(DIM):
            R = R.at[i,i].set(i+1)
        R = 1/2*R
        print(A)
        print(R)
        P = solve_continuous_are(A,B,Q,R)
        K = jnp.linalg.inv(R)@B.T@P
        print(K)
        # params = jnp.array([[4.0648, 1.6719],[2.4455, 4.1895]])
        params = (DIM+1)*jnp.ones((DIM,DIM))
        # params = K
        opt_state = optimizer.init(params)

        # x0 = jnp.array(np.random.uniform(-1,1,(2,2))).T
        x0 = jnp.ones((DIM,1))
        t1 = 20
        
        best_loss = float('inf')
        final_res = params
        
        res = loss(params, x0, t1, A, B, R)
        print(res)
        res_grad = jax.grad(loss)(params, x0, t1, A, B, R)
        print(res_grad)

        # loss_batch = jax.vmap(loss, (None, 1, None), 1)

        tmp_loss = jit(loss) 
        tmp_grad = jit(jax.grad(loss))
        # tmp_loss_batch = jit(loss_batch)
        # tmp_grad_batch = jit(jax.grad(loss_batch))
        for i in range(2000):
            val = tmp_loss(params, x0, t1, A, B, R)
            print(i, val)
            if val < best_loss:
                best_loss = val 
                final_res = copy.deepcopy(params) 
            
            grads = tmp_grad(params, x0, t1, A, B, R)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

        # theta_t1 = final_res[0]
        # theta_t2 = final_res[1]
        # theta_t3 = final_res[2]
        # theta_t4 = final_res[3]
        # theta_t5 = final_res[4]
        # theta_t6 = final_res[5]
        # theta_t7 = final_res[6]
        print(K)
        print(final_res)
        val1 = loss(K, x0, t1, A, B, R)
        print(val1)
        val2 = loss(final_res, x0, t1, A, B, R)
        print(val2)
        val3 = non_diff_cost(t1,A,B,K,R,x0)
        print(val3)
        val4 = non_diff_cost(t1,A,B,final_res,R,x0)
        print(val4)

        K_list.append(final_res)
        K_gt_list.append(K)
        loss_diff_list.append(float(val2))
        loss_diff_gt_list.append(float(val1))
        loss_true_list.append(val4)
        loss_true_gt_list.append(val3)

    with open('res.pkl','wb+') as f:
        pickle.dump((loss_diff_list, loss_diff_gt_list, loss_true_list, loss_true_gt_list, K_list, K_gt_list), f)

    plt.figure()
    plt.plot([*range(2,NUM)],loss_diff_list,'b')
    plt.plot([*range(2,NUM)],loss_diff_gt_list,'g')

    plt.figure()
    plt.plot([*range(2,NUM)],loss_true_list,'b')
    plt.plot([*range(2,NUM)],loss_true_gt_list,'g')
    plt.show()