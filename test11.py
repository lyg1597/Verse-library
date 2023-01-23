import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax.example_libraries import optimizers
from jax import jacfwd, jacrev, jit
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

def system_solution(t, t1, t2, t3, t4):
    A = jnp.array([[0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0]])
    traj0 = jnp.array([[30],[1],[10],[2],[0],[0],[2],[0]])
    traj1 = jscipy.linalg.expm(A*t1)@traj0
    # print(traj1)
    traj1 = traj1.at[4,0].set(-0.5)
    traj2 = jscipy.linalg.expm(A*(t2-t1))@traj1
    traj2 = traj2.at[4,0].set(0)
    traj3 = jscipy.linalg.expm(A*(t3-t2))@traj2
    traj3 = traj3.at[7,0].set(-0.5)
    traj4 = jscipy.linalg.expm(A*(t4-t3))@traj3
    traj4 = traj4.at[7,0].set(0)
    traj5 = jscipy.linalg.expm(A*(t-t4))@traj4
    x1 = traj5[0,0]
    v1 = traj5[1,0]
    x2 = traj5[2,0]
    v2 = traj5[3,0]
    x3 = traj5[5,0]
    v3 = traj5[6,0]
    return x1, v1, x2, v2, x3, v3

def loss_func_solution(t, theta_t1, theta_t2, theta_t3, theta_t4):
    end_state = system_solution(t, theta_t1, theta_t2, theta_t3, theta_t4)
    end_x1 = end_state[0]
    end_v1 = end_state[1]
    end_x2 = end_state[2]
    end_v2 = end_state[3]
    end_x3 = end_state[4]
    end_v3 = end_state[5]
    pos_diff1 = ((end_x1-end_x2)-5)**2
    pos_diff2 = ((end_x2-end_x3)-5)**2
    vel_diff1 = (end_v1-end_v2)**2
    vel_diff2 = (end_v2-end_v3)**2
    # loss = pos_diff + vel_diff*1000 + jax.nn.relu(theta_t1 - theta_t2)*100000
    loss = pos_diff1 + \
        pos_diff2 + vel_diff1*1000 + vel_diff2*1000 + \
        jax.nn.relu(-theta_t1)*100000 + jax.nn.relu(-theta_t2)*100000 + \
        jax.nn.relu(-theta_t3)*100000 + jax.nn.relu(-theta_t4)*100000 + \
        jax.nn.relu(-(theta_t2-theta_t1))*100000 + jax.nn.relu(-(theta_t3-theta_t2))*100000 + jax.nn.relu(-(theta_t4-theta_t3))*100000
    
    return loss

if __name__ == "__main__":

    learning_rate = 0.0001

    t = 60.0
    theta_t1 = 14.0 
    theta_t2 = 16.0
    theta_t3 = 16.0 
    theta_t4 = 20.0

    grad_loss = jax.grad(loss_func_solution, argnums=(1,2,3,4,))
    # grad_loss = jit(grad_loss)
    for i in range(500):
        loss_val = loss_func_solution(t,theta_t1, theta_t2, theta_t3, theta_t4)
        grad_t1, grad_t2, grad_t3, grad_t4 = grad_loss(t,theta_t1, theta_t2, theta_t3, theta_t4)
        print(i, theta_t1, theta_t2, theta_t3, theta_t4, loss_val, grad_t1, grad_t2, grad_t3, grad_t4)
        theta_t1 -= learning_rate * grad_t1
        theta_t2 -= learning_rate * grad_t2
        theta_t3 -= learning_rate * grad_t3
        theta_t4 -= learning_rate * grad_t4

    print(theta_t1, theta_t2, theta_t3, theta_t4, system_solution(t, theta_t1, theta_t2, theta_t3, theta_t4))