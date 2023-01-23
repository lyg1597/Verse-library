import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax.example_libraries import optimizers
from jax import jacfwd, jacrev, jit
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

def system_solution(t, t1, t2):
    A = jnp.array([[0,1,0,0,0],[0,0,0,0,0],[0,0,0,1,0],[0,0,0,0,1],[0,0,0,0,0]])
    traj0 = jnp.array([[20],[1],[0],[2],[0]])
    traj1 = jscipy.linalg.expm(A*t1)@traj0
    # print(traj1)
    traj1 = traj1.at[4,0].set(-0.5)
    traj2 = jscipy.linalg.expm(A*(t2-t1))@traj1
    # print(traj2)
    traj2 = traj2.at[4,0].set(0)
    traj3 = jscipy.linalg.expm(A*(t-t2))@traj2
    x1 = traj3[0,0]
    v1 = traj3[1,0]
    x2 = traj3[2,0]
    v2 = traj3[3,0]
    return x1, v1, x2, v2

def loss_func_solution(t, theta_t1, theta_t2, ref_pos):
    end_state = system_solution(t, theta_t1, theta_t2)
    end_x1 = end_state[0]
    end_v1 = end_state[1]
    end_x2 = end_state[2]
    end_v2 = end_state[3]
    pos_diff = ((end_x1-end_x2)-5)**2
    vel_diff = (end_v1-end_v2)**2
    # loss = pos_diff + vel_diff*1000 + jax.nn.relu(theta_t1 - theta_t2)*100000
    loss = pos_diff + vel_diff*1000 + jax.nn.relu(-theta_t1)*100000 + jax.nn.relu(-theta_t2)*100000
    
    # print(jax.numpy.asarray(loss))

    return loss

# if __name__ == "__main__":
#     t = 30.0
#     t1 = 10.0
#     t2 = 12.0
#     print(system_solution(t, t1, t2))

if __name__ == "__main__":
    t,x,v = 0.0,0.0,0.0
    theta = -0.5
    theta1, theta2, theta3 = 10.0,20.0,30.0
    t_max = 40.0

    # res = jacfwd(compute_trajectory)([t,x,v,theta, t_max])    
    # print("jacfwd result, with shape", res.shape)
    # print(res)

    # trajectory = jnp.array(compute_trajectory([t,x,v,theta, t_max]))
    # plt.plot(trajectory[:,0], trajectory[:,1])
    # plt.plot(trajectory[:,0], trajectory[:,2])
    # plt.show()
    # grad_loss = jit(grad_loss)

    learning_rate = 0.0001

    theta_a1 = 2.0
    theta_a2 = -1.0
    theta_t1 = 0.0
    theta_t2 = 15.0
    ref_pos = 250.0

    theta_a = -0.8003115
    
    # res = loss_func(t,x,v,theta_a1, theta_a2, t_max, ref_pos)
    # print(res)

    # trajectory = jnp.array(compute_trajectory([t,x,v,theta_a1, theta_a2, t_max]))
    # plt.plot(trajectory[:,0], trajectory[:,1])
    # plt.plot(trajectory[:,0], trajectory[:,2])
    # plt.show()

    t = 60.0
    theta_t1 = 10.0 
    theta_t2 = 20.0
    ref_pos = 250.0

    grad_loss = jax.grad(loss_func_solution, argnums=(1,2,))
    # grad_loss = jit(grad_loss)
    for i in range(500):
        loss_val = loss_func_solution(t,theta_t1, theta_t2,ref_pos)
        grad_t1, grad_t2, = grad_loss(t,theta_t1, theta_t2,ref_pos)
        print(i, theta_t1, theta_t2, loss_val, grad_t1, grad_t2)
        theta_t1 -= learning_rate * grad_t1
        theta_t2 -= learning_rate * grad_t2

    print(theta_t1, theta_t2, system_solution(t, theta_t1, theta_t2))