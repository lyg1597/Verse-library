import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
from jax import jacfwd, jacrev, jit
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

def system_solution(t, t1, t2):
    x = -1/2*0.5*(t-t2)**2+t1*t2-1/2*t1**2+t*t1-t1*t2 
    v = -0.5*(t-t2)+t1
    return x, v

def loss_func_solution(t, theta_t1, theta_t2, ref_pos):
    end_state = system_solution(t, theta_t1, theta_t2)
    end_pos = end_state[0]
    end_vel = end_state[1]
    pos_diff = (end_pos-ref_pos)**2
    vel_diff = (end_vel-0)**2
    loss = pos_diff + vel_diff*1000 + jax.nn.relu(theta_t1 - theta_t2)*100000
    # print(jax.numpy.asarray(loss))
    return loss

# if __name__ == "__main__":
#     t,x,v = 0.0,0.0,0.0
#     theta_t1 = 20.0
#     t_max = 30.0
#     ref_pos = 150.0
#     trajectory = jnp.array(compute_trajectory([t,x,v, theta_t1, t_max]))
#     plt.plot(trajectory[:,0], trajectory[:,1])
#     trajectory_smooth = jnp.array(compute_trajectory_smooth([t,x,v, theta_t1, t_max]))
#     plt.plot(trajectory_smooth[:,0], trajectory_smooth[:,1])

#     # import numpy as np 
#     # plt.figure()
#     # steps = np.linspace(0,50,100)
#     # res = []
#     # for step in steps:
#     #     tmp = decision_logic([step,0,0],[20])
#     #     res.append(tmp)
#     # plt.plot(steps, res)    
#     # res = []
#     # for step in steps:
#     #     tmp = decision_logic_smooth([step,0,0],[20])
#     #     res.append(tmp)
#     # plt.plot(steps, res)

#     # plt.figure()
#     # steps = np.linspace(10, 40, 300)
#     # res = []
#     # for step in steps:
#     #     tmp = loss_func(t,x,v,step,t_max,ref_pos)
#     #     print(tmp)
#     #     res.append(tmp)
#     # plt.plot(steps, res)    
#     # res = []
#     # for step in steps:
#     #     tmp = loss_func_smooth(t,x,v,step,t_max,ref_pos)
#     #     res.append(tmp)
#     # plt.plot(steps, res)    
#     plt.show()


# if __name__ == "__main__":
#     import numpy as np 
#     steps = np.linspace(0,50,100)
#     res = []
#     for step in steps:
#         tmp = decision_logic_smooth([step,0,0],[30])
#         res.append(tmp)
#     plt.plot(steps, res)
#     plt.show()

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

    t = 40.0
    theta_t1 = 0.0 
    theta_t2 = 0.0
    ref_pos = 250.0

    grad_loss = jax.grad(loss_func_solution, argnums=(1,2,))
    # grad_loss = jit(grad_loss)
    for i in range(500):
        loss_val = loss_func_solution(t,theta_t1, theta_t2,ref_pos)
        grad_t1, grad_t2, = grad_loss(t,theta_t1, theta_t2,ref_pos)
        print(i, theta_t1, theta_t2, loss_val, grad_t1, grad_t2)
        theta_t1 -= learning_rate * grad_t1
        theta_t2 -= learning_rate * grad_t2

    print(theta_t1, theta_t2, system_solution(40, theta_t1, theta_t2))