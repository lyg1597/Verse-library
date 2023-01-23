import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax import jacfwd, jacrev, jit
import optax
import matplotlib
import matplotlib.pyplot as plt

# Moon lander performs one up thrust followed by one right thrust followed by one up thrust
# No obstacles

def system_trajectory(t, t1, t2, t3, t4, t5, t6, y0, vy0, z0, vz0):
    A = jnp.array([[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,0,0,0],[0,0,0,0,1,0],[0,0,0,0,-0.01,1],[0,0,0,0,0,0]])
    traj0 = jnp.array([[z0],[vz0],[-0.1],[y0],[vy0],[0]])
    if t<t1:
        traj1 = jscipy.linalg.expm(A*t)@traj0
        return traj1 
    elif t<t2:
        traj1 = jscipy.linalg.expm(A*t1)@traj0
        traj1 = traj1.at[2,0].set(0.1)
        traj2 = jscipy.linalg.expm(A*(t-t1))@traj1 
        return traj2
    elif t<t3: 
        traj1 = jscipy.linalg.expm(A*t1)@traj0
        traj1 = traj1.at[2,0].set(0.1)
        traj2 = jscipy.linalg.expm(A*(t2-t1))@traj1 
        traj2 = traj2.at[2,0].set(-0.1)
        traj3 = jscipy.linalg.expm(A*(t-t2))@traj2 
        return traj3
    elif t<t4:
        traj1 = jscipy.linalg.expm(A*t1)@traj0
        traj1 = traj1.at[2,0].set(0.1)
        traj2 = jscipy.linalg.expm(A*(t2-t1))@traj1 
        traj2 = traj2.at[2,0].set(-0.1)
        traj3 = jscipy.linalg.expm(A*(t3-t2))@traj2 
        traj3 = traj3.at[5,0].set(-2.0)
        traj4 = jscipy.linalg.expm(A*(t-t3))@traj3
        return traj4 
    elif t<t5:
        traj1 = jscipy.linalg.expm(A*t1)@traj0
        traj1 = traj1.at[2,0].set(0.1)
        traj2 = jscipy.linalg.expm(A*(t2-t1))@traj1 
        traj2 = traj2.at[2,0].set(-0.1)
        traj3 = jscipy.linalg.expm(A*(t3-t2))@traj2 
        traj3 = traj3.at[5,0].set(-2.0)
        traj4 = jscipy.linalg.expm(A*(t4-t3))@traj3
        traj4 = traj4.at[5,0].set(0)
        traj5 = jscipy.linalg.expm(A*(t-t4))@traj4
        return traj5
    elif t<t6:
        traj1 = jscipy.linalg.expm(A*t1)@traj0
        traj1 = traj1.at[2,0].set(0.1)
        traj2 = jscipy.linalg.expm(A*(t2-t1))@traj1 
        traj2 = traj2.at[2,0].set(-0.1)
        traj3 = jscipy.linalg.expm(A*(t3-t2))@traj2 
        traj3 = traj3.at[5,0].set(-2.0)
        traj4 = jscipy.linalg.expm(A*(t4-t3))@traj3
        traj4 = traj4.at[5,0].set(0)
        traj5 = jscipy.linalg.expm(A*(t5-t4))@traj4
        traj5 = traj5.at[2,0].set(0.1)
        traj6 = jscipy.linalg.expm(A*(t-t5))@traj5
        return traj6
    else:
        traj1 = jscipy.linalg.expm(A*t1)@traj0
        traj1 = traj1.at[2,0].set(0.1)
        traj2 = jscipy.linalg.expm(A*(t2-t1))@traj1 
        traj2 = traj2.at[2,0].set(-0.1)
        traj3 = jscipy.linalg.expm(A*(t3-t2))@traj2 
        traj3 = traj3.at[5,0].set(-2.0)
        traj4 = jscipy.linalg.expm(A*(t4-t3))@traj3
        traj4 = traj4.at[5,0].set(0)
        traj5 = jscipy.linalg.expm(A*(t5-t4))@traj4
        traj5 = traj5.at[2,0].set(0.1)
        traj6 = jscipy.linalg.expm(A*(t6-t5))@traj5
        traj6 = traj6.at[2,0].set(-0.1)
        traj7 = jscipy.linalg.expm(A*(t-t6))@traj6
        return traj7

def system_solution(t, t1, t2, t3, t4, t5, t6, y0, vy0, z0, vz0):
    A = jnp.array([[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,0,0,0],[0,0,0,0,1,0],[0,0,0,0,-0.01,1],[0,0,0,0,0,0]])
    traj0 = jnp.array([[z0],[vz0],[-0.1],[y0],[vy0],[0]])
    traj1 = jscipy.linalg.expm(A*t1)@traj0
    traj1 = traj1.at[2,0].set(0.1)
    traj2 = jscipy.linalg.expm(A*(t2-t1))@traj1 
    traj2 = traj2.at[2,0].set(-0.1)
    traj3 = jscipy.linalg.expm(A*(t3-t2))@traj2 
    traj3 = traj3.at[5,0].set(-2.0)
    traj4 = jscipy.linalg.expm(A*(t4-t3))@traj3
    traj4 = traj4.at[5,0].set(0)
    traj5 = jscipy.linalg.expm(A*(t5-t4))@traj4
    traj5 = traj5.at[2,0].set(0.1)
    traj6 = jscipy.linalg.expm(A*(t6-t5))@traj5
    traj6 = traj6.at[2,0].set(-0.1)
    traj7 = jscipy.linalg.expm(A*(t-t6))@traj6
    return traj7


def loss_func(params, y0, vy0, z0, vz0, y_targ, z_targ):
    t = params[0]
    theta_t1 = params[1]
    theta_t2 = params[2]
    theta_t3 = params[3]
    theta_t4 = params[4]
    theta_t5 = params[5]
    theta_t6 = params[6]
    loss = loss_func_solution(t, theta_t1, theta_t2, theta_t3, theta_t4, theta_t5, theta_t6, y0, vy0, z0, vz0, y_targ, z_targ)
    return loss

def loss_func_solution(t, theta_t1, theta_t2, theta_t3, theta_t4, theta_t5, theta_t6, y0, vy0, z0, vz0, y_targ, z_targ):
    end_state = system_solution(t, theta_t1, theta_t2, theta_t3, theta_t4, theta_t5, theta_t6, y0, vy0, z0, vz0)
    end_z = end_state[0,0]
    end_vz = end_state[1,0]
    end_az = end_state[2,0]
    end_y = end_state[3,0]
    end_vy = end_state[4,0]
    end_ay = end_state[5,0]

    pos_loss = (z_targ - end_z)**2 + (y_targ - end_y)**2 
    vel_loss = end_vz**2 + end_vy**2
    seq_loss = jax.nn.relu(-(theta_t2-theta_t1))+\
        jax.nn.relu(-(theta_t3-theta_t2))+\
        jax.nn.relu(-(theta_t4-theta_t3))+\
        jax.nn.relu(-(theta_t5-theta_t4))+\
        jax.nn.relu(-(theta_t6-theta_t5))+\
        jax.nn.relu(-(t-theta_t6))
    above_0_loss = jax.nn.relu(-theta_t1)+\
        jax.nn.relu(-theta_t2)+\
        jax.nn.relu(-theta_t3)+\
        jax.nn.relu(-theta_t4)+\
        jax.nn.relu(-theta_t5)+\
        jax.nn.relu(-theta_t6)+\
        jax.nn.relu(-t)
    
    optimization_loss = jax.nn.relu(theta_t2 - theta_t1)+jax.nn.relu(theta_t4-theta_t3)+jax.nn.relu(theta_t6-theta_t5)

    loss = pos_loss*10 + vel_loss*10000 + seq_loss*100000 + above_0_loss*100000 + optimization_loss*1000
    return loss 

# if __name__ == "__main__":
#     import numpy as np 
#     y0 = 0
#     vy0 = 82
#     z0 = 800
#     vz0 = 0

#     t = 191.14265 
#     t1 = 29.671904 
#     t2 = 30.87238 
#     t3 = 56.221222  
#     t4 = 77.22123 
#     t5 = 91.602715 
#     t6 = 185.97371
#     res_y = []
#     res_z = []
#     res_vy = []
#     res_vz = []
#     res_ay = []
#     res_az = []
#     steps = np.linspace(0,t,1000)
#     for step in steps:
#         res = system_trajectory(step,t1,t2,t3,t4,t5,t6,y0,vy0,z0,vz0)
#         res_z.append(res[0,0])
#         res_vz.append(res[1,0])
#         res_az.append(res[2,0])
#         res_y.append(res[3,0])
#         res_vy.append(res[4,0])
#         res_ay.append(res[5,0])
#     print(system_solution(t,t1,t2,t3,t4,t5,t6,y0,vy0,z0,vz0))
#     print(system_trajectory(t,t1,t2,t3,t4,t5,t6,y0,vy0,z0,vz0))
#     plt.figure()
#     plt.plot(res_y, res_z)
#     plt.figure()
#     plt.plot(steps, res_y)
#     plt.figure()
#     plt.plot(steps, res_z)
#     plt.figure()
#     plt.plot(steps, res_vy)
#     plt.figure()
#     plt.plot(steps, res_vz)
#     plt.figure()
#     plt.plot(steps, res_ay)
#     plt.figure()
#     plt.plot(steps, res_az)
#     plt.show()

if __name__ == "__main__": 
    # schedule = optax.linear_schedule(5.0, 3.0, 0.001, 1500)
    start_learning_rate = 5.0
    optimizer = optax.adam(start_learning_rate)

    # Initialize parameters of the model + optimizer.
    params = jnp.array([10.0, 1.0, 2.0, 3.0, 4.0,5.0,6.0])
    opt_state = optimizer.init(params)
    
    y0 = 0
    vy0 = 82
    z0 = 800
    vz0 = 0
    y_targ = 4000
    z_targ = 0
    
    for i in range(2000):
        grads = jax.grad(loss_func)(params, y0, vy0, z0, vz0, y_targ, z_targ)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        print(i, loss_func(params, y0, vy0, z0, vz0, y_targ, z_targ), params)

    t = params[0]
    theta_t1 = params[1]
    theta_t2 = params[2]
    theta_t3 = params[3]
    theta_t4 = params[4]
    theta_t5 = params[5]
    theta_t6 = params[6]
    print(t, theta_t1, theta_t2, theta_t3, theta_t4, theta_t5, theta_t6, system_solution(t, theta_t1, theta_t2, theta_t3, theta_t4, theta_t5, theta_t6,y0,vy0,z0,vz0))

