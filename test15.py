import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax import jacfwd, jacrev, jit
import optax
import matplotlib
import matplotlib.pyplot as plt
jax.config.update('jax_platform_name', 'cpu')

# Moon lander performs one up thrust followed by one right thrust followed by one up thrust
# Let's add two obstacles

def func1(A, traj0, t, t1, t2, t3, t4, t5, t6, y0, vy0, z0, vz0):
    traj1 = jscipy.linalg.expm(A*t)@traj0
    return traj1 

def func2(A, traj0, t, t1, t2, t3, t4, t5, t6, y0, vy0, z0, vz0):
    traj1 = jscipy.linalg.expm(A*t1)@traj0
    traj1 = traj1.at[2,0].set(0.1)
    traj2 = jscipy.linalg.expm(A*(t-t1))@traj1 
    return traj2

def func3(A, traj0, t, t1, t2, t3, t4, t5, t6, y0, vy0, z0, vz0):
    traj1 = jscipy.linalg.expm(A*t1)@traj0
    traj1 = traj1.at[2,0].set(0.1)
    traj2 = jscipy.linalg.expm(A*(t2-t1))@traj1 
    traj2 = traj2.at[2,0].set(-0.1)
    traj3 = jscipy.linalg.expm(A*(t-t2))@traj2 
    return traj3

def func4(A, traj0, t, t1, t2, t3, t4, t5, t6, y0, vy0, z0, vz0):
    traj1 = jscipy.linalg.expm(A*t1)@traj0
    traj1 = traj1.at[2,0].set(0.1)
    traj2 = jscipy.linalg.expm(A*(t2-t1))@traj1 
    traj2 = traj2.at[2,0].set(-0.1)
    traj3 = jscipy.linalg.expm(A*(t3-t2))@traj2 
    traj3 = traj3.at[5,0].set(-2.0)
    traj4 = jscipy.linalg.expm(A*(t-t3))@traj3
    return traj4 

def func5(A, traj0, t, t1, t2, t3, t4, t5, t6, y0, vy0, z0, vz0):
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

def func6(A, traj0, t, t1, t2, t3, t4, t5, t6, y0, vy0, z0, vz0):
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

def false1(A, traj0, t, t1, t2, t3, t4, t5, t6, y0, vy0, z0, vz0):
    return jax.lax.cond(t<t2, func2, false2, A, traj0, t, t1, t2, t3, t4, t5, t6, y0, vy0, z0, vz0)

def false2(A, traj0, t, t1, t2, t3, t4, t5, t6, y0, vy0, z0, vz0):
    return jax.lax.cond(t<t3, func3, false3, A, traj0, t, t1, t2, t3, t4, t5, t6, y0, vy0, z0, vz0)

def false3(A, traj0, t, t1, t2, t3, t4, t5, t6, y0, vy0, z0, vz0):
    return jax.lax.cond(t<t4, func4, false4, A, traj0, t, t1, t2, t3, t4, t5, t6, y0, vy0, z0, vz0)

def false4(A, traj0, t, t1, t2, t3, t4, t5, t6, y0, vy0, z0, vz0):
    return jax.lax.cond(t<t5, func5, false5, A, traj0, t, t1, t2, t3, t4, t5, t6, y0, vy0, z0, vz0)

def false5(A, traj0, t, t1, t2, t3, t4, t5, t6, y0, vy0, z0, vz0):
    return jax.lax.cond(t<t6, func6, false6, A, traj0, t, t1, t2, t3, t4, t5, t6, y0, vy0, z0, vz0)

def false6(A, traj0, t, t1, t2, t3, t4, t5, t6, y0, vy0, z0, vz0):
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

def system_trajectory(t, t1, t2, t3, t4, t5, t6, y0, vy0, z0, vz0):
    A = jnp.array([[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,0,0,0],[0,0,0,0,1,0],[0,0,0,0,-0.01,1],[0,0,0,0,0,0]])
    traj0 = jnp.array([[z0],[vz0],[-0.1],[y0],[vy0],[0]])
    
    res = jax.lax.cond(t<t1, func1, false1, A, traj0, t, t1, t2, t3, t4, t5, t6, y0, vy0, z0, vz0)
    return res
    # if t<t1:
    #     traj1 = jscipy.linalg.expm(A*t)@traj0
    #     return traj1 
    # elif t<t2:
    #     traj1 = jscipy.linalg.expm(A*t1)@traj0
    #     traj1 = traj1.at[2,0].set(0.1)
    #     traj2 = jscipy.linalg.expm(A*(t-t1))@traj1 
    #     return traj2
    # elif t<t3: 
    #     traj1 = jscipy.linalg.expm(A*t1)@traj0
    #     traj1 = traj1.at[2,0].set(0.1)
    #     traj2 = jscipy.linalg.expm(A*(t2-t1))@traj1 
    #     traj2 = traj2.at[2,0].set(-0.1)
    #     traj3 = jscipy.linalg.expm(A*(t-t2))@traj2 
    #     return traj3
    # elif t<t4:
    #     traj1 = jscipy.linalg.expm(A*t1)@traj0
    #     traj1 = traj1.at[2,0].set(0.1)
    #     traj2 = jscipy.linalg.expm(A*(t2-t1))@traj1 
    #     traj2 = traj2.at[2,0].set(-0.1)
    #     traj3 = jscipy.linalg.expm(A*(t3-t2))@traj2 
    #     traj3 = traj3.at[5,0].set(-2.0)
    #     traj4 = jscipy.linalg.expm(A*(t-t3))@traj3
    #     return traj4 
    # elif t<t5:
    #     traj1 = jscipy.linalg.expm(A*t1)@traj0
    #     traj1 = traj1.at[2,0].set(0.1)
    #     traj2 = jscipy.linalg.expm(A*(t2-t1))@traj1 
    #     traj2 = traj2.at[2,0].set(-0.1)
    #     traj3 = jscipy.linalg.expm(A*(t3-t2))@traj2 
    #     traj3 = traj3.at[5,0].set(-2.0)
    #     traj4 = jscipy.linalg.expm(A*(t4-t3))@traj3
    #     traj4 = traj4.at[5,0].set(0)
    #     traj5 = jscipy.linalg.expm(A*(t-t4))@traj4
    #     return traj5
    # elif t<t6:
    #     traj1 = jscipy.linalg.expm(A*t1)@traj0
    #     traj1 = traj1.at[2,0].set(0.1)
    #     traj2 = jscipy.linalg.expm(A*(t2-t1))@traj1 
    #     traj2 = traj2.at[2,0].set(-0.1)
    #     traj3 = jscipy.linalg.expm(A*(t3-t2))@traj2 
    #     traj3 = traj3.at[5,0].set(-2.0)
    #     traj4 = jscipy.linalg.expm(A*(t4-t3))@traj3
    #     traj4 = traj4.at[5,0].set(0)
    #     traj5 = jscipy.linalg.expm(A*(t5-t4))@traj4
    #     traj5 = traj5.at[2,0].set(0.1)
    #     traj6 = jscipy.linalg.expm(A*(t-t5))@traj5
    #     return traj6
    # else:
    #     traj1 = jscipy.linalg.expm(A*t1)@traj0
    #     traj1 = traj1.at[2,0].set(0.1)
    #     traj2 = jscipy.linalg.expm(A*(t2-t1))@traj1 
    #     traj2 = traj2.at[2,0].set(-0.1)
    #     traj3 = jscipy.linalg.expm(A*(t3-t2))@traj2 
    #     traj3 = traj3.at[5,0].set(-2.0)
    #     traj4 = jscipy.linalg.expm(A*(t4-t3))@traj3
    #     traj4 = traj4.at[5,0].set(0)
    #     traj5 = jscipy.linalg.expm(A*(t5-t4))@traj4
    #     traj5 = traj5.at[2,0].set(0.1)
    #     traj6 = jscipy.linalg.expm(A*(t6-t5))@traj5
    #     traj6 = traj6.at[2,0].set(-0.1)
    #     traj7 = jscipy.linalg.expm(A*(t-t6))@traj6
    #     return traj7

def system_solution(t, t1, t2, t3, t4, t5, t6, y0, vy0, z0, vz0):
    A = jnp.array([[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,0,0,0],[0,0,0,0,1,0],[0,0,0,0,-0.01,1],[0,0,0,0,0,0]])
    traj0 = jnp.array([[z0],[vz0],[-0.1],[y0],[vy0],[0]])
    # Start upward thrust
    traj1 = jscipy.linalg.expm(A*t1)@traj0
    traj1 = traj1.at[2,0].set(0.1)
    # Stop upward thrust
    traj2 = jscipy.linalg.expm(A*(t2-t1))@traj1 
    traj2 = traj2.at[2,0].set(-0.1)
    # Start rightward thrust
    traj3 = jscipy.linalg.expm(A*(t3-t2))@traj2 
    traj3 = traj3.at[5,0].set(-2.0)
    # Stop rightward thrust
    traj4 = jscipy.linalg.expm(A*(t4-t3))@traj3
    traj4 = traj4.at[5,0].set(0)
    # Start upward thrust
    traj5 = jscipy.linalg.expm(A*(t5-t4))@traj4
    traj5 = traj5.at[2,0].set(0.1)
    traj6 = jscipy.linalg.expm(A*(t6-t5))@traj5
    # Stop upward thrust
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

def body_fun(i, val):
    steps, theta_t1, theta_t2, theta_t3, theta_t4, theta_t5, theta_t6, y0, vy0, z0, vz0, unsafe_loss = val
    step = steps[i]
    traj_state = system_trajectory(step, theta_t1,theta_t2,theta_t3,theta_t4,theta_t5,theta_t6,y0,vy0,z0,vz0)
    traj_z = traj_state[0,0]
    traj_vz = traj_state[1,0]
    traj_az = traj_state[2,0]
    traj_y = traj_state[3,0]
    traj_vy = traj_state[4,0]
    traj_ay = traj_state[5,0]
    unsafe_loss += jax.nn.relu(2000-traj_y)*jax.nn.relu(traj_y-1500)*jax.nn.relu(800-traj_z)
    unsafe_loss += jax.nn.relu(0-traj_z)
    return (steps, theta_t1, theta_t2, theta_t3, theta_t4, theta_t5, theta_t6, y0, vy0, z0, vz0, unsafe_loss)

def loss_func_solution(t, theta_t1, theta_t2, theta_t3, theta_t4, theta_t5, theta_t6, y0, vy0, z0, vz0, y_targ, z_targ):
    end_state = system_solution(t, theta_t1, theta_t2, theta_t3, theta_t4, theta_t5, theta_t6, y0, vy0, z0, vz0)
    end_z = end_state[0,0]
    end_vz = end_state[1,0]
    end_az = end_state[2,0]
    end_y = end_state[3,0]
    end_vy = end_state[4,0]
    end_ay = end_state[5,0]

    steps = jnp.linspace(0,t,1000)
    init_val = (steps, theta_t1, theta_t2, theta_t3, theta_t4, theta_t5, theta_t6, y0, vy0, z0, vz0, 0)
    _, _, _, _, _, _, _, _, _, _, _, unsafe_loss = jax.lax.fori_loop(0,1000,body_fun,init_val)
    
    # res = jnp.array([system_trajectory(step, theta_t1,theta_t2,theta_t3,theta_t4,theta_t5,theta_t6,y0,vy0,z0,vz0) for step in steps])
    # tmp = jax.nn.relu(2000-res[:,3,0])*jax.nn.relu(res[:,3,0]-1500)*jax.nn.relu(900-res[:,0,0])
    # unsafe_loss = jnp.sum(tmp)
    
    # unsafe_loss = 0
    # for step in steps:
    #     traj_state = system_trajectory(step, theta_t1,theta_t2,theta_t3,theta_t4,theta_t5,theta_t6,y0,vy0,z0,vz0)
    #     traj_z = traj_state[0,0]
    #     traj_vz = traj_state[1,0]
    #     traj_az = traj_state[2,0]
    #     traj_y = traj_state[3,0]
    #     traj_vy = traj_state[4,0]
    #     traj_ay = traj_state[5,0]
    #     unsafe_loss += jax.nn.relu(2000-traj_y)*jax.nn.relu(traj_y-1500)*jax.nn.relu(900-traj_z)

    pos_loss = (z_targ - end_z)**2 + (y_targ - end_y)**2 
    vel_loss = jax.nn.relu(-2-end_vz)**2 + end_vy**2
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

    loss = pos_loss + vel_loss*1000 + seq_loss*1000000 + above_0_loss*100000000 + unsafe_loss*1000
    return loss 

if __name__ == "__main__":
    import numpy as np 
    y0 = 0
    vy0 = 82
    z0 = 800
    vz0 = 0

    t = 221.89854 
    t1 = 5.081437 
    t2 = 16.966002 
    t3 = 56.311783 
    t4 = 77.31678 
    t5 = 112.204636 
    t6 = 201.269
    params = jnp.array([t, t1, t2, t3, t4, t5, t6])
    res_y = []
    res_z = []
    res_vy = []
    res_vz = []
    res_ay = []
    res_az = []
    steps = np.linspace(0,t,1000)
    tmp = jit(system_trajectory)
    for step in steps:
        print(step)
        res = tmp(step,t1,t2,t3,t4,t5,t6,y0,vy0,z0,vz0)
        res_z.append(res[0,0])
        res_vz.append(res[1,0])
        res_az.append(res[2,0])
        res_y.append(res[3,0])
        res_vy.append(res[4,0])
        res_ay.append(res[5,0])
    print(system_solution(t,t1,t2,t3,t4,t5,t6,y0,vy0,z0,vz0))
    print(tmp(t,t1,t2,t3,t4,t5,t6,y0,vy0,z0,vz0))
    print(loss_func(params, y0, vy0,z0,vz0,4000,0))
    plt.figure()
    plt.plot([1500,2000,2000,1500,1500],[0,0,800,800,0])
    plt.plot(res_y, res_z)
    plt.figure()
    plt.plot(steps, res_y)
    plt.figure()
    plt.plot(steps, res_z)
    plt.figure()
    plt.plot(steps, res_vy)
    plt.figure()
    plt.plot(steps, res_vz)
    plt.figure()
    plt.plot(steps, res_ay)
    plt.figure()
    plt.plot(steps, res_az)
    plt.show()

# if __name__ == "__main__": 
#     schedule = optax.linear_schedule(5.0, 1.0, 0.001, 3000)
#     start_learning_rate = 5.0
#     optimizer = optax.adam(schedule)

#     # Initialize parameters of the model + optimizer.
#     params = jnp.array([200.0, 0.0, 100.0, 110.0, 120.0,130.0,140.0])
#     opt_state = optimizer.init(params)
    
#     y0 = 0
#     vy0 = 82
#     z0 = 800
#     vz0 = 0
#     y_targ = 4000
#     z_targ = 0
    
#     tmp_loss = jit(loss_func) 
#     tmp = jit(jax.grad(loss_func))
#     final_res = []
#     best_loss = float('inf')
#     for i in range(5000):
#         val = tmp_loss(params, y0, vy0, z0, vz0, y_targ, z_targ)
#         if val < best_loss:
#             best_loss = val 
#             final_res = params
#         print(i, val, params)
#         grads = tmp(params, y0, vy0, z0, vz0, y_targ, z_targ)
#         updates, opt_state = optimizer.update(grads, opt_state)
#         params = optax.apply_updates(params, updates)

#     t = final_res[0]
#     theta_t1 = final_res[1]
#     theta_t2 = final_res[2]
#     theta_t3 = final_res[3]
#     theta_t4 = final_res[4]
#     theta_t5 = final_res[5]
#     theta_t6 = final_res[6]
#     print(t, theta_t1, theta_t2, theta_t3, theta_t4, theta_t5, theta_t6, system_solution(t, theta_t1, theta_t2, theta_t3, theta_t4, theta_t5, theta_t6,y0,vy0,z0,vz0))

