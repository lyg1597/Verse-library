import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax import jacfwd, jacrev, jit
from jax.experimental.ode import odeint 
import optax
import matplotlib
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController

jax.config.update('jax_platform_name', 'cpu')

# Moon lander performs one up thrust followed by one right thrust followed by one up thrust
# Let's add an obstacles
# Let's use diffrax to solve this problem

def func1(traj0, t, t1, t2, t3, t4, t5, t6, t7, y0, vy0, z0, vz0):
    # Start upward thrust
    traj1 = ode_solve(func, traj0, t)
    return traj1 

def func2(traj0, t, t1, t2, t3, t4, t5, t6, t7, y0, vy0, z0, vz0):
    # Start upward thrust
    traj1 = ode_solve(func, traj0, t1)
    traj1 = traj1.at[2,0].set(0.1)
    # Stop upward thrust
    traj2 = ode_solve(func, traj1, t-t1)
    return traj2

def func3(traj0, t, t1, t2, t3, t4, t5, t6, t7, y0, vy0, z0, vz0):
    # Start upward thrust
    traj1 = ode_solve(func, traj0, t1)
    traj1 = traj1.at[2,0].set(0.1)
    # Stop upward thrust
    traj2 = ode_solve(func, traj1, t2)
    traj2 = traj2.at[2,0].set(-0.1)
    # Start rightward thrust
    traj3 = ode_solve(func, traj2, t-(t2+t1))
    return traj3

def func4(traj0, t, t1, t2, t3, t4, t5, t6, t7, y0, vy0, z0, vz0):
    # Start upward thrust
    traj1 = ode_solve(func, traj0, t1)
    traj1 = traj1.at[2,0].set(0.1)
    # Stop upward thrust
    traj2 = ode_solve(func, traj1, t2)
    traj2 = traj2.at[2,0].set(-0.1)
    # Start rightward thrust
    traj3 = ode_solve(func, traj2, t3)
    traj3 = traj3.at[5,0].set(-2.0)
    # Stop rightward thrust
    traj4 = ode_solve(func, traj3, t-(t3+t2+t1))
    return traj4 

def func5(traj0, t, t1, t2, t3, t4, t5, t6, t7, y0, vy0, z0, vz0):
    # Start upward thrust
    traj1 = ode_solve(func, traj0, t1)
    traj1 = traj1.at[2,0].set(0.1)
    # Stop upward thrust
    traj2 = ode_solve(func, traj1, t2)
    traj2 = traj2.at[2,0].set(-0.1)
    # Start rightward thrust
    traj3 = ode_solve(func, traj2, t3)
    traj3 = traj3.at[5,0].set(-2.0)
    # Stop rightward thrust
    traj4 = ode_solve(func, traj3, t4)
    traj4 = traj4.at[5,0].set(0)
    # Start upward thrust
    traj5 = ode_solve(func, traj4, t-(t4+t3+t2+t1))
    return traj5

def func6(traj0, t, t1, t2, t3, t4, t5, t6, t7, y0, vy0, z0, vz0):
    # Start upward thrust
    traj1 = ode_solve(func, traj0, t1)
    traj1 = traj1.at[2,0].set(0.1)
    # Stop upward thrust
    traj2 = ode_solve(func, traj1, t2)
    traj2 = traj2.at[2,0].set(-0.1)
    # Start rightward thrust
    traj3 = ode_solve(func, traj2, t3)
    traj3 = traj3.at[5,0].set(-2.0)
    # Stop rightward thrust
    traj4 = ode_solve(func, traj3, t4)
    traj4 = traj4.at[5,0].set(0)
    # Start upward thrust
    traj5 = ode_solve(func, traj4, t5)
    traj5 = traj5.at[2,0].set(0.1)
    # Stop upward thrust
    traj6 = ode_solve(func, traj5, t-(t5+t4+t3+t2+t1))
    return traj6

def false1(traj0, t, t1, t2, t3, t4, t5, t6, t7, y0, vy0, z0, vz0):
    return jax.lax.cond(t<t2+t1, func2, false2, traj0, t, t1, t2, t3, t4, t5, t6, t7, y0, vy0, z0, vz0)

def false2(traj0, t, t1, t2, t3, t4, t5, t6, t7, y0, vy0, z0, vz0):
    return jax.lax.cond(t<t3+t2+t1, func3, false3, traj0, t, t1, t2, t3, t4, t5, t6, t7, y0, vy0, z0, vz0)

def false3(traj0, t, t1, t2, t3, t4, t5, t6, t7, y0, vy0, z0, vz0):
    return jax.lax.cond(t<t4+t3+t2+t1, func4, false4, traj0, t, t1, t2, t3, t4, t5, t6, t7, y0, vy0, z0, vz0)

def false4(traj0, t, t1, t2, t3, t4, t5, t6, t7, y0, vy0, z0, vz0):
    return jax.lax.cond(t<t5+t4+t3+t2+t1, func5, false5, traj0, t, t1, t2, t3, t4, t5, t6, t7, y0, vy0, z0, vz0)

def false5(traj0, t, t1, t2, t3, t4, t5, t6, t7, y0, vy0, z0, vz0):
    return jax.lax.cond(t<t6+t5+t4+t3+t2+t1, func6, false6, traj0, t, t1, t2, t3, t4, t5, t6, t7, y0, vy0, z0, vz0)

def false6(traj0, t, t1, t2, t3, t4, t5, t6, t7, y0, vy0, z0, vz0):
    # Start upward thrust
    traj1 = ode_solve(func, traj0, t1)
    traj1 = traj1.at[2,0].set(0.1)
    # Stop upward thrust
    traj2 = ode_solve(func, traj1, t2)
    traj2 = traj2.at[2,0].set(-0.1)
    # Start rightward thrust
    traj3 = ode_solve(func, traj2, t3)
    traj3 = traj3.at[5,0].set(-2.0)
    # Stop rightward thrust
    traj4 = ode_solve(func, traj3, t4)
    traj4 = traj4.at[5,0].set(0)
    # Start upward thrust
    traj5 = ode_solve(func, traj4, t5)
    traj5 = traj5.at[2,0].set(0.1)
    # Stop upward thrust
    traj6 = ode_solve(func, traj5, t6)
    traj6 = traj6.at[2,0].set(-0.1)
    traj7 = ode_solve(func, traj6, t-(t6+t5+t4+t3+t2+t1))
    return traj7

def ode_trajectory(func, y0, t1, sample_num = 100):
    term = ODETerm(func)
    solver = Dopri5()
    sample_array = jnp.linspace(0,t1, sample_num)
    saveat = SaveAt(ts = sample_array)
    
    dt = jax.lax.cond(t1>0, lambda arg: 0.1, lambda arg:-0.1, ())

    sol = diffeqsolve(term, solver, t0=0, t1=t1, dt0=dt, y0=y0, saveat=saveat)

    return sol.ys

def system_trajectory(t1, t2, t3, t4, t5, t6, t7, y0, vy0, z0, vz0):
    # A = jnp.array([[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,0,0,0],[0,0,0,0,1,0],[0,0,0,0,-0.01,1],[0,0,0,0,0,0]])
    traj = None 
    traj10 = jnp.array([[z0],[vz0],[-0.1],[y0],[vy0],[0]])
    traj1 = ode_trajectory(func, traj10, t1)
    traj = traj1
    
    traj20 = traj1[-1,:,:]
    traj20 = traj20.at[2,0].set(0.1)
    traj2 = ode_trajectory(func, traj20, t2)
    traj = jnp.concatenate((traj, traj2), axis = 0)
    
    traj30 = traj2[-1,:,:]
    traj30 = traj30.at[2,0].set(-0.1)
    traj3 = ode_trajectory(func, traj30, t3)
    traj = jnp.concatenate((traj, traj3), axis = 0)

    traj40 = traj3[-1,:,:]
    traj40 = traj40.at[5,0].set(-2.0)
    traj4 = ode_trajectory(func, traj40, t4)
    traj = jnp.concatenate((traj, traj4), axis = 0)

    traj50 = traj4[-1,:,:]
    traj50 = traj50.at[5,0].set(0)
    traj5 = ode_trajectory(func, traj50, t5)
    traj = jnp.concatenate((traj, traj5), axis = 0)
    
    traj60 = traj5[-1,:,:]
    traj60 = traj60.at[2,0].set(0.1)
    traj6 = ode_trajectory(func, traj60, t6)
    traj = jnp.concatenate((traj, traj6), axis = 0)
    
    traj70 = traj6[-1,:,:]
    traj70 = traj70.at[2,0].set(-0.1)
    traj7 = ode_trajectory(func, traj70, t7)
    traj = jnp.concatenate((traj, traj7), axis = 0)
    
    return traj

def system_trajectory_nojit(t, t1, t2, t3, t4, t5, t6, t7, y0, vy0, z0, vz0):
    traj0 = jnp.array([[z0],[vz0],[-0.1],[y0],[vy0],[0]])
    
def func(t,x, args):
    A = jnp.array([[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,0,0,0],[0,0,0,0,1,0],[0,0,0,0,-0.01,1],[0,0,0,0,0,0]])
    x_dot = A@x 
    return x_dot

def ode_solve(func, y0, t1):
    term = ODETerm(func)
    solver = Dopri5()
    saveat = SaveAt(ts = [t1])
    
    # if t1>0:
    #     dt = 0.1 
    # else:
    #     dt = -0.1 

    dt = jax.lax.cond(t1>0, lambda arg: 0.1, lambda arg:-0.1, ())

    sol = diffeqsolve(term, solver, t0=0, t1=t1, dt0=dt, y0=y0, saveat=saveat)

    return sol.ys[0]

def system_solution(t1, t2, t3, t4, t5, t6, t7, y0, vy0, z0, vz0):
    A = jnp.array([[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,0,0,0],[0,0,0,0,1,0],[0,0,0,0,-0.01,1],[0,0,0,0,0,0]])
    traj0 = jnp.array([[z0],[vz0],[-0.1],[y0],[vy0],[0]])
    # Start upward thrust
    traj1 = ode_solve(func, traj0, t1)
    traj1 = traj1.at[2,0].set(0.1)
    # Stop upward thrust
    traj2 = ode_solve(func, traj1, t2)
    traj2 = traj2.at[2,0].set(-0.1)
    # Start rightward thrust
    traj3 = ode_solve(func, traj2, t3)
    traj3 = traj3.at[5,0].set(-2.0)
    # Stop rightward thrust
    traj4 = ode_solve(func, traj3, t4)
    traj4 = traj4.at[5,0].set(0)
    # Start upward thrust
    traj5 = ode_solve(func, traj4, t5)
    traj5 = traj5.at[2,0].set(0.1)
    # Stop upward thrust
    traj6 = ode_solve(func, traj5, t6)
    traj6 = traj6.at[2,0].set(-0.1)
    traj7 = ode_solve(func, traj6, t7)
    return traj7

def loss_func(params, y0, vy0, z0, vz0, y_targ, z_targ):
    theta_t1 = params[0]
    theta_t2 = params[1]
    theta_t3 = params[2]
    theta_t4 = params[3]
    theta_t5 = params[4]
    theta_t6 = params[5]
    theta_t7 = params[6]
    loss = loss_func_solution(theta_t1, theta_t2, theta_t3, theta_t4, theta_t5, theta_t6, theta_t7, y0, vy0, z0, vz0, y_targ, z_targ)
    return loss

def body_fun(i, val):
    steps, theta_t1, theta_t2, theta_t3, theta_t4, theta_t5, theta_t6, theta_t7, y0, vy0, z0, vz0, unsafe_loss = val
    step = steps[i]
    traj_state = system_trajectory(step, theta_t1,theta_t2,theta_t3,theta_t4,theta_t5,theta_t6,theta_t7,y0,vy0,z0,vz0)
    traj_z = traj_state[0,0]
    traj_vz = traj_state[1,0]
    traj_az = traj_state[2,0]
    traj_y = traj_state[3,0]
    traj_vy = traj_state[4,0]
    traj_ay = traj_state[5,0]
    unsafe_loss += jax.nn.relu(2000-traj_y)*jax.nn.relu(traj_y-1500)*jax.nn.relu(800-traj_z)
    return (steps, theta_t1, theta_t2, theta_t3, theta_t4, theta_t5, theta_t6, theta_t7, y0, vy0, z0, vz0, unsafe_loss)

def loss_func_solution(theta_t1, theta_t2, theta_t3, theta_t4, theta_t5, theta_t6, theta_t7, y0, vy0, z0, vz0, y_targ, z_targ):
    traj = system_trajectory(
        theta_t1, theta_t2, theta_t3, theta_t4, theta_t5, theta_t6, theta_t7, y0, vy0, z0, vz0
    )
    end_z = traj[-1,0,0]
    end_vz = traj[-1,1,0]
    end_az = traj[-1,2,0]
    end_y = traj[-1,3,0]
    end_vy = traj[-1,4,0]
    end_ay = traj[-1,5,0]

    traj_y = traj[:,3,0]
    traj_z = traj[:,0,0]
    unsafe_loss = jnp.sum(
        jax.nn.relu(2000-traj_y)*
        jax.nn.relu(traj_y-1500)*
        jax.nn.relu(800-traj_z)
    ) + jnp.sum(jax.nn.relu(-traj_z))

    # steps = jnp.linspace(0.1,theta_t1+theta_t2+theta_t3+theta_t4+theta_t5+theta_t6+theta_t7,1000)
    # init_val = (steps, theta_t1, theta_t2, theta_t3, theta_t4, theta_t5, theta_t6, theta_t7, y0, vy0, z0, vz0, 0)
    # val = init_val 
    # for i in range(0,1000):
    #     val = body_fun(i,val)
    # _, _, _, _, _, _, _, _, _, _, _, unsafe_loss = val 
    # _, _, _, _, _, _, _, _, _, _, _, _, unsafe_loss = jax.lax.fori_loop(0,1000,body_fun,init_val)
    
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
    above_0_loss = jax.nn.relu(-theta_t1)+\
        jax.nn.relu(-theta_t2)+\
        jax.nn.relu(-theta_t3)+\
        jax.nn.relu(-theta_t4)+\
        jax.nn.relu(-theta_t5)+\
        jax.nn.relu(-theta_t6)+\
        jax.nn.relu(-theta_t7)
    optimize_goal = theta_t1+theta_t2+theta_t3+theta_t4+theta_t5+theta_t6

    loss = 5*(pos_loss + vel_loss*500 + above_0_loss*10000000 + unsafe_loss*1000)+optimize_goal
    return loss 

def test_res(t1, t2, t3 ,t4, t5, t6, t7, y0, vy0, z0, vz0):
    res_y = []
    res_z = []
    res_vy = []
    res_vz = []
    res_ay = []
    res_az = []
    tmp = jit(system_trajectory)
    res1 = system_solution(t1,t2,t3,t4,t5,t6,t7,y0,vy0,z0,vz0)
    # res2 = system_trajectory(t1,t2,t3,t4,t5,t6,t7,y0,vy0,z0,vz0)
    # res3 = tmp(t1,t2,t3,t4,t5,t6,t7,y0,vy0,z0,vz0)
    print(res1)
    # print(res2)
    # print(res3)
    res = system_trajectory(t1,t2,t3,t4,t5,t6,t7,y0,vy0,z0,vz0)
    res_z = res[:,0,0]
    res_vz = res[:,1,0]
    res_az = res[:,2,0]
    res_y = res[:,3,0]
    res_vy = res[:,4,0]
    res_ay = res[:,5,0]
    print(system_solution(t1,t2,t3,t4,t5,t6,t7,y0,vy0,z0,vz0))
    # print(tmp(t,t1,t2,t3,t4,t5,t6,t7,y0,vy0,z0,vz0))
    plt.figure()
    plt.plot([1500,2000,2000,1500,1500],[0,0,800,800,0])
    plt.plot(res_y, res_z)
    plt.show()

# if __name__ == "__main__":
#     y0 = 0
#     vy0 = 82
#     z0 = 800
#     vz0 = 0

#     # t = 222.32358 
#     # t1 = 5.0115414 
#     # t2 = 23.40875 
#     # t3 = 61.775997 
#     # t4 = 83.10098 
#     # t5 = 127.126884 
#     # t6 = 219.89009 

#     # t1 = 5.0115414 
#     # t2 = 18.3972086
#     # t3 = 38.367247 
#     # t4 = 21.324983 
#     # t5 = 44.025904 
#     # t6 = 92.763206 
#     # t7 = 2.43349
#     # t = t1+t2+t3+t4+t5+t6+t7
    
#     t1 = 5.1489 
#     t2 = 33.589314 
#     t3 = 17.315825 
#     t4 = 21.003927 
#     t5 = 83.62109 
#     t6 = 95.915016 
#     t7 = 3.154463

#     test_res(t1, t2, t3 ,t4, t5, t6, t7, y0, vy0, z0, vz0)

if __name__ == "__main__": 
    schedule = optax.linear_schedule(1.0, 0.1, 0.001, 20000)
    start_learning_rate = 5.0
    optimizer = optax.adam(schedule)
#     t1 = 5.081437 
#     t2 = 16.966002 
#     t3 = 56.311783 
#     t4 = 77.31678 
#     t5 = 112.204636 
#     t6 = 201.269

    # Initialize parameters of the model + optimizer.
    params = jnp.array([1.0 ,100.0 ,10.0 ,10.0 ,10.0 ,10.0 ,60.0])
    opt_state = optimizer.init(params)
    
    y0 = 0
    vy0 = 82
    z0 = 800
    vz0 = 0
    y_targ = 4000
    z_targ = 0
    
    res = loss_func(params, y0, vy0, z0, vz0, y_targ, z_targ)
    print(res)
    res_grad = jax.grad(loss_func)(params, y0, vy0, z0, vz0, y_targ, z_targ)
    print(res_grad)

    tmp_loss = jit(loss_func) 
    tmp_grad = jit(jax.grad(loss_func))
    for i in range(40000):
        print(i, tmp_loss(params, y0, vy0, z0, vz0, y_targ, z_targ), jnp.round(params,2))
        grads = tmp_grad(params, y0, vy0, z0, vz0, y_targ, z_targ)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

    theta_t1 = params[0]
    theta_t2 = params[1]
    theta_t3 = params[2]
    theta_t4 = params[3]
    theta_t5 = params[4]
    theta_t6 = params[5]
    theta_t7 = params[6]
    print(theta_t1, theta_t2, theta_t3, theta_t4, theta_t5, theta_t6, theta_t7, system_solution(theta_t1, theta_t2, theta_t3, theta_t4, theta_t5, theta_t6, theta_t7, y0,vy0,z0,vz0))

    test_res(theta_t1, theta_t2, theta_t3 ,theta_t4, theta_t5, theta_t6, theta_t7, y0, vy0, z0, vz0)
