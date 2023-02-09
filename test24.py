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

jax.config.update('jax_platform_name', 'cpu')

NUM_TRANS = 15

# Moon lander performs one up thrust followed by one right thrust followed by one left thrust
# N, UP, N, LEFT, N, RIGHT 
# Perform this sequence of six modes 15 times 
# Let's add an obstacles
# Let's use diffrax to solve this problem

def func(t,x, args):
    A = jnp.array([[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,0,0,0],[0,0,0,0,1,0],[0,0,0,0,-0.01,1],[0,0,0,0,0,0]])
    x_dot = A@x 
    return x_dot

def ode_trajectory(func, y0, t1, sample_num = 100):
    term = ODETerm(func)
    solver = Dopri5()
    sample_array = jnp.linspace(0,t1, sample_num)
    saveat = SaveAt(ts = sample_array)
    dt = jax.lax.cond(t1>0, lambda arg: 0.1, lambda arg:-0.1, ())
    sol = diffeqsolve(term, solver, t0=0, t1=t1, dt0=dt, y0=y0, saveat=saveat)
    return sol.ys

def body_func(i, val):
    t_list, traj10, traj = val
    t1 = t_list[i*6]
    t2 = t_list[i*6+1]
    t3 = t_list[i*6+2]
    t4 = t_list[i*6+3]
    t5 = t_list[i*6+4]
    t6 = t_list[i*6+5]

    traj1 = ode_trajectory(func, traj10, t1)
    traj = traj.at[i,0,:,:,:].set(traj1)
    
    traj20 = traj1[-1,:,:]
    traj20 = traj20.at[2,0].set(0.5)
    traj2 = ode_trajectory(func, traj20, t2)
    traj = traj.at[i,1,:,:,:].set(traj2)
    
    traj30 = traj2[-1,:,:]
    traj30 = traj30.at[2,0].set(-0.1)
    traj3 = ode_trajectory(func, traj30, t3)
    traj = traj.at[i,2,:,:,:].set(traj3)
    
    traj40 = traj3[-1,:,:]
    traj40 = traj40.at[5,0].set(-2.0)
    traj4 = ode_trajectory(func, traj40, t4)
    traj = traj.at[i,3,:,:,:].set(traj4)

    traj50 = traj4[-1,:,:]
    traj50 = traj50.at[5,0].set(0)
    traj5 = ode_trajectory(func, traj50, t5)
    traj = traj.at[i,4,:,:,:].set(traj5)
    
    traj60 = traj5[-1,:,:]
    traj60 = traj60.at[5,0].set(2.0)
    traj6 = ode_trajectory(func, traj60, t6)
    traj = traj.at[i,5,:,:,:].set(traj6)

    return t_list, traj[i,-1,-1,:,:], traj


def system_trajectory(params, y0, vy0, z0, vz0):
    traj0 = jnp.array([[z0],[vz0],[-0.1],[y0],[vy0],[0]])
    traj = jnp.zeros((NUM_TRANS,6,100,6,1))

    # _, _, traj = jax.lax.fori_loop(0,NUM_TRANS,body_func,(params, traj0, traj))
    val = (params, traj0, traj)
    for i in range(0, NUM_TRANS):
        val = body_func(i, val)
    # return val
    _,_,traj = val
    traj = jnp.reshape(traj, (NUM_TRANS*6*100,6,1))

    return traj

# def loss_func(params, y0, vy0, z0, vz0, y_targ, z_targ):
#     theta_t1 = params[0]
#     theta_t2 = params[1]
#     theta_t3 = params[2]
#     theta_t4 = params[3]
#     theta_t5 = params[4]
#     theta_t6 = params[5]
#     theta_t7 = params[6]
#     loss = loss_func_solution(theta_t1, theta_t2, theta_t3, theta_t4, theta_t5, theta_t6, theta_t7, y0, vy0, z0, vz0, y_targ, z_targ)
#     return loss

def loss_func(params, y0, vy0, z0, vz0, y_targ, z_targ):
    traj = system_trajectory(
        params, y0, vy0, z0, vz0
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
        jax.nn.relu(850-traj_z)
    ) + jnp.sum(jax.nn.relu(-traj_z))

    pos_loss = (z_targ - end_z)**2 + (y_targ - end_y)**2 
    vel_loss = jax.nn.relu(-2-end_vz)**2 + end_vy**2
    above_0_loss = jnp.sum(jax.nn.relu(-params))
    optimize_goal = jnp.sum(params)

    loss = 5*(pos_loss + vel_loss*500 + above_0_loss*10000000 + unsafe_loss*1000)
    return loss 

def test_res(params, y0, vy0, z0, vz0):
    res_y = []
    res_z = []
    res_vy = []
    res_vz = []
    res_ay = []
    res_az = []
    tmp = jit(system_trajectory)
    res = system_trajectory(final_res,y0,vy0,z0,vz0)
    print(res[-1,:,:])
    res_z = res[:,0,0]
    res_vz = res[:,1,0]
    res_az = res[:,2,0]
    res_y = res[:,3,0]
    res_vy = res[:,4,0]
    res_ay = res[:,5,0]
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
    schedule = optax.linear_schedule(0.1, 0.01, 0.0001, 20000)
    start_learning_rate = 5.0
    optimizer = optax.adam(schedule)

    # Initialize parameters of the model + optimizer.
    params = jnp.array([10.0 for i in range(NUM_TRANS*6)])
    opt_state = optimizer.init(params)
    
    y0 = 0
    vy0 = 82
    z0 = 800
    vz0 = 0
    y_targ = 4000
    z_targ = 0

    best_loss = float('inf')
    final_res = None
    
    res = loss_func(params, y0, vy0, z0, vz0, y_targ, z_targ)
    print(res)
    res_grad = jax.grad(loss_func)(params, y0, vy0, z0, vz0, y_targ, z_targ)
    print(res_grad)

    tmp_loss = jit(loss_func) 
    tmp_grad = jit(jax.grad(loss_func))
    for i in range(40000):
        val = tmp_loss(params, y0, vy0, z0, vz0, y_targ, z_targ)
        print(i, val)
        if val < best_loss:
            best_loss = val 
            final_res = copy.deepcopy(params) 
        
        grads = tmp_grad(params, y0, vy0, z0, vz0, y_targ, z_targ)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

    # theta_t1 = final_res[0]
    # theta_t2 = final_res[1]
    # theta_t3 = final_res[2]
    # theta_t4 = final_res[3]
    # theta_t5 = final_res[4]
    # theta_t6 = final_res[5]
    # theta_t7 = final_res[6]
    print(params)
    print(tmp_loss(final_res, y0, vy0, z0, vz0, y_targ, z_targ), system_trajectory(final_res, y0,vy0,z0,vz0)[-1,:,:])

    test_res(final_res, y0, vy0, z0, vz0)
