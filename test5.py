import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
from jax import jacfwd, jacrev, jit
import matplotlib.pyplot as plt

dt = 0.1

def vehicleDynamics(state, u):
    a = u
    t,x,v = state

    t_next = t + dt*1 
    x_next = x + dt*v 
    v_next = v + dt*a 

    return jnp.array([t_next, x_next, v_next])

def decision_logic_smooth(state, theta):
    t,x,v = state 
    theta_t1, = theta
    sig1 = 0
    sig2 = 0
    sig3 = 0
    sig1 = -1/(jnp.exp(-1*(t-10))+1)+1
    sig2 = (-0.5)/(jnp.exp(-1*(t-theta_t1))+1)
    sig3 = -(-0.5)/(jnp.exp(-1*(t-40))+1)
    return sig1+sig2+sig3

def decision_logic(state, theta):
    t,x,v = state 
    theta_t1, = theta
    if t > 40:
        a = 0 
    elif t<=40 and t>theta_t1:
        a = -0.5
    elif t<=theta_t1 and t>10:
        a = 0 
    else:
        a = 1
    return a 

def compute_trajectory_smooth(inp):
    t,x,v, theta_t1, t_max = inp
    # number_points = 5000
    # time_point = [round(i*dt, 10) for i in range(number_points)]
    theta = [theta_t1]
    init_state = jnp.array([t,x,v])

    state = init_state

    trajectory = [state]
    for _ in range(500):
        u = decision_logic_smooth(state, theta)
        new_state = jit(vehicleDynamics)(state, u)
        state = new_state
        trajectory.append(state)
    return trajectory
    # return state

def compute_trajectory(inp):
    t,x,v, theta_t1, t_max = inp
    # number_points = 5000
    # time_point = [round(i*dt, 10) for i in range(number_points)]
    theta = [theta_t1]
    init_state = jnp.array([t,x,v])

    state = init_state

    trajectory = [state]
    for _ in range(500):
        u = decision_logic(state, theta)
        new_state = jit(vehicleDynamics)(state, u)
        state = new_state
        trajectory.append(state)
    return trajectory
    # return state

def compute_endstate(inp):
    t,x,v, theta_t1, t_max = inp
    # number_points = 5000
    # time_point = [round(i*dt, 10) for i in range(number_points)]
    theta = [theta_t1]
    init_state = jnp.array([t,x,v])

    state = init_state

    trajectory = [state]
    for _ in range(500):
        u = decision_logic(state, theta)
        new_state = jit(vehicleDynamics)(state, u)
        state = new_state
    #     trajectory.append(state)
    # return trajectory
    return state

def compute_endstate_smooth(inp):
    t,x,v, theta_t1, t_max = inp
    # number_points = 5000
    # time_point = [round(i*dt, 10) for i in range(number_points)]
    theta = [theta_t1]
    init_state = jnp.array([t,x,v])

    state = init_state

    trajectory = [state]
    for _ in range(500):
        u = decision_logic_smooth(state, theta)
        new_state = jit(vehicleDynamics)(state, u)
        state = new_state
    #     trajectory.append(state)
    # return trajectory
    return state

def loss_func(t,x,v,theta_t1,t_max,ref_pos):
    inp = [t,x,v,theta_t1,t_max]
    end_state = compute_endstate(inp)
    end_pos = end_state[1]
    end_vel = end_state[2]
    pos_diff = (end_pos-ref_pos)**2
    vel_diff = (end_vel-0)**2
    loss = pos_diff + vel_diff*1000 
    # print(jax.numpy.asarray(loss))
    return loss

def loss_func_smooth(t,x,v,theta_t1,t_max,ref_pos):
    inp = [t,x,v,theta_t1,t_max]
    end_state = compute_endstate_smooth(inp)
    end_pos = end_state[1]
    end_vel = end_state[2]
    pos_diff = (end_pos-ref_pos)**2
    vel_diff = (end_vel-0)**2
    loss = pos_diff + vel_diff*1000 
    # print(jax.numpy.asarray(loss))
    return loss

# if __name__ == "__main__":
#     t,x,v = 0.0,0.0,0.0
#     theta_t1 = 20.0
#     t_max = 50.0
#     ref_pos = 250.0
#     trajectory = jnp.array(compute_trajectory([t,x,v, theta_t1, t_max]))
#     plt.plot(trajectory[:,0], trajectory[:,1])
#     trajectory_smooth = jnp.array(compute_trajectory_smooth([t,x,v, theta_t1, t_max]))
#     plt.plot(trajectory_smooth[:,0], trajectory_smooth[:,1])

#     import numpy as np 
#     plt.figure()
#     steps = np.linspace(0,50,100)
#     res = []
#     for step in steps:
#         tmp = decision_logic([step,0,0],[20])
#         res.append(tmp)
#     plt.plot(steps, res)    
#     res = []
#     for step in steps:
#         tmp = decision_logic_smooth([step,0,0],[20])
#         res.append(tmp)
#     plt.plot(steps, res)

#     plt.figure()
#     steps = np.linspace(10, 40, 300)
#     res = []
#     for step in steps:
#         tmp = loss_func(t,x,v,step,t_max,ref_pos)
#         print(tmp)
#         res.append(tmp)
#     plt.plot(steps, res)    
#     res = []
#     for step in steps:
#         tmp = loss_func_smooth(t,x,v,step,t_max,ref_pos)
#         res.append(tmp)
#     plt.plot(steps, res)    
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
    t_max = 50.0

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
    theta_t1 = 15.0
    theta_t2 = 15.0
    ref_pos = 250.0

    theta_a = -0.8003115
    
    # res = loss_func(t,x,v,theta_a1, theta_a2, t_max, ref_pos)
    # print(res)

    # trajectory = jnp.array(compute_trajectory([t,x,v,theta_a1, theta_a2, t_max]))
    # plt.plot(trajectory[:,0], trajectory[:,1])
    # plt.plot(trajectory[:,0], trajectory[:,2])
    # plt.show()

    grad_loss = jax.grad(loss_func_smooth, argnums=(3,))
    # grad_loss = jit(grad_loss)
    for i in range(500):
        loss_val = loss_func_smooth(t,x,v,theta_t1,t_max,ref_pos)
        grad_t1, = grad_loss(t,x,v,theta_t1,t_max,ref_pos)
        print(i, theta_t1, loss_val, grad_t1)
        theta_t1 -= learning_rate * grad_t1
        # theta_t2 -= learning_rate * grad_t2

    print(theta_t1)