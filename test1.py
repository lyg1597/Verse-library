import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
from jax import jacfwd, jacrev, jit
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

dt = 0.1

def vehicleDynamics(state, u):
    a = u
    t,x,v = state

    t_next = t + dt*1 
    x_next = x + dt*v 
    v_next = v + dt*a 

    return jnp.array([t_next, x_next, v_next])

def decision_logic(state, theta):
    t,x,v = state 
    theta_a1, theta_a2 = theta
    if t > 40:
        a = 0 
    elif t<=40 and t>20:
        a = theta_a2
    elif t<=20 and t>10:
        a = 0 
    else:
        a = theta_a1
    return a 

def compute_trajectory(inp):
    t,x,v, theta_a1, theta_a2, t_max = inp
    # number_points = 5000
    # time_point = [round(i*dt, 10) for i in range(number_points)]
    theta = jnp.array([theta_a1, theta_a2])
    init_state = jnp.array([t,x,v])

    state = init_state

    # trajectory = [state]
    for _ in range(500):
        u = decision_logic(state, theta)
        new_state = jit(vehicleDynamics)(state, u)
        state = new_state
        # trajectory.append(state)
    # return trajectory
    return state

def loss_func(t,x,v,theta_a1, theta_a2,t_max,ref_pos):
    inp = [t,x,v,theta_a1, theta_a2,t_max]
    end_state = compute_trajectory(inp)
    end_pos = end_state[1]
    end_vel = end_state[2]
    pos_diff = (end_pos-ref_pos)**2
    vel_diff = (end_vel-0)**2
    loss = pos_diff + vel_diff*1000 
    # print(jax.numpy.asarray(loss))
    return loss

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

    learning_rate = 0.000001

    theta_a1 = 2.0
    theta_a2 = -1.0
    theta_t = 30.0
    ref_pos = 250.0

    theta_a = -0.8003115
    
    # res = loss_func(t,x,v,theta_a1, theta_a2, t_max, ref_pos)
    # print(res)

    # trajectory = jnp.array(compute_trajectory([t,x,v,theta_a1, theta_a2, t_max]))
    # plt.plot(trajectory[:,0], trajectory[:,1])
    # plt.plot(trajectory[:,0], trajectory[:,2])
    # plt.show()

    grad_loss = jax.grad(loss_func, argnums=(3,4))
    # grad_loss = jit(grad_loss)
    for i in range(500):
        loss_val = loss_func(t,x,v,theta_a1,theta_a2,t_max,ref_pos)
        grad_a1, grad_a2 = grad_loss(t,x,v,theta_a1, theta_a2, t_max,ref_pos)
        print(i, theta_a1, theta_a2, loss_val, grad_a1, grad_a2)
        theta_a1 -= learning_rate * grad_a1
        theta_a2 -= learning_rate * grad_a2

    print(theta_a1, theta_a2)