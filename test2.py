import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
from jax import jacfwd, jacrev, jit
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

dt = 0.1

def final_pos(t2, t3):
    final_pos = 1/2*1*10**2 + 10*t2 + 10*t3 + 1/2*(-1)*t3**2
    return final_pos

def loss_func(t2, t3,ref_pos):
    # inp = [t,x,v,theta_a1, theta_a2,t_max]
    end_pos = final_pos(t2, t3)
    # end_pos = end_state[1]
    # end_vel = end_state[2]
    pos_diff = (end_pos-ref_pos)**2
    loss = pos_diff + t2 + t3 
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

    learning_rate = 0.005

    theta_a1 = 2.0
    theta_a2 = -1.0
    theta_t = 30.0
    ref_pos = 200.0

    theta_a = -0.8003115

    t2 = 0.0
    t3 = 0.0
    
    # res = loss_func(t,x,v,theta_a1, theta_a2, t_max, ref_pos)
    # print(res)

    # trajectory = jnp.array(compute_trajectory([t,x,v,theta_a1, theta_a2, t_max]))
    # plt.plot(trajectory[:,0], trajectory[:,1])
    # plt.plot(trajectory[:,0], trajectory[:,2])
    # plt.show()

    grad_loss = jax.grad(loss_func, argnums=(0,1))
    # grad_loss = jit(grad_loss)
    for i in range(10000):
        loss_val = loss_func(t2, t3, ref_pos)
        res = final_pos(t2,t3)
        grad_t2, grad_t3 = grad_loss(t2, t3, ref_pos)
        print(i, t2, t3, res, loss_val, grad_t2, grad_t3)
        t2 -= learning_rate * grad_t2
        t3 -= learning_rate * grad_t3

    print(t2, t3)