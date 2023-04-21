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
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import ode 
jax.config.update('jax_platform_name', 'cpu')

num_params = 4
Q = np.eye(3)

def end_state(params, x0, A_list):
    xt = x0
    for i in range(num_params):
        xt = jscipy.linalg.expm(A_list[i]*params[i])@xt
    return xt 

def cost(params, x0, A_list):
    xt = end_state(params, x0, A_list)
    return (xt.T@Q@xt)[0,0]

A0 = np.array([[-3.01543266, -3.79855661,  1.76747326],
       [-0.24781885,  0.94346251, -4.79410501],
       [-4.05235718,  1.73291361,  4.75043987]])
A1 = np.array([[ 1.29285307,  4.99714053, -3.54401244],
       [ 0.01084965, -3.78005681,  3.23872005],
       [ 0.91658178,  1.25065622, -0.192326  ]])
A2 = np.array([[-1.6151233 ,  4.62729827, -4.28605077],
       [ 0.73316211,  0.34576438, -3.50565487],
       [-3.86152252,  0.33095721,  0.57806628]])
A3 = np.array([[-3.69068301,  0.21493028,  1.19091545],
       [ 4.80865679,  4.20318553,  4.54577808],
       [ 4.84704884,  0.93460586, -2.51891471]])

if __name__ == "__main__":
    lr = 0.00000001
    A_list = []
    best_loss = 1000000
    A_list = [A0, A1, A2, A3]

    # x0 = np.random.uniform(0,1,(3,1))
    x0 = np.array([[-1.33504336],
       [ 2.19668654],
       [ 9.12047933]])

    # params = np.zeros(num_params)
    params = np.random.uniform(0,1,num_params)
    params_0 = copy.deepcopy(params)
    # params[1] = 0.09
    # params = np.array([0., 0.11908431, 0.14672016, 0.4896208])

    terminal_state = end_state(params, x0, A_list)

    res = cost(params, x0, A_list)

    res_grad = jax.grad(cost)(params, x0, A_list)

    
    res_grad = jax.grad(cost)(params, x0, A_list)

    tmp_loss = jit(cost) 
    tmp_grad = jit(jax.grad(cost))
    for i in range(1000000):
        val = tmp_loss(params, x0, A_list)
        # if i%1000==0:
        print(i, val, params)
        if val < best_loss:
            best_loss = val 
            final_res = copy.deepcopy(params) 
            res_grad = tmp_grad(params, x0, A_list)
            # print(res_grad)
    
        grads = tmp_grad(params, x0, A_list)
        params = params - lr*grads
        if params[0]<0:
            params = params.at[0].set(0)
        elif params[0]>1:
            params = params.at[0].set(1)
        if params[1]<0:
            params = params.at[1].set(0)
        elif params[1]>1:
            params = params.at[1].set(1)
        if params[2]<0:
            params = params.at[2].set(0)
        elif params[2]>1:
            params = params.at[2].set(1)
        if params[3]<0:
            params = params.at[3].set(0)
        elif params[3]>1:
            params = params.at[3].set(1)

    print("###################################")
    print(params_0)
    print(tmp_loss(params, x0, A_list), params)
    terminal_state = end_state(params, x0, A_list)
    print(terminal_state)
