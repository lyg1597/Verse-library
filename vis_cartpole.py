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

# Nonlinear example, cartpole
# Dynamics: http://www.matthewpeterkelly.com/tutorials/cartPole/index.html
'''
    \dot{x} = x_1
    \dot{x_1} = \frac{-gsin(\theta)-l\dot{\theta_1}}{cos(\theta)}
    \dot{\theta} = \theta_1
    \dot{\theta_1} = \frac{Fcos(\theta)+{m_2}l{\theta_1}^2sin(\theta)cos(\theta)+(m_1+m_2)gsin(\theta)}{{m_2}l(cos(\theta)^2)-l(m_1+m_2)}
'''
# F = {-2, 2}
# Perform this sequence of left, right modes ? times 
# Let's use diffrax to solve this problem

def func(t,state, args):
    x, x1, theta, theta1 = state[:,0]
    F = args 
    m1 = 1 
    m2 = 1 
    l = 1 
    g = 9.81
    theta_dot = theta1 
    theta1_dot = (F*jnp.cos(theta-jnp.pi)+m2*l*theta1**2*jnp.sin(theta-jnp.pi)*jnp.cos(theta-jnp.pi)+(m1+m2)*g*jnp.sin(theta-jnp.pi))\
        /(m2*l*jnp.cos(theta-jnp.pi)**2-l*(m1+m2))
    x_dot = x1 
    x1_dot = (-g*jnp.sin(theta-jnp.pi)-theta1_dot)/(jnp.cos(theta-jnp.pi))
    state_dot = jnp.array([[x_dot],[x1_dot],[theta_dot],[theta1_dot]])
    return state_dot

def ode_solve(func, y0, t1, arg=None, sample_num=100):
    term = ODETerm(func)
    solver = Dopri5()
    sample_array = jnp.linspace(0, t1, sample_num)
    saveat = SaveAt(ts = sample_array)
    dt = jax.lax.cond(t1>0, lambda arg: 0.05, lambda arg:-0.05, ())
    sol = diffeqsolve(term, solver, t0=0, t1=t1, dt0=dt, y0=y0, saveat=saveat, args=arg)
    return sol.ys

def body_func(i, val):
    params, traj10, traj = val
    t_list = params[:,0]
    a_list = params[:,1]
    t1 = t_list[i*2]
    t2 = t_list[i*2+1]
    a1 = a_list[i*2]
    a2 = a_list[i*2+1]
    a1 = jnp.clip(a1,-10,10)
    a2 = jnp.clip(a2,-10,10)

    traj1 = ode_solve(func, traj10, t1, arg=a1)
    traj = traj.at[i,0,:,:,:].set(traj1)
    
    traj20 = traj1[-1,:,:]
    traj2 = ode_solve(func, traj20, t2, arg=a2)
    traj = traj.at[i,1,:,:,:].set(traj2)
    
    return params, traj[i,-1,-1,:,:], traj


def system_trajectory(params, x0):
    traj0 = x0
    traj = jnp.zeros((NUM_TRANS,2,100,4,1))

    # _, _, traj = jax.lax.fori_loop(0,NUM_TRANS,body_func,(params, traj0, traj))
    val = (params, traj0, traj)
    for i in range(0, NUM_TRANS):
        val = body_func(i, val)
    # return val
    _,_,traj = val
    traj = jnp.reshape(traj, (NUM_TRANS*2*100,4,1))

    return traj

def loss(params, x0):
    traj = system_trajectory(params, x0)
    tmp = jnp.squeeze(traj)
    state_loss = jnp.sum(jnp.diag(tmp@jnp.array([[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]])@tmp.T))
    control_loss = jnp.sum(params)
    end_state_loss = tmp[-1,:]@jnp.array([[1,0,0,0],[0,0,0,0],[0,0,100,0],[0,0,0,100]])@tmp[-1,:].T
    above_0_loss = jnp.sum(jax.nn.relu(-params))
    loss = above_0_loss*10000000+end_state_loss*100
    return loss

def res_vis(params, x0):
    traj = system_trajectory(params, x0)
    plt.figure()
    for i in range(traj.shape[0]):
        p1 = [traj[i,0,0],0]
        p2 = [jnp.cos(traj[i,2,0]+jnp.pi/2)+p1[0], jnp.sin(traj[i,2,0]+jnp.pi/2)]
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]],'b')
        plt.xlim([-1, 100]) 
        plt.ylim([-1.1,1.1])   
        plt.pause(0.001)
        plt.clf()

if __name__ == "__main__":
# #     params = jnp.array([
# #         1.0071083,  1.0010864,  0.97229886, 0.93412995, 1.1405095,  1.0574526,
# #         1.1286395,  1.1425823,  1.1528772,  1.0958037,  1.0827278,  1.0399377,
# #         0.96794087, 0.83975846, 0.8847757,  0.5441705,  1.3095053,  1.8704956,
# #         1.4023228,  0.01450855])
#     params = jnp.array([
#         1.0347966,  1.0305216,  0.99658614, 0.94158167, 1.1585217,  1.2406237,
#         1.2758323,  1.3449851,  1.4486535,  1.2309204,  0.6687606,  0.6746257,
#         0.6164314,  0.51210374, 0.6183152,  0.46059418, 0.77532303, 1.8253607,
#         1.2668735,  0.05476815
#     ])
    params = jnp.array([
        [1.3834831e+00, 8.5092917e-02],
        [5.0391370e-01, 3.3167667e+00],
        [1.3541088e+00, 6.4527199e-02],
        [9.9541217e-01, 1.7519366e-02],
        [1.4769241e+00, 9.9451743e-02],
        [5.4064494e-01, 3.2144818e+00],
        [1.2481635e+00, 3.4860294e-02],
        [1.3161870e+00, 3.4533363e-02],
        [5.7910049e-01, 3.3814838e+00],
        [1.6115789e+00, 5.1746469e-02],
        [1.3257294e+00, 9.2742871e-03],
        [1.6381348e+00, 5.5335134e-02],
        [6.0565388e-01, 3.6907651e+00],
        [1.5880902e+00, 1.5708646e-02],
        [1.5791780e+00, 1.1889901e-02],
        [7.1006727e-01, 3.9041407e+00],
        [2.2755249e+00, 2.9632872e-02],
        [1.9378947e+00, 1.6236486e-03],
        [1.8230714e+00, 3.7534541e-04],
        [8.6707324e-01, 9.1173859e+00],    
    ])
    x0 = jnp.array([[0],[0],[1.0],[0]])
    res_vis(params, x0)
