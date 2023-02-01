from jax.experimental.ode import odeint 
import jax.numpy as jnp 
import jax

def func(x,t):
    p,v = x 
    p_dot = v 
    v_dot = 2.0
    return jnp.array([p_dot, v_dot])


def compute_int(p0, v0, T):
    t = jnp.linspace(0,T,1000)

    traj = odeint(func, jnp.array([p0,v0]), jnp.array([0,T]))

    return traj[-1,0]

tmp = compute_int(0.0,0.0,-2.0)
print(tmp)

res = jax.vmap(compute_int)(jnp.zeros(10),jnp.zeros(10),jnp.zeros(10)+2.0)

print(res.shape)

print(jax.grad(compute_int, argnums=1)(0.0,0.0,2.0))
print(type(compute_int(0.0,0.0,2.0)))