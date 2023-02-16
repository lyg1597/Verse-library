import numpy as np 
import jax.numpy as jnp
from jax import grad

# Try some small things 
# Consider differential equation \dot{x} = -2x, x0 = 2
# The solution of the differential equation is x(t) = 2e^{-2t}
# Consider mock condition x<c, c\in[0,2]

# Try a small hybrid system
# Consider system with two mode 
#   M1: \dot{x} = x
#   M2: \dot{x} = -x
#   x0 = 1
#   Transition condition M1->M2: x>=c
#   The overall trajectory of system is given by \xi(t,x0;c) = x0e^{t1}e^{-(t-t1)}
#       where t is the total time and t1 is the time when transition happen
#   For the initial setting, set c = 2, and t = 1.5 
#   Then \frac{d\xi(1.5,1;c)}{dc} can be estimated using 
#       \lim_{\delta c->0}\frac{\xi(1.5,t;c+\delta c)-\xi(1.5,t;c)}{\delta c}

def compute_xi(t, x0, c):
    assert c>=1
    t1 = jnp.log(c)
    assert t>t1
    xi = x0*jnp.exp(t1)*jnp.exp(-(t-t1))
    return xi

def ana_diff(t,x0,c):
    t1 = np.log(c)
    res = -(-compute_xi(t,x0,c))/c + np.exp(-(t-t1))
    return res

if __name__ == "__main__":
    c = 3.0
    print(compute_xi(2,1,c))
    dc = 0.0000001
    print(compute_xi(2,1,c+dc))
    diff = (compute_xi(2,1,c+dc)-compute_xi(2,1,c))/dc
    print(diff)
    print(ana_diff(2, 1, c))
    
    auto_diff = grad(compute_xi,argnums=2)
    res = auto_diff(2.0,1.0,c)
    print(res)