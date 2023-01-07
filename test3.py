import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
from jax import jacfwd, jacrev, jit
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def func(i, tmp):
    return tmp + 1

def test_func(ub):
    res = jax.lax.while_loop(0, ub, func, 0)
    return res 

if __name__ == "__main__":
    res1 = test_func(100)
    res2 = jax.grad(test_func, allow_int=True)(100)
    print(res1)
    print(res2)