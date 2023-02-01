from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController
import jax

def ode_solve(t1, y0):
    vector_field = lambda t, y, args: -y
    term = ODETerm(vector_field)
    solver = Dopri5()
    saveat = SaveAt(ts=[t1])
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

    if t1>0:
        dt = 0.1
    else:
        dt = -0.1

    sol = diffeqsolve(term, solver, t0=0, t1=t1, dt0=dt, y0=y0, saveat=saveat)

    return sol.ys[0]


sol = ode_solve(-3,1)
grad = jax.grad(ode_solve, argnums=(0,1))(-3.0,1.0)
print(grad)
# print(sol.ts)  # DeviceArray([0.   , 1.   , 2.   , 3.    ])
# print(sol.ys)  # DeviceArray([1.   , 0.368, 0.135, 0.0498])