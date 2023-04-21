import numpy as np 
from scipy.integrate import ode 
import matplotlib.pyplot as plt 

A = np.random.uniform(-1,1,size=(2,2))

def func(t,x):
    dot_x = A@x
    return dot_x 

if __name__ == "__main__":
    r = ode(func)
    init = np.random.uniform(-5,5,size=(2,1))
    r.set_initial_value(init)
    trace = [init]
    dt = 0.01
    for i in range(500):
        trace.append(r.integrate(r.t+dt))

    trace = np.array(trace)
    print(A, trace.shape)
    plt.figure()
    plt.plot(trace[:,0,:], trace[:,1,:])
    plt.plot(trace[0,0,0], trace[0,1,0], '*r')
    plt.figure()
    tmp = np.random.uniform(-1,1,size=(2,2))
    Q = tmp@tmp.T 
    cost = np.diag(trace[:,:,0]@Q@trace[:,:,0].T)
    plt.plot(cost)
    plt.show()