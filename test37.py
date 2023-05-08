import numpy as np 
from scipy.integrate import solve_ivp
import scipy.linalg
import matplotlib.pyplot as plt 

A1 = np.array([
    [0.3922,    0.7060,    0.0462],
    [0.6555,    0.0318,    0.0971],
    [0.1712,    0.2769,    0.8235]
])
A2 = np.array([
    [0.6948,    0.0344,    0.7655],
    [0.3171,    0.4387,    0.7952],
    [0.9502,    0.3816,    0.1869],
])
A3 = np.array([
    [0.4898,    0.7094,    0.6797],
    [0.4456,    0.7547,    0.6551],
    [0.6463,    0.2760,    0.1626],
])
x0 = np.array([
    0.1190,
    0.4984,
    0.9597
])

def func(t,x,A):
    # x = np.reshape(x, (3,1))
    x_dot = A@x 
    return x_dot

if __name__ == "__main__":
    fig = plt.figure('A1')
    ax = plt.axes(projection='3d')
    x0 = np.array([1,-1,0])
    th = 1
    t = np.linspace(0,th,1000)
    res = solve_ivp(func, (0,th), x0, method='RK45', t_eval = t, args=(A1, ))
    trace = res.y

    x = trace[0,:]
    y = trace[1,:]
    z = trace[2,:]

    
    ax.plot3D(x,y,z,'b')
    ax.plot3D(x0[0],x0[1],x0[2],'b*')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()

    