import numpy as np 
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import matplotlib.pyplot as plt 
import sys

# 3-d example, global optimal exist and is feasible. Checking convexity for this example

A1 = np.array([[ 0.1535, -0.8855,  0.2217],
 [ 0.1907, -0.9977, -0.3059],
 [ 0.1281,  0.9246, -0.5226],])
A2 = np.array([[-0.4510,  0.2605,  0.8095],
 [ 0.0659,  0.2352,  0.9419],
 [ 0.5653, -0.1497,  0.7590],])
A3 = np.array([[-0.9892,  0.8886 ,  0.0911],
 [-0.8447 ,  0.9989,  0.1088],
 [ 0.3062,  0.3690,  0.2870],])


def func(t,x,A):
    # x = np.reshape(x, (3,1))
    x_dot = A@x 
    return x_dot

if __name__ == "__main__":
    fig = plt.figure('path')
    ax = plt.axes(projection='3d')
    # x0 = np.array([1,-1,0])
    # th = 1
    # t = np.linspace(0,th,1000)
    # res = solve_ivp(func, (0,th), x0, method='RK45', t_eval = t, args=(A1, ))
    # trace = res.y

    # x = trace[0,:]
    # y = trace[1,:]
    # z = trace[2,:]

    
    # ax.plot3D(x,y,z,'b')
    # ax.plot3D(x0[0],x0[1],x0[2],'b*')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')

    # for j in range(10000):
    eta = 1e-4

    x0 = np.array([1,-1,0])

    P0 = np.array([1.99962683, 0.26641543, 0.61134131])
    Q = np.eye(3)
    xtf = expm(A3*P0[2])@expm(A2*P0[1])@expm(A1*P0[0])@x0 
    CP0 = xtf.T@Q@xtf
        
    P1 = np.array([1.7860, 0.4988, 0.5769])
    Q = np.eye(3)
    xtf = expm(A3*P1[2])@expm(A2*P1[1])@expm(A1*P1[0])@x0 
    CP1 = xtf.T@Q@xtf

    gamma = 0.4709
    P2 = gamma*P0+(1-gamma)*P1
    Q = np.eye(3)
    xtf = expm(A3*P2[2])@expm(A2*P2[1])@expm(A1*P2[0])@x0 
    CP2 = xtf.T@Q@xtf

    CP01 = gamma*CP0+(1-gamma)*CP1 

    # The function is convex is CP2=f(gamma*P0+(1-gamma)*P1)<CP01 = gamma*CP0+(1-gamma)*CP1
    # if CP2>CP01:
    #     print(j)
    #     print(P0)
    #     print(P1)
    #     print(gamma)
    #     # print(P2)
    #     print("non convexity found")

    # if CP2>max(CP0, CP1):
    #     print(P0)
    #     print(P1)
    #     print(P2)
    #     print(CP0)
    #     print(CP1)
    #     print(CP2)
    #     print(gamma)
    #     print("Non quasi convexity found")
    
    P_star = np.array([0.97848819, 0.60015272, 0.46271311])
    # Checking star convexity
    Q = np.eye(3)
    xtf = expm(A3*P_star[2])@expm(A2*P_star[1])@expm(A1*P_star[0])@x0 
    CP_star = xtf.T@Q@xtf

    gamma = 0.9518930587320524
    P3 = gamma*P0+ (1-gamma)*P_star
    Q = np.eye(3)
    xtf = expm(A3*P3[2])@expm(A2*P3[1])@expm(A1*P3[0])@x0 
    CP3 = xtf.T@Q@xtf

    if CP3 > gamma*CP0+(1-gamma)*CP_star:
        print(P0)
        print(gamma)
        print(CP3)
        print(gamma*CP0+(1-gamma)*CP_star)
        print("Non star convexity found")


    
    

    

    
    
