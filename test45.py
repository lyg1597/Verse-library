import numpy as np 
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import matplotlib.pyplot as plt 
import sys

# 3-d example, global optimal exist and is feasible. 
# Checking convexity for this example

A1 = np.array([[ 0.15349186, -0.88553462,  0.2217264 ],
 [ 0.19070735, -0.99773797, -0.30588691],
 [ 0.12806551,  0.92457259, -0.5226069 ],])
A2 = np.array([[-0.45100852,  0.26047977,  0.80949892],
 [ 0.06587522,  0.23522314,  0.94191645],
 [ 0.56533049, -0.14968034,  0.75901913],])
A3 = np.array([[-0.98921771,  0.8886189 ,  0.09112852],
 [-0.8446974 ,  0.99885956,  0.10878647],
 [ 0.30617329,  0.36903922,  0.28698174],])


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
    P_star = np.array([0.97848819, 0.60015272, 0.46271311])

    for j in range(10000):
        eta = 1e-4

        x0 = np.array([1,-1,0])

        P0 = np.array([np.random.uniform(0,2),np.random.uniform(0,1),np.random.uniform(0,1)])
        Q = np.eye(3)
        xtf = expm(A3*P0[2])@expm(A2*P0[1])@expm(A1*P0[0])@x0 
        CP0 = xtf.T@Q@xtf
            
        P1 = np.array([np.random.uniform(0,2),np.random.uniform(0,1),np.random.uniform(0,1)])
        Q = np.eye(3)
        xtf = expm(A3*P1[2])@expm(A2*P1[1])@expm(A1*P1[0])@x0 
        CP1 = xtf.T@Q@xtf

        gamma = np.random.uniform(0,1)
        P2 = gamma*P0+(1-gamma)*P1
        Q = np.eye(3)
        xtf = expm(A3*P2[2])@expm(A2*P2[1])@expm(A1*P2[0])@x0 
        CP2 = xtf.T@Q@xtf

        CP01 = gamma*CP0+(1-gamma)*CP1 

        # Checking convexity
        # The function is convex is CP2=f(gamma*P0+(1-gamma)*P1)<CP01 = gamma*CP0+(1-gamma)*CP1
        # if CP2>CP01:
        #     print(j)
        #     print(P0)
        #     print(P1)
        #     print(gamma)
        #     # print(P2)
        #     print("non convexity found")

        # Checking quasi convexsity
        # if CP2>max(CP0, CP1):
        #     print(j)
        #     print(P0)
        #     print(P1)
        #     print(gamma)
        #     print("Non quasi convexity found")
        #     break
        
        # Checking star convexity
        Q = np.eye(3)
        xtf = expm(A3*P_star[2])@expm(A2*P_star[1])@expm(A1*P_star[0])@x0 
        CP_star = xtf.T@Q@xtf

        gamma = np.random.uniform(0,1)
        P3 = gamma*P0+ (1-gamma)*P_star
        Q = np.eye(3)
        xtf = expm(A3*P3[2])@expm(A2*P3[1])@expm(A1*P3[0])@x0 
        CP3 = xtf.T@Q@xtf

        if CP3 > gamma*CP0+(1-gamma)*CP_star:
            print(CP3)
            print(gamma*CP0+(1-gamma)*CP_star)
            print(j)
            print(P0)
            print(gamma)
            print("Non star convexity found")
            break


        
        
