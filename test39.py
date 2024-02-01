import numpy as np 
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import matplotlib.pyplot as plt 
import sys

# 3-d example, global optimal exist and is feasible. Gold example to keep!!

A1 = np.array([[ 0.15349186, -0.88553462,  0.2217264 ],
 [ 0.19070735, -0.99773797, -0.30588691],
 [ 0.12806551,  0.92457259, -0.5226069 ],])
A2 = np.array([[-0.45100852,  0.26047977,  0.80949892],
 [ 0.06587522,  0.23522314,  0.94191645],
 [ 0.56533049, -0.14968034,  0.75901913],])
A3 = np.array([[-0.98921771,  0.8886189 ,  0.09112852],
 [-0.8446974 ,  0.99885956,  0.10878647],
 [ 0.30617329,  0.36903922,  0.28698174],])
# A4 = np.array([[ 0.10281819, -0.56175182, -0.39050629],
#  [-0.70471767,  0.2114341 ,  0.67865193],
#  [-0.90680211, -0.85267111, -0.66241973]])

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

    for j in range(10):
        print(j)
        # A1 = np.random.uniform(-1,1,(3,3))
        # A2 = np.random.uniform(-1,1,(3,3))
        # A3 = np.random.uniform(-1,1,(3,3))
        W,_ = np.linalg.eig(A1)
        if all([val.real>=0 for val in W]) or all([val.real<=0 for val in W]):
            continue
        W,_ = np.linalg.eig(A2)
        if all([val.real>=0 for val in W]) or all([val.real<=0 for val in W]):
            continue
        W,_ = np.linalg.eig(A3)
        if all([val.real>=0 for val in W]) or all([val.real<=0 for val in W]):
            continue

        eta = 1e-4

        x0 = np.array([1,-1,0])
        # x0 = np.array([
        #     0.1190,
        #     0.4984,
        #     0.9597
        # ])
        # P = np.array([[0.0],[0.0],[0.0]])
        P = np.array([np.random.uniform(0,2),np.random.uniform(0,1),np.random.uniform(0,1)])
        # P = np.array([0.99004004, 0.59243114, 0.46934092])
        P0 = (P[0],P[1],P[2])
        p1 = []
        p2 = []
        p3 = []
        Q = np.eye(3)
        fail = False

        for i in range(50000):
            p1.append(P[0])
            p2.append(P[1])
            p3.append(P[2])
            xtf = expm(A3*P[2])@expm(A2*P[1])@expm(A1*P[0])@x0 
            CP = xtf.T@Q@xtf
            
            DC1 = 2*x0@expm(A1.T*P[0])@A1.T@expm(A2.T*P[1])@expm(A3.T*P[2])@Q@xtf
            DC2 = 2*x0@expm(A1.T*P[0])@expm(A2.T*P[1])@A2.T@expm(A3.T*P[2])@Q@xtf
            DC3 = 2*x0@expm(A1.T*P[0])@expm(A2.T*P[1])@expm(A3.T*P[2])@A3.T@Q@xtf

            val1 = P[0]-eta*DC1
            val2 = P[1]-eta*DC2
            val3 = P[2]-eta*DC3

            P[0] = val1
            P[1] = val2 
            P[2] = val3

            
            # print(CP)
            # print(P)
            # print(f"{i}, ################")
            # if any(elem<0 for elem in P):
            #     fail = True 
            #     break

        print(CP)
        print(P)
        print(f"{i}, ################")
        ax.plot3D(p1,p2,p3,'b')
        ax.plot3D(P0[0],P0[1],P0[2],'b*')
        ax.plot3D(p1[-1],p2[-1],p3[-1],'g*')
        # if not fail:
        #     break
        # print("xxxxx Fail xxxxx")

    ax.set_xlabel('p1')
    ax.set_ylabel('p2')
    ax.set_zlabel('p3')
    plt.show()
    # print(A1)
    # print(A2)
    # print(A3)