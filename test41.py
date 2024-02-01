import numpy as np 
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import matplotlib.pyplot as plt 
import sys

# 3-d example, global optimal exist and is feasible. 

A1 = np.array([[-0.26451189, -0.44507302, -0.27353296],
 [-0.09530698,  0.93058662,  0.29888835],
 [ 0.3335902 ,  0.11226593, -0.94637758],])
B1 = np.array([-0.2093202,   0.69116982,  0.93632235])
A2 = np.array([[ 0.05939902, -0.79284431,  0.58041772],
 [ 0.22229864,  0.15493737,  0.53559285],
 [-0.09578686,  0.36867213, -0.95720527],])
B2 = np.array([0.60516792, 0.81991792, 0.6749214 ])
A3 = np.array([[-0.90522988, -0.00455259,  0.89326311],
 [-0.32821855,  0.84423832,  0.46007686],
 [ 0.83038975, -0.09769691, -0.79698896],])
B3 = np.array([-0.74164671,  0.92629276, -0.38736117])

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

    for j in range(1):
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

        x0 = np.array([
            0.1190,
            0.4984,
            0.9597
        ])


        eta = 1e-6

        # x0 = np.array([1,-1,0])
        # P = np.array([[0.0],[0.0],[0.0]])
        # P = np.array([np.random.uniform(2.5,3.5),np.random.uniform(7.0,8.0), np.random.uniform(2.0,3.0)])
        P = np.array([3.08385692, 7.39470173, 2.42733856])
        P0 = (P[0],P[1],P[2])
        p1 = []
        p2 = []
        p3 = []
        Q = np.eye(3)
        fail = False

        for i in range(1000000):
            p1.append(P[0])
            p2.append(P[1])
            p3.append(P[2])
            x1 = expm(A1*P[0])@x0 - np.linalg.inv(A1)@B1
            x2 = expm(A2*P[1])@x1 - np.linalg.inv(A2)@B2
            x3 = expm(A3*P[2])@x2 - np.linalg.inv(A3)@B3
            xtf = x3
            # xtf = expm(A3*P[2])@(expm(A2*P[1])@(expm(A1*P[0])@x0) - np.linalg.) - np.linalg.inv(A3)@B3 
            CP = xtf.T@Q@xtf
            
            D1 = expm(A3*P[2])@expm(A2*P[1])@A1@expm(A1*P[0])@x0
            D2 = expm(A3*P[2])@A2@expm(A2*P[1])@x1
            D3 = A3@expm(A3*P[2])@x2

            DC1 = 2*D1.T@Q@xtf
            DC2 = 2*D2.T@Q@xtf
            DC3 = 2*D3.T@Q@xtf

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