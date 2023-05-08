import numpy as np 
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import matplotlib.pyplot as plt 
import sys

if __name__ == "__main__":
    # fig = plt.figure('A1')
    # ax = plt.axes(projection='3d')
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

    for j in range(100000):
        A1 = np.random.uniform(-1,1,(3,3))
        A2 = np.random.uniform(-1,1,(3,3))
        A3 = np.random.uniform(-1,1,(3,3))

        B1 = np.random.uniform(-1,1,(3,))
        B2 = np.random.uniform(-1,1,(3,))
        B3 = np.random.uniform(-1,1,(3,))

        W,_ = np.linalg.eig(A1)
        if np.linalg.det(A1)==0 or all([val.real>=0 for val in W]) or all([val.real<=0 for val in W]):
            continue
        W,_ = np.linalg.eig(A2)
        if np.linalg.det(A2)==0 or all([val.real>=0 for val in W]) or all([val.real<=0 for val in W]):
            continue
        W,_ = np.linalg.eig(A3)
        if np.linalg.det(A3)==0 or all([val.real>=0 for val in W]) or all([val.real<=0 for val in W]):
            continue

        x0 = np.array([
            0.1190,
            0.4984,
            0.9597
        ])


        eta = 1e-4

        # x0 = np.array([1,-1,0])
        P = np.array([0.0,0.0,0.0])
        Q = np.eye(3)
        fail = False

        for i in range(20000):
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

            
            print(CP)
            print(P)
            print(f"{i}, ################")
            if any(elem<0 for elem in P):
                fail = True 
                break
    
        if not fail:
            break

    print(A1)
    print(B1)
    print(A2)
    print(B2)
    print(A3)
    print(B3)
