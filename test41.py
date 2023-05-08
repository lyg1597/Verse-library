import numpy as np 
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import matplotlib.pyplot as plt 
import sys

# 3-d example, global optimal exist and is feasible. 

A1 = np.array([[ 0.45797441, -0.62485492, -0.44152279],
 [-0.40043705, -0.04777642,  0.65826702],
 [ 0.98529847, -0.63715002, -0.78120171],])
B1 = np.array([ 0.18265337, -0.85801869, -0.56721677])
A2 = np.array([[ 0.72444012, -0.92108537, -0.1194663 ],
 [ 0.67627832,  0.39147163, -0.02213997],
 [-0.67135075, -0.40982228, -0.66738056],])
B2 = np.array([ 0.03000718, -0.13081883, -0.26529495])
A3 = np.array([[-0.93296787,  0.90780095,  0.01931887],
 [-0.57766271, -0.57317075, -0.8835233 ],
 [ 0.02707514, -0.76149803,  0.10329868],])
B3 = np.array([0.59383148, 0.30229878, 0.91392834])

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

        x0 = np.array([
            0.1190,
            0.4984,
            0.9597
        ])


        eta = 1e-4

        x0 = np.array([1,-1,0])
        # P = np.array([[0.0],[0.0],[0.0]])
        # P = np.random.uniform(0,2,(3,))
        P = [1.06896484, 0.67517184, 4.19103869]
        P0 = (P[0],P[1],P[2])
        p1 = []
        p2 = []
        p3 = []
        Q = np.eye(3)
        fail = False

        for i in range(100000):
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


            
            print(CP)
            print(P)
            print(f"{i}, ################")
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