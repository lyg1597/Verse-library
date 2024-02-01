import numpy as np 
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import matplotlib.pyplot as plt 
import sys

# Randomly sample a system with existing optimal

# A1 = np.array([[ 0.19837346,  0.6542015 , -0.21367801],
#  [ 0.73123647, -0.47246575,  0.41493367],
#  [ 0.85342198,  0.52359331,  0.03809274],])
# A2 = np.array([[ 0.0873275 ,  0.33826876,  0.48281116],
#  [ 0.22547549, -0.2611183 ,  0.44355073],
#  [ 0.77737517, -0.06798378, -0.05536394],])
# A3 = np.array([[-0.03014066,  0.14789939, -0.54474034],
#  [ 0.92327475,  0.11419394, -0.3863899 ],
#  [-0.84091924, -0.7624304 , -0.7638612 ],])

def func(t,x,A):
    # x = np.reshape(x, (3,1))
    x_dot = A@x 
    return x_dot

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
        A1 = np.random.uniform(-1,1,(2,2))
        A2 = np.random.uniform(-1,1,(2,2))
        # A3 = np.random.uniform(-1,1,(3,3))
        W,_ = np.linalg.eig(A1)
        if all([val.real>=0 for val in W]) or all([val.real<=0 for val in W]):
            continue
        W,_ = np.linalg.eig(A2)
        if all([val.real>=0 for val in W]) or all([val.real<=0 for val in W]):
            continue
        # W,_ = np.linalg.eig(A3)
        # if all([val.real>=0 for val in W]) or all([val.real<=0 for val in W]):
        #     continue

        x0 = np.array([
            0.1190,
            0.4984,
        ])


        eta = 1e-4

        x0 = np.array([1,-1])
        P = np.array([[0.0],[0.0]])
        Q = np.eye(2)
        fail = False

        for i in range(20000):
            xtf = expm(A2*P[1])@expm(A1*P[0])@x0 
            CP = xtf.T@Q@xtf
            
            DC1 = 2*xtf.T@expm(A1.T*P[0])@A1.T@expm(A2.T*P[1])@Q@xtf
            DC2 = 2*xtf.T@expm(A1.T*P[0])@expm(A2.T*P[1])@A2.T@Q@xtf

            val1 = P[0,0]-eta*DC1
            val2 = P[1,0]-eta*DC2

            P[0,0] = val1
            P[1,0] = val2 

            
            print(CP)
            print(P)
            print(f"{i}, ################")
            if any(elem<0 for elem in P):
                fail = True 
                break
    
        if not fail:
            break

    print(A1)
    print(A2)
