import numpy as np 
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import matplotlib.pyplot as plt 

def func(t,x,A):
    # x = np.reshape(x, (3,1))
    x_dot = A@x 
    return x_dot

A1 = np.array([[ 0.15349186, -0.88553462,  0.2217264 ],
 [ 0.19070735, -0.99773797, -0.30588691],
 [ 0.12806551,  0.92457259, -0.5226069 ],])
A2 = np.array([[-0.45100852,  0.26047977,  0.80949892],
 [ 0.06587522,  0.23522314,  0.94191645],
 [ 0.56533049, -0.14968034,  0.75901913],])
A3 = np.array([[-0.98921771,  0.8886189 ,  0.09112852],
 [-0.8446974 ,  0.99885956,  0.10878647],
 [ 0.30617329,  0.36903922,  0.28698174],])

x0 = np.array([1,-1,0])

P = np.array([0.99004004, 0.59243114, 0.46934092])

perturb = [-0.1,0,0.1]

plt.figure('path')
ax = plt.axes(projection='3d')
plt.figure("C")             

for i in range(3):
    for j in range(3):
        for k in range(3):
            if i==j==k==1:
                c = "b"
            else:
                c = "r" 

            P1_p = P[0] + perturb[i]
            P2_p = P[1] + perturb[j]
            P3_p = P[2] + perturb[k]

            P_p = np.array([P1_p, P2_p, P3_p])

            t = 0
            t_list = []
            state_list = []
            C_list = []
            while t<(P_p[0]+ P_p[1]+ P_p[2]):
                if t<=P_p[0]:
                    x = expm(A1*t)@x0 
                elif t>P_p[0] and t<=P_p[1]+P_p[0]:
                    x = expm(A2*(t-P_p[0]))@expm(A1*P_p[0])@x0 
                else:
                    x = expm(A3*(t-P_p[1]-P_p[0]))@expm(A2*P_p[1])@expm(A1*P_p[0])@x0
                C = x.T@x

                C_list.append(C)
                state_list.append(x)
                t_list.append(t)
                t+= 0.01

            state_list = np.array(state_list)

            plt.figure('path')

            ax.plot3D(state_list[:,0],state_list[:,1],state_list[:,2],f'{c}')
            ax.plot3D(state_list[0,0],state_list[0,1],state_list[0,2],f'{c}*')
            # ax.plot3D(p1[-1],p2[-1],p3[-1],'g*')
            # if not fail:
            #     break
            # print("xxxxx Fail xxxxx")

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            plt.figure("C") 
            plt.plot(C_list, c)
            plt.plot(len(C_list)-1, C_list[-1], f"{c}*")

P1_p = P[0] + perturb[1]
P2_p = P[1] + perturb[1]
P3_p = P[2] + perturb[1]

P_p = np.array([P1_p, P2_p, P3_p])

t = 0
t_list = []
state_list = []
C_list = []
while t<(P_p[0]+ P_p[1]+ P_p[2]):
    if t<=P_p[0]:
        x = expm(A1*t)@x0 
    elif t>P_p[0] and t<=P_p[1]+P_p[0]:
        x = expm(A2*(t-P_p[0]))@expm(A1*P_p[0])@x0 
    else:
        x = expm(A3*(t-P_p[1]-P_p[0]))@expm(A2*P_p[1])@expm(A1*P_p[0])@x0
    C = x.T@x

    C_list.append(C)
    state_list.append(x)
    t_list.append(t)
    t+= 0.01

state_list = np.array(state_list)

plt.figure('path')

ax.plot3D(state_list[:,0],state_list[:,1],state_list[:,2],f'b')
ax.plot3D(state_list[0,0],state_list[0,1],state_list[0,2],f'b*')
# ax.plot3D(p1[-1],p2[-1],p3[-1],'g*')
# if not fail:
#     break
# print("xxxxx Fail xxxxx")

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.figure("C") 
plt.plot(C_list, "b")
plt.plot(len(C_list)-1, C_list[-1], "b*")

plt.show()