import numpy as np
import scipy.integrate as integrate 
from scipy.linalg import expm, solve_continuous_are

DIM = 5

K_ana = np.array([
    [22.224327,  21.265285,  21.512522,  21.685617,  21.82094,  ],
    [10.632643,  11.8487425, 10.961647,  11.038477,  11.0972395,],
    [ 7.1708407,  7.307765,   8.192894,   7.421537,   7.4550896,],
    [ 5.4214044,  5.5192385,  5.5661526,  6.302893,   5.617014, ],
    [ 4.364188,   4.4388957,  4.473054,   4.4936113,  5.140092, ],
])

K_sol = np.array([
    [21.89348,   23.369263,  23.555174,  22.893055,  21.883415 ],
    [10.164404,  12.471903,  12.69359,   11.679326,  10.424602 ],
    [ 6.614687,   8.253322,   8.623323,   7.9350395,  6.9538713],
    [ 5.0669856,  6.021194,   6.3205366,  6.0351434,  5.54575  ],
    [ 4.1642,     4.7948194,  4.820076,   4.6402936,  4.9514174]
])

A_base = np.ones((DIM,DIM))
# for i in range(DIM):
#     for j in range(DIM):
#         A_base = A_base.at[i,j].set(i+j+1)
A = A_base@A_base.T
B = np.eye(DIM)
Q = np.eye(DIM)
R = 2*np.eye(DIM,DIM)
for i in range(DIM):
    R[i,i]=i+1
R = 1/2*R

x0 = np.ones((DIM,1))

P = solve_continuous_are(A,B,Q,R)
# K = np.linalg.inv(R)@B.T@P
K = K_sol

def func(t, args):
    A,B,K,R = args
    xt = expm((A-B@K)*t)@x0
    res = xt.T@xt + xt.T@K.T@R@K@xt 
    return res 

res = integrate.quad(func, 0, 20, args = (A,B,K,R))
print(res)