import matplotlib.pyplot as plt 
import pickle 
from scipy.linalg import expm 
from scipy.integrate import quad
import numpy as np
import jax.scipy as jscipy
import jax
jax.config.update('jax_platform_name', 'cpu')

def func_check(t, args):
    A,B,K,R,x0 = args
    # try:
    #     xt = expm((A-B@K)*t)@x0
    # except:
    #     xt = np.zeros(x0.shape)
    xt = jscipy.linalg.expm((A-B@K)*t)@x0
    res = xt.T@xt + xt.T@K.T@R@K@xt 
    return res 

def non_diff_cost(t1, A, B, K, R, x0):
    res = quad(func_check, 0, t1, args = ((A,B,K,R,x0),))
    return res[0]

with open('test28_long_1.pkl','rb') as f:
    loss_diff_list, loss_diff_gt_list, loss_true_list, loss_true_gt_list, K_list, K_gt_list = pickle.load(f)    

res_mean = []
res_std = []
for DIM in range(2, len(K_list)+2):
    print(DIM)
    loss_gt = []
    loss_dp = []
    loss_diff = []
    for i in range(len(K_list[DIM-2])):
        A,R,K_dp = K_list[DIM-2][i]
        _,_,K_gt = K_gt_list[DIM-2][i]

        cost_dp = non_diff_cost(3, A, np.eye(DIM), K_dp, R, x0 = np.ones((DIM,1)))
        cost_gt = non_diff_cost(3, A, np.eye(DIM), K_gt, R, x0 = np.ones((DIM,1)))
        
        loss_dp.append(cost_dp)
        loss_gt.append(cost_gt)
        loss_diff.append(abs(cost_dp-cost_gt)/cost_gt)
        print(DIM, i, abs(cost_dp-cost_gt)/cost_gt)
        if abs(cost_dp-cost_gt)/cost_gt>1:
            print("::::::", cost_dp, cost_gt)
    loss_mean = np.mean(loss_diff)
    loss_std = np.std(loss_diff)
    res_mean.append(loss_mean)
    res_std.append(loss_std)

plt.plot([*range(2,len(K_list)+2)], res_mean)
plt.errorbar([*range(2,len(K_list)+2)], res_mean, res_std, linestyle='None', marker='^')

plt.show()